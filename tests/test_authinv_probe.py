"""P5.1 probing tooling: labels, probes and baselines, CKA, the activation cache, and extraction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import build_tiny_benchmark, build_tiny_causal_lm

from authinv.policy import AttrRef, Condition, Entity, EntityRef, Policy, Request, Rule, Scope, evaluate
from authinv.probe import analysis, cache
from authinv.probe.labels import governing, probe_items

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("authinv_probe", ROOT / "scripts" / "authinv_probe.py")
runner = importlib.util.module_from_spec(spec)
sys.modules["authinv_probe"] = runner
spec.loader.exec_module(runner)
CFG = yaml.safe_load((ROOT / "configs/authinv/probe.yaml").read_text())
PCFG = {**CFG["probe"], "bootstrap_replicates": 200}


def test_governing_label_names_what_settles_the_decision():
    u1, u2, g, r = (
        EntityRef("User", "u1"),
        EntityRef("User", "u2"),
        EntityRef("Team", "t"),
        EntityRef("Repo", "r"),
    )
    pol = Policy(
        "p",
        (Entity(u1, (("level", 3),), (g,)), Entity(u2, (("level", 1),)), Entity(g), Entity(r)),
        ("read", "push"),
        (
            Rule(
                "r1",
                "permit",
                Scope("in", g),
                ("read", "push"),
                Scope("is", type="Repo"),
                (Condition(AttrRef("principal", "level"), ">=", 2),),
            ),
            Rule(
                "r2",
                "permit",
                Scope("eq", u2),
                ("read",),
                Scope("is", type="Repo"),
                (Condition(AttrRef("principal", "level"), ">=", 2),),
            ),
            Rule("r3", "forbid", Scope("is", type="User"), ("push",), Scope("eq", r)),
        ),
    )
    assert governing(pol, Request(u1, "read", r)) == "permit"
    assert governing(pol, Request(u1, "push", r)) == "forbid"  # a permit applies too; the forbid wins
    assert governing(pol, Request(u2, "read", r)) == "condition"  # r2 matches the scope, its condition fails
    assert governing(pol, Request(u2, "push", r)) == "forbid"
    assert evaluate(pol, Request(u2, "read", r))["decision"] == "deny"


def test_probe_items_carry_labels_and_metadata(tmp_path):
    pytest.importorskip("cedarpy")
    _, rows, worlds = build_tiny_benchmark(tmp_path, 2)
    items = probe_items(rows, worlds, ("application", "interpretation"), owner_slots=3)
    assert len(items) == len(rows)
    app = [it for it in items if it["task"] == "application"]
    assert all(
        it["labels"]["allow_deny"] == r["label"]
        for it, r in zip(app, [r for r in rows if r["task"] == "application"], strict=True)
    )
    assert {it["labels"]["governing"] for it in app} <= {"permit", "forbid", "condition", "scope"}
    interp = [it for it in items if it["task"] == "interpretation"]
    assert all(set(it["labels"]) == {"owner_0", "owner_1", "owner_2"} for it in interp)
    assert all("rendering" in it["meta"] and it["meta"]["n_rules"] >= 1 for it in items)
    # one instance's labels are identical across renderings (they come from the canonical policy)
    by_inst = {}
    for it in app:
        by_inst.setdefault((it["world_id"], it["assignment"], it["instance"]), set()).add(
            json.dumps(it["labels"], sort_keys=True)
        )
    assert all(len(v) == 1 for v in by_inst.values())


def _synthetic(n_worlds=60, per=8, d=24, seed=0):
    """Items in two renderings; layer 1 holds a shared label direction, layer 2 a rendering-specific one."""
    rng = np.random.default_rng(seed)
    items, rows = [], []
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    shared, da, db = (np.sqrt(d) * q[:, k] for k in range(3))  # orthogonal directions
    for w in range(n_worlds):
        for i in range(per):
            y = rng.integers(2)
            base = rng.normal(size=d)
            for g, own in (("a", da), ("b", db)):
                noise = rng.normal(size=(3, d))
                s = 2 * y - 1
                rows.append(
                    np.stack([noise[0], noise[1] + 1.5 * s * shared, noise[2] + 1.5 * s * own]) + 0.0 * base
                )
                items.append(
                    {
                        "world_id": f"w{w}",
                        "assignment": "orig",
                        "instance": f"q{i}",
                        "rendering": g,
                        "task": "application",
                        "labels": {"y": "allow" if y else "deny"},
                        "meta": {"rendering": g, "n_rules": 1},
                    }
                )
    return np.stack(rows).astype(np.float32), items


def test_transfer_recovers_planted_signal_and_baselines_stay_at_chance():
    acts, items = _synthetic()
    t = analysis.transfer(acts, items, "y", [0, 1, 2], PCFG)
    lay = t["layers"]
    assert lay["0"]["a->a"]["balanced_accuracy"] < 0.65  # noise layer
    assert (
        lay["1"]["a->b"]["balanced_accuracy"] > 0.85 and lay["1"]["b->a"]["balanced_accuracy"] > 0.85
    )  # shared
    assert (
        lay["2"]["a->a"]["balanced_accuracy"] > 0.85 and lay["2"]["a->b"]["balanced_accuracy"] < 0.65
    )  # specific
    lo, hi = lay["1"]["a->b"]["ci95"]
    assert lo <= lay["1"]["a->b"]["balanced_accuracy"] <= hi
    assert all(v["balanced_accuracy"] < 0.65 for v in t["shuffled_label_last_layer"].values())
    assert all(v["balanced_accuracy"] < 0.65 for v in t["metadata_only"].values() if v)
    assert all(v == 0.5 for v in t["majority_class"].values())
    # the split is by world: no world on both sides
    test = analysis.world_split([it["world_id"] for it in items], 0.3, PCFG["split_seed"])
    assert 0 < len(test) < 60 and test == analysis.world_split(
        [it["world_id"] for it in items], 0.3, PCFG["split_seed"]
    )


def test_probe_is_deterministic_and_multiclass():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 10))
    y = [["x", "y", "z"][int(np.argmax(r[:3]))] for r in X]
    p1, p2 = analysis.Probe().fit(X, y), analysis.Probe().fit(X, y)
    assert p1.predict(X) == p2.predict(X)
    assert analysis.balanced_accuracy(y, p1.predict(X)) > 0.9


def test_linear_cka_properties():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 30))
    Q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    assert analysis.linear_cka(X, X) == pytest.approx(1.0)
    assert analysis.linear_cka(X, 3.0 * X @ Q) == pytest.approx(1.0)  # orthogonal + isotropic scaling
    assert analysis.linear_cka(X, rng.normal(size=(200, 30))) < 0.3
    assert analysis.linear_cka(X[:20], X[:20] @ Q) == pytest.approx(1.0)  # n < d uses the Gram form
    acts, items = _synthetic(n_worlds=10)
    m = analysis.cka_matrix(acts, items, [0, 1, 2])
    assert set(m["layers"]["1"]) == {"a|b"} and m["layers"]["1"]["a|b"] > m["layers"]["0"]["a|b"]
    div = analysis.instance_divergence(acts, items, "a", "b", 1)
    assert len(div) == 10 * 8 and all(0 <= v <= 2 for v in div.values())


def test_cache_roundtrip_and_tamper_detection(tmp_path):
    acts, items = _synthetic(n_worlds=3)
    w = cache.CacheWriter(tmp_path / "c", [0, 1, 2], {"model": "m"})
    w.add([{**it, "prompt": "p"} for it in items[:20]], acts[:20])
    w.add(items[20:], acts[20:])
    m = w.close()
    got, its, man = cache.load(tmp_path / "c")
    assert got.shape == acts.shape and np.allclose(got, acts, atol=1e-2) and len(its) == len(items)
    assert "prompt" not in its[0] and man["items_sha256"] == m["items_sha256"]
    np.save(tmp_path / "c" / "acts_00001.npy", np.zeros_like(acts[20:], dtype=np.float16))
    with pytest.raises(ValueError, match="sha256"):
        cache.load(tmp_path / "c")
    with pytest.raises(ValueError, match="do not match"):
        w.add(items[:2], acts[:3])


def test_resolve_layers():
    from authinv.probe.extract import resolve_layers

    assert resolve_layers("all", 5) == [0, 1, 2, 3, 4]
    assert resolve_layers({"stride": 2}, 6) == [0, 2, 4, 5]
    with pytest.raises(ValueError):
        resolve_layers([0, 9], 5)


def test_extraction_batches_like_single_items_and_runner_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("cedarpy")
    from authinv.probe.extract import Extractor

    model_dir = build_tiny_causal_lm(tmp_path)
    ex = Extractor({"key": "tiny", "name": "tiny"}, str(model_dir), "float32")
    prompts = ["short", "a much longer prompt than the first one", "mid length"]
    batched = ex.hidden(prompts, [0, 1, 2])
    single = np.concatenate([ex.hidden([p], [0, 1, 2]) for p in prompts])
    assert batched.shape == (3, 3, 32) and np.allclose(batched, single, atol=1e-4)
    bench, rows, _ = build_tiny_benchmark(tmp_path, 2)
    cfg = {
        **CFG,
        "dataset_sha256": None,
        "worlds": "all",
        "layers": [0, 1, 2],
        "dtype": "float32",
        "batch_size": 7,
        "probe": PCFG,
    }
    cfg_path = tmp_path / "probe.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    sys.argv = [
        "x",
        "extract",
        "--config",
        str(cfg_path),
        "--dataset",
        str(bench),
        "--model",
        "smoke_qwen3_0_6b",
        "--model-dir",
        str(model_dir),
        "--output-root",
        str(tmp_path / "out"),
    ]
    runner.main()
    run = next((tmp_path / "out").iterdir())
    assert json.loads((run / "run_status.json").read_text())["status"] == "complete"
    acts, items, man = cache.load(run / "cache")
    assert acts.shape == (len(rows), 3, 32) and man["layers"] == [0, 1, 2]
    sys.argv = [
        "x",
        "analyze",
        "--config",
        str(cfg_path),
        "--cache",
        str(run),
        "--out",
        str(tmp_path / "a.json"),
    ]
    runner.main()
    a = json.loads((tmp_path / "a.json").read_text())
    assert {"allow_deny", "governing", "owner_0"} <= set(a["transfer"]) and set(a["cka"]) == {
        "application",
        "interpretation",
    }
