"""P4.1 SFT data: train-only domains, disjointness from the eval benchmark, examples, token matching."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from authinv import provenance
from authinv.benchmark import rename, swap_axis, swap_map
from authinv.eval.structured import parse_structured
from authinv.policy import to_dict
from authinv.render.renderers import round_trips
from authinv.sft import data
from authinv.sources.synthetic import DOMAINS, design_grid
from authinv.sources.synthetic import generate as eval_generate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("authinv_build_sft", ROOT / "scripts" / "authinv_build_sft.py")
builder = importlib.util.module_from_spec(spec)
sys.modules["authinv_build_sft"] = builder
spec.loader.exec_module(builder)


@pytest.fixture(scope="module")
def worlds():
    return {d: data.generate_worlds(d, 1, 7) for d in data.TRAIN_DOMAINS}


def test_every_domain_fills_the_design_grid_deterministically(worlds):
    for domain, ws in worlds.items():
        assert len(ws) == len(design_grid())
        again = data.generate_worlds(domain, 1, 7)
        assert [to_dict(p) for p, _ in again] == [to_dict(p) for p, _ in ws]
        cells = {
            (
                dict(p.meta)["tier_cell"],
                dict(p.meta)["roles"],
                dict(p.meta)["conditions"],
                dict(p.meta)["effect"],
            )
            for p, _ in ws
        }
        assert len(cells) == len(design_grid())
        for p, _ in ws:
            assert all(round_trips(p, g) for g in data.TRAIN_RENDERINGS)


def test_training_vocabulary_is_disjoint_from_the_eval_synthetic_worlds(worlds):
    train = [p for ws in worlds.values() for p, _ in ws]
    eval_policies = [to_dict(p) for d in DOMAINS for p in eval_generate(d, 1, 20261009)[0]]
    report = data.disjointness(train, eval_policies)
    assert report["passed"] and report["semantic_hash_overlap"] == 0
    # the check bites: an eval policy among the training ones is caught twice over
    leaked = data.disjointness(train + [eval_generate("repo", 1, 20261009)[0][0]], eval_policies)
    assert (
        not leaked["passed"]
        and leaked["semantic_hash_overlap"] == 1
        and leaked["vocabulary_overlap"]["types"]
    )


def test_examples_follow_the_benchmark_and_answers_parse_as_correct(worlds):
    p, _ = worlds["ledger"][5]
    d = data.TRAIN_DOMAINS["ledger"]
    exs = data.world_examples(p, d, data.TRAIN_RENDERINGS, per_label=2, seed=1)
    assert {e["rendering"] for e in exs} == set(data.TRAIN_RENDERINGS)  # Cedar and Rego never appear
    assert not any(
        "permit (" in e["messages"][0]["content"] or "package authz" in e["messages"][0]["content"]
        for e in exs
    )
    assert len({e["example_id"] for e in exs}) == len(exs)
    app = [e for e in exs if e["task"] == "application"]
    labels = [json.loads(e["messages"][1]["content"])["decision"] for e in app]
    assert labels.count("allow") == labels.count("deny")
    swapped = rename(p, swap_map(p, swap_axis(p, (d.principal_type,), (d.resource_type,))))
    for e in exs:
        want = json.loads(e["messages"][1]["content"])
        expected = want["decision"] if e["task"] == "application" else want["principals"]
        # the target is exactly what the T2 strict parser scores as correct
        assert parse_structured(e["messages"][1]["content"], e["task"], expected)["category"] == "correct"
    q = next(e for e in app if e["assignment"] == "swap")
    assert q["world_id"] == p.policy_id and swapped.policy_id == p.policy_id
    # each rendering of one query has the same facts, request and answer
    by_q = {}
    for e in app:
        by_q.setdefault((e["assignment"], e["instance"]), set()).add(e["messages"][1]["content"])
    assert all(len(v) == 1 for v in by_q.values())


def test_match_tokens_takes_the_shortest_prefix_and_refuses_a_small_pool():
    pool = [{"n": n} for n in (5, 5, 5, 5)]
    got, total = data.match_tokens(pool, 12, lambda e: e["n"])
    assert len(got) == 3 and total == 15 and got[0]["tokens"] == 5
    with pytest.raises(ValueError, match="too small"):
        data.match_tokens(pool, 21, lambda e: e["n"])


def test_build_writes_sets_manifest_and_refuses_overlap(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    eval_ws = eval_generate("repo", 1, 20261009)[0][:3]
    provenance.write_jsonl(
        bench / "worlds.jsonl",
        [{"world_id": w.policy_id, "orig": to_dict(w), "swap": to_dict(w)} for w in eval_ws],
    )
    provenance.write_json(bench / "dataset_manifest.json", {"sha256": "b" * 64, "rows": 0})
    cfg = yaml.safe_load((ROOT / "configs/authinv/sft_data.yaml").read_text())
    cfg.update(
        eval_benchmark_sha256="b" * 64,
        domains=["ledger", "archive"],
        replicates={"multi": 1, "val": 1, "control_pool": 8},
    )
    run = builder.build(cfg, bench, "chars", tmp_path / "out")
    m = json.loads((run / "manifest.json").read_text())
    assert m["disjointness"]["passed"] and m["held_out_renderings"] == ["executable", "rego"]
    assert set(m["sets"]["train_multi"]["by_rendering"]) == set(data.TRAIN_RENDERINGS)
    assert set(m["sets"]["control_qwen3_5_9b"]["by_rendering"]) == {"nl_statement"}
    for key in ("qwen3_5_9b", "gemma4_12b"):
        assert 1.0 <= m["tokens"][key]["control_over_train"] < 1.01
        assert m["no_sft_baseline"]["models"][key]["phase2_t2_run"].endswith(f"_authinv_t2_{key}")
    for name, info in m["files"].items():
        assert provenance.sha256_file(run / name) == info["sha256"]
    worlds_out = provenance.read_jsonl(run / "worlds.jsonl")
    assert {w["set"] for w in worlds_out} == {"multi", "val", "control_pool"}
    # a training world that is also an eval world is refused
    provenance.write_jsonl(
        bench / "worlds.jsonl",
        [
            {
                "world_id": "x",
                "orig": json.loads(json.dumps(worlds_out[0], default=str)),
                "swap": to_dict(eval_ws[0]),
            }
        ],
    )
    with pytest.raises(SystemExit, match="overlaps the eval benchmark"):
        builder.build(cfg, bench, "chars", tmp_path / "out2")
    with pytest.raises(SystemExit, match="partition"):
        builder.build({**cfg, "held_out_renderings": ["executable"]}, bench, "chars", tmp_path / "out3")
