"""D1.1 release builder and leaderboard: layout, license and proof checks, verification, the table."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from conftest import build_tiny_benchmark

from authinv import provenance, release
from authinv.policy import from_dict, policy_hash

pytest.importorskip("cedarpy")  # the executable rendering is Cedar text

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "authinv_leaderboard", ROOT / "scripts" / "authinv_leaderboard.py"
)
board = importlib.util.module_from_spec(spec)
sys.modules["authinv_leaderboard"] = board
spec.loader.exec_module(board)
APACHE = "Apache License\nVersion 2.0, January 2004\n"


def _setup(tmp_path: Path, upstream: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bench, rows, worlds = build_tiny_benchmark(tmp_path, 2)
    proofs = [
        {"policy_id": wid, "assignment": a, "passed": True, "policy_sha256": policy_hash(from_dict(w[a]))}
        for wid, w in worlds.items()
        for a in ("orig", "swap")
    ]
    provenance.write_jsonl(bench / "equivalence.jsonl", proofs)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "LICENSE").write_text("Copyright (c) 2015, Upstream Authors\nAll rights reserved.\n")
    (raw / "NOTICE").write_text("Upstream\nCopyright 2026 Upstream Authors\n")
    (tmp_path / "APACHE").write_text(APACHE)
    pin = {
        "repo": "https://example.org/upstream",
        "commit": "abc123",
        "license": "BSD-2-Clause",
        "file_sha256": {f: hashlib.sha256((raw / f).read_bytes()).hexdigest() for f in ("LICENSE", "NOTICE")},
    }
    (tmp_path / "pin.yaml").write_text(yaml.safe_dump(pin))
    cfg = yaml.safe_load((ROOT / "configs/authinv/release.yaml").read_text())
    cfg.update(
        benchmark_sha256=json.loads((bench / "dataset_manifest.json").read_text())["sha256"],
        release_license_file="APACHE",
        release_license_sha256=hashlib.sha256(APACHE.encode()).hexdigest(),
        sources={
            "synthetic": {"pin_config": "pin.yaml", "raw_dir": "raw", "license_files": ["LICENSE", "NOTICE"]}
        }
        if upstream
        else {"synthetic": {"license": "release"}},
    )
    return bench, rows, worlds, cfg


def test_release_layout_licenses_and_manifest(tmp_path):
    bench, rows, worlds, cfg = _setup(tmp_path)
    out = tmp_path / "release"
    m = release.build_release(bench, cfg, tmp_path, out, "# Datasheet\n\nbody")
    n = len(worlds)
    assert (
        m["counts"]["worlds"] == n
        and m["counts"]["renderings"] == n * 2 * 6
        and m["counts"]["requests"] == n * 2 * 4
    )
    assert m["counts"]["application"] + m["counts"]["interpretation"] == len(rows)
    for f in (
        "README.md",
        "LICENSE",
        "NOTICE",
        "licenses/synthetic/LICENSE",
        "licenses/synthetic/NOTICE",
        "data/policies.jsonl",
        "data/equivalence.jsonl",
        "data/application.jsonl",
    ):
        assert (out / f).exists(), f
    assert release.verify_release(out)["release"] == "authinv-bench"
    header = yaml.safe_load((out / "README.md").read_text().split("---")[1])
    assert header["license"] == "apache-2.0" and {c["config_name"] for c in header["configs"]} >= {
        "application",
        "policies",
        "renderings",
    }
    notice = (out / "NOTICE").read_text()
    assert "Copyright 2026 Upstream Authors" in notice  # the upstream NOTICE, verbatim
    assert "Copyright (c) 2015, Upstream Authors" in notice  # a BSD source keeps its copyright line
    pols = provenance.read_jsonl(out / "data/policies.jsonl")
    assert {p["license"] for p in pols} == {"BSD-2-Clause"} and all(
        set(p["policy_sha256"]) == {"orig", "swap"} for p in pols
    )
    reqs = provenance.read_jsonl(out / "data/requests.jsonl")
    assert {r["label"] for r in reqs} == {"allow", "deny"} and all("decision_owners" in r for r in reqs)
    app = provenance.read_jsonl(out / "data/application.jsonl")
    assert all(r["license"] == "BSD-2-Clause" for r in app)
    (out / "data/requests.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="differ from MANIFEST"):
        release.verify_release(out)


def test_release_refuses_unlicensed_sources_bad_pins_and_missing_proofs(tmp_path):
    bench, rows, worlds, cfg = _setup(tmp_path)
    with pytest.raises(SystemExit, match="compatible"):
        release.build_release(bench, {**cfg, "sources": {}}, tmp_path, tmp_path / "r1", "")
    (tmp_path / "raw" / "NOTICE").write_text("edited\n")
    with pytest.raises(SystemExit, match="pinned file"):
        release.build_release(bench, cfg, tmp_path, tmp_path / "r2", "")
    bench2, _, _, cfg2 = _setup(tmp_path / "b", upstream=False)
    proofs = provenance.read_jsonl(bench2 / "equivalence.jsonl")
    provenance.write_jsonl(bench2 / "equivalence.jsonl", proofs[1:])
    with pytest.raises(SystemExit, match="equivalence proof"):
        release.build_release(bench2, cfg2, tmp_path / "b", tmp_path / "r3", "")
    with pytest.raises(SystemExit, match="compatibility table"):
        release.check_licenses({"synthetic"}, {"synthetic": {"license": "MIT"}}, "GPL-3.0")


class FakeChat:
    def generate(self, prompts, schemas=None, max_tokens=None):
        return [
            {"text": self.answers[p], "finish_reason": "stop", "prompt_tokens": 10, "output_tokens": 5}
            for p in prompts
        ]


def test_leaderboard_scores_a_release_and_builds_the_table(tmp_path):
    bench, rows, worlds, cfg = _setup(tmp_path, upstream=False)
    out = tmp_path / "release"
    release.build_release(bench, cfg, tmp_path, out, "")
    app, sha = board.load_rows(out)
    assert sha == cfg["benchmark_sha256"] and all(r["task"] == "application" for r in app)
    chat = FakeChat()
    flip = {"allow": "deny", "deny": "allow"}
    chat.answers = {
        r["prompt"]: json.dumps(
            {
                "decision": flip[r["label"]]
                if r["rendering"] == "rego" and r["label"] == "deny"
                else r["label"]
            }
        )
        for r in app
    }
    run = tmp_path / "run"
    run.mkdir()
    preds = board.evaluate(chat, app, 7, run / "predictions.jsonl")
    t2 = yaml.safe_load((ROOT / "configs/authinv/t2.yaml").read_text())
    t2["analysis"]["bootstrap_replicates"] = 200
    row = board.finish(run, "org/model", "a" * 40, preds, t2, smoke=False)
    assert row["worst_rendering"] == "rego" and row["worst_case_accuracy"] == pytest.approx(0.5)
    assert row["deny_to_allow"]["rego"] == pytest.approx(1.0) and row["accuracy"]["table"] == 1.0
    text = board.table([row, {**row, "model": "smoke/model", "smoke": True}])
    assert "org/model@aaaaaaa" in text and "smoke/model" not in text and "| 50.0 | rego |" in text
    assert board.resolve_revision("org/model", "b" * 40) == "b" * 40
    assert json.loads((run / "run_status.json").read_text())["status"] == "complete"
