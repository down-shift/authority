"""Phase-0 harness: strict parser, invariance metrics, Gate-0 logic, dataset validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from authinv import provenance
from authinv.eval import metrics
from authinv.eval.parse import parse_answer

ROOT = Path(__file__).resolve().parents[1]
CANDS = ["Agent F04", "Agent F05"]


@pytest.mark.parametrize(
    ("text", "category", "selected"),
    [
        ("Agent F05", "correct", "Agent F05"),
        ("  `Agent F05`.\n", "correct", "Agent F05"),
        ("<think>\n\n</think>\n\nAgent F05", "correct", "Agent F05"),
        ('"Agent F04"', "incorrect", "Agent F04"),
        ("Agent F05 owns the filename.", "parse_failure", None),
        ("I cannot determine which actor owns it.", "refusal", None),
        ("Neither actor.", "abstain", None),
        ("", "parse_failure", None),
    ],
)
def test_strict_parser(text, category, selected):
    out = parse_answer(text, CANDS, "Agent F05")
    assert (out["category"], out["selected"]) == (category, selected)


def test_lenient_selection_is_secondary_only():
    out = parse_answer("The owner is Agent F05.", CANDS, "Agent F05")
    assert out["category"] == "parse_failure" and out["lenient_selected"] == "Agent F05"
    files = ["filename_1.txt", "filename_12.txt"]
    assert (
        parse_answer("Use filename_12.txt", files, "filename_1.txt")["lenient_selected"] == "filename_12.txt"
    )


def _rows(n_worlds=40, bad_rep="natural_language", bad_worlds=10, closed_rep=None):
    rows = []
    for w in range(n_worlds):
        for a in ("original", "swapped"):
            for rep in ("json", "natural_language", "table"):
                cat = "correct"
                if rep == bad_rep and w < bad_worlds:
                    cat = "incorrect"
                if rep == closed_rep and w == n_worlds - 1:
                    cat = "abstain"
                rows.append(
                    {
                        "world_id": f"w{w:02d}",
                        "assignment": a,
                        "representation": rep,
                        "task": "application",
                        "category": cat,
                        "selected": {"correct": "good", "incorrect": "bad"}.get(cat),
                    }
                )
    return rows


def test_task_metrics_gap_disagreement_and_flips():
    m = metrics.task_metrics(_rows(closed_rep="table"), "application", replicates=500, seed=1)
    assert m["worst_representation"] == "natural_language"
    assert m["worst_case_accuracy"] == pytest.approx(0.75)
    assert m["worst_case_gap"]["mean"] == pytest.approx(0.25)
    assert m["worst_case_gap"]["ci95"][0] > 0
    assert m["rendering_disagreement"]["mean"] == pytest.approx(11 / 40)
    assert m["deny_to_allow_count"] == 20 and m["allow_to_deny_count"] == 2
    assert m["pairwise_flips"]["json->natural_language"]["deny_to_allow"] == 20
    assert m["per_representation"]["table"]["fail_closed_rate"]["mean"] == pytest.approx(1 / 40)


def test_no_effect_gives_no_gap():
    m = metrics.task_metrics(_rows(bad_worlds=0), "application", replicates=200, seed=1)
    assert m["worst_case_gap"]["mean"] == 0 and m["rendering_disagreement"]["mean"] == 0


def test_holm():
    adj = metrics.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


def test_gate0_needs_two_large_models():
    hit = {"worst_case_gap": {"mean": 0.08, "ci95": [0.02, 0.12]}, "rendering_disagreement": {"mean": 0.01}}
    dis = {"worst_case_gap": {"mean": 0.01, "ci95": [-0.01, 0.03]}, "rendering_disagreement": {"mean": 0.06}}
    miss = {"worst_case_gap": {"mean": 0.08, "ci95": [-0.01, 0.12]}, "rendering_disagreement": {"mean": 0.01}}
    large = ["a", "b", "c"]
    assert metrics.gate0({"a": hit, "b": dis, "c": miss}, large)["recommendation"] == "PASS"
    assert metrics.gate0({"a": hit, "b": miss, "c": miss}, large)["recommendation"] == "FAIL"
    assert metrics.gate0({"a": hit}, large)["recommendation"] == "INCOMPLETE"


def _load_runner():
    spec = importlib.util.spec_from_file_location("authinv_phase0", ROOT / "scripts" / "authinv_phase0.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["authinv_phase0"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    worlds, rows = [], []
    for w in range(2):
        worlds.append({"world_id": f"w{w}"})
        for a in ("original", "swapped"):
            for rep in ("json", "table"):
                for task in ("application", "interpretation"):
                    rows.append(
                        {
                            "row_id": f"w{w}/{a}/{rep}/{task}",
                            "world_id": f"w{w}",
                            "assignment": a,
                            "representation": rep,
                            "task": task,
                            "prompt": "p",
                            "candidates": ["x", "y"],
                            "correct": "x",
                            "incorrect": "y",
                        }
                    )
    provenance.write_jsonl(src / "worlds.jsonl", worlds)
    provenance.write_jsonl(src / "dataset.jsonl", rows)
    provenance.write_json(src / "validation_report.json", {"passed": True})
    cfg = {
        "source": {
            "run_name": "fake",
            "worlds_sha256": provenance.sha256_file(src / "worlds.jsonl"),
            "dataset_sha256": provenance.sha256_file(src / "dataset.jsonl"),
            "expected": {"worlds": 2, "rows": 16, "assignments": 2, "representations": 2, "tasks": 2},
        }
    }
    return src, cfg


def test_build_dataset_validates_and_checks_hashes(tmp_path):
    runner = _load_runner()
    src, cfg = _fake_source(tmp_path)
    rows, report = runner.build_dataset(cfg, src)
    assert report["passed"] and len(rows) == 16 and set(rows[0]) == set(runner.KEEP)
    cfg["source"]["dataset_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="pinned hashes"):
        runner.build_dataset(cfg, src)


def test_phase0_config_matches_registry():
    import yaml

    from authinv import models

    cfg = yaml.safe_load((ROOT / "configs" / "authinv" / "phase0.yaml").read_text())
    reg = models.load_registry()
    assert set(cfg["large_models"]) <= set(reg["sets"][cfg["models_set"]])
    assert cfg["generation"]["temperature"] == 0.0
    assert json.dumps(cfg["gate0"]) and cfg["gate0"]["min_large_models_passing"] == 2


def test_read_head_without_git_binary(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text("# pack-refs\n" + "a" * 40 + " refs/heads/main\n")
    assert provenance._read_head(tmp_path) == "a" * 40
    (git / "refs" / "heads" / "main").write_text("b" * 40 + "\n")
    assert provenance._read_head(tmp_path) == "b" * 40


def test_aggregate_markdown_and_holm_family(tmp_path):
    import yaml

    runner = _load_runner()
    cfg_path = ROOT / "configs" / "authinv" / "phase0.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["analysis"]["bootstrap_replicates"] = 200
    small_cfg = tmp_path / "phase0.yaml"
    small_cfg.write_text(yaml.safe_dump(cfg))
    runs = []
    for key, bad in (("qwen3_8b", 0), ("qwen3_32b", 12), ("gemma3_27b", 0)):
        rows = [dict(r, lenient_selected=None) for r in _rows(bad_worlds=bad)]
        rows += [dict(r, task="interpretation") for r in rows]
        run = tmp_path / key
        run.mkdir()
        provenance.write_json(
            run / "config.json", {"config_sha256": provenance.sha256_json(cfg), "model": {"key": key}}
        )
        provenance.write_json(run / "run_status.json", {"status": "complete"})
        provenance.write_json(run / "metrics.json", runner.analyze(rows, cfg))
        runs.append(run)
    md = tmp_path / "agg.md"
    sys.argv = [
        "x",
        "aggregate",
        "--config",
        str(small_cfg),
        "--runs",
        *map(str, runs),
        "--markdown",
        str(md),
        "--out",
        str(tmp_path / "agg.json"),
    ]
    runner.main()
    result = json.loads((tmp_path / "agg.json").read_text())
    assert set(result["holm_adjusted_gap_p"]) == {"qwen3_32b", "gemma3_27b"}
    assert result["gate0"]["recommendation"] == "INCOMPLETE"
    assert "| qwen3_32b |" in md.read_text() and "Recommendation: INCOMPLETE" in md.read_text()


def test_split_harmony_takes_final_channel():
    from authinv.eval.generation import split_harmony

    raw = (
        "<|channel|>analysis<|message|>The listed owner is Agent F05.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Agent F05<|return|>"
    )
    out = split_harmony(raw)
    assert out["text"] == "Agent F05" and out["harmony_final_found"]
    assert out["reasoning"] == "The listed owner is Agent F05." and out["raw_text"] == raw
    truncated = split_harmony("<|channel|>analysis<|message|>Thinking about it")
    assert truncated["text"] == "" and not truncated["harmony_final_found"]
    assert parse_answer(truncated["text"], CANDS, "Agent F05")["category"] == "parse_failure"


def test_addendum_config_only_differs_where_intended():
    import yaml

    from authinv import models

    base = yaml.safe_load((ROOT / "configs" / "authinv" / "phase0.yaml").read_text())
    add = yaml.safe_load((ROOT / "configs" / "authinv" / "phase0_addendum.yaml").read_text())
    assert "gate0" not in add and add["source"] == base["source"] and add["analysis"] == base["analysis"]
    assert add["generation"] == base["generation"] and add["parser_version"] == base["parser_version"]
    reg = models.load_registry()
    assert set(add["large_models"]) == set(reg["sets"][add["models_set"]])
    assert set(add["generation_overrides"]) <= set(reg["sets"][add["models_set"]])
