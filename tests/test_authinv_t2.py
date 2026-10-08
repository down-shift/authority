"""Tier T2: strict JSON parsing, instance-level invariance metrics, dissociation, runner validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from authinv import provenance
from authinv.eval.invariance import dissociation, rendering_metrics
from authinv.eval.structured import answer_schema, extract_object, parse_structured

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "task", "expected", "category"),
    [
        ('{"decision": "allow"}', "application", "allow", "correct"),
        ('```json\n{"decision": "deny"}\n```', "application", "allow", "incorrect"),
        ('<think>\nhmm\n</think>\n{"decision": "deny"}', "application", "deny", "correct"),
        ('{"decision": "allow", "why": "x"}', "application", "allow", "parse_failure"),
        ('{"decision": "maybe"}', "application", "allow", "parse_failure"),
        ("The request is allowed.", "application", "allow", "parse_failure"),
        ("I’m sorry, but I can’t help with that.", "application", "allow", "refusal"),
        ("It cannot be determined from the policy; neither applies.", "application", "allow", "abstain"),
        ('{"principals": ["u_b2", "u_d7"]}', "interpretation", ["u_d7", "u_b2"], "correct"),
        ('{"principals": ["u_b2"]}', "interpretation", ["u_d7", "u_b2"], "incorrect"),
        ('{"principals": []}', "interpretation", [], "correct"),
        ('{"principals": ["u_b2", "u_b2"]}', "interpretation", ["u_b2"], "parse_failure"),
    ],
)
def test_parse_structured(text, task, expected, category):
    assert parse_structured(text, task, expected)["category"] == category


def test_extract_object_is_strict():
    assert extract_object('Sure! {"decision": "allow"}') is None
    assert extract_object('[{"decision": "allow"}]') is None
    assert answer_schema("application")["properties"]["decision"]["enum"] == ["allow", "deny"]
    assert answer_schema("interpretation", ["b", "a"])["properties"]["principals"]["items"]["enum"] == [
        "a",
        "b",
    ]


def _rows(n_worlds=30, bad=("nl", 8), refusal=None):
    rows = []
    for w in range(n_worlds):
        for a in ("orig", "swap"):
            for i, label in enumerate(["allow", "deny", "allow", "deny"]):
                for g in ("cedar", "json", "nl"):
                    value = label
                    if g == bad[0] and w < bad[1] and label == "deny":
                        value = "allow"  # fail-open on deny requests
                    cat = "correct" if value == label else "incorrect"
                    if refusal and g == refusal and w == 0 and i == 0:
                        cat, value = "refusal", None
                    rows.append(
                        {
                            "world_id": f"w{w:02d}",
                            "assignment": a,
                            "instance": f"q{i}",
                            "rendering": g,
                            "task": "application",
                            "label": label,
                            "value": value,
                            "category": cat,
                            "pair_key": f"read|r{i % 2}",
                        }
                    )
                    if i < 2:
                        rows.append(
                            {
                                "world_id": f"w{w:02d}",
                                "assignment": a,
                                "instance": f"o{i}",
                                "rendering": g,
                                "task": "interpretation",
                                "value": ("u",),
                                "category": "correct",
                                "pair_key": f"read|r{i}",
                            }
                        )
    return rows


def test_rendering_metrics_direction_gap_and_disagreement():
    m = rendering_metrics(_rows(), "application", replicates=300, seed=1)
    assert m["worst_rendering"] == "nl" and m["worst_case_gap"]["mean"] == pytest.approx(8 / 30 * 0.5)
    nl = m["per_rendering"]["nl"]
    assert nl["deny_to_allow_rate"]["mean"] == pytest.approx(8 / 30) and nl["allow_to_deny_rate"]["mean"] == 0
    assert m["rendering_disagreement"]["mean"] == pytest.approx(8 / 30 * 0.5)
    assert m["deny_to_allow_instances"]["mean"] == pytest.approx(8 / 30)
    assert m["allow_to_deny_instances"]["mean"] == 0
    assert m["pairwise_flips"]["cedar->nl"]["deny_to_allow"] == 8 * 2 * 2


def test_refusals_are_their_own_category():
    m = rendering_metrics(_rows(bad=("nl", 0), refusal="json"), "application", replicates=100, seed=1)
    js = m["per_rendering"]["json"]
    assert js["refusal_rate"]["mean"] == pytest.approx(2 / 8 / 30)  # both assignments of w00
    assert js["allow_to_deny_rate"]["mean"] == 0  # a refusal is not a deny
    assert m["allow_to_deny_instances"]["mean"] == 0


def test_dissociation_pairs_interpretation_and_application():
    d = dissociation(_rows(), replicates=100, seed=1)
    assert d["nl"]["joint"]["mean"] == pytest.approx(8 / 30 * 0.5)
    assert d["cedar"]["joint"]["mean"] == 0 and d["nl"]["paired_rows"] == 30 * 2 * 4


def _runner():
    spec = importlib.util.spec_from_file_location("authinv_eval", ROOT / "scripts" / "authinv_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["authinv_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def _dataset(tmp_path, drop_one=False):
    rows = []
    for w in range(2):
        for g in ("cedar", "nl"):
            rows.append(
                {
                    "row_id": f"w{w}/{g}/q0",
                    "world_id": f"w{w}",
                    "assignment": "orig",
                    "instance": "q0",
                    "rendering": g,
                    "task": "application",
                    "prompt": "p",
                    "expected": "allow",
                    "label": "allow",
                    "pair_key": "read|r",
                }
            )
    if drop_one:
        rows = rows[:-1]
    d = tmp_path / "ds"
    d.mkdir()
    provenance.write_jsonl(d / "dataset.jsonl", rows)
    provenance.write_json(
        d / "dataset_manifest.json",
        {"sha256": provenance.sha256_file(d / "dataset.jsonl"), "rows": len(rows)},
    )
    return d


def test_runner_dataset_only_and_validation(tmp_path):
    runner = _runner()
    rows, report = runner.load_dataset(_dataset(tmp_path))
    assert report["passed"] and len(rows) == 4
    with pytest.raises(SystemExit, match="every rendering"):
        runner.load_dataset(
            _dataset(tmp_path / "b", drop_one=True) if (tmp_path / "b").mkdir() is None else None
        )
    sys.argv = [
        "x",
        "run",
        "--config",
        str(ROOT / "configs/authinv/t2.yaml"),
        "--dataset",
        str(_dataset(tmp_path / "c") if (tmp_path / "c").mkdir() is None else None),
        "--model",
        "qwen3_5_4b",
        "--dataset-only",
        "--output-root",
        str(tmp_path / "out"),
    ]
    runner.main()
    run = next((tmp_path / "out").iterdir())
    assert json.loads((run / "run_status.json").read_text())["status"] == "dataset_only"
    assert json.loads((run / "config.json").read_text())["parser_version"] == "json-strict-v1"


def test_runner_analyze_on_synthetic_predictions():
    runner = _runner()
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/authinv/t2.yaml").read_text())
    cfg["analysis"]["bootstrap_replicates"] = 100
    out = runner.analyze(_rows(), cfg)
    assert out["tasks"]["application"]["worst_rendering"] == "nl" and "dissociation" in out
