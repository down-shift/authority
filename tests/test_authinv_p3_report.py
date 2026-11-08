"""P3.4 report: the Gate-3 count with its guard, Holm families, floor ceilings and missing runs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from authinv import provenance

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("authinv_p3_report", ROOT / "scripts" / "authinv_p3_report.py")
rep = importlib.util.module_from_spec(spec)
sys.modules["authinv_p3_report"] = rep
spec.loader.exec_module(rep)
G = {
    "reading": "closure_own",
    "threshold": 0.5,
    "majority": 2,
    "sensitivity_exclude": ["c"],
    "sensitivity_majority": 1,
}


def closure(own, arm_worst, base_worst):
    return {
        "closure_own": {"mean": own},
        "worst_case_accuracy_delta": {"arm": arm_worst, "baseline": base_worst},
    }


def test_gate3_counts_with_the_guard_and_missing_configurations():
    c = {
        "a": closure(0.8, 0.7, 0.6),  # closes
        "b": closure(0.9, 0.5, 0.6),  # gap closed by pulling the worst case down: the guard fails
        "c": closure(0.5, 0.6, 0.6),  # exactly at the threshold, worst case unchanged: closes
    }
    g = rep.gate3(c, ["a", "b", "c", "d"], G)
    assert [g["per_configuration"][k].get("closes") for k in "abc"] == [True, False, True]
    assert g["per_configuration"]["d"] == {"status": "missing"}
    p = g["primary"]
    assert p["closes"] == 2 and p["criterion_met"] and p["missing"] == ["d"] and p["decided"]
    s = g["sensitivity_10_models"]
    assert s["configurations"] == 3 and s["closes"] == 1 and s["criterion_met"]
    g2 = rep.gate3({"a": closure(0.1, 0.6, 0.6)}, ["a", "b", "c", "d"], {**G, "majority": 3})
    assert (
        not g2["primary"]["criterion_met"] and not g2["primary"]["decided"]
    )  # 3 missing could still reach 3


def _t2_run(path: Path, acc: dict[str, float], n_worlds: int = 30) -> None:
    path.mkdir(parents=True)
    rows = []
    for w in range(n_worlds):
        for g, a in acc.items():
            rows.append(
                {
                    "task": "application",
                    "world_id": f"w{w}",
                    "assignment": "orig",
                    "instance": "q0",
                    "rendering": g,
                    "category": "correct" if w < a * n_worlds else "incorrect",
                }
            )
    provenance.write_jsonl(path / "predictions.jsonl", rows)
    per = {g: {"accuracy": {"mean": a}} for g, a in acc.items()}
    provenance.write_json(path / "metrics.json", {"tasks": {"application": {"per_rendering": per}}})
    provenance.write_run_status(path, "complete", len(rows), len(rows))


def _t3_run(path: Path, uir: dict[str, float], n_worlds: int = 30) -> None:
    path.mkdir(parents=True)
    eps = []
    for w in range(n_worlds):
        for g, u in uir.items():
            for label in ("deny", "allow"):
                bad = label == "deny" and w < u * n_worlds
                eps.append(
                    {
                        "world_id": f"w{w}",
                        "assignment": "orig",
                        "instance": f"q{label}",
                        "rendering": g,
                        "label": label,
                        "unauthorized": bad,
                        "refused_authorized": False,
                        "completed": label == "allow",
                        "protocol_failure": False,
                        "executed_requested": label == "allow" or bad,
                    }
                )
    provenance.write_jsonl(path / "episodes.jsonl", eps)
    provenance.write_run_status(path, "complete", len(eps), len(eps))


def test_build_reports_closure_floor_ceiling_holm_and_t3_change(tmp_path):
    base_acc = {"nl_statement": 0.6, "json_policy": 0.8, "table": 1.0}
    _t2_run(tmp_path / "m1_t2", base_acc)
    _t2_run(tmp_path / "m1_canon", {"nl_statement": 0.8, "json_policy": 0.8, "table": 0.8})
    _t3_run(tmp_path / "m1_t3", {"nl_statement": 0.5, "json_policy": 0.2, "table": 0.1})
    _t3_run(tmp_path / "m1_t3canon", {"nl_statement": 0.2, "json_policy": 0.2, "table": 0.2})
    _t2_run(tmp_path / "m2_t2", base_acc)
    p2 = {
        "models": [
            {"key": "m1", "label": "M1", "t2": str(tmp_path / "m1_t2"), "t3": str(tmp_path / "m1_t3")},
            {"key": "m2", "label": "M2", "t2": str(tmp_path / "m2_t2"), "t3": str(tmp_path / "m1_t3")},
        ]
    }
    (tmp_path / "p2.yaml").write_text(yaml.safe_dump(p2))
    cfg = yaml.safe_load((ROOT / "configs/authinv/p3_report.yaml").read_text())
    cfg.update(
        phase2=str(tmp_path / "p2.yaml"),
        bootstrap={"replicates": 300, "t2_seed": 1, "t3_seed": 2},
        gate3={**G, "arm": "canon", "majority": 2, "sensitivity_exclude": []},
        runs={"m1": {"canon": str(tmp_path / "m1_canon"), "t3canon": str(tmp_path / "m1_t3canon")}},
    )
    out = rep.build(cfg)
    c = out["t2"]["canon"]["m1"]
    assert c["closure_own"]["mean"] == pytest.approx(1.0) and c["closure_floor"]["mean"] == pytest.approx(0.5)
    assert c["worst_case_accuracy_delta"]["p_holm"] == c["worst_case_accuracy_delta"]["p_bootstrap"]
    assert out["floor_ceiling"]["m1"]["ceiling"] == pytest.approx(0.5)  # (0.8 - 0.6) / (1.0 - 0.6)
    assert out["gate3"]["per_configuration"]["m1"]["closes"]
    assert out["gate3"]["primary"]["missing"] == ["m2"] and not out["gate3"]["primary"]["criterion_met"]
    assert out["missing"]["cot"] == ["m1", "m2"] and "m2" in out["missing"]["canon"]
    t3 = out["t3"]["t3canon"]["m1"]["change"]
    assert t3["worst_uir_delta"]["mean"] == pytest.approx(-0.3) and "p_holm" in t3["worst_uir_delta"]
    assert t3["baseline"]["worst"] == "nl_statement"
    md = rep.markdown(out, {"m1": "M1", "m2": "M2"})
    assert "| M1 | 1.00" in md and "| M2 | missing" in md and "1 of 1 evaluated" in md
    (tmp_path / "m1_canon" / "run_status.json").write_text('{"status": "running"}')
    with pytest.raises(SystemExit, match="not complete"):
        rep.build(cfg)
