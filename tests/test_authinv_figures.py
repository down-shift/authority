"""P2.9 figures and tables (scripts/authinv_figures.py): run-set checks, cross-model statistics, outputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from authinv import models, provenance

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("figures", ROOT / "scripts" / "authinv_figures.py")
fig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig)

REND = fig.REND
DATASET = "d" * 64


def ci(mean: float, p: float | None = None) -> dict:
    out = {"mean": mean, "ci95": [mean - 0.01, mean + 0.01], "worlds": 10}
    if p is not None:
        out["p_bootstrap"] = p
    return out


def app_metrics(acc: dict[str, float]) -> dict:
    worst, best = min(acc, key=acc.get), max(acc, key=acc.get)
    return {
        "per_rendering": {
            g: {
                "accuracy": ci(a),
                "deny_to_allow_rate": ci((1 - a) / 2),
                "allow_to_deny_rate": ci((1 - a) / 2),
                "refusal_rate": ci(0.0),
                "abstain_rate": ci(0.0),
                "parse_failure_rate": ci(0.01),
            }
            for g, a in acc.items()
        },
        "worst_rendering": worst,
        "best_rendering": best,
        "worst_case_accuracy": acc[worst],
        "worst_case_gap": ci(acc[best] - acc[worst], 0.0),
        "rendering_disagreement": ci(0.2),
        "all_renderings_correct": ci(0.6),
        "deny_to_allow_instances": ci(0.1),
        "allow_to_deny_instances": ci(0.1),
        "worlds": 10,
    }


def contrasts(acc: dict[str, float], p: float) -> dict:
    return {g: ci(a - (sum(acc.values()) - a) / 5, p) for g, a in acc.items()}


def t2_metrics(acc: dict[str, float], p: float) -> dict:
    app = app_metrics(acc)
    return {
        "tasks": {"application": app, "interpretation": app},
        "dissociation": {g: {"app_wrong_given_interp_correct": ci(0.05), "joint": ci(0.03)} for g in REND},
        "rendering_contrasts": contrasts(acc, p),
        "by_source_kind": {"real": app, "synthetic": app},
    }


def t1_metrics(acc: dict[str, float], p: float) -> dict:
    return {
        "application": app_metrics(acc),
        "rendering_contrasts": contrasts(acc, p),
        "mean_margin": {g: 5.0 for g in REND},
    }


def t3_metrics(uir: dict[str, float], p: float) -> dict:
    worst, best = max(uir, key=uir.get), min(uir, key=uir.get)
    rate = {
        g: {
            "uir_deny": ci(u),
            "completion": ci(0.9),
            "authorized_refusal": ci(0.05),
            "protocol_failure": ci(0.01),
        }
        for g, u in uir.items()
    }
    out = {
        "per_rendering": rate,
        "worst_uir_rendering": worst,
        "best_uir_rendering": best,
        "worst_case_uir": uir[worst],
        "uir_gap": ci(uir[worst] - uir[best], 0.0),
        "action_disagreement": ci(0.1),
        "uir_contrasts": {g: ci(u - (sum(uir.values()) - u) / 5, p) for g, u in uir.items()},
        "worlds": 10,
    }
    out["by_source_kind"] = {"real": dict(out), "synthetic": dict(out)}
    return out


T2_HASH = provenance.sha256_json(yaml.safe_load((ROOT / "configs/authinv/t2.yaml").read_text()))
ROWS = {"t2": 172296, "t1": 95232, "t3": 95232}


def make_run(root: Path, tier: str, key: str, metrics: dict, **override) -> Path:
    run = root / f"{tier}_{key}"
    run.mkdir()
    spec_ = models.model_spec(key)
    conf = {
        "dataset_sha256": DATASET,
        "config_sha256": T2_HASH,
        "model": {"key": key, "revision": spec_["revision"], "params_b": spec_["params_b"]},
    }
    if tier == "t3":
        conf = {"episodes": {"dataset_sha256": DATASET}, "model": conf["model"]}
    conf.update(override.get("config", {}))
    files = {
        "config.json": conf,
        "metadata.json": {
            "git": {"commit": "abc"},
            "runtime": {"vllm_version": "0.29.0", "gpu_type": ["H100"]},
        },
        "run_status.json": {
            "status": "complete",
            "completed_examples": ROWS[tier],
            "total_examples": ROWS[tier],
            **override.get("status", {}),
        },
        "metrics.json": metrics,
    }
    for name, value in files.items():
        (run / name).write_text(json.dumps(value))
    manifest = {name: provenance.sha256_file(run / name) for name in files}
    (run / "artifact_sha256.json").write_text(json.dumps(manifest))
    return run


def make_config(tmp_path: Path) -> dict:
    """Three configurations; bf16 and FP8 share a parameter count, as in the sweep."""
    base = {"nl_statement": 0.80, "owner_statement": 0.90, "table": 0.88, "json_policy": 0.86}
    accs = {
        "qwen3_5_4b": {**base, "executable": 0.85, "rego": 0.70},
        "qwen3_8_27b": {**base, "executable": 0.95, "rego": 0.50},
        "qwen3_8_27b_fp8": {**base, "executable": 0.94, "rego": 0.75},
    }
    entries = []
    for key, acc in accs.items():
        uir = {g: 1 - a for g, a in acc.items()}
        entries.append(
            {
                "key": key,
                "label": key,
                "t2": str(make_run(tmp_path, "t2", key, t2_metrics(acc, 0.0))),
                "t1": str(make_run(tmp_path, "t1", key, t1_metrics(acc, 0.2))),
                "t3": str(make_run(tmp_path, "t3", key, t3_metrics(uir, 0.01))),
            }
        )
    return {
        "dataset_sha256": DATASET,
        "t2_config": "configs/authinv/t2.yaml",
        "seeds": {"t1": 1, "t2": 1, "t3": 2},
        "permutations": 200,
        "alpha": 0.05,
        "expected_rows": ROWS,
        "quantization_pair": ["qwen3_8_27b", "qwen3_8_27b_fp8"],
        "scaling_descriptive_exclude": ["qwen3_8_27b_fp8"],
        "models": entries,
    }


def test_run_check_reports_incomplete_runs_and_unpinned_revisions(tmp_path):
    acc = dict.fromkeys(REND, 0.9)
    good = make_run(tmp_path, "t2", "qwen3_5_4b", t2_metrics(acc, 0.0))
    cfg = {"dataset_sha256": DATASET, "expected_rows": ROWS}
    assert fig.check_run(good, "t2", "qwen3_5_4b", cfg, T2_HASH, verify=True)["problems"] == []
    (good / "metrics.json").write_text("{}")
    assert (
        "artifact sha256 mismatch"
        in fig.check_run(good, "t2", "qwen3_5_4b", cfg, T2_HASH, True)["problems"][0]
    )
    part = make_run(
        tmp_path,
        "t1",
        "qwen3_5_4b",
        t1_metrics(acc, 0.0),
        status={"status": "running", "completed_examples": 10},
        config={"model": {"key": "qwen3_5_4b", "revision": "main", "params_b": 4.7}},
    )
    problems = fig.check_run(part, "t1", "qwen3_5_4b", cfg, T2_HASH, verify=False)["problems"]
    assert any(p.startswith("status running") for p in problems)
    assert any("registry pin" in p for p in problems)


def test_load_runs_refuses_a_run_set_with_a_failing_run(tmp_path):
    cfg = make_config(tmp_path)
    (Path(cfg["models"][0]["t2"]) / "config.json").write_text(
        json.dumps(
            {"dataset_sha256": "x", "config_sha256": T2_HASH, "model": models.model_spec("qwen3_5_4b")}
        )
    )
    with pytest.raises(SystemExit, match="run-set check failed"):
        fig.load_runs(cfg, verify=False)


def test_distil_holm_families_scaling_and_quantization(tmp_path):
    cfg = make_config(tmp_path)
    metrics, checks = fig.load_runs(cfg, verify=True)
    s = fig.distil(cfg, metrics, checks)
    # one Holm family per tier: models × renderings
    assert s["holm"]["t2"]["size"] == s["holm"]["t3"]["size"] == 18
    assert s["holm"]["t2"]["significant"] == 18  # p = 0 survives any Holm step
    assert all(abs(p - min(1.0, 18 * 0.01)) < 1e-12 for p in s["holm"]["t3"]["adjusted"].values())
    # scaling: three configurations, two of them tied at 27.8B; the descriptive set drops FP8
    assert s["scaling"]["t2"]["n"] == 3 and s["scaling"]["t2_without_variants"]["n"] == 2
    assert s["scaling"]["t2_without_variants"]["rho"] is None
    # quantization: FP8 − bf16 per rendering, compared with the bf16 gap (0.95 − 0.50)
    q = s["quantization"]["t2"]
    assert q["delta"]["rego"] == pytest.approx(0.25) and q["max_abs_delta"] == pytest.approx(0.25)
    assert q["base_gap"] == pytest.approx(0.45) and not q["exceeds_rendering_effect"]
    assert s["models"]["qwen3_8_27b"]["t2"]["worst"] == "rego"
    assert s["patterns"]["worst_rendering"]["qwen3_8_27b"] == {"t1": "rego", "t2": "rego", "t3": "rego"}


def test_tables_escape_for_latex_and_markdown(tmp_path):
    cfg = make_config(tmp_path)
    s = fig.distil(cfg, *fig.load_runs(cfg, verify=False))
    tabs = {t.name: t for t in fig.tables(s)}
    assert {"t2_accuracy", "t1_accuracy", "t3_uir", "direction", "dissociation", "scaling"} <= set(tabs)
    tex = tabs["t2_accuracy"].latex()
    assert "$\\rightarrow$" in tex and "→" not in tex and "\\label{tab:t2_accuracy}" in tex
    assert "\\|Δ\\|" in tabs["quantization"].markdown()  # a literal | must not split a Markdown cell
    assert "—" in tabs["scaling"].markdown()  # undefined rho (n < 3) is printed, not crashed on
    assert fig.tex("a_b 5% x|y") == "a\\_b 5\\% x$\\mid$y"


def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    cfg = make_config(tmp_path)
    s = fig.distil(cfg, *fig.load_runs(cfg, verify=False))
    out = tmp_path / "figs"
    out.mkdir()
    written = fig.figures(s, out, preview=None)
    assert len(written) == 7
    assert all((out / name).stat().st_size > 1000 for name in written)
