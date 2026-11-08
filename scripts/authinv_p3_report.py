#!/usr/bin/env python3
"""P3.4: the Phase-3 mitigation analysis over the sweep (docs/prereg/phase3.md), written before any result.

  python scripts/authinv_p3_report.py --config configs/authinv/p3_report.yaml --out FILE.json [--md FILE.md]

For every configuration and arm listed in the config, against the Phase-2 run of the same
configuration (configs/authinv/phase2_figures.yaml):
- T2 arms: gap closure (own / pair / floor) with CIs, the worst-case accuracy change with its
  two-sided paired bootstrap p, and per-rendering accuracy changes (`mitigate.metrics.gap_closure`);
- Gate 3 (decisions 1 and 2): arm 1 "closes >= 50%" on a configuration when its own-gap closure
  point estimate is >= 0.5 and its worst-case accuracy is not below the baseline's; the count is
  compared with the majority (>= 7 of 12), and the 10-model reading (>= 6 of 10) is reported;
- the floor-closure ceiling of a perfect conversion: (json_policy - worst) / (best - worst);
- Holm (alpha 0.05) over each T2 arm's worst-case accuracy changes, and over each P3.2b T3 arm's
  worst-case UIR changes (amendment 1);
- T3 arms: UIR, completion and their changes against the Phase-2 T3 run (`agent.metrics.uir_change`).
A configuration whose run is missing is listed as missing; the Gate-3 count says how many are.
The report is a recommendation; Jerzy decides G3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import provenance  # noqa: E402
from authinv.agent.metrics import agent_metrics, uir_change  # noqa: E402
from authinv.eval import metrics as emetrics  # noqa: E402
from authinv.mitigate.metrics import gap_closure  # noqa: E402

T2_FILE = "predictions.jsonl"
T3_FILE = "episodes.jsonl"


def _complete(run: Path) -> None:
    status = json.loads((run / "run_status.json").read_text())["status"]
    if status != "complete":
        raise SystemExit(f"{run}: status {status}, not complete")


def _rows(run: Path, name: str, task: str | None = None) -> list[dict]:
    _complete(run)
    rows = provenance.read_jsonl(run / name)
    return [r for r in rows if task is None or r.get("task") == task]


def floor_ceiling(per_rendering: dict) -> dict:
    acc = {g: v["accuracy"]["mean"] for g, v in per_rendering.items()}
    best, worst = max(acc.values()), min(acc.values())
    return {
        "json_policy": acc["json_policy"],
        "best": best,
        "worst": worst,
        "ceiling": (acc["json_policy"] - worst) / (best - worst) if best > worst else None,
    }


def gate3(closures: dict[str, dict], models: list[str], g: dict) -> dict:
    per = {}
    for key in models:
        c = closures.get(key)
        if c is None:
            per[key] = {"status": "missing"}
            continue
        own = c[g["reading"]]["mean"]
        guard = c["worst_case_accuracy_delta"]["arm"] >= c["worst_case_accuracy_delta"]["baseline"]
        per[key] = {
            "status": "ok",
            "closure": own,
            "guard": guard,
            "closes": own is not None and own >= g["threshold"] and guard,
        }

    def count(keys, majority):
        ok = [k for k in keys if per[k]["status"] == "ok"]
        n = sum(per[k]["closes"] for k in ok)
        return {
            "configurations": len(keys),
            "evaluated": len(ok),
            "missing": [k for k in keys if per[k]["status"] == "missing"],
            "closes": n,
            "majority": majority,
            "criterion_met": n >= majority,
            "decided": n >= majority or n + (len(keys) - len(ok)) < majority,
        }

    sens = [k for k in models if k not in g["sensitivity_exclude"]]
    return {
        "per_configuration": per,
        "primary": count(models, g["majority"]),
        "sensitivity_10_models": count(sens, g["sensitivity_majority"]),
    }


def build(cfg: dict) -> dict:
    p2 = yaml.safe_load((ROOT / cfg["phase2"]).read_text())
    models = [m["key"] for m in p2["models"]]
    base = {m["key"]: m for m in p2["models"]}
    b = cfg["bootstrap"]
    task = cfg["task"]
    out = {"t2": {}, "t3": {}, "floor_ceiling": {}, "missing": {}}
    for key in models:
        t2_metrics = json.loads((ROOT / base[key]["t2"] / "metrics.json").read_text())
        out["floor_ceiling"][key] = floor_ceiling(t2_metrics["tasks"][task]["per_rendering"])
    for arm in cfg["t2_arms"]:
        res = {}
        for key in models:
            run = (cfg.get("runs") or {}).get(key, {}).get(arm)
            if not run:
                out["missing"].setdefault(arm, []).append(key)
                continue
            base_rows = _rows(ROOT / base[key]["t2"], T2_FILE, task)
            arm_rows = _rows(ROOT / run, T2_FILE, task)
            res[key] = {"run": run, **gap_closure(arm_rows, base_rows, task, b["replicates"], b["t2_seed"])}
        holm = emetrics.holm({k: v["worst_case_accuracy_delta"]["p_bootstrap"] for k, v in res.items()})
        for k, v in res.items():
            v["worst_case_accuracy_delta"]["p_holm"] = holm[k]
        out["t2"][arm] = res
    out["gate3"] = gate3(out["t2"].get(cfg["gate3"]["arm"], {}), models, cfg["gate3"])
    for arm in cfg["t3_arms"]:
        res = {}
        for key in models:
            run = (cfg.get("runs") or {}).get(key, {}).get(arm)
            if not run:
                out["missing"].setdefault(arm, []).append(key)
                continue
            base_eps = _rows(ROOT / base[key]["t3"], T3_FILE)
            arm_eps = _rows(ROOT / run, T3_FILE)
            m = agent_metrics(arm_eps, b["replicates"], b["t3_seed"])
            res[key] = {
                "run": run,
                "per_rendering": m["per_rendering"],
                "worst_case_uir": m["worst_case_uir"],
                "change": uir_change(arm_eps, base_eps, b["replicates"], b["t3_seed"]),
            }
        if arm in cfg["holm_t3_arms"]:
            holm = emetrics.holm({k: v["change"]["worst_uir_delta"]["p_bootstrap"] for k, v in res.items()})
            for k, v in res.items():
                v["change"]["worst_uir_delta"]["p_holm"] = holm[k]
        out["t3"][arm] = res
    return out


def markdown(rep: dict, labels: dict[str, str]) -> str:
    g = rep["gate3"]
    lines = [
        "| configuration | own closure | guard | closes ≥ 50% | pair | floor | floor ceiling "
        "| worst Δ (pp) | Holm p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    canon = rep["t2"].get("canon", {})
    for key, label in labels.items():
        c = canon.get(key)
        fc = rep["floor_ceiling"][key]["ceiling"]
        if c is None:
            lines.append(f"| {label} | missing | | | | | {fc:.2f} | | |")
            continue

        def ci(x):
            lo, hi = x["ci95"]
            return f"{x['mean']:.2f} ({lo:.2f}–{hi:.2f})" if x["mean"] is not None and lo is not None else "—"

        d = c["worst_case_accuracy_delta"]
        p = g["per_configuration"][key]
        lines.append(
            f"| {label} | {ci(c['closure_own'])} | {'yes' if p['guard'] else 'no'} | "
            f"{'yes' if p['closes'] else 'no'} | {ci(c['closure_pair'])} | {ci(c['closure_floor'])} | "
            f"{fc:.2f} | "
            f"{100 * d['mean']:+.1f} | {d['p_holm']:.3g} |"
        )
    pr = g["primary"]
    lines.append("")
    lines.append(
        f"Gate 3 (decisions 1, 2): {pr['closes']} of {pr['evaluated']} evaluated configurations close "
        f"≥ 50% (majority {pr['majority']} of {pr['configurations']}; "
        f"missing: {', '.join(pr['missing']) or 'none'})."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rep = build(cfg)
    provenance.write_json(args.out, {"config": cfg, "config_sha256": provenance.sha256_json(cfg), **rep})
    if args.md:
        p2 = yaml.safe_load((ROOT / cfg["phase2"]).read_text())
        args.md.write_text(markdown(rep, {m["key"]: m["label"] for m in p2["models"]}))
    print(json.dumps(rep["gate3"], indent=2))


if __name__ == "__main__":
    main()
