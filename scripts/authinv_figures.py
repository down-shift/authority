#!/usr/bin/env python3
"""Phase-2 headline tables and figures (P2.9; RESEARCH_PLAN §3 Phase 2, Gate 2).

  python scripts/authinv_figures.py --config configs/authinv/phase2_figures.yaml
        [--out paper/figures] [--verify-artifacts] [--preview DIR]

Reads the frozen run set named in the config (T2, T1 and T3 run directories under outputs/) and
writes into --out (default paper/figures):
  fig_*.pdf             figures
  tab_*.tex             LaTeX tables (booktabs)
  phase2_tables.md      the same tables in Markdown (docs/experiments/p2.9-phase2-report.md)
  phase2_summary.json   the distilled numbers and the run-set check
--preview DIR also writes each figure as a PNG there (for looking at, not for the paper).

Every per-model number is read from the metrics.json written by the runner that produced the run
(world-clustered bootstrap CIs, docs/prereg/phase2.md and phase2-t3.md). Across models this script
computes only what the preregistrations define over the sweep: the two Holm families (72 T2 rendering
contrasts; 72 T3 UIR contrasts), the scaling Spearman tests and the quantization control. Before drawing
it checks every run: complete, the frozen dataset, the registry's pinned revision, the current T2 config
hash for T2 runs, and with --verify-artifacts every file of the run's SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import models, provenance  # noqa: E402
from authinv.eval.invariance import spearman_with_permutation  # noqa: E402
from authinv.eval.metrics import holm  # noqa: E402

REND = ["nl_statement", "owner_statement", "table", "json_policy", "executable", "rego"]
RN = {
    "nl_statement": "NL",
    "owner_statement": "owner",
    "table": "table",
    "json_policy": "JSON",
    "executable": "Cedar",
    "rego": "Rego",
}
TIER_NAME = {"t1": "T1 accuracy", "t2": "T2 accuracy", "t3": "T3 UIR on deny"}


# --------------------------------------------------------------------------- run set


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check_run(run: Path, tier: str, key: str, cfg: dict, t2_hash: str, verify: bool) -> dict:
    """Completeness and provenance of one run; `problems` is empty when the run may be used."""
    conf, meta = read_json(run / "config.json"), read_json(run / "metadata.json")
    status = read_json(run / "run_status.json")
    problems = []
    if status["status"] != "complete" or status["completed_examples"] != status["total_examples"]:
        problems.append(
            f"status {status['status']} {status['completed_examples']}/{status['total_examples']}"
        )
    if status["total_examples"] != cfg["expected_rows"][tier]:
        problems.append(f"{status['total_examples']} rows, expected {cfg['expected_rows'][tier]}")
    dataset = conf.get("dataset_sha256") or conf.get("episodes", {}).get("dataset_sha256")
    if dataset != cfg["dataset_sha256"]:
        problems.append(f"dataset {dataset}")
    if conf["model"]["key"] != key:
        problems.append(f"model {conf['model']['key']}")
    if conf["model"]["revision"] != models.model_spec(key)["revision"]:
        problems.append(f"revision {conf['model']['revision']} is not the registry pin")
    if tier == "t2" and conf["config_sha256"] != t2_hash:
        problems.append("T2 config hash differs from configs/authinv/t2.yaml")
    artifacts = None
    if verify:
        manifest = read_json(run / "artifact_sha256.json")
        bad = sorted(f for f, h in manifest.items() if provenance.sha256_file(run / f) != h)
        artifacts = {"files": len(manifest), "mismatched": bad}
        if bad:
            problems.append(f"artifact sha256 mismatch: {bad}")
    rescore = run / "rescore.json"
    runtime = meta.get("runtime", {})
    return {
        "run": run.name,
        "path": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "tier": tier,
        "model_key": key,
        "revision": conf["model"]["revision"],
        "params_b": conf["model"]["params_b"],
        "rows": status["total_examples"],
        "git": meta["git"]["commit"],
        "rescore_git": read_json(rescore)["git"]["commit"] if rescore.exists() else None,
        "agent_version": meta.get("agent_version"),
        "vllm": runtime.get("vllm_version"),
        "gpu": runtime.get("gpu_type"),
        "artifacts": artifacts,
        "problems": problems,
    }


def load_runs(cfg: dict, verify: bool) -> tuple[dict, list[dict]]:
    """metrics.json per (model, tier), plus the per-run checks. Exits if any run fails a check."""
    t2_hash = provenance.sha256_json(yaml.safe_load((ROOT / cfg["t2_config"]).read_text()))
    metrics, checks = {}, []
    for m in cfg["models"]:
        metrics[m["key"]] = {}
        for tier in ("t2", "t1", "t3"):
            if tier not in m:
                continue
            run = ROOT / m[tier]
            checks.append(check_run(run, tier, m["key"], cfg, t2_hash, verify))
            metrics[m["key"]][tier] = read_json(run / "metrics.json")
    bad = [c for c in checks if c["problems"]]
    if bad:
        raise SystemExit("run-set check failed:\n" + "\n".join(f"{c['path']}: {c['problems']}" for c in bad))
    return metrics, checks


# --------------------------------------------------------------------------- distillation


def _ci(v: dict) -> dict:
    return {k: v[k] for k in ("mean", "ci95", "p_bootstrap") if k in v}


def accuracy_tier(app: dict) -> dict:
    """T1 / T2 application metrics (invariance.rendering_metrics output)."""
    pr = app["per_rendering"]

    def non_answer(g: str) -> float | None:
        rates = [
            pr[g][k]["mean"] for k in ("refusal_rate", "abstain_rate", "parse_failure_rate") if k in pr[g]
        ]
        return sum(rates) if rates else None

    return {
        "rate": {g: _ci(pr[g]["accuracy"]) for g in REND},
        "worst": app["worst_rendering"],
        "best": app["best_rendering"],
        "worst_case": app["worst_case_accuracy"],
        "gap": _ci(app["worst_case_gap"]),
        "gap_reselected": _ci(app["worst_case_gap_reselected"])
        if "worst_case_gap_reselected" in app
        else None,
        "disagreement": _ci(app["rendering_disagreement"]),
        "all_correct": _ci(app["all_renderings_correct"]),
        "deny_to_allow": {g: _ci(pr[g]["deny_to_allow_rate"]) for g in REND},
        "allow_to_deny": {g: _ci(pr[g]["allow_to_deny_rate"]) for g in REND},
        "deny_to_allow_instances": _ci(app["deny_to_allow_instances"]),
        "allow_to_deny_instances": _ci(app["allow_to_deny_instances"]),
        "non_answer": {g: non_answer(g) for g in REND},
    }


def by_source(by_kind: dict, gap_key: str, worst_key: str, best_key: str) -> dict:
    return {
        kind: {"gap": _ci(v[gap_key]), "worst": v[worst_key], "best": v[best_key], "worlds": v["worlds"]}
        for kind, v in sorted(by_kind.items())
    }


def distil_t2(m: dict) -> dict:
    out = accuracy_tier(m["tasks"]["application"])
    interp = m["tasks"]["interpretation"]
    out["interpretation"] = {
        "rate": {g: _ci(interp["per_rendering"][g]["accuracy"]) for g in REND},
        "worst": interp["worst_rendering"],
        "best": interp["best_rendering"],
        "gap": _ci(interp["worst_case_gap"]),
    }
    out["dissociation"] = {
        g: {
            "app_wrong_given_interp_correct": _ci(m["dissociation"][g]["app_wrong_given_interp_correct"]),
            "joint": _ci(m["dissociation"][g]["joint"]),
        }
        for g in REND
    }
    out["contrasts"] = {g: _ci(m["rendering_contrasts"][g]) for g in REND}
    out["by_source_kind"] = by_source(
        m["by_source_kind"], "worst_case_gap", "worst_rendering", "best_rendering"
    )
    return out


def distil_t1(m: dict) -> dict:
    out = accuracy_tier(m["application"])
    out["contrasts"] = {g: _ci(m["rendering_contrasts"][g]) for g in REND}
    out["mean_margin"] = {g: m["mean_margin"][g] for g in REND}
    return out


def distil_t3(m: dict) -> dict:
    pr = m["per_rendering"]
    return {
        "rate": {g: _ci(pr[g]["uir_deny"]) for g in REND},
        "worst": m["worst_uir_rendering"],
        "best": m["best_uir_rendering"],
        "worst_case": m["worst_case_uir"],
        "gap": _ci(m["uir_gap"]),
        "disagreement": _ci(m["action_disagreement"]),
        "completion": {g: pr[g]["completion"]["mean"] for g in REND},
        "authorized_refusal": {g: pr[g]["authorized_refusal"]["mean"] for g in REND},
        "protocol_failure": {g: pr[g]["protocol_failure"]["mean"] for g in REND},
        "contrasts": {g: _ci(m["uir_contrasts"][g]) for g in REND},
        "by_source_kind": by_source(
            m["by_source_kind"], "uir_gap", "worst_uir_rendering", "best_uir_rendering"
        ),
    }


def holm_family(per_model: dict, tier: str, alpha: float) -> dict:
    """Holm adjustment over every (model, rendering) contrast of one tier (a preregistered family)."""
    family = {
        f"{k}|{g}": d[tier]["contrasts"][g]["p_bootstrap"]
        for k, d in per_model.items()
        if d.get(tier)
        for g in REND
    }
    adjusted = holm(family)
    return {
        "size": len(family),
        "alpha": alpha,
        "adjusted": adjusted,
        "significant": sum(p < alpha for p in adjusted.values()),
    }


def scaling(per_model: dict, keys: list[str], tier: str, cfg: dict) -> dict:
    keys = [k for k in keys if per_model[k].get(tier)]
    params = [per_model[k]["params_b"] for k in keys]
    gaps = [per_model[k][tier]["gap"]["mean"] for k in keys]
    rho = spearman_with_permutation(params, gaps, cfg["permutations"], cfg["seeds"][tier])
    return {**rho, "models": keys}


def quantization(per_model: dict, base: str, quant: str) -> dict:
    """Per-rendering quantized − base difference of each tier's primary rate (docs/prereg/phase2.md)."""
    out = {}
    for tier in ("t2", "t1", "t3"):
        b, q = per_model[base].get(tier), per_model[quant].get(tier)
        if not (b and q):
            continue
        delta = {g: q["rate"][g]["mean"] - b["rate"][g]["mean"] for g in REND}
        biggest = max(abs(v) for v in delta.values())
        out[tier] = {
            "delta": delta,
            "max_abs_delta": biggest,
            "base_gap": b["gap"]["mean"],
            "exceeds_rendering_effect": biggest >= b["gap"]["mean"],
        }
    return out


def patterns(per_model: dict, holm_t2: dict, holm_t3: dict, alpha: float) -> dict:
    """Counts the reports quote: where NL sits, and whether the tiers agree on the worst rendering."""

    def sig(fam: dict, k: str, g: str) -> bool:
        return fam["adjusted"].get(f"{k}|{g}", 1.0) < alpha

    out = {"nl_highest_deny_to_allow_t2": [], "nl_t2_below_mean_holm": [], "nl_t3_above_mean_holm": []}
    out["worst_rendering"] = {}
    for k, d in per_model.items():
        t2 = d["t2"]
        if max(REND, key=lambda g: t2["deny_to_allow"][g]["mean"]) == "nl_statement":
            out["nl_highest_deny_to_allow_t2"].append(k)
        if t2["contrasts"]["nl_statement"]["mean"] < 0 and sig(holm_t2, k, "nl_statement"):
            out["nl_t2_below_mean_holm"].append(k)
        if d["t3"]["contrasts"]["nl_statement"]["mean"] > 0 and sig(holm_t3, k, "nl_statement"):
            out["nl_t3_above_mean_holm"].append(k)
        out["worst_rendering"][k] = {tier: d[tier]["worst"] for tier in ("t1", "t2", "t3") if d.get(tier)}
    return out


def distil(cfg: dict, metrics: dict, checks: list[dict]) -> dict:
    params = {c["model_key"]: c["params_b"] for c in checks}
    per_model = {}
    for m in cfg["models"]:
        k, raw = m["key"], metrics[m["key"]]
        per_model[k] = {
            "label": m["label"],
            "params_b": params[k],
            "t2": distil_t2(raw["t2"]),
            "t1": distil_t1(raw["t1"]) if "t1" in raw else None,
            "t3": distil_t3(raw["t3"]),
        }
    keys = [m["key"] for m in cfg["models"]]
    core = [k for k in keys if k not in cfg["scaling_descriptive_exclude"]]
    holm_t2 = holm_family(per_model, "t2", cfg["alpha"])
    holm_t3 = holm_family(per_model, "t3", cfg["alpha"])
    gap_p = {k: per_model[k]["t2"]["gap"]["p_bootstrap"] for k in keys}
    return {
        "config": {k: cfg[k] for k in ("dataset_sha256", "seeds", "permutations", "alpha")},
        "runs": checks,
        "models": per_model,
        "order": keys,
        "holm": {"t2": holm_t2, "t3": holm_t3, "t2_gap": holm(gap_p)},
        "scaling": {
            "t2": scaling(per_model, keys, "t2", cfg),  # preregistered Q4 test
            "t2_without_variants": scaling(per_model, core, "t2", cfg),  # descriptive
            "t1": scaling(per_model, keys, "t1", cfg),  # descriptive
            "t3": scaling(per_model, keys, "t3", cfg),  # descriptive (phase2-t3.md)
        },
        "quantization": quantization(per_model, *cfg["quantization_pair"]),
        "patterns": patterns(per_model, holm_t2, holm_t3, cfg["alpha"]),
    }


# --------------------------------------------------------------------------- tables


def pct(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{nd}f}"


def signed_pp(x: float, nd: int = 1) -> str:
    s = f"{100 * x:+.{nd}f}"
    return s.replace("-", "−")


def gap_ci(v: dict, nd: int = 1) -> str:
    lo, hi = v["ci95"]
    return f"{pct(v['mean'], nd)} [{pct(lo, nd)}, {pct(hi, nd)}]"


def rho_text(rho: float | None, nd: int = 3) -> str:
    return "—" if rho is None else f"{rho:+.{nd}f}".replace("-", "−")


def p_text(p: float | None) -> str:
    if p is None:
        return "—"
    return "<5e-4" if p == 0 else f"{p:.2g}"


class Table:
    def __init__(self, name: str, caption: str, header: list[str], rows: list[list[str]], notes: str = ""):
        self.name, self.caption, self.header, self.rows, self.notes = name, caption, header, rows, notes

    def markdown(self) -> str:
        def row(cells: list[str]) -> str:
            return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

        lines = [f"### {self.caption}", "", row(self.header)]
        lines.append("|" + "|".join("---" if i == 0 else "---:" for i in range(len(self.header))) + "|")
        lines += [row(r) for r in self.rows]
        if self.notes:
            lines += ["", self.notes]
        return "\n".join(lines) + "\n"

    def latex(self) -> str:
        spec = "l" + "r" * (len(self.header) - 1)
        body = [
            "% generated by scripts/authinv_figures.py; do not edit",
            "\\begin{table*}[t]",
            "\\centering\\small",
            "% shrink to the text width only when the table is wider than it",
            "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
            f"\\begin{{tabular}}{{{spec}}}",
            "\\toprule",
            " & ".join(tex(h) for h in self.header) + " \\\\",
            "\\midrule",
        ]
        body += [" & ".join(tex(c) for c in r) + " \\\\" for r in self.rows]
        body += ["\\bottomrule", "\\end{tabular}}"]
        cap = tex(self.caption.rstrip(".") + ".") + (" " + tex(self.notes) if self.notes else "")
        body += [f"\\caption{{{cap}}}", f"\\label{{tab:{self.name}}}", "\\end{table*}", ""]
        return "\n".join(body)


_TEX = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "#": "\\#",
    "_": "\\_",
    "$": "\\$",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
    "→": "$\\rightarrow$",
    "−": "$-$",
    "ρ": "$\\rho$",
    "≥": "$\\geq$",
    "≤": "$\\leq$",
    "×": "$\\times$",
    "|": "$\\mid$",
    "—": "--",
    "–": "--",
    "<": "$<$",
    ">": "$>$",
    "·": "$\\cdot$",
    "Δ": "$\\Delta$",
    "∧": "$\\wedge$",
    "*": "$^{*}$",
}


def tex(s: str) -> str:
    return "".join(_TEX.get(ch, ch) for ch in s)


def tables(s: dict) -> list[Table]:
    M, order = s["models"], s["order"]
    rh = [RN[g] for g in REND]
    out = []

    def lab(k: str) -> str:
        return M[k]["label"]

    def params(k: str) -> str:
        return f"{M[k]['params_b']:g}"

    for tier, cap in (
        ("t2", "T2 application accuracy (%) by rendering"),
        ("t1", "T1 accuracy (%) by rendering"),
    ):
        rows = []
        for k in order:
            d = M[k][tier]
            if d is None:
                continue
            rows.append(
                [lab(k), params(k)]
                + [pct(d["rate"][g]["mean"]) for g in REND]
                + [f"{RN[d['worst']]} → {RN[d['best']]}", gap_ci(d["gap"]), pct(d["disagreement"]["mean"])]
                + [pct(d["all_correct"]["mean"])]
            )
        notes = (
            "Accuracies are world means; gap = best − worst rendering (point estimates), paired within "
            "world, 95% world-clustered bootstrap CI; disagree = share of instances answered differently "
            "across the six renderings; all-correct = share correct under all six."
        )
        if tier == "t1":
            notes += " T1 = sign of the log-prob margin; not defined for gpt-oss and thinking-on models."
        out.append(
            Table(
                f"{tier}_accuracy",
                cap,
                ["Model", "Params (B)"] + rh + ["worst → best", "gap [95% CI]", "disagree", "all-correct"],
                rows,
                notes,
            )
        )

    rows = []
    for k in order:
        d = M[k]["t3"]
        rows.append(
            [lab(k), params(k)]
            + [pct(d["rate"][g]["mean"]) for g in REND]
            + [f"{RN[d['worst']]} → {RN[d['best']]}", gap_ci(d["gap"]), pct(d["disagreement"]["mean"])]
        )
    out.append(
        Table(
            "t3_uir",
            "T3 unauthorized-action rate on deny requests (%) by rendering",
            ["Model", "Params (B)"] + rh + ["worst → best", "gap [95% CI]", "action disagree"],
            rows,
            "UIR on deny = share of deny-labelled episodes in which the agent executed a call the policy "
            "denies; gap = worst − best rendering, paired within world, 95% world-clustered bootstrap CI.",
        )
    )

    rows = []
    for k in order:
        d = M[k]["t2"]
        rows.append(
            [lab(k)]
            + [f"{pct(d['deny_to_allow'][g]['mean'])} / {pct(d['allow_to_deny'][g]['mean'])}" for g in REND]
            + [pct(d["deny_to_allow_instances"]["mean"]), pct(d["allow_to_deny_instances"]["mean"])]
        )
    out.append(
        Table(
            "direction",
            "T2 error direction: deny→allow / allow→deny rate (%) by rendering",
            ["Model"] + rh + ["any deny→allow", "any allow→deny"],
            rows,
            "A rate is computed among rows of that label; refusals, abstains and parse failures count "
            "toward neither direction. The last two columns: share of instances with a flip in that "
            "direction between some pair of renderings.",
        )
    )

    rows = []
    for k in order:
        d = M[k]["t2"]
        i = d["interpretation"]
        rows.append(
            [lab(k)]
            + [pct(d["dissociation"][g]["app_wrong_given_interp_correct"]["mean"]) for g in REND]
            + [f"{RN[i['worst']]} → {RN[i['best']]}", gap_ci(i["gap"])]
        )
    out.append(
        Table(
            "dissociation",
            "T2 dissociation: P(application wrong | interpretation correct), % by rendering",
            ["Model"] + rh + ["interp. worst → best", "interp. gap [95% CI]"],
            rows,
            "Application and interpretation rows matched by world, name assignment, rendering and request.",
        )
    )

    for tier, fam, cap in (
        ("t2", "t2", "T2 rendering contrasts: accuracy minus the mean of the other five renderings (pp)"),
        ("t3", "t3", "T3 rendering contrasts: UIR on deny minus the mean of the other five renderings (pp)"),
    ):
        adj = s["holm"][fam]["adjusted"]
        rows = [
            [lab(k)]
            + [
                signed_pp(M[k][tier]["contrasts"][g]["mean"])
                + ("*" if adj[f"{k}|{g}"] < s["config"]["alpha"] else "")
                for g in REND
            ]
            for k in order
        ]
        n_sig, n = s["holm"][fam]["significant"], s["holm"][fam]["size"]
        out.append(
            Table(
                f"{tier}_contrasts",
                cap,
                ["Model"] + rh,
                rows,
                f"* Holm-adjusted p < {s['config']['alpha']} over the {n} contrasts of this tier "
                f"({n_sig} of {n}). Paired within world; world-clustered bootstrap.",
            )
        )

    names = {
        "t2": "T2 accuracy gap, 12 configurations (preregistered Q4)",
        "t2_without_variants": "T2 accuracy gap, without FP8 and thinking (descriptive)",
        "t1": "T1 accuracy gap (descriptive)",
        "t3": "T3 UIR gap (descriptive)",
    }
    rows = [
        [names[t], str(v["n"]), rho_text(v["rho"]), p_text(v["p_permutation"])]
        for t, v in s["scaling"].items()
    ]
    out.append(
        Table(
            "scaling",
            "Scaling: Spearman ρ between parameter count and the worst-case gap",
            ["Test", "n", "ρ", "permutation p"],
            rows,
            f"Ties get average ranks; two-sided permutation p from {s['config']['permutations']:,} "
            "permutations.",
        )
    )

    q = s["quantization"]
    rows = [
        [TIER_NAME[t]]
        + [signed_pp(v["delta"][g], 2) for g in REND]
        + [
            pct(v["max_abs_delta"], 2),
            pct(v["base_gap"], 2),
            "yes" if v["exceeds_rendering_effect"] else "no",
        ]
        for t, v in q.items()
    ]
    out.append(
        Table(
            "quantization",
            "Quantization control: Qwen3.8-27B FP8 minus bf16 (pp) by rendering",
            ["Tier"] + rh + ["max |Δ|", "bf16 gap", "|Δ| ≥ gap"],
            rows,
            "The preregistered rule applies to T2: quantized configurations leave the headline tables only "
            "if max |Δ| ≥ the bf16 worst-case gap. T1 and T3 rows are descriptive.",
        )
    )

    rows = []
    for k in order:
        sk = M[k]["t2"]["by_source_kind"]
        tk = M[k]["t3"]["by_source_kind"]
        rows.append(
            [lab(k)]
            + [f"{gap_ci(sk[kind]['gap'])} ({RN[sk[kind]['worst']]})" for kind in ("real", "synthetic")]
            + [f"{gap_ci(tk[kind]['gap'])} ({RN[tk[kind]['worst']]})" for kind in ("real", "synthetic")]
        )
    out.append(
        Table(
            "source_kind",
            "Worst-case gap by source kind (pp, worst rendering in brackets)",
            ["Model", "T2 real", "T2 synthetic", "T3 real", "T3 synthetic"],
            rows,
            "Real = CedarBench and Quacky policies (544 worlds); synthetic = 1,440 generated worlds. Best "
            "and worst renderings are re-chosen within each subset.",
        )
    )

    by_model = {}
    for c in s["runs"]:
        by_model.setdefault(c["model_key"], {})[c["tier"]] = c
    rows = []
    for k in order:
        r = by_model[k]
        rows.append(
            [lab(k), r["t2"]["revision"][:12]]
            + [r[t]["run"][:22] if t in r else "—" for t in ("t2", "t1", "t3")]
            + [r["t3"]["agent_version"] or "—"]
        )
    out.append(
        Table(
            "runs",
            "Analysed runs (run-directory timestamps; all complete, frozen dataset, pinned revisions)",
            ["Model", "revision", "T2 run", "T1 run", "T3 run", "agent"],
            rows,
            f"Dataset sha256 {s['config']['dataset_sha256'][:16]}…; per-run git commits, vLLM versions and "
            "GPU types are in phase2_summary.json.",
        )
    )
    return out


# --------------------------------------------------------------------------- figures

# Reference palette (dataviz skill): sequential blue ramp, second sequential context orange,
# diverging blue <-> red around a neutral gray.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ORANGE = ["#fde5da", "#f8c0a6", "#f29a73", "#eb6834", "#c4521f", "#8f3a14"]
DIVERGE = ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b52d2c"]
INK, INK2, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f0efec"
SLOT1 = "#2a78d6"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlesize": 7.5,
            "axes.titlecolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "pdf.fonttype": 42,
            "svg.hashsalt": "authinv",
        }
    )
    return plt, LinearSegmentedColormap


def _save(fig, out: Path, name: str, preview: Path | None) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"Creator": None, "CreationDate": None})
    if preview:
        fig.savefig(preview / f"{name}.png", bbox_inches="tight", dpi=200)


def _cell_ink(rgba) -> str:
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def _heatmap(ax, values, cmap, vmin, vmax, text=None, mask=None):
    """values: rows × renderings (fractions); text: cell strings; mask: cells drawn neutral."""
    import numpy as np

    arr = np.array(values, dtype=float)
    shown = np.where(mask, np.nan, arr) if mask is not None else arr
    cm = cmap.with_extremes(bad=NEUTRAL)
    im = ax.imshow(shown, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if text is None or text[i][j] is None:
                continue
            masked = mask is not None and mask[i][j]
            ink = MUTED if masked else _cell_ink(cm((arr[i, j] - vmin) / (vmax - vmin)))
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6, color=ink)
    ax.set_xticks(range(len(REND)), [RN[g] for g in REND])
    ax.xaxis.tick_top()
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(REND))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, arr.shape[0])], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    return im


def _colorbar(fig, im, ax, label: str, fmt=None, horizontal: bool = False, extend: str = "neither"):
    if horizontal:
        cb = fig.colorbar(
            im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, aspect=40, extend=extend
        )
    else:
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, extend=extend)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    axis = cb.ax.xaxis if horizontal else cb.ax.yaxis
    axis.set_major_formatter(fmt or (lambda v, _: f"{100 * v:.0f}"))
    cb.set_label(label, fontsize=6.5, color=INK2)
    return cb


def _repel_labels(fig, ax, points: list[tuple[float, float]], texts: list[str], gap_px: float = 1.0) -> None:
    """Labels to the right of their points, pushed apart vertically where they would overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    anns = [
        ax.annotate(t, p, xytext=(4, 0), textcoords="offset points", fontsize=5.5, color=INK2, va="center")
        for p, t in zip(points, texts, strict=True)
    ]
    boxes = [a.get_window_extent(renderer) for a in anns]
    for a in anns:
        a.remove()
    base = [ax.transData.transform(p) for p in points]
    dy = [0.0] * len(anns)
    r = 3.5 * px_per_pt  # half-size of a marker, kept clear of other points' labels
    for _ in range(200):
        moved = False
        for i in range(len(anns)):
            for j, (mx, my) in enumerate(base):
                if j == i or boxes[i].x1 < mx - r or mx + r < boxes[i].x0:
                    continue
                yi = base[i][1] + dy[i]
                need = boxes[i].height / 2 + r + gap_px - abs(yi - my)
                if need > 0:
                    dy[i] += (need + 0.1) * (1 if yi >= my else -1)
                    moved = True
            for j in range(i + 1, len(anns)):
                bi, bj = boxes[i], boxes[j]
                if bi.x1 < bj.x0 or bj.x1 < bi.x0:
                    continue
                yi, yj = base[i][1] + dy[i], base[j][1] + dy[j]
                need = (bi.height + bj.height) / 2 + gap_px - abs(yi - yj)
                if need > 0:
                    step = need / 2 + 0.1
                    up, down = (i, j) if yi >= yj else (j, i)
                    dy[up] += step
                    dy[down] -= step
                    moved = True
        if not moved:
            break
    for p, t, d in zip(points, texts, dy, strict=True):
        line = (
            {"arrowstyle": "-", "color": MUTED, "lw": 0.4, "shrinkA": 0, "shrinkB": 2} if abs(d) > 2 else None
        )
        ax.annotate(
            t,
            p,
            xytext=(4, d / px_per_pt),
            textcoords="offset points",
            fontsize=5.5,
            color=INK2,
            va="center",
            arrowprops=line,
        )


def figures(s: dict, out: Path, preview: Path | None) -> list[str]:
    import numpy as np

    plt, LSC = _mpl()
    M, order = s["models"], s["order"]
    labels = [M[k]["label"] for k in order]
    blue = LSC.from_list("seq_blue", BLUE)
    orange = LSC.from_list("seq_orange", ORANGE)
    div = LSC.from_list("div", DIVERGE)
    div_r = LSC.from_list("div_r", DIVERGE[::-1])
    written = []

    # 1. T2 application accuracy by rendering
    acc = [[M[k]["t2"]["rate"][g]["mean"] for g in REND] for k in order]
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    im = _heatmap(ax, acc, blue, 0.4, 1.0, [[pct(v) for v in row] for row in acc])
    ax.set_yticks(range(len(order)), labels)
    _colorbar(fig, im, ax, "T2 application accuracy (%)")
    _save(fig, out, "fig_t2_accuracy", preview)
    written.append("fig_t2_accuracy.pdf")
    plt.close(fig)

    # 2. Rendering contrasts, T2 accuracy and T3 UIR; Holm-non-significant cells neutral
    clip = 0.10
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"wspace": 0.06})
    alpha = s["config"]["alpha"]
    for ax, tier, cmap, title in (
        (axes[0], "t2", div_r, "T2 accuracy: rendering − mean of the other five (pp)"),
        (axes[1], "t3", div, "T3 UIR on deny: rendering − mean of the other five (pp)"),
    ):
        vals = [[M[k][tier]["contrasts"][g]["mean"] for g in REND] for k in order]
        adj = s["holm"][tier]["adjusted"]
        mask = [[adj[f"{k}|{g}"] >= alpha for g in REND] for k in order]
        im = _heatmap(ax, vals, cmap, -clip, clip, [[signed_pp(v) for v in row] for row in vals], mask)
        ax.set_yticks(range(len(order)), labels if tier == "t2" else [])
        cb = _colorbar(
            fig,
            im,
            ax,
            title,
            lambda v, _: f"{100 * v:+.0f}".replace("-", "−"),
            horizontal=True,
            extend="both",
        )
        cb.set_ticks([-clip, -clip / 2, 0, clip / 2, clip])
    fig.text(
        0.5,
        -0.03,
        f"Red = riskier (lower accuracy, more unauthorized actions); colour clipped at ±{100 * clip:.0f} pp, "
        "cell text gives the value. Gray cells: Holm-adjusted p ≥ 0.05 within the tier's 72-test family.",
        ha="center",
        fontsize=6,
        color=INK2,
    )
    _save(fig, out, "fig_contrasts", preview)
    written.append("fig_contrasts.pdf")
    plt.close(fig)

    # 3/4. Forest plots: worst-case gap and disagreement, one panel per tier, shared scale
    def forest(name: str, metric: str, xlabel: str, tier_titles: dict) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True, sharex=True)
        y = np.arange(len(order))[::-1]
        for ax, (tier, title) in zip(axes, tier_titles.items(), strict=True):
            for yi, k in zip(y, order, strict=True):
                d = M[k][tier]
                if d is None:
                    ax.text(0.002, yi, "not defined", va="center", fontsize=6, color=MUTED)
                    continue
                v = d[metric]
                ax.plot(v["ci95"], [yi, yi], color=SLOT1, linewidth=1.4, solid_capstyle="round")
                ax.plot([v["mean"]], [yi], "o", color=SLOT1, markersize=4, markeredgecolor="white", mew=0.8)
            ax.set_title(title, loc="left")
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
            ax.set_xlabel(xlabel)
        axes[0].set_yticks(y, labels)
        axes[0].set_xlim(left=0)
        _save(fig, out, name, preview)
        written.append(f"{name}.pdf")
        plt.close(fig)

    forest(
        "fig_worst_case_gap",
        "gap",
        "worst-case gap (pp, 95% CI)",
        {
            "t1": "T1 accuracy (log-prob margin)",
            "t2": "T2 accuracy (generation)",
            "t3": "T3 UIR on deny (agent)",
        },
    )
    forest(
        "fig_disagreement",
        "disagreement",
        "instances with differing answers (%)",
        {"t1": "T1 decision", "t2": "T2 answer", "t3": "T3 action"},
    )

    # 5. Scaling: worst-case gap vs parameters (log), T2 (preregistered) and T3 (descriptive)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.22})
    variants = set(s["scaling"]["t2"]["models"]) - set(s["scaling"]["t2_without_variants"]["models"])
    for ax, tier, title in (
        (axes[0], "t2", "T2 accuracy gap (preregistered Q4)"),
        (axes[1], "t3", "T3 UIR-on-deny gap (descriptive)"),
    ):
        sc = s["scaling"][tier]
        for k in order:
            v, x = M[k][tier]["gap"], M[k]["params_b"]
            ax.plot([x, x], v["ci95"], color=SLOT1, linewidth=1.2, alpha=0.6)
            ax.plot(
                [x],
                [v["mean"]],
                "o",
                markersize=4.5,
                color="white" if k in variants else SLOT1,
                markeredgecolor=SLOT1,
                mew=1.0,
            )
        ax.set_xscale("log")
        ax.set_xlim(3.5, 260)
        ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
        ax.minorticks_off()
        ax.set_xlabel("parameters (B, log scale)")
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
        ax.set_ylabel("worst-case gap (pp)")
        top = max(M[k][tier]["gap"]["ci95"][1] for k in order) * 1.05
        ax.set_ylim(-0.04 * top, top)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        rho = rho_text(sc["rho"], 2)
        ax.set_title(
            f"{title}\nSpearman ρ = {rho}, permutation p = {p_text(sc['p_permutation'])}, n = {sc['n']}",
            loc="left",
        )
        _repel_labels(
            fig,
            ax,
            [(M[k]["params_b"], M[k][tier]["gap"]["mean"]) for k in order],
            [M[k]["label"].replace(" thinking", " think.").replace(" W8A8", "") for k in order],
        )
    fig.text(
        0.5,
        -0.04,
        "Hollow markers: Qwen3.8-27B FP8 and thinking (same base model). Bars: 95% world-clustered CIs.",
        ha="center",
        fontsize=6,
        color=INK2,
    )
    _save(fig, out, "fig_scaling", preview)
    written.append("fig_scaling.pdf")
    plt.close(fig)

    # 6. Error direction (T2): deny->allow and allow->deny rates
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"wspace": 0.12})
    for ax, key, cmap, title in (
        (axes[0], "deny_to_allow", blue, "deny→allow rate, among deny rows (%)"),
        (axes[1], "allow_to_deny", orange, "allow→deny rate, among allow rows (%)"),
    ):
        vals = [[M[k]["t2"][key][g]["mean"] for g in REND] for k in order]
        hi = max(v for row in vals for v in row)
        im = _heatmap(ax, vals, cmap, 0.0, hi, [[pct(v) for v in row] for row in vals])
        ax.set_yticks(range(len(order)), labels if key == "deny_to_allow" else [])
        _colorbar(fig, im, ax, title)
    _save(fig, out, "fig_direction", preview)
    written.append("fig_direction.pdf")
    plt.close(fig)

    # 7. Dissociation (T2): P(application wrong | interpretation correct)
    vals = [
        [M[k]["t2"]["dissociation"][g]["app_wrong_given_interp_correct"]["mean"] for g in REND] for k in order
    ]
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    hi = max(v for row in vals for v in row)
    im = _heatmap(ax, vals, blue, 0.0, hi, [[pct(v) for v in row] for row in vals])
    ax.set_yticks(range(len(order)), labels)
    _colorbar(fig, im, ax, "P(application wrong | interpretation correct), %")
    _save(fig, out, "fig_dissociation", preview)
    written.append("fig_dissociation.pdf")
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    ap.add_argument(
        "--verify-artifacts", action="store_true", help="re-hash every file of each run's manifest"
    )
    ap.add_argument("--preview", type=Path, help="also write PNG previews here")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    metrics, checks = load_runs(cfg, args.verify_artifacts)
    summary = distil(cfg, metrics, checks)
    summary["artifacts_verified"] = bool(args.verify_artifacts)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
    tabs = tables(summary)
    for t in tabs:
        (args.out / f"tab_{t.name}.tex").write_text(t.latex())
    header = "<!-- generated by scripts/authinv_figures.py from configs/authinv/phase2_figures.yaml -->\n\n"
    (args.out / "phase2_tables.md").write_text(header + "\n".join(t.markdown() for t in tabs))
    figs = figures(summary, args.out, args.preview)
    provenance.write_json(args.out / "phase2_summary.json", summary)
    print(f"{len(checks)} runs checked; {len(tabs)} tables, {len(figs)} figures -> {args.out}")


if __name__ == "__main__":
    main()
