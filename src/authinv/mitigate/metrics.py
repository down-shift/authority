"""Phase-3 arm metrics: conversion quality, garbage-in, gap closure vs the Phase-2 T2 run, cost.

The world is the statistical unit throughout (`authinv.eval.invariance`): a
world's rate is the mean over its rows, CIs are world-clustered percentile
bootstraps. Closure pairs the arm with the T2 baseline of the same model on
the same worlds and uses the same bootstrap draws for both.

Gap closure (all three reported; docs/prereg/phase3.md decision 1 makes ``own``
with a worst-case guard the Gate-3 reading):
- ``own``: 1 - (arm best - arm worst) / (base best - base worst), each arm
  picking its own best and worst rendering by point estimate;
- ``pair``: 1 - (arm[b] - arm[w]) / (base[b] - base[w]) on the baseline's
  best b and worst w;
- ``floor``: (arm worst - base worst) / (base best - base worst), which rises
  only when the worst case improves.
A ratio is undefined when the baseline gap is 0 in a bootstrap draw; those
draws are dropped from the interval and counted. The worst-case accuracy change
(arm worst - base worst, each its own worst rendering) carries a two-sided
paired bootstrap p (the Phase-3 Holm family).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from authinv.eval.invariance import _ci, _per_world, _summ
from authinv.eval.metrics import bootstrap_indices

CONVERSION_RATES = {
    "ir_rate": lambda c: c["conv_category"] == "ir",
    "valid_rate": lambda c: bool(c.get("ir_valid")),
    "exact_rate": lambda c: bool(c.get("ir_exact")),
    "equivalent_rate": lambda c: bool(c.get("ir_equivalent")),
    "refusal_rate": lambda c: c["conv_category"] == "refusal",
    "abstain_rate": lambda c: c["conv_category"] == "abstain",
    "parse_failure_rate": lambda c: c["conv_category"] == "parse_failure",
    "truncated_rate": lambda c: c.get("finish_reason") == "length",
}


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def conversion_metrics(convs: list[dict], replicates: int = 4000, seed: int = 0) -> dict:
    """Per rendering: how often the conversion parses, is valid, exact, and engine-equivalent."""
    worlds = sorted({c["world_id"] for c in convs})
    idx = bootstrap_indices(len(worlds), replicates, seed)
    out = {}
    for g in sorted({c["rendering"] for c in convs}):
        cs = [c for c in convs if c["rendering"] == g]
        d = {name: _summ(_per_world(cs, worlds, f), idx) for name, f in CONVERSION_RATES.items()}
        d["conversions"] = len(cs)
        d["mean_prompt_tokens"] = _mean(c.get("prompt_tokens") for c in cs)
        d["mean_output_tokens"] = _mean(c.get("output_tokens") for c in cs)
        out[g] = d
    point = {g: out[g]["equivalent_rate"]["mean"] for g in out}
    return {
        "per_rendering": out,
        "worst_equivalent_rendering": min(point, key=lambda g: (point[g], g)) if point else None,
        "worlds": len(worlds),
    }


def _wrong_ir(r) -> bool | None:
    """True when the decision stage saw an IR that is not engine-equivalent; None without a stage 2."""
    if r.get("conv_category") != "ir":
        return None
    return not r["ir_equivalent"]


GARBAGE_RATES = {
    # over every row of the rendering
    "garbage_in": lambda r: r["category"] == "correct" and _wrong_ir(r) is True,
    "garbage_in_request": lambda r: r["category"] == "correct" and r.get("ir_request_correct") is False,
    "wrong_ir_rate": lambda r: _wrong_ir(r) is True,
    # the decision a solver would take from the predicted IR (diagnostic; not this arm)
    "ir_engine_accuracy": lambda r: r.get("ir_request_correct") is True,
    # conditional on what the decision stage saw
    "accuracy_given_wrong_ir": lambda r: (r["category"] == "correct") if _wrong_ir(r) is True else None,
    "accuracy_given_equivalent_ir": lambda r: (r["category"] == "correct") if _wrong_ir(r) is False else None,
}


def garbage_in(preds: list[dict], task: str, replicates: int = 4000, seed: int = 0) -> dict:
    sel = [r for r in preds if r["task"] == task]
    worlds = sorted({r["world_id"] for r in sel})
    idx = bootstrap_indices(len(worlds), replicates, seed)
    return {
        g: {
            name: _summ(_per_world([r for r in sel if r["rendering"] == g], worlds, f), idx)
            for name, f in GARBAGE_RATES.items()
        }
        for g in sorted({r["rendering"] for r in sel})
    }


def _accuracy(rows: list[dict], worlds: list[str], rends: list[str]) -> dict[str, np.ndarray]:
    return {
        g: _per_world([r for r in rows if r["rendering"] == g], worlds, lambda r: r["category"] == "correct")
        for g in rends
    }


def _ratio_summary(num: np.ndarray, den: np.ndarray, point_num: float, point_den: float) -> dict:
    ok = den != 0
    vals = 1 - num[ok] / den[ok]
    return {
        "mean": None if point_den == 0 else float(1 - point_num / point_den),
        "ci95": _ci(vals) if ok.any() else [None, None],
        "undefined_draws": int((~ok).sum()),
    }


def _floor_summary(delta: np.ndarray, den: np.ndarray, point_delta: float, point_den: float) -> dict:
    ok = den != 0
    return {
        "mean": None if point_den == 0 else float(point_delta / point_den),
        "ci95": _ci(delta[ok] / den[ok]) if ok.any() else [None, None],
        "undefined_draws": int((~ok).sum()),
    }


def _p_two_sided(boot: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))


def gap_closure(arm: list[dict], base: list[dict], task: str, replicates: int = 4000, seed: int = 0) -> dict:
    """Closure of the worst-case gap, worst-case accuracy change, and per-rendering deltas vs the baseline."""
    arm = [r for r in arm if r["task"] == task]
    base = [r for r in base if r["task"] == task]
    key = lambda r: (r["world_id"], r["assignment"], r["instance"], r["rendering"])  # noqa: E731
    if {key(r) for r in arm} != {key(r) for r in base}:
        raise ValueError("arm and baseline rows differ: closure needs the same rows")
    worlds = sorted({r["world_id"] for r in arm})
    rends = sorted({r["rendering"] for r in arm})
    idx = bootstrap_indices(len(worlds), replicates, seed)
    a, b = _accuracy(arm, worlds, rends), _accuracy(base, worlds, rends)
    pa = {g: float(np.nanmean(a[g])) for g in rends}
    pb = {g: float(np.nanmean(b[g])) for g in rends}
    bb, bw = max(rends, key=lambda g: (pb[g], g)), min(rends, key=lambda g: (pb[g], g))
    ab, aw = max(rends, key=lambda g: (pa[g], g)), min(rends, key=lambda g: (pa[g], g))

    def boot(x):
        return np.nanmean(x[idx], axis=1)

    base_gap, own_gap, pair_gap = b[bb] - b[bw], a[ab] - a[aw], a[bb] - a[bw]
    gb, go, gp = boot(base_gap), boot(own_gap), boot(pair_gap)
    worst_delta = a[aw] - b[bw]
    gd = boot(worst_delta)
    return {
        "baseline": {"best": bb, "worst": bw, "gap": _summ(base_gap, idx)},
        "arm": {
            "best": ab,
            "worst": aw,
            "gap": _summ(own_gap, idx),
            "gap_on_baseline_pair": _summ(pair_gap, idx),
        },
        "closure_own": _ratio_summary(go, gb, float(np.nanmean(own_gap)), float(np.nanmean(base_gap))),
        "closure_pair": _ratio_summary(gp, gb, float(np.nanmean(pair_gap)), float(np.nanmean(base_gap))),
        "closure_floor": _floor_summary(gd, gb, float(np.nanmean(worst_delta)), float(np.nanmean(base_gap))),
        "worst_case_accuracy_delta": {
            **_summ(worst_delta, idx),
            "p_bootstrap": _p_two_sided(gd),
            "arm": pa[aw],
            "baseline": pb[bw],
        },
        "accuracy_delta": {g: _summ(a[g] - b[g], idx) for g in rends},
        "worlds": len(worlds),
    }


def cost(
    convs: list[dict], preds: list[dict], base: list[dict] | None = None, timing: list[dict] | None = None
) -> dict:
    """Tokens per query by rendering (conversion amortized over its queries) and wall time per stage."""
    per_case = defaultdict(int)  # arms without a shared conversion have no case_id: nothing to amortize
    for r in preds:
        per_case[r.get("case_id")] += 1
    conv_tokens = {c["case_id"]: (c.get("prompt_tokens") or 0) + (c.get("output_tokens") or 0) for c in convs}
    out = {}
    for g in sorted({r["rendering"] for r in preds}):
        rs = [r for r in preds if r["rendering"] == g]
        stage2 = [(r.get("prompt_tokens") or 0) + (r.get("output_tokens") or 0) for r in rs]
        amort = [conv_tokens.get(r.get("case_id"), 0) / per_case[r.get("case_id")] for r in rs]
        d = {
            "queries": len(rs),
            "decision_tokens_per_query": float(np.mean(stage2)),
            "conversion_tokens_per_query": float(np.mean(amort)),
            "total_tokens_per_query": float(np.mean(stage2) + np.mean(amort)),
            "conversion_tokens_per_conversion": _mean(
                conv_tokens[c["case_id"]] for c in convs if c["rendering"] == g
            ),
        }
        if base is not None:
            tasks = {x["task"] for x in rs}
            bs = [
                (r.get("prompt_tokens") or 0) + (r.get("output_tokens") or 0)
                for r in base
                if r["rendering"] == g and r["task"] in tasks
            ]
            d["baseline_tokens_per_query"] = float(np.mean(bs)) if bs else None
            d["token_ratio"] = d["total_tokens_per_query"] / d["baseline_tokens_per_query"] if bs else None
        out[g] = d
    stages = {}
    for t in timing or []:
        s = stages.setdefault(t["stage"], {"requests": 0, "seconds": 0.0, "batches": 0})
        s["requests"] += t["requests"]
        s["seconds"] += t["seconds"]
        s["batches"] += 1
    for s in stages.values():
        s["seconds_per_1k_requests"] = 1000 * s["seconds"] / s["requests"] if s["requests"] else None
    return {"per_rendering": out, "wall_time": stages}
