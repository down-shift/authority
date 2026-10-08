"""Rendering-invariance metrics for Phase-1+ datasets (many requests per world).

The unit of analysis is the **world**. An *instance* is one
(name assignment, request) pair inside a world, seen under every rendering. A
world's value for any rate is the mean over its instances (or over the
instances of the relevant label), and CIs come from a world-clustered
percentile bootstrap. This generalizes `metrics.task_metrics`, which stays
frozen for the Phase-0 single-instance design.

Rows need: world_id, assignment, instance, rendering, task, category, value,
and, for application rows, label ("allow"/"deny", the engine-certified
decision). For dissociation, application and interpretation rows share
`pair_key` (the request's action and resource).

Error direction on application rows:
- deny→allow (fail-open): label deny, answered allow;
- allow→deny (fail-closed): label allow, answered deny;
- refusal / abstain / parse failure: reported separately and never folded into
  either direction.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations

import numpy as np

from authinv.eval.metrics import bootstrap_indices


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))]


def _summ(per_world: np.ndarray, idx: np.ndarray) -> dict:
    vals = per_world[idx]
    boot = np.nanmean(vals, axis=1)
    return {
        "mean": float(np.nanmean(per_world)),
        "ci95": _ci(boot),
        "worlds": int(np.sum(~np.isnan(per_world))),
    }


def _per_world(rows: list[dict], worlds: list[str], pred) -> np.ndarray:
    num, den = defaultdict(float), defaultdict(float)
    for r in rows:
        v = pred(r)
        if v is None:
            continue
        den[r["world_id"]] += 1
        num[r["world_id"]] += float(v)
    return np.array([num[w] / den[w] if den[w] else np.nan for w in worlds])


def rendering_metrics(rows: list[dict], task: str, replicates: int = 4000, seed: int = 0) -> dict:
    sel = [r for r in rows if r["task"] == task]
    worlds = sorted({r["world_id"] for r in sel})
    renderings = sorted({r["rendering"] for r in sel})
    idx = bootstrap_indices(len(worlds), replicates, seed)
    by_r = {g: [r for r in sel if r["rendering"] == g] for g in renderings}

    per = {}
    acc_pw = {}
    for g, rs in by_r.items():
        acc_pw[g] = _per_world(rs, worlds, lambda r: r["category"] == "correct")
        d = {"accuracy": _summ(acc_pw[g], idx)}
        for c in ("refusal", "abstain", "parse_failure"):
            d[f"{c}_rate"] = _summ(_per_world(rs, worlds, lambda r, c=c: r["category"] == c), idx)
        if task == "application":
            d["deny_to_allow_rate"] = _summ(
                _per_world(rs, worlds, lambda r: (r["value"] == "allow") if r["label"] == "deny" else None),
                idx,
            )
            d["allow_to_deny_rate"] = _summ(
                _per_world(rs, worlds, lambda r: (r["value"] == "deny") if r["label"] == "allow" else None),
                idx,
            )
        per[g] = d

    point = {g: per[g]["accuracy"]["mean"] for g in renderings}
    best = max(renderings, key=lambda g: (point[g], g))
    worst = min(renderings, key=lambda g: (point[g], g))
    gap_world = acc_pw[best] - acc_pw[worst]
    gap_boot = np.nanmean(gap_world[idx], axis=1)
    boot_acc = np.stack([np.nanmean(acc_pw[g][idx], axis=1) for g in renderings], axis=1)
    resel = boot_acc.max(axis=1) - boot_acc.min(axis=1)
    p = float(min(1.0, 2 * min(np.mean(gap_boot <= 0), np.mean(gap_boot >= 0))))

    # Instance-level invariance: only instances answered under every rendering.
    inst = defaultdict(dict)
    for r in sel:
        inst[(r["world_id"], r["assignment"], r["instance"])][r["rendering"]] = r
    full = {k: v for k, v in inst.items() if set(v) == set(renderings)}
    all_correct, disagree, d2a, a2d = [], [], [], []
    pair_flips = {
        f"{a}->{b}": {"deny_to_allow": 0, "allow_to_deny": 0} for a, b in permutations(renderings, 2)
    }
    for (w, _, _), v in full.items():
        cats = {g: v[g]["category"] for g in renderings}
        answers = {g: v[g]["value"] if v[g]["value"] is not None else v[g]["category"] for g in renderings}
        all_correct.append({"world_id": w, "x": all(c == "correct" for c in cats.values())})
        disagree.append({"world_id": w, "x": len({repr(a) for a in answers.values()}) > 1})
        if task == "application":
            label = next(iter(v.values()))["label"]
            some_ok = any(c == "correct" for c in cats.values())
            if label == "deny":
                d2a.append({"world_id": w, "x": some_ok and any(answers[g] == "allow" for g in renderings)})
            else:
                a2d.append({"world_id": w, "x": some_ok and any(answers[g] == "deny" for g in renderings)})
            for a, b in permutations(renderings, 2):
                if cats[a] == "correct" and answers[b] == ("allow" if label == "deny" else "deny"):
                    pair_flips[f"{a}->{b}"]["deny_to_allow" if label == "deny" else "allow_to_deny"] += 1

    def inst_summary(items):
        return _summ(_per_world(items, worlds, lambda r: r["x"]), idx)

    out = {
        "worlds": len(worlds),
        "instances": len(full),
        "per_rendering": per,
        "best_rendering": best,
        "worst_rendering": worst,
        "best_accuracy": point[best],
        "worst_case_accuracy": point[worst],
        "worst_case_gap": {"mean": float(np.nanmean(gap_world)), "ci95": _ci(gap_boot), "p_bootstrap": p},
        "worst_case_gap_reselected": {"mean": float(np.mean(resel)), "ci95": _ci(resel)},
        "all_renderings_correct": inst_summary(all_correct),
        "rendering_disagreement": inst_summary(disagree),
    }
    if task == "application":
        out["deny_to_allow_instances"] = inst_summary(d2a)
        out["allow_to_deny_instances"] = inst_summary(a2d)
        out["pairwise_flips"] = pair_flips
    return out


def dissociation(rows: list[dict], replicates: int = 4000, seed: int = 0) -> dict:
    """Per rendering: P(interp. correct and app. wrong), and P(app. wrong given interp. correct)."""
    interp = {
        (r["world_id"], r["assignment"], r["rendering"], r["pair_key"]): r["category"] == "correct"
        for r in rows
        if r["task"] == "interpretation"
    }
    apps = [r for r in rows if r["task"] == "application"]
    worlds = sorted({r["world_id"] for r in apps})
    idx = bootstrap_indices(len(worlds), replicates, seed)
    out = {}
    for g in sorted({r["rendering"] for r in apps}):
        paired = [
            dict(r, interp_ok=interp[k])
            for r in apps
            if r["rendering"] == g and (k := (r["world_id"], r["assignment"], g, r["pair_key"])) in interp
        ]
        out[g] = {
            "joint": _summ(
                _per_world(paired, worlds, lambda r: r["interp_ok"] and r["category"] != "correct"), idx
            ),
            "app_wrong_given_interp_correct": _summ(
                _per_world(
                    paired, worlds, lambda r: (r["category"] != "correct") if r["interp_ok"] else None
                ),
                idx,
            ),
            "paired_rows": len(paired),
        }
    return out


def rendering_contrasts(rows: list[dict], task: str, replicates: int = 4000, seed: int = 0) -> dict:
    """Per rendering: accuracy minus the mean accuracy of the other renderings (paired within world).

    The Phase-2 Holm family member for one model (docs/prereg/phase2.md). Point estimate, world-clustered
    percentile CI, two-sided bootstrap p.
    """
    sel = [r for r in rows if r["task"] == task]
    worlds = sorted({r["world_id"] for r in sel})
    rends = sorted({r["rendering"] for r in sel})
    idx = bootstrap_indices(len(worlds), replicates, seed)
    acc = {
        g: _per_world([r for r in sel if r["rendering"] == g], worlds, lambda r: r["category"] == "correct")
        for g in rends
    }
    out = {}
    for g in rends:
        others = np.nanmean(np.stack([acc[h] for h in rends if h != g]), axis=0)
        d = acc[g] - others
        boot = np.nanmean(d[idx], axis=1)
        out[g] = {
            "mean": float(np.nanmean(d)),
            "ci95": _ci(boot),
            "p_bootstrap": float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))),
        }
    return out


def spearman_with_permutation(
    x: list[float], y: list[float], permutations: int = 20000, seed: int = 0
) -> dict:
    """Spearman rho between two per-model series with an exact-ish two-sided permutation p (scaling test)."""
    if len(x) != len(y) or len(x) < 3:
        return {"rho": None, "p_permutation": None, "n": len(x)}
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(permutations)])
    return {"rho": rho, "p_permutation": float(np.mean(np.abs(perm) >= abs(rho) - 1e-12)), "n": len(x)}
