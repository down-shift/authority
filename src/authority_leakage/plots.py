"""Plots regenerated entirely from predictions and derived metrics."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "authority_leakage_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from authority_leakage.generation.delegation import SCOPES


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _bar_stats(stats: dict, title: str, ylabel: str, path: Path) -> None:
    keys = list(stats)
    if not keys:
        return
    means = [stats[k]["mean"] for k in keys]
    errors = [[max(0, means[i] - stats[k]["ci95"][0]) for i, k in enumerate(keys)],
              [max(0, stats[k]["ci95"][1] - means[i]) for i, k in enumerate(keys)]]
    fig, ax = plt.subplots(figsize=(max(5, len(keys) * 1.4), 3.6))
    ax.bar(keys, means, color="#436a92")
    ax.errorbar(range(len(keys)), means, yerr=errors, fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title=title, ylabel=ylabel)
    _save(fig, path)


def plot_epistemic(metrics: dict, figure_dir: Path, model_name: str) -> None:
    p = metrics["paired"]
    if not p["n"]:
        return
    _bar_stats(metrics["by_strength"], "Leakage by evidence strength", "Mean paired difference", figure_dir / "epistemic_by_strength.png")
    _bar_stats(metrics["by_trust"], "Leakage by claim metadata", "Mean paired difference", figure_dir / "epistemic_by_trust.png")
    pairs = metrics["pairs"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([p["user_margin"] for p in pairs], [p["system_margin"] for p in pairs], alpha=0.45, s=15)
    bounds = [min(min(p["user_margin"], p["system_margin"]) for p in pairs),
              max(max(p["user_margin"], p["system_margin"]) for p in pairs)]
    ax.plot(bounds, bounds, color="black", linewidth=1)
    ax.set(xlabel="Claim in user: belief margin", ylabel="Claim in privileged role: belief margin", title="Role-swapped pairs")
    _save(fig, figure_dir / "epistemic_role_scatter.png")


def plot_delegation(metrics: dict, figure_dir: Path, model_name: str) -> None:
    matrix = np.full((len(SCOPES), len(SCOPES)), np.nan)
    for a, i in enumerate(SCOPES):
        for b, j in enumerate(SCOPES):
            if i != j:
                value = metrics["leakage_matrix"][f"{i}->{j}"]["rate"]
                matrix[a, b] = value if value is not None else np.nan
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    ax.set(xticks=range(len(SCOPES)), yticks=range(len(SCOPES)), xticklabels=SCOPES,
           yticklabels=SCOPES, xlabel="Attempted scope", ylabel="Authorized scope", title="Unauthorized-follow rate")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.colorbar(image, ax=ax)
    _save(fig, figure_dir / "delegation_leakage_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    values = [metrics["legitimate_compliance"][i]["rate"] for i in SCOPES]
    ax.bar(SCOPES, [np.nan if v is None else v for v in values], color="#327c68")
    ax.set(ylim=(0, 1), ylabel="Rate", title="Legitimate compliance by delegated scope")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, figure_dir / "delegation_compliance.png")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(SCOPES, [np.nan if metrics["selectivity"][i] is None else metrics["selectivity"][i] for i in SCOPES], color="#436a92")
    ax.set(ylim=(-1, 1), ylabel="Compliance minus mean leakage", title="Delegation selectivity")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, figure_dir / "delegation_selectivity.png")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    cats = [c for c in ("high", "medium", "low") if metrics["by_similarity"][c]["rate"] is not None]
    ax.bar(cats, [metrics["by_similarity"][c]["rate"] for c in cats], color="#9a6d46")
    ax.set(ylim=(0, 1), ylabel="Unauthorized-follow rate", title="Leakage by prespecified scope similarity")
    _save(fig, figure_dir / "delegation_similarity.png")


def plot_unified(metrics: dict, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    if metrics["experiment"] == "epistemic":
        value = metrics["paired"]["mean"]
        ax.barh(["Deontic → epistemic"], [0 if value is None else value], color="#436a92")
        ax.set(xlabel="Mean paired belief-margin shift", title="Authority leakage summary")
    else:
        value = metrics["controls"]["both"]["unauthorized_followed"]["rate"]
        compliance = metrics["controls"]["both"]["authorized_followed"]["rate"]
        ax.barh(["Cross-scope leakage", "Legitimate compliance"],
                [0 if value is None else value, 0 if compliance is None else compliance], color=["#9a6d46", "#327c68"])
        ax.set(xlim=(0, 1), xlabel="Rate", title="Authority leakage summary")
    _save(fig, figure_dir / "unified_summary.png")


def make_figures(metrics: dict, figure_dir: Path, model_name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    if metrics["experiment"] == "epistemic":
        plot_epistemic(metrics, figure_dir, model_name)
    else:
        plot_delegation(metrics, figure_dir, model_name)
    plot_unified(metrics, figure_dir)


def plot_combined_summary(epistemic: dict, delegation: dict, path: Path) -> None:
    """A joint figure with separate axes because the primary outcomes have different units."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    shift = epistemic["paired"]["mean"]
    axes[0].bar(["Deontic → epistemic"], [np.nan if shift is None else shift], color="#436a92")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(ylabel="Paired log-probability shift", title="Cross-type")
    leakage = delegation["controls"]["both"]["unauthorized_followed"]["rate"]
    compliance = delegation["controls"]["both"]["authorized_followed"]["rate"]
    axes[1].bar(["Unauthorized", "Authorized"], [np.nan if leakage is None else leakage,
                                                     np.nan if compliance is None else compliance],
                color=["#9a6d46", "#327c68"])
    axes[1].set(ylim=(0, 1), ylabel="Follow rate", title="Cross-scope")
    _save(fig, path)


def plot_model_comparison(metrics_list: list[dict], path: Path) -> None:
    """Compare independent model runs without claiming matched checkpoints."""
    if not metrics_list or len({m["experiment"] for m in metrics_list}) != 1:
        raise ValueError("Model comparison requires runs from one experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [m["model_name"] for m in metrics_list]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.6), 3.6))
    if metrics_list[0]["experiment"] == "epistemic":
        values = [m["paired"]["mean"] for m in metrics_list]
        ax.bar(names, [np.nan if v is None else v for v in values], color="#436a92")
        for k, m in enumerate(metrics_list):
            low, high = m["paired"]["ci95"]
            if low is not None and high is not None and values[k] is not None:
                ax.errorbar(k, values[k], yerr=[[values[k] - low], [high - values[k]]], fmt="none", color="black", capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(ylabel="Paired log-probability shift", title="Epistemic leakage by model")
    else:
        x = np.arange(len(names))
        leak = [m["controls"]["both"]["unauthorized_followed"]["rate"] for m in metrics_list]
        comp = [m["controls"]["both"]["authorized_followed"]["rate"] for m in metrics_list]
        ax.bar(x - 0.18, [np.nan if v is None else v for v in leak], width=0.36, label="Unauthorized", color="#9a6d46")
        ax.bar(x + 0.18, [np.nan if v is None else v for v in comp], width=0.36, label="Authorized", color="#327c68")
        ax.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Follow rate", title="Delegation by model")
        ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, path)
