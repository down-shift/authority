"""Paired bootstrap and exact/Monte Carlo tests."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import binomtest
from authority_leakage.progress import tqdm


def bootstrap_ci(values: list[float], seed: int = 0, n_boot: int = 4000) -> list[float | None]:
    if not values:
        return [None, None]
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[indices].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def bootstrap_mean_difference(a: list[float], b: list[float], seed: int = 0, n_boot: int = 4000) -> list[float | None]:
    if not a or not b:
        return [None, None]
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    xi = rng.integers(0, len(x), size=(n_boot, len(x)))
    yi = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = x[xi].mean(axis=1) - y[yi].mean(axis=1)
    return [float(v) for v in np.quantile(diffs, [0.025, 0.975])]


def paired_sign_permutation(values: list[float], seed: int = 0, n_perm: int = 10000) -> float | None:
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    observed = abs(x.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in tqdm(range(n_perm), total=n_perm, desc="Paired permutation test", unit="shuffle", leave=False):
        extreme += abs((x * rng.choice([-1, 1], size=len(x))).mean()) >= observed - 1e-12
    return (extreme + 1) / (n_perm + 1)


def paired_binary_pvalue(a: list[bool], b: list[bool]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    wins = sum(x and not y for x, y in zip(a, b))
    losses = sum(y and not x for x, y in zip(a, b))
    return float(binomtest(wins, wins + losses, p=0.5).pvalue) if wins + losses else 1.0


def benjamini_hochberg(pvalues: list[float | None]) -> list[float | None]:
    valid = [(i, p) for i, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    result: list[float | None] = [None] * len(pvalues)
    n = len(valid)
    prev = 1.0
    for rank_reverse, (i, p) in enumerate(sorted(valid, key=lambda item: item[1], reverse=True), 1):
        rank = n - rank_reverse + 1
        prev = min(prev, p * n / rank)
        result[i] = prev
    return result
