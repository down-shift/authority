"""Probes, baselines and CKA on cached activations (P5.1; numpy + scipy, no sklearn).

- **Linear probes:** L2-regularized multinomial logistic regression on standardized
  features (fit on the training split only), solved with L-BFGS. Deterministic.
- **Cross-rendering transfer:** for each layer and each pair (a, b), train on rendering a's
  items in the training worlds and test on rendering b's items in the test worlds. The
  split is by world, so an instance never sits on both sides. The diagonal is held-out
  worlds in the same rendering.
- **Metric:** balanced accuracy (mean per-class recall), with a percentile bootstrap over
  test worlds (the world is the unit).
- **Baselines:**
  - metadata-only: the same probe on surface features (rendering, tier, rule and condition
    counts, entity count, prompt length, option order), no activations;
  - shuffled-label control: train on labels permuted within the training split, test on the
    true labels;
  - majority class.
- **CKA:** linear CKA between two renderings' activations of the same instances, per layer
  (Kornblith et al. 2019), with feature-centered matrices.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize


def world_split(world_ids: list[str], test_fraction: float, seed: int) -> set[str]:
    """Deterministic test-world set: worlds whose salted hash falls under the fraction."""

    def u(w):
        return int(hashlib.sha256(f"{seed}|{w}".encode()).hexdigest()[:12], 16) / 16**12

    return {w for w in set(world_ids) if u(w) < test_fraction}


class Probe:
    def __init__(self, l2: float = 1.0, max_iter: int = 500):
        self.l2, self.max_iter = l2, max_iter

    def fit(self, X: np.ndarray, y: list) -> Probe:
        self.classes = sorted(set(y))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        idx = {c: i for i, c in enumerate(self.classes)}
        Y = np.zeros((len(y), len(self.classes)))
        Y[np.arange(len(y)), [idx[v] for v in y]] = 1
        n, d, k = Z.shape[0], Z.shape[1], len(self.classes)
        if k == 1:
            self.W, self.b = np.zeros((d, 1)), np.zeros(1)
            return self

        def loss(theta):
            W, b = theta[: d * k].reshape(d, k), theta[d * k :]
            logits = Z @ W + b
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            f = -np.sum(Y * np.log(p + 1e-12)) / n + 0.5 * self.l2 * np.sum(W * W) / n
            g = p - Y
            gW = Z.T @ g / n + self.l2 * W / n
            return f, np.concatenate([gW.ravel(), g.sum(0) / n])

        res = minimize(
            loss, np.zeros(d * k + k), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter}
        )
        self.W, self.b = res.x[: d * k].reshape(d, k), res.x[d * k :]
        return self

    def predict(self, X: np.ndarray) -> list:
        logits = ((X - self.mu) / self.sd) @ self.W + self.b
        return [self.classes[i] for i in logits.argmax(1)]


def balanced_accuracy(y: list, yhat: list) -> float:
    recalls = []
    for c in sorted(set(y)):
        sel = [i for i, v in enumerate(y) if v == c]
        recalls.append(sum(yhat[i] == c for i in sel) / len(sel))
    return float(np.mean(recalls)) if recalls else float("nan")


def bootstrap_ba(y: list, yhat: list, worlds: list[str], replicates: int, seed: int) -> list[float]:
    """Percentile 95% interval of balanced accuracy over test worlds."""
    by_w = defaultdict(list)
    for i, w in enumerate(worlds):
        by_w[w].append(i)
    keys = sorted(by_w)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(replicates):
        pick = [i for k in rng.choice(len(keys), len(keys)) for i in by_w[keys[k]]]
        ys = [y[i] for i in pick]
        if len(set(ys)) > 1:
            vals.append(balanced_accuracy(ys, [yhat[i] for i in pick]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [None, None]


def metadata_features(items: list[dict]) -> np.ndarray:
    keys = sorted({k for it in items for k in it["meta"]})
    cols = []
    for k in keys:
        vals = [it["meta"].get(k) for it in items]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            cols.append(np.array(vals, dtype=float)[:, None])
        else:
            levels = sorted({str(v) for v in vals})
            cols.append(np.array([[str(v) == lv for lv in levels] for v in vals], dtype=float))
    return np.hstack(cols) if cols else np.zeros((len(items), 0))


def _fit_eval(X, items, label, train_idx, test_idx, cfg, seed, shuffle=False):
    ytr = [items[i]["labels"][label] for i in train_idx]
    if shuffle:
        ytr = list(np.random.default_rng(seed).permutation(ytr))
    yte = [items[i]["labels"][label] for i in test_idx]
    if len(set(ytr)) < 2 or not yte:
        return None
    probe = Probe(cfg["l2"], cfg["max_iter"]).fit(X[train_idx], ytr)
    yhat = probe.predict(X[test_idx])
    return {
        "balanced_accuracy": balanced_accuracy(yte, yhat),
        "ci95": bootstrap_ba(
            yte, yhat, [items[i]["world_id"] for i in test_idx], cfg["bootstrap_replicates"], seed
        ),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def _majority(items, label, train_idx, test_idx) -> float | None:
    ytr = [items[i]["labels"][label] for i in train_idx]
    yte = [items[i]["labels"][label] for i in test_idx]
    if not ytr or not yte:
        return None
    top = max(sorted(set(ytr)), key=ytr.count)
    return balanced_accuracy(yte, [top] * len(yte))


def transfer(acts: np.ndarray, items: list[dict], label: str, layers: list[int], cfg: dict) -> dict:
    """Per layer, the (train rendering → test rendering) balanced-accuracy matrix, plus baselines."""
    test_worlds = world_split([it["world_id"] for it in items], cfg["test_fraction"], cfg["split_seed"])
    has = [i for i, it in enumerate(items) if label in it["labels"]]
    rends = sorted({items[i]["rendering"] for i in has})
    split = {
        (g, side): [
            i
            for i in has
            if items[i]["rendering"] == g and (items[i]["world_id"] in test_worlds) == (side == "test")
        ]
        for g in rends
        for side in ("train", "test")
    }
    out = {"label": label, "renderings": rends, "test_worlds": len(test_worlds), "layers": {}}
    for li, layer in enumerate(layers):
        X = acts[:, li, :]
        out["layers"][str(layer)] = {
            f"{a}->{b}": _fit_eval(
                X, items, label, split[(a, "train")], split[(b, "test")], cfg, cfg["split_seed"]
            )
            for a in rends
            for b in rends
        }
    meta = metadata_features(items)
    out["metadata_only"] = {
        f"{a}->{b}": _fit_eval(
            meta, items, label, split[(a, "train")], split[(b, "test")], cfg, cfg["split_seed"]
        )
        for a in rends
        for b in rends
    }
    top = len(layers) - 1
    out["shuffled_label_last_layer"] = {
        g: _fit_eval(
            acts[:, top, :],
            items,
            label,
            split[(g, "train")],
            split[(g, "test")],
            cfg,
            cfg["split_seed"],
            shuffle=True,
        )
        for g in rends
    }
    out["majority_class"] = {
        g: _majority(items, label, split[(g, "train")], split[(g, "test")]) for g in rends
    }
    return out


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA of two (n, d) matrices over the same n items."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    if X.shape[0] < X.shape[1]:  # n x n Gram form is cheaper
        kx, ky = X @ X.T, Y @ Y.T
        num, den = np.sum(kx * ky), np.linalg.norm(kx) * np.linalg.norm(ky)
    else:
        num = np.linalg.norm(Y.T @ X) ** 2
        den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else float("nan")  # a constant layer has no defined CKA


def aligned(items: list[dict], a: str, b: str, task: str) -> tuple[list[int], list[int]]:
    """Indices of rendering a and b items of the same (world, assignment, instance, task)."""
    key = lambda it: (it["world_id"], it["assignment"], it["instance"])  # noqa: E731
    ia = {key(it): i for i, it in enumerate(items) if it["rendering"] == a and it["task"] == task}
    ib = {key(it): i for i, it in enumerate(items) if it["rendering"] == b and it["task"] == task}
    common = sorted(set(ia) & set(ib))
    return [ia[k] for k in common], [ib[k] for k in common]


def cka_matrix(acts: np.ndarray, items: list[dict], layers: list[int], task: str = "application") -> dict:
    rends = sorted({it["rendering"] for it in items if it["task"] == task})
    out = {}
    for li, layer in enumerate(layers):
        m = {}
        for x, a in enumerate(rends):
            for b in rends[x + 1 :]:
                ia, ib = aligned(items, a, b, task)
                m[f"{a}|{b}"] = linear_cka(acts[ia, li, :], acts[ib, li, :]) if len(ia) > 2 else None
        out[str(layer)] = m
    return {"renderings": rends, "task": task, "layers": out}


def instance_divergence(
    acts: np.ndarray, items: list[dict], a: str, b: str, layer_index: int, task: str = "application"
) -> dict:
    """Cosine distance between the two renderings' vectors of each instance (for P5.3's disagreement test)."""
    ia, ib = aligned(items, a, b, task)
    X, Y = acts[ia, layer_index, :], acts[ib, layer_index, :]
    cos = np.sum(X * Y, 1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
    keys = [(items[i]["world_id"], items[i]["assignment"], items[i]["instance"]) for i in ia]
    return {"/".join(k): float(1 - c) for k, c in zip(keys, cos, strict=True)}
