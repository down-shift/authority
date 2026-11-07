"""Activation cache with integrity hashes (P5.1).

Layout of a cache directory:
- ``acts_<shard>.npy``: float16, shape (items, layers, hidden), one row per item in shard order;
- ``items.jsonl``: the items (id, world, assignment, instance, rendering, task, labels, meta) in
  cache order, without prompts;
- ``cache.json``: layers, hidden size, item count, per-shard sha256 and item ranges, and the
  sha256 of ``items.jsonl``.

`load` re-hashes every file and refuses on any mismatch, so an analysis never reads a
truncated or edited cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CacheWriter:
    def __init__(self, root: Path, layers: list[int], info: dict):
        self.root, self.layers, self.info = root, layers, info
        self.shards: list[dict] = []
        self.items: list[dict] = []
        self.hidden: int | None = None
        root.mkdir(parents=True, exist_ok=True)

    def add(self, items: list[dict], acts: np.ndarray) -> None:
        if acts.shape[:2] != (len(items), len(self.layers)):
            raise ValueError(
                f"activations {acts.shape} do not match {len(items)} items × {len(self.layers)} layers"
            )
        if self.hidden is None:
            self.hidden = int(acts.shape[2])
        path = self.root / f"acts_{len(self.shards):05d}.npy"
        np.save(path, acts.astype(np.float16))
        start = len(self.items)
        self.items += [{k: v for k, v in it.items() if k != "prompt"} for it in items]
        self.shards.append({"file": path.name, "start": start, "stop": len(self.items), "sha256": _sha(path)})

    def close(self) -> dict:
        items_path = self.root / "items.jsonl"
        items_path.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in self.items))
        manifest = {
            **self.info,
            "layers": self.layers,
            "hidden": self.hidden,
            "items": len(self.items),
            "items_sha256": _sha(items_path),
            "shards": self.shards,
        }
        (self.root / "cache.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest


def load(root: Path) -> tuple[np.ndarray, list[dict], dict]:
    """(activations (items, layers, hidden) as float32, items, manifest), after checking every hash."""
    manifest = json.loads((root / "cache.json").read_text())
    if _sha(root / "items.jsonl") != manifest["items_sha256"]:
        raise ValueError("items.jsonl sha256 differs from cache.json")
    items = [json.loads(line) for line in (root / "items.jsonl").read_text().splitlines() if line.strip()]
    parts = []
    for s in manifest["shards"]:
        path = root / s["file"]
        if _sha(path) != s["sha256"]:
            raise ValueError(f"{s['file']}: sha256 differs from cache.json")
        a = np.load(path)
        if a.shape[0] != s["stop"] - s["start"]:
            raise ValueError(f"{s['file']}: {a.shape[0]} rows, cache.json says {s['stop'] - s['start']}")
        parts.append(a.astype(np.float32))
    acts = (
        np.concatenate(parts)
        if parts
        else np.zeros((0, len(manifest["layers"]), manifest["hidden"] or 0), np.float32)
    )
    if len(items) != manifest["items"] or acts.shape[0] != len(items):
        raise ValueError("item count differs from cache.json")
    return acts, items, manifest
