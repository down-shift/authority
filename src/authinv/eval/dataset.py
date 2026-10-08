"""Frozen evaluation dataset loading and validation, shared by the T1 and T2 runners."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from authinv import provenance

REQUIRED = (
    "row_id",
    "world_id",
    "assignment",
    "instance",
    "rendering",
    "task",
    "prompt",
    "expected",
    "pair_key",
)


def load_dataset(path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((path / "dataset_manifest.json").read_text())
    got = provenance.sha256_file(path / "dataset.jsonl")
    if got != manifest["sha256"]:
        raise SystemExit(f"dataset.jsonl sha256 {got} != manifest {manifest['sha256']}")
    rows = provenance.read_jsonl(path / "dataset.jsonl")
    problems = []
    if len(rows) != manifest["rows"]:
        problems.append("row count")
    if len({r["row_id"] for r in rows}) != len(rows):
        problems.append("duplicate row_id")
    missing = {k for r in rows for k in REQUIRED if k not in r}
    if missing:
        problems.append(f"missing fields {sorted(missing)}")
    if any(r["task"] == "application" and r.get("label") not in ("allow", "deny") for r in rows):
        problems.append("application row without allow/deny label")
    cells = collections.Counter((r["world_id"], r["assignment"], r["instance"], r["task"]) for r in rows)
    renderings = {r["rendering"] for r in rows}
    if any(n != len(renderings) for n in cells.values()):
        problems.append("an instance is not rendered in every rendering")
    if problems:
        raise SystemExit(f"dataset validation failed: {problems}")
    return rows, {"passed": True, "rows": len(rows), "renderings": sorted(renderings), "manifest": manifest}
