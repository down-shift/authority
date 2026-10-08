"""Run-directory provenance: hashes, git state, run status, artifact manifest.

Adapted from the pilot runners' pattern (config.json, metadata.json,
run_status.json, artifact_sha256.json) so every authinv run is auditable and
resumable only when its provenance still matches.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git_state(root: Path = ROOT) -> dict:
    def run(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def source_hashes(paths: list[Path], root: Path = ROOT) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}


def write_run_status(run: Path, status: str, done: int, total: int, message: str | None = None) -> None:
    value = {
        "status": status,
        "completed_examples": done,
        "total_examples": total,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if message:
        value["message"] = message
    write_json(run / "run_status.json", value)


def finalize(run: Path) -> None:
    write_json(
        run / "artifact_sha256.json",
        {
            str(p.relative_to(run)): sha256_file(p)
            for p in sorted(run.rglob("*"))
            if p.is_file() and p.name != "artifact_sha256.json"
        },
    )
