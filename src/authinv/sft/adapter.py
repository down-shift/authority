"""A trained LoRA adapter as an evaluation input (P4.2b): checked, hashed, and described for provenance.

An SFT run directory (`scripts/authinv_sft_train.py`) holds `adapter/` (the peft adapter),
`metrics.json` (with `adapter_sha256`, one sha256 per adapter file) and `config.json` (model, arm,
seed, LoRA rank). `load_adapter` refuses unless the run belongs to the model being evaluated, its
status is `complete` (a `smoke` run only for a smoke evaluation), and every adapter file still has
the sha256 the trainer recorded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_adapter(run: Path, model_key: str, spec: dict, smoke: bool = False) -> dict:
    """{"path", "rank", "run", "arm", "seed", "files_sha256", "smoke"} for vLLM LoRA serving."""
    conf = json.loads((run / "config.json").read_text())
    status = json.loads((run / "run_status.json").read_text())["status"]
    mpath = run / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    problems = []
    if conf["model"].get("key") != model_key:
        problems.append(f"adapter trained on {conf['model'].get('key')}, not {model_key}")
    if conf["model"] != spec:
        problems.append("model spec differs from the registry")
    if status != "complete" and not (smoke and status == "smoke"):
        problems.append(f"status {status}")
    adapter_dir = run / "adapter"
    files = (
        {p.name: _sha(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()}
        if adapter_dir.is_dir()
        else {}
    )
    if files != metrics.get("adapter_sha256"):
        problems.append("adapter files differ from the sha256 the trainer recorded")
    if problems:
        raise SystemExit(f"adapter {run} refused: {problems}")
    return {
        "path": str(adapter_dir),
        "rank": int(conf["config"]["lora"]["r"]),
        "run": str(run),
        "arm": conf["arm"],
        "seed": conf["seed"],
        "files_sha256": files,
        "smoke": status == "smoke",
    }
