#!/usr/bin/env python3
"""Run one HF model and persist everything needed for independent analysis."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from authority_leakage.inference import run_inference, tokenization_audit, write_jsonl
from authority_leakage.models.hf import HFAdapter
from generate_dataset import build
from analyze_results import analyze_run


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["epistemic", "delegation"])
    parser.add_argument("--model", required=True, help="models.yaml key or HF repository/path")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models-config", default=Path("configs/models.yaml"), type=Path)
    parser.add_argument("--output-root", default=Path("outputs"), type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("experiment") != args.experiment:
        parser.error("--experiment and config experiment differ")
    model_entries = yaml.safe_load(args.models_config.read_text()).get("models", {}) if args.models_config.exists() else {}
    model_spec = model_entries.get(args.model, {"name": args.model, "revision": None})
    seed = int(config["seed"])
    examples = build(config)
    adapter = HFAdapter(
        model_spec["name"], model_spec.get("revision"), args.device,
        enable_thinking=model_spec.get("enable_thinking"),
        quantization=model_spec.get("quantization"),
        attention_implementation=model_spec.get("attention_implementation"),
    )
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.model)
    run_dir = args.output_root / f"{stamp}_{args.experiment}_{slug}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    full_config = {**config, "model": model_spec, "attention_implementation": adapter.attention_implementation or "model_default",
                   "decoding": {"do_sample": False, "num_beams": 1,
                                                                "max_new_tokens": int(config["max_new_tokens"]), "seed": seed},
                   "model_key": args.model}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(full_config, sort_keys=True))
    audit = tokenization_audit(examples, adapter)
    metadata = {"run_id": run_dir.name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(), "model": adapter.provenance(),
                "generation_config": full_config, "tokenization_audit": audit,
                "deterministic_algorithms_requested": True,
                "candidate_scoring": "sum full continuation-token conditional log probabilities"}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_jsonl(run_dir / "dataset.jsonl", (e.to_dict() for e in examples))
    predictions = run_inference(examples, adapter, int(config["max_new_tokens"]))
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    metrics = analyze_run(run_dir)
    print(f"Run: {run_dir}")
    print((run_dir / "pilot_report.txt").read_text().strip())
    if args.experiment == "epistemic" and (metrics["controls"].get("claim_absent", {}).get("accuracy", {}).get("rate") or 0) <= 0.95:
        print("Pilot gate failed: evidence-only accuracy <= 95%; reconsider the design before scaling.")
    if args.experiment == "delegation" and (metrics["authorized_only_compliance"]["rate"] or 0) <= 0.90:
        print("Pilot gate failed: authorized-only compliance <= 90%; reconsider the design before scaling.")


if __name__ == "__main__":
    main()
