#!/usr/bin/env python3
"""Run the two-scope deontic pilot only after Instrument C v2 passed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_leakage.inference import run_inference, write_jsonl, write_run_status
from authority_leakage.two_scope_pilot import (
    analyze_two_scope_pilot, dataset_sha256, generate_two_scope_pilot, validate_two_scope_matching,
)
from authority_leakage.models.hf import HFAdapter


def _commit():
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else None


def _report(metrics: dict) -> str:
    lines = ["# Filename × ordering two-scope pilot", "",
        "This is a two-scope deontic pilot using the frozen Instrument-C decision-owner representation and paired A/B candidate mappings. Numeric/factual answer scopes are excluded.", "",
        f"Worlds: {metrics['worlds']}; parse success: {metrics['parse_success']}; candidate scoring success: {metrics['candidate_scoring_success']}.", "",
        "## Authority comprehension", "", "| Field | Condition | Mapping-cancelled margin accuracy | Generated accuracy map 1 / map 2 |", "|---|---|---:|---:|"]
    for field, field_metrics in metrics["per_field"].items():
        for condition in ("00", "10", "01", "11"):
            comp = field_metrics['comprehension'][condition]
            gen = comp['generated_accuracy_by_mapping']
            lines.append(f"| {field} | {condition} | {comp['mapping_cancelled_margin_accuracy']} | {gen['1']} / {gen['2']} |")
    lines += ["", "## Conditions and manipulation checks", "", "| Field | Condition | Paired margin mean (95% CI) | Median | Margin-based source adoption | Generated adoption map 1 / map 2 | Mapping flips |", "|---|---|---:|---:|---:|---:|---:|"]
    for field, field_metrics in metrics["per_field"].items():
        for condition, values in field_metrics["by_condition"].items():
            lines.append(f"| {field} | {condition} | {values['mean']} ({values['bootstrap_ci95_world_cluster']}) | {values['median']} | {values['semantic_adoption_rate']} | {values['generated_adoption_by_mapping']['1']} / {values['generated_adoption_by_mapping']['2']} | {values['mapping_flip_rate']} |")
    lines += ["", "## Paired contrasts", "", "| Contrast | Mean (95% CI) | Median | Worlds |", "|---|---:|---:|---:|"]
    for key, effect in metrics["effects"].items():
        lines.append(f"| {key} | {effect['mean']} ({effect['bootstrap_ci95_world_cluster']}) | {effect['median']} | {effect['n_worlds']} |")
    lines += ["", "## Identifiability gates", ""]
    lines += [f"- {'PASS' if val else 'FAIL'} {key}" for key, val in metrics["gates"].items()]
    lines += ["", metrics["off_diagonal_interpretation"], "",
              "Primary Lambda values are continuous paired differences in mapping-cancelled source margins. Generated binary adoption is secondary. Behavioral effects do not establish a particular internal authority representation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="models.yaml key or Hugging Face ID/path")
    parser.add_argument("--validated-v2-run", type=Path, required=True,
                        help="Instrument validation v2 run directory whose validation.passed must be true")
    parser.add_argument("--config", type=Path, default=Path("configs/two_scope_pilot.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    v2_path = args.validated_v2_run / "metrics.json"
    if not v2_path.is_file() or not json.loads(v2_path.read_text()).get("validation", {}).get("passed"):
        parser.error("Stage 2 is blocked: the supplied Instrument-C v2 run has not passed every predefined gate")
    config = yaml.safe_load(args.config.read_text())
    if config.get("experiment") != "two_scope_pilot":
        parser.error("config experiment must be two_scope_pilot")
    seed = int(config["seed"])
    examples = generate_two_scope_pilot(config, seed)
    diffs = validate_two_scope_matching(examples)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.model)
    run_dir = args.output_root / f"{stamp}_two_scope_pilot_{slug}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**config, "model_key": args.model,
        "validated_v2_run": str(args.validated_v2_run)}, sort_keys=True))
    write_jsonl(run_dir / "dataset.jsonl", (e.to_dict() for e in examples))
    write_jsonl(run_dir / "prompt_diffs.jsonl", diffs)
    dsha = dataset_sha256(examples)
    (run_dir / "validation_report.json").write_text(json.dumps({"worlds": config["worlds"],
        "examples": len(examples), "paired_mapping_diffs": len(diffs), "all_matching_checks_passed": True,
        "dataset_sha256": dsha, "validated_v2_run": str(args.validated_v2_run),
        "two_scope_only": True, "full_matrix_auto_run": False}, indent=2) + "\n")
    models = yaml.safe_load(args.models_config.read_text()).get("models", {})
    spec = models.get(args.model, {"name": args.model, "revision": None})
    write_run_status(run_dir / "run_status.json", "loading_model", 0, len(examples))
    try:
        adapter = HFAdapter(spec["name"], spec.get("revision"), args.device,
            enable_thinking=spec.get("enable_thinking"), quantization=spec.get("quantization"),
            attention_implementation=spec.get("attention_implementation"))
    except BaseException as exc:
        write_run_status(run_dir / "run_status.json", "model_load_failed", 0, len(examples), repr(exc))
        raise
    provenance = adapter.provenance()
    metadata = {"run_id": run_dir.name, "git_commit": _commit(), "model": provenance,
        "model_id": provenance.get("model_name"), "model_revision": provenance.get("model_commit"),
        "tokenizer_id": provenance.get("tokenizer_name"), "tokenizer_revision": provenance.get("tokenizer_commit"),
        "dataset_sha256": dsha, "seed": seed,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": config["max_new_tokens"],
                     "candidate_scoring": "paired conditional log probability for A and B; constrained greedy label output"},
        "validated_v2_run": str(args.validated_v2_run), "full_matrix_auto_run": False}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_run_status(run_dir / "run_status.json", "running", 0, len(examples))
    predictions_path = run_dir / "predictions.jsonl"
    completed = 0
    def tick():
        nonlocal completed
        completed += 1
        if completed % 20 == 0 or completed == len(examples):
            write_run_status(run_dir / "run_status.json", "running", completed, len(examples))
    try:
        write_jsonl(predictions_path, run_inference(examples, adapter, int(config["max_new_tokens"])), on_row=tick)
    except BaseException as exc:
        write_run_status(run_dir / "run_status.json", "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         completed, len(examples), repr(exc))
        raise
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines() if line]
    metrics = analyze_two_scope_pilot(rows, seed=seed)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (run_dir / "pilot_report.md").write_text(_report(metrics))
    write_run_status(run_dir / "run_status.json", "complete", len(rows), len(examples))
    print(f"Run: {run_dir}")
    for name in ("R_filename_10_minus_00", "R_ordering_01_minus_00",
                 "Lambda_filename_to_ordering_10_minus_00", "Lambda_ordering_to_filename_01_minus_00"):
        e = metrics["effects"][name]
        print(f"{name}: mean={e['mean']} CI95={e['bootstrap_ci95_world_cluster']} n_worlds={e['n_worlds']}")
    print(metrics["off_diagonal_interpretation"])


if __name__ == "__main__":
    main()
