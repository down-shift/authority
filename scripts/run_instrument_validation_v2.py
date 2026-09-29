#!/usr/bin/env python3
"""Run Instrument C paired candidate-mapping validation on the selected work machine."""
from __future__ import annotations

import argparse
import csv
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
from authority_leakage.instrument_validation_v2 import (
    analyze_validation_v2, authority_template_sha256, dataset_sha256,
    generate_validation_v2, validate_v2_matching,
)
from authority_leakage.models.hf import HFAdapter


def _commit() -> str | None:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else None


def _report(metrics: dict, examples: list, diffs: list) -> str:
    lines = [
        "# Instrument validation v2: paired candidate mapping",
        "",
        "Purpose: test whether the eight prior Instrument-C NO errors were candidate-label sensitive. This is instrument validation, not a leakage experiment.",
        "",
        "Frozen Instrument-C authority wording is unchanged. Every world and authority condition is scored twice with reversed A/B mappings. Continuous candidate log-probability margins are oriented toward Source S; mapping averaging cancels additive A/B bias.",
        "",
        f"Pre-inference criteria: `{json.dumps(metrics.get('criteria_recorded_pre_inference', {}), sort_keys=True)}`",
        "",
    ]
    if metrics.get("validation"):
        lines += [f"Decision: **{metrics['validation']['decision']}**", ""]
        for name, passed in metrics["validation"]["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines += ["", "## Behavior and margins", "",
              "| Owner condition | Raw adoption, source=A | Raw adoption, source=B | Mapping-cancelled adoption | Mean margin | Median margin | World-cluster 95% CI | Mapping flips |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for auth in ("NO", "YES"):
        r = metrics["condition_results"][auth]
        raw = metrics["raw_mapping_adoption"][auth]
        lines.append(f"| {auth} | {raw['mapping_1']['source_adoption_rate']} | {raw['mapping_2']['source_adoption_rate']} | {r['semantic_adoption_rate']} | {r['mean']} | {r['median']} | {r['bootstrap_ci95_world_cluster']} | {metrics['mapping_flip_rate'][auth]} |")
    lines += ["", "## Semantic comprehension", "",
              "| Authority owner | Accuracy | Mean source-oriented margin | Median | World-cluster 95% CI |",
              "|---|---:|---:|---:|---:|"]
    for auth in ("NO", "YES"):
        c = metrics["comprehension"][auth]
        lines.append(f"| {auth} | {c['semantic_accuracy']} | {c['mean']} | {c['median']} | {c['bootstrap_ci95_world_cluster']} |")
    resp = metrics["intended_responsiveness_margin"]
    lines += ["", "## Intended responsiveness", "",
              f"Continuous paired responsiveness (YES owner minus NO owner): mean={resp['mean']}, median={resp['median']}, world-cluster 95% CI={resp['bootstrap_ci95_world_cluster']}, n_worlds={resp['n_worlds']}.",
              f"Secondary binary responsiveness: {metrics['binary_responsiveness']}.",
              f"Parse/scoring success: {metrics['parse_scoring_success']}; raw prediction rows: {metrics['raw_rows']}.",
              "", "## Candidate-label nuisance", "",
              f"Label-bias summaries: `{json.dumps({a: {k: v for k, v in x.items() if k != 'values'} for a, x in metrics['label_bias'].items()}, sort_keys=True)}`.",
              "The `m1` and `m2` arrays are preserved in `metrics.json`; per-world mappings and margins are in `per_example_scores.jsonl`.",
              "", "## Representative matched mapping prompts", ""]
    first = sorted({e.metadata["world_id"] for e in examples})[0]
    for auth in ("NO", "YES"):
        for task in ("filename_choice", "authority_comprehension"):
            lines.append(f"### {auth}, {task}, world {first}")
            for mapping in (1, 2):
                e = next(e for e in examples if e.metadata["world_id"] == first and e.metadata["authority"] == auth and e.metadata["task_type"] == task and e.metadata["mapping_id"] == mapping)
                lines += [f"Mapping {mapping}", "", "```text", e.messages[0].content, "```", ""]
    lines += ["Stage 2 is authorized only if every prospective validation gate passes. No full leakage matrix is run by this script.", ""]
    return "\n".join(lines)


def _write_plots(metrics: dict, run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Machine-readable figure source data is also retained in metrics.json and
    # per_example_scores.jsonl; this compact CSV supports direct plotting.
    with (run_dir / "plot_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["authority", "world_index", "m1", "m2", "semantic_source_margin", "label_bias"])
        for auth in ("NO", "YES"):
            data = metrics["m1_m2"][auth]
            margins = metrics["condition_results"][auth]["values"]
            biases = metrics["label_bias"][auth]["values"]
            for i, (m1, m2, margin, bias) in enumerate(zip(data["m1"], data["m2"], margins, biases)):
                writer.writerow([auth, i, m1, m2, margin, bias])

    fig, ax = plt.subplots()
    for auth, marker in (("NO", "o"), ("YES", "x")):
        ax.scatter(metrics["m1_m2"][auth]["m1"], metrics["m1_m2"][auth]["m2"], label=auth, marker=marker, alpha=.65)
    ax.axhline(0, color="grey", linewidth=.7); ax.axvline(0, color="grey", linewidth=.7)
    ax.set(xlabel="m1: log P(A) - log P(B), source=A", ylabel="m2: log P(B) - log P(A), source=B", title="Raw source-oriented margins by mapping")
    ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "m1_vs_m2.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    ax.boxplot([metrics["condition_results"][a]["values"] for a in ("NO", "YES")], showmeans=True)
    ax.set_xticks([1, 2], ["Default owner", "Source owner"])
    ax.axhline(0, color="grey", linewidth=.7); ax.set(ylabel="Mapping-cancelled source margin", title="Semantic margin by owner condition")
    fig.tight_layout(); fig.savefig(run_dir / "semantic_margin_by_owner.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist([metrics["label_bias"][a]["values"] for a in ("NO", "YES")], bins=25, label=["NO", "YES"], alpha=.65)
    ax.set(xlabel="(m1 - m2) / 2", ylabel="World count", title="Candidate-label bias by condition"); ax.legend()
    fig.tight_layout(); fig.savefig(run_dir / "label_bias.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(["NO", "YES"], [metrics["mapping_flip_rate"][a] or 0 for a in ("NO", "YES")])
    ax.set(ylim=(0, 1), ylabel="Rate", title="Semantic answer changes after A/B reversal")
    fig.tight_layout(); fig.savefig(run_dir / "mapping_flip_rate.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    no, yes = metrics["condition_results"]["NO"]["values"], metrics["condition_results"]["YES"]["values"]
    ax.scatter(no, yes, alpha=.65); ax.axhline(0, color="grey", linewidth=.7); ax.axvline(0, color="grey", linewidth=.7)
    ax.set(xlabel="Default owner margin", ylabel="Source owner margin", title="Paired NO → YES margin shifts")
    fig.tight_layout(); fig.savefig(run_dir / "paired_owner_shift.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="models.yaml key or Hugging Face ID/path")
    parser.add_argument("--config", type=Path, default=Path("configs/instrument_validation_v2.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plots-only", type=Path,
                        help="regenerate plots and finalize an existing completed-inference run directory; does not load a model")
    args = parser.parse_args()
    if args.plots_only:
        run_dir = args.plots_only
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.jsonl"
        if not metrics_path.is_file() or not predictions_path.is_file():
            parser.error("--plots-only requires metrics.json and predictions.jsonl in the run directory")
        metrics = json.loads(metrics_path.read_text())
        _write_plots(metrics, run_dir)
        completed = sum(1 for line in predictions_path.open(encoding="utf-8") if line.strip())
        write_run_status(run_dir / "run_status.json", "complete", completed, completed)
        print(f"Plots regenerated and run finalized: {run_dir}")
        return
    if not args.model:
        parser.error("--model is required unless --plots-only is used")
    config = yaml.safe_load(args.config.read_text())
    if config.get("experiment") != "instrument_validation_v2":
        parser.error("config experiment must be instrument_validation_v2")
    seed = int(config["seed"])
    examples = generate_validation_v2(config, seed)
    diffs = validate_v2_matching(examples)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.model)
    run_dir = args.output_root / f"{stamp}_instrument_validation_v2_{slug}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**config, "model_key": args.model}, sort_keys=True))
    write_jsonl(run_dir / "dataset.jsonl", (e.to_dict() for e in examples))
    write_jsonl(run_dir / "prompt_diffs.jsonl", diffs)
    dsha = dataset_sha256(examples)
    (run_dir / "validation_report.json").write_text(json.dumps({
        "worlds": config["worlds"], "examples": len(examples), "matched_mapping_pairs": len(diffs),
        "all_prompt_diff_checks_passed": True, "dataset_sha256": dsha,
        "criteria": config["criteria"], "stage2_or_full_matrix_auto_run": False,
    }, indent=2) + "\n")
    (run_dir / "run_status.json").write_text(json.dumps({"status": "dataset_ready_inference_not_started",
        "completed_examples": 0, "total_examples": len(examples)}, indent=2) + "\n")

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
    metadata = {"run_id": run_dir.name, "git_commit": _commit(), "model": adapter.provenance(),
                "dataset_sha256": dsha, "seed": seed,
                "decoding": {"do_sample": False, "num_beams": 1,
                             "max_new_tokens": config["max_new_tokens"],
                             "candidate_scoring": "unconstrained conditional log-probability of A and B; constrained greedy label output"},
                "authority_template_sha256": authority_template_sha256(),
                "stage2_auto_run": False}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_run_status(run_dir / "run_status.json", "running", 0, len(examples))
    predictions_path = run_dir / "predictions.jsonl"
    done = 0
    def tick() -> None:
        nonlocal done
        done += 1
        if done % 10 == 0 or done == len(examples):
            write_run_status(run_dir / "run_status.json", "running", done, len(examples))
    try:
        write_jsonl(predictions_path, run_inference(examples, adapter, int(config["max_new_tokens"])), on_row=tick)
    except BaseException as exc:
        write_run_status(run_dir / "run_status.json", "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         done, len(examples), repr(exc))
        raise
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines() if line]
    metrics = analyze_validation_v2(rows, seed=seed)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    per_scores = []
    for row in rows:
        md = row["metadata"]
        scores = row.get("candidate_logprobs") or {}
        src, default = (("A", "B") if md["mapping_id"] == 1 else ("B", "A"))
        per_scores.append({"world_id": md["world_id"], "authority": md["authority"],
            "task_type": md["task_type"], "mapping_id": md["mapping_id"],
            "m_source": scores.get(src) - scores.get(default) if src in scores and default in scores else None,
            "candidate_logprobs": scores, "raw_response": row.get("raw_response"),
            "outcome": row.get("outcome"), "label_map": md["label_map"]})
    write_jsonl(run_dir / "per_example_scores.jsonl", per_scores)
    (run_dir / "pilot_report.md").write_text(_report(metrics, examples, diffs))
    _write_plots(metrics, run_dir)
    write_run_status(run_dir / "run_status.json", "complete", len(rows), len(examples))
    print(f"Run: {run_dir}")
    print(metrics["validation"]["decision"])
    print(f"Mean intended margin responsiveness={metrics['intended_responsiveness_margin']['mean']} CI95={metrics['intended_responsiveness_margin']['bootstrap_ci95_world_cluster']}")
    print(f"NO adoption={metrics['condition_results']['NO']['semantic_adoption_rate']} YES adoption={metrics['condition_results']['YES']['semantic_adoption_rate']} parse/scoring={metrics['parse_scoring_success']}")


if __name__ == "__main__":
    main()
