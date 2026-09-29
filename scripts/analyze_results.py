#!/usr/bin/env python3
"""Reproduce metrics, pilot report, and figures from saved predictions only."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from authority_leakage.analysis import analyze, pilot_report, combined_pilot_report
from authority_leakage.inference import read_jsonl


def analyze_run(run_dir: Path) -> dict:
    from authority_leakage.plots import make_figures
    rows = read_jsonl(run_dir / "predictions.jsonl")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    metrics = analyze(rows)
    metrics["model_name"] = metadata["model"]["model_name"]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (run_dir / "pilot_report.txt").write_text(pilot_report(metrics) + "\n")
    make_figures(metrics, run_dir / "figures", metrics["model_name"])
    if metrics["experiment"] == "epistemic":
        with (run_dir / "bootstrap_ci.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stratum", "n", "mean", "ci95_low", "ci95_high"])
            if "effects" in metrics:
                for key, value in metrics["effects"].items():
                    writer.writerow([key, value["n_worlds"], value["mean"], *value["ci95"]])
            else:
                for kind in ("by_strength", "by_trust", "by_claim_truth", "by_template"):
                    for key, value in metrics[kind].items():
                        writer.writerow([f"{kind}:{key}", value["n"], value["mean"], *value["ci95"]])
                writer.writerow(["overall", metrics["paired"]["n"], metrics["paired"]["mean"], *metrics["paired"]["ci95"]])
    if metrics.get("experiment") == "scope" and "effects" in metrics:
        with (run_dir / "leakage_matrix.csv").open("w", newline="") as handle:
            writer=csv.writer(handle); writer.writerow(["granted_scope","target_scope","n_worlds","mean_effect","ci95_low","ci95_high"])
            for cell,value in metrics["leakage_matrix"].items():
                i,j=cell.split("->")
                writer.writerow([i,j,value["n_worlds"],value["mean"],*value["ci95"]])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="+", type=Path)
    parser.add_argument("--combined-output", type=Path, default=Path("outputs/combined"))
    args = parser.parse_args()
    analyzed = [analyze_run(run_dir) for run_dir in args.run_dir]
    for run_dir, metrics in zip(args.run_dir, analyzed):
        print(f"{run_dir}: {pilot_report(metrics)}")
    if len(analyzed) > 1:
        from authority_leakage.plots import plot_combined_summary, plot_model_comparison
        ep = [m for m in analyzed if m["experiment"] == "epistemic"]
        de = [m for m in analyzed if m["experiment"] == "delegation"]
        args.combined_output.mkdir(parents=True, exist_ok=True)
        if len(ep) == 1 and len(de) == 1:
            report = combined_pilot_report(ep[0], de[0])
            (args.combined_output / "pilot_report.txt").write_text(report + "\n")
            plot_combined_summary(ep[0], de[0], args.combined_output / "unified_summary.png")
            print(report)
        elif len(ep) == len(analyzed) or len(de) == len(analyzed):
            plot_model_comparison(analyzed, args.combined_output / "model_comparison.png")
        else:
            parser.error("Use runs from one experiment, or exactly one run from each experiment")


if __name__ == "__main__":
    main()
