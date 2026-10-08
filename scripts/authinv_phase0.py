#!/usr/bin/env python3
"""Phase 0 kill test: greedy generation + strict parsing on the pilot's Stage-2 prompts.

Subcommands:
  run        one model on the frozen 3,600-row Phase-0 dataset (resumable)
  aggregate  combine per-model runs into the Gate-0 table (a recommendation only)

  uv run python scripts/authinv_phase0.py run --config configs/authinv/phase0.yaml \
      --source-run outputs/<gemma stage-2 run> --model qwen3_8b [--model-dir DIR] [--tensor-parallel 2]
  uv run python scripts/authinv_phase0.py run ... --dataset-only
  uv run python scripts/authinv_phase0.py run ... --resume outputs/<phase0 run>
  uv run python scripts/authinv_phase0.py aggregate --config configs/authinv/phase0.yaml --runs RUN [RUN ...]

See docs/RESEARCH_PLAN.md §3 Phase 0 and docs/prereg/phase0.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import models, provenance  # noqa: E402
from authinv.eval import metrics  # noqa: E402
from authinv.eval.parse import PARSER_VERSION, parse_answer  # noqa: E402

KEEP = (
    "row_id",
    "world_id",
    "assignment",
    "representation",
    "task",
    "prompt",
    "candidates",
    "correct",
    "incorrect",
    "owner",
    "family",
    "correct_actor_position",
    "correct_value_position",
)
TRACKED = [
    ROOT / "scripts" / "authinv_phase0.py",
    ROOT / "src" / "authinv" / "eval" / "parse.py",
    ROOT / "src" / "authinv" / "eval" / "metrics.py",
    ROOT / "src" / "authinv" / "eval" / "generation.py",
    ROOT / "src" / "authinv" / "models.py",
    ROOT / "src" / "authinv" / "provenance.py",
]


def build_dataset(cfg: dict, source_run: Path) -> tuple[list[dict], dict]:
    """Verify the source run against the pinned hashes and project its rows."""
    src = cfg["source"]
    problems = []
    for name, key in (("worlds.jsonl", "worlds_sha256"), ("dataset.jsonl", "dataset_sha256")):
        got = provenance.sha256_file(source_run / name)
        if got != src[key]:
            problems.append(f"{name} sha256 {got} != pinned {src[key]}")
    if problems:
        raise SystemExit("source run does not match the pinned hashes: " + "; ".join(problems))
    validation = json.loads((source_run / "validation_report.json").read_text())
    rows = [{k: r.get(k) for k in KEEP} for r in provenance.read_jsonl(source_run / "dataset.jsonl")]
    exp = src["expected"]
    cells = collections.Counter((r["task"], r["representation"], r["assignment"]) for r in rows)
    checks = {
        "source_validation_passed": validation.get("passed") is True,
        "rows": len(rows) == exp["rows"],
        "worlds": len({r["world_id"] for r in rows}) == exp["worlds"],
        "balanced_cells": len(cells) == exp["tasks"] * exp["representations"] * exp["assignments"]
        and len(set(cells.values())) == 1,
        "unique_row_ids": len({r["row_id"] for r in rows}) == len(rows),
        "two_candidates_correct_listed": all(
            len(r["candidates"]) == 2
            and r["correct"] in r["candidates"]
            and r["incorrect"] in r["candidates"]
            for r in rows
        ),
    }
    report = {"passed": all(checks.values()), "checks": checks, "source_run": src["run_name"]}
    if not report["passed"]:
        raise SystemExit(f"Phase-0 dataset validation failed: {checks}")
    return rows, report


def analyze(rows: list[dict], cfg: dict) -> dict:
    a = cfg["analysis"]
    out = {"parser_version": PARSER_VERSION, "tasks": {}}
    for task in sorted({r["task"] for r in rows}):
        out["tasks"][task] = metrics.task_metrics(rows, task, a["bootstrap_replicates"], a["bootstrap_seed"])
    lenient = [{**r, "category": _lenient_category(r)} for r in rows]
    out["lenient_tasks"] = {
        task: metrics.task_metrics(lenient, task, a["bootstrap_replicates"], a["bootstrap_seed"])
        for task in sorted({r["task"] for r in rows})
    }
    out["category_counts"] = {
        task: dict(collections.Counter(r["category"] for r in rows if r["task"] == task))
        for task in sorted({r["task"] for r in rows})
    }
    return out


def _lenient_category(r: dict) -> str:
    if r["category"] in ("correct", "incorrect"):
        return r["category"]
    if r.get("lenient_selected"):
        return "correct" if r["lenient_selected"] == r["correct"] else "incorrect"
    return r["category"]


def report_md(model_key: str, spec: dict, m: dict, task_order=("application", "interpretation")) -> str:
    quant = spec["quantization"] or spec["precision"]
    lines = [f"# Phase 0 — {model_key} ({spec['name']}@{spec['revision'][:12]}, {quant})", ""]
    for task in task_order:
        t = m["tasks"][task]
        lines += [
            f"## {task}",
            "",
            "| rendering | accuracy (95% CI) | fail-open | fail-closed | parse failure |",
            "|---|---:|---:|---:|---:|",
        ]
        for rep, v in sorted(t["per_representation"].items()):
            lines.append(
                f"| {rep} | {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | "
                f"{v['fail_open_rate']['mean']:.3f} | {v['fail_closed_rate']['mean']:.3f} | "
                f"{v['parse_failure_rate']['mean']:.3f} |"
            )
        g = t["worst_case_gap"]
        lines += [
            "",
            f"- best {t['best_representation']} {t['best_accuracy']:.3f}; worst {t['worst_representation']} "
            f"{t['worst_case_accuracy']:.3f}; gap {100 * g['mean']:.1f} pp "
            f"[{100 * g['ci95'][0]:.1f}, {100 * g['ci95'][1]:.1f}] (bootstrap p={g['p_bootstrap']:.4f})",
            f"- all renderings correct: {t['all_renderings_correct']['mean']:.3f}; "
            f"rendering disagreement: {t['rendering_disagreement']['mean']:.3f} "
            f"[{t['rendering_disagreement']['ci95'][0]:.3f}, {t['rendering_disagreement']['ci95'][1]:.3f}]",
            f"- instances with a deny→allow flip: {t['deny_to_allow_count']}; with an allow→deny flip: "
            f"{t['allow_to_deny_count']} (of {t['instances']})",
            "",
        ]
    lines.append(
        "CIs resample worlds (each world averages its two name assignments). Behavioral evidence only."
    )
    return "\n".join(lines) + "\n"


def cmd_run(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    spec = models.model_spec(args.model)
    rows, validation = build_dataset(cfg, args.source_run)
    if args.resume:
        run = args.resume
        old = json.loads((run / "config.json").read_text())
        meta = json.loads((run / "dataset_metadata.json").read_text())
        mismatch = [
            k
            for k, ok in {
                "config": old["config_sha256"] == provenance.sha256_json(cfg),
                "model": old["model"] == spec,
                "dataset": meta["sha256"] == provenance.sha256_json(rows),
                "parser": old["parser_version"] == PARSER_VERSION,
            }.items()
            if not ok
        ]
        if mismatch:
            raise SystemExit(f"--resume refused: provenance differs ({mismatch})")
    else:
        run = args.output_root / f"{provenance.utc_stamp()}_authinv_phase0_{args.model}"
        run.mkdir(parents=True, exist_ok=False)
        provenance.write_json(
            run / "config.json",
            {
                "experiment": cfg["experiment"],
                "config": cfg,
                "config_sha256": provenance.sha256_json(cfg),
                "model": spec,
                "parser_version": PARSER_VERSION,
                "source_run": str(args.source_run),
                "smoke_limit": args.limit,
                "command": sys.argv,
            },
        )
        provenance.write_jsonl(run / "dataset.jsonl", rows)
        provenance.write_json(
            run / "dataset_metadata.json",
            {
                "sha256": provenance.sha256_json(rows),
                "file_sha256": provenance.sha256_file(run / "dataset.jsonl"),
                "rows": len(rows),
            },
        )
        provenance.write_json(run / "validation_report.json", validation)
    total = len(rows)
    if args.dataset_only:
        provenance.write_run_status(run, "dataset_only", 0, total)
        provenance.finalize(run)
        print(run)
        return

    done_path = run / "predictions.jsonl"
    done = {r["row_id"] for r in provenance.read_jsonl(done_path)} if done_path.exists() else set()
    todo = [r for r in rows if r["row_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    try:
        verification = None
        if args.model_dir and "upstream_manifest" in spec:
            manifest = json.loads((models.REGISTRY.parent / spec["upstream_manifest"]).read_text())
            verification = models.verify_model_dir(args.model_dir, manifest)
            provenance.write_json(run / "model_dir_verification.json", verification)
            if not verification["passed"]:
                raise SystemExit(f"model dir does not match the pinned upstream: {verification}")
        from authinv.eval.generation import VLLMChat

        engine = {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "max_num_seqs": args.max_num_seqs,
        }
        overrides = cfg.get("generation_overrides", {}).get(args.model, {})  # frozen per-model settings
        gen = {**cfg["generation"], **overrides, **engine}  # engine knobs are recorded, not frozen config
        chat = VLLMChat(spec, str(args.model_dir) if args.model_dir else None, gen, args.tensor_parallel)
        provenance.write_json(
            run / "metadata.json",
            {
                "model": spec,
                "model_dir": str(args.model_dir) if args.model_dir else None,
                "model_dir_verified": bool(verification and verification["passed"]),
                "runtime": chat.provenance(),
                "engine": {**engine, "tensor_parallel": args.tensor_parallel},
                "generation": gen,
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                "dataset_sha256": provenance.sha256_json(rows),
                "command": sys.argv,
                "parser_version": PARSER_VERSION,
            },
        )
        provenance.write_run_status(run, "running", len(done), total)
        step = cfg["generation"]["batch_size"]
        with done_path.open("a") as out:
            for start in range(0, len(todo), step):
                chunk = todo[start : start + step]
                for row, gen in zip(chunk, chat.generate([r["prompt"] for r in chunk]), strict=True):
                    parsed = parse_answer(gen["text"], row["candidates"], row["correct"])
                    out.write(json.dumps({**row, **gen, **parsed}, sort_keys=True) + "\n")
                out.flush()
                provenance.write_run_status(run, "running", len(done) + start + len(chunk), total)
        preds = provenance.read_jsonl(done_path)
        if args.limit:  # smoke runs never produce metrics
            provenance.write_run_status(run, "smoke", len(preds), total)
            provenance.finalize(run)
            print(done_path)
            return
        m = analyze(preds, cfg)
        provenance.write_json(run / "metrics.json", m)
        (run / "report.md").write_text(report_md(args.model, spec, m))
        provenance.write_run_status(run, "complete", len(preds), total)
        provenance.finalize(run)
        print(run / "report.md")
    except BaseException as error:
        n = len(provenance.read_jsonl(done_path)) if done_path.exists() else 0
        provenance.write_run_status(run, "failed", n, total, repr(error))
        provenance.finalize(run)
        raise


def cmd_aggregate(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    per_model, task = {}, cfg["analysis"]["primary_task"]
    for run in args.runs:
        conf = json.loads((run / "config.json").read_text())
        status = json.loads((run / "run_status.json").read_text())
        if status["status"] != "complete" or conf["config_sha256"] != provenance.sha256_json(cfg):
            raise SystemExit(f"{run}: not complete or config differs")
        per_model[conf["model"]["key"]] = json.loads((run / "metrics.json").read_text())
    # Holm family: the large models only (docs/prereg/phase0.md).
    pvals = {
        k: m["tasks"][task]["worst_case_gap"]["p_bootstrap"]
        for k, m in per_model.items()
        if k in cfg["large_models"]
    }
    parser_sensitive = {
        k: max(v["parse_failure_rate"]["mean"] for v in m["tasks"][task]["per_representation"].values())
        > 0.05
        for k, m in per_model.items()
    }
    g = cfg.get("gate0")  # absent for the addendum: reported outside the Gate-0 rule
    result = {
        "primary_task": task,
        "holm_adjusted_gap_p": metrics.holm(pvals),
        "gate0": metrics.gate0(
            {k: m["tasks"][task] for k, m in per_model.items()},
            cfg["large_models"],
            g["worst_case_gap_pp"],
            g["rendering_disagreement"],
        )
        if g
        else None,
        "parser_sensitive": parser_sensitive,
        "runs": [str(r) for r in args.runs],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        provenance.write_json(args.out, result)
    if args.markdown:
        args.markdown.write_text(aggregate_md(per_model, result, cfg))


def _pct(v: dict) -> str:
    return f"{100 * v['mean']:.1f} [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"


def aggregate_md(per_model: dict, result: dict, cfg: dict) -> str:
    """Cross-model tables for docs/experiments (strict parser; lenient as sensitivity)."""
    out = []
    for task in ("application", "interpretation"):
        reps = sorted(next(iter(per_model.values()))["tasks"][task]["per_representation"])
        out += [
            f"### {task}: accuracy % by rendering (95% CI)",
            "",
            "| model | " + " | ".join(reps) + " |",
            "|---|" + "---:|" * len(reps),
        ]
        for key, m in per_model.items():
            pr = m["tasks"][task]["per_representation"]
            out.append(f"| {key} | " + " | ".join(_pct(pr[r]) for r in reps) + " |")
        out += [
            "",
            f"### {task}: invariance summary",
            "",
            "| model | worst (rendering) | gap pp | Holm p | all-correct % | disagreement % | "
            "deny→allow inst. | allow→deny inst. | lenient gap pp |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key, m in per_model.items():
            t, lt = m["tasks"][task], m["lenient_tasks"][task]
            holm_p = result["holm_adjusted_gap_p"].get(key) if task == result["primary_task"] else None
            out.append(
                f"| {key} | {100 * t['worst_case_accuracy']:.1f} ({t['worst_representation']}) | "
                f"{_pct(t['worst_case_gap'])} | {'—' if holm_p is None else f'{holm_p:.4f}'} | "
                f"{_pct(t['all_renderings_correct'])} | {_pct(t['rendering_disagreement'])} | "
                f"{t['deny_to_allow_count']} | {t['allow_to_deny_count']} | {_pct(lt['worst_case_gap'])} |"
            )
        out.append("")
    g = result["gate0"]
    if g is None:
        out += ["Not a Gate-0 input (addendum; see its preregistration).", ""]
        return "\n".join(out)
    out += [
        "### Gate 0 (application, strict parser)",
        "",
        f"Rule: {g['rule']}.",
        "",
        "| large model | gap criterion | disagreement criterion | passes | parser-sensitive |",
        "|---|---|---|---|---|",
    ]
    for key, h in g["models"].items():
        if h is None:
            out.append(f"| {key} | not run | | | |")
        else:
            out.append(
                f"| {key} | {h['gap_criterion']} | {h['disagreement_criterion']} | {h['passes']} | "
                f"{result['parser_sensitive'].get(key)} |"
            )
    out += ["", f"**Recommendation: {g['recommendation']}** (for Jerzy's decision in G0).", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--source-run", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--model-dir", type=Path)
    r.add_argument("--tensor-parallel", type=int, default=1)
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.add_argument("--dataset-only", action="store_true")
    r.add_argument("--resume", type=Path)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--enforce-eager", action="store_true")
    r.add_argument(
        "--max-num-seqs", type=int, help="vLLM batch cap (hybrid Mamba models need <= their cache blocks)"
    )
    r.add_argument("--limit", type=int, help="engineering smoke test: first N rows only; never aggregated")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("aggregate")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--runs", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path)
    a.add_argument("--markdown", type=Path)
    a.set_defaults(func=cmd_aggregate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
