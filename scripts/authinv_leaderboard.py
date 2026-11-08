#!/usr/bin/env python3
"""Leaderboard (D1.1, RESEARCH_PLAN §6 deliverable 2): the worst-case table for any Hugging Face model.

  python scripts/authinv_leaderboard.py run --model ORG/NAME [--revision REV] --data DIR \
      [--thinking] [--harmony] [--tensor-parallel N] [--gpu-memory-utilization F] [--max-num-seqs N]
      [--limit N]
  python scripts/authinv_leaderboard.py table --runs RUN [RUN ...]
      [--phase2 configs/authinv/phase2_figures.yaml] [--out FILE.md]

`run` evaluates the model on the frozen application prompts with the frozen T2 settings
(configs/authinv/t2.yaml: greedy, 256 tokens, the json-strict-v3 parser; 4,096 tokens with
--thinking or --harmony, as for the Phase-2 reasoning models). DIR is a release built by
scripts/authinv_release.py (verified against its MANIFEST.json) or the frozen benchmark directory.
The revision is resolved to a commit sha and pinned in the run. Needs vLLM 0.29.0 (the pinned image).
`table` turns leaderboard runs, and optionally the paper's Phase-2 T2 runs, into the worst-case table.
`--limit` is a smoke test: such runs are marked and never ranked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import provenance  # noqa: E402
from authinv.eval import invariance  # noqa: E402
from authinv.eval.structured import JSON_PARSER_VERSION, parse_structured  # noqa: E402
from authinv.render.renderers import RENDERINGS  # noqa: E402

SHA = re.compile(r"^[0-9a-f]{40}$")


def resolve_revision(model: str, revision: str | None) -> str:
    """A commit sha: given, or resolved from the Hub (a branch or tag name is never stored)."""
    if revision and SHA.match(revision):
        return revision
    from huggingface_hub import HfApi

    return HfApi().model_info(model, revision=revision or "main").sha


def load_rows(data: Path) -> tuple[list[dict], str]:
    if (data / "MANIFEST.json").exists():
        from authinv.release import verify_release

        m = verify_release(data)
        return provenance.read_jsonl(data / "data" / "application.jsonl"), m["benchmark_sha256"]
    from authinv.eval.dataset import load_dataset

    rows, v = load_dataset(data)
    return [r for r in rows if r["task"] == "application"], v["manifest"]["sha256"]


def evaluate(chat, rows: list[dict], batch_size: int, out_path: Path) -> list[dict]:
    with out_path.open("w") as out:
        preds = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            for r, g in zip(chunk, chat.generate([x["prompt"] for x in chunk]), strict=True):
                p = {
                    **{k: v for k, v in r.items() if k != "prompt"},
                    **g,
                    **parse_structured(g["text"], "application", r["expected"]),
                }
                preds.append(p)
                out.write(json.dumps(p, sort_keys=True) + "\n")
    return preds


def leaderboard_row(model: str, revision: str, m: dict, smoke: bool) -> dict:
    pr = m["per_rendering"]
    return {
        "model": model,
        "revision": revision,
        "smoke": smoke,
        "worst_case_accuracy": m["worst_case_accuracy"],
        "worst_rendering": m["worst_rendering"],
        "best_rendering": m["best_rendering"],
        "worst_case_gap": m["worst_case_gap"]["mean"],
        "gap_ci95": m["worst_case_gap"]["ci95"],
        "accuracy": {g: pr[g]["accuracy"]["mean"] for g in pr},
        "deny_to_allow": {g: pr[g]["deny_to_allow_rate"]["mean"] for g in pr},
        "non_answer": {
            g: sum(pr[g][f"{c}_rate"]["mean"] for c in ("refusal", "abstain", "parse_failure")) for g in pr
        },
        "rendering_disagreement": m["rendering_disagreement"]["mean"],
    }


def cmd_run(args) -> None:
    t2 = yaml.safe_load((ROOT / "configs/authinv/t2.yaml").read_text())
    rows, dataset_sha = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    revision = resolve_revision(args.model, args.revision)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", args.model)
    spec = {
        "key": slug,
        "name": args.model,
        "revision": revision,
        "precision": args.dtype,
        "quantization": None,
    }
    if args.thinking is not None:
        spec["thinking"] = args.thinking
    if args.harmony:
        spec["output_format"] = "harmony"
    gen = dict(t2["generation"])
    if args.harmony or args.thinking:
        gen.update(max_tokens=4096, max_model_len=12288)
    gen.update(
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )
    run = args.output_root / f"{provenance.utc_stamp()}_authinv_leaderboard_{slug}"
    run.mkdir(parents=True, exist_ok=False)
    provenance.write_json(
        run / "config.json",
        {
            "model": spec,
            "generation": gen,
            "dataset_sha256": dataset_sha,
            "parser_version": JSON_PARSER_VERSION,
            "smoke_limit": args.limit,
            "command": sys.argv,
        },
    )
    from authinv.eval.generation import VLLMChat

    chat = VLLMChat(spec, None, gen, args.tensor_parallel)
    provenance.write_json(
        run / "metadata.json", {"runtime": chat.provenance(), "git": provenance.git_state()}
    )
    finish(
        run,
        args.model,
        revision,
        evaluate(chat, rows, gen["batch_size"], run / "predictions.jsonl"),
        t2,
        bool(args.limit),
    )
    print(run)


def finish(run: Path, model: str, revision: str, preds: list[dict], t2: dict, smoke: bool) -> dict:
    a = t2["analysis"]
    m = invariance.rendering_metrics(preds, "application", a["bootstrap_replicates"], a["bootstrap_seed"])
    provenance.write_json(run / "metrics.json", {"application": m})
    row = leaderboard_row(model, revision, m, smoke)
    provenance.write_json(run / "leaderboard_row.json", row)
    provenance.write_run_status(run, "smoke" if smoke else "complete", len(preds), len(preds))
    provenance.finalize(run)
    return row


def table(rows: list[dict]) -> str:
    rows = sorted((r for r in rows if not r.get("smoke")), key=lambda r: -r["worst_case_accuracy"])
    head = ["model", "worst-case acc.", "worst rendering", "gap (95% CI)", *RENDERINGS, "deny→allow (worst)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lo, hi = r["gap_ci95"]
        cells = [
            f"{r['model']}@{r['revision'][:7]}",
            f"{100 * r['worst_case_accuracy']:.1f}",
            r["worst_rendering"],
            f"{100 * r['worst_case_gap']:.1f} ({100 * lo:.1f}–{100 * hi:.1f})",
            *(f"{100 * r['accuracy'][g]:.1f}" for g in RENDERINGS),
            f"{100 * r['deny_to_allow'][r['worst_rendering']]:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def phase2_rows(config: Path) -> list[dict]:
    from authinv import models

    out = []
    for m in yaml.safe_load(config.read_text())["models"]:
        run = ROOT / m["t2"]
        app = json.loads((run / "metrics.json").read_text())["tasks"]["application"]
        spec = models.model_spec(m["key"])
        out.append(leaderboard_row(m["label"], spec["revision"], app, False))
    return out


def cmd_table(args) -> None:
    rows = [json.loads((r / "leaderboard_row.json").read_text()) for r in args.runs or []]
    if args.phase2:
        rows += phase2_rows(args.phase2)
    text = table(rows)
    if args.out:
        args.out.write_text(text)
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help="a Hugging Face model id")
    r.add_argument("--revision", help="commit sha, branch or tag; resolved to a sha and pinned")
    r.add_argument("--data", type=Path, required=True, help="a release directory or the frozen benchmark")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    r.add_argument("--harmony", action="store_true", help="gpt-oss style output: score the final channel")
    r.add_argument("--tensor-parallel", type=int, default=1)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--max-num-seqs", type=int)
    r.add_argument("--enforce-eager", action="store_true")
    r.add_argument("--limit", type=int, help="smoke test: first N rows; never ranked")
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("table")
    t.add_argument("--runs", type=Path, nargs="*")
    t.add_argument(
        "--phase2", type=Path, help="add the paper's Phase-2 T2 runs (configs/authinv/phase2_figures.yaml)"
    )
    t.add_argument("--out", type=Path)
    t.set_defaults(func=cmd_table)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
