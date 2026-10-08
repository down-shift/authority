#!/usr/bin/env python3
"""Tier-T1 evaluation: candidate log-probability margins on the frozen benchmark (P2.4, prereg phase2).

  python scripts/authinv_t1_eval.py run --config configs/authinv/t2.yaml --dataset DIR --model KEY
        [--model-dir DIR] [--max-batch-tokens N] [--limit N] [--resume RUN]

Application rows only. Each prompt is chat-templated with thinking disabled; the two complete answers
`{"decision": "allow"}` and `{"decision": "deny"}` are scored as continuations. margin = logp(correct) -
logp(incorrect); T1-correct iff margin > 0. Not defined for reasoning-output models (thinking on, harmony).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import models, provenance  # noqa: E402
from authinv.eval import invariance  # noqa: E402
from authinv.eval.dataset import load_dataset  # noqa: E402

CANDIDATES = {"allow": '{"decision": "allow"}', "deny": '{"decision": "deny"}'}
TRACKED = [
    ROOT / "scripts" / "authinv_t1_eval.py",
    ROOT / "src" / "authinv" / "eval" / "logprob.py",
    ROOT / "src" / "authinv" / "eval" / "invariance.py",
    ROOT / "src" / "authinv" / "eval" / "dataset.py",
]


def t1_applicable(spec: dict) -> bool:
    return not spec.get("thinking") and spec.get("output_format") != "harmony"


def to_prediction(row: dict, scores: dict) -> dict:
    other = "deny" if row["label"] == "allow" else "allow"
    margin = scores[CANDIDATES[row["label"]]]["sum_logprob"] - scores[CANDIDATES[other]]["sum_logprob"]
    tc = {d: scores[CANDIDATES[d]]["token_count"] for d in CANDIDATES}
    return {
        **{
            k: row[k]
            for k in (
                "row_id",
                "world_id",
                "assignment",
                "instance",
                "rendering",
                "task",
                "label",
                "source",
                "source_kind",
                "pair_key",
            )
        },
        "margin": margin,
        "category": "correct" if margin > 0 else "incorrect",
        "value": row["label"] if margin > 0 else other,
        "logp": {d: scores[CANDIDATES[d]]["sum_logprob"] for d in CANDIDATES},
        "token_counts": tc,
        "boundary_overlap": any(scores[CANDIDATES[d]]["boundary_overlap"] for d in CANDIDATES),
    }


def analyze(preds: list[dict], cfg: dict) -> dict:
    a = cfg["analysis"]
    out = {
        "tier": "T1",
        "application": invariance.rendering_metrics(
            preds, "application", a["bootstrap_replicates"], a["bootstrap_seed"]
        ),
        "rendering_contrasts": invariance.rendering_contrasts(
            preds, "application", a["bootstrap_replicates"], a["bootstrap_seed"]
        ),
    }
    out["mean_margin"] = {
        g: sum(p["margin"] for p in preds if p["rendering"] == g)
        / max(1, sum(p["rendering"] == g for p in preds))
        for g in sorted({p["rendering"] for p in preds})
    }
    out["matched_candidate_token_counts"] = all(len(set(p["token_counts"].values())) == 1 for p in preds)
    for kind in sorted({p.get("source_kind") for p in preds} - {None}):
        sub = [p for p in preds if p.get("source_kind") == kind]
        out.setdefault("by_source_kind", {})[kind] = invariance.rendering_metrics(
            sub, "application", a["bootstrap_replicates"], a["bootstrap_seed"]
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--model-dir", type=Path)
    r.add_argument("--max-batch-tokens", type=int, default=32768)
    r.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--max-num-seqs", type=int)
    r.add_argument("--chunk-rows", type=int, default=2048)
    r.add_argument("--limit", type=int, help="engineering smoke test only")
    r.add_argument("--resume", type=Path)
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    spec = models.model_spec(args.model)
    if not t1_applicable(spec):
        raise SystemExit(f"T1 is not defined for {args.model} (reasoning output); see docs/prereg/phase2.md")
    rows, validation = load_dataset(args.dataset)
    sha = validation["manifest"]["sha256"]
    if cfg.get("dataset_sha256") and sha != cfg["dataset_sha256"]:
        raise SystemExit(f"dataset {sha} is not the preregistered benchmark {cfg['dataset_sha256']}")
    rows = [x for x in rows if x["task"] == "application"]
    if args.resume:
        run = args.resume
        old = json.loads((run / "config.json").read_text())
        if (old["config_sha256"], old["model"], old["dataset_sha256"]) != (
            provenance.sha256_json(cfg),
            spec,
            sha,
        ):
            raise SystemExit("--resume refused: config, model, or dataset differs")
    else:
        run = args.output_root / f"{provenance.utc_stamp()}_authinv_t1_{args.model}"
        run.mkdir(parents=True, exist_ok=False)
        provenance.write_json(
            run / "config.json",
            {
                "tier": "T1",
                "config": cfg,
                "config_sha256": provenance.sha256_json(cfg),
                "model": spec,
                "dataset": str(args.dataset),
                "dataset_sha256": sha,
                "candidates": CANDIDATES,
                "smoke_limit": args.limit,
                "backend": args.backend,
                "command": sys.argv,
            },
        )
    done_path = run / "predictions.jsonl"
    done = {p["row_id"] for p in provenance.read_jsonl(done_path)} if done_path.exists() else set()
    todo = [x for x in rows if x["row_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    total = len(rows)
    try:
        mdir = str(args.model_dir) if args.model_dir else None
        if args.backend == "hf":
            from authinv.eval.logprob import HFScorer

            scorer = HFScorer({**spec, "thinking": False} if "thinking" in spec else spec, mdir)
        else:
            from authinv.eval.logprob import VLLMScorer

            gen = {
                **cfg["generation"],
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_seqs": args.max_num_seqs,
            }
            scorer = VLLMScorer(spec, mdir, gen)
        provenance.write_json(
            run / "metadata.json",
            {
                "model": spec,
                "runtime": scorer.provenance(),
                "git": provenance.git_state(),
                "dataset_sha256": sha,
                "source_sha256": provenance.source_hashes(TRACKED),
                "command": sys.argv,
            },
        )
        provenance.write_run_status(run, "running", len(done), total)
        with done_path.open("a") as out:
            for s in range(0, len(todo), args.chunk_rows):
                chunk = todo[s : s + args.chunk_rows]
                scored = scorer.score_many(
                    [(x["prompt"], list(CANDIDATES.values())) for x in chunk], args.max_batch_tokens
                )
                for x, sc in zip(chunk, scored, strict=True):
                    out.write(json.dumps(to_prediction(x, sc), sort_keys=True) + "\n")
                out.flush()
                provenance.write_run_status(run, "running", len(done) + s + len(chunk), total)
        preds = provenance.read_jsonl(done_path)
        if args.limit:
            provenance.write_run_status(run, "smoke", len(preds), total)
        else:
            provenance.write_json(run / "metrics.json", analyze(preds, cfg))
            provenance.write_run_status(run, "complete", len(preds), total)
        provenance.finalize(run)
        print(run)
    except BaseException as error:
        n = len(provenance.read_jsonl(done_path)) if done_path.exists() else 0
        provenance.write_run_status(run, "failed", n, total, repr(error))
        provenance.finalize(run)
        raise


if __name__ == "__main__":
    main()
