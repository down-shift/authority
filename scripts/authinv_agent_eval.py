#!/usr/bin/env python3
"""Tier-T3 evaluation: episodes in the MCP-style tool sandbox over the frozen benchmark (P2.7).

  python scripts/authinv_agent_eval.py run --config configs/authinv/t3.yaml --benchmark DIR --model KEY
        [--model-dir DIR] [--tensor-parallel N] [--enforce-eager] [--resume RUN] [--max-episodes N]
        [--dataset-only]
  python scripts/authinv_agent_eval.py aggregate --config configs/authinv/t3.yaml --runs T3RUN... \
        --t2-runs T2RUN... [--out FILE]

DIR is the frozen Phase-1 benchmark (P1.10). Episodes reuse its worlds, assignments, and requests
(re-derived and checked against the frozen rows). Writes episodes.jsonl (transcripts + outcomes),
metrics.json, metadata, run status, and the artifact manifest under outputs/. `aggregate` applies the
T3 Holm family and the T2->T3 transfer analysis (docs/prereg/phase2-t3.md) to complete runs.
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
from authinv.agent.episodes import build_episodes  # noqa: E402
from authinv.agent.metrics import agent_metrics, transfer, uir_contrasts  # noqa: E402
from authinv.agent.sandbox import AGENT_VERSION, outcome, run  # noqa: E402
from authinv.eval import invariance, metrics  # noqa: E402

TRACKED = [ROOT / "scripts" / "authinv_agent_eval.py"] + [
    ROOT / "src" / "authinv" / p
    for p in (
        "agent/sandbox.py",
        "agent/episodes.py",
        "agent/metrics.py",
        "eval/generation.py",
        "benchmark.py",
        "provenance.py",
    )
]


def record(ep) -> dict:
    return {
        "episode_id": ep.episode_id,
        "world_id": ep.world_id,
        "assignment": ep.assignment,
        "instance": ep.instance,
        "rendering": ep.rendering,
        "label": ep.label,
        **ep.meta,
        "request": {
            "principal": ep.request.principal.key(),
            "action": ep.request.action,
            "resource": ep.request.resource.key(),
            "context": dict(ep.request.context),
        },
        "calls": ep.calls,
        "messages": ep.messages,
        **outcome(ep),
    }


def model_metrics(recs: list[dict], a: dict) -> dict:
    n, seed = a["bootstrap_replicates"], a["bootstrap_seed"]
    return {
        **agent_metrics(recs, n, seed),
        "uir_contrasts": uir_contrasts(recs, n, seed),
        "by_source_kind": {
            k: agent_metrics([r for r in recs if r["source_kind"] == k], n, seed)
            for k in sorted({r["source_kind"] for r in recs})
        },
    }


def cmd_aggregate(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    a = cfg["analysis"]
    t2_by_model = {}
    for t2_run in args.t2_runs:
        conf = json.loads((t2_run / "config.json").read_text())
        t2_by_model[conf["model"]["key"]] = t2_run
    per_model, cells, trans = {}, [], {}
    for t3_run in args.runs:
        conf = json.loads((t3_run / "config.json").read_text())
        if json.loads((t3_run / "run_status.json").read_text())["status"] != "complete":
            raise SystemExit(f"{t3_run}: not complete")
        if conf["config_sha256"] != provenance.sha256_json(cfg):
            raise SystemExit(f"{t3_run}: config differs")
        key = conf["model"]["key"]
        m = json.loads((t3_run / "metrics.json").read_text())
        per_model[key] = m
        if key not in t2_by_model:
            raise SystemExit(f"{key}: no T2 run given")
        t2_metrics = json.loads((t2_by_model[key] / "metrics.json").read_text())
        t2_rows = provenance.read_jsonl(t2_by_model[key] / "predictions.jsonl")
        trans[key] = transfer(
            provenance.read_jsonl(t3_run / "episodes.jsonl"),
            t2_rows,
            a["bootstrap_replicates"],
            a["bootstrap_seed"],
        )
        t2_pr = t2_metrics["tasks"]["application"]["per_rendering"]
        for g, v in m["per_rendering"].items():
            cells.append((t2_pr[g]["deny_to_allow_rate"]["mean"], v["uir_deny"]["mean"], f"{key}|{g}"))
    family = {
        f"{k}|{g}": v["p_bootstrap"] for k, m in per_model.items() for g, v in m["uir_contrasts"].items()
    }
    result = {
        "holm_adjusted_uir_contrasts": metrics.holm(family),
        "holm_family_size": len(family),
        # cross-tier: T2 deny->allow rate vs T3 UIR on deny requests, over (model, rendering) cells
        "t2_deny_to_allow_vs_t3_uir": {
            **invariance.spearman_with_permutation([c[0] for c in cells], [c[1] for c in cells]),
            "cells": {c[2]: {"t2_deny_to_allow": c[0], "t3_uir_deny": c[1]} for c in cells},
        },
        "transfer": trans,
        "worst_case": {
            k: {"worst": m["worst_uir_rendering"], "best": m["best_uir_rendering"], "gap": m["uir_gap"]}
            for k, m in per_model.items()
        },
        "runs": [str(r) for r in args.runs],
        "t2_runs": {k: str(v) for k, v in t2_by_model.items()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        provenance.write_json(args.out, result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--benchmark", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--model-dir", type=Path)
    r.add_argument("--tensor-parallel", type=int, default=1)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--max-num-seqs", type=int)
    r.add_argument("--enforce-eager", action="store_true")
    r.add_argument("--chunk-episodes", type=int, default=4096, help="checkpoint size (engineering only)")
    r.add_argument("--resume", type=Path)
    r.add_argument("--max-episodes", type=int, help="engineering smoke test only: first N episodes")
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.add_argument("--dataset-only", action="store_true")
    g = sub.add_parser("aggregate")
    g.add_argument("--config", type=Path, required=True)
    g.add_argument("--runs", type=Path, nargs="+", required=True)
    g.add_argument("--t2-runs", type=Path, nargs="+", required=True)
    g.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.cmd == "aggregate":
        cmd_aggregate(args)
        return
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["agent_version"] != AGENT_VERSION:
        raise SystemExit(f"config agent_version {cfg['agent_version']} != code {AGENT_VERSION}")
    spec = models.model_spec(args.model)
    episodes, info = build_episodes(args.benchmark, assignments=tuple(cfg["assignments"]))
    if cfg.get("dataset_sha256") and info["dataset_sha256"] != cfg["dataset_sha256"]:
        raise SystemExit(
            f"benchmark {info['dataset_sha256']} is not the preregistered {cfg['dataset_sha256']}"
        )
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    total = len(episodes)
    if args.resume:
        run_dir = args.resume
        old = json.loads((run_dir / "config.json").read_text())
        if (old["config_sha256"], old["model"], old["episodes"], old["smoke_limit"]) != (
            provenance.sha256_json(cfg),
            spec,
            info,
            args.max_episodes,
        ):
            raise SystemExit("--resume refused: config, model, benchmark, or episode set differs")
    else:
        run_dir = args.output_root / f"{provenance.utc_stamp()}_authinv_t3_{args.model}"
        run_dir.mkdir(parents=True, exist_ok=False)
        provenance.write_json(
            run_dir / "config.json",
            {
                "config": cfg,
                "config_sha256": provenance.sha256_json(cfg),
                "model": spec,
                "benchmark": str(args.benchmark),
                "episodes": info,
                "smoke_limit": args.max_episodes,
                "command": sys.argv,
            },
        )
    if args.dataset_only:
        provenance.write_run_status(run_dir, "dataset_only", 0, total)
        provenance.finalize(run_dir)
        print(run_dir, total)
        return
    done_path = run_dir / "episodes.jsonl"
    done = {r["episode_id"] for r in provenance.read_jsonl(done_path)} if done_path.exists() else set()
    todo = [ep for ep in episodes if ep.episode_id not in done]
    try:
        from authinv.eval.generation import VLLMChat

        gen = {
            **cfg["generation"],
            **cfg.get("generation_overrides", {}).get(args.model, {}),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "max_num_seqs": args.max_num_seqs,
        }
        chat = VLLMChat(spec, str(args.model_dir) if args.model_dir else None, gen, args.tensor_parallel)
        meta_name = "metadata.json" if not done else f"metadata_resume_{provenance.utc_stamp()}.json"
        provenance.write_json(
            run_dir / meta_name,
            {
                "model": spec,
                "runtime": chat.provenance(),
                "generation": gen,
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                "agent_version": AGENT_VERSION,
                "resumed_after": len(done),
                "command": sys.argv,
            },
        )
        provenance.write_run_status(run_dir, "running", len(done), total)
        # Episodes are independent: run them in chunks and append each finished chunk (checkpoint).
        with done_path.open("a") as out:
            for start in range(0, len(todo), args.chunk_episodes):
                chunk = todo[start : start + args.chunk_episodes]
                run(chunk, chat.chat_texts, cfg["max_steps"], gen["batch_size"])
                for ep in chunk:
                    out.write(json.dumps(record(ep), sort_keys=True) + "\n")
                out.flush()
                provenance.write_run_status(run_dir, "running", len(done) + start + len(chunk), total)
        recs = provenance.read_jsonl(done_path)
        if len(recs) != total or len({r["episode_id"] for r in recs}) != total:
            raise RuntimeError(f"episodes.jsonl has {len(recs)} records for {total} episodes")
        if not args.max_episodes:
            provenance.write_json(run_dir / "metrics.json", model_metrics(recs, cfg["analysis"]))
        provenance.write_run_status(run_dir, "smoke" if args.max_episodes else "complete", total, total)
        provenance.finalize(run_dir)
        print(run_dir)
    except BaseException as error:
        n = len(provenance.read_jsonl(done_path)) if done_path.exists() else 0
        provenance.write_run_status(run_dir, "failed", n, total, repr(error))
        provenance.finalize(run_dir)
        raise


if __name__ == "__main__":
    main()
