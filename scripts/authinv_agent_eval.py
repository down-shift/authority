#!/usr/bin/env python3
"""Tier-T3 evaluation: episodes in the MCP-style tool sandbox over the frozen benchmark (P2.7).

  python scripts/authinv_agent_eval.py run --config configs/authinv/t3.yaml --benchmark DIR --model KEY
        [--model-dir DIR] [--tensor-parallel N] [--max-episodes N] [--dataset-only]

DIR is the frozen Phase-1 benchmark (P1.10). Episodes reuse its worlds, assignments, and requests
(re-derived and checked against the frozen rows). Writes episodes.jsonl (transcripts + outcomes),
metrics.json, metadata, run status, and the artifact manifest under outputs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import models, provenance  # noqa: E402
from authinv.agent.episodes import build_episodes  # noqa: E402
from authinv.agent.metrics import agent_metrics  # noqa: E402
from authinv.agent.sandbox import AGENT_VERSION, outcome, run  # noqa: E402

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
    r.add_argument("--max-episodes", type=int, help="engineering smoke test only: first N episodes")
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.add_argument("--dataset-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["agent_version"] != AGENT_VERSION:
        raise SystemExit(f"config agent_version {cfg['agent_version']} != code {AGENT_VERSION}")
    spec = models.model_spec(args.model)
    episodes, info = build_episodes(args.benchmark, assignments=tuple(cfg["assignments"]))
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
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
        provenance.write_run_status(run_dir, "dataset_only", 0, len(episodes))
        provenance.finalize(run_dir)
        print(run_dir, len(episodes))
        return
    try:
        from authinv.eval.generation import VLLMChat

        gen = {
            **cfg["generation"],
            **cfg.get("generation_overrides", {}).get(args.model, {}),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
        }
        chat = VLLMChat(spec, str(args.model_dir) if args.model_dir else None, gen, args.tensor_parallel)
        provenance.write_json(
            run_dir / "metadata.json",
            {
                "model": spec,
                "runtime": chat.provenance(),
                "generation": gen,
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                "agent_version": AGENT_VERSION,
                "command": sys.argv,
            },
        )
        provenance.write_run_status(run_dir, "running", 0, len(episodes))
        run(episodes, chat.chat_texts, cfg["max_steps"], gen["batch_size"])
        recs = [record(ep) for ep in episodes]
        provenance.write_jsonl(run_dir / "episodes.jsonl", recs)
        if not args.max_episodes:
            a = cfg["analysis"]
            provenance.write_json(
                run_dir / "metrics.json", agent_metrics(recs, a["bootstrap_replicates"], a["bootstrap_seed"])
            )
        provenance.write_run_status(
            run_dir, "smoke" if args.max_episodes else "complete", len(recs), len(recs)
        )
        provenance.finalize(run_dir)
        print(run_dir)
    except BaseException as error:
        provenance.write_run_status(run_dir, "failed", 0, len(episodes), repr(error))
        provenance.finalize(run_dir)
        raise


if __name__ == "__main__":
    main()
