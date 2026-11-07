"""Batch loop shared by the Phase-3 arm runners: resumable JSONL output plus a per-batch timing log.

`chat` is anything with ``generate(prompts, schemas=None, max_tokens=None) -> list[dict]``
returning at least ``text``, ``finish_reason``, ``prompt_tokens`` and ``output_tokens``
per prompt (`authinv.eval.generation.VLLMChat`, or a fake in tests).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from authinv import provenance
from authinv.eval.structured import JSON_PARSER_VERSION


def done_ids(path: Path, id_key: str) -> set[str]:
    return {r[id_key] for r in provenance.read_jsonl(path)} if path.exists() else set()


def run_stage(
    chat,
    items: list[dict],
    out_path: Path,
    timing_path: Path,
    *,
    stage: str,
    id_key: str,
    batch_size: int,
    score: Callable[[dict, dict], dict],
    schema_of: Callable[[dict], dict | None] | None = None,
    max_tokens: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Generate for every item not yet in `out_path`; returns the number generated now.

    Each output row is ``{**item, **generation, **score(item, generation)}``; the
    timing log gets one line per batch (stage, requests, seconds, tokens).
    """
    done = done_ids(out_path, id_key)
    todo = [x for x in items if x[id_key] not in done]
    with out_path.open("a") as out, timing_path.open("a") as tlog:
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            schemas = [schema_of(x) for x in chunk] if schema_of else None
            t0 = time.perf_counter()
            gens = chat.generate([x["prompt"] for x in chunk], schemas, max_tokens=max_tokens)
            seconds = time.perf_counter() - t0
            for x, g in zip(chunk, gens, strict=True):
                out.write(json.dumps({**x, **g, **score(x, g)}, sort_keys=True) + "\n")
            out.flush()
            tlog.write(
                json.dumps(
                    {
                        "stage": stage,
                        "requests": len(chunk),
                        "seconds": seconds,
                        "prompt_tokens": sum(g.get("prompt_tokens") or 0 for g in gens),
                        "output_tokens": sum(g.get("output_tokens") or 0 for g in gens),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            tlog.flush()
            if progress:
                progress(len(done) + start + len(chunk))
    return len(todo)


def load_baseline(run: Path, spec: dict, dataset_sha: str) -> tuple[list[dict], dict]:
    """A Phase-2 T2 run usable as an arm's baseline: complete, same dataset, model spec and parser."""
    conf = json.loads((run / "config.json").read_text())
    status = json.loads((run / "run_status.json").read_text())["status"]
    problems = []
    if status != "complete":
        problems.append(f"status {status}")
    if conf.get("dataset_sha256") != dataset_sha:
        problems.append("dataset differs")
    if conf.get("model") != spec:
        problems.append("model spec differs")
    if conf.get("parser_version") != JSON_PARSER_VERSION:
        problems.append(f"parser {conf.get('parser_version')}")
    if problems:
        raise SystemExit(f"baseline {run} refused: {problems}")
    path = run / "predictions.jsonl"
    return provenance.read_jsonl(path), {"run": str(run), "predictions_sha256": provenance.sha256_file(path)}
