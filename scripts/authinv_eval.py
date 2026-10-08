#!/usr/bin/env python3
"""Tier-T2 evaluation: structured generation on a frozen authinv dataset (P2.2).

  python scripts/authinv_eval.py run --config configs/authinv/t2.yaml --dataset DIR --model KEY
        [--model-dir DIR] [--tensor-parallel N] [--resume RUN] [--dataset-only] [--tokenizer-audit-only]
  python scripts/authinv_eval.py aggregate --config configs/authinv/t2.yaml --runs RUN [RUN ...] [--out F]

DIR is a frozen dataset directory built by P1.10: dataset.jsonl plus dataset_manifest.json
({"sha256": ..., "rows": ...}). Each row carries: row_id, world_id, assignment, instance,
rendering, task (application | interpretation), prompt, expected (decision or principal ids),
label (application), pair_key, and principal_ids (interpretation).
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
from authinv.eval import invariance, metrics  # noqa: E402
from authinv.eval.structured import JSON_PARSER_VERSION, answer_schema, parse_structured  # noqa: E402

REQUIRED = (
    "row_id",
    "world_id",
    "assignment",
    "instance",
    "rendering",
    "task",
    "prompt",
    "expected",
    "pair_key",
)
TRACKED = [ROOT / "scripts" / "authinv_eval.py"] + [
    ROOT / "src" / "authinv" / p
    for p in (
        "eval/structured.py",
        "eval/invariance.py",
        "eval/generation.py",
        "eval/parse.py",
        "models.py",
        "provenance.py",
    )
]


def load_dataset(path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((path / "dataset_manifest.json").read_text())
    got = provenance.sha256_file(path / "dataset.jsonl")
    if got != manifest["sha256"]:
        raise SystemExit(f"dataset.jsonl sha256 {got} != manifest {manifest['sha256']}")
    rows = provenance.read_jsonl(path / "dataset.jsonl")
    problems = []
    if len(rows) != manifest["rows"]:
        problems.append("row count")
    if len({r["row_id"] for r in rows}) != len(rows):
        problems.append("duplicate row_id")
    missing = {k for r in rows for k in REQUIRED if k not in r}
    if missing:
        problems.append(f"missing fields {sorted(missing)}")
    if any(r["task"] == "application" and r.get("label") not in ("allow", "deny") for r in rows):
        problems.append("application row without allow/deny label")
    cells = collections.Counter((r["world_id"], r["assignment"], r["instance"], r["task"]) for r in rows)
    renderings = {r["rendering"] for r in rows}
    if any(n != len(renderings) for n in cells.values()):
        problems.append("an instance is not rendered in every rendering")
    if problems:
        raise SystemExit(f"dataset validation failed: {problems}")
    return rows, {"passed": True, "rows": len(rows), "renderings": sorted(renderings), "manifest": manifest}


def analyze(preds: list[dict], cfg: dict) -> dict:
    a = cfg["analysis"]
    tasks = sorted({r["task"] for r in preds})
    out = {
        "parser_version": JSON_PARSER_VERSION,
        "tasks": {
            t: invariance.rendering_metrics(preds, t, a["bootstrap_replicates"], a["bootstrap_seed"])
            for t in tasks
        },
        "category_counts": {
            t: dict(collections.Counter(r["category"] for r in preds if r["task"] == t)) for t in tasks
        },
    }
    if "application" in tasks:
        out["rendering_contrasts"] = invariance.rendering_contrasts(
            preds, "application", a["bootstrap_replicates"], a["bootstrap_seed"]
        )
        for kind in sorted({r.get("source_kind") for r in preds} - {None}):
            sub = [r for r in preds if r.get("source_kind") == kind]
            out.setdefault("by_source_kind", {})[kind] = invariance.rendering_metrics(
                sub, "application", a["bootstrap_replicates"], a["bootstrap_seed"]
            )
    if {"application", "interpretation"} <= set(tasks):
        out["dissociation"] = invariance.dissociation(preds, a["bootstrap_replicates"], a["bootstrap_seed"])
    return out


def tokenizer_audit(rows: list[dict], spec: dict, model_dir: Path | None) -> dict:
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(
        str(model_dir) if model_dir else spec["name"], revision=None if model_dir else spec["revision"]
    )
    kwargs = {"enable_thinking": bool(spec["thinking"])} if "thinking" in spec else {}
    lengths = collections.defaultdict(list)
    for r in rows:
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, **kwargs
        )
        lengths[r["rendering"]].append(len(tok(text, add_special_tokens=False)["input_ids"]))
    return {
        g: {"min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v), "mean": sum(v) / len(v)}
        for g, v in sorted(lengths.items())
    }


def cmd_run(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    spec = models.model_spec(args.model)
    rows, validation = load_dataset(args.dataset)
    dataset_sha = validation["manifest"]["sha256"]
    if args.resume:
        run = args.resume
        old = json.loads((run / "config.json").read_text())
        if (old["config_sha256"], old["model"], old["dataset_sha256"], old["parser_version"]) != (
            provenance.sha256_json(cfg),
            spec,
            dataset_sha,
            JSON_PARSER_VERSION,
        ):
            raise SystemExit("--resume refused: config, model, dataset, or parser differs")
    else:
        run = args.output_root / f"{provenance.utc_stamp()}_authinv_t2_{args.model}"
        run.mkdir(parents=True, exist_ok=False)
        provenance.write_json(
            run / "config.json",
            {
                "experiment": cfg["experiment"],
                "config": cfg,
                "config_sha256": provenance.sha256_json(cfg),
                "model": spec,
                "dataset": str(args.dataset),
                "dataset_sha256": dataset_sha,
                "parser_version": JSON_PARSER_VERSION,
                "command": sys.argv,
            },
        )
        provenance.write_json(run / "validation_report.json", validation)
    total = len(rows)
    if args.dataset_only:
        provenance.write_run_status(run, "dataset_only", 0, total)
        provenance.finalize(run)
        print(run)
        return
    if args.tokenizer_audit_only:
        provenance.write_json(run / "tokenizer_audit.json", tokenizer_audit(rows, spec, args.model_dir))
        provenance.write_run_status(run, "tokenizer_audit_only", 0, total)
        provenance.finalize(run)
        print(run / "tokenizer_audit.json")
        return
    done_path = run / "predictions.jsonl"
    done = {r["row_id"] for r in provenance.read_jsonl(done_path)} if done_path.exists() else set()
    todo = [r for r in rows if r["row_id"] not in done]
    if args.limit:  # engineering smoke test only; never analysed or aggregated
        todo = todo[: args.limit]
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
        provenance.write_json(
            run / "metadata.json",
            {
                "model": spec,
                "runtime": chat.provenance(),
                "generation": gen,
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                "dataset_sha256": dataset_sha,
                "command": sys.argv,
                "parser_version": JSON_PARSER_VERSION,
            },
        )
        provenance.write_run_status(run, "running", len(done), total)
        constrained = cfg["generation"].get("constrained", False)
        with done_path.open("a") as out:
            for start in range(0, len(todo), gen["batch_size"]):
                chunk = todo[start : start + gen["batch_size"]]
                schemas = (
                    [answer_schema(r["task"], r.get("principal_ids")) for r in chunk] if constrained else None
                )
                for row, g in zip(chunk, chat.generate([r["prompt"] for r in chunk], schemas), strict=True):
                    out.write(
                        json.dumps(
                            {**row, **g, **parse_structured(g["text"], row["task"], row["expected"])},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                out.flush()
                provenance.write_run_status(run, "running", len(done) + start + len(chunk), total)
        preds = provenance.read_jsonl(done_path)
        if args.limit:
            provenance.write_run_status(run, "smoke", len(preds), total)
            provenance.finalize(run)
            print(run)
            return
        provenance.write_json(run / "metrics.json", analyze(preds, cfg))
        provenance.write_run_status(run, "complete", len(preds), total)
        provenance.finalize(run)
        print(run)
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
        if json.loads((run / "run_status.json").read_text())["status"] != "complete":
            raise SystemExit(f"{run}: not complete")
        if conf["config_sha256"] != provenance.sha256_json(cfg):
            raise SystemExit(f"{run}: config differs")
        per_model[conf["model"]["key"]] = json.loads((run / "metrics.json").read_text())
    pvals = {k: m["tasks"][task]["worst_case_gap"]["p_bootstrap"] for k, m in per_model.items()}
    # Holm family (docs/prereg/phase2.md): each (model, rendering) contrast vs the other renderings.
    family = {
        f"{k}|{g}": v["p_bootstrap"]
        for k, m in per_model.items()
        for g, v in m.get("rendering_contrasts", {}).items()
    }
    params = {k: models.model_spec(k)["params_b"] for k in per_model}
    scaling = invariance.spearman_with_permutation(
        [params[k] for k in sorted(per_model)],
        [per_model[k]["tasks"][task]["worst_case_gap"]["mean"] for k in sorted(per_model)],
    )
    quant = {}
    for base, q in cfg.get("quantization_pairs", {}).items():
        if base in per_model and q in per_model:
            b, qq = per_model[base]["tasks"][task], per_model[q]["tasks"][task]
            delta = {
                g: qq["per_rendering"][g]["accuracy"]["mean"] - b["per_rendering"][g]["accuracy"]["mean"]
                for g in b["per_rendering"]
            }
            quant[f"{base}->{q}"] = {
                "accuracy_delta": delta,
                "max_abs_delta": max(abs(v) for v in delta.values()),
                "base_worst_case_gap": b["worst_case_gap"]["mean"],
                "exceeds_rendering_effect": max(abs(v) for v in delta.values())
                >= b["worst_case_gap"]["mean"],
            }
    result = {
        "primary_task": task,
        "holm_adjusted_gap_p": metrics.holm(pvals),
        "holm_adjusted_rendering_contrasts": metrics.holm(family) if family else {},
        "scaling_worst_case_gap_vs_params": scaling,
        "quantization_control": quant,
        "worst_case": {
            k: {"worst": m["tasks"][task]["worst_rendering"], "gap": m["tasks"][task]["worst_case_gap"]}
            for k, m in per_model.items()
        },
        "runs": [str(r) for r in args.runs],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        provenance.write_json(args.out, result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--model-dir", type=Path)
    r.add_argument("--tensor-parallel", type=int, default=1)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--enforce-eager", action="store_true")
    r.add_argument("--max-num-seqs", type=int)
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.add_argument("--resume", type=Path)
    r.add_argument("--dataset-only", action="store_true")
    r.add_argument("--tokenizer-audit-only", action="store_true")
    r.add_argument("--limit", type=int, help="engineering smoke test: first N rows only; never analysed")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("aggregate")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--runs", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path)
    a.set_defaults(func=cmd_aggregate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
