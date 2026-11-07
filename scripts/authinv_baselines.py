#!/usr/bin/env python3
"""Phase-3 baseline arms (P3.2): zero-shot CoT, restate-as-JSON, self-consistency, solver pipeline.

  python scripts/authinv_baselines.py run --arm {cot,restate} --config configs/authinv/p3_baselines.yaml
        --dataset DIR --model KEY [--baseline T2RUN] [--model-dir DIR] [--tensor-parallel N]
        [--resume RUN] [--dataset-only] [--limit N]
  python scripts/authinv_baselines.py derive --arm self_consistency --config ... --t2-run T2RUN
  python scripts/authinv_baselines.py derive --arm solver --config ... --dataset DIR --canon-run RUN
        [--baseline T2RUN]

`run` asks every query of the configured tasks with the frozen T2 prompt, its answer instruction
replaced (src/authinv/mitigate/baselines.py), and writes predictions.jsonl, timing.jsonl and
metrics.json. `derive` builds an arm from frozen runs without a model: self-consistency from one
Phase-2 T2 run (analysis only, not deployable), the solver pipeline from one P3.1 canonicalization
run (the model's conversions, decided by the engine). T2RUN is the Phase-2 T2 run of the same model
(same dataset, model spec and parser); with it, metrics.json adds gap closure and the token ratio.
The enforcement proxy runs in T3: scripts/authinv_agent_eval.py with configs/authinv/p3_proxy_t3.yaml.
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
from authinv.eval import invariance  # noqa: E402
from authinv.eval.dataset import load_dataset  # noqa: E402
from authinv.eval.structured import JSON_PARSER_VERSION  # noqa: E402
from authinv.mitigate import baselines as bl  # noqa: E402
from authinv.mitigate import canonicalize as canon  # noqa: E402
from authinv.mitigate import metrics as mm  # noqa: E402
from authinv.mitigate.prompts import request_of  # noqa: E402
from authinv.mitigate.runner import load_baseline, run_stage  # noqa: E402
from authinv.policy import evaluate, from_dict  # noqa: E402

TRACKED = [ROOT / "scripts" / "authinv_baselines.py"] + [
    ROOT / "src" / "authinv" / p
    for p in (
        "mitigate/baselines.py",
        "mitigate/canonicalize.py",
        "mitigate/prompts.py",
        "mitigate/metrics.py",
        "mitigate/runner.py",
        "eval/structured.py",
        "eval/invariance.py",
        "eval/generation.py",
        "eval/parse.py",
        "models.py",
        "provenance.py",
    )
]


def check_config(cfg: dict) -> None:
    want = {
        "baseline_version": bl.BASELINE_VERSION,
        "final_parser_version": bl.FINAL_PARSER_VERSION,
        "canon_version": canon.CANON_VERSION,
        "parser_version": JSON_PARSER_VERSION,
    }
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit(f"config versions differ from the code: {bad}")


def arm_settings(cfg: dict, arm: str, model: str) -> tuple[dict, dict]:
    over = cfg.get("overrides", {}).get(model, {})
    arm_cfg = {**cfg["arms"][arm], **over.get(arm, {})}
    gen = {**cfg["generation"], **over.get("generation", {})}
    if "max_tokens" in arm_cfg:
        gen["max_tokens"] = arm_cfg["max_tokens"]
    return arm_cfg, gen


def summarize(preds: list[dict], cfg: dict, base: list[dict] | None, timing: list[dict], convs=None) -> dict:
    a = cfg["analysis"]
    reps, seed = a["bootstrap_replicates"], a["bootstrap_seed"]
    tasks = sorted({r["task"] for r in preds})
    out = {
        "tasks": {t: invariance.rendering_metrics(preds, t, reps, seed) for t in tasks},
        "category_counts": {
            t: dict(collections.Counter(r["category"] for r in preds if r["task"] == t)) for t in tasks
        },
        "cost": mm.cost(convs or [], preds, base, timing),
    }
    if "application" in tasks:
        out["rendering_contrasts"] = invariance.rendering_contrasts(preds, "application", reps, seed)
    if base is not None:
        out["closure"] = {t: mm.gap_closure(preds, base, t, reps, seed) for t in tasks}
    return out


class Scorer:
    def __init__(self, arm: str, worlds: dict[str, dict]):
        self.arm, self.worlds, self.policies, self.ir_cache = arm, worlds, {}, {}

    def policy(self, w: str, a: str):
        if (w, a) not in self.policies:
            self.policies[(w, a)] = from_dict(self.worlds[w][a])
        return self.policies[(w, a)]

    def __call__(self, item: dict, g: dict) -> dict:
        out = bl.parse_final(g["text"], item["task"], item["expected"])
        if self.arm != "restate":
            return out
        ir = bl.restated_ir(g["text"])
        out.update(conv_category=ir["conv_category"], ir_valid=False, ir_exact=False, ir_equivalent=False)
        out["ir_request_correct"] = None
        if ir["conv_category"] == "ir":
            w, a = item["world_id"], item["assignment"]
            key = (w, a, ir["ir_text"])
            if key not in self.ir_cache:
                types = (tuple(self.worlds[w]["principal_types"]), tuple(self.worlds[w]["resource_types"]))
                self.ir_cache[key] = canon.score_ir(self.policy(w, a), ir["ir_text"], *types)
            scores, pred = self.ir_cache[key]
            out.update(scores)
            if pred is not None and item["task"] == "application":
                decision = evaluate(pred, request_of(item, self.policy(w, a)))["decision"]
                out["ir_request_correct"] = decision == item["label"]
        return out


def cmd_run(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    check_config(cfg)
    if args.arm not in bl.MODEL_ARMS:
        raise SystemExit(f"--arm {args.arm} is not a model arm; use derive")
    spec = models.model_spec(args.model)
    rows, validation = load_dataset(args.dataset)
    dataset_sha = validation["manifest"]["sha256"]
    if cfg.get("dataset_sha256") and dataset_sha != cfg["dataset_sha256"]:
        raise SystemExit(f"dataset {dataset_sha} is not the preregistered benchmark {cfg['dataset_sha256']}")
    tasks = tuple(cfg["tasks"])
    arm_cfg, gen = arm_settings(cfg, args.arm, args.model)
    items = [
        {**{k: v for k, v in r.items() if k != "prompt"}, "prompt": bl.arm_prompt(r["prompt"], args.arm)}
        for r in rows
        if r["task"] in tasks
    ]
    worlds = {w["world_id"]: w for w in provenance.read_jsonl(args.dataset / "worlds.jsonl")}
    baseline = load_baseline(args.baseline, spec, dataset_sha) if args.baseline else None
    identity = (provenance.sha256_json(cfg), spec, dataset_sha, args.arm)
    if args.resume:
        run = args.resume
        old = json.loads((run / "config.json").read_text())
        if (old["config_sha256"], old["model"], old["dataset_sha256"], old["arm"]) != identity:
            raise SystemExit("--resume refused: config, model, dataset, or arm differs")
    else:
        run = args.output_root / f"{provenance.utc_stamp()}_authinv_p3{args.arm}_{args.model}"
        run.mkdir(parents=True, exist_ok=False)
        provenance.write_json(
            run / "config.json",
            {
                "experiment": cfg["experiment"],
                "arm": args.arm,
                "config": cfg,
                "config_sha256": identity[0],
                "model": spec,
                "dataset": str(args.dataset),
                "dataset_sha256": dataset_sha,
                "arm_settings": arm_cfg,
                "command": sys.argv,
            },
        )
        provenance.write_json(run / "validation_report.json", {**validation, "items": len(items)})
    if args.limit:  # engineering smoke test only; never analysed or aggregated
        items = items[: args.limit]
    total = len(items)
    if args.dataset_only:
        provenance.write_run_status(run, "dataset_only", 0, total)
        provenance.finalize(run)
        print(run)
        return
    pred_path, timing_path = run / "predictions.jsonl", run / "timing.jsonl"
    try:
        from authinv.eval.generation import VLLMChat

        gen.update(
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            max_num_seqs=args.max_num_seqs,
        )
        chat = VLLMChat(spec, str(args.model_dir) if args.model_dir else None, gen, args.tensor_parallel)
        provenance.write_json(
            run / "metadata.json",
            {
                "model": spec,
                "arm": args.arm,
                "runtime": chat.provenance(),
                "generation": gen,
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                "dataset_sha256": dataset_sha,
                "command": sys.argv,
                "versions": {
                    k: cfg[k]
                    for k in ("baseline_version", "final_parser_version", "canon_version", "parser_version")
                },
            },
        )
        execute(chat, items, worlds, args.arm, gen, run, total)
        keep = {x["row_id"] for x in items}
        preds = [p for p in provenance.read_jsonl(pred_path) if p["row_id"] in keep]
        if args.limit:
            provenance.write_run_status(run, "smoke", len(preds), total)
            provenance.finalize(run)
            print(run)
            return
        provenance.write_json(
            run / "metrics.json",
            arm_metrics(args.arm, preds, cfg, baseline, provenance.read_jsonl(timing_path)),
        )
        provenance.write_run_status(run, "complete", len(preds), total)
        provenance.finalize(run)
        print(run)
    except BaseException as error:
        n = len(provenance.read_jsonl(pred_path)) if pred_path.exists() else 0
        provenance.write_run_status(run, "failed", n, total, repr(error))
        provenance.finalize(run)
        raise


def execute(chat, items: list[dict], worlds: dict, arm: str, gen: dict, run: Path, total: int) -> None:
    run_stage(
        chat,
        items,
        run / "predictions.jsonl",
        run / "timing.jsonl",
        stage=arm,
        id_key="row_id",
        batch_size=gen["batch_size"],
        score=Scorer(arm, worlds),
        max_tokens=gen["max_tokens"],
        progress=lambda n: provenance.write_run_status(run, "running", n, total),
    )


def arm_metrics(arm: str, preds: list[dict], cfg: dict, baseline, timing: list[dict]) -> dict:
    base = baseline[0] if baseline else None
    out = {"arm": arm, **summarize(preds, cfg, base, timing), "baseline": baseline[1] if baseline else None}
    if arm == "restate":
        a = cfg["analysis"]
        out["restatement"] = mm.conversion_metrics(preds, a["bootstrap_replicates"], a["bootstrap_seed"])
        out["garbage_in"] = {
            t: mm.garbage_in(preds, t, a["bootstrap_replicates"], a["bootstrap_seed"])
            for t in sorted({r["task"] for r in preds})
        }
    return out


def _derived_run(args, cfg: dict, spec: dict, arm: str, sources: dict) -> Path:
    run = args.output_root / f"{provenance.utc_stamp()}_authinv_p3{arm}_{spec['key']}"
    run.mkdir(parents=True, exist_ok=False)
    provenance.write_json(
        run / "config.json",
        {
            "experiment": cfg["experiment"],
            "arm": arm,
            "config": cfg,
            "config_sha256": provenance.sha256_json(cfg),
            "model": spec,
            "dataset_sha256": sources["dataset_sha256"],
            "sources": sources,
            "command": sys.argv,
        },
    )
    return run


def _complete(run: Path) -> dict:
    if json.loads((run / "run_status.json").read_text())["status"] != "complete":
        raise SystemExit(f"{run}: not complete")
    return json.loads((run / "config.json").read_text())


def cmd_derive(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    check_config(cfg)
    a = cfg["analysis"]
    tasks = tuple(cfg["tasks"])
    if args.arm == "self_consistency":
        conf = _complete(args.t2_run)
        spec = conf["model"]
        if spec != models.model_spec(spec["key"]):
            raise SystemExit("T2 run model spec differs from the registry")
        base, info = load_baseline(args.t2_run, spec, conf["dataset_sha256"])
        if cfg.get("dataset_sha256") and conf["dataset_sha256"] != cfg["dataset_sha256"]:
            raise SystemExit("T2 run is not on the preregistered benchmark")
        rows, stats = [], {}
        for t in tasks:
            got, stats[t] = bl.vote_rows(base, t, cfg["arms"]["self_consistency"]["tie"])
            rows += got
        sources = {"t2_run": info, "dataset_sha256": conf["dataset_sha256"]}
        run = _derived_run(args, cfg, spec, args.arm, sources)
        metrics = {
            "arm": args.arm,
            "analysis_only": True,
            "votes": stats,
            **summarize(rows, cfg, base, []),
            "baseline": info,
        }
    elif args.arm == "solver":
        conf = _complete(args.canon_run)
        spec = conf["model"]
        if spec != models.model_spec(spec["key"]) or conf["canon_version"] != canon.CANON_VERSION:
            raise SystemExit("canonicalization run model spec or canon version differs from the code")
        ds_rows, validation = load_dataset(args.dataset)
        dataset_sha = validation["manifest"]["sha256"]
        if conf["dataset_sha256"] != dataset_sha or (
            cfg.get("dataset_sha256") and dataset_sha != cfg["dataset_sha256"]
        ):
            raise SystemExit("canonicalization run and dataset are not the preregistered benchmark")
        worlds = {w["world_id"]: w for w in provenance.read_jsonl(args.dataset / "worlds.jsonl")}
        convs = provenance.read_jsonl(args.canon_run / "conversions.jsonl")
        rows, stats = bl.solver_rows(
            [r for r in ds_rows if r["task"] in tasks], convs, worlds, cfg["arms"]["solver"]["cedar_check"]
        )
        baseline = load_baseline(args.baseline, spec, dataset_sha) if args.baseline else None
        base = baseline[0] if baseline else None
        sources = {
            "canon_run": str(args.canon_run),
            "conversions_sha256": provenance.sha256_file(args.canon_run / "conversions.jsonl"),
            "dataset_sha256": dataset_sha,
            "t2_run": baseline[1] if baseline else None,
        }
        run = _derived_run(args, cfg, spec, args.arm, sources)
        canon_timing = provenance.read_jsonl(args.canon_run / "timing.jsonl")
        timing = [t for t in canon_timing if t["stage"] == "conversion"] + [
            {"stage": "engine", "requests": stats["queries"], "seconds": stats["engine_seconds"]}
        ]
        metrics = {
            "arm": args.arm,
            "engine": stats,
            **summarize(rows, cfg, base, timing, convs),
            "fail_closed": summarize(bl.fail_closed(rows), cfg, base, timing, convs),
            "baseline": baseline[1] if baseline else None,
        }
    else:
        raise SystemExit(f"--arm {args.arm} is not a derived arm")
    metrics["bootstrap"] = {"replicates": a["bootstrap_replicates"], "seed": a["bootstrap_seed"]}
    provenance.write_jsonl(run / "predictions.jsonl", rows)
    provenance.write_json(run / "metrics.json", metrics)
    provenance.write_json(
        run / "metadata.json",
        {
            "git": provenance.git_state(),
            "source_sha256": provenance.source_hashes(TRACKED),
            "command": sys.argv,
        },
    )
    provenance.write_run_status(run, "complete", len(rows), len(rows))
    provenance.finalize(run)
    print(run)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=bl.MODEL_ARMS)
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--baseline", type=Path, help="Phase-2 T2 run of the same model (gap closure)")
    r.add_argument("--model-dir", type=Path)
    r.add_argument("--tensor-parallel", type=int, default=1)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--enforce-eager", action="store_true")
    r.add_argument("--max-num-seqs", type=int)
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.add_argument("--resume", type=Path)
    r.add_argument("--dataset-only", action="store_true")
    r.add_argument("--limit", type=int, help="engineering smoke test: first N rows only; never analysed")
    r.set_defaults(func=cmd_run)
    d = sub.add_parser("derive")
    d.add_argument("--arm", required=True, choices=("self_consistency", "solver"))
    d.add_argument("--config", type=Path, required=True)
    d.add_argument("--t2-run", type=Path, help="self_consistency: the Phase-2 T2 run")
    d.add_argument("--dataset", type=Path, help="solver: the frozen benchmark")
    d.add_argument("--canon-run", type=Path, help="solver: the P3.1 canonicalization run of the model")
    d.add_argument("--baseline", type=Path, help="solver: the Phase-2 T2 run (gap closure)")
    d.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    d.set_defaults(func=cmd_derive)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
