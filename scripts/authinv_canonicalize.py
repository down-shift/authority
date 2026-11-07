#!/usr/bin/env python3
"""Phase-3 arm 1, policy canonicalization (P3.1): rendering -> canonical JSON IR -> decision.

  python scripts/authinv_canonicalize.py run --config configs/authinv/p3_canonicalize.yaml --dataset DIR
        --model KEY [--baseline T2RUN] [--model-dir DIR] [--tensor-parallel N] [--resume RUN]
        [--dataset-only] [--limit N]
  python scripts/authinv_canonicalize.py analyze --config configs/authinv/p3_canonicalize.yaml --run RUN
        [--baseline T2RUN] --out FILE

DIR is the frozen Phase-1 benchmark (dataset.jsonl, dataset_manifest.json, worlds.jsonl).
Stage 1 converts each (world, assignment, rendering) cell once (conversions.jsonl); stage 2
asks every query of the configured tasks with the frozen T2 prompt, its policy block replaced
by the predicted IR (predictions.jsonl). A cell whose conversion did not parse gets no stage 2:
its rows keep the conversion's category (refusal / abstain / parse_failure) with
failed_stage = "conversion". timing.jsonl logs every batch (stage, requests, seconds, tokens).

T2RUN is the Phase-2 T2 run of the same model (same dataset, model spec, and parser); with it,
metrics.json adds gap closure and the token ratio. `analyze` recomputes the metrics of a complete
run into FILE, outside the run directory. --dataset-only builds and validates the conversion
cases without a model.
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
from authinv.eval.structured import JSON_PARSER_VERSION, parse_structured  # noqa: E402
from authinv.mitigate import canonicalize as canon  # noqa: E402
from authinv.mitigate import metrics as mm  # noqa: E402
from authinv.mitigate.prompts import query_of, replace_policy, request_of  # noqa: E402
from authinv.mitigate.runner import done_ids, run_stage  # noqa: E402
from authinv.policy import decision_owners, evaluate, from_dict  # noqa: E402

TRACKED = [ROOT / "scripts" / "authinv_canonicalize.py"] + [
    ROOT / "src" / "authinv" / p
    for p in (
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


def settings(cfg: dict, model: str) -> tuple[dict, dict, dict]:
    """(conversion, decision, generation) settings with the model's overrides applied."""
    over = cfg.get("overrides", {}).get(model, {})
    conv = {**cfg["conversion"], **over.get("conversion", {})}
    dec = {**cfg["decision"], **over.get("decision", {})}
    gen = {**cfg["generation"], **over.get("generation", {}), "max_tokens": dec["max_tokens"]}
    if conv["decoding"] not in ("constrained", "free"):
        raise SystemExit(f"unknown conversion decoding {conv['decoding']!r}")
    return conv, dec, gen


class Arm:
    """Scoring state for one run: canonical policies, schemas, and caches keyed by IR text."""

    def __init__(self, worlds: dict[str, dict]):
        self.worlds = worlds
        self.policies: dict[tuple, object] = {}
        self.schemas: dict[tuple, dict] = {}
        self.scored: dict[tuple, tuple] = {}
        self.case_ir: dict[str, str] = {}  # case_id -> canonical IR text, filled before stage 2

    def policy(self, w: str, a: str):
        if (w, a) not in self.policies:
            self.policies[(w, a)] = from_dict(self.worlds[w][a])
        return self.policies[(w, a)]

    def types(self, w: str) -> tuple[tuple, tuple]:
        return tuple(self.worlds[w]["principal_types"]), tuple(self.worlds[w]["resource_types"])

    def schema(self, case: dict) -> dict:
        key = (case["world_id"], case["assignment"])
        if key not in self.schemas:
            self.schemas[key] = canon.ir_schema(self.policy(*key))
        return self.schemas[key]

    def ir(self, w: str, a: str, text: str) -> tuple[dict, object]:
        key = (w, a, text)
        if key not in self.scored:
            self.scored[key] = canon.score_ir(self.policy(w, a), text, *self.types(w))
        return self.scored[key]

    def score_conversion(self, case: dict, g: dict) -> dict:
        out = canon.parse_conversion(g["text"])
        if out["conv_category"] != "ir":
            return {**out, "ir_valid": False, "ir_exact": False, "ir_equivalent": False}
        scores, _ = self.ir(case["world_id"], case["assignment"], out["ir_text"])
        return {**out, **scores}

    def score_decision(self, item: dict, g: dict) -> dict:
        out = parse_structured(g["text"], item["task"], item["expected"])
        out["ir_request_correct"] = None
        if not item["ir_valid"]:
            return out
        w, a = item["world_id"], item["assignment"]
        _, pred = self.ir(w, a, self.case_ir[item["case_id"]])
        gold = self.policy(w, a)
        if item["task"] == "application":
            decision = evaluate(pred, request_of(item, gold))["decision"]
            out.update(ir_decision=decision, ir_request_correct=decision == item["label"])
        else:
            action, res, ctx = query_of(item, gold)
            owners = [e.id for e in decision_owners(pred, action, res, ctx, self.types(w)[0])]
            out["ir_request_correct"] = owners == item["expected"]
        return out


def decision_items(
    rows: list[dict], convs: list[dict], tasks: tuple[str, ...]
) -> tuple[list[dict], list[dict]]:
    """(stage-2 items to generate, rows whose conversion failed, already final)."""
    by_case = {c["case_id"]: c for c in convs}
    ready, failed = [], []
    for r in rows:
        if r["task"] not in tasks:
            continue
        cid = f"{r['world_id']}/{r['assignment']}/{r['rendering']}"
        if cid not in by_case:
            continue
        c = by_case[cid]
        item = {
            **{k: v for k, v in r.items() if k != "prompt"},
            "case_id": cid,
            **{k: c.get(k) for k in ("conv_category", "ir_valid", "ir_exact", "ir_equivalent")},
        }
        if c["conv_category"] == "ir":
            ready.append({**item, "prompt": replace_policy(r["prompt"], r["task"], c["ir_text"])})
        else:
            failed.append(
                {
                    **item,
                    "category": c["conv_category"],
                    "value": None,
                    "failed_stage": "conversion",
                    "ir_request_correct": None,
                    "text": None,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                }
            )
    return ready, failed


def load_baseline(run: Path, spec: dict, dataset_sha: str) -> tuple[list[dict], dict]:
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


def analyze(
    convs: list[dict], preds: list[dict], cfg: dict, base: list[dict] | None, timing: list[dict]
) -> dict:
    a = cfg["analysis"]
    reps, seed = a["bootstrap_replicates"], a["bootstrap_seed"]
    tasks = sorted({r["task"] for r in preds})
    out = {
        "canon_version": canon.CANON_VERSION,
        "parser_version": JSON_PARSER_VERSION,
        "conversion": mm.conversion_metrics(convs, reps, seed),
        "tasks": {t: invariance.rendering_metrics(preds, t, reps, seed) for t in tasks},
        "category_counts": {
            t: dict(collections.Counter(r["category"] for r in preds if r["task"] == t)) for t in tasks
        },
        "conversion_failed_rows": {
            t: sum(r.get("failed_stage") == "conversion" for r in preds if r["task"] == t) for t in tasks
        },
        "garbage_in": {t: mm.garbage_in(preds, t, reps, seed) for t in tasks},
        "cost": mm.cost(convs, preds, base, timing),
    }
    if "application" in tasks:
        out["rendering_contrasts"] = invariance.rendering_contrasts(preds, "application", reps, seed)
    if base is not None:
        out["closure"] = {t: mm.gap_closure(preds, base, t, reps, seed) for t in tasks}
    return out


def cmd_run(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    spec = models.model_spec(args.model)
    rows, validation = load_dataset(args.dataset)
    dataset_sha = validation["manifest"]["sha256"]
    if cfg.get("dataset_sha256") and dataset_sha != cfg["dataset_sha256"]:
        raise SystemExit(f"dataset {dataset_sha} is not the preregistered benchmark {cfg['dataset_sha256']}")
    if cfg["parser_version"] != JSON_PARSER_VERSION or cfg["canon_version"] != canon.CANON_VERSION:
        raise SystemExit("config parser or canon version differs from the code")
    tasks = tuple(cfg["tasks"])
    conv_cfg, dec_cfg, gen = settings(cfg, args.model)
    worlds = {w["world_id"]: w for w in provenance.read_jsonl(args.dataset / "worlds.jsonl")}
    cases, report = canon.build_cases(rows, worlds, tasks)
    baseline = load_baseline(args.baseline, spec, dataset_sha) if args.baseline else None
    identity = (provenance.sha256_json(cfg), spec, dataset_sha, JSON_PARSER_VERSION, canon.CANON_VERSION)
    if args.resume:
        run = args.resume
        old = json.loads((run / "config.json").read_text())
        if (
            old["config_sha256"],
            old["model"],
            old["dataset_sha256"],
            old["parser_version"],
            old["canon_version"],
        ) != identity:
            raise SystemExit("--resume refused: config, model, dataset, parser, or canon version differs")
    else:
        run = args.output_root / f"{provenance.utc_stamp()}_authinv_p3canon_{args.model}"
        run.mkdir(parents=True, exist_ok=False)
        provenance.write_json(
            run / "config.json",
            {
                "experiment": cfg["experiment"],
                "config": cfg,
                "config_sha256": identity[0],
                "model": spec,
                "dataset": str(args.dataset),
                "dataset_sha256": dataset_sha,
                "parser_version": JSON_PARSER_VERSION,
                "canon_version": canon.CANON_VERSION,
                "conversion": conv_cfg,
                "decision": dec_cfg,
                "command": sys.argv,
            },
        )
        provenance.write_json(run / "validation_report.json", {**validation, "cases": report})
    if args.limit:  # engineering smoke test only; never analysed or aggregated
        cases = cases[: args.limit]
    keep = {c["case_id"] for c in cases}
    n_rows = sum(c["queries"] for c in cases)
    total = len(cases) + n_rows
    if args.dataset_only:
        provenance.write_run_status(run, "dataset_only", 0, total)
        provenance.finalize(run)
        print(run)
        return
    conv_path, pred_path, timing_path = (
        run / "conversions.jsonl",
        run / "predictions.jsonl",
        run / "timing.jsonl",
    )
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
                "runtime": chat.provenance(),
                "generation": gen,
                "conversion": conv_cfg,
                "decision": dec_cfg,
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                "dataset_sha256": dataset_sha,
                "command": sys.argv,
                "parser_version": JSON_PARSER_VERSION,
                "canon_version": canon.CANON_VERSION,
            },
        )
        execute(chat, rows, worlds, cases, tasks, conv_cfg, dec_cfg, gen["batch_size"], run, total)
        convs = [c for c in provenance.read_jsonl(conv_path) if c["case_id"] in keep]
        preds = [p for p in provenance.read_jsonl(pred_path) if p["case_id"] in keep]
        if args.limit:
            provenance.write_run_status(run, "smoke", len(convs) + len(preds), total)
            provenance.finalize(run)
            print(run)
            return
        base = baseline[0] if baseline else None
        metrics = analyze(convs, preds, cfg, base, provenance.read_jsonl(timing_path))
        metrics["baseline"] = baseline[1] if baseline else None
        provenance.write_json(run / "metrics.json", metrics)
        provenance.write_run_status(run, "complete", len(convs) + len(preds), total)
        provenance.finalize(run)
        print(run)
    except BaseException as error:
        n = sum(len(provenance.read_jsonl(p)) for p in (conv_path, pred_path) if p.exists())
        provenance.write_run_status(run, "failed", n, total, repr(error))
        provenance.finalize(run)
        raise


def execute(chat, rows, worlds, cases, tasks, conv_cfg, dec_cfg, batch_size, run, total) -> None:
    """Stage 1 then stage 2, both resumable; conversion-failed rows are appended without generation."""
    arm = Arm(worlds)
    conv_path, pred_path, timing_path = (
        run / "conversions.jsonl",
        run / "predictions.jsonl",
        run / "timing.jsonl",
    )
    constrained = conv_cfg["decoding"] == "constrained"
    for c in cases:
        if constrained:
            c["schema_sha256"] = canon.schema_sha256(arm.schema(c))
    run_stage(
        chat,
        cases,
        conv_path,
        timing_path,
        stage="conversion",
        id_key="case_id",
        batch_size=batch_size,
        score=arm.score_conversion,
        schema_of=arm.schema if constrained else None,
        max_tokens=conv_cfg["max_tokens"],
        progress=lambda n: provenance.write_run_status(run, "running", n, total),
    )
    keep = {c["case_id"] for c in cases}
    convs = [c for c in provenance.read_jsonl(conv_path) if c["case_id"] in keep]
    arm.case_ir = {c["case_id"]: c["ir_text"] for c in convs if c["conv_category"] == "ir"}
    ready, failed = decision_items(rows, convs, tasks)
    run_stage(
        chat,
        ready,
        pred_path,
        timing_path,
        stage="decision",
        id_key="row_id",
        batch_size=batch_size,
        score=arm.score_decision,
        max_tokens=dec_cfg["max_tokens"],
        progress=lambda n: provenance.write_run_status(run, "running", len(convs) + n, total),
    )
    have = done_ids(pred_path, "row_id")
    with pred_path.open("a") as out:
        for r in failed:
            if r["row_id"] not in have:
                out.write(json.dumps(r, sort_keys=True) + "\n")


def cmd_analyze(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    conf = json.loads((args.run / "config.json").read_text())
    if json.loads((args.run / "run_status.json").read_text())["status"] != "complete":
        raise SystemExit(f"{args.run}: not complete")
    if conf["config_sha256"] != provenance.sha256_json(cfg):
        raise SystemExit(f"{args.run}: config differs")
    baseline = load_baseline(args.baseline, conf["model"], conf["dataset_sha256"]) if args.baseline else None
    out = analyze(
        provenance.read_jsonl(args.run / "conversions.jsonl"),
        provenance.read_jsonl(args.run / "predictions.jsonl"),
        cfg,
        baseline[0] if baseline else None,
        provenance.read_jsonl(args.run / "timing.jsonl"),
    )
    out.update(run=str(args.run), baseline=baseline[1] if baseline else None, git=provenance.git_state())
    provenance.write_json(args.out, out)
    print(args.out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
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
    r.add_argument(
        "--limit", type=int, help="engineering smoke test: first N conversion cases only; never analysed"
    )
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("analyze")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--baseline", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=cmd_analyze)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
