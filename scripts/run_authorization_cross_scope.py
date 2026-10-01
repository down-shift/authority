#!/usr/bin/env python3
"""Measure filename application interference from an irrelevant ordering scope."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from tqdm import tqdm

from authority_leakage.inference import write_jsonl, write_run_status
from authority_leakage.models.continuation import continuation_encoding
from authority_leakage.models.hf import HFAdapter
from authority_leakage.schemas import Message
from authorization_competence.design import REPRESENTATIONS
from authorization_competence.lexical_symmetry import ACTOR_IDENTIFIERS, ASSIGNMENTS, TASKS
from authorization_competence.cross_scope import (
    CONDITIONS, build_cross_scope_rows, derive_cross_scope_worlds,
    validate_cross_scope_worlds,
)
from authorization_competence.cross_scope_analysis import cross_scope_analysis


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokenizer_audit(tokenizer, rows):
    actor_ids = {name: tokenizer(name, add_special_tokens=False)["input_ids"] for name in ACTOR_IDENTIFIERS}
    actor_token_counts = {len(ids) for ids in actor_ids.values()}
    audit_rows = []
    for row in rows:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        encodings = [continuation_encoding(tokenizer, rendered, candidate) for candidate in row["candidates"]]
        full_prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        audit_rows.append({
            "row_id": row["row_id"], "world_id": row["world_id"],
            "representation": row["representation"], "condition": row["scope_condition"],
            "assignment": row["assignment"], "task": row["task"],
            "query_scope_position": row["queried_scope_position"],
            "prompt_token_count": len(full_prompt_ids),
            "policy_token_count": len(tokenizer(row["policy_text"], add_special_tokens=False)["input_ids"]),
            "policy_actor_name_counts": {actor: row["policy_text"].count(actor) for actor in row["actor_order"]},
            "candidate_token_counts": [x["token_count"] for x in encodings],
            "candidate_token_count_mismatch": len({x["token_count"] for x in encodings}) > 1,
            "boundary_error": any(x["token_count"] <= 0 for x in encodings),
            "boundary_overlap": any(x["boundary_overlap"] for x in encodings),
            "prompt_prefix_retokenized": any(x["prompt_prefix_retokenized"] for x in encodings),
            "boundary_modes": [x["boundary_mode"] for x in encodings],
            "prompt_prefix_retokenization_by_candidate": [x["prompt_prefix_retokenized"] for x in encodings],
        })
    indexed = {(r["world_id"], r["representation"], r["assignment"], r["task"], r["condition"]): r
               for r in audit_rows}
    condition_pairs = []
    for world in {r["world_id"] for r in audit_rows}:
        for rep in REPRESENTATIONS:
            for assignment in ASSIGNMENTS:
                for task in TASKS:
                    c = indexed[(world, rep, assignment, task, "congruent")]
                    x = indexed[(world, rep, assignment, task, "conflicting")]
                    condition_pairs.append({
                        "world_id": world, "representation": rep, "assignment": assignment, "task": task,
                        "candidate_counts_match": c["candidate_token_counts"] == x["candidate_token_counts"],
                        "prompt_token_counts_match": c["prompt_token_count"] == x["prompt_token_count"],
                        "policy_token_counts_match": c["policy_token_count"] == x["policy_token_count"],
                        "policy_actor_name_counts_match": c["policy_actor_name_counts"] == x["policy_actor_name_counts"],
                    })
    return {
        "actor_identifiers": {name: {"token_ids": ids, "token_count": len(ids)} for name, ids in actor_ids.items()},
        "actor_token_counts_matched": len(actor_token_counts) == 1,
        "actor_character_lengths_matched": len({len(a) for a in ACTOR_IDENTIFIERS}) == 1,
        "candidate_token_count_mismatch_rows": sum(r["candidate_token_count_mismatch"] for r in audit_rows),
        "candidate_boundary_error_rows": sum(r["boundary_error"] for r in audit_rows),
        "candidate_boundary_overlap_rows": sum(r["boundary_overlap"] for r in audit_rows),
        "candidate_prefix_retokenization_rows": sum(r["prompt_prefix_retokenized"] for r in audit_rows),
        "condition_pair_candidate_token_count_mismatch_rows": sum(not r["candidate_counts_match"] for r in condition_pairs),
        "condition_pair_prompt_token_count_mismatch_rows": sum(not r["prompt_token_counts_match"] for r in condition_pairs),
        "condition_pair_policy_token_count_mismatch_rows": sum(not r["policy_token_counts_match"] for r in condition_pairs),
        "condition_pair_policy_actor_name_count_mismatch_rows": sum(not r["policy_actor_name_counts_match"] for r in condition_pairs),
        "condition_pairs": condition_pairs,
        "rows": audit_rows,
    }


def report_text(metrics):
    lines = ["# Experiment 2: cross-scope authorization interference", "",
             f"Worlds: {metrics['worlds']}; scored rows: {metrics['rows']}. Every condition contains both scopes. Margins are lexical-name symmetrized; CIs resample semantic worlds.", "",
             "## Filename application", "",
             "| Representation | Δ scope margin (95% CI) | Congruent accuracy | Conflicting accuracy | Harm | Rescue | Raw Δ original | Raw Δ swapped |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    app = metrics["tasks"]["application"]
    for rep in REPRESENTATIONS:
        cell = app["by_representation"][rep]
        delta = cell["mean_delta_scope"]
        lines.append(f"| {rep} | {delta['mean']:.3f} [{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}] | {cell['accuracy_congruent']['mean']:.3f} | {cell['accuracy_conflicting']['mean']:.3f} | {cell['categorical_harm_rate']['mean']:.3f} | {cell['categorical_rescue_rate']['mean']:.3f} | {cell['mean_delta_original_assignment']['mean']:.3f} | {cell['mean_delta_swapped_assignment']['mean']:.3f} |")
    lines += ["", f"Across representations, mean paired Δ scope margin = {app['mean_delta_scope']['mean']:.3f} [{app['mean_delta_scope']['ci95'][0]:.3f}, {app['mean_delta_scope']['ci95'][1]:.3f}].",
              f"Overall congruent accuracy = {app['accuracy_congruent']['mean']:.3f}; conflicting accuracy = {app['accuracy_conflicting']['mean']:.3f}.",
              f"Paired categorical harm rate = {app['categorical_harm_rate']['mean']:.3f}; rescue rate = {app['categorical_rescue_rate']['mean']:.3f}.",
              "", "## Scope-order control", "",
              "| Queried filename policy position | Δ scope margin | Congruent accuracy | Conflicting accuracy | Harm | Rescue |",
              "|---|---:|---:|---:|---:|---:|"]
    for position in ("first", "second"):
        cell = metrics["scope_order"]["application"]["by_queried_scope_position"][position]
        lines.append(f"| {position} | {cell['mean_delta_scope']['mean']:.3f} [{cell['mean_delta_scope']['ci95'][0]:.3f}, {cell['mean_delta_scope']['ci95'][1]:.3f}] | {cell['accuracy_congruent']['mean']:.3f} | {cell['accuracy_conflicting']['mean']:.3f} | {cell['categorical_harm_rate']['mean']:.3f} | {cell['categorical_rescue_rate']['mean']:.3f} |")
    order = metrics["scope_order"]["application"]["delta_second_minus_first"]
    lines.append(f"\nScope-order interaction (filename second minus first): {order['mean']:.3f} [{order['ci95'][0]:.3f}, {order['ci95'][1]:.3f}].")
    deltas = {rep: app["by_representation"][rep]["mean_delta_scope"]["mean"] for rep in REPRESENTATIONS}
    most_negative = min(deltas, key=deltas.get)
    least_negative = max(deltas, key=deltas.get)
    lexical = app["lexical_swap_interaction"]
    net_harm = app["net_categorical_harm"]
    app_delta = app["mean_delta_scope"]
    supports_exp3 = app_delta["ci95"][1] < 0 or net_harm["ci95"][0] > 0
    lines += ["", "## Interpretation secondary outcome", "",
              f"Mean paired Δ interpretation margin = {metrics['tasks']['interpretation']['mean_delta_scope']['mean']:.3f} [{metrics['tasks']['interpretation']['mean_delta_scope']['ci95'][0]:.3f}, {metrics['tasks']['interpretation']['mean_delta_scope']['ci95'][1]:.3f}].",
              f"Filename-owner interpretation accuracy: congruent {metrics['tasks']['interpretation']['accuracy_congruent']['mean']:.3f}; conflicting {metrics['tasks']['interpretation']['accuracy_conflicting']['mean']:.3f}.",
              f"Interpretation harm/rescue: {metrics['tasks']['interpretation']['categorical_harm_rate']['mean']:.3f}/{metrics['tasks']['interpretation']['categorical_rescue_rate']['mean']:.3f}.",
              "", "## Requested conclusions", "",
              f"A. Negative application-margin evidence: {'yes' if app_delta['ci95'][1] < 0 else 'not conclusive'}; Δ={app_delta['mean']:.3f} [{app_delta['ci95'][0]:.3f}, {app_delta['ci95'][1]:.3f}].",
              f"B. Categorical filename harm/rescue: {app['categorical_harm_rate']['mean']:.3f}/{app['categorical_rescue_rate']['mean']:.3f}; error-rate increase = {app['accuracy_congruent']['mean'] - app['accuracy_conflicting']['mean']:.3f}.",
              f"C. Interpretation: congruent/conflicting accuracy {metrics['tasks']['interpretation']['accuracy_congruent']['mean']:.3f}/{metrics['tasks']['interpretation']['accuracy_conflicting']['mean']:.3f}, paired Δ margin {metrics['tasks']['interpretation']['mean_delta_scope']['mean']:.3f}.",
              f"D. Descriptive scope effects: most negative for {most_negative} ({deltas[most_negative]:.3f}); least negative for {least_negative} ({deltas[least_negative]:.3f}). Treat as exploratory until replicated.",
              f"E. Scope-order contrast (second minus first): {order['mean']:.3f} [{order['ci95'][0]:.3f}, {order['ci95'][1]:.3f}].",
              f"F. Lexical-swap interaction on application Δ (swapped minus original): {lexical['mean']:.3f} [{lexical['ci95'][0]:.3f}, {lexical['ci95'][1]:.3f}].",
              f"G. Evidence supports testing canonicalization as Experiment 3: {'yes' if supports_exp3 else 'not on this result alone'}. This is a follow-up rationale, not evidence that canonicalization will mitigate interference.",
              "", "## Representation interactions and lexical swap", "",
              "Paired representation contrasts, raw assignment deltas, scope-order strata, and lexical-swap interaction CIs are in `metrics.json`.",
              "Interpret negative deltas as behavioral cross-scope interference.", ""]
    return "\n".join(lines)


def finalize(run):
    (run / "artifact_sha256.json").write_text(json.dumps({
        str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "artifact_sha256.json"}, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-run", type=Path, required=True,
                        help="Completed Experiment 1 single-scope lexical-invariance run")
    parser.add_argument("--model", default="qwen3_8b_int8", choices=["qwen3_8b_int8"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset-only", action="store_true", help="Generate, validate, and save the 7,200-row dataset without loading a model")
    parser.add_argument("--resume", type=Path, help="Run inference on a previously generated dataset-only run")
    args = parser.parse_args()
    source = args.stage2_run
    status = json.loads((source / "run_status.json").read_text())
    if status.get("status") != "complete":
        parser.error("Experiment 2 requires a complete frozen Stage 2 run")
    source_validation = json.loads((source / "validation_report.json").read_text())
    if source_validation.get("passed") is not True or source_validation.get("worlds") != 180:
        parser.error("Stage 2 source must contain the validated 180-world lexical-symmetry dataset")
    source_calibration = json.loads((source / "source_calibration.json").read_text())
    if source_calibration.get("gate", {}).get("passed") is not True:
        parser.error("The lexical-symmetry competence calibration gate must have passed")
    source_meta = json.loads((source / "metadata.json").read_text())
    model_meta = source_meta.get("model", {})
    if model_meta.get("model_name") != "Qwen/Qwen3-8B" or model_meta.get("quantization") != "bitsandbytes_int8":
        parser.error("Stage 2 source must use Qwen3-8B int8")
    model_revision = model_meta.get("model_commit")
    if not model_revision:
        parser.error("Stage 2 metadata is missing the resolved model revision")
    frozen_worlds = load_jsonl(source / "worlds.jsonl")
    worlds = derive_cross_scope_worlds(frozen_worlds)
    rows = build_cross_scope_rows(worlds)
    validation = validate_cross_scope_worlds(worlds, rows)
    if len(rows) != 7200:
        raise RuntimeError(f"Expected 7,200 rows before inference, got {len(rows)}")
    run = args.resume or args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                              "_authorization_cross_scope_qwen3_8b_int8")
    config = yaml.safe_load((ROOT / "configs/authorization_cross_scope.yaml").read_text())
    prereg = {
        "primary_estimand": "M_sym(conflicting) - M_sym(congruent), orientation-correct filename margin",
        "conditions": "both scopes explicitly encode owner and non-owner status for both actors; only ordering-scope owner changes",
        "application_context": "filename proposals only; no ordering values are shown",
        "lexical_symmetry": "average original and swapped assignment margins within world/representation/condition/task",
        "scope_order": "filename policy first/second balanced 90/90 and held constant within each condition pair",
        "bootstrap_unit": "semantic world", "row_count": 7200,
        "representations": list(REPRESENTATIONS), "conditions": list(CONDITIONS),
        "tasks": list(TASKS), "model_revision": model_revision,
        "no_canonicalization_or_other_experiments": True,
    }
    if args.resume:
        if not run.is_dir():
            parser.error("--resume requires an existing dataset run")
        resume_status = json.loads((run / "run_status.json").read_text()).get("status")
        if resume_status not in {"dataset_complete", "failed"}:
            parser.error("--resume requires a dataset-only run or a failed run with a completed tokenizer audit")
        saved_worlds = load_jsonl(run / "worlds.jsonl")
        saved_rows = load_jsonl(run / "dataset.jsonl")
        if saved_worlds != worlds or saved_rows != rows or json.loads((run / "validation_report.json").read_text()).get("passed") is not True:
            parser.error("Prepared dataset differs from the frozen worlds or fails semantic validation")
        if json.loads((run / "source_stage2.json").read_text()).get("model_revision") != model_revision:
            parser.error("Prepared dataset model revision differs from frozen Stage 2")
        if resume_status == "failed":
            audit_path = run / "tokenization_audit.json"
            if not audit_path.is_file():
                parser.error("Failed run has no completed tokenizer audit")
            saved_audit = json.loads(audit_path.read_text())
            if any(saved_audit.get(key, 0) for key in (
                "candidate_token_count_mismatch_rows", "candidate_boundary_error_rows",
                "candidate_boundary_overlap_rows", "candidate_prefix_retokenization_rows",
                "condition_pair_candidate_token_count_mismatch_rows",
                "condition_pair_policy_actor_name_count_mismatch_rows")):
                parser.error("Failed run's tokenizer audit did not pass")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", {**config, "model": args.model, "model_revision": model_revision,
                                         "stage2_source_run": str(source), "device": args.device})
        write_json(run / "preregistration.json", prereg)
        write_jsonl(run / "worlds.jsonl", worlds)
        write_jsonl(run / "dataset.jsonl", rows)
        write_json(run / "validation_report.json", validation)
        dataset_hash = sha(run / "dataset.jsonl")
        write_json(run / "dataset_metadata.json", {"sha256": dataset_hash, "rows": len(rows), "worlds": len(worlds)})
        write_json(run / "source_stage2.json", {"run": str(source), "dataset_sha256": json.loads((source / "dataset_metadata.json").read_text())["sha256"],
                                                   "model_revision": model_revision})
    print(f"Expected rows before inference: {len(rows)}", flush=True)
    if args.dataset_only:
        write_run_status(run / "run_status.json", "dataset_complete", 0, len(rows))
        finalize(run)
        print(run, flush=True)
        return
    write_run_status(run / "run_status.json", "tokenizer_audit", 0, len(rows))
    try:
        import torch
        import transformers
        model_spec = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"][args.model]
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_spec["name"], revision=model_revision,
                                                                trust_remote_code=False)
        audit = tokenizer_audit(tokenizer, rows)
        write_json(run / "tokenization_audit.json", audit)
        stage2_audit = json.loads((source / "tokenization_audit.json").read_text())
        if not audit["actor_token_counts_matched"] or not audit["actor_character_lengths_matched"]:
            write_run_status(run / "run_status.json", "actor_token_audit_failed", 0, len(rows))
            finalize(run)
            print(run)
            return
        if any(audit["actor_identifiers"][name]["token_ids"] != stage2_audit["identifiers"][name]["token_ids"]
               for name in ACTOR_IDENTIFIERS):
            write_run_status(run / "run_status.json", "tokenizer_identity_mismatch", 0, len(rows))
            finalize(run)
            print(run)
            return
        if audit["candidate_token_count_mismatch_rows"] or audit["candidate_boundary_error_rows"] or \
           audit["candidate_boundary_overlap_rows"] or audit["candidate_prefix_retokenization_rows"] or \
           audit["condition_pair_candidate_token_count_mismatch_rows"] or \
           audit["condition_pair_policy_actor_name_count_mismatch_rows"]:
            write_run_status(run / "run_status.json", "candidate_scoring_audit_failed", 0, len(rows))
            finalize(run)
            print(run)
            return
        adapter = HFAdapter(model_spec["name"], model_revision, args.device,
                            enable_thinking=model_spec.get("enable_thinking"),
                            quantization=model_spec.get("quantization"),
                            attention_implementation=model_spec.get("attention_implementation"))
        torch.set_num_threads(int(config.get("torch_threads", 4)))
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        tracked = [ROOT / "scripts/run_authorization_cross_scope.py",
                   ROOT / "src/authorization_competence/cross_scope.py",
                   ROOT / "src/authorization_competence/cross_scope_analysis.py",
                   ROOT / "src/authorization_competence/lexical_symmetry.py",
                   ROOT / "src/authority_leakage/models/hf.py",
                   ROOT / "src/authority_leakage/models/continuation.py"]
        write_json(run / "metadata.json", {
            "model": adapter.provenance(), "git_commit": commit,
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                               capture_output=True, text=True).stdout.strip()),
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in tracked},
            "dataset_sha256": dataset_hash, "command": sys.argv,
            "scoring": "direct sum log probability for exact semantic answer strings",
            "bootstrap_unit": "semantic world", "stage2_source_run": str(source),
        })
        audit_by_id = {x["row_id"]: x for x in audit["rows"]}
        predictions = []
        done = 0
        with (run / "predictions.jsonl").open("w") as output:
            for row in tqdm(rows, desc="Cross-scope interference", unit="row"):
                scores = adapter.score_candidates_detailed([Message("user", row["prompt"])], sorted(row["candidates"]))
                correct = scores[row["correct"]]["sum_logprob"]
                incorrect = scores[row["incorrect"]]["sum_logprob"]
                selected = max(sorted(row["candidates"]), key=lambda c: scores[c]["sum_logprob"])
                row_audit = audit_by_id[row["row_id"]]
                prediction = {**row,
                    "candidate_scores": {c: {k: v for k, v in score.items() if k != "input_ids"}
                                         for c, score in scores.items()},
                    "margin": correct - incorrect, "selected": selected,
                    "score_audit": {"candidate_token_count_mismatch": row_audit["candidate_token_count_mismatch"],
                                    "boundary_error": row_audit["boundary_error"],
                                    "boundary_overlap": row_audit["boundary_overlap"],
                                    "prompt_prefix_retokenized": row_audit["prompt_prefix_retokenized"],
                                    "candidate_token_counts": row_audit["candidate_token_counts"]}}
                predictions.append(prediction)
                output.write(json.dumps(prediction, sort_keys=True) + "\n")
                output.flush()
                done += 1
                if done % 100 == 0:
                    write_run_status(run / "run_status.json", "running", done, len(rows))
        metrics = cross_scope_analysis(predictions, int(config["seed"]))
        write_json(run / "metrics.json", metrics)
        (run / "report.md").write_text(report_text(metrics))
        write_run_status(run / "run_status.json", "complete", len(predictions), len(rows))
        finalize(run)
        print(run / "report.md", flush=True)
    except BaseException as error:
        write_run_status(run / "run_status.json", "failed", 0, len(rows), repr(error))
        finalize(run)
        raise


if __name__ == "__main__":
    main()
