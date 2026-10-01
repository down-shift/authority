#!/usr/bin/env python3
"""Run E3 canonicalization mitigation over the frozen E2 cross-scope dataset."""
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
    CONDITIONS, build_cross_scope_rows, derive_cross_scope_worlds, validate_cross_scope_worlds,
)
from authorization_competence.canonicalization import (
    APPLICATION_ORDERS, CANONICALIZATION_ARMS, IR_PLACEHOLDER, canonical_answer_prompt,
    canonical_candidates, decode_status_policy, make_e3_datasets, render_ir, validate_e3_datasets,
)
from authorization_competence.canonicalization_analysis import canonicalization_analysis


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def path_from_record(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def source_hashes(path):
    if not path.exists():
        return {}
    return {str(file.relative_to(path)): sha(file) for file in sorted(path.rglob("*")) if file.is_file()}


def source_code_hashes():
    paths = [
        ROOT / "scripts/run_authorization_canonicalization.py",
        ROOT / "configs/models.yaml",
        ROOT / "src/authorization_competence/canonicalization.py",
        ROOT / "src/authorization_competence/canonicalization_analysis.py",
        ROOT / "src/authorization_competence/cross_scope.py",
        ROOT / "src/authorization_competence/cross_scope_analysis.py",
        ROOT / "src/authorization_competence/lexical_symmetry.py",
        ROOT / "src/authority_leakage/models/hf.py",
        ROOT / "src/authority_leakage/models/continuation.py",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def git_provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def validate_e2_source(e2_run, model_key):
    status_path = e2_run / "run_status.json"
    if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "complete":
        raise ValueError("E3 requires a completed E2 run")
    config = json.loads((e2_run / "config.json").read_text())
    revision = config.get("model_revision")
    if config.get("model") != model_key or not revision:
        raise ValueError("E2 config model must match --model and pin its revision")
    predictions_path = e2_run / "predictions.jsonl"
    if not predictions_path.is_file():
        raise ValueError("E2 predictions are required to reuse the raw arm")
    e2_predictions = load_jsonl(predictions_path)
    expected = 180 * len(REPRESENTATIONS) * len(CONDITIONS) * len(ASSIGNMENTS) * len(TASKS)
    if len(e2_predictions) != expected or len({r["row_id"] for r in e2_predictions}) != expected:
        raise ValueError("E2 must contain 7,200 unique completed scored rows")
    bad_audit = [r["row_id"] for r in e2_predictions if any(r.get("score_audit", {}).get(k, False) for k in (
        "candidate_token_count_mismatch", "boundary_error", "boundary_overlap", "prompt_prefix_retokenized"))]
    if bad_audit:
        raise ValueError("E2 raw-score candidate audit contains failures")
    stage2_run = path_from_record(config["stage2_source_run"])
    stage2_meta = json.loads((stage2_run / "metadata.json").read_text())
    model_spec = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"][model_key]
    e2_model = json.loads((e2_run / "metadata.json").read_text()).get("model", {})
    if e2_model.get("model_name") != model_spec["name"] or e2_model.get("quantization") != model_spec.get("quantization"):
        raise ValueError("E2 model and quantization do not match --model")
    actual_revision = stage2_meta["model"]["model_commit"]
    if model_spec.get("revision") and actual_revision != model_spec.get("revision"):
        raise ValueError("E2 model revision differs from pinned --model revision")
    if actual_revision != revision:
        raise ValueError("E2 and its Stage 2 source do not use the same model revision")
    stage2_worlds_path = stage2_run / "worlds.jsonl"
    if not stage2_worlds_path.is_file():
        raise ValueError("Frozen Stage 2 worlds are missing")
    frozen_worlds = load_jsonl(stage2_worlds_path)
    worlds = derive_cross_scope_worlds(frozen_worlds)
    e2_dataset = build_cross_scope_rows(worlds)
    validation = validate_cross_scope_worlds(worlds, e2_dataset)
    base_by_id = {r["row_id"]: r for r in e2_dataset}
    for pred in e2_predictions:
        base = base_by_id.get(pred["row_id"])
        if base is None:
            raise ValueError("E2 predictions do not map to the frozen E2 dataset")
        for field in ("prompt", "policy_text", "policy", "candidates", "correct", "incorrect", "filename_owner", "values"):
            if pred[field] != base[field]:
                raise ValueError(f"Frozen E2 prediction differs from reconstructed dataset field {field}")
    return config, revision, stage2_run, worlds, e2_dataset, e2_predictions, validation


def ir_from_text(text):
    value = json.loads(text)
    if set(value) != {"filename_owner", "ordering_owner"}:
        raise ValueError("Predicted canonical IR must preserve both scope owners")
    if any(actor not in ACTOR_IDENTIFIERS for actor in value.values()):
        raise ValueError("Predicted canonical IR contains an unknown actor")
    return value


def chat_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def audit_candidate_rows(tokenizer, prompts):
    rows = []
    for row_id, prompt, candidates in prompts:
        rendered = chat_prompt(tokenizer, prompt)
        encodings = [continuation_encoding(tokenizer, rendered, c) for c in candidates]
        rows.append({
            "row_id": row_id, "candidate_token_counts": [e["token_count"] for e in encodings],
            "candidate_token_count_mismatch": len({e["token_count"] for e in encodings}) > 1,
            "boundary_error": any(e["token_count"] <= 0 for e in encodings),
            "boundary_overlap": any(e["boundary_overlap"] for e in encodings),
            "prompt_prefix_retokenized": any(e["prompt_prefix_retokenized"] for e in encodings),
            "prompt_prefix_retokenization_by_candidate": [e["prompt_prefix_retokenized"] for e in encodings],
            "boundary_modes": [e["boundary_mode"] for e in encodings],
        })
    return rows


def audit_failures(rows):
    fields = ("candidate_token_count_mismatch", "boundary_error", "boundary_overlap", "prompt_prefix_retokenized")
    return [r for r in rows if any(r.get(f, False) for f in fields)]


def report_text(metrics):
    lines = ["# E3: canonicalization mitigation", "",
             f"Worlds: {metrics['worlds']}; conversion cases: {metrics['conversion_cases']}; answer rows: {metrics['answer_rows']}. All CIs bootstrap semantic worlds.",
             "Application margins average both filename proposal orders before averaging original/swapped actor-name assignments.", "",
             "## Canonical conversion", "",
             "| Input representation | Exact two-scope conversion | Filename owner correct | Ordering owner correct |",
             "|---|---:|---:|---:|"]
    for rep, cell in metrics["conversion"]["by_representation"].items():
        exact = cell["exact_conversion_correct"]
        fn = cell["filename_owner_correct"]
        order = cell["ordering_owner_correct"]
        lines.append(f"| {rep} | {exact['mean']:.3f} [{exact['ci95'][0]:.3f}, {exact['ci95'][1]:.3f}] | {fn['mean']:.3f} | {order['mean']:.3f} |")
    exact = metrics["conversion"]["overall"]["exact_conversion_correct"]
    lines += ["", f"Overall exact conversion accuracy: {exact['mean']:.3f} [{exact['ci95'][0]:.3f}, {exact['ci95'][1]:.3f}]. Conversion chose the model's highest-scoring legal IR; gold was used only to score conversion correctness.", ""]
    for task in TASKS:
        label = "Filename application" if task == "application" else "Filename-owner interpretation"
        lines += [f"## {label}", "", "| Arm | Condition | Accuracy | Mean margin |", "|---|---|---:|---:|"]
        for arm in CANONICALIZATION_ARMS:
            for condition in CONDITIONS:
                acc = metrics["tasks"][task]["by_arm"][arm]["accuracy"][condition]
                margin = metrics["tasks"][task]["by_arm"][arm]["margin"][condition]
                lines.append(f"| {arm} | {condition} | {acc['mean']:.3f} [{acc['ci95'][0]:.3f}, {acc['ci95'][1]:.3f}] | {margin['mean']:.3f} [{margin['ci95'][0]:.3f}, {margin['ci95'][1]:.3f}] |")
        eff = metrics["tasks"][task]["interference_reduction"]
        lines.append(f"\nRaw Δ scope: {eff['delta_scope_raw']['mean']:.3f} [{eff['delta_scope_raw']['ci95'][0]:.3f}, {eff['delta_scope_raw']['ci95'][1]:.3f}]; canonicalized Δ scope: {eff['delta_scope_canonicalized']['mean']:.3f} [{eff['delta_scope_canonicalized']['ci95'][0]:.3f}, {eff['delta_scope_canonicalized']['ci95'][1]:.3f}].")
        red = eff["interference_reduction"]
        lines.append(f"Interference reduction (canonicalized Δ − raw Δ): {red['mean']:.3f} [{red['ci95'][0]:.3f}, {red['ci95'][1]:.3f}]. Positive indicates mitigation.")
        accuracy_label = "application" if task == "application" else "interpretation"
        lines += ["", f"| Representation | Raw Δ scope | Canonicalized Δ scope | Interference reduction | Raw {accuracy_label} accuracy C/X | Canonicalized {accuracy_label} accuracy C/X |", "|---|---:|---:|---:|---:|---:|"]
        for rep, cell in metrics["tasks"][task]["by_representation"].items():
            raw_ci = cell["raw"]["delta_scope"]
            can_ci = cell["canonicalized"]["delta_scope"]
            reduction_ci = cell["interference_reduction"]
            raw_acc = cell["raw"]["accuracy_congruent"]["mean"], cell["raw"]["accuracy_conflicting"]["mean"]
            can_acc = cell["canonicalized"]["accuracy_congruent"]["mean"], cell["canonicalized"]["accuracy_conflicting"]["mean"]
            fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
            lines.append(f"| {rep} | {fmt(raw_ci)} | {fmt(can_ci)} | {fmt(reduction_ci)} | {raw_acc[0]:.3f}/{raw_acc[1]:.3f} | {can_acc[0]:.3f}/{can_acc[1]:.3f} |")
    lines += ["", "Conversion errors and answer errors are separated in `metrics.json`; canonicalized answers are also summarized conditional on correct conversion. All canonical answer prompts use the model-selected IR, never the gold IR.", ""]
    return "\n".join(lines)


def finalize(run):
    write_json(run / "artifact_sha256.json", {
        str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "artifact_sha256.json"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2-run", type=Path, required=True, help="Completed E2 cross-scope run")
    parser.add_argument("--model", default="qwen3_8b_int8",
                        choices=["qwen3_8b_int8", "gemma3_12b_it_int8", "gemma3_12b_it_nf4"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume a prepared dataset-only run")
    args = parser.parse_args()
    e2_run = args.e2_run if args.e2_run.is_absolute() else ROOT / args.e2_run
    try:
        e2_config, revision, stage2_run, worlds, e2_rows, e2_predictions, e2_validation = validate_e2_source(e2_run, args.model)
    except Exception as exc:
        parser.error(f"Invalid frozen E2 source: {exc}")
    conversion_rows, answer_rows = make_e3_datasets(worlds, e2_rows)
    validation = validate_e3_datasets(worlds, e2_rows, conversion_rows, answer_rows)
    validation["reconstructed_e2_validation"] = e2_validation
    if validation["passed"] is not True:
        parser.error("E3 validation failed")
    run = args.resume or args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                              f"_authorization_canonicalization_{args.model}")
    config = {"experiment": "authorization_canonicalization_mitigation", "model": args.model,
              "model_revision": revision, "e2_source_run": str(e2_run), "stage2_source_run": str(stage2_run),
              "worlds": 180, "representations": list(REPRESENTATIONS), "conditions": list(CONDITIONS),
              "assignments": list(ASSIGNMENTS), "tasks": list(TASKS),
              "canonicalization_arms": list(CANONICALIZATION_ARMS),
              "application_candidate_orders": list(APPLICATION_ORDERS),
              "expected_conversion_cases": len(conversion_rows), "expected_answer_rows": len(answer_rows),
              "bootstrap_unit": "semantic world", "seed": 20261006}
    prereg = {"primary_estimand": "interference_reduction = delta_scope_canonicalized - delta_scope_raw",
              "delta_scope": "M_sym(CONFLICTING) - M_sym(CONGRUENT)",
              "symmetrization_order": "average correct-oriented margins over frozen/reversed filename proposal order, then original/swapped actor-name assignment",
              "canonical_ir": "JSON with independent filename_owner and ordering_owner fields",
              "ir_selection": "highest-scoring candidate among all four legal owner combinations; gold policy is used only to calculate conversion accuracy",
              "raw_arm": "reuse exact E2 direct-answer scores for frozen-order application and interpretation; score reversed filename order",
              "canonicalized_arm": "score answers from the model-selected canonical IR",
              "bootstrap_unit": "semantic world", "no_model_sweep_or_new_task_family": True}
    e2_file_hashes = source_hashes(e2_run)
    stage2_file_hashes = source_hashes(stage2_run)
    lineage = {"e2_run": str(e2_run), "e2_config": json.loads((e2_run / "config.json").read_text()),
               "e2_artifact_hashes_present": e2_file_hashes,
               "e2_artifact_manifest": json.loads((e2_run / "artifact_sha256.json").read_text()) if (e2_run / "artifact_sha256.json").is_file() else None,
               "stage2_run": str(stage2_run), "stage2_dataset_sha256": json.loads((stage2_run / "dataset_metadata.json").read_text())["sha256"],
               "stage2_model_revision": revision, "stage2_artifact_hashes_present": stage2_file_hashes}
    if args.resume:
        if not run.is_dir():
            parser.error("--resume requires an existing E3 dataset run")
        resume_status = json.loads((run / "run_status.json").read_text()).get("status")
        if resume_status not in {"dataset_complete", "failed"}:
            parser.error("--resume requires a dataset-only run or a clean failed run")
        if load_jsonl(run / "conversion_dataset.jsonl") != conversion_rows or load_jsonl(run / "answer_dataset.jsonl") != answer_rows:
            parser.error("Prepared E3 dataset differs from frozen E2 source")
        if json.loads((run / "source_run_lineage.json").read_text()) != lineage:
            parser.error("Prepared run source lineage differs")
        if resume_status == "failed":
            if (run / "conversion_predictions.jsonl").exists() or (run / "predictions.jsonl").exists():
                parser.error("Cannot resume a run with partial model predictions; use its frozen dataset in a new run")
            audit_path = run / "tokenization_audit.json"
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text())
                if audit.get("preflight_failures", 0) or audit.get("canonicalized_answer_preflight_failures", 0):
                    parser.error("Failed run contains tokenizer audit failures")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", config)
        write_json(run / "preregistration.json", prereg)
        write_jsonl(run / "worlds.jsonl", worlds)
        write_jsonl(run / "conversion_dataset.jsonl", conversion_rows)
        write_jsonl(run / "answer_dataset.jsonl", answer_rows)
        write_json(run / "validation_report.json", validation)
        combined_hash = hashlib.sha256((sha(run / "conversion_dataset.jsonl") + sha(run / "answer_dataset.jsonl")).encode()).hexdigest()
        write_json(run / "dataset_metadata.json", {
            "conversion_rows": len(conversion_rows), "answer_rows": len(answer_rows),
            "worlds": len(worlds), "conversion_dataset_sha256": sha(run / "conversion_dataset.jsonl"),
            "answer_dataset_sha256": sha(run / "answer_dataset.jsonl"), "combined_dataset_sha256": combined_hash,
            "worlds_sha256": sha(run / "worlds.jsonl"),
            "e2_predictions_sha256": e2_file_hashes["predictions.jsonl"]})
        write_json(run / "source_run_lineage.json", lineage)
        write_json(run / "source_stage2.json", {"run": str(stage2_run), "dataset_sha256": lineage["stage2_dataset_sha256"],
                                                  "model_revision": revision})
        write_json(run / "source_e2.json", {"run": str(e2_run), "predictions_sha256": e2_file_hashes["predictions.jsonl"],
                                              "metrics_sha256": e2_file_hashes.get("metrics.json"),
                                              "model_revision": revision})
        commit, dirty = git_provenance()
        selected_model = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"][args.model]
        write_json(run / "metadata.json", {
            "model": {"model_name": selected_model["name"], "model_commit": revision,
                      "quantization": selected_model.get("quantization"),
                      "quantization_config": ({"load_in_8bit": True} if selected_model.get("quantization") == "bitsandbytes_int8" else
                                              {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                               "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"}
                                              if selected_model.get("quantization") == "bitsandbytes_nf4" else None),
                      "device_requested": args.device,
                      "status": "not_loaded_yet"},
            "git_commit": commit, "git_dirty": dirty, "source_sha256": source_code_hashes(),
            "dataset_sha256": combined_hash, "source_run_lineage": str(run / "source_run_lineage.json"),
            "command": sys.argv, "scoring": "direct semantic continuation sum log probability",
            "primary_margin": "correct minus incorrect; symmetrized over candidate order and lexical assignment"})
    print(f"Validated E3 dataset: {len(conversion_rows)} conversion cases + {len(answer_rows)} answer rows", flush=True)
    if args.dataset_only:
        write_run_status(run / "run_status.json", "dataset_complete", 0, len(answer_rows) + len(conversion_rows))
        finalize(run)
        print(run, flush=True)
        return

    try:
        import torch
        import transformers
        model_spec = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"][args.model]
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_spec["name"], revision=revision,
                                                                trust_remote_code=False)
        # Preflight every conversion candidate and every raw reversed-order answer.
        # Frozen-order raw predictions inherit E2's already-passed audit.
        conversion_audit_prompts = [(r["row_id"], r["prompt"], r["candidates"]) for r in conversion_rows]
        raw_reverse = [r for r in answer_rows if r["arm"] == "raw" and r["task"] == "application" and r["candidate_order"] == "reversed"]
        raw_audit_prompts = [(r["row_id"], r["prompt"], r["candidates"]) for r in raw_reverse]
        audit_rows = audit_candidate_rows(tokenizer, conversion_audit_prompts + raw_audit_prompts)
        write_json(run / "tokenization_audit.json", {"pre_conversion_and_raw_reversed": audit_rows,
                    "preflight_failures": len(audit_failures(audit_rows)),
                    "frozen_raw_rows_reuse_e2_audit": True})
        if audit_failures(audit_rows):
            write_run_status(run / "run_status.json", "tokenizer_audit_failed", 0, len(answer_rows) + len(conversion_rows))
            finalize(run)
            print(run, flush=True)
            return
        torch.set_num_threads(4)
        adapter = HFAdapter(model_spec["name"], revision, args.device,
                            enable_thinking=model_spec.get("enable_thinking"),
                            quantization=model_spec.get("quantization"),
                            attention_implementation=model_spec.get("attention_implementation"))
        model_provenance = adapter.provenance()
        commit, dirty = git_provenance()
        write_json(run / "metadata.json", {"model": model_provenance, "git_commit": commit, "git_dirty": dirty,
                    "source_sha256": source_code_hashes(), "dataset_sha256": json.loads((run / "dataset_metadata.json").read_text())["combined_dataset_sha256"],
                    "source_run_lineage": str(run / "source_run_lineage.json"), "command": sys.argv,
                    "scoring": "direct semantic continuation sum log probability", "bootstrap_unit": "semantic world"})
        write_run_status(run / "run_status.json", "canonical_conversion", 0, len(conversion_rows) + len(answer_rows))

        conversion_predictions = []
        conversion_audit_by_id = {r["row_id"]: r for r in audit_rows if r["row_id"] in {x["row_id"] for x in conversion_rows}}
        with (run / "conversion_predictions.jsonl").open("w") as out:
            for i, row in enumerate(tqdm(conversion_rows, desc="Canonical conversion", unit="policy")):
                scores = adapter.score_candidates_detailed([Message("user", row["prompt"])], sorted(row["candidates"]))
                selected_text = max(sorted(row["candidates"]), key=lambda c: scores[c]["sum_logprob"])
                selected_ir = ir_from_text(selected_text)
                prediction = {**row, "candidate_scores": {c: {k: v for k, v in score.items() if k != "input_ids"}
                                                              for c, score in scores.items()},
                              "selected_ir_text": selected_text, "selected_ir": selected_ir,
                              "exact_conversion_correct": selected_ir == row["gold_ir"],
                              "filename_owner_correct": selected_ir["filename_owner"] == row["gold_ir"]["filename_owner"],
                              "ordering_owner_correct": selected_ir["ordering_owner"] == row["gold_ir"]["ordering_owner"],
                              "score_audit": {key: conversion_audit_by_id[row["row_id"]][key] for key in (
                                  "candidate_token_count_mismatch", "boundary_error", "boundary_overlap",
                                  "prompt_prefix_retokenized", "candidate_token_counts")}}
                conversion_predictions.append(prediction)
                out.write(json.dumps(prediction, sort_keys=True) + "\n")
                if (i + 1) % 100 == 0:
                    write_run_status(run / "run_status.json", "canonical_conversion", i + 1, len(conversion_rows) + len(answer_rows))
        ir_by_key = {(r["world_id"], r["representation"], r["scope_condition"], r["assignment"]): r
                     for r in conversion_predictions}

        canonical_rows = [r for r in answer_rows if r["arm"] == "canonicalized"]
        canonical_audit_prompts = []
        for row in canonical_rows:
            conv = ir_by_key[(row["world_id"], row["representation"], row["scope_condition"], row["assignment"])]
            prompt = row["prompt"].replace(IR_PLACEHOLDER, conv["selected_ir_text"])
            canonical_audit_prompts.append((row["row_id"], prompt, row["candidates"]))
        canonical_audit = audit_candidate_rows(tokenizer, canonical_audit_prompts)
        audit_doc = json.loads((run / "tokenization_audit.json").read_text())
        audit_doc["canonicalized_answer_rows"] = canonical_audit
        audit_doc["canonicalized_answer_preflight_failures"] = len(audit_failures(canonical_audit))
        write_json(run / "tokenization_audit.json", audit_doc)
        if audit_failures(canonical_audit):
            write_run_status(run / "run_status.json", "canonicalized_answer_tokenizer_audit_failed", 0, len(answer_rows) + len(conversion_rows))
            finalize(run)
            print(run, flush=True)
            return

        raw_e2_by_id = {r["row_id"]: r for r in e2_predictions}
        reverse_audit = {r["row_id"]: r for r in audit_rows if r["row_id"] in {x["row_id"] for x in raw_reverse}}
        canonical_audit_by_id = {r["row_id"]: r for r in canonical_audit}
        prediction_rows = []
        with (run / "predictions.jsonl").open("w") as out:
            for index, row in enumerate(tqdm(answer_rows, desc="E3 paired answers", unit="row")):
                if row["arm"] == "raw" and row["candidate_order"] in ("none", "frozen"):
                    source = raw_e2_by_id[row["source_e2_row_id"]]
                    scores = source["candidate_scores"]
                    audit = source["score_audit"]
                    prompt = row["prompt"]
                    reused = True
                else:
                    if row["arm"] == "canonicalized":
                        conv = ir_by_key[(row["world_id"], row["representation"], row["scope_condition"], row["assignment"])]
                        prompt = row["prompt"].replace(IR_PLACEHOLDER, conv["selected_ir_text"])
                        audit = canonical_audit_by_id[row["row_id"]]
                    else:
                        prompt = row["prompt"]
                        audit = reverse_audit[row["row_id"]]
                    raw_scores = adapter.score_candidates_detailed([Message("user", prompt)], sorted(row["candidates"]))
                    scores = {c: {k: v for k, v in score.items() if k != "input_ids"} for c, score in raw_scores.items()}
                    reused = False
                correct_score = scores[row["correct"]]["sum_logprob"]
                incorrect_score = scores[row["incorrect"]]["sum_logprob"]
                selected = max(sorted(row["candidates"]), key=lambda c: scores[c]["sum_logprob"])
                pred = {**row, "prompt": prompt, "candidate_scores": scores,
                        "margin": correct_score - incorrect_score, "selected": selected,
                        "reused_e2_score": reused,
                        "score_audit": {"candidate_token_count_mismatch": audit.get("candidate_token_count_mismatch", False),
                                        "boundary_error": audit.get("boundary_error", False),
                                        "boundary_overlap": audit.get("boundary_overlap", False),
                                        "prompt_prefix_retokenized": audit.get("prompt_prefix_retokenized", False),
                                        "candidate_token_counts": audit.get("candidate_token_counts", [])}}
                if row["arm"] == "canonicalized":
                    conv = ir_by_key[(row["world_id"], row["representation"], row["scope_condition"], row["assignment"])]
                    pred["selected_ir"] = conv["selected_ir"]
                    pred["conversion_correct"] = conv["exact_conversion_correct"]
                prediction_rows.append(pred)
                out.write(json.dumps(pred, sort_keys=True) + "\n")
                if (index + 1) % 100 == 0:
                    write_run_status(run / "run_status.json", "answer_scoring", len(conversion_rows) + index + 1,
                                     len(conversion_rows) + len(answer_rows))
        write_jsonl(run / "conversion_predictions.jsonl", conversion_predictions)
        metrics = canonicalization_analysis(conversion_predictions, prediction_rows)
        write_json(run / "metrics.json", metrics)
        (run / "report.md").write_text(report_text(metrics))
        write_run_status(run / "run_status.json", "complete", len(conversion_rows) + len(prediction_rows),
                         len(conversion_rows) + len(answer_rows))
        finalize(run)
        print(run / "report.md", flush=True)
    except BaseException as exc:
        write_run_status(run / "run_status.json", "failed", 0, len(conversion_rows) + len(answer_rows), repr(exc))
        finalize(run)
        raise


if __name__ == "__main__":
    main()
