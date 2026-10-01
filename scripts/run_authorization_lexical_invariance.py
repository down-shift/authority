#!/usr/bin/env python3
"""Run single-scope representation invariance after lexical-symmetry competence passes."""
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
from authorization_competence.lexical_symmetry import (
    ACTOR_IDENTIFIERS, ASSIGNMENTS, TASKS, build_lexical_representation_rows,
    decode_lexical_policy, validate_lexical_representation_rows,
)
from authorization_competence.lexical_analysis import lexical_representation_analysis


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokenization_audit(tokenizer, rows):
    identifiers = {}
    for actor in ACTOR_IDENTIFIERS:
        ids = tokenizer(actor, add_special_tokens=False)["input_ids"]
        identifiers[actor] = {"token_ids": ids, "token_count": len(ids), "character_length": len(actor)}
    token_counts = {x["token_count"] for x in identifiers.values()}
    char_lengths = {x["character_length"] for x in identifiers.values()}
    row_audit = []
    for row in rows:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        enc = [continuation_encoding(tokenizer, rendered, candidate) for candidate in row["candidates"]]
        row_audit.append({"row_id": row["row_id"], "candidate_token_counts": [x["token_count"] for x in enc],
                          "candidate_token_count_mismatch": len({x["token_count"] for x in enc}) > 1,
                          "boundary_error": any(x["token_count"] <= 0 for x in enc),
                          "boundary_overlap": any(x["boundary_overlap"] for x in enc),
                          "prefix_retokenization": any(x["prompt_prefix_retokenized"] for x in enc),
                          "boundary_modes": [x["boundary_mode"] for x in enc],
                          "prompt_prefix_retokenized": [x["prompt_prefix_retokenized"] for x in enc]})
    return {"passed": len(token_counts) == 1 and len(char_lengths) == 1,
            "matched_actor_token_count": len(token_counts) == 1,
            "matched_actor_character_length": len(char_lengths) == 1,
            "identifiers": identifiers,
            "candidate_rows": row_audit,
            "candidate_token_count_mismatch_rows": sum(x["candidate_token_count_mismatch"] for x in row_audit)}


def report(metrics):
    lines = ["# Single-scope authorization representation invariance with lexical symmetry", "",
             f"Worlds: {metrics['worlds']}; rows: {metrics['rows']}. Each policy and task was evaluated under original and swapped actor-name assignments; margins below average the two assignments.", "",
             "| Task | Representation | Symmetrized accuracy (95% CI) | Mean symmetrized margin (95% CI) | Swap-flip rate |",
             "|---|---|---:|---:|---:|"]
    for task in TASKS:
        for rep in REPRESENTATIONS:
            cell = metrics["tasks"][task][rep]
            acc, margin, flips = cell["symmetrized_accuracy"], cell["symmetrized_margin"], cell["swap_flip_rate"]
            lines.append(f"| {task} | {rep} | {acc['mean']:.3f} [{acc['ci95'][0]:.3f}, {acc['ci95'][1]:.3f}] | {margin['mean']:.3f} [{margin['ci95'][0]:.3f}, {margin['ci95'][1]:.3f}] | {flips['mean']:.3f} |")
    lines += ["", "## Invariance", "",
              "| Task | Worlds with categorical disagreement | Mean within-world margin variance |",
              "|---|---:|---:|"]
    for task in TASKS:
        d = metrics["categorical_disagreement_fraction"][task]
        v = metrics["within_world_margin_variance"][task]
        lines.append(f"| {task} | {d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] | {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    lines += ["", "Pairwise world-paired margin and accuracy contrasts are in `metrics.json`. CIs resample semantic worlds.",
              "", "The raw disagreement metrics compare winners across representations separately within each of the 360 world-assignment pairs. The symmetrized disagreement metric compares winners after averaging the two name-assignment margins; it does not represent literal single-prompt answer flips.",
              "", "All five representation renderers were strictly decoded back to the same canonical owner, and queries, values, candidates, world IDs, and name assignments were held fixed.", ""]
    return "\n".join(lines)


def finalize(run):
    write_json(run / "artifact_sha256.json", {
        str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "artifact_sha256.json"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-run", type=Path, required=True,
                        help="Passing lexical-symmetry calibration run directory")
    parser.add_argument("--model", default="qwen3_8b_int8", choices=["qwen3_8b_int8"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    calibration = args.calibration_run
    status_path = calibration / "run_status.json"
    if not status_path.is_file() or json.loads(status_path.read_text()).get("gate", {}).get("passed") is not True:
        parser.error("Representation invariance requires a passing lexical-symmetry competence gate")
    if not (calibration / "predictions.jsonl").is_file():
        parser.error("Calibration run is missing saved predictions")
    source_metadata = json.loads((calibration / "metadata.json").read_text())
    if source_metadata.get("model", {}).get("model_name") != "Qwen/Qwen3-8B" or \
       source_metadata.get("model", {}).get("quantization") != "bitsandbytes_int8":
        parser.error("Calibration must use the registered Qwen3-8B int8 model")
    model_revision = source_metadata.get("model", {}).get("model_commit")
    if not model_revision:
        parser.error("Calibration metadata is missing the resolved model revision")
    worlds = load_jsonl(calibration / "worlds.jsonl")
    rows = build_lexical_representation_rows(worlds)
    validation = validate_lexical_representation_rows(worlds, rows)
    model_spec = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"][args.model]
    run = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                              "_authorization_lexical_invariance_qwen3_8b_int8")
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", {"experiment": "single_scope_lexical_symmetrized_representation_invariance",
                                     "calibration_run": str(calibration), "model": args.model,
                                     "representations": list(REPRESENTATIONS), "assignments": list(ASSIGNMENTS),
                                     "tasks": list(TASKS), "worlds": len(worlds)})
    write_json(run / "preregistration.json", {
        "source_calibration_gate_passed": True,
        "worlds_and_name_assignments": "exactly reused from the lexical-symmetry calibration",
        "primary_margin": "correct semantic candidate sum log probability minus incorrect semantic candidate sum log probability",
        "per_representation_symmetrized_margin": "mean of original and swapped assignment margins, each oriented correct minus incorrect",
        "symmetrized_disagreement": "within-world difference in semantic winner across representations after name-swap symmetrization; not literal single-prompt flips",
        "raw_disagreement": "representation winner differences separately for each world and actor-name assignment",
        "margin_dispersion": "within-world variance across five symmetrized representation margins",
        "bootstrap_unit": "semantic world",
        "representation_invariance_only": True,
    })
    write_jsonl(run / "worlds.jsonl", worlds)
    write_jsonl(run / "dataset.jsonl", rows)
    write_json(run / "validation_report.json", validation)
    dataset_hash = sha(run / "dataset.jsonl")
    write_json(run / "dataset_metadata.json", {"sha256": dataset_hash, "rows": len(rows), "worlds": len(worlds)})
    write_json(run / "source_calibration.json", {
        "run": str(calibration), "dataset_sha256": json.loads((calibration / "dataset_metadata.json").read_text())["sha256"],
        "gate": json.loads(status_path.read_text())["gate"],
    })
    try:
        import torch
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_spec["name"], revision=model_revision,
                                                                trust_remote_code=False)
        audit = tokenization_audit(tokenizer, rows)
        write_json(run / "tokenization_audit.json", audit)
        prior_audit = json.loads((calibration / "actor_identifier_audit.json").read_text())
        bad_rows = [r for r in audit["candidate_rows"] if
                    r["candidate_token_count_mismatch"] or r["boundary_error"] or
                    r["boundary_overlap"] or r["prefix_retokenization"]]
        audit["boundary_overlap_rows"] = sum(r["boundary_overlap"] for r in audit["candidate_rows"])
        audit["prefix_retokenization_rows"] = sum(r["prefix_retokenization"] for r in audit["candidate_rows"])
        audit["passed"] = audit["passed"] and not bad_rows
        write_json(run / "tokenization_audit.json", audit)
        if not audit["passed"]:
            write_run_status(run / "run_status.json", "tokenizer_audit_failed", 0, len(rows), "Representation candidate scoring token audit failed")
            finalize(run)
            return
        if any(audit["identifiers"][name]["token_ids"] != prior_audit["identifiers"][name]["token_ids"]
               for name in ACTOR_IDENTIFIERS):
            write_run_status(run / "run_status.json", "tokenizer_identity_mismatch", 0, len(rows), "Tokenizer actor IDs differ from calibration")
            finalize(run)
            return
        torch.set_num_threads(4)
        adapter = HFAdapter(model_spec["name"], model_revision, args.device,
                            enable_thinking=model_spec.get("enable_thinking"),
                            quantization=model_spec.get("quantization"),
                            attention_implementation=model_spec.get("attention_implementation"))
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        tracked = [ROOT / "scripts/run_authorization_lexical_invariance.py",
                   ROOT / "src/authorization_competence/lexical_symmetry.py",
                   ROOT / "src/authorization_competence/lexical_analysis.py",
                   ROOT / "src/authority_leakage/models/hf.py",
                   ROOT / "src/authority_leakage/models/continuation.py"]
        write_json(run / "metadata.json", {
            "model": adapter.provenance(), "git_commit": commit,
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                               capture_output=True, text=True).stdout.strip()),
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in tracked},
            "dataset_sha256": dataset_hash, "command": sys.argv,
            "scoring": "direct sum log probability for complete semantic answer strings",
            "bootstrap_unit": "semantic world", "source_calibration_run": str(calibration),
        })
        predictions = []
        candidate_audit_by_row = {x["row_id"]: x for x in audit["candidate_rows"]}
        with (run / "predictions.jsonl").open("w") as output:
            for row in tqdm(rows, desc="Symmetrized representations", unit="row"):
                scores = adapter.score_candidates_detailed([Message("user", row["prompt"])], sorted(row["candidates"]))
                correct = scores[row["correct"]]["sum_logprob"]
                incorrect = scores[row["incorrect"]]["sum_logprob"]
                selected = max(sorted(row["candidates"]), key=lambda c: scores[c]["sum_logprob"])
                row_audit = candidate_audit_by_row[row["row_id"]]
                prediction = {**row,
                    "candidate_scores": {c: {k: v for k, v in value.items() if k != "input_ids"}
                                         for c, value in scores.items()},
                    "margin": correct - incorrect, "selected": selected,
                    "score_audit": {"candidate_token_count_mismatch": row_audit["candidate_token_count_mismatch"],
                                    "boundary_error": row_audit["boundary_error"],
                                    "boundary_overlap": row_audit["boundary_overlap"],
                                    "prompt_prefix_retokenized": row_audit["prefix_retokenization"],
                                    "candidate_token_counts": row_audit["candidate_token_counts"]}}
                predictions.append(prediction)
                output.write(json.dumps(prediction, sort_keys=True) + "\n")
                output.flush()
        metrics = lexical_representation_analysis(predictions)
        write_json(run / "metrics.json", metrics)
        (run / "report.md").write_text(report(metrics))
        write_run_status(run / "run_status.json", "complete", len(predictions), len(rows))
        finalize(run)
        print(run / "report.md", flush=True)
    except BaseException as error:
        write_run_status(run / "run_status.json", "failed", 0, len(rows), repr(error))
        finalize(run)
        raise


if __name__ == "__main__":
    main()
