"""Inference and deterministic JSONL storage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from authority_leakage.models.base import ModelAdapter
from authority_leakage.schemas import Example
from authority_leakage.scoring import grade


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_inference(examples: list[Example], adapter: ModelAdapter, max_new_tokens: int) -> list[dict]:
    rows = []
    for example in examples:
        candidates = example.metadata["labels"] if example.experiment == "epistemic" else []
        candidate_scores = adapter.score_candidates(example.messages, candidates) if candidates else None
        generation = adapter.generate(example.messages, max_new_tokens)
        outcome = grade(example, generation.text, candidate_scores)
        rows.append({
            **example.to_dict(),
            "rendered_prompt": generation.rendered_prompt,
            "raw_response": generation.text,
            "generated_token_logprobs": generation.generated_token_logprobs,
            "candidate_logprobs": candidate_scores,
            "outcome": outcome,
        })
    return rows


def tokenization_audit(examples: list[Example], adapter: ModelAdapter) -> dict:
    labels = sorted({label for e in examples if e.experiment == "epistemic" for label in e.metadata["labels"]})
    ids = adapter.candidate_token_ids(labels)
    counts = {label: len(token_ids) for label, token_ids in ids.items()}
    unequal_pairs = sorted({tuple(e.metadata["labels"]) for e in examples if e.experiment == "epistemic" and counts[e.metadata["labels"][0]] != counts[e.metadata["labels"][1]]})
    audit = {"candidate_token_ids": ids, "candidate_token_counts": counts, "unequal_length_pairs": unequal_pairs,
             "all_single_token": all(n == 1 for n in counts.values())}
    if hasattr(adapter, "render") and hasattr(adapter, "tokenizer"):
        by_condition: dict[str, list[int]] = {}
        by_pair: dict[str, dict[str, int]] = {}
        for e in examples:
            prompt = adapter.render(e.messages)
            n = len(adapter.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            by_condition.setdefault(e.condition, []).append(n)
            by_pair.setdefault(e.pair_id, {})[e.condition] = n
        audit["prompt_tokens_by_condition"] = {
            key: {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
            for key, values in by_condition.items()
        }
        diffs = []
        for group in by_pair.values():
            privileged = group.get("claim_system", group.get("claim_developer"))
            user = group.get("claim_user")
            if privileged is not None and user is not None:
                diffs.append(privileged - user)
        audit["paired_prompt_token_difference"] = {
            "n": len(diffs), "min": min(diffs) if diffs else None,
            "max": max(diffs) if diffs else None,
            "mean": sum(diffs) / len(diffs) if diffs else None,
        }
    return audit
