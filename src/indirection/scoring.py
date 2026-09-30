"""Semantic scoring; positive sum-logprob margin always favors ground truth."""
import math
from authority_leakage.schemas import Message


def score_example(row, adapter, model):
    messages = [Message(role="user", content=row["prompt"])]
    candidates = [row["correct_candidate"], row["incorrect_candidate"]]
    scores = adapter.score_candidates_detailed(messages, candidates)
    if any(not math.isfinite(s["sum_logprob"]) for s in scores.values()):
        raise ValueError("Nonfinite candidate score")
    margin = scores[candidates[0]]["sum_logprob"] - scores[candidates[1]]["sum_logprob"]
    return {**row, "model": model, "rendered_prompt": adapter.render(messages),
            "candidate_scores": scores, "margin": margin, "correct": margin > 0,
            "tie": margin == 0}
