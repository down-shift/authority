"""Joint tokenization with explicit continuation boundaries.

A token overlapping the character boundary is included in the scored suffix.
This is a joint-token suffix likelihood, conditional on the stable token prefix;
it is not a probability of partial bytes within a token. Such overlaps are audited.
Slow tokenizers fall back to the longest common token prefix, conservatively
including every retokenized suffix token.
"""
from __future__ import annotations


def continuation_encoding(tokenizer, prompt: str, candidate: str) -> dict:
    if not prompt or not candidate:
        raise ValueError("Prompt and candidate must be nonempty")
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    try:
        encoded = tokenizer(prompt + candidate, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        positions = [i for i, (_, end) in enumerate(offsets) if end > len(prompt)]
        if not positions:
            raise ValueError("Candidate has no continuation tokens")
        start = positions[0]
        overlap = offsets[start][0] < len(prompt)
        mode = "joint_offsets"
        # Never condition on prompt tokens that joint tokenization changed.
        common = 0
        for a, b in zip(prefix, ids):
            if a != b:
                break
            common += 1
        retokenized_prefix = common < min(len(prefix), start)
        start = min(start, common)
    except (NotImplementedError, TypeError):
        ids = tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"]
        start = 0
        for a, b in zip(prefix, ids):
            if a != b:
                break
            start += 1
        overlap = start < len(prefix)
        mode = "joint_common_prefix"
        retokenized_prefix = overlap
    if start < 1 or start >= len(ids):
        raise ValueError("Continuation requires a nonempty conditioning prefix and scored suffix")
    return {"input_ids": ids, "continuation_start": start,
            "continuation_ids": ids[start:], "token_count": len(ids) - start,
            "boundary_overlap": overlap, "boundary_mode": mode,
            "prompt_prefix_retokenized": retokenized_prefix,
            "prompt_token_count": len(prefix)}


def continuation_logprob(logits, input_ids, start: int, torch) -> tuple[float, int]:
    """Sum autoregressive log probabilities for ids[start:], using logits at t-1."""
    if start < 1 or start >= input_ids.shape[-1]:
        raise ValueError("Invalid continuation start")
    selected_logits = logits[0, start - 1:input_ids.shape[-1] - 1, :].float()
    targets = input_ids[0, start:]
    token_logprobs = torch.log_softmax(selected_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return float(token_logprobs.sum().item()), int(targets.numel())
