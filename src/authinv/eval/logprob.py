"""Tier T1: candidate log-probability margins (copied and adapted from the pilot scorer).

`continuation_encoding` jointly tokenizes prompt + candidate and scores the
candidate's tokens conditional on the stable prompt prefix. A token
straddling the boundary is included in the scored suffix and audited
(`boundary_overlap`). Prompt tokens that joint tokenization changed are never
conditioned on (`prompt_prefix_retokenized`). `sum_logprob` is a NumPy
reference for the per-candidate sum. `HFScorer` (transformers, imported
lazily) scores all candidates of a row in one right-padded forward pass,
which gives the same logits per position as scoring them one by one.
"""

from __future__ import annotations

import hashlib
from typing import Any


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
        common = 0
        for a, b in zip(prefix, ids, strict=False):
            if a != b:
                break
            common += 1
        retokenized_prefix = common < min(len(prefix), start)
        start = min(start, common)
    except (NotImplementedError, TypeError):
        ids = tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"]
        start = 0
        for a, b in zip(prefix, ids, strict=False):
            if a != b:
                break
            start += 1
        overlap = start < len(prefix)
        mode = "joint_common_prefix"
        retokenized_prefix = overlap
    if start < 1 or start >= len(ids):
        raise ValueError("Continuation requires a nonempty conditioning prefix and scored suffix")
    return {
        "input_ids": ids,
        "continuation_start": start,
        "continuation_ids": ids[start:],
        "token_count": len(ids) - start,
        "boundary_overlap": overlap,
        "boundary_mode": mode,
        "prompt_prefix_retokenized": retokenized_prefix,
        "prompt_token_count": len(prefix),
    }


def sum_logprob(logits, input_ids: list[int], start: int) -> tuple[float, int]:
    """NumPy reference: sum of log p(ids[t] | ids[:t]) for t >= start, using logits at t-1."""
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    if start < 1 or start >= len(input_ids):
        raise ValueError("Invalid continuation start")
    rows = logits[start - 1 : len(input_ids) - 1]
    rows = rows - rows.max(axis=1, keepdims=True)
    logp = rows - np.log(np.exp(rows).sum(axis=1, keepdims=True))
    targets = np.asarray(input_ids[start:])
    return float(logp[np.arange(len(targets)), targets].sum()), int(len(targets))


def candidate_audit(tokenizer, prompt: str, candidates: list[str]) -> dict:
    """Token counts and boundary flags per candidate; matched counts are reported, not enforced."""
    enc = {c: continuation_encoding(tokenizer, prompt, c) for c in candidates}
    counts = {c: e["token_count"] for c, e in enc.items()}
    return {
        "token_counts": counts,
        "matched_token_counts": len(set(counts.values())) == 1,
        "boundary_overlap": any(e["boundary_overlap"] for e in enc.values()),
        "prompt_prefix_retokenized": any(e["prompt_prefix_retokenized"] for e in enc.values()),
    }


class HFScorer:
    """Pinned-revision HF model; scores candidates as continuations of the chat-templated prompt."""

    def __init__(self, spec: dict[str, Any], model_path: str | None = None, device_map: str = "auto"):
        import torch
        import transformers

        self.torch, self.spec = torch, spec
        src = model_path or spec["name"]
        rev = None if model_path else spec["revision"]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(src, revision=rev)
        dtype = getattr(torch, spec["precision"])
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                src, revision=rev, torch_dtype=dtype, device_map=device_map
            )
        except ValueError:  # multimodal checkpoints (Gemma 4, Qwen3.5+) expose the LM via image-text-to-text
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(
                src, revision=rev, torch_dtype=dtype, device_map=device_map
            )
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.template_kwargs = {"enable_thinking": bool(spec["thinking"])} if "thinking" in spec else {}

    def render(self, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )

    def score(self, user_prompt: str, candidates: list[str]) -> dict[str, dict]:
        torch = self.torch
        prompt = self.render(user_prompt)
        encs = {c: continuation_encoding(self.tokenizer, prompt, c) for c in candidates}
        width = max(len(e["input_ids"]) for e in encs.values())
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids = torch.full((len(candidates), width), pad, dtype=torch.long)
        mask = torch.zeros((len(candidates), width), dtype=torch.long)
        for i, c in enumerate(candidates):
            row = encs[c]["input_ids"]
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        with torch.inference_mode():
            logits = self.model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
            ).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        out = {}
        for i, c in enumerate(candidates):
            e, n = encs[c], len(encs[c]["input_ids"])
            targets = ids[i, e["continuation_start"] : n].to(logp.device)
            rows = logp[i, e["continuation_start"] - 1 : n - 1]
            total = float(rows.gather(1, targets[:, None]).sum().item())
            out[c] = {k: v for k, v in e.items() if k != "input_ids"} | {
                "sum_logprob": total,
                "mean_logprob": total / e["token_count"],
            }
        return out

    def provenance(self) -> dict:
        import torch
        import transformers

        template = self.tokenizer.chat_template or ""
        return {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": self.spec["precision"],
            "device": str(self.device),
            "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest(),
            "template_kwargs": self.template_kwargs,
        }
