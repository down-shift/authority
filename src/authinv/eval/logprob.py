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

    def score_many(self, items: list[tuple[str, list[str]]], max_batch_tokens: int = 32768) -> list[dict]:
        """Score many (user_prompt, candidates) items, packing candidate sequences into padded batches.

        Same per-sequence result as `score` (right padding leaves earlier positions' logits unchanged).
        """
        torch = self.torch
        seqs = []  # (item index, candidate, encoding)
        for i, (prompt, cands) in enumerate(items):
            rendered = self.render(prompt)
            for c in cands:
                seqs.append((i, c, continuation_encoding(self.tokenizer, rendered, c)))
        order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]["input_ids"]))
        results: list[dict] = [{} for _ in items]
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        b = 0
        while b < len(order):
            e = b + 1  # sorted ascending, so the next sequence is the batch's longest
            while e < len(order) and (e - b + 1) * len(seqs[order[e]][2]["input_ids"]) <= max_batch_tokens:
                e += 1
            batch = [seqs[k] for k in order[b:e]]
            width = max(len(x[2]["input_ids"]) for x in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for r, (_, _, enc) in enumerate(batch):
                ids[r, : len(enc["input_ids"])] = torch.tensor(enc["input_ids"])
                mask[r, : len(enc["input_ids"])] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
            for r, (i, c, enc) in enumerate(batch):
                n, st = len(enc["input_ids"]), enc["continuation_start"]
                lp = torch.log_softmax(logits[r, st - 1 : n - 1].float(), dim=-1)
                tgt = ids[r, st:n].to(lp.device)
                total = float(lp.gather(1, tgt[:, None]).sum().item())
                results[i][c] = {k: v for k, v in enc.items() if k != "input_ids"} | {
                    "sum_logprob": total,
                    "mean_logprob": total / enc["token_count"],
                }
            b = e
        return results

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


class VLLMScorer:
    """T1 via vLLM prompt log-probabilities: same joint tokenization (exact token ids), same continuation sum.

    Each (prompt, candidate) sequence is passed as token ids; vLLM returns the log-probability of every prompt
    token, and prefix caching shares the prompt between a row's candidates. Engineering backend only: the
    quantity is identical to `HFScorer.score` up to kernel numerics (checked against HF before the sweep).
    """

    def __init__(self, spec: dict[str, Any], model_path: str | None, gen: dict[str, Any]):
        from vllm import LLM, SamplingParams

        self.spec = spec
        self.llm = LLM(
            model=model_path or spec["name"],
            revision=None if model_path else spec["revision"],
            tokenizer_revision=None if model_path else spec["revision"],
            dtype=spec["precision"],
            seed=gen["seed"],
            max_model_len=gen["max_model_len"],
            gpu_memory_utilization=gen.get("gpu_memory_utilization", 0.9),
            enable_prefix_caching=True,
            **({"max_num_seqs": gen["max_num_seqs"]} if gen.get("max_num_seqs") else {}),
            **({"language_model_only": True} if spec.get("multimodal") else {}),
        )
        self.params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        self.tokenizer = self.llm.get_tokenizer()
        self.template_kwargs = {"enable_thinking": False} if "thinking" in spec else {}

    def render(self, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )

    def score_many(self, items: list[tuple[str, list[str]]], max_batch_tokens: int = 0) -> list[dict]:
        from vllm.inputs import TokensPrompt

        seqs = []
        for i, (prompt, cands) in enumerate(items):
            rendered = self.render(prompt)
            for c in cands:
                seqs.append((i, c, continuation_encoding(self.tokenizer, rendered, c)))
        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=e["input_ids"]) for _, _, e in seqs], self.params, use_tqdm=False
        )
        results: list[dict] = [{} for _ in items]
        for (i, c, enc), o in zip(seqs, outs, strict=True):
            ids, st = enc["input_ids"], enc["continuation_start"]
            total = float(sum(o.prompt_logprobs[t][ids[t]].logprob for t in range(st, len(ids))))
            results[i][c] = {k: v for k, v in enc.items() if k != "input_ids"} | {
                "sum_logprob": total,
                "mean_logprob": total / enc["token_count"],
            }
        return results

    def provenance(self) -> dict:
        import torch
        import vllm

        template = getattr(self.tokenizer, "chat_template", None) or ""
        return {
            "backend": "vllm",
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu_type": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "dtype": self.spec["precision"],
            "template_kwargs": self.template_kwargs,
            "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest(),
        }
