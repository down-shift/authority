"""Greedy chat generation through vLLM (tier T2 / Phase 0).

vLLM is imported lazily and is not part of the locked environment: runs use the
pinned vllm/vllm-openai image (H100) or a venv with the same vLLM version
(A100). The versions actually used are returned by `provenance()`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_HARMONY_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)", re.S
)
_HARMONY_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)


def split_harmony(raw: str) -> dict:
    """gpt-oss (harmony) output: the answer is the final channel; the analysis channel is kept apart.

    No final channel (e.g. truncated reasoning) yields an empty answer, which the
    strict parser labels a parse failure; the raw text is always stored.
    """
    final = _HARMONY_FINAL.search(raw)
    analysis = _HARMONY_ANALYSIS.search(raw)
    return {
        "text": final.group(1).strip() if final else "",
        "harmony_final_found": bool(final),
        "reasoning": analysis.group(1).strip() if analysis else None,
        "raw_text": raw,
    }


def _with_schema(params, structured):
    p = params.clone()
    p.structured_outputs = structured
    return p


class VLLMChat:
    def __init__(
        self, spec: dict[str, Any], model_path: str | None, gen: dict[str, Any], tensor_parallel: int = 1
    ):
        from vllm import LLM, SamplingParams

        self.spec = spec
        self.llm = LLM(
            model=model_path or spec["name"],
            revision=None if model_path else spec["revision"],
            tokenizer_revision=None if model_path else spec["revision"],
            dtype=spec["precision"],
            tensor_parallel_size=tensor_parallel,
            seed=gen["seed"],
            max_model_len=gen["max_model_len"],
            gpu_memory_utilization=gen.get("gpu_memory_utilization", 0.9),
            enforce_eager=gen.get("enforce_eager", False),
            **({"max_num_seqs": gen["max_num_seqs"]} if gen.get("max_num_seqs") else {}),
            # Multimodal checkpoints are scored on text prompts only: skip the vision/audio towers.
            **({"language_model_only": True} if spec.get("multimodal") else {}),
        )
        self.harmony = spec.get("output_format") == "harmony"
        self.params = SamplingParams(
            temperature=0.0,
            max_tokens=gen["max_tokens"],
            seed=gen["seed"],
            # Harmony channel markers are special tokens; keep them to find the final channel.
            skip_special_tokens=not self.harmony,
        )
        self.chat_kwargs = {"enable_thinking": bool(spec["thinking"])} if "thinking" in spec else None

    def generate(self, prompts: list[str], schemas: list[dict | None] | None = None) -> list[dict]:
        """Greedy chat generation; `schemas` (one per prompt) enables constrained JSON decoding."""
        convs = [[{"role": "user", "content": p}] for p in prompts]
        params = self.params
        if schemas is not None:
            from vllm.sampling_params import StructuredOutputsParams

            params = [
                self.params.clone()
                if sch is None
                else _with_schema(self.params, StructuredOutputsParams(json=sch))
                for sch in schemas
            ]
        outs = self.llm.chat(convs, params, use_tqdm=False, chat_template_kwargs=self.chat_kwargs)
        rows = []
        for o in outs:
            row = {
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "output_tokens": len(o.outputs[0].token_ids),
                "prompt_tokens": len(o.prompt_token_ids),
            }
            if self.harmony:
                row.update(split_harmony(row["text"]))
            rows.append(row)
        return rows

    def provenance(self) -> dict:
        import torch
        import transformers
        import vllm

        tok = self.llm.get_tokenizer()
        template = getattr(tok, "chat_template", None) or ""
        return {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_type": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "chat_template_kwargs": self.chat_kwargs,
        }
