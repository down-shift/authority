"""Greedy chat generation through vLLM (tier T2 / Phase 0).

vLLM is imported lazily and is not part of the locked environment: runs use the
pinned vllm/vllm-openai image (H100) or a venv with the same vLLM version
(A100). The versions actually used are returned by `provenance()`.
"""

from __future__ import annotations

import hashlib
from typing import Any


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
        )
        self.params = SamplingParams(temperature=0.0, max_tokens=gen["max_tokens"], seed=gen["seed"])
        self.chat_kwargs = {"enable_thinking": bool(spec["thinking"])} if "thinking" in spec else None

    def generate(self, prompts: list[str]) -> list[dict]:
        convs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(convs, self.params, use_tqdm=False, chat_template_kwargs=self.chat_kwargs)
        return [
            {
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "output_tokens": len(o.outputs[0].token_ids),
                "prompt_tokens": len(o.prompt_token_ids),
            }
            for o in outs
        ]

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
