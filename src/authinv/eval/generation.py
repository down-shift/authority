"""Greedy chat generation through vLLM (tier T2 / Phase 0).

vLLM is imported lazily and is not part of the locked environment: runs use the
pinned vllm/vllm-openai image (H100) or a venv with the same vLLM version
(A100). The versions actually used are returned by `provenance()`.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_HARMONY_FINAL = re.compile(
    # The final header may carry a content-type token (`final <|constrain|>json<|message|>`), which gpt-oss
    # emits when the prompt lists tools (T3); T2 outputs never contain one.
    r"<\|channel\|>final(?:\s*<\|constrain\|>[^<]*)?<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)",
    re.S,
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


_HARMONY_BODY_END = re.compile(r"<\|end\|>|<\|call\|>|<\|return\|>")
_HARMONY_CHANNEL = re.compile(r"<\|channel\|>\s*(\w+)")
_HARMONY_RECIPIENT = re.compile(r"\bto=([^\s<]+)")
_TOOL_NAMESPACE = re.compile(r"^(?:functions|tools?)\.")


def harmony_agent_turn(raw: str) -> str:
    """T3 (agent-v4): the text a gpt-oss turn is scored on. T2 keeps `split_harmony`.

    gpt-oss may call a tool natively: a message addressed to a recipient,
    `<|channel|>commentary to=delete <|constrain|>json<|message|>{"resource": "r_lynx"}`,
    instead of writing the protocol object in the final channel. The first
    message that has a recipient or is in the final channel is the turn:
    - final channel: the same text as `split_harmony` (agent-v3), unchanged;
    - a recipient: a body that is already a protocol object is the turn as
      written; any other JSON object body is the arguments of the recipient,
      after one leading `functions.`, `tool.` or `tools.` namespace is removed
      (the name must then match a tool exactly, or the sandbox scores it
      unknown); a non-JSON body is returned as is and fails to parse.
    A message without a `<|message|>` token is malformed and is not read.
    """
    for segment in raw.split("<|start|>assistant"):
        header, sep, body = segment.partition("<|message|>")
        if not sep:
            continue
        recipient = _HARMONY_RECIPIENT.search(header)
        if recipient:
            body = _HARMONY_BODY_END.split(body, maxsplit=1)[0].strip()
            try:
                obj = json.loads(body)
            except ValueError:
                return body
            if not isinstance(obj, dict):
                return body
            if set(obj) == {"tool", "arguments"}:
                return body
            name = _TOOL_NAMESPACE.sub("", recipient.group(1))
            return json.dumps({"tool": name, "arguments": obj})
        channel = _HARMONY_CHANNEL.search(header)
        if channel and channel.group(1) == "final":
            break
    return split_harmony(raw)["text"]


def _with_schema(params, structured):
    p = params.clone()
    p.structured_outputs = structured
    return p


class VLLMChat:
    """`lora` (P4.2b): {"path": adapter dir, "rank": r} serves one LoRA adapter on the base model;
    every request then goes through it. Without `lora` the engine is exactly the T2 engine."""

    def __init__(
        self,
        spec: dict[str, Any],
        model_path: str | None,
        gen: dict[str, Any],
        tensor_parallel: int = 1,
        lora: dict[str, Any] | None = None,
    ):
        from vllm import LLM, SamplingParams

        self.spec = spec
        self.lora_request = None
        lora_kw = {}
        if lora:
            from vllm.lora.request import LoRARequest

            lora_kw = {"enable_lora": True, "max_lora_rank": int(lora["rank"]), "max_loras": 1}
            self.lora_request = LoRARequest("adapter", 1, str(lora["path"]))
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
            **lora_kw,
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

    def generate(
        self, prompts: list[str], schemas: list[dict | None] | None = None, max_tokens: int | None = None
    ) -> list[dict]:
        """Greedy chat generation; `schemas` (one per prompt) enables constrained JSON decoding.

        `max_tokens` overrides the configured limit for this call (Phase-3 arms run two
        stages with different limits); T2 never passes it.
        """
        return self.generate_chats([[{"role": "user", "content": p}] for p in prompts], schemas, max_tokens)

    def generate_chats(
        self,
        convs: list[list[dict]],
        schemas: list[dict | None] | None = None,
        max_tokens: int | None = None,
    ) -> list[dict]:
        """`generate` on whole conversations (system + user turns; the BFCL check, P4.2b)."""
        params = self.params
        if max_tokens is not None:
            params = self.params.clone()
            params.max_tokens = max_tokens
        if schemas is not None:
            from vllm.sampling_params import StructuredOutputsParams

            base = params
            params = [
                base.clone() if sch is None else _with_schema(base, StructuredOutputsParams(json=sch))
                for sch in schemas
            ]
        outs = self.llm.chat(
            convs, params, use_tqdm=False, chat_template_kwargs=self.chat_kwargs, **self._lora()
        )
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

    def chat_texts(self, conversations: list[list[dict]]) -> list[str] | list[dict]:
        """Greedy reply text for multi-turn conversations (T3).

        Harmony models return `{"text": <scored turn>, "raw": <raw output>}` (agent-v4,
        `harmony_agent_turn`) so the raw harmony text is stored; other models return plain text.
        """
        outs = self.llm.chat(
            conversations, self.params, use_tqdm=False, chat_template_kwargs=self.chat_kwargs, **self._lora()
        )
        texts = [o.outputs[0].text for o in outs]
        if self.harmony:
            return [{"text": harmony_agent_turn(t), "raw": t} for t in texts]
        return texts

    def _lora(self) -> dict:
        return {"lora_request": self.lora_request} if self.lora_request is not None else {}

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
