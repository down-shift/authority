"""Hugging Face causal LM adapter using the tokenizer's official chat template."""
from __future__ import annotations

from authority_leakage.models.base import Generation, ModelAdapter
from authority_leakage.schemas import Message
from authority_leakage.scoring import conditional_logprob
import hashlib


class HFAdapter(ModelAdapter):
    def __init__(self, model_name: str, revision: str | None = None, device: str = "auto",
                 enable_thinking: bool | None = None, quantization: str | None = None,
                 attention_implementation: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HF inference requires torch and transformers; run `uv sync --extra inference`") from exc
        self.torch = torch
        self.model_name = model_name
        self.requested_revision = revision
        self.enable_thinking = enable_thinking
        self.quantization = quantization
        self.attention_implementation = attention_implementation or (
            "sdpa" if quantization == "bitsandbytes_int8" else None
        )
        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, **kwargs)
        if self.tokenizer.chat_template is None:
            raise ValueError(f"{model_name} has no official tokenizer chat template")
        if quantization not in (None, "bitsandbytes_int8"):
            raise ValueError(f"Unsupported quantization: {quantization}")
        if quantization == "bitsandbytes_int8":
            if device not in ("auto", "cuda"):
                raise ValueError("bitsandbytes int8 loading requires device='auto' or 'cuda'")
            if not torch.cuda.is_available():
                raise RuntimeError("bitsandbytes int8 loading requires a CUDA device in this adapter")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("Install the quantization extra for bitsandbytes int8 loading") from exc
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=False, torch_dtype="auto",
                quantization_config=quant_config, device_map="auto",
                attn_implementation=self.attention_implementation, **kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=False, torch_dtype="auto", **kwargs,
            )
        if self.enable_thinking is None and getattr(self.model.config, "model_type", "") == "qwen3":
            self.enable_thinking = False
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.requested_device = device
        if quantization is None:
            self.model.to(device)
        self.input_device = self.model.get_input_embeddings().weight.device
        self.device = str(self.input_device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def render(self, messages: list[Message]) -> str:
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False, add_generation_prompt=True, **template_kwargs,
        )

    def _prompt_ids(self, messages: list[Message]):
        prompt = self.render(messages)
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.input_device)
        return prompt, ids

    def candidate_token_ids(self, candidates: list[str]) -> dict[str, list[int]]:
        return {candidate: self.tokenizer(candidate, add_special_tokens=False)["input_ids"] for candidate in candidates}

    def score_candidates(self, messages: list[Message], candidates: list[str]) -> dict[str, float]:
        torch = self.torch
        _, prompt_ids = self._prompt_ids(messages)
        if prompt_ids.shape[1] == 0:
            raise ValueError("Empty rendered prompt")
        scores: dict[str, float] = {}
        for candidate, token_ids in self.candidate_token_ids(candidates).items():
            if not token_ids:
                raise ValueError(f"Empty candidate: {candidate!r}")
            suffix = torch.tensor([token_ids], dtype=prompt_ids.dtype, device=self.input_device)
            full = torch.cat((prompt_ids, suffix), dim=1)
            with torch.inference_mode():
                logits = self.model(input_ids=full).logits
                positions = torch.arange(prompt_ids.shape[1] - 1, full.shape[1] - 1, device=self.input_device)
                next_token_logits = logits[0, positions, :].float()
                selected_logits = next_token_logits.cpu().numpy()
            scores[candidate] = conditional_logprob(selected_logits, token_ids)
        return scores

    def generate(self, messages: list[Message], max_new_tokens: int) -> Generation:
        torch = self.torch
        prompt, input_ids = self._prompt_ids(messages)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
        generated_ids = output.sequences[0, input_ids.shape[1]:]
        selected_logprobs = []
        for step, logits in enumerate(output.scores):
            token_id = generated_ids[step]
            selected_logprobs.append(float(torch.log_softmax(logits[0].float(), dim=-1)[token_id].item()))
        return Generation(
            text=self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            rendered_prompt=prompt,
            generated_token_logprobs=selected_logprobs,
        )

    def provenance(self) -> dict:
        import transformers
        return {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer.name_or_path,
            "requested_revision": self.requested_revision,
            "model_commit": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_commit": self.tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_revision": self.requested_revision,
            "chat_template_sha256": hashlib.sha256((self.tokenizer.chat_template or "").encode()).hexdigest(),
            "device": self.device,
            "requested_device": self.requested_device,
            "quantization": self.quantization,
            "enable_thinking": self.enable_thinking,
            "attention_implementation": self.attention_implementation or "model_default",
            "model_dtype": str(next(self.model.parameters()).dtype),
            "transformers_version": transformers.__version__,
            "torch_version": self.torch.__version__,
            "model_generation_config": self.model.generation_config.to_dict(),
        }
