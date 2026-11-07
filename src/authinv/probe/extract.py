"""Residual-stream extraction on T1 prompts (P5.1).

The prompt is the T1 / T2 user prompt, chat-templated with the generation prompt and the
model's `chat_template_kwargs` (thinking off), as the T1 scorer renders it. The extracted
vector is each selected layer's hidden state at the **last prompt token**: the position
whose next-token distribution T1 scores. Hidden-state index 0 is the embedding output;
index i is the output of block i. Batches are right-padded, and the last real token is
gathered per row, so batching does not change a row's vector (beyond floating-point
order effects, which the tests bound).

torch and transformers are imported lazily (the `inference` extra).
"""

from __future__ import annotations

import numpy as np


def resolve_layers(spec, n_hidden: int) -> list[int]:
    """`"all"`, an explicit list, or {"stride": s} (always including the last layer)."""
    if spec == "all":
        return list(range(n_hidden))
    if isinstance(spec, dict):
        out = list(range(0, n_hidden, spec["stride"]))
        return out if out[-1] == n_hidden - 1 else out + [n_hidden - 1]
    bad = [i for i in spec if not 0 <= i < n_hidden]
    if bad:
        raise ValueError(f"layers {bad} out of range 0..{n_hidden - 1}")
    return list(spec)


class Extractor:
    def __init__(self, spec: dict, model_path: str | None, dtype: str = "bfloat16"):
        import torch
        import transformers

        self.torch = torch
        src, rev = model_path or spec["name"], None if model_path else spec.get("revision")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(src, revision=rev)
        kw = {"revision": rev, "dtype": getattr(torch, dtype)}
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(src, **kw)
        except ValueError:  # multimodal checkpoints, as the T1 scorer
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(src, **kw)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.template_kwargs = {"enable_thinking": bool(spec["thinking"])} if "thinking" in spec else {}
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def render(self, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )

    def n_hidden(self) -> int:
        out = self.hidden(["x"], None)
        return out.shape[1]

    def hidden(self, prompts: list[str], layers: list[int] | None) -> np.ndarray:
        """(len(prompts), len(layers), hidden) float32 at the last prompt token."""
        torch = self.torch
        encs = [self.tokenizer(self.render(p), add_special_tokens=False)["input_ids"] for p in prompts]
        width = max(len(e) for e in encs)
        ids = torch.full((len(encs), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(encs), width), dtype=torch.long)
        for i, e in enumerate(encs):
            ids[i, : len(e)] = torch.tensor(e)
            mask[i, : len(e)] = 1
        with torch.no_grad():
            out = self.model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device), output_hidden_states=True
            )
        states = out.hidden_states
        pick = list(range(len(states))) if layers is None else layers
        last = torch.tensor([len(e) - 1 for e in encs], device=self.device)
        rows = torch.arange(len(encs), device=self.device)
        return np.stack([states[i][rows, last].float().cpu().numpy() for i in pick], axis=1)
