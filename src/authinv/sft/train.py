"""LoRA SFT for Phase 4 (P4.2, RESEARCH_PLAN §3 Phase 4 task 2).

One run = one (model, arm, seed). The arm picks the P4.1 set it trains on:
``multi`` (`train_multi.jsonl`, four renderings) or ``control`` (the token-matched
single-format set of that model). The no-SFT arm is the base model and has no run.

**Training sequences match inference.** The prompt part is exactly what the eval
harness sends: the model's chat template over the single user message with
``add_generation_prompt=True`` and the model's ``chat_template_kwargs`` (e.g.
``enable_thinking: False``, which makes Qwen's template open and close an empty
think block). The target is the answer text plus the end-of-turn text the template
puts after an assistant message, tokenized after the prompt. Loss is on the target
tokens only (prompt labels are -100).

Determinism: the seed fixes the LoRA initialization (`torch.manual_seed` before
`get_peft_model`), the example order (a seeded shuffle per epoch) and dropout.
Heavy dependencies (torch, transformers, peft) are imported lazily.

`TRAINER_VERSION` covers the sequence construction, the loss, the schedule and the
batching. Bump it for any change.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

TRAINER_VERSION = "sft-trainer-v1"


def load_set(data_dir: Path, name: str) -> list[dict]:
    """A P4.1 set, checked against its manifest (sha256, rows) and the disjointness verdict."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if not manifest["disjointness"]["passed"]:
        raise ValueError("SFT data failed its disjointness check")
    info = manifest["files"].get(name)
    if info is None:
        raise ValueError(f"{name} is not in the SFT data manifest")
    path = data_dir / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError(f"{name}: sha256 differs from the manifest")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != info["rows"]:
        raise ValueError(f"{name}: {len(rows)} rows, manifest says {info['rows']}")
    return rows


def set_name(arm: str, model_key: str, arms: dict) -> str:
    if arm not in arms:
        raise ValueError(f"unknown arm {arm!r}; config arms: {sorted(arms)}")
    return arms[arm].format(model=model_key)


def _ids(enc) -> list[int]:
    ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
    return list(ids[0] if ids and isinstance(ids[0], list) else ids)


def encode(tokenizer, messages: list[dict], kwargs: dict, max_len: int) -> dict:
    """input_ids = generation prompt + answer + end-of-turn; labels mask the prompt."""
    user, answer = messages
    if user["role"] != "user" or answer["role"] != "assistant":
        raise ValueError("an SFT example is one user turn and one assistant turn")
    prompt = _ids(
        tokenizer.apply_chat_template(
            [user], add_generation_prompt=True, tokenize=True, return_dict=True, **kwargs
        )
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    at = full_text.rfind(answer["content"])
    if at < 0:
        raise ValueError("chat template does not contain the answer text")
    end_of_turn = full_text[at + len(answer["content"]) :]
    target = _ids(tokenizer(answer["content"] + end_of_turn, add_special_tokens=False))
    ids = prompt + target
    if len(ids) > max_len:
        raise ValueError(f"example of {len(ids)} tokens exceeds max_len {max_len}")
    return {"input_ids": ids, "labels": [-100] * len(prompt) + target}


def batches(encoded: list[dict], batch_size: int, order: list[int], pad_id: int):
    import torch

    for start in range(0, len(order), batch_size):
        chunk = [encoded[i] for i in order[start : start + batch_size]]
        width = max(len(e["input_ids"]) for e in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, e in enumerate(chunk):
            n = len(e["input_ids"])
            ids[i, :n] = torch.tensor(e["input_ids"])
            labels[i, :n] = torch.tensor(e["labels"])
            mask[i, :n] = 1
        yield {"input_ids": ids, "labels": labels, "attention_mask": mask}


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup, then cosine decay to 0."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def mean_loss(model, encoded: list[dict], batch_size: int, pad_id: int, device) -> float:
    """Token-weighted mean loss on the target tokens."""
    import torch

    total, tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for b in batches(encoded, batch_size, list(range(len(encoded))), pad_id):
            b = {k: v.to(device) for k, v in b.items()}
            n = int((b["labels"][:, 1:] != -100).sum())
            total += float(model(**b).loss) * n
            tokens += n
    model.train()
    return total / tokens if tokens else float("nan")


def train(
    spec: dict,
    model_path: str | None,
    train_rows: list[dict],
    val_rows: list[dict],
    cfg: dict,
    seed: int,
    out: Path,
    max_steps: int | None = None,
) -> dict:
    """Train one LoRA adapter; writes train_log.jsonl and adapter/ under `out`, returns the metrics."""
    import peft
    import torch
    import transformers

    t0 = time.perf_counter()
    lcfg, tcfg = cfg["lora"], cfg["training"]
    random.seed(seed)
    torch.manual_seed(seed)
    source = model_path or spec["name"]
    revision = None if model_path else spec["revision"]
    tok = transformers.AutoTokenizer.from_pretrained(source, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = {"enable_thinking": bool(spec["thinking"])} if "thinking" in spec else {}
    auto = getattr(transformers, cfg.get("auto_class", {}).get(spec["key"], "AutoModelForCausalLM"))
    dtype = getattr(torch, tcfg.get("dtype", "bfloat16"))
    model = auto.from_pretrained(source, revision=revision, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if tcfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    torch.manual_seed(seed)  # LoRA initialization
    lora = peft.LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = [encode(tok, r["messages"], kwargs, tcfg["max_len"]) for r in train_rows]
    val = [encode(tok, r["messages"], kwargs, tcfg["max_len"]) for r in val_rows]
    bs, accum = tcfg["batch_size"], tcfg["grad_accum"]
    per_epoch = math.ceil(len(enc) / bs)
    total = math.ceil(per_epoch * tcfg["epochs"] / accum)
    if max_steps:
        total = min(total, max_steps)
    warmup = max(1, int(tcfg["warmup_ratio"] * total))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    rng = random.Random(seed)
    step, micro, seen_tokens = 0, 0, 0
    log = (out / "train_log.jsonl").open("w")
    model.train()
    done = False
    for epoch in range(math.ceil(tcfg["epochs"])):
        order = list(range(len(enc)))
        rng.shuffle(order)
        for b in batches(enc, bs, order, tok.pad_token_id):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss / accum
            loss.backward()
            micro += 1
            seen_tokens += int((b["labels"] != -100).sum())
            if micro % accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, tcfg["learning_rate"])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % tcfg["log_every"] == 0 or step == total:
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(loss.detach()) * accum,
                            "lr": opt.param_groups[0]["lr"],
                            "grad_norm": float(norm),
                            "target_tokens": seen_tokens,
                        }
                    )
                    + "\n"
                )
                log.flush()
            if step >= total:
                done = True
                break
        if done:
            break
    log.close()
    val_loss = mean_loss(model, val, bs, tok.pad_token_id, device)
    model.save_pretrained(out / "adapter")
    adapter = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((out / "adapter").iterdir())
        if p.is_file()
    }
    return {
        "trainer_version": TRAINER_VERSION,
        "steps": step,
        "planned_steps": total,
        "warmup_steps": warmup,
        "train_examples": len(enc),
        "val_examples": len(val),
        "target_tokens_seen": seen_tokens,
        "trainable_parameters": trainable,
        "val_loss": val_loss,
        "adapter_sha256": adapter,
        "seconds": time.perf_counter() - t0,
        "device": str(device),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "chat_template_kwargs": kwargs,
    }
