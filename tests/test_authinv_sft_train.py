"""P4.2 LoRA SFT trainer: config and data checks, the schedule, and a CPU smoke test on a tiny model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from authinv.sft import train as sft

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("authinv_sft_train", ROOT / "scripts" / "authinv_sft_train.py")
runner = importlib.util.module_from_spec(spec)
sys.modules["authinv_sft_train"] = runner
spec.loader.exec_module(runner)
CFG = yaml.safe_load((ROOT / "configs/authinv/sft_lora.yaml").read_text())


def _example(i: int, decision: str) -> dict:
    prompt = f"Policy: rule r{i} allows read. Request {i}: read? Answer with JSON only."
    return {
        "example_id": f"e{i}",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps({"decision": decision})},
        ],
    }


def _data(tmp_path: Path, n: int = 8, passed: bool = True) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True)
    sets = {
        "train_multi.jsonl": [_example(i, "allow" if i % 2 else "deny") for i in range(n)],
        "val_multi.jsonl": [_example(100 + i, "deny") for i in range(2)],
        "control_qwen3_5_9b.jsonl": [_example(200 + i, "allow") for i in range(n)],
    }
    files = {}
    for name, rows in sets.items():
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[name] = {"sha256": hashlib.sha256((d / name).read_bytes()).hexdigest(), "rows": len(rows)}
    (d / "manifest.json").write_text(json.dumps({"files": files, "disjointness": {"passed": passed}}))
    return d


def test_plan_lists_three_seeds_per_model_and_arm():
    runs = runner.plan(CFG)
    assert len(runs) == 2 * 2 * 3 and {r["arm"] for r in runs} == {"multi", "control"}
    assert len(CFG["seeds"]) == 3 and 16 <= CFG["lora"]["r"] <= 64


def test_load_set_and_run_checks_refuse_mismatches(tmp_path):
    d = _data(tmp_path)
    assert len(sft.load_set(d, "train_multi.jsonl")) == 8
    assert sft.set_name("control", "qwen3_5_9b", CFG["arms"]) == "control_qwen3_5_9b.jsonl"
    with pytest.raises(ValueError, match="unknown arm"):
        sft.set_name("no_sft", "qwen3_5_9b", CFG["arms"])
    (d / "val_multi.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="sha256"):
        sft.load_set(d, "val_multi.jsonl")
    bad = _data(tmp_path / "b", passed=False)
    with pytest.raises(ValueError, match="disjointness"):
        sft.load_set(bad, "train_multi.jsonl")
    with pytest.raises(SystemExit, match="data manifest"):
        runner.check(CFG, _data(tmp_path / "c"), "qwen3_5_9b", "multi", CFG["seeds"][0], False)
    good = {
        **CFG,
        "data_manifest_sha256": hashlib.sha256(
            (tmp_path / "c" / "data" / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    runner.check(good, tmp_path / "c" / "data", "qwen3_5_9b", "multi", CFG["seeds"][0], False)
    with pytest.raises(SystemExit, match="seed"):
        runner.check(good, tmp_path / "c" / "data", "qwen3_5_9b", "multi", 7, False)


def test_lr_schedule_warms_up_then_decays_to_zero():
    lrs = [sft.lr_at(s, 100, 10, 1e-4) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-5) and lrs[9] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:], strict=False)) and lrs[-1] < 1e-6


def _tiny_model(root: Path) -> Path:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, pre_tokenizers

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for t in ("<pad>", "</s>", "<s>"):
        vocab[t] = len(vocab)
    tk = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(["<pad>", "</s>", "<s>"])
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", eos_token="</s>", bos_token="<s>"
    )
    tok.chat_template = (
        "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</s>{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    d = root / "tiny"
    tok.save_pretrained(d)
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    transformers.LlamaForCausalLM(cfg).save_pretrained(d)
    return d


def _smoke_cfg() -> dict:
    return {
        **CFG,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.05, "target_modules": "all-linear"},
        "training": {
            **CFG["training"],
            "batch_size": 2,
            "grad_accum": 1,
            "max_len": 512,
            "gradient_checkpointing": False,
            "dtype": "float32",
            "log_every": 1,
        },
    }


def test_encode_masks_the_prompt_and_trains_on_answer_and_end_of_turn(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(_tiny_model(tmp_path))
    ex = _example(1, "allow")
    e = sft.encode(tok, ex["messages"], {}, 512)
    prompt = tok.apply_chat_template([ex["messages"][0]], add_generation_prompt=True, tokenize=False)
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert e["labels"][:n_prompt] == [-100] * n_prompt
    target = tok.decode([t for t in e["labels"] if t != -100])
    assert target == '{"decision": "allow"}</s>'
    with pytest.raises(ValueError, match="exceeds max_len"):
        sft.encode(tok, ex["messages"], {}, 10)


def test_cpu_smoke_training_is_seed_deterministic(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    model_dir = _tiny_model(tmp_path)
    data = _data(tmp_path)
    cfg = _smoke_cfg()
    rows, val = sft.load_set(data, "train_multi.jsonl"), sft.load_set(data, "val_multi.jsonl")
    spec_ = {"key": "tiny", "name": "tiny", "revision": None}
    outs = []
    for i, seed in enumerate((1, 1, 2)):
        out = tmp_path / f"run{i}"
        out.mkdir()
        outs.append(sft.train(spec_, str(model_dir), rows, val, cfg, seed, out, max_steps=3))
    a, b, c = outs
    assert a["steps"] == 3 and a["val_loss"] == a["val_loss"] and a["trainable_parameters"] > 0
    assert a["adapter_sha256"] == b["adapter_sha256"] != c["adapter_sha256"]
    assert "adapter_model.safetensors" in a["adapter_sha256"]
    log = [json.loads(line) for line in (tmp_path / "run0" / "train_log.jsonl").read_text().splitlines()]
    assert [x["step"] for x in log] == [1, 2, 3]


def test_runner_smoke_run_writes_provenance(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    model_dir = _tiny_model(tmp_path)
    data = _data(tmp_path)
    cfg = {
        **_smoke_cfg(),
        "data_manifest_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    sys.argv = [
        "x",
        "run",
        "--config",
        str(cfg_path),
        "--model",
        "qwen3_5_9b",
        "--arm",
        "control",
        "--seed",
        "5",
        "--data",
        str(data),
        "--model-dir",
        str(model_dir),
        "--output-root",
        str(tmp_path / "out"),
        "--max-steps",
        "2",
    ]
    runner.main()
    run = next((tmp_path / "out").iterdir())
    assert json.loads((run / "run_status.json").read_text())["status"] == "smoke"
    conf = json.loads((run / "config.json").read_text())
    assert conf["data"]["train_set"] == "control_qwen3_5_9b.jsonl" and conf["arm"] == "control"
    m = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((run / "artifact_sha256.json").read_text())
    assert any(k.startswith("adapter/") for k in manifest) and m["steps"] == 2
