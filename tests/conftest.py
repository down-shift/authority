import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_tiny_benchmark(root: Path, n_worlds: int = 3) -> tuple[Path, list[dict], dict[str, dict]]:
    """A frozen-benchmark directory (dataset.jsonl, manifest, worlds.jsonl) from a few synthetic worlds.

    Same layout as scripts/authinv_build_benchmark.py; rendering needs cedarpy (Cedar text).
    """
    from authinv import provenance
    from authinv.benchmark import World, non_degenerate, world_rows
    from authinv.policy import to_dict
    from authinv.sources.synthetic import generate

    ws, _ = generate("repo", replicates=1, seed=3)
    picked = [
        w
        for w in (World(p, ("User",), ("Repo",), "synthetic", "synthetic", dict(p.meta)) for p in ws)
        if non_degenerate(w, 4)
    ]
    rows, out = [], []
    for w in picked[:n_worlds]:
        rs, extra = world_rows(w, per_label=2, seed=1)
        rows += rs
        out.append(
            {
                "world_id": w.policy.policy_id,
                "source": w.source,
                "source_kind": w.source_kind,
                "principal_types": list(w.principal_types),
                "resource_types": list(w.resource_types),
                "orig": to_dict(w.policy),
                "swap": to_dict(extra["swap_policy"]),
                "swap_map": extra["swap_map"],
            }
        )
    d = root / "bench"
    d.mkdir()
    provenance.write_jsonl(d / "dataset.jsonl", rows)
    provenance.write_json(
        d / "dataset_manifest.json",
        {"sha256": provenance.sha256_file(d / "dataset.jsonl"), "rows": len(rows)},
    )
    provenance.write_jsonl(d / "worlds.jsonl", out)
    return d, rows, {w["world_id"]: w for w in out}


def build_tiny_causal_lm(root: Path) -> Path:
    """A tiny random Llama, byte-level tokenizer and chat template, saved under root/tiny (no download)."""
    import pytest

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
