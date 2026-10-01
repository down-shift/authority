from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_gemma_replication_profiles_are_pinned_and_quantization_is_explicit():
    models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]
    revision = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
    for key in ("gemma3_12b_it_int8", "gemma3_12b_it_nf4"):
        assert models[key]["name"] == "google/gemma-3-12b-it"
        assert models[key]["revision"] == revision
        assert models[key]["attention_implementation"] == "sdpa"
    assert models["gemma3_12b_it_int8"]["quantization"] == "bitsandbytes_int8"
    assert models["gemma3_12b_it_nf4"]["quantization"] == "bitsandbytes_nf4"


def test_gemma_calibration_keeps_the_frozen_world_count_and_gate():
    qwen = yaml.safe_load((ROOT / "configs/authorization_lexical_symmetry.yaml").read_text())
    gemma = yaml.safe_load((ROOT / "configs/authorization_lexical_symmetry_gemma.yaml").read_text())
    assert gemma["worlds"] == qwen["worlds"] == 180
    assert gemma["gate"] == qwen["gate"]
    assert gemma["model"] == "gemma3_12b_it_int8"
