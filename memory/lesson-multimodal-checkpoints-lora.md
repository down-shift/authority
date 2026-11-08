# Lesson: multimodal checkpoints change the model class and the LoRA targets

**Type:** lesson. **Audience:** every agent loading Gemma 4 / Qwen3.5+ checkpoints for training or probing.

The Phase-2 models are multimodal checkpoints, used as text models. P4.2 (2026-11-07),
checked on meta-device builds of the real configs:
- `AutoModelForCausalLM` gives `Qwen3_5ForCausalLM` (text only) for Qwen3.5-9B, but
  `Gemma4UnifiedForConditionalGeneration` for Gemma-4-12B-it. The latter has
  `embed_vision` / `embed_audio` linear projections.
- peft's `target_modules="all-linear"` therefore put LoRA on the vision and audio
  projections too. They get no gradient from text training, but they end up in the
  adapter and can trip vLLM LoRA serving (which runs these models `language_model_only`).
- The T1 scorer already fell back to `AutoModelForImageTextToText` when the causal-LM
  class refuses a checkpoint. New loaders must do the same.

**How to apply:**
- Exclude non-text modules (`src/authinv/sft/train.py` `NON_TEXT_MODULES`), and assert
  that no LoRA module matches them.
- List the targeted modules on a meta-device build
  (`with torch.device("meta"): AutoModelForCausalLM.from_config(cfg)`) before the first
  GPU run. It needs only the config download.
- Qwen3.5 is hybrid: its linear-attention projections (`in_proj_*`, `out_proj`) are
  legitimate LoRA targets alongside the attention and MLP projections.

See [[lesson-qwen38-serving-traps]].
