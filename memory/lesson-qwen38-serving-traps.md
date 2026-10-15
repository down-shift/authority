# Lesson: Qwen3.8 serving traps (think opener in the prompt, Mamba cache)

**Type:** lesson. **Audience:** every agent running Qwen3.8 (or any hybrid or
thinking model) through the authinv runners.

P2.5 (2026-10-12 to 2026-10-14) hit two problems with Qwen3.8-27B:

- **The think opener lives in the prompt.** With `enable_thinking: true`, the
  chat template ends the prompt with `<think>`. The output starts with the
  reasoning and contains only the closing `</think>`. Parser `json-strict-v2`
  stripped only a block that opens with `<think>`, so the whole thinking run
  scored 0% (99% `parse_failure`). Fixed in `json-strict-v3` (prereg
  amendment, `docs/CHANGELOG.md`). Before trusting a new thinking model, look
  at a few raw `text` fields from the first batch, not just the category
  counts.
- **Mamba cache limits concurrency.** Qwen3.8 is hybrid Mamba-attention. vLLM's
  default `max_num_seqs` of 1,024 fails at engine start with "max_num_seqs
  (1024) exceeds available Mamba cache blocks (129)" at 0.9 memory on one H100.
  Pass `--max-num-seqs 128` and disclose it in the report. It changes
  throughput, not greedy outputs.
- **Thinking off doesn't stop it reasoning.** With thinking disabled, Qwen3.8
  still writes Markdown step-by-step reasoning on some renderings (above all
  Rego) and hits the 256-token T2 limit. Those rows are `parse_failure`, kept
  as their own category. T1 is the decision-level check.

**How to apply:**
- Smoke-test the first batch of any new model family by reading raw outputs.
- For hybrid models, set `--max-num-seqs` from the error message.
- Never change a frozen token limit or parser after seeing data. Bump the
  version, amend the prereg, and re-score or rerun.

See [[reference-dgx-usage]].
