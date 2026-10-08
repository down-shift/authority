# Phase 0 addendum preregistration: the same kill test on current models

- status: preregistered (committed before any addendum-model inference)
- step: P0.5 · approved by Jerzy 2026-10-08 (model-roster refresh) · spec: `docs/RESEARCH_PLAN.md` §3 Phase 0 addendum
- config: `configs/authinv/phase0_addendum.yaml` · harness: `scripts/authinv_phase0.py` (same code path as Phase 0)

## Why

The Phase-0 models (Qwen3, Gemma 3, Llama 3.3) have newer open-weight
successors. This addendum asks the Phase-0 question of current models, so the
G0 decision can take them into account. **It is not a Gate-0 input.** Gate 0
stays exactly as preregistered in `docs/prereg/phase0.md` (`89b91c0`), and is
computed on the four Phase-0 models only.

## Everything identical to Phase 0

- the same frozen data: the 3,600 rows, hash-pinned in the config;
- the same strict-v1 parser and lenient sensitivity analysis;
- the same metric definitions, flip definitions, bootstrap (4,000
  world-clustered replicates, seed 20261008), and primary task (application);
- greedy decoding (temperature 0), vLLM 0.29.0;
- no exclusions; failed runs are resumed with provenance checks.

`docs/prereg/phase0.md` defines all of these, and they apply here verbatim.

## What differs

| key | checkpoint @ revision | settings |
|---|---|---|
| qwen3_8_27b | Qwen/Qwen3.8-27B @ 1d4bf0f2 | bf16, thinking off, text-only loading |
| gemma4_31b | google/gemma-4-31B-it @ 842da379 | bf16, thinking off, text-only loading |
| gpt_oss_120b | openai/gpt-oss-120b @ b5c939de | native MXFP4, checkpoint-default reasoning effort |

- **Text-only loading:** Qwen3.8 and Gemma 4 are multimodal checkpoints.
  They're loaded with `language_model_only=True`, so the vision and audio
  towers are skipped. The prompts are text, so this doesn't change the
  computation for these inputs.
- **gpt-oss answers in the harmony format.** It writes an analysis channel,
  then a final channel. The **answer is the final channel's content**, and
  that's what the strict parser sees. The analysis channel and the raw text
  are stored, but they're never parsed or scored. If no final channel is
  produced within the token budget, the answer is empty and is labelled
  `parse_failure`. Its budget is `max_tokens 4096` with `max_model_len 8192`
  (from the config's `generation_overrides`). Every other model keeps 64
  tokens. Reasoning effort is left at the checkpoint default and is not tuned.
- **Holm family:** the three addendum models (application worst-case-gap
  bootstrap p-values).

## Reported, not decided

For each model, the report gives the same table as Phase 0: per-rendering
accuracy with fail-open, fail-closed and parse-failure rates; worst-case gap;
all-renderings-correct; rendering disagreement; deny→allow and allow→deny
flips; lenient sensitivity. These appear next to the Phase-0 models.

For descriptive comparison only, the report also says whether each model
*would* meet the Phase-0 per-model criteria:
- a gap of ≥ 5 pp with the CI excluding 0, or
- disagreement ≥ 5%.

This is labelled as outside Gate 0. The loop recommends nothing about G0 from
these numbers; Jerzy decides.

## Disclosure of pre-registration engineering

Before this file was committed, the new code paths were smoke-tested on H100
GPU 7. The two tests used models that are **not** in the addendum set,
on at most 20 rows each, and the outputs were discarded:
- the harmony final-channel extraction, with `gpt_oss_20b`;
- text-only multimodal loading, with `gemma4_e4b`.

No addendum model was run before this commit.
