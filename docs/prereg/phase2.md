# Phase 2 preregistration: tiers T1 and T2 on the model sweep

- status: preregistered (committed before any Phase-2 sweep inference)
- step: P2.3 (runs in P2.4–P2.6; report in P2.9) · spec: `docs/RESEARCH_PLAN.md` §3 Phase 2 (amended 2026-10-08), §4
- frozen input: Phase-1 benchmark v2, `dataset.jsonl` sha256 `0b602a1701bc52aaaf676973f24ea87b1e27acd9029f5e0523174d51894a4f83`. G1 passed on it (`docs/experiments/g1-spotcheck-round2.md`), and the runner refuses any other dataset.
- config: `configs/authinv/t2.yaml` · harness: `scripts/authinv_eval.py`, `src/authinv/eval/{structured,invariance,generation,logprob}.py`
- T3 is preregistered separately, before its runs (P2.8).

## Questions

- **Q1 (headline).** On engine-certified equivalent renderings of real and
  synthetic authorization policies, does a model's application accuracy
  depend on the rendering? Measured as the worst-case gap.
- **Q2.** When models err, is the error direction deny→allow (fail-open) or
  allow→deny (fail-closed)? Does the direction depend on the rendering or the
  model?
- **Q3.** Is "interpretation correct but application wrong" (the
  dissociation) rendering-dependent?
- **Q4.** How does the worst-case gap relate to model scale across the sweep?

We state no directional prediction for Q1–Q4. Phase 0 found the effect at
8B but not at ≥27B on toy worlds, which isn't informative for this
benchmark's harder worlds.

## Data (frozen)

- 1,984 worlds: 544 real (CedarBench 494, Quacky 50) and 1,440 synthetic.
- 6 renderings:
  - nl_statement (nl-v2), owner_statement (owner-v2), table (table-v2),
    json_policy (json-v2);
  - executable / Cedar (cedar-v1), rego (rego-v1).
- 2 name assignments (orig / swap) × 4 requests per world (2 allow, 2 deny).
- 172,296 rows: application 95,232 and interpretation 77,064. Prompt template
  prompt-v1.

## Models (`configs/authinv/models.yaml`, set `phase2_sweep`, pinned revisions)

| key | checkpoint | params (B) | precision / format | T1 | T2 |
|---|---|---:|---|:-:|:-:|
| qwen3_5_4b | Qwen/Qwen3.5-4B | 4.7 | bf16, thinking off | ✓ | ✓ |
| qwen3_5_9b | Qwen/Qwen3.5-9B | 9.7 | bf16, thinking off | ✓ | ✓ |
| gemma4_e4b | google/gemma-4-E4B-it | 8.0 | bf16, thinking off | ✓ | ✓ |
| gemma4_12b | google/gemma-4-12B-it | 12.0 | bf16, thinking off | ✓ | ✓ |
| ministral3_14b | mistralai/Ministral-3-14B-Instruct-2512 | 13.9 | checkpoint-native | ✓ | ✓ |
| gpt_oss_20b | openai/gpt-oss-20b | 20.9 | MXFP4, harmony | — | ✓ |
| qwen3_8_27b | Qwen/Qwen3.8-27B | 27.8 | bf16, thinking off | ✓ | ✓ |
| qwen3_8_27b_thinking | Qwen/Qwen3.8-27B | 27.8 | bf16, thinking **on** | — | ✓ |
| qwen3_8_27b_fp8 | Qwen/Qwen3.8-27B-FP8 | 27.8 | FP8 (quantization control) | ✓ | ✓ |
| gemma4_31b | google/gemma-4-31B-it | 31.3 | bf16, thinking off | ✓ | ✓ |
| llama3_3_70b_int8 | RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8 | 70 | INT8 W8A8 (legacy anchor) | ✓* | ✓ |
| gpt_oss_120b | openai/gpt-oss-120b | 116.8 | MXFP4, harmony | — | ✓ |

- **T1 is not defined for reasoning-output models** (thinking on, harmony),
  because their answer follows generated reasoning.
- \* T1 runs wherever the HF stack loads the checkpoint as published. If it
  can't, the cell is reported as missing, never approximated.
- The optional models (`phase2_optional`: Qwen3.6-35B-A3B, Gemma-4-26B-A4B,
  Granite-4.2-8B/30B) run only if time permits. They're reported in a
  separate, labelled table, outside the Holm family and the scaling test.

## Procedure

- **T2** (headline). vLLM 0.29.0, greedy, temperature 0, and seed 20261010.
  One user message per prompt, rendered through the model's own chat
  template.
  - Token budget: `max_tokens` 256 and `max_model_len` 8192. For gpt-oss and
    Qwen3.8-27B thinking: `max_tokens` 4096 and `max_model_len` 12288.
  - Decoding is unconstrained (`constrained: false`), so refusals stay
    observable.
  - gpt-oss answers are its harmony final channel. Thinking models' leading
    `<think>` block is stripped before parsing.
  - All 172,296 rows are run per model.
- **T1.** The application prompt is chat-templated with thinking disabled.
  The model scores the two complete answer strings `{"decision": "allow"}` and
  `{"decision": "deny"}` as continuations (sum log-probability, joint
  tokenization with boundary audit: `src/authinv/eval/logprob.py`).
  - margin = logp(correct) − logp(incorrect);
  - T1-correct = margin > 0;
  - T1 covers application rows only.
- **Hardware** may differ by model (H100 80 GB, A100 40 GB, RTX 16 GB, in
  bf16 where the format is bf16; V100 is not used because it lacks bf16).
  Hardware is recorded per run and isn't analysed.

## Parsing (`json-strict-v2`, frozen)

- **Accepted answer.** Exactly one JSON object, bare or in one ```json fence,
  with one trailing period allowed. It must contain only the task's field:
  - application: `{"decision": "allow"|"deny"}`;
  - interpretation: `{"principals": [ids]}`, an exact set match with no
    duplicates.
- **Otherwise**, the answer is `refusal` or `abstain` if it matches the
  strict-v2 patterns (typographic apostrophes folded), and `parse_failure` if
  not.
- These three categories are reported on their own. They aren't correct, and
  they're never dropped or re-labelled.

## Metrics (per model; `src/authinv/eval/invariance.py`)

The unit of analysis is the world. An instance is one (assignment, request)
pair. Per-world rates average a world's instances. CIs come from a
world-clustered percentile bootstrap with 4,000 replicates (seed 20261010).
Every table is reported for **all worlds** and separately for **real** and
**synthetic** worlds (`source_kind`).

- **Accuracy per rendering**, with refusal, abstain, and parse-failure rates.
- **Worst-case gap** (primary). Best and worst renderings are taken from the
  point estimates, and the gap is the paired within-world difference with a
  bootstrap CI and two-sided p. Secondary: the gap with best and worst
  re-selected in every replicate.
- **Rendering disagreement**: the share of instances whose answers differ
  across renderings.
- **All-renderings-correct**.
- **Direction (application)**:
  - deny→allow rate: among deny-labelled rows, answered allow;
  - allow→deny rate: among allow-labelled rows, answered deny.

  Refusals count toward neither. Instance-level flip shares and ordered
  pairwise flip counts are also reported.
- **Dissociation (Q3)**: P(interpretation correct ∧ application wrong), and
  P(application wrong | interpretation correct), per rendering. Pairs are
  matched by world, assignment, rendering, and `pair_key`.
- **Rendering contrasts (the Holm family)**: for each rendering, accuracy
  minus the mean accuracy of the other five, paired within world, with a
  bootstrap CI and p.
- **T1**: the same accuracy, gap, and disagreement metrics on T1-correct,
  plus mean margins per rendering.

## Inference and multiplicity

- **Holm family:** every (model, rendering) rendering contrast on T2
  application accuracy for the 12 sweep models: 12 × 6 = 72 tests. Holm
  adjustment at α = 0.05 across the whole family. Effect sizes and CIs lead;
  p-values are secondary.
- **Scaling (Q4):** Spearman ρ between parameter count and T2 application
  worst-case gap across the 12 sweep models, with a two-sided permutation p
  (20,000 permutations). It is reported once more without the FP8 and
  thinking variants, which duplicate the 27B base. That second version is
  descriptive.
- **Quantization control:** Qwen3.8-27B bf16 vs its official FP8 checkpoint.
  For each rendering, the accuracy difference is computed. **If the largest
  absolute difference is ≥ the bf16 worst-case gap**, quantization changes
  results as much as rendering does. Quantized configurations
  (qwen3_8_27b_fp8, llama3_3_70b_int8) then leave the headline tables and are
  reported in a labelled secondary table. gpt-oss's MXFP4 is its published
  native format, not a quantized configuration.

## Exclusions

None. Every row of every completed run is analysed. A run that fails is
resumed (`--resume`, which checks config, dataset, model, and parser
provenance) and never mixed with another configuration. A model that can't
be run is reported as missing, with the reason. Prompts, renderers, parser,
and config are frozen: any change gets a version bump, a dated amendment
below, and a rerun.

## Pre-registration engineering disclosure

Before this file was committed, the T2 runner was smoke-tested end to end on
the v2 benchmark with **Qwen/Qwen3-0.6B** (`smoke_qwen3_0_6b`; not a sweep
model). It ran 300 rows on gpubox (RTX 5080), and the outputs were discarded.
The purpose was to check installation, chat templating, and parsing; all 300
answers parsed.

That run used the harness one commit before the frozen config, so it had
`max_tokens` 96 and no dataset pin. It wasn't an analysis run.

No sweep model was run on the benchmark before this commit.

## Amendments

(none)
