# Phase 4 preregistration: format-diverse LoRA SFT, Cedar held out

- status: **draft for review**. This file is the preregistration only once Jerzy has settled the
  choices below and it is merged to `main`. That must happen before any Phase-4 training run on a
  real model.
- step: P4.3 (training and evaluation in P4.4) · spec: `docs/RESEARCH_PLAN.md` §3 Phase 4, §4
- frozen inputs:
  - eval benchmark v2 (`dataset.jsonl` sha256 `0b602a17…`);
  - SFT data `sft-data-v1`, `outputs/20261107T214145755710Z_authinv_sft_data`, manifest sha256
    `14f063c1…` (`docs/experiments/p4.1-sft-data.md`);
  - trainer `sft-trainer-v1` (`scripts/authinv_sft_train.py`, `configs/authinv/sft_lora.yaml`).
- What was known when this was written: every Phase-2 result, including both SFT models' T2
  numbers (below). No adapter has been trained on a real model. The trainer has only run CPU smoke
  tests, on a tiny random model.

## Choices for review (Jerzy), to settle before merge

Each choice lists the draft's recommendation first.

1. **Primary held-out measure.** Recommended: application accuracy on Cedar (`executable`), SFT −
   no-SFT and SFT − control. That is "generalizes to an unseen policy language", as the plan reads.
   Alternative: the worst-case gap over all six renderings.
2. **Rego.** It is unseen in training as well. Recommended: report it as a secondary held-out
   format, with Cedar as the primary, as the plan says. Alternative: treat both executable formats
   as one held-out family.
3. **Control rendering.** nl_statement, as built in P4.1. Alternative: json_policy, which needs a
   P4.1 rebuild.
4. **The BFCL regression check.** Recommended: the BFCL v3 `simple` and `multiple` categories, AST
   evaluation, no execution, all their items, with the BFCL prompt and our pinned models. A drop of
   more than 2 pp with a CI that excludes 0 is reported as a regression. There is no harness yet;
   it needs a step (proposed P4.2b, below).
5. **Hyperparameters.** Recommended: those of `configs/authinv/sft_lora.yaml` as they stand (r 32,
   1 epoch, lr 1e-4), with no search and no early stopping. The validation loss is reported, never
   used for selection. Alternative: a small r ∈ {16, 64} sensitivity on one seed.
6. **Tasks in training.** Recommended: application and interpretation, as built. Alternative:
   application only, which needs a P4.1 rebuild.

**Needed before any training (proposed new step P4.2b, a plan change):**
- adapter evaluation in the T2 harness: vLLM `enable_lora` with a `LoRARequest`, the adapter
  sha256 recorded in the run, and otherwise the frozen T2 config and parser;
- the BFCL-subset harness of choice 4;
- a GPU smoke test of the trainer and of LoRA serving on both real models, H100 GPU 7 with
  `--max-steps`, outputs discarded.

## What is known from Phase 2 (the no-SFT arm)

| model | nl | owner | table | json | Cedar | Rego | worst-case gap (best − worst) |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 77.2 | 77.7 | 75.8 | 75.1 | **78.2** | 72.4 | 5.8 (Cedar − Rego) |
| Gemma-4-12B | 85.1 | 89.4 | 85.9 | 90.7 | 89.5 | 91.2 | 6.1 (Rego − NL) |

(T2 application accuracy in %, `json-strict-v3`, from the Phase-2 runs in
`configs/authinv/phase2_figures.yaml`.)

Cedar is already Qwen3.5-9B's best rendering, and is mid-ranked for Gemma-4-12B. Neither model's
worst rendering is Cedar. For Qwen, the worst is Rego, which training also never shows. So the
held-out question (does Cedar accuracy rise?) and the gap question (does the worst case rise?) can
come apart, which is why they are reported separately.

## Questions

- **Q13 (primary, held-out transfer).** Does multi-format SFT raise application accuracy on Cedar,
  a format absent from training, over the no-SFT model and over the matched-token single-format
  control?
- **Q14 (held-in).** Does it shrink the worst-case gap over the four training renderings?
- **Q15 (all six).** Does it shrink the worst-case gap over all six renderings, and raise
  worst-case accuracy?
- **Q16.** Rego, the second unseen format, as in Q13.
- **Q17 (regression).** Does SFT lower general function-calling accuracy on the BFCL subset?

## Arms and runs (P4.4)

- **Models:** Qwen3.5-9B and Gemma-4-12B-it at the pinned revisions (amended 2026-10-08).
- **Arms:**
  - no-SFT: the base model, i.e. the Phase-2 T2 run of each model; no new run;
  - multi: `train_multi`, four renderings, 15.6M / 15.9M target-model tokens;
  - control: `control_<model>`, nl_statement only, token-matched.
- **Seeds:** 20261111, 20261112 and 20261113 for both trained arms. That makes 2 models × 2 arms
  × 3 seeds = 12 adapters.
- **Evaluation:** the frozen benchmark through the T2 harness (`configs/authinv/t2.yaml`: prompt-v1,
  greedy, 256 tokens, `json-strict-v3`), all six renderings, both tasks. The adapter is served by
  vLLM LoRA (P4.2b), with the base model at its pinned revision. BFCL subset per adapter and per
  base model.

## Metrics

The world is the unit, with a world-clustered percentile bootstrap: 4,000 replicates, seed
20261010, the Phase-2 draws.

- **Per adapter:** the full T2 `rendering_metrics` (accuracy, direction, non-answers, worst-case gap
  and its reselected variant) and the T2 contrasts.
- **Per arm (three seeds):** a world's value is its mean over the three seeds' rows, so the
  bootstrap carries across worlds and seeds come in as replicated measurement. The per-seed
  estimates and their range are reported too.
- **Contrasts (paired within world):**
  - Cedar accuracy: multi − no-SFT, multi − control, control − no-SFT;
  - the same for Rego (Q16);
  - held-in worst-case gap over the four training renderings (Q14);
  - six-rendering worst-case gap and worst-case accuracy (Q15).
- **BFCL:** accuracy per arm, and its difference from the base model.

## Decision rule

Per model:
- **H1:** multi − no-SFT on Cedar accuracy > 0.
- **H2:** multi − control on Cedar accuracy > 0.
- **H3:** the multi arm's six-rendering worst-case gap < no-SFT's.

Each is tested two-sided by paired bootstrap. The 2 models × 3 hypotheses = 6 tests form one Holm
family, α = 0.05. Q14, Q16 and Q17 are reported with CIs, outside the family.

Result classification:
- **positive** if H1 and H2 hold on both models;
- **neutral** if they hold on one;
- **negative** otherwise.

H3 and the BFCL change are reported alongside, whatever the classification. No gate follows Phase
4. The P4.4 report ends with a recommendation for the paper.

## Exclusions

None. Every adapter and every row is analysed. A failed training run is rerun from scratch with
the same seed (the trainer is seed-deterministic). It is never resumed from a partial state, and
the failure is disclosed. A configuration that can't be run is reported as missing, with the
reason.

## Pre-registration engineering disclosure

- P4.1 data build:
  - the vocabulary check caught five words shared with CedarBench in the first domain draft;
  - a transformers-5 token-count trap was fixed before the frozen build.
- P4.2 trainer: CPU smoke tests on a tiny random Llama only. Sequence construction was checked on
  both real chat templates; no weights were loaded.
- GPU smoke test: pending, part of P4.2b.

## Amendments

None yet.
