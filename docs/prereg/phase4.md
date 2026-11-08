# Phase 4 preregistration: format-diverse LoRA SFT, Cedar held out

- status: **preregistration**, settled 2026-11-08 and merged to `main` before any Phase-4 training run
  on a real model. Its commit and timestamp are in `docs/prereg/MANIFEST.md`.
- step: P4.3 (training and evaluation in P4.4) · spec: `docs/RESEARCH_PLAN.md` §3 Phase 4, §4
- frozen inputs:
  - eval benchmark v2 (`dataset.jsonl` sha256 `0b602a17…`);
  - SFT data `sft-data-v1`, `outputs/20261107T214145755710Z_authinv_sft_data`, manifest sha256
    `14f063c1…` (`docs/experiments/p4.1-sft-data.md`);
  - trainer `sft-trainer-v1` (`scripts/authinv_sft_train.py`, `configs/authinv/sft_lora.yaml`).
- What was known when this was written: every Phase-2 result, including both SFT models' T2
  numbers (below). No adapter has been trained on a real model. The trainer has only run CPU smoke
  tests, on a tiny random model.

## Decisions (2026-11-08)

Jerzy delegated these six choices to the draft's recommendations ("On decisions: proceed with your
best judgement. Go ahead with your suggestions.", 2026-11-08, in the session). The alternatives the
draft listed are not run.

1. **Primary held-out measure:** application accuracy on Cedar (`executable`), as the contrasts
   multi − no-SFT and multi − control. The six-rendering worst-case gap is H3, reported alongside.
2. **Rego:** a secondary held-out format (Q16), reported with CIs outside the Holm family. Cedar is
   the primary.
3. **Control rendering:** nl_statement, as built in P4.1 (`sft-data-v1`, no rebuild).
4. **BFCL regression check:** the BFCL v3 `simple` and `multiple` categories, AST evaluation, no
   execution, every item, with the BFCL prompt and our pinned models. A drop of more than 2 pp whose
   95% CI excludes 0 is reported as a regression. The harness is the new step P4.2b; its version,
   data revision and item count are named here by a dated amendment before any BFCL run.
5. **Hyperparameters:** `configs/authinv/sft_lora.yaml` as it stands (r 32, alpha 64, 1 epoch,
   lr 1e-4), no search and no early stopping. The validation loss is reported, never used for
   selection.
6. **Tasks in training:** application and interpretation, as built.

**Before any training (step P4.2b, adopted):**
- adapter evaluation in the T2 harness: vLLM `enable_lora` with a `LoRARequest`, the adapter
  sha256 recorded in the run, and otherwise the frozen T2 config and parser;
- the BFCL-subset harness of decision 4;
- a GPU smoke test of the trainer and of LoRA serving on both real models, H100 GPU 7 with
  `--max-steps`, outputs discarded and disclosed below.

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
