# Phase 5 preregistration: probes and CKA across renderings (optional phase)

- status: **preregistration**, settled 2026-11-08 and merged to `main` before any extraction run on
  a real model. Phase 5 is optional (RESEARCH_PLAN §3); it runs last on the H100, after the Phase-3
  and Phase-4 runs. Its commit and timestamp are in `docs/prereg/MANIFEST.md`.
- step: P5.2 (runs and report in P5.3; the causal check in P5.4) · spec: `docs/RESEARCH_PLAN.md` §3
  Phase 5, §4
- frozen inputs: benchmark v2 (`0b602a17…`); tooling from P5.1 (`scripts/authinv_probe.py`,
  `src/authinv/probe/`, `configs/authinv/probe.yaml`).
- What was known when this was written: every Phase-2 result. No activation has been extracted
  from a real model. The tooling has run on synthetic activations and a tiny random model only.

## Decisions (2026-11-08)

Jerzy delegated these choices to the draft's recommendations ("On decisions: proceed with your
best judgement. Go ahead with your suggestions.", 2026-11-08, in the session).

1. **Phase 5 runs**, at the lowest priority on the H100: after the Phase-3 arms and the Phase-4
   training and evaluation. It costs about one GPU-day for two models. The probing claim stays a
   single figure unless P5.4 finds a causal effect.
2. **Models:** Qwen3.5-9B and Gemma-4-12B-it, the Phase-4 SFT models, at their pinned revisions.
   Both have Phase-2 T1 runs, mid-sized gaps (T2 5.8 / 6.1 pp; T1 5.7 / 5.8 pp) and high rendering
   disagreement (28.6% / 22.6%).
3. **Items:** the 400 hash-selected worlds of `configs/authinv/probe.yaml`, with all six
   renderings, both assignments and both tasks: 34,884 items per model.
4. **Layers:** every other hidden-state index, plus the last.
5. **Thresholds:** as in Decision rules below.

## Questions

- **Q18.** Are allow/deny, the governing clause and decision-owner identity linearly decodable from
  the residual stream at the last prompt token, above the metadata-only baseline?
- **Q19.** Does a probe trained on one rendering transfer to the others? Is transfer worst into the
  model's worst T2 rendering?
- **Q20.** Do the renderings' representations converge with depth (linear CKA between renderings
  per layer)?
- **Q21 (P5.3).** Does an instance's cross-rendering divergence (cosine distance at the
  best-transfer layer) predict that the model's T2 answers on it disagree across renderings?

These are behavioural-alignment questions only. Decodability is not use; P5.4 is the causal
check, and without a causal effect probing stays one figure.

## Procedure

- **Extraction:** `scripts/authinv_probe.py extract`, at the pinned revisions, bf16. The T1 prompt
  is chat-templated with thinking off, and the vector is the last prompt token's hidden state at
  the selected layers. Cache integrity is verified by sha256 before any analysis.
- **Probes:** L2 multinomial logistic regression (l2 = 1.0, L-BFGS, 500 iterations) on features
  standardized on the training split.
  - The split is by world: 30% of worlds are held out (salted hash, seed 20261115).
  - Each probe trains on one rendering's training worlds and is tested on every rendering's
    test worlds.
- **Labels:**
  - `allow_deny` and `governing` (permit / forbid / condition / scope) on application items;
  - `owner_0`–`owner_4` on interpretation items, read slot by slot.
- **Metric:** balanced accuracy, with a percentile bootstrap over test worlds (2,000 replicates).
- **Baselines per cell:** metadata-only probe, shuffled-label probe (last layer), majority class.
- **CKA:** linear CKA per layer for every pair of renderings, on the same instances.

## Decision rules

Per model and label:
- **Decodable (Q18)** at a layer: the in-rendering balanced accuracy's CI lower bound exceeds the
  metadata-only probe's point estimate for that rendering by at least 0.05. This must hold in all
  six renderings.
- **Transfer gap (Q19):** in-rendering minus cross-rendering balanced accuracy, averaged over
  pairs, at the layer with the best mean in-rendering accuracy, with its CI (bootstrap over test
  worlds).
- **Q20** is descriptive: CKA by layer, and the layer of maximum mean CKA.
- **Q21:** AUC of instance divergence for T2 cross-rendering disagreement (the Phase-2 T2 runs),
  with a world-clustered bootstrap CI. "Predictive" means the CI lower bound is above 0.5.

Multiplicity: the Q21 AUCs (2 models × 15 rendering pairs) form one Holm family. Q18 and Q19 are
reported with CIs as effect sizes.

## Exclusions

None. A failed extraction is rerun from scratch, since a run is deterministic given the pinned
model and inputs. A cache that fails its hash check is never analysed.

## Pre-registration engineering disclosure

- P5.1 tooling was tested on synthetic activations (a planted shared vs rendering-specific signal)
  and on a tiny random Llama. On benchmark v2, CPU only, the 400-world subset gives 34,884 items;
  `forbid` is the rare governing class (70 per rendering).
- A GPU smoke test of extraction on both models (H100 GPU 7, `--limit`, outputs discarded) runs
  before the first P5.3 extraction and is disclosed here by a dated amendment. It checks the
  engine path, not probe accuracy.

## Amendments

None yet.
