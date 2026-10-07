# Research Plan: Representation Invariance of Authorization Policies for LLM Agents

**Repository:** `down-shift/authority`
**Target venue:** NeurIPS 2027 main track (deadline ~May 2027); fallback NeurIPS Datasets & Benchmarks track, then ACL ARR.
**Compute:** open-weight models only, on local H100/V100 cluster. No proprietary API models. Budget target: $0 in API spend.
**Audience of this file:** Claude Code. Read it fully before touching the repo. Sections marked `[DECISION]` are settled; sections marked `[ASK JERZY]` require a human answer before proceeding.

---

## 0. One-paragraph thesis

Authorization policies handed to LLM agents (tool allowlists, resource permissions, decision ownership) are not representation-invariant: the same policy rendered as prose, a decision-owner statement, a permission table, a JSON policy, or executable rule code (Cedar/Rego) produces different model decisions and different agent actions, even when an authorization engine certifies the renderings equivalent. We measure this as **worst-case accuracy over equivalent renderings** and, in an agentic harness, as **worst-case unauthorized-action rate**, with special attention to **deny→allow flips**. We then test whether canonicalizing any rendering into a single intermediate representation (IR) before deciding, or format-diverse fine-tuning, closes the worst-case gap.

## 1. What already exists and what we keep

Current repo state (see README and `docs/legacy_experiments.md`):

- A Gemma 3 12B NF4 and Qwen3-8B int8 single-scope pilot: interpretation at ceiling for all five renderings; application accuracy format-sensitive (NL ≈ 91% symmetrized, others ≈ 100%), 12.8% name-swap flips on NL, residual position gap (100% vs 94%).
- Toy worlds: filename / ordering / destination owner fields, two semantic candidates, sum-log-prob margin scoring.
- Methodological assets worth keeping **verbatim in spirit**: paired worlds, lexical-symmetry (name-swap) audits, position balancing, tokenizer/token-boundary audits, preregistered competence gates, world-clustered bootstrap, frozen datasets with SHA-256 manifests, `run_status.json`, `--dataset-only` and `--resume` discipline, prompts fixed in renderers and never tuned on outcomes.
- Legacy indirection-depth line: effect reversed under controls. **Out of scope** for this paper; do not touch.

`[DECISION]` New work lives in a new package `src/authinv/` with its own configs under `configs/authinv/`. Do not import from `authorization_invariance` (legacy) or `authorization_competence`. Reuse code by copying and adapting, not by importing, so the paper's code is self-contained.

## 2. Prior work we must position against (do not re-derive the phenomenon)

Claude Code: fetch and skim the abstracts of these before designing anything; cite them in `docs/related_work.md`.

| Role | Paper | ID |
|---|---|---|
| Generic format sensitivity (established) | Sclar et al., spurious prompt features | ICLR 2024 |
| | He et al., prompt formatting impact | arXiv 2411.10541 |
| | Not as Sweet by Another Name (document format robustness, metamorphic invariance relations) | arXiv 2607.27648 |
| | Multi-Format Training for cross-format robustness | arXiv 2606.11643 |
| Closest policy-format precedent (2 formats, DLP, withdrawn) | PolicyGuard DLP | arXiv 2608.02687 |
| LLMs misapply real IAM policies, formal ground truth | PolicySummarizer / Neurosymbolic characterization | arXiv 2510.20692 |
| Unauthorized tool invocation rate (UIR) metric | Prompts Don't Protect (MCP proxy) | arXiv 2605.18414 |
| Representation sensitivity of *attacks*, policy fixed | Threat-Preserving Representation Sensitivity | arXiv 2610.03585 |
| Canonicalize-then-decide baselines | Policy-as-Logic (ASP + Clingo) | arXiv 2608.11905 |
| | Ghost in the Context (Policy IR) | arXiv 2605.12535 |
| | MetaPermit | arXiv 2609.31039 |
| | Structured Decomposition (NL→JSON IR→Rego) | arXiv 2609.24036 |
| Policy sources | CedarBench / AutoCedar | arXiv 2607.03656 |
| | Quacky (AWS IAM, 587 policies) | via 2510.20692 |
| | ACRE dataset | via Prose2Policy arXiv 2603.15799 |
| Probing precedents | How Language Models Choose Sides | arXiv 2608.28648 |
| | System Prompt Illusion | arXiv 2609.38205 |
| Multi-constraint counterpoint | Phase transitions in constraint satisfaction ("not pairwise interference") | arXiv 2608.12426 |
| Agent benchmarks | AgentDojo, Agent Security Bench, GrantBox (2603.28166), AuthBench (2605.14859) | |

**Novelty we claim:** (1) five engine-certified-equivalent renderings of *authorization* policies; (2) worst-case-over-renderings as a security metric, with deny→allow flips; (3) agentic unauthorized-action rate with policy rendering as the independent variable; (4) interpretation/application dissociation that *varies by rendering*; (5) policy-side canonicalization evaluated by closure of the worst-case gap; (6) per-rendering probes of owner/permission representations. Cross-scope interference is demoted to an appendix experiment.

## 3. Phases, gates, and acceptance criteria

Each phase ends with a committed report under `outputs/` and an entry in `docs/PROGRESS.md`. Do not start a phase until the previous gate is recorded as passed or explicitly waived by Jerzy.

### Phase 0 — Kill test on open models (Weeks 1–2)

Goal: confirm the effect survives (a) generation instead of log-probs and (b) model scale.

Tasks:
1. Write `scripts/authinv_phase0.py` that takes the existing Gemma Stage-2 `worlds.jsonl` (180 worlds, both name assignments, five renderings) and evaluates with **greedy generation** plus a strict parser, not candidate log-probs.
2. Models: Qwen3-32B (bf16), Llama-3.3-70B (int8), Gemma-3-27B (bf16), Qwen3-8B (bf16, as a bridge to the pilot). Serve via vLLM.
3. Metrics per model: mean accuracy per rendering; worst-case accuracy = min over renderings; per-world rendering disagreement rate; deny→allow vs allow→deny flip counts; world-clustered bootstrap CIs.

Gate 0 (preregister in `docs/prereg/phase0.md` before running):
- PASS if, on at least two of the three ≥27B models, worst-case accuracy is ≥5 pp below the best rendering with a 95% CI excluding zero, **or** per-world rendering disagreement ≥5%.
- FAIL → pivot: paper becomes "format-diverse SFT yields representation-invariant small open agents" (Phase 4 becomes primary, Phases 2–3 shrink). Record the decision; do not quietly continue.

### Phase 1 — Benchmark with engine-certified equivalence (Weeks 3–8)

Goal: replace toy worlds with real policies whose renderings are provably equivalent.

`[DECISION]` Primary engine: **Cedar** (use the `cedar-policy` CLI or Python bindings; the evaluator is formally verified). Secondary: **OPA/Rego** (`opa eval`). AWS IAM only if Quacky's SMT equivalence check can be run locally without pain; otherwise IAM policies are translated into Cedar and the translation is validated by request-level differential testing.

Tasks:
1. `src/authinv/policy/` — canonical policy object (principal, action, resource, effect, conditions, owner). This is the single source of truth.
2. `src/authinv/render/` — five renderers, each a pure function of the canonical object:
   - `nl_statement` (natural-language permission statement)
   - `owner_statement` (decision-owner statement: "X decides Y")
   - `table` (Markdown permission table)
   - `json_policy`
   - `executable` (Cedar; Rego as a sixth rendering if time permits)
   Prompts are fixed templates. **Never edit a renderer after seeing model results**; if a template must change, bump a version string and rerun everything.
3. `src/authinv/equivalence/` — for every world, generate a request set (balanced allow/deny, including boundary requests) and verify every rendering decodes to the canonical object, and that the executable rendering produces the same allow/deny on every request via the real engine. A world ships only if all five renderings pass. Log equivalence proofs to `equivalence.jsonl` with hashes.
4. Policy sources:
   - Real: CedarBench (221), Quacky AWS IAM (587 → Cedar), ACRE statements (NL → canonical, then engine-checked).
   - Synthetic: MCP tool allowlists and GitHub-style repo permissions generated from a grammar, balanced over effect, role count, and condition count.
   - Difficulty tiers: single rule; multi-rule with deny-overrides; conditioned (attribute-based).
5. Query types per world: **interpretation** ("who/what governs this decision?") and **application** ("is request R allowed?" / "which proposed value is valid?"). Keep both; the dissociation is a planned result.
6. Balance and audits: actor name-swap symmetrization, option position swap, request order, rendering order in any multi-rendering prompt, tokenizer length audits (renderings should differ in length; record it and treat length as a covariate, not something to equalize away).
7. Target size: ≥1,000 worlds across tiers; ≥5 renderings; ≥4 requests per world → ≥20k application rows per model.

Gate 1: dataset-only run passes all equivalence checks; a 20-world human spot-check (Jerzy) finds no rendering that reads as ambiguous; `docs/datasheet.md` written.

### Phase 2 — Three evaluation tiers on the model sweep (Weeks 7–12, overlaps Phase 1)

`[DECISION]` Model sweep (all open weights, local):

| Family | Sizes | Precision |
|---|---|---|
| Qwen3 | 4B, 8B, 14B, 32B; 235B-A22B if it fits | bf16 (≤32B); int8/FP8 for 235B |
| Llama 3.x | 3.1-8B, 3.3-70B | bf16 (8B); **70B in both bf16 and int8** as the quantization control |
| Gemma 3 | 12B, 27B | bf16 |
| Mistral | Small 3.x (24B) | bf16 |
| Reasoning variant | Qwen3-32B thinking mode on, vs off | bf16 |

Report quantized-vs-bf16 deltas for Llama-70B explicitly; if they exceed the rendering effect, drop quantized configs from headline tables.

Tiers:
- **T1 Log-prob margins** (port the existing scorer): cheap, mechanistically clean, used for probing alignment.
- **T2 Structured generation**: greedy decoding, strict JSON answer schema, parse-failure rate reported separately. Headline metrics come from T2.
- **T3 Agentic execution**: `src/authinv/agent/` — a minimal MCP tool sandbox (file ops, repo ops, HTTP stubs). Policy goes in the system prompt in one rendering; tasks require tool calls; some tools/resources are forbidden. Metrics: unauthorized-action rate (UIR, as in arXiv 2605.18414), authorized-action refusal rate, task completion. Policy rendering is the only varied factor. Use the same policies as T2 so T2→T3 transfer can be reported.

Headline tables (per model, per tier): mean accuracy by rendering; worst-case accuracy; rendering disagreement; deny→allow flip rate; dissociation rate (interpretation correct ∧ application wrong, by rendering); scaling trend of the worst-case gap vs parameter count.

Gate 2: all sweep runs complete with frozen datasets, provenance, and bootstrap CIs; figures auto-generated by `scripts/authinv_figures.py`.

### Phase 3 — Mitigations (Weeks 12–15)

Arms, all evaluated by **closure of the worst-case gap** and by T3 UIR:
1. **Policy canonicalization (ours):** model maps any rendering → canonical JSON IR (constrained decoding over the IR schema, or candidate-likelihood selection as in the current E3 design); then decides from the IR. Report conversion accuracy and answer accuracy separately, and the "garbage-in" rate where a wrong IR still yields a right answer.
2. **Baselines:** zero-shot CoT; "restate the policy as JSON, then answer" (single prompt); self-consistency across the five renderings (majority vote — note this is an analysis tool, not deployable in one prompt, say so); Policy-as-Logic-style solver pipeline (LLM extracts facts, Clingo/engine decides) as the strong neurosymbolic baseline; external enforcement proxy (upper bound, by construction 0% UIR).
3. Cost accounting: tokens and latency per arm.

Gate 3: canonicalization closes ≥50% of the worst-case gap on the majority of models, or we report honestly that it does not and the solver pipeline is the recommendation.

### Phase 4 — Format-diverse LoRA SFT (Weeks 13–16, parallel with Phase 3)

Following arXiv 2606.11643's recipe, applied to authorization:
1. Training data: synthetic policies (not from the eval set) in **four** renderings; hold out one rendering entirely (Cedar) to test generalization to an unseen format.
2. Models: Qwen3-8B and Gemma-3-12B, LoRA r=16–64, 3 seeds.
3. Controls: single-format SFT with matched token count; no-SFT.
4. Report: worst-case gap on held-in and held-out renderings; any safety/utility regression on a general tool-use eval (e.g., a BFCL subset).

### Phase 5 — Probing (Weeks 14–17, optional, raises tier)

1. Linear probes on residual stream per layer for: owner identity, allow/deny, and "which field governs" — trained on T1 prompts of one rendering, tested on the others (cross-rendering transfer).
2. Representational similarity (CKA) between renderings per layer: where do renderings converge, and does divergence predict behavioral disagreement?
3. Causal check: steering or activation patching from the best-performing rendering into the NL rendering; does application accuracy recover? Without a causal result, keep probing to one figure.
4. Baselines as in arXiv 2608.28648: metadata-only probes and shuffled-label controls.

### Appendix experiment — Cross-scope interference (only if time)

Reuse E2 design but hold constraint count fixed and vary semantic independence. Address arXiv 2608.12426's "not pairwise" finding directly. If the effect is small, it becomes two paragraphs in the appendix.

## 4. Statistics and preregistration rules

- Unit of analysis: world. All CIs are world-clustered bootstrap (≥2,000 resamples). Never pool across renderings without reporting per-rendering values.
- Every phase has a `docs/prereg/phaseN.md` committed **before** inference, stating hypotheses, metrics, gate thresholds, and exclusion rules.
- Multiple comparisons across models × renderings: report Holm-corrected p-values alongside CIs; the paper leads with effect sizes.
- Report parse failures, refusals, and "abstain" responses as their own category; never silently count them as wrong or drop them.
- Any prompt or renderer change after seeing results → new version, full rerun, noted in `docs/CHANGELOG.md`.

## 5. Engineering conventions

- `uv` for everything; extras: `inference`, `quantization`, `engines` (Cedar/OPA), `probing`, `agent`.
- Every runner supports `--dataset-only`, `--tokenizer-audit-only`, `--resume RUN_DIR`, writes `config.json`, dataset hashes, model revision, `metadata.json`, `run_status.json`, and a SHA-256 manifest. Copy the pattern from the existing runners.
- vLLM for T2/T3 generation; HF transformers for T1 log-probs and probing. Pin model revisions in `configs/authinv/models.yaml`.
- Tests: unit tests for renderers (round-trip canonical → rendering → decode), equivalence checker (known-equivalent and known-different pairs), parsers, and metrics. CI must pass before any inference run.
- Determinism: fixed seeds, temperature 0, record vLLM version and GPU type.
- Never write results into README; README only documents how to run. Results live in `outputs/*/report.md` and `docs/PROGRESS.md`.

## 6. Deliverables

1. `authinv` benchmark release (policies, renderings, equivalence proofs, request sets, datasheet, license) on HF Datasets + GitHub.
2. Leaderboard script producing the worst-case table for any HF model.
3. T3 agent sandbox as a reusable harness.
4. Paper draft in `paper/` (LaTeX, NeurIPS style): 9 pages + appendix. Figures auto-generated from `outputs/`.
5. `docs/related_work.md` with the table in §2 expanded and kept current (re-search arXiv cs.CR/cs.CL monthly; most close neighbours appeared May–Oct 2026).

## 7. Timeline (16–17 weeks from start)

| Weeks | Work |
|---|---|
| 1–2 | Phase 0 kill test, prereg, related-work doc |
| 3–8 | Phase 1 benchmark + engines; Phase 2 T1/T2 harness in parallel |
| 7–12 | Phase 2 sweep (T1, T2, then T3) |
| 12–15 | Phase 3 mitigations |
| 13–16 | Phase 4 SFT (parallel) |
| 14–17 | Phase 5 probing (optional) |
| 15–18 | Paper writing, figure polish, benchmark release |

## 8. `[ASK JERZY]` Open questions before Phase 1

1. Is a Rego rendering worth the sixth renderer, or is Cedar alone enough for "executable rule"?
2. Which GPUs are available for the 70B bf16 runs, and for how long? (Determines whether 235B-A22B is in or out.)
3. Should the T3 agent sandbox be built fresh or adapted from AgentDojo's harness? Adapting saves time but ties us to its task style.
4. Co-authors / student allocation: who owns the benchmark, who owns the agent harness?
5. Do we want to invite the PolicySummarizer (Eiers, Stevens) group as collaborators for the IAM/Quacky part, or keep it internal?

## 9. What success looks like

A reviewer reads: "Across N open-weight models from 4B to 70B+, an engine-certified-equivalent policy loses X pp of accuracy in its worst rendering, and in Y% of worlds a denial becomes an allow; in an agent sandbox this raises unauthorized actions by Z pp; canonicalizing the policy to an IR before deciding closes W% of the gap, and format-diverse SFT generalizes to an unseen policy language." If X, Y, Z are small on the largest models, the paper is honest about it and the SFT/IR result becomes the lead.

---

## Repo mapping (added when the plan was entered into the loop, 2026-10-07)

How this plan maps onto the `/continue` loop; where the two differ, this section says which wins.

- **Steps:** every task and gate above is a step in `docs/PLAN.md` (ids `P<phase>.<n>`, gates `G<phase>`, human questions `H0.1`). Gates and §8 are `[HUMAN]` steps the loop never claims.
- **Reports:** `outputs/` is git-ignored, so the committed copy of each phase report lives in `docs/experiments/<step-id>-<slug>.md`; `outputs/*/report.md` stays as the run-local copy. `docs/PROGRESS.md` is the phase/gate/daily log.
- **Preregistration:** `docs/prereg/phaseN.md` is committed and pushed before inference; its commit goes into `docs/prereg/MANIFEST.md`.
- **Paper format:** `paper/` currently uses the ACL template (set up at the human's request before this plan arrived). Switching to NeurIPS style is a separate decision for Jerzy.
