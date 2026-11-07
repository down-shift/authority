# Phase 3 preregistration: mitigation arms on the model sweep

- status: **draft for review**. This file is the preregistration only once Jerzy has settled the
  choices below and it is merged to `main`. That must happen before any Phase-3 inference on a sweep
  model. Until then no arm runs on a sweep model, and no derived arm is computed on real data.
- step: P3.3 (runs and report in P3.4) · spec: `docs/RESEARCH_PLAN.md` §3 Phase 3 and Gate 3, §4
- frozen input: Phase-1 benchmark v2, `dataset.jsonl` sha256
  `0b602a1701bc52aaaf676973f24ea87b1e27acd9029f5e0523174d51894a4f83`, the same as Phase 2. Every
  runner refuses any other benchmark.
- baselines: the Phase-2 runs analysed in P2.9 (`configs/authinv/phase2_figures.yaml`): the
  `json-strict-v3` T2 runs and the agent-v4 (gpt-oss) / agent-v3 T3 runs, one per configuration.
- harness and configs:

  | arm | code | config | version |
  |---|---|---|---|
  | 1 canonicalization | `scripts/authinv_canonicalize.py`, `src/authinv/mitigate/{canonicalize,prompts,metrics,runner}.py` | `configs/authinv/p3_canonicalize.yaml` | canon-v1 |
  | CoT, restate | `scripts/authinv_baselines.py run`, `src/authinv/mitigate/baselines.py` | `configs/authinv/p3_baselines.yaml` | baseline-v1, json-final-v1 |
  | self-consistency, solver | `scripts/authinv_baselines.py derive` | `configs/authinv/p3_baselines.yaml` | baseline-v1 |
  | enforcement proxy (T3) | `scripts/authinv_agent_eval.py`, `src/authinv/agent/sandbox.py` | `configs/authinv/p3_proxy_t3.yaml` | agent-v4 + proxy-v1 |
- What was known when this was written: every Phase-2 result (`docs/experiments/p2.9-phase2-report.md`).
  No Phase-3 arm has run on a sweep model. The only Phase-3 model calls were engineering smoke tests on
  `smoke_qwen3_0_6b`, a non-sweep model (see the disclosure below).

## Choices for review (Jerzy), to settle before merge

Each choice lists the draft's recommendation first. Whatever is chosen replaces this section with
"Decisions (Jerzy, date)".

1. **What "closes ≥50% of the worst-case gap" means.** Three readings, all computed (`src/authinv/mitigate/metrics.py`):
   - *own* (recommended as primary): 1 − (arm best − arm worst) / (baseline best − baseline worst).
     Each side picks its own best and worst rendering by point estimate, so the gap shrinks no matter
     how it shrinks.
   - *pair*: the same ratio on the baseline's best/worst pair.
   - *floor*: (arm worst − baseline worst) / (baseline best − baseline worst). This one rises only when
     the worst case improves.

   The recommendation adds a guard: a model counts only if the arm's worst-case accuracy is not
   lower than the baseline's (point estimate). That way a gap closed by pulling every rendering down
   doesn't count. The alternative is the gate verbatim, with no guard.
2. **Which models count for "the majority".** Recommended: all 12 Phase-2 configurations, so a
   majority is ≥ 7 of 12. Alternative: the 10 distinct models (dropping Qwen3.8-27B-FP8 and
   Qwen3.8-27B thinking), so ≥ 6 of 10, with the 12 reported as sensitivity. All 12 baseline gaps
   are non-zero (Holm-significant, P2.9), so the ratio is defined for each.
3. **Decoding for the reasoning models' conversions** (gpt-oss-20b, gpt-oss-120b, Qwen3.8-27B
   thinking). Recommended: free decoding with strict parsing, because a grammar from the first token
   would block the reasoning channel. The other nine use the vocabulary-only JSON schema.
   Alternative: constrained decoding for all.
4. **T3 scope.** The plan evaluates every arm by T3 UIR too. Two T3 arms are defined below. Only
   the proxy has code today.
   - Recommended: the proxy on all 12 configurations.
   - Also recommended, needing a new step (proposed P3.2b): canonicalization in T3 (the agent's
     system prompt carries the arm-1 IR in place of the rendering) and a solver-enforced agent (the
     proxy checks calls against the model's own arm-1 IR instead of the certified policy).
   - CoT and restate don't fit the one-JSON-object-per-turn protocol, and self-consistency has no
     T3 analogue. None of the three runs in T3.
5. **The `respond` collision** (10 CedarBench worlds have an action named `respond`, which the
   sandbox's reply tool shadows; ≤ 0.1 pp in P2.8). Recommended: keep agent-v4 for Phase-3 T3, so
   arms pair with the Phase-2 T3 runs episode by episode, and disclose it. Alternative: agent-v5,
   which renames the reply tool in those worlds only, plus a rerun of their raw episodes.
6. **Budget.** The heaviest arm is restate, about 95k long generations per model. It's estimated in
   "Runs" below. Recommended: all arms on all 12 configurations. Alternative: restate (and CoT) on
   the `orig` assignment only, which halves their cost.

## What the IR choice implies, known from Phase 2

A conversion that reproduces the rules and their ids gives a stage-2 prompt byte-identical to the
T2 json_policy prompt. So with perfect conversion, the arm's accuracy on every rendering is the
model's Phase-2 json_policy accuracy. The *own* closure is then 1 for every model, and the
*floor* closure is (json_policy − worst) / (best − worst):

| configuration | T2 best (rendering) | T2 worst (rendering) | json_policy | gap | floor-closure ceiling | best-rendering loss |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 71.3 (owner_statement) | 65.8 (rego) | 67.1 | 5.4 | 0.24 | 4.1 |
| Gemma-4-E4B | 79.5 (executable) | 75.2 (table) | 75.3 | 4.3 | 0.02 | 4.2 |
| Qwen3.5-9B | 78.2 (executable) | 72.4 (rego) | 75.1 | 5.8 | 0.46 | 3.1 |
| Gemma-4-12B | 91.2 (rego) | 85.1 (nl_statement) | 90.7 | 6.1 | 0.92 | 0.5 |
| Ministral-3-14B | 74.1 (owner_statement) | 66.8 (rego) | 69.8 | 7.3 | 0.40 | 4.3 |
| gpt-oss-20b | 99.9 (rego) | 97.7 (nl_statement) | 99.6 | 2.2 | 0.87 | 0.3 |
| Qwen3.8-27B | 95.3 (owner_statement) | 49.5 (rego) | 93.6 | 45.7 | 0.96 | 1.7 |
| Qwen3.8-27B thinking | 99.8 (rego) | 99.1 (nl_statement) | 99.8 | 0.7 | 0.95 | 0.0 |
| Qwen3.8-27B-FP8 | 95.1 (owner_statement) | 79.6 (rego) | 93.0 | 15.4 | 0.87 | 2.0 |
| Gemma-4-31B | 99.1 (json_policy) | 96.7 (nl_statement) | 99.1 | 2.4 | 1.00 | 0.0 |
| Llama-3.3-70B W8A8 | 84.4 (owner_statement) | 78.7 (json_policy) | 78.7 | 5.7 | 0.00 | 5.7 |
| gpt-oss-120b | 99.9 (executable) | 98.7 (nl_statement) | 99.8 | 1.2 | 0.90 | 0.1 |

(Accuracy in %, T2 application, point estimates from `paper/figures/phase2_summary.json`.)

For the four smallest models and Llama-3.3-70B, json_policy is the 5th or 6th best rendering. On
those models canonicalizing to JSON can only close the gap by pulling the best rendering down.
Under the floor reading, perfect conversion passes at most 7 of 12 configurations, which is exactly
the majority line. These are ceilings: conversion errors only lower them. The report gives them
next to every closure.

## Questions

- **Q8 (Gate 3, primary).** Does canonicalization close ≥ 50% of the worst-case T2 application gap
  on the majority of models (as operationalized under choices 1 and 2)?
- **Q9.** Where does the arm lose accuracy? Per rendering: conversion (parsed, valid, exact,
  engine-equivalent), the decision from an equivalent IR, and garbage-in (a right answer from a
  non-equivalent IR).
- **Q10.** How do the baselines compare with arm 1 on closure and worst-case accuracy? The baselines
  are CoT, restate, the solver pipeline, self-consistency (analysis only) and the proxy (upper
  bound, by construction).
- **Q11 (T3).** Under the proxy, how often does each rendering lead the agent to attempt an
  unauthorized call (`attempted_unauthorized`, the UIR the proxy prevents)? What happens to
  completion and authorized refusal once denied calls are blocked?
- **Q12.** What does each arm cost in tokens per query and in wall time?

We make no directional prediction beyond Gate 3's threshold.

## Arms (procedure)

The unit, rows and requests are those of Phase 2: 1,984 worlds × 2 assignments × 4 application
requests × 6 renderings = 95,232 application rows per model. Interpretation rows are not asked
(`tasks: [application]`). Decoding is vLLM 0.29.0, greedy, temperature 0, seed 20261010 (T2 arms)
or 20261013 (T3), at the Phase-2 pinned revisions. Throughput settings are recorded and don't
change greedy outputs.

1. **Canonicalization (arm 1, canon-v1).**
   - Stage 1: one conversion per (world, assignment, rendering) cell, 23,808 per model. The
     conversion prompt has the facts block, one rendering and the IR format description.
     Constrained decoding uses a per-world JSON schema holding only vocabulary the prompt shows;
     the reasoning models decode freely (choice 3). Limits: 6,144 tokens, or 12,288 for the
     reasoning models.
   - Stage 2: the frozen T2 prompt with only its policy block replaced by the canonically serialized
     IR. Limit 256 tokens, or 4,096 for the reasoning models, as T2.
   - A cell whose conversion doesn't parse gets no stage 2. Its rows keep the conversion's category
     (refusal / abstain / parse_failure) and count as not correct.
2. **Zero-shot CoT (baseline-v1).** The frozen T2 prompt with only its answer instruction replaced
   ("think it through step by step, then give your final answer on the last line as JSON only").
   Limit 2,048 tokens, or 4,096 for the reasoning models.
3. **Restate-as-JSON-then-answer (baseline-v1, one prompt).** The answer instruction is replaced
   by "first restate the policy in the canonical format (canon-v1), then answer". Limit 6,144
   tokens, or 12,288. The restated IR is scored like a conversion.
4. **Self-consistency (analysis only).** The majority of the six Phase-2 T2 answers to an
   instance, given to every rendering. A tie or no parsed answer is an abstention. It isn't
   deployable, since no single prompt has all six renderings, and is reported as such.
5. **Solver pipeline.** The model's arm-1 conversions decided by the engine: the reference
   evaluator, cross-checked with Cedar on every IR that renders as Cedar. An unusable IR (not
   parsed, or invalid) gets its own category. The fail-closed variant (deny) is reported alongside.
6. **External enforcement proxy (upper bound).** In T2 the engine decides on the certified policy,
   so accuracy is 1 by construction and nothing runs. In T3 (proxy-v1), every call the policy
   denies is blocked, and the agent sees
   `{"ok": false, "error": "blocked by the authorization proxy: ..."}`. Everything else is the
   Phase-2 agent-v4 protocol, prompt and limits.

T3 arms pending code (choice 4): canonicalization in T3 and the solver-enforced agent. Their
harness has to be built, tested and named here before any of their inference runs.

Answers are parsed with the T2 parser (`json-strict-v3`) for arm 1's stage 2, and with
`json-final-v1` for CoT and restate: the last JSON object holding exactly the task's field.
Refusal, abstention and parse failure are kept apart and never dropped.

## Metrics (per model and arm)

The world is the unit. CIs are world-clustered percentile bootstraps with 4,000 replicates, seed
20261010 (T2) and 20261013 (T3). These are the Phase-2 draws, so arm and baseline share them.

- **Per rendering:** accuracy, deny→allow and allow→deny rates, and refusal / abstain /
  parse-failure rates (`invariance.rendering_metrics`).
- **Gap closure against the Phase-2 T2 run of the same configuration,** on identical rows:
  - *own*, *pair* and *floor* closure, each with a paired bootstrap CI (draws where the baseline
    gap is 0 are dropped and counted);
  - the worst-case accuracy change, arm worst − baseline worst;
  - the per-rendering accuracy change.
- **Arm 1 and restate:** conversion rates per rendering (parsed, valid, exact, engine-equivalent,
  truncated) and garbage-in: right answer ∧ non-equivalent IR; right answer ∧ IR wrong on that
  request; accuracy given an equivalent / non-equivalent IR; and, as a diagnostic, the engine's
  decision on the predicted IR.
- **Solver:** Cedar agreement and engine time.
- **Proxy (T3):** `attempted_uir_deny` per rendering and its worst-case gap. Also completion,
  authorized refusal, protocol failures and blocked calls per episode.
- **Cost:** tokens per query by rendering, with conversion tokens amortized over the cell's
  queries; the ratio to the Phase-2 T2 tokens; wall time per stage from `timing.jsonl`.

## Gate 3 and inference

Gate 3, verbatim (`docs/RESEARCH_PLAN.md` §3): *"canonicalization closes ≥50% of the worst-case
gap on the majority of models, or we report honestly that it does not and the solver pipeline is
the recommendation."*

- Per configuration, arm 1 "closes ≥ 50%" when its primary closure point estimate is ≥ 0.50
  (choice 1, guard included if chosen). The count over the models of choice 2 is compared with the
  majority. CIs are reported for every closure; the gate itself uses point estimates, as the gate
  states.
- The P3.4 report ends with a Gate-3 recommendation. Jerzy decides G3.
- **Secondary inference:** per arm, the 12 worst-case-accuracy changes form one Holm family
  (α = 0.05), two-sided bootstrap p. Effect sizes and CIs lead.
- **Not tested:** cross-arm differences. They are compared descriptively on shared rows, since
  every arm uses the same instances and bootstrap draws.

## Runs (P3.4)

- **T2:** arm 1 on the 12 configurations; CoT and restate on the 12 (choice 6); self-consistency
  and solver derived for the 12. The solver needs arm-1 runs.
- **T3:** proxy on the 12 (choice 4).
- **Hardware:** H100 GPU 7, compose only (`memory/reference-dgx-usage.md`).
- **Rough cost** (an assumption, not a measurement), per non-reasoning configuration: about 10M
  output tokens of conversions, 30M of CoT and 50M of restate. Reasoning models are several times
  that. At Phase-2 throughput this comes to about 2 GPU-days for the T2 arms over the 12
  configurations. The proxy T3 sweep takes about as long as P2.8 did (2–3 GPU-days). The `orig`-only
  option in choice 6 saves about a day.

## Exclusions

None. Every row of every completed run is analysed. A failed run is resumed (`--resume` checks
config, model, dataset and versions) and never mixed with another configuration. A configuration
that can't be run is reported as missing, with the reason. Prompts, IR format, parsers, sandbox and
configs are frozen: any change gets a version bump, a dated amendment here and a rerun.

## Pre-registration engineering disclosure

- **P3.1 oracle dry run (CPU, no model call):** stage-2 answers were taken from Qwen3.5-4B's own
  Phase-2 T2 json_policy outputs. With oracle IRs, every stage-2 prompt matched a T2 json_policy
  prompt; arm accuracy was 0.672 on all six renderings. This only re-reads Phase-2 data.
- **P3.3 smoke test (2026-11-07, H100 GPU 7, vLLM 0.29.0, git `a880d3a`).** Model
  `smoke_qwen3_0_6b` (not a sweep model), `--limit` runs only, outputs discarded and not analysed.
  It checks the engine path and the parsers, not accuracy.
  - The first attempt failed at import: the vLLM image lacks scipy, which the P2.9 Spearman fix had
    made a module-level import. Fixed in PR #61 with a lazy import; no other change.
  - Canonicalization, constrained (72 cells, the first six worlds): the per-world schema compiled in
    vLLM's structured-output backend; 72/72 outputs parsed as IRs, all stopped normally (median 130,
    max 263 output tokens); 12/72 exact. Stage 2 ran on all 288 rows.
  - Canonicalization, free decoding (36 cells): 4/36 parsed. In the rest, the model wrote scopes as
    bare strings or malformed conditions, which are not IR objects, so they are parse failures, as
    defined. Their 128 rows were kept as `failed_stage: conversion`.
  - CoT (120 rows): all answers parsed with json-final-v1 (median 67 output tokens).
  - Restate (60 rows): the model wrote the IR and stopped without a final answer (54), or looped to
    the 6,144-token limit (6). All 60 are parse failures, which is correct; 10/60 restated IRs
    parsed.
  - Proxy T3 (96 episodes): 0 executed unauthorized calls, 48 blocked attempts, and 48 protocol
    failures from the tiny model.
  - No code or limit changed because of the smoke test.

## Amendments

None yet.
