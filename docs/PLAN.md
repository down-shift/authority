# Master plan & claim ledger

This is the **single living source of truth** for the road to the paper. It is
both the roadmap (ordered, dependency-aware steps) and the **claim ledger** that
lets several Claude Code agents (and humans, on GPU nodes) work in parallel
without colliding. The loop that drives it is `/continue`; the full protocol is
in `docs/AUTONOMY.md`. The research spec and the reasoning behind every step is
`docs/RESEARCH_PLAN.md` (read it before touching research code); positioning
against prior work is `docs/novelty_check.md`.

**How to work this file:** don't hand-edit statuses during the loop — the
scripts do it atomically. To advance the project, run `/continue` (or say
«продолжи»), which does **exactly one step** then asks before continuing.
**Changes to the step set (adding / splitting / re-sequencing steps) must be
proposed to a human and confirmed before they are applied** — the loop never
self-edits the plan's structure.

**Thesis (frozen):** `docs/RESEARCH_PLAN.md` §0 — authorization policies are not
representation-invariant; measured as worst-case accuracy over engine-certified
equivalent renderings, worst-case unauthorized-action rate, and deny→allow flips;
mitigations judged by closure of the worst-case gap.
**Target:** NeurIPS 2027 main track (deadline ~May 2027); fallbacks NeurIPS D&B, then ACL ARR.
**Compute:** open-weight models only, local H100/V100 cluster, $0 API spend.
**Gates:** G0–G3 are `[HUMAN]` steps. A phase starts only after its gate is
recorded as passed or waived by Jerzy (`mark.py G<n> done "<pass|waived>: …"`).
Gate 0 FAIL → pivot (SFT becomes primary); the human rewrites the plan.

## Step format (machine-parsed by scripts/claim.py & scripts/mark.py)

```
### <id> — <title>
- status: todo | claimed | in_review | done | blocked | cut
- owner: —                      (set by claim.py)
- claimed_at: —                 (set by claim.py)
- deps: <id> <id> | —           (a step is eligible only when all deps are done)
- source: <which RQ / plan section / request motivates it>
- done-when: <concrete, checkable acceptance criteria>
```

`cut` is terminal: a step retired by human decision (never re-claimed).
`[HUMAN]` in a title marks a decision only a human can close (gates, open
questions): `claim.py` never claims it and reports it as "waiting on human".
Steps that need a GPU say so in their title/`done-when` (`[GPU]`, or
`[GPU: <class, memory>]`) and must only be claimed on a machine that has one —
see `docs/AUTONOMY.md` § Hardware-aware claiming.

---

## E0 — Bootstrap

### E0.1 — Autonomous loop, ledger, gate, CI, memory/, ACL paper skeleton
- status: done
- owner: bootstrap
- claimed_at: 2026-10-07
- deps: —
- source: user request 2026-10-07 (port the latent-underspecification-research loop)
- done-when: docs/PLAN.md + docs/AUTONOMY.md + .claude/commands/continue.md + scripts/{bootstrap,agent_id,check}.sh + scripts/{claim,mark}.py + tests/test_loop.py + .github/workflows/gate.yml + memory/ + paper/ committed; gate green.

### H0.1 — [HUMAN] Answer the open questions in RESEARCH_PLAN §8
- status: done
- note: answered by Jerzy 2026-10-08; see RESEARCH_PLAN §8
- owner: —
- claimed_at: —
- deps: —
- source: RESEARCH_PLAN §8
- done-when: Jerzy records answers in docs/RESEARCH_PLAN.md §8 (or docs/PROGRESS.md): (1) Rego as a sixth rendering yes/no; (2) GPUs + hours for 70B bf16 runs, and whether Qwen3-235B-A22B is in; (3) T3 sandbox fresh vs adapted from AgentDojo; (4) owners of benchmark / agent harness; (5) Eiers group collaboration yes/no. Then `mark.py H0.1 done`.

### R0.1 — Related-work doc with verified citations
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T00:27:11Z
- deps: —
- source: RESEARCH_PLAN §2, §6.5; docs/novelty_check.md
- done-when: docs/related_work.md expands the §2 table: each paper's arXiv ID, title, authors, and withdrawal/version status checked against its arXiv abstract page (mismatches flagged, not silently fixed); one-paragraph overlap + differentiator per paper; the novelty-claim table from novelty_check.md mapped to planned results; BibTeX entries for all verified papers added to paper/references.bib; a "last re-searched" date line for the monthly arXiv recheck.

## Phase 0 — Kill test on open models (Weeks 1–2)

### P0.1 — authinv package skeleton, configs, logs
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T00:27:14Z
- deps: —
- source: RESEARCH_PLAN §1 [DECISION], §4, §5
- done-when: src/authinv/__init__.py (+ subpackages policy/, render/, equivalence/, eval/, agent/ as empty modules with docstrings); configs/authinv/models.yaml listing the Phase-0 and Phase-2 sweep models with pinned HF revisions (revision: null + a TODO only where a revision can't be resolved offline, flagged in the PR); pyproject extras `engines`, `probing`, `agent` declared (vllm in `inference`, lazily imported); docs/CHANGELOG.md (renderer/prompt version log) created; a test asserts authinv imports nothing from authorization_invariance / authorization_competence; gate green.

### P0.2 — Phase-0 generation harness (greedy + strict parser)
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T00:31:56Z
- deps: P0.1
- source: RESEARCH_PLAN §3 Phase 0 tasks 1 & 3
- done-when: scripts/authinv_phase0.py + configs/authinv/phase0.yaml read the Gemma Stage-2 worlds.jsonl (path in config, hash verified), build generation prompts for all 180 worlds × both name assignments × five renderings × interpretation/application, decode greedily via vLLM (lazy import, temperature 0, vLLM version + GPU type recorded), and parse with a strict parser whose failures/refusals/abstains are their own category; src/authinv/eval/metrics.py computes mean accuracy per rendering, worst-case accuracy, per-world rendering disagreement, deny→allow vs allow→deny flip counts (mapping for the toy worlds defined in the config: selecting the non-owner's value = allow of an unauthorized action), world-clustered bootstrap CIs (≥2,000 resamples), Holm-corrected p-values; --dataset-only, --resume, config.json, metadata.json, run_status.json, SHA-256 manifest as in the existing runners; unit tests for parser and metrics on synthetic rows; gate green.

### P0.3 — Preregister Phase 0
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T00:46:50Z
- deps: P0.2
- source: RESEARCH_PLAN §3 Gate 0, §4
- done-when: docs/prereg/phase0.md states hypotheses, models (Qwen3-8B bf16, Qwen3-32B bf16, Gemma-3-27B bf16, Llama-3.3-70B int8), metrics, exclusion rules, the flip definition, and Gate 0 verbatim (PASS iff on ≥2 of the 3 ≥27B models worst-case accuracy is ≥5 pp below the best rendering with 95% CI excluding zero, or per-world rendering disagreement ≥5%; FAIL → SFT pivot); committed and pushed to main before any Phase-0 inference; MANIFEST row added.

### P0.4 — [GPU: H100] Phase-0 run and report
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T00:47:38Z
- deps: P0.3
- source: RESEARCH_PLAN §3 Phase 0 task 2
- done-when: all four models run on the frozen Phase-0 dataset with full provenance; docs/experiments/p0.4-phase0-kill-test.md (result positive/neutral/negative) reports every preregistered metric with CIs per model, per rendering, and ends with the Gate 0 recommendation for Jerzy (not a decision); docs/PROGRESS.md entry added.

### P0.5 — [GPU: H100] Phase-0 addendum on current models
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T01:16:10Z
- deps: P0.3
- source: Jerzy 2026-10-08 (model roster refresh); RESEARCH_PLAN §3 Phase 0 addendum
- done-when: harness supports the new models (multimodal checkpoints loaded text-only; gpt-oss harmony output with the answer taken from the final channel); docs/prereg/phase0-addendum.md committed and pushed before any addendum inference (same frozen data, parser, metrics; per-model generation settings fixed in configs/authinv/phase0_addendum.yaml); Qwen3.8-27B, Gemma-4-31B-it, gpt-oss-120b run with full provenance; docs/experiments/p0.5-phase0-addendum.md reports them next to Phase 0, explicitly outside the Gate-0 rule.

### G0 — [HUMAN] Gate 0 decision: proceed or pivot
- status: done
- note: waived by Jerzy 2026-10-08: proceed to Phase 1
- owner: —
- claimed_at: —
- deps: P0.4 P0.5
- source: RESEARCH_PLAN §3 Gate 0
- done-when: Jerzy records PASS / FAIL / waiver in docs/PROGRESS.md. PASS → `mark.py G0 done`. FAIL → Jerzy rewrites the remaining plan for the SFT-primary paper before anything else is claimed.

## Phase 1 — Benchmark with engine-certified equivalence (Weeks 3–8)

### P1.1 — Canonical policy object
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:26:49Z
- deps: G0
- source: RESEARCH_PLAN §3 Phase 1 task 1
- done-when: src/authinv/policy/ defines the canonical policy (principal, action, resource, effect, conditions, owner) with difficulty tiers (single rule; multi-rule deny-overrides; attribute-conditioned), deterministic serialization + SHA-256 hash, and a reference evaluator for requests; unit tests cover each tier incl. deny-overrides precedence; gate green.

### P1.2 — Five versioned renderers + decoders
- status: done
- note: landed on main as b5d929f without a PR (process slip); gate + CI green
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:30:37Z
- deps: P1.1
- source: RESEARCH_PLAN §3 Phase 1 task 2
- done-when: src/authinv/render/ has pure functions nl_statement, owner_statement, table (Markdown), json_policy, executable (Cedar), each with a RENDERER_VERSION string and a decoder back to the canonical object; round-trip tests (canonical → rendering → decode == canonical) over all tiers; renderer versions logged in docs/CHANGELOG.md; gate green.

### P1.3 — Cedar engine + request-set generator
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:30:39Z
- deps: P1.1
- source: RESEARCH_PLAN §3 Phase 1 [DECISION] engine, task 3
- done-when: `engines` extra / documented install for the cedar-policy CLI or Python bindings; src/authinv/equivalence/engine.py evaluates (policy, request) via the real engine; request generator yields balanced allow/deny sets incl. boundary requests per policy; engine tests are skipped when the engine is absent and run in CI if it can be installed there (decision noted in the PR); gate green.

### P1.4 — Equivalence checker
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:38:21Z
- deps: P1.2 P1.3
- source: RESEARCH_PLAN §3 Phase 1 task 3
- done-when: src/authinv/equivalence/ checks per world that every rendering decodes to the canonical object and that the engine's allow/deny on the executable rendering matches the reference evaluator on every request; a world ships only if all renderings pass; proofs written to equivalence.jsonl with hashes; tests with known-equivalent and known-different pairs; gate green.

### P1.5 — OPA/Rego rendering and engine (if H0.1 says yes)
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:40:05Z
- deps: P1.4 H0.1
- source: RESEARCH_PLAN §3 Phase 1 [DECISION] secondary engine, §8.1
- done-when: if H0.1 answered "no", the human marks this step cut. Otherwise a versioned Rego renderer + decoder, `opa eval` wrapper, and Rego included in equivalence checks with tests; gate green.

### P1.6 — Synthetic policy sources (MCP allowlists, repo permissions)
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:40:07Z
- deps: P1.4
- source: RESEARCH_PLAN §3 Phase 1 task 4 (synthetic)
- done-when: grammar-based generators for MCP tool allowlists and GitHub-style repo permissions, seeded from config, balanced over effect, role count, condition count, and tier; all generated worlds pass the equivalence checker; balance table emitted by --dataset-only; tests; gate green.

### P1.7 — CedarBench import
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:40:09Z
- deps: P1.4
- source: RESEARCH_PLAN §3 Phase 1 task 4 (real); arXiv 2607.03656
- done-when: fetch script (data/raw git-ignored) with pinned source URL/commit and license recorded; CedarBench policies mapped to canonical objects where expressible (unsupported constructs counted and reported, not dropped silently); imported worlds engine-checked; tests on a small fixture; gate green.

### P1.8 — Quacky AWS IAM import → Cedar
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T08:01:06Z
- deps: P1.7
- source: RESEARCH_PLAN §3 Phase 1 [DECISION] IAM handling
- done-when: the Quacky policy dataset pinned (source URL/commit, license, and the **actual** policy count: the "587 policies" figure is unverified per docs/related_work.md, so the size is whatever the pinned source contains) and fetched; either local SMT equivalence (if it runs without pain) or IAM→Cedar translation validated by request-level differential testing against an IAM evaluator; method choice and failure counts documented; tests on fixtures; gate green.

### P1.9 — ACRE import (NL → canonical, engine-checked)
- status: cut
- note: dropped by Jerzy 2026-10-08: no licensed public ACRE source
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:40:12Z
- deps: P1.4
- source: RESEARCH_PLAN §3 Phase 1 task 4 (real); arXiv 2603.15799
- done-when: ACRE statements fetched with pinned source + license; mapped to canonical objects (mapping rules deterministic and tested; ambiguous statements excluded with counts reported); engine-checked; gate green.

### P1.6.2 — Synthetic top-up
- status: claimed
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T11:06:46Z
- deps: P1.6
- source: Jerzy 2026-10-08 (fill the ≥1,000-world target synthetically, alongside enlarged real sources)
- done-when: configs/authinv/synthetic.yaml replicates raised to 30 (1,440 worlds), regenerated and certified on all six renderings (incl. Rego); report docs/experiments/p1.6-synthetic-sources.md updated with the new counts (old counts kept as history); gate green.

### P1.7.2 — CedarBench importer v2: enlarged universes + exact `||` split
- status: todo
- owner: —
- claimed_at: —
- deps: P1.7
- source: Jerzy 2026-10-08 (approved universe enlargement); docs/experiments/p1.7-cedarbench-import.md caveats
- done-when: importer version bump (recorded in each world's meta and docs/CHANGELOG.md): the request universe adds every schema action as a probe and extra deterministic entities per type so worlds are non-degenerate where the policy allows; `||` at the top of a when-clause is split exactly into separate rules (DNF over the disjunction only; anything else still excluded); every world re-certified (all six renderings) and faithful to the original Cedar text on the enlarged universe; report updated with v1-vs-v2 counts incl. the number meeting the ≥4 allow / ≥4 deny bar, split by the paper's 221 tasks vs the 5 stress scenarios; gate green.

### P1.8.2 — Quacky importer v2: wildcard witnesses + exact Not* complements
- status: todo
- owner: —
- claimed_at: —
- deps: P1.8
- source: Jerzy 2026-10-08 (approved universe enlargement); docs/experiments/p1.8-quacky-import.md finding
- done-when: importer version bump: one deterministic concrete witness per wildcard pattern added to the closed universe; NotAction / NotResource translated as exact complements over the closed universe (NotPrincipal only if exact); every world faithful to the original IAM JSON under the independent IAM evaluator and certified on all six renderings; report updated with v1-vs-v2 counts incl. the non-degenerate count; gate green.

### P1.10 — Query builder, balancing audits, frozen benchmark
- status: todo
- owner: —
- claimed_at: —
- deps: P1.6.2 P1.7.2 P1.8.2
- source: RESEARCH_PLAN §3 Phase 1 tasks 5–7; Jerzy 2026-10-08 (ACRE dropped; real/synthetic split reported separately)
- done-when: worlds drawn from CedarBench v2, Quacky v2, and synthetic sources, with source kind (real / synthetic) carried on every row and reported separately in every table; interpretation and application queries per world; name-swap, option-position, request-order, and rendering-order balancing; tokenizer length audit recorded as a covariate (not equalized); scripts/authinv_build.py --dataset-only produces a frozen benchmark of ≥1,000 worlds across tiers, ≥5 renderings, ≥4 requests/world (≥20k application rows/model) with dataset hash + manifest; balance and size audits pass; tests; gate green.

### P1.11 — Datasheet and spot-check packet
- status: todo
- owner: —
- claimed_at: —
- deps: P1.10
- source: RESEARCH_PLAN §3 Gate 1
- done-when: docs/datasheet.md (motivation, composition, sources + licenses, equivalence method, known limitations); a seeded 20-world spot-check packet (all renderings side by side, with a form for marking ambiguity) generated by script for Jerzy; equivalence summary for the full dataset-only run committed in docs/experiments/p1.11-benchmark-audit.md.

### G1 — [HUMAN] Gate 1: benchmark sign-off
- status: todo
- owner: —
- claimed_at: —
- deps: P1.11
- source: RESEARCH_PLAN §3 Gate 1
- done-when: Jerzy completes the 20-world spot-check (no rendering reads as ambiguous, or fixes requested as new steps) and records the gate in docs/PROGRESS.md.

## Phase 2 — Three evaluation tiers on the model sweep (Weeks 7–12)

### P2.0 — Parser strict-v2 (Unicode apostrophes)
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:26:51Z
- deps: G0
- source: P0.5 finding (curly-apostrophe refusals labelled parse_failure); approved by Jerzy 2026-10-08
- done-when: PARSER_VERSION strict-v2 accepts typographic apostrophes/quotes in refusal and abstention patterns (and in normalization); docs/CHANGELOG.md entry; tests with the P0.5 refusal strings; strict-v1 remains importable so Phase-0 results stay reproducible; Phase 2+ configs use strict-v2; gate green.

### P2.1 — T1 log-prob scorer for authinv
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:44:33Z
- deps: P1.2
- source: RESEARCH_PLAN §3 Phase 2 T1
- done-when: the existing candidate log-prob scorer copied and adapted into src/authinv/eval/ (no imports from legacy packages), scoring authinv rendered prompts; candidate token-count audit; tests with a fake tokenizer/logits; gate green.

### P2.2 — T2 structured-generation harness
- status: done
- note: marked done prematurely before merge; corrected
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:53:12Z
- deps: P0.2 P1.2 P2.0
- source: RESEARCH_PLAN §3 Phase 2 T2, headline tables
- done-when: vLLM greedy decoding with a strict JSON answer schema over authinv datasets; parse failures / refusals / abstains as separate categories; metrics extended with dissociation rate (interpretation correct ∧ application wrong, by rendering) and deny→allow flip rate on engine-labelled requests; scripts/authinv_eval.py with --dataset-only / --tokenizer-audit-only / --resume and full provenance; tests; gate green.

### P2.3 — Preregister Phase 2 (T1/T2)
- status: todo
- owner: —
- claimed_at: —
- deps: P2.1 P2.2 G1
- source: RESEARCH_PLAN §4
- done-when: docs/prereg/phase2.md (hypotheses, sweep table from §3 Phase 2, metrics, Holm family, exclusion rules, quantization-control rule for Llama-70B) committed and pushed before sweep inference; MANIFEST row.

### P2.4 — [GPU: H100] Sweep A: ≤14B models (T1 + T2)
- status: todo
- owner: —
- claimed_at: —
- deps: P2.3
- source: RESEARCH_PLAN §3 Phase 2 sweep
- done-when: the ≤14B models of the amended roster (Qwen3.5-4B/9B, Gemma-4-E4B/12B, gpt-oss-20b, Ministral-3-14B) complete T1 and T2 on the frozen benchmark with provenance; per-run reports referenced from docs/experiments/p2.4-sweep-a.md. (Roster amended 2026-10-08.)

### P2.5 — [GPU: H100] Sweep B: 27–31B models and controls (T1 + T2)
- status: todo
- owner: —
- claimed_at: —
- deps: P2.3
- source: RESEARCH_PLAN §3 Phase 2 sweep
- done-when: Qwen3.8-27B (thinking off and on), Qwen3.8-27B-FP8 (quantization control: bf16-vs-FP8 delta reported against the rendering effect), Gemma-4-31B complete T1 and T2; docs/experiments/p2.5-sweep-b.md. (Roster amended 2026-10-08.)

### P2.6 — [GPU: H100] Sweep C: largest models
- status: todo
- owner: —
- claimed_at: —
- deps: P2.3 H0.1
- source: RESEARCH_PLAN §3 Phase 2 sweep, §8.2
- done-when: gpt-oss-120b and the Llama-3.3-70B W8A8 anchor complete T1 (where applicable) and T2; optional models (Qwen3.6-35B-A3B, Gemma-4-26B-A4B, Granite-4.2) only if H0.1 allows the GPU time; docs/experiments/p2.6-sweep-c.md. (Roster amended 2026-10-08.)

### P2.7 — T3 agent sandbox
- status: todo
- owner: —
- claimed_at: —
- deps: P1.10 H0.1
- source: RESEARCH_PLAN §3 Phase 2 T3, §8.3
- done-when: src/authinv/agent/ minimal MCP tool sandbox (file ops, repo ops, HTTP stubs) built fresh (H0.1 decision 2026-10-08), reusing only the UIR / refusal / completion metric definitions; tasks derived from the same policies as T2, with forbidden tools/resources; policy in the system prompt in one rendering; metrics UIR (as in arXiv 2605.18414), authorized-action refusal rate, task completion; tests with a scripted mock agent covering each metric; gate green.

### P2.8 — [GPU: H100] T3 runs over the sweep
- status: todo
- owner: —
- claimed_at: —
- deps: P2.7 P2.3
- source: RESEARCH_PLAN §3 Phase 2 T3
- done-when: T3 preregistration appended to docs/prereg/phase2.md (or phase2-t3.md) before inference; sweep models run in the sandbox with rendering as the only varied factor; T2→T3 transfer reported; docs/experiments/p2.8-agentic-uir.md.

### P2.9 — Figures and Phase-2 report
- status: todo
- owner: —
- claimed_at: —
- deps: P2.4 P2.5 P2.6 P2.8
- source: RESEARCH_PLAN §3 Phase 2 headline tables, Gate 2
- done-when: scripts/authinv_figures.py regenerates all headline tables/figures from outputs/ (mean by rendering, worst-case, disagreement, deny→allow flips, dissociation, scaling of worst-case gap vs parameters) into paper/figures/; docs/experiments/p2.9-phase2-report.md with the Gate 2 recommendation; docs/PROGRESS.md entry.

### G2 — [HUMAN] Gate 2: sweep complete
- status: todo
- owner: —
- claimed_at: —
- deps: P2.9
- source: RESEARCH_PLAN §3 Gate 2
- done-when: Jerzy confirms all sweep runs are complete with frozen datasets, provenance, and CIs, and records it in docs/PROGRESS.md.

## Phase 3 — Mitigations (Weeks 12–15)

### P3.1 — Policy canonicalization arm
- status: todo
- owner: —
- claimed_at: —
- deps: G2
- source: RESEARCH_PLAN §3 Phase 3 arm 1
- done-when: rendering → canonical JSON IR via constrained decoding over the IR schema (or candidate-likelihood selection as in E3), then decision from the IR; conversion accuracy, answer accuracy, and garbage-in rate (wrong IR, right answer) reported separately; token + latency accounting; tests; gate green.

### P3.2 — Baseline arms
- status: todo
- owner: —
- claimed_at: —
- deps: G2
- source: RESEARCH_PLAN §3 Phase 3 arm 2, §6 recommendations
- done-when: zero-shot CoT; restate-as-JSON-then-answer (single prompt); self-consistency across renderings (labelled analysis-only, not deployable); Policy-as-Logic-style pipeline (LLM extracts facts, Clingo/engine decides); external enforcement proxy upper bound (0% UIR by construction); token + latency accounting per arm; tests; gate green.

### P3.3 — Preregister Phase 3
- status: todo
- owner: —
- claimed_at: —
- deps: P3.1 P3.2
- source: RESEARCH_PLAN §3 Gate 3, §4
- done-when: docs/prereg/phase3.md with Gate 3 verbatim (canonicalization closes ≥50% of the worst-case gap on the majority of models, else the solver pipeline is the recommendation) committed and pushed before inference; MANIFEST row.

### P3.4 — [GPU: H100] Mitigation runs and report
- status: todo
- owner: —
- claimed_at: —
- deps: P3.3
- source: RESEARCH_PLAN §3 Phase 3
- done-when: all arms evaluated on T2 and T3 across the sweep (or the preregistered subset); gap closure with CIs; docs/experiments/p3.4-mitigations.md with the Gate 3 recommendation; docs/PROGRESS.md entry.

### G3 — [HUMAN] Gate 3: mitigation verdict
- status: todo
- owner: —
- claimed_at: —
- deps: P3.4
- source: RESEARCH_PLAN §3 Gate 3
- done-when: Jerzy records the Gate 3 outcome and the paper's mitigation recommendation in docs/PROGRESS.md.

## Phase 4 — Format-diverse LoRA SFT (Weeks 13–16, parallel with Phase 3)

### P4.1 — SFT data (Cedar held out)
- status: todo
- owner: —
- claimed_at: —
- deps: G2
- source: RESEARCH_PLAN §3 Phase 4 task 1, 3
- done-when: synthetic training policies disjoint from the eval benchmark (hash-checked), rendered in four renderings with Cedar held out; matched-token single-format control set; no-SFT baseline defined; dataset hashes + manifest; tests for disjointness; gate green.

### P4.2 — LoRA SFT trainer
- status: todo
- owner: —
- claimed_at: —
- deps: P4.1
- source: RESEARCH_PLAN §3 Phase 4 task 2
- done-when: config-driven LoRA trainer (Qwen3.5-9B, Gemma-4-12B-it — amended 2026-10-08; r ∈ {16..64}; seeds from config, 3 per arm) with provenance and adapter hashes; a smoke test on a tiny model skipped without torch; gate green.

### P4.3 — Preregister Phase 4
- status: todo
- owner: —
- claimed_at: —
- deps: P4.2
- source: RESEARCH_PLAN §3 Phase 4, §4
- done-when: docs/prereg/phase4.md (held-in vs held-out worst-case gap, controls, BFCL-subset regression check) committed and pushed before training; MANIFEST row.

### P4.4 — [GPU: H100] Train and evaluate
- status: todo
- owner: —
- claimed_at: —
- deps: P4.3
- source: RESEARCH_PLAN §3 Phase 4 task 4
- done-when: all arms × models × seeds trained and evaluated; worst-case gap on held-in and held-out (Cedar) renderings with CIs; general tool-use regression on a BFCL subset; docs/experiments/p4.4-format-diverse-sft.md; docs/PROGRESS.md entry.

## Phase 5 — Probing (Weeks 14–17, optional)

### P5.1 — Activation extraction, probes, CKA tooling
- status: todo
- owner: —
- claimed_at: —
- deps: G2
- source: RESEARCH_PLAN §3 Phase 5 tasks 1, 2, 4
- done-when: `probing` extra; residual-stream extraction on T1 prompts with cache integrity hashes; per-layer linear probes for owner identity, allow/deny, governing field, trained on one rendering and tested on the others; metadata-only and shuffled-label baselines; per-layer CKA between renderings; tests on synthetic activations; gate green.

### P5.2 — Preregister Phase 5
- status: todo
- owner: —
- claimed_at: —
- deps: P5.1
- source: RESEARCH_PLAN §4
- done-when: docs/prereg/phase5.md committed and pushed before extraction runs; MANIFEST row.

### P5.3 — [GPU: H100] Probing and CKA run
- status: todo
- owner: —
- claimed_at: —
- deps: P5.2
- source: RESEARCH_PLAN §3 Phase 5 tasks 1, 2
- done-when: probes + CKA on the preregistered models; whether divergence predicts behavioural disagreement; docs/experiments/p5.3-probing.md.

### P5.4 — [GPU: H100] Causal check (steering / patching)
- status: todo
- owner: —
- claimed_at: —
- deps: P5.3
- source: RESEARCH_PLAN §3 Phase 5 task 3
- done-when: steering or activation patching from the best rendering into NL, with application-accuracy recovery and CIs; docs/experiments/p5.4-causal.md. Without a causal effect, the report says probing stays to one figure.

## Appendix and release

### A1.1 — [GPU: H100] Cross-scope interference with constraint count fixed (only if time)
- status: todo
- owner: —
- claimed_at: —
- deps: G2
- source: RESEARCH_PLAN §3 Appendix; arXiv 2608.12426
- done-when: E2-style design re-implemented in authinv with constraint count held fixed and semantic independence varied; preregistered in docs/prereg/appendix-cross-scope.md before inference; report docs/experiments/a1.1-cross-scope.md engaging with the "not pairwise" finding.

### D1.1 — Benchmark release and leaderboard script
- status: todo
- owner: —
- claimed_at: —
- deps: G1 P2.9
- source: RESEARCH_PLAN §6 deliverables 1–3
- done-when: release builder producing the HF Datasets layout (policies, renderings, equivalence proofs, request sets, datasheet, license compatible with every source); leaderboard script producing the worst-case table for any HF model id; T3 sandbox documented as a reusable harness; publishing itself is left to Jerzy.

### W0.1 — Threat model for format-choice exploitability
- status: done
- owner: jrzkaminski@Jerzy-Pro.local
- claimed_at: 2026-10-08T07:26:53Z
- deps: R0.1
- source: docs/novelty_check.md §5 and Novelty Assessment ("exploitable through format choice: open but needs a threat model"); approved by Jerzy 2026-10-08
- done-when: docs/threat_model.md defines the attacker who chooses a *legitimate* policy's rendering (tenant-written MCP allowlists, repo-synced policy files) vs one who injects fake policy text (Policy Puppetry, role confusion), states assumptions/capabilities/goals, maps each claim to the experiments that can support it (P2.x deny→allow / fail-closed rates, T3 UIR), and lists what the paper must not claim; cited related work from docs/related_work.md.

### W1.1 — Paper draft
- status: todo
- owner: —
- claimed_at: —
- deps: R0.1 W0.1 P2.9 P3.4 P4.4
- source: RESEARCH_PLAN §6 deliverable 4, §9
- done-when: paper/sections/* drafted from committed reports only (no numbers absent from docs/experiments), figures from scripts/authinv_figures.py, prereg appendix table mirroring docs/prereg/MANIFEST.md; compiles with latexmk. ACL template (Jerzy 2026-10-08: keep ACL for now).
