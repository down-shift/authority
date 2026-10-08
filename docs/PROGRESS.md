# PROGRESS — phase log, daily log & gate decisions

One line per working day (what landed, what's next) plus **every gate decision**,
recorded by the human who made it. Newest at the bottom. Format:

```
YYYY-MM-DD — <what landed / what's next>
YYYY-MM-DD — GATE <id>: <pass|fail> — <decision + who decided>
```

2026-10-07 — Autonomous `/continue` loop ported (ledger, runbook, gate, CI, memory/, paper/ ACL skeleton); plan pending from the human.
2026-10-07 — Research plan entered: docs/RESEARCH_PLAN.md + docs/novelty_check.md; steps E0.1–W1.1 in docs/PLAN.md. Waiting on human: H0.1 (§8 questions).
2026-10-08 — P0.1 authinv skeleton: package boundaries, pinned model registry (12 models), Gemma upstream manifests (gated repos reproduced from byte-identical mirrors; Llama mirrors are not identical, so the Llama bf16 access path is open for Phase 2).
2026-10-08 — R0.1 related-work doc: 77 papers verified, 7 mismatches flagged.
2026-10-08 — P0.2 Phase-0 harness merged; end-to-end smoke test on A100 with Qwen3-0.6B (not a Phase-0 model, 20 rows, discarded).
2026-10-08 — P0.3 Phase 0 preregistered (`89b91c0`); Phase-0 inference may start.
2026-10-08 — Jerzy approved the model-roster refresh: Phase-2 sweep amended to current releases (Qwen3.5/3.8, Gemma 4, gpt-oss, Ministral 3; Llama-3.3-70B W8A8 kept as anchor); new step P0.5 (Phase-0 addendum on Qwen3.8-27B, Gemma-4-31B, gpt-oss-120b), G0 now waits for it.
2026-10-08 — P0.5 harness: smoke-tested on H100 with gpt_oss_20b and gemma4_e4b (not addendum models, 20 rows each, discarded).
2026-10-08 — P0.5 addendum preregistered (`64b3b52`).
2026-10-08 — P0.4 Phase 0 complete: Gate-0 recommendation **FAIL** (≥27B models: gap ≤1.1 pp, disagreement ≤2.2%; Llama-70B perfect). Qwen3-8B gap 8.6 pp, all deny→allow. Waiting on human: G0 (after P0.5).
2026-10-08 — P0.5 addendum complete (outside Gate 0): Qwen3.8-27B 0.6 pp, Gemma-4-31B 0.0 pp, gpt-oss-120b 7.5 pp on natural language, errors are refusals (fail-closed; strict-v1 labels them parse failures — curly apostrophe; strict-v2 proposed). All A100/H100 work cleaned up; runs archived on V100. Waiting on human: G0, H0.1.
2026-10-08 — GATE G0: **waived** by Jerzy: proceed to Phase 1 (Gate-0 rule recommended FAIL; toy worlds at ceiling for ≥27B, gpt-oss-120b 7.5 pp; Phase 1 is the real test). H0.1 answered: Rego in; H100 unlimited; T3 sandbox built fresh; ownership deferred; Eiers cite only. Approved: strict-v2 parser (P2.0), Quacky pinning (P1.8), threat model (W0.1), Phase-4 models refreshed; paper stays ACL. .DS_Store untracked.
2026-10-08 — P1.1 canonical policy object (Cedar semantics: default deny, forbid overrides; scopes any/eq/is/in; schema-checked conditions; derived decision ownership; canonical + semantic hashes).
2026-10-08 — P2.0 parser strict-v2 (typographic apostrophes); Phase 0 stays on strict-v1.
2026-10-08 — W0.1 threat model written (docs/threat_model.md).
2026-10-08 — P1.3 Cedar engine (cedarpy 4.12.1) + request sets: reference semantics agree with Cedar on 23,976 requests over 165 random policies (0 engine errors); strict schema validation; balanced boundary-first sampling. Added the is_in scope kind (Cedar `principal is T in G`).
2026-10-08 — P1.2 five versioned renderers (nl-v1, owner-v1, table-v1, json-v1, cedar-v1) with exact decoders; every rendering round-trips on fuzzed policies of all three tiers; Cedar decoding uses Cedar's own parser.
2026-10-08 — P1.4 equivalence checker: a world ships only if all five renderings decode to the canonical rules, give the reference decision on every request in the universe, and (executable) the Cedar engine agrees with zero errors and the strict typechecker accepts it; hashed proofs to equivalence.jsonl. Tampered renderings (flipped effect ×5, threshold, dropped rule, garbage, unless-clause) are caught.
2026-10-08 — P1.6 synthetic sources: 720 worlds (repo 360, mcp 360; 24-cell grid × 15), all certified; balanced over roles/conditions/effect (docs/experiments/p1.6-synthetic-sources.md).
2026-10-08 — P2.1 T1 log-prob scorer ported into authinv (continuation boundaries + audit, NumPy reference, batched HF scorer verified against the reference on a tiny model).
2026-10-08 — P1.5 Rego rendering (rego-v1) + OPA 1.21.1 engine check; reference agrees with OPA on 41,958 requests over 165 random policies.
