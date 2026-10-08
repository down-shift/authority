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
