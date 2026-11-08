# Pre-registration record

This folder documents the pre-registered decision rules referenced in the
paper (Appendix, Pre-registration Record).

Each phase's rules live in `docs/prereg/phaseN.md` (`docs/RESEARCH_PLAN.md`
§4), committed and pushed before the run it governs. If a prereg file is later
amended, freeze the original as `phaseN@<sha>.md` with
`git show <sha>:docs/prereg/phaseN.md > docs/prereg/phaseN@<sha>.md` and log
the amendment and its reason here.

The "first result" column gives the earliest timestamp of the run's output
(the run directory's timestamp under `outputs/`, or its earliest
`run_status.json` / `metadata.json` entry). Every run also records the git
hash of the code that produced it.

| rule (paper location) | pre-registration commit | committed (UTC) | first result (UTC) | file |
|---|---|---|---|---|
| Phase 0 kill test: Gate-0 rule, strict-v1 parser, worst-case gap, disagreement, flip definitions | `89b91c0` | 2026-10-08 00:47:05 | 2026-10-08 00:48:38 (qwen3_8b run dir) | `phase0.md` |
| Phase 0 addendum (current models; outside Gate 0): harmony final-channel answer, text-only loading, Holm over 3 models | `64b3b52` | 2026-10-08 01:26:29 | 2026-10-08 01:27:07 (gpt_oss_120b run dir) | `phase0-addendum.md` |
| Phase 2 T1/T2: worst-case gap, direction, dissociation, Holm over 72 (model, rendering) contrasts, scaling test, quantization rule | `91cf920` | 2026-10-08 17:34:28 | 2026-10-08 17:46:27 (qwen3_5_4b T2 run dir) | `phase2.md` (original: `phase2@91cf920.md`) |
| Phase 2 T3: worst-case UIR gap on deny requests, separate T3 Holm family over 72 (model, rendering) UIR contrasts, T2→T3 transfer, cross-tier Spearman, agent-v2 protocol | `4694983` | 2026-10-16 16:46:08 | 2026-10-16 16:49:55 (agent-v2 gpt_oss_120b run dir) | `phase2-t3.md` (original: `phase2-t3@4694983.md`) |
| Phase 3 mitigations: Gate-3 own-gap closure with a worst-case guard over all 12 configurations (pair / floor secondary), canonicalization canon-v1, baselines baseline-v1 / json-final-v1, enforcement proxy proxy-v1, Holm over 12 worst-case-accuracy changes per arm | `93d8d7a` | 2026-11-08 16:20:02 | — (P3.4 pending) | `phase3.md` |
| Phase 4 LoRA SFT: Cedar held out; H1–H3 (Cedar transfer vs no-SFT and vs the token-matched control; six-rendering gap), Holm over 6; BFCL v3 simple + multiple regression (> 2 pp, CI excludes 0) | `3ebec83` | 2026-11-08 16:22:17 | — (P4.4 pending) | `phase4.md` |
| Phase 5 probing (optional; runs last): Q18 decodability (CI lower bound ≥ metadata probe + 0.05 in all six renderings), Q19 transfer gap, Q20 CKA, Q21 divergence AUC, Holm over 30 AUCs; Qwen3.5-9B and Gemma-4-12B-it, 400 worlds | `6ce99bd` | 2026-11-08 16:24:29 | — (P5.3 pending) | `phase5.md` |

## Amendments

| file | original frozen as | amendment commit | committed (UTC) | what and why |
|---|---|---|---|---|
| `phase2.md` | `phase2@91cf920.md` | `7a65522` | 2026-10-08 17:44:17 | T1 log-probabilities via vLLM prompt log-probs instead of HF transformers (same token ids and continuation sum; 100/100 T1-correct agreement on a smoke-only model); committed before the first sweep run |
| `phase2.md` | `phase2@91cf920.md` | `318bc5c` (PR #48 branch; squash-merged to `main` as `b55a75e`, 2026-10-15 09:51:19) | 2026-10-14 16:53:11 | parser `json-strict-v3`: strip the reasoning before `</think>` when the chat template opened `<think>` in the prompt (v2 scored the Qwen3.8-27B thinking run 99% `parse_failure`); every T2 run re-scored offline at 2026-10-14 20:27, only the thinking run changes; approved by Jerzy |
| `phase2-t3.md` | `phase2-t3@4694983.md` | `5df80dd` | 2026-10-16 21:41:28 | agent-v3: read the harmony final channel when its header has a content type (the agent-v2 gpt-oss-120b run returned empty turns); all 12 configurations rerun |
| `phase2-t3.md` | `phase2-t3@4694983.md` | `1c723be` | 2026-11-04 19:27:59 | score a non-string `resource` as the preregistered protocol failure (it crashed the run); gpt-oss-120b resumed across the fix, on Jerzy's approval |
| `phase2-t3.md` | `phase2-t3@4694983.md` | `801606c` | 2026-11-06 15:28:54 | agent-v4: read gpt-oss native tool calls (19.4% gpt-oss-20b protocol failures under agent-v3); both gpt-oss models rerun, the other 10 agent-v3 runs kept |
| `phase2-t3.md` | `phase2-t3@4694983.md` | the P2.8 report PR | 2026-11-06 | erratum: one diagnostic count in the agent-v4 amendment (207 → 219 of 520); no rule changes |
| `phase3.md` | `phase3@93d8d7a.md` | `75d4238` | 2026-11-08 16:39:43 | amendment 1: names the two P3.2b T3 arms (canon-t3-v1, solver-proxy-v1), their configs, metrics and a Holm family over the 12 worst-case-UIR changes per arm, as decision 4 required; committed before any of their inference, with the arm-1 T2 sweep started but no output read |
| `phase4.md` | `phase4@3ebec83.md` | `e26c1b3` | 2026-11-08 16:54:57 | amendment 1: names the P4.2b harness: T2 adapter evaluation through vLLM LoRA (hash-checked adapters) and the BFCL check bfcl-ast-v1 (BFCL v3 simple + multiple, 600 items at HF revision 61fc0608, gorilla 6ea5797 prompt/decoder/checker, per-arm seed-mean items, regression = all-item drop > 2 pp with CI excluding 0); committed before any BFCL, adapter-evaluation or training run |
