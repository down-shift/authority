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
| Phase 2 T1/T2: worst-case gap, direction, dissociation, Holm over 72 (model, rendering) contrasts, scaling test, quantization rule | `91cf920` | 2026-10-08 17:34:28 | — (P2.4) | `phase2.md` |
| Phase 2 T3: worst-case UIR gap on deny requests, separate T3 Holm family over 72 (model, rendering) UIR contrasts, T2→T3 transfer, cross-tier Spearman, agent-v2 protocol | `4694983` | 2026-10-16 16:46:08 | 2026-10-16 16:49:55 (agent-v2 gpt_oss_120b run dir) | `phase2-t3.md` (original: `phase2-t3@4694983.md`) |

## Amendments

| file | original frozen as | amendment commit | committed (UTC) | what and why |
|---|---|---|---|---|
| `phase2-t3.md` | `phase2-t3@4694983.md` | `5df80dd` | 2026-10-16 21:41:28 | agent-v3: read the harmony final channel when its header has a content type (the agent-v2 gpt-oss-120b run returned empty turns); all 12 configurations rerun |
| `phase2-t3.md` | `phase2-t3@4694983.md` | `1c723be` | 2026-11-04 19:27:59 | score a non-string `resource` as the preregistered protocol failure (it crashed the run); gpt-oss-120b resumed across the fix, on Jerzy's approval |
| `phase2-t3.md` | `phase2-t3@4694983.md` | `801606c` | 2026-11-06 15:28:54 | agent-v4: read gpt-oss native tool calls (19.4% gpt-oss-20b protocol failures under agent-v3); both gpt-oss models rerun, the other 10 agent-v3 runs kept |
| `phase2-t3.md` | `phase2-t3@4694983.md` | the P2.8 report PR | 2026-11-06 | erratum: one diagnostic count in the agent-v4 amendment (207 → 219 of 520); no rule changes |
