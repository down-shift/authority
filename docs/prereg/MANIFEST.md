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
| Phase 0 addendum (current models; outside Gate 0): harmony final-channel answer, text-only loading, Holm over 3 models | `64b3b52` | 2026-10-08 01:26:29 | — (P0.5) | `phase0-addendum.md` |
