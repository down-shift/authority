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
| Phase 0 kill test: Gate-0 rule, strict-v1 parser, worst-case gap, disagreement, flip definitions | `89b91c0` | 2026-10-08 00:47:05 | — (P0.4) | `phase0.md` |
