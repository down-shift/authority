# Renderer / prompt changelog

Every renderer, prompt template, or answer-parser change is versioned and
logged here (`docs/RESEARCH_PLAN.md` §4). Changing one after seeing model
results requires a new version and a full rerun of everything that used it.

| date | component | version | change | reason | rerun required |
|---|---|---|---|---|---|
| 2026-10-08 | parser (`src/authinv/eval/parse.py`) | strict-v2 | Typographic apostrophes (’ ‘ ʼ) and double quotes (“ ”) folded to ASCII for refusal/abstention pattern matching; “…” and ‘…’ accepted as a wrapping quote pair. Exact-candidate match still compares the unfolded text with the candidates as given. strict-v1 kept unchanged and selectable (`parse_answer(..., version="strict-v1")`); `DEFAULT_PARSER_VERSION = "strict-v2"` for new configs; the Phase-0 runner reads `parser_version` from its config. | P0.5: gpt-oss refusals with typographic apostrophes were labelled parse_failure | no — Phase 0 stays on strict-v1; Phase 2+ use strict-v2 |
| 2026-10-08 | nl_statement renderer | nl-v1 | initial template + exact decoder | P1.2 | n/a (first version) |
| 2026-10-08 | owner_statement renderer | owner-v1 | initial template + exact decoder | P1.2 | n/a (first version) |
| 2026-10-08 | table renderer | table-v1 | initial template + exact decoder | P1.2 | n/a (first version) |
| 2026-10-08 | json_policy renderer | json-v1 | initial template + exact decoder | P1.2 | n/a (first version) |
| 2026-10-08 | executable renderer (Cedar, cedarpy 4.12.1 formatter) | cedar-v1 | initial template + exact decoder | P1.2 | n/a (first version) |
| 2026-10-08 | rego renderer (Rego v1 module, OPA v1.21.1) | rego-v1 | initial template + exact decoder (OPA's own parser, `opa parse`): fixed preamble (default deny with `default permit/forbid := false`, `allow if { permit; not forbid }`, transitive and reflexive `member` via `graph.reachable`) and one `permit`/`forbid` body per rule under `# rule <id>`; entity facts passed as OPA `data`, not shown in the rendering; sixth rendering, engine-checked with `opa eval` | P1.5 (H0.1: add Rego) | n/a (first version) |
| 2026-10-08 | CedarBench importer (`src/authinv/sources/cedarbench.py`) | cedarbench-import-v2 | Exact universe enlargement and an exact `\|\|` split; no policy's meaning changes. (1) Every schema action whose `appliesTo` admits the world's principal/resource types and declares the context keys the policy reads joins `actions` as a default-deny probe; an unconstrained `action` covers exactly those actions. (2) Every synthesized attribute/membership combination is instantiated twice (`ENTITY_REPLICAS = 2`, falling back to 1 under the 64-entity / 20,000-request caps). (3) A `when` clause whose top-level expression is a disjunction of supported conjunctions becomes one rule per disjunct (ids `<id>_or<k>`, several clauses multiply out, ≤ 32 rules); nested `\|\|` stays excluded as `nested_\|\|`. Per-world `n_allow`/`n_deny`/boundary/`non_degenerate`/`dead_rules` recorded. v1 kept selectable (`--importer cedarbench-import-v1`). | P1.7.2 (Jerzy 2026-10-08: worlds too small, `\|\|` the largest blocker) | yes — re-import (done in P1.7.2); no model results used CedarBench yet |
