# Renderer / prompt changelog

Every renderer, prompt template, or answer-parser change is versioned and
logged here (`docs/RESEARCH_PLAN.md` §4). Changing one after seeing model
results requires a new version and a full rerun of everything that used it.

| date | component | version | change | reason | rerun required |
|---|---|---|---|---|---|
| 2026-10-08 | parser (`src/authinv/eval/parse.py`) | strict-v2 | Typographic apostrophes (’ ‘ ʼ) and double quotes (“ ”) folded to ASCII for refusal/abstention pattern matching; “…” and ‘…’ accepted as a wrapping quote pair. Exact-candidate match still compares the unfolded text with the candidates as given. strict-v1 kept unchanged and selectable (`parse_answer(..., version="strict-v1")`); `DEFAULT_PARSER_VERSION = "strict-v2"` for new configs; the Phase-0 runner reads `parser_version` from its config. | P0.5: gpt-oss refusals with typographic apostrophes were labelled parse_failure | no — Phase 0 stays on strict-v1; Phase 2+ use strict-v2 |
