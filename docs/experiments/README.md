# Experiment & validation reports

Every experiment, gate, or empirical validation in this project produces a
committed report here — **including negative or failed results**. A "fail" is a
finding, not something to bury: silently dropping a negative result is
forbidden (see the Results rule in `CLAUDE.md` / `docs/AUTONOMY.md`). Stage
gates (e.g. a competence calibration that must pass before representation
testing) are themselves experiments and get reports.

## Rules

- One report per experiment: `docs/experiments/<step-id>-<slug>.md`.
- A step that makes an empirical claim is **not `done` until its report is
  committed** with a result.
- The report must contain:
  1. **Question / hypothesis** — what is being tested.
  2. **Method** — design (worlds, conditions, row counts), model + pinned
     revision + quantization, seed, and the **exact reproduce command(s)**
     (script + config; every run is config-driven).
  3. **Decision rule** — pre-registered *before* running (what counts as
     positive / neutral / negative; gate thresholds), to avoid p-hacking.
  4. **Data** — the numbers / table with world-clustered bootstrap intervals;
     the `outputs/<run>` directory name, dataset hash, git hash, and config
     hash recorded.
  5. **Result** — classified **positive | neutral | negative**.
  6. **Conclusion & implication** — what we now believe; what changes; what is
     *not* claimed (behavioral evidence ≠ internal mechanism). Gate reports end
     with the recommendation handed to the human — the decision itself is made
     by the human and recorded in `docs/PROGRESS.md`.
- **Raw artifacts stay out of git** (`outputs/` is git-ignored and
  regenerable). Commit only the report + distilled numbers.
- Pre-register before you run: `docs/prereg/phaseN.md` (hypotheses, metrics,
  gate thresholds, exclusion rules — `docs/RESEARCH_PLAN.md` §4) is committed
  and pushed before any inference it governs, and its commit is added to
  `docs/prereg/MANIFEST.md`. The report then cites it and fills in data +
  result (see `memory/lesson-preregistration-git-trail.md`).
- `outputs/<run>/report.md` is the run-local copy; the committed copy with
  distilled numbers lives here, since `outputs/` is git-ignored.

## Template

```markdown
# <Title> (<step-id>)
- status: planned | done
- result: — | positive | neutral | negative

## Question / hypothesis
## Method (+ reproduce command)
## Decision rule (pre-registered)
## Data
## Result
## Conclusion & implication
```

## Index

- `p0.4-phase0-kill-test.md` — Phase 0, 4 models: Gate-0 recommendation FAIL (negative); 8B effect reproduced.
- `p0.5-phase0-addendum.md` — current models (Qwen3.8-27B, Gemma-4-31B invariant; gpt-oss-120b 7.5 pp fail-closed on natural language); neutral, outside Gate 0.
- `p1.6-synthetic-sources.md` — 720 synthetic MCP/repo worlds, all engine-certified (data audit).
