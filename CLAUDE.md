# CLAUDE.md — working agreement for Claude Code in this repo

This file is auto-loaded into every Claude Code session. It is the single
source of truth for how agents (and humans) work here. **Read it fully before
changing anything.** The research spec — thesis, phases, gates, decisions — is
`docs/RESEARCH_PLAN.md`; positioning against prior work is
`docs/novelty_check.md`. The step ledger is `docs/PLAN.md`; team-shared lessons
live under `memory/` (read all of it at session start); the daily log and gate
decisions live in `docs/PROGRESS.md`.

## What this project is (north star)

**Representation invariance of authorization policies for LLM agents**
(`docs/RESEARCH_PLAN.md` §0). The same authorization policy rendered as a
natural-language statement, a decision-owner statement, a permission table, a
JSON policy, or executable rules (Cedar; Rego optional) — renderings an
authorization engine certifies equivalent — yields different model decisions
and agent actions. We measure **worst-case accuracy over renderings**,
**worst-case unauthorized-action rate** in an agent sandbox, and **deny→allow
flips**, then test policy canonicalization to an IR and format-diverse LoRA SFT
as mitigations judged by closure of the worst-case gap.

Target: NeurIPS 2027 main track (~May 2027). Open-weight models only on the
local H100/V100 cluster; $0 API spend. New work lives in `src/authinv/` with
configs in `configs/authinv/`. The pre-existing packages (`authority_leakage`,
`authorization_*`, `indirection*`) are the pilot that motivated the plan:
copy and adapt from them, **never import them** from `authinv`. The legacy
indirection-depth line is out of scope — don't touch it.

## Hard invariants — do NOT break these

These protect the scientific validity of the results. A change that violates
one is wrong even if it runs. If a task seems to require breaking one, **stop
and raise it** instead of working around it.

1. **The thesis, `[DECISION]`s, and pre-registered rules are frozen**
   (`docs/RESEARCH_PLAN.md`). Do not drift a question, a gate threshold, or a
   primary metric after seeing data. `[ASK JERZY]` items wait for Jerzy.
2. **Gates are human decisions.** G0–G3 are `[HUMAN]` steps; a phase starts only
   after its gate is recorded as passed or waived. A runner may *refuse* to
   start when a prerequisite gate failed, but never launches the next phase,
   a sweep, or the Gate-0 pivot on its own. The loop reports; a human decides.
3. **Every experiment = one script + one config.** No notebook-only results.
   The seed lives in the config; world generation is deterministic.
4. **Equivalence is engine-certified; renderers are frozen.** A world ships only
   if every rendering decodes to the canonical policy and the engine agrees on
   every request. **Never edit a renderer or prompt after seeing model
   results** — bump its version, rerun everything, log it in
   `docs/CHANGELOG.md`.
5. **Freeze, validate, hash before inference.** Datasets are generated and
   semantically validated (and tokenizer-audited where the design needs it)
   *before* any model is loaded; the dataset hash is recorded. `--resume`
   only proceeds when config, data, model, and source provenance match.
6. **Provenance on every run:** model name + pinned revision + quantization,
   git hash, config, dataset hash, invocation, and an artifact SHA-256
   manifest, under a timestamped `outputs/` run directory. Generation runs
   also record vLLM version and GPU type; temperature 0, fixed seeds.
7. **World is the statistical unit.** Bootstrap intervals cluster by world;
   paired contrasts are within-world. Do not pool across models, experiments,
   or task families unless the plan says so; never pool renderings without
   per-rendering values; ≥2,000 bootstrap resamples; Holm-corrected p-values
   across models × renderings, effect sizes first.
8. **Tiers:** T1 log-prob margins, T2 structured generation (**headline
   metrics**), T3 agentic execution (UIR). Parse failures, refusals, and
   abstains are their own category — never counted as wrong or dropped. No LLM
   judge unless the plan adds one.
9. **Behavioral claims only.** Accuracy and likelihood margins do not
   establish an internal mechanism — reports and the paper must say so.
10. **Never commit secrets or heavy artifacts.** Tokens live only in `.env` /
   the HF cache (git-ignored, read-denied to agents). `outputs/` is
   git-ignored; distilled numbers go in `docs/experiments/` reports.
11. **Negative results are committed, never dropped.** See the Results rule.

## Repository map

```
README.md                # how to run — never results
docs/RESEARCH_PLAN.md    # the frozen research spec (read before research code)
docs/novelty_check.md    # prior-work scan the plan positions against
docs/related_work.md     # verified related work (step R0.1)
docs/threat_model.md     # attacker model + claim→evidence map (step W0.1)
docs/CHANGELOG.md        # renderer / prompt version changes
docs/PROGRESS.md         # phase log: one line per day + every gate decision
docs/PLAN.md             # step ledger driving the autonomous loop
docs/AUTONOMY.md         # loop protocol; .claude/commands/continue.md is the runbook
docs/experiments/        # committed experiment reports (incl. negatives)
docs/prereg/             # phaseN.md committed before inference + MANIFEST.md
memory/                  # team-shared agent knowledge (committed, PR-reviewed)
configs/                 # models.yaml + one YAML per experiment
src/authinv/             # the paper's code: policy/ render/ equivalence/ eval/ agent/
src/                     # pilot packages (authority_leakage, authorization_*, indirection*) — don't import
configs/authinv/         # authinv configs + pinned models.yaml
scripts/                 # loop mechanics + thin CLI runners, one per experiment
tests/                   # pytest; heavy deps via pytest.importorskip
outputs/                 # git-ignored timestamped run directories
paper/                   # ACL LaTeX (main.tex + sections/*.tex)
```

## Code style

- Python ≥ 3.10; match the surrounding module's idiom.
- **Ruff** (config in `pyproject.toml`) is enforced on every Python file you
  add or change: `uv run --extra dev ruff check <files>` and
  `uv run --extra dev ruff format <files>`. Pre-existing untouched files are
  exempt; don't mass-reformat them in an unrelated PR.
- Heavy deps (`torch`, `transformers`, `bitsandbytes`) live in the
  `inference` / `quantization` extras and are imported lazily, so the gate and
  CI never need them; tests that need them use `pytest.importorskip`.
- Keep dependencies lean — no new heavy dependency without discussion in the PR.

## Collaboration workflow (multiple agents, one repo)

- **One step → one short-lived branch** off the latest `main`
  (`feat/… | fix/… | docs/… | chore/…`). **Never commit to `main`** except the
  claim/mark ledger commits made by the scripts.
- `git pull --rebase origin main` before starting and before pushing.
- **Keep PRs small and single-purpose** so parallel agents don't collide.
- **Before every commit, the local gate is mandatory:** `bash scripts/check.sh`
  (ruff on changed files + `pytest -q`). CI mirrors it.
- Imperative commit subjects; include the Claude `Co-Authored-By:` trailer on
  agent-made commits.
- Update `docs/PROGRESS.md`, run instructions in `README.md`, and any touched
  docs in the **same** PR. Results never go in the README.

## Shared memory protocol

- `memory/*.md` is **team-shared, committed, PR-reviewed** agent knowledge.
  **Read it at session start.** One fact per file, kebab-case names, a short
  type tag (`feedback`, `lesson`, `reference`, `decision`).
- When you learn a durable, generalizable lesson (a tokenizer trap, a
  quantization quirk, a reviewer-facing constraint), **promote it to
  `memory/` via a PR** so every teammate's agent inherits it. Per-user
  `~/.claude` memory is private scratch, not a substitute.

## Running & verifying

```bash
bash scripts/bootstrap.sh              # uv sync (+ inference extras on GPU nodes)
bash scripts/check.sh                  # the gate
uv run python scripts/<runner>.py --config configs/<exp>.yaml --dataset-only   # no model
```

Inference runs are GPU-bound — run them deliberately on a model-capable
machine (see `docs/AUTONOMY.md` § Hardware-aware claiming), never as casual
checks.

## Autonomous research loop

This repo advances via a self-driving loop. The plan and claim ledger is
`docs/PLAN.md`; the runbook is the `/continue` slash command; the full protocol
is `docs/AUTONOMY.md`. To make progress, run `/continue` (or say «продолжи»):
Claude claims the next eligible step, implements its `done-when`, runs the
gate, opens a PR, and **auto-merges when green**, then marks the step done.

Conventions that keep parallel agents safe:
- **Never hand-edit step statuses** in `docs/PLAN.md` — `scripts/claim.py` and
  `scripts/mark.py` update them atomically (git push is the lock; the loser
  re-picks).
- One claimed step per agent; stay within its scope. **Plan changes (new /
  split / re-sequenced steps) need human confirmation** — propose, don't
  self-edit.
- **`[HUMAN]` steps** (gates, open questions) are never claimed; `claim.py`
  skips them and reports "waiting on human".
- **Hardware-aware claiming:** don't claim `[GPU…]` steps on a machine without
  a suitable GPU; release a mis-claim with `mark.py <id> todo "<reason>"`.
- **Cadence:** one step per `/continue`; announce before you start; then report
  and **ask before continuing** (no auto-advance).
- **Blocked > forced:** unresolvable conflicts, ambiguity, or a failed research
  gate → `scripts/mark.py <id> blocked` and stop for a human.

## Results & experiments

Empirical results are committed, never discarded. Any step that runs an
experiment / gate / validation is preregistered in `docs/prereg/phaseN.md`
(committed before inference) and writes a report to
`docs/experiments/<id>-<slug>.md` and commits it — **including negative or
failed results** (a fail is a finding; silently dropping it is forbidden). The
report carries: question/hypothesis, method + exact reproduce command, a
**pre-registered decision rule**, the data, a result classified
**positive / neutral / negative**, and the conclusion. Raw artifacts stay in
git-ignored `outputs/` — commit only the report + distilled numbers. A step
that makes an empirical claim is **not `done` until its report is committed**.
See `docs/experiments/README.md`.
