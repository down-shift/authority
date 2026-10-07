# Autonomous research loop

This repo can drive itself toward the finished paper, one reviewed step at a
time, with several Claude Code agents (on the laptop and on GPU nodes) working
in parallel without colliding. This is the human-readable spec; the executable
runbook is the `/continue` slash command (`.claude/commands/continue.md`), and
the mechanics live in `scripts/`.

## The UX

```
git clone https://github.com/down-shift/authority.git && cd authority
claude                       # start Claude Code
> продолжи        (or: /continue)
```

Claude picks the next unclaimed step from `docs/PLAN.md`, implements it, opens a
PR, and **auto-merges when the gate is green**, then stops (or loops if you ask).

## Source of truth: `docs/PLAN.md`

`docs/PLAN.md` is both the roadmap and the **claim ledger**. Each step is a
block with `status / owner / claimed_at / deps / source / done-when`. The
`status/owner/claimed_at` fields mark who is working on what. **Don't hand-edit
statuses while the loop runs** — `scripts/claim.py` and `scripts/mark.py`
update them atomically.

The plan may evolve, but **changes to the step set need a human**: the loop
*proposes* new / split / re-sequenced steps and applies them only after
confirmation. The loop never self-edits the plan's structure — it only updates
step **statuses** atomically (claim / in_review / done / blocked).

## Human steps and gates

Steps whose title carries `[HUMAN]` — the phase gates G0–G3 and the open
questions in `docs/RESEARCH_PLAN.md` §8 (H0.1) — are decisions only Jerzy can
make. `claim.py` never claims them; when nothing else is eligible it prints
`waiting on human: <ids>`. The human closes one with
`python3 scripts/mark.py <id> done "<decision>"` after recording it in
`docs/PROGRESS.md`. Every phase depends on its predecessor's gate, so the loop
pauses at each gate by construction.

## Cadence & human checkpoints

`/continue` runs **exactly one step**, with a human checkpoint on each side:

1. **Announce first.** After claiming, the agent states (2–4 lines) which step
   it took and how it will satisfy the `done-when` — *before* writing code.
2. **Implement → gate → PR → auto-merge** (below).
3. **Report & ask.** The agent summarizes what merged, says what it would do
   next, and **asks whether to continue** — it does not auto-advance. Progress
   resumes only when a human says «продолжи».

## How parallel agents stay out of each other's way

Git is the arbiter (an atomic ref update wins). To claim a step, `claim.py`:

1. `git fetch origin main && git reset --hard origin/main` (sync to the truth),
2. picks the first `todo` step whose `deps` are all `done`,
3. flips it to `claimed` (with owner + timestamp), commits, and **pushes to main**,
4. if the push is **rejected** (someone else pushed first), discards the local
   claim, re-syncs, and re-picks — looping until its push lands or nothing is left.

So two agents that grab the same step at the same moment can't both win: exactly
one push lands; the other re-picks the next eligible step. The claim is visible
to everyone on the next `git pull`. Both scripts refuse to run with uncommitted
changes to tracked files, since the hard reset would discard them.

## Hardware-aware claiming

Code, tests, dataset generation (`--dataset-only`), analysis, and writing run
anywhere. Model inference needs a CUDA GPU with enough memory for the
configured model and quantization (`configs/models.yaml`), and gated models
(e.g. `google/gemma-3-*`) need a Hugging Face token. Steps that need a GPU say
so in their title/`done-when` (`[GPU]`, `[GPU: <class, memory>]`).

**Before claiming, check whether this machine can actually run the next eligible
step.** If it can't, do **not** claim it — report to the human which step is
next and where it should run. If you discover the mismatch only *after*
claiming, `scripts/mark.py <id> todo "needs <hw>, this machine lacks it"` to
release it, and stop for the human. Never sit on a claim you cannot execute.

## Merge policy

- Work happens on a per-step branch `feat/<id>-<slug>`; **never commit to
  `main`** except the tiny claim/mark ledger commits the scripts make.
- **Auto-merge requires the gate to be green:** `scripts/check.sh` runs
  `ruff check` + `ruff format --check` on the Python files changed relative to
  `origin/main` (the pre-existing code predates ruff, so it is grandfathered
  until touched) and the full `pytest -q`. The same gate runs in CI via
  `.github/workflows/gate.yml` on every PR into `main` and every push to
  `main` — the local gate stays authoritative, but PRs cannot land red.
- Code/doc PRs: `gh pr merge --squash --delete-branch`. **Experiment PRs:**
  `gh pr merge --merge --delete-branch`, so the pre-registration commit
  (decision rule before results) survives on `main`.
- **Conflicts:** bring the branch up to date with `origin/main`, resolve,
  re-gate, re-merge.
- **Blocked:** anything unresolvable (bad conflict, failing upstream, ambiguous
  spec, a failed go/no-go gate) → `scripts/mark.py <id> blocked "<reason>"` and
  **stop for a human**. The loop never forces a merge, never hacks around a
  blocker, and **never decides a research gate, model sweep, or pivot itself**.

## Prerequisites (per machine)

`scripts/bootstrap.sh` is idempotent and sets up most of it:

- **uv** — installed if missing; `uv sync --extra dev` from `uv.lock` (plus
  `--extra inference --extra quantization` on GPU nodes).
- **gh** — must be authenticated (`gh auth login`) for PR + auto-merge.
  Bootstrap warns if not.
- **Hugging Face token** — for gated models on GPU nodes (`HF_TOKEN` or
  `huggingface-cli login`). Bootstrap warns if missing.

## Safety / guardrails

- One claimed step per agent at a time; stay within the step's `done-when` scope.
- Never force-push `main`; never commit secrets or heavy artifacts (`outputs/`
  is git-ignored; `.env` is deny-listed in `.claude/settings.json`).
- Every code change ships tests; the gate is mandatory before any merge.
- Respect the hard invariants in `CLAUDE.md` and `memory/`.

## Scripts

| Script | Does |
|---|---|
| `scripts/bootstrap.sh` | idempotent env setup; warns on missing prereqs |
| `scripts/agent_id.sh` | stable claim owner id (`name@host`) |
| `scripts/check.sh` | the gate: ruff on changed files + pytest |
| `scripts/claim.py` | atomically claim the next eligible step (race-safe); `--dry-run` peeks |
| `scripts/mark.py` | set a step's status on main (`in_review` / `done` / `blocked` / `todo`) |
| `.github/workflows/gate.yml` | runs `scripts/check.sh` on every PR / push to main |

## Results rule

Experiments commit their findings. Any experiment / gate / validation step
writes a report under `docs/experiments/<id>-<slug>.md` and commits it —
**negative and failed results included** (a fail is a finding, never silently
dropped; a failed calibration gate is *especially* a finding). Each report
states the question, method + exact reproduce command, a **pre-registered
decision rule**, the data, a `positive | neutral | negative` result, and the
conclusion. Run directories stay in git-ignored `outputs/`; only the report +
distilled numbers land in git. An empirical-claim step is not `done` until its
report exists. Each phase is preregistered in `docs/prereg/phaseN.md`,
committed before inference and recorded in `docs/prereg/MANIFEST.md`. Full
convention: `docs/experiments/README.md`.

`docs/PROGRESS.md` is the complementary **daily log**: one line per day +
every gate decision.
