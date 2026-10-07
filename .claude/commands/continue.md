---
description: Pick up the next unclaimed step in docs/PLAN.md and drive it to a merged PR (the autonomous research loop).
---

You are advancing this project autonomously. Follow this loop **exactly**.
Read `CLAUDE.md`, `docs/RESEARCH_PLAN.md`, `docs/AUTONOMY.md`, `docs/PLAN.md`,
`docs/PROGRESS.md`, `README.md`, and every `memory/*.md` first — the hard invariants there (frozen questions and
decision rules, human-decided gates, freeze-validate-hash before inference,
run provenance, world-clustered statistics) are non-negotiable.

**Cadence (strict):** do **exactly one** step per `/continue`. Announce your plan
*before* you start, and at the end report what you did, say what you'd do next,
and **ask the human before continuing** — never auto-advance to the next step.
Stop and report immediately if you hit a blocker.

**Plan changes need a human.** If the work suggests new, split, or
re-sequenced steps, **propose it and wait for confirmation** before editing the
step set in `docs/PLAN.md`. Atomic status updates via `claim.py`/`mark.py` are
not plan changes. **Research gates, model sweeps, and pivots are always human
decisions** — the loop reports the numbers, it never decides. Steps tagged
`[HUMAN]` (gates G0–G3, open questions) are never claimed; if `claim.py`
reports "waiting on human", relay which decision is pending and stop.

### 0. Sync & bootstrap
```
export PATH="$HOME/.local/bin:$PATH"
git switch main && git pull --rebase origin main
bash scripts/bootstrap.sh        # idempotent: uv sync; warns on missing gh / GPU / HF token
```
If `bootstrap.sh` warns that `gh` is unauthenticated, stop and ask the human to
run `gh auth login` (the loop needs it to open/merge PRs).

### 1. Claim the next step (race-safe, hardware-aware)
First peek at the next eligible step (`python3 scripts/claim.py --dry-run`). If
its title/`done-when` needs hardware this machine lacks (`[GPU…]` without a
suitable GPU, a gated model without an HF token), **do not claim** — report
which step is next and where it should run, then stop. Otherwise:
```
python3 scripts/claim.py
```
- Prints `CLAIMED <id>` + the step block, or `NONE` (then stop — nothing eligible).
- `claim.py` already handles the parallel-agent race: it commits the claim to
  `main` and pushes; if another agent won the race it re-syncs and re-picks.
  Trust its output — the step it printed is yours.
- If you claimed something you can't actually run, release it:
  `python3 scripts/mark.py <id> todo "needs <hw>; this machine lacks it"`
  and stop for the human.
- **Then announce** (before any code): post 2–4 lines — the claimed step id and
  how you'll satisfy its `done-when`.

### 2. Branch & implement
```
git switch -c feat/<id>-<short-slug>
```
- Implement **exactly** the step's `done-when`, nothing more. Keep the diff
  small and in scope. Match the codebase style in `CLAUDE.md`.
- **Add or extend tests** under `tests/` for any code change.
- Never violate the hard invariants: frozen questions / decision rules; one
  script + one config with the seed in the config; datasets frozen, validated,
  and hashed before any model loads; pinned model revision + full provenance
  per run; world-clustered bootstrap, no cross-model pooling; behavioral
  claims only; no secrets or `outputs/` artifacts in git.
- **Preregistration steps** (`docs/prereg/phaseN.md`) are committed and
  pushed to `main` **before** any inference they govern; add the commit to
  `docs/prereg/MANIFEST.md`.
- **Experiment steps:** run only against a committed prereg; write
  `docs/experiments/<id>-<slug>.md` with data + result (positive / neutral /
  negative — a negative is a finding, not a failure to hide); gate reports end
  with a recommendation, never a decision. Update `docs/PROGRESS.md` (one line
  for today + any gate result) in the same PR. Results never go in the README.

### 3. Gate
```
bash scripts/check.sh            # ruff (changed files) + pytest -q
```
Fix until green. Run `uv run --extra dev ruff format <your changed files>`
before committing so the format gate stays green.

### 4. Commit, PR, auto-merge
```
git add -A && git commit -m "<imperative subject>"   # + Claude Co-Authored-By trailer
git push -u origin feat/<id>-<short-slug>
gh pr create --fill --base main
python3 scripts/mark.py <id> "in_review (#<pr-number>)"
gh pr merge --squash --delete-branch <pr-number>    # experiment PRs: --merge (keeps the prereg commit)
```
- **Auto-merge only when the gate is green** (local and CI).
- **If the merge reports a conflict:** `git fetch origin main && git rebase origin/main`,
  resolve the conflict, re-run the gate, `git push --force-with-lease`, retry the merge.
  On experiment branches prefer merging `origin/main` in over rebasing, so the
  pre-registration commit's date survives.
- **If genuinely stuck** (unresolvable conflict, failing upstream, ambiguous
  spec, a research gate that didn't pass):
  `python3 scripts/mark.py <id> blocked "<one-line reason>"` and **stop,
  reporting to the human**. Do not force or hack around a blocker.

### 5. Close out
```
git switch main && git pull --rebase origin main
python3 scripts/mark.py <id> done
```
Optionally propose refinements to `docs/PLAN.md` (new/split/re-sequenced steps)
— but apply them only after human confirmation, using the same commit-and-push
to `main` so concurrent agents stay consistent.

### 6. Report & ask
Summarize what merged (step id, PR #, what changed, key numbers with intervals
if it was an experiment), state what you'd do next (the next eligible step and
which machine it needs), and **ask whether to continue. Do not auto-proceed**
to the next step — wait for the human's «продолжи».
