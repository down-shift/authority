# Decision: the autonomous /continue research loop

**Type:** decision. **Audience:** every Claude in this repo.

The project advances via a single mechanism: **`docs/PLAN.md` (the living plan +
claim ledger) driven by the `/continue` runbook** (`.claude/commands/continue.md`)
and the helpers in `scripts/`. Full spec: `docs/AUTONOMY.md`.

**Key rules to remember:**
- Saying «продолжи» / `/continue` = claim the next eligible step, implement its
  `done-when`, gate (`scripts/check.sh`), open a PR, **auto-merge only when
  green**, then `mark.py <id> done`. **Exactly one step per invocation**, with
  human checkpoints: **announce the plan before starting**, and at the end
  **report + ask before continuing** (never auto-advance).
- **Never hand-edit step statuses** while working — `scripts/claim.py` /
  `scripts/mark.py` do it atomically (git push is the lock; losers re-pick).
  This is what makes parallel agents safe.
- **`[HUMAN]` steps** (gates G0–G3, open question H0.1) are never claimed;
  `claim.py` skips them and prints "waiting on human". Relay and stop.
- **Hardware-aware claiming:** steps tagged `[GPU…]` are only claimed on
  machines that can run them. Check with `claim.py --dry-run` first; release a
  mis-claim with `mark.py <id> todo "<reason>"`. A stuck claim is schedule damage.
- **Plan changes need a human.** Adding/splitting/re-sequencing steps must be
  *proposed and confirmed* before editing `docs/PLAN.md`'s step set; the loop
  only updates statuses automatically.
- **Blocked > forced.** Unresolvable conflict / failing upstream / ambiguous
  spec / a failed research gate → `mark.py <id> blocked "<reason>"` and stop
  for a human. Never force a merge. Stage gates, model sweeps, and pivots are
  **always** human decisions.

**How to apply:** when the user wants progress, run `/continue`. When you finish
a step, leave `docs/PLAN.md` and `docs/PROGRESS.md` accurate so the next agent (or
you) resumes cleanly. See also [[lesson-preregistration-git-trail]].
