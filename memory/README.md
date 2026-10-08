# `memory/` — team-shared Claude knowledge

This directory is **committed, PR-reviewed agent memory**: durable lessons,
constraints, and feedback that every contributor's Claude Code session should
load. It exists because per-user Claude memory (under `~/.claude/...`) is
private to one machine — when a colleague's Claude learns something important,
the rest of the team's agents never see it. Committing it here fixes that.

## Protocol

- **Read everything in `memory/*.md` at session start.** `CLAUDE.md` instructs
  agents to do this.
- **One fact per file.** Use a kebab-case name and a short type tag in the body
  (`feedback`, `lesson`, `reference`, `decision`).
- **Promote, don't hoard.** When you learn a durable, generalizable lesson in a
  session (a tokenizer trap, a quantization quirk, a reviewer-facing
  constraint), write it here and include it in your PR so the team inherits it.
  Volatile, task-local notes stay in your private `~/.claude` memory.
- **Changes go through PR review** like any other repo content — these files
  steer every agent, so they deserve scrutiny.

## Current entries

- `decision-autonomy-loop.md` — how the `/continue` loop works and its
  non-negotiable rules.
- `reference-dgx-usage.md` — compute resources (A100, RTX PCs, H100, V100),
  the claim priority, and each machine's rules (H100: compose only, files only
  under `/data/storage/kaluzhnaya_jhub/`, GPU #7 only, clean up).
- `lesson-preregistration-git-trail.md` — squash/rebase erase the
  rule-before-result evidence; merge experiment PRs with `--merge`; only say
  "pre-registered" with a timestamped commit (see `docs/prereg/`).
