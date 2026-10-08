# Lesson: never let a pipe or an unprotected branch hide a red gate

**Type:** lesson. **Audience:** every agent merging PRs here.

On 2026-10-08 a PR (#35) merged with a failing gate. Two things combined:

- `bash scripts/check.sh | tail -1 && …` takes **`tail`'s** exit status, not
  the gate's, so the chain carried on after `Found 3 errors.`
- `main` has **no branch protection**, so `gh pr merge` merges a PR whose CI
  failed, and `gh pr checks --watch` only waits; it doesn't block.

**How to apply:**
- Run the gate as `bash scripts/check.sh > log 2>&1; [ $? -eq 0 ]`, or with
  `set -o pipefail`. Never put `| tail` before an `&&`.
- Merge only after reading the CI conclusion:
  `gh pr view N --json statusCheckRollup -q '.statusCheckRollup[0].conclusion'`
  must be `SUCCESS`.
- Mark a step done only after `gh pr view N --json state` returns `MERGED`.
- Turning on branch protection for `main` (required `gate` check) would make
  this structural. That's a repo-settings decision for the human.
