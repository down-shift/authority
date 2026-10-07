# Lesson: keep the pre-registration trail visible in git

**Type:** lesson. **Audience:** every Claude in this repo.

Reviewers ask "where is the pre-registration?". The answer is the git trail:
the report with its decision rule is committed (status `planned`) **before**
the run, and every run directory records the code's git hash. The trail is
fragile in three ways:

- **Squash-merging erases it.** A squash leaves only the combined commit on
  `main`; the rule-before-result commit survives only in GitHub PR history.
  **Merge experiment PRs with `gh pr merge --merge`** (not `--squash`).
- **Rebasing resets committer dates.** Author dates and file contents survive.
  If you rebase, cite author dates and record the pre→post-rebase SHA mapping in
  the report; state that the runner script is byte-identical to the hash the
  run recorded.
- **Only call it pre-registered if a timestamped commit exists.** Say
  "pre-registered" only when a commit with the rule precedes the first result.
  `docs/prereg/MANIFEST.md` is the record, and the paper's appendix mirrors it.
  Extend both whenever a new pre-registered result enters the paper.

**How to apply:** for experiment steps, commit and push the planned report
first, then run, then commit results as a separate commit; merge with
`--merge`; add a MANIFEST row.

(Inherited from the latent-underspecification-research project, where an
AAAI-27 reviewer raised exactly this question.)
