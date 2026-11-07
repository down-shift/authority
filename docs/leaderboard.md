# authinv leaderboard: Phase-2 reference rows

The worst-case table for the paper's 12 Phase-2 configurations: T2 application accuracy (%) on the
frozen benchmark v2, with prompt-v1 and the `json-strict-v3` parser. Rows are sorted by worst-case
accuracy. The gap is best minus worst rendering, with a world-clustered 95% CI. Each model is shown
at its pinned revision. Regenerate the table with

```sh
python scripts/authinv_leaderboard.py table --phase2 configs/authinv/phase2_figures.yaml [--runs <leaderboard runs>]
```

New models are added with `scripts/authinv_leaderboard.py run --model <HF id> --data <release dir>`,
under the same settings.

| model | worst-case acc. | worst rendering | gap (95% CI) | nl_statement | owner_statement | table | json_policy | executable | rego | deny→allow (worst) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B thinking@1d4bf0f | 99.1 | nl_statement | 0.7 (0.5–1.0) | 99.1 | 99.8 | 99.7 | 99.8 | 99.7 | 99.8 | 1.3 |
| gpt-oss-120b@b5c939d | 98.7 | nl_statement | 1.2 (1.0–1.5) | 98.7 | 99.8 | 99.9 | 99.8 | 99.9 | 99.9 | 2.5 |
| gpt-oss-20b@6cee5e8 | 97.7 | nl_statement | 2.2 (1.9–2.6) | 97.7 | 99.3 | 99.8 | 99.6 | 99.9 | 99.9 | 4.5 |
| Gemma-4-31B@842da37 | 96.7 | nl_statement | 2.4 (2.0–2.8) | 96.7 | 98.4 | 98.6 | 99.1 | 98.9 | 98.7 | 4.8 |
| Gemma-4-12B@707f0a3 | 85.1 | nl_statement | 6.1 (5.3–6.8) | 85.1 | 89.4 | 85.9 | 90.7 | 89.5 | 91.2 | 16.5 |
| Qwen3.8-27B-FP8@017b9c7 | 79.6 | rego | 15.4 (14.6–16.3) | 91.0 | 95.1 | 92.5 | 93.0 | 94.4 | 79.6 | 4.9 |
| Llama-3.3-70B W8A8@b1e1d82 | 78.7 | json_policy | 5.7 (5.0–6.4) | 81.0 | 84.4 | 80.5 | 78.7 | 81.7 | 79.5 | 10.2 |
| Gemma-4-E4B@ee0ef60 | 75.2 | table | 4.3 (3.6–5.0) | 75.3 | 79.0 | 75.2 | 75.3 | 79.5 | 76.5 | 25.9 |
| Qwen3.5-9B@c202236 | 72.4 | rego | 5.8 (5.2–6.4) | 77.2 | 77.7 | 75.8 | 75.1 | 78.2 | 72.4 | 5.3 |
| Ministral-3-14B@29439f8 | 66.8 | rego | 7.3 (6.6–8.0) | 74.0 | 74.1 | 70.9 | 69.8 | 70.5 | 66.8 | 3.1 |
| Qwen3.5-4B@851bf6e | 65.8 | rego | 5.4 (4.6–6.2) | 70.1 | 71.3 | 70.7 | 67.1 | 69.6 | 65.8 | 4.8 |
| Qwen3.8-27B@1d4bf0f | 49.5 | rego | 45.7 (44.2–47.3) | 91.1 | 95.3 | 93.1 | 93.6 | 94.7 | 49.5 | 1.6 |

Caveats, from `docs/experiments/p2.9-phase2-report.md`:
- Qwen3.8-27B with thinking off loses mostly on Rego answers cut off at the 256-token limit. Those
  are parse failures, a format effect; its T1 (log-prob) gap is 4.8 pp.
- Non-answers (refusal, abstention, parse failure) count as not correct; the run files report each
  one separately.
