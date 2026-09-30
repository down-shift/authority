# Instruction Indirection Gap

We study the instruction indirection gap: whether language models become less reliable when an intended action is specified through declarative policy structure and multi-step semantic bindings rather than as a direct executable instruction. Using matched synthetic tasks, we hold the intended action constant while varying the number of semantic dereferences required to infer it. We separately measure policy comprehension and policy application using exact semantic continuation likelihoods.

This is a behavioral study and does not by itself establish an internal mechanism. Direct and indirect conditions are matched at the world level. The primary outcome is the likelihood margin between semantic candidates, not arbitrary A/B output. Policy comprehension and execution are analyzed separately. The independent unit is the synthetic world; depth and task measurements are repeated observations.

## Frozen pilot design

The initial ladder has five levels: (0) direct final action, (1) direct relational instruction, (2) explicit field-owner lookup, (3) role-mediated ownership, and (4) compositional symbolic role/field rule. Each world is generated once and rendered at all depths for filename selection, literal sequence ordering, and destination selection. Within a world, the correct action and alternative action never change. Source versus default ownership is balanced within each family.

Each world/depth has two prompts. Comprehension asks which source determines the field; application asks for the final semantic action. For each task both semantic candidates are scored, saving sum log probability, token count, and mean log probability. The primary margin is correct minus incorrect sum log probability. Positive means the correct candidate is preferred; token-length normalization remains diagnostic only. The pilot uses a single `canonical_v1` surface form and makes no claim about paraphrase robustness.

## Run

```bash
uv sync --extra inference --extra test
uv run python scripts/run_indirection_experiment.py --config configs/indirection_pilot.yaml --dataset-only
uv run python scripts/run_indirection_experiment.py --config configs/indirection_pilot.yaml --model qwen3_4b
uv run python scripts/run_indirection_experiment.py --config configs/indirection.yaml --model qwen3_4b
uv run pytest -q
```

The pilot config defaults to four worlds per family for a fast engineering check. The main config has 100 worlds per family, three families, five depths, and two measurement tasks (3,000 rows). Rows can be regenerated deterministically with the configured seed. Model checkpoint configuration is in `configs/models.yaml`; run artifacts preserve the exact dataset hash, prompt audit, candidate tokenization audit, provenance, semantic scores, per-world rows, aggregate metrics, confidence intervals, figures and plot CSV, run status, and report. A completed dataset-only run stops before model loading.

Continuation scoring tokenizes rendered prompt and candidate jointly, then scores continuation tokens autoregressively. If tokenization changes across the boundary, the first token overlapping the candidate boundary is included and the overlap is recorded; a tokenizer cannot assign a partial-token conditional probability. This makes the chosen continuation estimand explicit.

## Interpretation

For each depth, report accuracy and margin for comprehension and application, paired change from depth zero, and family-specific curves. The bootstrap resamples worlds, retaining repeated measurements together. Application failure with correct comprehension is reported separately. Curves may be null, monotonic, threshold-like, or non-monotonic; the implementation does not assume a linear or monotonic effect. Depth-zero accuracy below 0.95 produces an engineering warning. Poor direct performance means a task family needs repair before interpreting deeper conditions.

A later study may compare indirect prompts with length-matched neutral filler and flattened rules. These controls are outside the initial pilot. Multi-model comparisons should follow an interpretable single-model pilot; base/instruct contrasts describe post-training differences and do not identify a causal training mechanism.

## Retained legacy experiments

`src/authority_leakage/` and historical outputs remain available as **legacy / exploratory authority-leakage experiments**. Their authority-specific analyses are not part of the new indirection study. Generic Hugging Face inference, provenance, durable JSONL, and bootstrap utilities are reused where appropriate.
