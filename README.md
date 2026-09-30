# Authorization Representation Invariance

This repository's current experiment asks whether semantically equivalent authorization policies produce equivalent model behavior. The canonical policy object is created first, then rendered as a natural-language permission statement, decision-owner statement, permission table, JSON policy, or executable rule. Each rendering is independently decoded and checked against the canonical semantics before prompts are scored.

The pilot uses deterministic matched worlds across filename, ordering, and destination actions. Authorized and unauthorized scopes are balanced. It asks policy interpretation (who determines a field?) and policy application (which supplied value is selected?) separately, with direct semantic candidate sum-log-probability margins as the primary outcome. It measures accuracy and margins by representation, within-world disagreement, pairwise representation effects, and interpretation/application disagreement. Disagreement is an outcome, not a manipulation failure.

The two-scope condition combines two independently specified fields and queries each scope separately. Its matched single-scope condition isolates the additional policy and context. A mitigation arm asks the model to map each raw representation to one of the legal canonical policies, selected by semantic candidate likelihood, and then answers from that policy. The converter is constrained over two or four legal policies, depending on the number of scopes; it never receives the correct authorization values as a hint. Conversion accuracy is reported separately, so answer changes can be interpreted alongside canonicalization quality. No LLM judge or model sweep is used.

## Run the Qwen3-4B pilot

The checked-in pilot uses 12 paired worlds (four authorization combinations for each of three action pairs), 480 raw prompts, and 960 answer rows across raw and canonicalized arms. This is a small descriptive pilot; world is the bootstrap unit. Use the repository's configured Qwen3-4B revision and a machine with enough model memory:

```sh
uv run --extra inference python scripts/run_authorization_invariance.py \
  --config configs/authorization_invariance_pilot.yaml \
  --model qwen3_4b --device cuda
```

The runner writes a timestamped directory under `outputs/` with `config.json`, canonical `worlds.jsonl`, rendered `dataset.jsonl`, representative prompts, semantic validation and dataset hash, `metadata.json` with model and source provenance, raw and canonicalization candidate scores, `predictions.jsonl`, world-cluster bootstrap metrics, plot data and figures, `pilot_report.md`, run status, and an artifact SHA-256 manifest. `--resume RUN_DIRECTORY` resumes a partial run only when config, data, model, and source provenance still match. `--dataset-only` generates and validates the frozen dataset without loading a model.

The report gives interpretation and application accuracy/margins by representation and scope count, disagreement and stability rankings, pairwise effects, comprehension/application dissociations, two-scope changes, and canonicalization recovery. It records the exact invocation. Representation prompts are fixed in the renderer and are not tuned against model outcomes.

## Legacy experiments

The prior scoped-authority and instruction-indirection experiments, their design notes, and historical findings are preserved in [docs/legacy_experiments.md](docs/legacy_experiments.md). Their code, configurations, tests, and outputs remain available and are not pooled with this experiment.
