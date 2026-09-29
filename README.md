# Authority Is Not Trust

This repository measures **behavioral authority influence** in controlled synthetic tasks. Its primary estimand is the matched change caused by granting authority to a source over dimension `i` on the source's influence over dimension `j`:

`Λ[i→j] = Influence_j(authority_i) − Influence_j(no_authority)`

The diagonal measures intended authority responsiveness. Off diagonal cells measure incremental authority leakage. Raw source adoption without authority is reported separately; it is not leakage by itself because models may already follow an unauthorized proposal at a high rate. These behavioral results do not establish how a model internally represents authority. A broad source-global influence heuristic is one behaviorally consistent interpretation, not a mechanistic conclusion.

## Primary experiments

`epistemic` tests deontic `output_format` authority → influence on a factual claim. Every world is rendered with `NO_AUTHORITY`, `AUTHORITY_I` (format only), and an epistemic `AUTHORITY_J` positive control. Evidence-only, no-evidence, and explicit-denial controls diagnose competence and source following separately. Source claim, evidence, candidate values, query, and ordering remain fixed across the three primary authority variants. The source-claim log-probability margin is `log P(claim) − log P(other)`; the primary leakage is its paired change under format authority versus no authority. Claims are balanced true/false, candidate order is counterbalanced, and neutral templates are crossed. HF epistemic generation is constrained to the two candidate continuations; unconstrained candidate log-probability scoring remains the inferential outcome.

`instrument_validation` is the required calibration stage for the one-scope `filename` task. It compares preregistered A/B/C authority representations and a D direct-control anchor across matched NO/YES conditions. The default config uses 60 shared worlds and separate authority-comprehension and behavioral-choice tasks. Selection uses only comprehension and intended filename adoption; it does not calculate leakage. Do not run the scope matrix unless an eligible instrument passes every fixed calibration gate. The later `scope` matrix uses `output_format`, `ordering`, `filename`, `tool_choice`, and `numeric_answer`, matched across no authority, each individual grant, and full authority. Its cell outcome is source adoption on the target dimension; each matrix cell is the paired adoption difference from no authority. Parse success, compliance, adoption, and joint valid-and-correct rates are distinct.

The output contract and parser share one canonical JSON key schema. Malformed output is counted as parse failure and excluded from conditional compliance/adoption denominators, while joint valid-and-correct is reported separately. Epistemic parse failures are excluded from accuracy/following denominators and representative unchanged raw responses are included in the metrics for diagnosis. There is no LLM-as-judge scoring.

Natural system/developer/user role comparisons are secondary ecological/generalization experiments. They do not define the primary causal effect because role swaps also change sequence position and related factors. Base-versus-instruction-tuned comparisons are called post-training differences; they do not isolate hierarchy training.

## Repository and retained legacy code

`src/authority_leakage/clean.py` contains deterministic synthetic world generation, authority rendering, matching validation, and content hashing. Existing role-swap and earlier delegation generators remain available as legacy code for reading old artifacts; legacy raw unauthorized-follow rates are explicitly labeled as raw rates, not leakage. The new configs and primary documented design use `matched_authority_v1`. Existing HF inference, full continuation log-prob scoring, durable JSONL output, provenance capture, and plotting infrastructure are retained.

Generated records share `pair_id`/`world_id`; variants are never independently regenerated. Dataset generation is deterministic from seed. `validate_matching` checks answer, world fields, and that rendered prompts differ only in authority declaration. The primary unit is the independently generated world, and analysis pairs variants by that ID.

## Install and run

This repository uses `uv` to select Python, resolve dependencies, run scripts, and run tests. Install `uv` on the machine where you will work, then create and commit the lockfile once with:

```bash
uv lock
uv sync --locked --extra inference --extra test
uv run --locked python scripts/generate_dataset.py --config configs/epistemic.yaml --output outputs/epistemic-dataset.jsonl
uv run --locked python scripts/generate_dataset.py --config configs/scope_instrument_pilot.yaml --output outputs/scope-instrument-pilot.jsonl
uv run --locked python scripts/audit_scope_instrument.py outputs/scope-instrument-pilot.jsonl --worlds 5 --output outputs/scope-instrument-audit.md
uv run --locked --extra test pytest -q
```

The project supports Python 3.10 and newer. `uv.lock` pins the resolved dependency graph; regenerate it deliberately with `uv lock` after changing `pyproject.toml`.

Clean epistemic pilot:

```bash
uv run --locked --extra inference python scripts/run_experiment.py --experiment epistemic --model qwen3_4b --config configs/epistemic.yaml
```

Authority-instrument validation (required before any scope matrix):

```bash
uv run --locked --extra inference python scripts/run_instrument_validation.py --model qwen3_4b --config configs/instrument_validation.yaml
```

The validation config uses 60 independent worlds shared across the four representations. A/B/C are eligible for selection; D is an anchor only. If no instrument passes the fixed gates, stop and do not run the 2×2 or five-scope matrix. `--model` accepts a `configs/models.yaml` key or a Hugging Face model ID/path. Model and tokenizer revisions are pinned through that config when supplied. Greedy decoding settings, software versions, checkpoint commits, tokenizer identity, seed, git commit, dataset hash, raw outputs, per-example scores, metrics, confidence intervals, validation information, prompt diffs, selection record, and a human-readable report are saved under each run directory.

Each run stores `config.yaml`, `metadata.json`, `dataset.jsonl`, `predictions.jsonl`, `metrics.json`, `pilot_report.txt`, run status, and figures. `raw_response` is preserved unchanged. The output JSON and predictions are machine readable.

## Pilot diagnostics and interpretation

Pilot reports distinguish no-authority adoption from baseline-corrected leakage and provide world-paired estimates with bootstrap intervals. Engineering warnings flag parse rate below 98%, unauthorized baseline adoption near ceiling, weak authorized responsiveness, weak epistemic positive controls, and low evidence-only accuracy where available. Approximate 0.8 compliance/adoption and 0.9 evidence-accuracy values are diagnostics, not inferential acceptance thresholds. Do not scale until the prompts, parser, matching checks, and dynamic range have been inspected.

The framework supports multiple Hugging Face IDs/revisions and model families. Inference and interpretation should stay modest: results describe observed behavior under these synthetic tasks, not a model's internal representation or a causal effect of a particular post-training procedure.
