# Instruction Indirection Gap

This repository studies whether language models become less reliable when an intended action must be derived through declarative provider-resolution links. It is a behavioral study: observed accuracy and semantic likelihood do not establish an internal mechanism.

Worlds hold the action, candidate values, and resolved terminal provider constant across matched conditions. The primary outcome is the sum-log-probability margin between semantic candidates; candidate token counts and mean token log probability are also saved. Policy comprehension and action application are separate prompts and repeated measurements from the same synthetic world.

## Controlled experiment: v2

The current primary design is indirection_v2_controlled. It varies relevant provider-resolution path depth from 1 to 5 while keeping six total relation edges in every graph. Relevant and disconnected distractor edges use the same resolves-to grammar. Distractor edges cannot connect to the queried field or either terminal provider. Source versus default terminal providers are exactly balanced within each task family, and relation statement order is randomized deterministically with relevant-edge positions recorded.

The three families are filename, literal three-symbol sequence ordering, and destination. Every world/depth has comprehension and application questions. For depths 3–5, application also has a full-path neutral-note control, a full-path explicit bridge, and a flattened graph with one relevant edge and six total links. Bridge and neutral use the same insertion slot; tokenizer preflight checks their exact token-count match before loading model weights.

The main config creates 100 worlds per family. It yields 3,000 primary path/task rows and 2,700 deep-condition control rows, 5,700 total. Bootstrap resampling clusters by world; results remain separated by task family. The experiment does not assume a monotonic depth effect.

### Dataset and tokenizer audit

Dataset only: uv run python scripts/run_indirection_v2.py --config configs/indirection_v2_pilot.yaml --dataset-only

Tokenizer audit only: uv run --extra inference python scripts/run_indirection_v2.py --config configs/indirection_v2_pilot.yaml --tokenizer-audit-only

The first command verifies graph paths, distractor disconnection, fixed link count, terminal balance, semantic matching, and bridge/neutral insertion. The second audits exact rendered prompt and candidate token lengths using the configured tokenizer, before loading model weights. It rejects candidate length mismatches, path-depth prompt spreads above the configured tolerance, bridge/neutral length differences, or flattened/path differences above tolerance.

### Controlled Qwen3-4B run

Run this on the designated model-capable machine:

    uv run --extra inference python scripts/run_indirection_v2.py --config configs/indirection_v2.yaml --model qwen3_4b --device cuda

The tokenizer audit runs before model loading. The run saves the frozen dataset and hash, graph audit, tokenizer metadata and audit, model provenance, predictions with raw candidate scores, world-level outcomes, paired metrics and bootstrap intervals, plot-source CSV/JSON, figures, run.log, run status, and pilot_report.md. A pilot config with four worlds per family is for engineering checks; it is not suitable for inference.

## Exploratory pilot: v1

indirection_v1_exploratory is retained as historical data and code. It used five hand-written constructions whose wording, statement count, and prompt length changed along with depth. Qwen3-4B performance dropped sharply around its levels 2–3, but that result does not isolate semantic path depth. Do not pool v1 and v2 or describe v1 as a causal depth effect.

Reproduce v1 with scripts/run_indirection_experiment.py and configs/indirection.yaml. Earlier scoped-authority experiments remain in src/authority_leakage/ as legacy exploratory work.

## Inference and interpretation

Continuation scoring jointly tokenizes the rendered chat prompt and candidate, records boundary behavior, and scores the continuation autoregressively. Sum log probability is primary; length-normalized scores are diagnostic.

The independent unit is the synthetic world. Depth variants, task types, and control conditions are repeated observations. Report paired world-level contrasts and family-specific results. A comprehension-correct/application-wrong pair comes from separate prompts for the same world and depth; it describes a behavioral dissociation, not a sequential internal process.

Do not add chain-of-thought prompting or run a multi-model sweep until the fixed-length depth effect, bridge-specific recovery, or comprehension/application dissociation survives v2 controls. Base/instruct differences, if studied later, describe post-training differences and do not identify a causal training mechanism.
