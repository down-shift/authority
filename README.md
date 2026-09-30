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

## Final design validation: v3 factorial

`indirection_v3_factorial` independently crosses relevant provider path depth (1–5), disconnected relation count (0, 2, 4, 6), and statement layout (compact, dispersed), with identical graph semantics across layouts. Each matched semantic world supplies filename, ordering, and destination values, separate comprehension and application queries, and a direct-action application reference. Tokenizer-aware neutral notes control prompt length; relevant edge statement and token positions, span, mean gap, and relation-block length are recorded. Deep bridge, neutral, and flattened controls are targeted to depths 3–5 at four disconnected edges and dispersed layout.

The 30-world-per-family pilot contains 7,200 primary factorial rows (30 worlds × 3 families × 5 depths × 4 distractor counts × 2 layouts × 2 queries), plus 900 direct and targeted control rows. Each row scores two semantic candidates, for 16,200 candidate scores. Generate the dataset with `python3 scripts/run_indirection_v3.py --config configs/indirection_v3_pilot.yaml --dataset-only`; run the five-world tokenizer smoke audit with `--smoke`; audit the full pilot tokenizer design with `--tokenizer-audit-only`. The runner prints workload counts before tokenizer/model work and gates inference on exact matched prompt lengths and design independence.

The current 30-world tokenizer audit passes, with exact within-world/task prompt counts, exact bridge/neutral/flattened control counts, and zero observed correlations of depth or distractor count with prompt token count. Five real Qwen3-4B scoring prompts averaged 88 seconds on CPU, and no CUDA or MPS device is available here. The full pilot was not launched because its estimated CPU runtime is about 8–11 days; its behavioral questions remain unanswered.

V3 is the final single-model design validation. Do not begin a multi-model sweep until depth effects survive distractor and layout controls, direct application remains near ceiling, and bridge/flattened controls remain coherent. A vanished depth effect is valid evidence that v2's effect depended on correlated structure.

## Exploratory pilot: v1

indirection_v1_exploratory is retained as historical data and code. It used five hand-written constructions whose wording, statement count, and prompt length changed along with depth. Qwen3-4B performance dropped sharply around its levels 2–3, but that result does not isolate semantic path depth. Do not pool v1 and v2 or describe v1 as a causal depth effect.

Reproduce v1 with scripts/run_indirection_experiment.py and configs/indirection.yaml. Earlier scoped-authority experiments remain in src/authority_leakage/ as legacy exploratory work.

## Inference and interpretation

Continuation scoring jointly tokenizes the rendered chat prompt and candidate, records boundary behavior, and scores the continuation autoregressively. Sum log probability is primary; length-normalized scores are diagnostic.

The independent unit is the synthetic world. Depth variants, task types, and control conditions are repeated observations. Report paired world-level contrasts and family-specific results. A comprehension-correct/application-wrong pair comes from separate prompts for the same world and depth; it describes a behavioral dissociation, not a sequential internal process.

Do not add chain-of-thought prompting or run a multi-model sweep until the fixed-length depth effect, bridge-specific recovery, or comprehension/application dissociation survives v2 controls. Base/instruct differences, if studied later, describe post-training differences and do not identify a causal training mechanism.
