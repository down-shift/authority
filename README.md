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

## Competence-gated authorization study

The earlier 12-world representation pilot is exploratory. Its interpretation responses showed a fixed Source S preference and its application choices showed a first-listed-value preference, so those disagreements do not establish competent authorization reasoning.

The next design uses neutral, balanced actors and a single canonical JSON calibration over 120 frozen worlds. It scores direct actor-name and filename continuations; exact generated answers are secondary. Actor identity, authorized-owner status, actor order, and proposal order are crossed and balanced. Stage 1 requires at least 90% accuracy on both tasks, position gaps no greater than 0.15, actor-identity accuracy range no greater than 0.25, and matched candidate token counts. A failure stops representation testing and writes the same frozen worlds for a single Qwen3-8B follow-up.

```sh
uv run --extra inference python scripts/run_authorization_competence.py \
  --config configs/authorization_competence.yaml --model qwen3_4b --device cuda
```

Only a passing Stage 1 opens Stage 2 (single-scope representation testing). Two-scope composition requires at least 90% accuracy per task and representation at Stage 2. Canonicalization runs only after adequate competence through Stage 3 and observed raw representation disagreement. Each stage saves its own predictions, world-clustered metrics, and report under `outputs/`; `run_status.json` records which gates stopped or opened later stages. A failed Qwen3-4B Stage 1 report includes a command to evaluate the exact same `worlds.jsonl` with Qwen3-8B.

## Lexical-symmetry competence calibration

The follow-up calibration creates 180 fresh semantic worlds, each with original and swapped assignments drawn from 36 equal-length actor identifiers across six predeclared identifier families. It preserves the canonical JSON and Stage-1 wording, audits identifier tokenization and authorization/position frequencies, and reports raw and world-paired symmetrized margins, name-swap flips, position effects, and family effects. The preregistered gate must pass before a separate representation-invariance run is considered; this runner never starts that stage.

```sh
uv run --extra inference --extra quantization python scripts/run_authorization_lexical_symmetry.py \
  --config configs/authorization_lexical_symmetry.yaml \
  --model qwen3_8b_int8 --device cuda
```

Use `--dataset-only` to generate and validate the fresh paired dataset without loading a model.

After the lexical-symmetry report passes its gate, single-scope representation invariance can reuse those exact worlds and both actor-name assignments:

```sh
uv run --extra inference --extra quantization python scripts/run_authorization_lexical_invariance.py \
  --calibration-run outputs/20261001T002047991833Z_authorization_lexical_symmetry_qwen3_8b_int8 \
  --model qwen3_8b_int8 --device cuda
```

The representation runner refuses to start without a passing calibration gate and verifies the resolved model revision and actor-token IDs against the calibration artifacts.

### Gemma 3 12B IT NF4 findings

The Gemma 3 12B IT NF4 lexical-symmetry calibration passed its preregistered gate on 180 worlds. The subsequent single-scope representation run scored 3,600 rows over the same worlds and both actor-name assignments. Dataset semantics, candidate scoring, and token-boundary audits passed.

Interpretation was at 100% symmetrized accuracy for all five representations, with no name-swap flips or representation disagreement. Application was format-sensitive: decision-owner statements, JSON, and permission tables were at 100% symmetrized accuracy; executable rules were at 100% after name-swap averaging but 98.3% across individual assignments; natural-language policies were at 91.1% symmetrized accuracy (88.1% across individual assignments), with a 12.8% name-swap flip rate. Across representations, application winners disagreed in 8.9% of worlds after symmetrization and in 12.5% of individual world-assignment pairs. Symmetrization is an analysis measure, not a deployable single-prompt behavior. Application accuracy also retained a position gap: 100% when the correct value was first versus 94% when it was second.

This supports a representation-dependent application result for this synthetic single-scope task; it does not establish a general authorization failure or an internal mechanism. The calibration gate qualifies this task for representation testing but does not remove the observed single-prompt sensitivity. The run has not yet been extended to Gemma cross-scope testing. To test whether the effect transfers to cross-scope interference, use the completed Gemma Stage 2 run as the E2 source:

```sh
uv run --extra inference --extra quantization python scripts/run_authorization_cross_scope.py \
  --stage2-run outputs/20261001T203345458674Z_authorization_lexical_invariance_gemma3_12b_it_nf4 \
  --model gemma3_12b_it_nf4 --device cuda
```

See the [Gemma single-scope report](outputs/20261001T203345458674Z_authorization_lexical_invariance_gemma3_12b_it_nf4/report.md) and [metrics](outputs/20261001T203345458674Z_authorization_lexical_invariance_gemma3_12b_it_nf4/metrics.json) for world-bootstrap intervals and detailed contrasts. The existing Qwen3-8B E2/E3 results are a separate model-specific sequence and should not be treated as evidence that these Gemma effects transfer.

## Experiment 2: cross-scope authorization interference

This experiment reuses the Stage 2 worlds and model revision. Every CONGRUENT/CONFLICTING pair contains filename and ordering scopes, with both actors explicitly marked owner/non-owner in each scope; only the ordering owner changes. Application prompts include filename proposals only. Both lexical assignments and all five representations are scored for filename interpretation and application. Filename policy order is balanced first/second within conditions. The runner validates and writes exactly 7,200 rows before inference and does not launch canonicalization or another model.

```sh
uv run --extra inference --extra quantization python scripts/run_authorization_cross_scope.py \
  --stage2-run outputs/20261001T120524351420Z_authorization_lexical_invariance_qwen3_8b_int8 \
  --model qwen3_8b_int8 --device cuda
```

## Legacy authorization-invariance pilot

`authorization_invariance` is the original asymmetric `Source S` / `Default policy` exploratory pilot. Its package, config, runner, and outputs are retained for provenance only. New authorization experiments should use `authorization_competence` and must not import the legacy package.

## Experiment 3: canonicalization mitigation

E3 reuses the complete E2 run and its pinned Qwen3-8B int8 revision. The model converts each raw two-scope policy by scoring the four legal canonical owner mappings; it then answers from the selected IR. Application margins average frozen and reversed filename proposal order before lexical-name symmetrization. Conversion and answer accuracy are reported separately. A dataset-only run validates and saves all datasets and provenance before inference; use `--resume` on the same host to score that exact prepared run.

```sh
uv run --extra inference --extra quantization python scripts/run_authorization_canonicalization.py \
  --e2-run outputs/20261001T130103102194Z_authorization_cross_scope_qwen3_8b_int8 \
  --model qwen3_8b_int8 --device cuda
```

## Development workflow

Work advances through an autonomous, human-checkpointed loop: `docs/PLAN.md` is the step ledger, `/continue` in Claude Code (or «продолжи») claims and ships one step per invocation, and `bash scripts/check.sh` is the mandatory gate (also run in CI). The Rego engine check needs the pinned OPA binary: `bash scripts/fetch_opa.sh` downloads it into the git-ignored `tools/` and verifies its sha256 (`scripts/bootstrap.sh` and CI run it; Rego tests skip locally without it). The research spec is [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md); see also [CLAUDE.md](CLAUDE.md) and [docs/AUTONOMY.md](docs/AUTONOMY.md). Experiment reports, including negative results, go in `docs/experiments/`. The paper is in `paper/` (ACL template; `cd paper && latexmk -pdf main`).

## authinv Phase 0: generation kill test

Spec: `docs/RESEARCH_PLAN.md` §3 Phase 0; preregistration: `docs/prereg/phase0.md`. The runner consumes the pilot's Gemma Stage-2 run (pinned by hash in `configs/authinv/phase0.yaml`) and needs vLLM 0.29.0 (the `vllm/vllm-openai:v0.29.0` image, or a separate venv; vLLM is not in `uv.lock`).

```sh
python scripts/authinv_phase0.py run --config configs/authinv/phase0.yaml \
  --source-run outputs/20261001T203345458674Z_authorization_lexical_invariance_gemma3_12b_it_nf4 \
  --model qwen3_8b                       # add --tensor-parallel N, --model-dir DIR (verified mirror), --resume RUN
python scripts/authinv_phase0.py aggregate --config configs/authinv/phase0.yaml --runs outputs/<run> ...
```

`--dataset-only` validates and freezes the 3,600-row dataset without loading a model. On hosts whose system `nvcc` is older than FlashInfer requires, set `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy decoding does not use it).

## authinv Phase 1: CedarBench import

Fetches the pinned CedarBench snapshot (`configs/authinv/sources/cedarbench.yaml`: repo, commit, Apache-2.0 license, tree hash) into the git-ignored `data/raw/cedarbench/`, then converts every reference policy to a canonical world, certifies it, and checks it against the original Cedar text. No model is loaded.

```sh
uv run --extra engines python scripts/fetch_cedarbench.py            # or --verify-only
uv run --extra engines python scripts/authinv_import_cedarbench.py --dataset-only
# importer v1 (P1.7) instead of the default v2 (P1.7.2):
uv run --extra engines python scripts/authinv_import_cedarbench.py --dataset-only --importer cedarbench-import-v1
```

## authinv Phase 1: Quacky AWS IAM import

Fetches the pinned Quacky AWS IAM policy set (`configs/authinv/sources/quacky.yaml`: repo, commit, BSD-2-Clause license, tree hash, 587 files) into the git-ignored `data/raw/quacky/`. It then translates each IAM policy to a canonical world over a closed request universe and differential-tests the translation against a reference IAM evaluator (`src/authinv/sources/iam_eval.py`) run on the original JSON. Each world is certified with Cedar and OPA. No model is loaded. The importer version is set in the manifest (`importer`, currently `quacky-import-v2`: wildcard witnesses and closed-world `NotAction`/`NotResource` complements); `--importer quacky-import-v1` reproduces the P1.8 run.

```sh
uv run --extra engines python scripts/fetch_quacky.py                # or --verify-only
uv run --extra engines python scripts/authinv_import_quacky.py --dataset-only
uv run --extra engines python scripts/authinv_import_quacky.py --dataset-only --importer quacky-import-v1
```

## authinv Phase 1: frozen benchmark

```sh
uv run --extra engines python scripts/authinv_build_benchmark.py --config configs/authinv/benchmark.yaml \
  --source synthetic=outputs/<synthetic run> --source cedarbench=outputs/<cedarbench v2 run> --source quacky=outputs/<quacky v2 run>
```

Writes `dataset.jsonl` + `dataset_manifest.json` (input to `scripts/authinv_eval.py`), `worlds.jsonl`, `equivalence.jsonl`, and `audit.json`. See `docs/experiments/p1.10-benchmark.md`.

## authinv tier T3: agent sandbox

```sh
python scripts/authinv_agent_eval.py run --config configs/authinv/t3.yaml \
  --benchmark outputs/<frozen benchmark run> --model <key>     # --dataset-only builds and verifies episodes
```

Episodes reuse the benchmark's worlds and requests. Needs vLLM 0.29.0 and `cedarpy` (`--extra engines`). See `src/authinv/agent/sandbox.py` for the protocol and metrics.
