# Lesson: the vLLM run image is not the project environment

**Type:** lesson. **Audience:** every agent adding imports to code that runs on the H100.

Runs use the `vllm/vllm-openai:v0.29.0` image, plus `pip install cedarpy==4.12.1`.
It has torch, transformers, numpy, yaml and huggingface_hub, but **not scipy**.
The P2.9 Spearman fix made `from scipy.stats import rankdata` a module-level import
of `src/authinv/eval/invariance.py`, which every runner imports. The P3.3 smoke test
(2026-11-07) then failed at import in all five runs. It was fixed in PR #61 with a
lazy import inside the one function that needs it.

**How to apply:**
- Keep analysis-only dependencies (scipy, sklearn, matplotlib) out of the import path
  of the runners: import them inside the function that uses them.
- `tests/test_authinv_t2.py::test_run_modules_import_without_scipy` imports the run
  modules with scipy blocked. Extend it when you add a run module.
- Smoke-test a new runner in the actual image (a `--limit` run on `smoke_qwen3_0_6b`)
  before a sweep. The local venv will not show this class of bug.

See [[reference-dgx-usage]].
