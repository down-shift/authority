"""Skip marker for tests that run the OPA binary (fetched by scripts/fetch_opa.sh; CI always has it)."""

from __future__ import annotations

import os

import pytest

from authinv.equivalence.rego import find_opa

requires_opa = pytest.mark.skipif(
    find_opa() is None and not os.environ.get("CI"),
    reason="OPA binary not found: run `bash scripts/fetch_opa.sh`",
)
