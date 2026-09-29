#!/usr/bin/env python3
"""Offline structural smoke test; use run_experiment.py for model inference."""
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from authority_leakage.clean import generate_clean, dataset_sha256, validate_matching
from authority_leakage.scoring import grade, parse_canonical_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    e_config = yaml.safe_load((root / "configs/epistemic.yaml").read_text())
    d_config = yaml.safe_load((root / "configs/delegation.yaml").read_text())
    e_config["worlds"] = 3
    d_config["worlds"] = 3
    epi = generate_clean("epistemic", e_config, e_config["seed"])
    scope = generate_clean("scope", d_config, d_config["seed"])
    validate_matching(epi); validate_matching(scope)
    assert epi == generate_clean("epistemic", e_config, e_config["seed"])
    assert scope == generate_clean("scope", d_config, d_config["seed"])
    assert parse_canonical_json('{"output_format":"json","ordering":"ascending","filename":"result.txt","tool_choice":"calculator","numeric_answer":42}')
    assert grade(epi[0], epi[0].correct_answer)["parse_success"]
    print(json.dumps({"epistemic_examples": len(epi), "scope_examples": len(scope),
                      "epistemic_sha256": dataset_sha256(epi), "scope_sha256": dataset_sha256(scope),
                      "matched_worlds_checked": 6, "structural_checks": "passed"}))


if __name__ == "__main__":
    main()
