#!/usr/bin/env python3
"""Offline structural smoke test; use run_experiment.py for model inference."""
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from authority_leakage.generation.delegation import assert_delegation_controls, generate_delegation
from authority_leakage.generation.epistemic import assert_epistemic_pairs, generate_epistemic
from authority_leakage.scoring import grade


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    e_config = yaml.safe_load((root / "configs/epistemic.yaml").read_text())
    d_config = yaml.safe_load((root / "configs/delegation.yaml").read_text())
    epi = generate_epistemic(e_config, e_config["seed"])
    delegation = generate_delegation(d_config, d_config["seed"])
    assert_epistemic_pairs(epi)
    assert_delegation_controls(delegation)
    assert epi == generate_epistemic(e_config, e_config["seed"])
    assert delegation == generate_delegation(d_config, d_config["seed"])
    assert grade(epi[0], epi[0].correct_answer)["accuracy"]
    print(json.dumps({"epistemic_examples": len(epi), "epistemic_pairs": len({e.pair_id for e in epi}),
                      "delegation_examples": len(delegation), "delegation_pairs": len({e.pair_id for e in delegation}),
                      "structural_checks": "passed"}))


if __name__ == "__main__":
    main()
