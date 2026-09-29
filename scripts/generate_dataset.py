#!/usr/bin/env python3
"""Generate a deterministic dataset without loading a model."""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from authority_leakage.generation.delegation import assert_delegation_controls, generate_delegation
from authority_leakage.generation.epistemic import assert_epistemic_pairs, generate_epistemic
from authority_leakage.inference import write_jsonl
from authority_leakage.clean import generate_clean, dataset_sha256


def build(config: dict):
    if config.get("design") == "matched_authority_v1":
        return generate_clean(config["experiment"], config, int(config["seed"]))
    if config["experiment"] == "epistemic":
        examples = generate_epistemic(config, int(config["seed"]))
        assert_epistemic_pairs(examples)
    elif config["experiment"] == "delegation":
        examples = generate_delegation(config, int(config["seed"]))
        assert_delegation_controls(examples)
    else:
        raise ValueError("Unknown experiment")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    examples = build(config)
    write_jsonl(args.output, (e.to_dict() for e in examples))
    print(json.dumps({"examples": len(examples), "worlds": len({e.pair_id for e in examples}), "sha256": dataset_sha256(examples), "output": str(args.output)}))


if __name__ == "__main__":
    main()
