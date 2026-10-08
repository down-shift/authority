#!/usr/bin/env python3
"""Generate and certify the synthetic policy worlds (P1.6).

  uv run --extra engines python scripts/authinv_build_synthetic.py --config configs/authinv/synthetic.yaml

Writes outputs/<stamp>_authinv_synthetic/: policies.jsonl (canonical), equivalence.jsonl (one proof
per world),
balance.json (design-factor counts), report.json (generation + certification counts), artifact manifest.
A world is kept only if certification passes for every rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import provenance  # noqa: E402
from authinv.equivalence.check import certify, write_proofs  # noqa: E402
from authinv.policy import to_dict  # noqa: E402
from authinv.sources.synthetic import DOMAINS, balance_table, generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run = args.output_root / f"{provenance.utc_stamp()}_authinv_synthetic"
    run.mkdir(parents=True, exist_ok=False)
    provenance.write_json(
        run / "config.json",
        {
            "config": cfg,
            "config_sha256": provenance.sha256_json(cfg),
            "git": provenance.git_state(),
            "command": sys.argv,
        },
    )
    kept, proofs, reports = [], [], []
    for domain in cfg["domains"]:
        worlds, rep = generate(domain, cfg["replicates"], cfg["seed"], cfg["min_per_label"])
        d = DOMAINS[domain]
        failed = 0
        for w in worlds:
            proof = certify(w, (d.principal_type,), (d.resource_type,))
            proofs.append(proof)
            if proof["passed"]:
                kept.append(w)
            else:
                failed += 1
        rep["certification_failed"] = failed
        reports.append(rep)
    provenance.write_jsonl(run / "policies.jsonl", [to_dict(w) for w in kept])
    proofs_sha = write_proofs(proofs, run / "equivalence.jsonl")
    provenance.write_json(run / "balance.json", balance_table(kept))
    provenance.write_json(
        run / "report.json",
        {
            "domains": reports,
            "kept": len(kept),
            "equivalence_sha256": proofs_sha,
            "policies_sha256": provenance.sha256_file(run / "policies.jsonl"),
        },
    )
    provenance.write_run_status(run, "complete", len(kept), sum(r["worlds"] for r in reports))
    provenance.finalize(run)
    print(json.dumps({"run": str(run), "kept": len(kept), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
