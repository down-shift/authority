#!/usr/bin/env python3
"""Build the seeded 20-world spot-check packet for Gate 1 (P1.11).

  uv run --extra engines python scripts/authinv_spotcheck.py --benchmark outputs/<run> --out spotcheck.json

Picks 20 worlds stratified by source (cedarbench / quacky / synthetic repo / synthetic mcp) and tier,
seeded, and writes, per world: the facts block, all six renderings of the `orig` policy, and two sample
requests with their engine-certified labels. The packet feeds the human review form; no model output.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authinv import provenance  # noqa: E402
from authinv.benchmark import render_facts  # noqa: E402
from authinv.policy import from_dict, policy_hash, tier  # noqa: E402
from authinv.render.renderers import RENDERINGS, render  # noqa: E402

QUOTA = {"cedarbench": 6, "quacky": 4, "synthetic:repo": 5, "synthetic:mcp": 5}


def stratum(w: dict) -> str:
    if w["source"] != "synthetic":
        return w["source"]
    return "synthetic:" + dict(map(tuple, w["orig"]["meta"]))["domain"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()
    worlds = provenance.read_jsonl(args.benchmark / "worlds.jsonl")
    rows = provenance.read_jsonl(args.benchmark / "dataset.jsonl")
    rng = random.Random(args.seed)
    picked = []
    for s, n in QUOTA.items():
        pool = sorted((w for w in worlds if stratum(w) == s), key=lambda w: w["world_id"])
        by_tier = {}
        for w in pool:
            by_tier.setdefault(tier(from_dict(w["orig"])), []).append(w)
        order = []
        tiers = sorted(by_tier)
        while len(order) < min(n, len(pool)):  # round-robin over tiers so each stratum spans tiers
            for t in tiers:
                rest = [w for w in by_tier[t] if w not in order]
                if rest and len(order) < n:
                    order.append(rng.choice(rest))
        picked += [(s, w) for w in order]
    packet = []
    for i, (s, w) in enumerate(picked, 1):
        pol = from_dict(w["orig"])
        reqs = [
            r
            for r in rows
            if r["world_id"] == w["world_id"]
            and r["assignment"] == "orig"
            and r["task"] == "application"
            and r["rendering"] == "nl_statement"
        ]
        sample = [
            {"request": r["prompt"].split("Request: ", 1)[1].split("\n")[0], "label": r["label"]}
            for r in reqs[:2]
        ]
        packet.append(
            {
                "n": i,
                "world_id": w["world_id"],
                "stratum": s,
                "tier": tier(pol),
                "policy_sha256": policy_hash(pol),
                "facts": render_facts(pol),
                "renderings": {g: render(pol, g) for g in RENDERINGS},
                "sample_requests": sample,
            }
        )
    meta = {
        "benchmark": str(args.benchmark),
        "dataset_sha256": json.loads((args.benchmark / "dataset_manifest.json").read_text())["sha256"],
        "seed": args.seed,
        "quota": QUOTA,
        "renderings": list(RENDERINGS),
        "worlds": len(packet),
    }
    args.out.write_text(json.dumps({"meta": meta, "worlds": packet}, indent=1, ensure_ascii=False))
    print(args.out, len(packet))


if __name__ == "__main__":
    main()
