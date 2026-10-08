#!/usr/bin/env python3
"""Build the frozen Phase-1 benchmark from certified source runs (P1.10).

  uv run --extra engines python scripts/authinv_build_benchmark.py --config configs/authinv/benchmark.yaml \
      --source synthetic=outputs/<run> --source cedarbench=outputs/<v2 run> --source quacky=outputs/<v2 run>

Writes outputs/<stamp>_authinv_benchmark/: dataset.jsonl + dataset_manifest.json (the frozen T2 input),
worlds.jsonl (canonical orig + swapped policies), equivalence.jsonl (re-certification of every
swapped policy),
audit.json (balance, label/boundary/option-order counts, rendering lengths, source split), artifact manifest.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import provenance  # noqa: E402
from authinv.benchmark import PROMPT_VERSION, World, dedupe, non_degenerate, world_rows  # noqa: E402
from authinv.equivalence.check import certify, write_proofs  # noqa: E402
from authinv.policy import from_dict, policy_hash, semantic_hash, to_dict  # noqa: E402
from authinv.render.renderers import RENDERINGS, VERSIONS  # noqa: E402


def load_source(name: str, run: Path, spec: dict) -> tuple[list[World], dict]:
    proofs = {p["policy_id"]: p for p in provenance.read_jsonl(run / "equivalence.jsonl")}
    groups = {}
    if spec.get("group_from_records"):
        groups = {
            r["policy_id"]: r[spec["group_from_records"]]
            for r in provenance.read_jsonl(run / "records.jsonl")
        }
    worlds, skipped = [], collections.Counter()
    for d in provenance.read_jsonl(run / "policies.jsonl"):
        pol = from_dict(d)
        proof = proofs.get(pol.policy_id)
        if proof is None or not proof["passed"]:
            skipped["no_passing_proof"] += 1
            continue
        if proof["policy_sha256"] != policy_hash(pol):
            skipped["proof_hash_mismatch"] += 1
            continue
        worlds.append(
            World(
                pol,
                tuple(proof["principal_types"]),
                tuple(proof["resource_types"]),
                spec["kind"],
                name,
                dict(pol.meta) | ({"group": groups[pol.policy_id]} if groups else {}),
            )
        )
    return worlds, {"run": str(run), "loaded": len(worlds), "skipped": dict(skipped)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--source", action="append", required=True, help="NAME=RUN_DIR")
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run = args.output_root / f"{provenance.utc_stamp()}_authinv_benchmark"
    run.mkdir(parents=True, exist_ok=False)
    audit: dict = {"sources": {}, "prompt_version": PROMPT_VERSION, "renderer_versions": VERSIONS}
    selected: list[World] = []
    for item in args.source:
        name, path = item.split("=", 1)
        spec = cfg["sources"][name]
        worlds, info = load_source(name, Path(path), spec)
        worlds, dupes = dedupe(worlds)
        keep = [w for w in worlds if non_degenerate(w, cfg["min_per_label"])]
        info.update({"duplicates_removed": dupes, "non_degenerate": len(keep)})
        if spec.get("cap_per_group"):
            per = collections.Counter()
            capped = []
            # Within a group, select by semantic hash: deterministic and independent of file order.
            for w in sorted(keep, key=lambda w: (w.meta["group"], semantic_hash(w.policy))):
                g = w.meta["group"]
                if per[g] < spec["cap_per_group"]:
                    per[g] += 1
                    capped.append(w)
            info["capped_out"] = len(keep) - len(capped)
            info["groups"] = len(per)
            keep = capped
        info["selected"] = len(keep)
        audit["sources"][name] = info
        selected += keep
    rows, worlds_out, proofs = [], [], []
    for w in selected:
        r, extra = world_rows(w, cfg["per_label"], cfg["seed"])
        rows += r
        proof = certify(extra["swap_policy"], w.principal_types, w.resource_types)
        proofs.append(proof)
        if not proof["passed"]:
            raise SystemExit(f"{w.policy.policy_id}: swapped policy failed certification")
        worlds_out.append(
            {
                "world_id": w.policy.policy_id,
                "source": w.source,
                "source_kind": w.source_kind,
                "principal_types": list(w.principal_types),
                "resource_types": list(w.resource_types),
                "orig": to_dict(w.policy),
                "swap": to_dict(extra["swap_policy"]),
                "swap_map": extra["swap_map"],
            }
        )
    provenance.write_jsonl(run / "dataset.jsonl", rows)
    provenance.write_json(
        run / "dataset_manifest.json",
        {
            "sha256": provenance.sha256_file(run / "dataset.jsonl"),
            "rows": len(rows),
            "prompt_version": PROMPT_VERSION,
        },
    )
    provenance.write_jsonl(run / "worlds.jsonl", worlds_out)
    write_proofs(proofs, run / "equivalence.jsonl")
    app = [r for r in rows if r["task"] == "application"]
    audit.update(
        {
            "worlds": len(selected),
            "worlds_by_source_kind": dict(collections.Counter(w.source_kind for w in selected)),
            "worlds_by_source": dict(collections.Counter(w.source for w in selected)),
            "worlds_by_source_group": {
                src: dict(collections.Counter(str(w.meta.get("group")) for w in selected if w.source == src))
                for src in sorted({w.source for w in selected})
                if any("group" in w.meta for w in selected if w.source == src)
            },
            "worlds_by_tier": dict(
                collections.Counter(
                    r["tier"]
                    for r in app
                    if r["rendering"] == RENDERINGS[0] and r["instance"] == "q0" and r["assignment"] == "orig"
                )
            ),
            "rows": len(rows),
            "rows_by_task": dict(collections.Counter(r["task"] for r in rows)),
            "application_label": dict(collections.Counter(r["label"] for r in app)),
            "application_boundary_share": sum(r["boundary"] for r in app) / max(1, len(app)),
            "option_order": dict(collections.Counter(r["option_order"] for r in app)),
            "rendering_chars": {
                g: {
                    "median": statistics.median(x["rendering_chars"] for x in app if x["rendering"] == g),
                    "max": max(x["rendering_chars"] for x in app if x["rendering"] == g),
                }
                for g in RENDERINGS
            },
            "swap_recertified": f"{sum(p['passed'] for p in proofs)}/{len(proofs)}",
            "meets_target": len(selected) >= cfg["target_worlds"],
        }
    )
    provenance.write_json(run / "audit.json", audit)
    provenance.write_json(
        run / "config.json",
        {
            "config": cfg,
            "config_sha256": provenance.sha256_json(cfg),
            "sources": args.source,
            "git": provenance.git_state(),
            "command": sys.argv,
        },
    )
    provenance.write_run_status(run, "complete", len(rows), len(rows))
    provenance.finalize(run)
    print(
        json.dumps(
            {
                "run": str(run),
                **{k: audit[k] for k in ("worlds", "worlds_by_source", "rows", "meets_target")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
