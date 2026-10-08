#!/usr/bin/env python3
"""Import CedarBench reference policies as canonical worlds (steps P1.7, P1.7.2). No model is loaded.

  uv run --extra engines python scripts/fetch_cedarbench.py
  uv run --extra engines python scripts/authinv_import_cedarbench.py --dataset-only
  uv run --extra engines python scripts/authinv_import_cedarbench.py --dataset-only \
      --importer cedarbench-import-v1   # reproduce the P1.7 (v1) import

The default importer is cedarbench-import-v2 (enlarged universes, exact `||` split).

Verifies the pinned snapshot (configs/authinv/sources/cedarbench.yaml), converts
every references/*.cedar file, certifies each converted world, checks the
conversion against the original Cedar text, and writes
outputs/<stamp>_cedarbench_import/:

  records.jsonl        one row per reference file: status, constructs, checks
  policies.jsonl       canonical JSON of the shipped worlds (unique by semantic hash)
  equivalence.jsonl    certify() proof + faithfulness check for every converted file
  import_report.json   counts (total, converted, excluded by construct, certified, shipped),
                       overall and split into the paper's 221 tasks vs the 5 stress scenarios
  metadata.json, run_status.json, artifact_sha256.json
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from authinv import provenance  # noqa: E402
from authinv.equivalence.check import write_proofs  # noqa: E402
from authinv.policy import canonical_json  # noqa: E402
from authinv.sources import cedarbench  # noqa: E402

MANIFEST = ROOT / "configs" / "authinv" / "sources" / "cedarbench.yaml"
TRACKED = [
    ROOT / "src" / "authinv" / "sources" / "cedarbench.py",
    ROOT / "src" / "authinv" / "policy" / "model.py",
    ROOT / "src" / "authinv" / "equivalence" / "check.py",
    ROOT / "src" / "authinv" / "equivalence" / "cedar.py",
    ROOT / "src" / "authinv" / "equivalence" / "requests.py",
    ROOT / "src" / "authinv" / "render" / "renderers.py",
    ROOT / "scripts" / "authinv_import_cedarbench.py",
    MANIFEST,
]


def run_import(
    scenarios: Path, out: Path, *, progress: bool = True, importer: str = cedarbench.IMPORTER_VERSION
) -> dict:
    """Convert every reference under `scenarios`; write artifacts into `out`; return the report."""
    schemas: dict[str, tuple] = {}
    for d in cedarbench.scenario_dirs(scenarios):
        text = (d / "schema.cedarschema").read_text()
        try:
            schemas[str(d.relative_to(scenarios))] = (cedarbench.parse_schema(text), text, None)
        except cedarbench.SchemaError as e:
            schemas[str(d.relative_to(scenarios))] = (None, text, str(e))
    records, proofs, shipped = [], [], {}
    refs = list(cedarbench.iter_references(scenarios))
    for i, (scen, ref) in enumerate(refs):
        schema, text, err = schemas[scen]
        rec, policy, proof = cedarbench.import_reference(scen, ref, schema, text, err, importer)
        records.append(rec)
        if proof is not None:
            proofs.append(proof)
        if rec.get("shipped"):
            shipped.setdefault(rec["semantic_sha256"], (policy, []))[1].append(rec["policy_id"])
        if progress and (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(refs)}", flush=True)
    provenance.write_jsonl(out / "records.jsonl", records)
    proof_sha = write_proofs(proofs, out / "equivalence.jsonl")
    worlds = sorted(shipped.values(), key=lambda pv: pv[0].policy_id)
    (out / "policies.jsonl").write_text("".join(canonical_json(p) + "\n" for p, _ in worlds))
    report = summarize(records, len(schemas), importer)
    report["splits"] = {
        name: summarize(
            [r for r in records if keep(r)], len({s for s in schemas if keep({"scenario": s})}), importer
        )
        for name, keep in (
            ("paper_221", lambda r: not cedarbench.is_stress(r["scenario"])),
            ("stress_5", lambda r: cedarbench.is_stress(r["scenario"])),
        )
    }
    report["equivalence_sha256"] = proof_sha
    report["policies_sha256"] = provenance.sha256_file(out / "policies.jsonl")
    provenance.write_json(out / "import_report.json", report)
    return report


def _quantiles(xs: list[int]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return {"min": xs[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p90": q(0.9), "max": xs[-1]}


def summarize(records: list[dict], n_scenarios: int, importer: str = cedarbench.IMPORTER_VERSION) -> dict:
    excluded = [r for r in records if r["status"] == "excluded"]
    converted = [r for r in records if r["status"] == "converted"]
    by_construct = collections.Counter(c for r in excluded for c in r["constructs"])
    sole = collections.Counter(r["constructs"][0] for r in excluded if len(r["constructs"]) == 1)
    shipped = [r for r in records if r.get("shipped")]
    first: dict[str, dict] = {}  # one record per unique world (semantic hash), first by policy id
    for r in sorted(shipped, key=lambda r: r["policy_id"]):
        first.setdefault(r["semantic_sha256"], r)
    out = {
        "importer": importer,
        "scenarios": n_scenarios,
        "scenarios_with_a_shipped_world": len({r["scenario"] for r in shipped}),
        "reference_files": len(records),
        "unique_reference_texts": len({r["source_sha256"] for r in records}),
        "source_validates": sum(bool(r.get("source_validates")) for r in records),
        "converted": len(converted),
        "excluded": len(excluded),
        "excluded_files_with_construct": dict(by_construct.most_common()),
        "excluded_files_with_only_this_construct": dict(sole.most_common()),
        "certified": sum(bool(r.get("certified")) for r in converted),
        "faithful": sum(bool(r.get("faithful")) for r in converted),
        "converted_but_source_invalid": sum(not r.get("source_validates") for r in converted),
        "shipped_reference_files": len(shipped),
        "shipped_unique_worlds": len(first),
        "shipped_tiers": dict(collections.Counter(r["tier"] for r in shipped)),
        "max_requests_per_world": max((r["n_requests"] for r in shipped), default=0),
        "requests_per_unique_world": _quantiles([r["n_requests"] for r in first.values()]),
        "rules_per_unique_world": dict(
            sorted(collections.Counter(r["n_rules"] for r in first.values()).items())
        ),
    }
    if importer != cedarbench.IMPORTER_V1:
        nd = [r for r in first.values() if r.get("non_degenerate")]
        out.update(
            converted_with_disjunction_split=sum(r.get("split_rules", 0) > 0 for r in converted),
            non_degenerate_files=sum(bool(r.get("non_degenerate")) for r in shipped),
            non_degenerate_unique_worlds=len(nd),
            non_degenerate_unique_worlds_without_dead_rules=sum(r["dead_rules"] == 0 for r in nd),
            unique_worlds_with_both_labels=sum(r["n_allow"] > 0 and r["n_deny"] > 0 for r in first.values()),
            scenarios_with_a_non_degenerate_world=len({r["scenario"] for r in nd}),
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-only", action="store_true", help="the only mode: data import, no model")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "cedarbench")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--importer", choices=cedarbench.IMPORTERS, default=cedarbench.IMPORTER_VERSION)
    args = p.parse_args()
    import fetch_cedarbench

    m = yaml.safe_load(args.manifest.read_text())
    problems = fetch_cedarbench.verify(m, args.data)
    if problems:
        raise SystemExit(
            "snapshot does not match the pin; run scripts/fetch_cedarbench.py:\n  " + "\n  ".join(problems)
        )
    suffix = "" if args.importer == cedarbench.IMPORTER_V1 else "_v2"
    run = args.output_root / f"{provenance.utc_stamp()}_cedarbench_import{suffix}"
    run.mkdir(parents=True)
    provenance.write_json(
        run / "metadata.json",
        {
            "step": "P1.7" if args.importer == cedarbench.IMPORTER_V1 else "P1.7.2",
            "importer": args.importer,
            "invocation": sys.argv,
            "source": {k: m[k] for k in ("repo", "commit", "license", "tree_sha256", "n_files")},
            "manifest_sha256": provenance.sha256_file(args.manifest),
            "git": provenance.git_state(),
            "source_sha256": provenance.source_hashes(TRACKED),
            "limits": {
                "max_entities_per_type": cedarbench.MAX_ENTITIES_PER_TYPE,
                "max_requests": cedarbench.MAX_REQUESTS,
                "entity_replicas": cedarbench.ENTITY_REPLICAS,
                "max_split_rules": cedarbench.MAX_SPLIT_RULES,
                "min_per_label": cedarbench.MIN_PER_LABEL,
            },
            "cedarpy": _version("cedarpy"),
        },
    )
    provenance.write_run_status(run, "running", 0, m["expected_reference_files"])
    try:
        report = run_import(args.data / "scenarios", run, importer=args.importer)
    except Exception as e:
        provenance.write_run_status(run, "failed", 0, m["expected_reference_files"], repr(e))
        provenance.finalize(run)
        raise
    provenance.write_run_status(run, "dataset_only", report["reference_files"], report["reference_files"])
    provenance.finalize(run)
    print(yaml.safe_dump(report, sort_keys=False))
    print(f"wrote {run}")


def _version(pkg: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
