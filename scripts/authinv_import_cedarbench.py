#!/usr/bin/env python3
"""Import CedarBench reference policies as canonical worlds (step P1.7). No model is loaded.

  uv run --extra engines python scripts/fetch_cedarbench.py
  uv run --extra engines python scripts/authinv_import_cedarbench.py --dataset-only

Verifies the pinned snapshot (configs/authinv/sources/cedarbench.yaml), converts
every references/*.cedar file, certifies each converted world, checks the
conversion against the original Cedar text, and writes
outputs/<stamp>_cedarbench_import/:

  records.jsonl        one row per reference file: status, constructs, checks
  policies.jsonl       canonical JSON of the shipped worlds (unique by semantic hash)
  equivalence.jsonl    certify() proof + faithfulness check for every converted file
  import_report.json   counts (total, converted, excluded by construct, certified, shipped)
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


def run_import(scenarios: Path, out: Path, *, progress: bool = True) -> dict:
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
        rec, policy, proof = cedarbench.import_reference(scen, ref, schema, text, err)
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
    report = summarize(records, len(schemas), worlds)
    report["equivalence_sha256"] = proof_sha
    report["policies_sha256"] = provenance.sha256_file(out / "policies.jsonl")
    provenance.write_json(out / "import_report.json", report)
    return report


def summarize(records: list[dict], n_scenarios: int, worlds: list) -> dict:
    excluded = [r for r in records if r["status"] == "excluded"]
    converted = [r for r in records if r["status"] == "converted"]
    by_construct = collections.Counter(c for r in excluded for c in r["constructs"])
    sole = collections.Counter(r["constructs"][0] for r in excluded if len(r["constructs"]) == 1)
    shipped = [r for r in records if r.get("shipped")]
    return {
        "importer": cedarbench.IMPORTER_VERSION,
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
        "shipped_unique_worlds": len(worlds),
        "shipped_tiers": dict(collections.Counter(r["tier"] for r in shipped)),
        "max_requests_per_world": max((r["n_requests"] for r in shipped), default=0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-only", action="store_true", help="the only mode: data import, no model")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "cedarbench")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = p.parse_args()
    import fetch_cedarbench

    m = yaml.safe_load(args.manifest.read_text())
    problems = fetch_cedarbench.verify(m, args.data)
    if problems:
        raise SystemExit(
            "snapshot does not match the pin; run scripts/fetch_cedarbench.py:\n  " + "\n  ".join(problems)
        )
    run = args.output_root / f"{provenance.utc_stamp()}_cedarbench_import"
    run.mkdir(parents=True)
    provenance.write_json(
        run / "metadata.json",
        {
            "step": "P1.7",
            "invocation": sys.argv,
            "source": {k: m[k] for k in ("repo", "commit", "license", "tree_sha256", "n_files")},
            "manifest_sha256": provenance.sha256_file(args.manifest),
            "git": provenance.git_state(),
            "source_sha256": provenance.source_hashes(TRACKED),
            "limits": {
                "max_entities_per_type": cedarbench.MAX_ENTITIES_PER_TYPE,
                "max_requests": cedarbench.MAX_REQUESTS,
            },
            "cedarpy": _version("cedarpy"),
        },
    )
    provenance.write_run_status(run, "running", 0, m["expected_reference_files"])
    try:
        report = run_import(args.data / "scenarios", run)
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
