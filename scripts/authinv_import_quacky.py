#!/usr/bin/env python3
"""Import the Quacky AWS IAM policies as canonical worlds (step P1.8). No model is loaded.

  uv run --extra engines python scripts/fetch_quacky.py
  uv run --extra engines python scripts/authinv_import_quacky.py --dataset-only

Verifies the pinned snapshot (configs/authinv/sources/quacky.yaml), translates
every IAM file in the import set, differential-tests each translation against
the reference IAM evaluator on the original JSON, certifies each converted
world, and writes outputs/<stamp>_quacky_import/:

  records.jsonl        one row per IAM file: status, constructs, flags, checks
  policies.jsonl       canonical JSON of the shipped worlds (unique by semantic hash)
  equivalence.jsonl    certify() proof + IAM-evaluator faithfulness check for every converted file
  import_report.json   counts (total, converted, excluded by construct, faithful, certified, shipped)
  metadata.json, run_status.json, artifact_sha256.json
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from authinv import provenance  # noqa: E402
from authinv.equivalence.check import write_proofs  # noqa: E402
from authinv.policy import canonical_json  # noqa: E402
from authinv.sources import quacky  # noqa: E402

MANIFEST = ROOT / "configs" / "authinv" / "sources" / "quacky.yaml"
TRACKED = [
    ROOT / "src" / "authinv" / "sources" / "quacky.py",
    ROOT / "src" / "authinv" / "sources" / "iam_eval.py",
    ROOT / "src" / "authinv" / "policy" / "model.py",
    ROOT / "src" / "authinv" / "equivalence" / "check.py",
    ROOT / "src" / "authinv" / "equivalence" / "cedar.py",
    ROOT / "src" / "authinv" / "equivalence" / "rego.py",
    ROOT / "src" / "authinv" / "equivalence" / "requests.py",
    ROOT / "src" / "authinv" / "render" / "renderers.py",
    ROOT / "scripts" / "authinv_import_quacky.py",
    MANIFEST,
]


def run_import(samples: Path, globs: list[str], out: Path, *, expand_wildcards: bool = True) -> dict:
    """Translate every IAM file matched under `samples`; write artifacts into `out`; return the report."""
    records, proofs, shipped = [], [], {}
    for path in quacky.iter_policies(samples, globs):
        rec, policy, proof = quacky.import_policy(path, samples, expand_wildcards=expand_wildcards)
        records.append(rec)
        if proof is not None:
            proofs.append(proof)
        if rec.get("shipped"):
            shipped.setdefault(rec["semantic_sha256"], (policy, []))[1].append(rec["policy_id"])
    provenance.write_jsonl(out / "records.jsonl", records)
    proof_sha = write_proofs(proofs, out / "equivalence.jsonl")
    worlds = sorted(shipped.values(), key=lambda pv: pv[0].policy_id)
    (out / "policies.jsonl").write_text("".join(canonical_json(p) + "\n" for p, _ in worlds))
    report = summarize(records, worlds)
    report["equivalence_sha256"] = proof_sha
    report["policies_sha256"] = provenance.sha256_file(out / "policies.jsonl")
    provenance.write_json(out / "import_report.json", report)
    return report


def _dist(xs: list[int]) -> dict:
    if not xs:
        return {}
    return {"min": min(xs), "median": statistics.median(xs), "max": max(xs), "n": len(xs)}


def summarize(records: list[dict], worlds: list) -> dict:
    excluded = [r for r in records if r["status"] == "excluded"]
    converted = [r for r in records if r["status"] == "converted"]
    shipped = [r for r in records if r.get("shipped")]
    by_construct = collections.Counter(c for r in excluded for c in r["constructs"])
    sole = collections.Counter(r["constructs"][0] for r in excluded if len(r["constructs"]) == 1)
    first_world: dict[str, dict] = {}
    for r in shipped:
        first_world.setdefault(r["semantic_sha256"], r)
    uniq = list(first_world.values())

    def by_subset(rows: list[dict]) -> dict:
        return dict(collections.Counter(r["subset"] for r in rows))

    def no_wild(r: dict) -> bool:
        return "wildcard_expanded" not in r.get("flags", [])

    return {
        "importer": quacky.IMPORTER_VERSION,
        "policy_files": len(records),
        "policy_files_by_subset": by_subset(records),
        "unique_source_texts": len({r["source_sha256"] for r in records}),
        "converted": len(converted),
        "converted_by_subset": by_subset(converted),
        "excluded": len(excluded),
        "excluded_files_with_construct": dict(by_construct.most_common()),
        "excluded_files_with_only_this_construct": dict(sole.most_common()),
        "faithful": sum(bool(r.get("faithful")) for r in converted),
        "certified": sum(bool(r.get("certified")) for r in converted),
        "shipped_policy_files": len(shipped),
        "shipped_by_subset": by_subset(shipped),
        "shipped_unique_worlds": len(worlds),
        "shipped_unique_source_texts": len({r["source_sha256"] for r in shipped}),
        "shipped_flags": dict(collections.Counter(f for r in shipped for f in r.get("flags", []))),
        "shipped_files_without_wildcard_expansion": sum(no_wild(r) for r in shipped),
        "shipped_unique_worlds_without_wildcard_expansion": sum(no_wild(r) for r in uniq),
        "shipped_tiers": dict(collections.Counter(r["tier"] for r in shipped)),
        "requests_per_file": _dist([r["n_requests"] for r in shipped]),
        "requests_per_unique_world": _dist([r["n_requests"] for r in uniq]),
        "rules_per_unique_world": _dist([r["n_rules"] for r in uniq]),
        "non_degenerate_files": sum(r["non_degenerate"] for r in shipped),
        "non_degenerate_unique_worlds": sum(r["non_degenerate"] for r in uniq),
        "non_degenerate_unique_worlds_without_wildcard_expansion": sum(
            r["non_degenerate"] and no_wild(r) for r in uniq
        ),
        "all_allow_unique_worlds": sum(r["n_allow"] == r["n_requests"] for r in uniq),
        "all_deny_unique_worlds": sum(r["n_allow"] == 0 for r in uniq),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-only", action="store_true", help="the only mode: data import, no model")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "quacky")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = p.parse_args()
    import fetch_quacky

    m = yaml.safe_load(args.manifest.read_text())
    problems = fetch_quacky.verify(m, args.data)
    if problems:
        raise SystemExit(
            "snapshot does not match the pin; run scripts/fetch_quacky.py:\n  " + "\n  ".join(problems)
        )
    run = args.output_root / f"{provenance.utc_stamp()}_quacky_import"
    run.mkdir(parents=True)
    provenance.write_json(
        run / "metadata.json",
        {
            "step": "P1.8",
            "invocation": sys.argv,
            "source": {k: m[k] for k in ("repo", "commit", "license", "tree_sha256", "n_files")},
            "manifest_sha256": provenance.sha256_file(args.manifest),
            "git": provenance.git_state(),
            "source_sha256": provenance.source_hashes(TRACKED),
            "method": "IAM->canonical translation, differential-tested against authinv.sources.iam_eval",
            "expand_wildcards": m["expand_wildcards"],
            "limits": {"max_rules": quacky.MAX_RULES, "max_requests": quacky.MAX_REQUESTS},
            "cedarpy": _version("cedarpy"),
        },
    )
    provenance.write_run_status(run, "running", 0, m["expected_policies"])
    try:
        report = run_import(
            args.data / "samples", m["import_globs"], run, expand_wildcards=m["expand_wildcards"]
        )
    except Exception as e:
        provenance.write_run_status(run, "failed", 0, m["expected_policies"], repr(e))
        provenance.finalize(run)
        raise
    provenance.write_run_status(run, "dataset_only", report["policy_files"], report["policy_files"])
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
