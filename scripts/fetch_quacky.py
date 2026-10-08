#!/usr/bin/env python3
"""Fetch the pinned Quacky AWS IAM policy set into data/raw/quacky/ (git-ignored) and verify it.

  uv run python scripts/fetch_quacky.py              # download + extract + verify
  uv run python scripts/fetch_quacky.py --verify-only

The source, commit, license, and expected hashes live in
configs/authinv/sources/quacky.yaml. Only LICENSE, the READMEs, and the AWS IAM
sample directories the importer reads are extracted. Verification hashes the
extracted tree (sorted `path<TAB>sha256` lines), as for CedarBench (P1.7), and
counts the import set. Any mismatch is an error: re-pinning is a deliberate
manifest change.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fetch_cedarbench import fetch, tree_sha256  # noqa: E402

MANIFEST = ROOT / "configs" / "authinv" / "sources" / "quacky.yaml"
DEST = ROOT / "data" / "raw" / "quacky"


def verify(m: dict, dest: Path) -> list[str]:
    from authinv.sources.quacky import iter_policies, subset_of

    problems = []
    if not dest.is_dir():
        return [f"{dest} does not exist; run without --verify-only"]
    got, n = tree_sha256(dest)
    if got != m["tree_sha256"]:
        problems.append(f"tree_sha256 {got} != pinned {m['tree_sha256']}")
    if n != m["n_files"]:
        problems.append(f"{n} files != pinned {m['n_files']}")
    for rel, want in m["file_sha256"].items():
        p = dest / rel
        have = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if have != want:
            problems.append(f"{rel}: {have} != {want}")
    samples = dest / "samples"
    files = iter_policies(samples, m["import_globs"])
    n_orig = sum(subset_of(p.relative_to(samples).as_posix()) == "original" for p in files)
    if len(files) != m["expected_policies"]:
        problems.append(f"{len(files)} policies != expected {m['expected_policies']}")
    if n_orig != m["expected_originals"]:
        problems.append(f"{n_orig} originals != expected {m['expected_originals']}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--dest", type=Path, default=DEST)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    m = yaml.safe_load(args.manifest.read_text())
    if not args.verify_only:
        if args.dest.exists() and any(args.dest.iterdir()):
            raise SystemExit(f"{args.dest} is not empty; remove it or use --verify-only")
        fetch(m, args.dest)
    problems = verify(m, args.dest)
    if problems:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"verified {m['repo']}@{m['commit'][:12]} ({m['license']}): {m['expected_policies']} policies, "
        f"tree_sha256 {m['tree_sha256'][:16]}…"
    )


if __name__ == "__main__":
    main()
