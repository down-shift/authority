#!/usr/bin/env python3
"""Fetch the pinned CedarBench snapshot into data/raw/cedarbench/ (git-ignored) and verify it.

  uv run python scripts/fetch_cedarbench.py              # download + extract + verify
  uv run python scripts/fetch_cedarbench.py --verify-only

The source, commit, license, and expected hashes live in
configs/authinv/sources/cedarbench.yaml. Only the files the importer reads are
extracted: cedarbench/scenarios/** plus LICENSE, NOTICE, and the CedarBench
README. Verification hashes the extracted tree (sorted `path<TAB>sha256` lines),
not the tarball, because GitHub does not promise byte-stable archives. Any
mismatch is an error: re-pinning is a deliberate manifest change.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "authinv" / "sources" / "cedarbench.yaml"
DEST = ROOT / "data" / "raw" / "cedarbench"


def tree_sha256(root: Path) -> tuple[str, int]:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            lines.append(f"{p.relative_to(root).as_posix()}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest(), len(lines)


def verify(m: dict, dest: Path) -> list[str]:
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
    scen = dest / "scenarios"
    n_scen = len(list(scen.rglob("schema.cedarschema")))
    n_refs = len(list(scen.rglob("references/*.cedar")))
    if n_scen != m["expected_scenarios"]:
        problems.append(f"{n_scen} scenarios != expected {m['expected_scenarios']}")
    if n_refs != m["expected_reference_files"]:
        problems.append(f"{n_refs} reference files != expected {m['expected_reference_files']}")
    return problems


def fetch(m: dict, dest: Path) -> None:
    url = m["archive_url"].format(commit=m["commit"])
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - pinned https URL from the manifest
        data = r.read()
    prefix = f"{m['repo'].split('/')[-1]}-{m['commit']}/"
    keep = {f"{prefix}{k}": v for k, v in m["extract"].items()}
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            target = None
            for src, dst in keep.items():
                if member.name == src or (src.endswith("/") and member.name.startswith(src)):
                    target = dst + member.name[len(src) :] if src.endswith("/") else dst
            if target is None:
                continue
            out = (dest / target).resolve()
            if dest.resolve() not in out.parents:
                raise SystemExit(f"refusing to extract outside {dest}: {member.name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tar.extractfile(member).read())
            n += 1
    print(f"extracted {n} files into {dest}")


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
    print(f"verified {m['repo']}@{m['commit'][:12]} ({m['license']}): tree_sha256 {m['tree_sha256'][:16]}…")


if __name__ == "__main__":
    main()
