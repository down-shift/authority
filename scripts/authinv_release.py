#!/usr/bin/env python3
"""Build the authinv benchmark release in the Hugging Face Datasets layout (D1.1). Local only.

  uv run --extra engines python scripts/fetch_cedarbench.py && uv run python scripts/fetch_quacky.py
  uv run --extra engines python scripts/authinv_release.py build --config configs/authinv/release.yaml \
      --benchmark outputs/<frozen benchmark run> --out <dir> [--datasheet docs/datasheet.md]
  python scripts/authinv_release.py verify --release <dir>

The fetch scripts put the pinned upstream license files under data/raw/ (git-ignored). `build`
refuses on any unlicensed or incompatible source, any license file that isn't the pinned one,
any shipped policy without a passed equivalence proof, and any rendering or request that differs
from the frozen prompts (src/authinv/release.py). `verify` re-hashes a release against its
MANIFEST.json. Publishing is Jerzy's decision; nothing here uploads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import release  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--benchmark", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--datasheet", type=Path, default=ROOT / "docs" / "datasheet.md")
    v = sub.add_parser("verify")
    v.add_argument("--release", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "build":
        cfg = yaml.safe_load(args.config.read_text())
        m = release.build_release(args.benchmark, cfg, ROOT, args.out, args.datasheet.read_text())
        print(json.dumps({"out": str(args.out), "counts": m["counts"], "files": len(m["files"])}, indent=2))
    else:
        m = release.verify_release(args.release)
        print(f"verified {len(m['files'])} files of {m['release']} {m['version']}")


if __name__ == "__main__":
    main()
