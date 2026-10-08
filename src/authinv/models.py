"""Model registry (configs/authinv/models.yaml) and model-directory verification.

Gated upstream checkpoints are reproduced without sharing anyone's HF token:
LFS weights come from an ungated mirror whose files are byte-identical to the
pinned upstream revision, the upstream's small files are copied from a machine
that has access, and `verify_model_dir` checks every file in the assembled
directory against the committed upstream manifest before a model is loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "authinv" / "models.yaml"
REQUIRED_KEYS = ("name", "revision", "params_b", "precision", "quantization")


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    for key, spec in data["models"].items():
        missing = [k for k in REQUIRED_KEYS if k not in spec]
        if missing:
            raise ValueError(f"model {key!r} missing {missing}")
    for set_name, members in data.get("sets", {}).items():
        unknown = [m for m in members if m not in data["models"]]
        if unknown:
            raise ValueError(f"set {set_name!r} names unknown models {unknown}")
    return data


def model_spec(key: str, path: Path = REGISTRY) -> dict[str, Any]:
    return {"key": key, **load_registry(path)["models"][key]}


def git_blob_sha1(data: bytes) -> str:
    """The object id git (and the HF tree API) reports for a non-LFS file."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_dir(
    model_dir: Path, manifest: dict[str, Any], skip: tuple[str, ...] = ("README.md",)
) -> dict:
    """Check every manifest file exists in `model_dir` with the upstream hash.

    Extra files in `model_dir` (e.g. a mirror's chat_template.jinja) are
    reported, because loaders may prefer them over the upstream's files.
    """
    problems: list[str] = []
    for rel, want in sorted(manifest["files"].items()):
        if rel in skip:
            continue
        path = model_dir / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif "sha256" in want:
            if file_sha256(path) != want["sha256"]:
                problems.append(f"sha256 mismatch {rel}")
        elif git_blob_sha1(path.read_bytes()) != want["git_blob_sha1"]:
            problems.append(f"blob mismatch {rel}")
    extra = sorted(
        str(p.relative_to(model_dir))
        for p in model_dir.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(model_dir).parts)
        and str(p.relative_to(model_dir)) not in manifest["files"]
    )
    return {
        "passed": not problems and not extra,
        "upstream": f"{manifest['repo']}@{manifest['revision']}",
        "problems": problems,
        "extra_files": extra,
    }
