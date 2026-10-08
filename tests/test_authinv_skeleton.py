"""authinv package boundaries, model registry, and model-directory verification."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from authinv import FORBIDDEN_IMPORTS, models

ROOT = Path(__file__).resolve().parents[1]


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_authinv_never_imports_pilot_packages():
    sources = list((ROOT / "src" / "authinv").rglob("*.py")) + list((ROOT / "scripts").glob("authinv_*.py"))
    for path in sources:
        bad = _imported_roots(path) & set(FORBIDDEN_IMPORTS)
        assert not bad, f"{path} imports {bad}"


def test_registry_is_pinned_and_consistent():
    reg = models.load_registry()
    for key, spec in reg["models"].items():
        assert len(spec["revision"]) == 40 and int(spec["revision"], 16) >= 0, key
        if "weights_from" in spec:
            assert len(spec["weights_from"]["revision"]) == 40, key
            manifest = json.loads((models.REGISTRY.parent / spec["upstream_manifest"]).read_text())
            assert manifest["repo"] == spec["name"] and manifest["revision"] == spec["revision"], key
    assert {"qwen3_8b", "qwen3_32b", "gemma3_27b", "llama3_3_70b_int8"} == set(reg["sets"]["phase0"])


def test_git_blob_sha1_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert models.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_verify_model_dir(tmp_path):
    (tmp_path / "config.json").write_bytes(b"{}\n")
    (tmp_path / "w.safetensors").write_bytes(b"weights")
    manifest = {
        "repo": "org/m",
        "revision": "0" * 40,
        "files": {
            "config.json": {"git_blob_sha1": models.git_blob_sha1(b"{}\n"), "size": 3},
            "w.safetensors": {"sha256": models.file_sha256(tmp_path / "w.safetensors"), "size": 7},
            "README.md": {"git_blob_sha1": "x", "size": 1},
        },
    }
    assert models.verify_model_dir(tmp_path, manifest)["passed"]
    (tmp_path / "chat_template.jinja").write_text("mirror template")
    result = models.verify_model_dir(tmp_path, manifest)
    assert not result["passed"] and result["extra_files"] == ["chat_template.jinja"]
    (tmp_path / "chat_template.jinja").unlink()
    (tmp_path / "config.json").write_bytes(b"{ }\n")
    assert models.verify_model_dir(tmp_path, manifest)["problems"] == ["blob mismatch config.json"]
