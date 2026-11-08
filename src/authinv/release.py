"""Benchmark release builder (D1.1, RESEARCH_PLAN §6 deliverable 1): the Hugging Face Datasets layout.

Layout written under the output directory (format `RELEASE_FORMAT`):

    README.md                  dataset card: HF YAML header (license, configs) + the datasheet
    LICENSE                    the release license text
    NOTICE                     the release notice, with every upstream notice and copyright line
    licenses/<source>/...      each upstream source's own license files, byte for byte
    data/policies.jsonl        one row per world: canonical orig and swap policies, swap map, license
    data/renderings.jsonl      one row per (world, assignment, rendering): the exact rendered text
    data/requests.jsonl        one row per (world, assignment, instance): request, label, owners
    data/equivalence.jsonl     the certification proofs, as built (P1.10, P1.13)
    data/application.jsonl     the frozen T2 application rows (prompts), with license
    data/interpretation.jsonl  the frozen T2 interpretation rows (prompts), with license
    MANIFEST.json              counts, versions, source pins, sha256 of every file

The builder refuses unless:
- every world comes from a source whose license is known and compatible with the release
  license (an unlicensed source such as ACRE never ships);
- every upstream license file matches its pinned sha256;
- every world and assignment has a passed proof whose `policy_sha256` is the shipped policy's;
- every rendering equals the policy block of the frozen prompts, and every request read
  back from a prompt reproduces its row's label and pair_key.

Publishing is Jerzy's decision. Nothing here uploads anything.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from authinv import provenance
from authinv.benchmark import PROMPT_VERSION
from authinv.mitigate.prompts import pair_key, request_of, split_prompt
from authinv.policy import decision_owners, evaluate, from_dict, policy_hash
from authinv.render.renderers import RENDERINGS, VERSIONS, render

RELEASE_FORMAT = "authinv-release-v1"
# release license -> source licenses that may be redistributed under it (with their notices kept)
COMPATIBLE = {"Apache-2.0": {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT"}}
HF_LICENSE_IDS = {"Apache-2.0": "apache-2.0"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_licenses(cfg: dict, root: Path) -> dict[str, dict]:
    """source -> {"license", "files": {name: path}, "pin"}; upstream files checked against their pins."""
    out = {}
    for name, s in cfg["sources"].items():
        if s.get("license") == "release":
            out[name] = {"license": cfg["release_license"], "files": {}, "pin": "this project"}
            continue
        pin = yaml.safe_load((root / s["pin_config"]).read_text())
        files = {}
        for fname in s.get("license_files", []):
            path = root / s["raw_dir"] / fname
            want = pin.get("file_sha256", {}).get(fname)
            if want is None or not path.exists() or sha256(path) != want:
                raise SystemExit(
                    f"{name}: {fname} missing or not the pinned file (run the source's fetch script)"
                )
            files[fname] = path
        out[name] = {"license": pin.get("license"), "files": files, "pin": f"{pin['repo']}@{pin['commit']}"}
    return out


def check_licenses(used: set[str], licenses: dict[str, dict], release_license: str) -> None:
    allowed = COMPATIBLE.get(release_license)
    if allowed is None:
        raise SystemExit(f"release license {release_license} has no compatibility table")
    bad = {
        s: licenses.get(s, {}).get("license")
        for s in used
        if licenses.get(s, {}).get("license") not in allowed
    }
    if bad:
        raise SystemExit(f"sources without a license compatible with {release_license}: {bad}")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_jsonl(path, rows)


def notice(cfg: dict, licenses: dict[str, dict]) -> str:
    parts = [cfg["notice_header"].rstrip(), ""]
    for name, info in sorted(licenses.items()):
        if not info["files"]:
            continue
        parts.append(
            f"This release includes material derived from {name} ({info['pin']}), {info['license']}."
        )
        for fname, path in sorted(info["files"].items()):
            if fname == "NOTICE":
                parts += ["Its NOTICE reads:", "", path.read_text().rstrip(), ""]
            elif fname == "LICENSE" and info["license"].startswith("BSD"):
                parts += [f"Its copyright notice: {path.read_text().splitlines()[0].strip()}", ""]
        parts += [f"Its license is in licenses/{name}/.", ""]
    return "\n".join(parts).rstrip() + "\n"


def card(cfg: dict, counts: dict, datasheet: str) -> str:
    configs = [
        {"config_name": name, "data_files": [{"split": "test", "path": f"data/{name}.jsonl"}]}
        for name in ("application", "interpretation", "policies", "renderings", "requests", "equivalence")
    ]
    header = {
        "license": HF_LICENSE_IDS.get(cfg["release_license"], "other"),
        "pretty_name": cfg["pretty_name"],
        "language": ["en"],
        "size_categories": ["100K<n<1M"],
        "task_categories": ["text-classification"],
        "tags": [
            "authorization",
            "access-control",
            "cedar",
            "rego",
            "llm-evaluation",
            "representation-invariance",
        ],
        "configs": configs,
    }
    body = [
        f"# {cfg['pretty_name']}",
        "",
        "Engine-certified equivalent renderings of authorization policies, to measure whether an LLM's",
        "decision depends on the policy's format: worst-case accuracy over renderings and deny→allow flips.",
        "",
        f"- worlds: {counts['worlds']:,}; application rows: {counts['application']:,}; "
        f"interpretation rows: {counts['interpretation']:,}",
        "- renderings: "
        + ", ".join(f"{g} ({VERSIONS[g]})" for g in RENDERINGS)
        + f"; prompt {PROMPT_VERSION}",
        f"- frozen benchmark sha256: `{cfg['benchmark_sha256']}`",
        "- every row carries its source's license; see LICENSE, NOTICE and licenses/.",
        "",
        "Evaluate a model with the leaderboard script of the authinv repository",
        "(`scripts/authinv_leaderboard.py`), which reports the worst-case table.",
        "",
        "---",
        "",
        datasheet.rstrip(),
        "",
    ]
    return "---\n" + yaml.safe_dump(header, sort_keys=False) + "---\n\n" + "\n".join(body)


def build_release(benchmark: Path, cfg: dict, root: Path, out: Path, datasheet: str) -> dict:
    from authinv.eval.dataset import load_dataset

    rows, validation = load_dataset(benchmark)
    if validation["manifest"]["sha256"] != cfg["benchmark_sha256"]:
        raise SystemExit(
            f"benchmark {validation['manifest']['sha256']} is not the release's {cfg['benchmark_sha256']}"
        )
    worlds = {w["world_id"]: w for w in provenance.read_jsonl(benchmark / "worlds.jsonl")}
    licenses = source_licenses(cfg, root)
    check_licenses({w["source"] for w in worlds.values()}, licenses, cfg["release_license"])
    proofs = {
        (p["policy_id"], p["assignment"]): p for p in provenance.read_jsonl(benchmark / "equivalence.jsonl")
    }
    policies, pols = [], {}
    for wid, w in sorted(worlds.items()):
        hashes = {}
        for a in ("orig", "swap"):
            pol = from_dict(w[a])
            pols[(wid, a)] = pol
            proof = proofs.get((wid, a))
            if not proof or not proof["passed"] or proof["policy_sha256"] != policy_hash(pol):
                raise SystemExit(f"{wid}/{a}: no passed equivalence proof for the shipped policy")
            hashes[a] = proof["policy_sha256"]
        policies.append(
            {
                "world_id": wid,
                "source": w["source"],
                "source_kind": w["source_kind"],
                "license": licenses[w["source"]]["license"],
                "principal_types": w["principal_types"],
                "resource_types": w["resource_types"],
                "orig": w["orig"],
                "swap": w["swap"],
                "swap_map": w["swap_map"],
                "policy_sha256": hashes,
            }
        )
    cells, requests = {}, {}
    for r in rows:
        key = (r["world_id"], r["assignment"], r["rendering"])
        if key not in cells:
            cells[key] = split_prompt(r["prompt"], r["task"])[1]
        if r["task"] == "application":
            pol = pols[(r["world_id"], r["assignment"])]
            q = request_of(r, pol)
            if (
                evaluate(pol, q)["decision"] != r["label"]
                or pair_key(q.action, q.resource, q.context) != r["pair_key"]
            ):
                raise SystemExit(f"{r['row_id']}: request read back does not match the row")
            pt = tuple(worlds[r["world_id"]]["principal_types"])
            requests.setdefault(
                (r["world_id"], r["assignment"], r["instance"]),
                {
                    "world_id": r["world_id"],
                    "assignment": r["assignment"],
                    "instance": r["instance"],
                    "principal": q.principal.key(),
                    "action": q.action,
                    "resource": q.resource.key(),
                    "context": dict(q.context),
                    "label": r["label"],
                    "boundary": r["boundary"],
                    "pair_key": r["pair_key"],
                    "decision_owners": [
                        e.id for e in decision_owners(pol, q.action, q.resource, q.context, pt)
                    ],
                },
            )
    renderings = []
    for (wid, a, g), text in sorted(cells.items()):
        if render(pols[(wid, a)], g).rstrip() != text:
            raise SystemExit(f"{wid}/{a}/{g}: rendering differs from the frozen prompts")
        renderings.append(
            {"world_id": wid, "assignment": a, "rendering": g, "renderer_version": VERSIONS[g], "text": text}
        )
    out.mkdir(parents=True, exist_ok=False)
    lic = {w["world_id"]: w["license"] for w in policies}
    _write(out / "data/policies.jsonl", policies)
    _write(out / "data/renderings.jsonl", renderings)
    _write(out / "data/requests.jsonl", [requests[k] for k in sorted(requests)])
    for task in ("application", "interpretation"):
        _write(
            out / f"data/{task}.jsonl",
            [{**r, "license": lic[r["world_id"]]} for r in rows if r["task"] == task],
        )
    shutil.copyfile(benchmark / "equivalence.jsonl", out / "data/equivalence.jsonl")
    for name, info in licenses.items():
        for fname, path in info["files"].items():
            (out / "licenses" / name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out / "licenses" / name / fname)
    lic_file = root / cfg["release_license_file"]
    if sha256(lic_file) != cfg["release_license_sha256"]:
        raise SystemExit("release license file is not the pinned text")
    shutil.copyfile(lic_file, out / "LICENSE")
    (out / "NOTICE").write_text(notice(cfg, licenses))
    counts = {
        "worlds": len(policies),
        "application": sum(r["task"] == "application" for r in rows),
        "interpretation": sum(r["task"] == "interpretation" for r in rows),
        "renderings": len(renderings),
        "requests": len(requests),
        "proofs": len(proofs),
        "worlds_by_source": dict(sorted(Counter(w["source"] for w in policies).items())),
        "worlds_by_license": dict(sorted(Counter(w["license"] for w in policies).items())),
    }
    (out / "README.md").write_text(card(cfg, counts, datasheet))
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {
        "format": RELEASE_FORMAT,
        "release": cfg["release"],
        "version": cfg["version"],
        "release_license": cfg["release_license"],
        "benchmark_sha256": cfg["benchmark_sha256"],
        "prompt_version": PROMPT_VERSION,
        "renderer_versions": VERSIONS,
        "sources": {k: {"license": v["license"], "pin": v["pin"]} for k, v in licenses.items()},
        "counts": counts,
        "files": files,
        "git": provenance.git_state(),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_release(out: Path) -> dict:
    """Re-hash every file against MANIFEST.json (what the leaderboard checks before reading a release)."""
    manifest = json.loads((out / "MANIFEST.json").read_text())
    bad = [f for f, h in manifest["files"].items() if not (out / f).exists() or sha256(out / f) != h]
    if bad:
        raise SystemExit(f"release files differ from MANIFEST.json: {bad[:5]}")
    return manifest
