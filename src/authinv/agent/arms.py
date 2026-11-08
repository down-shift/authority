"""Phase-3 T3 arms built on a model's own arm-1 conversions (P3.2b; docs/prereg/phase3.md decision 4).

Both arms read a complete canonicalization run (canon-v1, `scripts/authinv_canonicalize.py`) of the
same model on the same benchmark, and use the conversion of each episode's (world, assignment,
rendering) cell. Everything else is the agent-v4 protocol, prompt, tools and limits.

- `canon_t3` (canon-t3-v1): the agent's system prompt carries the cell's predicted IR, canonically
  serialized (`canonicalize.ir_text`, the text T2 stage 2 uses), in place of the rendering. A
  perfect conversion makes the prompt byte-identical to the agent-v4 json_policy episode. A cell
  whose conversion did not parse as an IR gets no episode: status `conversion_failure`, nothing is
  executed, and it counts as a protocol failure (its own category; never a refusal or a violation),
  as an unparsed conversion is a parse failure in T2. A parsed but invalid IR is still shown, as in
  T2 stage 2.
- `solver_proxy` (solver-proxy-v1): the agent sees the rendering, as in agent-v4, and an enforcement
  proxy blocks every call that the cell's predicted IR denies (reference evaluator). The proxy's
  observation is proxy-v1's. An unusable IR (not parsed, or invalid) blocks every call
  (fail-closed). UIR counts executed calls the *certified* policy denies, so an IR that wrongly
  allows lets real violations through; `wrongly_blocked` counts calls the certified policy allows
  that the proxy blocked.

Never change either arm after seeing results: bump its version and log it in docs/CHANGELOG.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from authinv import provenance
from authinv.agent.sandbox import Episode
from authinv.mitigate import canonicalize as canon

CANON_T3_VERSION = "canon-t3-v1"
SOLVER_PROXY_VERSION = "solver-proxy-v1"
ARM_VERSIONS = {"canon_t3": CANON_T3_VERSION, "solver_proxy": SOLVER_PROXY_VERSION}


def cell_id(ep: Episode) -> str:
    """The canonicalization case id of an episode's (world, assignment, rendering) cell."""
    return f"{ep.world_id}/{ep.assignment}/{ep.rendering}"


def load_conversions(run: Path, spec: dict, dataset_sha: str) -> tuple[dict[str, dict], dict]:
    """The conversions of a complete arm-1 run of this model on this benchmark, by case id."""
    conf = json.loads((run / "config.json").read_text())
    status = json.loads((run / "run_status.json").read_text())["status"]
    problems = []
    if status != "complete":
        problems.append(f"status {status}")
    if conf.get("model") != spec:
        problems.append("model spec differs")
    if conf.get("dataset_sha256") != dataset_sha:
        problems.append("dataset differs")
    if conf.get("canon_version") != canon.CANON_VERSION:
        problems.append(f"canon version {conf.get('canon_version')}")
    if problems:
        raise SystemExit(f"canonicalization run {run} refused: {problems}")
    path = run / "conversions.jsonl"
    convs = {c["case_id"]: c for c in provenance.read_jsonl(path)}
    return convs, {
        "canon_run": str(run),
        "canon_config_sha256": conf.get("config_sha256"),
        "conversions_sha256": provenance.sha256_file(path),
    }


def apply_arm(episodes: list[Episode], arm: str, convs: dict[str, dict]) -> dict:
    """Set up each episode for `arm` from its cell's conversion; returns counts of the IR states."""
    if arm not in ARM_VERSIONS:
        raise ValueError(f"unknown T3 arm {arm!r}")
    missing = sorted({cell_id(ep) for ep in episodes} - set(convs))
    if missing:
        raise SystemExit(f"{len(missing)} cells have no conversion, e.g. {missing[0]}")
    counts = {"episodes": len(episodes), "no_ir": 0, "ir_invalid": 0, "ir_valid": 0}
    for ep in episodes:
        c = convs[cell_id(ep)]
        parsed = c["conv_category"] == "ir"
        usable = parsed and bool(c["ir_valid"])
        counts["ir_valid" if usable else "ir_invalid" if parsed else "no_ir"] += 1
        ep.arm = arm
        ep.meta = {
            **ep.meta,
            "conv_category": c["conv_category"],
            "ir_valid": bool(c["ir_valid"]),
            "ir_exact": bool(c["ir_exact"]),
            "ir_equivalent": bool(c["ir_equivalent"]),
        }
        if arm == "canon_t3":
            if parsed:
                ep.rendering_text = c["ir_text"]
            else:
                ep.done, ep.status = True, "conversion_failure"
        else:
            ep.enforce = True
            if usable:
                ep.enforce_policy = canon.ir_policy(ep.policy, c["ir_text"])
            else:
                ep.enforce_fail_closed = True
    return counts
