"""Engine-certified equivalence of renderings (RESEARCH_PLAN §3 Phase 1, task 3).

A world ships only if every rendering passes every check:

1. **decodes**: the rendering decodes back to the canonical rules (normal form);
2. **decisions**: the decoded rules give the reference decision on *every*
   request in the universe. This is semantic, not just structural;
3. **engine**: the Cedar engine, run on the exact executable text shown to the
   model, gives the reference decision on every request, with no errors;
4. **typecheck**: Cedar's strict validator accepts the executable text
   against the generated schema.

`certify_texts` takes the rendered texts explicitly, so tests can feed
tampered renderings and confirm they are caught. The proof record carries
hashes of the policy, every rendering, and the labelled universe, so a shipped
world can be re-verified later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from authinv.equivalence import cedar
from authinv.equivalence.requests import request_to_dict, universe
from authinv.policy import Policy, evaluate, policy_hash, semantic_hash
from authinv.render.renderers import RENDERINGS, VERSIONS, RenderError, decode, normal_form, render

MAX_UNIVERSE = 200_000


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def certify_texts(
    policy: Policy,
    texts: dict[str, str],
    principal_types: tuple[str, ...],
    resource_types: tuple[str, ...],
) -> dict:
    reqs = universe(policy, principal_types, resource_types)
    if len(reqs) > MAX_UNIVERSE:
        raise ValueError(f"universe of {len(reqs)} requests exceeds {MAX_UNIVERSE}; shrink the world")
    truth = [evaluate(policy, r)["decision"] for r in reqs]
    want_nf = normal_form(policy.rules)
    checks: dict[str, dict] = {}
    for rendering, text in sorted(texts.items()):
        c: dict = {"version": VERSIONS[rendering], "sha256": _sha(text)}
        try:
            rules = decode(text, rendering)
            c["decodes"] = normal_form(rules) == want_nf
            decoded = Policy(
                policy.policy_id, policy.entities, policy.actions, tuple(rules), policy.context_schema
            )
            got = [evaluate(decoded, r)["decision"] for r in reqs]
            c["decision_mismatches"] = sum(a != b for a, b in zip(got, truth, strict=True))
        except (RenderError, KeyError, ValueError) as e:
            c["decodes"], c["decision_mismatches"], c["error"] = False, None, repr(e)
        if rendering == "executable":
            engine = cedar.cedar_decisions(policy, reqs, policy_text=text)
            c["engine_mismatches"] = sum(e["decision"] != t for e, t in zip(engine, truth, strict=True))
            c["engine_errors"] = sum(bool(e["errors"]) for e in engine)
            v = cedar.validate_with_schema_text(policy, text, principal_types, resource_types)
            c["typecheck"] = v["passed"]
            if v["errors"]:
                c["typecheck_errors"] = v["errors"][:5]
        c["passed"] = (
            c["decodes"]
            and c["decision_mismatches"] == 0
            and (
                rendering != "executable"
                or (c["engine_mismatches"] == 0 and c["engine_errors"] == 0 and c["typecheck"])
            )
        )
        checks[rendering] = c
    universe_doc = [{**request_to_dict(r), "decision": d} for r, d in zip(reqs, truth, strict=True)]
    return {
        "policy_id": policy.policy_id,
        "policy_sha256": policy_hash(policy),
        "semantic_sha256": semantic_hash(policy),
        "principal_types": list(principal_types),
        "resource_types": list(resource_types),
        "n_requests": len(reqs),
        "n_allow": truth.count("allow"),
        "universe_sha256": _sha(json.dumps(universe_doc, sort_keys=True)),
        "engine": {"name": "cedar", "cedarpy": _cedarpy_version()},
        "renderings": checks,
        "passed": set(checks) == set(RENDERINGS) and all(c["passed"] for c in checks.values()),
    }


def certify(policy: Policy, principal_types: tuple[str, ...], resource_types: tuple[str, ...]) -> dict:
    texts = {r: render(policy, r) for r in RENDERINGS}
    return certify_texts(policy, texts, principal_types, resource_types)


def write_proofs(records: list[dict], path: Path) -> str:
    """equivalence.jsonl (one proof per world); returns the file's sha256."""
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    path.write_text(data)
    return _sha(data)


def _cedarpy_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("cedarpy")
    except Exception:  # noqa: BLE001 - provenance only
        return None
