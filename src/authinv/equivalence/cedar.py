"""Canonical policy → Cedar (policies, entities, schema) and evaluation with the real Cedar engine.

`cedar_policy_text` is the exact text the executable rendering shows the
model, formatted by Cedar's own formatter. Equivalence is certified by
evaluating that same text with `cedarpy`, which embeds the Rust cedar-policy
engine, and comparing against the reference semantics in `authinv.policy`.
`validate_with_schema` also has Cedar's typechecker accept the policies in
strict mode against a schema generated from the entities.

Requires the `engines` extra (`cedarpy`). It is imported lazily.
"""

from __future__ import annotations

import json
from typing import Any

from authinv.policy import AttrRef, Condition, EntityRef, Policy, Request, Rule, Scope

_CEDAR_TYPE = {"str": "String", "int": "Long", "bool": "Boolean"}


def _uid(ref: EntityRef) -> str:
    return f"{ref.type}::{json.dumps(ref.id)}"


def _lit(v: str | int | bool) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return json.dumps(v)


def _operand(x: AttrRef | str | int | bool) -> str:
    return f"{x.subject}.{x.attr}" if isinstance(x, AttrRef) else _lit(x)


def _scope(var: str, s: Scope) -> str:
    if s.kind == "any":
        return var
    if s.kind == "eq":
        return f"{var} == {_uid(s.entity)}"
    if s.kind == "is":
        return f"{var} is {s.type}"
    if s.kind == "in":
        return f"{var} in {_uid(s.entity)}"
    return f"{var} is {s.type} in {_uid(s.entity)}"


def _actions(actions: tuple[str, ...]) -> str:
    acts = sorted(actions)
    if len(acts) == 1:
        return f"action == Action::{json.dumps(acts[0])}"
    return "action in [" + ", ".join(f"Action::{json.dumps(a)}" for a in acts) + "]"


def _condition(c: Condition) -> str:
    return f"{_operand(c.left)} {c.op} {_operand(c.right)}"


def _rule(r: Rule) -> str:
    head = f"@id({json.dumps(r.rule_id)})\n{r.effect} ("
    head += f"{_scope('principal', r.principal)}, {_actions(r.actions)}, "
    head += f"{_scope('resource', r.resource)})"
    if r.conditions:
        head += " when { " + " && ".join(_condition(c) for c in r.conditions) + " }"
    return head + ";"


def cedar_policy_text(policy: Policy) -> str:
    """Deterministic Cedar source (rules sorted by id), normalized by Cedar's formatter."""
    import cedarpy

    raw = "\n\n".join(_rule(r) for r in sorted(policy.rules, key=lambda r: r.rule_id))
    return cedarpy.format_policies(raw).strip() + "\n"


def cedar_entities(policy: Policy) -> list[dict]:
    return [
        {
            "uid": {"type": e.ref.type, "id": e.ref.id},
            "attrs": dict(e.attrs),
            "parents": [{"type": p.type, "id": p.id} for p in e.parents],
        }
        for e in sorted(policy.entities, key=lambda e: e.ref.key())
    ]


def cedar_schema(policy: Policy, principal_types: tuple[str, ...], resource_types: tuple[str, ...]) -> dict:
    """Cedar JSON schema: entity types (required attributes, membership); actions over the given types."""
    attrs: dict[str, dict] = {}
    member_of: dict[str, set[str]] = {}
    for e in policy.entities:
        attrs.setdefault(e.ref.type, {k: {"type": _CEDAR_TYPE[_vt(v)], "required": True} for k, v in e.attrs})
        member_of.setdefault(e.ref.type, set()).update(p.type for p in e.parents)
    ctx = {k: {"type": _CEDAR_TYPE[t], "required": True} for k, t in policy.context_schema}
    return {
        "": {
            "entityTypes": {
                t: {"memberOfTypes": sorted(member_of[t]), "shape": {"type": "Record", "attributes": a}}
                for t, a in sorted(attrs.items())
            },
            "actions": {
                a: {
                    "appliesTo": {
                        "principalTypes": list(principal_types),
                        "resourceTypes": list(resource_types),
                        "context": {"type": "Record", "attributes": ctx},
                    }
                }
                for a in sorted(policy.actions)
            },
        }
    }


def _vt(v: str | int | bool) -> str:
    return "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else "str"


def validate_with_schema(
    policy: Policy, principal_types: tuple[str, ...], resource_types: tuple[str, ...]
) -> dict:
    """Cedar's strict typechecker on the rendered policies; {"passed", "errors"}."""
    return validate_with_schema_text(policy, cedar_policy_text(policy), principal_types, resource_types)


def validate_with_schema_text(
    policy: Policy, text: str, principal_types: tuple[str, ...], resource_types: tuple[str, ...]
) -> dict:
    """Typecheck the given Cedar text (e.g. the exact executable rendering) against the generated schema."""
    import cedarpy

    result = cedarpy.validate_policies(
        text, json.dumps(cedar_schema(policy, principal_types, resource_types))
    )
    errors = [str(e) for e in (result.errors or [])]
    return {"passed": bool(result.validation_passed) and not errors, "errors": errors}


def _cedar_request(req: Request) -> dict[str, Any]:
    return {
        "principal": _uid(req.principal),
        "action": f"Action::{json.dumps(req.action)}",
        "resource": _uid(req.resource),
        "context": dict(req.context),
    }


def cedar_decisions(policy: Policy, requests: list[Request], policy_text: str | None = None) -> list[dict]:
    """Evaluate requests with the Cedar engine. Any engine error is returned, never swallowed."""
    import cedarpy

    text = policy_text if policy_text is not None else cedar_policy_text(policy)
    results = cedarpy.is_authorized_batch([_cedar_request(r) for r in requests], text, cedar_entities(policy))
    out = []
    for res in results:
        errors = [str(e) for e in (res.diagnostics.errors or [])]
        out.append({"decision": "allow" if res.allowed else "deny", "errors": errors})
    return out
