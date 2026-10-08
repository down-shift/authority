"""Canonical authorization policy: the single source of truth every rendering derives from.

Semantics follow Cedar's: a request is allowed iff at least one `permit` rule
matches and no `forbid` rule matches (default deny, deny-overrides). A rule
matches when its principal scope, action set, resource scope, and every
condition (a conjunction) hold.

Every attribute a condition reads must be declared for the entity types it can
reach (`validate`). That removes the missing-attribute cases where Cedar, Rego,
and IAM disagree, so engine-certified equivalence is well defined.

Decision ownership ("who decides X") is not stored. It is derived from the
rules by `decision_owners`, so it can never disagree with them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Value = str | int | bool
EFFECTS = ("permit", "forbid")
SCOPE_KINDS = ("any", "eq", "is", "in", "is_in")
OPS = ("==", "!=", "<", "<=", ">", ">=")
SUBJECTS = ("principal", "resource", "context")


@dataclass(frozen=True)
class EntityRef:
    type: str
    id: str

    def key(self) -> str:
        return f"{self.type}::{self.id}"


@dataclass(frozen=True)
class Entity:
    ref: EntityRef
    attrs: tuple[tuple[str, Value], ...] = ()
    parents: tuple[EntityRef, ...] = ()  # group / role / container membership (transitive)

    def attr(self, name: str) -> Value:
        return dict(self.attrs)[name]


@dataclass(frozen=True)
class Scope:
    """Scope: any, == entity, is <type>, in <group> (transitive), or is <type> in <group>."""

    kind: Literal["any", "eq", "is", "in", "is_in"] = "any"
    entity: EntityRef | None = None  # for eq / in / is_in
    type: str | None = None  # for is / is_in


@dataclass(frozen=True)
class AttrRef:
    subject: Literal["principal", "resource", "context"]
    attr: str


@dataclass(frozen=True)
class Condition:
    left: AttrRef
    op: Literal["==", "!=", "<", "<=", ">", ">="]
    right: AttrRef | Value


@dataclass(frozen=True)
class Rule:
    rule_id: str
    effect: Literal["permit", "forbid"]
    principal: Scope
    actions: tuple[str, ...]
    resource: Scope
    conditions: tuple[Condition, ...] = ()


@dataclass(frozen=True)
class Request:
    principal: EntityRef
    action: str
    resource: EntityRef
    context: tuple[tuple[str, Value], ...] = ()


@dataclass(frozen=True)
class Policy:
    policy_id: str
    entities: tuple[Entity, ...]
    actions: tuple[str, ...]
    rules: tuple[Rule, ...]
    context_schema: tuple[tuple[str, str], ...] = ()  # (key, "str" | "int" | "bool")
    meta: tuple[tuple[str, Value], ...] = field(default=(), compare=False)

    # ---- lookup -------------------------------------------------------------
    def entity(self, ref: EntityRef) -> Entity:
        for e in self.entities:
            if e.ref == ref:
                return e
        raise KeyError(ref.key())

    def ancestors(self, ref: EntityRef) -> set[EntityRef]:
        seen: set[EntityRef] = set()
        stack = list(self.entity(ref).parents)
        while stack:
            p = stack.pop()
            if p not in seen:
                seen.add(p)
                stack.extend(self.entity(p).parents)
        return seen


class PolicyError(ValueError):
    pass


_TYPE_OF = {"str": str, "int": int, "bool": bool}


def _value_type(v: Value) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    raise PolicyError(f"unsupported value {v!r}")


def _attr_schema(policy: Policy) -> dict[str, dict[str, str]]:
    """type -> {attr: value type}; every entity of a type must declare the same attributes."""
    schema: dict[str, dict[str, str]] = {}
    for e in policy.entities:
        decl = {k: _value_type(v) for k, v in e.attrs}
        if e.ref.type in schema and schema[e.ref.type] != decl:
            raise PolicyError(f"entities of type {e.ref.type!r} declare different attributes")
        schema.setdefault(e.ref.type, decl)
    return schema


def _scope_types(policy: Policy, scope: Scope) -> set[str]:
    """Entity types a scope can match (what its conditions may read)."""
    if scope.kind == "eq":
        return {scope.entity.type}
    if scope.kind in ("is", "is_in"):
        return {scope.type}
    if scope.kind == "in":
        return {
            e.ref.type
            for e in policy.entities
            if scope.entity in policy.ancestors(e.ref) or e.ref == scope.entity
        }
    return {e.ref.type for e in policy.entities}


def validate(policy: Policy) -> None:
    """Raise PolicyError unless the policy is well formed and every condition is total."""
    refs = [e.ref for e in policy.entities]
    if len(set(refs)) != len(refs):
        raise PolicyError("duplicate entity")
    known = set(refs)
    for e in policy.entities:
        for p in e.parents:
            if p not in known:
                raise PolicyError(f"{e.ref.key()} has unknown parent {p.key()}")
    for e in policy.entities:  # membership must be acyclic
        if e.ref in policy.ancestors(e.ref):
            raise PolicyError(f"membership cycle through {e.ref.key()}")
    schema = _attr_schema(policy)
    ctx = dict(policy.context_schema)
    if not policy.rules:
        raise PolicyError("policy has no rules")
    if len({r.rule_id for r in policy.rules}) != len(policy.rules):
        raise PolicyError("duplicate rule_id")
    for r in policy.rules:
        if r.effect not in EFFECTS:
            raise PolicyError(f"{r.rule_id}: bad effect {r.effect!r}")
        if not r.actions or not set(r.actions) <= set(policy.actions):
            raise PolicyError(f"{r.rule_id}: actions must be a non-empty subset of {policy.actions}")
        for side, scope in (("principal", r.principal), ("resource", r.resource)):
            if scope.kind not in SCOPE_KINDS:
                raise PolicyError(f"{r.rule_id}: bad {side} scope {scope.kind!r}")
            if scope.kind in ("eq", "in", "is_in") and scope.entity not in known:
                raise PolicyError(f"{r.rule_id}: unknown {side} entity")
            if scope.kind in ("is", "is_in") and scope.type not in schema:
                raise PolicyError(f"{r.rule_id}: unknown {side} type {scope.type!r}")
        reach = {"principal": _scope_types(policy, r.principal), "resource": _scope_types(policy, r.resource)}
        for c in r.conditions:
            if c.op not in OPS:
                raise PolicyError(f"{r.rule_id}: bad op {c.op!r}")
            types = set()
            for ref in [c.left] + ([c.right] if isinstance(c.right, AttrRef) else []):
                if ref.subject == "context":
                    if ref.attr not in ctx:
                        raise PolicyError(f"{r.rule_id}: undeclared context key {ref.attr!r}")
                    types.add(ctx[ref.attr])
                    continue
                for t in reach[ref.subject]:
                    if ref.attr not in schema.get(t, {}):
                        raise PolicyError(
                            f"{r.rule_id}: {ref.subject} of type {t!r} lacks attribute {ref.attr!r}"
                        )
                    types.add(schema[t][ref.attr])
            if not isinstance(c.right, AttrRef):
                types.add(_value_type(c.right))
            if len(types) != 1:
                raise PolicyError(f"{r.rule_id}: condition compares mismatched types {sorted(types)}")
            if c.op in ("<", "<=", ">", ">=") and types != {"int"}:
                raise PolicyError(f"{r.rule_id}: ordering comparison on non-integers")


# ---- evaluation (reference semantics) -----------------------------------------


def _in_scope(policy: Policy, scope: Scope, ref: EntityRef) -> bool:
    if scope.kind == "any":
        return True
    if scope.kind == "eq":
        return ref == scope.entity
    if scope.kind == "is":
        return ref.type == scope.type
    member = ref == scope.entity or scope.entity in policy.ancestors(ref)
    return member and (scope.kind == "in" or ref.type == scope.type)


def _resolve(policy: Policy, x: AttrRef | Value, req: Request) -> Value:
    if not isinstance(x, AttrRef):
        return x
    if x.subject == "context":
        return dict(req.context)[x.attr]
    ent = policy.entity(req.principal if x.subject == "principal" else req.resource)
    return ent.attr(x.attr)


_CMP = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def rule_matches(policy: Policy, rule: Rule, req: Request) -> bool:
    return (
        _in_scope(policy, rule.principal, req.principal)
        and req.action in rule.actions
        and _in_scope(policy, rule.resource, req.resource)
        and all(
            _CMP[c.op](_resolve(policy, c.left, req), _resolve(policy, c.right, req)) for c in rule.conditions
        )
    )


def evaluate(policy: Policy, req: Request) -> dict[str, Any]:
    """Reference decision: {"decision": "allow"|"deny", "permits": [...], "forbids": [...]}."""
    matched = [r for r in policy.rules if rule_matches(policy, r, req)]
    permits = sorted(r.rule_id for r in matched if r.effect == "permit")
    forbids = sorted(r.rule_id for r in matched if r.effect == "forbid")
    return {
        "decision": "allow" if permits and not forbids else "deny",
        "permits": permits,
        "forbids": forbids,
    }


def decision_owners(
    policy: Policy,
    action: str,
    resource: EntityRef,
    context: tuple = (),
    principal_types: tuple[str, ...] | None = None,
) -> list[EntityRef]:
    """Principals allowed `action` on `resource` (the derived 'who decides' view).

    `in` scopes follow Cedar: `principal in G` also holds for G itself, so pass
    `principal_types` (e.g. ("User",)) to list only real principals.
    """
    return sorted(
        (
            e.ref
            for e in policy.entities
            if (principal_types is None or e.ref.type in principal_types)
            and evaluate(policy, Request(e.ref, action, resource, context))["decision"] == "allow"
        ),
        key=EntityRef.key,
    )


def tier(policy: Policy) -> str:
    """Difficulty tier (RESEARCH_PLAN §3 Phase 1): single / multi_rule / conditioned."""
    if any(r.conditions for r in policy.rules):
        return "conditioned"
    if len(policy.rules) > 1:
        return "multi_rule"
    return "single"


# ---- canonical serialization ---------------------------------------------------


def _ref(r: EntityRef | None) -> dict | None:
    return None if r is None else {"type": r.type, "id": r.id}


def _scope(s: Scope) -> dict:
    return {"kind": s.kind, "entity": _ref(s.entity), "type": s.type}


def _operand(x: AttrRef | Value) -> dict:
    if isinstance(x, AttrRef):
        return {"attr": {"subject": x.subject, "attr": x.attr}}
    return {"value": x, "value_type": _value_type(x)}


def to_dict(policy: Policy) -> dict:
    """Canonical form: entities, actions, and rules sorted; order carries no meaning."""
    return {
        "policy_id": policy.policy_id,
        "actions": sorted(policy.actions),
        "context_schema": sorted([k, t] for k, t in policy.context_schema),
        "entities": sorted(
            (
                {
                    "ref": _ref(e.ref),
                    "attrs": sorted([k, v] for k, v in e.attrs),
                    "parents": sorted((_ref(p) for p in e.parents), key=lambda d: (d["type"], d["id"])),
                }
                for e in policy.entities
            ),
            key=lambda d: (d["ref"]["type"], d["ref"]["id"]),
        ),
        "rules": sorted(
            (
                {
                    "rule_id": r.rule_id,
                    "effect": r.effect,
                    "principal": _scope(r.principal),
                    "actions": sorted(r.actions),
                    "resource": _scope(r.resource),
                    "conditions": [
                        {"left": _operand(c.left), "op": c.op, "right": _operand(c.right)}
                        for c in r.conditions
                    ],
                }
                for r in policy.rules
            ),
            key=lambda d: d["rule_id"],
        ),
        "meta": sorted([k, v] for k, v in policy.meta),
    }


def _unref(d: dict | None) -> EntityRef | None:
    return None if d is None else EntityRef(d["type"], d["id"])


def _unoperand(d: dict) -> AttrRef | Value:
    if "attr" in d:
        return AttrRef(d["attr"]["subject"], d["attr"]["attr"])
    v, t = d["value"], d["value_type"]
    if not isinstance(v, _TYPE_OF[t]) or (t == "int" and isinstance(v, bool)):
        raise PolicyError(f"value {v!r} is not {t}")
    return v


def from_dict(d: dict) -> Policy:
    return Policy(
        policy_id=d["policy_id"],
        actions=tuple(d["actions"]),
        context_schema=tuple((k, t) for k, t in d["context_schema"]),
        entities=tuple(
            Entity(
                _unref(e["ref"]), tuple((k, v) for k, v in e["attrs"]), tuple(_unref(p) for p in e["parents"])
            )
            for e in d["entities"]
        ),
        rules=tuple(
            Rule(
                r["rule_id"],
                r["effect"],
                Scope(r["principal"]["kind"], _unref(r["principal"]["entity"]), r["principal"]["type"]),
                tuple(r["actions"]),
                Scope(r["resource"]["kind"], _unref(r["resource"]["entity"]), r["resource"]["type"]),
                tuple(
                    Condition(_unoperand(c["left"]), c["op"], _unoperand(c["right"])) for c in r["conditions"]
                ),
            )
            for r in d["rules"]
        ),
        meta=tuple((k, v) for k, v in d["meta"]),
    )


def canonical_json(policy: Policy) -> str:
    return json.dumps(to_dict(policy), sort_keys=True, separators=(",", ":"))


def policy_hash(policy: Policy) -> str:
    return hashlib.sha256(canonical_json(policy).encode()).hexdigest()


def semantic_hash(policy: Policy) -> str:
    """Hash of the semantics only (entities, actions, context schema, rules); ignores id and meta."""
    d = to_dict(policy)
    d.pop("policy_id"), d.pop("meta")
    for r in d["rules"]:
        r.pop("rule_id")
    d["rules"].sort(key=lambda r: json.dumps(r, sort_keys=True))
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
