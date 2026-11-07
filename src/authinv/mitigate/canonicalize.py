"""Phase-3 arm 1: policy canonicalization (P3.1, RESEARCH_PLAN §3 Phase 3).

Two model calls per query, as in the pilot's E3 design (the model converts,
then the model decides):

1. **Conversion.** The model rewrites one rendering of a policy (with the
   shared facts block) as the canonical JSON IR: a ``{"rules": [...]}`` object
   in the json-v2 rule format. One conversion per (world, assignment,
   rendering) cell, shared by every request of that cell. Decoding is either
   ``constrained`` (a per-world JSON schema, `ir_schema`) or ``free``
   (unconstrained, strict parsing, so refusals stay observable).
2. **Decision.** The frozen T2 prompt with only its policy block replaced by
   the predicted IR, serialized canonically (`ir_text`: the json-v2 header,
   rules sorted by id, actions sorted). A conversion that reproduces the rules
   and their ids gives a stage-2 prompt byte-identical to the T2 json_policy
   prompt, so the arm's ceiling is the measured T2 json_policy accuracy.

The schema holds only vocabulary the prompt already shows (entity refs and
types from the facts block, the declared actions, attribute and context
names). It never encodes which rules exist.

Scoring a conversion (all against the engine-certified canonical policy):

- ``ir``: the output parses as an IR object (else refusal / abstain /
  parse_failure, kept apart and never dropped);
- ``ir_valid``: the IR is a well-formed policy over the world's entities
  (`authinv.policy.validate`; an empty rule list is valid and denies all);
- ``ir_exact``: same rules as the canonical policy up to rule ids and order
  (`normal_form`);
- ``ir_equivalent``: same decision as the canonical policy on every request of
  the world's request universe, built over both policies' context literals.

`CANON_VERSION` covers the conversion prompt, the IR format and the
serializer. Never edit them after seeing model results: bump the version and
log it in docs/CHANGELOG.md.
"""

from __future__ import annotations

import hashlib
import json

from authinv.equivalence.requests import universe
from authinv.eval.parse import _ABSTAIN, _REFUSAL, fold_typographic
from authinv.eval.structured import JSON_PARSER_VERSION, extract_object, strip_think
from authinv.policy import Policy, PolicyError, Rule, evaluate, validate
from authinv.policy.model import OPS, SUBJECTS
from authinv.render.renderers import (
    JSON_COMBINING_RULE,
    JSON_CONDITION_SEMANTICS,
    RenderError,
    decode_json,
    normal_form,
)

CANON_VERSION = "canon-v1"

CONVERSION_INTRO = (
    "You are an authorization assistant. Rewrite the policy below in a canonical JSON format. "
    "Keep its meaning exactly: every rule with its id, effect, actions, principal and resource scopes, "
    "and conditions."
)

IR_FORMAT = """\
Canonical format: one JSON object {"rules": [<rule>, ...]} with one entry per rule of the policy. Each rule is
{"id": "<rule id>", "effect": "allow" or "deny", "principal": <scope>, "actions": ["<action>", ...], \
"resource": <scope>, "conditions_mode": "all", "conditions": [<condition>, ...]}.
A scope is one of:
- {"match": "any"}: every entity;
- {"match": "exactly", "entity": {"type": "<type>", "id": "<id>"}}: that entity only;
- {"match": "any_of_type", "type": "<type>"}: every entity of that type;
- {"match": "entity_or_member_of", "entity": {"type": "<type>", "id": "<id>"}}: that entity itself or \
anything that is a member of it, directly or through other groups;
- {"match": "type_and_member_of", "type": "<type>", "entity": {"type": "<type>", "id": "<id>"}}: entities of \
that type that are that entity or members of it.
A condition is {"left": <attribute>, "op": <op>, "right": <operand>} with <op> one of "==", "!=", "<", \
"<=", ">", ">=". An <attribute> is {"attr": "principal.<name>"}, {"attr": "resource.<name>"}, or \
{"attr": "context.<key>"}; an <operand> is an <attribute> or a literal string, integer, or boolean.
A rule applies when the principal matches its principal scope, the action is one of its actions, the \
resource matches its resource scope, and all of its conditions hold. A request is allowed if at least one \
allow rule applies and no deny rule applies; otherwise it is denied."""


def conversion_prompt(facts: str, rendering_text: str) -> str:
    """Identical across renderings except for the policy text."""
    return (
        f"{CONVERSION_INTRO}\n\nFacts:\n{facts}\n\nPolicy:\n{rendering_text.rstrip()}\n\n{IR_FORMAT}\n"
        "Answer with the JSON object only."
    )


# ---- IR schema (constrained decoding) -------------------------------------------------

_MATCH_FIELDS = {
    "any": (),
    "exactly": ("entity",),
    "any_of_type": ("type",),
    "entity_or_member_of": ("entity",),
    "type_and_member_of": ("type", "entity"),
}
RULE_FIELDS = ("id", "effect", "principal", "actions", "resource", "conditions_mode", "conditions")


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def ir_schema(policy: Policy) -> dict:
    """JSON schema of the IR over this world's vocabulary (vLLM StructuredOutputsParams(json=...)).

    Property order is the json-v2 key order, so grammar-guided output follows it.
    Uses only: type, properties, required, additionalProperties, enum, const,
    anyOf, items, minItems, maxItems, $defs/$ref (see `schema_accepts`).
    """
    ents = sorted(policy.entities, key=lambda e: e.ref.key())
    attrs = sorted(
        {f"{s}.{k}" for s in ("principal", "resource") for e in ents for k, _ in e.attrs}
        | {f"context.{k}" for k, _ in policy.context_schema}
    )
    entity = {"anyOf": [_obj({"type": {"const": e.ref.type}, "id": {"const": e.ref.id}}) for e in ents]}
    kind = {"enum": sorted({e.ref.type for e in ents})}
    field = {"type": kind, "entity": {"$ref": "#/$defs/entity"}}
    scope = {
        "anyOf": [
            _obj({"match": {"const": m}, **{f: field[f] for f in fields}})
            for m, fields in _MATCH_FIELDS.items()
        ]
    }
    literal = [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}]
    defs = {"entity": entity, "scope": scope}
    if attrs:  # a condition's left side is always an attribute (canonical `Condition`)
        defs["attr"] = _obj({"attr": {"enum": attrs}})
        defs["operand"] = {"anyOf": [{"$ref": "#/$defs/attr"}, *literal]}
        cond = _obj(
            {
                "left": {"$ref": "#/$defs/attr"},
                "op": {"enum": list(OPS)},
                "right": {"$ref": "#/$defs/operand"},
            }
        )
        conditions = {"type": "array", "items": cond}
    else:  # nothing to condition on
        conditions = {"type": "array", "maxItems": 0}
    rule = _obj(
        {
            "id": {"type": "string"},
            "effect": {"enum": ["allow", "deny"]},
            "principal": {"$ref": "#/$defs/scope"},
            "actions": {"type": "array", "items": {"enum": sorted(policy.actions)}, "minItems": 1},
            "resource": {"$ref": "#/$defs/scope"},
            "conditions_mode": {"const": "all"},
            "conditions": conditions,
        }
    )
    return {
        **_obj({"rules": {"type": "array", "items": {"$ref": "#/$defs/rule"}}}),
        "$defs": {**defs, "rule": rule},
    }


_JSON_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def schema_accepts(schema: dict, value, root: dict | None = None) -> bool:
    """Check `value` against the JSON-schema subset `ir_schema` emits (no external dependency)."""
    root = root or schema
    if "$ref" in schema:
        return schema_accepts(root["$defs"][schema["$ref"].rsplit("/", 1)[1]], value, root)
    if "anyOf" in schema:
        return any(schema_accepts(s, value, root) for s in schema["anyOf"])
    if "const" in schema and not (value == schema["const"] and type(value) is type(schema["const"])):
        return False
    if "enum" in schema and not any(value == e and type(value) is type(e) for e in schema["enum"]):
        return False
    if "type" in schema and not _JSON_TYPES[schema["type"]](value):
        return False
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        if any(k not in value for k in schema.get("required", ())):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(props):
            return False
        if not all(schema_accepts(props[k], v, root) for k, v in value.items() if k in props):
            return False
    if isinstance(value, list) and len(value) < schema.get("minItems", 0):
        return False
    if isinstance(value, list) and len(value) > schema.get("maxItems", len(value)):
        return False
    items = schema.get("items")
    return not (isinstance(value, list) and items and not all(schema_accepts(items, v, root) for v in value))


def schema_sha256(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---- parsing and canonical serialization ------------------------------------------------

_BAD = object()


def _entity(d):
    if isinstance(d, dict) and set(d) == {"type", "id"} and all(isinstance(d[k], str) and d[k] for k in d):
        return {"type": d["type"], "id": d["id"]}
    return _BAD


def _scope(d):
    if not isinstance(d, dict) or d.get("match") not in _MATCH_FIELDS:
        return _BAD
    fields = _MATCH_FIELDS[d["match"]]
    if set(d) != {"match", *fields}:
        return _BAD
    out = {"match": d["match"]}
    if "type" in fields:
        if not (isinstance(d["type"], str) and d["type"]):
            return _BAD
        out["type"] = d["type"]
    if "entity" in fields:
        out["entity"] = _entity(d["entity"])
        if out["entity"] is _BAD:
            return _BAD
    return out


def _operand(x):
    if isinstance(x, dict):
        if set(x) != {"attr"} or not isinstance(x["attr"], str):
            return _BAD
        subject, dot, attr = x["attr"].partition(".")
        return {"attr": x["attr"]} if dot and attr and subject in SUBJECTS else _BAD
    return x if isinstance(x, (str, bool, int)) else _BAD


def _rule(r):
    if not isinstance(r, dict) or set(r) != set(RULE_FIELDS):
        return _BAD
    if not (isinstance(r["id"], str) and r["id"]) or r["effect"] not in ("allow", "deny"):
        return _BAD
    if r["conditions_mode"] != "all" or not isinstance(r["conditions"], list):
        return _BAD
    acts = r["actions"]
    if not (isinstance(acts, list) and acts and all(isinstance(a, str) and a for a in acts)):
        return _BAD
    principal, resource = _scope(r["principal"]), _scope(r["resource"])
    conds = []
    for c in r["conditions"]:
        if not (isinstance(c, dict) and set(c) == {"left", "op", "right"} and c["op"] in OPS):
            return _BAD
        left, right = _operand(c["left"]), _operand(c["right"])
        if not isinstance(left, dict) or right is _BAD:  # the left side must be an attribute
            return _BAD
        conds.append({"left": left, "op": c["op"], "right": right})
    if principal is _BAD or resource is _BAD:
        return _BAD
    return {
        "id": r["id"],
        "effect": r["effect"],
        "principal": principal,
        "actions": sorted(set(acts)),
        "resource": resource,
        "conditions_mode": "all",
        "conditions": conds,
    }


def ir_rules(obj) -> list[dict] | None:
    """Normalized IR rules of a parsed JSON object, or None if it is not an IR object."""
    if not (isinstance(obj, dict) and set(obj) == {"rules"} and isinstance(obj["rules"], list)):
        return None
    rules = [_rule(r) for r in obj["rules"]]
    return None if any(r is _BAD for r in rules) else rules


def ir_text(rules: list[dict]) -> str:
    """Canonical serialization: the json-v2 document, rules sorted by id (equals render_json on gold)."""
    doc = {
        "combining_rule": JSON_COMBINING_RULE,
        "condition_semantics": JSON_CONDITION_SEMANTICS,
        "rules": sorted(rules, key=lambda r: r["id"]),
    }
    return json.dumps(doc, indent=2) + "\n"


def gold_ir(policy: Policy) -> dict:
    """The canonical policy's IR object (what a perfect conversion outputs, up to key order)."""
    from authinv.render.renderers import render_json

    return {"rules": json.loads(render_json(policy))["rules"]}


def parse_conversion(text: str, version: str = JSON_PARSER_VERSION) -> dict:
    """{"conv_category": "ir" | "refusal" | "abstain" | "parse_failure", "ir_text": str | None}."""
    obj = extract_object(text, version)
    if obj is not None:
        rules = ir_rules(obj)
        if rules is not None:
            return {"conv_category": "ir", "ir_text": ir_text(rules)}
    folded = fold_typographic(strip_think(text, version))
    if _REFUSAL.search(folded):
        return {"conv_category": "refusal", "ir_text": None}
    if _ABSTAIN.search(folded) and obj is None:
        return {"conv_category": "abstain", "ir_text": None}
    return {"conv_category": "parse_failure", "ir_text": None}


# ---- semantic scoring ---------------------------------------------------------------------


def ir_policy(gold: Policy, text: str) -> Policy:
    """The IR as a policy over the world's entities; rule ids are re-keyed (labels carry no meaning)."""
    rules = [
        Rule(f"ir{i}", r.effect, r.principal, r.actions, r.resource, r.conditions)
        for i, r in enumerate(decode_json(text))
    ]
    return Policy(gold.policy_id, gold.entities, gold.actions, tuple(rules), gold.context_schema)


def score_ir(
    gold: Policy, text: str, principal_types: tuple[str, ...], resource_types: tuple[str, ...]
) -> tuple[dict, Policy | None]:
    """(scores, predicted policy or None when invalid) for one canonical IR text."""
    pred = ir_policy(gold, text)
    out = {"ir_rules": len(pred.rules), "ir_valid": True, "ir_error": None}
    try:
        if pred.rules:  # an empty rule list is a well-formed deny-all policy
            validate(pred)
    except (PolicyError, RenderError, KeyError, TypeError, ValueError) as e:
        out.update(ir_valid=False, ir_error=str(e)[:200], ir_exact=False, ir_equivalent=False)
        return out, None
    out["ir_exact"] = normal_form(pred.rules) == normal_form(gold.rules)
    merged = Policy(gold.policy_id, gold.entities, gold.actions, gold.rules + pred.rules, gold.context_schema)
    reqs = universe(merged, principal_types, resource_types)
    diff = sum(evaluate(gold, q)["decision"] != evaluate(pred, q)["decision"] for q in reqs)
    out.update(ir_equivalent=diff == 0, ir_disagreements=diff, ir_universe=len(reqs))
    return out, pred


# ---- conversion cases from the frozen benchmark ----------------------------------------


def build_cases(rows: list[dict], worlds: dict[str, dict], tasks: tuple[str, ...]) -> tuple[list[dict], dict]:
    """One conversion case per (world, assignment, rendering), validated against the frozen rows.

    Refuses unless, for every cell, all its rows share one facts block and one
    policy text, the facts equal `render_facts` of the stored canonical policy,
    the policy text equals the frozen renderer's output, every row's request (or
    question) read back from its prompt reproduces the row's label (or decision
    owners) and pair_key, and every canonical policy's IR passes its own schema
    and serializes back to its json_policy rendering.
    """
    from authinv.benchmark import render_facts
    from authinv.mitigate.prompts import FACTS_OPEN, facts_of, pair_key, query_of, request_of, split_prompt
    from authinv.policy import decision_owners, from_dict
    from authinv.render.renderers import render, render_json

    sel = [r for r in rows if r["task"] in tasks]
    cells: dict[tuple, list] = {}
    for r in sel:
        head, text, _ = split_prompt(r["prompt"], r["task"])
        key = (r["world_id"], r["assignment"], r["rendering"])
        if key not in cells:
            cells[key] = [head, text, 0, r]
        elif cells[key][:2] != [head, text]:
            raise ValueError(f"{key}: rows of one cell differ outside the request")
        cells[key][2] += 1
    policies: dict[tuple, Policy] = {}
    for w, a in sorted({k[:2] for k in cells}):
        if w not in worlds:
            raise ValueError(f"{w}: not in worlds.jsonl")
        policies[(w, a)] = from_dict(worlds[w][a])
    cases = []
    for (w, a, g), (head, text, n, r) in sorted(cells.items()):
        pol = policies[(w, a)]
        if head != FACTS_OPEN + render_facts(pol):
            raise ValueError(f"{(w, a, g)}: facts block differs from the canonical policy")
        if text != render(pol, g).rstrip():
            raise ValueError(f"{(w, a, g)}: policy text differs from the frozen renderer")
        cases.append(
            {
                "case_id": f"{w}/{a}/{g}",
                "world_id": w,
                "assignment": a,
                "rendering": g,
                "source": r.get("source"),
                "source_kind": r.get("source_kind"),
                "queries": n,
                "prompt": conversion_prompt(facts_of(head), text),
            }
        )
    for r in sel:
        pol, pt = policies[(r["world_id"], r["assignment"])], tuple(worlds[r["world_id"]]["principal_types"])
        if r["task"] == "application":
            q = request_of(r, pol)
            ok = (
                evaluate(pol, q)["decision"] == r["label"]
                and pair_key(q.action, q.resource, q.context) == r["pair_key"]
            )
        else:
            action, res, ctx = query_of(r, pol)
            owners = [e.id for e in decision_owners(pol, action, res, ctx, pt)]
            ok = owners == r["expected"] and pair_key(action, res, ctx) == r["pair_key"]
        if not ok:
            raise ValueError(f"{r['row_id']}: request read back from the prompt does not match the row")
    for (w, a), pol in policies.items():
        g = gold_ir(pol)
        if not schema_accepts(ir_schema(pol), g):
            raise ValueError(f"{(w, a)}: canonical IR fails its own schema")
        if ir_text(ir_rules(g)) != render_json(pol):
            raise ValueError(f"{(w, a)}: canonical IR does not serialize to the json_policy rendering")
    return cases, {
        "passed": True,
        "cases": len(cases),
        "rows": len(sel),
        "tasks": list(tasks),
        "policies": len(policies),
        "canon_version": CANON_VERSION,
    }
