"""Quacky AWS IAM policies → canonical policies (step P1.8).

Source: the Quacky artifact (Eiers et al., ASE '22, DOI 10.1145/3551349.3559530;
github.com/vlab-cs-ucsb/quacky, BSD-2-Clause). The AWS IAM set is the 41
`samples/{ec2,iam,s3}/exp_single` policies the paper analyzes plus the 546
`samples/mutations` policies generated from them, 587 files in all.

Method (RESEARCH_PLAN §3 Phase 1 [DECISION]): Quacky's SMT route needs the ABC
model counter built from source, so we use the fallback. Each IAM document is
**translated deterministically** into a canonical `Policy`, and the translation
is **validated by request-level differential testing**: the independent
reference evaluator in `authinv.sources.iam_eval`, run on the ORIGINAL IAM
JSON, must give the canonical reference decision on every request in the
universe. `certify()` then checks every rendering with Cedar and OPA. Only
policies with zero mismatches ship.

Closed request universe (recorded in each world's meta):

- **actions**: every concrete (wildcard-free) action string in the policy, plus
  one probe action `PROBE_ACTION` that no concrete string names;
- **resources**: every concrete resource ARN in the policy, plus `PROBE_RESOURCE`;
- **principals**: for an identity policy (no `Principal` element) a single
  caller whose policy this is; for a resource policy, every principal the
  policy names plus one unrelated caller `PROBE_PRINCIPAL`;
- **context**: one key per condition key, with the literals the policy uses
  plus neighbours (`authinv.equivalence.requests.context_values`). Every
  request supplies every key.

Wildcard patterns (`s3:*`, `arn:aws:s3:::bucket/*`, `"*"`) are **expanded over
this finite universe**: a statement covers exactly the universe members its
patterns match. This is exact on the universe and nowhere else, so worlds that
rely on it carry `wildcard_expanded: true`. Statements whose resource or
principal set is neither one member nor the whole axis are split into one rule
per member, and a positive condition operator with several values (an OR) is
split into one rule per value. Both splits are exact. Negated operators with
several values become a conjunction of `!=`.

Everything else is **excluded and counted per construct**: `NotAction`,
`NotResource`, `NotPrincipal`, operators beyond `iam_eval.OPERATORS`,
`IfExists`, `ForAnyValue:` / `ForAllValues:`, policy variables, account or
wildcard principals, non-integer numbers. Nothing is approximated.

Identifiers are mapped injectively to the renderers' `SAFE` alphabet (`:` →
`.`, any other unsafe character → `_`; condition keys → `[A-Za-z0-9_]`). The
map is stored in the world's meta, and a collision excludes the policy.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from authinv.policy import (
    AttrRef,
    Condition,
    Entity,
    EntityRef,
    Policy,
    PolicyError,
    Request,
    Rule,
    Scope,
    evaluate,
    validate,
)
from authinv.render.renderers import SAFE, RenderError, check_safe
from authinv.sources import iam_eval

IMPORTER_VERSION = "quacky-import-v1"
PRINCIPAL_TYPE = "Principal"
RESOURCE_TYPE = "Resource"
PRINCIPAL_TYPES = (PRINCIPAL_TYPE,)
RESOURCE_TYPES = (RESOURCE_TYPE,)
CALLER_ID = "caller"  # the identity-policy caller (no Principal element anywhere)
PROBE_ACTION = "authinv-probe:UnlistedAction"
PROBE_RESOURCE = "arn:aws:authinv-probe:::unlisted-resource"
PROBE_PRINCIPAL = ("AWS", "arn:aws:iam::000000000000:user/authinv-unlisted-caller")
UNSEEN = "__unseen__"  # same unseen string authinv.equivalence.requests adds to context keys
MAX_RULES = 64
MAX_REQUESTS = 20_000
NON_DEGENERATE = 4  # P1.6 bar: >= 4 allow and >= 4 deny requests

# the translator's own operator table (independent of iam_eval's)
_OPS = {
    "StringEquals": ("str", "==", False),
    "StringNotEquals": ("str", "!=", True),
    "NumericEquals": ("int", "==", False),
    "NumericNotEquals": ("int", "!=", True),
    "NumericLessThan": ("int", "<", False),
    "NumericLessThanEquals": ("int", "<=", False),
    "NumericGreaterThan": ("int", ">", False),
    "NumericGreaterThanEquals": ("int", ">=", False),
    "Bool": ("bool", "==", False),
}
_USER_OR_ROLE = re.compile(r"^arn:aws:iam::[0-9]+:(user|role)/[A-Za-z0-9+=,.@_/-]+$")


@dataclass
class Conversion:
    status: str  # "converted" | "excluded"
    constructs: list[str]
    policy: Policy | None = None
    detail: str | None = None
    flags: list[str] = field(default_factory=list)
    names: dict | None = None  # canonical id -> original string, per axis


def _list(x) -> list:
    return x if isinstance(x, list) else [x]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _wild(s: str) -> bool:
    return "*" in s or "?" in s


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s.replace(":", "."))


def _safe_attr(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return out if re.match(r"[A-Za-z_]", out) else "k_" + out


def _matches(pattern: str, value: str, *, fold: bool) -> bool:
    if "[" in pattern:  # IAM has no character classes; keep fnmatch from inventing them
        pattern = pattern.replace("[", "[[]")
    return (
        fnmatch.fnmatchcase(value.lower(), pattern.lower()) if fold else fnmatch.fnmatchcase(value, pattern)
    )


def _int(v) -> int:
    if isinstance(v, bool):
        raise ValueError(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str) and re.fullmatch(r"-?[0-9]+", v):
        return int(v)
    raise ValueError(v)


def _value(v, kind: str):
    if kind == "str":
        if not isinstance(v, str):
            raise ValueError(v)
        return v
    if kind == "int":
        return _int(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    raise ValueError(v)


# ---- construct scan ------------------------------------------------------------


def scan(doc) -> tuple[list[dict], set[str], set[str]]:
    """(statements, excluded constructs, flags). Constructs outside the subset are named, not guessed."""
    found: set[str] = set()
    flags: set[str] = set()
    if not isinstance(doc, dict) or "Statement" not in doc:
        return [], {"no_statement"}, flags
    if "${" in json.dumps(doc):
        found.add("policy_variable")
    stmts = _list(doc["Statement"])
    if not stmts or not all(isinstance(s, dict) for s in stmts):
        return [], found | {"no_statement"}, flags
    has_principal = {"Principal" in s or "NotPrincipal" in s for s in stmts}
    if len(has_principal) > 1:
        found.add("mixed_principal_elements")
    key_kinds: dict[str, set[str]] = {}
    for s in stmts:
        for el in ("NotAction", "NotResource", "NotPrincipal"):
            if el in s:
                found.add(el)
        if s.get("Effect") not in ("Allow", "Deny"):
            found.add("bad_effect")
        if "Action" not in s and "NotAction" not in s:
            found.add("missing_action")
        if "Resource" not in s and "NotResource" not in s:
            found.add("missing_resource")
        for a in _list(s.get("Action", [])):
            if not isinstance(a, str):
                found.add("non_string_action")
            elif _wild(a):
                flags.add("action_wildcard")
        for r in _list(s.get("Resource", [])):
            if not isinstance(r, str):
                found.add("non_string_resource")
            elif _wild(r):
                flags.add("resource_wildcard")
        if "Principal" in s:
            p = s["Principal"]
            if p == "*":
                flags.add("principal_wildcard_all")
            elif not isinstance(p, dict):
                found.add("principal_form")
            else:
                for ptype, vals in p.items():
                    for v in _list(vals):
                        if not isinstance(v, str):
                            found.add("principal_form")
                        elif ptype == "AWS" and v == "*":
                            flags.add("principal_wildcard_all")
                        elif _wild(v):
                            found.add("principal_wildcard_partial")
                        elif ptype == "AWS" and (v.endswith(":root") or re.fullmatch(r"[0-9]+", v)):
                            found.add("principal_account")
                        elif ptype == "AWS" and not _USER_OR_ROLE.match(v):
                            found.add("principal_form")
                        elif ptype not in ("AWS", "Service"):
                            found.add(f"principal_type:{ptype}")
        cond = s.get("Condition", {})
        if not isinstance(cond, dict):
            found.add("condition_form")
            continue
        for op, block in cond.items():
            if ":" in op:
                found.add("condition_set_qualifier")  # ForAnyValue: / ForAllValues:
                if op.endswith("IfExists"):
                    found.add("condition_if_exists")
                continue
            if op.endswith("IfExists"):
                found.add("condition_if_exists")
                continue
            if op not in _OPS:
                found.add(f"condition_op:{op}")
                continue
            kind = _OPS[op][0]
            if not isinstance(block, dict):
                found.add("condition_form")
                continue
            for key, vals in block.items():
                key_kinds.setdefault(key, set()).add(kind)
                vs = _list(vals)
                if not vs:
                    found.add("condition_empty_values")
                for v in vs:
                    try:
                        _value(v, kind)
                    except ValueError:
                        found.add("condition_value_type")
                if len(vs) > 1:
                    flags.add("condition_multi_value")
    if any(len(k) > 1 for k in key_kinds.values()):
        found.add("condition_key_type_conflict")
    return stmts, found, flags


# ---- translation ---------------------------------------------------------------


class _Exclude(Exception):
    pass


def _injective(originals: list[str], fn, axis: str) -> dict[str, str]:
    """original -> canonical id; raises _Exclude on a collision or an unsafe result."""
    out: dict[str, str] = {}
    seen: dict[str, str] = {}
    for o in originals:
        c = fn(o)
        if not SAFE.match(c):
            raise _Exclude("unsafe_identifier", f"{axis}: {o!r}")
        if c in seen and seen[c] != o:
            raise _Exclude("identifier_collision", f"{axis}: {seen[c]!r} / {o!r}")
        seen[c], out[o] = o, c
    return out


def _conjunct_alternatives(cond: dict, key_ids: dict, str_ids: dict) -> list[list[Condition]]:
    """DNF of a statement's Condition block: a list of alternative conjunctions."""
    per_key: list[list[list[Condition]]] = []
    for op in sorted(cond):
        kind, cop, negated = _OPS[op]
        for key in sorted(cond[op]):
            ref = AttrRef("context", key_ids[key])
            vals = [_value(v, kind) for v in _list(cond[op][key])]
            if kind == "str":
                vals = [str_ids[key][v] for v in vals]
            vals = sorted(set(vals), key=repr)
            if negated:
                per_key.append([[Condition(ref, cop, v) for v in vals]])
            else:
                per_key.append([[Condition(ref, cop, v)] for v in vals])
    return [sum(combo, []) for combo in itertools.product(*per_key)] or [[]]


def convert(doc, policy_id: str, source_sha256: str, *, expand_wildcards: bool = True) -> Conversion:
    """Translate one IAM document into a canonical Policy over its closed universe, or explain why not."""
    stmts, found, flags = scan(doc)
    if not expand_wildcards:
        found |= {f for f in flags if f in ("action_wildcard", "resource_wildcard", "principal_wildcard_all")}
    if found:
        return Conversion("excluded", sorted(found), flags=sorted(flags))
    try:
        return _translate(stmts, policy_id, source_sha256, flags)
    except _Exclude as e:
        return Conversion("excluded", [e.args[0]], detail=e.args[1] if len(e.args) > 1 else None)


def _translate(stmts: list[dict], policy_id: str, source_sha256: str, flags: set[str]) -> Conversion:
    identity = not any("Principal" in s for s in stmts)
    # --- action axis: concrete strings (case-insensitive in IAM) + probe
    concrete_actions: dict[str, str] = {}
    for s in stmts:
        for a in _list(s["Action"]):
            if not _wild(a):
                prev = concrete_actions.setdefault(a.lower(), a)
                if prev != a:
                    raise _Exclude("action_case_variant", f"{prev!r} / {a!r}")
    actions = sorted(concrete_actions.values()) + [PROBE_ACTION]
    act_id = _injective(actions, _safe_id, "action")
    # --- resource axis
    resources = sorted({r for s in stmts for r in _list(s["Resource"]) if not _wild(r)}) + [PROBE_RESOURCE]
    res_id = _injective(resources, _safe_id, "resource")
    # --- principal axis
    if identity:
        principals: list[tuple[str, str] | None] = [None]
        prin_id = {None: CALLER_ID}
    else:
        named = set()
        for s in stmts:
            p = s["Principal"]
            if isinstance(p, dict):
                for ptype, vals in p.items():
                    named |= {(ptype, v) for v in _list(vals) if v != "*"}
        principals = sorted(named) + [PROBE_PRINCIPAL]
        ids = _injective([v for _, v in principals], _safe_id, "principal")
        prin_id = {p: ids[p[1]] for p in principals}
        if len(set(prin_id.values())) != len(prin_id):
            raise _Exclude("identifier_collision", "principal values shared across principal types")
    # --- context axis
    kinds: dict[str, str] = {}
    lits: dict[str, set] = {}
    for s in stmts:
        for op, block in s.get("Condition", {}).items():
            for key, vals in block.items():
                kinds[key] = _OPS[op][0]
                lits.setdefault(key, set()).update(_value(v, _OPS[op][0]) for v in _list(vals))
    key_ids = _injective(sorted(kinds), _safe_attr, "condition key")
    str_ids: dict[str, dict[str, str]] = {}
    for key in sorted(kinds):
        if kinds[key] == "str":
            str_ids[key] = _injective(sorted(lits[key]) + [UNSEEN], _safe_id, f"value of {key}")
    # --- rules
    rules: list[Rule] = []
    dropped = 0
    for i, s in enumerate(stmts):
        sid = s.get("Sid")
        base = f"s{i}" + (f"_{_safe_id(sid)}" if isinstance(sid, str) and sid else "")
        acts = [a for a in actions if any(_matches(p, a, fold=True) for p in _list(s["Action"]))]
        ress = [r for r in resources if any(_matches(p, r, fold=False) for p in _list(s["Resource"]))]
        if identity:
            pscopes = [Scope()]
        else:
            spec = s["Principal"]
            if spec == "*" or (isinstance(spec, dict) and "*" in _list(spec.get("AWS", []))):
                hit = list(principals)
            else:
                listed = {(t, v) for t, vals in spec.items() for v in _list(vals)}
                hit = [p for p in principals if p in listed]
            pscopes = (
                [Scope()]
                if len(hit) == len(principals)
                else [Scope("eq", EntityRef(PRINCIPAL_TYPE, prin_id[p])) for p in hit]
            )
        rscopes = (
            [Scope()]
            if len(ress) == len(resources)
            else [Scope("eq", EntityRef(RESOURCE_TYPE, res_id[r])) for r in ress]
        )
        alts = _conjunct_alternatives(s.get("Condition", {}), key_ids, str_ids)
        if not acts or not pscopes or not rscopes:
            dropped += 1
            continue
        combos = list(itertools.product(pscopes, rscopes, alts))
        if len(combos) > 1:
            flags.add("statement_split")
        for j, (ps, rs, conds) in enumerate(combos):
            rid = base if len(combos) == 1 else f"{base}_{j}"
            effect = "permit" if s["Effect"] == "Allow" else "forbid"
            rules.append(Rule(rid, effect, ps, tuple(act_id[a] for a in acts), rs, tuple(conds)))
        if len(rules) > MAX_RULES:
            raise _Exclude("too_many_rules", f"> {MAX_RULES}")
    if dropped:
        flags.add("statement_matches_nothing_in_universe")
    if not rules:
        raise _Exclude("no_statement_matches_universe")
    if any("_wildcard" in f for f in flags):
        flags.add("wildcard_expanded")
    entities = tuple(
        sorted(
            [Entity(EntityRef(PRINCIPAL_TYPE, prin_id[p])) for p in principals]
            + [Entity(EntityRef(RESOURCE_TYPE, res_id[r])) for r in resources],
            key=lambda e: (e.ref.type, e.ref.id),
        )
    )
    names = {
        "action": {act_id[a]: a for a in actions},
        "resource": {res_id[r]: r for r in resources},
        "principal": {prin_id[p]: (list(p) if p else None) for p in principals},
        "context_key": {key_ids[k]: k for k in kinds},
        "context_value": {key_ids[k]: {c: o for o, c in str_ids[k].items()} for k in str_ids},
    }
    policy = Policy(
        policy_id=policy_id,
        entities=entities,
        actions=tuple(sorted(act_id[a] for a in actions)),
        rules=tuple(rules),
        context_schema=tuple(sorted((key_ids[k], kinds[k]) for k in kinds)),
        meta=(
            ("source", "quacky"),
            ("importer", IMPORTER_VERSION),
            ("source_sha256", source_sha256),
            ("entities_synthesized", True),
            ("identity_policy", identity),
            ("wildcard_expanded", "wildcard_expanded" in flags),
            ("names", json.dumps(names, sort_keys=True)),
            ("principal_types", PRINCIPAL_TYPE),
            ("resource_types", RESOURCE_TYPE),
        ),
    )
    try:
        validate(policy)
    except PolicyError as e:
        raise _Exclude("canonical_invalid", str(e)) from e
    try:
        check_safe(policy)
    except RenderError as e:
        raise _Exclude("unsafe_identifier", str(e)[:200]) from e
    from authinv.equivalence.requests import universe

    n = len(universe(policy, PRINCIPAL_TYPES, RESOURCE_TYPES))
    if n > MAX_REQUESTS:
        raise _Exclude("universe_too_large", f"{n} requests")
    return Conversion("converted", [], policy, flags=sorted(flags), names=names)


# ---- differential test against the IAM evaluator -------------------------------


def to_iam_request(req: Request, names: dict) -> iam_eval.IamRequest:
    """Map a canonical request back to the original IAM strings."""
    p = names["principal"][req.principal.id]
    ctx = {}
    for k, v in req.context:
        orig = names["context_key"][k]
        if isinstance(v, str):
            v = names["context_value"][k].get(v, v)
        ctx[orig] = v
    return iam_eval.IamRequest(
        action=names["action"][req.action],
        resource=names["resource"][req.resource.id],
        principal=tuple(p) if p else None,
        context=ctx,
    )


def faithfulness(policy: Policy, doc: dict, names: dict) -> dict:
    """The ORIGINAL IAM document (via iam_eval) vs the canonical reference decision, on every request."""
    from authinv.equivalence.requests import universe

    reqs = universe(policy, PRINCIPAL_TYPES, RESOURCE_TYPES)
    mismatches = errors = 0
    example = None
    for r in reqs:
        want = evaluate(policy, r)["decision"]
        try:
            got = iam_eval.evaluate(doc, to_iam_request(r, names))
        except iam_eval.Unsupported as e:
            errors += 1
            example = example or {"error": str(e)}
            continue
        if got != want:
            mismatches += 1
            example = example or {"request": repr(r), "iam": got, "canonical": want}
    out = {"n_requests": len(reqs), "mismatches": mismatches, "evaluator_errors": errors}
    out["passed"] = mismatches == 0 and errors == 0 and len(reqs) > 0
    if example:
        out["example"] = example
    return out


# ---- dataset walk --------------------------------------------------------------


def iter_policies(samples: Path, globs: list[str]) -> list[Path]:
    """The import set: every file matched by the manifest's globs under samples/, sorted, deduplicated."""
    return sorted({p for g in globs for p in samples.glob(g) if p.is_file()})


def subset_of(rel: str) -> str:
    return "mutation" if rel.startswith("mutations/") else "original"


def import_policy(path: Path, samples: Path, *, expand_wildcards: bool = True):
    """Full pipeline for one IAM file; returns (record, policy | None, proof | None)."""
    from authinv.equivalence.check import certify
    from authinv.policy import semantic_hash, tier

    rel = path.relative_to(samples).as_posix()
    text = path.read_text()
    src = _sha(text)
    pid = "quacky/" + rel.removesuffix(".json")
    rec: dict = {
        "policy_id": pid,
        "path": rel,
        "subset": subset_of(rel),
        "service": rel.split("/")[1] if rel.startswith("mutations/") else rel.split("/")[0],
        "source_sha256": src,
    }
    try:
        doc = json.loads(text)
    except ValueError as e:
        rec.update(status="excluded", constructs=["json_parse_error"], detail=str(e)[:200])
        return rec, None, None
    conv = convert(doc, pid, src, expand_wildcards=expand_wildcards)
    rec.update(status=conv.status, constructs=conv.constructs, flags=conv.flags)
    if conv.detail:
        rec["detail"] = conv.detail
    if conv.policy is None:
        return rec, None, None
    proof = certify(conv.policy, PRINCIPAL_TYPES, RESOURCE_TYPES)
    faith = faithfulness(conv.policy, doc, conv.names)
    proof["faithfulness"] = faith
    proof["source"] = {"path": rel, "sha256": src}
    n, n_allow = proof["n_requests"], proof["n_allow"]
    rec.update(
        certified=proof["passed"],
        faithful=faith["passed"],
        n_requests=n,
        n_allow=n_allow,
        non_degenerate=n_allow >= NON_DEGENERATE and n - n_allow >= NON_DEGENERATE,
        n_rules=len(conv.policy.rules),
        tier=tier(conv.policy),
        semantic_sha256=semantic_hash(conv.policy),
    )
    rec["shipped"] = bool(proof["passed"] and faith["passed"])
    return rec, conv.policy, proof
