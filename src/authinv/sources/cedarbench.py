"""CedarBench reference policies → canonical policies (step P1.7).

CedarBench (Vatsa et al., arXiv 2607.03656; github.com/neselab/cedar-synthesis-engine,
Apache-2.0) ships, per scenario, a prose spec, a Cedar schema
(`schema.cedarschema`), a verification plan, and `references/*.cedar`: Cedar
policy sets the authors wrote as formal witnesses for each check (ceilings and
floors). There is no single "answer" policy per scenario and no entity data,
so the import unit is one reference file, and the entity universe is
synthesized deterministically from the schema and the policy's own literals
(recorded as `entities_synthesized` in every world's meta).

A reference converts only when every construct is inside the canonical model
(`authinv.policy`): scopes any/==/is/in/is-in on literal entities, actions as
literals or literal lists, and `when` clauses that are conjunctions of
comparisons between principal/resource/context attributes and primitive
literals. Two exact rewrites are applied, both typed by Cedar's validator:
`!x.a` becomes `x.a == false` and a bare boolean `x.a` becomes `x.a == true`.
A missing action constraint expands to every schema action. Everything else
(`unless`, `||`, `has`, `in` expressions, sets, records, extension types,
templates, entity-valued attributes, attribute chains, action groups,
namespaces, optional attributes, non-SAFE identifiers) is **counted per
construct and excluded**. Nothing is approximated.

A converted world ships only if (1) `certify` passes on every rendering and
(2) the ORIGINAL reference text, run through the Cedar engine on the full
request universe with the synthesized entities, agrees with the canonical
policy's reference decision on every request (conversion faithfulness).

Importer v2 (`cedarbench-import-v2`, step P1.7.2; v1 stays selectable with
`importer="cedarbench-import-v1"`) changes only the universe and one exact
rewrite, never a policy's meaning:

- **action probes**: every schema action whose `appliesTo` admits the
  policy's principal and resource types (and declares every context key the
  policy reads, with the same type) joins `actions`. A rule still lists only
  the actions it names, so requests for the other actions are probes the
  policy denies by default. An unconstrained `action` covers exactly these
  actions (the schema-valid ones for the world's types);
- **extra entities**: every synthesized combination of attribute values and
  group memberships is instantiated `ENTITY_REPLICAS` times, so each side of
  each literal, threshold, and membership has two witnesses. Falls back to one
  copy when the caps would be exceeded;
- **exact `||` split**: a `when` clause whose top-level expression is a
  disjunction of supported conjunctions becomes one rule per disjunct (same
  effect, scopes, and actions; rule ids get `_or<k>` suffixes). This is exact
  for `permit` (a request matches some disjunct iff it matches the clause)
  and for `forbid` (deny-overrides: forbid(A or B) denies iff forbid(A) or
  forbid(B) does). Several `when` clauses multiply out (at most
  `MAX_SPLIT_RULES` rules per reference rule). A `||` anywhere below the
  top level (`A && (B || C)`) is still excluded, counted as `nested_||`.

Every v2 world records `n_allow`, `n_deny`, boundary counts, and whether it
meets P1.6's non-degeneracy bar (`degeneracy`), but non-degeneracy does not
gate shipping; P1.10 filters on it.

Requires the `engines` extra (`cedarpy`), imported lazily.
"""

from __future__ import annotations

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

IMPORTER_V1 = "cedarbench-import-v1"
IMPORTER_V2 = "cedarbench-import-v2"
IMPORTER_VERSION = IMPORTER_V2  # default for new imports
IMPORTERS = (IMPORTER_V1, IMPORTER_V2)
ENTITY_REPLICAS = 2  # v2: copies of every synthesized attribute/membership combination
MAX_SPLIT_RULES = 32  # v2: rules one reference rule may split into
MIN_PER_LABEL = 4  # P1.6 non-degeneracy bar: >= 4 allow and >= 4 deny, >= 2 boundary each
# The pinned tree has 226 scenarios; the paper's 221 tasks are all but these harness-stress ones.
STRESS_SCENARIOS = (
    "realworld/fifty_role_matrix",
    "realworld/hundred_check_scale",
    "realworld/hundred_tenant_isolation",
    "realworld/mega_scale_1000_checks",
    "realworld/mega_scale_500_checks",
)
MAX_ENTITIES_PER_TYPE = 64
MAX_REQUESTS = 20_000
OTHER_STR = "__unseen__"  # same unseen value authinv.equivalence.requests uses for context strings
_PRIM = {"String": "str", "Long": "int", "Bool": "bool"}
_OPS = ("==", "!=", "<", "<=", ">", ">=")


class SchemaError(ValueError):
    pass


# ---- Cedar schema (human-readable syntax) ---------------------------------------


@dataclass
class EntityDecl:
    attrs: dict[str, tuple[object, bool]] = field(default_factory=dict)  # name -> (type, required)
    member_of: tuple[str, ...] = ()


@dataclass
class ActionDecl:
    principals: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    context: dict[str, tuple[object, bool]] = field(default_factory=dict)
    member_of: tuple[str, ...] = ()


@dataclass
class Schema:
    entities: dict[str, EntityDecl]
    actions: dict[str, ActionDecl]
    namespaced: bool = False


_TOKEN = re.compile(r'\s+|//[^\n]*|"(?:[^"\\]|\\.)*"|::|[A-Za-z_][A-Za-z0-9_]*|[{}\[\]<>,;:=?@()]|.', re.S)


def _tokens(text: str) -> list[str]:
    out = []
    for m in _TOKEN.finditer(text):
        t = m.group(0)
        if t.isspace() or t.startswith("//"):
            continue
        out.append(t)
    return out


class _Parser:
    def __init__(self, text: str):
        self.t = _tokens(text)
        self.i = 0
        self.aliases: dict[str, object] = {}
        self.entities: dict[str, EntityDecl] = {}
        self.actions: dict[str, ActionDecl] = {}
        self.namespaced = False

    def peek(self, k: int = 0) -> str | None:
        return self.t[self.i + k] if self.i + k < len(self.t) else None

    def take(self, want: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise SchemaError(f"expected {want!r} at token {self.i}, got {tok!r}")
        self.i += 1
        return tok

    def name(self) -> str:
        tok = self.take()
        if tok.startswith('"'):
            return json.loads(tok)
        if not re.match(r"[A-Za-z_]", tok):
            raise SchemaError(f"expected a name, got {tok!r}")
        return tok

    def path(self) -> str:
        parts = [self.name()]
        while self.peek() == "::" and self.peek(1) and not self.peek(1).startswith('"'):
            self.take("::")
            parts.append(self.name())
        return "::".join(parts)

    def annotations(self) -> None:
        while self.peek() == "@":
            self.take("@")
            self.name()
            if self.peek() == "(":
                self.take("(")
                self.take()
                self.take(")")

    def parse(self) -> Schema:
        while self.peek() is not None:
            self.annotations()
            if self.peek() == "namespace":
                self.take()
                self.namespaced = True
                self.path()
                self.take("{")
                while self.peek() != "}":
                    self.annotations()
                    self.decl()
                self.take("}")
            else:
                self.decl()
        for decl in list(self.entities.values()) + list(self.actions.values()):  # resolve aliases
            for attrs in [decl.attrs] if isinstance(decl, EntityDecl) else [decl.context]:
                for k, (ty, req) in attrs.items():
                    attrs[k] = (self.resolve(ty), req)
        return Schema(self.entities, self.actions, self.namespaced)

    def resolve(self, ty: object, depth: int = 0) -> object:
        if isinstance(ty, str) and ty in self.aliases and depth < 20:
            return self.resolve(self.aliases[ty], depth + 1)
        return ty

    def decl(self) -> None:
        kw = self.take()
        if kw == "entity":
            self.entity()
        elif kw == "action":
            self.action()
        elif kw == "type":
            n = self.name()
            self.take("=")
            self.aliases[n] = self.type_()
            self.take(";")
        else:
            raise SchemaError(f"unknown declaration {kw!r}")

    def names(self, path: bool = False) -> list[str]:
        out = [self.path() if path else self.name()]
        while self.peek() == ",":
            self.take(",")
            out.append(self.path() if path else self.name())
        return out

    def typelist(self) -> list[str]:
        if self.peek() != "[":
            return [self.path()]
        self.take("[")
        out = []
        while self.peek() != "]":
            out.append(self.path())
            if self.peek() == ",":
                self.take(",")
        self.take("]")
        return out

    def entity(self) -> None:
        names = self.names()
        decl = EntityDecl()
        if self.peek() == "enum":
            self.take()
            self.take("[")
            while self.peek() != "]":
                self.take()
            self.take("]")
            decl.member_of = ("__enum__",)
        if self.peek() == "in":
            self.take()
            decl.member_of = tuple(self.typelist())
        if self.peek() == "=":
            self.take()
        if self.peek() == "{":
            rec = self.type_()
            decl.attrs = dict(rec[1])  # type: ignore[index]
        if self.peek() == "tags":
            self.take()
            self.type_()
        self.take(";")
        for n in names:
            self.entities[n] = EntityDecl(dict(decl.attrs), decl.member_of)

    def action_ref(self) -> str:
        if self.peek() and self.peek().startswith('"'):
            return self.name()
        p = self.path()
        if self.peek() == "::":
            self.take("::")
            return self.name()
        return p

    def action(self) -> None:
        names = self.names()
        decl = ActionDecl()
        if self.peek() == "in":
            self.take()
            if self.peek() == "[":
                self.take("[")
                groups = []
                while self.peek() != "]":
                    groups.append(self.action_ref())
                    if self.peek() == ",":
                        self.take(",")
                self.take("]")
            else:
                groups = [self.action_ref()]
            decl.member_of = tuple(groups)
        if self.peek() == "appliesTo":
            self.take()
            self.take("{")
            while self.peek() != "}":
                key = self.take()
                self.take(":")
                if key == "principal":
                    decl.principals = tuple(self.typelist())
                elif key == "resource":
                    decl.resources = tuple(self.typelist())
                elif key == "context":
                    ctx = self.resolve(self.type_())
                    if not (isinstance(ctx, tuple) and ctx[0] == "record"):
                        raise SchemaError("context is not a record")
                    decl.context = dict(ctx[1])
                else:
                    raise SchemaError(f"unknown appliesTo key {key!r}")
                if self.peek() == ",":
                    self.take(",")
            self.take("}")
        self.take(";")
        for n in names:
            self.actions[n] = ActionDecl(decl.principals, decl.resources, dict(decl.context), decl.member_of)

    def type_(self) -> object:
        if self.peek() == "{":
            self.take("{")
            attrs: dict[str, tuple[object, bool]] = {}
            while self.peek() != "}":
                self.annotations()
                n = self.name()
                req = True
                if self.peek() == "?":
                    self.take("?")
                    req = False
                self.take(":")
                attrs[n] = (self.type_(), req)
                if self.peek() == ",":
                    self.take(",")
            self.take("}")
            return ("record", tuple(attrs.items()))
        p = self.path()
        if p in ("Set", "__cedar::Set"):
            self.take("<")
            inner = self.type_()
            self.take(">")
            return ("set", inner)
        return p


def parse_schema(text: str) -> Schema:
    """Parse the subset of Cedar's human-readable schema syntax CedarBench uses."""
    return _Parser(text).parse()


def _prim(ty: object) -> str | None:
    if isinstance(ty, str):
        return _PRIM.get(ty.removeprefix("__cedar::"))
    return None


# ---- policy conversion ----------------------------------------------------------


@dataclass
class Conversion:
    status: str  # "converted" | "excluded"
    constructs: list[str]  # unsupported constructs found (sorted, deduplicated)
    policy: Policy | None = None
    principal_types: tuple[str, ...] = ()
    resource_types: tuple[str, ...] = ()
    detail: str | None = None
    split_rules: int = 0  # v2: reference rules split at a top-level `||`


class _Unsupported(Exception):
    pass


def _operand(d: dict, found: set[str]) -> AttrRef | str | int | bool | None:
    if "." in d:
        left = d["."]["left"]
        if "Var" not in left:
            found.add("attr_chain")
            return None
        if left["Var"] not in ("principal", "resource", "context"):
            found.add("action_attr")
            return None
        return AttrRef(left["Var"], d["."]["attr"])
    if "Value" in d:
        v = d["Value"]
        if isinstance(v, (bool, int, str)):
            return v
        found.add("entity_literal" if isinstance(v, dict) and "__entity" in v else "non_primitive_literal")
        return None
    if "Var" in d:
        found.add("entity_comparison")
        return None
    ((key, _),) = d.items()
    found.add(_construct_name(key))
    return None


_EXT = {
    "ip",
    "decimal",
    "datetime",
    "duration",
    "offset",
    "durationSince",
    "toTime",
    "toDate",
    "toMilliseconds",
    "toSeconds",
    "toMinutes",
    "toHours",
    "toDays",
    "isInRange",
    "isIpv4",
    "isIpv6",
    "isLoopback",
    "isMulticast",
    "lessThan",
    "lessThanOrEqual",
    "greaterThan",
    "greaterThanOrEqual",
}


def _construct_name(key: str) -> str:
    if key in _EXT:
        return "extension_type"
    if key in ("contains", "containsAll", "containsAny", "isEmpty", "Set"):
        return "set"
    if key in ("hasTag", "getTag"):
        return "tags"
    if key in ("+", "-", "*", "neg"):
        return "arithmetic"
    return {"in": "in_expression", "is": "is_expression", "Record": "record"}.get(key, key)


def _disjuncts(d: dict) -> list[dict]:
    """Operands of a top-level `||` chain (any association); [d] when d is not a disjunction."""
    if "||" in d:
        return _disjuncts(d["||"]["left"]) + _disjuncts(d["||"]["right"])
    return [d]


def _conjuncts(d: dict, found: set[str], nested_or: str = "||") -> list[Condition]:
    if "&&" in d:
        return _conjuncts(d["&&"]["left"], found, nested_or) + _conjuncts(d["&&"]["right"], found, nested_or)
    ((key, body),) = d.items()
    if key == "||":
        found.add(nested_or)
        return []
    if key in _OPS:
        left, right = _operand(body["left"], found), _operand(body["right"], found)
        if left is None or right is None:
            return []
        if not isinstance(left, AttrRef):
            if not isinstance(right, AttrRef):
                found.add("literal_condition")
                return []
            flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(key, key)
            return [Condition(right, flip, left)]
        return [Condition(left, key, right)]
    if key == "!":
        arg = body["arg"]
        if "." in arg:
            ref = _operand(arg, found)
            return [Condition(ref, "==", False)] if isinstance(ref, AttrRef) else []
        found.add("negation")
        return []
    if key == ".":
        ref = _operand(d, found)
        return [Condition(ref, "==", True)] if isinstance(ref, AttrRef) else []
    if key == "Value":
        found.add("literal_condition")
        return []
    found.add(_construct_name(key))
    return []


def _scope(d: dict, found: set[str]) -> Scope:
    op = d["op"]
    if op == "All":
        return Scope()
    if op in ("==", "in") and "entity" in d:
        return Scope("eq" if op == "==" else "in", EntityRef(d["entity"]["type"], d["entity"]["id"]))
    if op == "is" and "in" in d and "entity" in d["in"]:
        e = d["in"]["entity"]
        return Scope("is_in", EntityRef(e["type"], e["id"]), d["entity_type"])
    if op == "is" and "in" not in d:
        return Scope("is", type=d["entity_type"])
    found.add("template_slot" if "slot" in json.dumps(d) else "scope_" + op)
    return Scope()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _compatible_actions(
    schema: Schema, ptypes: tuple[str, ...], rtypes: tuple[str, ...], ctx_types: dict[str, str]
) -> list[str]:
    """Schema actions a request over (ptypes, rtypes) with the policy's context keys is valid for."""
    out = []
    for name, a in sorted(schema.actions.items()):
        if not (set(ptypes) <= set(a.principals) and set(rtypes) <= set(a.resources)):
            continue
        ok = True
        for k, t in ctx_types.items():
            ty, req = a.context.get(k, (None, False))
            ok = ok and req and _prim(ty) == t
        if ok:
            out.append(name)
    return out


def convert(text: str, schema: Schema, policy_id: str, importer: str = IMPORTER_VERSION) -> Conversion:
    """Convert one Cedar policy set to a canonical Policy, or explain why not."""
    import cedarpy

    if importer not in IMPORTERS:
        raise ValueError(f"unknown importer {importer!r}; expected one of {IMPORTERS}")
    v2 = importer == IMPORTER_V2

    try:
        est = json.loads(cedarpy.policies_to_json_str(text))
    except ValueError as e:
        return Conversion("excluded", ["cedar_parse_error"], detail=str(e)[:200])
    found: set[str] = set()
    if schema.namespaced:
        found.add("namespace")
    if est.get("templates"):
        found.add("template")
    pols = est.get("staticPolicies", {})
    if not pols:
        found.add("empty_policy_set")
    rules = []
    unconstrained: set[str] = set()  # v2: ids of rules whose action is unconstrained
    split = 0  # v2: reference rules split at a top-level `||`
    for key in sorted(pols):
        pol = pols[key]
        act = pol["action"]
        rid = pol.get("annotations", {}).get("id", key)
        if act["op"] == "All":
            acts = sorted(schema.actions)
        elif act["op"] == "==":
            acts = [act["entity"]["id"]]
        elif act["op"] == "in" and "entities" in act:
            acts = [e["id"] for e in act["entities"]]
        else:
            found.add("action_group")
            acts = []
        alts: list[list[Condition]] = [[]]  # the rule's conditions in DNF (v1: one conjunction)
        for c in pol["conditions"]:
            if c["kind"] != "when":
                found.add("unless")
                continue
            if not v2:
                alts[0] += _conjuncts(c["body"], found)
                continue
            parts = [_conjuncts(x, found, "nested_||") for x in _disjuncts(c["body"])]
            alts = [a + b for a in alts for b in parts]
        if len(alts) > MAX_SPLIT_RULES:
            found.add("disjunction_too_large")
        principal, resource = _scope(pol["principal"], found), _scope(pol["resource"], found)
        ids = [rid] if len(alts) == 1 else [f"{rid}_or{k}" for k in range(1, len(alts) + 1)]
        if v2 and act["op"] == "All":
            unconstrained.update(ids)
        split += len(alts) > 1
        for rule_id, conds in zip(ids, alts, strict=True):
            rules.append(Rule(rule_id, pol["effect"], principal, tuple(acts), resource, tuple(conds)))
    if found:
        return Conversion("excluded", sorted(found))

    named = [r for r in rules if r.rule_id not in unconstrained] or rules
    actions = sorted({a for r in named for a in r.actions})
    unknown = [a for a in actions if a not in schema.actions]
    if unknown:
        return Conversion("excluded", ["unknown_action"], detail=str(unknown))
    applies = {(schema.actions[a].principals, schema.actions[a].resources) for a in actions}
    if len(applies) != 1:
        return Conversion("excluded", ["heterogeneous_applies_to"])
    ((ptypes, rtypes),) = applies
    if not ptypes or not rtypes:
        return Conversion("excluded", ["action_without_applies_to"])
    try:
        policy = _build_world(
            policy_id, rules, actions, schema, tuple(ptypes), tuple(rtypes), text, importer, unconstrained
        )
    except _Unsupported as e:
        return Conversion("excluded", [e.args[0]], detail=e.args[1] if len(e.args) > 1 else None)
    try:
        validate(policy)
    except PolicyError as e:
        return Conversion("excluded", ["canonical_invalid"], detail=str(e))
    try:
        check_safe(policy)
        bad = [x for e in policy.entities for x in (e.ref.type, e.ref.id) if not SAFE.match(x)]
        bad += [k for k, _ in policy.context_schema if not SAFE.match(k)]
        if bad:
            raise RenderError(f"identifiers outside {SAFE.pattern}: {bad}")
    except RenderError as e:
        return Conversion("excluded", ["unsafe_identifier"], detail=str(e)[:200])
    return Conversion("converted", [], policy, tuple(ptypes), tuple(rtypes), split_rules=split)


def _build_world(
    policy_id: str,
    rules: list[Rule],
    actions: list[str],
    schema: Schema,
    ptypes: tuple[str, ...],
    rtypes: tuple[str, ...],
    text: str,
    importer: str = IMPORTER_V1,
    unconstrained: set[str] = frozenset(),
) -> Policy:
    """Deterministic entity universe: every combination of attribute values and group memberships.

    v2 also adds the schema's other compatible actions as probes and instantiates each
    combination ENTITY_REPLICAS times (once when that would exceed the caps).
    """
    for t in ptypes + rtypes:
        if t not in schema.entities:
            raise _Unsupported("unknown_entity_type", t)
    v2 = importer == IMPORTER_V2
    ctx_decl = schema.actions[actions[0]].context
    reach = {"principal": ptypes, "resource": rtypes}
    # attribute keys: ("context", attr) or (type, attr); types from the schema
    attr_type: dict[tuple[str, str], str] = {}
    reads: list[tuple[str, str]] = []  # (subject, attr)
    for r in rules:
        for c in r.conditions:
            for x in (c.left, c.right):
                if isinstance(x, AttrRef):
                    reads.append((x.subject, x.attr))
    for subj, attr in reads:
        if subj == "context":
            if attr not in ctx_decl:
                raise _Unsupported("undeclared_attribute", f"context.{attr}")
            ty, req = ctx_decl[attr]
            if not req:
                raise _Unsupported("optional_attribute", f"context.{attr}")
            if _prim(ty) is None:
                raise _Unsupported("non_primitive_attribute", f"context.{attr}")
            attr_type[("context", attr)] = _prim(ty)
            continue
        for t in reach[subj]:
            decl = schema.entities[t].attrs
            if attr not in decl:
                continue  # canonical validate() rejects it if a rule can reach this type
            ty, req = decl[attr]
            if not req:
                raise _Unsupported("optional_attribute", f"{t}.{attr}")
            if _prim(ty) is None:
                raise _Unsupported("non_primitive_attribute", f"{t}.{attr}")
            attr_type[(t, attr)] = _prim(ty)

    def keys_of(x: AttrRef) -> list[tuple[str, str]]:
        if x.subject == "context":
            return [("context", x.attr)]
        return [(t, x.attr) for t in reach[x.subject] if (t, x.attr) in attr_type]

    # literal domains, merged across attribute-attribute comparisons (union-find)
    parent = {k: k for k in attr_type}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    lits: dict[tuple[str, str], set] = {k: set() for k in attr_type}
    for r in rules:
        for c in r.conditions:
            left = keys_of(c.left)
            if isinstance(c.right, AttrRef):
                for a, b in itertools.product(left, keys_of(c.right)):
                    parent[find(a)] = find(b)
            else:
                for k in left:
                    lits[k].add(c.right)
    group_lits: dict[tuple[str, str], set] = {}
    for k in attr_type:
        group_lits.setdefault(find(k), set()).update(lits[k])

    def domain(k) -> list:
        t, vals = attr_type[k], group_lits[find(k)]
        if t == "bool":
            return [False, True]
        if t == "int":
            return sorted({v + d for v in vals for d in (-1, 0, 1)} or {0, 1})
        return sorted(vals | {OTHER_STR} | (set() if vals else {"v0"}))

    # literal entities in scopes (groups / named principals / resources)
    literal = sorted(
        {s.entity for r in rules for s in (r.principal, r.resource) if s.entity is not None},
        key=EntityRef.key,
    )
    for e in literal:
        if e.type not in schema.entities:
            raise _Unsupported("unknown_entity_type", e.type)
    types = sorted(set(ptypes) | set(rtypes) | {e.type for e in literal})
    ctx_schema = tuple(sorted((a, attr_type[("context", a)]) for (s, a) in attr_type if s == "context"))
    probes: list[str] = []
    if v2:
        universe_actions = _compatible_actions(schema, ptypes, rtypes, dict(ctx_schema))
        if not set(actions) <= set(universe_actions):  # not expected for a validating source
            raise _Unsupported("incompatible_action", str(sorted(set(actions) - set(universe_actions))))
        probes = [a for a in universe_actions if a not in actions]
        rules = [
            Rule(r.rule_id, r.effect, r.principal, tuple(universe_actions), r.resource, r.conditions)
            if r.rule_id in unconstrained
            else r
            for r in rules
        ]
        actions = list(universe_actions)
    from authinv.equivalence.requests import universe

    why: tuple[str, str] = ("universe_too_large", "")
    for replicas in (ENTITY_REPLICAS, 1) if v2 else (1,):
        entities = _entities(types, ptypes, rtypes, literal, attr_type, schema, domain, replicas, v2)
        if entities is None:
            continue  # too many entities of some type with this many replicas
        meta = (
            ("source", "cedarbench"),
            ("importer", importer),
            ("entities_synthesized", True),
            ("source_sha256", _sha(text)),
            ("principal_types", ",".join(ptypes)),
            ("resource_types", ",".join(rtypes)),
        )
        if v2:
            meta += (("entity_replicas", replicas), ("probe_actions", ",".join(probes)))
        policy = Policy(
            policy_id=policy_id,
            entities=tuple(entities),
            actions=tuple(actions),
            rules=tuple(rules),
            context_schema=ctx_schema,
            meta=meta,
        )
        n = len(universe(policy, ptypes, rtypes))
        if n <= MAX_REQUESTS:
            return policy
        why = ("universe_too_large", f"{n} requests")
    raise _Unsupported(*why)


def _entities(types, ptypes, rtypes, literal, attr_type, schema, domain, replicas: int, v2: bool):
    """Entities for one replica count; None (v2 only) when a type would exceed MAX_ENTITIES_PER_TYPE."""
    entities: list[Entity] = []
    for t in types:
        attrs = sorted(a for (tt, a) in attr_type if tt == t)
        doms = [domain((t, a)) for a in attrs]
        groups = [g for g in literal if g.type in schema.entities[t].member_of]
        combos = list(itertools.product(*doms)) or [()]
        made = []
        if t in ptypes or t in rtypes:
            members = [
                tuple(g for g, m in zip(groups, bits, strict=True) if m)
                for bits in itertools.product((False, True), repeat=len(groups))
            ]
            n = len(combos) * len(members) * replicas
            if n > MAX_ENTITIES_PER_TYPE:
                if v2 and replicas > 1:
                    return None
                raise _Unsupported("universe_too_large", f"{t}: {n} entities")
            taken = {e.id for e in literal if e.type == t}
            i = 0
            for _, combo, mem in itertools.product(range(replicas), combos, members):
                while f"{t}_{i}" in taken:
                    i += 1
                made.append(Entity(EntityRef(t, f"{t}_{i}"), tuple(zip(attrs, combo, strict=True)), mem))
                i += 1
        for lit in (e for e in literal if e.type == t):
            made.append(Entity(lit, tuple(zip(attrs, combos[0], strict=True)), ()))
        entities += made
    return entities


# ---- checks ---------------------------------------------------------------------


def source_validates(text: str, schema_text: str) -> dict:
    """Cedar's validator on the ORIGINAL reference against the ORIGINAL schema."""
    import cedarpy

    try:
        res = cedarpy.validate_policies(text, schema_text)
    except ValueError as e:
        return {"passed": False, "errors": [str(e)[:200]]}
    errors = [str(e)[:200] for e in (res.errors or [])]
    return {"passed": bool(res.validation_passed) and not errors, "errors": errors[:5]}


def faithfulness(
    policy: Policy, original_text: str, principal_types: tuple[str, ...], resource_types: tuple[str, ...]
) -> dict:
    """The original Cedar text vs the canonical reference decision on every request in the universe."""
    from authinv.equivalence.cedar import cedar_decisions
    from authinv.equivalence.requests import universe

    reqs: list[Request] = universe(policy, principal_types, resource_types)
    truth = [evaluate(policy, r)["decision"] for r in reqs]
    got = cedar_decisions(policy, reqs, policy_text=original_text)
    mismatches = sum(g["decision"] != t for g, t in zip(got, truth, strict=True))
    errors = sum(bool(g["errors"]) for g in got)
    return {
        "n_requests": len(reqs),
        "mismatches": mismatches,
        "engine_errors": errors,
        "passed": mismatches == 0 and errors == 0,
    }


def degeneracy(
    policy: Policy,
    principal_types: tuple[str, ...],
    resource_types: tuple[str, ...],
    min_per_label: int = MIN_PER_LABEL,
) -> dict:
    """P1.6's non-degeneracy bar on the world's universe (recorded, not a shipping gate).

    `non_degenerate`: >= min_per_label allow and deny requests with >= min_per_label // 2
    boundary requests each (as `synthetic.is_non_degenerate`); `dead_rules` counts rules
    whose removal changes no decision (reported separately).
    """
    from authinv.equivalence.requests import label_universe

    labelled = label_universe(policy, principal_types, resource_types)
    out: dict = {}
    for label in ("allow", "deny"):
        pool = [x for x in labelled if x["decision"] == label]
        out[f"n_{label}"] = len(pool)
        out[f"boundary_{label}"] = sum(x["boundary"] for x in pool)
    out["non_degenerate"] = all(
        out[f"n_{x}"] >= min_per_label and out[f"boundary_{x}"] >= min_per_label // 2
        for x in ("allow", "deny")
    )
    reqs = [x["request"] for x in labelled]
    base = [x["decision"] for x in labelled]
    dead = 0
    for r in policy.rules:
        rest = tuple(q for q in policy.rules if q is not r)
        if not rest:
            continue
        reduced = Policy(policy.policy_id, policy.entities, policy.actions, rest, policy.context_schema)
        dead += [evaluate(reduced, q)["decision"] for q in reqs] == base
    out["dead_rules"] = dead
    return out


# ---- dataset walk ---------------------------------------------------------------


def iter_references(scenarios: Path):
    """(scenario name, reference path) for every references/*.cedar, sorted."""
    for ref in sorted(scenarios.rglob("references/*.cedar")):
        yield str(ref.parent.parent.relative_to(scenarios)), ref


def scenario_dirs(scenarios: Path) -> list[Path]:
    return sorted(p.parent for p in scenarios.rglob("schema.cedarschema"))


def is_stress(scenario: str) -> bool:
    """True for the five harness-stress scenarios outside the paper's 221 tasks."""
    return scenario in STRESS_SCENARIOS


def import_reference(
    scenario: str,
    ref: Path,
    schema: Schema | None,
    schema_text: str,
    schema_error: str | None,
    importer: str = IMPORTER_VERSION,
):
    """Full pipeline for one reference file; returns (record, policy | None, proof | None)."""
    from authinv.equivalence.check import certify

    text = ref.read_text()
    pid = f"cedarbench/{scenario}/{ref.stem}"
    rec: dict = {
        "policy_id": pid,
        "scenario": scenario,
        "stress": is_stress(scenario),
        "reference": ref.name,
        "importer": importer,
        "source_sha256": _sha(text),
    }
    if schema is None:
        rec.update(status="excluded", constructs=["schema_parse_error"], detail=schema_error)
        return rec, None, None
    src = source_validates(text, schema_text)
    rec["source_validates"] = src["passed"]
    if not src["passed"]:
        rec["source_errors"] = src["errors"]
    conv = convert(text, schema, pid, importer)
    rec.update(status=conv.status, constructs=conv.constructs)
    if conv.detail:
        rec["detail"] = conv.detail
    if conv.policy is None:
        return rec, None, None
    proof = certify(conv.policy, conv.principal_types, conv.resource_types)
    faith = faithfulness(conv.policy, text, conv.principal_types, conv.resource_types)
    proof["faithfulness"] = faith
    proof["source"] = {"scenario": scenario, "reference": ref.name, "sha256": rec["source_sha256"]}
    from authinv.policy import semantic_hash, tier

    rec.update(
        certified=proof["passed"],
        faithful=faith["passed"],
        n_requests=proof["n_requests"],
        tier=tier(conv.policy),
        semantic_sha256=semantic_hash(conv.policy),
        n_rules=len(conv.policy.rules),
        n_actions=len(conv.policy.actions),
        split_rules=conv.split_rules,
    )
    if importer == IMPORTER_V2:
        rec.update(degeneracy(conv.policy, conv.principal_types, conv.resource_types))
    rec["shipped"] = bool(proof["passed"] and faith["passed"] and src["passed"])
    return rec, conv.policy, proof
