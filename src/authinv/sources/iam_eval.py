"""A small reference evaluator for AWS IAM policy documents (step P1.8).

This module is the **independent oracle** for the Quacky import. It reads the
ORIGINAL IAM JSON and decides one request, with no reference to the canonical
policy or the translator in `authinv.sources.quacky`. The translator is
validated by comparing the two on every request in the universe
(request-level differential testing, RESEARCH_PLAN §3 Phase 1 [DECISION]).

Semantics: one policy document evaluated on its own, as Quacky analyzes it:

- a request is **denied** if any `Deny` statement matches (explicit deny wins);
- otherwise it is **allowed** if any `Allow` statement matches;
- otherwise it is denied (implicit default deny).

A statement matches when its principal (if it has a `Principal` element), its
action, its resource, and every condition hold.

- `Action` patterns use `*` / `?` globbing and are matched **case-insensitively**.
- `Resource` patterns use `*` / `?` globbing and are matched **case-sensitively**.
- `Principal`: `"*"` (or `{"AWS": "*"}`) matches every caller; otherwise the
  caller matches when its `(principal type, value)` is listed exactly.
- `Condition`: all operators AND together, and all keys within an operator AND
  together. A list of values for a positive operator is an OR over the values;
  for a negated operator (`StringNotEquals`, `NumericNotEquals`) it holds when
  the request value equals none of them, as AWS documents.
- A request supplies every condition key the policy reads. Missing-key
  behaviour (`...IfExists`, `Null`) is outside the supported subset.

Only the subset below is supported. Anything else raises `Unsupported`, so the
oracle never silently guesses: `NotAction`, `NotResource`, `NotPrincipal`,
operators other than those in `OPERATORS`, set qualifiers (`ForAnyValue:`,
`ForAllValues:`), `IfExists`, and policy variables (`${...}`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

OPERATORS = {
    "StringEquals": "str",
    "StringNotEquals": "str",
    "NumericEquals": "int",
    "NumericNotEquals": "int",
    "NumericLessThan": "int",
    "NumericLessThanEquals": "int",
    "NumericGreaterThan": "int",
    "NumericGreaterThanEquals": "int",
    "Bool": "bool",
}
_NEGATED = {"StringNotEquals", "NumericNotEquals"}
_TEST = {
    "StringEquals": lambda a, b: a == b,
    "StringNotEquals": lambda a, b: a == b,  # negated after the OR over values
    "NumericEquals": lambda a, b: a == b,
    "NumericNotEquals": lambda a, b: a == b,
    "NumericLessThan": lambda a, b: a < b,
    "NumericLessThanEquals": lambda a, b: a <= b,
    "NumericGreaterThan": lambda a, b: a > b,
    "NumericGreaterThanEquals": lambda a, b: a >= b,
    "Bool": lambda a, b: a == b,
}


class Unsupported(ValueError):
    """The document uses a construct outside the evaluator's subset."""


@dataclass(frozen=True)
class IamRequest:
    action: str
    resource: str
    principal: tuple[str, str] | None = None  # (principal type, value), e.g. ("AWS", "arn:...:user/a")
    context: dict = field(default_factory=dict, hash=False, compare=False)


def _as_list(x) -> list:
    return x if isinstance(x, list) else [x]


def glob_match(pattern: str, value: str, *, case_sensitive: bool) -> bool:
    """IAM wildcard match: `*` is any run of characters, `?` is one character."""
    rx = "".join(".*" if ch == "*" else "." if ch == "?" else re.escape(ch) for ch in pattern)
    return re.fullmatch(rx, value, 0 if case_sensitive else re.IGNORECASE | re.DOTALL) is not None


def statements(doc: dict) -> list[dict]:
    if not isinstance(doc, dict) or "Statement" not in doc:
        raise Unsupported("no Statement element")
    return [s for s in _as_list(doc["Statement"])]


def _principal_matches(spec, caller: tuple[str, str] | None) -> bool:
    if spec == "*":
        return True
    if not isinstance(spec, dict):
        raise Unsupported(f"Principal {spec!r}")
    for ptype, values in spec.items():
        for v in _as_list(values):
            if not isinstance(v, str):
                raise Unsupported(f"Principal value {v!r}")
            if v == "*" and ptype == "AWS":
                return True
            if "*" in v or "?" in v:
                raise Unsupported("wildcard in Principal value")
            if caller is not None and caller == (ptype, v):
                return True
    return False


def _parse(value, kind: str):
    if kind == "str":
        if not isinstance(value, str):
            raise Unsupported(f"string condition value {value!r}")
        return value
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise Unsupported(f"Bool condition value {value!r}")
    if isinstance(value, bool):
        raise Unsupported(f"numeric condition value {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    raise Unsupported(f"non-integer numeric condition value {value!r}")


def _condition_holds(cond: dict, ctx: dict) -> bool:
    if not isinstance(cond, dict):
        raise Unsupported("Condition is not an object")
    for op, block in cond.items():
        if ":" in op:
            raise Unsupported(f"set qualifier {op}")
        if op.endswith("IfExists"):
            raise Unsupported(f"IfExists operator {op}")
        if op not in OPERATORS:
            raise Unsupported(f"condition operator {op}")
        kind = OPERATORS[op]
        for key, values in block.items():
            if key not in ctx:
                raise Unsupported(f"request lacks condition key {key!r}")
            got = ctx[key]
            if kind == "int" and (isinstance(got, bool) or not isinstance(got, int)):
                raise Unsupported(f"context {key!r} is not an integer")
            if kind == "bool" and not isinstance(got, bool):
                raise Unsupported(f"context {key!r} is not a boolean")
            if kind == "str" and not isinstance(got, str):
                raise Unsupported(f"context {key!r} is not a string")
            vals = [_parse(v, kind) for v in _as_list(values)]
            if not vals:
                raise Unsupported(f"empty value list for {key!r}")
            hit = any(_TEST[op](got, v) for v in vals)
            if (not hit) if op not in _NEGATED else hit:
                return False
    return True


def check_supported(stmt: dict) -> None:
    """Raise Unsupported unless the statement lies in the evaluator's subset."""
    if not isinstance(stmt, dict):
        raise Unsupported("statement is not an object")
    if stmt.get("Effect") not in ("Allow", "Deny"):
        raise Unsupported(f"Effect {stmt.get('Effect')!r}")
    for bad in ("NotAction", "NotResource", "NotPrincipal"):
        if bad in stmt:
            raise Unsupported(bad)
    if "Action" not in stmt or "Resource" not in stmt:
        raise Unsupported("statement without Action or Resource")
    if "Principal" in stmt:
        _principal_matches(stmt["Principal"], None)
    cond = stmt.get("Condition", {})
    if not isinstance(cond, dict):
        raise Unsupported("Condition is not an object")
    for op, block in cond.items():
        if ":" in op:
            raise Unsupported(f"set qualifier {op}")
        if op.endswith("IfExists"):
            raise Unsupported(f"IfExists operator {op}")
        if op not in OPERATORS or not isinstance(block, dict):
            raise Unsupported(f"condition operator {op}")
        for values in block.values():
            if not _as_list(values):
                raise Unsupported("empty condition value list")
            for v in _as_list(values):
                _parse(v, OPERATORS[op])


def statement_matches(stmt: dict, req: IamRequest) -> bool:
    check_supported(stmt)
    if "Principal" in stmt and not _principal_matches(stmt["Principal"], req.principal):
        return False
    if not any(glob_match(p, req.action, case_sensitive=False) for p in _as_list(stmt["Action"])):
        return False
    if not any(glob_match(p, req.resource, case_sensitive=True) for p in _as_list(stmt["Resource"])):
        return False
    return _condition_holds(stmt.get("Condition", {}), req.context)


def evaluate(doc: dict, req: IamRequest) -> str:
    """'allow' or 'deny' for one request against one IAM policy document."""
    if "${" in repr(doc):
        raise Unsupported("policy variable")
    stmts = statements(doc)
    for stmt in stmts:  # refuse the whole document up front, not only the statements a request reaches
        check_supported(stmt)
    allowed = False
    for stmt in stmts:
        if statement_matches(stmt, req):
            if stmt["Effect"] == "Deny":
                return "deny"
            allowed = True
    return "allow" if allowed else "deny"
