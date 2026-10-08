"""Canonical policy → Rego v1 module, entity data, and evaluation with the real OPA engine.

`rego_policy_text` is the exact text the `rego` rendering shows the model: one
deterministic Rego v1 module (`package authz`) with a fixed preamble (default
deny, forbid overrides permit, transitive membership) and one `permit if {…}`
or `forbid if {…}` body per canonical rule, each preceded by `# rule <id>`.
The text is already in `opa fmt` form.

Entity facts are not part of the rendering. As with Cedar's entity list, they
are passed separately, here as OPA `data`:

    data.entities["<type>::<id>"].attrs   the entity's attributes
    data.parents["<type>::<id>"]          the entity's direct parents (keys)

A request is OPA `input`: `{"principal": {"type", "id"}, "action": <name>,
"resource": {"type", "id"}, "context": {...}}`.

`opa_decisions` evaluates a batch of requests in a single `opa eval` call
(the requests ride along as `data.requests`). The binary is the pinned release
fetched by `scripts/fetch_opa.sh` into `tools/opa`; `$OPA_BIN` or an `opa` on
PATH are used otherwise. It is never imported or installed implicitly.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

from authinv.policy import AttrRef, EntityRef, Policy, Request, Rule, Scope

OPA_VERSION = "1.21.1"  # pinned in scripts/fetch_opa.sh
_REPO_ROOT = Path(__file__).resolve().parents[3]

PREAMBLE = """\
package authz

# input.principal and input.resource are {"type": ..., "id": ...}; input.action is the
# action name; input.context is the request context. data.entities["<type>::<id>"].attrs
# holds an entity's attributes; data.parents["<type>::<id>"] lists its direct groups.

default allow := false

default permit := false

default forbid := false

allow if {
\tpermit
\tnot forbid
}

principal := data.entities[key(input.principal)].attrs

resource := data.entities[key(input.resource)].attrs

key(e) := concat("::", [e.type, e.id])

# member(e, group): e is group itself or belongs to it, directly or transitively.
member(e, group) if key(group) in graph.reachable(data.parents, {key(e)})
"""

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEYWORDS = {
    "as",
    "contains",
    "default",
    "else",
    "every",
    "false",
    "if",
    "import",
    "in",
    "not",
    "null",
    "package",
    "some",
    "true",
    "with",
}


class OpaError(RuntimeError):
    pass


# ---- the binary ------------------------------------------------------------------


def find_opa() -> str | None:
    """$OPA_BIN, then the pinned tools/opa, then `opa` on PATH."""
    for cand in (os.environ.get("OPA_BIN"), str(_REPO_ROOT / "tools" / "opa")):
        if cand and os.access(cand, os.X_OK):
            return cand
    return shutil.which("opa")


def _opa() -> str:
    path = find_opa()
    if path is None:
        raise OpaError("OPA binary not found: run `bash scripts/fetch_opa.sh` (or set $OPA_BIN)")
    return path


@cache
def opa_version() -> str | None:
    path = find_opa()
    if path is None:
        return None
    out = subprocess.run([path, "version"], capture_output=True, text=True, check=False).stdout
    m = re.search(r"^Version: (\S+)", out, re.M)
    return m.group(1) if m else None


# ---- rendering -------------------------------------------------------------------


def _lit(v: str | int | bool) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return json.dumps(v)


def _field(base: str, name: str) -> str:
    if _IDENT.match(name) and name not in _KEYWORDS:
        return f"{base}.{name}"
    return f"{base}[{json.dumps(name)}]"


def _operand(x: AttrRef | str | int | bool) -> str:
    if not isinstance(x, AttrRef):
        return _lit(x)
    base = "input.context" if x.subject == "context" else x.subject
    return _field(base, x.attr)


def _ent(ref: EntityRef) -> str:
    return f'{{"type": {json.dumps(ref.type)}, "id": {json.dumps(ref.id)}}}'


def _scope(side: str, s: Scope) -> list[str]:
    var = f"input.{side}"
    if s.kind == "any":
        return []
    if s.kind == "eq":
        return [f"{var} == {_ent(s.entity)}"]
    if s.kind == "is":
        return [f"{var}.type == {json.dumps(s.type)}"]
    if s.kind == "in":
        return [f"member({var}, {_ent(s.entity)})"]
    return [f"{var}.type == {json.dumps(s.type)}", f"member({var}, {_ent(s.entity)})"]


def _actions(actions: tuple[str, ...]) -> str:
    acts = sorted(actions)
    if len(acts) == 1:
        return f"input.action == {json.dumps(acts[0])}"
    return "input.action in {" + ", ".join(json.dumps(a) for a in acts) + "}"


def _rule(r: Rule) -> str:
    body = [_actions(r.actions), *_scope("principal", r.principal), *_scope("resource", r.resource)]
    body += [f"{_operand(c.left)} {c.op} {_operand(c.right)}" for c in r.conditions]
    return f"# rule {r.rule_id}\n{r.effect} if {{\n" + "".join(f"\t{b}\n" for b in body) + "}\n"


def rego_policy_text(policy: Policy) -> str:
    """Deterministic Rego v1 module: preamble, then one body per rule (sorted by id)."""
    rules = sorted(policy.rules, key=lambda r: r.rule_id)
    return PREAMBLE + "".join("\n" + _rule(r) for r in rules)


def rego_data(policy: Policy) -> dict:
    ents = sorted(policy.entities, key=lambda e: e.ref.key())
    return {
        "entities": {e.ref.key(): {"attrs": dict(e.attrs)} for e in ents},
        "parents": {e.ref.key(): sorted(p.key() for p in e.parents) for e in ents},
    }


def rego_input(req: Request) -> dict[str, Any]:
    return {
        "principal": {"type": req.principal.type, "id": req.principal.id},
        "action": req.action,
        "resource": {"type": req.resource.type, "id": req.resource.id},
        "context": dict(req.context),
    }


# ---- parsing (OPA's own parser) --------------------------------------------------


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "location"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def parse_module(text: str) -> dict:
    """`opa parse --format json` of the text: {"package", "rules", "comments", ...}.

    Rules keep their head row as `row`; locations are otherwise stripped.
    Comments become [{"row", "text"}]. Raises ValueError on a parse error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.rego"
        path.write_text(text)
        res = subprocess.run(
            [_opa(), "parse", "--format", "json", "--json-include", "comments,locations", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if res.returncode != 0:
        raise ValueError(f"opa parse failed: {(res.stderr or res.stdout).strip()[:500]}")
    ast = json.loads(res.stdout)
    rules = []
    for r in ast.get("rules", []):
        row = r["head"]["location"]["row"]
        rules.append({**_strip(r), "row": row})
    comments = [
        {"row": c["Location"]["row"], "text": base64.b64decode(c["Text"]).decode()}
        for c in ast.get("comments", [])
    ]
    out = {k: _strip(v) for k, v in ast.items() if k not in ("rules", "comments")}
    return {**out, "rules": rules, "comments": comments}


@cache
def preamble_rules() -> tuple[str, ...]:
    """The preamble's rules as canonical JSON strings (what a decoded module must contain verbatim)."""
    return tuple(
        json.dumps({k: v for k, v in r.items() if k != "row"}, sort_keys=True)
        for r in parse_module(PREAMBLE)["rules"]
    )


# ---- evaluation ------------------------------------------------------------------

_QUERY = "results := [[i, d] | some i, r in data.requests; d := data.authz.allow with input as r]"


def opa_decisions(policy: Policy, requests: list[Request], policy_text: str | None = None) -> list[dict]:
    """Evaluate requests with OPA in one `opa eval` call. Errors are returned per request, never swallowed.

    A request whose `allow` is undefined or not a boolean counts as an error
    (decision "deny"), as does any OPA compile or evaluation error.
    """
    text = policy_text if policy_text is not None else rego_policy_text(policy)
    data = {**rego_data(policy), "requests": [rego_input(r) for r in requests]}
    with tempfile.TemporaryDirectory() as tmp:
        pol, dat = Path(tmp) / "policy.rego", Path(tmp) / "data.json"
        pol.write_text(text)
        dat.write_text(json.dumps(data))
        res = subprocess.run(
            [_opa(), "eval", "--format", "json", "--data", str(pol), "--data", str(dat), _QUERY],
            capture_output=True,
            text=True,
            check=False,
        )
    try:
        doc = json.loads(res.stdout) if res.stdout.strip() else {}
    except json.JSONDecodeError:
        doc = {}
    if res.returncode != 0 or "errors" in doc or "result" not in doc:
        err = json.dumps(doc.get("errors")) if doc.get("errors") else (res.stderr or res.stdout).strip()
        return [{"decision": "deny", "errors": [f"opa eval failed: {err[:500]}"]} for _ in requests]
    got = dict(doc["result"][0]["bindings"]["results"])
    out = []
    for i in range(len(requests)):
        d = got.get(i)
        if isinstance(d, bool):
            out.append({"decision": "allow" if d else "deny", "errors": []})
        else:
            out.append({"decision": "deny", "errors": [f"allow is {'undefined' if d is None else repr(d)}"]})
    return out
