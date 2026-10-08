"""Five renderings of a canonical policy's rules, each with an exact decoder.

Each renderer is a pure function of the canonical policy: the same policy gives
byte-identical text. Templates are fixed. **Never edit one after seeing model
results**: bump its version, log it in docs/CHANGELOG.md, and rerun.

A rendering shows the rules only. Entity facts (attributes, memberships) and
the combining rule (default deny, forbid overrides permit) are given once in a
shared prompt block that is identical for every rendering (P1.10), so the
rendering is the only thing that varies.

Decoders invert their renderer. `normal_form` gives the comparison key:
rules split into one rule per action, ids dropped, sorted. The owner statement
is organised per action rather than per rule, so it round-trips at that level.
Identifiers (entity ids and types, actions, attribute names, string values)
must match `SAFE` so the textual grammars stay unambiguous.
"""

from __future__ import annotations

import json
import re

from authinv.policy import AttrRef, Condition, EntityRef, Policy, Rule, Scope

SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
VERSIONS = {
    "nl_statement": "nl-v1",
    "owner_statement": "owner-v1",
    "table": "table-v1",
    "json_policy": "json-v1",
    "executable": "cedar-v1",
}
RENDERINGS = tuple(VERSIONS)


class RenderError(ValueError):
    pass


def check_safe(policy: Policy) -> None:
    names = list(policy.actions)
    for r in policy.rules:
        names.append(r.rule_id)
        for s in (r.principal, r.resource):
            if s.entity:
                names += [s.entity.type, s.entity.id]
            if s.type:
                names.append(s.type)
        for c in r.conditions:
            for x in (c.left, c.right):
                if isinstance(x, AttrRef):
                    names.append(x.attr)
                elif isinstance(x, str):
                    names.append(x)
    bad = [n for n in names if not SAFE.match(n)]
    if bad:
        raise RenderError(f"identifiers outside {SAFE.pattern}: {bad}")


def _rules(policy: Policy) -> list[Rule]:
    check_safe(policy)
    return sorted(policy.rules, key=lambda r: r.rule_id)


def normal_form(rules) -> list[str]:
    """Order- and id-insensitive key: one entry per (rule, action), conditions sorted."""
    out = []
    for r in rules:
        conds = sorted(json.dumps([_od(c.left), c.op, _od(c.right)]) for c in r.conditions)
        for a in r.actions:
            out.append(json.dumps([r.effect, _sd(r.principal), a, _sd(r.resource), conds]))
    return sorted(out)


def _sd(s: Scope) -> list:
    return [s.kind, None if s.entity is None else [s.entity.type, s.entity.id], s.type]


def _od(x) -> list:
    if isinstance(x, AttrRef):
        return ["attr", x.subject, x.attr]
    return ["bool" if isinstance(x, bool) else "int" if isinstance(x, int) else "str", x]


# ---- shared phrase pieces ----------------------------------------------------

_OP_WORDS = {
    "==": "is",
    "!=": "is not",
    "<": "is less than",
    "<=": "is at most",
    ">": "is greater than",
    ">=": "is at least",
}
_WORD_OPS = {v: k for k, v in _OP_WORDS.items()}
_SUBJ_WORDS = {"principal": "the principal's", "resource": "the resource's", "context": "the request's"}
_WORD_SUBJ = {v: k for k, v in _SUBJ_WORDS.items()}


def _lit(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return f'"{v}"'


def _unlit(t: str):
    if t in ("true", "false"):
        return t == "true"
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    m = re.fullmatch(r'"([^"]*)"', t)
    if not m:
        raise RenderError(f"bad literal {t!r}")
    return m.group(1)


def _ent(e: EntityRef) -> str:
    return f'{e.type} "{e.id}"'


_ENT = r'([A-Za-z0-9_.-]+) "([A-Za-z0-9_.-]+)"'


def _unent(t: str) -> EntityRef:
    m = re.fullmatch(_ENT, t)
    if not m:
        raise RenderError(f"bad entity {t!r}")
    return EntityRef(m.group(1), m.group(2))


def _join_or(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + ", or " + items[-1]


def _split_or(t: str) -> list[str]:
    t = t.replace(", or ", ", ")
    if ", " in t:
        return t.split(", ")
    return t.split(" or ")


# ---- 1. natural-language permission statement --------------------------------


def _nl_scope(s: Scope, side: str) -> str:
    anyone = "anyone" if side == "principal" else "any resource"
    member = "anyone" if side == "principal" else "anything"
    if s.kind == "any":
        return anyone
    if s.kind == "eq":
        return _ent(s.entity)
    if s.kind == "is":
        return f"any {s.type}"
    if s.kind == "in":
        return f"{member} in {_ent(s.entity)}"
    return f"any {s.type} in {_ent(s.entity)}"


def _nl_unscope(t: str, side: str) -> Scope:
    anyone = "anyone" if side == "principal" else "any resource"
    member = "anyone" if side == "principal" else "anything"
    if t == anyone:
        return Scope()
    if m := re.fullmatch(rf"any ([A-Za-z0-9_.-]+) in {_ENT}", t):
        return Scope("is_in", EntityRef(m.group(2), m.group(3)), m.group(1))
    if m := re.fullmatch(rf"{member} in {_ENT}", t):
        return Scope("in", EntityRef(m.group(1), m.group(2)))
    if m := re.fullmatch(r"any ([A-Za-z0-9_.-]+)", t):
        return Scope("is", type=m.group(1))
    return Scope("eq", _unent(t))


def _nl_operand(x) -> str:
    return f"{_SUBJ_WORDS[x.subject]} {x.attr}" if isinstance(x, AttrRef) else _lit(x)


def _nl_unoperand(t: str):
    for words, subj in _WORD_SUBJ.items():
        if t.startswith(words + " "):
            return AttrRef(subj, t[len(words) + 1 :])
    return _unlit(t)


def _nl_cond(c: Condition) -> str:
    return f"{_nl_operand(c.left)} {_OP_WORDS[c.op]} {_nl_operand(c.right)}"


_OP_ALT = "|".join(sorted((re.escape(w) for w in _WORD_OPS), key=len, reverse=True))


def _nl_uncond(t: str) -> Condition:
    m = re.fullmatch(rf"(.+?) ({_OP_ALT}) (.+)", t)
    if not m:
        raise RenderError(f"bad condition {t!r}")
    return Condition(_nl_unoperand(m.group(1)), _WORD_OPS[m.group(2)], _nl_unoperand(m.group(3)))


def render_nl(policy: Policy) -> str:
    lines = []
    for r in _rules(policy):
        verb = "may" if r.effect == "permit" else "must never"
        s = f"Rule {r.rule_id}: {_nl_scope(r.principal, 'principal')} {verb} {_join_or(sorted(r.actions))} "
        s += _nl_scope(r.resource, "resource")
        if r.conditions:
            s += ", but only when " + " and ".join(_nl_cond(c) for c in r.conditions)
        lines.append(s + ".")
    return "\n".join(lines) + "\n"


_NL_LINE = re.compile(
    r"Rule ([A-Za-z0-9_.-]+): (.+?) (may|must never) (.+?) ((?:any|anything|anyone)\b.*?|"
    + _ENT.replace("(", "(?:")
    + r")(?:, but only when (.+))?\."
)


def decode_nl(text: str) -> list[Rule]:
    rules = []
    for line in text.strip().splitlines():
        m = _NL_LINE.fullmatch(line)
        if not m:
            raise RenderError(f"bad nl line {line!r}")
        rid, p, verb, acts, res, conds = m.groups()
        rules.append(
            Rule(
                rid,
                "permit" if verb == "may" else "forbid",
                _nl_unscope(p, "principal"),
                tuple(_split_or(acts)),
                _nl_unscope(res, "resource"),
                tuple(_nl_uncond(c) for c in conds.split(" and ")) if conds else (),
            )
        )
    return rules


# ---- 2. decision-owner statement ("X decides Y"), one line per action ---------


def render_owner(policy: Policy) -> str:
    lines = []
    for r in _rules(policy):
        for a in sorted(r.actions):
            who, what = _nl_scope(r.principal, "principal"), _nl_scope(r.resource, "resource")
            if r.effect == "permit":
                s = f"{who} decides whether to {a} {what}"
            else:
                s = f"{who} never decides whether to {a} {what}, whatever any other line says"
            if r.conditions:
                s += ", but only when " + " and ".join(_nl_cond(c) for c in r.conditions)
            lines.append(f"[{r.rule_id}] " + s[0].upper() + s[1:] + ".")
    return "\n".join(lines) + "\n"


_OWNER_LINE = re.compile(
    r"\[([A-Za-z0-9_.-]+)\] (.+?) (never decides|decides) whether to ([A-Za-z0-9_.-]+) "
    r"(.+?)(, whatever any other line says)?(?:, but only when (.+))?\."
)


def decode_owner(text: str) -> list[Rule]:
    rules = []
    for i, line in enumerate(text.strip().splitlines()):
        m = _OWNER_LINE.fullmatch(line)
        if not m:
            raise RenderError(f"bad owner line {line!r}")
        rid, who, verb, act, what, _, conds = m.groups()
        who = who[0].lower() + who[1:] if who.split(" ")[0] in ("Anyone", "Any") else who
        rules.append(
            Rule(
                f"{rid}#{i}",
                "forbid" if verb == "never decides" else "permit",
                _nl_unscope(who, "principal"),
                (act,),
                _nl_unscope(what, "resource"),
                tuple(_nl_uncond(c) for c in conds.split(" and ")) if conds else (),
            )
        )
    return rules


# ---- 3. Markdown permission table ---------------------------------------------


def _tab_scope(s: Scope) -> str:
    if s.kind == "any":
        return "any"
    if s.kind == "eq":
        return _ent(s.entity)
    if s.kind == "is":
        return f"is {s.type}"
    if s.kind == "in":
        return f"in {_ent(s.entity)}"
    return f"is {s.type} in {_ent(s.entity)}"


def _tab_unscope(t: str) -> Scope:
    if t == "any":
        return Scope()
    if m := re.fullmatch(rf"is ([A-Za-z0-9_.-]+) in {_ENT}", t):
        return Scope("is_in", EntityRef(m.group(2), m.group(3)), m.group(1))
    if m := re.fullmatch(rf"in {_ENT}", t):
        return Scope("in", EntityRef(m.group(1), m.group(2)))
    if m := re.fullmatch(r"is ([A-Za-z0-9_.-]+)", t):
        return Scope("is", type=m.group(1))
    return Scope("eq", _unent(t))


def _tab_operand(x) -> str:
    return f"{x.subject}.{x.attr}" if isinstance(x, AttrRef) else _lit(x)


def _tab_unoperand(t: str):
    if m := re.fullmatch(r"(principal|resource|context)\.([A-Za-z0-9_.-]+)", t):
        return AttrRef(m.group(1), m.group(2))
    return _unlit(t)


_TAB_HEAD = "| rule | effect | principal | actions | resource | conditions (all must hold) |"


def render_table(policy: Policy) -> str:
    rows = [_TAB_HEAD, "|---|---|---|---|---|---|"]
    for r in _rules(policy):
        conds = (
            "; ".join(f"{_tab_operand(c.left)} {c.op} {_tab_operand(c.right)}" for c in r.conditions) or "—"
        )
        effect = "allow" if r.effect == "permit" else "deny"
        rows.append(
            f"| {r.rule_id} | {effect} | {_tab_scope(r.principal)} | {', '.join(sorted(r.actions))} | "
            f"{_tab_scope(r.resource)} | {conds} |"
        )
    return "\n".join(rows) + "\n"


def decode_table(text: str) -> list[Rule]:
    lines = text.strip().splitlines()
    if lines[0] != _TAB_HEAD or not re.fullmatch(r"\|(---\|){6}", lines[1]):
        raise RenderError("bad table header")
    rules = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            raise RenderError(f"bad table row {line!r}")
        rid, eff, p, acts, res, conds = cells
        cs = []
        if conds != "—":
            for c in conds.split("; "):
                m = re.fullmatch(r"(\S+) (==|!=|<=|>=|<|>) (\S+)", c)
                if not m:
                    raise RenderError(f"bad table condition {c!r}")
                cs.append(Condition(_tab_unoperand(m.group(1)), m.group(2), _tab_unoperand(m.group(3))))
        rules.append(
            Rule(
                rid,
                {"allow": "permit", "deny": "forbid"}[eff],
                _tab_unscope(p),
                tuple(acts.split(", ")),
                _tab_unscope(res),
                tuple(cs),
            )
        )
    return rules


# ---- 4. JSON policy -------------------------------------------------------------


def _js_scope(s: Scope) -> dict:
    d: dict = {"match": s.kind}
    if s.type:
        d["type"] = s.type
    if s.entity:
        d["entity"] = {"type": s.entity.type, "id": s.entity.id}
    return d


def _js_unscope(d: dict) -> Scope:
    ent = d.get("entity")
    return Scope(d["match"], EntityRef(ent["type"], ent["id"]) if ent else None, d.get("type"))


def _js_operand(x):
    return {"attr": f"{x.subject}.{x.attr}"} if isinstance(x, AttrRef) else x


def _js_unoperand(x):
    if isinstance(x, dict):
        subj, attr = x["attr"].split(".", 1)
        return AttrRef(subj, attr)
    return x


def render_json(policy: Policy) -> str:
    doc = {
        "rules": [
            {
                "id": r.rule_id,
                "effect": "allow" if r.effect == "permit" else "deny",
                "principal": _js_scope(r.principal),
                "actions": sorted(r.actions),
                "resource": _js_scope(r.resource),
                "conditions": [
                    {"left": _js_operand(c.left), "op": c.op, "right": _js_operand(c.right)}
                    for c in r.conditions
                ],
            }
            for r in _rules(policy)
        ]
    }
    return json.dumps(doc, indent=2) + "\n"


def decode_json(text: str) -> list[Rule]:
    return [
        Rule(
            r["id"],
            {"allow": "permit", "deny": "forbid"}[r["effect"]],
            _js_unscope(r["principal"]),
            tuple(r["actions"]),
            _js_unscope(r["resource"]),
            tuple(
                Condition(_js_unoperand(c["left"]), c["op"], _js_unoperand(c["right"]))
                for c in r["conditions"]
            ),
        )
        for r in json.loads(text)["rules"]
    ]


# ---- 5. executable rules (Cedar) ------------------------------------------------


def render_executable(policy: Policy) -> str:
    from authinv.equivalence.cedar import cedar_policy_text

    _rules(policy)
    return cedar_policy_text(policy)


def _est_scope(d: dict) -> Scope:
    op = d["op"]
    if op == "All":
        return Scope()
    if op == "==":
        return Scope("eq", EntityRef(d["entity"]["type"], d["entity"]["id"]))
    if op == "in":
        return Scope("in", EntityRef(d["entity"]["type"], d["entity"]["id"]))
    if op == "is" and "in" in d:
        return Scope("is_in", EntityRef(d["in"]["entity"]["type"], d["in"]["entity"]["id"]), d["entity_type"])
    if op == "is":
        return Scope("is", type=d["entity_type"])
    raise RenderError(f"unsupported Cedar scope {d}")


def _est_operand(d: dict):
    if "." in d:
        return AttrRef(d["."]["left"]["Var"], d["."]["attr"])
    if "Value" in d:
        return d["Value"]
    raise RenderError(f"unsupported Cedar operand {d}")


def _est_conj(d: dict) -> list[Condition]:
    if "&&" in d:
        return _est_conj(d["&&"]["left"]) + _est_conj(d["&&"]["right"])
    ((op, body),) = d.items()
    if op not in ("==", "!=", "<", "<=", ">", ">="):
        raise RenderError(f"unsupported Cedar condition {op}")
    return [Condition(_est_operand(body["left"]), op, _est_operand(body["right"]))]


def decode_executable(text: str) -> list[Rule]:
    """Parse with Cedar's own parser (cedarpy) and convert its JSON form back."""
    import cedarpy

    est = json.loads(cedarpy.policies_to_json_str(text))
    rules = []
    for pol in est["staticPolicies"].values():
        act = pol["action"]
        acts = [act["entity"]["id"]] if act["op"] == "==" else [e["id"] for e in act["entities"]]
        conds = []
        for c in pol["conditions"]:
            if c["kind"] != "when":
                raise RenderError("unless-clauses are not produced by the renderer")
            conds += _est_conj(c["body"])
        rules.append(
            Rule(
                pol["annotations"]["id"],
                pol["effect"],
                _est_scope(pol["principal"]),
                tuple(acts),
                _est_scope(pol["resource"]),
                tuple(conds),
            )
        )
    return rules


RENDER = {
    "nl_statement": render_nl,
    "owner_statement": render_owner,
    "table": render_table,
    "json_policy": render_json,
    "executable": render_executable,
}
DECODE = {
    "nl_statement": decode_nl,
    "owner_statement": decode_owner,
    "table": decode_table,
    "json_policy": decode_json,
    "executable": decode_executable,
}


def render(policy: Policy, rendering: str) -> str:
    return RENDER[rendering](policy)


def decode(text: str, rendering: str) -> list[Rule]:
    return DECODE[rendering](text)


def round_trips(policy: Policy, rendering: str) -> bool:
    return normal_form(decode(render(policy, rendering), rendering)) == normal_form(policy.rules)
