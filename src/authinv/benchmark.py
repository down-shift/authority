"""Phase-1 benchmark builder: certified worlds → frozen evaluation rows (P1.10).

Each world is a certified canonical policy plus its principal and resource
types. The builder samples balanced requests (allow/deny, boundary requests
first) and turns each request into two queries, both asked under every
rendering:

- **application**: is this request allowed? (JSON allow/deny; the label is the
  engine-certified decision)
- **interpretation**: which principals may perform this action on this
  resource in this context? (JSON set, the derived decision owners)

The two share `pair_key` (action, resource, context), which is what the
dissociation metric pairs on.

**Name swap.** The `swap` assignment renames principal ids by a fixed
derangement (rotation of the sorted ids), consistently in entities, rules, and
requests. Semantics are unchanged (re-certified), but which name holds which
role changes, so lexical name effects average out across assignments.

**Answer-option order.** The order of "allow"/"deny" in the instruction
alternates across instances and assignments, and is recorded as
`option_order`.

The prompt template (`PROMPT_VERSION`) is identical across renderings: a
shared facts block, the combining rule, then the rendering text. Never edit it
after seeing model results; bump the version and log it in docs/CHANGELOG.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from authinv.equivalence.requests import label_universe, sample_requests
from authinv.policy import (
    Entity,
    EntityRef,
    Policy,
    Request,
    Rule,
    Scope,
    decision_owners,
    evaluate,
    semantic_hash,
    tier,
)
from authinv.render.renderers import RENDERINGS, render

PROMPT_VERSION = "prompt-v1"

INTRO = (
    "You are an authorization assistant. Decide strictly from the facts and the policy below.\n"
    "Combining rule: a request is allowed only if at least one rule allows it and no rule forbids it. "
    "A forbidding rule always wins, and anything not allowed is denied."
)


@dataclass(frozen=True)
class World:
    policy: Policy
    principal_types: tuple[str, ...]
    resource_types: tuple[str, ...]
    source_kind: str  # "real" | "synthetic"
    source: str  # e.g. "synthetic", "cedarbench", "quacky"
    meta: dict = field(default_factory=dict)


# ---- facts and prompt text ------------------------------------------------------


def _val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return f'"{v}"'


def _ent(r: EntityRef) -> str:
    return f'{r.type} "{r.id}"'


def render_facts(policy: Policy) -> str:
    lines = []
    for e in sorted(policy.entities, key=lambda e: e.ref.key()):
        bits = []
        if e.attrs:
            bits.append(", ".join(f"{k} = {_val(v)}" for k, v in sorted(e.attrs)))
        if e.parents:
            bits.append("member of " + ", ".join(_ent(p) for p in sorted(e.parents, key=EntityRef.key)))
        lines.append(f"- {_ent(e.ref)}" + (": " + "; ".join(bits) if bits else ""))
    return "\n".join(lines)


def _context(ctx: tuple) -> str:
    return ", ".join(f"{k} = {_val(v)}" for k, v in ctx) if ctx else "none"


def application_prompt(policy: Policy, rendering_text: str, req: Request, option_order: int) -> str:
    opts = ['{"decision": "allow"}', '{"decision": "deny"}']
    if option_order:
        opts.reverse()
    return (
        f"{INTRO}\n\nFacts:\n{render_facts(policy)}\n\nPolicy:\n{rendering_text.rstrip()}\n\n"
        f"Request: {_ent(req.principal)} asks to {req.action} {_ent(req.resource)}. "
        f"Request context: {_context(req.context)}.\n"
        f"Is this request allowed? Answer with JSON only: {opts[0]} or {opts[1]}."
    )


def interpretation_prompt(
    policy: Policy, rendering_text: str, req: Request, principal_types: tuple[str, ...]
) -> str:
    who = " or ".join(principal_types)
    return (
        f"{INTRO}\n\nFacts:\n{render_facts(policy)}\n\nPolicy:\n{rendering_text.rstrip()}\n\n"
        f"Question: which {who} entities may {req.action} {_ent(req.resource)} when the request context is "
        f"{_context(req.context)}? Consider every {who} listed in the facts.\n"
        'Answer with JSON only: {"principals": [...]} listing their exact quoted ids, or {"principals": []} '
        "if none."
    )


# ---- name swap ---------------------------------------------------------------------


def swap_map(policy: Policy, principal_types: tuple[str, ...]) -> dict[EntityRef, EntityRef]:
    ids = sorted((e.ref for e in policy.entities if e.ref.type in principal_types), key=EntityRef.key)
    if len(ids) < 2:
        raise ValueError(f"{policy.policy_id}: name swap needs at least two principals")
    return {ids[i]: EntityRef(ids[i].type, ids[(i + 1) % len(ids)].id) for i in range(len(ids))}


def _rename_ref(r: EntityRef | None, m: dict) -> EntityRef | None:
    return None if r is None else m.get(r, r)


def rename(policy: Policy, m: dict[EntityRef, EntityRef]) -> Policy:
    def scope(s: Scope) -> Scope:
        return Scope(s.kind, _rename_ref(s.entity, m), s.type)

    ents = tuple(
        Entity(m.get(e.ref, e.ref), e.attrs, tuple(m.get(p, p) for p in e.parents)) for e in policy.entities
    )
    rules = tuple(
        Rule(r.rule_id, r.effect, scope(r.principal), r.actions, scope(r.resource), r.conditions)
        for r in policy.rules
    )
    return Policy(policy.policy_id, ents, policy.actions, rules, policy.context_schema, policy.meta)


def rename_request(req: Request, m: dict) -> Request:
    return Request(
        m.get(req.principal, req.principal), req.action, m.get(req.resource, req.resource), req.context
    )


# ---- selection -----------------------------------------------------------------------


def non_degenerate(w: World, min_per_label: int) -> bool:
    labelled = label_universe(w.policy, w.principal_types, w.resource_types)
    for label in ("allow", "deny"):
        pool = [x for x in labelled if x["decision"] == label]
        if len(pool) < min_per_label or sum(x["boundary"] for x in pool) < min_per_label // 2:
            return False
    return len([e for e in w.policy.entities if e.ref.type in w.principal_types]) >= 2


def dedupe(worlds: list[World]) -> tuple[list[World], int]:
    seen, out = set(), []
    for w in worlds:
        h = (w.source, semantic_hash(w.policy))
        if h not in seen:
            seen.add(h)
            out.append(w)
    return out, len(worlds) - len(out)


# ---- rows -------------------------------------------------------------------------------


def world_rows(w: World, per_label: int, seed: int) -> tuple[list[dict], dict]:
    """Rows for one world under both assignments; returns (rows, the swapped policy for re-certification)."""
    picked = sample_requests(w.policy, w.principal_types, w.resource_types, per_label, seed)
    m = swap_map(w.policy, w.principal_types)
    variants = {"orig": (w.policy, {}), "swap": (rename(w.policy, m), m)}
    rows = []
    for ai, (assignment, (pol, mapping)) in enumerate(variants.items()):
        texts = {g: render(pol, g) for g in RENDERINGS}
        asked: set[str] = set()  # interpretation questions depend only on pair_key
        for qi, item in enumerate(picked):
            req = rename_request(item["request"], mapping) if mapping else item["request"]
            label = evaluate(pol, req)["decision"]
            if label != item["decision"]:
                raise AssertionError(f"{w.policy.policy_id}: name swap changed a decision")
            owners = [
                r.id for r in decision_owners(pol, req.action, req.resource, req.context, w.principal_types)
            ]
            option_order = (qi + ai) % 2
            pair_key = f"{req.action}|{req.resource.key()}|{_context(req.context)}"
            base = {
                "world_id": w.policy.policy_id,
                "source": w.source,
                "source_kind": w.source_kind,
                "tier": tier(w.policy),
                "assignment": assignment,
                "instance": f"q{qi}",
                "boundary": item["boundary"],
                "pair_key": pair_key,
                "prompt_version": PROMPT_VERSION,
            }
            principal_ids = sorted(e.ref.id for e in pol.entities if e.ref.type in w.principal_types)
            for g in RENDERINGS:
                rows.append(
                    {
                        **base,
                        "rendering": g,
                        "task": "application",
                        "label": label,
                        "expected": label,
                        "option_order": option_order,
                        "row_id": f"{w.policy.policy_id}/{assignment}/q{qi}/{g}/application",
                        "prompt": application_prompt(pol, texts[g], req, option_order),
                        "rendering_chars": len(texts[g]),
                    }
                )
                if pair_key in asked:
                    continue
                rows.append(
                    {
                        **base,
                        "rendering": g,
                        "task": "interpretation",
                        "expected": owners,
                        "principal_ids": principal_ids,
                        "row_id": f"{w.policy.policy_id}/{assignment}/q{qi}/{g}/interpretation",
                        "prompt": interpretation_prompt(pol, texts[g], req, w.principal_types),
                        "rendering_chars": len(texts[g]),
                    }
                )
            asked.add(pair_key)
    return rows, {"swap_policy": variants["swap"][0], "swap_map": {k.key(): v.key() for k, v in m.items()}}
