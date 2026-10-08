"""Request sets for a policy: balanced allow/deny, with decision-boundary requests prioritized.

The request universe is every (principal, action, resource, context) built
from the given principal and resource types, all actions, and per-key context
values. Context values are the literals the policy's conditions compare
against, plus their integer neighbours (c-1, c, c+1), plus one unseen string or
both booleans. A request is a **boundary** request when changing a single
coordinate (the principal, action, resource, or one context key) flips the
reference decision. Those requests probe exactly the distinctions a rendering
must preserve.

Sampling is deterministic given the seed. It takes up to `per_label` allow and
`per_label` deny requests, boundary requests first, so the set is balanced
whenever both labels are available.
"""

from __future__ import annotations

import itertools
import random

from authinv.policy import AttrRef, Policy, Request, evaluate

UNSEEN = "__unseen__"


def context_values(policy: Policy) -> dict[str, list]:
    lits: dict[str, set] = {k: set() for k, _ in policy.context_schema}
    for r in policy.rules:
        for c in r.conditions:
            for side, other in ((c.left, c.right), (c.right, c.left)):
                if isinstance(side, AttrRef) and side.subject == "context" and not isinstance(other, AttrRef):
                    lits[side.attr].add(other)
    out = {}
    for k, t in policy.context_schema:
        if t == "bool":
            out[k] = [False, True]
        elif t == "int":
            vals = {v + d for v in lits[k] for d in (-1, 0, 1)} or {0}
            out[k] = sorted(vals)
        else:
            out[k] = sorted(lits[k]) + [UNSEEN]
    return out


def universe(
    policy: Policy, principal_types: tuple[str, ...], resource_types: tuple[str, ...]
) -> list[Request]:
    principals = sorted(
        (e.ref for e in policy.entities if e.ref.type in principal_types), key=lambda r: r.key()
    )
    resources = sorted(
        (e.ref for e in policy.entities if e.ref.type in resource_types), key=lambda r: r.key()
    )
    ctx = context_values(policy)
    keys = sorted(ctx)
    contexts = [tuple(zip(keys, combo, strict=True)) for combo in itertools.product(*(ctx[k] for k in keys))]
    return [
        Request(p, a, r, c)
        for p in principals
        for a in sorted(policy.actions)
        for r in resources
        for c in contexts
    ]


def _neighbours(req: Request, axes: dict[str, list]) -> list[Request]:
    out = []
    for p in axes["principal"]:
        if p != req.principal:
            out.append(Request(p, req.action, req.resource, req.context))
    for a in axes["action"]:
        if a != req.action:
            out.append(Request(req.principal, a, req.resource, req.context))
    for r in axes["resource"]:
        if r != req.resource:
            out.append(Request(req.principal, req.action, r, req.context))
    ctx = dict(req.context)
    for k, vals in axes["context"].items():
        for v in vals:
            if v != ctx[k]:
                out.append(
                    Request(req.principal, req.action, req.resource, tuple(sorted({**ctx, k: v}.items())))
                )
    return out


def label_universe(
    policy: Policy, principal_types: tuple[str, ...], resource_types: tuple[str, ...]
) -> list[dict]:
    """Every request in the universe with its reference decision and boundary flag."""
    reqs = universe(policy, principal_types, resource_types)
    decision = {r: evaluate(policy, r)["decision"] for r in reqs}
    axes = {
        "principal": sorted({r.principal for r in reqs}, key=lambda x: x.key()),
        "action": sorted(policy.actions),
        "resource": sorted({r.resource for r in reqs}, key=lambda x: x.key()),
        "context": context_values(policy),
    }
    return [
        {
            "request": r,
            "decision": decision[r],
            "boundary": any(decision[n] != decision[r] for n in _neighbours(r, axes)),
        }
        for r in reqs
    ]


def sample_requests(
    policy: Policy,
    principal_types: tuple[str, ...],
    resource_types: tuple[str, ...],
    per_label: int,
    seed: int,
) -> list[dict]:
    """Balanced allow/deny sample, boundary requests first; deterministic given the seed."""
    rng = random.Random(seed)
    labelled = label_universe(policy, principal_types, resource_types)
    picked = []
    for label in ("allow", "deny"):
        pool = [x for x in labelled if x["decision"] == label]
        boundary = [x for x in pool if x["boundary"]]
        rest = [x for x in pool if not x["boundary"]]
        rng.shuffle(boundary)
        rng.shuffle(rest)
        picked += (boundary + rest)[:per_label]
    rng.shuffle(picked)
    return picked


def request_to_dict(req: Request) -> dict:
    return {
        "principal": {"type": req.principal.type, "id": req.principal.id},
        "action": req.action,
        "resource": {"type": req.resource.type, "id": req.resource.id},
        "context": dict(req.context),
    }
