"""Probe items and labels from the frozen benchmark (P5.1).

Labels come from the canonical policy (reference semantics), never from a rendering:

- ``allow_deny`` (application rows): the certified decision.
- ``governing`` (application rows): what settles it.
  - ``permit``: a permit rule applies and no forbid does;
  - ``forbid``: a forbid rule applies (it wins whether or not a permit does);
  - ``condition``: no rule applies, but some permit rule matches the principal, action and
    resource and fails a condition;
  - ``scope``: no rule even matches the principal, action and resource (default deny).
- ``owner_<k>`` (interpretation rows): whether the k-th principal in the facts list (key
  order, the order `render_facts` prints) may perform the asked action on the asked
  resource in the asked context. It is the decision-owner set, read slot by slot.

The request (or question) of a row is read back from its prompt
(`authinv.mitigate.prompts`) and checked against the row's frozen label or owners.
"""

from __future__ import annotations

from authinv.mitigate.prompts import pair_key, query_of, request_of
from authinv.policy import Policy, Request, Rule, decision_owners, evaluate, from_dict
from authinv.policy.model import rule_matches

GOVERNING = ("permit", "forbid", "condition", "scope")


def governing(policy: Policy, req: Request) -> str:
    result = evaluate(policy, req)
    if result["forbids"]:
        return "forbid"
    if result["permits"]:
        return "permit"
    unconditioned = (
        Rule(r.rule_id, r.effect, r.principal, r.actions, r.resource)
        for r in policy.rules
        if r.effect == "permit"
    )
    return "condition" if any(rule_matches(policy, r, req) for r in unconditioned) else "scope"


def principal_slots(policy: Policy, principal_types: tuple[str, ...]) -> list:
    return sorted((e.ref for e in policy.entities if e.ref.type in principal_types), key=lambda r: r.key())


def metadata(row: dict, policy: Policy) -> dict:
    """Surface features for the metadata-only baseline (no activation, no label)."""
    return {
        "rendering": row["rendering"],
        "source_kind": row.get("source_kind"),
        "tier": row.get("tier"),
        "assignment": row["assignment"],
        "n_rules": len(policy.rules),
        "n_forbid": sum(r.effect == "forbid" for r in policy.rules),
        "n_conditions": sum(len(r.conditions) for r in policy.rules),
        "n_entities": len(policy.entities),
        "prompt_chars": len(row["prompt"]),
        "option_order": row.get("option_order", -1),
    }


def probe_items(
    rows: list[dict], worlds: dict[str, dict], tasks: tuple[str, ...], owner_slots: int
) -> list[dict]:
    """One item per dataset row of the given tasks, with its labels and metadata (prompt kept)."""
    cache, out = {}, []
    for r in rows:
        if r["task"] not in tasks:
            continue
        key = (r["world_id"], r["assignment"])
        if key not in cache:
            cache[key] = from_dict(worlds[r["world_id"]][r["assignment"]])
        pol, pt = cache[key], tuple(worlds[r["world_id"]]["principal_types"])
        if r["task"] == "application":
            req = request_of(r, pol)
            decision = evaluate(pol, req)["decision"]
            if decision != r["label"] or pair_key(req.action, req.resource, req.context) != r["pair_key"]:
                raise ValueError(f"{r['row_id']}: request read back does not match the row")
            labels = {"allow_deny": decision, "governing": governing(pol, req)}
        else:
            action, res, ctx = query_of(r, pol)
            owners = {e.id for e in decision_owners(pol, action, res, ctx, pt)}
            if sorted(owners) != sorted(r["expected"]):
                raise ValueError(f"{r['row_id']}: owners read back do not match the row")
            slots = principal_slots(pol, pt)[:owner_slots]
            labels = {
                f"owner_{k}": ("owner" if ref.id in owners else "not_owner") for k, ref in enumerate(slots)
            }
        out.append(
            {
                "item_id": r["row_id"],
                "world_id": r["world_id"],
                "assignment": r["assignment"],
                "instance": r["instance"],
                "rendering": r["rendering"],
                "task": r["task"],
                "prompt": r["prompt"],
                "labels": labels,
                "meta": metadata(r, pol),
            }
        )
    return out
