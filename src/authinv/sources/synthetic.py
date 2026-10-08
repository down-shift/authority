"""Synthetic policy grammars: MCP tool allowlists and GitHub-style repository permissions (P1.6).

Worlds are generated over a balanced design grid for each domain:

| tier        | roles   | conditions | effect pattern               |
|-------------|---------|------------|------------------------------|
| single      | 1       | 0          | permit_only                  |
| multi_rule  | 1, 2, 3 | 0          | permit_only (roles ≥ 2), with_forbid |
| conditioned | 1, 2, 3 | 1, 2, 3    | permit_only, with_forbid     |

`roles` is the number of permit rules, each granted to a different group
(role). `with_forbid` adds one deny-overrides rule. `conditions` is the total
number of conditions spread over the rules. Every cell gets `replicates`
worlds.

A world is accepted only if it is non-degenerate:
- its request universe has at least `min_per_label` allow and deny requests,
  including boundary ones;
- every rule changes at least one decision (no dead rules).

Rejected draws are retried with the next attempt index, and the attempt count
is recorded. Generation is deterministic in (seed, domain, cell, replicate,
attempt).

Identifiers are neutral codes (`u_k3`, `g_amber`, `r_heron`, …), so names carry
no authority cues. Name-swap symmetrization happens later, in P1.10.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass

from authinv.equivalence.requests import label_universe
from authinv.policy import (
    AttrRef,
    Condition,
    Entity,
    EntityRef,
    Policy,
    PolicyError,
    Rule,
    Scope,
    evaluate,
    tier,
    validate,
)


@dataclass(frozen=True)
class Domain:
    name: str
    principal_type: str
    group_type: str
    resource_type: str
    actions: tuple[str, ...]
    principal_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]


REPO = Domain(
    "repo",
    "User",
    "Team",
    "Repo",
    ("delete", "merge", "push", "read"),
    tuple(f"u_{c}{d}" for c, d in zip("bdfhkmnp", "27394658", strict=True)),
    ("g_amber", "g_cobalt", "g_jade", "g_onyx"),
    ("r_heron", "r_lynx", "r_otter", "r_wren"),
)
MCP = Domain(
    "mcp",
    "Agent",
    "Role",
    "Tool",
    ("call", "configure", "list"),
    tuple(f"a_{c}{d}" for c, d in zip("cgjlqrtv", "51846273", strict=True)),
    ("g_basalt", "g_ivory", "g_ochre", "g_slate"),
    ("t_alder", "t_birch", "t_cedar", "t_maple"),
)
DOMAINS = {"repo": REPO, "mcp": MCP}
N_PRINCIPALS, N_GROUPS, N_RESOURCES = 5, 3, 3


def design_grid() -> list[dict]:
    cells = [{"tier": "single", "roles": 1, "conditions": 0, "effect": "permit_only"}]
    for roles in (1, 2, 3):
        for effect in ("permit_only", "with_forbid"):
            if effect == "permit_only" and roles == 1:
                continue  # a single permit rule is the `single` tier
            cells.append({"tier": "multi_rule", "roles": roles, "conditions": 0, "effect": effect})
    for roles, conds, effect in itertools.product((1, 2, 3), (1, 2, 3), ("permit_only", "with_forbid")):
        cells.append({"tier": "conditioned", "roles": roles, "conditions": conds, "effect": effect})
    return cells


def _entities(d: Domain, rng: random.Random) -> tuple[tuple[Entity, ...], list[EntityRef], list[EntityRef]]:
    pids = rng.sample(d.principal_ids, N_PRINCIPALS)
    gids = rng.sample(d.group_ids, N_GROUPS)
    rids = rng.sample(d.resource_ids, N_RESOURCES)
    groups = [EntityRef(d.group_type, g) for g in gids]
    ents = [Entity(g) for g in groups]
    principals = []
    for i, pid in enumerate(pids):
        ref = EntityRef(d.principal_type, pid)
        principals.append(ref)
        member = (groups[i % N_GROUPS],) if i < N_GROUPS else tuple(rng.sample(groups, rng.randint(0, 2)))
        if d.name == "repo":
            attrs = (("level", rng.randint(0, 5)), ("team", rng.choice(["blue", "green"])))
        else:
            attrs = (("clearance", rng.randint(1, 5)), ("env", rng.choice(["prod", "staging"])))
        ents.append(Entity(ref, attrs, member))
    resources = []
    for rid in rids:
        ref = EntityRef(d.resource_type, rid)
        resources.append(ref)
        if d.name == "repo":
            attrs = (
                ("team", rng.choice(["blue", "green"])),
                ("visibility", rng.choice(["private", "public"])),
            )
        else:
            attrs = (("category", rng.choice(["db", "fs", "net"])), ("risk", rng.randint(1, 5)))
        ents.append(Entity(ref, attrs))
    return tuple(ents), principals, resources


def _condition(d: Domain, rng: random.Random) -> tuple[Condition, str | None]:
    """A random condition and the context key it needs (if any)."""
    P, R, C = (
        (lambda a: AttrRef("principal", a)),
        (lambda a: AttrRef("resource", a)),
        (lambda a: AttrRef("context", a)),
    )
    if d.name == "repo":
        options = [
            (Condition(P("level"), ">=", rng.randint(1, 4)), None),
            (Condition(P("team"), "==", R("team")), None),
            (Condition(R("visibility"), "==", rng.choice(["private", "public"])), None),
            (Condition(C("mfa"), "==", True), "mfa"),
            (Condition(C("hour"), "<", rng.choice([9, 12, 18])), "hour"),
        ]
    else:
        options = [
            (Condition(P("clearance"), ">=", R("risk")), None),
            (Condition(R("category"), "==", rng.choice(["db", "fs", "net"])), None),
            (Condition(P("env"), "==", rng.choice(["prod", "staging"])), None),
            (Condition(C("approved"), "==", True), "approved"),
            (Condition(R("risk"), "<=", rng.randint(1, 4)), None),
        ]
    return rng.choice(options)


_CTX_TYPES = {"mfa": "bool", "hour": "int", "approved": "bool"}


def _draw(d: Domain, cell: dict, rng: random.Random, policy_id: str) -> Policy:
    ents, principals, resources = _entities(d, rng)
    groups = [e.ref for e in ents if e.ref.type == d.group_type]
    n_cond = cell["conditions"]
    per_rule = [0] * cell["roles"]
    for _ in range(n_cond):
        per_rule[rng.randrange(cell["roles"])] += 1
    rules, ctx = [], set()
    for i in range(cell["roles"]):
        conds = []
        for _ in range(per_rule[i]):
            c, key = _condition(d, rng)
            conds.append(c)
            if key:
                ctx.add(key)
        reads_principal = any(
            isinstance(x, AttrRef) and x.subject == "principal" for c in conds for x in (c.left, c.right)
        )
        scope_kind = "is_in" if reads_principal else rng.choice(["in", "is_in"])
        principal = Scope(scope_kind, groups[i], d.principal_type if scope_kind == "is_in" else None)
        reads_resource = any(
            isinstance(x, AttrRef) and x.subject == "resource" for c in conds for x in (c.left, c.right)
        )
        resource = (
            Scope("is", type=d.resource_type)
            if reads_resource or rng.random() < 0.6
            else Scope("eq", rng.choice(resources))
        )
        actions = tuple(sorted(rng.sample(d.actions, rng.randint(1, len(d.actions) - 1))))
        rules.append(Rule(f"r{i + 1}", "permit", principal, actions, resource, tuple(conds)))
    if cell["effect"] == "with_forbid":
        if rng.random() < 0.5:
            fp, fr = Scope("eq", rng.choice(principals)), Scope("is", type=d.resource_type)
        else:
            fp, fr = Scope("is", type=d.principal_type), Scope("eq", rng.choice(resources))
        actions = tuple(sorted(rng.sample(d.actions, rng.randint(1, 2))))
        rules.append(Rule(f"r{cell['roles'] + 1}", "forbid", fp, actions, fr))
    context_schema = tuple(sorted((k, _CTX_TYPES[k]) for k in ctx))
    return Policy(
        policy_id,
        ents,
        d.actions,
        tuple(rules),
        context_schema,
        meta=(
            ("domain", d.name),
            ("tier_cell", cell["tier"]),
            ("roles", cell["roles"]),
            ("conditions", n_cond),
            ("effect", cell["effect"]),
        ),
    )


def is_non_degenerate(policy: Policy, d: Domain, min_per_label: int) -> tuple[bool, str]:
    labelled = label_universe(policy, (d.principal_type,), (d.resource_type,))
    for label in ("allow", "deny"):
        pool = [x for x in labelled if x["decision"] == label]
        if len(pool) < min_per_label:
            return False, f"too_few_{label}"
        if sum(x["boundary"] for x in pool) < min_per_label // 2:
            return False, f"too_few_boundary_{label}"
    reqs = [x["request"] for x in labelled]
    base = [x["decision"] for x in labelled]
    for r in policy.rules:
        rest = tuple(q for q in policy.rules if q is not r)
        if not rest:
            continue
        reduced = Policy(policy.policy_id, policy.entities, policy.actions, rest, policy.context_schema)
        if [evaluate(reduced, q)["decision"] for q in reqs] == base:
            return False, "dead_rule"
    return True, "ok"


def generate(
    domain: str, replicates: int, seed: int, min_per_label: int = 4, max_attempts: int = 200
) -> tuple[list[Policy], dict]:
    d = DOMAINS[domain]
    worlds, rejections, attempts_used = [], Counter(), []
    for ci, cell in enumerate(design_grid()):
        for rep in range(replicates):
            pid = f"syn-{domain}-c{ci:02d}-{rep:03d}"
            for attempt in range(max_attempts):
                rng = random.Random(f"{seed}|{domain}|{ci}|{rep}|{attempt}")
                p = _draw(d, cell, rng, pid)
                try:
                    validate(p)
                except PolicyError:
                    rejections["invalid"] += 1
                    continue
                if tier(p) != cell["tier"]:
                    rejections["tier_mismatch"] += 1
                    continue
                ok, why = is_non_degenerate(p, d, min_per_label)
                if ok:
                    worlds.append(
                        Policy(
                            p.policy_id,
                            p.entities,
                            p.actions,
                            p.rules,
                            p.context_schema,
                            p.meta + (("attempt", attempt),),
                        )
                    )
                    attempts_used.append(attempt)
                    break
                rejections[why] += 1
            else:
                raise RuntimeError(f"{pid}: no non-degenerate world in {max_attempts} attempts ({cell})")
    report = {
        "domain": domain,
        "cells": len(design_grid()),
        "replicates": replicates,
        "worlds": len(worlds),
        "rejections": dict(sorted(rejections.items())),
        "max_attempt": max(attempts_used, default=0),
    }
    return worlds, report


def balance_table(worlds: list[Policy]) -> list[dict]:
    """Counts per design factor level (domain, tier, roles, conditions, effect)."""
    rows = []
    for factor in ("domain", "tier_cell", "roles", "conditions", "effect"):
        counts = Counter(dict(w.meta)[factor] for w in worlds)
        rows += [{"factor": factor, "level": str(k), "worlds": v} for k, v in sorted(counts.items(), key=str)]
    return rows
