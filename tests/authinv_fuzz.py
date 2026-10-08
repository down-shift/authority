"""Shared random-policy generator for authinv tests (small universes, all tiers)."""

from __future__ import annotations

import random

from authinv.policy import AttrRef, Condition, Entity, EntityRef, Policy, PolicyError, Rule, Scope, validate

GROUPS = [EntityRef("Group", g) for g in ("eng", "ops", "admins")]
USERS = [EntityRef("User", u) for u in ("alice", "bob", "carol", "dan")]
REPOS = [EntityRef("Repo", r) for r in ("core", "wiki", "infra")]
ACTIONS = ("read", "push", "delete")


def world(rng: random.Random) -> tuple[Entity, ...]:
    ents = [Entity(GROUPS[0]), Entity(GROUPS[1]), Entity(GROUPS[2], (), (GROUPS[0],))]  # admins ⊂ eng
    for u in USERS:
        ents.append(
            Entity(
                u,
                (("level", rng.randint(0, 5)), ("team", rng.choice(["eng", "ops"]))),
                tuple(rng.sample(GROUPS, rng.randint(0, 2))),
            )
        )
    for r in REPOS:
        ents.append(Entity(r, (("team", rng.choice(["eng", "ops"])), ("public", rng.random() < 0.5))))
    return tuple(ents)


def random_scope(rng, side):
    kind = rng.choice(["any", "eq", "is", "in", "is_in"] if side == "principal" else ["any", "eq", "is"])
    if kind == "eq":
        return Scope("eq", rng.choice(USERS if side == "principal" else REPOS))
    if kind == "is":
        return Scope("is", type="User" if side == "principal" else "Repo")
    if kind in ("in", "is_in"):
        return Scope(kind, rng.choice(GROUPS), "User" if kind == "is_in" else None)
    return Scope()


def random_condition(rng):
    return rng.choice(
        [
            Condition(
                AttrRef("principal", "level"),
                rng.choice(["<", "<=", ">", ">=", "==", "!="]),
                rng.randint(0, 5),
            ),
            Condition(AttrRef("principal", "team"), rng.choice(["==", "!="]), AttrRef("resource", "team")),
            Condition(AttrRef("resource", "public"), "==", rng.random() < 0.5),
            Condition(AttrRef("context", "hour"), rng.choice(["<", ">="]), rng.randint(6, 20)),
            Condition(AttrRef("context", "mfa"), "==", True),
        ]
    )


def random_policy(seed: int) -> Policy | None:
    rng = random.Random(seed)
    rules = []
    for i in range(rng.randint(1, 4)):
        rules.append(
            Rule(
                f"r{i}",
                rng.choice(["permit", "permit", "forbid"]),
                random_scope(rng, "principal"),
                tuple(rng.sample(ACTIONS, rng.randint(1, 3))),
                random_scope(rng, "resource"),
                tuple(random_condition(rng) for _ in range(rng.randint(0, 2))),
            )
        )
    p = Policy(f"fuzz{seed}", world(rng), ACTIONS, tuple(rules), (("hour", "int"), ("mfa", "bool")))
    try:
        validate(p)
    except PolicyError:
        return None  # e.g. `in` scope + principal attribute: reaches a Group without attributes
    return p
