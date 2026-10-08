"""Cedar translation and engine agreement with the reference semantics; request sets."""

from __future__ import annotations

import random

import pytest

from authinv.equivalence.requests import context_values, label_universe, sample_requests, universe
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
    validate,
)

cedarpy = pytest.importorskip("cedarpy")
from authinv.equivalence import cedar  # noqa: E402

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


def test_cedar_text_is_deterministic_and_formatted():
    p = random_policy(3) or random_policy(4)
    text = cedar.cedar_policy_text(p)
    assert text == cedar.cedar_policy_text(p) and text.startswith('@id("r0")')
    assert cedarpy.format_policies(text).strip() + "\n" == text


def test_reference_semantics_match_the_cedar_engine_on_random_policies():
    checked = requests = 0
    for seed in range(300):
        p = random_policy(seed)
        if p is None:
            continue
        reqs = universe(p, ("User",), ("Repo",))
        engine = cedar.cedar_decisions(p, reqs)
        for req, got in zip(reqs, engine, strict=True):
            assert not got["errors"], (seed, got["errors"])
            assert got["decision"] == evaluate(p, req)["decision"], (seed, req, cedar.cedar_policy_text(p))
        checked += 1
        requests += len(reqs)
    assert checked >= 150 and requests > 20_000


def test_schema_validation_accepts_valid_policies():
    for seed in range(40):
        p = random_policy(seed)
        if p is not None:
            v = cedar.validate_with_schema(p, ("User", "Group"), ("Repo",))
            assert v["passed"], (seed, v["errors"], cedar.cedar_policy_text(p))


def test_schema_validation_rejects_ill_typed_text():
    p = random_policy(5) or random_policy(6)
    bad = 'permit (principal, action == Action::"read", resource) when { principal.level == "high" };'
    result = cedarpy.validate_policies(
        bad, __import__("json").dumps(cedar.cedar_schema(p, ("User",), ("Repo",)))
    )
    assert not result.validation_passed


def test_context_values_cover_boundaries():
    p = Policy(
        "c",
        world(random.Random(0)),
        ACTIONS,
        (
            Rule(
                "r0", "permit", Scope(), ("read",), Scope(), (Condition(AttrRef("context", "hour"), "<", 18),)
            ),
        ),
        (("hour", "int"), ("mfa", "bool")),
    )
    validate(p)
    assert context_values(p) == {"hour": [17, 18, 19], "mfa": [False, True]}


def test_sample_is_balanced_boundary_first_and_deterministic():
    for seed in range(20):
        p = random_policy(seed)
        if p is None:
            continue
        labelled = label_universe(p, ("User",), ("Repo",))
        allow = [x for x in labelled if x["decision"] == "allow"]
        deny = [x for x in labelled if x["decision"] == "deny"]
        if len(allow) < 4 or len(deny) < 4:
            continue
        s = sample_requests(p, ("User",), ("Repo",), per_label=4, seed=7)
        assert [x["request"] for x in s] == [
            x["request"] for x in sample_requests(p, ("User",), ("Repo",), 4, 7)
        ]
        assert sum(x["decision"] == "allow" for x in s) == 4 and sum(x["decision"] == "deny" for x in s) == 4
        for label in ("allow", "deny"):
            n_boundary = sum(x["boundary"] for x in labelled if x["decision"] == label)
            got = sum(x["boundary"] for x in s if x["decision"] == label)
            assert got == min(4, n_boundary)
        return
    pytest.fail("no fuzz policy had both labels")
