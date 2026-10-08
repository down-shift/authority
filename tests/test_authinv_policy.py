"""Canonical policy: validation, Cedar-style evaluation, derived ownership, serialization."""

from __future__ import annotations

import pytest

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
    decision_owners,
    evaluate,
    from_dict,
    policy_hash,
    semantic_hash,
    tier,
    to_dict,
    validate,
)

ALICE, BOB, CAROL = (EntityRef("User", n) for n in ("alice", "bob", "carol"))
ENG, OPS = EntityRef("Group", "eng"), EntityRef("Group", "ops")
REPO, WIKI = EntityRef("Repo", "core"), EntityRef("Repo", "wiki")


def entities(alice_level=3):
    return (
        Entity(ENG),
        Entity(OPS),
        Entity(ALICE, (("level", alice_level), ("team", "eng")), (ENG,)),
        Entity(BOB, (("level", 1), ("team", "ops")), (OPS,)),
        Entity(CAROL, (("level", 5), ("team", "eng")), (ENG,)),
        Entity(REPO, (("team", "eng"), ("public", False))),
        Entity(WIKI, (("team", "ops"), ("public", True))),
    )


def policy(rules, ctx=(("hour", "int"),), **kw):
    return Policy("p1", kw.get("ents", entities()), ("read", "push", "delete"), tuple(rules), ctx)


PERMIT_ENG_READ = Rule("r1", "permit", Scope("in", ENG), ("read", "push"), Scope("is", type="Repo"))
FORBID_DELETE = Rule("r2", "forbid", Scope(), ("delete", "push"), Scope("eq", REPO))
PERMIT_ALL_DELETE = Rule("r3", "permit", Scope("is", type="User"), ("delete",), Scope())


def req(p, a, r, **ctx):
    return Request(p, a, r, tuple(sorted(ctx.items())))


def test_default_deny_permit_and_group_membership():
    p = policy([PERMIT_ENG_READ])
    validate(p)
    assert evaluate(p, req(ALICE, "read", REPO, hour=9))["decision"] == "allow"
    assert evaluate(p, req(BOB, "read", REPO, hour=9))["decision"] == "deny"  # not in eng: default deny
    assert evaluate(p, req(ALICE, "delete", REPO, hour=9))["decision"] == "deny"  # action not covered


def test_forbid_overrides_permit():
    p = policy([PERMIT_ENG_READ, FORBID_DELETE, PERMIT_ALL_DELETE])
    validate(p)
    out = evaluate(p, req(ALICE, "push", REPO, hour=9))
    assert out == {"decision": "deny", "permits": ["r1"], "forbids": ["r2"]}
    assert evaluate(p, req(ALICE, "push", WIKI, hour=9))["decision"] == "allow"
    assert evaluate(p, req(BOB, "delete", WIKI, hour=9))["decision"] == "allow"
    assert evaluate(p, req(BOB, "delete", REPO, hour=9))["decision"] == "deny"
    assert tier(p) == "multi_rule"


def test_conditions_on_attributes_and_context():
    rule = Rule(
        "r1",
        "permit",
        Scope("is", type="User"),
        ("push",),
        Scope("is", type="Repo"),
        (
            Condition(AttrRef("principal", "team"), "==", AttrRef("resource", "team")),
            Condition(AttrRef("principal", "level"), ">=", 3),
            Condition(AttrRef("context", "hour"), "<", 18),
        ),
    )
    p = policy([rule])
    validate(p)
    assert tier(p) == "conditioned"
    assert evaluate(p, req(ALICE, "push", REPO, hour=9))["decision"] == "allow"
    assert evaluate(p, req(ALICE, "push", REPO, hour=20))["decision"] == "deny"  # context
    assert evaluate(p, req(ALICE, "push", WIKI, hour=9))["decision"] == "deny"  # team mismatch
    assert (
        evaluate(policy([rule], ents=entities(alice_level=2)), req(ALICE, "push", REPO, hour=9))["decision"]
        == "deny"
    )


def test_decision_owners_is_derived_from_rules():
    p = policy([PERMIT_ENG_READ, FORBID_DELETE])
    assert decision_owners(p, "push", REPO) == []  # forbid wins for everyone
    assert decision_owners(p, "read", REPO, principal_types=("User",)) == [ALICE, CAROL]
    # Cedar semantics: `principal in Group::eng` also matches the group entity itself.
    assert decision_owners(p, "read", REPO) == [ENG, ALICE, CAROL]
    assert tier(policy([PERMIT_ENG_READ])) == "single"


@pytest.mark.parametrize(
    ("rules", "match"),
    [
        ([], "no rules"),
        ([Rule("r1", "permit", Scope(), ("fly",), Scope())], "subset"),
        ([Rule("r1", "allow", Scope(), ("read",), Scope())], "effect"),
        (
            [Rule("r1", "permit", Scope("eq", EntityRef("User", "zed")), ("read",), Scope())],
            "unknown principal",
        ),
        (
            [
                Rule(
                    "r1",
                    "permit",
                    Scope("is", type="User"),
                    ("read",),
                    Scope(),
                    (Condition(AttrRef("resource", "level"), ">", 1),),
                )
            ],
            "lacks attribute",
        ),
        (
            [
                Rule(
                    "r1",
                    "permit",
                    Scope("is", type="User"),
                    ("read",),
                    Scope("is", type="Repo"),
                    (Condition(AttrRef("principal", "team"), "<", "eng"),),
                )
            ],
            "non-integers",
        ),
        (
            [
                Rule(
                    "r1",
                    "permit",
                    Scope("is", type="User"),
                    ("read",),
                    Scope("is", type="Repo"),
                    (Condition(AttrRef("principal", "level"), "==", "3"),),
                )
            ],
            "mismatched types",
        ),
        (
            [
                Rule(
                    "r1",
                    "permit",
                    Scope(),
                    ("read",),
                    Scope(),
                    (Condition(AttrRef("context", "ip"), "==", "x"),),
                )
            ],
            "context key",
        ),
        ([PERMIT_ENG_READ, PERMIT_ENG_READ], "duplicate rule_id"),
    ],
)
def test_validate_rejects(rules, match):
    with pytest.raises(PolicyError, match=match):
        validate(policy(rules))


def test_validate_rejects_inconsistent_schema_and_cycles():
    ents = list(entities())
    ents[2] = Entity(ALICE, (("level", 3),), (ENG,))  # alice lacks 'team'
    with pytest.raises(PolicyError, match="different attributes"):
        validate(policy([PERMIT_ENG_READ], ents=tuple(ents)))
    cyc = (Entity(ENG, (), (OPS,)), Entity(OPS, (), (ENG,)), *entities()[2:])
    with pytest.raises(PolicyError, match="cycle"):
        validate(policy([PERMIT_ENG_READ], ents=cyc))


def test_serialization_round_trip_and_hashes():
    p = policy([FORBID_DELETE, PERMIT_ENG_READ, PERMIT_ALL_DELETE])
    q = policy([PERMIT_ALL_DELETE, PERMIT_ENG_READ, FORBID_DELETE])  # same rules, other order
    assert to_dict(p) == to_dict(q) and policy_hash(p) == policy_hash(q)
    back = from_dict(to_dict(p))
    assert policy_hash(back) == policy_hash(p)
    for r in (req(ALICE, "push", REPO, hour=1), req(BOB, "delete", WIKI, hour=1)):
        assert evaluate(back, r) == evaluate(p, r)
    renamed = Policy(
        "other-id",
        p.entities,
        p.actions,
        tuple(
            Rule("x" + r.rule_id, r.effect, r.principal, r.actions, r.resource, r.conditions) for r in p.rules
        ),
        p.context_schema,
    )
    assert policy_hash(renamed) != policy_hash(p) and semantic_hash(renamed) == semantic_hash(p)
    bad = to_dict(p)
    bad["rules"][0]["conditions"] = [
        {
            "left": {"attr": {"subject": "context", "attr": "hour"}},
            "op": "<",
            "right": {"value": True, "value_type": "int"},
        }
    ]
    with pytest.raises(PolicyError, match="is not int"):
        from_dict(bad)
