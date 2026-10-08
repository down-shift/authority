"""Five renderings: determinism, exact round trips across tiers, decoder strictness."""

from __future__ import annotations

import pytest
from authinv_fuzz import random_policy, world

from authinv.policy import AttrRef, Condition, EntityRef, Policy, Rule, Scope, tier, validate
from authinv.render.renderers import (
    RENDERINGS,
    VERSIONS,
    RenderError,
    decode,
    normal_form,
    render,
    round_trips,
)

pytest.importorskip("cedarpy")  # the executable rendering is Cedar text


def fuzz_policies(n=200):
    return [p for p in (random_policy(s) for s in range(n)) if p is not None]


def test_every_rendering_round_trips_on_random_policies_of_every_tier():
    tiers = set()
    for p in fuzz_policies():
        tiers.add(tier(p))
        for rendering in RENDERINGS:
            assert round_trips(p, rendering), (p.policy_id, rendering, render(p, rendering))
    assert tiers == {"single", "multi_rule", "conditioned"}


def test_renderings_are_deterministic_versioned_and_distinct():
    p = fuzz_policies(10)[0]
    texts = {r: render(p, r) for r in RENDERINGS}
    assert texts == {r: render(p, r) for r in RENDERINGS}
    assert len(set(texts.values())) == len(RENDERINGS) and set(VERSIONS) == set(RENDERINGS)


def test_owner_statement_has_one_line_per_rule_action():
    for p in fuzz_policies(30):
        assert len(render(p, "owner_statement").strip().splitlines()) == sum(len(r.actions) for r in p.rules)


def _example():
    import random

    ents = world(random.Random(1))
    rules = (
        Rule(
            "r1",
            "permit",
            Scope("is_in", EntityRef("Group", "eng"), "User"),
            ("push", "read"),
            Scope("is", type="Repo"),
            (
                Condition(AttrRef("principal", "level"), ">=", 3),
                Condition(AttrRef("principal", "team"), "==", AttrRef("resource", "team")),
                Condition(AttrRef("context", "mfa"), "==", True),
            ),
        ),
        Rule(
            "r2",
            "forbid",
            Scope("eq", EntityRef("User", "bob")),
            ("delete",),
            Scope("eq", EntityRef("Repo", "core")),
        ),
        Rule(
            "r3",
            "permit",
            Scope(),
            ("read",),
            Scope("is", type="Repo"),
            (Condition(AttrRef("resource", "public"), "!=", False),),
        ),
    )
    p = Policy("ex", ents, ("read", "push", "delete"), rules, (("hour", "int"), ("mfa", "bool")))
    validate(p)
    return p


def test_golden_texts_for_a_fixed_policy():
    p = _example()
    assert render(p, "nl_statement").splitlines() == [
        'Rule r1: any User in Group "eng" may push or read any Repo, but only when the principal\'s level '
        "is at least 3 and the principal's team is the resource's team and the request's mfa is true.",
        'Rule r2: User "bob" must never delete Repo "core".',
        "Rule r3: anyone may read any Repo, but only when the resource's public is not false.",
    ]
    assert render(p, "owner_statement").splitlines()[-2:] == [
        '[r2] User "bob" never decides whether to delete Repo "core", whatever any other line says.',
        "[r3] Anyone decides whether to read any Repo, but only when the resource's public is not false.",
    ]
    assert render(p, "table").splitlines()[3] == '| r2 | deny | User "bob" | delete | Repo "core" | — |'
    assert '"effect": "deny"' in render(p, "json_policy")
    assert 'principal is User in Group::"eng"' in render(p, "executable")
    for r in RENDERINGS:
        assert normal_form(decode(render(p, r), r)) == normal_form(p.rules)


def test_unsafe_identifiers_are_rejected():
    p = _example()
    bad = Policy(
        p.policy_id,
        p.entities,
        p.actions,
        p.rules + (Rule("r 4", "permit", Scope(), ("read",), Scope()),),
        p.context_schema,
    )
    with pytest.raises(RenderError, match="identifiers"):
        render(bad, "nl_statement")


@pytest.mark.parametrize(
    ("rendering", "text"),
    [
        ("nl_statement", "Rule r1: anyone might read any resource.\n"),
        ("owner_statement", "[r1] Anyone decides to read any resource.\n"),
        ("table", "| rule | effect |\n|---|---|\n"),
        (
            "nl_statement",
            "Rule r1: anyone may read any resource, but only when the principal's level is maybe 3.\n",
        ),
    ],
)
def test_decoders_reject_malformed_text(rendering, text):
    with pytest.raises(RenderError):
        decode(text, rendering)
