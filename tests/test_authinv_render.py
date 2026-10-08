"""Six renderings: determinism, exact round trips across tiers, decoder strictness."""

from __future__ import annotations

import json

import pytest
from authinv_fuzz import random_policy, world

from authinv.policy import AttrRef, Condition, EntityRef, Policy, Rule, Scope, tier, validate
from authinv.render.renderers import (
    NL_HEADER,
    OWNER_HEADER,
    RENDERINGS,
    TABLE_CAPTION,
    VERSIONS,
    RenderError,
    decode,
    normal_form,
    render,
    round_trips,
)

pytest.importorskip("cedarpy")  # the executable rendering is Cedar text
from authinv.equivalence.rego import find_opa  # noqa: E402

# Rendering Rego needs no binary; decoding it parses with OPA (CI fetches the pinned build).
DECODABLE = [r for r in RENDERINGS if r != "rego" or find_opa()]


def fuzz_policies(n=200):
    return [p for p in (random_policy(s) for s in range(n)) if p is not None]


def test_every_rendering_round_trips_on_random_policies_of_every_tier():
    tiers = set()
    for p in fuzz_policies():
        tiers.add(tier(p))
        for rendering in DECODABLE:
            assert round_trips(p, rendering), (p.policy_id, rendering, render(p, rendering))
    assert tiers == {"single", "multi_rule", "conditioned"}


def test_renderings_are_deterministic_versioned_and_distinct():
    p = fuzz_policies(10)[0]
    texts = {r: render(p, r) for r in RENDERINGS}
    assert texts == {r: render(p, r) for r in RENDERINGS}
    assert len(set(texts.values())) == len(RENDERINGS) and set(VERSIONS) == set(RENDERINGS)


def test_owner_statement_has_one_line_per_rule_action():
    for p in fuzz_policies(30):
        lines = render(p, "owner_statement").strip().splitlines()
        assert lines[0] == OWNER_HEADER and lines[1] == ""
        assert len(lines) - 2 == sum(len(r.actions) for r in p.rules)


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
    nl = render(p, "nl_statement").splitlines()
    assert nl[0] == NL_HEADER and nl[1] == ""
    assert nl[2:] == [
        'Rule r1 allows any User in Group "eng" to push or read any Repo, whenever all of these hold: '
        "the principal's level is at least 3; the principal's team is the resource's team; "
        "the request's mfa is true.",
        'Rule r2 forbids User "bob" to delete Repo "core".',
        "Rule r3 allows anyone to read any Repo, whenever the resource's public is not false.",
    ]
    assert render(p, "owner_statement").splitlines()[-2:] == [
        '[r2] User "bob" may never delete Repo "core".',
        "[r3] Anyone decides whether to read any Repo, whenever the resource's public is not false.",
    ]
    table = render(p, "table").splitlines()
    assert table[0] == TABLE_CAPTION and table[5] == '| r2 | deny | User "bob" | delete | Repo "core" | — |'
    doc = json.loads(render(p, "json_policy"))
    assert doc["combining_rule"].startswith("deny-overrides") and doc["rules"][0]["conditions_mode"] == "all"
    assert doc["rules"][0]["principal"] == {
        "match": "type_and_member_of",
        "type": "User",
        "entity": {"type": "Group", "id": "eng"},
    }
    assert 'principal is User in Group::"eng"' in render(p, "executable")
    assert '\tmember(input.principal, {"type": "Group", "id": "eng"})' in render(p, "rego")
    for r in DECODABLE:
        assert normal_form(decode(render(p, r), r)) == normal_form(p.rules)


def test_in_scope_says_the_group_itself_is_included():
    p = _example()
    rule = Rule("r9", "permit", Scope("in", EntityRef("Group", "eng")), ("read",), Scope("is", type="Repo"))
    q = Policy(p.policy_id, p.entities, p.actions, (rule,), p.context_schema)
    validate(q)
    assert 'Group "eng" itself or any member of it' in render(q, "nl_statement")
    assert '[r9] Group "eng" itself or any member of it decides whether to read any Repo.' in render(
        q, "owner_statement"
    )
    assert '| Group "eng" itself or any member |' in render(q, "table")
    assert '"match": "entity_or_member_of"' in render(q, "json_policy")
    for r in DECODABLE:
        assert normal_form(decode(render(q, r), r)) == normal_form(q.rules)


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
        ("nl_statement", "Rule r1: anyone may read any resource.\n"),  # v1 text, no header
        (
            "nl_statement",
            NL_HEADER + "\n\nRule r1 allows anyone to read any resource, but only when x is 3.\n",
        ),
        ("owner_statement", "[r1] Anyone decides whether to read any resource.\n"),  # missing definition line
        ("owner_statement", OWNER_HEADER + "\n\n[r1] Anyone never decides whether to read any resource.\n"),
        ("table", "| rule | effect |\n|---|---|\n"),
        ("json_policy", '{"rules": []}'),
    ],
)
def test_decoders_reject_malformed_text(rendering, text):
    with pytest.raises((RenderError, KeyError)):
        decode(text, rendering)
