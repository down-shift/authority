"""Rego rendering (rego-v1) and OPA engine agreement with the reference semantics."""

from __future__ import annotations

import random
import subprocess

import pytest
from authinv_fuzz import ACTIONS, GROUPS, USERS, random_policy, world
from authinv_opa import requires_opa

from authinv.equivalence import rego
from authinv.equivalence.requests import universe
from authinv.policy import AttrRef, Condition, Entity, EntityRef, Policy, Rule, Scope, evaluate, validate
from authinv.render.renderers import RenderError, decode, normal_form, render, round_trips

pytestmark = requires_opa


def fuzz_policies(n):
    return [p for p in (random_policy(s) for s in range(n)) if p is not None]


def test_rego_text_is_deterministic_and_already_opa_formatted(tmp_path):
    for p in fuzz_policies(20):
        text = render(p, "rego")
        assert text == render(p, "rego") and text.startswith("package authz\n")
        f = tmp_path / "p.rego"
        f.write_text(text)
        res = subprocess.run([rego.find_opa(), "fmt", "--list", str(f)], capture_output=True, text=True)
        assert res.returncode == 0 and res.stdout == "", (p.policy_id, res.stdout, res.stderr)


def test_rego_round_trips_on_random_policies():
    ps = fuzz_policies(200)
    assert len(ps) > 100
    for p in ps:
        assert round_trips(p, "rego"), (p.policy_id, render(p, "rego"))


def test_reference_semantics_match_opa_on_random_policies():
    checked = requests = 0
    for seed in range(300):
        p = random_policy(seed)
        if p is None:
            continue
        # Groups as principals too: exercises `in` on the group itself (Cedar: G in G).
        reqs = universe(p, ("User", "Group"), ("Repo",))
        engine = rego.opa_decisions(p, reqs)
        for req, got in zip(reqs, engine, strict=True):
            assert not got["errors"], (seed, got["errors"])
            assert got["decision"] == evaluate(p, req)["decision"], (seed, req, render(p, "rego"))
        checked += 1
        requests += len(reqs)
    print(f"OPA {rego.opa_version()}: agreement on {requests} requests over {checked} policies")
    assert checked >= 100 and requests > 20_000


def _policy(rules, entities=None):
    p = Policy(
        "t", entities or world(random.Random(0)), ACTIONS, tuple(rules), (("hour", "int"), ("mfa", "bool"))
    )
    validate(p)
    return p


def test_membership_is_transitive_and_reflexive():
    # admins ⊂ eng in the fuzz world; a User only in admins is still in eng.
    ents = list(world(random.Random(0)))
    ents = [e for e in ents if e.ref != USERS[0]] + [
        Entity(USERS[0], (("level", 1), ("team", "eng")), (GROUPS[2],))
    ]
    p = _policy([Rule("r0", "permit", Scope("in", GROUPS[0]), ("read",), Scope())], tuple(ents))
    reqs = universe(p, ("User", "Group"), ("Repo",))
    got = rego.opa_decisions(p, reqs)
    assert [g["decision"] for g in got] == [evaluate(p, r)["decision"] for r in reqs]
    allowed = {r.principal for r, g in zip(reqs, got, strict=True) if g["decision"] == "allow"}
    assert USERS[0] in allowed and GROUPS[0] in allowed and GROUPS[2] in allowed and GROUPS[1] not in allowed


def test_awkward_attribute_names_are_quoted_and_round_trip():
    ents = tuple(
        Entity(e.ref, (*e.attrs, ("in", 1), ("x.y", "a")) if e.ref.type == "User" else e.attrs, e.parents)
        for e in world(random.Random(2))
    )
    p = _policy(
        [
            Rule(
                "r-1.a",
                "permit",
                Scope("is", type="User"),
                ("read", "push"),
                Scope("eq", EntityRef("Repo", "core")),
                (
                    Condition(AttrRef("principal", "in"), ">=", -1),
                    Condition(AttrRef("principal", "x.y"), "==", "a"),
                ),
            )
        ],
        ents,
    )
    text = render(p, "rego")
    assert 'principal["in"] >= -1' in text and 'principal["x.y"] == "a"' in text and "# rule r-1.a" in text
    assert normal_form(decode(text, "rego")) == normal_form(p.rules)
    reqs = universe(p, ("User",), ("Repo",))
    assert [g["decision"] for g in rego.opa_decisions(p, reqs)] == [evaluate(p, r)["decision"] for r in reqs]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("# rule r0\n", "\n"),  # rule id lost
        ("package authz", "package other"),
        ("package authz\n", "package authz\n\nimport rego.v1\n"),
        ("default allow := false", "default allow := true"),
        ("\tinput.action", "\tnot input.action"),  # negation is not in the grammar
        ("\tinput.action", '\tinput.context.mfa == true with input.context as {"mfa": true}\n\tinput.action'),
        ("\tinput.action", "\tinput.action\n\tinput.action"),  # body order / extra test
        ("permit if {", "permit if {\n\tinput.principal.level == 1"),
    ],
)
def test_decoder_rejects_text_outside_the_renderer_grammar(old, new):
    p = _policy([Rule("r0", "permit", Scope("is", type="User"), ("read",), Scope())])
    text = render(p, "rego")
    assert old in text
    with pytest.raises((RenderError, ValueError)):
        decode(text.replace(old, new, 1), "rego")


def test_opa_errors_are_surfaced_per_request():
    p = fuzz_policies(5)[0]
    reqs = universe(p, ("User",), ("Repo",))[:5]
    bad = rego.opa_decisions(p, reqs, policy_text="package authz\n\nallow if {\n")
    assert all(b["errors"] and b["errors"][0].startswith("opa eval failed") for b in bad)
    undefined = rego.opa_decisions(p, reqs, policy_text="package authz\n\nallow if false\n")
    assert all(u["errors"] == ["allow is undefined"] and u["decision"] == "deny" for u in undefined)
