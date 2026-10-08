"""Cedar translation and engine agreement with the reference semantics; request sets."""

from __future__ import annotations

import random

import pytest
from authinv_fuzz import ACTIONS, random_policy, world

from authinv.equivalence.requests import context_values, label_universe, sample_requests, universe
from authinv.policy import AttrRef, Condition, Policy, Rule, Scope, evaluate, validate

cedarpy = pytest.importorskip("cedarpy")
from authinv.equivalence import cedar  # noqa: E402


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
