"""Benchmark builder: name swap preserves semantics, prompts differ only in the policy, rows are balanced."""

from __future__ import annotations

import collections

import pytest

from authinv.benchmark import World, non_degenerate, rename, swap_map, world_rows
from authinv.equivalence.requests import universe
from authinv.policy import evaluate
from authinv.render.renderers import RENDERINGS
from authinv.sources.synthetic import generate

pytest.importorskip("cedarpy")


def worlds(n=6):
    ws, _ = generate("repo", replicates=1, seed=3)
    return [World(p, ("User",), ("Repo",), "synthetic", "synthetic", dict(p.meta)) for p in ws[:n]]


def test_swap_is_a_derangement_and_preserves_decisions():
    for w in worlds():
        m = swap_map(w.policy, w.principal_types)  # synthetic worlds have >= 2 principals
        assert all(k != v for k, v in m.items()) and len(set(m.values())) == len(m)
        swapped = rename(w.policy, m)
        for req in universe(w.policy, w.principal_types, w.resource_types):
            from authinv.benchmark import rename_request

            assert (
                evaluate(swapped, rename_request(req, m))["decision"] == evaluate(w.policy, req)["decision"]
            )


def test_rows_are_balanced_and_prompts_differ_only_in_the_policy():
    w = next(x for x in worlds(24) if non_degenerate(x, 4))
    rows, extra = world_rows(w, per_label=2, seed=1)
    app = [r for r in rows if r["task"] == "application"]
    assert collections.Counter(r["label"] for r in app) == {
        "allow": 2 * 2 * len(RENDERINGS),
        "deny": 2 * 2 * len(RENDERINGS),
    }
    assert (
        collections.Counter(r["option_order"] for r in app)[0]
        == collections.Counter(r["option_order"] for r in app)[1]
    )
    by_inst = collections.defaultdict(dict)
    for r in app:
        by_inst[(r["assignment"], r["instance"])][r["rendering"]] = r["prompt"]
    for prompts in by_inst.values():
        assert set(prompts) == set(RENDERINGS)
        heads = {p.split("\nPolicy:\n")[0] for p in prompts.values()}
        tails = {p.split("\n\nRequest:")[1] for p in prompts.values()}
        assert len(heads) == 1 and len(tails) == 1 and len(set(prompts.values())) == len(RENDERINGS)
    assert len({r["row_id"] for r in rows}) == len(rows)


def test_interpretation_expected_matches_decision_owners_and_pairs_exist():
    w = next(x for x in worlds(24) if non_degenerate(x, 4))
    rows, _ = world_rows(w, per_label=2, seed=1)
    interp = [r for r in rows if r["task"] == "interpretation"]
    app_keys = {(r["assignment"], r["pair_key"]) for r in rows if r["task"] == "application"}
    assert interp and {(r["assignment"], r["pair_key"]) for r in interp} == app_keys
    for r in interp:
        assert set(r["expected"]) <= set(r["principal_ids"])


def test_swapped_policy_is_recertified():
    from authinv.equivalence.check import certify

    w = worlds(1)[0]
    _, extra = world_rows(w, per_label=2, seed=1)
    try:
        assert certify(extra["swap_policy"], w.principal_types, w.resource_types)["passed"]
    except FileNotFoundError:
        pytest.skip("OPA binary not installed")


def test_single_principal_worlds_swap_resources():
    from authinv.benchmark import rename_request, swap_axis
    from authinv.policy import Entity, EntityRef, Policy, Rule, Scope, validate

    me = EntityRef("Caller", "c1")
    b1, b2, b3 = (EntityRef("Bucket", x) for x in ("b1", "b2", "b3"))
    p = Policy(
        "iam",
        (Entity(me), Entity(b1), Entity(b2), Entity(b3)),
        ("get", "put"),
        (Rule("r1", "permit", Scope("eq", me), ("get",), Scope("eq", b1)),),
        (),
    )
    validate(p)
    axis = swap_axis(p, ("Caller",), ("Bucket",))
    assert axis == ("Bucket",)
    m = swap_map(p, axis)
    q = rename(p, m)
    for req in universe(p, ("Caller",), ("Bucket",)):
        assert evaluate(q, rename_request(req, m))["decision"] == evaluate(p, req)["decision"]
