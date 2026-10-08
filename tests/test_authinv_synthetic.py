"""Synthetic grammars: balanced design, determinism, non-degeneracy, certification."""

from __future__ import annotations

from collections import Counter

import pytest

from authinv.policy import policy_hash, tier
from authinv.sources.synthetic import DOMAINS, balance_table, design_grid, generate, is_non_degenerate


def test_design_grid_is_the_documented_24_cells():
    grid = design_grid()
    assert len(grid) == 24
    assert Counter(c["tier"] for c in grid) == {"single": 1, "multi_rule": 5, "conditioned": 18}
    assert {
        (c["tier"], c["roles"], c["conditions"], c["effect"]) for c in grid if c["tier"] == "multi_rule"
    } == {
        ("multi_rule", 1, 0, "with_forbid"),
        ("multi_rule", 2, 0, "permit_only"),
        ("multi_rule", 2, 0, "with_forbid"),
        ("multi_rule", 3, 0, "permit_only"),
        ("multi_rule", 3, 0, "with_forbid"),
    }


@pytest.mark.parametrize("domain", ["repo", "mcp"])
def test_generation_is_deterministic_balanced_and_non_degenerate(domain):
    worlds, report = generate(domain, replicates=2, seed=11)
    again, _ = generate(domain, replicates=2, seed=11)
    assert [policy_hash(w) for w in worlds] == [policy_hash(w) for w in again]
    assert len(worlds) == 48 and report["worlds"] == 48
    d = DOMAINS[domain]
    for w in worlds:
        meta = dict(w.meta)
        assert tier(w) == meta["tier_cell"]
        assert sum(r.effect == "forbid" for r in w.rules) == (meta["effect"] == "with_forbid")
        assert sum(len(r.conditions) for r in w.rules) == meta["conditions"]
        assert is_non_degenerate(w, d, 4)[0]
    counts = {(r["factor"], r["level"]): r["worlds"] for r in balance_table(worlds)}
    assert counts[("tier_cell", "conditioned")] == 36 and counts[("effect", "with_forbid")] == 2 * (3 + 9)


def test_different_seeds_give_different_worlds():
    a, _ = generate("repo", replicates=1, seed=1)
    b, _ = generate("repo", replicates=1, seed=2)
    assert [policy_hash(w) for w in a] != [policy_hash(w) for w in b]


def test_generated_worlds_are_certified():
    pytest.importorskip("cedarpy")
    from authinv.equivalence.check import certify

    for domain in ("repo", "mcp"):
        d = DOMAINS[domain]
        worlds, _ = generate(domain, replicates=1, seed=5)
        for w in worlds:
            proof = certify(w, (d.principal_type,), (d.resource_type,))
            assert proof["passed"], (
                w.policy_id,
                {k: v for k, v in proof["renderings"].items() if not v["passed"]},
            )
