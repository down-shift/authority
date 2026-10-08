"""Equivalence checker: certified worlds pass; tampered renderings are caught."""

from __future__ import annotations

import json

import pytest
from authinv_fuzz import random_policy

from authinv.render.renderers import RENDERINGS, render

pytest.importorskip("cedarpy")
from authinv.equivalence.check import certify, certify_texts, write_proofs  # noqa: E402

TYPES = (("User",), ("Repo",))


def policies(n):
    return [p for p in (random_policy(s) for s in range(n)) if p is not None]


def test_random_worlds_are_certified_with_full_proofs(tmp_path):
    records = [certify(p, *TYPES) for p in policies(60)]
    assert records and all(r["passed"] for r in records), [r for r in records if not r["passed"]][:1]
    r = records[0]
    assert set(r["renderings"]) == set(RENDERINGS) and r["n_requests"] > 0
    ex = r["renderings"]["executable"]
    assert ex["engine_mismatches"] == 0 and ex["engine_errors"] == 0 and ex["typecheck"]
    digest = write_proofs(records, tmp_path / "equivalence.jsonl")
    lines = (tmp_path / "equivalence.jsonl").read_text().splitlines()
    assert len(lines) == len(records) and json.loads(lines[0])["policy_sha256"] == r["policy_sha256"]
    assert len(digest) == 64


def _texts(p):
    return {r: render(p, r) for r in RENDERINGS}


def _first_conditioned(n=200):
    for p in policies(n):
        if any(
            c.op in (">=", "<", ">", "<=") and isinstance(c.right, int) for r in p.rules for c in r.conditions
        ):
            return p
    pytest.fail("no policy with an integer threshold")


@pytest.mark.parametrize(
    "rendering", ["nl_statement", "owner_statement", "table", "json_policy", "executable"]
)
def test_flipping_an_effect_is_caught(rendering):
    p = policies(10)[0]
    texts = _texts(p)
    swaps = {
        "nl_statement": (" may ", " must never "),
        "owner_statement": (" decides whether", " never decides whether"),
        "table": ("| allow |", "| deny |"),
        "json_policy": ('"effect": "allow"', '"effect": "deny"'),
        "executable": ("permit (", "forbid ("),
    }
    a, b = swaps[rendering]
    if a not in texts[rendering]:
        a, b = b, a
    texts[rendering] = texts[rendering].replace(a, b, 1)
    out = certify_texts(p, texts, *TYPES)
    assert not out["passed"] and not out["renderings"][rendering]["passed"]
    assert all(out["renderings"][r]["passed"] for r in RENDERINGS if r != rendering)


def test_changing_a_threshold_is_caught_semantically():
    import re

    p = _first_conditioned()
    texts = _texts(p)
    texts["json_policy"] = re.sub(
        r'("right": )(\d+)', lambda m: f"{m.group(1)}{int(m.group(2)) + 7}", texts["json_policy"], count=1
    )
    c = certify_texts(p, texts, *TYPES)["renderings"]["json_policy"]
    assert not c["decodes"] and not c["passed"]


def test_dropping_a_rule_and_garbage_text_are_caught():
    p = next(q for q in policies(100) if len(q.rules) >= 2)
    texts = _texts(p)
    texts["table"] = "\n".join(texts["table"].splitlines()[:-1]) + "\n"
    texts["nl_statement"] = "Everyone may do anything.\n"
    out = certify_texts(p, texts, *TYPES)["renderings"]
    assert not out["table"]["passed"] and not out["nl_statement"]["passed"] and "error" in out["nl_statement"]


def test_engine_disagreement_is_caught_even_if_decoding_matched():
    p = policies(10)[0]
    texts = _texts(p)
    # An `unless` clause the decoder refuses: decode fails and the engine sees different semantics.
    texts["executable"] = texts["executable"].rstrip().rstrip(";") + " unless { true };\n"
    c = certify_texts(p, texts, *TYPES)["renderings"]["executable"]
    assert not c["passed"]
