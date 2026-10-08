"""Equivalence checker: certified worlds pass; tampered renderings are caught."""

from __future__ import annotations

import json

import pytest
from authinv_fuzz import random_policy

from authinv.render.renderers import RENDERINGS, render

pytest.importorskip("cedarpy")
from authinv_opa import requires_opa  # noqa: E402

from authinv.equivalence.check import certify, certify_texts, write_proofs  # noqa: E402

pytestmark = requires_opa  # certify() runs both engines; CI fetches the pinned OPA

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
    rg = r["renderings"]["rego"]
    assert rg["version"] == "rego-v1" and rg["engine_mismatches"] == 0 and rg["engine_errors"] == 0
    assert r["engine"]["opa"]
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
    "rendering", ["nl_statement", "owner_statement", "table", "json_policy", "executable", "rego"]
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
        "rego": ("\npermit if {", "\nforbid if {"),
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


def test_changing_a_rego_threshold_is_caught_by_decoder_and_engine():
    import re

    p = _first_conditioned()
    texts = _texts(p)
    texts["rego"] = re.sub(
        r"(principal\.level (?:>=|<=|<|>) )(\d+)",
        lambda m: f"{m.group(1)}{int(m.group(2)) + 7}",
        texts["rego"],
    )
    texts["rego"] = re.sub(
        r"(input\.context\.hour (?:>=|<) )(\d+)",
        lambda m: f"{m.group(1)}{int(m.group(2)) + 7}",
        texts["rego"],
    )
    out = certify_texts(p, texts, *TYPES)
    c = out["renderings"]["rego"]
    assert not c["decodes"] and not c["passed"] and not out["passed"]
    # The engine runs the tampered text; the decoder reads the same text, so both see the same rules.
    assert c["engine_errors"] == 0 and c["engine_mismatches"] == c["decision_mismatches"]


def test_rego_preamble_tampering_is_caught_even_if_rule_bodies_match():
    p = policies(10)[0]
    texts = _texts(p)
    texts["rego"] = texts["rego"].replace("\tnot forbid\n", "")  # forbid no longer overrides
    c = certify_texts(p, texts, *TYPES)["renderings"]["rego"]
    assert not c["passed"] and not c["decodes"] and "preamble" in c["error"]


def out_n(p):
    from authinv.equivalence.requests import universe

    return len(universe(p, *TYPES))


def test_broken_rego_is_an_engine_error_not_a_silent_deny():
    p = policies(10)[0]
    texts = _texts(p)
    texts["rego"] = texts["rego"].replace("package authz", "package other")
    c = certify_texts(p, texts, *TYPES)["renderings"]["rego"]
    assert not c["passed"] and c["engine_errors"] == out_n(p)


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
