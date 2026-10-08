"""Quacky AWS IAM import (P1.8) on a hand-written fixture: evaluator, translation, exclusion, faithfulness."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from authinv_opa import requires_opa

from authinv.equivalence.requests import universe
from authinv.policy import AttrRef, Condition, EntityRef, Scope, evaluate, from_dict, to_dict, validate
from authinv.sources import iam_eval, quacky
from authinv.sources.iam_eval import IamRequest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "quacky" / "samples"
GLOBS = yaml.safe_load((ROOT / "configs" / "authinv" / "sources" / "quacky.yaml").read_text())["import_globs"]
sys.path.insert(0, str(ROOT / "scripts"))
needs_engines = pytest.mark.skipif(importlib.util.find_spec("cedarpy") is None, reason="needs cedarpy")
Q1, Q2 = "arn:aws:s3:::reports/q1.csv", "arn:aws:s3:::reports/q2.csv"


def _doc(rel: str) -> dict:
    return json.loads((FIX / rel).read_text())


def _convert(rel: str, **kw):
    return quacky.convert(_doc(rel), f"fx/{rel}", "sha", **kw)


# ---- the reference IAM evaluator -------------------------------------------------


def test_evaluator_deny_overrides_allow_and_defaults_to_deny():
    doc = _doc("s3/exp_single/deny_overrides/policy.json")
    assert iam_eval.evaluate(doc, IamRequest("s3:GetObject", Q1)) == "allow"
    assert iam_eval.evaluate(doc, IamRequest("s3:DeleteObject", Q1)) == "deny"  # explicit deny
    assert iam_eval.evaluate(doc, IamRequest("s3:DeleteObject", Q2)) == "allow"
    assert iam_eval.evaluate(doc, IamRequest("s3:PutObject", Q2)) == "deny"  # implicit deny


def test_evaluator_matching_rules():
    assert iam_eval.glob_match("s3:Get*", "S3:getobject", case_sensitive=False)
    assert not iam_eval.glob_match("arn:aws:s3:::Reports/*", Q1, case_sensitive=True)
    assert iam_eval.glob_match("arn:aws:s3:::reports/q?.csv", Q1, case_sensitive=True)
    assert not iam_eval.glob_match("arn:aws:s3:::reports", Q1, case_sensitive=True)
    doc = _doc("s3/exp_single/bucket_principal/policy.json")
    alice = ("AWS", "arn:aws:iam::111122223333:user/alice")
    ctx = {"aws:RequestedRegion": "us-east-1", "aws:SourceVpc": "vpc-2"}
    assert iam_eval.evaluate(doc, IamRequest("s3:GetObject", Q1, alice, ctx)) == "allow"
    assert iam_eval.evaluate(doc, IamRequest("s3:GetObject", Q1, ("AWS", "bob"), ctx)) == "deny"
    # StringNotEquals with several values: holds only when the value is none of them
    assert (
        iam_eval.evaluate(doc, IamRequest("s3:GetObject", Q1, alice, {**ctx, "aws:SourceVpc": "x"})) == "deny"
    )
    assert iam_eval.evaluate(
        doc, IamRequest("s3:GetObject", Q1, alice, {**ctx, "aws:RequestedRegion": "x"})
    ) == ("deny")


def test_evaluator_refuses_unsupported_constructs():
    with pytest.raises(iam_eval.Unsupported):
        iam_eval.evaluate(_doc("iam/exp_single/notaction/policy.json"), IamRequest("iam:CreateUser", "x"))
    with pytest.raises(iam_eval.Unsupported):
        iam_eval.evaluate(_doc("ec2/exp_single/if_exists/policy.json"), IamRequest("ec2:StartInstances", "x"))


# ---- translation -----------------------------------------------------------------


def test_convertible_allow_maps_to_an_exact_canonical_policy():
    conv = _convert("s3/exp_single/allow_basic/policy.json")
    assert conv.status == "converted" and conv.constructs == []
    p = conv.policy
    validate(p)
    (rule,) = p.rules
    assert rule.rule_id == "s0_ReadOverTls" and rule.effect == "permit"
    assert rule.actions == ("s3.GetObject", "s3.PutObject") and rule.principal == Scope()
    assert rule.resource == Scope("eq", EntityRef("Resource", "arn.aws.s3...reports_q1.csv"))
    assert set(rule.conditions) == {
        Condition(AttrRef("context", "aws_SecureTransport"), "==", True),
        Condition(AttrRef("context", "s3_max_keys"), "<", 10),
    }
    assert set(p.actions) == {"s3.GetObject", "s3.PutObject", "authinv-probe.UnlistedAction"}
    assert [e.ref.id for e in p.entities if e.ref.type == "Principal"] == ["caller"]
    meta = dict(p.meta)
    assert meta["identity_policy"] is True and meta["wildcard_expanded"] is False
    assert json.loads(meta["names"])["resource"]["arn.aws.s3...reports_q1.csv"] == Q1
    reqs = universe(p, quacky.PRINCIPAL_TYPES, quacky.RESOURCE_TYPES)
    assert len(reqs) == 3 * 2 * 2 * 3  # actions x resources x bool x int neighbours of 10
    assert {evaluate(p, r)["decision"] for r in reqs} == {"allow", "deny"}
    assert _convert("s3/exp_single/allow_basic/policy.json").policy == p  # deterministic
    assert from_dict(json.loads(json.dumps(to_dict(p)))) == p


def test_deny_statement_becomes_a_forbid_that_overrides():
    conv = _convert("s3/exp_single/deny_overrides/policy.json")
    p = conv.policy
    rules = {r.rule_id: r for r in p.rules}
    assert rules["s1_NoDelete"].effect == "forbid"
    q1 = EntityRef("Resource", "arn.aws.s3...reports_q1.csv")
    assert rules["s1_NoDelete"].resource == Scope("eq", q1)
    # the Allow names two of the three universe resources, so it splits into one rule per resource
    assert {r for r in rules if r.startswith("s0")} == {"s0_0", "s0_1"}
    from authinv.policy import Request

    caller = EntityRef("Principal", "caller")
    assert evaluate(p, Request(caller, "s3.DeleteObject", q1))["decision"] == "deny"
    assert evaluate(p, Request(caller, "s3.GetObject", q1))["decision"] == "allow"


def test_resource_policy_principals_wildcards_and_multi_value_split():
    conv = _convert("s3/exp_single/bucket_principal/policy.json")
    assert conv.status == "converted"
    assert {"wildcard_expanded", "condition_multi_value", "statement_split"} <= set(conv.flags)
    p = conv.policy
    principals = sorted(e.ref.id for e in p.entities if e.ref.type == "Principal")
    assert principals == [
        "arn.aws.iam..000000000000.user_authinv-unlisted-caller",
        "arn.aws.iam..111122223333.user_alice",
    ]
    permits = [r for r in p.rules if r.effect == "permit"]
    assert len(permits) == 2  # one per StringEquals value (OR)
    assert {c.right for r in permits for c in r.conditions} == {"eu-west-1", "us-east-1"}
    assert all(r.actions == ("s3.GetObject",) for r in permits)  # s3:Get* over the closed universe
    (forbid,) = [r for r in p.rules if r.effect == "forbid"]
    assert forbid.principal == Scope() and forbid.resource == Scope()
    assert {(c.op, c.right) for c in forbid.conditions} == {("!=", "vpc-1"), ("!=", "vpc-2")}
    # strict mode refuses wildcards instead of expanding them
    strict = _convert("s3/exp_single/bucket_principal/policy.json", expand_wildcards=False)
    assert strict.status == "excluded" and "action_wildcard" in strict.constructs


def test_unsupported_constructs_are_counted_and_excluded():
    conv = _convert("iam/exp_single/notaction/policy.json")
    assert conv.status == "excluded" and conv.policy is None and conv.constructs == ["NotAction"]
    conv = _convert("ec2/exp_single/if_exists/policy.json")
    assert set(conv.constructs) == {"condition_if_exists", "condition_set_qualifier"}
    base = {"Effect": "Allow", "Action": "s3:GetObject", "Resource": Q1}
    cases = {
        "condition_op:StringLike": {**base, "Condition": {"StringLike": {"s3:prefix": "a*"}}},
        "policy_variable": {**base, "Resource": "arn:aws:s3:::b/${aws:username}"},
        "NotResource": {"Effect": "Deny", "Action": "s3:GetObject", "NotResource": Q1},
        "principal_account": {**base, "Principal": {"AWS": "arn:aws:iam::111122223333:root"}},
        "condition_value_type": {**base, "Condition": {"NumericEquals": {"k": "1.5"}}},
    }
    for construct, stmt in cases.items():
        got = quacky.convert({"Statement": [stmt]}, "x", "y")
        assert got.status == "excluded" and construct in got.constructs, construct
    only_wild = {"Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "arn:aws:s3:::b/*"}]}
    assert quacky.convert(only_wild, "x", "y").constructs == ["no_statement_matches_universe"]


# ---- differential test: translation faithfulness ----------------------------------


def test_faithfulness_compares_the_original_json_and_catches_a_mismatch():
    doc = _doc("s3/exp_single/bucket_principal/policy.json")
    conv = quacky.convert(doc, "fx", "sha")
    ok = quacky.faithfulness(conv.policy, doc, conv.names)
    assert ok["passed"] and ok["n_requests"] == 2 * 3 * 2 * 3 * 3
    tampered = copy.deepcopy(doc)
    tampered["Statement"][1]["Condition"]["StringNotEquals"]["aws:SourceVpc"] = ["vpc-1"]
    bad = quacky.faithfulness(conv.policy, tampered, conv.names)
    assert not bad["passed"] and bad["mismatches"] > 0 and "request" in bad["example"]
    # a wrong translation is caught too: flip the forbid into a permit
    from dataclasses import replace

    wrong = replace(conv.policy, rules=tuple(replace(r, effect="permit") for r in conv.policy.rules))
    assert quacky.faithfulness(wrong, doc, conv.names)["mismatches"] > 0


@needs_engines
@requires_opa  # certify() runs Cedar and OPA
def test_import_policy_certifies_and_ships():
    rec, policy, proof = quacky.import_policy(FIX / "s3/exp_single/bucket_principal/policy.json", FIX)
    assert rec["shipped"] and rec["certified"] and rec["faithful"] and rec["subset"] == "original"
    assert proof["passed"] and proof["faithfulness"]["passed"]
    assert rec["non_degenerate"] and rec["n_requests"] == 108
    assert rec["policy_id"] == "quacky/s3/exp_single/bucket_principal/policy"


@needs_engines
@requires_opa
def test_run_import_on_the_fixture_reports_every_file(tmp_path):
    import authinv_import_quacky as cli

    report = cli.run_import(FIX, GLOBS, tmp_path)
    assert report["policy_files"] == 6
    assert report["policy_files_by_subset"] == {"original": 5, "mutation": 1}
    assert report["converted"] == 4 and report["excluded"] == 2
    assert report["faithful"] == report["certified"] == report["shipped_policy_files"] == 4
    assert report["excluded_files_with_construct"] == {
        "NotAction": 1,
        "condition_if_exists": 1,
        "condition_set_qualifier": 1,
    }
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    # the mutation's second statement covers nothing in its universe and is dropped
    mut = next(r for r in rows if r["subset"] == "mutation")
    assert "statement_matches_nothing_in_universe" in mut["flags"]
    assert len((tmp_path / "equivalence.jsonl").read_text().splitlines()) == 4
    assert len((tmp_path / "policies.jsonl").read_text().splitlines()) == report["shipped_unique_worlds"]


def test_fetch_verify_detects_a_changed_tree(tmp_path):
    import fetch_quacky
    from fetch_cedarbench import tree_sha256

    (tmp_path / "samples" / "s3" / "exp_single" / "a").mkdir(parents=True)
    (tmp_path / "samples" / "s3" / "exp_single" / "a" / "policy.json").write_text('{"Statement": []}')
    (tmp_path / "samples" / "mutations").mkdir()
    (tmp_path / "samples" / "mutations" / "m.json").write_text('{"Statement": []}')
    (tmp_path / "LICENSE").write_text("BSD")
    tree, n = tree_sha256(tmp_path)
    m = {
        "tree_sha256": tree,
        "n_files": n,
        "file_sha256": {"LICENSE": hashlib.sha256(b"BSD").hexdigest()},
        "import_globs": GLOBS,
        "expected_policies": 2,
        "expected_originals": 1,
    }
    assert fetch_quacky.verify(m, tmp_path) == []
    (tmp_path / "LICENSE").write_text("changed")
    assert len(fetch_quacky.verify(m, tmp_path)) == 2
