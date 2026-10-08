"""Quacky AWS IAM import (P1.8, v2 P1.8.2) on a hand-written fixture: evaluator, translation, faithfulness."""

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
V1, V2 = quacky.IMPORTER_V1, quacky.IMPORTER_V2


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
        iam_eval.evaluate(_doc("ec2/exp_single/if_exists/policy.json"), IamRequest("ec2:StartInstances", "x"))
    user = {"AWS": "arn:aws:iam::111122223333:user/alice"}  # matched via its account too: not single-identity
    doc = {"Statement": [{"Effect": "Deny", "NotPrincipal": user, "Action": "s3:*", "Resource": "*"}]}
    with pytest.raises(iam_eval.Unsupported):
        iam_eval.evaluate(doc, IamRequest("s3:GetObject", Q1, ("AWS", "x")))
    both = {"Statement": [{"Effect": "Allow", "Action": "s3:*", "NotAction": "s3:Get*", "Resource": "*"}]}
    with pytest.raises(iam_eval.Unsupported):
        iam_eval.evaluate(both, IamRequest("s3:GetObject", Q1))


def test_evaluator_not_elements_are_complements():
    doc = _doc("iam/exp_single/notaction/policy.json")  # Allow NotAction iam:DeleteUser on *
    assert iam_eval.evaluate(doc, IamRequest("iam:CreateUser", "x")) == "allow"
    assert iam_eval.evaluate(doc, IamRequest("IAM:deleteuser", "x")) == "deny"  # actions fold case
    nr = {"Statement": [{"Effect": "Allow", "Action": "s3:*", "NotResource": "arn:aws:s3:::reports/*"}]}
    assert iam_eval.evaluate(nr, IamRequest("s3:GetObject", Q1)) == "deny"
    assert iam_eval.evaluate(nr, IamRequest("s3:GetObject", "arn:aws:s3:::other/a")) == "allow"
    svc = {"Service": "lambda.amazonaws.com"}
    np = {"Statement": [{"Effect": "Allow", "NotPrincipal": svc, "Action": "s3:*", "Resource": "*"}]}
    assert (
        iam_eval.evaluate(np, IamRequest("s3:GetObject", Q1, ("Service", "lambda.amazonaws.com"))) == "deny"
    )
    assert iam_eval.evaluate(np, IamRequest("s3:GetObject", Q1, ("AWS", "arn:aws:iam::1:user/b"))) == "allow"


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
    conv = _convert("s3/exp_single/bucket_principal/policy.json", version=V1)
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
    strict = _convert("s3/exp_single/bucket_principal/policy.json", expand_wildcards=False, version=V1)
    assert strict.status == "excluded" and "action_wildcard" in strict.constructs


def test_unsupported_constructs_are_counted_and_excluded():
    conv = _convert("iam/exp_single/notaction/policy.json", version=V1)
    assert conv.status == "excluded" and conv.policy is None and conv.constructs == ["NotAction"]
    conv = _convert("ec2/exp_single/if_exists/policy.json")
    assert set(conv.constructs) == {"condition_if_exists", "condition_set_qualifier"}
    base = {"Effect": "Allow", "Action": "s3:GetObject", "Resource": Q1}
    cases = {
        "condition_op:StringLike": {**base, "Condition": {"StringLike": {"s3:prefix": "a*"}}},
        "policy_variable": {**base, "Resource": "arn:aws:s3:::b/${aws:username}"},
        "principal_account": {**base, "Principal": {"AWS": "arn:aws:iam::111122223333:root"}},
        "condition_value_type": {**base, "Condition": {"NumericEquals": {"k": "1.5"}}},
        "NotPrincipal": {**base, "NotPrincipal": {"AWS": "arn:aws:iam::111122223333:user/alice"}},
    }
    for construct, stmt in cases.items():
        for version in (V1, V2):
            got = quacky.convert({"Statement": [stmt]}, "x", "y", version=version)
            assert got.status == "excluded" and construct in got.constructs, (construct, version)
    nr = {"Effect": "Deny", "Action": "s3:GetObject", "NotResource": Q1}
    assert "NotResource" in quacky.convert({"Statement": [nr]}, "x", "y", version=V1).constructs
    only_wild = {"Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "arn:aws:s3:::b/*"}]}
    assert quacky.convert(only_wild, "x", "y", version=V1).constructs == ["no_statement_matches_universe"]
    with pytest.raises(ValueError):
        quacky.convert(only_wild, "x", "y", version="quacky-import-v9")


# ---- differential test: translation faithfulness ----------------------------------


def test_faithfulness_compares_the_original_json_and_catches_a_mismatch():
    doc = _doc("s3/exp_single/bucket_principal/policy.json")
    conv = quacky.convert(doc, "fx", "sha", version=V1)
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


# ---- v2: wildcard witnesses and closed-world complements ---------------------------


def test_witness_matches_its_pattern_and_avoids_the_others():
    pats = ["s3:Get*", "s3:Put*", "s3:*"]
    w, extra = quacky.witness("s3:Get*", pats, set(), axis="action")
    assert w == "s3:GetZzWitness" and extra == ["s3:*"]  # s3:* is more general: unavoidable
    assert iam_eval.glob_match("s3:Get*", w, case_sensitive=False)
    assert not iam_eval.glob_match("s3:Put*", w, case_sensitive=False)
    assert quacky.witness("s3:*", pats, set(), axis="action") == ("s3:ZzWitness", [])
    # an avoidable overlap is avoided by the next token
    assert quacky.witness("s3:*", ["s3:*", "s3:Z*"], set(), axis="action") == ("s3:QqWitness", [])
    # a taken string (case-folded for actions) is skipped
    assert quacky.witness("s3:Get*", pats, {"s3:getzzwitness"}, axis="action")[0] == "s3:GetQqWitness"
    res = "arn:aws:s3:::reports/*"
    assert quacky.witness(res, [res], set(), axis="resource") == ("arn:aws:s3:::reports/zz-witness", [])
    w, _ = quacky.witness("arn:aws:s3:::reports/q?.csv", [], set(), axis="resource")
    assert w == "arn:aws:s3:::reports/qz.csv"
    assert iam_eval.glob_match("arn:aws:s3:::reports/q?.csv", w, case_sensitive=True)
    # witnesses enter the universe and the meta; every one matches its own pattern
    doc = _doc("s3/exp_single/bucket_principal/policy.json")
    conv = quacky.convert(doc, "fx", "sha")
    assert conv.status == "converted" and "wildcard_witness" in conv.flags
    wit = json.loads(dict(conv.policy.meta)["witnesses"])
    assert wit["action"] == {"s3:Get*": {"witness": "s3:GetZzWitness", "also_matches": []}}
    assert set(wit["resource"]) == {"*", "arn:aws:s3:::reports/*"}
    for pat, t in wit["resource"].items():
        assert iam_eval.glob_match(pat, t["witness"], case_sensitive=True)
    assert wit["principal"]["*"]["witness"] == list(quacky.PRINCIPAL_WITNESS)
    assert "s3.GetZzWitness" in conv.policy.actions
    ok = quacky.faithfulness(conv.policy, doc, conv.names)
    assert ok["passed"] and ok["n_requests"] == 3 * 4 * 4 * 3 * 3  # principals x actions x resources x ctx


def test_not_action_and_not_resource_are_exact_complements():
    doc = {
        "Statement": [
            {"Effect": "Allow", "NotAction": ["s3:Delete*", "s3:PutObject"], "Resource": "*"},
            {"Effect": "Deny", "Action": "s3:*", "NotResource": ["arn:aws:s3:::reports/*"]},
        ]
    }
    conv = quacky.convert(doc, "fx/not", "sha")
    assert conv.status == "converted"
    assert {"not_action", "not_resource", "closed_world_complement"} <= set(conv.flags)
    meta = dict(conv.policy.meta)
    assert meta["closed_world_complement"] is True and meta["importer"] == V2
    (allow,) = [r for r in conv.policy.rules if r.effect == "permit"]
    forbid_res = {r.resource.entity.id for r in conv.policy.rules if r.effect == "forbid"}
    assert forbid_res == {"arn.aws.authinv-probe...unlisted-resource", "zz-witness"}  # not reports/*
    assert "s3.DeleteZzWitness" not in allow.actions and "s3.PutObject" not in allow.actions
    assert {"s3.ZzWitness", "authinv-probe.UnlistedAction"} <= set(allow.actions)
    # the canonical decision equals the IAM evaluator's on the original JSON for every request
    reqs = universe(conv.policy, quacky.PRINCIPAL_TYPES, quacky.RESOURCE_TYPES)
    for r in reqs:
        assert evaluate(conv.policy, r)["decision"] == iam_eval.evaluate(
            doc, quacky.to_iam_request(r, conv.names)
        )
    faith = quacky.faithfulness(conv.policy, doc, conv.names)
    assert faith["passed"] and {evaluate(conv.policy, r)["decision"] for r in reqs} == {"allow", "deny"}
    # not vacuous: reading NotAction as Action changes the IAM decisions
    flipped = copy.deepcopy(doc)
    flipped["Statement"][0]["Action"] = flipped["Statement"][0].pop("NotAction")
    assert quacky.faithfulness(conv.policy, flipped, conv.names)["mismatches"] > 0


def test_not_principal_service_is_an_exact_complement():
    from authinv.policy import Request

    svc = {"Service": "lambda.amazonaws.com"}
    doc = {
        "Statement": [
            {"Effect": "Allow", "Principal": "*", "Action": "s3:GetObject", "Resource": Q1},
            {"Effect": "Deny", "NotPrincipal": svc, "Action": "s3:GetObject", "Resource": Q1},
        ]
    }
    conv = quacky.convert(doc, "fx/np", "sha")
    assert conv.status == "converted" and "not_principal" in conv.flags
    faith = quacky.faithfulness(conv.policy, doc, conv.names)
    assert faith["passed"] and faith["n_requests"] == 3 * 2 * 2  # {lambda, witness, probe} x 2 x 2
    lam = next(k for k, v in conv.names["principal"].items() if v == ["Service", "lambda.amazonaws.com"])
    q1 = next(k for k, v in conv.names["resource"].items() if v == Q1)
    for pid in conv.names["principal"]:
        req = Request(EntityRef("Principal", pid), "s3.GetObject", EntityRef("Resource", q1))
        assert evaluate(conv.policy, req)["decision"] == ("allow" if pid == lam else "deny")
    assert quacky.convert(doc, "fx/np", "sha", version=V1).constructs == ["NotPrincipal"]


def test_policies_v1_excluded_now_ship_under_v2():
    only_wild = {"Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "arn:aws:s3:::b/*"}]}
    notaction = _doc("iam/exp_single/notaction/policy.json")
    for doc in (only_wild, notaction):
        assert quacky.convert(doc, "x", "y", version=V1).status == "excluded"
        conv = quacky.convert(doc, "x", "y")
        assert conv.status == "converted" and dict(conv.policy.meta)["importer"] == V2
        assert quacky.faithfulness(conv.policy, doc, conv.names)["passed"]
        reqs = universe(conv.policy, quacky.PRINCIPAL_TYPES, quacky.RESOURCE_TYPES)
        assert {evaluate(conv.policy, r)["decision"] for r in reqs} == {"allow", "deny"}
    conv = quacky.convert(only_wild, "x", "y")
    assert "arn.aws.s3...b_zz-witness" in {e.ref.id for e in conv.policy.entities}


def test_v1_meta_is_unchanged_and_v2_adds_its_keys():
    v1 = dict(_convert("s3/exp_single/allow_basic/policy.json", version=V1).policy.meta)
    v2 = dict(_convert("s3/exp_single/allow_basic/policy.json").policy.meta)
    assert v1["importer"] == V1 and "witnesses" not in v1 and "closed_world_complement" not in v1
    assert v2["importer"] == V2 and v2["closed_world_complement"] is False
    assert quacky.IMPORTER_VERSION == V2


@needs_engines
@requires_opa
def test_v1_excluded_policy_certifies_under_v2():
    rec, policy, proof = quacky.import_policy(FIX / "iam/exp_single/notaction/policy.json", FIX)
    assert rec["importer"] == V2 and rec["shipped"] and rec["certified"] and rec["faithful"]
    assert "closed_world_complement" in rec["flags"]
    old, _, _ = quacky.import_policy(FIX / "iam/exp_single/notaction/policy.json", FIX, version=V1)
    assert old["status"] == "excluded" and old["importer"] == V1


@needs_engines
@requires_opa  # certify() runs Cedar and OPA
def test_import_policy_certifies_and_ships():
    rec, policy, proof = quacky.import_policy(FIX / "s3/exp_single/bucket_principal/policy.json", FIX)
    assert rec["shipped"] and rec["certified"] and rec["faithful"] and rec["subset"] == "original"
    assert proof["passed"] and proof["faithfulness"]["passed"]
    assert rec["non_degenerate"] and rec["n_requests"] == 432
    assert rec["n_boundary_allow"] >= 2 and rec["n_boundary_deny"] >= 2
    v1, _, _ = quacky.import_policy(FIX / "s3/exp_single/bucket_principal/policy.json", FIX, version=V1)
    assert v1["shipped"] and v1["n_requests"] == 108
    assert rec["policy_id"] == "quacky/s3/exp_single/bucket_principal/policy"


@needs_engines
@requires_opa
def test_run_import_on_the_fixture_reports_every_file(tmp_path):
    import authinv_import_quacky as cli

    (tmp_path / "v1").mkdir()
    report = cli.run_import(FIX, GLOBS, tmp_path / "v1", version=V1)
    assert report["importer"] == V1 and report["converted"] == 4
    assert "NotAction" in report["excluded_files_with_construct"]
    report = cli.run_import(FIX, GLOBS, tmp_path)
    assert report["importer"] == V2
    assert report["policy_files"] == 6
    assert report["policy_files_by_subset"] == {"original": 5, "mutation": 1}
    assert report["converted"] == 5 and report["excluded"] == 1
    assert report["faithful"] == report["certified"] == report["shipped_policy_files"] == 5
    assert report["excluded_files_with_construct"] == {"condition_if_exists": 1, "condition_set_qualifier": 1}
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(r["importer"] == V2 for r in rows)
    # v1 dropped the mutation's second statement (it covered nothing); v2's witness keeps it
    mut = next(r for r in rows if r["subset"] == "mutation")
    assert "statement_matches_nothing_in_universe" not in mut["flags"] and "wildcard_witness" in mut["flags"]
    v1_rows = [json.loads(line) for line in (tmp_path / "v1" / "records.jsonl").read_text().splitlines()]
    v1_mut = next(r for r in v1_rows if r["subset"] == "mutation")
    assert "statement_matches_nothing_in_universe" in v1_mut["flags"]
    assert len((tmp_path / "equivalence.jsonl").read_text().splitlines()) == 5
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
