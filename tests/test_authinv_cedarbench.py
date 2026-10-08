"""CedarBench import (P1.7) on a hand-written fixture: schema, conversion, exclusion, faithfulness."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from authinv.policy import AttrRef, Condition, EntityRef, Scope, evaluate, from_dict, to_dict, validate

cedarpy = pytest.importorskip("cedarpy")
from authinv_opa import requires_opa  # noqa: E402

from authinv.sources import cedarbench  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "cedarbench" / "scenarios"
sys.path.insert(0, str(ROOT / "scripts"))


def _load(scenario: str, ref: str):
    d = FIX / scenario
    schema_text = (d / "schema.cedarschema").read_text()
    return cedarbench.parse_schema(schema_text), schema_text, (d / "references" / f"{ref}.cedar").read_text()


def test_schema_parser_reads_attrs_aliases_optionals_and_applies_to():
    s, _, _ = _load("docs_basic", "ceiling_view")
    assert s.entities["User"].member_of == ("Group",)
    assert s.entities["User"].attrs["level"] == ("Long", True)  # alias Clearance resolved
    assert s.entities["User"].attrs["nickname"] == ("String", False)
    assert s.entities["Doc"].attrs["tags"] == (("set", "String"), True)
    assert s.actions["view"].principals == ("User",) and s.actions["edit"].resources == ("Doc",)
    assert s.actions["view"].context == {"mfa": ("Bool", True)}
    assert s.actions["archive"].principals == ("User",) and not s.namespaced
    ns = cedarbench.parse_schema(
        "namespace App { entity U; action a appliesTo { principal: U, resource: U }; }"
    )
    assert ns.namespaced and "a" in ns.actions
    with pytest.raises(cedarbench.SchemaError):
        cedarbench.parse_schema("entity U = { a: String ")


def test_convertible_reference_maps_to_an_exact_canonical_policy():
    schema, schema_text, text = _load("docs_basic", "ceiling_view")
    assert cedarbench.source_validates(text, schema_text)["passed"]
    conv = cedarbench.convert(text, schema, "fx/view")
    assert conv.status == "converted" and conv.constructs == []
    p = conv.policy
    validate(p)
    (rule,) = p.rules
    assert rule.principal == Scope("is", type="User") and rule.actions == ("view",)
    assert set(rule.conditions) == {
        Condition(AttrRef("principal", "role"), "==", "editor"),
        Condition(AttrRef("principal", "level"), ">=", AttrRef("resource", "classification")),
        Condition(AttrRef("resource", "archived"), "==", False),
        Condition(AttrRef("context", "mfa"), "==", True),
    }
    assert dict(p.meta)["entities_synthesized"] is True
    assert p.context_schema == (("mfa", "bool"),)
    # the synthesized universe exercises both decisions
    decisions = {evaluate(p, r)["decision"] for r in _universe(p, conv)}
    assert decisions == {"allow", "deny"}
    # deterministic
    again = cedarbench.convert(text, schema, "fx/view").policy
    assert again == p


def _universe(p, conv):
    from authinv.equivalence.requests import universe

    return universe(p, conv.principal_types, conv.resource_types)


def test_groups_action_lists_and_forbids_convert():
    schema, _, text = _load("docs_basic", "floor_admins_edit")
    conv = cedarbench.convert(text, schema, "fx/admins")
    assert conv.status == "converted"
    rules = {r.rule_id: r for r in conv.policy.rules}
    assert rules["admins_edit"].principal == Scope("in", EntityRef("Group", "admins"))
    assert rules["admins_edit"].actions == ("view", "edit")
    assert rules["no_archived_edit"].effect == "forbid"
    users = [e for e in conv.policy.entities if e.ref.type == "User"]
    assert sorted(len(u.parents) for u in users) == [0, 1]  # member and non-member


def test_unsupported_constructs_are_counted_and_excluded():
    schema, _, text = _load("docs_basic", "ceiling_tagged")
    conv = cedarbench.convert(text, schema, "fx/tagged")
    assert conv.status == "excluded" and conv.policy is None
    assert {"||", "unless"} <= set(conv.constructs)
    schema, _, text = _load("realworld/shared_repo", "ceiling_pull")
    conv = cedarbench.convert(text, schema, "fx/pull")
    assert conv.status == "excluded" and "||" in conv.constructs
    # the outermost unsupported construct of each conjunct is what gets counted
    head = 'permit (principal is User, action == Action::"pull", resource is Repo) when '
    single = head + "{ principal in resource.readers };"
    assert cedarbench.convert(single, schema, "x").constructs == ["in_expression"]
    eq = head + "{ resource.owner == principal };"
    assert cedarbench.convert(eq, schema, "x").constructs == ["entity_comparison"]


def test_faithfulness_compares_the_original_text_and_catches_a_mismatch():
    schema, _, text = _load("docs_basic", "ceiling_view")
    conv = cedarbench.convert(text, schema, "fx/view")
    ok = cedarbench.faithfulness(conv.policy, text, conv.principal_types, conv.resource_types)
    assert ok["passed"] and ok["n_requests"] > 0
    tampered = text.replace('"editor"', '"admin"')
    bad = cedarbench.faithfulness(conv.policy, tampered, conv.principal_types, conv.resource_types)
    assert not bad["passed"] and bad["mismatches"] > 0


@requires_opa  # certify() runs Cedar and OPA
def test_import_reference_certifies_and_ships():
    schema, schema_text, _ = _load("docs_basic", "ceiling_view")
    rec, policy, proof = cedarbench.import_reference(
        "docs_basic", FIX / "docs_basic" / "references" / "ceiling_view.cedar", schema, schema_text, None
    )
    assert rec["shipped"] and rec["certified"] and rec["faithful"] and rec["source_validates"]
    assert proof["passed"] and proof["faithfulness"]["passed"]
    assert from_dict(json.loads(json.dumps(to_dict(policy)))) == policy


@requires_opa
def test_run_import_on_the_fixture_reports_every_file(tmp_path):
    import authinv_import_cedarbench as cli

    report = cli.run_import(FIX, tmp_path, progress=False)
    assert report["scenarios"] == 2 and report["reference_files"] == 5
    assert report["converted"] == 3 and report["excluded"] == 2
    assert report["certified"] == report["faithful"] == report["shipped_reference_files"] == 3
    assert report["excluded_files_with_construct"]["||"] == 2
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and all(r["status"] in ("converted", "excluded") for r in rows)
    assert len((tmp_path / "policies.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "equivalence.jsonl").read_text().splitlines()) == 3


def test_fetch_verify_detects_a_changed_tree(tmp_path):
    import fetch_cedarbench

    (tmp_path / "scenarios" / "a" / "references").mkdir(parents=True)
    (tmp_path / "scenarios" / "a" / "schema.cedarschema").write_text("entity U;")
    (tmp_path / "scenarios" / "a" / "references" / "r.cedar").write_text(
        "permit(principal, action, resource);"
    )
    (tmp_path / "LICENSE").write_text("Apache")
    tree, n = fetch_cedarbench.tree_sha256(tmp_path)
    m = {
        "tree_sha256": tree,
        "n_files": n,
        "file_sha256": {"LICENSE": hashlib.sha256(b"Apache").hexdigest()},
        "expected_scenarios": 1,
        "expected_reference_files": 1,
    }
    assert fetch_cedarbench.verify(m, tmp_path) == []
    (tmp_path / "LICENSE").write_text("changed")
    assert len(fetch_cedarbench.verify(m, tmp_path)) == 2
