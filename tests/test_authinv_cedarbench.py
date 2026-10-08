"""CedarBench import (P1.7, v2 P1.7.2) on a hand-written fixture: schema, conversion, exclusion,
faithfulness, and v2's exact universe enlargement and `||` split."""

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

V1, V2 = cedarbench.IMPORTER_V1, cedarbench.IMPORTER_V2
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
    assert sorted(len(u.parents) for u in users) == [0, 0, 1, 1]  # two members, two non-members
    v1 = cedarbench.convert(text, schema, "fx/admins", V1).policy
    assert sorted(len(e.parents) for e in v1.entities if e.ref.type == "User") == [0, 1]


def test_unsupported_constructs_are_counted_and_excluded():
    schema, _, text = _load("docs_basic", "ceiling_tagged")
    conv = cedarbench.convert(text, schema, "fx/tagged", V1)
    assert conv.status == "excluded" and conv.policy is None
    assert {"||", "unless"} <= set(conv.constructs)
    # v2 splits the top-level `||`, so what remains is counted per disjunct
    conv = cedarbench.convert(text, schema, "fx/tagged")
    assert conv.status == "excluded" and conv.constructs == ["has", "set", "unless"]
    schema, _, text = _load("realworld/shared_repo", "ceiling_pull")
    conv = cedarbench.convert(text, schema, "fx/pull", V1)
    assert conv.status == "excluded" and "||" in conv.constructs
    conv = cedarbench.convert(text, schema, "fx/pull")
    assert conv.constructs == ["entity_comparison", "in_expression"]
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
    assert report["importer"] == V2
    assert report["scenarios"] == 2 and report["reference_files"] == 5
    assert report["converted"] == 3 and report["excluded"] == 2
    assert report["certified"] == report["faithful"] == report["shipped_reference_files"] == 3
    assert "||" not in report["excluded_files_with_construct"]
    assert report["splits"]["paper_221"]["reference_files"] == 5
    assert report["splits"]["stress_5"]["reference_files"] == 0
    assert report["non_degenerate_unique_worlds"] <= report["shipped_unique_worlds"]
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and all(r["status"] in ("converted", "excluded") for r in rows)
    assert all(r["importer"] == V2 for r in rows)
    assert all("non_degenerate" in r for r in rows if r["status"] == "converted")
    pols = [json.loads(line) for line in (tmp_path / "policies.jsonl").read_text().splitlines()]
    assert len(pols) == 3 and all(dict(map(tuple, p["meta"]))["importer"] == V2 for p in pols)
    assert len((tmp_path / "equivalence.jsonl").read_text().splitlines()) == 3
    # v1 stays reproducible
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    v1 = cli.run_import(FIX, v1_dir, progress=False, importer=V1)
    assert v1["importer"] == V1 and v1["excluded_files_with_construct"]["||"] == 2
    assert v1["converted"] == 3


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


# ---- importer v2 (P1.7.2) --------------------------------------------------------

_HEAD = 'permit (principal is User, action == Action::"view", resource is Doc)'


def _faithful(text: str, schema) -> tuple:
    conv = cedarbench.convert(text, schema, "fx/x")
    assert conv.status == "converted", conv.constructs
    res = cedarbench.faithfulness(conv.policy, text, conv.principal_types, conv.resource_types)
    return conv, res


def test_top_level_disjunction_splits_exactly_into_one_rule_per_disjunct():
    schema, _, _ = _load("docs_basic", "ceiling_view")
    text = _HEAD + ' when { principal.role == "editor" || (resource.classification < 3 && context.mfa) };'
    assert cedarbench.convert(text, schema, "fx/x", V1).constructs == ["||"]
    conv, res = _faithful(text, schema)
    assert conv.split_rules == 1
    rules = {r.rule_id: r for r in conv.policy.rules}
    assert sorted(rules) == ["policy0_or1", "policy0_or2"]
    assert rules["policy0_or1"].conditions == (Condition(AttrRef("principal", "role"), "==", "editor"),)
    assert set(rules["policy0_or2"].conditions) == {
        Condition(AttrRef("resource", "classification"), "<", 3),
        Condition(AttrRef("context", "mfa"), "==", True),
    }
    assert all(r.effect == "permit" and r.actions == ("view",) for r in rules.values())
    # the original text and the split agree on the whole (enlarged) universe, with both outcomes
    assert res["passed"] and res["n_requests"] > 0
    assert {evaluate(conv.policy, r)["decision"] for r in _universe(conv.policy, conv)} == {"allow", "deny"}
    # and the check is sharp: dropping a disjunct breaks faithfulness
    dropped = _HEAD + ' when { principal.role == "editor" };'
    bad = cedarbench.faithfulness(conv.policy, dropped, conv.principal_types, conv.resource_types)
    assert not bad["passed"] and bad["mismatches"] > 0


def test_forbid_disjunction_and_several_when_clauses_split_exactly():
    schema, _, _ = _load("docs_basic", "ceiling_view")
    text = (
        'permit (principal is User, action in [Action::"view", Action::"edit"], resource is Doc);\n'
        'forbid (principal is User, action == Action::"edit", resource is Doc)'
        " when { resource.archived || principal.level < 2 };"
    )
    conv, res = _faithful(text, schema)
    forbids = sorted(r.rule_id for r in conv.policy.rules if r.effect == "forbid")
    assert forbids == ["policy1_or1", "policy1_or2"] and res["passed"]
    text = (
        _HEAD
        + ' when { principal.role == "a" || principal.role == "b" }'
        + " when { context.mfa || resource.archived };"
    )
    conv, res = _faithful(text, schema)
    assert len(conv.policy.rules) == 4 and conv.split_rules == 1 and res["passed"]


def test_nested_or_mixed_disjunction_stays_excluded():
    schema, _, _ = _load("docs_basic", "ceiling_view")
    text = _HEAD + ' when { principal.role == "editor" && (resource.archived || context.mfa) };'
    conv = cedarbench.convert(text, schema, "fx/x")
    assert conv.status == "excluded" and conv.constructs == ["nested_||"]
    text = _HEAD + " when { !(resource.archived || context.mfa) };"
    assert cedarbench.convert(text, schema, "fx/x").constructs == ["negation"]


def test_schema_actions_join_the_universe_as_default_deny_probes():
    schema, _, text = _load("docs_basic", "floor_admins_edit")
    conv = cedarbench.convert(text, schema, "fx/admins")
    p = conv.policy
    assert p.actions == ("archive", "edit", "view")  # archive is a probe
    assert dict(p.meta)["probe_actions"] == "archive"
    assert {r.rule_id: r.actions for r in p.rules} == {
        "admins_edit": ("view", "edit"),
        "no_archived_edit": ("edit",),
    }
    reqs = _universe(p, conv)
    assert {evaluate(p, r)["decision"] for r in reqs if r.action == "archive"} == {"deny"}
    assert cedarbench.faithfulness(p, text, conv.principal_types, conv.resource_types)["passed"]
    assert cedarbench.convert(text, schema, "fx/admins", V1).policy.actions == ("edit", "view")
    # a probe must declare the context keys the policy reads: archive has no `mfa`
    schema, _, text = _load("docs_basic", "ceiling_view")
    p = cedarbench.convert(text, schema, "fx/view").policy
    assert p.actions == ("edit", "view") and dict(p.meta)["probe_actions"] == "edit"


def test_unconstrained_action_covers_exactly_the_compatible_actions():
    schema, _, _ = _load("docs_basic", "ceiling_view")
    text = "permit (principal, action, resource is Doc) when { resource.archived == false };"
    conv, res = _faithful(text, schema)
    assert conv.policy.rules[0].actions == conv.policy.actions == ("archive", "edit", "view")
    assert res["passed"]


def test_synthesized_entities_give_each_condition_both_outcomes_twice():
    schema, _, text = _load("docs_basic", "ceiling_view")
    p = cedarbench.convert(text, schema, "fx/view").policy
    assert dict(p.meta)["entity_replicas"] == 2
    users = [e for e in p.entities if e.ref.type == "User"]
    docs = [e for e in p.entities if e.ref.type == "Doc"]
    for ents, attr, holds in (
        (users, "role", lambda v: v == "editor"),
        (docs, "archived", lambda v: v is False),
    ):
        sides = [holds(e.attr(attr)) for e in ents]
        assert sides.count(True) >= 2 and sides.count(False) >= 2
    assert all(cedarbench.SAFE.match(e.ref.id) for e in p.entities)
    assert len({e.ref for e in p.entities}) == len(p.entities)
    v1 = cedarbench.convert(text, schema, "fx/view", V1).policy
    assert len(v1.entities) * 2 == len(p.entities) and "entity_replicas" not in dict(v1.meta)


def test_version_is_recorded_and_degeneracy_is_reported():
    schema, schema_text, _ = _load("docs_basic", "floor_admins_edit")
    conv = cedarbench.convert(
        (FIX / "docs_basic" / "references" / "floor_admins_edit.cedar").read_text(), schema, "fx/a"
    )
    assert dict(conv.policy.meta)["importer"] == V2 == cedarbench.IMPORTER_VERSION
    d = cedarbench.degeneracy(conv.policy, conv.principal_types, conv.resource_types)
    assert d["n_allow"] >= 4 and d["n_deny"] >= 4 and d["non_degenerate"] and d["dead_rules"] == 0
    assert cedarbench.is_stress("realworld/mega_scale_500_checks") and not cedarbench.is_stress("docs_basic")
    with pytest.raises(ValueError):
        cedarbench.convert("permit(principal, action, resource);", schema, "x", "cedarbench-import-v9")
