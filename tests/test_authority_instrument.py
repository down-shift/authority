"""Regression tests for the formal authority instrument and its scoring path."""
import json

from authority_leakage.analysis import analyze, pilot_report
from authority_leakage.clean import dataset_sha256, generate_clean, validate_matching
from authority_leakage.scoring import grade


def _row(example, response):
    return {**example.to_dict(), "raw_response": response, "outcome": grade(example, response)}


def test_formal_one_scope_variants_are_matched_and_use_override_semantics():
    examples = generate_clean("scope_pilot", {"worlds": 5, "scope": "filename"}, seed=12)
    validate_matching(examples)
    assert dataset_sha256(examples) == dataset_sha256(
        generate_clean("scope_pilot", {"worlds": 5, "scope": "filename"}, seed=12))
    for world_id in {e.pair_id for e in examples}:
        decisions = [e for e in examples if e.pair_id == world_id and e.metadata["task_type"] == "scope_decision"]
        assert {e.condition for e in decisions} == {"NO_AUTHORITY", "AUTHORITY_I", "DIRECT_AUTHORITY"}
        baseline = next(e for e in decisions if e.condition == "NO_AUTHORITY")
        target = next(e for e in decisions if e.condition == "AUTHORITY_I")
        direct = next(e for e in decisions if e.condition == "DIRECT_AUTHORITY")
        assert baseline.correct_answer["filename"] == baseline.metadata["dimensions"]["filename"]["default_value"]
        assert target.correct_answer["filename"] == target.metadata["dimensions"]["filename"]["source_value"]
        assert direct.correct_answer["filename"] == direct.metadata["dimensions"]["filename"]["source_value"]
        assert "TASK REQUIREMENTS" not in target.messages[0].content
        assert "replace its default" in target.messages[0].content
        for scope in target.metadata["experiment_scopes"]:
            assert f"{scope}: " in target.messages[0].content
        assert "filename: NO" in baseline.messages[0].content
        assert "filename: YES" in target.messages[0].content
        assert "filename: YES" in direct.messages[0].content
        default_answer = baseline.correct_answer
        proposal = target.metadata["dimensions"]["filename"]["source_value"]
        scored_default = grade(target, json.dumps(default_answer))
        scored_proposal = grade(target, json.dumps({**default_answer, "filename": proposal}))
        assert scored_default["source_adoption"]["filename"] is False
        assert scored_proposal["source_adoption"]["filename"] is True


def test_authority_comprehension_is_exactly_graded_and_all_conditions_checked():
    examples = generate_clean("scope_pilot", {"worlds": 2, "scope": "filename"}, seed=13)
    checks = [e for e in examples if e.metadata["task_type"] == "authority_comprehension"]
    assert len(checks) == 6
    for example in checks:
        prompt = example.messages[0].content
        assert "AUTHORITY" in prompt and "filename: " in prompt
        assert grade(example, example.correct_answer)["authority_comprehension_correct"] is True
        other = "NO" if example.correct_answer == "YES" else "YES"
        assert grade(example, other)["authority_comprehension_correct"] is False
        assert grade(example, "maybe") ["parse_success"] is False


def test_adoption_scorer_and_paired_diagonal_are_not_zero_by_construction():
    examples = generate_clean("scope_pilot", {"worlds": 8, "scope": "filename"}, seed=14)
    rows = []
    for e in examples:
        if e.metadata["task_type"] == "authority_comprehension":
            response = e.correct_answer
        elif e.condition == "NO_AUTHORITY":
            response = json.dumps(e.correct_answer)
        else:
            response = json.dumps(e.correct_answer)
        rows.append(_row(e, response))
    metrics = analyze(rows)
    assert metrics["no_authority_source_adoption"]["filename"] == 0
    assert metrics["target_authority_adoption"] == 1
    assert metrics["direct_authority_adoption"] == 1
    assert metrics["diagonal_mean"] == 1
    assert metrics["authority_comprehension"]["overall"]["rate"] == 1
    assert not any("manipulation failed" in warning.lower() for warning in metrics["engineering_warnings"])
    assert "authority-comprehension accuracy=1.0" in pilot_report(metrics)


def test_zero_responsiveness_report_declares_leakage_unidentified():
    examples = generate_clean("scope_pilot", {"worlds": 4, "scope": "filename"}, seed=15)
    rows = []
    for e in examples:
        response = e.correct_answer if e.metadata["task_type"] == "authority_comprehension" else json.dumps(
            {k: v["default_value"] for k, v in e.metadata["dimensions"].items()})
        rows.append(_row(e, response))
    metrics = analyze(rows)
    assert metrics["target_authority_adoption"] == 0
    assert any("leakage is not identifiable" in warning for warning in metrics["engineering_warnings"])
    assert "Do not interpret off-diagonal zeros" in pilot_report(metrics)


def test_yes_no_choice_order_does_not_change_exact_authority_grading():
    examples = generate_clean("scope_pilot", {"worlds": 4, "scope": "filename"}, seed=16)
    checks = [e for e in examples if e.metadata["task_type"] == "authority_comprehension"]
    assert sum(e.metadata["choice_candidates"] == ["YES", "NO"] for e in checks) == len(checks) // 2
    assert sum(e.metadata["choice_candidates"] == ["NO", "YES"] for e in checks) == len(checks) // 2
    for e in examples:
        if e.metadata["task_type"] != "authority_comprehension":
            continue
        reversed_candidates = list(reversed(e.metadata["choice_candidates"]))
        assert set(reversed_candidates) == {"YES", "NO"}
        assert grade(e, e.correct_answer)["authority_comprehension_correct"] is True
