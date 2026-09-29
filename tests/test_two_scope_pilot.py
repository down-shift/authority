from authority_leakage.scoring import grade
import pytest
from authority_leakage.two_scope_pilot import (
    CONDITIONS, SCOPES, analyze_two_scope_pilot, generate_two_scope_pilot,
    validate_two_scope_matching,
)


def _examples(worlds=3, seed=19):
    return generate_two_scope_pilot({"worlds": worlds}, seed)


def test_every_world_has_all_four_conditions_and_both_candidate_mappings():
    examples = _examples()
    assert len(examples) == 3 * 4 * 2 * 2 * 2
    assert len(validate_two_scope_matching(examples)) == 3 * 4 * 2 * 2
    for world in {e.pair_id for e in examples}:
        rows = [e for e in examples if e.pair_id == world]
        assert {e.metadata["authority_condition"] for e in rows} == set(CONDITIONS)
        assert {e.metadata["field"] for e in rows} == {"filename", "ordering"}
        assert "numeric_answer" not in {e.metadata["field"] for e in rows}


def test_matched_conditions_change_only_owner_assignments_and_values_are_stable():
    examples = _examples(1)
    for field in SCOPES:
        for task in ("scope_value_choice", "authority_comprehension"):
            rows = [e for e in examples if e.metadata["field"] == field and e.metadata["task_type"] == task and e.metadata["mapping_id"] == 1]
            assert {e.metadata["authority_condition"] for e in rows} == set(CONDITIONS)
            assert len({e.metadata["default_values"][field] for e in rows}) == 1
            assert len({e.metadata["source_values"][field] for e in rows}) == 1
            prompts = {_normalize(e.messages[0].content) for e in rows}
            assert len(prompts) == 1
    for e in examples:
        md = e.metadata
        assert md["default_values"][md["field"]] != md["source_values"][md["field"]]
        assert md["expected_owner"] == ("Source S" if md["authority_condition"][SCOPES.index(md["field"])] == "1" else "Default policy")


def _normalize(prompt):
    lines = prompt.splitlines()
    i = lines.index("FINAL DECISION OWNER") + 1
    lines[i:i + 2] = ["filename: <OWNER>", "ordering: <OWNER>"]
    for j, line in enumerate(lines):
        if line.startswith("A: "):
            lines[j:j + 2] = ["A: <A>", "B: <B>"]
            break
    return "\n".join(lines)


def test_handwritten_source_and_default_answers_score_correctly():
    examples = _examples(1)
    for e in examples:
        if e.metadata["task_type"] != "scope_value_choice":
            continue
        out = grade(e, e.correct_answer)
        assert out["parse_success"] is True
        assert out["accuracy"] is True
        assert out["source_adoption"] is (e.metadata["authority_condition"][SCOPES.index(e.metadata["field"])] == "1")


def _synthetic_results(examples):
    expected_margins = {
        "filename": {"00": -2, "10": 3, "01": -1, "11": 3},
        "ordering": {"00": -2, "10": -1, "01": 4, "11": 4},
    }
    rows = []
    for e in examples:
        md = e.metadata
        field, condition, mapping, task = md["field"], md["authority_condition"], md["mapping_id"], md["task_type"]
        source_margin = expected_margins[field][condition]
        world_offset = int(md["world_id"].split("-")[-1]) * .1
        source_margin += world_offset
        if task == "authority_comprehension":
            source_margin = 6 if md["expected_owner"] == "Source S" else -6
        bias = 1.25
        oriented = source_margin + bias if mapping == 1 else source_margin - bias
        scores = {"A": oriented, "B": 0.0} if mapping == 1 else {"B": oriented, "A": 0.0}
        answer = max(scores, key=scores.get)
        rows.append({**e.to_dict(), "candidate_logprobs": scores, "outcome": grade(e, answer, scores)})
    return rows


def test_effects_are_paired_baseline_differences_and_interactions():
    metrics = analyze_two_scope_pilot(_synthetic_results(_examples(4)), seed=2)
    assert metrics["effects"]["R_filename_10_minus_00"]["values"] == pytest.approx([5.0] * 4)
    assert metrics["effects"]["R_ordering_01_minus_00"]["values"] == pytest.approx([6.0] * 4)
    assert metrics["effects"]["Lambda_filename_to_ordering_10_minus_00"]["values"] == pytest.approx([1.0] * 4)
    assert metrics["effects"]["Lambda_ordering_to_filename_01_minus_00"]["values"] == pytest.approx([1.0] * 4)
    assert metrics["effects"]["I_filename"]["values"] == pytest.approx([-1.0] * 4)
    assert metrics["effects"]["I_ordering"]["values"] == pytest.approx([-1.0] * 4)
    assert metrics["effects"]["R_filename_10_minus_00"]["n_worlds"] == 4
    assert metrics["per_field"]["filename"]["comprehension"]["00"]["generated_accuracy_by_mapping"] == {"1": 1.0, "2": 1.0}
    assert metrics["gates"]["off_diagonal_interpretable"] is True


def test_off_diagonal_not_identified_when_diagonal_fails():
    rows = _synthetic_results(_examples(2))
    # Force the filename intended effect to zero in each matched condition.
    for row in rows:
        md = row["metadata"]
        if md["field"] == "filename" and md["task_type"] == "scope_value_choice":
            row["candidate_logprobs"] = {"A": 0.0, "B": 0.0}
    metrics = analyze_two_scope_pilot(rows)
    assert metrics["gates"]["R_filename_positive_ci"] is False
    assert metrics["gates"]["off_diagonal_interpretable"] is False
    assert "not identified" in metrics["off_diagonal_interpretation"]
