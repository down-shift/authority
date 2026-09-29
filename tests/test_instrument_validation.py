from authority_leakage.instrument_validation import (
    INSTRUMENTS, analyze_instrument_validation, dataset_sha256, generate_instrument_validation,
    select_instrument, validate_instrument_matching,
)
from authority_leakage.scoring import grade


def _examples(worlds=60, seed=91):
    return generate_instrument_validation({"worlds": worlds}, seed)


def test_all_encodings_have_matched_no_yes_pairs_and_randomized_distinct_values():
    examples = _examples(60)
    assert len({e.pair_id for e in examples}) == 60
    assert len(examples) == 60 * 4 * 2 * 2
    assert len(validate_instrument_matching(examples)) == 60 * 4 * 2
    for wid in {e.pair_id for e in examples}:
        rows = [e for e in examples if e.pair_id == wid]
        for instrument in INSTRUMENTS:
            paired = [e for e in rows if e.metadata["instrument"] == instrument]
            no = next(e for e in paired if e.metadata["authority"] == "NO" and e.metadata["task_type"] == "filename_choice")
            yes = next(e for e in paired if e.metadata["authority"] == "YES" and e.metadata["task_type"] == "filename_choice")
            assert no.metadata["default_filename"] == yes.metadata["default_filename"]
            assert no.metadata["source_filename"] == yes.metadata["source_filename"]
            assert no.metadata["default_filename"] != no.metadata["source_filename"]
            assert no.metadata["label_map"] == yes.metadata["label_map"]
            assert no.metadata["expected_final_filename"] == no.metadata["default_filename"]
            assert yes.metadata["expected_final_filename"] == yes.metadata["source_filename"]


def test_seed_reproduces_worlds_and_dataset_hash():
    a = _examples(8, seed=55)
    b = _examples(8, seed=55)
    assert a == b
    assert dataset_sha256(a) == dataset_sha256(b)


def test_candidate_labels_are_balanced_and_shared_within_world():
    examples = _examples(60)
    for instrument in INSTRUMENTS:
        label_counts = {"source": {"A": 0, "B": 0}, "default": {"A": 0, "B": 0}}
        for e in examples:
            if e.metadata["instrument"] != instrument or e.metadata["task_type"] != "filename_choice" or e.metadata["authority"] != "NO":
                continue
            for label, value in e.metadata["label_map"].items():
                target = "source" if value == e.metadata["source_filename"] else "default"
                label_counts[target][label] += 1
        assert label_counts == {"source": {"A": 30, "B": 30}, "default": {"A": 30, "B": 30}}


def test_comprehension_uses_exact_matching_authority_treatment_including_direct_anchor():
    examples = _examples(2)
    for instrument in INSTRUMENTS:
        for authority in ("NO", "YES"):
            decision = next(e for e in examples if e.metadata["instrument"] == instrument and
                            e.metadata["authority"] == authority and e.metadata["task_type"] == "filename_choice")
            check = next(e for e in examples if e.pair_id == decision.pair_id and
                         e.metadata["instrument"] == instrument and e.metadata["authority"] == authority and
                         e.metadata["task_type"] == "authority_comprehension")
            block = {
                "A": "SOURCE S AUTHORITY",
                "B": "PERMISSION GRANTED TO SOURCE S:",
                "C": "FINAL DECISION OWNER",
                "D": "DIRECT CONTROL",
            }[instrument]
            assert block in decision.messages[0].content
            assert block in check.messages[0].content
            # Both task prompts share the same authority plus world facts verbatim.
            decision_lines = decision.messages[0].content.split("\n\nApply the policy above")[0]
            check_lines = check.messages[0].content.split("\n\nDoes Source S determine")[0]
            assert decision_lines == check_lines
            if instrument == "D":
                assert "Use Source S's proposed filename." in check.messages[0].content if authority == "YES" else "Use the default policy's filename." in check.messages[0].content


def test_a_b_c_prompt_diffs_are_limited_to_declared_authority_encoding():
    examples = _examples(3)
    diffs = validate_instrument_matching(examples)
    assert {d["instrument"] for d in diffs} == set(INSTRUMENTS)
    assert all(d["normalized_prompts_equal"] for d in diffs)
    for instrument in ("A", "B", "C"):
        for row in diffs:
            if row["instrument"] == instrument:
                assert row["authority_no_prompt"] != row["authority_yes_prompt"]


def test_handwritten_yes_no_answers_score_expected_final_values():
    examples = _examples(1)
    for e in examples:
        if e.metadata["task_type"] == "authority_comprehension":
            outcome = grade(e, e.correct_answer)
            assert outcome["parse_success"] and outcome["authority_comprehension_correct"] is True
        else:
            outcome = grade(e, e.correct_answer)
            assert outcome["parse_success"] and outcome["accuracy"] is True
            assert outcome["source_adoption"] is (e.metadata["authority"] == "YES")


def test_metrics_keep_conditions_separate_and_pair_responsiveness_by_world():
    examples = _examples(6)
    rows = []
    for e in examples:
        response = e.correct_answer
        rows.append({**e.to_dict(), "raw_response": response, "candidate_logprobs": None,
                     "outcome": grade(e, response)})
    metrics = analyze_instrument_validation(rows)
    assert "leakage_matrix" not in metrics
    assert metrics["instruments"]["A"]["NO"]["comprehension_accuracy"]["rate"] == 1
    assert metrics["instruments"]["A"]["YES"]["comprehension_accuracy"]["rate"] == 1
    assert metrics["instruments"]["A"]["NO"]["source_adoption"]["rate"] == 0
    assert metrics["instruments"]["A"]["YES"]["source_adoption"]["rate"] == 1
    assert metrics["instruments"]["A"]["paired_responsiveness"]["n_worlds"] == 6
    assert metrics["instruments"]["A"]["paired_responsiveness"]["mean"] == 1
    assert metrics["direct_control_anchor"]["adoption_yes"] == 1
    assert metrics["instruments"]["D"]["eligible_for_selection"] is False


def test_selection_uses_predefined_gates_and_never_reads_leakage_results():
    examples = _examples(10)
    rows = [{**e.to_dict(), "outcome": grade(e, e.correct_answer)} for e in examples]
    metrics = analyze_instrument_validation(rows)
    metrics["leakage_matrix"] = {"poison": 0}
    selection = select_instrument(metrics)
    assert selection["selected_instrument"] in {"A", "B", "C"}
    assert selection["uses_leakage_metrics"] is False
    assert selection["direct_control_is_selection_eligible"] is False


def test_no_passing_instrument_stops_and_does_not_select_anchor():
    examples = _examples(4)
    rows = []
    for e in examples:
        response = e.correct_answer
        if e.metadata["instrument"] in {"A", "B", "C"} and e.metadata["task_type"] == "filename_choice":
            if e.metadata["authority"] == "NO":
                response = next(label for label, value in e.metadata["label_map"].items()
                                if value == e.metadata["source_filename"])
            else:
                response = next(label for label, value in e.metadata["label_map"].items()
                                if value == e.metadata["default_filename"])
        rows.append({**e.to_dict(), "outcome": grade(e, response)})
    metrics = analyze_instrument_validation(rows)
    assert metrics["selection"]["selected_instrument"] is None
    assert "do not run a leakage matrix" in metrics["selection"]["decision"]
