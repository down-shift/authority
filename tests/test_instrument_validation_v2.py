from authority_leakage.instrument_validation_v2 import (
    analyze_validation_v2, dataset_sha256, generate_validation_v2, validate_v2_matching,
)
from authority_leakage.scoring import grade


def _examples(worlds=4, seed=27):
    return generate_validation_v2({"worlds": worlds}, seed)


def test_each_world_condition_and_task_has_both_candidate_mappings():
    examples = _examples()
    assert len(examples) == 4 * 2 * 2 * 2
    assert len(validate_v2_matching(examples)) == 4 * 2 * 2
    for world in {e.pair_id for e in examples}:
        rows = [e for e in examples if e.pair_id == world]
        assert {(e.metadata["authority"], e.metadata["mapping_id"], e.metadata["task_type"])
                for e in rows} == {(a, m, t) for a in ("NO", "YES") for m in (1, 2)
                                   for t in ("filename_choice", "authority_comprehension")}


def test_mapping_reversal_changes_only_candidate_label_assignment():
    examples = _examples(1)
    for authority in ("NO", "YES"):
        for task in ("filename_choice", "authority_comprehension"):
            pair = [e for e in examples if e.metadata["authority"] == authority and e.metadata["task_type"] == task]
            one, two = sorted(pair, key=lambda e: e.metadata["mapping_id"])
            assert one.metadata["world_id"] == two.metadata["world_id"]
            for key in ("default_filename", "source_filename", "owner", "expected_final_filename"):
                assert one.metadata[key] == two.metadata[key]
            assert one.metadata["label_map"]["A"] == two.metadata["label_map"]["B"]
            assert one.metadata["label_map"]["B"] == two.metadata["label_map"]["A"]
            assert _normalize(one.messages[0].content) == _normalize(two.messages[0].content)


def _normalize(prompt):
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("A: "):
            lines[i:i + 2] = ["A: <A>", "B: <B>"]
            break
    return "\n".join(lines)


def test_frozen_c_authority_wording_is_unchanged_and_semantics_are_correct():
    examples = _examples(1)
    no = next(e for e in examples if e.metadata["authority"] == "NO" and e.metadata["mapping_id"] == 1 and e.metadata["task_type"] == "filename_choice")
    yes = next(e for e in examples if e.metadata["authority"] == "YES" and e.metadata["mapping_id"] == 1 and e.metadata["task_type"] == "filename_choice")
    assert "FINAL DECISION OWNER\nfilename: Default policy\n\nThe final decision owner determines the final filename." in no.messages[0].content
    assert "FINAL DECISION OWNER\nfilename: Source S\n\nThe final decision owner determines the final filename." in yes.messages[0].content
    assert no.metadata["expected_final_filename"] == no.metadata["default_filename"]
    assert yes.metadata["expected_final_filename"] == yes.metadata["source_filename"]


def test_source_oriented_mapping_margins_cancel_synthetic_a_label_bias():
    examples = _examples(worlds=1)
    rows = []
    for e in examples:
        if e.metadata["task_type"] != "filename_choice":
            continue
        # semantic preference=2, A-label bias=3: m1=5; m2=-1.
        scores = {"A": 5.0, "B": 0.0} if e.metadata["mapping_id"] == 1 else {"A": 1.0, "B": 0.0}
        # On mapping 2 source is B, so oriented m2 = B - A = -1; instead set B=2, A=3 -> -1.
        if e.metadata["mapping_id"] == 2:
            scores = {"A": 3.0, "B": 2.0}
        chosen = max(scores, key=scores.get)
        rows.append({**e.to_dict(), "candidate_logprobs": scores,
                     "outcome": grade(e, chosen, scores)})
    metrics = analyze_validation_v2(rows)
    no = metrics["condition_results"]["NO"]
    assert no["values"] == [2.0]
    assert metrics["label_bias"]["NO"]["values"] == [3.0]
    assert metrics["m1_m2"]["NO"] == {"m1": [5.0], "m2": [-1.0]}


def test_world_is_the_bootstrap_unit_and_responsiveness_pairs_by_world():
    examples = _examples(worlds=3)
    rows = []
    for e in examples:
        task, auth, mapping = e.metadata["task_type"], e.metadata["authority"], e.metadata["mapping_id"]
        if task != "filename_choice":
            continue
        # NO semantic margin=-2, YES=+4; repeated A/B versions encode same semantic margin.
        semantic = -2.0 if auth == "NO" else 4.0
        bias = 1.5
        m = semantic + bias if mapping == 1 else semantic - bias
        scores = {"A": m, "B": 0.0} if mapping == 1 else {"B": m, "A": 0.0}
        chosen = max(scores, key=scores.get)
        rows.append({**e.to_dict(), "candidate_logprobs": scores,
                     "outcome": grade(e, chosen, scores)})
    metrics = analyze_validation_v2(rows, seed=4)
    assert metrics["condition_results"]["NO"]["n_worlds"] == 3
    assert metrics["intended_responsiveness_margin"]["n_worlds"] == 3
    assert metrics["intended_responsiveness_margin"]["values"] == [6.0, 6.0, 6.0]


def test_seed_reproducibility_and_hash_stability():
    assert _examples(5, 88) == _examples(5, 88)
    assert dataset_sha256(_examples(5, 88)) == dataset_sha256(_examples(5, 88))


def test_comprehension_is_semantic_and_condition_specific():
    examples = _examples(2)
    for e in examples:
        if e.metadata["task_type"] != "authority_comprehension":
            continue
        prompt = e.messages[0].content
        assert "Who determines the final filename?" in prompt
        assert "YES" not in prompt and "NO" not in prompt
        assert e.correct_answer == ("A" if e.metadata["mapping_id"] == (1 if e.metadata["authority"] == "YES" else 2) else "B")
