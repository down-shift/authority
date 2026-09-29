import json

from authority_leakage.clean import SCHEMA, dataset_sha256, generate_clean, validate_matching
from authority_leakage.scoring import parse_canonical_json, grade
from authority_leakage.analysis import analyze, pilot_report
from authority_leakage.plots import make_figures


def test_matched_authority_worlds_are_deterministic_and_only_grant_changes():
    cfg = {"worlds": 8, "templates": ["direct", "plain", "question"]}
    a = generate_clean("epistemic", cfg, 19)
    assert a == generate_clean("epistemic", cfg, 19)
    assert dataset_sha256(a) == dataset_sha256(generate_clean("epistemic", cfg, 19))
    validate_matching(a)
    assert len({x.pair_id for x in a}) == 8
    assert all(len([x for x in a if x.pair_id == wid]) == 6 for wid in {x.pair_id for x in a})
    assert all(x.correct_answer == next(y.correct_answer for y in a if y.pair_id == x.pair_id) for x in a)


def test_scope_schema_and_paired_baseline_subtraction():
    rows = generate_clean("scope", {"worlds": 2, "scopes": ["output_format", "ordering"]}, 4)
    good = {"output_format": "xml", "ordering": "descending", "filename": "wrong.txt", "tool_choice": "shell", "numeric_answer": 17}
    raw = json.dumps(good)
    assert parse_canonical_json(raw) == good
    assert parse_canonical_json(raw + " prose") is None
    assert parse_canonical_json('{"output_format":"xml","ordering":"ascending","filename":"x","tool_choice":"shell","numeric_answer":true}') is None
    baseline = next(x for x in rows if x.condition == "NO_AUTHORITY")
    out = grade(baseline, json.dumps({"output_format":"json","ordering":"ascending","filename":"result.txt","tool_choice":"calculator","numeric_answer":42}))
    assert out["parse_success"] and out["source_adoption"]["ordering"] is False
    assert set(SCHEMA) == set(out["source_adoption"])


def test_claim_truth_and_answer_position_balanced():
    rows = generate_clean("epistemic", {"worlds": 40}, 22)
    worlds = {r.pair_id: r for r in rows if r.condition == "NO_AUTHORITY"}
    assert sum(bool(r.metadata["claim_truth"]) for r in worlds.values()) == 20
    assert sum(r.metadata["labels"].index(r.metadata["claim_label"]) == 0 for r in worlds.values()) == 20


def test_scope_leakage_is_paired_baseline_subtraction_and_plots(tmp_path):
    examples = generate_clean("scope", {"worlds": 4, "scopes": ["output_format", "ordering"]}, 5)
    rows=[]
    for e in examples:
        scope=e.metadata.get("authority_scope")
        # all source proposals adopted only when a scope has an explicit grant
        answers={k:v["correct_value"] for k,v in e.metadata["dimensions"].items()}
        if e.condition == "AUTHORITY_I": answers[scope]=e.metadata["dimensions"][scope]["source_value"]
        if e.condition == "FULL_AUTHORITY": answers={k:v["source_value"] for k,v in e.metadata["dimensions"].items()}
        response=json.dumps(answers)
        rows.append({**e.to_dict(),"raw_response":response,"outcome":grade(e,response)})
    metrics=analyze(rows)
    assert metrics["leakage_matrix"]["output_format->output_format"]["mean"] == 1
    assert metrics["leakage_matrix"]["output_format->ordering"]["mean"] == 0
    assert metrics["no_authority_source_adoption"]["ordering"] == 0
    assert "Raw adoption" in pilot_report(metrics)
    make_figures(metrics,tmp_path,"synthetic")
    assert (tmp_path/"leakage_matrix_heatmap.png").exists()
