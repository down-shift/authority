import json
import subprocess
import sys
from pathlib import Path

from authority_leakage.analysis import analyze, pilot_report, combined_pilot_report
from authority_leakage.generation.delegation import generate_delegation
from authority_leakage.generation.epistemic import generate_epistemic
from authority_leakage.inference import read_jsonl, write_jsonl
from authority_leakage.plots import make_figures, plot_combined_summary, plot_model_comparison
from authority_leakage.scoring import grade


def _delegation_response(answer):
    fields = {"ordering": "3,2,1" if answer["ordering"] == "descending" else "1,2,3",
              "numeric_answer": answer["numeric_answer"], "filename": answer["filename"]}
    if answer["format"] == "JSON":
        return json.dumps(fields)
    return "<result>" + "".join(f"<{k}>{v}</{k}>" for k, v in fields.items()) + "</result>"


def test_epistemic_saved_prediction_analysis_and_figures(tmp_path):
    examples = generate_epistemic({"templates": ["reports", "readings"], "evidence_strengths": [1],
                                   "trust_metadata": ["ordinary", "untrusted"], "worlds_per_cell": 2}, 3)
    rows = []
    for e in examples:
        claim = e.metadata["claim_label"]
        other = next(x for x in e.metadata["labels"] if x != claim)
        logprobs = {claim: -1. if e.condition == "claim_system" else -2., other: -2.}
        response = e.correct_answer
        rows.append({**e.to_dict(), "raw_response": response, "candidate_logprobs": logprobs,
                     "rendered_prompt": "\n".join(m.content for m in e.messages),
                     "outcome": grade(e, response, logprobs)})
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, rows)
    metrics = analyze(read_jsonl(path))
    assert metrics["paired"]["n"] == 16
    assert metrics["paired"]["mean"] == 1.
    assert metrics["controls"]["claim_absent"]["accuracy"]["rate"] == 1.
    assert "untrusted" in pilot_report(metrics)
    (tmp_path / "metadata.json").write_text(json.dumps({"model": {"model_name": "fake"}}))
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_results.py"
    subprocess.run([sys.executable, str(script), str(tmp_path)], check=True, capture_output=True, text=True)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "bootstrap_ci.csv").exists()
    assert (tmp_path / "pilot_report.txt").exists()
    assert (tmp_path / "figures" / "epistemic_role_scatter.png").exists()


def test_delegation_saved_prediction_analysis_and_figures(tmp_path):
    examples = generate_delegation({"templates": ["direct"], "repeats_per_pair": 1}, 3)
    rows = []
    for e in examples:
        response = _delegation_response(e.correct_answer)
        rows.append({**e.to_dict(), "raw_response": response, "outcome": grade(e, response)})
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, rows)
    metrics = analyze(read_jsonl(path))
    assert metrics["authorized_only_compliance"]["rate"] == 1.
    assert metrics["controls"]["both"]["unauthorized_followed"]["rate"] == 0.
    assert all(value == 1. for value in metrics["selectivity"].values())
    make_figures(metrics, tmp_path / "figures", "fake")
    assert (tmp_path / "figures" / "legacy_raw_unauthorized_follow_matrix.png").exists()


def test_combined_report_and_figure(tmp_path):
    epi = {"paired": {"mean": 0.5, "ci95": [0.2, 0.8], "n": 10},
           "by_trust": {"ordinary": {"mean": 0.6}, "untrusted": {"mean": 0.4}},
           "by_template": {"reports": {"mean": 0.5}}}
    delegation = {"authorized_only_compliance": {"rate": 1.0},
                  "controls": {"both": {"unauthorized_followed": {"rate": 0.2},
                                          "authorized_followed": {"rate": 0.9}}},
                  "by_similarity": {"high": {"rate": 0.3}, "medium": {"rate": 0.2}, "low": {"rate": 0.1}}}
    assert "Role-swap belief shift" in combined_pilot_report(epi, delegation)
    path = tmp_path / "unified_summary.png"
    plot_combined_summary(epi, delegation, path)
    assert path.exists()
    comparison = tmp_path / "model_comparison.png"
    plot_model_comparison([{**epi, "experiment": "epistemic", "model_name": "model-a"},
                           {**epi, "experiment": "epistemic", "model_name": "model-b"}], comparison)
    assert comparison.exists()
