import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_indirection_v2 import recover_report,report_text


def test_report_accepts_summary_strata_with_additional_dimensions():
    summary={"filename|template|qwen|application|path|3|3": {
        "n_worlds":2,"accuracy":0.5,"accuracy_ci95_world":[0.2,0.8],
        "mean_margin":0.1,"margin_ci95_world":[-0.5,0.7],"mean_prompt_tokens":100.0}}
    summary["filename|template|qwen|scoring_variant|application|path|4|4"]={
        "n_worlds":2,"accuracy":0.75,"accuracy_ci95_world":[0.4,1.0],
        "mean_margin":0.4,"margin_ci95_world":[-0.1,0.9],"mean_prompt_tokens":101.0}
    metrics={"summaries":summary,"path_depth_vs_depth1":{},
             "application_control_contrasts":{},"comprehension_application_dissociation":{}}
    audit={"worlds":2,"K":6}
    report=report_text({},"abc",audit,{},metrics)
    assert "| filename | application | 3 |" in report
    assert "| filename | application | 4 |" in report


def test_report_recovery_uses_existing_metrics_without_model_inference(tmp_path):
    (tmp_path/"config.yaml").write_text("model: fixture\n")
    (tmp_path/"dataset.jsonl").write_text('{"world_id":"w"}\n')
    (tmp_path/"prompt_audit.json").write_text('{"worlds":1,"K":6}')
    (tmp_path/"tokenization_audit.json").write_text('{}')
    metrics={"n_rows":1,"summaries":{},"path_depth_vs_depth1":{},
             "application_control_contrasts":{},"comprehension_application_dissociation":{}}
    (tmp_path/"metrics.json").write_text(__import__("json").dumps(metrics))
    sha,total=recover_report(tmp_path)
    assert len(sha)==64 and total==1
    assert (tmp_path/"pilot_report.md").is_file()
    assert '"status": "complete"' in (tmp_path/"run_status.json").read_text()
