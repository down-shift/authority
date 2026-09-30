#!/usr/bin/env python3
"""Run or dataset-audit the fixed-graph provider-resolution v2 experiment."""
from __future__ import annotations
import argparse,hashlib,json,logging,re,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import yaml
from authority_leakage.inference import write_jsonl,write_run_status
from authority_leakage.progress import tqdm
from authority_leakage.models.hf import HFAdapter
from authority_leakage.models.continuation import continuation_encoding
from authority_leakage.schemas import Message
from indirection_v2.config import generate
from indirection_v2.scoring import audit_tokenization,score_example
from indirection_v2.analysis import analyze
from indirection_v2.plots import plot

class TokenizerRenderer:
    def __init__(self,tokenizer,enable_thinking=None):
        self.tokenizer=tokenizer;self.enable_thinking=enable_thinking
    def render(self,messages):
        kwargs={}
        if self.enable_thinking is not None: kwargs["enable_thinking"]=self.enable_thinking
        return self.tokenizer.apply_chat_template(
            [{"role":m.role,"content":m.content} for m in messages],
            tokenize=False,add_generation_prompt=True,**kwargs)

def report_text(config,sha,audit_report,token_audit,metrics=None):
    out=["# Instruction Indirection Gap v2", "",
      f"Dataset SHA256: {sha}; worlds: {audit_report['worlds']}; graph links K={audit_report['K']}.",
      "This is the controlled provider-resolution experiment. V1 remains exploratory and is not pooled with v2.",
      "",
      "## Frozen design","",
      f"Each world has exactly {audit_report['K']} resolution links at relevant path depths 1–5. All relevant and distractor links use the same resolves-to grammar. The resolved terminal provider is balanced by task family. Primary rows include separate comprehension and application questions. At depths 3–5, application also has matched path, neutral, bridge, and flattened conditions.",
      "",
      "The neutral and bridge lines are inserted into the same deep prompt at the same position. Flattened prompts replace the relevant path with one direct edge and retain K total links. Statements are independently shuffled by world and depth; relevant edge positions and span are saved.",
      "",
      "## Graph and counterbalance audit","",
      f"{json.dumps(audit_report,sort_keys=True)}",""]
    if token_audit:
        compact={k:v for k,v in token_audit.items() if "counts_by_world" not in k and "examples" not in k}
        out += ["## Tokenization audit","",json.dumps(compact,sort_keys=True),""]
    if metrics is None:
        out += ["## Inference status","", "No model predictions were collected in this dataset/tokenizer audit run.",""]
        return "\n".join(out)
    out += ["## Path-depth results","",
      "| Family | Task | Depth | Worlds | Accuracy (95% world CI) | Mean margin (95% world CI) | Prompt tokens |",
      "|---|---|---:|---:|---:|---:|---:|"]
    for key,v in sorted(metrics["summaries"].items()):
        # Prefer named dimensions; tolerate older metrics whose keys only
        # encode the stratum as a pipe-delimited string.
        stratum=v.get("stratum")
        if stratum:
            fam=stratum["task_family"];task=stratum["measurement_type"]
            condition=stratum["condition_type"];depth=stratum["indirection_depth"]
        else:
            parts=key.split("|")
            if len(parts)<7:
                raise ValueError(f"Unexpected summary stratum key: {key!r}")
            fam=parts[0];task=parts[-4];condition=parts[-3];depth=parts[-2]
        if condition=="path":
            out.append(f"| {fam} | {task} | {depth} | {v['n_worlds']} | {v['accuracy']:.3f} ({v['accuracy_ci95_world']}) | {v['mean_margin']:.3f} ({v['margin_ci95_world']}) | {v['mean_prompt_tokens']:.1f} |")
    out += ["","## Paired depth and control contrasts","",
      "Contrasts are paired by world and reported separately by task family. Bridge vs neutral is the primary bridge-specific estimate.",""]
    for key,v in sorted(metrics["path_depth_vs_depth1"].items()):
        out.append(f"- Depth contrast {key}: margin difference {v['mean_margin_difference']:.3f}, 95% CI {v['margin_ci95_world']}; accuracy difference {v['mean_accuracy_difference']:.3f}, CI {v['accuracy_ci95_world']}.")
    for key,v in sorted(metrics["application_control_contrasts"].items()):
        out.append(f"- Control contrast {key}: margin difference {v['mean_margin_difference']:.3f}, 95% CI {v['margin_ci95_world']}; accuracy difference {v['mean_accuracy_difference']:.3f}, CI {v['accuracy_ci95_world']}.")
    out += ["","## Comprehension/application dissociation",""]
    for key,v in sorted(metrics["comprehension_application_dissociation"].items()):
        out.append(f"- {key}: application wrong given comprehension correct = {v['comprehension_correct_application_wrong']}/{v['n_comprehension_correct']} ({v['p_application_wrong_given_comprehension_correct']}). This is a paired world-depth pattern across separate prompts, not evidence of sequential internal stages.")
    out += ["","Bootstrap resampling uses world as the unit. Results are behavioral and do not identify an internal mechanism.",""]
    return "\n".join(out)

def recover_report(run_dir):
    """Rebuild the final report/status from artifacts after a late report failure."""
    run=Path(run_dir)
    required=("config.yaml","dataset.jsonl","prompt_audit.json",
              "tokenization_audit.json","metrics.json")
    missing=[name for name in required if not (run/name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot rebuild report; missing run artifacts: {', '.join(missing)}")
    config=yaml.safe_load((run/"config.yaml").read_text())
    audit_report=json.loads((run/"prompt_audit.json").read_text())
    token_audit=json.loads((run/"tokenization_audit.json").read_text())
    metrics=json.loads((run/"metrics.json").read_text())
    sha=hashlib.sha256((run/"dataset.jsonl").read_bytes()).hexdigest()
    (run/"pilot_report.md").write_text(report_text(config,sha,audit_report,token_audit,metrics))
    total=int(metrics.get("n_rows",0))
    write_run_status(run/"run_status.json","complete",total,total)
    return sha,total

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("configs/indirection_v2.yaml"))
    p.add_argument("--model",default=None);p.add_argument("--models-config",type=Path,default=Path("configs/models.yaml"))
    p.add_argument("--output-root",type=Path,default=Path("outputs"));p.add_argument("--device",default="auto")
    p.add_argument("--dataset-only",action="store_true");p.add_argument("--tokenizer-audit-only",action="store_true")
    p.add_argument("--report-only",action="store_true",help="Rebuild the report/status from an existing completed analysis")
    p.add_argument("--run-dir",type=Path,default=None,help="Existing run directory for --report-only")
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",datefmt="%Y-%m-%dT%H:%M:%S")
    logger=logging.getLogger("indirection_v2")
    if a.report_only:
        if a.run_dir is None: p.error("--report-only requires --run-dir")
        run=a.run_dir
        fh=logging.FileHandler(run/"run.log",encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"));logger.addHandler(fh)
        sha,total=recover_report(run)
        logger.info("Recovered report from existing artifacts run=%s sha256=%s rows=%d; no inference rerun",run,sha,total)
        print(run);return
    cfg=yaml.safe_load(a.config.read_text());model=a.model or cfg["model"]
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");slug=re.sub(r"[^A-Za-z0-9_-]+","_",model)
    run=a.output_root/f"{stamp}_indirection_v2_{slug}_s{cfg['seed']}";run.mkdir(parents=True,exist_ok=False)
    fh=logging.FileHandler(run/"run.log",encoding="utf-8");fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"));logger.addHandler(fh)
    logger.info("Starting v2 config=%s model=%s seed=%s worlds_per_family=%d K=%d dataset_only=%s tokenizer_audit_only=%s",
                a.config,model,cfg["seed"],cfg["worlds_per_family"],cfg["total_relation_edges"],a.dataset_only,a.tokenizer_audit_only)
    (run/"config.yaml").write_text(yaml.safe_dump({**cfg,"model":model},sort_keys=True))
    total=int(cfg["worlds_per_family"])*3*19
    write_run_status(run/"run_status.json","generating_dataset",0,total)
    started=time.monotonic()
    try: rows,audit_report=generate(cfg)
    except BaseException as exc:
        write_run_status(run/"run_status.json","dataset_generation_failed",0,total,repr(exc));logger.exception("Dataset generation/audit failed");raise
    logger.info("Dataset generated rows=%d worlds=%d graph_audit_passed=%s elapsed=%.2fs",len(rows),audit_report["worlds"],audit_report["passed"],time.monotonic()-started)
    write_jsonl(run/"dataset.jsonl",tqdm(rows,total=len(rows),desc="Writing v2 dataset",unit="prompt",leave=False))
    sha=hashlib.sha256((run/"dataset.jsonl").read_bytes()).hexdigest()
    (run/"prompt_audit.json").write_text(json.dumps(audit_report,indent=2,sort_keys=True)+"\n")
    exemplar=next(w for w in rows if w["world_id"]=="filename-0000")
    sample=["# Representative v2 graph audit","",
      "One filename world, with its resolved provider and proposed values fixed across depths. Relevant and distractor edges use the same relation grammar.",""]
    for d in range(1,6):
        row=next(r for r in rows if r["world_id"]==exemplar["world_id"] and r["target_depth"]==d and r["condition_type"]=="path" and r["measurement_type"]=="application")
        sample += [f"## Path depth {d}: {row['graph']['total_edges']} total edges; relevant positions {row['graph']['relevant_edge_positions']}; span {row['graph']['relevant_edge_span']}",
                   "",row["prompt"],""]
    for cond in ("neutral","bridge","flattened"):
        row=next(r for r in rows if r["world_id"]==exemplar["world_id"] and r["target_depth"]==4 and r["condition_type"]==cond)
        sample += [f"## Depth 4 {cond}","","".join(row["prompt"]),""]
    (run/"representative_prompts.md").write_text("\n".join(sample))
    write_run_status(run/"run_status.json","dataset_ready",0,total)
    if a.dataset_only:
        (run/"pilot_report.md").write_text(report_text(cfg,sha,audit_report,None))
        logger.info("Dataset-only audit complete sha256=%s; inference was not started",sha);print(run);return

    registry=yaml.safe_load(a.models_config.read_text()).get("models",{});spec=registry.get(model,{"name":model})
    # Check token-count controls before loading model weights.
    write_run_status(run/"run_status.json","loading_tokenizer",0,total)
    logger.info("Loading tokenizer for pre-inference length audit: %s revision=%s",spec["name"],spec.get("revision"))
    try:
        import transformers
        tokenizer=transformers.AutoTokenizer.from_pretrained(spec["name"],revision=spec.get("revision"),trust_remote_code=False)
        if tokenizer.chat_template is None: raise ValueError("Model tokenizer has no chat template")
        renderer=TokenizerRenderer(tokenizer,spec.get("enable_thinking"))
        token_rows,token_audit=audit_tokenization(rows,renderer)
        write_jsonl(run/"tokenization_audit.jsonl",token_rows)
        (run/"tokenization_audit.json").write_text(json.dumps(token_audit,indent=2,sort_keys=True)+"\n")
        tokenizer_meta={"tokenizer_name":tokenizer.name_or_path,"requested_revision":spec.get("revision"),
            "tokenizer_commit":tokenizer.init_kwargs.get("_commit_hash"),
            "chat_template_sha256":hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()}
        (run/"tokenizer_metadata.json").write_text(json.dumps(tokenizer_meta,indent=2,sort_keys=True)+"\n")
        logger.info("Tokenizer audit candidates=%d unequal_pairs=%d max_path_prompt_spread=%d bridge_neutral_mismatch=%d flat_path_mismatch=%d",
            len(token_rows),token_audit["unequal_candidate_token_pairs"],token_audit["path_prompt_token_spread_max"],
            token_audit["bridge_neutral_token_mismatches"],token_audit["flattened_path_token_mismatches"])
        if token_audit["unequal_candidate_token_pairs"] or token_audit["path_prompt_token_spread_over_tolerance"] or token_audit["bridge_neutral_token_mismatches"] or token_audit["flattened_path_token_mismatches"]:
            raise ValueError("Token-balance audit exceeded frozen tolerances; see tokenization_audit.json")
    except BaseException as exc:
        write_run_status(run/"run_status.json","tokenization_audit_failed",0,total,repr(exc))
        logger.exception("Pre-inference token audit failed; model weights were not loaded");raise
    if a.tokenizer_audit_only:
        (run/"pilot_report.md").write_text(report_text(cfg,sha,audit_report,token_audit))
        write_run_status(run/"run_status.json","tokenization_audit_complete",0,total)
        logger.info("Tokenizer-only audit complete; model weights were not loaded")
        print(run);return

    write_run_status(run/"run_status.json","loading_model",0,total)
    logger.info("Loading model checkpoint=%s requested_revision=%s device=%s",spec["name"],spec.get("revision"),a.device)
    try:
        adapter=HFAdapter(spec["name"],spec.get("revision"),a.device,
            enable_thinking=spec.get("enable_thinking"),quantization=spec.get("quantization"),
            attention_implementation=spec.get("attention_implementation"))
    except BaseException as exc:
        write_run_status(run/"run_status.json","model_load_failed",0,total,repr(exc));logger.exception("Model loading failed");raise
    provenance=adapter.provenance()
    metadata={"git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
        "model":provenance,"dataset_sha256":sha,"seed":cfg["seed"],
        "candidate_scoring_mode":"joint_offsets_with_boundary_overlap_audit",
        "decoding":{"do_sample":False,"candidate_scoring":"sum log probability primary; mean token log probability diagnostic"}}
    (run/"metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    logger.info("Model ready device=%s revision=%s transformers=%s torch=%s",provenance.get("device"),
                provenance.get("model_commit"),provenance.get("transformers_version"),provenance.get("torch_version"))
    scored=[];write_run_status(run/"run_status.json","running",0,total)
    def prediction_rows():
        for i,row in enumerate(tqdm(rows,total=total,desc="Scoring v2 candidates",unit="prompt"),1):
            result=score_example({**row,"model":model},adapter,model);scored.append(result)
            if i%20==0 or i==total:
                write_run_status(run/"run_status.json","running",i,total);logger.info("Scored %d/%d prompts",i,total)
            yield result
    try: write_jsonl(run/"predictions.jsonl",prediction_rows())
    except BaseException as exc:
        write_run_status(run/"run_status.json","interrupted" if isinstance(exc,KeyboardInterrupt) else "scoring_failed",len(scored),total,repr(exc))
        logger.exception("Scoring failed after %d/%d prompts",len(scored),total);raise
    write_jsonl(run/"per_world_metrics.jsonl",scored)
    try: metrics=analyze(scored,int(cfg["seed"]),int(cfg["bootstrap_replicates"]))
    except BaseException as exc:
        write_run_status(run/"run_status.json","analysis_failed",len(scored),total,repr(exc));logger.exception("Analysis failed");raise
    (run/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    write_jsonl(run/"world_outcomes.jsonl",metrics["comprehension_application_by_world"])
    plot(scored,metrics,run/"figures")
    (run/"pilot_report.md").write_text(report_text(cfg,sha,audit_report,token_audit,metrics))
    write_run_status(run/"run_status.json","complete",total,total)
    logger.info("v2 run complete output=%s rows=%d",run,total);print(run)

if __name__=="__main__":main()
