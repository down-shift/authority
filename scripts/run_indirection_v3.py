#!/usr/bin/env python3
"""Generate, audit, and run the v3 factorial experiment."""
from __future__ import annotations
import argparse,hashlib,json,logging,re,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import yaml
from authority_leakage.inference import write_jsonl,write_run_status
from authority_leakage.models.hf import HFAdapter
from authority_leakage.schemas import Message
from indirection_v3.design import generate,render,generate_worlds,TERMINALS
from indirection_v3.audit import pad_to_world_task,audit_tokens
from indirection_v3.analysis import analyze
from indirection_v3.plots import plot
from indirection.scoring import score_example

class Renderer:
 def __init__(self,t,thinking=None):self.t=t;self.thinking=thinking
 def render(self,messages):
  kw={} if self.thinking is None else {"enable_thinking":self.thinking}
  return self.t.apply_chat_template([{"role":m.role,"content":m.content} for m in messages],tokenize=False,add_generation_prompt=True,**kw)

def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def report(audit,token_audit=None,metrics=None,stage="dataset"):
 lines=["# Instruction Indirection Gap v3 factorial","",f"Stage: {stage}.",
  f"Graph audit passed: {audit['passed']}; worlds: {audit['worlds']}; primary rows: {audit['primary_rows']}; total rows: {audit['total_rows']}.",
  f"Provider balance exact: {audit['provider_balance_exact']}; terminal-provider counts: `{json.dumps(audit['terminal_balance'],sort_keys=True)}`.",
  "","Primary design: depth 1–5 × 0/2/4/6 disconnected edges × compact/dispersed layout × three families × comprehension/application.",
  "Relevant and disconnected facts use one provider-link grammar. DIRECT is a separate action-selection reference. Bridge, neutral, and flattened controls target depths 3–5 at k=4 and dispersed layout.",""]
 if token_audit:
  lines += [f"Tokenizer audit passed: {token_audit['passed']}; exact primary prompt equality within world/task: {token_audit['token_counts_equal_within_world_task']}; deep-control token mismatches: {token_audit['deep_control_token_mismatches']}; candidate-length mismatch rows: {token_audit['candidate_length_mismatch_rows']}.",
   f"Prompt-token correlations: `{json.dumps(token_audit['primary_correlations'],sort_keys=True)}`; span summaries: `{json.dumps(token_audit['span_by_layout'],sort_keys=True)}`.",""]
 if metrics is None:
  lines += ["No factorial model predictions are present; questions about depth, distractors, layout, interactions, controls, and DIRECT accuracy remain unanswered.",
   "The single-model pilot must complete before deciding whether to freeze a multi-model benchmark.",""]
 else:
  lines += [f"Primary rows scored: {metrics['n_primary_rows']}; independent unit: {metrics['independent_unit']}.",
   "Cell summaries and paired estimates are in `metrics.json`; the figure source data are in `figures/plot_data.csv`.",""]
 return "\n".join(lines)

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("configs/indirection_v3_pilot.yaml"));p.add_argument("--model");p.add_argument("--models-config",type=Path,default=Path("configs/models.yaml"));p.add_argument("--output-root",type=Path,default=Path("outputs"));p.add_argument("--device",default="auto");p.add_argument("--dataset-only",action="store_true");p.add_argument("--tokenizer-audit-only",action="store_true");p.add_argument("--smoke",action="store_true");a=p.parse_args()
 cfg=yaml.safe_load(a.config.read_text());model=a.model or cfg["model"];stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");run=a.output_root/f"{stamp}_indirection_v3_factorial_{re.sub(r'[^A-Za-z0-9_-]+','_',model)}_s{cfg['seed']}";run.mkdir(parents=True,exist_ok=False)
 logging.basicConfig(filename=run/"run.log",level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s");log=logging.getLogger("indirection_v3")
 (run/"config.yaml").write_text(yaml.safe_dump({**cfg,"model":model},sort_keys=True))
 override=None
 if a.smoke:
  pool=generate_worlds(2,int(cfg["seed"]));override=[]
  for fam in ("filename","ordering"):
   vals=[w for w in pool if w.task_family==fam];override.extend([next(w for w in vals if w.resolved_provider==TERMINALS[0]),next(w for w in vals if w.resolved_provider==TERMINALS[1])])
  override.append(next(w for w in pool if w.task_family=="destination"))
 rows,graph_audit=generate(cfg,override);write_jsonl(run/"dataset.jsonl",rows);write_json(run/"graph_audit.json",graph_audit)
 sha=hashlib.sha256((run/"dataset.jsonl").read_bytes()).hexdigest();write_json(run/"dataset_metadata.json",{"sha256":sha,"rows":len(rows),"seed":cfg["seed"]})
 (run/"representative_prompts.md").write_text("\n\n".join(f"## {r['factorial_cell_id']} {r['layout']}\n\n{r['prompt']}\n\nGraph metrics: {json.dumps(r['graph'],sort_keys=True)}" for r in rows if r["world_id"]==rows[0]["world_id"] and r["task"]=="application" and r["condition"]=="BASE" and (r["depth"],r["distractor_count"]) in ((1,0),(3,4),(5,6))))
 primary=[r for r in rows if r["condition"]=="BASE"];expected=len(primary);print(f"primary rows={expected}; total rows={len(rows)}; candidate scoring calls={2*len(rows)}")
 if a.dataset_only:
  (run/"pilot_report.md").write_text(report(graph_audit,stage="dataset-only"));write_json(run/"run_status.json",{"status":"dataset_complete","rows":len(rows)});print(run);return
 registry=yaml.safe_load(a.models_config.read_text()).get("models",{});spec=registry.get(model,{"name":model})
 import transformers
 tok=transformers.AutoTokenizer.from_pretrained(spec["name"],revision=spec.get("revision"),trust_remote_code=False)
 renderer=Renderer(tok,spec.get("enable_thinking"));token_audit=pad_to_world_task(rows,renderer,tok)
 write_json(run/"tokenization_audit.json",token_audit)
 if not token_audit["passed"]:raise ValueError("Factorial token-count independence audit failed; inference not started")
 write_jsonl(run/"dataset.jsonl",rows)
 dataset_sha=hashlib.sha256((run/"dataset.jsonl").read_bytes()).hexdigest();write_json(run/"dataset_metadata.json",{"sha256":dataset_sha,"rows":len(rows),"seed":cfg["seed"]})
 if a.tokenizer_audit_only:
  (run/"pilot_report.md").write_text(report(graph_audit,token_audit,stage="full-pilot tokenizer audit only"));write_json(run/"run_status.json",{"status":"tokenizer_audit_complete","rows":len(rows)});print(run);return
 if a.smoke:
  (run/"pilot_report.md").write_text(report(graph_audit,token_audit,stage="five-world tokenizer smoke"));write_json(run/"run_status.json",{"status":"smoke_tokenizer_audit_complete","rows":len(rows)});print(run);return
 adapter=HFAdapter(spec["name"],spec.get("revision"),a.device,enable_thinking=spec.get("enable_thinking"),quantization=spec.get("quantization"),attention_implementation=spec.get("attention_implementation"))
 predictions=[];t0=time.monotonic()
 for i,row in enumerate(rows,1):
  result=score_example(row,adapter,model);predictions.append(result)
  if i%100==0:log.info("scored %d/%d elapsed=%.1fs",i,len(rows),time.monotonic()-t0)
 write_jsonl(run/"predictions.jsonl",predictions);metrics=analyze(predictions,cfg["seed"],cfg.get("bootstrap_replicates",2000));write_json(run/"metrics.json",metrics);plot(predictions,run/"figures")
 (run/"pilot_report.md").write_text(report(graph_audit,token_audit,metrics,stage="Qwen3-4B pilot"))
 write_json(run/"run_status.json",{"status":"complete","rows":len(rows),"elapsed_seconds":time.monotonic()-t0});print(run)
if __name__=="__main__":main()
