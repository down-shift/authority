#!/usr/bin/env python3
"""Generate, audit, score, and analyze a frozen instruction-indirection dataset."""
from __future__ import annotations
import argparse, hashlib, json, logging, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import yaml
from authority_leakage.inference import write_jsonl, write_run_status
from authority_leakage.progress import tqdm
from authority_leakage.models.hf import HFAdapter
from authority_leakage.models.continuation import continuation_encoding
from authority_leakage.schemas import Message
from indirection.config import generate
from indirection.scoring import score_example
from indirection.analysis import analyze
from indirection.plots import plot

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/indirection.yaml'));p.add_argument('--model',default=None);p.add_argument('--models-config',type=Path,default=Path('configs/models.yaml'));p.add_argument('--output-root',type=Path,default=Path('outputs'));p.add_argument('--device',default='auto');p.add_argument('--dataset-only',action='store_true');a=p.parse_args()
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%dT%H:%M:%S')
 logger=logging.getLogger('indirection')
 cfg=yaml.safe_load(a.config.read_text()); model=a.model or cfg['model']
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');slug=re.sub(r'[^A-Za-z0-9_-]+','_',model)
 run=a.output_root/f'{stamp}_indirection_{slug}_s{cfg["seed"]}';run.mkdir(parents=True,exist_ok=False)
 file_handler=logging.FileHandler(run/'run.log',encoding='utf-8');file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
 logger.addHandler(file_handler)
 logger.info('Starting indirection run directory=%s config=%s model=%s seed=%s worlds_per_family=%s dataset_only=%s',run,a.config,model,cfg['seed'],cfg['worlds_per_family'],a.dataset_only)
 (run/'config.yaml').write_text(yaml.safe_dump({**cfg,'model':model},sort_keys=True))
 expected_rows=int(cfg['worlds_per_family'])*3*5*2
 write_run_status(run/'run_status.json','generating_dataset',0,expected_rows)
 started=time.monotonic();logger.info('Generating structured worlds and matched prompts')
 try:
  rows,audit_report=generate(cfg)
 except BaseException as exc:
  write_run_status(run/'run_status.json','dataset_generation_failed',0,expected_rows,repr(exc))
  logger.exception('Dataset generation or semantic audit failed');raise
 logger.info('Dataset generation complete worlds=%d prompts=%d audit_passed=%s elapsed_seconds=%.2f',audit_report['worlds'],len(rows),audit_report['passed'],time.monotonic()-started)
 logger.info('Writing frozen dataset')
 write_jsonl(run/'dataset.jsonl',tqdm(rows,total=len(rows),desc='Writing dataset',unit='prompt',leave=False))
 dsha=hashlib.sha256((run/'dataset.jsonl').read_bytes()).hexdigest()
 logger.info('Frozen dataset written sha256=%s',dsha)
 (run/'prompt_audit.json').write_text(json.dumps(audit_report,indent=2)+'\n')
 write_run_status(run/'run_status.json','dataset_ready',0,len(rows))
 if a.dataset_only:
  report=['# Instruction Indirection Gap dataset audit','',f'Dataset SHA256: `{dsha}`; worlds: {audit_report["worlds"]}; rows: {len(rows)}.','',
   'The independent unit is the world. Each world has five depth renderings for comprehension and application. The semantic correct and incorrect candidates are held fixed across all depths. Source ownership and proposal presentation order are each balanced 50/50 within every task family. No model predictions were collected in this dataset-only run.','',
   '## Task families','',
   '- Filename: select the exact proposed filename controlled by the policy owner.','- Ordering: return one of two literal three-symbol sequences proposed by the source and default policy.','- Destination: return one of two synthetic route identifiers proposed by the source and default policy.','',
   '## Frozen depth ladder','',
   '- 0: give the final semantic action directly and identify the owner.','- 1: state directly that the owner determines the field and use that owner’s proposal.','- 2: declare the field’s final decision owner and refer to the owner’s proposal.','- 3: bind the owner to a role, state that the role determines the field, then resolve the entity through its role.','- 4: bind owner to role, role to symbolic field, state the ownership rule, then apply the rule to the field.','',
   '## Audit','',f'`{json.dumps(audit_report["balance"],sort_keys=True)}`','',
   'The audit confirms semantic matching and counterbalancing. Inference, tokenization balance, depth-zero performance gates, bootstrap intervals, and behavioral conclusions remain unmeasured. Known limits include increasing prompt length with depth, one surface form per depth, no length-matched filler or flattened-rule control, and a direct depth-zero statement that explicitly names the owner. These prevent attributing any future difference solely to reasoning depth without follow-up controls.','']
  (run/'pilot_report.md').write_text('\n'.join(report))
  logger.info('Dataset-only audit complete; model inference was not started')
  print(run);return
 registry=yaml.safe_load(a.models_config.read_text()).get('models',{});spec=registry.get(model,{'name':model})
 write_run_status(run/'run_status.json','loading_model',0,len(rows))
 logger.info('Loading model checkpoint=%s requested_revision=%s device=%s',spec['name'],spec.get('revision'),a.device)
 model_started=time.monotonic()
 try:
  adapter=HFAdapter(spec['name'],spec.get('revision'),a.device,enable_thinking=spec.get('enable_thinking'),quantization=spec.get('quantization'),attention_implementation=spec.get('attention_implementation'))
 except BaseException as exc:
  write_run_status(run/'run_status.json','model_load_failed',0,len(rows),repr(exc));logger.exception('Model loading failed');raise
 prov=adapter.provenance();meta={'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),'model':prov,'model_name':spec['name'],'model_revision':prov.get('model_commit'),'tokenizer_revision':prov.get('tokenizer_commit'),'chat_template_sha256':prov.get('chat_template_sha256'),'transformers_version':prov.get('transformers_version'),'torch_version':prov.get('torch_version'),'seed':cfg['seed'],'dataset_sha256':dsha,'decoding':{'do_sample':False,'max_new_tokens':int(cfg.get('max_new_tokens',32)),'candidate_scoring':'joint prompt+candidate tokenization; sum log probability primary, mean diagnostic'},'candidate_scoring_mode':'joint_offsets_with_boundary_overlap_audit'}
 logger.info('Model ready actual_device=%s model_revision=%s tokenizer_revision=%s transformers=%s torch=%s load_seconds=%.2f',prov.get('device'),prov.get('model_commit'),prov.get('tokenizer_commit'),prov.get('transformers_version'),prov.get('torch_version'),time.monotonic()-model_started)
 (run/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
 # Use exactly same frozen prompts in token audit and predictions.
 counts=[]
 audit_started=time.monotonic()
 for r in tqdm(rows,total=len(rows),desc='Auditing candidate tokenization',unit='prompt',leave=False):
  rendered=adapter.render([Message('user',r['prompt'])])
  for c in (r['correct_candidate'],r['incorrect_candidate']):
   x=continuation_encoding(adapter.tokenizer,rendered,c)
   counts.append({'world_id':r['world_id'],'family':r['task_family'],'depth':r['indirection_depth'],'measurement':r['measurement_type'],'candidate':c,'joint_token_count':len(x['input_ids']),'candidate_token_count':x['token_count'],'boundary_overlap':x['boundary_overlap'],'prompt_prefix_retokenized':x['prompt_prefix_retokenized']})
 write_jsonl(run/'tokenization_audit.jsonl',counts)
 pairs={}
 for item in counts:
  pairs.setdefault((item['world_id'],item['depth'],item['measurement']),[]).append(item['candidate_token_count'])
 imbalance=[{'world_id':wid,'depth':depth,'measurement':task,'token_counts':vals}
            for (wid,depth,task),vals in pairs.items() if len(vals)!=2 or vals[0]!=vals[1]]
 (run/'tokenization_audit.json').write_text(json.dumps({
  'candidate_continuation_tokens':len(counts),
  'unequal_token_length_pairs':len(imbalance),
  'unequal_token_length_pair_examples':imbalance[:50],
  'note':'Boundary-overlap details and both candidate token counts are in tokenization_audit.jsonl.'
 },indent=2)+'\n')
 overlap_count=sum(bool(item['boundary_overlap']) for item in counts)
 retokenized_count=sum(bool(item['prompt_prefix_retokenized']) for item in counts)
 logger.info('Tokenization audit complete candidates=%d unequal_pairs=%d boundary_overlaps=%d retokenized_prefixes=%d elapsed_seconds=%.2f',len(counts),len(imbalance),overlap_count,retokenized_count,time.monotonic()-audit_started)
 if imbalance: logger.warning('Found %d unequal candidate token-length pairs; inspect tokenization_audit.json',len(imbalance))
 write_run_status(run/'run_status.json','running',0,len(rows));pred=[]
 def scored_rows():
  scoring_started=time.monotonic()
  for i,r in enumerate(tqdm(rows,total=len(rows),desc='Scoring semantic candidates',unit='prompt'),1):
   item=score_example(r,adapter,model);pred.append(item)
   if i%20==0 or i==len(rows):
    write_run_status(run/'run_status.json','running',i,len(rows))
    logger.info('Scored prompts=%d/%d elapsed_seconds=%.1f',i,len(rows),time.monotonic()-scoring_started)
   yield item
 try:
  write_jsonl(run/'predictions.jsonl',scored_rows())
 except BaseException as exc:
  write_run_status(run/'run_status.json','interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',len(pred),len(rows),repr(exc))
  logger.exception('Candidate scoring failed after %d/%d prompts',len(pred),len(rows));raise
 write_jsonl(run/'per_world_metrics.jsonl',({'world_id':r['world_id'],'task_family':r['task_family'],'indirection_depth':r['indirection_depth'],'template_id':r['template_id'],'model':r['model'],'measurement_type':r['measurement_type'],'margin':r['margin'],'correct':r['correct'],'candidate_scores':r['candidate_scores']} for r in pred))
 analysis_started=time.monotonic()
 try:
  metrics=analyze(pred,int(cfg['seed']),int(cfg.get('bootstrap_replicates',4000)))
 except BaseException as exc:
  write_run_status(run/'run_status.json','failed',len(pred),len(rows),repr(exc))
  logger.exception('World-level analysis failed');raise
 logger.info('Analysis complete worlds=%d rows=%d strata=%d paired_strata=%d bootstrap_replicates=%d elapsed_seconds=%.2f',metrics['n_worlds'],metrics['n_prediction_rows'],len(metrics['summaries']),len(metrics['paired_change']),int(cfg.get('bootstrap_replicates',4000)),time.monotonic()-analysis_started)
 for key,value in sorted(metrics['summaries'].items()):
  logger.info('Metric stratum=%s worlds=%d accuracy=%.4f accuracy_ci95=%s mean_margin=%.5f margin_ci95=%s',key,value['n_worlds'],value['accuracy'],value['accuracy_ci95_world'],value['mean_margin'],value['margin_ci95_world'])
 write_jsonl(run/'world_outcomes.jsonl',metrics['outcomes_by_world'])
 (run/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
 plot_started=time.monotonic();logger.info('Generating figures and plot-source data')
 plot(pred,metrics,run/'figures')
 logger.info('Plot generation complete elapsed_seconds=%.2f',time.monotonic()-plot_started)
 gates={}
 for task in ('application','comprehension'):
  x=[r['correct'] for r in pred if r['indirection_depth']==0 and r['measurement_type']==task]
  gates[f'{task}_depth0_accuracy']=sum(x)/len(x)
  gates[f'warn_{task}_below_0_95']=gates[f'{task}_depth0_accuracy']<.95
  logger.info('Engineering gate task=%s depth0_accuracy=%.4f threshold=0.95 warning=%s',task,gates[f'{task}_depth0_accuracy'],gates[f'warn_{task}_below_0_95'])
  if gates[f'warn_{task}_below_0_95']:
   logger.warning('Depth-zero %s accuracy is below 0.95; do not interpret deeper conditions as an indirection effect.',task)
 report=['# Instruction Indirection Gap pilot','',f'Model: {model}; dataset SHA256: `{dsha}`; scored rows: {len(pred)}.','',f'Depth 0 gates: `{json.dumps(gates,sort_keys=True)}`.','', 'The independent unit is the synthetic world. All confidence intervals resample worlds; comprehension and application are repeated measures. Positive margins favor the correct semantic candidate.','', 'Metrics and plot source data are saved alongside predictions. This behavioral pilot does not establish an internal mechanism.','']
 report += ['', '## Depth-wise results', '', '| Task family | Task | Depth | Worlds | Accuracy (95% world CI) | Mean margin (95% world CI) |', '|---|---|---:|---:|---:|---:|']
 for key,value in sorted(metrics['summaries'].items()):
  family,template,mdl,task,depth=key.split('|')
  report.append(f"| {family} | {task} | {depth} | {value['n_worlds']} | {value['accuracy']:.3f} ({value['accuracy_ci95_world']}) | {value['mean_margin']:.3f} ({value['margin_ci95_world']}) |")
 report += ['', '## Direct-to-indirect paired change', '', '| Family | Task | Depth | Mean margin change (95% world CI) | Accuracy change (95% world CI) |', '|---|---|---:|---:|---:|']
 for key,value in sorted(metrics['paired_change'].items()):
  family,template,mdl,task,depth=key.split('|')
  report.append(f"| {family} | {task} | {depth} | {value['mean_margin_change_from_depth0']:.3f} ({value['margin_change_ci95_world']}) | {value['mean_accuracy_change_from_depth0']:.3f} ({value['accuracy_change_ci95_world']}) |")
 report += ['', '## Comprehension and application outcomes', '', '| Family / depth | CC/AC | CW/AW | CC/AW | CW/AC |', '|---|---:|---:|---:|---:|']
 for key,c in sorted(metrics['comprehension_application_categories'].items()):
  family,template,mdl,d=key.split('|')
  report.append(f"| {family} / {d} | {c['comprehension_correct_application_correct']} | {c['comprehension_wrong_application_wrong']} | {c['comprehension_correct_application_wrong']} | {c['comprehension_wrong_application_correct']} |")
 report += ['', 'CC=both correct; CW/AW=both wrong; CC/AW=comprehension correct/application wrong; CW/AC=comprehension wrong/application correct.', '']
 (run/'pilot_report.md').write_text('\n'.join(report));write_run_status(run/'run_status.json','complete',len(pred),len(rows));print(run)
 logger.info('Run complete output=%s scored=%d',run,len(pred))
if __name__=='__main__': main()
