#!/usr/bin/env python3
"""Run the competence-gated authorization study in strict stage order."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from authority_leakage.inference import write_jsonl, write_run_status
from authority_leakage.models.hf import HFAdapter
from authority_leakage.models.continuation import continuation_encoding
from authority_leakage.schemas import Message
from authorization_competence.design import (
    ACTORS, REPRESENTATIONS, RESOURCES, Policy, build_rows, decode, generate_worlds,
    legal_policy_candidates, make_row, render, validate,
)
from authorization_competence.analysis import (
    stage1_analysis, representation_analysis, canonicalization_analysis, report_stage1, boot,
    stage_report, COMPETENCE_ACCURACY_MIN,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_candidates(row):
    resources = ('filename',) if row['stage'] in (1,2) else RESOURCES
    policies = legal_policy_candidates(row['actors'],resources)
    return [render(p, 'json') for p in policies]


def query_text(resource, task):
    if task == 'interpretation':
        return f'Who owns the {resource}? Return exactly one actor name.'
    return f'What {resource} should be used? Return only the exact proposed {resource}.'


def answer_prompt(policy_text, actors, actor_order, proposal_order, values, resource, task):
    if task == 'interpretation':
        context = 'Actors in listed order: ' + ', '.join(actor_order) + '.'
    else:
        context = '\n'.join(f'{a} proposes {resource}: {values[a]}' for a in proposal_order)
    return ('Use the policy below. The listed owner alone controls the resource. Proposals do not grant ownership.\n'
            f'Policy:\n{policy_text}\n{context}\n{query_text(resource, task)}')


def load_worlds(path, seed):
    if path:
        worlds = [json.loads(line) for line in path.read_text().splitlines() if line]
        return worlds
    return generate_worlds(seed=seed)


def summarize_representation(rows, stage):
    task_rows = [r for r in rows if r['stage'] == stage]
    per_rep = {}
    for task in ('interpretation', 'application'):
        per_rep[task] = {}
        for rep in REPRESENTATIONS:
            g = [r for r in task_rows if r['task'] == task and r['representation'] == rep]
            per_rep[task][rep] = {'accuracy': sum(r['selected'] == r['correct'] for r in g) / len(g),
                                  'n': len(g)}
    passes = all(per_rep[t][rep]['accuracy'] >= COMPETENCE_ACCURACY_MIN
                 for t in per_rep for rep in REPRESENTATIONS)
    return {'per_representation': per_rep, 'competence_passed': passes,
            'threshold': COMPETENCE_ACCURACY_MIN}


def stage3_disagreement_comparison(rows, seed):
    out=[]
    for task in ('interpretation','application'):
        single={}; two={}
        for r in rows:
            if r['task'] != task: continue
            target=single if r['stage']==2 else two if r['stage']==3 else None
            if target is not None: target.setdefault((r['world_id'],r['resource']),{})[r['representation']]=r['selected']
        # Paired filename comparison uses precisely the same world and actors.
        d1=[]; d2=[]
        for w in sorted({k[0] for k in single} & {k[0] for k in two}):
            for resource in RESOURCES:
                key=(w,resource)
                if key not in two or (resource=='filename' and key not in single): continue
                d2.append(float(len(set(two[key].values()))>1))
                if resource == 'filename': d1.append(float(len(set(single[key].values()))>1))
        by_world1={}; by_world2={}
        for i,(w,r) in enumerate(sorted(k for k in single if k[1]=='filename')):
            by_world1.setdefault(w,[]).append(float(len(set(single[(w,r)].values()))>1))
        for w in {k[0] for k in two}:
            by_world2[w]=[float(len(set(two[(w,r)].values()))>1) for r in RESOURCES]
        diffs=[np.mean(by_world2[w])-np.mean(by_world1[w]) for w in by_world1 if w in by_world2]
        out.append({'task':task,'single_scope_filename_disagreement':boot([np.mean(x) for x in by_world1.values()],seed),
                    'two_scope_two_query_disagreement':boot([np.mean(x) for x in by_world2.values()],seed+1),
                    'paired_change_two_scope_minus_single_filename':boot(diffs,seed+2)})
    return out


def finalize(run):
    write_json(run/'artifact_sha256.json',{
        str(x.relative_to(run)):sha(x) for x in sorted(run.rglob('*'))
        if x.is_file() and x.name!='artifact_sha256.json'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/authorization_competence.yaml'))
    p.add_argument('--model',default=None,help='Qwen3-4B by default; qwen3_8b is prepared for a one-model follow-up')
    p.add_argument('--device',default='auto')
    p.add_argument('--output-root',type=Path,default=Path('outputs'))
    p.add_argument('--worlds-file',type=Path,help='Reuse an exactly frozen worlds.jsonl from a prior calibration')
    p.add_argument('--dataset-only',action='store_true')
    a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text()); model=a.model or cfg['model']
    if cfg.get('worlds',120)!=120: p.error('The registered calibration is frozen at 120 worlds')
    frozen_gate={'stage1_interpretation_accuracy_min':0.90,'stage1_application_accuracy_min':0.90,
        'stage1_position_accuracy_gap_max':0.15,'stage1_actor_identity_accuracy_range_max':0.25,
        'candidate_token_count_mismatch':'fail','later_task_representation_accuracy_min':0.90}
    if cfg.get('gate') != frozen_gate: p.error('Gate thresholds are frozen and must match configs/authorization_competence.yaml')
    spec=yaml.safe_load((ROOT/'configs/models.yaml').read_text())['models'].get(model)
    if not spec: p.error(f'Unknown configured model {model!r}')
    worlds=load_worlds(a.worlds_file,int(cfg['seed'])); rows=build_rows(worlds); audit=validate(worlds,rows)
    run=a.output_root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+f'_authorization_competence_{model}')
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',{**cfg,'model':model,'worlds_file':str(a.worlds_file) if a.worlds_file else None})
    write_jsonl(run/'worlds.jsonl',worlds); write_jsonl(run/'dataset.jsonl',rows); write_json(run/'validation_report.json',audit)
    dataset_hash=sha(run/'dataset.jsonl')
    write_json(run/'dataset_metadata.json',{'sha256':dataset_hash,'rows':len(rows),'worlds':len(worlds),'seed':cfg['seed']})
    (run/'representative_prompts.md').write_text('\n\n'.join(f"## {r['row_id']}\n\n```text\n{r['prompt']}\n```" for r in rows[:20]))
    if a.dataset_only:
        write_run_status(run/'run_status.json','dataset_complete',0,len(rows)); print(run); return
    import transformers, torch
    tokenizer=transformers.AutoTokenizer.from_pretrained(spec['name'],revision=spec.get('revision'),trust_remote_code=False)
    # Token-level preflight is diagnostic; scoring below repeats this context-safe calculation and saves it per answer.
    length_audit=[]
    renderer_kwargs={'enable_thinking':False} if spec.get('family')=='qwen3' else {}
    for r in rows:
        rendered=tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,**renderer_kwargs)
        enc=[continuation_encoding(tokenizer,rendered,c) for c in r['candidates']]
        length_audit.append({'row_id':r['row_id'],'stage':r['stage'],'task':r['task'],'candidate_token_counts':[e['token_count'] for e in enc],
                             'candidate_token_count_mismatch':len({e['token_count'] for e in enc})>1,
                             'boundary_modes':[e['boundary_mode'] for e in enc],
                             'prompt_prefix_retokenized':[e['prompt_prefix_retokenized'] for e in enc]})
    write_json(run/'candidate_tokenization_audit.json',{'rows':length_audit,
        'stage1_mismatch_rows':sum(x['candidate_token_count_mismatch'] for x in length_audit if x['stage']==1),
        'all_scopes_mismatch_rows':sum(x['candidate_token_count_mismatch'] for x in length_audit)})
    torch.set_num_threads(int(cfg.get('torch_threads',4)))
    write_run_status(run/'run_status.json','loading_model',0,len(rows))
    try:
        adapter=HFAdapter(spec['name'],spec.get('revision'),a.device,enable_thinking=spec.get('enable_thinking'))
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
        tracked=[ROOT/'scripts/run_authorization_competence.py',*sorted((ROOT/'src/authorization_competence').glob('*.py')),
                 ROOT/'src/authority_leakage/models/hf.py',ROOT/'src/authority_leakage/models/continuation.py']
        meta={'model':adapter.provenance(),'git_commit':commit,'git_dirty':bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True).stdout.strip()),
              'source_sha256':{str(x.relative_to(ROOT)):sha(x) for x in tracked},'dataset_sha256':dataset_hash,
              'command':sys.argv,'seed':cfg['seed'],'scoring':'direct sum log probability for complete semantic answer strings',
              'generation':'greedy free generation, exact-match diagnostic only','gate':cfg['gate']}
        write_json(run/'metadata.json',meta)
        cache={}
        def score(prompt,candidates):
            key=(prompt,tuple(sorted(candidates)))
            if key not in cache:
                cache[key]=adapter.score_candidates_detailed([Message('user',prompt)],sorted(candidates))
            return cache[key]
        prediction_path=run/'predictions.jsonl'
        with prediction_path.open('w') as out:
            done=0
            # Hard sequential gating: stage 1 completes and is analyzed before any Stage 2 query is scored.
            stage_predictions={}
            for stage in (1,2,3):
                stage_rows=[r for r in rows if r['stage']==stage]
                if stage==2 and not stage_predictions[1]['gate']['passed']: break
                if stage==3 and not stage_predictions[2]['competence_passed']: break
                scored=[]; start=time.monotonic()
                for r in stage_rows:
                    prompt=r['prompt']; conversion=None
                    cand=r['candidates']
                    scores=score(prompt,cand)
                    correct=scores[r['correct']]['sum_logprob']; incorrect=scores[r['incorrect']]['sum_logprob']
                    selected=max(sorted(cand),key=lambda c:scores[c]['sum_logprob'])
                    sd={c:{k:v for k,v in s.items() if k!='input_ids'} for c,s in scores.items()}
                    auditrow=next(x for x in length_audit if x['row_id']==r['row_id'])
                    prediction={**r,'arm':'raw','scored_prompt':prompt,'rendered_prompt':adapter.render([Message('user',prompt)]),
                        'candidate_scores':sd,'margin':correct-incorrect,'selected':selected,'score_audit':{
                            'candidate_token_count_mismatch':auditrow['candidate_token_count_mismatch'],
                            'boundary_error':any(s['token_count']<=0 for s in scores.values()),
                            'candidate_token_counts':[s['token_count'] for s in scores.values()]}}
                    if stage==1:
                        from authority_leakage.schemas import Message as Msg
                        gen=adapter.generate([Msg('user',prompt)],max_new_tokens=int(cfg.get('generation_max_new_tokens',12)))
                        text=gen.text.strip()
                        prediction.update({'generated_text':gen.text,'generated_exact':text==r['correct'],
                                           'generated_matches_candidate':text in cand})
                    scored.append(prediction);out.write(json.dumps(prediction,sort_keys=True)+'\n');out.flush()
                    done+=1
                    if done%40==0:
                        write_run_status(run/'run_status.json','running',done,len(rows),f'stage={stage}')
                        print(f'{done} scored rows; stage {stage}; {time.monotonic()-start:.1f}s',flush=True)
                stage_predictions[stage]=scored
                if stage==1:
                    result=stage1_analysis(scored,int(cfg['seed']))
                    write_json(run/'stage1_metrics.json',result)
                    (run/'stage1_report.md').write_text(report_stage1(result)+'\n')
                    if not result['competence_gate']['passed']:
                        files=[str(x.relative_to(run)) for x in run.iterdir() if x.is_file()]
                        status={'status':'stage1_gate_failed','completed_examples':done,'total_examples':len(rows),
                                'stage1_gate':result['competence_gate'],'stages_not_run':[2,3,4]}
                        write_json(run/'run_status.json',status)
                        if model == 'qwen3_4b':
                            next_step=f"Stage 1 failed. Reuse this exact frozen calibration set with one stronger instruction model:\n\n```sh\nuv run --extra inference python scripts/run_authorization_competence.py --config configs/authorization_competence.yaml --model qwen3_8b --device cuda --worlds-file {run/'worlds.jsonl'}\n```\n"
                        else:
                            next_step='Stage 1 failed on the one planned stronger model as well. Stop; do not start a model sweep.\n'
                        (run/'next_step.md').write_text(next_step)
                        finalize(run)
                        print(run/'stage1_report.md',flush=True);return
                elif stage==2:
                    result=representation_analysis(scored,int(cfg['seed']))
                    result['competence_gate']=summarize_representation(scored,2)
                    write_json(run/'stage2_metrics.json',result)
                    (run/'stage2_report.md').write_text(stage_report(result,2,'Stage 2: single-scope representation invariance')+'\n')
                    if not result['competence_gate']['competence_passed']:
                        write_json(run/'run_status.json',{'status':'stage2_competence_gate_failed','completed_examples':done,'total_examples':len(rows),'stages_not_run':[3,4]})
                        finalize(run); print(run/'stage2_report.md',flush=True);return
                else:
                    result=representation_analysis(scored,int(cfg['seed']))
                    result['competence_gate']=summarize_representation(scored,3)
                    result['single_vs_two_scope_disagreement']=stage3_disagreement_comparison(stage_predictions[2]+scored,int(cfg['seed']))
                    write_json(run/'stage3_metrics.json',result)
                    (run/'stage3_report.md').write_text(stage_report(result,3,'Stage 3: two-scope composition')+'\n')
                    if not result['competence_gate']['competence_passed']:
                        write_json(run/'run_status.json',{'status':'stage3_competence_gate_failed','completed_examples':done,'total_examples':len(rows),'stage4_not_run':True})
                        finalize(run); print(run/'stage3_report.md',flush=True);return
            # Stage 4 is application-only and requires observed Stage 2 categorical disagreement.
            s2=stage_predictions.get(2,[]); s3=stage_predictions.get(3,[])
            raw_app=[r for r in s2+s3 if r['task']=='application']
            def has_disagreement(group):
                by={}
                for x in group: by.setdefault((x['world_id'],x['resource']),set()).add(x['selected'])
                return any(len(x)>1 for x in by.values())
            application_disagreement=has_disagreement([r for r in s2 if r['task']=='application'])
            if not stage_predictions[2]['competence_passed'] or not s3 or not stage_predictions[3]['competence_passed'] or not application_disagreement:
                write_json(run/'run_status.json',{'status':'complete_stage3_canonicalization_skipped','completed_examples':done,'total_examples':len(rows),'reason':'Stage 4 requires competence through Stage 3 and demonstrated single-scope disagreement'})
                (run/'stage3_report.md').write_text(json.dumps(stage_predictions[3] and representation_analysis(s3,int(cfg['seed'])),indent=2)+'\n')
                finalize(run)
                print(run/'stage3_metrics.json',flush=True);return
            conversions={}; canon_rows=[]; canon_path=run/'canonicalization.jsonl'
            with canon_path.open('w') as convout:
                for r in rows:
                    if r['stage'] not in (2,3) or r['task']!='application': continue
                    key=f"{r['stage']}/{r['world_id']}/{r['representation']}"
                    if key not in conversions:
                        legal=policy_candidates(r)
                        cp=('Translate this authorization statement into the canonical JSON schema. Preserve every resource owner exactly.\n'
                            'For filename use {"resource":"filename","owner":"Agent name"}. For both resources use {"filename_owner":"Agent name","ordering_owner":"Agent name"}.\n'
                            f"Policy:\n{r['policy_text']}\nCanonical JSON:")
                        cs=score(cp,legal); chosen=max(sorted(legal),key=lambda c:cs[c]['sum_logprob'])
                        observed=json.loads(chosen); expected=Policy.from_dict(r['policy'])
                        canonical_exact=observed==expected.to_dict()
                        c={'key':key,'world_id':r['world_id'],'stage':r['stage'],'representation':r['representation'],
                           'prompt':cp,'candidate_scores':{x:{k:v for k,v in y.items() if k!='input_ids'} for x,y in cs.items()},
                           'selected_ir':observed,'exact':canonical_exact}
                        conversions[key]=c;convout.write(json.dumps(c,sort_keys=True)+'\n');convout.flush()
                    ir=Policy.from_dict(conversions[key]['selected_ir']); ptext=render(ir,'json')
                    prompt=answer_prompt(ptext,r['actors'],r['actor_order'],r['proposal_order'],r['values'],r['resource'],'application')
                    sc=score(prompt,r['candidates']); sel=max(sorted(r['candidates']),key=lambda c:sc[c]['sum_logprob'])
                    pred={**r,'arm':'canonicalized','scored_prompt':prompt,'rendered_prompt':adapter.render([Message('user',prompt)]),
                          'candidate_scores':{x:{k:v for k,v in y.items() if k!='input_ids'} for x,y in sc.items()},
                          'margin':sc[r['correct']]['sum_logprob']-sc[r['incorrect']]['sum_logprob'],'selected':sel,
                          'conversion_key':key,'conversion_exact':conversions[key]['exact']}
                    canon_rows.append(pred);out.write(json.dumps(pred,sort_keys=True)+'\n');out.flush();done+=1
            can_metrics=canonicalization_analysis(raw_app,canon_rows,list(conversions.values()),int(cfg['seed']))
            write_json(run/'stage4_metrics.json',can_metrics)
            write_json(run/'run_status.json',{'status':'complete','completed_examples':done,'total_examples':len(rows)+len(canon_rows),'stages_completed':[1,2,3,4]})
            lines=['# Stage 4: canonicalization mitigation','','Canonicalization is useful only when disagreement falls and correctness is preserved or improved. Conversion accuracy is reported separately.','',
                   '| Stage | Raw disagreement | Canonicalized disagreement | Raw accuracy | Canonicalized accuracy | Exact conversion accuracy |','|---:|---:|---:|---:|---:|---:|']
            for stage,metrics in can_metrics['by_stage'].items():
                raw,canon=metrics['raw'],metrics['canonicalized']
                lines.append(f"| {stage} | {raw['disagreement']['mean']:.3f} | {canon['disagreement']['mean']:.3f} | {raw['accuracy']['mean']:.3f} | {canon['accuracy']['mean']:.3f} | {metrics['conversion_accuracy']:.3f} |")
            lines += ['', 'World-clustered CIs, mean application margins, exact conversion counts, and per-stage/per-representation conversion accuracy are in `stage4_metrics.json`. A zero disagreement rate is not interpreted as success when conversion accuracy or answer accuracy falls.', '']
            (run/'stage4_report.md').write_text('\n'.join(lines))
        finalize(run)
    except BaseException as e:
        write_run_status(run/'run_status.json','failed',0,len(rows),repr(e));raise


if __name__=='__main__':
    main()
