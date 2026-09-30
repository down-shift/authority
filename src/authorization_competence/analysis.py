"""World-clustered summaries and preregistered competence gates."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

REPRESENTATIONS = ('json', 'decision_owner', 'natural_language', 'permission_table', 'executable_rule')
BIAS_POSITION_MAX_GAP = 0.15
BIAS_IDENTITY_MAX_RANGE = 0.25
COMPETENCE_ACCURACY_MIN = 0.90


def boot(values, seed, n=4000):
    x = np.asarray(list(values), dtype=float)
    if not len(x): return {'mean': None, 'ci95': [None, None], 'n_worlds': 0}
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), (n, len(x)))
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(x[ix].mean(1), [.025,.975])], 'n_worlds': len(x)}


def world_means(rows, metric, group=()):
    df = pd.DataFrame(rows)
    keys = ['world_id', *group]
    return df.groupby(keys, dropna=False)[metric].mean().reset_index()


def stage1_analysis(rows, seed=20261003):
    df = pd.DataFrame([r for r in rows if r['stage'] == 1]).copy()
    df['correctness'] = (df['margin'] > 0).astype(float)
    summaries = {}
    for task in ('interpretation', 'application'):
        g = df[df.task == task]
        summaries[task] = {
            'accuracy': boot(g.groupby('world_id').correctness.mean(), seed),
            'margin': boot(g.groupby('world_id').margin.mean(), seed + 1),
            'raw_correct': int(g.correctness.sum()), 'n_rows': len(g),
        }
    bias = {}
    for task in ('interpretation', 'application'):
        g = df[df.task == task]
        # Position of the authorized actor among the displayed actors/proposals.
        position = 'correct_actor_position' if task == 'interpretation' else 'correct_value_position'
        bias[task] = {}
        for pos, h in g.groupby(position):
            bias[task][f'correct_answer_position_{int(pos)+1}'] = {
                'accuracy': boot(h.groupby('world_id').correctness.mean(), seed + int(pos) + 10),
                'margin': boot(h.groupby('world_id').margin.mean(), seed + int(pos) + 20),
                'n_rows': len(h),
            }
        by_actor = {}
        for actor, h in g.groupby('owner'):
            by_actor[actor] = {'accuracy': boot(h.groupby('world_id').correctness.mean(), seed + len(by_actor) + 30),
                               'margin': boot(h.groupby('world_id').margin.mean(), seed + len(by_actor) + 40),
                               'n_rows': len(h)}
        bias[task]['authorized_actor_identity'] = by_actor
    # Explicit proposal-order diagnostic, separate from owner identity.
    app = df[df.task == 'application']
    bias['application']['first_listed_proposal_actor'] = {}
    for first, h in app.assign(first_proposal_owner=app.correct_value_position.eq(0)).groupby('first_proposal_owner'):
        bias['application']['first_listed_proposal_actor']['owner_first' if first else 'owner_second'] = {
            'accuracy': boot(h.groupby('world_id').correctness.mean(), seed + 60 + int(first)),
            'margin': boot(h.groupby('world_id').margin.mean(), seed + 70 + int(first)), 'n_rows': len(h)}
    # Determine visible bias from group accuracies at the actual trial level.
    gaps = {}
    for task in ('interpretation', 'application'):
        poscol = 'correct_actor_position' if task == 'interpretation' else 'correct_value_position'
        g = df[df.task == task]
        posrates = g.groupby(poscol).correctness.mean().to_dict()
        idrates = g.groupby('owner').correctness.mean().to_dict()
        gaps[task] = {'position_accuracy_gap': abs(posrates.get(0, 0) - posrates.get(1, 0)),
                      'identity_accuracy_range': max(idrates.values()) - min(idrates.values())}
    accuracy_pass = all(summaries[t]['raw_correct'] / summaries[t]['n_rows'] >= COMPETENCE_ACCURACY_MIN for t in summaries)
    position_pass = all(x['position_accuracy_gap'] <= BIAS_POSITION_MAX_GAP for x in gaps.values())
    identity_pass = all(x['identity_accuracy_range'] <= BIAS_IDENTITY_MAX_RANGE for x in gaps.values())
    score_issue = any(r.get('score_audit', {}).get('candidate_token_count_mismatch', False) or
                      r.get('score_audit', {}).get('boundary_error', False) for r in rows if r['stage'] == 1)
    gate = {'interpretation_accuracy_at_least_0_90': summaries['interpretation']['raw_correct']/summaries['interpretation']['n_rows'] >= COMPETENCE_ACCURACY_MIN,
            'application_accuracy_at_least_0_90': summaries['application']['raw_correct']/summaries['application']['n_rows'] >= COMPETENCE_ACCURACY_MIN,
            'position_gap_at_most_0_15': position_pass,
            'actor_identity_range_at_most_0_25': identity_pass,
            'candidate_scoring_audit_passed': not score_issue,
            'passed': accuracy_pass and position_pass and identity_pass and not score_issue,
            'thresholds': {'accuracy_min': COMPETENCE_ACCURACY_MIN, 'position_gap_max': BIAS_POSITION_MAX_GAP,
                           'actor_identity_range_max': BIAS_IDENTITY_MAX_RANGE}}
    gate['failure_categories'] = [name for name, failed in (
        ('interpretation', not gate['interpretation_accuracy_at_least_0_90']),
        ('application', not gate['application_accuracy_at_least_0_90']),
        ('position_bias', not gate['position_gap_at_most_0_15']),
        ('actor_identity_bias', not gate['actor_identity_range_at_most_0_25']),
        ('candidate_scoring_issue', not gate['candidate_scoring_audit_passed'])) if failed]
    return {'worlds': df.world_id.nunique(), 'summaries': summaries, 'bias_diagnostics': bias,
            'bias_gaps': gaps, 'competence_gate': gate,
            'generation_diagnostics': generation_summary(rows)}


def generation_summary(rows):
    out = {}
    for task in ('interpretation', 'application'):
        g = [r for r in rows if r.get('stage') == 1 and r['task'] == task]
        exact = [r.get('generated_exact') for r in g if r.get('generated_exact') is not None]
        out[task] = {'exact_accuracy': float(np.mean(exact)) if exact else None, 'n': len(exact)}
    return out


def representation_analysis(rows, seed=20261003):
    df = pd.DataFrame(rows)
    df['correctness'] = (df.margin > 0).astype(float)
    cells=[]; disagreement=[]; dispersion=[]; pairs=[]
    for (stage, task, rep), g in df.groupby(['stage','task','representation'], sort=True):
        cells.append({'stage':int(stage),'task':task,'representation':rep,
          'accuracy':boot(g.groupby('world_id').correctness.mean(),seed),
          'margin':boot(g.groupby('world_id').margin.mean(),seed+1), 'rows':len(g)})
    for (stage,task),g in df.groupby(['stage','task'],sort=True):
        choices=g.pivot(index=['world_id','resource'],columns='representation',values='selected')
        margins=g.pivot(index=['world_id','resource'],columns='representation',values='margin')
        dis=choices.nunique(axis=1).gt(1).astype(float)
        var=margins.var(axis=1,ddof=0)
        disagreement.append({'stage':int(stage),'task':task,'fraction_worlds_with_disagreement':boot(dis.groupby('world_id').mean(),seed+2),
                             'disagreement_count':int(dis.sum()),'queries':len(dis)})
        dispersion.append({'stage':int(stage),'task':task,'within_world_margin_variance':boot(var.groupby('world_id').mean(),seed+3),
                           'mean_range':boot((margins.max(axis=1)-margins.min(axis=1)).groupby('world_id').mean(),seed+4)})
        for a,b in combinations(REPRESENTATIONS,2):
            d=margins[b]-margins[a]
            pairs.append({'stage':int(stage),'task':task,'contrast':f'{b} minus {a}',
                          'margin_difference':boot(d.groupby('world_id').mean(),seed+5),
                          'choice_disagreement':boot(choices[a].ne(choices[b]).groupby('world_id').mean(),seed+6)})
    return {'cells':cells,'categorical_disagreement':disagreement,'margin_dispersion':dispersion,
            'pairwise_representation_effects':pairs}


def canonicalization_analysis(raw_rows, canonicalized_rows, conversions, seed=20261003):
    rows = list(raw_rows) + list(canonicalized_rows)
    result={}
    for stage in sorted({r['stage'] for r in rows}):
        stage_result={}
        world_outcomes={}
        for arm in ('raw','canonicalized'):
            g=[r for r in rows if r['stage']==stage and r['arm']==arm]
            by=defaultdict(list); choices={}
            for r in g:
                by[r['world_id']].append(r)
                choices.setdefault((r['world_id'],r['resource']),{})[r['representation']]=r['selected']
            flags=defaultdict(list)
            for (world,_resource),answers in choices.items():
                flags[world].append(float(len(set(answers.values()))>1))
            stage_result[arm]={
                'accuracy':boot([np.mean([r['margin']>0 for r in rs]) for rs in by.values()],seed),
                'mean_margin':boot([np.mean([r['margin'] for r in rs]) for rs in by.values()],seed+1),
                'disagreement':boot([np.mean(v) for v in flags.values()],seed+2),
            }
            world_outcomes[arm]={w:{'accuracy':float(np.mean([r['margin']>0 for r in rs])),
                                    'margin':float(np.mean([r['margin'] for r in rs]))} for w,rs in by.items()}
        shared=sorted(world_outcomes['raw'].keys() & world_outcomes['canonicalized'].keys())
        stage_result['paired_change_canonicalized_minus_raw']={
            metric:boot([world_outcomes['canonicalized'][w][metric]-world_outcomes['raw'][w][metric]
                         for w in shared],seed+10+i)
            for i,metric in enumerate(('accuracy','margin'))}
        stage_result['conversion_accuracy']=float(np.mean([c['exact'] for c in conversions if c['stage']==stage])) if any(c['stage']==stage for c in conversions) else None
        stage_result['conversion_exact_count']=sum(c['exact'] for c in conversions if c['stage']==stage)
        stage_result['conversion_count']=sum(c['stage']==stage for c in conversions)
        result[str(stage)]=stage_result
    return {'by_stage':result,
            'conversion_accuracy_by_stage_representation':{
                f"{stage}/{rep}":sum(c['exact'] for c in conversions if c['stage']==stage and c['representation']==rep)/
                    max(1,sum(c['stage']==stage and c['representation']==rep for c in conversions))
                for stage in sorted({c['stage'] for c in conversions}) for rep in REPRESENTATIONS}}


def stage_report(metrics, stage, heading):
    lines=[f'# {heading}','',f"Bootstrap unit: world. Competence threshold: {metrics['competence_gate']['threshold']:.2f} per task and representation.",'',
      '| Task | Representation | Accuracy (95% CI) | Margin (95% CI) |','|---|---|---:|---:|']
    def fmt(x): return 'NA' if x['mean'] is None else f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    for c in metrics['cells']:
        lines.append(f"| {c['task']} | {c['representation']} | {fmt(c['accuracy'])} | {fmt(c['margin'])} |")
    lines += ['', '## Cross-representation outcomes', '',
      '| Task | Worlds with any different selected answer (95% CI) | Mean within-world margin variance (95% CI) | Mean margin range (95% CI) |',
      '|---|---:|---:|---:|']
    for d,v in zip(metrics['categorical_disagreement'],metrics['margin_dispersion']):
        lines.append(f"| {d['task']} | {fmt(d['fraction_worlds_with_disagreement'])} | {fmt(v['within_world_margin_variance'])} | {fmt(v['mean_range'])} |")
    lines += ['', '## Competence gate', '',
      f"Gate passed: **{metrics['competence_gate']['competence_passed']}**.",
      'The pairwise contrasts and their world-cluster bootstrap intervals are in the accompanying metrics JSON. No best/worst format ranking is used.', '']
    if 'single_vs_two_scope_disagreement' in metrics:
        lines += ['## Single-scope versus two-scope disagreement', '',
                  '| Task | Single-scope filename | Two-scope queries averaged | Paired change |',
                  '|---|---:|---:|---:|']
        for x in metrics['single_vs_two_scope_disagreement']:
            lines.append(f"| {x['task']} | {fmt(x['single_scope_filename_disagreement'])} | {fmt(x['two_scope_two_query_disagreement'])} | {fmt(x['paired_change_two_scope_minus_single_filename'])} |")
        lines.append('')
    return '\n'.join(lines)


def report_stage1(metrics):
    g=metrics['competence_gate']; s=metrics['summaries']
    lines=['# Stage 1: canonical authorization competence calibration','',
      f"Worlds: {metrics['worlds']}. One JSON policy formulation; direct semantic candidate scores are primary. Accuracy is based on which semantic continuation has higher sum log probability. Bootstrap resamples worlds.",'',
      '| Task | Accuracy (95% world-cluster CI) | Correct | Mean margin (95% CI) |','|---|---:|---:|---:|']
    for task in ('interpretation','application'):
      x=s[task]
      lines.append(f"| {task} | {x['raw_correct']/x['n_rows']:.3f} [{x['accuracy']['ci95'][0]:.3f}, {x['accuracy']['ci95'][1]:.3f}] | {x['raw_correct']}/{x['n_rows']} | {x['margin']['mean']:.3f} [{x['margin']['ci95'][0]:.3f}, {x['margin']['ci95'][1]:.3f}] |")
    lines += ['', '## Bias diagnostics', '', '| Task | Position accuracy gap | Authorized actor identity range |', '|---|---:|---:|']
    for t,x in metrics['bias_gaps'].items(): lines.append(f"| {t} | {x['position_accuracy_gap']:.3f} | {x['identity_accuracy_range']:.3f} |")
    lines += ['', 'Accuracy and margin by authorized actor, list position, and exact generation diagnostics are in `stage1_metrics.json`.', '', '## Frozen gate', '',
      f"- Interpretation >= 0.90: {g['interpretation_accuracy_at_least_0_90']}.",
      f"- Application >= 0.90: {g['application_accuracy_at_least_0_90']}.",
      f"- Position gap <= {g['thresholds']['position_gap_max']:.2f}: {g['position_gap_at_most_0_15']}.",
      f"- Actor identity range <= {g['thresholds']['actor_identity_range_max']:.2f}: {g['actor_identity_range_at_most_0_25']}.",
      f"- Candidate scoring audit: {g['candidate_scoring_audit_passed']}.", '',
      f"Failure categories: {', '.join(g['failure_categories']) if g['failure_categories'] else 'none'}.", '',
      f"**Gate: {'PASS' if g['passed'] else 'FAIL'}.** Representation stages {'may proceed' if g['passed'] else 'were not run'}.", '',
      'The Stage 1 world file is frozen and can be reused unchanged with one stronger instruction model if Qwen3-4B fails. No prompt tuning is performed.', '']
    return '\n'.join(lines)
