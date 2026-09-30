from indirection_v2.worlds import generate_worlds
from indirection_v2.render import examples
from indirection_v2.analysis import analyze
from indirection_v2.validation import summarize_tokenization


def test_world_bootstrap_pairs_depth_and_control_contrasts():
    rows=examples(generate_worlds(2,13),6,13)
    for r in rows:
        # A deterministic fixture with a depth penalty and a bridge rescue.
        if r['condition_type']=='path': margin=5-r['target_depth']
        elif r['condition_type']=='bridge': margin=4
        elif r['condition_type']=='neutral': margin=1-r['target_depth']
        else: margin=4
        if r['measurement_type']=='comprehension': margin=max(margin,1)
        r.update(model='fixture',margin=float(margin),correct=margin>0,
                 candidate_scores={r['correct_candidate']:{'prompt_token_count':100}})
    m=analyze(rows,seed=4,n_boot=50)
    assert m['n_worlds']==6
    assert 'filename|resolution_grammar_v2|fixture|application|path|1|1' in m['summaries']
    assert m['summaries']['filename|resolution_grammar_v2|fixture|application|path|1|1']['stratum']['task_family']=='filename'
    key='filename|fixture|application|5'
    assert m['path_depth_vs_depth1'][key]['n_worlds']==2
    control='filename|fixture|5|bridge_vs_neutral'
    assert m['application_control_contrasts'][control]['mean_margin_difference']>0
    assert len(m['comprehension_application_by_world'])==6*5


def test_token_audit_reports_unbalanced_candidates_and_prompt_spread():
    rows=[{'world_id':'w','task_family':'filename','measurement_type':'application','condition_type':'path','target_depth':1},
          {'world_id':'w','task_family':'filename','measurement_type':'application','condition_type':'path','target_depth':2},
          {'world_id':'w','task_family':'filename','measurement_type':'application','condition_type':'path','target_depth':3},
          {'world_id':'w','task_family':'filename','measurement_type':'application','condition_type':'bridge','target_depth':3},
          {'world_id':'w','task_family':'filename','measurement_type':'application','condition_type':'neutral','target_depth':3},
          {'world_id':'w','task_family':'filename','measurement_type':'application','condition_type':'flattened','target_depth':3}]
    counts=[]
    for r,n in zip(rows,(10,12,14,16,14,14)):
        for candidate,tokens in (('correct',3),('wrong',4 if r['target_depth']==2 else 3)):
            counts.append({**r,'candidate':candidate,'prompt_token_count':n,'continuation_token_count':tokens})
    report=summarize_tokenization(rows,counts)
    assert report['unequal_candidate_token_pairs']==1
    assert report['path_prompt_token_spread_max']==4
    assert report['bridge_neutral_token_mismatches']==1
