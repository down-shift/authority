import copy
import json
from collections import Counter

import pytest

from authorization_competence.analysis import boot, representation_analysis, stage1_analysis
from authorization_competence.design import (
    ACTORS, REPRESENTATIONS, RESOURCES, Policy, build_rows, decode,
    generate_worlds, legal_policy_candidates, render, validate,
)
from authorization_competence.lexical_analysis import lexical_symmetry_analysis
from authorization_competence.lexical_analysis import lexical_representation_analysis
from authorization_competence.lexical_symmetry import (
    ACTOR_FAMILIES, ACTOR_IDENTIFIERS, ASSIGNMENTS, build_lexical_rows,
    build_lexical_representation_rows, decode_lexical_policy,
    generate_semantic_worlds, validate_lexical_worlds,
    validate_lexical_representation_rows,
)


def test_frozen_world_allocation_balances_actor_identity_and_positions():
    worlds = generate_worlds(seed=20261003)
    assert len(worlds) == 120
    for resource in RESOURCES:
        for actor in ACTORS:
            owned = [w for w in worlds if w['owner_by_resource'][resource] == actor]
            assert len(owned) == 20
            assert sum(w['actor_order'][0] == actor for w in owned) == 10
            assert sum(w['actor_order'][1] == actor for w in owned) == 10
            assert sum(w['proposal_order'][resource][0] == actor for w in owned) == 10
            assert sum(w['proposal_order'][resource][1] == actor for w in owned) == 10
    assert worlds == generate_worlds(seed=20261003)


def test_every_renderer_decodes_to_same_single_and_two_scope_policy():
    actors = ('Agent K', 'Agent M')
    for owners in (('Agent K',), ('Agent M',), ('Agent K', 'Agent M'), ('Agent M', 'Agent K')):
        resources = ('filename',) if len(owners) == 1 else RESOURCES
        policy = Policy(tuple(zip(resources, owners)))
        for rep in REPRESENTATIONS:
            encoded = render(policy, rep)
            assert decode(encoded, rep, actors) == policy


def test_dataset_has_canonical_semantics_fixed_queries_values_and_direct_candidates():
    worlds = generate_worlds()
    rows = build_rows(worlds)
    audit = validate(worlds, rows)
    assert audit['passed']
    assert (audit['stage1_rows'], audit['stage2_rows'], audit['stage3_rows']) == (240, 1200, 2400)
    for stage in (2, 3):
        groups = {}
        for row in rows:
            if row['stage'] != stage:
                continue
            key = row['world_id'], row['resource'], row['task']
            normalized = row['prompt'].replace(row['policy_text'], '<POLICY>')
            groups.setdefault(key, set()).add(normalized)
        assert all(len(v) == 1 for v in groups.values())
    # The Stage-1 calibration is exactly the JSON single-scope cell of Stage 2.
    index={(r['world_id'],r['task'],r['stage'],r['representation']):r for r in rows}
    for world in worlds:
        for task in ('interpretation','application'):
            one=index[(world['world_id'],task,1,'json')]
            two=index[(world['world_id'],task,2,'json')]
            assert one['policy']==two['policy']
            assert one['values']==two['values']
            assert one['candidates']==two['candidates']
            assert one['prompt']==two['prompt']
    for row in rows:
        world = next(w for w in worlds if w['world_id'] == row['world_id'])
        assert row['values'] == world['values'][row['resource']]
        assert row['correct'] in row['candidates']
        assert row['incorrect'] in row['candidates']
        assert len(row['candidates']) == 2
        assert all(not x.startswith(('A ', 'B ')) for x in row['candidates'])


def test_stage_one_actor_and_value_correct_positions_exactly_balanced():
    rows = [r for r in build_rows(generate_worlds()) if r['stage'] == 1]
    interp = [r for r in rows if r['task'] == 'interpretation']
    app = [r for r in rows if r['task'] == 'application']
    assert Counter(r['correct_actor_position'] for r in interp) == {0: 60, 1: 60}
    assert Counter(r['correct_value_position'] for r in app) == {0: 60, 1: 60}


def test_two_scope_queries_show_only_one_resource_values_and_share_policy():
    rows = [r for r in build_rows(generate_worlds()) if r['stage'] == 3]
    by_world_rep = {}
    for r in rows:
        by_world_rep.setdefault((r['world_id'],r['representation']),{})[(r['resource'],r['task'])] = r
    for items in by_world_rep.values():
        for resource in RESOURCES:
            for task in ('interpretation','application'):
                row = items[(resource,task)]
                assert set(row['values']) == set(row['actors'])
                assert set(row['policy']) == {'filename_owner','ordering_owner'}
                assert resource in row['prompt']
                assert set(row['policy']) == {'filename_owner','ordering_owner'}
                assert f'Who owns the {resource}?' in row['prompt'] if task == 'interpretation' else f'What {resource} should be used?' in row['prompt']


def test_world_cluster_bootstrap_counts_worlds_not_rendered_rows():
    result = boot([0.0, 1.0, 1.0, 0.0], seed=4, n=500)
    assert result['n_worlds'] == 4
    assert result['mean'] == 0.5
    assert 0 <= result['ci95'][0] <= result['ci95'][1] <= 1


def test_canonical_ir_is_minimal_and_rejects_extra_fields():
    single = Policy((('filename','Agent K'),))
    two = Policy((('filename','Agent K'),('ordering','Agent M')))
    assert single.to_dict() == {'resource':'filename','owner':'Agent K'}
    assert two.to_dict() == {'filename_owner':'Agent K','ordering_owner':'Agent M'}
    assert Policy.from_dict(single.to_dict()) == single
    assert Policy.from_dict(two.to_dict()) == two
    with pytest.raises(ValueError):
        Policy.from_dict({'resource':'filename','owner':'Agent K','default_owner':'Agent M'})


def test_canonicalization_candidates_include_exact_gold_ir_without_using_it_to_select():
    actors=('Agent K','Agent M')
    one=Policy((('filename','Agent M'),))
    two=Policy((('filename','Agent M'),('ordering','Agent K')))
    assert one in legal_policy_candidates(actors,('filename',))
    assert two in legal_policy_candidates(actors,RESOURCES)
    assert len(legal_policy_candidates(actors,('filename',))) == 2
    assert len(legal_policy_candidates(actors,RESOURCES)) == 4


def test_validation_rejects_changed_candidate_or_policy():
    worlds = generate_worlds()
    rows = build_rows(worlds)
    bad = copy.deepcopy(rows)
    bad[0]['correct'] = bad[0]['incorrect']
    with pytest.raises(ValueError):
        validate(worlds,bad)
    bad = copy.deepcopy(rows)
    bad[50]['policy_text'] = render(Policy(((bad[50]['resource'],next(a for a in bad[50]['actors'] if a != bad[50]['owner'])),)),bad[50]['representation'])
    with pytest.raises(ValueError):
        validate(worlds,bad)


def test_stage1_gate_has_explicit_competence_and_bias_thresholds():
    rows = []
    worlds = generate_worlds()
    # Perfect answers with full balanced metadata satisfy the frozen gate.
    stage1 = [r for r in build_rows(worlds) if r['stage'] == 1]
    for r in stage1:
        rows.append({**r,'margin':1.0,'score_audit':{'candidate_token_count_mismatch':False,'boundary_error':False},'generated_exact':True})
    passed = stage1_analysis(rows)
    assert passed['competence_gate']['passed']
    assert passed['competence_gate']['thresholds']['accuracy_min'] == .90
    flips=0
    for r in rows:
        if r['task']=='interpretation' and flips<20:
            r['margin']=-1.0; flips+=1
    failed=stage1_analysis(rows)
    assert not failed['competence_gate']['interpretation_accuracy_at_least_0_90']


def test_representation_analysis_bootstraps_120_worlds_not_600_rows():
    rows=[r for r in build_rows(generate_worlds()) if r['stage']==2]
    for r in rows:
        r.update({'margin':1.0,'selected':r['correct']})
    result=representation_analysis(rows,seed=9)
    for c in result['cells']:
        assert c['accuracy']['n_worlds']==120
        assert c['margin']['n_worlds']==120


def test_lexical_symmetry_worlds_are_fresh_balanced_and_swap_semantics_exactly():
    worlds = generate_semantic_worlds(seed=20261004)
    rows = build_lexical_rows(worlds)
    audit = validate_lexical_worlds(worlds, rows)
    assert audit['passed']
    assert len(worlds) == 180 and len(rows) == 720
    assert worlds == generate_semantic_worlds(seed=20261004)
    assert len(ACTOR_IDENTIFIERS) == 36
    assert len(ACTOR_FAMILIES) == 6
    by_key = {(r['world_id'], r['assignment'], r['task']): r for r in rows}
    for world in worlds:
        original = by_key[(world['world_id'], 'original', 'application')]
        swapped = by_key[(world['world_id'], 'swapped', 'application')]
        assert original['owner_logical'] == swapped['owner_logical']
        assert original['values_by_logical_actor'] == swapped['values_by_logical_actor']
        assert original['correct'] == swapped['correct']
        assert original['name_map']['logical_actor_1'] == swapped['name_map']['logical_actor_2']
        assert original['name_map']['logical_actor_2'] == swapped['name_map']['logical_actor_1']
        for task in ('interpretation', 'application'):
            assert by_key[(world['world_id'], 'original', task)]['prompt'] != by_key[(world['world_id'], 'swapped', task)]['prompt']
    for family in ACTOR_FAMILIES:
        group = [w for w in worlds if w['family'] == family]
        assert len(group) == 30
        assert sum(w['owner_logical'] == 'logical_actor_1' for w in group) == 15
        assert sum(w['owner_logical'] == 'logical_actor_2' for w in group) == 15


def test_lexical_symmetry_analysis_averages_oriented_margins_by_world():
    rows = []
    for row in build_lexical_rows(generate_semantic_worlds(seed=20261004)):
        rows.append({**row, 'margin': 2.0,
                     'score_audit': {'candidate_token_count_mismatch': False, 'boundary_error': False}})
    metrics = lexical_symmetry_analysis(rows, seed=18)
    assert metrics['tasks']['application']['raw_accuracy']['n_worlds'] == 180
    assert metrics['tasks']['application']['symmetrized_accuracy']['mean'] == 1.0
    assert metrics['tasks']['application']['symmetrized_margin']['mean'] == 2.0
    assert metrics['tasks']['application']['mean_identity_sensitivity']['mean'] == 0.0
    assert metrics['tasks']['application']['swap_flip_rate']['mean'] == 0.0
    assert metrics['application_identifier_family']['accuracy_range'] == 0.0
    assert metrics['gate']['passed']

    # One opposing pair has a sign flip and zero symmetrized margin; it must
    # count as a flip and as incorrect after averaging.
    target = next(r for r in rows if r['world_id'] == 'lex-world-0000'
                  and r['assignment'] == 'original' and r['task'] == 'application')
    target['margin'] = -2.0
    failed = lexical_symmetry_analysis(rows, seed=18)
    assert failed['tasks']['application']['swap_flip_rate']['mean'] == pytest.approx(1 / 180)
    assert failed['tasks']['application']['symmetrized_accuracy']['mean'] == pytest.approx(179 / 180)


def test_lexical_representation_renderers_preserve_paired_semantics_and_queries():
    worlds = generate_semantic_worlds(seed=20261004)
    rows = build_lexical_representation_rows(worlds)
    audit = validate_lexical_representation_rows(worlds, rows)
    assert audit['passed']
    assert len(rows) == 3600
    for row in rows:
        assert decode_lexical_policy(row['policy_text'], row['representation']) == row['policy']
    grouped = {}
    for row in rows:
        grouped.setdefault((row['world_id'], row['assignment'], row['task']), set()).add(
            row['prompt'].replace(row['policy_text'], '<POLICY>'))
    assert all(len(prompts) == 1 for prompts in grouped.values())


def test_lexical_representation_analysis_bootstraps_worlds_and_detects_disagreement():
    rows = []
    for row in build_lexical_representation_rows(generate_semantic_worlds(seed=20261004)):
        rows.append({**row, 'margin': 1.0,
                     'score_audit': {'candidate_token_count_mismatch': False, 'boundary_error': False}})
    metrics = lexical_representation_analysis(rows, seed=29)
    assert metrics['tasks']['application']['json']['symmetrized_accuracy']['n_worlds'] == 180
    assert metrics['categorical_disagreement_fraction']['application']['mean'] == 0.0
    assert metrics['within_world_margin_variance']['application']['mean'] == 0.0
    victim = next(r for r in rows if r['world_id'] == 'lex-world-0000'
                  and r['representation'] == 'json' and r['assignment'] == 'original'
                  and r['task'] == 'application')
    victim['margin'] = -3.0
    metrics = lexical_representation_analysis(rows, seed=29)
    assert metrics['categorical_disagreement_fraction']['application']['mean'] == pytest.approx(1 / 180)
