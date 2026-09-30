from indirection.worlds import generate_worlds
from indirection.render import examples, DEPTH_RENDERERS
from indirection.validation import audit

def test_seed_stable_semantics_and_balanced_source():
    a=generate_worlds(20,41); b=generate_worlds(20,41)
    assert [w.to_dict() for w in a]==[w.to_dict() for w in b]
    for fam in ('filename','ordering','destination'):
        ws=[w for w in a if w.task_family==fam]
        assert sum(w.owner==w.source for w in ws)==10
        assert sum(w.source_first for w in ws)==10

def test_all_depths_match_semantics_and_contract():
    rows=examples(generate_worlds(4,2)); assert audit(rows)['passed']
    for wid in {r['world_id'] for r in rows}:
        rs=[r for r in rows if r['world_id']==wid and r['measurement_type']=='application']
        assert len({r['correct_candidate'] for r in rs})==1
        assert len({r['incorrect_candidate'] for r in rs})==1
        assert len({r['prompt'].split('Return only')[1] for r in rs})==1

def test_each_depth_is_actual_distinct_rule_structure():
    w=generate_worlds(2,3)[0]
    prompts=[DEPTH_RENDERERS[d](w,'filename') for d in range(5)]
    assert len(set(prompts))==5
    assert 'Role ' in prompts[3] and 'owns field' in prompts[4]

def test_template_ids_are_real_and_direct_value_is_ground_truth():
    from indirection.render import TEMPLATES, render
    assert set(TEMPLATES)=={'canonical_v1'}
    for w in generate_worlds(2,9):
        prompt=render(w,0,'application')
        assert 'Final ' in prompt and f'Use {w.correct_value}.' in prompt

def test_audit_rejects_semantic_mutation_across_depths():
    import pytest
    rows=examples(generate_worlds(2,10))
    row=next(r for r in rows if r['measurement_type']=='application' and r['indirection_depth']==4)
    row['correct_candidate']='mutated'
    with pytest.raises(AssertionError): audit(rows)

def test_dataset_hash_is_stable():
    from indirection.worlds import dataset_sha256
    a=examples(generate_worlds(4,112));b=examples(generate_worlds(4,112))
    assert dataset_sha256(a)==dataset_sha256(b)
