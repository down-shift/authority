from indirection.worlds import generate_worlds
from indirection.render import examples
from indirection.validation import audit
from indirection.analysis import analyze

def test_world_cluster_pairs_and_categories():
    rows=examples(generate_worlds(2,7)); audit(rows)
    for r in rows:
        r.update(model='toy',margin=1.0,correct=True)
    m=analyze(rows,n_boot=50)
    assert m['n_worlds']==6 and m['n_prediction_rows']==60
    assert all(x['mean_margin_change_from_depth0']==0 for x in m['paired_change'].values())
    assert len(m['outcomes_by_world'])==30
    assert m['comprehension_application_categories_plot']['4']['comprehension_correct_application_correct']==6
    assert all(len(key.split('|'))==4 and key.split('|')[-1] in {'0','1','2','3','4'}
               for key in m['comprehension_application_categories'])
