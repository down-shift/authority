from collections import Counter,defaultdict
from indirection_v2.worlds import generate_worlds,make_graph,TERMINALS
from indirection_v2.render import examples,render
from indirection_v2.validation import audit,_reachable


def test_deterministic_worlds_and_exact_terminal_balance():
    a=generate_worlds(20,111);b=generate_worlds(20,111)
    assert [w.to_dict() for w in a]==[w.to_dict() for w in b]
    for family in ('filename','ordering','destination'):
        ws=[w for w in a if w.task_family==family]
        assert Counter(w.resolved_provider for w in ws)=={TERMINALS[0]:10,TERMINALS[1]:10}


def test_fixed_graph_size_path_depth_and_disconnected_distractors():
    worlds=generate_worlds(2,7)
    for world in worlds:
        for depth in range(1,6):
            graph=make_graph(world,depth,6,7)
            assert len(graph['edges'])==6
            assert graph['relevant_path_depth']==depth
            assert graph['distractor_edge_count']==6-depth
            assert len(_reachable(graph['edges'],'Field F'))==1
            assert _reachable(graph['edges'],'Field F')[0][0]==world.resolved_provider
            assert len(_reachable(graph['edges'],'Field F')[0][1])==depth
            relevant={n for e in graph['edges'] if e['relevant'] for n in (e['source'],e['target'])}
            assert all(e['source'] not in relevant and e['target'] not in relevant
                       for e in graph['edges'] if not e['relevant'])


def test_full_dataset_matching_and_total_edge_counts():
    worlds=generate_worlds(2,9);rows=examples(worlds,6,9)
    report=audit(rows,6,9)
    assert report['passed'] and report['rows']==len(worlds)*19
    for wid in {r['world_id'] for r in rows}:
        rs=[r for r in rows if r['world_id']==wid]
        assert {r['condition_type'] for r in rs}>={'path','neutral','bridge','flattened'}
        for r in rs:
            assert r['total_relation_edges']==6
            assert len(r['graph']['edges'])==6
            assert sum(' resolves to ' in line for line in r['prompt'].splitlines())==6
        app=[r for r in rs if r['measurement_type']=='application']
        assert len({(r['correct_candidate'],r['incorrect_candidate']) for r in app})==1


def test_homogeneous_relation_grammar_and_randomized_deterministic_order():
    w=generate_worlds(2,33)[0]
    rows=[make_graph(w,d,6,33) for d in range(1,6)]
    assert len({tuple(g['edge_order']) for g in rows})>1
    assert all(g['edge_order']==[e['edge_id'] for e in g['edges']] for g in rows)
    assert all(' resolves to ' in e['source'] + ' resolves to ' + e['target'] for g in rows for e in g['edges'])
    assert all(len(g['relevant_edge_positions'])==d for d,g in enumerate(rows,1))


def test_graph_structure_and_statement_order_are_family_independent():
    worlds=generate_worlds(4,91)
    by_family={f:{w.world_id.rsplit('-',1)[-1]:w for w in worlds if w.task_family==f}
               for f in ('filename','ordering','destination')}
    for i in range(4):
        graphs=[make_graph(by_family[f][f'{i:04d}'],4,6,91) for f in by_family]
        def abstract(g):
            return [(e['source'], 'TERMINAL' if e['target'] in TERMINALS else e['target'],e['relevant'])
                    for e in g['edges']]
        assert abstract(graphs[0])==abstract(graphs[1])==abstract(graphs[2])


def test_bridge_and_neutral_are_single_note_insertions_and_flattened_preserves_action():
    w=generate_worlds(2,51)[0]
    for d in (3,4,5):
        base,_=render(w,d,'application',6,'path',51)
        bridge,bg=render(w,d,'application',6,'bridge',51)
        neutral,ng=render(w,d,'application',6,'neutral',51)
        bline=f"DERIVED NOTE: The resolved provider for Field F is {w.resolved_provider}."
        nline="DERIVED NOTE: The resolved provider for Node P is Node Q."
        assert bridge.replace(bline+'\n\n','')==base
        assert neutral.replace(nline+'\n\n','')==base
        assert bg==ng
        flat,fg=render(w,d,'application',6,'flattened',51)
        assert fg['total_edges']==6 and fg['relevant_path_depth']==1
        assert sum(e['relevant'] for e in fg['edges'])==1
        assert w.correct_value in (w.source_value,w.default_value)
        assert w.correct_value==w.source_value if w.resolved_provider==TERMINALS[0] else w.correct_value==w.default_value


def test_task_families_have_all_depths_and_controls():
    rows=examples(generate_worlds(2,5),6,5)
    cells={(r['task_family'],r['target_depth'],r['measurement_type'],r['condition_type']) for r in rows}
    for family in ('filename','ordering','destination'):
        for d in range(1,6):
            assert (family,d,'comprehension','path') in cells
            assert (family,d,'application','path') in cells
        for d in (3,4,5):
            assert all((family,d,'application',c) in cells for c in ('bridge','neutral','flattened'))
