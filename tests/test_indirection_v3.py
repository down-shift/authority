from indirection_v3.design import generate,generate_worlds,make_graph,render,validate,DEPTHS,DISTRACTORS,LAYOUTS,TERMINALS
from indirection_v3.analysis import analyze
from indirection_v3.audit import check_independence

def test_factorial_graph_depth_distractors_and_semantic_layout():
    worlds=generate_worlds(2,17);rows=[]
    for w in worlds:
      for d in DEPTHS:
       for k in DISTRACTORS:
        a=make_graph(w,d,k,"compact",17);b=make_graph(w,d,k,"dispersed",17)
        assert sum(e["relevant"] for e in a["edges"])==d
        assert sum(not e["relevant"] for e in a["edges"])==k
        assert a["terminal_provider"]==b["terminal_provider"]==w.resolved_provider
        assert sorted((e["source"],e["target"],e["relevant"]) for e in a["edges"])==sorted((e["source"],e["target"],e["relevant"]) for e in b["edges"])
        assert a["relevant_edge_positions"]==list(range(d))
        if k and d>1: assert b["relevant_span"]>=a["relevant_span"]
        for e in a["edges"]:
          if not e["relevant"]: assert e["source"].startswith("Node X") and e["target"].startswith("Node X")

def test_all_primary_cells_complete_and_provider_balanced():
    worlds=generate_worlds(2,18);rows=[]
    for w in worlds:
      for d in DEPTHS:
       for k in DISTRACTORS:
        for layout in LAYOUTS:
         for task in ("comprehension","application"):rows.append(render(w,d,k,layout,task,18))
    report=validate(rows,worlds)
    assert report["passed"] and report["primary_rows"]==len(worlds)*80

def test_validator_rejects_correlated_cell_fixture():
    worlds=generate_worlds(2,19); rows=[]
    for w in worlds:
      for d in DEPTHS:
       for k in DISTRACTORS:
        for layout in LAYOUTS:
         for task in ("comprehension","application"):rows.append(render(w,d,k,layout,task,19))
    # A fixture that overwrites k=6 at all depths with k=0 graphs must fail structural audit.
    victim=next(r for r in rows if r["depth"]==5 and r["distractor_count"]==6)
    victim["graph"]["distractor_edge_count"]=0
    try:validate(rows,worlds)
    except AssertionError:pass
    else:raise AssertionError("correlated/invalid factorial fixture was accepted")

def test_semantic_candidate_orientation_and_direct_target():
    w=generate_worlds(2,20)[0]
    r=render(w,3,2,"dispersed","application",20)
    assert r["correct_candidate"]==w.correct_value and r["incorrect_candidate"]==w.alternative_value
    assert w.resolved_provider in TERMINALS

def test_depth_has_exactly_requested_number_of_provider_edges():
    w=generate_worlds(2,21)[0]
    for d in DEPTHS:
        g=make_graph(w,d,4,"compact",21)
        assert g["relevant_path_depth"]==sum(e["relevant"] for e in g["edges"])==d

def test_distractors_exactly_requested_and_disconnected_from_path():
    w=generate_worlds(2,22)[0]
    for k in DISTRACTORS:
        g=make_graph(w,4,k,"dispersed",22); rel={x for e in g["edges"] if e["relevant"] for x in (e["source"],e["target"])}
        ds=[e for e in g["edges"] if not e["relevant"]]
        assert len(ds)==k and all(e["source"] not in rel and e["target"] not in rel for e in ds)

def test_terminal_provider_and_values_are_stable_across_factorial_cells():
    w=generate_worlds(2,23)[0]
    rows=[render(w,d,k,l,t,23) for d in DEPTHS for k in DISTRACTORS for l in LAYOUTS for t in ("application","comprehension")]
    assert {r["world"]["resolved_provider"] for r in rows}=={w.resolved_provider}
    assert {(r["world"]["source_value"],r["world"]["default_value"]) for r in rows}=={(w.source_value,w.default_value)}
    assert {r["task_family"] for r in rows}=={w.task_family}

def test_compact_and_dispersed_layouts_are_deterministic_and_semantically_equal():
    w=generate_worlds(2,24)[0]
    compact=make_graph(w,4,4,"compact",24);dispersed=make_graph(w,4,4,"dispersed",24)
    edge_semantics=lambda g:sorted((e["source"],e["target"],e["relevant"]) for e in g["edges"])
    assert edge_semantics(compact)==edge_semantics(dispersed)
    assert make_graph(w,4,4,"dispersed",24)==dispersed
    assert compact["relevant_edge_positions"]==[0,1,2,3]
    assert dispersed["relevant_span"]>compact["relevant_span"]

def test_layout_span_metrics_and_statement_positions_are_saved():
    w=generate_worlds(2,25)[0]
    row=render(w,4,4,"dispersed","application",25)
    g=row["graph"]
    assert g["first_relevant_position"]==min(g["relevant_edge_positions"])
    assert g["last_relevant_position"]==max(g["relevant_edge_positions"])
    assert g["relevant_span"]==g["last_relevant_position"]-g["first_relevant_position"]+1
    assert g["relation_block_length"]==8 and g["mean_relevant_gap"]>1

def test_full_dataset_direct_bridge_neutral_and_flattened_semantics():
    rows,audit=generate({"seed":26,"worlds_per_family":2})
    assert audit["passed"] and audit["provider_balance_exact"]
    world=rows[0]["world"];direct=next(r for r in rows if r["world_id"]==rows[0]["world_id"] and r["condition"]=="DIRECT")
    assert direct["correct_candidate"]==world["correct_value"]
    bridge=next(r for r in rows if r["world_id"]==direct["world_id"] and r["condition"]=="BRIDGE" and r["depth"]==3)
    neutral=next(r for r in rows if r["world_id"]==direct["world_id"] and r["condition"]=="NEUTRAL" and r["depth"]==3)
    flat=next(r for r in rows if r["world_id"]==direct["world_id"] and r["condition"]=="FLATTENED" and r["depth"]==3)
    assert f"Resolved provider for Field F: {world['resolved_provider']}" in bridge["prompt"]
    assert "Reference note: marker R17 is recorded." in neutral["prompt"]
    neutral_line=next(line for line in neutral["prompt"].splitlines() if line.startswith("Reference note:"))
    assert all(provider not in neutral_line for provider in TERMINALS)
    assert flat["graph"]["relevant_path_depth"]==1 and flat["graph"]["terminal_provider"]==world["resolved_provider"]

def test_disconnected_edges_cannot_reach_query_or_terminal_provider():
    w=generate_worlds(2,27)[0];g=make_graph(w,5,6,"dispersed",27)
    graph_nodes={v for e in g["edges"] for v in (e["source"],e["target"])}
    distractor_nodes={v for e in g["edges"] if not e["relevant"] for v in (e["source"],e["target"])}
    assert "Field F" not in distractor_nodes and not distractor_nodes.intersection(TERMINALS)
    assert len(graph_nodes)==len({v for e in g["edges"] for v in (e["source"],e["target"])})

def test_direct_condition_is_explicitly_not_depth_zero():
    rows,_=generate({"seed":28,"worlds_per_family":2})
    ds=[r for r in rows if r["condition"]=="DIRECT"]
    assert ds and all(r["depth"]=="DIRECT" and r["factorial_cell_id"]=="DIRECT" for r in ds)

def test_factorial_rows_have_unique_world_cell_and_semantic_candidate_pairs():
    worlds=generate_worlds(2,29);rows=[render(w,d,k,l,t,29) for w in worlds for d in DEPTHS for k in DISTRACTORS for l in LAYOUTS for t in ("application","comprehension")]
    ids=[(r["world_id"],r["depth"],r["distractor_count"],r["layout"],r["task"]) for r in rows]
    assert len(ids)==len(set(ids))
    for r in rows:
        assert r["correct_candidate"]!=r["incorrect_candidate"]
        assert r["graph"]["relevant_path_depth"]==r["depth"]

def test_design_audit_detects_intentionally_correlated_fixture():
    d=[1,1,2,2,3,3,4,4,5,5];k=[0,0,2,2,4,4,6,6,6,6];tokens=[100+x*7 for x in d]
    report=check_independence(d,k,tokens)
    assert not report["passed"] and abs(report["correlations"]["depth_tokens"])>.9

def test_analysis_bootstraps_world_rows_and_keeps_joint_queries_secondary():
    rows=[]
    for i,(c,a) in enumerate(((True,False),(False,True))):
      for task,correct in (("comprehension",c),("application",a)):
       rows.append({"world_id":f"w{i}","task_family":"filename","task":task,"condition":"BASE","depth":1,"distractor_count":0,"layout":"compact","margin":1.0 if correct else -1.0,"correct":correct})
    result=analyze(rows,seed=4,n_boot=20)
    cell=result["primary_tables"]["application|filename|1|0|compact"]
    assert cell["n_worlds"]==2 and cell["accuracy"]==.5 and len(cell["accuracy_ci95_world"])==2
    joint=result["comprehension_application_decomposition"]["filename|1|0|compact"]
    assert joint["comp_correct_app_wrong"]==1 and joint["comp_wrong_app_correct"]==1
