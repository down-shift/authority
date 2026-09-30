"""Graph, matching, intervention, and tokenizer audits for v2."""
from collections import defaultdict
from dataclasses import fields
from .worlds import World, TERMINALS
from .render import render, K_DEFAULT
from authority_leakage.progress import tqdm

EXPECTED_CONDITIONS = {(d,m,c) for d in range(1,6) for m,c in (
    ("comprehension","path"),("application","path"))} | {
    (d,"application",c) for d in (3,4,5) for c in ("neutral","bridge","flattened")}


def _reachable(edges, start):
    adjacency=defaultdict(list)
    for edge in edges: adjacency[edge["source"]].append(edge["target"])
    stack=[(start,[])]; paths=[]
    while stack:
        node,path=stack.pop()
        if node in TERMINALS:
            paths.append((node,path)); continue
        for target in adjacency[node]:
            if target in path: raise AssertionError("Cycle in provider graph")
            stack.append((target,path+[node]))
    return paths


def audit(rows,total_edges=K_DEFAULT,seed=20261001):
    by_world=defaultdict(list); seen=set()
    for row in tqdm(rows,total=len(rows),desc="Auditing v2 graph conditions",unit="prompt",leave=False):
        raw=row["world"]; world=World(**{f.name:raw[f.name] for f in fields(World)})
        assert raw==world.to_dict()
        ident=(row["world_id"],row["target_depth"],row["measurement_type"],row["condition_type"])
        assert ident not in seen,f"duplicate row {ident}"; seen.add(ident); by_world[world.world_id].append(row)
        depth=row["target_depth"]; cond=row["condition_type"]; graph=row["graph"]
        assert row["indirection_depth"]==depth and 1<=depth<=5
        assert row["total_relation_edges"]==total_edges==graph["total_edges"]
        expected_path=1 if cond=="flattened" else depth
        assert row["relevant_path_depth"]==expected_path==graph["relevant_path_depth"]
        assert len(graph["edges"])==total_edges
        assert sum(e["relevant"] for e in graph["edges"])==expected_path
        assert graph["distractor_edge_count"]==total_edges-expected_path
        assert graph["edge_order"]==[e["edge_id"] for e in graph["edges"]]
        assert graph["relevant_edge_positions"]==[i for i,e in enumerate(graph["edges"]) if e["relevant"]]
        assert len(graph["edge_order"])==len(set(graph["edge_order"]))
        assert all(e["source"]!=e["target"] for e in graph["edges"])
        paths=_reachable(graph["edges"],"Field F")
        assert len(paths)==1 and paths[0][0]==world.resolved_provider
        assert len(paths[0][1])==expected_path
        rel_nodes={n for e in graph["edges"] if e["relevant"] for n in (e["source"],e["target"])}
        assert all(e["source"] not in rel_nodes and e["target"] not in rel_nodes
                   for e in graph["edges"] if not e["relevant"])
        assert row["correct_candidate"]==(world.resolved_provider if row["measurement_type"]=="comprehension" else world.correct_value)
        wrong=(TERMINALS[1] if world.resolved_provider==TERMINALS[0] else TERMINALS[0]) if row["measurement_type"]=="comprehension" else world.alternative_value
        assert row["incorrect_candidate"]==wrong
        assert row["prompt"]==render(world,depth,row["measurement_type"],total_edges,cond,seed)[0]
        assert sum(1 for line in row["prompt"].splitlines() if " resolves to " in line)==total_edges
    balance={}
    for wid,items in by_world.items():
        assert {(r["target_depth"],r["measurement_type"],r["condition_type"]) for r in items}==EXPECTED_CONDITIONS
        world=items[0]["world"]
        assert all(r["world"]==world for r in items)
        assert len({(r["correct_candidate"],r["incorrect_candidate"]) for r in items if r["measurement_type"]=="application"})==1
        balance.setdefault(world["task_family"],{"source_terminal":0,"default_terminal":0})[
          "source_terminal" if world["resolved_provider"]==TERMINALS[0] else "default_terminal"]+=1
        for d in (3,4,5):
            controls={r["condition_type"]:r for r in items if r["target_depth"]==d and r["measurement_type"]=="application"}
            base=controls["path"]["prompt"]; bridge=controls["bridge"]["prompt"]; neutral=controls["neutral"]["prompt"]
            bline=f"DERIVED NOTE: The resolved provider for Field F is {world['resolved_provider']}."
            nline="DERIVED NOTE: The resolved provider for Node P is Node Q."
            assert bridge.replace(bline+"\n\n","")==base
            assert neutral.replace(nline+"\n\n","")==base
            assert controls["bridge"]["graph"]==controls["neutral"]["graph"]==controls["path"]["graph"]
            graph_nodes={x for e in controls["neutral"]["graph"]["edges"] for x in (e["source"],e["target"])}
            assert not graph_nodes.intersection(("Node P","Node Q"))
    assert len(set(v["source_terminal"] for v in balance.values()))==1
    assert all(v["source_terminal"]==v["default_terminal"] for v in balance.values())
    return {"passed":True,"worlds":len(by_world),"rows":len(rows),"K":total_edges,
            "conditions_per_world":len(EXPECTED_CONDITIONS),"terminal_balance":balance}


def summarize_tokenization(rows,token_rows,max_depth_spread=2,max_bridge_neutral_delta=0):
    prompt_counts={}; candidate_pairs=defaultdict(list); length_strata=defaultdict(list)
    for item in token_rows:
        key=(item["world_id"],item["measurement_type"],item["condition_type"],item["target_depth"])
        prompt_counts[key]=item["prompt_token_count"]
        candidate_pairs[key].append(item["continuation_token_count"])
        if item["condition_type"]=="path":
            length_strata[(item["task_family"],item["measurement_type"],item["target_depth"])].append(item["prompt_token_count"])
    unequal=[{"key":k,"counts":v} for k,v in candidate_pairs.items() if len(v)!=2 or v[0]!=v[1]]
    paths=defaultdict(list)
    for (wid,task,condition,d),count in prompt_counts.items():
        if condition=="path": paths[(wid,task)].append((d,count))
    spreads={f"{wid}|{task}":max(c for _,c in vals)-min(c for _,c in vals)
             for (wid,task),vals in paths.items()}
    controls=defaultdict(dict)
    for (wid,task,condition,d),count in prompt_counts.items():
        if condition in ("bridge","neutral"): controls[(wid,d)][condition]=count
    control_mismatch=[{"world_id":w,"depth":d,"bridge":v.get("bridge"),"neutral":v.get("neutral")}
                      for (w,d),v in controls.items() if abs(v.get("bridge",0)-v.get("neutral",0))>max_bridge_neutral_delta]
    flat_mismatch=[]
    for row in rows:
        if row["condition_type"]=="flattened":
            key=(row["world_id"],row["measurement_type"],"flattened",row["target_depth"])
            base=(row["world_id"],row["measurement_type"],"path",row["target_depth"])
            if abs(prompt_counts[key]-prompt_counts[base])>max_depth_spread:
                flat_mismatch.append({"world_id":row["world_id"],"depth":row["target_depth"],
                                      "path":prompt_counts[base],"flattened":prompt_counts[key]})
    return {"candidate_pairs":len(candidate_pairs),"unequal_candidate_token_pairs":len(unequal),
       "path_prompt_token_spread_max":max(spreads.values(),default=0),
       "path_prompt_token_spread_over_tolerance":sum(v>max_depth_spread for v in spreads.values()),
       "bridge_neutral_token_mismatches":len(control_mismatch),
       "flattened_path_token_mismatches":len(flat_mismatch),
       "path_prompt_token_summary_by_family_task_depth":{
          "|".join(map(str,k)):{"n":len(v),"mean":sum(v)/len(v),"min":min(v),"max":max(v)}
          for k,v in sorted(length_strata.items())},
       "path_prompt_token_counts_by_world_task_depth":{f"{w}|{t}|{d}":n for (w,t,c,d),n in prompt_counts.items() if c=="path"},
       "unequal_pair_examples":unequal[:30],"bridge_neutral_mismatch_examples":control_mismatch[:30],
       "flattened_path_mismatch_examples":flat_mismatch[:30],
       "max_depth_spread_tolerance":max_depth_spread,"max_bridge_neutral_delta_tolerance":max_bridge_neutral_delta}
