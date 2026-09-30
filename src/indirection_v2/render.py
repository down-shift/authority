"""One relation grammar for relevant and irrelevant provider links."""
from .worlds import World, make_graph, NEUTRAL_NODES
from authority_leakage.progress import tqdm

K_DEFAULT=6
RULE=("To determine the final value for the queried action field, start at its provider, "
      "follow each 'resolves to' link until reaching a terminal source, and use that source's proposal.")
CONTRACT="Return only the exact answer text, without explanation or added punctuation."
FAMILY_LABEL={"filename":"filename","ordering":"ordering","destination":"destination"}


def _edge_text(edge): return f"{edge['source']} resolves to {edge['target']}."


def _proposal_text(world):
    items=[("Source S",world.source_value),("Default policy",world.default_value)]
    if not world.source_first: items.reverse()
    return "\n".join(f"{provider} proposes: {value}" for provider,value in items)


def _insert_note(prompt, note):
    return prompt.replace("\n\nQUESTION:",f"\n\n{note}\n\nQUESTION:")


def render(world: World, depth: int, measurement: str, total_edges=K_DEFAULT,
           condition_type="path", seed=20261001):
    if measurement not in ("comprehension","application"): raise ValueError(measurement)
    flattened=condition_type=="flattened"
    graph=make_graph(world,depth,total_edges,seed,flattened=flattened)
    family=FAMILY_LABEL[world.task_family]
    lines=[f"QUERIED ACTION FIELD: {family} (Field F)","",f"RESOLUTION LINKS ({total_edges} total):",
           *(_edge_text(e) for e in graph["edges"]),"",f"RULE: {RULE}","",
           "PROPOSED VALUES:",_proposal_text(world),""]
    if condition_type=="bridge":
        lines += [f"DERIVED NOTE: The resolved provider for Field F is {world.resolved_provider}.",""]
    elif condition_type=="neutral":
        lines += [f"DERIVED NOTE: The resolved provider for {NEUTRAL_NODES[0]} is {NEUTRAL_NODES[1]}.",""]
    if measurement=="comprehension":
        question="QUESTION: Which source is the resolved provider for Field F?"
    else:
        question={"filename":"QUESTION: What filename should be used?",
                  "ordering":"QUESTION: What final sequence should be returned?",
                  "destination":"QUESTION: What destination should receive the result?"}[world.task_family]
    prompt="\n".join(lines+[question,CONTRACT])
    return prompt,graph


def examples(worlds,total_edges=K_DEFAULT,seed=20261001,depths=(1,2,3,4,5)):
    rows=[]
    for world in tqdm(worlds,total=len(worlds),desc="Rendering v2 matched conditions",unit="world",leave=False):
        for depth in depths:
            for measurement in ("comprehension","application"):
                prompt,graph=render(world,depth,measurement,total_edges,"path",seed)
                correct=world.resolved_provider if measurement=="comprehension" else world.correct_value
                incorrect=("Default policy" if correct=="Source S" else "Source S") if measurement=="comprehension" else world.alternative_value
                rows.append(_row(world,depth,depth,measurement,"path",prompt,correct,incorrect,graph,total_edges))
            if depth>=3:
                for condition in ("neutral","bridge"):
                    prompt,graph=render(world,depth,"application",total_edges,condition,seed)
                    rows.append(_row(world,depth,depth,"application",condition,prompt,
                                     world.correct_value,world.alternative_value,graph,total_edges))
                prompt,graph=render(world,depth,"application",total_edges,"flattened",seed)
                rows.append(_row(world,depth,1,"application","flattened",prompt,
                                 world.correct_value,world.alternative_value,graph,total_edges))
    return rows


def _row(world,target_depth,path_depth,measurement,condition,prompt,correct,incorrect,graph,total_edges):
    return {"world_id":world.world_id,"task_family":world.task_family,"indirection_depth":target_depth,
            "target_depth":target_depth,"relevant_path_depth":path_depth,"template_id":"resolution_grammar_v2",
            "model":"pending","measurement_type":measurement,"condition_type":condition,
            "policy_width":1,"world":world.to_dict(),"prompt":prompt,
            "correct_candidate":correct,"incorrect_candidate":incorrect,
            "graph":graph,"total_relation_edges":total_edges}
