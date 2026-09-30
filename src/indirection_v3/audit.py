"""Tokenizer-aware neutral padding and factorial independence audit."""
from collections import defaultdict
import numpy as np
from authority_leakage.schemas import Message
from authority_leakage.models.continuation import continuation_encoding

FILLERS=("NOTE Z17.","Reference tag M42.","Auxiliary marker P31.","Record code R28.","Index label T63.")

def check_independence(depth,distractors,tokens,tolerance=.02):
    corr=lambda a,b:float(np.corrcoef(a,b)[0,1]) if np.std(a) and np.std(b) else 0.
    values={"depth_distractors":corr(np.asarray(depth,float),np.asarray(distractors,float)),
            "depth_tokens":corr(np.asarray(depth,float),np.asarray(tokens,float)),
            "distractors_tokens":corr(np.asarray(distractors,float),np.asarray(tokens,float))}
    return {"correlations":values,"passed":all(abs(x)<tolerance for x in values.values())}

def pad_to_world_task(rows,renderer,tokenizer):
    """Pad primary conditions within world/task to a common rendered-token count."""
    groups=defaultdict(list)
    for r in rows:
        if r["condition"]=="BASE": groups[(r["world_id"],r["task"])].append(r)
    def count(prompt):
        rendered=renderer.render([Message(role="user",content=prompt)])
        return len(tokenizer(rendered,add_special_tokens=False)["input_ids"])
    filler_counts={s:count("\n".join((s,))) for s in FILLERS}
    if any("->" in s or any(t in s for t in ("Source S","Default policy")) for s in FILLERS): raise AssertionError("unsafe filler")
    for _,items in groups.items():
        counts=[count(r["prompt"]) for r in items]
        empty_counts=[count(r["prompt"].replace("QUESTION:","NOTE Z17.\n\nQUESTION:",1)) for r in items]
        target=max(empty_counts)+32
        for row,n in zip(items,counts):
            # Qwen's tokenizer gives each repeated "marker" one token, making
            # the neutral note a deterministic fine-grained length control.
            base=max(0,target-empty_counts[items.index(row)]); chosen=None
            for repeat in (base,*[x for delta in range(1,6) for x in (base-delta,base+delta) if x>=0]):
                line="NOTE "+("marker "*repeat)+"Z17."
                candidate=row["prompt"].replace("QUESTION:",line+"\n\nQUESTION:",1)
                if count(candidate)==target: chosen=(candidate,[line]);break
            if chosen is None: raise ValueError(f"cannot exactly token-match {row['world_id']} {row['task']} with neutral marker padding")
            row["prompt"],row["neutral_padding"]=chosen
            filler="\n".join(row["neutral_padding"])
            assert "->" not in filler and " resolves to " not in filler
            assert all(term not in filler for term in ("Source S","Default policy"))
            assert all(value not in filler for value in (row["world"]["source_value"],row["world"]["default_value"]))
    # Deep controls are matched to their same-world, same-task BASE prompt.
    control_groups=defaultdict(list)
    for r in rows:
        if r["condition"] in ("NEUTRAL","BRIDGE","FLATTENED"):
            control_groups[(r["world_id"],r["task"],r["depth"])].append(r)
    for (wid,task,depth),controls in control_groups.items():
        base=next(r for r in rows if r["world_id"]==wid and r["task"]==task and r["condition"]=="BASE" and r["depth"]==depth and r["distractor_count"]==4 and r["layout"]=="dispersed")
        base_n=count(base["prompt"]);control_ns=[count(r["prompt"]) for r in controls]
        if max(control_ns)>base_n: raise ValueError(f"deep control exceeds matched factorial token budget for {wid} depth {depth}")
        for r in controls:_pad_row_exact(r,base_n,renderer,tokenizer)
    return audit_tokens(rows,renderer,tokenizer)

def _pad_row_exact(row,target,renderer,tokenizer):
    rendered=renderer.render([Message(role="user",content=row["prompt"])])
    n=len(tokenizer(rendered,add_special_tokens=False)["input_ids"])
    base=max(0,target-n-3)
    for repeat in (base,*[x for delta in range(1,10) for x in (base-delta,base+delta) if x>=0]):
        line="NOTE "+("marker "*repeat)+"Z17."
        prompt=row["prompt"].replace("QUESTION:",line+"\n\nQUESTION:",1)
        rendered=renderer.render([Message(role="user",content=prompt)])
        if len(tokenizer(rendered,add_special_tokens=False)["input_ids"])==target:
            row["prompt"]=prompt;row["neutral_padding"]=[line];return
    raise ValueError(f"could not match control token budget for {row['world_id']} {row['condition']}")

def audit_tokens(rows,renderer,tokenizer):
    groups=defaultdict(list); records=[]
    for r in rows:
        rendered=renderer.render([Message(role="user",content=r["prompt"])])
        try:
            full_encoding=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
            ids=full_encoding["input_ids"]; offsets=full_encoding.get("offset_mapping")
        except (NotImplementedError,TypeError):
            ids=tokenizer(rendered,add_special_tokens=False)["input_ids"];offsets=None
        r["prompt_token_count"]=len(ids)
        candidate_encodings=[continuation_encoding(tokenizer,rendered,c) for c in (r["correct_candidate"],r["incorrect_candidate"])]
        r["candidate_token_counts"]=[x["token_count"] for x in candidate_encodings]
        if r["condition"]=="BASE":
            positions=r["graph"]["relevant_edge_statement_indices"]
            # Includes all text before the relation block and preceding edge/newline tokens.
            pre=rendered.find("PROVIDER LINKS\n")+len("PROVIDER LINKS\n")
            offsets=[];cursor=pre
            for edge in r["graph"]["edges"]:
                text=f"{edge['source']} -> {edge['target']}"
                a=rendered.find(text,cursor); b=a+len(text); cursor=b
                offsets.append((a,b))
            rel=[offsets[i] for i in positions]
            edge_token_positions=[]
            for a,b in rel:
                if offsets is not None: edge_token_positions.append(next((i for i,(_,end) in enumerate(offsets) if end>a),len(offsets)))
                else: edge_token_positions.append(len(tokenizer(rendered[:a],add_special_tokens=False)["input_ids"]))
            r["graph"].update({"relevant_edge_token_indices":edge_token_positions,
              "first_relevant_statement_position":positions[0],"last_relevant_statement_position":positions[-1],
              "relevant_span":positions[-1]-positions[0]+1,"relation_block_length":len(offsets)})
            groups[(r["world_id"],r["task"])].append(r)
        records.append({"world_id":r["world_id"],"task_family":r["task_family"],"depth":r["depth"],"distractor_count":r["distractor_count"],"layout":r["layout"],"task":r["task"],"condition":r["condition"],"prompt_token_count":len(ids)})
    unequal=[]
    for key,rs in groups.items():
        vals=[r["prompt_token_count"] for r in rs]
        if len(set(vals))!=1: unequal.append({"key":key,"counts":sorted(set(vals))})
    primary=[r for r in rows if r["condition"]=="BASE"]
    d=np.array([r["depth"] for r in primary],float);k=np.array([r["distractor_count"] for r in primary],float);t=np.array([r["prompt_token_count"] for r in primary],float)
    independence=check_independence(d,k,t)
    spans={x:{"mean":float(np.mean([r["graph"]["relevant_span"] for r in primary if r["layout"]==x])),"max":max(r["graph"]["relevant_span"] for r in primary if r["layout"]==x)} for x in ("compact","dispersed")}
    unequal_candidates=[r["world_id"] for r in rows if len(set(r["candidate_token_counts"]))>1]
    record_map={(r["world_id"],r["task"],str(r["depth"]),r["condition"]):r["prompt_token_count"] for r in rows}
    control_mismatch=[]
    for r in rows:
        if r["condition"] in ("NEUTRAL","BRIDGE","FLATTENED"):
            base=record_map.get((r["world_id"],r["task"],str(r["depth"]),"BASE"))
            if base!=r["prompt_token_count"]: control_mismatch.append(r["world_id"])
    audit={"token_counts_equal_within_world_task":not unequal,"unequal_groups":unequal[:20],"deep_control_token_mismatches":len(control_mismatch),"candidate_length_mismatch_rows":len(unequal_candidates),"prompt_token_count_min":min(r["prompt_token_count"] for r in rows),"prompt_token_count_max":max(r["prompt_token_count"] for r in rows),"primary_correlations":independence["correlations"],"span_by_layout":spans,"records":records}
    audit["passed"]=not unequal and not control_mismatch and independence["passed"]
    return audit
