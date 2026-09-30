"""World-clustered v2 summaries, paired contrasts, and C/A dissociation."""
from collections import defaultdict
import numpy as np

KEYS=("world_id","task_family","indirection_depth","template_id","model","measurement_type",
      "condition_type","relevant_path_depth")


def _ci(values, rng, n_boot):
    a=np.asarray(values,dtype=float)
    if not len(a): return [None,None]
    ix=rng.integers(0,len(a),size=(n_boot,len(a)))
    return [float(x) for x in np.quantile(a[ix].mean(axis=1),[.025,.975])]


def _paired(left,right,rng,n_boot):
    a={r["world_id"]:r for r in left}; b={r["world_id"]:r for r in right}
    ids=sorted(set(a)&set(b))
    if not ids: return None
    dm=[a[w]["margin"]-b[w]["margin"] for w in ids]
    da=[float(a[w]["correct"])-float(b[w]["correct"]) for w in ids]
    return {"n_worlds":len(ids),"mean_margin_difference":float(np.mean(dm)),
            "margin_ci95_world":_ci(dm,rng,n_boot),
            "mean_accuracy_difference":float(np.mean(da)),
            "accuracy_ci95_world":_ci(da,rng,n_boot)}


def analyze(rows,seed=0,n_boot=4000):
    for r in rows:
        assert all(k in r for k in KEYS)
        assert r["measurement_type"] in ("comprehension","application")
        assert r["correct"]==(r["margin"]>0)
    rng=np.random.default_rng(seed)
    strata=defaultdict(list)
    for r in rows:
        key=(r["task_family"],r["template_id"],r["model"],r["measurement_type"],
             r["condition_type"],r["indirection_depth"],r["relevant_path_depth"])
        strata[key].append(r)
    summaries={}
    for key,rs in sorted(strata.items()):
        margins=[r["margin"] for r in rs]; acc=[float(r["correct"]) for r in rs]
        prompt=[r["candidate_scores"][r["correct_candidate"]]["prompt_token_count"] for r in rs]
        summaries["|".join(map(str,key))]={"stratum":dict(zip(
            ("task_family","template_id","model","measurement_type","condition_type",
             "indirection_depth","relevant_path_depth"),key)),
            "n_worlds":len({r["world_id"] for r in rs}),
            "accuracy":float(np.mean(acc)),"accuracy_ci95_world":_ci(acc,rng,n_boot),
            "mean_margin":float(np.mean(margins)),"median_margin":float(np.median(margins)),
            "margin_ci95_world":_ci(margins,rng,n_boot),
            "mean_prompt_tokens":float(np.mean(prompt))}

    # Path-depth contrasts use depth 1 as the fixed-total-edge reference.
    depth_effects={}
    for family in sorted({r["task_family"] for r in rows}):
      for model in sorted({r["model"] for r in rows}):
       for task in ("application","comprehension"):
        for depth in range(1,6):
         left=[r for r in rows if r["task_family"]==family and r["model"]==model and
               r["measurement_type"]==task and r["condition_type"]=="path" and r["indirection_depth"]==depth]
         right=[r for r in rows if r["task_family"]==family and r["model"]==model and
               r["measurement_type"]==task and r["condition_type"]=="path" and r["indirection_depth"]==1]
         value=_paired(left,right,rng,n_boot)
         if value: depth_effects["|".join(map(str,(family,model,task,depth)))]=value

    control_effects={}
    for family in sorted({r["task_family"] for r in rows}):
      for model in sorted({r["model"] for r in rows}):
       for depth in (3,4,5):
        app=[r for r in rows if r["task_family"]==family and r["model"]==model and
             r["measurement_type"]=="application" and r["indirection_depth"]==depth]
        by={c:[r for r in app if r["condition_type"]==c] for c in ("path","neutral","bridge","flattened")}
        for label,a,b in (("bridge_vs_neutral","bridge","neutral"),
                          ("bridge_vs_path","bridge","path"),
                          ("neutral_vs_path","neutral","path"),
                          ("flattened_vs_path","flattened","path")):
            value=_paired(by[a],by[b],rng,n_boot)
            if value: control_effects["|".join(map(str,(family,model,depth,label)))]=value

    by_world=defaultdict(dict)
    for r in rows:
        if r["condition_type"]=="path":
            by_world[(r["world_id"],r["task_family"],r["model"],r["indirection_depth"])][r["measurement_type"]]=r
    outcomes=[]; dissociation=defaultdict(list)
    for (wid,fam,model,depth),pair in by_world.items():
        if set(pair)!={"application","comprehension"}: continue
        c=bool(pair["comprehension"]["correct"]); a=bool(pair["application"]["correct"])
        outcomes.append({"world_id":wid,"task_family":fam,"model":model,"indirection_depth":depth,
                         "comprehension_correct":c,"application_correct":a})
        dissociation[(fam,model,depth)].append((c,a))
    dissociation_metrics={}
    for key,vals in sorted(dissociation.items()):
        n_c=sum(c for c,_ in vals); fail=sum(c and not a for c,a in vals)
        conditional=[float(a) for c,a in vals if c]
        dissociation_metrics["|".join(map(str,key))]={
            "n_worlds":len(vals),"both_correct":sum(c and a for c,a in vals),
            "both_wrong":sum(not c and not a for c,a in vals),
            "comprehension_correct_application_wrong":fail,
            "comprehension_wrong_application_correct":sum(not c and a for c,a in vals),
            "p_application_wrong_given_comprehension_correct":(fail/n_c if n_c else None),
            "application_accuracy_given_comprehension_correct":(float(np.mean(conditional)) if conditional else None),
            "n_comprehension_correct":n_c}

    return {"stratification_keys":list(KEYS),"independent_unit":"world_id",
       "summaries":{"|".join(map(str,k)):v for k,v in summaries.items()},
       "path_depth_vs_depth1":depth_effects,"application_control_contrasts":control_effects,
       "comprehension_application_by_world":outcomes,
       "comprehension_application_dissociation":dissociation_metrics,
       "n_rows":len(rows),"n_worlds":len({r["world_id"] for r in rows})}
