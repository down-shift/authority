"""Equal-cell summaries and world-clustered contrasts."""
from collections import defaultdict
import numpy as np

def _boot(a,rng,n):
    a=np.asarray(a,float); ix=rng.integers(0,len(a),(n,len(a)))
    return [float(x) for x in np.quantile(a[ix].mean(axis=1),[.025,.975])]

def _family_boot(values,seed,n_boot):
    grouped=defaultdict(list)
    for key,value in values.items():
      family=key[0] if isinstance(key,tuple) else key
      if isinstance(key,tuple):grouped[family].append(float(value))
      else:grouped[family].extend(list(value))
    values=grouped
    rng=np.random.default_rng(seed);means={f:float(np.mean(v)) for f,v in values.items() if len(v)}
    point=float(np.mean(list(means.values()))) if means else None;draws=[]
    if means:
      for _ in range(n_boot):draws.append(float(np.mean([np.mean(np.asarray(values[f])[rng.integers(0,len(values[f]),len(values[f]))]) for f in means])))
    return {"estimate":point,"ci95":[float(x) for x in np.quantile(draws,[.025,.975])] if draws else [None,None],"by_family":means,"n_worlds_by_family":{f:len(values[f]) for f in means}}

def _world_means(rows,task,selector,metric):
    grouped=defaultdict(list)
    for r in rows:
      if r["task"]==task and selector(r):grouped[(r["task_family"],r["world_id"])].append(float(r[metric]))
    return {k:float(np.mean(v)) for k,v in grouped.items()}

def _world_contrast(rows,task,left,right,metric="margin"):
    a=_world_means(rows,task,left,metric);b=_world_means(rows,task,right,metric)
    return {k:a[k]-b[k] for k in a.keys()&b.keys()}

def factorial_contrasts(rows,seed=0,n_boot=4000):
    """World-paired contrasts averaged over the remaining factorial cells."""
    output={}
    for task in ("application","comprehension"):
      base=lambda r:r["condition"]=="BASE"
      for metric in ("margin","correct"):
       effects={}
       for d in range(2,6):
        effects[f"depth_{d}_minus_1"]=_family_boot(_world_contrast(rows,task,lambda r,d=d:base(r) and r["depth"]==d,lambda r:base(r) and r["depth"]==1,metric),seed+d+(0 if metric=="margin" else 20),n_boot)
       effects["distractor_6_minus_0"]=_family_boot(_world_contrast(rows,task,lambda r:base(r) and r["distractor_count"]==6,lambda r:base(r) and r["distractor_count"]==0,metric),seed+30,n_boot)
       effects["dispersed_minus_compact"]=_family_boot(_world_contrast(rows,task,lambda r:base(r) and r["layout"]=="dispersed",lambda r:base(r) and r["layout"]=="compact",metric),seed+31,n_boot)
       effects["depth_5_minus_1_by_distractor_layout"]={}
       for k in (0,2,4,6):
        for layout in ("compact","dispersed"):
         effects["depth_5_minus_1_by_distractor_layout"][f"k{k}_{layout}"]=_family_boot(_world_contrast(rows,task,lambda r,k=k,l=layout:base(r) and r["depth"]==5 and r["distractor_count"]==k and r["layout"]==l,lambda r,k=k,l=layout:base(r) and r["depth"]==1 and r["distractor_count"]==k and r["layout"]==l,metric),seed+34+k+(layout=="dispersed"),n_boot)
       effects["distractor_6_minus_0_by_depth_layout"]={}
       for d in range(1,6):
        for layout in ("compact","dispersed"):
         effects["distractor_6_minus_0_by_depth_layout"][f"d{d}_{layout}"]=_family_boot(_world_contrast(rows,task,lambda r,d=d,l=layout:base(r) and r["depth"]==d and r["distractor_count"]==6 and r["layout"]==l,lambda r,d=d,l=layout:base(r) and r["depth"]==d and r["distractor_count"]==0 and r["layout"]==l,metric),seed+50+d+(layout=="dispersed"),n_boot)
       effects["layout_dispersed_minus_compact_by_depth_distractor"]={}
       for d in range(1,6):
        for k in (0,2,4,6):
         effects["layout_dispersed_minus_compact_by_depth_distractor"][f"d{d}_k{k}"]=_family_boot(_world_contrast(rows,task,lambda r,d=d,k=k:base(r) and r["depth"]==d and r["distractor_count"]==k and r["layout"]=="dispersed",lambda r,d=d,k=k:base(r) and r["depth"]==d and r["distractor_count"]==k and r["layout"]=="compact",metric),seed+70+d+k,n_boot)
       # Difference-in-differences, paired within each semantic world.
       hi=_world_contrast(rows,task,lambda r:base(r) and r["depth"]==5 and r["distractor_count"]==6,lambda r:base(r) and r["depth"]==1 and r["distractor_count"]==6,metric)
       lo=_world_contrast(rows,task,lambda r:base(r) and r["depth"]==5 and r["distractor_count"]==0,lambda r:base(r) and r["depth"]==1 and r["distractor_count"]==0,metric)
       effects["depth_by_distractor_interaction"]=_family_boot({f:[hi[(f,w)]-lo[(f,w)] for ff,w in hi if ff==f and (f,w) in lo] for f in ("filename","ordering","destination")},seed+32,n_boot)
       hd=_world_contrast(rows,task,lambda r:base(r) and r["depth"]==5 and r["layout"]=="dispersed",lambda r:base(r) and r["depth"]==1 and r["layout"]=="dispersed",metric)
       hc=_world_contrast(rows,task,lambda r:base(r) and r["depth"]==5 and r["layout"]=="compact",lambda r:base(r) and r["depth"]==1 and r["layout"]=="compact",metric)
       effects["depth_by_layout_interaction"]=_family_boot({f:[hd[(f,w)]-hc[(f,w)] for ff,w in hd if ff==f and (f,w) in hc] for f in ("filename","ordering","destination")},seed+33,n_boot)
       output[f"{task}|{metric}"]=effects
      if task=="application":
       for name,left,right in (("bridge_minus_neutral",lambda r:r["condition"]=="BRIDGE",lambda r:r["condition"]=="NEUTRAL"),("flattened_minus_deep",lambda r:r["condition"]=="FLATTENED",lambda r:r["condition"]=="BASE" and r["depth"] in (3,4,5) and r["distractor_count"]==4 and r["layout"]=="dispersed")):
        output[f"application|margin"][name]=_family_boot(_world_contrast(rows,task,left,right,"margin"),seed+40,n_boot)
    return output

def analyze(rows,seed=20261001,n_boot=2000):
    rng=np.random.default_rng(seed); prim=[r for r in rows if r["condition"]=="BASE"]
    tables={}
    for task in ("application","comprehension"):
      for fam in ("filename","ordering","destination"):
       for d in range(1,6):
        for k in (0,2,4,6):
         for layout in ("compact","dispersed"):
          rs=[r for r in prim if (r["task"],r["task_family"],r["depth"],r["distractor_count"],r["layout"])==(task,fam,d,k,layout)]
          if not rs:continue
          margins=[r["margin"] for r in rs];acc=[float(r["correct"]) for r in rs]
          key="|".join(map(str,(task,fam,d,k,layout)))
          tables[key]={"n_worlds":len(rs),"accuracy":float(np.mean(acc)),"mean_margin":float(np.mean(margins)),"median_margin":float(np.median(margins)),"accuracy_ci95_world":_boot(acc,rng,n_boot),"margin_ci95_world":_boot(margins,rng,n_boot)}
    def contrast(a,b,label):
      x={r["world_id"]:r for r in a};y={r["world_id"]:r for r in b};ids=sorted(x.keys()&y.keys())
      if not ids:return None
      dm=[x[i]["margin"]-y[i]["margin"] for i in ids];da=[float(x[i]["correct"])-float(y[i]["correct"]) for i in ids]
      return {"label":label,"n_worlds":len(ids),"mean_margin_difference":float(np.mean(dm)),"margin_ci95_world":_boot(dm,rng,n_boot),"accuracy_difference":float(np.mean(da)),"accuracy_ci95_world":_boot(da,rng,n_boot)}
    contrasts={}
    for task in ("application","comprehension"):
      for fam in ("filename","ordering","destination","ALL_EQUAL_FAMILY"):
       families=(fam,) if fam!="ALL_EQUAL_FAMILY" else ("filename","ordering","destination")
       for k in (0,2,4,6):
        for layout in ("compact","dispersed"):
         for d in range(2,6):
          left=[r for r in prim if r["task"]==task and r["task_family"] in families and r["depth"]==d and r["distractor_count"]==k and r["layout"]==layout]
          right=[r for r in prim if r["task"]==task and r["task_family"] in families and r["depth"]==1 and r["distractor_count"]==k and r["layout"]==layout]
          # Aggregate rows are paired within family to prevent family composition weighting.
          pieces=[contrast([r for r in left if r["task_family"]==f],[r for r in right if r["task_family"]==f],f) for f in families]
          pieces=[p for p in pieces if p]
          if pieces:
            contrasts[f"depth|{task}|{fam}|d{d}|k{k}|{layout}"]={"mean_margin_difference":float(np.mean([p["mean_margin_difference"] for p in pieces])),"family_contrasts":pieces}
       # k and layout effects at fixed depth.
       for d in range(1,6):
        for layout in ("compact","dispersed"):
         a=[r for r in prim if r["task"]==task and r["task_family"] in families and r["depth"]==d and r["distractor_count"]==6 and r["layout"]==layout]
         b=[r for r in prim if r["task"]==task and r["task_family"] in families and r["depth"]==d and r["distractor_count"]==0 and r["layout"]==layout]
         contrasts[f"distractor|{task}|{fam}|d{d}|{layout}"]=contrast(a,b,"k6-k0")
        for k in (0,2,4,6):
         a=[r for r in prim if r["task"]==task and r["task_family"] in families and r["depth"]==d and r["distractor_count"]==k and r["layout"]=="dispersed"]
         b=[r for r in prim if r["task"]==task and r["task_family"] in families and r["depth"]==d and r["distractor_count"]==k and r["layout"]=="compact"]
         contrasts[f"layout|{task}|{fam}|d{d}|k{k}"]=contrast(a,b,"dispersed-compact")
    extra={}
    for task in ("application","comprehension"):
      for fam in ("filename","ordering","destination"):
       for d in range(1,6):
        a=[r for r in prim if r["task"]==task and r["task_family"]==fam and r["depth"]==d and r["distractor_count"]==4 and r["layout"]=="dispersed"]
        b=[r for r in prim if r["task"]==task and r["task_family"]==fam and r["depth"]==1 and r["distractor_count"]==4 and r["layout"]=="dispersed"]
        extra[f"depth5_minus_depth1|{task}|{fam}|k4|dispersed|d{d}"]=contrast(a,b,"depth-d1")
    control_rows=[r for r in rows if r["condition"] in ("BASE","NEUTRAL","BRIDGE","FLATTENED","DIRECT") and r["task"]=="application"]
    for fam in ("filename","ordering","destination"):
      for d in (3,4,5):
       by={c:[r for r in control_rows if r["task_family"]==fam and r["condition"]==c and (r["depth"]==d if c!="DIRECT" else False)] for c in ("BASE","NEUTRAL","BRIDGE","FLATTENED")}
       # Primary BASE match uses depth d, k=4 and dispersed layout.
       by["BASE"]=[r for r in prim if r["task"]=="application" and r["task_family"]==fam and r["depth"]==d and r["distractor_count"]==4 and r["layout"]=="dispersed"]
       for label,left,right in (("bridge_minus_neutral","BRIDGE","NEUTRAL"),("bridge_minus_base","BRIDGE","BASE"),("neutral_minus_base","NEUTRAL","BASE"),("flattened_minus_deep","FLATTENED","BASE")):
        extra[f"{label}|{fam}|d{d}"]=contrast(by[left],by[right],label)
      direct=[r for r in rows if r["condition"]=="DIRECT" and r["task_family"]==fam]
      extra[f"direct_accuracy|{fam}"]={"n_worlds":len(direct),"accuracy":float(np.mean([r["correct"] for r in direct])) if direct else None,"mean_margin":float(np.mean([r["margin"] for r in direct])) if direct else None}
    joint=defaultdict(dict)
    for r in prim:
      joint[(r["world_id"],r["task_family"],r["depth"],r["distractor_count"],r["layout"])][r["task"]]=bool(r["correct"])
    decomposed=defaultdict(lambda:defaultdict(int))
    for (wid,fam,d,k,layout),outcome in joint.items():
      if set(outcome)!={"application","comprehension"}:continue
      c=outcome["comprehension"];a=outcome["application"]
      label=("comp_correct_app_correct" if c and a else "comp_correct_app_wrong" if c else "comp_wrong_app_correct" if a else "comp_wrong_app_wrong")
      decomposed["|".join(map(str,(fam,d,k,layout)))][label]+=1
    return {"independent_unit":"world_id","primary_tables":tables,"contrasts":contrasts,"factorial_aggregate_contrasts":factorial_contrasts(rows,seed,n_boot),"control_and_direct":extra,"comprehension_application_decomposition":{k:dict(v) for k,v in decomposed.items()},"n_primary_rows":len(prim)}
