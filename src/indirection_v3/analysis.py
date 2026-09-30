"""Equal-cell summaries and world-clustered contrasts."""
from collections import defaultdict
import numpy as np

def _boot(a,rng,n):
    a=np.asarray(a,float); ix=rng.integers(0,len(a),(n,len(a)))
    return [float(x) for x in np.quantile(a[ix].mean(axis=1),[.025,.975])]

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
    return {"independent_unit":"world_id","primary_tables":tables,"contrasts":contrasts,"control_and_direct":extra,"comprehension_application_decomposition":{k:dict(v) for k,v in decomposed.items()},"n_primary_rows":len(prim)}
