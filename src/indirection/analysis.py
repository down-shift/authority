"""Repeated-measures summaries and world-cluster bootstrap intervals."""
from collections import defaultdict
import numpy as np

KEYS = ("world_id", "task_family", "indirection_depth", "template_id", "model", "measurement_type")


def _ci(values, rng, n_boot):
    if not values: return [None, None]
    a = np.asarray(values, dtype=float)
    ix = rng.integers(0, len(a), size=(n_boot, len(a)))
    return [float(x) for x in np.quantile(a[ix].mean(axis=1), [.025,.975])]


def analyze(rows, seed=0, n_boot=4000):
    for row in rows:
        assert all(k in row for k in KEYS)
        assert row["measurement_type"] in ("application", "comprehension")
    worlds = defaultdict(dict)
    for r in rows:
        k = (r["task_family"], r["template_id"], r["model"], r["measurement_type"], r["indirection_depth"])
        worlds[r["world_id"]][k] = r
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for wid, measures in worlds.items():
        for k, row in measures.items():
            strata[k].append((wid, float(row["margin"]), float(bool(row["correct"]))))
    summaries = {}
    for k, vals in sorted(strata.items()):
        margins = [v[1] for v in vals]; acc = [v[2] for v in vals]
        summaries[k] = {"n_worlds": len(vals), "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)), "margin_ci95_world": _ci(margins,rng,n_boot),
            "accuracy": float(np.mean(acc)), "accuracy_ci95_world": _ci(acc,rng,n_boot)}
    paired = {}
    for family in sorted({k[0] for k in strata}):
      for template in sorted({k[1] for k in strata}):
       for model in sorted({k[2] for k in strata}):
        for task in ("application", "comprehension"):
         for depth in range(5):
          k=(family,template,model,task,depth); base=(family,template,model,task,0)
          a={wid:(m,ac) for wid,m,ac in strata.get(k,[])}; b={wid:(m,ac) for wid,m,ac in strata.get(base,[])}
          ids=sorted(set(a)&set(b))
          if not ids: continue
          dm=[a[i][0]-b[i][0] for i in ids]; da=[a[i][1]-b[i][1] for i in ids]
          ix=rng.integers(0,len(ids),size=(n_boot,len(ids)))
          paired[k]={"n_worlds":len(ids),"mean_margin_change_from_depth0":float(np.mean(dm)),
                     "margin_change_ci95_world":[float(x) for x in np.quantile(np.asarray(dm)[ix].mean(axis=1),[.025,.975])],
                     "mean_accuracy_change_from_depth0":float(np.mean(da)),
                     "accuracy_change_ci95_world":[float(x) for x in np.quantile(np.asarray(da)[ix].mean(axis=1),[.025,.975])]}
    joined = defaultdict(dict)
    for wid, measures in worlds.items():
        for k,r in measures.items():
            fam,template,model,task,depth=k
            joined[(wid,fam,template,model,depth)][task]=r
    categories=defaultdict(int); outcomes_by_world=[]
    for (wid,family,template,model,depth), pair in joined.items():
        if set(pair)=={"application","comprehension"}:
            comp=bool(pair['comprehension']['correct']); app=bool(pair['application']['correct'])
            categories[(family,template,model,depth,comp,app)]+=1
            outcomes_by_world.append({"world_id":wid,"task_family":family,"template_id":template,
                "model":model,"indirection_depth":depth,"comprehension_correct":comp,
                "application_correct":app})
    labels={(True,True):"comprehension_correct_application_correct",
            (False,False):"comprehension_wrong_application_wrong",
            (True,False):"comprehension_correct_application_wrong",
            (False,True):"comprehension_wrong_application_correct"}
    strata_categories={}
    for family,template,model,depth,_,_ in categories:
        key=(family,template,model,depth)
        strata_categories.setdefault(key,{label:0 for label in labels.values()})
    for (family,template,model,depth,comp,app),n in categories.items():
        strata_categories[(family,template,model,depth)][labels[(comp,app)]]=n
    # Aggregate only for the overview plot; inferential reporting keeps strata.
    plot_categories={str(d):{label:sum(v[label] for (fam,tmp,mdl,dep),v in strata_categories.items() if dep==d)
                    for label in labels.values()} for d in range(5)}
    return {"stratification_keys":list(KEYS),"independent_unit":"world_id",
            "summaries":{"|".join(map(str,k)):v for k,v in summaries.items()},
            "paired_change":{"|".join(map(str,k)):v for k,v in paired.items()},
            "comprehension_application_categories":{"|".join(map(str,k)):v for k,v in strata_categories.items()},
            "comprehension_application_categories_plot":plot_categories,
            "outcomes_by_world":outcomes_by_world,
            "n_prediction_rows":len(rows),"n_worlds":len(worlds)}
