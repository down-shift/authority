"""Required v3 figures with raw machine-readable plot data."""
from pathlib import Path
import csv
import numpy as np

def plot(rows,out):
 import matplotlib.pyplot as plt
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 fields=("world_id","task_family","task","condition","depth","distractor_count","layout","margin","correct","prompt_token_count","relevant_span","mean_relevant_gap")
 with (out/"plot_data.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:(r.get(k) if k not in ("relevant_span","mean_relevant_gap") else (r.get("graph") or {}).get(k)) for k in fields})
 primary=[r for r in rows if r["condition"]=="BASE"]; app=[r for r in primary if r["task"]=="application"];comp=[r for r in primary if r["task"]=="comprehension"]
 def curve(rs,metric,name,title,ylabel,filters=()):
  fig,ax=plt.subplots()
  for label,select,style in filters:
   vals=[]
   for d in range(1,6):
    z=[r for r in rs if r["depth"]==d and select(r)]
    vals.append(float(np.mean([metric(r) for r in z])) if z else np.nan)
   ax.plot(range(1,6),vals,style,marker="o",label=label)
  ax.set(xlabel="Relevant path depth (edges)",ylabel=ylabel,title=title);ax.set_xticks(range(1,6));ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/name,dpi=150);plt.close(fig)
 curve(app,lambda r:r["margin"],"application_margin_by_distractor.png","Application margin by distractor count","Semantic margin",[(f"k={k}",lambda r,k=k:r["distractor_count"]==k,"-") for k in (0,2,4,6)])
 curve(app,lambda r:r["margin"],"application_margin_by_layout.png","Application margin by layout","Semantic margin",[(x,lambda r,x=x:r["layout"]==x,"-") for x in ("compact","dispersed")])
 curve(app,lambda r:float(r["correct"]),"application_accuracy_by_family.png","Application accuracy by family","Accuracy",[(f,lambda r,f=f:r["task_family"]==f,"-") for f in ("filename","ordering","destination")])
 curve(comp,lambda r:float(r["correct"]),"comprehension_accuracy_by_depth.png","Comprehension accuracy by depth","Accuracy",[(f,lambda r,f=f:r["task_family"]==f,"-") for f in ("filename","ordering","destination")])
 fig,ax=plt.subplots()
 for d in range(1,6):
  vals=[np.mean([r["margin"] for r in app if r["depth"]==d and r["distractor_count"]==k]) for k in (0,2,4,6)];ax.plot((0,2,4,6),vals,marker="o",label=f"depth {d}")
 ax.set(xlabel="Disconnected edges",ylabel="Application margin",title="Distractor effect by depth");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"distractor_effect_by_depth.png",dpi=150);plt.close(fig)
 fig,ax=plt.subplots()
 for d in range(1,6):
  vals=[np.mean([r["margin"] for r in app if r["depth"]==d and r["layout"]==l]) for l in ("compact","dispersed")];ax.plot((0,1),vals,marker="o",label=f"depth {d}")
 ax.set_xticks((0,1),("compact","dispersed"));ax.set(ylabel="Application margin",title="Layout effect by depth");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"layout_effect_by_depth.png",dpi=150);plt.close(fig)
 direct=[r for r in rows if r["condition"]=="DIRECT"];fig,ax=plt.subplots();
 ds=[np.mean([r["correct"] for r in app if r["depth"]==d]) for d in range(1,6)];ax.plot(range(1,6),ds,marker="o",label="factorial application");ax.axhline(np.mean([r["correct"] for r in direct]),color="red",label="DIRECT");ax.set(xlabel="Depth",ylabel="Accuracy",ylim=(0,1),title="Direct reference vs factorial depths");ax.legend();fig.tight_layout();fig.savefig(out/"direct_reference.png",dpi=150);plt.close(fig)
 def controlplot(a,b,name,title):
  fig,ax=plt.subplots();
  for label,condition in ((a,a),(b,b)):
   rs=[r for r in rows if r["condition"]==condition]
   if condition=="BASE":rs=[r for r in rs if r["distractor_count"]==4 and r["layout"]=="dispersed"]
   ax.plot((3,4,5),[np.mean([r["margin"] for r in rs if r["depth"]==d]) for d in (3,4,5)],marker="o",label=label)
  ax.set(xlabel="Depth",ylabel="Application margin",title=title);ax.legend();fig.tight_layout();fig.savefig(out/name,dpi=150);plt.close(fig)
 controlplot("BRIDGE","NEUTRAL","bridge_vs_neutral.png","Bridge and neutral")
 controlplot("FLATTENED","BASE","flattened_vs_deep.png","Flattened and deep paths")
 fig,ax=plt.subplots();spans=sorted({r["graph"]["relevant_span"] for r in app});ax.plot(spans,[np.mean([r["margin"] for r in app if r["graph"]["relevant_span"]==s]) for s in spans],marker="o");ax.set(xlabel="Relevant statement span",ylabel="Application margin",title="Relevant span and performance");fig.tight_layout();fig.savefig(out/"span_performance.png",dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,(x,label) in zip(axes,(("depth","depth"),("distractor_count","distractors"),("prompt_token_count","tokens"))):
  ax.scatter([r[x] for r in app],[r["prompt_token_count"] for r in app] if x!="prompt_token_count" else [r["depth"] for r in app],s=7);ax.set_xlabel(label);ax.set_ylabel("prompt tokens" if x!="prompt_token_count" else "depth")
 fig.tight_layout();fig.savefig(out/"design_independence.png",dpi=150);plt.close(fig)
