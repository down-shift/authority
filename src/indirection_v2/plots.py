"""Minimal figures with full per-example source data in CSV/JSON."""
from pathlib import Path
import csv,json
import numpy as np

def plot(rows,metrics,out):
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fields=["world_id","task_family","model","measurement_type","indirection_depth",
            "relevant_path_depth","condition_type","margin","correct","prompt_token_count"]
    with (out/"plot_data.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            w.writerow({**{k:r.get(k) for k in fields},
                        "prompt_token_count":r["candidate_scores"][r["correct_candidate"]]["prompt_token_count"]})
    (out/"aggregate_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")

    path=[r for r in rows if r["condition_type"]=="path"]
    families=sorted({r["task_family"] for r in path})
    def family_curve(task,value,filename,ylabel):
        fig,ax=plt.subplots()
        for fam in families:
            means=[]
            for d in range(1,6):
                rs=[r for r in path if r["task_family"]==fam and r["measurement_type"]==task and r["indirection_depth"]==d]
                means.append(float(np.mean([value(r) for r in rs])))
            ax.plot(range(1,6),means,marker="o",label=fam)
        ax.set(xlabel="Relevant path depth (edges)",ylabel=ylabel,title=f"{task.title()} by family")
        ax.set_xticks(range(1,6));ax.legend();fig.tight_layout();fig.savefig(out/filename,dpi=150);plt.close(fig)
    family_curve("application",lambda r:float(r["correct"]),"application_accuracy_by_family.png","Accuracy")
    family_curve("comprehension",lambda r:float(r["correct"]),"comprehension_accuracy_by_family.png","Accuracy")
    family_curve("application",lambda r:r["margin"],"application_margin_by_family.png","Semantic margin (sum log probability)")
    family_curve("comprehension",lambda r:r["margin"],"comprehension_margin_by_family.png","Semantic margin (sum log probability)")

    fig,ax=plt.subplots()
    for fam in families:
      for task,style in (("application","-o"),("comprehension","--s")):
        means=[np.mean([r["margin"] for r in path if r["task_family"]==fam and r["measurement_type"]==task and r["indirection_depth"]==d]) for d in range(1,6)]
        ax.plot(range(1,6),means,style,label=f"{fam} {task}")
    ax.axhline(0,color="grey",lw=.7);ax.set(xlabel="Relevant path depth (edges)",ylabel="Mean semantic margin");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"depth_curves_by_family.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots()
    for fam in families:
        vals=[np.mean([r["candidate_scores"][r["correct_candidate"]]["prompt_token_count"] for r in path if r["task_family"]==fam and r["measurement_type"]=="application" and r["indirection_depth"]==d]) for d in range(1,6)]
        ax.plot(range(1,6),vals,marker="o",label=fam)
    ax.set(xlabel="Relevant path depth (edges)",ylabel="Mean rendered prompt tokens",title="Prompt length by path depth");ax.set_xticks(range(1,6));ax.legend();fig.tight_layout();fig.savefig(out/"prompt_tokens_by_depth.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots()
    for fam in families:
        vals=[]
        for d in range(1,6):
            metric=metrics["comprehension_application_dissociation"][f"{fam}|{rows[0]['model']}|{d}"]
            vals.append(metric["application_accuracy_given_comprehension_correct"])
        ax.plot(range(1,6),vals,marker="o",label=fam)
    ax.set(xlabel="Relevant path depth (edges)",ylabel="P(application correct | comprehension correct)",ylim=(0,1));ax.set_xticks(range(1,6));ax.legend();fig.tight_layout();fig.savefig(out/"application_given_comprehension.png",dpi=150);plt.close(fig)

    names=("both_correct","both_wrong","comprehension_correct_application_wrong","comprehension_wrong_application_correct")
    fig,ax=plt.subplots(); bottom=np.zeros(5)
    for name in names:
        vals=[]
        for d in range(1,6):
            vals.append(sum(m[name] for k,m in metrics["comprehension_application_dissociation"].items() if int(k.split("|")[-1])==d))
        ax.bar(range(1,6),vals,bottom=bottom,label=name);bottom+=np.asarray(vals)
    ax.set(xlabel="Relevant path depth (edges)",ylabel="World count");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"comprehension_application_categories.png",dpi=150);plt.close(fig)

    deep=[r for r in rows if r["measurement_type"]=="application" and r["condition_type"] in ("path","neutral","bridge") and r["indirection_depth"]>=3]
    fig,ax=plt.subplots()
    for condition in ("path","neutral","bridge"):
        vals=[np.mean([r["margin"] for r in deep if r["condition_type"]==condition and r["indirection_depth"]==d]) for d in (3,4,5)]
        ax.plot((3,4,5),vals,marker="o",label=condition)
    ax.axhline(0,color="grey",lw=.7);ax.set(xlabel="Target depth",ylabel="Mean application margin");ax.legend();fig.tight_layout();fig.savefig(out/"bridge_neutral_path.png",dpi=150);plt.close(fig)

    flat=[r for r in rows if r["measurement_type"]=="application" and r["indirection_depth"]>=3 and r["condition_type"] in ("path","flattened")]
    fig,ax=plt.subplots()
    for condition in ("path","flattened"):
        vals=[np.mean([r["correct"] for r in flat if r["condition_type"]==condition and r["indirection_depth"]==d]) for d in (3,4,5)]
        ax.plot((3,4,5),vals,marker="o",label=condition)
    ax.set(xlabel="Target depth",ylabel="Application accuracy",ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/"flattened_vs_deep.png",dpi=150);plt.close(fig)
