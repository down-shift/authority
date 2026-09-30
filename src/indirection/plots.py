from pathlib import Path
import csv
import json
import numpy as np


def plot(rows, metrics, out):
    import matplotlib.pyplot as plt
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    # Preserve an exact machine-readable source table.
    with (out/'plot_data.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['world_id','task_family','depth','template_id','model','measurement_type','margin','correct'])
        for r in rows: w.writerow([r['world_id'],r['task_family'],r['indirection_depth'],r['template_id'],r['model'],r['measurement_type'],r['margin'],int(r['correct'])])
    json.dump(metrics, (out/'aggregate_metrics.json').open('w'), indent=2)
    groups={}
    for r in rows: groups.setdefault((r['task_family'],r['indirection_depth'],r['measurement_type']),[]).append(r)
    def curve(task, measure, split=False):
        fig,ax=plt.subplots()
        fams=sorted({k[0] for k in groups}) if split else [None]
        for fam in fams:
            ys=[]
            for d in range(5):
                vals=[float(r['correct'] if measure=='accuracy' else r['margin']) for r in rows if r['indirection_depth']==d and r['measurement_type']==task and (fam is None or r['task_family']==fam)]
                ys.append(sum(vals)/len(vals) if vals else float('nan'))
            ax.plot(range(5),ys,marker='o',label=fam or task)
        ax.set_xticks(range(5)); ax.set_xlabel('Indirection depth'); ax.set_ylabel('Accuracy' if measure=='accuracy' else 'Semantic margin (sum log probability)')
        ax.axhline(.5 if measure=='accuracy' else 0,color='grey',lw=.7); ax.legend(); fig.tight_layout()
        fig.savefig(out/f'{task}_{measure}{"_by_family" if split else ""}.png',dpi=150); plt.close(fig)
    for task in ('application','comprehension'):
        curve(task,'accuracy'); curve(task,'margin'); curve(task,'margin',True)
    fig,ax=plt.subplots()
    for task in ('application','comprehension'):
        means=[]; lows=[]; highs=[]
        for d in range(5):
            values=[]
            for r in rows:
                if r['measurement_type']==task and r['indirection_depth']==d:
                    baseline=next(x for x in rows if x['world_id']==r['world_id'] and x['measurement_type']==task and x['indirection_depth']==0 and x['model']==r['model'] and x['template_id']==r['template_id'])
                    values.append(r['margin']-baseline['margin'])
            means.append(float(np.mean(values)))
            rng=np.random.default_rng(9182+d)
            sample=np.asarray(values)[rng.integers(0,len(values),size=(4000,len(values)))].mean(axis=1)
            lows.append(float(np.quantile(sample,.025))); highs.append(float(np.quantile(sample,.975)))
        ax.plot(range(5),means,marker='o',label=task)
        ax.fill_between(range(5),lows,highs,alpha=.15)
    ax.set(xlabel='Indirection depth',ylabel='Paired mean margin change from depth 0'); ax.axhline(0,color='grey',lw=.7);ax.legend();fig.tight_layout();fig.savefig(out/'paired_depth_curve.png',dpi=150);plt.close(fig)
    cats=metrics['comprehension_application_categories_plot']; names=list(next(iter(cats.values())))
    fig,ax=plt.subplots(); bottom=[0]*5
    for name in names:
        vals=[cats[str(d)][name] for d in range(5)];ax.bar(range(5),vals,bottom=bottom,label=name);bottom=[a+b for a,b in zip(bottom,vals)]
    ax.set(xlabel='Indirection depth',ylabel='World count');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'outcome_decomposition.png',dpi=150);plt.close(fig)
