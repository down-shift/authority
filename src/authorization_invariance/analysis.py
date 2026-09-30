"""Paired estimates with the entire synthetic world as bootstrap cluster."""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from authority_leakage.statistics import bootstrap_ci
from .design import REPRESENTATIONS


def estimate(values, seed, n_boot):
    values = list(values)
    return {'mean': float(np.mean(values)) if values else None,
            'ci95': bootstrap_ci(values, seed, n_boot), 'n_worlds': len(values)}


def analyze(rows, seed=0, n_boot=2000):
    df = pd.DataFrame(rows)
    keys = ['world_id', 'scope_count', 'field', 'representation', 'task', 'arm']
    if df.duplicated(keys).any() or df.margin.isna().any() or not np.isfinite(df.margin).all():
        raise ValueError('Duplicate or invalid predictions')
    expected = len(df.world_id.unique()) * 80
    if len(df) != expected:
        raise ValueError('Incomplete predictions')
    df['accuracy'] = (df.margin > 0).astype(float)  # Exact ties are incorrect; report separately.
    cells = []
    for key, g in df.groupby(['scope_count', 'task', 'arm', 'representation'], sort=True):
        cells.append(dict(zip(['scope_count', 'task', 'arm', 'representation'], key),
            accuracy=estimate(g.groupby('world_id').accuracy.mean(), seed, n_boot),
            margin=estimate(g.groupby('world_id').margin.mean(), seed, n_boot),
            ties=int((g.margin == 0).sum()), n_rows=len(g)))
    families = []
    for key, g in df.groupby(['scope_count', 'field', 'task', 'arm', 'representation'], sort=True):
        families.append(dict(zip(['scope_count', 'field', 'task', 'arm', 'representation'], key),
            accuracy=estimate(g.groupby('world_id').accuracy.mean(), seed, n_boot),
            margin=estimate(g.groupby('world_id').margin.mean(), seed, n_boot)))
    disagreements, pairs = [], []
    for key, g in df.groupby(['scope_count', 'task', 'arm'], sort=True):
        margins = g.pivot(index=['world_id', 'field'], columns='representation', values='margin')
        choices = g.pivot(index=['world_id', 'field'], columns='representation', values='selected')
        if margins.isna().any().any() or set(margins.columns) != set(REPRESENTATIONS):
            raise ValueError('Incomplete representation group')
        measures = pd.DataFrame({'any_disagreement': choices.nunique(axis=1).gt(1).astype(float),
            'margin_range': margins.max(axis=1) - margins.min(axis=1)}, index=margins.index)
        disagree = dict(zip(['scope_count', 'task', 'arm'], key))
        disagree.update({col: estimate(measures[col].groupby('world_id').mean(), seed, n_boot) for col in measures})
        stability = {}
        for rep in REPRESENTATIONS:
            rate = sum(choices[rep].ne(choices[other]).astype(float) for other in REPRESENTATIONS if other != rep) / 4
            stability[rep] = estimate(rate.groupby('world_id').mean(), seed, n_boot)
        disagree['representation_pairwise_disagreement'] = stability
        disagreements.append(disagree)
        for a, b in combinations(REPRESENTATIONS, 2):
            diff = margins[b] - margins[a]
            rate = choices[b].ne(choices[a]).astype(float)
            pairs.append(dict(zip(['scope_count', 'task', 'arm'], key), contrast=f'{b} minus {a}',
                margin_effect=estimate(diff.groupby('world_id').mean(), seed, n_boot),
                disagreement=estimate(rate.groupby('world_id').mean(), seed, n_boot)))
    dissociations = []
    for key, g in df.groupby(['scope_count', 'arm', 'representation']):
        pivot = g.pivot(index=['world_id', 'field'], columns='task', values='accuracy')
        measures = {'disagreement': pivot.interpretation.ne(pivot.application).astype(float),
                    'interpretation_correct_application_wrong': ((pivot.interpretation == 1) & (pivot.application == 0)).astype(float),
                    'interpretation_wrong_application_correct': ((pivot.interpretation == 0) & (pivot.application == 1)).astype(float)}
        dissociations.append(dict(zip(['scope_count', 'arm', 'representation'], key),
            **{k: estimate(v.groupby('world_id').mean(), seed, n_boot) for k, v in measures.items()}))
    contrasts = []
    for task in ('interpretation', 'application'):
        for rep in REPRESENTATIONS:
            g = df[(df.task == task) & (df.representation == rep)]
            for measure in ('margin', 'accuracy'):
                pivot = g.pivot(index=['world_id', 'field'], columns=['arm', 'scope_count'], values=measure)
                for arm in ('raw', 'canonicalized'):
                    diff = pivot[(arm, 2)] - pivot[(arm, 1)]
                    contrasts.append({'task': task, 'representation': rep, 'measure': measure,
                        'contrast': f'two_minus_single/{arm}', **estimate(diff.groupby('world_id').mean(), seed, n_boot)})
                for scopes in (1, 2):
                    diff = pivot[('canonicalized', scopes)] - pivot[('raw', scopes)]
                    contrasts.append({'task': task, 'representation': rep, 'measure': measure,
                        'contrast': f'canonicalized_minus_raw/{scopes}', **estimate(diff.groupby('world_id').mean(), seed, n_boot)})
    recovery = []
    for key, g in df.groupby(['scope_count', 'task']):
        flags = {}
        for arm, h in g.groupby('arm'):
            p = h.pivot(index=['world_id', 'field'], columns='representation', values='selected')
            flags[arm] = p.nunique(axis=1).gt(1).astype(float)
        diff = flags['raw'] - flags['canonicalized']
        recovery.append(dict(zip(['scope_count', 'task'], key),
            reduction=estimate(diff.groupby('world_id').mean(), seed, n_boot)))
    return {'worlds': df.world_id.nunique(), 'rows': len(df), 'independent_unit': 'world',
        'accuracy_rule': 'correct semantic candidate has strictly positive sum-logprob margin; ties incorrect',
        'cells': cells, 'family_cells': families, 'within_world_disagreement': disagreements,
        'pairwise_representation_effects': pairs, 'comprehension_application_disagreement': dissociations,
        'paired_scope_and_mitigation_effects': contrasts, 'canonicalization_disagreement_recovery': recovery}


def report(metrics, conversions):
    lines = ['# Authorization representation invariance: Qwen3-4B pilot', '',
        f"{metrics['worlds']} independent worlds; {metrics['rows']} answer rows. Sum log probability of direct semantic candidates is primary. Accuracy uses margin > 0. Intervals resample whole worlds (both queried fields together).", '',
        'Prompts were fixed before inference; no representation-specific tuning. Semantic equivalence is validated independently of model behavior. Disagreement is the outcome, never a manipulation-failure gate.', '',
        'Canonicalization uses model likelihood to choose among all legal canonical policies for the named scopes (2 or 4 candidates); no ground-truth authorization enters selection. The chosen policy alone replaces the raw policy before answering. This constrained semantic MAP converter is a mitigation experiment, not unconstrained JSON generation.', '',
        '| Scopes | Task | Arm | Representation | Accuracy (95% CI) | Margin (95% CI) |',
        '|---|---|---|---|---|---|']
    def fmt(e):
        return f"{e['mean']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
    for c in metrics['cells']:
        lines.append(f"| {c['scope_count']} | {c['task']} | {c['arm']} | {c['representation']} | {fmt(c['accuracy'])} | {fmt(c['margin'])} |")
    lines += ['', '## Representation disagreement and stability', '',
              '| Scopes | Task | Arm | Any disagreement (95% CI) | Most stable | Least stable |',
              '|---|---|---|---|---|---|']
    for d in metrics['within_world_disagreement']:
        s = d['representation_pairwise_disagreement']
        low, high = min(v['mean'] for v in s.values()), max(v['mean'] for v in s.values())
        best = ', '.join(k for k, v in s.items() if v['mean'] == low)
        worst = ', '.join(k for k, v in s.items() if v['mean'] == high)
        lines.append(f"| {d['scope_count']} | {d['task']} | {d['arm']} | {fmt(d['any_disagreement'])} | {best} | {worst} |")
    lines += ['', 'Stability is mean disagreement with the other four representations; tied rankings are retained. Agreement can include unanimous incorrect answers.', '',
              '## Canonicalization recovery', '']
    for r in metrics['canonicalization_disagreement_recovery']:
        lines.append(f"- {r['scope_count']} scopes / {r['task']}: raw minus canonicalized disagreement {fmt(r['reduction'])}.")
    lines += ['', '## Two-scope change (two minus matched single)', '',
              '| Task | Representation | Raw accuracy change | Raw margin change |', '|---|---|---|---|']
    for task in ('interpretation', 'application'):
        for rep in REPRESENTATIONS:
            vals = {r['measure']: r for r in metrics['paired_scope_and_mitigation_effects'] if r['task'] == task and r['representation'] == rep and r['contrast'] == 'two_minus_single/raw'}
            lines.append(f"| {task} | {rep} | {fmt(vals['accuracy'])} | {fmt(vals['margin'])} |")
    lines += ['', '## Conversion accuracy', '']
    conv = pd.DataFrame(conversions)
    for (scopes, rep), g in conv.groupby(['scope_count', 'representation']):
        lines.append(f"- {scopes} scopes / {rep}: exact policy {g.exact.mean():.3f}; scope authorization {g.scope_accuracy.mean():.3f}.")
    lines += ['', 'Family-specific estimates, all ten paired representation effects, comprehension/application dissociations, and world-cluster bootstrap intervals are in metrics.json. Raw candidate token IDs/counts, boundary audit, sum/mean log probabilities, model provenance, and prompts are retained.', '',
        'This small synthetic single-model pilot measures these fixed renderers, including their wording and token lengths. It does not establish an internal mechanism or a universal representation ranking. Two-scope change includes additional policy/value context. No LLM judge or multi-model sweep is used.', '']
    return '\n'.join(lines)


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df['accuracy'] = (df.margin > 0).astype(float)
    df.to_csv(output / 'plot_data.csv', index=False)
    for measure in ('accuracy', 'margin'):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for i, scopes in enumerate((1, 2)):
            for j, task in enumerate(('interpretation', 'application')):
                ax = axes[i, j]
                for arm in ('raw', 'canonicalized'):
                    g = df[(df.scope_count == scopes) & (df.task == task) & (df.arm == arm)]
                    means = g.groupby('representation')[measure].mean().reindex(REPRESENTATIONS)
                    ax.plot(range(5), means, marker='o', label=arm)
                ax.set_xticks(range(5), REPRESENTATIONS, rotation=25, ha='right')
                ax.set_title(f'{scopes} scopes: {task}')
                ax.set_ylabel(measure)
                ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{measure}.png', dpi=160)
        plt.close(fig)
