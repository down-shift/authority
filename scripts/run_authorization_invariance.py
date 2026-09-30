#!/usr/bin/env python3
"""Frozen single-model authorization representation experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from authority_leakage.inference import write_jsonl, write_run_status
from authority_leakage.models.hf import HFAdapter
from authority_leakage.schemas import Message
from authorization_invariance.design import Policy, generate, validate, render, canonical_candidates, answer_prompt
from authorization_invariance.analysis import analyze, report, plot


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/authorization_invariance_pilot.yaml'))
    parser.add_argument('--model', default='qwen3_4b', choices=['qwen3_4b'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--dataset-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cpu-float32', action='store_true', help='Avoid emulated BF16 CPU operations on ARM')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['experiment'] != 'authorization_representation_invariance':
        parser.error('Wrong experiment')
    worlds, rows = generate(cfg['worlds_per_pair'], cfg['seed'])
    audit = validate(worlds, rows)
    run = args.resume or args.output_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_authorization_invariance_qwen3_4b')
    if args.resume:
        if json.loads((run / 'config.json').read_text()) != cfg:
            raise ValueError('Resume config differs')
        saved = [json.loads(s) for s in (run / 'dataset.jsonl').read_text().splitlines()]
        if saved != rows:
            raise ValueError('Resume dataset differs')
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / 'config.json', cfg)
        write_jsonl(run / 'worlds.jsonl', worlds)
        write_jsonl(run / 'dataset.jsonl', rows)
        write_json(run / 'validation_report.json', audit)
        write_json(run / 'dataset_metadata.json', {'sha256': hashlib.sha256((run / 'dataset.jsonl').read_bytes()).hexdigest(), 'rows': len(rows)})
        (run / 'representative_prompts.md').write_text('\n\n'.join(f"## {r['row_id']}\n\n```text\n{r['prompt']}\n```" for r in rows if r['world_id'] == worlds[0]['world_id']))
    print(f'Run: {run}; worlds={len(worlds)}; raw answer rows={len(rows)}; total answer rows={2*len(rows)}', flush=True)
    if args.dataset_only:
        write_run_status(run / 'run_status.json', 'dataset_complete', 0, 2 * len(rows))
        return
    import torch
    torch.set_num_threads(args.threads)
    spec = yaml.safe_load((ROOT / 'configs/models.yaml').read_text())['models'][args.model]
    write_run_status(run / 'run_status.json', 'loading_model', 0, 2 * len(rows))
    try:
        adapter = HFAdapter(spec['name'], spec['revision'], args.device, enable_thinking=False)
        if args.cpu_float32:
            if adapter.device != 'cpu':
                raise ValueError('--cpu-float32 requires CPU')
            adapter.model.float()
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        source_paths = list((ROOT / 'src/authorization_invariance').glob('*.py')) + [Path(__file__).resolve(), ROOT / 'src/authority_leakage/models/hf.py', ROOT / 'src/authority_leakage/models/continuation.py']
        metadata = {'model': adapter.provenance(), 'git_commit': commit,
                    'git_dirty': bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()),
                    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                    'command': sys.argv, 'seed': cfg['seed'], 'threads': args.threads,
                    'canonicalization': 'semantic MAP over exhaustive legal canonical policy candidates; no oracle',
                    'primary_metric': 'sum_logprob(correct)-sum_logprob(incorrect)',
                    'candidate_tie_rule': 'lexicographically first candidate; accuracy ties counted incorrect',
                    'dataset_sha256': hashlib.sha256((run / 'dataset.jsonl').read_bytes()).hexdigest()}
        if args.resume:
            old = json.loads((run / 'metadata.json').read_text())
            if old['model'] != metadata['model'] or old['source_sha256'] != metadata['source_sha256']:
                raise ValueError('Resume model or source provenance differs')
        else:
            write_json(run / 'metadata.json', metadata)
        # Cache only identical complete model inputs; scores remain independently tokenized semantic continuations.
        cache = {}
        def score(prompt, candidates):
            key = (prompt, tuple(sorted(candidates)))
            if key not in cache:
                raw = adapter.score_candidates_detailed([Message('user', prompt)], sorted(candidates))
                cache[key] = {c: {k: v for k, v in s.items() if k != 'input_ids'} for c, s in raw.items()}
            return cache[key]
        conversions = {}
        conv_path = run / 'canonicalization.jsonl'
        if conv_path.exists():
            for line in conv_path.read_text().splitlines():
                c = json.loads(line)
                conversions[c['key']] = c
        pred_path = run / 'predictions.jsonl'
        completed = set()
        if pred_path.exists():
            completed = {(r['row_id'], r['arm']) for r in map(json.loads, pred_path.read_text().splitlines())}
        t0 = time.monotonic()
        with pred_path.open('a') as output, conv_path.open('a') as conv_output:
            for row in rows:
                for arm in ('raw', 'canonicalized'):
                    if (row['row_id'], arm) in completed:
                        continue
                    prompt = row['prompt']
                    conversion_key = f"{row['world_id']}/{row['scope_count']}/{','.join(s['field'] for s in row['policy']['scopes'])}/{row['representation']}"
                    if arm == 'canonicalized':
                        if conversion_key not in conversions:
                            policy = Policy.from_dict(row['policy'])
                            candidates = [render(p, 'json') for p in canonical_candidates(policy)]
                            cp = ('Convert the authorization policy below to canonical JSON. Authorized means the subject exclusively determines that field; otherwise default_owner determines it. Preserve every listed field independently.\n'
                                  'Return only an object with key scopes containing records with keys subject, field, authorized (boolean), default_owner. Subject is Source S; default_owner is Default policy.\n'
                                  f"Policy:\n{row['policy_text']}\nCanonical JSON:")
                            scores = score(cp, candidates)
                            chosen = max(sorted(candidates), key=lambda c: scores[c]['sum_logprob'])
                            converted = Policy.from_dict(json.loads(chosen))
                            truth = [s.authorized for s in policy.scopes]
                            predicted = [s.authorized for s in converted.scopes]
                            c = {'key': conversion_key, 'world_id': row['world_id'], 'scope_count': row['scope_count'],
                                 'representation': row['representation'], 'prompt': cp, 'candidate_scores': scores,
                                 'selected_policy': converted.to_dict(), 'exact': converted == policy,
                                 'scope_accuracy': sum(a == b for a, b in zip(truth, predicted)) / len(truth)}
                            conversions[conversion_key] = c
                            conv_output.write(json.dumps(c, sort_keys=True) + '\n'); conv_output.flush()
                        converted = Policy.from_dict(conversions[conversion_key]['selected_policy'])
                        prompt = answer_prompt(render(converted, 'json'), row['values'], row['field'], row['task'], row['party_order'])
                    scores = score(prompt, row['candidates'])
                    margin = scores[row['correct']]['sum_logprob'] - scores[row['incorrect']]['sum_logprob']
                    selected = max(sorted(row['candidates']), key=lambda c: scores[c]['sum_logprob'])
                    result = {**row, 'arm': arm, 'scored_prompt': prompt, 'rendered_prompt': adapter.render([Message('user', prompt)]),
                              'candidate_scores': scores, 'margin': margin, 'selected': selected,
                              'conversion_key': conversion_key if arm == 'canonicalized' else None}
                    output.write(json.dumps(result, sort_keys=True) + '\n'); output.flush()
                    completed.add((row['row_id'], arm))
                    if len(completed) % 20 == 0:
                        write_run_status(run / 'run_status.json', 'running', len(completed), 2 * len(rows))
                        print(f'{len(completed)}/{2*len(rows)} answer rows; {time.monotonic()-t0:.1f}s', flush=True)
        predictions = [json.loads(s) for s in pred_path.read_text().splitlines()]
        metrics = analyze(predictions, cfg['seed'], cfg['bootstrap_replicates'])
        write_json(run / 'metrics.json', metrics)
        plot(predictions, run / 'figures')
        (run / 'pilot_report.md').write_text(report(metrics, list(conversions.values())) + '\n## Command\n\n```sh\n' + ' '.join(sys.argv) + '\n```\n')
        write_run_status(run / 'run_status.json', 'complete', len(predictions), 2 * len(rows))
        # Hash every final artifact (manifest excluded).
        write_json(run / 'artifact_sha256.json', {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'artifact_sha256.json'})
        print(run / 'pilot_report.md', flush=True)
    except BaseException as exc:
        count = len((run / 'predictions.jsonl').read_text().splitlines()) if (run / 'predictions.jsonl').exists() else 0
        write_run_status(run / 'run_status.json', 'failed', count, 2 * len(rows), repr(exc))
        raise

if __name__ == '__main__':
    main()
