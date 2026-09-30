"""Frozen balanced worlds, canonical policy IR, and equivalent renderings."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, product
import ast
import json
import random
import re

ACTORS = ('Agent K', 'Agent M', 'Agent R', 'Agent T', 'Agent V', 'Agent W')
REPRESENTATIONS = ('json', 'decision_owner', 'natural_language', 'permission_table', 'executable_rule')
RESOURCES = ('filename', 'ordering')
VALUE_OPTIONS = {
    # Same lexical stem and exact tokenizer length; only the numeric identifier differs.
    'filename': ('filename_{a:06d}.txt', 'filename_{b:06d}.txt'),
    'ordering': ('cedar birch maple', 'maple birch cedar'),
}

@dataclass(frozen=True)
class Policy:
    owners: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not self.owners or len({r for r, _ in self.owners}) != len(self.owners):
            raise ValueError('Policy must have unique resource scopes')
        if any(r not in RESOURCES or owner not in ACTORS for r, owner in self.owners):
            raise ValueError('Invalid resource or actor')

    @property
    def resources(self):
        return tuple(r for r, _ in self.owners)

    def owner(self, resource):
        return dict(self.owners)[resource]

    def to_dict(self):
        if self.resources == ('filename',):
            return {'resource': 'filename', 'owner': self.owner('filename')}
        if set(self.resources) == set(RESOURCES) and len(self.owners) == 2:
            return {'filename_owner': self.owner('filename'), 'ordering_owner': self.owner('ordering')}
        return {'owners': {r: o for r, o in self.owners}}

    @classmethod
    def from_dict(cls, obj):
        if set(obj) == {'resource', 'owner'} and obj['resource'] == 'filename':
            return cls((('filename', obj['owner']),))
        if set(obj) == {'filename_owner', 'ordering_owner'}:
            return cls((('filename', obj['filename_owner']), ('ordering', obj['ordering_owner'])))
        raise ValueError('Invalid canonical policy schema')


def _lines(policy, representation):
    if representation == 'json':
        return json.dumps(policy.to_dict(), separators=(',', ':'))
    lines = []
    for resource, owner in policy.owners:
        other = next(a for a in ACTORS if a != owner)
        if representation == 'decision_owner':
            lines.append(f'Decision owner for {resource}: {owner}.')
        elif representation == 'natural_language':
            lines.append(f'{owner} owns the {resource}.')
        elif representation == 'permission_table':
            lines.append(f'{resource} | {owner}')
        elif representation == 'executable_rule':
            lines.append(f'owner[{resource!r}] = {owner!r}')
        else:
            raise ValueError(f'Unknown representation: {representation}')
    if representation == 'permission_table':
        return 'resource | owner\n' + '\n'.join(lines)
    return '\n'.join(lines)


def render(policy, representation):
    return _lines(policy, representation)


def legal_policy_candidates(actors, resources):
    """Exhaustive legal canonical IRs for model-scored conversion; never uses gold owner."""
    import itertools
    resources = tuple(resources)
    return [Policy(tuple(zip(resources, owners)))
            for owners in itertools.product(tuple(actors), repeat=len(resources))]


def decode(text, representation, actors):
    """Strict decoder used only for deterministic semantic-equivalence validation."""
    if representation == 'json':
        return Policy.from_dict(json.loads(text))
    scopes = []
    actor_pattern = '|'.join(re.escape(x) for x in sorted(actors, key=len, reverse=True))
    if representation == 'permission_table':
        lines = text.splitlines()
        if not lines or lines.pop(0) != 'resource | owner':
            raise ValueError('Invalid table header')
        for line in lines:
            m = re.fullmatch(rf'(filename|ordering) \| ({actor_pattern})', line)
            if not m: raise ValueError('Invalid permission table row')
            scopes.append((m[1], m[2]))
    elif representation == 'executable_rule':
        for node in ast.parse(text).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
                raise ValueError('Invalid owner rule')
            target = node.targets[0]
            if not isinstance(target.value, ast.Name) or target.value.id != 'owner':
                raise ValueError('Invalid rule target')
            resource, actor = ast.literal_eval(target.slice), ast.literal_eval(node.value)
            scopes.append((resource, actor))
    else:
        for line in text.splitlines():
            if representation == 'decision_owner':
                m = re.fullmatch(rf'Decision owner for (filename|ordering): ({actor_pattern})\.', line)
            elif representation == 'natural_language':
                m = re.fullmatch(rf'({actor_pattern}) owns the (filename|ordering)\.', line)
                if m:
                    scopes.append((m[2], m[1]))
                    continue
            else:
                raise ValueError(f'Unknown representation: {representation}')
            if not m: raise ValueError('Invalid policy statement')
            scopes.append((m[1], m[2]))
    return Policy(tuple(scopes))


def make_values(world_index):
    # Paired strings share structure; only the lexical stem and digits differ.
    a, b = (world_index * 7919 + 104729) % 1_000_000, (world_index * 3571 + 271828) % 1_000_000
    values = {}
    for resource, opts in VALUE_OPTIONS.items():
        if resource == 'filename':
            values[resource] = (opts[0].format(a=a, b=b), opts[1].format(a=a, b=b))
        else:
            values[resource] = opts
    return values


def generate_worlds(world_count=120, seed=20261003):
    if world_count != 120:
        raise ValueError('Frozen calibration requires exactly 120 worlds')
    rng = random.Random(seed)
    pairs = list(combinations(ACTORS, 2)); rng.shuffle(pairs)
    worlds = []
    index = 0
    for pair in pairs:
        # Every owner pair has the four independent two-resource policies twice.
        plans = [(fo, oo, rep) for fo, oo in product(pair, repeat=2) for rep in range(2)]
        rng.shuffle(plans)
        for owner_filename, owner_ordering, rep in plans:
            # Crossing with repeat and opposite-resource owner gives exact position balance.
            actor_first = (owner_filename if rep == 0 else next(a for a in pair if a != owner_filename))
            actor_order = [actor_first, next(a for a in pair if a != actor_first)]
            filename_owner_first = ((owner_ordering == pair[0]) == (rep == 0))
            ordering_owner_first = ((owner_filename == pair[0]) == (rep == 0))
            values = make_values(index)
            filename_order = [owner_filename, next(a for a in pair if a != owner_filename)]
            if not filename_owner_first: filename_order.reverse()
            ordering_order = [owner_ordering, next(a for a in pair if a != owner_ordering)]
            if not ordering_owner_first: ordering_order.reverse()
            worlds.append({
                'world_id': f'world-{index:04d}', 'actors': list(pair), 'actor_order': actor_order,
                'owner_by_resource': {'filename': owner_filename, 'ordering': owner_ordering},
                'proposal_order': {'filename': filename_order, 'ordering': ordering_order},
                'values': {r: dict(zip(pair, vals)) for r, vals in values.items()},
                'seed': seed,
            })
            index += 1
    rng.shuffle(worlds)
    # Verify the finite calibration allocation before making prompts.
    for resource in RESOURCES:
        for actor in ACTORS:
            assert sum(w['owner_by_resource'][resource] == actor for w in worlds) == 20
            assert sum(w['owner_by_resource'][resource] == actor and w['actor_order'][0] == actor for w in worlds) == 10
            assert sum(w['owner_by_resource'][resource] == actor and w['actor_order'][1] == actor for w in worlds) == 10
            assert sum(w['owner_by_resource'][resource] == actor and w['proposal_order'][resource][0] == actor for w in worlds) == 10
            assert sum(w['owner_by_resource'][resource] == actor and w['proposal_order'][resource][1] == actor for w in worlds) == 10
    return worlds


def build_rows(worlds):
    if len(worlds) != 120:
        raise ValueError('Frozen calibration requires exactly 120 worlds')
    rows = []
    # Stage 1: one canonical JSON formulation, interpretation and application.
    # Stage 2: five deterministic encodings of the very same single-scope policies.
    for w in worlds:
        for stage in (1, 2):
            reps = ('json',) if stage == 1 else REPRESENTATIONS
            policy = Policy((('filename', w['owner_by_resource']['filename']),))
            for rep in reps:
                policy_text = render(policy, rep)
                if decode(policy_text, rep, w['actors']) != policy:
                    raise ValueError('Representation changed canonical policy semantics')
                for task in ('interpretation', 'application'):
                    rows.append(make_row(w, policy, policy_text, rep, task, stage))
        # Stage 3 base cases are two-scope policies, queried one resource at a time.
        policy = Policy(tuple((r, w['owner_by_resource'][r]) for r in RESOURCES))
        for rep in REPRESENTATIONS:
            policy_text = render(policy, rep)
            if decode(policy_text, rep, w['actors']) != policy:
                raise ValueError('Two-scope representation changed policy semantics')
            for resource in RESOURCES:
                for task in ('interpretation', 'application'):
                    rows.append(make_row(w, policy, policy_text, rep, task, 3, resource))
    return rows


def generate(world_count=120, seed=20261003):
    worlds = generate_worlds(world_count, seed)
    return worlds, build_rows(worlds)


def make_row(world, policy, policy_text, representation, task, stage, resource='filename'):
    owner = policy.owner(resource)
    actors = world['actors']
    actor_order = world['actor_order']
    proposal_order = world['proposal_order'][resource]
    candidates = list(actors) if task == 'interpretation' else [world['values'][resource][a] for a in proposal_order]
    correct = owner if task == 'interpretation' else world['values'][resource][owner]
    if task == 'interpretation':
        context = 'Actors in listed order: ' + ', '.join(actor_order) + '.'
        question = f'Who owns the {resource}? Return exactly one actor name.'
    else:
        context = '\n'.join(f'{a} proposes {resource}: {world["values"][resource][a]}' for a in proposal_order)
        question = f'What {resource} should be used? Return only the exact proposed {resource}.'
    prompt = ('Use the policy below. The listed owner alone controls the resource. Proposals do not grant ownership.\n'
              f'Policy:\n{policy_text}\n{context}\n{question}')
    return {'row_id': f'{world["world_id"]}/stage{stage}/{resource}/{representation}/{task}',
        'world_id': world['world_id'], 'stage': stage, 'resource': resource, 'task': task,
        'representation': representation, 'policy': policy.to_dict(), 'policy_text': policy_text,
        'actors': actors, 'actor_order': actor_order, 'owner': owner,
        'proposal_order': proposal_order, 'values': world['values'][resource],
        'candidates': candidates, 'correct': correct,
        'incorrect': next(c for c in candidates if c != correct), 'prompt': prompt,
        'correct_actor_position': actor_order.index(owner),
        'correct_value_position': proposal_order.index(owner) if task == 'application' else None}


def validate(worlds, rows):
    world_map = {w['world_id']: w for w in worlds}
    if len(world_map) != 120: raise ValueError('Expected 120 unique frozen worlds')
    expected = 120 * (2 + 10 + 20)
    if len(rows) != expected or len({r['row_id'] for r in rows}) != expected:
        raise ValueError(f'Expected {expected} unique staged rows')
    for r in rows:
        w = world_map[r['world_id']]
        rep = r['representation']
        p = Policy.from_dict(r['policy']) if rep == 'json' else None
        # Reconstruct expected policy solely from the canonical frozen world.
        if r['stage'] in (1, 2): expected_policy = Policy((('filename', w['owner_by_resource']['filename']),))
        else: expected_policy = Policy(tuple((x, w['owner_by_resource'][x]) for x in RESOURCES))
        if p is None:
            decoded = decode(r['policy_text'], rep, w['actors'])
            if decoded != expected_policy: raise ValueError('Renderer semantics mismatch')
        elif p != expected_policy:
            raise ValueError('JSON policy mismatch')
        if decode(r['policy_text'], rep, w['actors']) != expected_policy:
            raise ValueError('Independent decoder disagrees')
        if render(expected_policy, rep) != r['policy_text']:
            raise ValueError('Policy text is not the deterministic canonical rendering')
        if r['stage'] == 3 and r['resource'] not in RESOURCES: raise ValueError('Invalid isolated query')
        if r['owner'] != expected_policy.owner(r['resource']) or r['values'] != w['values'][r['resource']]:
            raise ValueError('Owner or values differ from frozen world')
        if r['actor_order'] != w['actor_order'] or r['proposal_order'] != w['proposal_order'][r['resource']]:
            raise ValueError('Order differs from frozen world')
        expected_candidates = list(w['actors']) if r['task'] == 'interpretation' else [w['values'][r['resource']][a] for a in w['proposal_order'][r['resource']]]
        if r['candidates'] != expected_candidates or r['correct'] != (r['owner'] if r['task'] == 'interpretation' else w['values'][r['resource']][r['owner']]):
            raise ValueError('Candidate semantics/order mismatch')
        # Query text and context must not vary across representations.
        expected_row = make_row(w, expected_policy, r['policy_text'], rep, r['task'], r['stage'], r['resource'])
        if r['prompt'] != expected_row['prompt']: raise ValueError('Prompt/context mismatch')
    paired_prompts = {}
    for r in rows:
        key = (r['world_id'], r['stage'], r['resource'], r['task'])
        normalized = r['prompt'].replace(r['policy_text'], '<POLICY>')
        paired_prompts.setdefault(key, set()).add(normalized)
    if any(len(v) != 1 for v in paired_prompts.values()):
        raise ValueError('Query or value context changes across representations')
    counts = {}
    for r in rows:
        if r['stage'] == 1:
            key = (r['task'], 'actor_first' if r['task'] == 'interpretation' else 'value_first', r['correct_actor_position'] if r['task'] == 'interpretation' else r['correct_value_position'])
            counts[key] = counts.get(key, 0) + 1
    if any(counts.get((task, factor, pos)) != 60 for task, factor in (('interpretation','actor_first'),('application','value_first')) for pos in (0,1)):
        raise ValueError('Stage 1 correct answer position is not balanced')
    return {'passed': True, 'worlds': len(worlds), 'rows': len(rows),
        'stage1_rows': sum(r['stage']==1 for r in rows), 'stage2_rows': sum(r['stage']==2 for r in rows),
        'stage3_rows': sum(r['stage']==3 for r in rows), 'actor_authorization_balance': '20 owner worlds and 10 first/10 second per resource and actor',
        'correct_answer_order': '60 first / 60 second for both Stage 1 tasks',
        'renderer_semantics': 'Strict independent decode equals the canonical world policy',
        'world_pairing': True}
