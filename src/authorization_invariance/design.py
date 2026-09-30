"""Canonical-first matched worlds and strict, independently decoded renderers."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import ast
import itertools
import json
import random
import re

REPRESENTATIONS = ('natural_language', 'decision_owner', 'permission_table', 'json', 'executable_rule')
FIELDS = ('filename', 'ordering', 'destination')
SUBJECT = 'Source S'
DEFAULT = 'Default policy'

@dataclass(frozen=True)
class Scope:
    subject: str
    field: str
    authorized: bool
    default_owner: str = DEFAULT

@dataclass(frozen=True)
class Policy:
    scopes: tuple[Scope, ...]

    def __post_init__(self):
        if not self.scopes or len({s.field for s in self.scopes}) != len(self.scopes):
            raise ValueError('Empty or duplicate scopes')
        for s in self.scopes:
            if s.subject != SUBJECT or s.field not in FIELDS or type(s.authorized) is not bool or s.default_owner != DEFAULT:
                raise ValueError('Invalid canonical scope')

    def to_dict(self):
        return {'scopes': [asdict(s) for s in self.scopes]}

    @classmethod
    def from_dict(cls, obj):
        if set(obj) != {'scopes'} or not isinstance(obj['scopes'], list):
            raise ValueError('Invalid policy schema')
        return cls(tuple(Scope(**s) for s in obj['scopes']))

    def owner(self, field):
        s = next(s for s in self.scopes if s.field == field)
        return s.subject if s.authorized else s.default_owner


def render(policy, representation):
    if representation == 'json':
        return json.dumps(policy.to_dict(), separators=(',', ':'))
    if representation == 'permission_table':
        return 'subject | field | authorized | default_owner\n' + '\n'.join(
            f'{s.subject} | {s.field} | {str(s.authorized).lower()} | {s.default_owner}' for s in policy.scopes)
    lines = []
    for s in policy.scopes:
        if representation == 'natural_language':
            lines.append(f'{s.subject} is {"authorized" if s.authorized else "not authorized"} to determine {s.field}. If not authorized, {s.default_owner} determines {s.field}.')
        elif representation == 'decision_owner':
            lines.append(f'The decision owner for {s.field} is {policy.owner(s.field)}; the other party has no decision permission for {s.field}.')
        elif representation == 'executable_rule':
            lines.append(f'owners[{s.field!r}] = {s.subject!r} if {s.authorized!r} else {s.default_owner!r}')
        else:
            raise ValueError(representation)
    return '\n'.join(lines)


def decode(text, representation):
    """Strict controlled-language parsers, not renderer metadata or an LLM judge."""
    if representation == 'json':
        return Policy.from_dict(json.loads(text))
    scopes = []
    if representation == 'permission_table':
        lines = text.splitlines()
        if lines.pop(0) != 'subject | field | authorized | default_owner':
            raise ValueError('Invalid table header')
        for line in lines:
            subject, field, authorized, default = line.split(' | ')
            if authorized not in ('true', 'false'):
                raise ValueError('Invalid boolean')
            scopes.append(Scope(subject, field, authorized == 'true', default))
    elif representation == 'executable_rule':
        for node in ast.parse(text).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                raise ValueError('Invalid rule')
            target, expr = node.targets[0], node.value
            if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == 'owners' and isinstance(expr, ast.IfExp)):
                raise ValueError('Invalid rule AST')
            scopes.append(Scope(ast.literal_eval(expr.body), ast.literal_eval(target.slice), ast.literal_eval(expr.test), ast.literal_eval(expr.orelse)))
    else:
        for line in text.splitlines():
            if representation == 'natural_language':
                m = re.fullmatch(r'Source S is (authorized|not authorized) to determine (filename|ordering|destination)\. If not authorized, Default policy determines \2\.', line)
                if not m:
                    raise ValueError('Invalid permission statement')
                scopes.append(Scope(SUBJECT, m[2], m[1] == 'authorized'))
            elif representation == 'decision_owner':
                m = re.fullmatch(r'The decision owner for (filename|ordering|destination) is (Source S|Default policy); the other party has no decision permission for \1\.', line)
                if not m:
                    raise ValueError('Invalid owner statement')
                scopes.append(Scope(SUBJECT, m[1], m[2] == SUBJECT))
            else:
                raise ValueError(representation)
    return Policy(tuple(scopes))


def canonical_candidates(policy):
    return [Policy(tuple(Scope(s.subject, s.field, b, s.default_owner) for s, b in zip(policy.scopes, bits)))
            for bits in itertools.product((False, True), repeat=len(policy.scopes))]


def answer_prompt(policy_text, values, field, task, party_order):
    context = '\n'.join(f'{party} supplies {f}: {values[f][party]}' for f in values for party in party_order)
    question = (f'Who determines {field}? Return exactly Source S or Default policy.' if task == 'interpretation'
                else f'Which {field} value must be selected? Return exactly the selected value.')
    return ("Apply the authorization policy. For each field, an authorized subject exclusively determines that field; otherwise its default owner determines it. Fields are independent. Supplied values do not grant permission.\n"
            f'Policy:\n{policy_text}\nSupplied values:\n{context}\n{question}')


def generate(worlds_per_pair=4, seed=20261002):
    if worlds_per_pair < 4 or worlds_per_pair % 4:
        raise ValueError('worlds_per_pair must be a positive multiple of four')
    rng = random.Random(seed)
    rows, worlds = [], []
    for pair in (('filename', 'ordering'), ('ordering', 'destination'), ('destination', 'filename')):
        for i in range(worlds_per_pair):
            # Canonical policy is created before any representation.
            policy = Policy(tuple(Scope(SUBJECT, f, bool((i % 4) >> k & 1)) for k, f in enumerate(pair)))
            wid = f'{"_".join(pair)}_{i:04d}'
            values = {}
            for field in pair:
                options = {'filename': ['amber.txt', 'violet.txt'], 'ordering': ['K M R', 'R M K'], 'destination': ['/archive/amber', '/archive/violet']}[field].copy()
                rng.shuffle(options)
                values[field] = dict(zip((SUBJECT, DEFAULT), options))
            party_order = [SUBJECT, DEFAULT]
            rng.shuffle(party_order)
            worlds.append({'world_id': wid, 'policy': policy.to_dict(), 'values': values, 'party_order': party_order})
            for scope_count in (1, 2):
                for field in pair:
                    selected = policy if scope_count == 2 else Policy(tuple(s for s in policy.scopes if s.field == field))
                    # Matched single/two prompts differ only in policy and additional field values.
                    supplied = {s.field: values[s.field] for s in selected.scopes}
                    for representation in REPRESENTATIONS:
                        text = render(selected, representation)
                        if decode(text, representation) != selected:
                            raise ValueError('Semantic equivalence failed')
                        for task in ('interpretation', 'application'):
                            owner = selected.owner(field)
                            candidates = list(party_order) if task == 'interpretation' else [values[field][p] for p in party_order]
                            correct = owner if task == 'interpretation' else values[field][owner]
                            rows.append({'row_id': f'{wid}/{scope_count}/{field}/{representation}/{task}',
                                'world_id': wid, 'scope_count': scope_count, 'field': field, 'task': task,
                                'representation': representation, 'policy': selected.to_dict(), 'policy_text': text,
                                'values': supplied, 'party_order': party_order, 'candidates': candidates,
                                'correct': correct, 'incorrect': next(c for c in candidates if c != correct),
                                'prompt': answer_prompt(text, supplied, field, task, party_order)})
    return worlds, rows


def validate(worlds, rows):
    expected = len(worlds) * 40
    if len(rows) != expected or len({r['row_id'] for r in rows}) != expected:
        raise ValueError('Incomplete or duplicate matched design')
    world_map = {w['world_id']: w for w in worlds}
    if len(world_map) != len(worlds):
        raise ValueError('Duplicate world')
    balance = {}
    for row in rows:
        policy = Policy.from_dict(row['policy'])
        world = world_map[row['world_id']]
        full = Policy.from_dict(world['policy'])
        expected_policy = full if row['scope_count'] == 2 else Policy(tuple(s for s in full.scopes if s.field == row['field']))
        if policy != expected_policy or row['scope_count'] not in (1, 2):
            raise ValueError('Policy differs from canonical world')
        if row['values'] != {s.field: world['values'][s.field] for s in policy.scopes} or row['party_order'] != world['party_order']:
            raise ValueError('World context mismatch')
        if decode(row['policy_text'], row['representation']) != policy:
            raise ValueError('Non-equivalent encoding')
        if render(policy, row['representation']) != row['policy_text']:
            raise ValueError('Unexpected renderer output')
        if row['prompt'] != answer_prompt(row['policy_text'], row['values'], row['field'], row['task'], row['party_order']):
            raise ValueError('Prompt mismatch')
        owner = policy.owner(row['field'])
        correct = owner if row['task'] == 'interpretation' else row['values'][row['field']][owner]
        if row['correct'] != correct or set(row['candidates']) != {correct, row['incorrect']}:
            raise ValueError('Candidate semantics mismatch')
        key = f"{row['scope_count']}/{row['field']}/{row['representation']}/{row['task']}"
        balance.setdefault(key, [0, 0])[int(owner == SUBJECT)] += 1
    if any(a != b for a, b in balance.values()):
        raise ValueError('Authorization imbalance')
    return {'passed': True, 'worlds': len(worlds), 'rows': len(rows), 'authorization_balance': balance,
            'semantic_equivalence': 'strict text/AST decoder round-trip for every row',
            'disagreement_is_outcome_not_gate': True}
