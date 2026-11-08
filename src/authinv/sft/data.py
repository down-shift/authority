"""SFT training data for Phase 4 (P4.1): synthetic policies disjoint from the eval benchmark.

Worlds come from three **train-only domains** (`TRAIN_DOMAINS`). Their entity types,
ids, actions, attribute names, string values and context keys share nothing with the
eval benchmark; `disjointness` checks this against the benchmark's own vocabulary and
refuses any overlap. They use the P1.6 design grid (`sources.synthetic.design_grid`)
and its non-degeneracy rule, with a data-driven copy of its drawing procedure, so the
training policies have the eval synthetic worlds' structure but none of their words.

Each world is asked like the benchmark (`authinv.benchmark`, prompt-v1): balanced
allow/deny requests, boundary requests first, under both name assignments (orig and
the fixed derangement swap), as application and interpretation queries. Every query
is rendered in the **training renderings** only (nl_statement, owner_statement,
table, json_policy). Cedar (`executable`) is held out entirely, as RESEARCH_PLAN
§3 Phase 4 says, and Rego is unseen too. The target is the exact answer object the
prompt asks for. Each training rendering must round-trip through its exact decoder.

Sets (worlds are assigned by replicate index within every design cell, so each set
covers the whole grid):
- ``train_multi``: the multi-format set, every query in all four training renderings;
- ``val_multi``: the same over held-out training worlds (selection only; never Cedar);
- ``control_<model>``: the single-format control, one rendering (config), drawn
  from a separate pool of worlds until its token count under that model's own
  tokenizer and chat template matches ``train_multi``'s.

`DATA_VERSION` covers the domains, the drawing procedure, the example format and the
set construction. Bump it for any change.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass

from authinv.benchmark import (
    _context,
    application_prompt,
    interpretation_prompt,
    rename,
    rename_request,
    swap_axis,
    swap_map,
)
from authinv.equivalence.requests import sample_requests
from authinv.policy import (
    AttrRef,
    Condition,
    Entity,
    EntityRef,
    Policy,
    PolicyError,
    Rule,
    Scope,
    decision_owners,
    evaluate,
    semantic_hash,
    tier,
    validate,
)
from authinv.render.renderers import render, round_trips
from authinv.sources.synthetic import design_grid, is_non_degenerate

DATA_VERSION = "sft-data-v1"
TRAIN_RENDERINGS = ("nl_statement", "owner_statement", "table", "json_policy")
N_PRINCIPALS, N_GROUPS, N_RESOURCES = 5, 3, 3


@dataclass(frozen=True)
class TrainDomain:
    """A train-only domain. Attributes: (name, ("int", lo, hi) | ("str", choices) | ("bool",)).

    Condition templates: (left, op, right). left is (subject, attr); right is
    ("attr", subject, attr) | ("int", choices) | ("str", choices) | ("bool", value).
    A context attribute needs its type in `context_types`.
    """

    name: str
    principal_type: str
    group_type: str
    resource_type: str
    actions: tuple[str, ...]
    principal_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    principal_attrs: tuple[tuple, ...]
    resource_attrs: tuple[tuple, ...]
    conditions: tuple[tuple, ...]
    context_types: tuple[tuple[str, str], ...] = ()


LEDGER = TrainDomain(
    "ledger",
    "Clerk",
    "Bureau",
    "Ledger",
    ("journal", "reconcile", "seal", "tally"),
    ("ck_d4", "ck_f7", "ck_j2", "ck_m9", "ck_q3", "ck_s8", "ck_w5", "ck_z6"),
    ("bu_fennel", "bu_sorrel", "bu_thyme", "bu_yarrow"),
    ("lg_beryl", "lg_garnet", "lg_jasper", "lg_quartz"),
    (("rank", ("int", 0, 5)), ("desk", ("str", ("east", "west")))),
    (("desk", ("str", ("east", "west"))), ("state", ("str", ("balanced", "unbalanced")))),
    (
        (("principal", "rank"), ">=", ("int", (1, 2, 3, 4))),
        (("principal", "desk"), "==", ("attr", "resource", "desk")),
        (("resource", "state"), "==", ("str", ("balanced", "unbalanced"))),
        (("context", "notarized"), "==", ("bool", True)),
        (("context", "weekday"), "<", ("int", (3, 5, 6))),
    ),
    (("notarized", "bool"), ("weekday", "int")),
)
HANGAR = TrainDomain(
    "hangar",
    "Pilot",
    "Squadron",
    "Aircraft",
    ("board", "refuel", "taxi"),
    ("pl_b6", "pl_c3", "pl_g8", "pl_h1", "pl_k7", "pl_n4", "pl_r2", "pl_v9"),
    ("sq_corvid", "sq_grebe", "sq_plover", "sq_tern"),
    ("ac_bora", "ac_chinook", "ac_foehn", "ac_sirocco"),
    (("hours", ("int", 1, 5)), ("base", ("str", ("kilo_field", "lima_field")))),
    (("base", ("str", ("kilo_field", "lima_field"))), ("payload", ("int", 1, 5))),
    (
        (("principal", "hours"), ">=", ("attr", "resource", "payload")),
        (("resource", "base"), "==", ("str", ("kilo_field", "lima_field"))),
        (("principal", "base"), "==", ("str", ("kilo_field", "lima_field"))),
        (("context", "cleared"), "==", ("bool", True)),
        (("resource", "payload"), "<=", ("int", (1, 2, 3, 4))),
    ),
    (("cleared", "bool"),),
)
ARCHIVE = TrainDomain(
    "archive",
    "Scholar",
    "Faculty",
    "Codex",
    ("borrow", "catalog", "gloss", "rebind"),
    ("sc_a5", "sc_e2", "sc_i9", "sc_l4", "sc_o7", "sc_u1", "sc_x8", "sc_y3"),
    ("fa_lumen", "fa_quill", "fa_strata", "fa_vellum"),
    ("cx_ash", "cx_hazel", "cx_rowan", "cx_yew"),
    (("tenure", ("int", 0, 5)), ("wing", ("str", ("north_wing", "south_wing")))),
    (("wing", ("str", ("north_wing", "south_wing"))), ("fragile", ("bool",))),
    (
        (("principal", "tenure"), ">=", ("int", (1, 2, 3, 4))),
        (("principal", "wing"), "==", ("attr", "resource", "wing")),
        (("resource", "fragile"), "==", ("bool", False)),
        (("context", "supervised"), "==", ("bool", True)),
        (("context", "term"), "<", ("int", (2, 3))),
    ),
    (("supervised", "bool"), ("term", "int")),
)
TRAIN_DOMAINS = {d.name: d for d in (LEDGER, HANGAR, ARCHIVE)}


# ---- drawing (a data-driven copy of sources.synthetic._draw) -------------------------------------


def _attr_value(spec: tuple, rng: random.Random):
    if spec[0] == "int":
        return rng.randint(spec[1], spec[2])
    if spec[0] == "str":
        return rng.choice(spec[1])
    return rng.random() < 0.5


def _entities(
    d: TrainDomain, rng: random.Random
) -> tuple[tuple[Entity, ...], list[EntityRef], list[EntityRef]]:
    pids = rng.sample(d.principal_ids, N_PRINCIPALS)
    gids = rng.sample(d.group_ids, N_GROUPS)
    rids = rng.sample(d.resource_ids, N_RESOURCES)
    groups = [EntityRef(d.group_type, g) for g in gids]
    ents = [Entity(g) for g in groups]
    principals = []
    for i, pid in enumerate(pids):
        ref = EntityRef(d.principal_type, pid)
        principals.append(ref)
        member = (groups[i % N_GROUPS],) if i < N_GROUPS else tuple(rng.sample(groups, rng.randint(0, 2)))
        attrs = tuple((name, _attr_value(spec, rng)) for name, spec in d.principal_attrs)
        ents.append(Entity(ref, attrs, member))
    resources = []
    for rid in rids:
        ref = EntityRef(d.resource_type, rid)
        resources.append(ref)
        ents.append(Entity(ref, tuple((name, _attr_value(spec, rng)) for name, spec in d.resource_attrs)))
    return tuple(ents), principals, resources


def _condition(d: TrainDomain, rng: random.Random) -> tuple[Condition, str | None]:
    (subject, attr), op, right = rng.choice(d.conditions)
    if right[0] == "attr":
        value = AttrRef(right[1], right[2])
    elif right[0] == "bool":
        value = right[1]
    else:
        value = rng.choice(right[1])
    return Condition(AttrRef(subject, attr), op, value), (attr if subject == "context" else None)


def _reads(conds, subject: str) -> bool:
    return any(isinstance(x, AttrRef) and x.subject == subject for c in conds for x in (c.left, c.right))


def draw(d: TrainDomain, cell: dict, rng: random.Random, policy_id: str) -> Policy:
    ents, principals, resources = _entities(d, rng)
    groups = [e.ref for e in ents if e.ref.type == d.group_type]
    per_rule = [0] * cell["roles"]
    for _ in range(cell["conditions"]):
        per_rule[rng.randrange(cell["roles"])] += 1
    rules, ctx = [], set()
    for i in range(cell["roles"]):
        conds = []
        for _ in range(per_rule[i]):
            c, key = _condition(d, rng)
            conds.append(c)
            if key:
                ctx.add(key)
        kind = "is_in" if _reads(conds, "principal") else rng.choice(["in", "is_in"])
        principal = Scope(kind, groups[i], d.principal_type if kind == "is_in" else None)
        resource = (
            Scope("is", type=d.resource_type)
            if _reads(conds, "resource") or rng.random() < 0.6
            else Scope("eq", rng.choice(resources))
        )
        actions = tuple(sorted(rng.sample(d.actions, rng.randint(1, len(d.actions) - 1))))
        rules.append(Rule(f"r{i + 1}", "permit", principal, actions, resource, tuple(conds)))
    if cell["effect"] == "with_forbid":
        if rng.random() < 0.5:
            fp, fr = Scope("eq", rng.choice(principals)), Scope("is", type=d.resource_type)
        else:
            fp, fr = Scope("is", type=d.principal_type), Scope("eq", rng.choice(resources))
        actions = tuple(sorted(rng.sample(d.actions, rng.randint(1, 2))))
        rules.append(Rule(f"r{cell['roles'] + 1}", "forbid", fp, actions, fr))
    types = dict(d.context_types)
    return Policy(
        policy_id,
        ents,
        d.actions,
        tuple(rules),
        tuple(sorted((k, types[k]) for k in ctx)),
        meta=(
            ("domain", d.name),
            ("tier_cell", cell["tier"]),
            ("roles", cell["roles"]),
            ("conditions", cell["conditions"]),
            ("effect", cell["effect"]),
        ),
    )


def generate_worlds(
    domain: str, replicates: int, seed: int, min_per_label: int = 4, max_attempts: int = 200
) -> list[tuple[Policy, int]]:
    """(policy, replicate) for every design cell × replicate; deterministic in the seed."""
    d = TRAIN_DOMAINS[domain]
    out = []
    for ci, cell in enumerate(design_grid()):
        for rep in range(replicates):
            pid = f"sft-{domain}-c{ci:02d}-{rep:03d}"
            for attempt in range(max_attempts):
                p = draw(d, cell, random.Random(f"{seed}|{domain}|{ci}|{rep}|{attempt}"), pid)
                try:
                    validate(p)
                except PolicyError:
                    continue
                if tier(p) == cell["tier"] and is_non_degenerate(p, d, min_per_label)[0]:
                    out.append(
                        (
                            Policy(
                                p.policy_id,
                                p.entities,
                                p.actions,
                                p.rules,
                                p.context_schema,
                                p.meta + (("attempt", attempt),),
                            ),
                            rep,
                        )
                    )
                    break
            else:
                raise RuntimeError(f"{pid}: no non-degenerate world in {max_attempts} attempts ({cell})")
    return out


# ---- examples --------------------------------------------------------------------------------------


def answer_text(task: str, value) -> str:
    """The exact object the prompt asks for (what T2's strict parser accepts)."""
    return json.dumps({"decision": value}) if task == "application" else json.dumps({"principals": value})


def world_examples(
    policy: Policy,
    domain: TrainDomain,
    renderings: tuple[str, ...],
    per_label: int,
    seed: int,
    tasks: tuple[str, ...] = ("application", "interpretation"),
) -> list[dict]:
    """Every query of one world under both assignments in the given renderings (as `benchmark.world_rows`)."""
    pt, rt = (domain.principal_type,), (domain.resource_type,)
    picked = sample_requests(policy, pt, rt, per_label, seed)
    m = swap_map(policy, swap_axis(policy, pt, rt))
    out = []
    for ai, (assignment, pol, mapping) in enumerate((("orig", policy, {}), ("swap", rename(policy, m), m))):
        bad = [g for g in renderings if not round_trips(pol, g)]
        if bad:
            raise ValueError(f"{policy.policy_id}/{assignment}: renderings do not round-trip: {bad}")
        texts = {g: render(pol, g) for g in renderings}
        asked: set[str] = set()
        for qi, item in enumerate(picked):
            req = rename_request(item["request"], mapping) if mapping else item["request"]
            label = evaluate(pol, req)["decision"]
            if label != item["decision"]:
                raise AssertionError(f"{policy.policy_id}: name swap changed a decision")
            pair_key = f"{req.action}|{req.resource.key()}|{_context(req.context)}"
            owners = [r.id for r in decision_owners(pol, req.action, req.resource, req.context, pt)]
            base = {
                "world_id": policy.policy_id,
                "domain": domain.name,
                "assignment": assignment,
                "instance": f"q{qi}",
                "pair_key": pair_key,
            }
            for g in renderings:
                if "application" in tasks:
                    prompt = application_prompt(pol, texts[g], req, (qi + ai) % 2)
                    out.append(_example(base, g, "application", prompt, label))
                if "interpretation" in tasks and pair_key not in asked:
                    prompt = interpretation_prompt(pol, texts[g], req, pt)
                    out.append(_example(base, g, "interpretation", prompt, owners))
            asked.add(pair_key)
    return out


def _example(base: dict, rendering: str, task: str, prompt: str, value) -> dict:
    return {
        **base,
        "example_id": f"{base['world_id']}/{base['assignment']}/{base['instance']}/{rendering}/{task}",
        "rendering": rendering,
        "task": task,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_text(task, value)},
        ],
    }


# ---- disjointness from the eval benchmark ----------------------------------------------------------


def vocabulary(policy_dicts: list[dict]) -> dict[str, set]:
    """Identifiers and string values of canonical policies (`policy.to_dict` form)."""
    v = {
        "types": set(),
        "ids": set(),
        "actions": set(),
        "attributes": set(),
        "context_keys": set(),
        "string_values": set(),
    }
    for p in policy_dicts:
        v["actions"] |= set(p["actions"])
        v["context_keys"] |= {k for k, _ in p["context_schema"]}
        for e in p["entities"]:
            v["types"].add(e["ref"]["type"])
            v["ids"].add(e["ref"]["id"])
            for k, val in e["attrs"]:
                v["attributes"].add(k)
                if isinstance(val, str):
                    v["string_values"].add(val)
        for r in p["rules"]:
            for c in r["conditions"]:
                for side in (c["left"], c["right"]):
                    if "attr" in side:
                        v["attributes" if side["attr"]["subject"] != "context" else "context_keys"].add(
                            side["attr"]["attr"]
                        )
                    elif isinstance(side["value"], str):
                        v["string_values"].add(side["value"])
    return v


def disjointness(train: list[Policy], eval_policies: list[dict]) -> dict:
    """Semantic-hash and vocabulary overlap between training and eval policies (both must be empty)."""
    from authinv.policy import from_dict, to_dict

    eval_hashes = {semantic_hash(from_dict(p)) for p in eval_policies}
    train_hashes = [semantic_hash(p) for p in train]
    tv, ev = vocabulary([to_dict(p) for p in train]), vocabulary(eval_policies)
    overlap = {k: sorted(tv[k] & ev[k]) for k in tv}
    return {
        "train_policies": len(train),
        "eval_policies": len(eval_policies),
        "semantic_hash_overlap": sum(h in eval_hashes for h in train_hashes),
        "duplicate_train_hashes": len(train_hashes) - len(set(train_hashes)),
        "vocabulary_overlap": overlap,
        "passed": not any(overlap.values())
        and not any(h in eval_hashes for h in train_hashes)
        and len(set(train_hashes)) == len(train_hashes),
    }


# ---- token-matched control -------------------------------------------------------------------------


def match_tokens(pool: list[dict], target: int, count: Callable[[dict], int]) -> tuple[list[dict], int]:
    """The shortest prefix of `pool` whose token count reaches `target` (refuses if the pool is too small)."""
    total, out = 0, []
    for ex in pool:
        if total >= target:
            break
        n = count(ex)
        out.append({**ex, "tokens": n})
        total += n
    if total < target:
        raise ValueError(f"control pool too small: {total} < {target} tokens")
    return out, total
