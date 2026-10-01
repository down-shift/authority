"""Fresh single-scope calibration worlds paired across swapped actor names."""
from __future__ import annotations

from collections import Counter
import json
import random
import re

from authorization_competence.design import REPRESENTATIONS

ACTOR_FAMILIES = tuple("ABCDEF")
ACTOR_IDS_PER_FAMILY = 6
ACTOR_IDENTIFIERS = tuple(
    f"Agent {family}{index:02d}"
    for family in ACTOR_FAMILIES
    for index in range(1, ACTOR_IDS_PER_FAMILY + 1)
)
ASSIGNMENTS = ("original", "swapped")
TASKS = ("interpretation", "application")


def actor_family(actor: str) -> str:
    if actor not in ACTOR_IDENTIFIERS:
        raise ValueError(f"Unknown lexical-symmetry actor: {actor}")
    return actor[6]


def generate_semantic_worlds(world_count: int = 180, seed: int = 20261004):
    if world_count != 180:
        raise ValueError("Lexical-symmetry calibration is frozen at exactly 180 worlds")
    rng = random.Random(seed)
    worlds = []
    index = 0
    # Each family contributes 30 worlds. Five traversals of six adjacent-ID
    # pairs make every identifier occur in exactly ten semantic worlds.
    for family_index, family in enumerate(ACTOR_FAMILIES):
        family_actors = [a for a in ACTOR_IDENTIFIERS if actor_family(a) == family]
        plans = []
        for traversal in range(5):
            for pair_index in range(ACTOR_IDS_PER_FAMILY):
                # Alternating across pair/traversal gives exactly 15 worlds per
                # family owned by each logical actor.
                owner = "logical_actor_1" if (traversal + pair_index) % 2 == 0 else "logical_actor_2"
                plans.append((traversal, pair_index, owner))
        rng.shuffle(plans)
        # Independently balance actor order and correct filename position.
        value_first_flags = [True] * 15 + [False] * 15
        rng.shuffle(value_first_flags)
        actor_first_flags = {}
        for logical_owner in ("logical_actor_1", "logical_actor_2"):
            owner_indices = [i for i, plan in enumerate(plans) if plan[2] == logical_owner]
            first_count = (7 if family_index < 3 else 8) if logical_owner == "logical_actor_1" else (8 if family_index < 3 else 7)
            flags = [True] * first_count + [False] * (len(owner_indices) - first_count)
            rng.shuffle(flags)
            actor_first_flags.update(zip(owner_indices, flags))
        for local_index, (traversal, pair_index, owner) in enumerate(plans):
            logical_pair = ("logical_actor_1", "logical_actor_2")
            identity_pair = (family_actors[pair_index], family_actors[(pair_index + 1) % len(family_actors)])
            actor_order = list(logical_pair if actor_first_flags[local_index] else reversed(logical_pair))
            values = {
                "logical_actor_1": f"filename_{(index * 7919 + 104729) % 1_000_000:06d}.txt",
                "logical_actor_2": f"filename_{(index * 3571 + 271828) % 1_000_000:06d}.txt",
            }
            value_order = list((owner, "logical_actor_2" if owner == "logical_actor_1" else "logical_actor_1")
                               if value_first_flags[local_index] else
                               ("logical_actor_2" if owner == "logical_actor_1" else "logical_actor_1", owner))
            worlds.append({
                "world_id": f"lex-world-{index:04d}",
                "seed": seed,
                "family": family,
                "family_index": family_index,
                "traversal": traversal,
                "pair_index": pair_index,
                "logical_actors": list(logical_pair),
                "identity_pair": list(identity_pair),
                "owner_logical": owner,
                "actor_order_logical": actor_order,
                "proposal_order_logical": value_order,
                "values_by_logical_actor": values,
            })
            index += 1
    rng.shuffle(worlds)
    return worlds


def name_map(world, assignment):
    a, b = world["identity_pair"]
    return ({"logical_actor_1": a, "logical_actor_2": b}
            if assignment == "original" else
            {"logical_actor_1": b, "logical_actor_2": a})


def render_policy(owner_name, representation="json"):
    """Deterministically render one owner from the same canonical semantics."""
    if representation == "json":
        return json.dumps({"resource": "filename", "owner": owner_name}, separators=(",", ":"))
    if representation == "decision_owner":
        return f"Decision owner for filename: {owner_name}."
    if representation == "natural_language":
        return f"{owner_name} owns the filename."
    if representation == "permission_table":
        return f"resource | owner\nfilename | {owner_name}"
    if representation == "executable_rule":
        return f"owner['filename'] = {owner_name!r}"
    raise ValueError(f"Unknown policy representation: {representation}")


def decode_lexical_policy(text, representation):
    """Strictly decode the lexical-symmetry single-scope renderer outputs."""
    if representation == "json":
        value = json.loads(text)
        if set(value) != {"resource", "owner"} or value["resource"] != "filename":
            raise ValueError("Invalid canonical JSON policy")
        owner = value["owner"]
    elif representation == "decision_owner":
        match = re.fullmatch(r"Decision owner for filename: (Agent [A-F][0-9]{2})\.", text)
        if not match:
            raise ValueError("Invalid decision-owner policy")
        owner = match.group(1)
    elif representation == "natural_language":
        match = re.fullmatch(r"(Agent [A-F][0-9]{2}) owns the filename\.", text)
        if not match:
            raise ValueError("Invalid natural-language policy")
        owner = match.group(1)
    elif representation == "permission_table":
        match = re.fullmatch(r"resource \| owner\nfilename \| (Agent [A-F][0-9]{2})", text)
        if not match:
            raise ValueError("Invalid permission table policy")
        owner = match.group(1)
    elif representation == "executable_rule":
        match = re.fullmatch(r"owner\['filename'\] = ('Agent [A-F][0-9]{2}')", text)
        if not match:
            raise ValueError("Invalid executable rule policy")
        owner = ast_literal_string(match.group(1))
    else:
        raise ValueError(f"Unknown policy representation: {representation}")
    if owner not in ACTOR_IDENTIFIERS:
        raise ValueError("Unknown owner identifier")
    return {"resource": "filename", "owner": owner}


def ast_literal_string(source):
    # Keep executable-rule decoding literal-only; eval is never used.
    import ast
    value = ast.literal_eval(source)
    if not isinstance(value, str):
        raise ValueError("Executable rule owner must be a string literal")
    return value


def make_lexical_row(world, assignment, task, representation="json", stage=1):
    mapping = name_map(world, assignment)
    owner_name = mapping[world["owner_logical"]]
    policy_text = render_policy(owner_name, representation)
    actor_order = [mapping[a] for a in world["actor_order_logical"]]
    proposal_order_logical = world["proposal_order_logical"]
    proposal_order = [mapping[a] for a in proposal_order_logical]
    values = {mapping[logical]: value for logical, value in world["values_by_logical_actor"].items()}
    candidates = actor_order.copy() if task == "interpretation" else [values[a] for a in proposal_order]
    correct = owner_name if task == "interpretation" else values[owner_name]
    if task == "interpretation":
        context = "Actors in listed order: " + ", ".join(actor_order) + "."
        question = "Who owns the filename? Return exactly one actor name."
    else:
        context = "\n".join(f"{a} proposes filename: {values[a]}" for a in proposal_order)
        question = "What filename should be used? Return only the exact proposed filename."
    prompt = ("Use the policy below. The listed owner alone controls the resource. Proposals do not grant ownership.\n"
              f"Policy:\n{policy_text}\n{context}\n{question}")
    return {
        "row_id": f"{world['world_id']}/stage{stage}/{assignment}/{representation}/{task}",
        "world_id": world["world_id"], "assignment": assignment, "task": task,
        "representation": representation, "stage": stage, "family": world["family"],
        "policy": {"resource": "filename", "owner": owner_name}, "policy_text": policy_text,
        "name_map": mapping, "logical_actors": world["logical_actors"],
        "identity_pair": world["identity_pair"], "actor_order": actor_order,
        "actor_order_logical": world["actor_order_logical"],
        "owner": owner_name, "owner_logical": world["owner_logical"],
        "proposal_order": proposal_order, "proposal_order_logical": proposal_order_logical,
        "values": values, "values_by_logical_actor": world["values_by_logical_actor"],
        "candidates": candidates, "correct": correct,
        "incorrect": next(c for c in candidates if c != correct), "prompt": prompt,
        "correct_actor_position": actor_order.index(owner_name),
        "correct_value_position": proposal_order.index(owner_name) if task == "application" else None,
    }


def build_lexical_rows(worlds):
    return [make_lexical_row(w, assignment, task, "json", 1)
            for w in worlds for assignment in ASSIGNMENTS for task in TASKS]


def build_lexical_representation_rows(worlds):
    return [make_lexical_row(w, assignment, task, representation, 2)
            for w in worlds for representation in REPRESENTATIONS
            for assignment in ASSIGNMENTS for task in TASKS]


def validate_lexical_worlds(worlds, rows):
    if len(worlds) != 180 or len({w["world_id"] for w in worlds}) != 180:
        raise ValueError("Expected exactly 180 unique semantic worlds")
    if len(rows) != 720 or len({r["row_id"] for r in rows}) != 720:
        raise ValueError("Expected exactly 720 paired task rows")
    world_by_id = {w["world_id"]: w for w in worlds}
    row_by_key = {(r["world_id"], r["assignment"], r["task"]): r for r in rows}
    if len(row_by_key) != len(rows):
        raise ValueError("Duplicate world/assignment/task rows")
    owner_counts = Counter()
    identifier_world_counts = Counter()
    identity_frequency = Counter()
    for world in worlds:
        if world["owner_logical"] not in world["logical_actors"]:
            raise ValueError("Semantic owner must be a logical actor")
        if set(world["values_by_logical_actor"]) != set(world["logical_actors"]):
            raise ValueError("Each logical actor must have exactly one proposed value")
        if len(set(world["values_by_logical_actor"].values())) != 2:
            raise ValueError("Actor proposal values must be distinct")
        for actor in world["identity_pair"]:
            identifier_world_counts[actor] += 1
        owner_counts[(world["family"], world["owner_logical"])] += 1
        for assignment in ASSIGNMENTS:
            mapping = name_map(world, assignment)
            if set(mapping) != set(world["logical_actors"]) or len(set(mapping.values())) != 2:
                raise ValueError("Name assignment must map logical actors bijectively")
            owner_name = mapping[world["owner_logical"]]
            for task in TASKS:
                row = row_by_key[(world["world_id"], assignment, task)]
                expected = make_lexical_row(world, assignment, task)
                if row != expected:
                    raise ValueError("Rendered row differs from the canonical semantic world")
                if row["policy"] != {"resource": "filename", "owner": owner_name}:
                    raise ValueError("JSON policy owner does not follow the name assignment")
                identity_frequency[(owner_name, "authorized_policy_rows")] += 1
                identity_frequency[(row["actor_order"][0], "interpretation_first")] += int(task == "interpretation")
                identity_frequency[(row["actor_order"][1], "interpretation_second")] += int(task == "interpretation")
                identity_frequency[(row["proposal_order"][0], "proposal_first")] += int(task == "application")
                identity_frequency[(row["proposal_order"][1], "proposal_second")] += int(task == "application")
                if task == "interpretation":
                    identity_frequency[(row["candidates"][0], "candidate_occurrences")] += 1
                    identity_frequency[(row["candidates"][1], "candidate_occurrences")] += 1
        original = row_by_key[(world["world_id"], "original", "application")]
        swapped = row_by_key[(world["world_id"], "swapped", "application")]
        if original["correct"] != swapped["correct"]:
            raise ValueError("Correct semantic filename changed under name swap")
        if original["values_by_logical_actor"] != swapped["values_by_logical_actor"]:
            raise ValueError("Logical proposals changed under name swap")
        if original["owner_logical"] != swapped["owner_logical"]:
            raise ValueError("Semantic owner changed under name swap")
    if set(world_by_id) != {r["world_id"] for r in rows}:
        raise ValueError("Rows and worlds do not pair exactly")
    for family in ACTOR_FAMILIES:
        if sum(w["family"] == family for w in worlds) != 30:
            raise ValueError("Each identifier family must contribute 30 worlds")
        if owner_counts[(family, "logical_actor_1")] != 15 or owner_counts[(family, "logical_actor_2")] != 15:
            raise ValueError("Logical owner must be balanced within each identifier family")
    if set(identifier_world_counts) != set(ACTOR_IDENTIFIERS) or any(n != 10 for n in identifier_world_counts.values()):
        raise ValueError("Every identifier must appear in exactly ten semantic worlds")
    for actor in ACTOR_IDENTIFIERS:
        expected_counts = {"authorized_policy_rows": 20, "interpretation_first": 10,
                           "interpretation_second": 10, "proposal_first": 10,
                           "proposal_second": 10, "candidate_occurrences": 20}
        if any(identity_frequency[(actor, key)] != count for key, count in expected_counts.items()):
            raise ValueError(f"Identifier authorization/position frequency is not balanced for {actor}")
    app = [r for r in rows if r["task"] == "application"]
    if sum(r["correct_value_position"] == 0 for r in app) != 180:
        raise ValueError("Correct value must be first in half of application rows")
    if sum(r["correct_actor_position"] == 0 for r in rows if r["task"] == "interpretation") != 180:
        raise ValueError("Correct actor must be first in half of interpretation rows")
    return {
        "passed": True, "worlds": len(worlds), "rows": len(rows),
        "families": {family: sum(w["family"] == family for w in worlds) for family in ACTOR_FAMILIES},
        "identifiers": len(ACTOR_IDENTIFIERS),
        "world_pairing": "each semantic world has original and swapped assignments for both tasks",
        "logical_owner_balance": "90/90 overall and 15/15 within each identifier family",
        "correct_actor_position": "180 first / 180 second across paired interpretation rows",
        "correct_value_position": "180 first / 180 second across paired application rows",
        "semantic_value_unchanged_under_swap": True,
    }


def validate_lexical_representation_rows(worlds, rows):
    if len(worlds) != 180 or len(rows) != 3600:
        raise ValueError("Expected 180 worlds and 3,600 representation rows")
    world_map = {w["world_id"]: w for w in worlds}
    if len(world_map) != 180:
        raise ValueError("World identifiers must be unique")
    keys = {(r["world_id"], r["representation"], r["assignment"], r["task"]) for r in rows}
    expected_keys = {(w["world_id"], rep, assignment, task)
                     for w in worlds for rep in REPRESENTATIONS
                     for assignment in ASSIGNMENTS for task in TASKS}
    if keys != expected_keys or len(keys) != len(rows):
        raise ValueError("Representation rows are not fully paired by world, encoding, assignment, and task")
    for row in rows:
        world = world_map[row["world_id"]]
        expected = make_lexical_row(world, row["assignment"], row["task"], row["representation"], 2)
        if row != expected:
            raise ValueError("Representation row changed policy semantics, query, candidates, or values")
        decoded = decode_lexical_policy(row["policy_text"], row["representation"])
        if decoded != row["policy"]:
            raise ValueError("Representation renderer changed canonical authorization semantics")
    normalized_prompts = {}
    for row in rows:
        key = (row["world_id"], row["assignment"], row["task"])
        normalized_prompts.setdefault(key, set()).add(row["prompt"].replace(row["policy_text"], "<POLICY>"))
    if any(len(prompts) != 1 for prompts in normalized_prompts.values()):
        raise ValueError("Query, actor order, or proposed values changed across representations")
    return {"passed": True, "worlds": len(worlds), "rows": len(rows),
            "representations": list(REPRESENTATIONS), "assignments": list(ASSIGNMENTS),
            "paired_worlds": True, "semantic_equivalence": True,
            "same_queries_values_and_candidates": True}
