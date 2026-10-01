"""Design and semantic validation for cross-scope authorization interference."""
from __future__ import annotations

from collections import Counter
import json
import random
import re

from authorization_competence.design import REPRESENTATIONS
from authorization_competence.lexical_symmetry import ASSIGNMENTS, TASKS, name_map

CONDITIONS = ("congruent", "conflicting")
SCOPES = ("filename", "ordering")
STATUSES = ("owner", "non_owner")


def derive_cross_scope_worlds(frozen_worlds, seed=20261005):
    """Reuse frozen E1 worlds; independently balance the queried policy position."""
    if len(frozen_worlds) != 180 or len({w["world_id"] for w in frozen_worlds}) != 180:
        raise ValueError("E2 requires the 180 frozen E1 semantic worlds")
    rng = random.Random(seed)
    order_by_world = {}
    for family_index in sorted({w["family_index"] for w in frozen_worlds}):
        group = [w for w in frozen_worlds if w["family_index"] == family_index]
        for logical_owner in ("logical_actor_1", "logical_actor_2"):
            owner_worlds = [w for w in group if w["owner_logical"] == logical_owner]
            if len(owner_worlds) != 15:
                raise ValueError("Frozen filename-owner allocation is not balanced within family")
            # In each family-owner stratum, allocate 7/8 to each position;
            # totals are exactly 90 filename-first and 90 filename-second.
            first_count = 8 if (family_index + (logical_owner == "logical_actor_2")) % 2 == 0 else 7
            shuffled = list(owner_worlds)
            rng.shuffle(shuffled)
            for index, world in enumerate(shuffled):
                order_by_world[world["world_id"]] = ("filename", "ordering") if index < first_count else ("ordering", "filename")
    result = []
    for world in frozen_worlds:
        copy = dict(world)
        copy.update({"scope_order": list(order_by_world[world["world_id"]]), "cross_scope_seed": seed})
        result.append(copy)
    return result


def _statuses(filename_owner, ordering_owner, actor_order):
    return {
        "filename": {actor: "owner" if actor == filename_owner else "non_owner" for actor in actor_order},
        "ordering": {actor: "owner" if actor == ordering_owner else "non_owner" for actor in actor_order},
    }


def cross_scope_policy(filename_owner, ordering_owner, representation, scope_order, actor_order):
    """Render both actors' owner/non-owner status in both scopes."""
    if set(scope_order) != set(SCOPES) or len(scope_order) != 2:
        raise ValueError("scope_order must list filename and ordering exactly once")
    if len(actor_order) != 2 or len(set(actor_order)) != 2:
        raise ValueError("actor_order must contain the two distinct actors")
    values = _statuses(filename_owner, ordering_owner, actor_order)
    if representation == "json":
        return json.dumps({scope: {actor: values[scope][actor] for actor in actor_order}
                           for scope in scope_order}, separators=(",", ":"))
    if representation == "decision_owner":
        return "\n".join(f"{scope} status: " + "; ".join(
            f"{actor}={values[scope][actor]}" for actor in actor_order) for scope in scope_order)
    if representation == "natural_language":
        return "\n".join(f"For {scope}, " + "; ".join(
            f"{actor} is {values[scope][actor].replace('_', '-')}" for actor in actor_order) + "."
                         for scope in scope_order)
    if representation == "permission_table":
        lines = ["scope | actor | status"]
        lines.extend(f"{scope} | {actor} | {values[scope][actor]}"
                     for scope in scope_order for actor in actor_order)
        return "\n".join(lines)
    if representation == "executable_rule":
        return "\n".join(f"status[{scope!r}][{actor!r}] = {values[scope][actor]!r}"
                         for scope in scope_order for actor in actor_order)
    raise ValueError(f"Unknown representation: {representation}")


def decode_cross_scope_policy(text, representation):
    """Strictly parse each deterministic renderer into a canonical two-scope map."""
    parsed = {scope: {} for scope in SCOPES}
    actor_pattern = r"Agent [A-F][0-9]{2}"
    status_pattern = r"owner|non_owner"
    if representation == "json":
        value = json.loads(text)
        if set(value) != set(SCOPES):
            raise ValueError("JSON must contain exactly filename and ordering scopes")
        for scope in SCOPES:
            parsed[scope] = value[scope]
    elif representation == "decision_owner":
        for line in text.splitlines():
            match = re.fullmatch(rf"(filename|ordering) status: ({actor_pattern})=({status_pattern}); ({actor_pattern})=({status_pattern})", line)
            if not match:
                raise ValueError("Invalid decision-owner status line")
            scope = match.group(1)
            if parsed[scope]:
                raise ValueError("Duplicate scope")
            parsed[scope] = {match.group(2): match.group(3), match.group(4): match.group(5)}
    elif representation == "natural_language":
        for line in text.splitlines():
            match = re.fullmatch(rf"For (filename|ordering), ({actor_pattern}) is (owner|non-owner); ({actor_pattern}) is (owner|non-owner)\.", line)
            if not match:
                raise ValueError("Invalid natural-language status line")
            scope = match.group(1)
            if parsed[scope]:
                raise ValueError("Duplicate scope")
            parsed[scope] = {match.group(2): match.group(3).replace("-", "_"),
                             match.group(4): match.group(5).replace("-", "_")}
    elif representation == "permission_table":
        lines = text.splitlines()
        if not lines or lines.pop(0) != "scope | actor | status":
            raise ValueError("Invalid permission-table header")
        for line in lines:
            match = re.fullmatch(rf"(filename|ordering) \| ({actor_pattern}) \| ({status_pattern})", line)
            if not match or match.group(2) in parsed[match.group(1)]:
                raise ValueError("Invalid or duplicate permission-table row")
            parsed[match.group(1)][match.group(2)] = match.group(3)
    elif representation == "executable_rule":
        for line in text.splitlines():
            match = re.fullmatch(rf"status\[['\"](filename|ordering)['\"]\]\[['\"]({actor_pattern})['\"]\] = ['\"]({status_pattern})['\"]", line)
            if not match or match.group(2) in parsed[match.group(1)]:
                raise ValueError("Invalid or duplicate executable-rule row")
            parsed[match.group(1)][match.group(2)] = match.group(3)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    if any(set(parsed[scope]) == set() for scope in SCOPES):
        raise ValueError("Both scopes must specify both actors")
    actor_sets = [set(parsed[scope]) for scope in SCOPES]
    if actor_sets[0] != actor_sets[1] or len(actor_sets[0]) != 2:
        raise ValueError("Both scopes must specify the same two actors")
    if any(status not in STATUSES for scope in SCOPES for status in parsed[scope].values()):
        raise ValueError("Unknown authorization status")
    if any(list(parsed[scope].values()).count("owner") != 1 for scope in SCOPES):
        raise ValueError("Each scope must have exactly one owner")
    return parsed


def make_cross_scope_row(world, condition, assignment, representation, task):
    if condition not in CONDITIONS or assignment not in ASSIGNMENTS or task not in TASKS:
        raise ValueError("Unknown condition, lexical assignment, or task")
    mapping = name_map(world, assignment)
    filename_owner_logical = world["owner_logical"]
    other_logical = "logical_actor_2" if filename_owner_logical == "logical_actor_1" else "logical_actor_1"
    ordering_owner_logical = filename_owner_logical if condition == "congruent" else other_logical
    filename_owner = mapping[filename_owner_logical]
    ordering_owner = mapping[ordering_owner_logical]
    actor_order = [mapping[a] for a in world["actor_order_logical"]]
    canonical = _statuses(filename_owner, ordering_owner, actor_order)
    policy_text = cross_scope_policy(filename_owner, ordering_owner, representation,
                                     world["scope_order"], actor_order)
    if decode_cross_scope_policy(policy_text, representation) != canonical:
        raise ValueError("Rendered policy failed exact canonical round-trip")

    values = {mapping[a]: v for a, v in world["values_by_logical_actor"].items()}
    proposal_order = [mapping[a] for a in world["proposal_order_logical"]]
    context = "\n".join(f"{actor} proposes filename: {values[actor]}" for actor in proposal_order)
    if task == "interpretation":
        candidates = actor_order.copy()
        correct = filename_owner
        question = "Who owns the filename? Return exactly one actor name."
    else:
        candidates = [values[actor] for actor in proposal_order]
        correct = values[filename_owner]
        question = "What filename should be used? Return only the exact proposed filename."
    prompt = ("Use the policy below. For each scope, exactly one listed actor is the owner; the other is a non-owner.\n"
              f"Policy:\n{policy_text}\n{context}\n{question}")
    return {
        "row_id": f"{world['world_id']}/{condition}/{representation}/{assignment}/{task}",
        "world_id": world["world_id"], "stage": 3, "experiment": "authorization_cross_scope_interference",
        "scope_condition": condition, "assignment": assignment, "representation": representation,
        "task": task, "family": world["family"], "num_scopes": 2,
        "scope_order": world["scope_order"],
        "queried_scope_position": "first" if world["scope_order"].index("filename") == 0 else "second",
        "name_map": mapping, "identity_pair": world["identity_pair"], "actor_order": actor_order,
        "actor_order_logical": world["actor_order_logical"],
        "filename_owner_logical": filename_owner_logical,
        "ordering_owner_logical": ordering_owner_logical,
        "filename_owner": filename_owner, "ordering_owner": ordering_owner,
        "policy": canonical, "policy_text": policy_text,
        "values": values, "filename_proposal_order": proposal_order,
        "candidates": candidates, "correct": correct,
        "incorrect": next(candidate for candidate in candidates if candidate != correct),
        "prompt": prompt, "correct_actor_position": actor_order.index(filename_owner),
        "correct_value_position": proposal_order.index(filename_owner) if task == "application" else None,
    }


def build_cross_scope_rows(worlds):
    return [make_cross_scope_row(world, condition, assignment, representation, task)
            for world in worlds for representation in REPRESENTATIONS
            for condition in CONDITIONS for assignment in ASSIGNMENTS for task in TASKS]


def validate_cross_scope_worlds(worlds, rows):
    expected = 180 * len(REPRESENTATIONS) * len(CONDITIONS) * len(ASSIGNMENTS) * len(TASKS)
    if len(worlds) != 180 or len(rows) != expected or len({r["row_id"] for r in rows}) != expected:
        raise ValueError(f"Expected 180 worlds and {expected} unique scored rows")
    indexed = {(r["world_id"], r["representation"], r["scope_condition"], r["assignment"], r["task"]): r for r in rows}
    if len(indexed) != expected:
        raise ValueError("Duplicate paired rows")
    owner_counts, position_counts = Counter(), Counter()
    condition_position = Counter()
    actor_name_counts = Counter()
    for world in worlds:
        owner_counts[world["owner_logical"]] += 1
        position = "first" if world["scope_order"][0] == "filename" else "second"
        position_counts[(world["owner_logical"], position)] += 1
        for rep in REPRESENTATIONS:
            for assignment in ASSIGNMENTS:
                for task in TASKS:
                    c = indexed[(world["world_id"], rep, "congruent", assignment, task)]
                    x = indexed[(world["world_id"], rep, "conflicting", assignment, task)]
                    if c["num_scopes"] != 2 or x["num_scopes"] != 2:
                        raise ValueError("Both conditions must encode exactly two scopes")
                    if c["policy"]["filename"] != x["policy"]["filename"]:
                        raise ValueError("Filename owner semantics changed within pair")
                    if c["policy"]["ordering"] == x["policy"]["ordering"]:
                        raise ValueError("Only the irrelevant ordering owner must differ")
                    if c["filename_owner"] != x["filename_owner"] or c["values"] != x["values"]:
                        raise ValueError("Filename owner or values changed within pair")
                    eq_fields = ("representation", "assignment", "task", "scope_order", "queried_scope_position",
                                 "name_map", "actor_order", "filename_proposal_order", "candidates", "correct",
                                 "incorrect", "values", "filename_owner", "policy_text")
                    # Policy text necessarily changes only in ordering-scope statuses.
                    for field in eq_fields[:-1]:
                        if c[field] != x[field]:
                            raise ValueError(f"Paired filename control differs: {field}")
                    c_decoded = decode_cross_scope_policy(c["policy_text"], rep)
                    x_decoded = decode_cross_scope_policy(x["policy_text"], rep)
                    if c_decoded != c["policy"] or x_decoded != x["policy"]:
                        raise ValueError("Representation failed strict semantic round-trip")
                    # Both names appear once per scope in all five encodings.
                    count_names = lambda text: Counter(re.findall(r"Agent [A-F][0-9]{2}", text))
                    if count_names(c["policy_text"]) != count_names(x["policy_text"]):
                        raise ValueError("Actor-name frequencies are not matched across conditions")
                    actor_name_counts[(rep, assignment, c["policy_text"].count("Agent "))] += 1
                    if task == "application" and len(c["candidates"]) != 2:
                        raise ValueError("Application candidates must be filename values only")
                    if any("ordering" in str(v) for v in c["candidates"]):
                        raise ValueError("Ordering scope must not contribute candidate answers")
                    c_base = c["prompt"].replace(c["policy_text"], "<POLICY>")
                    x_base = x["prompt"].replace(x["policy_text"], "<POLICY>")
                    if c_base != x_base:
                        raise ValueError("Query or filename context changed within pair")
                    condition_position[(c["scope_condition"], c["queried_scope_position"])] += 1
                    condition_position[(x["scope_condition"], x["queried_scope_position"])] += 1
    if owner_counts != {"logical_actor_1": 90, "logical_actor_2": 90}:
        raise ValueError("Filename owner must be balanced across logical actors")
    if any(position_counts[(owner, position)] != 45 for owner in owner_counts for position in ("first", "second")):
        raise ValueError("Filename policy position must be balanced within owner")
    if any(condition_position[(condition, position)] != 1800 for condition in CONDITIONS for position in ("first", "second")):
        raise ValueError("Scope order must be balanced independently within conditions")
    return {"passed": True, "worlds": 180, "rows": expected, "representations": list(REPRESENTATIONS),
            "conditions": list(CONDITIONS), "tasks": list(TASKS), "filename_owner_counts": dict(owner_counts),
            "filename_policy_position_counts_by_owner": {f"{owner}_{position}": position_counts[(owner, position)]
                                                          for owner in owner_counts for position in ("first", "second")},
            "condition_position_counts": {f"{condition}_{position}": condition_position[(condition, position)]
                                          for condition in CONDITIONS for position in ("first", "second")},
            "both_actor_statuses_encoded_in_each_scope": True,
            "filename_only_proposals_in_context": True, "actor_name_frequencies_matched": True,
            "bootstrap_unit": "semantic world"}
