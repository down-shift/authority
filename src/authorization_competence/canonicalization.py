"""Frozen E3 dataset construction and canonical two-scope IR helpers."""
from __future__ import annotations

from collections import Counter
import json

from authorization_competence.design import REPRESENTATIONS
from authorization_competence.lexical_symmetry import ASSIGNMENTS, TASKS
from authorization_competence.cross_scope import CONDITIONS, SCOPES, build_cross_scope_rows, decode_cross_scope_policy

CANONICALIZATION_ARMS = ("raw", "canonicalized")
APPLICATION_ORDERS = ("frozen", "reversed")
IR_PLACEHOLDER = "{{PREDICTED_CANONICAL_IR}}"


def canonical_ir(filename_owner: str, ordering_owner: str) -> dict:
    return {"filename_owner": filename_owner, "ordering_owner": ordering_owner}


def render_ir(ir: dict) -> str:
    if set(ir) != {"filename_owner", "ordering_owner"}:
        raise ValueError("Canonical IR must encode filename and ordering owners separately")
    return json.dumps({"filename_owner": ir["filename_owner"], "ordering_owner": ir["ordering_owner"]}, separators=(",", ":"))


def decode_status_policy(status_policy: dict) -> dict:
    if set(status_policy) != set(SCOPES):
        raise ValueError("Raw policy must preserve both scopes")
    result = {}
    for scope in SCOPES:
        owners = [actor for actor, status in status_policy[scope].items() if status == "owner"]
        if len(owners) != 1 or len(status_policy[scope]) != 2 or any(
            status not in ("owner", "non_owner") for status in status_policy[scope].values()
        ):
            raise ValueError(f"Invalid owner statuses for {scope}")
        result[f"{scope}_owner"] = owners[0]
    return result


def canonical_candidates(actor_order):
    if len(actor_order) != 2 or len(set(actor_order)) != 2:
        raise ValueError("Canonical IR candidates require exactly two actors")
    return [render_ir(canonical_ir(filename_owner, ordering_owner))
            for filename_owner in actor_order for ordering_owner in actor_order]


def conversion_prompt(policy_text):
    return ("Convert the authorization policy below into canonical JSON with exactly two fields: "
            '"filename_owner" and "ordering_owner". Preserve each scope independently. '
            "Return exactly one JSON object and no explanation.\nPolicy:\n" + policy_text + "\nCanonical JSON:")


def _filename_context(values, order):
    return "\n".join(f"{actor} proposes filename: {values[actor]}" for actor in order)


def _question(task):
    if task == "interpretation":
        return "Who owns the filename? Return exactly one actor name."
    return "What filename should be used? Return only the exact proposed filename."


def canonical_answer_prompt(ir_text, values, proposal_order, task):
    return ("Use this canonical two-scope policy. Each field names the owner for that scope.\n"
            f"Canonical policy IR:\n{ir_text}\n{_filename_context(values, proposal_order)}\n{_question(task)}")


def make_e3_datasets(worlds, e2_rows):
    """Build one conversion case and paired raw/canonical answer rows per frozen E2 cell."""
    indexed = {(r["world_id"], r["representation"], r["scope_condition"], r["assignment"], r["task"]): r
               for r in e2_rows}
    expected_e2 = 180 * len(REPRESENTATIONS) * len(CONDITIONS) * len(ASSIGNMENTS) * len(TASKS)
    if len(indexed) != expected_e2 or len(e2_rows) != expected_e2:
        raise ValueError("E3 requires the complete 7,200-row frozen E2 dataset")
    conversion_rows = []
    answer_rows = []
    conversion_keys = set()
    for world in worlds:
        for representation in REPRESENTATIONS:
            for condition in CONDITIONS:
                for assignment in ASSIGNMENTS:
                    source = indexed[(world["world_id"], representation, condition, assignment, "interpretation")]
                    key = (world["world_id"], representation, condition, assignment)
                    if key in conversion_keys:
                        raise ValueError("Duplicate conversion case")
                    conversion_keys.add(key)
                    gold_ir = decode_status_policy(decode_cross_scope_policy(source["policy_text"], representation))
                    conversion_rows.append({
                        "row_id": "/".join((world["world_id"], representation, condition, assignment, "conversion")),
                        "world_id": world["world_id"], "representation": representation,
                        "scope_condition": condition, "assignment": assignment,
                        "actor_order": source["actor_order"], "filename_owner": source["filename_owner"],
                        "ordering_owner": source["ordering_owner"], "source_e2_row_id": source["row_id"],
                        "policy_text": source["policy_text"], "prompt": conversion_prompt(source["policy_text"]),
                        "candidates": canonical_candidates(source["actor_order"]),
                        "gold_ir": gold_ir, "gold_ir_text": render_ir(gold_ir),
                    })
                    for task in TASKS:
                        source_task = indexed[(world["world_id"], representation, condition, assignment, task)]
                        orders = ("none",) if task == "interpretation" else APPLICATION_ORDERS
                        for arm in CANONICALIZATION_ARMS:
                            for order_name in orders:
                                if order_name == "reversed":
                                    proposal_order = list(reversed(source_task["filename_proposal_order"]))
                                else:
                                    proposal_order = list(source_task["filename_proposal_order"])
                                if arm == "raw":
                                    if order_name in ("none", "frozen"):
                                        prompt = source_task["prompt"]
                                    else:
                                        old_context = _filename_context(source_task["values"], source_task["filename_proposal_order"])
                                        new_context = _filename_context(source_task["values"], proposal_order)
                                        if source_task["prompt"].count(old_context) != 1:
                                            raise ValueError("Could not identify frozen E2 filename proposal context")
                                        prompt = source_task["prompt"].replace(old_context, new_context)
                                else:
                                    prompt = canonical_answer_prompt(IR_PLACEHOLDER, source_task["values"], proposal_order, task)
                                candidates = list(source_task["candidates"])
                                if task == "application":
                                    candidates = [source_task["values"][actor] for actor in proposal_order]
                                answer_rows.append({
                                    "row_id": "/".join((world["world_id"], representation, condition, assignment,
                                                       arm, task, order_name)),
                                    "world_id": world["world_id"], "representation": representation,
                                    "scope_condition": condition, "assignment": assignment,
                                    "arm": arm, "task": task, "candidate_order": order_name,
                                    "queried_scope_position": source_task["queried_scope_position"],
                                    "filename_owner": source_task["filename_owner"],
                                    "ordering_owner": source_task["ordering_owner"],
                                    "values": source_task["values"], "filename_proposal_order": proposal_order,
                                    "candidates": candidates, "correct": source_task["correct"],
                                    "incorrect": source_task["incorrect"], "prompt": prompt,
                                    "source_e2_row_id": source_task["row_id"],
                                })
    return conversion_rows, answer_rows


def validate_e3_datasets(worlds, e2_rows, conversion_rows, answer_rows):
    expected_conversion = 180 * len(REPRESENTATIONS) * len(CONDITIONS) * len(ASSIGNMENTS)
    expected_answers = expected_conversion * 6
    if len(conversion_rows) != expected_conversion or len(answer_rows) != expected_answers:
        raise ValueError(f"Expected {expected_conversion} conversions and {expected_answers} answer rows")
    if len({r["row_id"] for r in conversion_rows}) != expected_conversion or len({r["row_id"] for r in answer_rows}) != expected_answers:
        raise ValueError("E3 row IDs must be unique")
    e2 = {(r["row_id"]): r for r in e2_rows}
    conversion = {(r["world_id"], r["representation"], r["scope_condition"], r["assignment"]): r
                  for r in conversion_rows}
    for row in conversion_rows:
        raw_decoded = decode_cross_scope_policy(row["policy_text"], row["representation"])
        if decode_status_policy(raw_decoded) != row["gold_ir"]:
            raise ValueError("Golden IR disagrees with strictly decoded raw policy")
        if row["gold_ir_text"] != render_ir(row["gold_ir"]):
            raise ValueError("Canonical IR serialization mismatch")
        if row["gold_ir_text"] not in row["candidates"]:
            raise ValueError("The legal conversion candidate set omits the correct policy")
    groups = {}
    for row in answer_rows:
        source = e2.get(row["source_e2_row_id"])
        if source is None:
            raise ValueError("Answer row has no source E2 row")
        group_key = (row["world_id"], row["representation"], row["scope_condition"],
                     row["assignment"], row["task"], row["candidate_order"])
        groups.setdefault(group_key, {})[row["arm"]] = row
        if row["filename_owner"] != source["filename_owner"] or row["values"] != source["values"]:
            raise ValueError("E3 changed frozen filename semantics")
        if row["arm"] == "raw" and row["candidate_order"] in ("none", "frozen") and row["prompt"] != source["prompt"]:
            raise ValueError("Raw E3 prompt differs from frozen E2 prompt")
        if row["task"] == "application" and len(row["candidates"]) != 2:
            raise ValueError("Application candidates must be the two filename values only")
    if len(groups) != expected_conversion * 3 or any(set(x) != set(CANONICALIZATION_ARMS) for x in groups.values()):
        raise ValueError("Every query/order cell requires raw and canonicalized answer arms")
    for key, arms in groups.items():
        raw, canon = arms["raw"], arms["canonicalized"]
        if raw["task"] != canon["task"] or raw["candidate_order"] != canon["candidate_order"]:
            raise ValueError("Raw and canonical arms are not paired on task and value order")
        if raw["candidates"] != canon["candidates"] or raw["correct"] != canon["correct"]:
            raise ValueError("Canonicalization changed filename answer values")
        expected_ir = conversion[(raw["world_id"], raw["representation"], raw["scope_condition"], raw["assignment"])]
        if expected_ir["gold_ir"]["filename_owner"] != raw["filename_owner"]:
            raise ValueError("Canonical IR filename owner does not match E2 world")
    counts = Counter((r["task"], r["arm"], r["candidate_order"]) for r in answer_rows)
    if counts != Counter({
        ("interpretation", arm, "none"): expected_conversion for arm in CANONICALIZATION_ARMS
    } | {
        ("application", arm, order): expected_conversion for arm in CANONICALIZATION_ARMS for order in APPLICATION_ORDERS
    }):
        raise ValueError("Task, arm, or candidate-order cell counts are unbalanced")
    return {"passed": True, "worlds": len(worlds), "e2_rows": len(e2_rows),
            "conversion_rows": len(conversion_rows), "answer_rows": len(answer_rows),
            "all_answer_rows_including_conversion": len(conversion_rows) + len(answer_rows),
            "representations": list(REPRESENTATIONS), "conditions": list(CONDITIONS),
            "assignments": list(ASSIGNMENTS), "arms": list(CANONICALIZATION_ARMS),
            "candidate_orders": list(APPLICATION_ORDERS),
            "candidate_order_symmetrization": "orientation-correct margins averaged over frozen and reversed filename proposal order",
            "bootstrap_unit": "semantic world", "gold_used_to_select_ir": False}
