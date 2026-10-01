"""World-paired analysis for the lexical actor-name symmetry calibration."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from authorization_competence.analysis import boot, BIAS_POSITION_MAX_GAP, COMPETENCE_ACCURACY_MIN
from authorization_competence.lexical_symmetry import ASSIGNMENTS, TASKS, ACTOR_FAMILIES

SWAP_FLIP_MAX = 0.10
FAMILY_ACCURACY_RANGE_MAX = 0.15


def lexical_symmetry_analysis(rows, seed=20261004):
    stage_rows = [r for r in rows if r.get("stage") == 1]
    by_pair = defaultdict(dict)
    for row in stage_rows:
        by_pair[(row["world_id"], row["task"])][row["assignment"]] = row
    if not by_pair or any(set(pair) != set(ASSIGNMENTS) for pair in by_pair.values()):
        raise ValueError("Each world/task requires exactly one original and one swapped row")
    world_meta = {}
    for row in stage_rows:
        world_meta[row["world_id"]] = row

    result = {"worlds": len(world_meta), "rows": len(stage_rows), "tasks": {}, "gate": {}}
    task_worlds = {}
    for task_index, task in enumerate(TASKS):
        pairs = []
        raw_rows = [r for r in stage_rows if r["task"] == task]
        for (world_id, row_task), assignments in sorted(by_pair.items()):
            if row_task != task:
                continue
            original, swapped = (assignments[a] for a in ASSIGNMENTS)
            if original["owner_logical"] != swapped["owner_logical"]:
                raise ValueError("Logical owner changed under actor-name swap")
            if original["values_by_logical_actor"] != swapped["values_by_logical_actor"]:
                raise ValueError("Logical proposals changed under actor-name swap")
            if task == "application" and original["correct"] != swapped["correct"]:
                raise ValueError("Correct application value changed under actor-name swap")
            m_original, m_swapped = float(original["margin"]), float(swapped["margin"])
            family = original["family"]
            if family != swapped["family"]:
                raise ValueError("Identifier family changed inside a matched pair")
            pairs.append({"world_id": world_id, "family": family,
                          "margin_original": m_original, "margin_swapped": m_swapped,
                          "margin_symmetrized": (m_original + m_swapped) / 2,
                          "identity_sensitivity": abs(m_original - m_swapped),
                          "swap_flip": float(np.sign(m_original) != np.sign(m_swapped)),
                          "sym_correct": float((m_original + m_swapped) / 2 > 0),
                          "original_correct": float(m_original > 0),
                          "swapped_correct": float(m_swapped > 0),
                          "correct_position": original["correct_actor_position"] if task == "interpretation" else original["correct_value_position"]})
        task_worlds[task] = pairs
        raw_by_world = defaultdict(list)
        margin_by_world = defaultdict(list)
        for row in raw_rows:
            raw_by_world[row["world_id"]].append(float(float(row["margin"]) > 0))
            margin_by_world[row["world_id"]].append(float(row["margin"]))
        per_assignment = {}
        for assignment_index, assignment in enumerate(ASSIGNMENTS):
            group = [r for r in raw_rows if r["assignment"] == assignment]
            per_assignment[assignment] = {
                "accuracy": boot([float(float(r["margin"]) > 0) for r in group], seed + task_index * 10 + assignment_index),
                "margin": boot([float(r["margin"]) for r in group], seed + task_index * 10 + assignment_index + 2),
                "n_rows": len(group),
            }
        result["tasks"][task] = {
            "raw_accuracy": boot([np.mean(x) for x in raw_by_world.values()], seed + task_index),
            "raw_margin": boot([np.mean(x) for x in margin_by_world.values()], seed + task_index + 1),
            "by_assignment": per_assignment,
            "symmetrized_accuracy": boot([p["sym_correct"] for p in pairs], seed + task_index + 20),
            "symmetrized_margin": boot([p["margin_symmetrized"] for p in pairs], seed + task_index + 21),
            "mean_identity_sensitivity": boot([p["identity_sensitivity"] for p in pairs], seed + task_index + 22),
            "swap_flip_rate": boot([p["swap_flip"] for p in pairs], seed + task_index + 23),
            "n_worlds": len(pairs),
        }

    position = {}
    app_pairs = task_worlds["application"]
    for pos in (0, 1):
        group = [p for p in app_pairs if p["correct_position"] == pos]
        position[f"correct_value_position_{pos + 1}"] = {
            "accuracy": boot([p["sym_correct"] for p in group], seed + 40 + pos),
            "margin": boot([p["margin_symmetrized"] for p in group], seed + 50 + pos),
            "n_worlds": len(group),
        }
    pos_accuracy = [position[f"correct_value_position_{p + 1}"]["accuracy"]["mean"] for p in (0, 1)]
    position_gap = abs(pos_accuracy[0] - pos_accuracy[1])
    result["application_position"] = {"by_correct_value_position": position, "accuracy_gap": position_gap}

    family_result = {}
    for family_index, family in enumerate(ACTOR_FAMILIES):
        group = [p for p in app_pairs if p["family"] == family]
        family_result[family] = {
            "accuracy": boot([p["sym_correct"] for p in group], seed + 60 + family_index),
            "margin": boot([p["margin_symmetrized"] for p in group], seed + 70 + family_index),
            "n_worlds": len(group),
        }
    family_rates = [v["accuracy"]["mean"] for v in family_result.values()]
    family_range = max(family_rates) - min(family_rates)
    result["application_identifier_family"] = {"by_family": family_result, "accuracy_range": family_range,
                                               "predefined_max_range": FAMILY_ACCURACY_RANGE_MAX}

    score_issue = any(r.get("score_audit", {}).get("candidate_token_count_mismatch", False) or
                      r.get("score_audit", {}).get("boundary_error", False) for r in stage_rows)
    interp_acc = result["tasks"]["interpretation"]["symmetrized_accuracy"]["mean"]
    app_acc = result["tasks"]["application"]["symmetrized_accuracy"]["mean"]
    flip = result["tasks"]["application"]["swap_flip_rate"]["mean"]
    failures = []
    if interp_acc < COMPETENCE_ACCURACY_MIN: failures.append("interpretation_symmetrized_accuracy")
    if app_acc < COMPETENCE_ACCURACY_MIN: failures.append("application_symmetrized_accuracy")
    if flip > SWAP_FLIP_MAX: failures.append("application_swap_flip_rate")
    if position_gap > BIAS_POSITION_MAX_GAP: failures.append("application_position_bias")
    if family_range > FAMILY_ACCURACY_RANGE_MAX: failures.append("identifier_family_effect")
    if score_issue: failures.append("candidate_scoring_audit")
    result["gate"] = {
        "passed": not failures,
        "failures": failures,
        "interpretation_symmetrized_accuracy_at_least_0_90": interp_acc >= COMPETENCE_ACCURACY_MIN,
        "application_symmetrized_accuracy_at_least_0_90": app_acc >= COMPETENCE_ACCURACY_MIN,
        "application_swap_flip_rate_at_most_0_10": flip <= SWAP_FLIP_MAX,
        "application_position_gap_at_most_0_15": position_gap <= BIAS_POSITION_MAX_GAP,
        "identifier_family_accuracy_range_at_most_0_15": family_range <= FAMILY_ACCURACY_RANGE_MAX,
        "candidate_scoring_audit_passed": not score_issue,
        "thresholds": {"symmetrized_accuracy_min": COMPETENCE_ACCURACY_MIN,
                       "application_swap_flip_rate_max": SWAP_FLIP_MAX,
                       "application_position_gap_max": BIAS_POSITION_MAX_GAP,
                       "identifier_family_accuracy_range_max": FAMILY_ACCURACY_RANGE_MAX},
    }
    return result
