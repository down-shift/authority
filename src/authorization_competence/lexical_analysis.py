"""World-paired analysis for the lexical actor-name symmetry calibration."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
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


def lexical_representation_analysis(rows, seed=20261005):
    """Analyze five encodings after averaging the two matched name assignments."""
    from authorization_competence.design import REPRESENTATIONS

    by_pair = defaultdict(dict)
    for row in rows:
        if row.get("stage") != 2:
            continue
        by_pair[(row["world_id"], row["representation"], row["task"])][row["assignment"]] = row
    if len(by_pair) != 180 * len(REPRESENTATIONS) * len(TASKS) or any(
        set(assignments) != set(ASSIGNMENTS) for assignments in by_pair.values()
    ):
        raise ValueError("Stage 2 requires both name assignments for every world, representation, and task")
    per_cell = {}
    world_rep_answers = defaultdict(dict)
    world_rep_margins = defaultdict(dict)
    raw_app_position = defaultdict(list)
    for task_index, task in enumerate(TASKS):
        per_cell[task] = {}
        for rep_index, rep in enumerate(REPRESENTATIONS):
            pairs = []
            raw = []
            for (world_id, row_rep, row_task), assignments in by_pair.items():
                if row_rep != rep or row_task != task:
                    continue
                original, swapped = (assignments[a] for a in ASSIGNMENTS)
                m0, m1 = float(original["margin"]), float(swapped["margin"])
                msym = (m0 + m1) / 2
                correct = float(msym > 0)
                pairs.append((world_id, msym, correct))
                world_rep_answers[(world_id, task)][rep] = correct
                world_rep_margins[(world_id, task)][rep] = msym
                raw.extend([float(m0 > 0), float(m1 > 0)])
                if task == "application":
                    raw_app_position[original["correct_value_position"]].append((world_id, float(m0 > 0), float(m1 > 0)))
            per_cell[task][rep] = {
                "symmetrized_accuracy": boot([x[2] for x in pairs], seed + task_index * 100 + rep_index),
                "symmetrized_margin": boot([x[1] for x in pairs], seed + task_index * 100 + rep_index + 20),
                "raw_accuracy_both_assignments": boot(
                    [np.mean([float(by_pair[(world_id, rep, task)][a]["margin"] > 0)
                              for a in ASSIGNMENTS]) for world_id, _, _ in pairs],
                    seed + task_index * 100 + rep_index + 40),
                "swap_flip_rate": boot([
                    float(np.sign(float(by_pair[(world_id, rep, task)]["original"]["margin"])) !=
                          np.sign(float(by_pair[(world_id, rep, task)]["swapped"]["margin"])))
                    for world_id, _, _ in pairs], seed + task_index * 100 + rep_index + 60),
                "n_worlds": len(pairs),
            }
    disagreement = {}
    dispersion = {}
    for task_index, task in enumerate(TASKS):
        answers = [world_rep_answers[(world_id, task)] for world_id in sorted({k[0] for k in world_rep_answers if k[1] == task})]
        margins = [world_rep_margins[(world_id, task)] for world_id in sorted({k[0] for k in world_rep_margins if k[1] == task})]
        disagreement_values = [float(len(set(x.values())) > 1) for x in answers]
        dispersion_values = [float(np.var(list(x.values()))) for x in margins]
        disagreement[task] = boot(disagreement_values, seed + 200 + task_index)
        dispersion[task] = boot(dispersion_values, seed + 210 + task_index)
    pairwise = {task: {} for task in TASKS}
    for task_index, task in enumerate(TASKS):
        for contrast_index, (rep_a, rep_b) in enumerate(combinations(REPRESENTATIONS, 2)):
            worlds = sorted({key[0] for key in world_rep_margins if key[1] == task})
            margin_diffs = [world_rep_margins[(w, task)][rep_a] - world_rep_margins[(w, task)][rep_b] for w in worlds]
            accuracy_diffs = [world_rep_answers[(w, task)][rep_a] - world_rep_answers[(w, task)][rep_b] for w in worlds]
            pairwise[task][f"{rep_a}_minus_{rep_b}"] = {
                "symmetrized_margin_difference": boot(margin_diffs, seed + 300 + task_index * 100 + contrast_index),
                "symmetrized_accuracy_difference": boot(accuracy_diffs, seed + 400 + task_index * 100 + contrast_index),
            }
    pos_by = {}
    for pos in (0, 1):
        entries = raw_app_position[pos]
        by_world = defaultdict(list)
        for world_id, original_correct, swapped_correct in entries:
            by_world[world_id].append((original_correct + swapped_correct) / 2)
        # Position is repeated for five representations; average within world first.
        values = [np.mean(x) for x in by_world.values()]
        pos_by[pos] = boot(values, seed + 500 + pos)
    position_gap = abs(pos_by[0]["mean"] - pos_by[1]["mean"])
    score_issue = any(r.get("score_audit", {}).get("candidate_token_count_mismatch", False) or
                      r.get("score_audit", {}).get("boundary_error", False) for r in rows)
    return {
        "worlds": len({r["world_id"] for r in rows}), "rows": len(rows),
        "tasks": per_cell,
        "categorical_disagreement_fraction": disagreement,
        "within_world_margin_variance": dispersion,
        "pairwise_representation_contrasts": pairwise,
        "application_position": {"accuracy_by_correct_value_position": {
            "first": pos_by[0], "second": pos_by[1]}, "accuracy_gap": position_gap},
        "candidate_scoring_audit_passed": not score_issue,
        "analysis_note": "All margins and representation outcomes are paired-name symmetrized; bootstrap resamples semantic worlds.",
    }
