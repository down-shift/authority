"""World-paired analysis for E3 canonicalization mitigation."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from authorization_competence.analysis import boot
from authorization_competence.canonicalization import APPLICATION_ORDERS, CANONICALIZATION_ARMS
from authorization_competence.cross_scope import CONDITIONS
from authorization_competence.design import REPRESENTATIONS
from authorization_competence.lexical_symmetry import ASSIGNMENTS, TASKS


def summary(values, seed):
    return boot(values, seed)


def canonicalization_analysis(conversion_rows, answer_rows, seed=20261006):
    conversions = {(r["world_id"], r["representation"], r["scope_condition"], r["assignment"]): r
                   for r in conversion_rows}
    answer_cells = defaultdict(dict)
    for row in answer_rows:
        key = (row["world_id"], row["representation"], row["scope_condition"], row["arm"], row["task"], row["assignment"])
        if row["task"] == "application":
            answer_cells[key][row["candidate_order"]] = row
        else:
            answer_cells[key]["none"] = row
    worlds = sorted({r["world_id"] for r in conversion_rows})
    expected_conversion = 180 * len(REPRESENTATIONS) * len(CONDITIONS) * len(ASSIGNMENTS)
    if len(conversion_rows) != expected_conversion or len(worlds) != 180:
        raise ValueError("E3 requires all 180 worlds and 3,600 conversion cases")
    if len(answer_cells) != expected_conversion * len(TASKS) * len(CANONICALIZATION_ARMS):
        raise ValueError("Incomplete paired answer arms")

    conversion_metrics = {"overall": {}, "by_representation": {}, "by_condition": {}}
    conversion_measures = ("exact_conversion_correct", "filename_owner_correct", "ordering_owner_correct")
    def conversion_value(row, measure):
        if measure == "exact_conversion_correct": return float(row["selected_ir"] == row["gold_ir"])
        field = "filename_owner" if measure == "filename_owner_correct" else "ordering_owner"
        return float(row["selected_ir"].get(field) == row["gold_ir"][field])
    for measure_index, measure in enumerate(conversion_measures):
        by_world = defaultdict(list)
        for row in conversion_rows:
            by_world[row["world_id"]].append(conversion_value(row, measure))
        conversion_metrics["overall"][measure] = summary([np.mean(by_world[w]) for w in worlds], seed + measure_index)
        for rep_index, rep in enumerate(REPRESENTATIONS):
            rep_world = defaultdict(list)
            for row in conversion_rows:
                if row["representation"] == rep:
                    rep_world[row["world_id"]].append(conversion_value(row, measure))
            conversion_metrics["by_representation"].setdefault(rep, {})[measure] = summary(
                [np.mean(rep_world[w]) for w in worlds], seed + 10 + measure_index * 10 + rep_index)
        for cond_index, condition in enumerate(CONDITIONS):
            cond_world = defaultdict(list)
            for row in conversion_rows:
                if row["scope_condition"] == condition:
                    cond_world[row["world_id"]].append(conversion_value(row, measure))
            conversion_metrics["by_condition"].setdefault(condition, {})[measure] = summary(
                [np.mean(cond_world[w]) for w in worlds], seed + 30 + measure_index * 10 + cond_index)

    # First average the two proposal-order margins within each lexical assignment,
    # then average original/swapped assignments within each semantic world.
    world_arm = defaultdict(dict)
    raw_assignment = {task: defaultdict(dict) for task in TASKS}
    conversion_conditionals = defaultdict(list)
    for task in TASKS:
        for world_id in worlds:
            for rep in REPRESENTATIONS:
                for condition in CONDITIONS:
                    for arm in CANONICALIZATION_ARMS:
                        assignment_values = {}
                        for assignment in ASSIGNMENTS:
                            cell_key = (world_id, rep, condition, arm, task, assignment)
                            cell = answer_cells[cell_key]
                            expected_orders = set(APPLICATION_ORDERS) if task == "application" else {"none"}
                            if set(cell) != expected_orders:
                                raise ValueError("Missing candidate order in E3 answer cell")
                            margins = {order: float(cell[order]["margin"]) for order in expected_orders}
                            assignment_values[assignment] = float(np.mean(list(margins.values())))
                            raw_assignment[task][(world_id, rep, condition, arm)][assignment] = assignment_values[assignment]
                        sym_margin = float(np.mean(list(assignment_values.values())))
                        world_arm[(world_id, rep, task, arm)][condition] = {
                            "margin": sym_margin, "correct": float(sym_margin > 0),
                            "original_margin": assignment_values["original"],
                            "swapped_margin": assignment_values["swapped"],
                        }
                        if arm == "canonicalized":
                            for assignment in ASSIGNMENTS:
                                conv = conversions[(world_id, rep, condition, assignment)]
                                rowkey = (world_id, rep, condition, arm, task, assignment)
                                if task == "application":
                                    correct = float(np.mean([answer_cells[rowkey][o]["margin"] for o in APPLICATION_ORDERS]) > 0)
                                else:
                                    correct = float(answer_cells[rowkey]["none"]["margin"] > 0)
                                conversion_conditionals[(rep, condition, task)].append(
                                    (world_id, float(conv["selected_ir"] == conv["gold_ir"]), correct))

    metrics = {"worlds": len(worlds), "conversion_cases": len(conversion_rows),
               "answer_rows": len(answer_rows), "bootstrap_unit": "semantic world",
               "candidate_order_symmetrization": "average orientation-correct filename margins over frozen and reversed proposal order, then average original/swapped actor-name assignments",
               "conversion": conversion_metrics, "tasks": {}, "representation_interactions": {},
               "candidate_scoring_audit_passed": not any(
                   r.get("score_audit", {}).get(k, False) for r in list(answer_rows) + list(conversion_rows)
                   for k in ("candidate_token_count_mismatch", "boundary_error", "boundary_overlap", "prompt_prefix_retokenized"))}
    world_deltas = {task: {} for task in TASKS}
    for task_index, task in enumerate(TASKS):
        task_metrics = {"by_arm": {}, "interference_reduction": {}, "by_representation": {}}
        for arm_index, arm in enumerate(CANONICALIZATION_ARMS):
            cond_acc, cond_margin = {}, {}
            for condition_index, condition in enumerate(CONDITIONS):
                per_world_acc, per_world_margin = {}, {}
                for world_id in worlds:
                    cells = [world_arm[(world_id, rep, task, arm)][condition] for rep in REPRESENTATIONS]
                    per_world_acc[world_id] = float(np.mean([c["correct"] for c in cells]))
                    per_world_margin[world_id] = float(np.mean([c["margin"] for c in cells]))
                cond_acc[condition] = summary([per_world_acc[w] for w in worlds], seed + 100 + task_index * 100 + arm_index * 20 + condition_index)
                cond_margin[condition] = summary([per_world_margin[w] for w in worlds], seed + 110 + task_index * 100 + arm_index * 20 + condition_index)
            task_metrics["by_arm"][arm] = {"accuracy": cond_acc, "margin": cond_margin}
        reduction_by_world = {}
        raw_delta_by_world, canon_delta_by_world = {}, {}
        for world_id in worlds:
            raw_delta_by_world[world_id] = float(np.mean([
                world_arm[(world_id, rep, task, "raw")]["conflicting"]["margin"] -
                world_arm[(world_id, rep, task, "raw")]["congruent"]["margin"] for rep in REPRESENTATIONS]))
            canon_delta_by_world[world_id] = float(np.mean([
                world_arm[(world_id, rep, task, "canonicalized")]["conflicting"]["margin"] -
                world_arm[(world_id, rep, task, "canonicalized")]["congruent"]["margin"] for rep in REPRESENTATIONS]))
            reduction_by_world[world_id] = canon_delta_by_world[world_id] - raw_delta_by_world[world_id]
            world_deltas[task][world_id] = {"raw": raw_delta_by_world[world_id],
                                            "canonicalized": canon_delta_by_world[world_id],
                                            "interference_reduction": reduction_by_world[world_id]}
        task_metrics["interference_reduction"] = {
            "delta_scope_raw": summary([raw_delta_by_world[w] for w in worlds], seed + 300 + task_index),
            "delta_scope_canonicalized": summary([canon_delta_by_world[w] for w in worlds], seed + 310 + task_index),
            "interference_reduction": summary([reduction_by_world[w] for w in worlds], seed + 320 + task_index),
            "positive_means_mitigation": True,
        }
        task_metrics["by_representation"] = {}
        for rep_index, rep in enumerate(REPRESENTATIONS):
            rep_effects = {}
            for arm in CANONICALIZATION_ARMS:
                cell_world = {}
                for world_id in worlds:
                    cong = world_arm[(world_id, rep, task, arm)]["congruent"]
                    conf = world_arm[(world_id, rep, task, arm)]["conflicting"]
                    cell_world[world_id] = {"delta": conf["margin"] - cong["margin"],
                                            "accuracy_congruent": cong["correct"],
                                            "accuracy_conflicting": conf["correct"],
                                            "harm": float(cong["correct"] == 1 and conf["correct"] == 0),
                                            "rescue": float(cong["correct"] == 0 and conf["correct"] == 1)}
                rep_effects[arm] = {
                    "delta_scope": summary([cell_world[w]["delta"] for w in worlds], seed + 400 + task_index * 100 + rep_index * 2 + (arm == "canonicalized")),
                    "accuracy_congruent": summary([cell_world[w]["accuracy_congruent"] for w in worlds], seed + 500 + task_index * 100 + rep_index * 2 + (arm == "canonicalized")),
                    "accuracy_conflicting": summary([cell_world[w]["accuracy_conflicting"] for w in worlds], seed + 600 + task_index * 100 + rep_index * 2 + (arm == "canonicalized")),
                    "harm_rate": summary([cell_world[w]["harm"] for w in worlds], seed + 700 + task_index * 100 + rep_index * 2 + (arm == "canonicalized")),
                    "rescue_rate": summary([cell_world[w]["rescue"] for w in worlds], seed + 800 + task_index * 100 + rep_index * 2 + (arm == "canonicalized")),
                }
            per_world_reduction = [
                (world_arm[(w, rep, task, "canonicalized")]["conflicting"]["margin"] - world_arm[(w, rep, task, "canonicalized")]["congruent"]["margin"])
                - (world_arm[(w, rep, task, "raw")]["conflicting"]["margin"] - world_arm[(w, rep, task, "raw")]["congruent"]["margin"])
                for w in worlds]
            rep_effects["interference_reduction"] = summary(per_world_reduction, seed + 900 + task_index * 100 + rep_index)
            task_metrics["by_representation"][rep] = rep_effects
        # Raw and canonicalized filename-owner interpretation correctness conditional on exact conversion.
        if task == "interpretation":
            task_metrics["canonicalized_accuracy_by_conversion_correct"] = {}
            for rep_index, rep in enumerate(REPRESENTATIONS):
                vals = [item for cond in CONDITIONS for item in conversion_conditionals[(rep, cond, task)]]
                overall = defaultdict(list)
                for world_id, conv_ok, answer_ok in vals:
                    overall[world_id].append((conv_ok, answer_ok))
                correct_world = [np.mean([a for c, a in overall[w] if c]) if any(c for c, _ in overall[w]) else np.nan for w in worlds]
                selected = [x for x in correct_world if not np.isnan(x)]
                task_metrics["canonicalized_accuracy_by_conversion_correct"][rep] = summary(selected, seed + 950 + rep_index)
        task_metrics["canonicalized_answer_accuracy_given_conversion_correct"] = {}
        for rep_index, rep in enumerate(REPRESENTATIONS):
            vals = [item for cond in CONDITIONS for item in conversion_conditionals[(rep, cond, task)]]
            by_world = defaultdict(list)
            for world_id, conv_ok, answer_ok in vals:
                if conv_ok:
                    by_world[world_id].append(answer_ok)
            nonempty = [np.mean(by_world[w]) for w in worlds if by_world[w]]
            task_metrics["canonicalized_answer_accuracy_given_conversion_correct"][rep] = summary(nonempty, seed + 970 + task_index * 10 + rep_index)
        metrics["tasks"][task] = task_metrics

    # How often the model's canonical conversion is fully or partially correct,
    # and how canonical answer errors relate to a failed conversion.
    metrics["error_decomposition"] = {
        "conversion_accuracy_is_reported_separately": True,
        "answer_accuracy_is_reported_separately": True,
        "canonicalized_answer_accuracy_given_conversion_correct": metrics["tasks"]["application"]["canonicalized_answer_accuracy_given_conversion_correct"],
    }
    metrics["analysis_note"] = "All application margins symmetrize candidate order before lexical assignment. Every interval resamples semantic worlds, keeping all representations and paired conditions together."
    return metrics
