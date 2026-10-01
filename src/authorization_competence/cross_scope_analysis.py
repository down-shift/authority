"""World-paired estimands for irrelevant-scope authorization interference."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import numpy as np

from authorization_competence.analysis import boot
from authorization_competence.design import REPRESENTATIONS
from authorization_competence.lexical_symmetry import ASSIGNMENTS, TASKS
from authorization_competence.cross_scope import CONDITIONS


def _summary(values, seed):
    return boot(values, seed)


def cross_scope_analysis(rows, seed=20261005):
    indexed = defaultdict(dict)
    for row in rows:
        indexed[(row["world_id"], row["representation"], row["scope_condition"], row["task"])][row["assignment"]] = row
    expected_groups = 180 * len(REPRESENTATIONS) * len(CONDITIONS) * len(TASKS)
    if len(indexed) != expected_groups or any(set(x) != set(ASSIGNMENTS) for x in indexed.values()):
        raise ValueError("Each world/representation/condition/task needs both lexical assignments")

    paired = defaultdict(dict)
    for world_id in sorted({key[0] for key in indexed}):
        for rep in REPRESENTATIONS:
            for task in TASKS:
                for condition in CONDITIONS:
                    group = indexed[(world_id, rep, condition, task)]
                    margins = {a: float(group[a]["margin"]) for a in ASSIGNMENTS}
                    sym = sum(margins.values()) / 2
                    paired[(world_id, rep, task)][condition] = {
                        "margin": sym, "correct": float(sym > 0),
                        "original_margin": margins["original"], "swapped_margin": margins["swapped"],
                        "order": group["original"]["queried_scope_position"],
                    }

    metrics = {"worlds": len({key[0] for key in indexed}), "rows": len(rows),
               "tasks": {}, "representation_interactions": {}, "scope_order": {},
               "candidate_scoring_audit_passed": True}
    world_cells_by_task = {}
    for task_index, task in enumerate(TASKS):
        per_rep = {}
        world_overall = defaultdict(dict)
        for rep_index, rep in enumerate(REPRESENTATIONS):
            cell_by_world = {}
            for world_id in sorted({key[0] for key in indexed}):
                c, x = (paired[(world_id, rep, task)][condition] for condition in CONDITIONS)
                delta = x["margin"] - c["margin"]
                cell_by_world[world_id] = {
                    "margin_congruent": c["margin"], "margin_conflicting": x["margin"],
                    "delta_scope": delta, "correct_congruent": c["correct"],
                    "correct_conflicting": x["correct"],
                    "harm": float(c["correct"] == 1 and x["correct"] == 0),
                    "rescue": float(c["correct"] == 0 and x["correct"] == 1),
                    "scope_position": c["order"],
                    "delta_original": x["original_margin"] - c["original_margin"],
                    "delta_swapped": x["swapped_margin"] - c["swapped_margin"],
                    "correct_congruent_original": float(c["original_margin"] > 0),
                    "correct_congruent_swapped": float(c["swapped_margin"] > 0),
                    "correct_conflicting_original": float(x["original_margin"] > 0),
                    "correct_conflicting_swapped": float(x["swapped_margin"] > 0),
                }
                world_overall[world_id][rep] = cell_by_world[world_id]
            per_rep[rep] = {
                "mean_delta_scope": _summary([v["delta_scope"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index),
                "accuracy_congruent": _summary([v["correct_congruent"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 20),
                "accuracy_conflicting": _summary([v["correct_conflicting"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 40),
                "categorical_harm_rate": _summary([v["harm"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 60),
                "categorical_rescue_rate": _summary([v["rescue"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 80),
                "net_categorical_harm": _summary([v["harm"] - v["rescue"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 90),
                "mean_delta_original_assignment": _summary([v["delta_original"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 100),
                "mean_delta_swapped_assignment": _summary([v["delta_swapped"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 120),
                "accuracy_congruent_original_assignment": _summary([v["correct_congruent_original"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 140),
                "accuracy_conflicting_original_assignment": _summary([v["correct_conflicting_original"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 160),
                "accuracy_congruent_swapped_assignment": _summary([v["correct_congruent_swapped"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 180),
                "accuracy_conflicting_swapped_assignment": _summary([v["correct_conflicting_swapped"] for v in cell_by_world.values()], seed + task_index * 100 + rep_index + 200),
                "n_worlds": len(cell_by_world),
            }
        world_means = {}
        for world_id, rep_cells in world_overall.items():
            world_means[world_id] = {key: float(np.mean([x[key] for x in rep_cells.values()]))
            for key in ("delta_scope", "correct_congruent", "correct_conflicting", "harm", "rescue",
                         "delta_original", "delta_swapped")}
            world_means[world_id]["net_harm"] = world_means[world_id]["harm"] - world_means[world_id]["rescue"]
        metrics["tasks"][task] = {
            "by_representation": per_rep,
            "mean_delta_scope": _summary([x["delta_scope"] for x in world_means.values()], seed + task_index),
            "accuracy_congruent": _summary([x["correct_congruent"] for x in world_means.values()], seed + task_index + 1),
            "accuracy_conflicting": _summary([x["correct_conflicting"] for x in world_means.values()], seed + task_index + 2),
            "categorical_harm_rate": _summary([x["harm"] for x in world_means.values()], seed + task_index + 3),
            "categorical_rescue_rate": _summary([x["rescue"] for x in world_means.values()], seed + task_index + 4),
            "net_categorical_harm": _summary([x["net_harm"] for x in world_means.values()], seed + task_index + 8),
            "mean_delta_original_assignment": _summary([x["delta_original"] for x in world_means.values()], seed + task_index + 5),
            "mean_delta_swapped_assignment": _summary([x["delta_swapped"] for x in world_means.values()], seed + task_index + 6),
            "lexical_swap_interaction": _summary([x["delta_swapped"] - x["delta_original"] for x in world_means.values()], seed + task_index + 7),
            "n_worlds": len(world_means),
        }
        world_cells_by_task[task] = world_overall

        order_values = {}
        for order_index, order in enumerate(("first", "second")):
            group = [world_means[w] for w in world_means
                     if paired[(w, REPRESENTATIONS[0], task)]["congruent"]["order"] == order]
            order_values[order] = {
                "mean_delta_scope": _summary([x["delta_scope"] for x in group], seed + task_index * 10 + order_index),
                "accuracy_congruent": _summary([x["correct_congruent"] for x in group], seed + task_index * 10 + order_index + 2),
                "accuracy_conflicting": _summary([x["correct_conflicting"] for x in group], seed + task_index * 10 + order_index + 4),
                "categorical_harm_rate": _summary([x["harm"] for x in group], seed + task_index * 10 + order_index + 6),
                "categorical_rescue_rate": _summary([x["rescue"] for x in group], seed + task_index * 10 + order_index + 8),
                "net_categorical_harm": _summary([x["harm"] - x["rescue"] for x in group], seed + task_index * 10 + order_index + 10),
                "n_worlds": len(group),
            }
        scope_order_diffs = []
        for world_id, values in world_means.items():
            order = paired[(world_id, REPRESENTATIONS[0], task)]["congruent"]["order"]
            scope_order_diffs.append((order, values["delta_scope"]))
        # Contrast is second-position delta minus first-position delta; bootstrap
        # worlds within their randomized order groups.
        first = [d for order, d in scope_order_diffs if order == "first"]
        second = [d for order, d in scope_order_diffs if order == "second"]
        rng = np.random.default_rng(seed + task_index + 900)
        draws = 4000
        first_ix = rng.integers(0, len(first), (draws, len(first)))
        second_ix = rng.integers(0, len(second), (draws, len(second)))
        differences = np.mean(np.asarray(second)[second_ix], axis=1) - np.mean(np.asarray(first)[first_ix], axis=1)
        metrics["scope_order"][task] = {
            "by_queried_scope_position": order_values,
            "delta_second_minus_first": {"mean": float(np.mean(second) - np.mean(first)),
                                          "ci95": [float(x) for x in np.quantile(differences, [.025, .975])],
                                          "n_worlds": len(first) + len(second)},
            "bootstrap_unit": "world, resampled within randomized queried-scope-position strata",
        }

    # Paired representation interactions: differences between each pair of
    # representation-specific scope deltas within the same semantic world.
    for task_index, task in enumerate(TASKS):
        contrasts = {}
        for contrast_index, (rep_a, rep_b) in enumerate(combinations(REPRESENTATIONS, 2)):
            diffs = []
            for world_id in world_cells_by_task[task]:
                a = world_cells_by_task[task][world_id][rep_a]["delta_scope"]
                b = world_cells_by_task[task][world_id][rep_b]["delta_scope"]
                diffs.append(a - b)
            contrasts[f"{rep_a}_minus_{rep_b}"] = _summary(diffs, seed + 1000 + task_index * 100 + contrast_index)
        metrics["representation_interactions"][task] = contrasts

    metrics["candidate_scoring_audit_passed"] = not any(
        row.get("score_audit", {}).get("candidate_token_count_mismatch", False) or
        row.get("score_audit", {}).get("boundary_error", False) or
        row.get("score_audit", {}).get("boundary_overlap", False) or
        row.get("score_audit", {}).get("prompt_prefix_retokenized", False) for row in rows)
    metrics["bootstrap_unit"] = "semantic world"
    metrics["categorical_rule"] = "margin > 0 is correct; margin <= 0 is incorrect"
    return metrics
