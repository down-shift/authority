"""Fail closed on unmatched semantics, missing repeats, or mutated prompt text."""
from collections import defaultdict
from dataclasses import fields
from .worlds import World, FAMILIES
from .render import render, CONTRACT
from authority_leakage.progress import tqdm


def audit(rows):
    groups = defaultdict(list)
    for row in tqdm(rows, total=len(rows), desc="Auditing semantic matches", unit="prompt", leave=False):
        groups[row["world_id"]].append(row)
        raw = row["world"]
        w = World(**{f.name: raw[f.name] for f in fields(World)})
        assert raw == w.to_dict(), "Derived ground truth changed"
        assert row["world_id"] == w.world_id and row["task_family"] == w.task_family
        assert row["control_type"] == "depth_ladder" and row["policy_width"] == 1
        assert row["prompt"] == render(w, row["indirection_depth"], row["measurement_type"], row["template_id"]), "Prompt mutation"
        assert row["prompt"].endswith(CONTRACT)
        if row["measurement_type"] == "application":
            assert (row["correct_candidate"], row["incorrect_candidate"]) == (w.correct_value, w.alternative_value)
        else:
            assert row["correct_candidate"] == w.owner
            assert row["incorrect_candidate"] == (w.default if w.owner == w.source else w.source)
    balance = {}
    for world_id, items in groups.items():
        assert len(items) == 10, f"Missing/duplicate repeats: {world_id}"
        assert {(r["indirection_depth"], r["measurement_type"]) for r in items} == {(d,m) for d in range(5) for m in ("comprehension","application")}
        assert all(r["world"] == items[0]["world"] for r in items), "Unmatched world"
        w = items[0]["world"]
        tally = balance.setdefault(w["task_family"], {"source_correct": 0, "default_correct": 0,
                                                       "source_first": 0, "default_first": 0})
        tally["source_correct" if w["owner"] == w["source"] else "default_correct"] += 1
        tally["source_first" if w["source_first"] else "default_first"] += 1
    assert set(balance) == set(FAMILIES)
    assert len({sum(x.values()) for x in balance.values()}) == 1
    assert all(x["source_correct"] == x["default_correct"] and x["source_first"] == x["default_first"] for x in balance.values())
    return {"passed": True, "worlds": len(groups), "examples": len(rows), "balance": balance,
            "template_ids": ["canonical_v1"], "manual_review_required": "Inspect representative prompts before interpreting inference."}
