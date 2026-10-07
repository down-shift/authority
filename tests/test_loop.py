"""Guard the autonomous-loop plumbing: the docs/PLAN.md format that
scripts/claim.py and scripts/mark.py depend on, and claim.py's safety checks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {"status", "owner", "claimed_at", "deps", "source", "done-when"}
# "cut" is a terminal status for a step retired by human decision: like "done"
# it is finished and never re-claimed, but it did not satisfy its done-when.
VALID_BASE_STATUSES = {"todo", "claimed", "in_review", "done", "blocked", "cut"}


def _load_claim_module():
    spec = importlib.util.spec_from_file_location("claim", ROOT / "scripts" / "claim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["claim"] = mod
    spec.loader.exec_module(mod)
    return mod


def _plan_blocks(claim):
    return claim.parse((ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8").splitlines())


def test_plan_ledger_parses_and_is_consistent():
    claim = _load_claim_module()
    blocks = _plan_blocks(claim)  # an empty plan (no steps yet) is valid

    ids = [b["id"] for b in blocks]
    assert len(ids) == len(set(ids)), "duplicate step ids in docs/PLAN.md"

    id_set = set(ids)
    for b in blocks:
        missing = REQUIRED_FIELDS - set(b["fields"])
        assert not missing, f"step {b['id']} missing fields: {missing}"
        status = claim.status_of(b)
        base = status.split(" ", 1)[0].split("(", 1)[0]
        assert base in VALID_BASE_STATUSES, f"step {b['id']} has odd status {status!r}"
        for dep in claim.deps_of(b):
            assert dep in id_set, f"step {b['id']} depends on unknown step {dep!r}"


def test_plan_has_a_frontier_or_is_human_gated():
    """The ledger must never truly deadlock: unfinished todo steps remain but
    none is eligible, nothing is in flight, and nothing is blocked for a human."""
    claim = _load_claim_module()
    blocks = _plan_blocks(claim)
    statuses = [claim.status_of(b) for b in blocks]
    if all(s in {"done", "cut"} for s in statuses):  # also true for an empty plan
        return
    in_flight = any(s.startswith(("claimed", "in_review")) for s in statuses)
    human_gated = any(s.startswith("blocked") for s in statuses) or bool(claim.waiting_on_human(blocks))
    assert claim.pick(blocks) is not None or in_flight or human_gated, (
        "docs/PLAN.md is deadlocked: unfinished todo steps but none eligible, "
        "nothing in flight, and nothing blocked for a human to unblock"
    )


def test_parse_and_pick_follow_deps_and_skip_fences():
    claim = _load_claim_module()
    lines = """
```
### X0 — inside a fence, ignored
- status: todo
```
### A1 — first
- status: done
- deps: —
### A2 — second
- status: todo
- deps: A3
### A3 — third
- status: todo
- deps: A1
""".splitlines()
    blocks = claim.parse(lines)
    assert [b["id"] for b in blocks] == ["A1", "A2", "A3"]
    assert claim.pick(blocks)["id"] == "A3"


def test_pick_skips_human_steps():
    claim = _load_claim_module()
    lines = """
### G0 — [HUMAN] Gate 0 decision
- status: todo
- deps: —
### P1.1 — canonical policy
- status: todo
- deps: G0
### R0.1 — related work
- status: todo
- deps: —
""".splitlines()
    blocks = claim.parse(lines)
    assert claim.pick(blocks)["id"] == "R0.1"
    assert claim.waiting_on_human(blocks) == ["G0"]


def test_real_plan_gates_are_human():
    claim = _load_claim_module()
    gates = [b for b in _plan_blocks(claim) if b["id"].startswith("G")]
    assert gates and all(claim.is_human(b) for b in gates)


def test_claim_rejects_unknown_args_before_touching_git(monkeypatch):
    # `claim.py --help` must not perform a real claim and a hard reset.
    claim = _load_claim_module()

    def no_git(*a, **k):
        raise AssertionError("git must not be called")

    monkeypatch.setattr(claim, "git", no_git)
    monkeypatch.setattr(sys, "argv", ["claim.py", "--help"])
    assert claim.main() == 2


def test_claim_refuses_with_dirty_tracked_files(monkeypatch):
    claim = _load_claim_module()
    calls = []

    class R:
        stdout = " M docs/PLAN.md\n"

    def fake_git(*a, **k):
        calls.append(a)
        if a[0] == "status":
            return R()
        raise AssertionError(f"reset/checkout must not run: {a}")

    monkeypatch.setattr(claim, "git", fake_git)
    monkeypatch.setattr(sys, "argv", ["claim.py"])
    assert claim.main() == 2
    assert calls == [("status", "--porcelain", "--untracked-files=no")]
