"""P3.2b T3 arms built on the model's own arm-1 IR: canonicalization in T3 and the solver-enforced proxy."""

from __future__ import annotations

import json

import pytest
from test_authinv_agent import act, episodes, scripted

from authinv import provenance
from authinv.agent.arms import apply_arm, cell_id, load_conversions
from authinv.agent.metrics import agent_metrics, arm_metrics
from authinv.agent.sandbox import outcome, run, start
from authinv.mitigate import canonicalize as canon

pytest.importorskip("cedarpy")


def conv(ep, ir_obj=None, text=None):
    """A conversion record for an episode's cell, scored as the arm-1 runner scores it."""
    out = canon.parse_conversion(text if text is not None else json.dumps(ir_obj))
    if out["conv_category"] != "ir":
        return {"case_id": cell_id(ep), **out, "ir_valid": False, "ir_exact": False, "ir_equivalent": False}
    scores, _ = canon.score_ir(ep.policy, out["ir_text"], ep.principal_types, ep.resource_types)
    return {"case_id": cell_id(ep), **out, **scores}


def gold_convs(eps):
    return {cell_id(e): conv(e, canon.gold_ir(e.policy)) for e in eps}


ALLOW_ALL = {
    "rules": [
        {
            "id": "r1",
            "effect": "allow",
            "principal": {"match": "any"},
            "actions": None,  # filled per policy
            "resource": {"match": "any"},
            "conditions_mode": "all",
            "conditions": [],
        }
    ]
}


def allow_all(policy):
    ir = json.loads(json.dumps(ALLOW_ALL))
    ir["rules"][0]["actions"] = sorted(policy.actions)
    return ir


def requester(eps):
    """The mock agent always attempts the requested call, then responds."""
    index = {}
    for e in eps:
        if not e.done:
            start(e)
            index[(e.messages[0]["content"], e.messages[1]["content"])] = e

    def execute(msgs):
        e = index[(msgs[0]["content"], msgs[1]["content"])]
        return act(e.request.action, resource=e.request.resource.id)

    return scripted(execute)


def test_canon_t3_with_a_perfect_ir_is_the_json_policy_episode():
    eps = episodes(renderings=("nl_statement", "json_policy"))
    ref = {(e.world_id, e.instance): e for e in episodes(renderings=("json_policy",))}
    counts = apply_arm(eps, "canon_t3", gold_convs(eps))
    assert counts["ir_valid"] == len(eps) and counts["no_ir"] == 0
    for e in eps:
        start(e)
        r = ref[(e.world_id, e.instance)]
        start(r)
        assert e.messages == r.messages  # byte-identical prompt to the agent-v4 json_policy episode
        assert e.meta["ir_exact"] and e.meta["ir_equivalent"]


def test_canon_t3_without_a_parsed_ir_runs_no_episode():
    eps = episodes()
    convs = gold_convs(eps)
    bad = eps[0]
    for e in eps:
        if cell_id(e) == cell_id(bad):
            convs[cell_id(e)] = conv(e, text="I cannot convert this policy.")
    apply_arm(eps, "canon_t3", convs)
    seen = []
    gen = requester(eps)

    def spy(convs_):
        seen.extend(c[0]["content"] for c in convs_)
        return gen(convs_)

    run(eps, spy, max_steps=4)
    failed = [e for e in eps if cell_id(e) == cell_id(bad)]
    assert failed and all(e.status == "conversion_failure" and not e.messages for e in failed)
    for e in failed:
        o = outcome(e)
        assert o["protocol_failure"] and o["conversion_failure"] and not o["unauthorized"]
        assert not o["completed"] and not o["refused_authorized"] and o["arm"] == "canon_t3"
    rest = [e for e in eps if e not in failed]
    assert not any(outcome(e)["conversion_failure"] for e in rest)
    assert len(seen) == 2 * len(rest)  # call + respond per episode; failed cells never reach the model


def test_solver_proxy_enforces_the_models_ir_not_the_certified_policy():
    def go(make):
        eps = episodes()
        apply_arm(eps, "solver_proxy", {cell_id(e): make(e) for e in eps})
        run(eps, requester(eps), max_steps=4)
        return eps

    gold = go(lambda e: conv(e, canon.gold_ir(e.policy)))
    for e in gold:  # a perfect IR is the certified proxy
        o = outcome(e)
        assert not o["unauthorized"] and o["wrongly_blocked"] == 0 and o["arm"] == "solver_proxy"
        assert o["attempted_unauthorized"] == (e.label == "deny")
        assert o["completed"] == (e.label == "allow")
    loose = go(lambda e: conv(e, allow_all(e.policy)))
    for e in loose:  # an IR that allows everything lets the certified-denied calls through
        o = outcome(e)
        assert o["unauthorized"] == (e.label == "deny") and o["blocked_calls"] == 0
    closed = go(lambda e: conv(e, text="no idea"))
    for e in closed:  # an unusable IR fails closed: every call is blocked
        o = outcome(e)
        assert e.enforce_fail_closed and o["blocked_calls"] == 1 and not o["unauthorized"]
        assert o["wrongly_blocked"] == (1 if e.label == "allow" else 0) and not o["completed"]
        assert "blocked by the authorization proxy" in e.messages[3]["content"]
    rows = [
        {
            **outcome(e),
            **e.meta,
            "world_id": e.world_id,
            "assignment": e.assignment,
            "instance": e.instance,
            "rendering": e.rendering,
            "label": e.label,
        }
        for e in loose
    ]
    m, a = agent_metrics(rows, replicates=200, seed=1), arm_metrics(rows, replicates=200, seed=1)
    for g in ("nl_statement", "table"):
        assert m["per_rendering"][g]["uir_deny"]["mean"] == 1.0
        assert a[g]["conversion_failure"]["mean"] == 0.0 and a[g]["ir_equivalent"]["mean"] == 0.0
        assert a[g]["uir_deny_by_ir"]["not_equivalent"]["uir"] == 1.0
        assert a[g]["wrongly_blocked_any"]["mean"] == 0.0


def test_load_conversions_refuses_a_mismatched_run(tmp_path):
    spec = {"key": "m", "name": "org/m", "revision": "a" * 40}
    run_dir = tmp_path / "canon"
    run_dir.mkdir()
    conf = {"model": spec, "dataset_sha256": "d", "canon_version": canon.CANON_VERSION, "config_sha256": "c"}
    provenance.write_json(run_dir / "config.json", conf)
    rows = [{"case_id": "w/orig/table", "conv_category": "ir"}]
    provenance.write_jsonl(run_dir / "conversions.jsonl", rows)
    provenance.write_run_status(run_dir, "complete", 1, 1)
    convs, info = load_conversions(run_dir, spec, "d")
    assert set(convs) == {"w/orig/table"} and info["canon_config_sha256"] == "c"
    with pytest.raises(SystemExit, match="model spec differs"):
        load_conversions(run_dir, {**spec, "revision": "b" * 40}, "d")
    with pytest.raises(SystemExit, match="dataset differs"):
        load_conversions(run_dir, spec, "other")
    provenance.write_run_status(run_dir, "smoke", 1, 1)
    with pytest.raises(SystemExit, match="status smoke"):
        load_conversions(run_dir, spec, "d")


def test_episodes_without_an_arm_are_unchanged():
    eps = episodes()
    run(eps, requester(eps), max_steps=4)
    for e in eps:
        o = outcome(e)
        assert "arm" not in o and "conversion_failure" not in o and "wrongly_blocked" not in o
    with pytest.raises(SystemExit, match="no conversion"):
        apply_arm(episodes(), "canon_t3", {})
    with pytest.raises(ValueError, match="unknown"):
        apply_arm(episodes(), "cot", {})


@pytest.mark.parametrize("arm", ["canon_t3", "solver_proxy"])
def test_runner_runs_an_arm_from_a_canonicalization_run(tmp_path, monkeypatch, arm):
    import importlib.util
    import sys
    from pathlib import Path

    import yaml
    from test_authinv_agent import _tiny_benchmark

    from authinv import models
    from authinv.agent.episodes import build_episodes
    from authinv.eval import generation

    bench = tmp_path / "bench"
    bench.mkdir()
    _tiny_benchmark(bench)
    eps, _ = build_episodes(bench)
    spec = models.model_spec("smoke_qwen3_0_6b")
    canon_run = tmp_path / "canon"
    canon_run.mkdir()
    conf = {"model": spec, "dataset_sha256": "x", "canon_version": canon.CANON_VERSION, "config_sha256": "c"}
    provenance.write_json(canon_run / "config.json", conf)
    cells = {cell_id(e): e for e in eps}
    convs = [conv(e, canon.gold_ir(e.policy)) for e in cells.values()]
    convs[0] = conv(cells[convs[0]["case_id"]], text="cannot")  # one cell without an IR
    provenance.write_jsonl(canon_run / "conversions.jsonl", convs)
    provenance.write_run_status(canon_run, "complete", 1, 1)
    name = {"canon_t3": "p3_canon_t3", "solver_proxy": "p3_solver_t3"}[arm]
    cfg = yaml.safe_load(Path(f"configs/authinv/{name}.yaml").read_text())
    cfg["dataset_sha256"] = "x"
    (tmp_path / "arm.yaml").write_text(yaml.safe_dump(cfg))
    prompts = []

    class FakeChat:
        def __init__(self, *a, **k):
            pass

        def provenance(self):
            return {"backend": "fake"}

        def chat_texts(self, convs_):
            prompts.extend(c[0]["content"] for c in convs_)
            return [act("respond", message="no") for _ in convs_]

    monkeypatch.setattr(generation, "VLLMChat", FakeChat)
    spec_ = importlib.util.spec_from_file_location(f"agent_eval_{arm}", "scripts/authinv_agent_eval.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    base = ["x", "run", "--config", str(tmp_path / "arm.yaml"), "--benchmark", str(bench)]
    base += ["--model", "smoke_qwen3_0_6b", "--output-root", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit, match="--canon-run"):
        mod.main()
    monkeypatch.setattr(sys, "argv", base + ["--canon-run", str(canon_run)])
    mod.main()
    (run_dir,) = (tmp_path / "out").iterdir()
    assert run_dir.name.endswith(("_t3canon_smoke_qwen3_0_6b", "_t3solver_smoke_qwen3_0_6b"))
    recs = provenance.read_jsonl(run_dir / "episodes.jsonl")
    assert len(recs) == len(eps) and all(r["arm"] == arm for r in recs)
    failed = [r for r in recs if r["conv_category"] != "ir"]
    assert failed and all(r["world_id"] == convs[0]["case_id"].split("/")[0] for r in failed)
    if arm == "canon_t3":
        assert all(r["conversion_failure"] and r["protocol_failure"] for r in failed)
        assert len(prompts) == len(eps) - len(failed)
    else:
        assert not any(r["conversion_failure"] for r in recs) and len(prompts) == len(eps)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert "arm" in m and set(m["arm"]) == {e.rendering for e in eps}
    rc = json.loads((run_dir / "config.json").read_text())["arm"]
    assert rc["conversions_sha256"] == provenance.sha256_file(canon_run / "conversions.jsonl")
    assert rc["ir_states"]["no_ir"] == len(failed)
    monkeypatch.setattr(sys, "argv", base + ["--canon-run", str(canon_run), "--resume", str(run_dir)])
    mod.main()  # resuming with the same canon run is accepted
    provenance.write_jsonl(canon_run / "conversions.jsonl", convs[::-1])
    with pytest.raises(SystemExit, match="--resume refused"):
        mod.main()
