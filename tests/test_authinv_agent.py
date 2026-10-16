"""T3 sandbox: protocol parsing, call checking, outcomes and metrics with scripted mock agents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from authinv.agent.metrics import agent_metrics, transfer, uir_contrasts
from authinv.agent.sandbox import Episode, outcome, parse_action, run, start
from authinv.benchmark import World, non_degenerate
from authinv.equivalence.requests import sample_requests
from authinv.render.renderers import render
from authinv.sources.synthetic import generate

pytest.importorskip("cedarpy")


def episodes(n_worlds=6, renderings=("nl_statement", "table")):
    ws, _ = generate("repo", replicates=1, seed=3)
    out = []
    for p in ws:
        w = World(p, ("User",), ("Repo",), "synthetic", "synthetic", {})
        if not non_degenerate(w, 4):
            continue
        for qi, item in enumerate(sample_requests(p, ("User",), ("Repo",), 2, 1)):
            for g in renderings:
                out.append(
                    Episode(
                        f"{p.policy_id}/q{qi}/{g}",
                        p.policy_id,
                        "orig",
                        f"q{qi}",
                        g,
                        p,
                        item["request"],
                        item["decision"],
                        ("User",),
                        ("Repo",),
                        render(p, g),
                    )
                )
        if len({e.world_id for e in out}) >= n_worlds:
            break
    return out


def act(tool, **args):
    return json.dumps({"tool": tool, "arguments": args})


def scripted(policy_fn):
    """A mock agent: decides from the episode's own task message (the last user turn of the first round)."""

    def gen(convs):
        out = []
        for msgs in convs:
            if msgs[-1]["content"].startswith("Tool result:"):
                out.append(act("respond", message="done"))
            else:
                out.append(policy_fn(msgs))
        return out

    return gen


def by_episode(eps):
    return {e.episode_id: e for e in eps}


def test_parse_action_is_strict():
    assert parse_action('{"tool": "push", "arguments": {"resource": "r_lynx"}}')["tool"] == "push"
    assert (
        parse_action('```json\n{"tool": "respond", "arguments": {"message": "no"}}\n```')["tool"] == "respond"
    )
    assert parse_action('I will push. {"tool": "push", "arguments": {}}') is None
    assert parse_action('{"tool": "push"}') is None
    assert parse_action('{"tool": "push", "arguments": {"resource": "r_lynx"}}.')["tool"] == "push"


def test_parse_action_strips_think_blocks():
    call = '{"tool": "push", "arguments": {"resource": "r_lynx"}}'
    assert parse_action(f"<think>\nr1 allows push.\n</think>\n\n{call}")["tool"] == "push"
    # agent-v2: the chat template put the opening <think> in the prompt (Qwen3.8 thinking on)
    assert parse_action(f"r1 allows push. {{maybe}}\n</think>\n\n{call}")["tool"] == "push"
    assert parse_action(f"r1 allows push.\n{call}") is None


def _requested(msgs, eps_by_task):
    return eps_by_task[msgs[1]["content"] + msgs[0]["content"]]


def test_always_execute_agent_has_full_uir_on_denied_requests():
    eps = episodes()
    lookup = {e.episode_id: e for e in eps}
    for e in eps:
        start(e)
    index = {(e.messages[0]["content"], e.messages[1]["content"]): e for e in eps}

    def execute(msgs):
        e = index[(msgs[0]["content"], msgs[1]["content"])]
        return act(e.request.action, resource=e.request.resource.id)

    run(eps, scripted(execute), max_steps=4)
    rows = [
        {
            **outcome(e),
            "world_id": e.world_id,
            "assignment": e.assignment,
            "instance": e.instance,
            "rendering": e.rendering,
            "label": e.label,
        }
        for e in lookup.values()
    ]
    m = agent_metrics(rows, replicates=200, seed=1)
    for g in ("nl_statement", "table"):
        assert m["per_rendering"][g]["uir_deny"]["mean"] == 1.0
        assert m["per_rendering"][g]["completion"]["mean"] == 1.0
        assert m["per_rendering"][g]["authorized_refusal"]["mean"] == 0.0
    assert all(o["status"] == "responded" for o in (outcome(e) for e in eps))


def test_refusing_agent_never_violates_but_refuses_authorized():
    eps = episodes()
    run(eps, scripted(lambda msgs: act("respond", message="I can't do that.")), max_steps=4)
    rows = [
        {
            **outcome(e),
            "world_id": e.world_id,
            "assignment": e.assignment,
            "instance": e.instance,
            "rendering": e.rendering,
            "label": e.label,
        }
        for e in eps
    ]
    m = agent_metrics(rows, replicates=200, seed=1)
    pr = m["per_rendering"]["table"]
    assert (
        pr["uir_deny"]["mean"] == 0
        and pr["authorized_refusal"]["mean"] == 1
        and pr["completion"]["mean"] == 0
    )


def test_oracle_agent_is_perfect_and_rendering_gap_is_measured():
    eps = episodes()
    for e in eps:
        start(e)
    index = {(e.messages[0]["content"], e.messages[1]["content"]): e for e in eps}

    def oracle_except_nl(msgs):
        e = index[(msgs[0]["content"], msgs[1]["content"])]
        if e.label == "allow" or e.rendering == "nl_statement":  # fail open on NL only
            return act(e.request.action, resource=e.request.resource.id)
        return act("respond", message="Not permitted.")

    run(eps, scripted(oracle_except_nl), max_steps=4)
    rows = [
        {
            **outcome(e),
            "world_id": e.world_id,
            "assignment": e.assignment,
            "instance": e.instance,
            "rendering": e.rendering,
            "label": e.label,
        }
        for e in eps
    ]
    m = agent_metrics(rows, replicates=200, seed=1)
    assert (
        m["per_rendering"]["table"]["uir_deny"]["mean"] == 0
        and m["per_rendering"]["table"]["completion"]["mean"] == 1
    )
    assert m["worst_uir_rendering"] == "nl_statement" and m["uir_gap"]["mean"] == 1.0
    assert m["action_disagreement"]["mean"] == pytest.approx(0.5)  # deny instances disagree, allow agree


def test_protocol_failures_and_wrong_resource_calls():
    eps = episodes(n_worlds=2)
    run(eps, scripted(lambda msgs: "Sure, done!"), max_steps=4)
    assert all(outcome(e)["protocol_failure"] and not outcome(e)["unauthorized"] for e in eps)
    eps = episodes(n_worlds=2)
    run(eps, scripted(lambda msgs: act("delete", resource="r_nonexistent")), max_steps=4)
    assert all(outcome(e)["status"] == "protocol_failure" for e in eps)


def test_unauthorized_other_call_counts_even_on_allowed_requests():
    eps = [e for e in episodes() if e.label == "allow"]
    for e in eps:
        start(e)
    index = {(e.messages[0]["content"], e.messages[1]["content"]): e for e in eps}
    from authinv.policy import Request, evaluate

    def misuse(msgs):
        e = index[(msgs[0]["content"], msgs[1]["content"])]
        for a in sorted(e.policy.actions):
            for r in sorted(
                {x.ref for x in e.policy.entities if x.ref.type == "Repo"}, key=lambda x: x.key()
            ):
                if (
                    evaluate(e.policy, Request(e.request.principal, a, r, e.request.context))["decision"]
                    == "deny"
                ):
                    return act(a, resource=r.id)
        return act("respond", message="nothing to misuse")

    run(eps, scripted(misuse), max_steps=4)
    outs = [outcome(e) for e in eps]
    assert any(o["unauthorized"] for o in outs) and not any(o["completed"] for o in outs if o["unauthorized"])


def _tiny_benchmark(tmp_path):
    from authinv import provenance
    from authinv.benchmark import world_rows
    from authinv.policy import to_dict

    ws, _ = generate("mcp", replicates=1, seed=4)
    w = next(
        World(p, ("Agent",), ("Tool",), "synthetic", "synthetic", {})
        for p in ws
        if non_degenerate(World(p, ("Agent",), ("Tool",), "synthetic", "synthetic", {}), 4)
    )
    rows, extra = world_rows(w, per_label=2, seed=9)
    provenance.write_jsonl(tmp_path / "dataset.jsonl", rows)
    provenance.write_json(tmp_path / "dataset_manifest.json", {"sha256": "x", "rows": len(rows)})
    provenance.write_json(tmp_path / "config.json", {"config": {"per_label": 2, "seed": 9}})
    provenance.write_jsonl(
        tmp_path / "worlds.jsonl",
        [
            {
                "world_id": w.policy.policy_id,
                "source": "synthetic",
                "source_kind": "synthetic",
                "principal_types": ["Agent"],
                "resource_types": ["Tool"],
                "orig": to_dict(w.policy),
                "swap": to_dict(extra["swap_policy"]),
                "swap_map": extra["swap_map"],
            }
        ],
    )


def test_episode_builder_matches_frozen_rows(tmp_path):
    from authinv import provenance
    from authinv.agent.episodes import build_episodes

    _tiny_benchmark(tmp_path)
    eps, info = build_episodes(tmp_path)
    assert info["instances_checked"] == 8 and len(eps) == 8 * 6
    provenance.write_json(tmp_path / "config.json", {"config": {"per_label": 2, "seed": 10}})
    with pytest.raises(SystemExit, match="does not match"):
        build_episodes(tmp_path)


def _rows(eps):
    return [
        {
            **outcome(e),
            "world_id": e.world_id,
            "assignment": e.assignment,
            "instance": e.instance,
            "rendering": e.rendering,
            "label": e.label,
        }
        for e in eps
    ]


def _fail_open_on_nl(eps):
    for e in eps:
        start(e)
    index = {(e.messages[0]["content"], e.messages[1]["content"]): e for e in eps}

    def agent(msgs):
        e = index[(msgs[0]["content"], msgs[1]["content"])]
        if e.label == "allow" or e.rendering == "nl_statement":
            return act(e.request.action, resource=e.request.resource.id)
        return act("respond", message="Not permitted.")

    run(eps, scripted(agent), max_steps=4)
    return _rows(eps)


def test_uir_contrasts_are_paired_within_world():
    rows = _fail_open_on_nl(episodes())
    c = uir_contrasts(rows, replicates=200, seed=1)
    assert c["nl_statement"]["mean"] == pytest.approx(1.0) and c["table"]["mean"] == pytest.approx(-1.0)
    assert c["nl_statement"]["ci95"] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_transfer_matches_t2_rows_and_splits_uir_by_t2_answer():
    rows = _fail_open_on_nl(episodes())
    # T2 answered allow on every NL row and correctly elsewhere; one table row is a parse failure
    t2 = []
    for i, r in enumerate(rows):
        value = "allow" if r["rendering"] == "nl_statement" else r["label"]
        cat = "correct" if value == r["label"] else "incorrect"
        if i == 1 and r["rendering"] == "table":
            value, cat = None, "parse_failure"
        t2.append(
            {
                **{k: r[k] for k in ("world_id", "assignment", "instance", "rendering", "label")},
                "task": "application",
                "value": value,
                "category": cat,
            }
        )
    t = transfer(rows, t2, replicates=200, seed=1)
    assert t["all"]["agreement"]["mean"] == pytest.approx(1.0)
    u = t["all"]["uir_given_t2"]
    assert u["allow"]["uir"] == 1.0 and u["deny"]["uir"] == 0.0
    assert t["per_rendering"]["table"]["uir_given_t2"]["allow"]["episodes"] == 0
    with pytest.raises(ValueError):
        transfer(rows, t2[1:], replicates=10, seed=1)


def test_runner_checkpoints_and_resumes(tmp_path, monkeypatch):
    import importlib.util
    import sys

    import yaml

    from authinv import provenance
    from authinv.eval import generation

    bench = tmp_path / "bench"
    bench.mkdir()
    _tiny_benchmark(bench)
    cfg = yaml.safe_load(Path("configs/authinv/t3.yaml").read_text())
    cfg["dataset_sha256"] = "x"
    (tmp_path / "t3.yaml").write_text(yaml.safe_dump(cfg))
    calls = {"n": 0, "crash_after": 1}

    class FakeChat:
        def __init__(self, *a, **k):
            pass

        def provenance(self):
            return {"backend": "fake"}

        def chat_texts(self, convs):
            if calls["n"] >= calls["crash_after"] and calls["crash_after"]:
                raise RuntimeError("simulated crash")
            calls["n"] += 1
            return [act("respond", message="no") for _ in convs]

    monkeypatch.setattr(generation, "VLLMChat", FakeChat)
    spec = importlib.util.spec_from_file_location("agent_eval", "scripts/authinv_agent_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = ["x", "run", "--config", str(tmp_path / "t3.yaml"), "--benchmark", str(bench)]
    base += ["--model", "smoke_qwen3_0_6b", "--output-root", str(tmp_path / "out"), "--chunk-episodes", "12"]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(RuntimeError, match="simulated"):
        mod.main()
    (run_dir,) = (tmp_path / "out").iterdir()
    assert len(provenance.read_jsonl(run_dir / "episodes.jsonl")) == 12
    calls["crash_after"] = 0
    monkeypatch.setattr(sys, "argv", base + ["--resume", str(run_dir)])
    mod.main()
    recs = provenance.read_jsonl(run_dir / "episodes.jsonl")
    assert len({r["episode_id"] for r in recs}) == len(recs) == 48
    assert json.loads((run_dir / "run_status.json").read_text())["status"] == "complete"
    assert "uir_contrasts" in json.loads((run_dir / "metrics.json").read_text())
    cfg["max_steps"] = 3
    (tmp_path / "t3.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="--resume refused"):
        mod.main()
