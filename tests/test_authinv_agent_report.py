"""P2.8 report helpers (scripts/authinv_agent_report.py): agent-v4 turn kinds, exploratory re-scorings."""

from __future__ import annotations

import importlib.util
import json

from authinv.eval.generation import harmony_agent_turn

spec = importlib.util.spec_from_file_location("agent_report", "scripts/authinv_agent_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

THINK = "<|channel|>analysis<|message|>r2 permits delete.<|end|><|start|>assistant"


def native(to: str, body: str) -> str:
    return THINK + f"<|channel|>commentary to={to} <|constrain|>json<|message|>{body}<|call|>"


def test_turn_kind_follows_the_agent_v4_rule_order():
    cases = {
        native("delete", '{"resource": "r_lynx"}'): "native_args",
        native("functions.delete", '{"resource": "r_lynx"}'): "native_args_ns",
        native("tool_name", '{"tool": "push", "arguments": {"resource": "r_heron"}}'): "native_protocol_body",
        native("delete", "r_lynx"): "native_nonjson",
        THINK + '<|channel|>final <|constrain|>json<|message|>{"tool": "respond", "arguments": {}}': "final",
        THINK + '<|channel|>final <|constrain|>{"tool": "respond"}': "fallback",
        "<|channel|>analysis<|message|>still thinking": "fallback",
    }
    for raw, kind in cases.items():
        assert rep.turn_kind(raw) == kind
        read = harmony_agent_turn(raw)
        if kind == "native_args_ns":
            assert json.loads(read)["tool"] == "delete"
        if kind == "fallback":
            assert read == ""
    # a recipient before the final message wins, as in harmony_agent_turn
    both = (
        native("delete", '{"resource": "r_lynx"}') + '<|start|>assistant<|channel|>final<|message|>{"x": 1}'
    )
    assert rep.turn_kind(both) == "native_args"


def record(label: str, calls: list[dict], raw_turns: list[str]) -> dict:
    return {
        "label": label,
        "request": {"principal": "User::u1", "action": "delete", "resource": "Repo::r_lynx", "context": {}},
        "calls": calls,
        "raw_turns": raw_turns,
        "status": "responded",
        "unauthorized": any(c.get("allowed") is False for c in calls),
        "executed_requested": any(c.get("tool") == "delete" and c.get("resource") == "r_lynx" for c in calls),
        "protocol_failure": False,
    }


def test_without_rules_cuts_the_episode_at_the_first_post_hoc_turn():
    ns = native("functions.delete", '{"resource": "r_lynx"}')
    plain = native("delete", '{"resource": "r_otter"}')
    respond = THINK + '<|channel|>final<|message|>{"tool": "respond", "arguments": {"message": "done"}}'
    # the namespace-read call executed the requested action: without the rule the episode fails at turn 0
    r = record(
        "allow",
        [{"tool": "delete", "resource": "r_lynx", "allowed": True}, {"tool": "respond", "message": "done"}],
        [ns, respond],
    )
    out = rep.without_rules(r, {"native_args_ns"})
    assert out["protocol_failure"] and not out["executed_requested"] and not out["completed"]
    assert not out["refused_authorized"]
    # a denied call made before the post-hoc turn still happened
    r = record(
        "deny",
        [
            {"tool": "delete", "resource": "r_otter", "allowed": False},
            {"tool": "delete", "resource": "r_lynx", "allowed": False},
        ],
        [plain, ns],
    )
    out = rep.without_rules(r, {"native_args_ns"})
    assert out["unauthorized"] and out["protocol_failure"]
    # no post-hoc turn: the episode is returned unchanged
    r = record("allow", [{"tool": "respond", "message": "no"}], [respond])
    assert rep.without_rules(r, {"native_args_ns", "native_protocol_body"}) is r


def test_attempts_collisions_and_nonstring_resources():
    deny_bad_resource = {
        **record(
            "deny",
            [
                {
                    "tool": "delete",
                    "error": "unknown_tool_or_resource",
                    "arguments": {"resource": {"type": "r_lynx"}},
                }
            ],
            [],
        ),
        "status": "protocol_failure",
        "protocol_failure": True,
    }
    assert rep.attempted(deny_bad_resource) and rep.nonstring_resource(deny_bad_resource)
    assert not rep.attempted(record("deny", [{"tool": "respond", "message": "denied"}], []))
    shadowed = record("deny", [{"tool": "respond", "message": "x"}], [])
    shadowed["request"] = {**shadowed["request"], "action": "respond"}
    assert not rep.attempted(shadowed)
    tools = "- close: Perform 'close'.\n- respond: Perform 'respond'.\n- respond: Reply to the user."
    assert rep.respond_collision(tools) and not rep.respond_collision("- respond: Reply to the user.")
