"""Prompt surgery shared by the Phase-3 arms: the frozen T2 prompt with only its policy block changed.

A frozen T2 prompt (`authinv.benchmark`, prompt-v1) is

    <INTRO>\\n\\nFacts:\\n<facts>\\n\\nPolicy:\\n<rendering>\\n\\n<Request: ... | Question: ...>

`split_prompt` cuts it at the first policy opener and the last task marker, so
a rendering may itself contain blank lines. The request (or question) is read
back from the prompt text and checked against the row's frozen label by the
caller, since dataset rows store prompts, not request objects.
"""

from __future__ import annotations

import re

from authinv.benchmark import INTRO, _context
from authinv.policy import EntityRef, Policy, Request

POLICY_OPEN = "\n\nPolicy:\n"
FACTS_OPEN = INTRO + "\n\nFacts:\n"
TAIL_MARKERS = {"application": "\n\nRequest: ", "interpretation": "\n\nQuestion: "}

_ID = r"[^\s\"]+"
_ENT = rf'({_ID}) "({_ID})"'
_REQUEST = re.compile(
    rf"^\n\nRequest: {_ENT} asks to (\S+) {_ENT}\. Request context: (.*)\.\nIs this request allowed\?"
)
_QUESTION = re.compile(
    rf"^\n\nQuestion: which .+? entities may (\S+) {_ENT} when the request context is (.*)\? "
)


def split_prompt(prompt: str, task: str) -> tuple[str, str, str]:
    """(head, policy text, tail): head ends before the policy opener, tail starts at the task marker."""
    head, sep, rest = prompt.partition(POLICY_OPEN)
    text, marker, tail = rest.rpartition(TAIL_MARKERS[task])
    if not sep or not marker:
        raise ValueError("prompt lacks the frozen policy block")
    return head, text, marker + tail


def replace_policy(prompt: str, task: str, text: str) -> str:
    head, _, tail = split_prompt(prompt, task)
    return head + POLICY_OPEN + text.rstrip() + tail


def facts_of(head: str) -> str:
    if not head.startswith(FACTS_OPEN):
        raise ValueError("prompt head is not the frozen intro + facts block")
    return head[len(FACTS_OPEN) :]


def _value(text: str, kind: str):
    if kind == "bool":
        if text not in ("true", "false"):
            raise ValueError(f"bad bool {text!r}")
        return text == "true"
    if kind == "int":
        return int(text)
    if not (len(text) >= 2 and text[0] == text[-1] == '"'):
        raise ValueError(f"bad string {text!r}")
    return text[1:-1]


def _parse_context(text: str, policy: Policy) -> tuple:
    if text == "none":
        return ()
    kinds = dict(policy.context_schema)
    out = []
    for item in text.split(", "):
        key, sep, val = item.partition(" = ")
        if not sep or key not in kinds:
            raise ValueError(f"bad context item {item!r}")
        out.append((key, _value(val, kinds[key])))
    return tuple(out)


def request_of(row: dict, policy: Policy) -> Request:
    """The application row's request, read back from its prompt (checked by `check_request`)."""
    _, _, tail = split_prompt(row["prompt"], "application")
    m = _REQUEST.match(tail)
    if not m:
        raise ValueError(f"{row['row_id']}: request line not found")
    pt, pid, action, rt, rid, ctx = m.groups()
    return Request(EntityRef(pt, pid), action, EntityRef(rt, rid), _parse_context(ctx, policy))


def query_of(row: dict, policy: Policy) -> tuple[str, EntityRef, tuple]:
    """(action, resource, context) of an interpretation row, read back from its prompt."""
    _, _, tail = split_prompt(row["prompt"], "interpretation")
    m = _QUESTION.match(tail)
    if not m:
        raise ValueError(f"{row['row_id']}: question line not found")
    action, rt, rid, ctx = m.groups()
    return action, EntityRef(rt, rid), _parse_context(ctx, policy)


def pair_key(action: str, resource: EntityRef, context: tuple) -> str:
    return f"{action}|{resource.key()}|{_context(context)}"
