"""Strict JSON answer parsing for tier T2 (structured generation).

Answers must be one JSON object (bare, or in a single ```json fence) with exactly
the task's field:

- application:    {"decision": "allow" | "deny"}
- interpretation: {"principals": [<principal id>, ...]}  (exact set match, order-free)

Anything else is a refusal or abstention (strict-v2 patterns, with typographic
apostrophes folded) or a parse failure. These categories are kept apart and are
never counted as wrong or dropped (RESEARCH_PLAN §4). A rule change needs a new
JSON_PARSER_VERSION and a docs/CHANGELOG.md entry.
"""

from __future__ import annotations

import json
import re

from authinv.eval.parse import _ABSTAIN, _REFUSAL, _THINK, fold_typographic

JSON_PARSER_VERSION = "json-strict-v1"
_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```$", re.S)


def extract_object(text: str) -> dict | None:
    s = _THINK.sub("", text, count=1).strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _value(obj: dict, task: str):
    if task == "application":
        if set(obj) == {"decision"} and obj["decision"] in ("allow", "deny"):
            return obj["decision"]
        return None
    if set(obj) == {"principals"} and isinstance(obj["principals"], list):
        items = obj["principals"]
        if all(isinstance(x, str) for x in items) and len(set(items)) == len(items):
            return tuple(sorted(items))
    return None


def parse_structured(text: str, task: str, expected) -> dict:
    """{"category", "value"} where value is the parsed answer (decision str or sorted principal tuple)."""
    if task not in ("application", "interpretation"):
        raise ValueError(f"unknown task {task!r}")
    want = tuple(sorted(expected)) if task == "interpretation" else expected
    obj = extract_object(text)
    if obj is not None:
        value = _value(obj, task)
        if value is not None:
            return {"category": "correct" if value == want else "incorrect", "value": value}
    folded = fold_typographic(_THINK.sub("", text, count=1))
    if _REFUSAL.search(folded):
        return {"category": "refusal", "value": None}
    if _ABSTAIN.search(folded) and obj is None:
        return {"category": "abstain", "value": None}
    return {"category": "parse_failure", "value": None}


def answer_schema(task: str, principal_ids: list[str] | None = None) -> dict:
    """JSON schema for optional constrained decoding (vLLM StructuredOutputsParams(json=...))."""
    if task == "application":
        return {
            "type": "object",
            "properties": {"decision": {"enum": ["allow", "deny"]}},
            "required": ["decision"],
            "additionalProperties": False,
        }
    items = {"type": "string"} if principal_ids is None else {"enum": sorted(principal_ids)}
    return {
        "type": "object",
        "properties": {"principals": {"type": "array", "items": items, "uniqueItems": True}},
        "required": ["principals"],
        "additionalProperties": False,
    }
