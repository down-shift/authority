"""Strict answer parser for generated answers (tier T2 / Phase 0).

An answer counts only if, after light normalization, it is exactly one of the
row's candidates. Refusals, abstentions, and anything else are separate
categories and are never counted as wrong or dropped (RESEARCH_PLAN §4).
Changing these rules requires a new PARSER_VERSION and a changelog entry.
"""

from __future__ import annotations

import re

PARSER_VERSION = "strict-v1"
CATEGORIES = ("correct", "incorrect", "refusal", "abstain", "parse_failure")

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_QUOTES = ("``", "**", '"', "'", "`", "*")
_REFUSAL = re.compile(
    r"\b(i can(?:no|')t|i am unable|i'm unable|i won't|i will not|unable to (?:help|comply|assist))\b", re.I
)
_ABSTAIN = re.compile(
    r"\b(neither|none of|no one|nobody|cannot (?:be )?determine|can't (?:be )?determine|"
    r"not enough information|insufficient information|unclear|ambiguous|unknown|both)\b",
    re.I,
)


def normalize(text: str) -> str:
    """Strip an empty/closed think block, whitespace, one wrapping quote pair, a final period."""
    s = _strip_period(_THINK.sub("", text, count=1).strip())
    for q in _QUOTES:
        if len(s) > 2 * len(q) and s.startswith(q) and s.endswith(q):
            s = s[len(q) : -len(q)].strip()
            break
    return _strip_period(s)


def _strip_period(s: str) -> str:
    return s[:-1].rstrip() if s.endswith(".") and not s.endswith("..") else s


def lenient_selection(s: str, candidates: list[str]) -> str | None:
    """Secondary analysis only: the single candidate that occurs in the answer, if exactly one does."""
    found = [c for c in candidates if re.search(r"(?<![\w.])" + re.escape(c) + r"(?![\w])", s)]
    return found[0] if len(found) == 1 else None


def parse_answer(text: str, candidates: list[str], correct: str) -> dict:
    """Return {"category", "selected", "normalized", "lenient_selected"} for one generated answer."""
    s = normalize(text)
    lenient = lenient_selection(s, candidates)
    if s in candidates:
        return {
            "category": "correct" if s == correct else "incorrect",
            "selected": s,
            "normalized": s,
            "lenient_selected": s,
        }
    if _REFUSAL.search(s):
        category = "refusal"
    elif _ABSTAIN.search(s):
        category = "abstain"
    else:
        category = "parse_failure"
    return {"category": category, "selected": None, "normalized": s, "lenient_selected": lenient}
