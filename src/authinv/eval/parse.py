"""Strict answer parser for generated answers (tier T2).

An answer counts only if, after light normalization, it is exactly one of the
row's candidates. Refusals, abstentions, and anything else are separate
categories and are never counted as wrong or dropped (RESEARCH_PLAN §4).
Changing these rules requires a new parser version and a changelog entry.

Versions (selected with ``parse_answer(..., version=...)``):

- ``strict-v1`` — frozen; Phase 0 and the P0.5 addendum were scored with it.
  Its code below must never change.
- ``strict-v2`` — default for Phase 2+. Same rules, but typographic
  apostrophes (’ ‘ ʼ) and double quotes (“ ”) are folded to ASCII for the
  refusal / abstention patterns and accepted as a wrapping quote pair. The
  exact-candidate match still compares the *unfolded* text against the
  candidates as given, so folding can never turn a non-candidate into one.
"""

from __future__ import annotations

import re

STRICT_V1 = "strict-v1"
STRICT_V2 = "strict-v2"
DEFAULT_PARSER_VERSION = STRICT_V2  # for new (Phase 2+) configs
PARSER_VERSION = DEFAULT_PARSER_VERSION
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


def parse_answer(
    text: str, candidates: list[str], correct: str, version: str = DEFAULT_PARSER_VERSION
) -> dict:
    """Return {"category", "selected", "normalized", "lenient_selected"} for one generated answer."""
    try:
        parser = PARSERS[version]
    except KeyError:
        raise ValueError(f"unknown parser version {version!r}; known: {sorted(PARSERS)}") from None
    return parser(text, candidates, correct)


# --- strict-v1 (frozen: do not edit) ---------------------------------------


def _parse_v1(text: str, candidates: list[str], correct: str) -> dict:
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


# --- strict-v2 -------------------------------------------------------------

_TYPO_FOLD = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"'})
_WRAP_V2 = (
    ("``", "``"),
    ("**", "**"),
    ('"', '"'),
    ("\u201c", "\u201d"),
    ("'", "'"),
    ("\u2018", "\u2019"),
    ("`", "`"),
    ("*", "*"),
)


def fold_typographic(s: str) -> str:
    """Map typographic apostrophes/double quotes to ASCII (pattern matching only)."""
    return s.translate(_TYPO_FOLD)


def normalize_v2(text: str) -> str:
    """strict-v1 normalization, also unwrapping one typographic quote pair; interior left unfolded."""
    s = _strip_period(_THINK.sub("", text, count=1).strip())
    for open_q, close_q in _WRAP_V2:
        if len(s) > len(open_q) + len(close_q) and s.startswith(open_q) and s.endswith(close_q):
            s = s[len(open_q) : -len(close_q)].strip()
            break
    return _strip_period(s)


def _parse_v2(text: str, candidates: list[str], correct: str) -> dict:
    s = normalize_v2(text)
    lenient = lenient_selection(s, candidates)
    if s in candidates:  # exact match on the unfolded text, against candidates as given
        return {
            "category": "correct" if s == correct else "incorrect",
            "selected": s,
            "normalized": s,
            "lenient_selected": s,
        }
    folded = fold_typographic(s)
    if _REFUSAL.search(folded):
        category = "refusal"
    elif _ABSTAIN.search(folded):
        category = "abstain"
    else:
        category = "parse_failure"
    return {"category": category, "selected": None, "normalized": s, "lenient_selected": lenient}


PARSERS = {STRICT_V1: _parse_v1, STRICT_V2: _parse_v2}
