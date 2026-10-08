"""Parser versions: strict-v1 frozen (Phase 0), strict-v2 folds typographic apostrophes/quotes (P2.0)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from authinv.eval import parse
from authinv.eval.parse import DEFAULT_PARSER_VERSION, PARSERS, parse_answer

ROOT = Path(__file__).resolve().parents[1]
CANDS = ["Agent F04", "Agent F05"]
FILES = ["filename_1.txt", "filename_12.txt"]
TYPO = "‘’ʼ“”"

# The gpt-oss-120b refusals found in P0.5 (typographic apostrophe U+2019).
P05_REFUSALS = [
    "I’m sorry, but I can’t comply with that request.",
    "I’m sorry, but I can’t provide that.",
    "I’m sorry, but I can’t provide a filename in this case.",
    "I’m sorry, but I can’t provide that information.",
]

# Fixed strings -> strict-v1 (category, selected). Phase-0 results depend on these; never edit them.
V1_GOLDEN = [
    ("Agent F05", "correct", "Agent F05"),
    ("Agent F04", "incorrect", "Agent F04"),
    ("  `Agent F05`.\n", "correct", "Agent F05"),
    ('"Agent F04"', "incorrect", "Agent F04"),
    ("**Agent F05**", "correct", "Agent F05"),
    ("<think>\n\n</think>\n\nAgent F05", "correct", "Agent F05"),
    ("“Agent F05”", "parse_failure", None),
    ("‘Agent F05’", "parse_failure", None),
    ("Agent F05 owns the filename.", "parse_failure", None),
    ("I cannot determine which actor owns it.", "refusal", None),
    ("I can't help with that.", "refusal", None),
    ("I am unable to answer.", "refusal", None),
    ("I won't answer.", "refusal", None),
    ("Neither actor.", "abstain", None),
    ("It is unclear.", "abstain", None),
    ("Both.", "abstain", None),
    ("One can't determine the owner.", "abstain", None),
    ("", "parse_failure", None),
    *[(t, "parse_failure", None) for t in P05_REFUSALS],
    ("I’m unable to say.", "parse_failure", None),
    ("I won’t answer.", "parse_failure", None),
    ("One can’t determine the owner.", "parse_failure", None),
]


def test_registry_and_default():
    assert set(PARSERS) == {"strict-v1", "strict-v2"}
    assert DEFAULT_PARSER_VERSION == parse.PARSER_VERSION == "strict-v2"
    with pytest.raises(ValueError):
        parse_answer("Agent F05", CANDS, "Agent F05", version="strict-v9")


@pytest.mark.parametrize(("text", "category", "selected"), V1_GOLDEN)
def test_strict_v1_categories_frozen(text, category, selected):
    out = parse_answer(text, CANDS, "Agent F05", version="strict-v1")
    assert (out["category"], out["selected"]) == (category, selected)


@pytest.mark.parametrize("text", P05_REFUSALS + [t.replace("’", "'") for t in P05_REFUSALS])
def test_strict_v2_catches_p05_refusals(text):
    out = parse_answer(text, FILES, "filename_1.txt", version="strict-v2")
    assert out["category"] == "refusal" and out["selected"] is None


@pytest.mark.parametrize(
    "text", ["I’m unable to say.", "I‘m unable to tell.", "I canʼt do that.", "I won’t answer."]
)
def test_strict_v2_refusal_apostrophe_variants(text):
    assert parse_answer(text, CANDS, "Agent F05", version="strict-v2")["category"] == "refusal"


@pytest.mark.parametrize(
    "text",
    ["One can’t determine the owner.", "One can't determine the owner.", "“Neither.”", "Neither actor."],
)
def test_strict_v2_abstentions(text):
    assert parse_answer(text, CANDS, "Agent F05", version="strict-v2")["category"] == "abstain"


@pytest.mark.parametrize(
    ("text", "category", "selected"),
    [
        ("“Agent F05”", "correct", "Agent F05"),
        ("‘Agent F04’.", "incorrect", "Agent F04"),
        ('"Agent F05"', "correct", "Agent F05"),
        ("**Agent F05**", "correct", "Agent F05"),
    ],
)
def test_strict_v2_unwraps_typographic_quotes(text, category, selected):
    out = parse_answer(text, CANDS, "Agent F05", version="strict-v2")
    assert (out["category"], out["selected"]) == (category, selected)


def test_strict_v2_folding_never_creates_a_candidate():
    # Interior typographic characters are not folded before the exact match, so a
    # curly-apostrophe variant of an apostrophe-bearing candidate stays a non-match.
    cands = ["O'Brien", "Agent F05"]
    out = parse_answer("O’Brien", cands, "O'Brien", version="strict-v2")
    assert out["category"] == "parse_failure" and out["selected"] is None
    assert out["normalized"] == "O’Brien"
    # Mismatched or one-sided typographic quotes are not unwrapped.
    for text in ("”Agent F05“", "“Agent F05", "Agent F05’"):
        assert parse_answer(text, CANDS, "Agent F05", version="strict-v2")["selected"] is None


def test_strict_v2_matches_v1_without_typographic_characters():
    for text, _, _ in V1_GOLDEN:
        if any(ch in text for ch in TYPO):
            continue
        v1 = parse_answer(text, CANDS, "Agent F05", version="strict-v1")
        assert parse_answer(text, CANDS, "Agent F05", version="strict-v2") == v1


def test_phase0_configs_pin_strict_v1():
    for name in ("phase0.yaml", "phase0_addendum.yaml"):
        assert (
            yaml.safe_load((ROOT / "configs" / "authinv" / name).read_text())["parser_version"] == "strict-v1"
        )
