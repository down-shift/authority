import numpy as np
import pytest

from authority_leakage.models.continuation import continuation_encoding, continuation_logprob
from indirection.scoring import score_example

torch = pytest.importorskip("torch")


class BoundaryTokenizer:
    def __init__(self):
        self.chat_template = ""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # ' prompt' is one token spanning a boundary if joined; standalone prompt differs.
        if text == "P":
            ids = [1]
            offsets = [(0, 1)]
        elif text == "P cat":
            ids = [1, 2, 3]
            offsets = [(0, 1), (1, 2), (2, 5)]
        elif text == "P!":
            ids = [1, 4]
            offsets = [(0, 1), (1, 2)]
        else:
            ids = [ord(c) for c in text]
            offsets = [(i, i + 1) for i in range(len(text))]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_joint_encoding_and_continuation_boundary_whitespace_punctuation_multitoken():
    t = BoundaryTokenizer()
    a = continuation_encoding(t, "P", " cat")
    assert a["input_ids"] == [1, 2, 3] and a["continuation_start"] == 1 and len(a["continuation_ids"]) == 2
    b = continuation_encoding(t, "P", "!")
    assert b["continuation_start"] == 1 and b["continuation_ids"] == [4]


def test_joint_boundary_token_is_included_after_stable_prefix():
    class MergeTokenizer:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            if text == "PP":
                ids, offs = [1, 2], [(0, 1), (1, 2)]
            else:
                ids, offs = [1, 8, 3], [(0, 1), (1, 3), (3, 4)]
            out = {"input_ids": ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offs
            return out

    result = continuation_encoding(MergeTokenizer(), "PP", "X")
    assert result["boundary_overlap"] is True
    assert result["continuation_start"] == 1
    assert result["continuation_ids"] == [8, 3]


def test_toy_logits_score_every_continuation_token():
    # At each position, chosen next-token logprob is -ln(2); summed over 2 tokens.
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]])
    full = torch.tensor([[1, 2, 3]])
    total, count = continuation_logprob(logits, full, 1, torch)
    assert count == 2
    assert np.isclose(total, -2 * np.log(5), rtol=1e-6)


def test_semantic_margin_oriented_to_correct_candidate():
    class A:
        def render(self, m):
            return m[0].content

        def score_candidates_detailed(self, m, c):
            return {c[0]: {"sum_logprob": -1, "token_count": 2}, c[1]: {"sum_logprob": -4, "token_count": 1}}

    row = {"prompt": "p", "correct_candidate": "right", "incorrect_candidate": "wrong"}
    result = score_example(row, A(), "toy")
    assert result["margin"] == 3 and result["correct"]
