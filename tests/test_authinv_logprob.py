"""T1 scorer: continuation boundaries, NumPy reference sums, candidate audit."""

from __future__ import annotations

import numpy as np
import pytest

from authinv.eval.logprob import candidate_audit, continuation_encoding, sum_logprob


class BoundaryTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        table = {
            "P": ([1], [(0, 1)]),
            "P cat": ([1, 2, 3], [(0, 1), (1, 2), (2, 5)]),
            "P!": ([1, 4], [(0, 1), (1, 2)]),
        }
        ids, offs = table.get(text, ([ord(c) for c in text], [(i, i + 1) for i in range(len(text))]))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


class MergeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs = ([1, 2], [(0, 1), (1, 2)]) if text == "PP" else ([1, 8, 3], [(0, 1), (1, 3), (3, 4)])
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


class SlowTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if return_offsets_mapping:
            raise NotImplementedError
        return {"input_ids": [ord(c) for c in text]}


def test_boundaries_whitespace_punctuation_and_multitoken():
    t = BoundaryTokenizer()
    a = continuation_encoding(t, "P", " cat")
    assert a["input_ids"] == [1, 2, 3] and a["continuation_start"] == 1 and a["token_count"] == 2
    assert continuation_encoding(t, "P", "!")["continuation_ids"] == [4]


def test_straddling_token_is_scored_and_flagged():
    r = continuation_encoding(MergeTokenizer(), "PP", "X")
    assert r["boundary_overlap"] and r["continuation_start"] == 1 and r["continuation_ids"] == [8, 3]


def test_slow_tokenizer_fallback():
    r = continuation_encoding(SlowTokenizer(), "ab", "cd")
    assert r["boundary_mode"] == "joint_common_prefix" and r["continuation_ids"] == [ord("c"), ord("d")]


def test_sum_logprob_matches_closed_form():
    logits = np.zeros((3, 5))
    total, n = sum_logprob(logits, [1, 2, 3], 1)
    assert n == 2 and np.isclose(total, -2 * np.log(5))
    logits[1, 3] = 10.0  # position 1 predicts token 3 strongly
    total2, _ = sum_logprob(logits, [1, 2, 3], 1)
    assert total2 > total
    with pytest.raises(ValueError):
        sum_logprob(logits, [1, 2, 3], 0)


def test_candidate_audit_reports_mismatched_lengths():
    audit = candidate_audit(BoundaryTokenizer(), "P", [" cat", "!"])
    assert audit["token_counts"] == {" cat": 2, "!": 1} and not audit["matched_token_counts"]


def test_hf_scorer_matches_reference_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import transformers

    from authinv.eval.logprob import HFScorer

    tok = transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    model = transformers.AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    scorer = HFScorer.__new__(HFScorer)
    scorer.torch, scorer.spec, scorer.tokenizer, scorer.model = (
        torch,
        {"precision": "float32"},
        tok,
        model.eval(),
    )
    scorer.device, scorer.template_kwargs = torch.device("cpu"), {}
    got = scorer.score("Owner: Agent", [" A", " Bob the builder"])
    for c, s in got.items():
        e = continuation_encoding(tok, "Owner: Agent", c)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([e["input_ids"]])).logits[0].numpy()
        ref, _ = sum_logprob(logits, e["input_ids"], e["continuation_start"])
        assert np.isclose(s["sum_logprob"], ref, atol=1e-4)


def test_score_many_matches_score_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import transformers

    from authinv.eval.logprob import HFScorer

    tok = transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    model = transformers.AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    scorer = HFScorer.__new__(HFScorer)
    scorer.torch, scorer.spec, scorer.tokenizer, scorer.model = (
        torch,
        {"precision": "float32"},
        tok,
        model.eval(),
    )
    scorer.device, scorer.template_kwargs = torch.device("cpu"), {}
    items = [
        ("Owner: Agent", [" A", " Bob the builder"]),
        ("A much longer prompt about policies", [" allow", " deny"]),
    ]
    many = scorer.score_many(items, max_batch_tokens=40)
    for (prompt, cands), got in zip(items, many, strict=True):
        one = scorer.score(prompt, cands)
        for c in cands:
            assert np.isclose(got[c]["sum_logprob"], one[c]["sum_logprob"], atol=1e-4)


def test_t1_prediction_and_applicability():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("authinv_t1_eval", root / "scripts" / "authinv_t1_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["authinv_t1_eval"] = mod
    spec.loader.exec_module(mod)
    assert not mod.t1_applicable({"thinking": True}) and not mod.t1_applicable({"output_format": "harmony"})
    assert mod.t1_applicable({"thinking": False})
    row = {
        "row_id": "r",
        "world_id": "w",
        "assignment": "orig",
        "instance": "q0",
        "rendering": "table",
        "task": "application",
        "label": "deny",
        "source": "s",
        "source_kind": "synthetic",
        "pair_key": "k",
    }
    sc = {
        mod.CANDIDATES["allow"]: {"sum_logprob": -1.0, "token_count": 6, "boundary_overlap": False},
        mod.CANDIDATES["deny"]: {"sum_logprob": -3.0, "token_count": 6, "boundary_overlap": False},
    }
    p = mod.to_prediction(row, sc)
    assert p["margin"] == -2.0 and p["category"] == "incorrect" and p["value"] == "allow"
