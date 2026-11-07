"""P3.2 baseline arms: prompt variants, the final-answer parser, self-consistency, solver pipeline, runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from conftest import build_tiny_benchmark

from authinv import models, provenance
from authinv.mitigate import baselines as bl
from authinv.mitigate import canonicalize as canon
from authinv.mitigate.prompts import request_of, split_prompt
from authinv.policy import evaluate, from_dict

pytest.importorskip("cedarpy")  # the executable rendering is Cedar text

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("authinv_baselines", ROOT / "scripts" / "authinv_baselines.py")
runner = importlib.util.module_from_spec(spec)
sys.modules["authinv_baselines"] = runner
spec.loader.exec_module(runner)
CFG = yaml.safe_load((ROOT / "configs/authinv/p3_baselines.yaml").read_text())


def test_arm_prompts_change_only_the_answer_instruction(tmp_path):
    _, rows, _ = build_tiny_benchmark(tmp_path, 1)
    for r in rows:
        for arm in bl.MODEL_ARMS:
            p = bl.arm_prompt(r["prompt"], arm)
            head, text, tail = split_prompt(r["prompt"], r["task"])
            assert split_prompt(p, r["task"])[:2] == (head, text)  # facts and policy byte-identical
            assert bl.ANSWER_MARK not in p and p.endswith(r["prompt"].split(bl.ANSWER_MARK)[1])
    assert canon.IR_FORMAT in bl.arm_prompt(rows[0]["prompt"], "restate")
    with pytest.raises(ValueError, match="unique answer"):
        bl.arm_prompt("no instruction here", "cot")


@pytest.mark.parametrize(
    "text,category,value",
    [
        ('r1 allows merge for members.\nSo the answer:\n{"decision": "allow"}', "correct", "allow"),
        (
            'Maybe {"decision": "deny"}? No: r1 applies.\n```json\n{"decision": "allow"}\n```',
            "correct",
            "allow",
        ),
        ('{"decision": "allow"}\nOn reflection the forbid wins.\n{"decision": "deny"}', "incorrect", "deny"),
        ('{"decision": "allow"}\nHope this helps.', "correct", "allow"),
        ('<think>{"decision": "deny"}</think>\n{"decision": "allow"}', "correct", "allow"),
        ("Step 1: both rules... I can't help with authorization decisions.", "refusal", None),
        ("Rule r1 is unknown to me, but r2 says...\nIt is ambiguous.", "abstain", None),
        ('Step 1: r1 applies.\n{"decision": "maybe"}', "parse_failure", None),
        ('The answer is {"decision": "allow"', "parse_failure", None),
    ],
)
def test_parse_final_takes_the_last_valid_answer(text, category, value):
    assert bl.parse_final(text, "application", "allow") == {"category": category, "value": value}


def test_parse_final_interpretation_and_restated_ir():
    out = bl.parse_final(
        'first {"principals": ["u2"]}, finally {"principals": ["u2", "u1"]}', "interpretation", ["u1", "u2"]
    )
    assert out == {"category": "correct", "value": ("u1", "u2")}
    ir = {"rules": []}
    text = f'Restated:\n{json.dumps(ir)}\nNo rule allows it.\n{{"decision": "deny"}}'
    assert bl.restated_ir(text) == {"conv_category": "ir", "ir_text": canon.ir_text([])}
    assert bl.parse_final(text, "application", "deny")["category"] == "correct"
    assert bl.restated_ir('{"decision": "deny"}')["conv_category"] == "parse_failure"


def _t2_rows(rows, wrong=("rego",), unparsed=()):
    """T2-like predictions: correct except on `wrong` renderings, parse failures on `unparsed`."""
    out = []
    for r in rows:
        if r["task"] != "application":
            continue
        flip = "deny" if r["label"] == "allow" else "allow"
        if r["rendering"] in unparsed:
            cat, val = "parse_failure", None
        elif r["rendering"] in wrong:
            cat, val = "incorrect", flip
        else:
            cat, val = "correct", r["label"]
        out.append(
            {
                **{k: v for k, v in r.items() if k != "prompt"},
                "category": cat,
                "value": val,
                "prompt_tokens": 100,
                "output_tokens": 5,
            }
        )
    return out


def test_vote_rows_majority_ties_and_tokens(tmp_path):
    _, rows, _ = build_tiny_benchmark(tmp_path, 2)
    t2 = _t2_rows(rows, wrong=("rego", "table"))
    votes, stats = bl.vote_rows(t2, "application", "abstain")
    assert len(votes) == len(t2) and stats["split"] == stats["instances"]
    assert all(v["category"] == "correct" and v["prompt_tokens"] == 600 for v in votes)  # 4 right vs 2 wrong
    tied = _t2_rows(rows, wrong=("rego", "table", "json_policy"))
    v_abs, s_abs = bl.vote_rows(tied, "application", "abstain")
    assert s_abs["tie"] == s_abs["instances"] and {v["category"] for v in v_abs} == {"abstain"}
    v_deny, _ = bl.vote_rows(tied, "application", "deny")
    assert all(
        v["value"] == "deny" and (v["category"] == "correct") == (v["label"] == "deny") for v in v_deny
    )
    none, s_none = bl.vote_rows(
        _t2_rows(rows, unparsed=tuple(r["rendering"] for r in rows)), "application", "abstain"
    )
    assert s_none["no_votes"] == s_none["instances"]
    with pytest.raises(ValueError):
        bl.vote_rows(t2, "interpretation", "deny")


def _conversions(rows, worlds, broken=()):
    """Oracle conversions; `broken` renderings get an invalid IR (unknown attribute), nl a refusal."""
    out = []
    for c in canon.build_cases(rows, worlds, ("application",))[0]:
        gold = from_dict(worlds[c["world_id"]][c["assignment"]])
        if c["rendering"] == "nl_statement":
            out.append(
                {
                    **c,
                    "conv_category": "refusal",
                    "ir_text": None,
                    "ir_valid": False,
                    "ir_exact": False,
                    "ir_equivalent": False,
                }
            )
            continue
        g = canon.gold_ir(gold)
        if c["rendering"] in broken:
            g["rules"][0]["conditions"] = [{"left": {"attr": "principal.salary"}, "op": "==", "right": 1}]
        text = canon.ir_text(canon.ir_rules(g))
        scores, _ = canon.score_ir(gold, text, ("User",), ("Repo",))
        out.append(
            {
                **c,
                "conv_category": "ir",
                "ir_text": text,
                **scores,
                "prompt_tokens": 400,
                "output_tokens": 200,
            }
        )
    return out


def test_solver_rows_decide_with_the_engine_and_keep_unusable_irs_apart(tmp_path):
    _, rows, worlds = build_tiny_benchmark(tmp_path, 2)
    convs = _conversions(rows, worlds, broken=("rego",))
    out, stats = bl.solver_rows(rows, convs, worlds, cedar_check=True)
    app = [r for r in rows if r["task"] == "application"]
    assert len(out) == len(app) and stats["queries"] == len(app)
    assert stats["cedar_checked"] == stats["cedar_agree"] > 0  # Cedar agrees with the reference evaluator
    for r in out:
        if r["rendering"] == "nl_statement":
            assert r["category"] == "refusal" and r["failed_stage"] == "conversion"
        elif r["rendering"] == "rego":
            assert r["category"] == "parse_failure" and r["failed_stage"] == "ir_invalid"
        else:
            assert r["category"] == "correct"
        if r.get("failed_stage"):
            assert r["value_fail_closed"] == "deny" and (r["category_fail_closed"] == "correct") == (
                r["label"] == "deny"
            )
    with pytest.raises(ValueError, match="no conversion"):
        bl.solver_rows(rows, convs[:3], worlds, cedar_check=False)


class FakeChat:
    def __init__(self, worlds, arm):
        self.worlds, self.arm, self.calls = worlds, arm, []

    def generate(self, prompts, schemas=None, max_tokens=None):
        self.calls.append(max_tokens)
        return [
            {
                "text": self.answer(p),
                "finish_reason": "stop",
                "prompt_tokens": len(p) // 4,
                "output_tokens": 50,
            }
            for p in prompts
        ]

    def answer(self, p):
        return self.lookup[p]


def _items(rows, arm):
    return [
        {**{k: v for k, v in r.items() if k != "prompt"}, "prompt": bl.arm_prompt(r["prompt"], arm)}
        for r in rows
        if r["task"] == "application"
    ]


def test_model_arms_execute_score_and_summarize(tmp_path):
    _, rows, worlds = build_tiny_benchmark(tmp_path, 2)
    for arm in bl.MODEL_ARMS:
        items = _items(rows, arm)
        chat = FakeChat(worlds, arm)
        chat.lookup = {}
        for it, r in zip(items, [r for r in rows if r["task"] == "application"], strict=True):
            pol = from_dict(worlds[r["world_id"]][r["assignment"]])
            dec = evaluate(pol, request_of(r, pol))["decision"]
            restated = json.dumps(canon.gold_ir(pol)) + "\n" if arm == "restate" else "Step by step: "
            chat.lookup[it["prompt"]] = f'{restated}The rules give:\n{{"decision": "{dec}"}}'
        run = tmp_path / arm
        run.mkdir()
        _, gen = runner.arm_settings(CFG, arm, "qwen3_5_4b")
        runner.execute(chat, items, worlds, arm, {**gen, "batch_size": 40}, run, len(items))
        assert set(chat.calls) == {CFG["arms"][arm]["max_tokens"]}
        preds = provenance.read_jsonl(run / "predictions.jsonl")
        assert len(preds) == len(items) and all(p["category"] == "correct" for p in preds)
        base = _t2_rows(rows, wrong=("rego",))
        m = runner.arm_metrics(
            arm, preds, CFG, (base, {"run": "t2"}), provenance.read_jsonl(run / "timing.jsonl")
        )
        c = m["closure"]["application"]
        assert c["baseline"]["worst"] == "rego" and c["closure_own"]["mean"] == pytest.approx(1.0)
        assert m["cost"]["per_rendering"]["table"]["token_ratio"] > 0
        if arm == "restate":
            assert all(p["ir_exact"] and p["ir_request_correct"] for p in preds)
            assert m["restatement"]["per_rendering"]["rego"]["equivalent_rate"]["mean"] == 1.0
            assert m["garbage_in"]["application"]["table"]["garbage_in"]["mean"] == 0.0


def _run_dir(path, conf, rows, extra=None):
    path.mkdir()
    provenance.write_json(path / "config.json", conf)
    provenance.write_json(path / "run_status.json", {"status": "complete"})
    provenance.write_jsonl(path / "predictions.jsonl", rows)
    for name, value in (extra or {}).items():
        provenance.write_jsonl(path / name, value)
    return path


def test_derive_self_consistency_and_solver_runs(tmp_path):
    bench, rows, worlds = build_tiny_benchmark(tmp_path, 2)
    sha = json.loads((bench / "dataset_manifest.json").read_text())["sha256"]
    spec_ = models.model_spec("qwen3_5_4b")
    t2 = _run_dir(
        tmp_path / "t2",
        {"dataset_sha256": sha, "model": spec_, "parser_version": "json-strict-v3"},
        _t2_rows(rows, wrong=("rego", "table")),
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in CFG.items() if k != "dataset_sha256"}))
    out = tmp_path / "out"
    sys.argv = [
        "x",
        "derive",
        "--arm",
        "self_consistency",
        "--config",
        str(cfg_path),
        "--t2-run",
        str(t2),
        "--output-root",
        str(out),
    ]
    runner.main()
    sc = next(out.glob("*_authinv_p3self_consistency_qwen3_5_4b"))
    m = json.loads((sc / "metrics.json").read_text())
    assert m["analysis_only"] and m["tasks"]["application"]["worst_case_gap"]["mean"] == 0.0
    assert m["closure"]["application"]["closure_own"]["mean"] == pytest.approx(1.0)
    assert m["cost"]["per_rendering"]["rego"]["token_ratio"] == pytest.approx(6.0)
    convs = _conversions(rows, worlds, broken=("rego",))
    canon_run = _run_dir(
        tmp_path / "canon",
        {"dataset_sha256": sha, "model": spec_, "canon_version": canon.CANON_VERSION},
        [],
        {
            "conversions.jsonl": convs,
            "timing.jsonl": [{"stage": "conversion", "requests": len(convs), "seconds": 2.0}],
        },
    )
    sys.argv = [
        "x",
        "derive",
        "--arm",
        "solver",
        "--config",
        str(cfg_path),
        "--dataset",
        str(bench),
        "--canon-run",
        str(canon_run),
        "--baseline",
        str(t2),
        "--output-root",
        str(out),
    ]
    runner.main()
    so = next(out.glob("*_authinv_p3solver_qwen3_5_4b"))
    m = json.loads((so / "metrics.json").read_text())
    pr = m["tasks"]["application"]["per_rendering"]
    assert pr["table"]["accuracy"]["mean"] == 1.0 and pr["nl_statement"]["refusal_rate"]["mean"] == 1.0
    assert pr["rego"]["parse_failure_rate"]["mean"] == 1.0
    fc = m["fail_closed"]["tasks"]["application"]["per_rendering"]
    assert fc["rego"]["deny_to_allow_rate"]["mean"] == 0.0 and fc["rego"]["allow_to_deny_rate"]["mean"] == 1.0
    assert m["engine"]["cedar_agree"] == m["engine"]["cedar_checked"] > 0
    assert m["cost"]["wall_time"]["engine"]["requests"] == m["engine"]["queries"]
    assert json.loads((so / "config.json").read_text())["sources"]["conversions_sha256"]


def test_config_versions_and_proxy_config_matches_t3():
    runner.check_config(CFG)
    with pytest.raises(SystemExit, match="versions differ"):
        runner.check_config({**CFG, "final_parser_version": "json-final-v0"})
    t3 = yaml.safe_load((ROOT / "configs/authinv/t3.yaml").read_text())
    proxy = yaml.safe_load((ROOT / "configs/authinv/p3_proxy_t3.yaml").read_text())
    drop = {"experiment", "enforce", "proxy_version", "non_harmony_compatible_agent_versions"}
    assert {k: v for k, v in proxy.items() if k not in drop} == {k: v for k, v in t3.items() if k not in drop}
    assert proxy["enforce"] is True and proxy["proxy_version"] == "proxy-v1"
