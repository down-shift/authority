"""P3.1 canonicalization arm: IR schema, parsing, semantic scoring, case building, two-stage runner."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from conftest import build_tiny_benchmark

from authinv import provenance
from authinv.benchmark import World, non_degenerate, world_rows
from authinv.mitigate import canonicalize as canon
from authinv.mitigate import metrics as mm
from authinv.mitigate.prompts import replace_policy, request_of, split_prompt
from authinv.policy import evaluate
from authinv.render.renderers import RENDERINGS, render, render_json
from authinv.sources.synthetic import generate

pytest.importorskip("cedarpy")  # the executable rendering is Cedar text

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "authinv_canonicalize", ROOT / "scripts" / "authinv_canonicalize.py"
)
runner = importlib.util.module_from_spec(spec)
sys.modules["authinv_canonicalize"] = runner
spec.loader.exec_module(runner)


def worlds(domain="repo", n=4):
    ws, _ = generate(domain, replicates=1, seed=3)
    out = [
        World(
            p, ("User",), ("Repo",) if domain == "repo" else ("Tool",), "synthetic", "synthetic", dict(p.meta)
        )
        for p in ws
    ]
    return [w for w in out if non_degenerate(w, 4)][:n]


def conditioned_world():
    ws, _ = generate("repo", replicates=1, seed=3)
    return next(p for p in ws if any(r.conditions for r in p.rules) and len(p.rules) >= 2)


def ir_obj(policy):
    return canon.gold_ir(policy)


# ---- IR format ---------------------------------------------------------------------------


def test_gold_ir_passes_its_schema_and_serializes_to_the_json_rendering():
    for domain in ("repo", "mcp"):
        ws, _ = generate(domain, replicates=1, seed=3)
        for pol in ws:
            g = ir_obj(pol)
            assert canon.schema_accepts(canon.ir_schema(pol), g)
            assert canon.ir_text(canon.ir_rules(g)) == render_json(pol)
            fenced = "```json\n" + json.dumps(g) + "\n```"
            assert canon.parse_conversion(fenced)["ir_text"] == render_json(pol)


def test_schema_holds_vocabulary_only_and_rejects_unknown_names():
    pol = conditioned_world()
    sch, g = canon.ir_schema(pol), ir_obj(pol)
    # the schema is a function of the vocabulary alone: other rules over the same world give the same schema
    assert canon.ir_schema(dataclasses.replace(pol, rules=pol.rules[:1])) == sch
    bad_entity = json.loads(json.dumps(g))
    bad_entity["rules"][0]["principal"] = {"match": "exactly", "entity": {"type": "User", "id": "u_nobody"}}
    bad_action = json.loads(json.dumps(g))
    bad_action["rules"][0]["actions"] = ["fly"]
    extra = json.loads(json.dumps(g))
    extra["rules"][0]["note"] = "x"
    for obj in (bad_entity, bad_action, extra, {"rules": g["rules"], "combining_rule": "x"}):
        assert not canon.schema_accepts(sch, obj)
    assert canon.schema_accepts(sch, {"rules": []})  # an empty policy is well formed (deny all)


def test_conversion_prompt_differs_only_in_the_policy_text():
    pol = conditioned_world()
    prompts = {g: canon.conversion_prompt('- User "u"', render(pol, g)) for g in RENDERINGS}
    heads = {p.split("\n\nPolicy:\n")[0] for p in prompts.values()}
    tails = {p.rsplit("\n\nCanonical format:", 1)[1] for p in prompts.values()}
    assert len(heads) == 1 and len(tails) == 1 and len(set(prompts.values())) == len(RENDERINGS)


@pytest.mark.parametrize(
    "text,category",
    [
        ("I can't help with rewriting this policy.", "refusal"),
        ('{"rules": [{"id": "r1", "effect": "allow"', "parse_failure"),
        ('{"rules": [{"id": "r1"}]}', "parse_failure"),
        ('{"policy": []}', "parse_failure"),
        (
            '{"rules": [{"id": "r1", "effect": "allow", "principal": {"match": "any"}, "actions": ["read"], '
            '"resource": {"match": "any"}, "conditions_mode": "all", '
            '"conditions": [{"left": 1, "op": "==", "right": {"attr": "principal.level"}}]}]}',
            "parse_failure",
        ),  # a condition's left side must be an attribute
        ("The policy is ambiguous.", "abstain"),
        ('<think>first the rules</think>{"rules": []}', "ir"),
        ('the rules are</think>\n{"rules": []}', "ir"),  # json-strict-v3: think opener in the prompt
    ],
)
def test_parse_conversion_categories(text, category):
    assert canon.parse_conversion(text)["conv_category"] == category


def _variant(policy, edit):
    g = json.loads(json.dumps(ir_obj(policy)))
    edit(g["rules"])
    return canon.ir_text(canon.ir_rules(g))


def test_score_ir_separates_exact_equivalent_valid():
    pol = conditioned_world()
    types = (("User",), ("Repo",))

    def score(text):
        return canon.score_ir(pol, text, *types)[0]

    def rename_and_reverse(rules):
        for i, r in enumerate(rules):
            r["id"] = f"x{i}"
        rules.reverse()

    def flip_effect(rules):
        rules[0]["effect"] = "deny" if rules[0]["effect"] == "allow" else "allow"

    negate = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}

    def add_dead_copy(rules):  # a copy that also requires the negation of its own condition never applies
        r = json.loads(json.dumps(next(x for x in rules if x["conditions"])))
        c = r["conditions"][0]
        r["id"] = "dup"
        r["conditions"].append({**c, "op": negate[c["op"]]})
        rules.append(r)

    def unknown_attr(rules):
        rules[0]["conditions"] = [{"left": {"attr": "principal.salary"}, "op": ">", "right": 3}]

    s = score(_variant(pol, rename_and_reverse))
    assert s["ir_valid"] and s["ir_exact"] and s["ir_equivalent"]
    s = score(_variant(pol, flip_effect))
    assert s["ir_valid"] and not s["ir_exact"] and not s["ir_equivalent"] and s["ir_disagreements"] > 0
    s = score(_variant(pol, add_dead_copy))
    assert s["ir_valid"] and not s["ir_exact"] and s["ir_equivalent"]
    s = score(_variant(pol, unknown_attr))
    assert not s["ir_valid"] and "salary" in s["ir_error"]
    s = score(canon.ir_text([]))
    assert s["ir_valid"] and s["ir_rules"] == 0 and not s["ir_equivalent"]


# ---- the frozen benchmark and the runner -----------------------------------------------------


def _benchmark(tmp_path):
    return build_tiny_benchmark(tmp_path)


def test_build_cases_validates_against_the_frozen_rows(tmp_path):
    _, rows, ws = _benchmark(tmp_path)
    cases, report = canon.build_cases(rows, ws, ("application",))
    assert report["cases"] == len(ws) * 2 * len(RENDERINGS) == len(cases)
    assert sum(c["queries"] for c in cases) == sum(r["task"] == "application" for r in rows)
    tampered = [dict(r) for r in rows]
    app = next(i for i, r in enumerate(tampered) if r["task"] == "application")
    tampered[app]["label"] = "allow" if tampered[app]["label"] == "deny" else "deny"
    with pytest.raises(ValueError, match="request read back"):
        canon.build_cases(tampered, ws, ("application",))
    edited = [dict(r) for r in rows]
    for r in edited:
        if r["rendering"] == "table" and r["task"] == "application":
            r["prompt"] = r["prompt"].replace("| allow |", "| deny |", 1)
    with pytest.raises(ValueError, match="frozen renderer|differ outside"):
        canon.build_cases(edited, ws, ("application",))


class FakeChat:
    """Oracle conversions (or a refusal for chosen renderings) and engine-correct decisions."""

    def __init__(self, gold_by_prompt, decision_by_prompt, refuse=()):
        self.gold, self.decide, self.refuse, self.calls = gold_by_prompt, decision_by_prompt, refuse, []

    def generate(self, prompts, schemas=None, max_tokens=None):
        self.calls.append({"n": len(prompts), "schemas": schemas is not None, "max_tokens": max_tokens})
        out = []
        for p in prompts:
            if p.startswith(canon.CONVERSION_INTRO):
                rendering = self.gold[p][1]
                text = "I cannot rewrite this." if rendering in self.refuse else json.dumps(self.gold[p][0])
            else:
                text = json.dumps({"decision": self.decide[p]})
            out.append(
                {"text": text, "finish_reason": "stop", "prompt_tokens": len(p) // 4, "output_tokens": 7}
            )
        return out


def _cfg():
    return yaml.safe_load((ROOT / "configs/authinv/p3_canonicalize.yaml").read_text())


def _oracle(rows, ws, cases):
    pols = {}
    for w in ws.values():
        for a in ("orig", "swap"):
            pols[(w["world_id"], a)] = runner.from_dict(w[a])
    gold = {c["prompt"]: (ir_obj(pols[(c["world_id"], c["assignment"])]), c["rendering"]) for c in cases}
    decide = {}
    for r in rows:
        if r["task"] == "application":
            pol = pols[(r["world_id"], r["assignment"])]
            p = replace_policy(r["prompt"], "application", render_json(pol))
            decide[p] = evaluate(pol, request_of(r, pol))["decision"]
    return gold, decide


def test_two_stage_run_with_oracle_conversions(tmp_path):
    _, rows, ws = _benchmark(tmp_path)
    cfg = _cfg()
    cases, _ = canon.build_cases(rows, ws, ("application",))
    gold, decide = _oracle(rows, ws, cases)
    chat = FakeChat(gold, decide, refuse=("nl_statement",))
    run = tmp_path / "run"
    run.mkdir()
    conv, dec, _ = runner.settings(cfg, "qwen3_5_4b")
    runner.execute(chat, rows, ws, cases, ("application",), conv, dec, 50, run, 0)
    convs = provenance.read_jsonl(run / "conversions.jsonl")
    preds = provenance.read_jsonl(run / "predictions.jsonl")
    app = [r for r in rows if r["task"] == "application"]
    assert len(convs) == len(cases) and len(preds) == len(app)
    # stage 1 is grammar-constrained with the conversion limit; stage 2 uses the T2 limit
    assert chat.calls[0]["schemas"] and chat.calls[0]["max_tokens"] == 6144
    assert not chat.calls[-1]["schemas"] and chat.calls[-1]["max_tokens"] == 256
    assert all(c["schema_sha256"] for c in convs)
    ok = [c for c in convs if c["rendering"] != "nl_statement"]
    assert all(c["ir_exact"] and c["ir_equivalent"] for c in ok)
    # an oracle IR turns every rendering's stage-2 prompt into the frozen T2 json_policy prompt
    t2_json = {
        (r["world_id"], r["assignment"], r["instance"]): r["prompt"]
        for r in app
        if r["rendering"] == "json_policy"
    }
    for p in preds:
        if p["rendering"] != "nl_statement":
            assert p["prompt"] == t2_json[(p["world_id"], p["assignment"], p["instance"])]
            assert p["category"] == "correct" and p["ir_request_correct"] is True
        else:  # refused conversion: kept as its own category, never sent to stage 2
            assert p["category"] == "refusal" and p["failed_stage"] == "conversion" and p["text"] is None
    # resume generates nothing new
    n_calls = len(chat.calls)
    runner.execute(chat, rows, ws, cases, ("application",), conv, dec, 50, run, 0)
    assert len(chat.calls) == n_calls and len(provenance.read_jsonl(run / "predictions.jsonl")) == len(app)

    # metrics against a baseline that gets owner_statement wrong everywhere
    base = [
        {
            **r,
            "category": "incorrect" if r["rendering"] == "owner_statement" else "correct",
            "value": ("deny" if r["label"] == "allow" else "allow")
            if r["rendering"] == "owner_statement"
            else r["label"],
            "prompt_tokens": 100,
            "output_tokens": 5,
        }
        for r in app
    ]
    m = runner.analyze(convs, preds, cfg, base, provenance.read_jsonl(run / "timing.jsonl"))
    conv_nl = m["conversion"]["per_rendering"]["nl_statement"]
    assert conv_nl["refusal_rate"]["mean"] == 1.0 and conv_nl["equivalent_rate"]["mean"] == 0.0
    assert m["conversion"]["per_rendering"]["table"]["equivalent_rate"]["mean"] == 1.0
    t = m["tasks"]["application"]
    assert (
        t["worst_rendering"] == "nl_statement"
        and t["per_rendering"]["nl_statement"]["refusal_rate"]["mean"] == 1.0
    )
    assert m["conversion_failed_rows"]["application"] == sum(r["rendering"] == "nl_statement" for r in app)
    gi = m["garbage_in"]["application"]["table"]
    assert gi["garbage_in"]["mean"] == 0.0 and gi["ir_engine_accuracy"]["mean"] == 1.0
    c = m["closure"]["application"]
    assert c["baseline"]["worst"] == "owner_statement" and c["baseline"]["gap"]["mean"] == 1.0
    assert c["closure_pair"]["mean"] == pytest.approx(1.0)  # owner_statement fixed, the best stays correct
    assert c["arm"]["worst"] == "nl_statement" and c["closure_own"]["mean"] == pytest.approx(0.0)
    cost = m["cost"]
    assert cost["per_rendering"]["table"]["conversion_tokens_per_query"] > 0
    assert cost["per_rendering"]["nl_statement"]["decision_tokens_per_query"] == 0
    assert cost["wall_time"]["conversion"]["requests"] == len(cases)


def test_garbage_in_counts_right_answers_from_wrong_irs():
    rows = []
    for w in range(4):
        for wrong in (True, False):
            rows.append(
                {
                    "world_id": f"w{w}",
                    "assignment": "orig",
                    "instance": f"q{int(wrong)}",
                    "rendering": "table",
                    "task": "application",
                    "category": "correct",
                    "conv_category": "ir",
                    "ir_equivalent": not wrong,
                    "ir_request_correct": not wrong,
                }
            )
    gi = mm.garbage_in(rows, "application", 200, 0)["table"]
    assert gi["garbage_in"]["mean"] == 0.5 and gi["garbage_in_request"]["mean"] == 0.5
    assert gi["accuracy_given_wrong_ir"]["mean"] == 1.0 and gi["ir_engine_accuracy"]["mean"] == 0.5


def test_closure_refuses_mismatched_rows():
    a = [
        {
            "world_id": "w",
            "assignment": "orig",
            "instance": "q0",
            "rendering": g,
            "task": "application",
            "category": "correct",
        }
        for g in ("a", "b")
    ]
    with pytest.raises(ValueError, match="same rows"):
        mm.gap_closure(a, a[:1], "application", 10, 0)


def test_settings_overrides_and_dataset_only_run(tmp_path):
    cfg = _cfg()
    conv, dec, gen = runner.settings(cfg, "gpt_oss_120b")
    assert conv["decoding"] == "free" and dec["max_tokens"] == 4096 and gen["max_tokens"] == 4096
    conv, dec, gen = runner.settings(cfg, "qwen3_5_4b")
    assert conv["decoding"] == "constrained" and gen["max_tokens"] == 256 and gen["max_model_len"] == 16384
    bench, rows, _ = _benchmark(tmp_path)
    unpinned = tmp_path / "cfg.yaml"
    unpinned.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "dataset_sha256"}))
    argv = [
        "x",
        "run",
        "--dataset",
        str(bench),
        "--model",
        "qwen3_5_4b",
        "--dataset-only",
        "--output-root",
        str(tmp_path / "out"),
    ]
    sys.argv = argv[:2] + ["--config", str(ROOT / "configs/authinv/p3_canonicalize.yaml")] + argv[2:]
    with pytest.raises(SystemExit, match="preregistered benchmark"):
        runner.main()
    sys.argv = argv[:2] + ["--config", str(unpinned)] + argv[2:]
    runner.main()
    run = next((tmp_path / "out").iterdir())
    assert json.loads((run / "run_status.json").read_text())["status"] == "dataset_only"
    report = json.loads((run / "validation_report.json").read_text())["cases"]
    assert report["passed"] and report["canon_version"] == "canon-v1"


def test_baseline_must_match_model_dataset_and_parser(tmp_path):
    from authinv import models

    run = tmp_path / "t2"
    run.mkdir()
    spec_ = models.model_spec("qwen3_5_4b")
    provenance.write_json(
        run / "config.json", {"dataset_sha256": "d", "model": spec_, "parser_version": "json-strict-v2"}
    )
    provenance.write_json(run / "run_status.json", {"status": "complete"})
    provenance.write_jsonl(run / "predictions.jsonl", [])
    with pytest.raises(SystemExit, match="parser json-strict-v2"):
        runner.load_baseline(run, spec_, "d")
    with pytest.raises(SystemExit, match="dataset differs"):
        runner.load_baseline(run, spec_, "e")


def test_split_prompt_keeps_blank_lines_inside_a_rendering():
    pol = conditioned_world()
    rows, _ = world_rows(World(pol, ("User",), ("Repo",), "synthetic", "synthetic"), per_label=2, seed=1)
    rego = next(r for r in rows if r["rendering"] == "rego" and r["task"] == "application")
    _, text, tail = split_prompt(rego["prompt"], "application")
    assert "\n\n" in text and text == render(pol, "rego").rstrip() and tail.startswith("\n\nRequest: ")
