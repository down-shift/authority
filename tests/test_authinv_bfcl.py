"""P4.2b: the BFCL-subset check (bfcl-ast-v1) and adapter evaluation inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from authinv import bfcl, provenance
from authinv.sft.adapter import load_adapter

ROOT = Path(__file__).resolve().parents[1]

AREA = {
    "name": "geometry.triangle_area",
    "description": "Area of a triangle.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer", "description": "Base."},
            "height": {"type": "float", "description": "Height."},
            "unit": {"type": "string", "description": "Unit."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags."},
        },
        "required": ["base", "height"],
    },
}
OTHER = {
    "name": "weather.today",
    "description": "Weather.",
    "parameters": {
        "type": "dict",
        "properties": {"city": {"type": "string", "description": "City."}},
        "required": ["city"],
    },
}
GT = [
    {"geometry.triangle_area": {"base": [10], "height": [5.0], "unit": ["cm", ""], "tags": [["a", "b"], ""]}}
]


def entry(functions, i="simple_0"):
    return {
        "id": i,
        "question": [[{"role": "user", "content": "Area of a 10 by 5 triangle?"}]],
        "function": functions,
    }


def test_prompt_is_the_official_default_format():
    msgs = bfcl.messages(entry([AREA]))
    sp = msgs[0]["content"]
    assert msgs[0]["role"] == "system" and msgs[1] == {
        "role": "user",
        "content": "Area of a 10 by 5 triangle?",
    }
    assert sp.startswith(
        "You are an expert in composing functions.You are given a question"
    )  # official quirk
    assert "func_name2(params)].  You SHOULD NOT include any other text" in sp
    assert "Here is a list of functions in json format that you can invoke.\n[\n    {" in sp
    docs = json.loads(sp.split("you can invoke.\n", 1)[1])
    assert (
        docs[0]["description"] == "Area of a triangle. Note that the provided function is in Python 3 syntax."
    )
    assert AREA["description"] == "Area of a triangle."  # the input is not mutated
    with_system = entry([AREA])
    with_system["question"][0].insert(0, {"role": "system", "content": "Be brief."})
    m2 = bfcl.messages(with_system)
    assert len(m2) == 2 and m2[0]["content"].endswith("\n\nBe brief.")


def test_decode_and_check():
    ok = "[geometry.triangle_area(base=10, height=5, unit='CM')]"
    assert bfcl.decode(ok) == [{"geometry.triangle_area": {"base": 10, "height": 5, "unit": "CM"}}]
    assert bfcl.decode("```geometry.triangle_area(base=2*5, height=-1.5)```") == [
        {"geometry.triangle_area": {"base": 10, "height": -1.5}}
    ]
    a = {"ground_truth": GT}
    e = entry([AREA])
    assert bfcl.score(e, a, ok, "simple")["valid"]  # int -> float allowed; strings case-insensitive
    assert bfcl.score(e, a, "geometry.triangle_area(base=10, height=5.0, tags=['A', 'b'])", "simple")["valid"]
    cases = {
        "geometry.triangle_area(base=11, height=5)": "value_error:others",
        "geometry.triangle_area(base=10)": "simple_function_checker:missing_required",
        "geometry.triangle_area(base=10, height=5, color='red')": "simple_function_checker:unexpected_param",
        "geometry.triangle_area(base='10', height=5)": "type_error:simple",
        "triangle_area(base=10, height=5)": "simple_function_checker:wrong_func_name",
        "[geometry.triangle_area(base=10, height=5), geometry.triangle_area(base=10, height=5)]": (
            "simple_function_checker:wrong_count"
        ),
        "I cannot compute that.": "ast_decoder:decoder_failed",
        "geometry.triangle_area(base=__import__('os').getcwd(), height=5)": "type_error:simple",
        "geometry.triangle_area(base=10**999, height=5)": "ast_decoder:decoder_failed",
    }
    for text, err in cases.items():
        r = bfcl.score(e, a, text, "simple")
        assert not r["valid"] and r["error_type"] == err, (text, r)
    m = entry([OTHER, AREA], "multiple_0")
    assert bfcl.score(m, a, ok, "multiple")["valid"]
    assert bfcl.score(m, a, "weather.today(city='Paris')", "multiple")["error_type"] == (
        "simple_function_checker:wrong_func_name"
    )
    with pytest.raises(ValueError):
        bfcl.ast_check("parallel", [AREA], [], GT)


def test_accuracy_and_regression_rule():
    base = [{"id": f"s{i}", "category": "simple", "valid": i % 10 != 0} for i in range(200)]
    same = [dict(r) for r in base]
    worse = [{**r, "valid": r["valid"] and int(r["id"][1:]) % 4 != 1} for r in base]
    acc = bfcl.accuracy(base, 500, 1)
    assert acc["simple"]["mean"] == pytest.approx(0.9) and acc["all"]["items"] == 200
    assert not bfcl.paired_difference(same, base, 500, 1, 0.02)["regression"]
    d = bfcl.paired_difference(worse, base, 500, 1, 0.02)
    assert d["mean"] < -0.02 and d["ci95"][1] < 0 and d["regression"]
    with pytest.raises(ValueError):
        bfcl.paired_difference(worse[1:], base, 10, 1, 0.02)


def _sft_run(root: Path, key: str, status: str = "complete") -> Path:
    from authinv import models

    run = root / f"sft_{key}"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "adapter" / "adapter_config.json").write_text("{}")
    shas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run / "adapter").iterdir())}
    cfg = yaml.safe_load((ROOT / "configs/authinv/sft_lora.yaml").read_text())
    provenance.write_json(
        run / "config.json",
        {"config": cfg, "model": models.model_spec(key), "arm": "multi", "seed": 20261111},
    )
    provenance.write_json(run / "metrics.json", {"adapter_sha256": shas})
    provenance.write_run_status(run, status, 1, 1)
    return run


def test_load_adapter_checks_model_status_and_hashes(tmp_path):
    from authinv import models

    spec = models.model_spec("smoke_qwen3_0_6b")
    run = _sft_run(tmp_path, "smoke_qwen3_0_6b")
    a = load_adapter(run, "smoke_qwen3_0_6b", spec)
    assert a["rank"] == 32 and a["arm"] == "multi" and a["path"].endswith("adapter") and not a["smoke"]
    with pytest.raises(SystemExit, match="trained on"):
        load_adapter(run, "qwen3_5_9b", models.model_spec("qwen3_5_9b"))
    smoke = _sft_run(tmp_path / "s", "smoke_qwen3_0_6b", "smoke")
    with pytest.raises(SystemExit, match="status smoke"):
        load_adapter(smoke, "smoke_qwen3_0_6b", spec)
    assert load_adapter(smoke, "smoke_qwen3_0_6b", spec, smoke=True)["smoke"]
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"edited")
    with pytest.raises(SystemExit, match="sha256"):
        load_adapter(run, "smoke_qwen3_0_6b", spec)


class FakeChat:
    seen: list = []

    def __init__(self, spec, model_dir, gen, tp, lora=None):
        FakeChat.lora = lora

    def provenance(self):
        return {"backend": "fake"}

    def generate_chats(self, convs, schemas=None, max_tokens=None):
        FakeChat.seen += convs
        return [
            {"text": "geometry.triangle_area(base=10, height=5)", "finish_reason": "stop", "output_tokens": 9}
            if "triangle" in c[-1]["content"]
            else {"text": "no call", "finish_reason": "stop", "output_tokens": 2}
            for c in convs
        ]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bfcl_runner_scores_base_and_adapter_and_compares(tmp_path, monkeypatch):
    from authinv.eval import generation

    data = tmp_path / "data"
    (data / "possible_answer").mkdir(parents=True)
    qs = {
        "simple": [entry([AREA], f"simple_{i}") for i in range(3)],
        "multiple": [entry([OTHER, AREA], "multiple_0")],
    }
    qs["simple"][2]["question"][0][0]["content"] = "Weather in Paris?"
    for cat, items in qs.items():
        (data / f"BFCL_v3_{cat}.json").write_text("\n".join(json.dumps(q) for q in items))
        (data / "possible_answer" / f"BFCL_v3_{cat}.json").write_text(
            "\n".join(json.dumps({"id": q["id"], "ground_truth": GT}) for q in items)
        )
    cfg = yaml.safe_load((ROOT / "configs/authinv/bfcl.yaml").read_text())
    cfg["files"] = {f: hashlib.sha256((data / f).read_bytes()).hexdigest() for f in cfg["files"]}
    cfg["categories"] = {"simple": 3, "multiple": 1}
    cfg["analysis"]["bootstrap_replicates"] = 200
    (tmp_path / "bfcl.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(generation, "VLLMChat", FakeChat)
    mod = _module("authinv_bfcl_t", ROOT / "scripts" / "authinv_bfcl.py")
    base = ["x", "run", "--config", str(tmp_path / "bfcl.yaml"), "--data", str(data)]
    base += ["--model", "smoke_qwen3_0_6b", "--output-root", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", base)
    mod.main()
    sft = _sft_run(tmp_path, "smoke_qwen3_0_6b")
    monkeypatch.setattr(sys, "argv", base + ["--adapter", str(sft)])
    mod.main()
    assert FakeChat.lora["arm"] == "multi" and FakeChat.lora["rank"] == 32
    b_run, a_run = sorted((tmp_path / "out").iterdir())
    assert a_run.name.endswith("_bfcl_smoke_qwen3_0_6b_multi_s20261111")
    m = json.loads((b_run / "metrics.json").read_text())
    assert (
        m["accuracy"]["simple"]["mean"] == pytest.approx(2 / 3) and m["accuracy"]["multiple"]["mean"] == 1.0
    )
    assert m["error_types"]["simple"] == {"ast_decoder:decoder_failed": 1}
    assert FakeChat.seen[0][0]["role"] == "system" and "Python 3 syntax" in FakeChat.seen[0][0]["content"]
    out = tmp_path / "cmp.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "compare", "--config", str(tmp_path / "bfcl.yaml"), "--base", str(b_run)]
        + ["--runs", str(a_run), "--out", str(out)],
    )
    mod.main()
    c = json.loads(out.read_text())["runs"]["multi_s20261111"]["difference"]
    assert c["all"]["mean"] == 0.0 and not c["all"]["regression"]
    arm = json.loads(out.read_text())["arms"]["multi"]
    assert arm["seeds"] == 1 and arm["difference"]["all"]["mean"] == 0.0
    (data / "BFCL_v3_simple.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit, match="differs from the pinned"):
        mod.main()


def test_t2_runner_evaluates_an_adapter(tmp_path, monkeypatch):
    from conftest import build_tiny_benchmark

    from authinv.eval import generation

    pytest.importorskip("cedarpy")
    bench, rows, _ = build_tiny_benchmark(tmp_path, 1)

    class T2Chat(FakeChat):
        def generate(self, prompts, schemas=None, max_tokens=None):
            return [{"text": '{"decision": "deny"}', "finish_reason": "stop"} for _ in prompts]

    monkeypatch.setattr(generation, "VLLMChat", T2Chat)
    mod = _module("authinv_eval_t", ROOT / "scripts" / "authinv_eval.py")
    cfg = yaml.safe_load((ROOT / "configs/authinv/t2.yaml").read_text())
    cfg.pop("dataset_sha256", None)
    (tmp_path / "t2.yaml").write_text(yaml.safe_dump(cfg))
    sft = _sft_run(tmp_path, "smoke_qwen3_0_6b", "smoke")
    base = [
        "x",
        "run",
        "--config",
        str(tmp_path / "t2.yaml"),
        "--dataset",
        str(bench),
        "--model",
        "smoke_qwen3_0_6b",
    ]
    base += ["--output-root", str(tmp_path / "out"), "--adapter", str(sft)]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit, match="status smoke"):  # a smoke adapter only for a smoke evaluation
        mod.main()
    monkeypatch.setattr(sys, "argv", base + ["--limit", "8"])
    mod.main()
    runs = [r for r in (tmp_path / "out").iterdir() if (r / "metadata.json").exists()]
    (run,) = runs
    assert run.name.endswith("_t2sft_smoke_qwen3_0_6b_multi_s20261111")
    conf = json.loads((run / "config.json").read_text())
    assert (
        conf["adapter"]["files_sha256"]
        == load_adapter(sft, "smoke_qwen3_0_6b", conf["model"], True)["files_sha256"]
    )
    assert T2Chat.lora["path"] == conf["adapter"]["path"]
