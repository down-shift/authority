#!/usr/bin/env python3
"""Build the Phase-4 SFT data (P4.1): train-only synthetic worlds, four renderings, Cedar held out.

  uv run --extra inference python scripts/authinv_build_sft.py --config configs/authinv/sft_data.yaml \
      --benchmark outputs/<frozen benchmark run> [--token-counter hf|chars] [--output-root DIR]

Writes outputs/<stamp>_authinv_sft_data/: worlds.jsonl (canonical training policies with their
set), train_multi.jsonl, val_multi.jsonl, control_<model>.jsonl (one token-matched control per
model), and manifest.json: file hashes, counts, tokens per set and model, the disjointness
report against the benchmark (refuses on any overlap), the held-out renderings, and the no-SFT
baseline. `--token-counter hf` (the default) counts tokens with each model's pinned tokenizer and
chat template; `chars` (len/4) is for tests only and is recorded as such.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import models, provenance  # noqa: E402
from authinv.benchmark import rename, swap_axis, swap_map  # noqa: E402
from authinv.policy import to_dict  # noqa: E402
from authinv.render.renderers import RENDERINGS  # noqa: E402
from authinv.sft import data  # noqa: E402

TRACKED = [ROOT / "scripts" / "authinv_build_sft.py"] + [
    ROOT / "src" / "authinv" / p
    for p in ("sft/data.py", "benchmark.py", "sources/synthetic.py", "render/renderers.py", "provenance.py")
]


MIN_TOKENS_PER_EXAMPLE = 50


def hf_counter(key: str) -> tuple:
    """(count(example) -> tokens, provenance) with the model's pinned tokenizer and chat template."""
    import transformers

    spec = models.model_spec(key)
    tok = transformers.AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"])
    kwargs = {"enable_thinking": bool(spec["thinking"])} if "thinking" in spec else {}

    def count(ex: dict) -> int:
        # transformers 5 returns an encoding dict here, so read input_ids (len() of it would count its keys)
        ids = tok.apply_chat_template(ex["messages"], tokenize=True, return_dict=True, **kwargs)["input_ids"]
        return len(ids[0] if ids and isinstance(ids[0], list) else ids)

    template = getattr(tok, "chat_template", None) or ""
    return count, {
        "counter": "hf",
        "name": spec["name"],
        "revision": spec["revision"],
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "chat_template_kwargs": kwargs,
        "transformers_version": transformers.__version__,
    }


def chars_counter(key: str) -> tuple:
    return (lambda ex: sum(len(m["content"]) for m in ex["messages"]) // 4), {
        "counter": "chars/4 (tests only)"
    }


def summary(examples: list[dict]) -> dict:
    return {
        "examples": len(examples),
        "worlds": len({e["world_id"] for e in examples}),
        "by_rendering": dict(sorted(collections.Counter(e["rendering"] for e in examples).items())),
        "by_task": dict(sorted(collections.Counter(e["task"] for e in examples).items())),
        "by_domain": dict(sorted(collections.Counter(e["domain"] for e in examples).items())),
        "application_labels": dict(
            sorted(
                collections.Counter(
                    json.loads(e["messages"][1]["content"])["decision"]
                    for e in examples
                    if e["task"] == "application"
                ).items()
            )
        ),
    }


def no_sft_baseline(cfg: dict) -> dict:
    figs = yaml.safe_load((ROOT / cfg["phase2_figures"]).read_text())
    runs = {m["key"]: Path(m["t2"]).name for m in figs["models"]}
    return {
        "definition": "each base model at its pinned revision, without an adapter, on the frozen benchmark "
        "through the Phase-2 T2 harness (configs/authinv/t2.yaml): the Phase-2 T2 run of that model",
        "models": {
            k: {"spec": models.model_spec(k), "phase2_t2_run": runs.get(k)} for k in cfg["token_models"]
        },
    }


def build(cfg: dict, benchmark: Path, counter: str, out_root: Path) -> Path:
    manifest = json.loads((benchmark / "dataset_manifest.json").read_text())
    if manifest["sha256"] != cfg["eval_benchmark_sha256"]:
        raise SystemExit(
            f"benchmark {manifest['sha256']} is not the eval benchmark {cfg['eval_benchmark_sha256']}"
        )
    if cfg["data_version"] != data.DATA_VERSION:
        raise SystemExit(f"config data_version {cfg['data_version']} != code {data.DATA_VERSION}")
    train_r, held = tuple(cfg["train_renderings"]), set(cfg["held_out_renderings"])
    if (
        held & set(train_r)
        or set(train_r) | held != set(RENDERINGS)
        or cfg["control_rendering"] not in train_r
    ):
        raise SystemExit(
            "renderings: training and held-out must partition all six; the control must be a training one"
        )
    reps = cfg["replicates"]
    n_rep = reps["multi"] + reps["val"] + reps["control_pool"]
    sets = {"multi": [], "val": [], "control_pool": []}
    for di, domain in enumerate(cfg["domains"]):
        for policy, rep in data.generate_worlds(domain, n_rep, cfg["seed"], cfg["min_per_label"]):
            name = (
                "multi"
                if rep < reps["multi"]
                else "val"
                if rep < reps["multi"] + reps["val"]
                else "control_pool"
            )
            sets[name].append((rep, di, policy))
    # the control pool is read in (replicate, domain, cell) order, so any prefix covers the whole grid evenly
    sets["control_pool"].sort(key=lambda t: (t[0], t[1], t[2].policy_id))
    all_train = []
    for _, _, p in (x for s in sets.values() for x in s):
        d = data.TRAIN_DOMAINS[dict(p.meta)["domain"]]
        pt, rt = (d.principal_type,), (d.resource_type,)
        all_train += [p, rename(p, swap_map(p, swap_axis(p, pt, rt)))]
    eval_policies = [
        w[a] for w in provenance.read_jsonl(benchmark / "worlds.jsonl") for a in ("orig", "swap")
    ]
    disjoint = data.disjointness(all_train, eval_policies)
    if not disjoint["passed"]:
        raise SystemExit(f"training data overlaps the eval benchmark: {json.dumps(disjoint)[:2000]}")

    def examples(items, renderings):
        out = []
        for _, _, p in items:
            d = data.TRAIN_DOMAINS[dict(p.meta)["domain"]]
            out += data.world_examples(p, d, renderings, cfg["per_label"], cfg["seed"], tuple(cfg["tasks"]))
        return out

    train = examples(sets["multi"], train_r)
    val = examples(sets["val"], train_r)
    pool = examples(sets["control_pool"], (cfg["control_rendering"],))
    make = {"hf": hf_counter, "chars": chars_counter}[counter]
    tokens, controls, tokenizers = {}, {}, {}
    for key in cfg["token_models"]:
        count, info = make(key)
        tokenizers[key] = info
        t_train = sum(count(e) for e in train)
        if t_train < MIN_TOKENS_PER_EXAMPLE * len(train):  # a prompt-v1 query is hundreds of tokens
            raise SystemExit(f"{key}: implausible token count ({t_train / len(train):.1f} per example)")
        controls[key], t_control = data.match_tokens(pool, t_train, count)
        tokens[key] = {
            "train_multi": t_train,
            "val_multi": sum(count(e) for e in val),
            "control": t_control,
            "control_over_train": t_control / t_train,
        }
    run = out_root / f"{provenance.utc_stamp()}_authinv_sft_data"
    run.mkdir(parents=True, exist_ok=False)
    provenance.write_json(
        run / "config.json",
        {
            "config": cfg,
            "config_sha256": provenance.sha256_json(cfg),
            "benchmark": str(benchmark),
            "token_counter": counter,
            "command": sys.argv,
        },
    )
    worlds_out = [
        {**to_dict(p), "set": name, "replicate": rep} for name, s in sets.items() for rep, _, p in s
    ]
    files = {"worlds.jsonl": worlds_out, "train_multi.jsonl": train, "val_multi.jsonl": val}
    files.update({f"control_{k}.jsonl": v for k, v in controls.items()})
    for name, rows in files.items():
        provenance.write_jsonl(run / name, rows)
    provenance.write_json(
        run / "manifest.json",
        {
            "data_version": data.DATA_VERSION,
            "files": {
                name: {"sha256": provenance.sha256_file(run / name), "rows": len(rows)}
                for name, rows in files.items()
            },
            "sets": {
                "train_multi": summary(train),
                "val_multi": summary(val),
                **{f"control_{k}": summary(v) for k, v in controls.items()},
            },
            "worlds": {name: len(s) for name, s in sets.items()},
            "control_worlds_used": {k: len({e["world_id"] for e in v}) for k, v in controls.items()},
            "tokens": tokens,
            "tokenizers": tokenizers,
            "token_counter": counter,
            "train_renderings": list(train_r),
            "held_out_renderings": sorted(held),
            "control_rendering": cfg["control_rendering"],
            "disjointness": disjoint,
            "eval_benchmark_sha256": manifest["sha256"],
            "no_sft_baseline": no_sft_baseline(cfg),
            "git": provenance.git_state(),
            "source_sha256": provenance.source_hashes(TRACKED),
        },
    )
    provenance.write_run_status(run, "complete", len(train) + len(val), len(train) + len(val))
    provenance.finalize(run)
    return run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--token-counter", choices=("hf", "chars"), default="hf")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = p.parse_args()
    print(
        build(yaml.safe_load(args.config.read_text()), args.benchmark, args.token_counter, args.output_root)
    )


if __name__ == "__main__":
    main()
