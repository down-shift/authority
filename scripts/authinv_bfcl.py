#!/usr/bin/env python3
"""Phase-4 BFCL-subset regression check (P4.2b): general function calling, base model vs adapters.

  python scripts/authinv_bfcl.py fetch --config configs/authinv/bfcl.yaml [--output-root DIR]
  python scripts/authinv_bfcl.py run --config configs/authinv/bfcl.yaml --data DIR --model KEY
        [--adapter SFTRUN] [--model-dir DIR] [--gpu-memory-utilization F] [--max-num-seqs N] [--limit N]
  python scripts/authinv_bfcl.py compare --config configs/authinv/bfcl.yaml --base RUN --runs RUN [RUN ...]
        [--out FILE]

`fetch` downloads the pinned BFCL v3 files (simple, multiple, their possible answers) from the
Hugging Face dataset revision in the config and refuses unless every sha256 matches. `run` scores
one model (or one LoRA adapter on it, `--adapter`, an SFT run of that model) with bfcl-ast-v1
(`src/authinv/bfcl.py`): greedy, the official default prompt, AST check, no execution. `compare`
gives each run's accuracy change against the base run on the same items (paired bootstrap) and
flags a regression (a drop > threshold whose 95% CI excludes 0). `--limit` is a smoke test only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import bfcl, models, provenance  # noqa: E402
from authinv.eval.structured import JSON_STRICT_V3, strip_think  # noqa: E402

TRACKED = [ROOT / "scripts" / "authinv_bfcl.py"] + [
    ROOT / "src" / "authinv" / p for p in ("bfcl.py", "eval/generation.py", "sft/adapter.py", "provenance.py")
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_data(data: Path, cfg: dict) -> dict:
    bad = {
        f: _sha(data / f)
        for f in cfg["files"]
        if not (data / f).is_file() or _sha(data / f) != cfg["files"][f]
    }
    if bad:
        raise SystemExit(f"BFCL data in {data} differs from the pinned files: {sorted(bad)}")
    return {"dir": str(data), "hf_revision": cfg["hf_revision"], "files_sha256": dict(cfg["files"])}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_items(data: Path, cfg: dict) -> list[dict]:
    items = []
    for cat, n in cfg["categories"].items():
        qs = _jsonl(data / f"BFCL_v3_{cat}.json")
        ans = {a["id"]: a for a in _jsonl(data / "possible_answer" / f"BFCL_v3_{cat}.json")}
        if len(qs) != n or {q["id"] for q in qs} != set(ans):
            raise SystemExit(f"{cat}: {len(qs)} items or answer ids differ from the pinned {n}")
        items += [{"id": q["id"], "category": cat, "entry": q, "answer": ans[q["id"]]} for q in qs]
    return items


def cmd_fetch(args) -> None:
    from huggingface_hub import hf_hub_download

    cfg = yaml.safe_load(args.config.read_text())
    out = args.output_root / f"{provenance.utc_stamp()}_authinv_bfcl_data"
    out.mkdir(parents=True, exist_ok=False)
    for f in cfg["files"]:
        src = hf_hub_download(cfg["hf_dataset"], f, repo_type="dataset", revision=cfg["hf_revision"])
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_bytes(Path(src).read_bytes())
    info = check_data(out, cfg)
    provenance.write_json(
        out / "data_manifest.json", {**info, "hf_dataset": cfg["hf_dataset"], "license": cfg["license"]}
    )
    provenance.finalize(out)
    print(out)


def score_rows(items: list[dict], outputs: list[dict]) -> list[dict]:
    rows = []
    for it, g in zip(items, outputs, strict=True):
        text = strip_think(g["text"], JSON_STRICT_V3)
        rows.append(
            {
                "id": it["id"],
                "category": it["category"],
                **{k: g[k] for k in ("text", "finish_reason", "prompt_tokens", "output_tokens") if k in g},
                **bfcl.score(it["entry"], it["answer"], text, it["category"]),
            }
        )
    return rows


def cmd_run(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["bfcl_version"] != bfcl.BFCL_VERSION or cfg["gorilla_commit"] != bfcl.GORILLA_COMMIT:
        raise SystemExit(
            f"config {cfg['bfcl_version']} / {cfg['gorilla_commit']} != code {bfcl.BFCL_VERSION}"
        )
    data_info = check_data(args.data, cfg)
    items = load_items(args.data, cfg)
    if args.limit:
        items = items[: args.limit]
    spec = models.model_spec(args.model)
    adapter = None
    if args.adapter:
        from authinv.sft.adapter import load_adapter

        adapter = load_adapter(args.adapter, args.model, spec, smoke=bool(args.limit))
    tag = f"bfcl_{args.model}_{adapter['arm']}_s{adapter['seed']}" if adapter else f"bfcl_{args.model}"
    run = args.output_root / f"{provenance.utc_stamp()}_authinv_{tag}"
    run.mkdir(parents=True, exist_ok=False)
    provenance.write_json(
        run / "config.json",
        {
            "experiment": cfg["experiment"],
            "config": cfg,
            "config_sha256": provenance.sha256_json(cfg),
            "model": spec,
            "data": data_info,
            "bfcl_version": bfcl.BFCL_VERSION,
            **({"adapter": adapter} if adapter else {}),
            "smoke_limit": args.limit,
            "command": sys.argv,
        },
    )
    total = len(items)
    try:
        from authinv.eval.generation import VLLMChat

        gen = {
            **cfg["generation"],
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "max_num_seqs": args.max_num_seqs,
        }
        chat = VLLMChat(spec, str(args.model_dir) if args.model_dir else None, gen, 1, lora=adapter)
        provenance.write_json(
            run / "metadata.json",
            {
                "model": spec,
                "runtime": chat.provenance(),
                "generation": gen,
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                **({"adapter": adapter} if adapter else {}),
                "command": sys.argv,
            },
        )
        provenance.write_run_status(run, "running", 0, total)
        rows = []
        for start in range(0, total, gen["batch_size"]):
            chunk = items[start : start + gen["batch_size"]]
            rows += score_rows(chunk, chat.generate_chats([bfcl.messages(it["entry"]) for it in chunk]))
            provenance.write_run_status(run, "running", len(rows), total)
        provenance.write_jsonl(run / "predictions.jsonl", rows)
        a = cfg["analysis"]
        metrics = {
            "bfcl_version": bfcl.BFCL_VERSION,
            "accuracy": bfcl.accuracy(rows, a["bootstrap_replicates"], a["bootstrap_seed"]),
            "error_types": {
                c: dict(
                    sorted(
                        {
                            e: sum(1 for r in rows if r["category"] == c and r["error_type"] == e)
                            for e in {r["error_type"] for r in rows if r["category"] == c and r["error_type"]}
                        }.items()
                    )
                )
                for c in sorted({r["category"] for r in rows})
            },
        }
        provenance.write_json(run / "metrics.json", metrics)
        provenance.write_run_status(run, "smoke" if args.limit else "complete", total, total)
        provenance.finalize(run)
        print(run)
    except BaseException as error:
        provenance.write_run_status(run, "failed", 0, total, repr(error))
        provenance.finalize(run)
        raise


def _complete_run(run: Path) -> tuple[dict, list[dict]]:
    if json.loads((run / "run_status.json").read_text())["status"] != "complete":
        raise SystemExit(f"{run}: not complete")
    return json.loads((run / "config.json").read_text()), provenance.read_jsonl(run / "predictions.jsonl")


def cmd_compare(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    a = cfg["analysis"]
    base_conf, base_rows = _complete_run(args.base)
    if base_conf.get("adapter"):
        raise SystemExit(f"{args.base}: the base run must not use an adapter")
    out = {"base": str(args.base), "model": base_conf["model"]["key"], "runs": {}, "arms": {}}

    def diff(rows: list[dict]) -> dict:
        return {
            cat: bfcl.paired_difference(
                [r for r in rows if cat == "all" or r["category"] == cat],
                [r for r in base_rows if cat == "all" or r["category"] == cat],
                a["bootstrap_replicates"],
                a["bootstrap_seed"],
                a["regression_threshold"],
            )
            for cat in (*cfg["categories"], "all")
        }

    by_arm: dict[str, list[list[dict]]] = {}
    for run in args.runs:
        conf, rows = _complete_run(run)
        if conf["model"] != base_conf["model"] or conf["config_sha256"] != base_conf["config_sha256"]:
            raise SystemExit(f"{run}: model or config differs from the base run")
        if not conf.get("adapter"):
            raise SystemExit(f"{run}: not an adapter run")
        arm = conf["adapter"]["arm"]
        by_arm.setdefault(arm, []).append(rows)
        out["runs"][f"{arm}_s{conf['adapter']['seed']}"] = {
            "run": str(run),
            "accuracy": json.loads((run / "metrics.json").read_text())["accuracy"],
            "difference": diff(rows),
        }
    # per arm: an item's value is the mean of its seeds' validity (phase4.md amendment 1)
    for arm, runs in sorted(by_arm.items()):
        vals: dict[str, list[float]] = {}
        cats = {}
        for rows in runs:
            for r in rows:
                vals.setdefault(r["id"], []).append(float(r["valid"]))
                cats[r["id"]] = r["category"]
        mean_rows = [{"id": i, "category": cats[i], "valid": sum(v) / len(v)} for i, v in vals.items()]
        out["arms"][arm] = {"seeds": len(runs), "difference": diff(mean_rows)}
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out:
        provenance.write_json(args.out, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--config", type=Path, required=True)
    f.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    f.set_defaults(func=cmd_fetch)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--adapter", type=Path, help="an SFT run of this model: evaluate its LoRA adapter")
    r.add_argument("--model-dir", type=Path)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    r.add_argument("--max-num-seqs", type=int)
    r.add_argument("--enforce-eager", action="store_true")
    r.add_argument("--limit", type=int, help="engineering smoke test: first N items only; never compared")
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--out", type=Path)
    c.set_defaults(func=cmd_compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
