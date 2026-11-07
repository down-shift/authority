#!/usr/bin/env python3
"""Phase-4 LoRA SFT (P4.2): train one adapter per (model, arm, seed) on the frozen P4.1 data.

  uv run --extra inference --extra training python scripts/authinv_sft_train.py run \
      --config configs/authinv/sft_lora.yaml --model KEY --arm multi|control --seed N \
      [--data DIR] [--model-dir DIR] [--output-root DIR] [--max-steps N]
  python scripts/authinv_sft_train.py plan --config configs/authinv/sft_lora.yaml

`run` writes outputs/<stamp>_authinv_sft_<model>_<arm>_s<seed>/: config.json, metadata.json,
train_log.jsonl, metrics.json (val loss, steps, tokens, adapter sha256) and adapter/ (the
peft adapter), plus the artifact manifest. It refuses unless the data manifest hash, the
trainer version, the model, the arm and the seed are the config's. `--max-steps` is an
engineering smoke test (status "smoke"), never used for results. `plan` lists every run.
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

from authinv import models, provenance  # noqa: E402
from authinv.sft import train as sft  # noqa: E402

TRACKED = [
    ROOT / "scripts" / "authinv_sft_train.py",
    ROOT / "src" / "authinv" / "sft" / "train.py",
    ROOT / "src" / "authinv" / "provenance.py",
]


def plan(cfg: dict) -> list[dict]:
    return [
        {"model": m, "arm": a, "seed": s} for m in cfg["models"] for a in cfg["arms"] for s in cfg["seeds"]
    ]


def check(cfg: dict, data_dir: Path, model: str, arm: str, seed: int, smoke: bool) -> None:
    problems = []
    if cfg["trainer_version"] != sft.TRAINER_VERSION:
        problems.append(f"trainer_version {cfg['trainer_version']} != code {sft.TRAINER_VERSION}")
    got = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if got != cfg["data_manifest_sha256"]:
        problems.append(f"data manifest {got} != config {cfg['data_manifest_sha256']}")
    if model not in cfg["models"]:
        problems.append(f"model {model} not in config models")
    if arm not in cfg["arms"]:
        problems.append(f"arm {arm} not in config arms")
    if seed not in cfg["seeds"] and not smoke:
        problems.append(f"seed {seed} not in config seeds")
    if problems:
        raise SystemExit(f"run refused: {problems}")


def cmd_run(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    data_dir = args.data or ROOT / cfg["data_dir"]
    check(cfg, data_dir, args.model, args.arm, args.seed, bool(args.max_steps))
    spec = models.model_spec(args.model)
    set_name = sft.set_name(args.arm, args.model, cfg["arms"])
    train_rows = sft.load_set(data_dir, set_name)
    val_rows = sft.load_set(data_dir, cfg["val_set"])
    run = args.output_root / f"{provenance.utc_stamp()}_authinv_sft_{args.model}_{args.arm}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    provenance.write_json(
        run / "config.json",
        {
            "experiment": cfg["experiment"],
            "config": cfg,
            "config_sha256": provenance.sha256_json(cfg),
            "model": spec,
            "arm": args.arm,
            "seed": args.seed,
            "data": {
                "dir": str(data_dir),
                "manifest_sha256": cfg["data_manifest_sha256"],
                "train_set": set_name,
                "train_sha256": manifest["files"][set_name]["sha256"],
                "val_set": cfg["val_set"],
                "val_sha256": manifest["files"][cfg["val_set"]]["sha256"],
            },
            "smoke_max_steps": args.max_steps,
            "command": sys.argv,
        },
    )
    provenance.write_json(
        run / "metadata.json",
        {
            "git": provenance.git_state(),
            "source_sha256": provenance.source_hashes(TRACKED),
            "model_dir": str(args.model_dir) if args.model_dir else None,
        },
    )
    provenance.write_run_status(run, "running", 0, len(train_rows))
    try:
        metrics = sft.train(
            spec,
            str(args.model_dir) if args.model_dir else None,
            train_rows,
            val_rows,
            cfg,
            args.seed,
            run,
            args.max_steps,
        )
        provenance.write_json(run / "metrics.json", metrics)
        provenance.write_run_status(
            run, "smoke" if args.max_steps else "complete", len(train_rows), len(train_rows)
        )
        provenance.finalize(run)
        print(run)
    except BaseException as error:
        provenance.write_run_status(run, "failed", 0, len(train_rows), repr(error))
        provenance.finalize(run)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--data", type=Path)
    r.add_argument("--model-dir", type=Path, help="local weights (else the pinned HF revision)")
    r.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    r.add_argument("--max-steps", type=int, help="engineering smoke test: stop after N optimizer steps")
    r.set_defaults(func=cmd_run)
    q = sub.add_parser("plan")
    q.add_argument("--config", type=Path, required=True)
    q.set_defaults(func=lambda a: print(json.dumps(plan(yaml.safe_load(a.config.read_text())), indent=1)))
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
