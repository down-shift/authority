#!/usr/bin/env python3
"""Phase-5 probing tooling (P5.1): extract residual-stream activations, then probe and compare.

  uv run --extra inference python scripts/authinv_probe.py extract --config configs/authinv/probe.yaml \
      --dataset DIR --model KEY [--model-dir DIR] [--output-root DIR] [--limit N]
  python scripts/authinv_probe.py analyze --config configs/authinv/probe.yaml --cache RUN --out FILE

`extract` selects the configured worlds (hash-selected, every rendering and assignment kept),
builds the probe items and their labels (src/authinv/probe/labels.py), and writes the
activation cache with integrity hashes into outputs/<stamp>_authinv_probe_<model>/ (plus config,
metadata, status and the artifact manifest). `analyze` re-verifies the cache, then writes per
label and layer the cross-rendering transfer matrix with the metadata-only, shuffled-label and
majority baselines, and per layer the linear CKA between renderings. `--limit` is an engineering
smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from authinv import models, provenance  # noqa: E402
from authinv.eval.dataset import load_dataset  # noqa: E402
from authinv.probe import analysis, cache  # noqa: E402
from authinv.probe.labels import probe_items  # noqa: E402

TRACKED = [ROOT / "scripts" / "authinv_probe.py"] + [
    ROOT / "src" / "authinv" / p
    for p in ("probe/labels.py", "probe/cache.py", "probe/extract.py", "probe/analysis.py", "provenance.py")
]


def select_worlds(world_ids: list[str], n, seed: int) -> set[str]:
    ids = sorted(set(world_ids))
    if n == "all":
        return set(ids)
    return set(sorted(ids, key=lambda w: hashlib.sha256(f"{seed}|{w}".encode()).hexdigest())[: int(n)])


def cmd_extract(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    spec = models.model_spec(args.model)
    rows, validation = load_dataset(args.dataset)
    sha = validation["manifest"]["sha256"]
    if cfg.get("dataset_sha256") and sha != cfg["dataset_sha256"]:
        raise SystemExit(f"dataset {sha} is not the preregistered benchmark {cfg['dataset_sha256']}")
    worlds = {w["world_id"]: w for w in provenance.read_jsonl(args.dataset / "worlds.jsonl")}
    keep = select_worlds([r["world_id"] for r in rows], cfg["worlds"], cfg["world_seed"])
    items = probe_items(
        [r for r in rows if r["world_id"] in keep], worlds, tuple(cfg["tasks"]), cfg["owner_slots"]
    )
    if args.limit:  # engineering smoke test only
        items = items[: args.limit]
    run = args.output_root / f"{provenance.utc_stamp()}_authinv_probe_{args.model}"
    run.mkdir(parents=True, exist_ok=False)
    provenance.write_json(
        run / "config.json",
        {
            "config": cfg,
            "config_sha256": provenance.sha256_json(cfg),
            "model": spec,
            "dataset_sha256": sha,
            "worlds": len(keep),
            "items": len(items),
            "smoke_limit": args.limit,
            "command": sys.argv,
        },
    )
    provenance.write_run_status(run, "running", 0, len(items))
    try:
        from authinv.probe.extract import Extractor, resolve_layers

        ex = Extractor(spec, str(args.model_dir) if args.model_dir else None, cfg["dtype"])
        layers = resolve_layers(cfg["layers"], ex.n_hidden())
        provenance.write_json(
            run / "metadata.json",
            {
                "git": provenance.git_state(),
                "source_sha256": provenance.source_hashes(TRACKED),
                "layers": layers,
                "device": str(ex.device),
            },
        )
        writer = cache.CacheWriter(
            run / "cache",
            layers,
            {
                "model": spec["key"],
                "revision": spec.get("revision"),
                "dataset_sha256": sha,
                "template_kwargs": ex.template_kwargs,
            },
        )
        for start in range(0, len(items), cfg["batch_size"]):
            chunk = items[start : start + cfg["batch_size"]]
            writer.add(chunk, ex.hidden([it["prompt"] for it in chunk], layers))
            provenance.write_run_status(run, "running", start + len(chunk), len(items))
        writer.close()
        provenance.write_run_status(run, "smoke" if args.limit else "complete", len(items), len(items))
        provenance.finalize(run)
        print(run)
    except BaseException as error:
        provenance.write_run_status(run, "failed", 0, len(items), repr(error))
        provenance.finalize(run)
        raise


def cmd_analyze(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    acts, items, manifest = cache.load(args.cache / "cache")
    layers = manifest["layers"]
    labels = [lab for lab in cfg["labels"] if any(lab in it["labels"] for it in items)]
    out = {
        "cache": str(args.cache),
        "model": manifest["model"],
        "items": len(items),
        "layers": layers,
        "transfer": {lab: analysis.transfer(acts, items, lab, layers, cfg["probe"]) for lab in labels},
        "cka": {t: analysis.cka_matrix(acts, items, layers, t) for t in sorted({it["task"] for it in items})},
        "git": provenance.git_state(),
    }
    provenance.write_json(args.out, out)
    print(args.out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--dataset", type=Path, required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--model-dir", type=Path)
    e.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    e.add_argument("--limit", type=int, help="engineering smoke test: first N items only")
    e.set_defaults(func=cmd_extract)
    a = sub.add_parser("analyze")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=cmd_analyze)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
