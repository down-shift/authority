import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_tiny_benchmark(root: Path, n_worlds: int = 3) -> tuple[Path, list[dict], dict[str, dict]]:
    """A frozen-benchmark directory (dataset.jsonl, manifest, worlds.jsonl) from a few synthetic worlds.

    Same layout as scripts/authinv_build_benchmark.py; rendering needs cedarpy (Cedar text).
    """
    from authinv import provenance
    from authinv.benchmark import World, non_degenerate, world_rows
    from authinv.policy import to_dict
    from authinv.sources.synthetic import generate

    ws, _ = generate("repo", replicates=1, seed=3)
    picked = [
        w
        for w in (World(p, ("User",), ("Repo",), "synthetic", "synthetic", dict(p.meta)) for p in ws)
        if non_degenerate(w, 4)
    ]
    rows, out = [], []
    for w in picked[:n_worlds]:
        rs, extra = world_rows(w, per_label=2, seed=1)
        rows += rs
        out.append(
            {
                "world_id": w.policy.policy_id,
                "source": w.source,
                "source_kind": w.source_kind,
                "principal_types": list(w.principal_types),
                "resource_types": list(w.resource_types),
                "orig": to_dict(w.policy),
                "swap": to_dict(extra["swap_policy"]),
                "swap_map": extra["swap_map"],
            }
        )
    d = root / "bench"
    d.mkdir()
    provenance.write_jsonl(d / "dataset.jsonl", rows)
    provenance.write_json(
        d / "dataset_manifest.json",
        {"sha256": provenance.sha256_file(d / "dataset.jsonl"), "rows": len(rows)},
    )
    provenance.write_jsonl(d / "worlds.jsonl", out)
    return d, rows, {w["world_id"]: w for w in out}
