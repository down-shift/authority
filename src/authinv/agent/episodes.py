"""Build T3 episodes from the frozen Phase-1 benchmark (same worlds, assignments, and requests as T2).

The benchmark rows store prompts, not request objects, so requests are
re-derived with the builder's own seeded sampler (`sample_requests`, using the
benchmark config's seed and per_label) and the stored swap maps. Every derived
(world, assignment, instance) is then checked against the frozen rows: the
label and the pair_key must match exactly, or the build refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

from authinv import provenance
from authinv.agent.sandbox import Episode
from authinv.benchmark import _context
from authinv.equivalence.requests import sample_requests
from authinv.policy import EntityRef, Request, evaluate, from_dict
from authinv.render.renderers import RENDERINGS, render


def _ref(key: str) -> EntityRef:
    t, i = key.split("::", 1)
    return EntityRef(t, i)


def build_episodes(
    benchmark: Path, renderings: tuple[str, ...] = RENDERINGS, assignments: tuple[str, ...] = ("orig", "swap")
) -> tuple[list[Episode], dict]:
    cfg = json.loads((benchmark / "config.json").read_text())["config"]
    rows = provenance.read_jsonl(benchmark / "dataset.jsonl")
    want = {
        (r["world_id"], r["assignment"], r["instance"]): (r["label"], r["pair_key"])
        for r in rows
        if r["task"] == "application"
    }
    episodes, checked = [], 0
    for w in provenance.read_jsonl(benchmark / "worlds.jsonl"):
        pt, rt = tuple(w["principal_types"]), tuple(w["resource_types"])
        orig = from_dict(w["orig"])
        variants = {
            "orig": (orig, {}),
            "swap": (from_dict(w["swap"]), {_ref(k): _ref(v) for k, v in w["swap_map"].items()}),
        }
        picked = sample_requests(orig, pt, rt, cfg["per_label"], cfg["seed"])
        for assignment in assignments:
            pol, m = variants[assignment]
            texts = {g: render(pol, g) for g in renderings}
            for qi, item in enumerate(picked):
                r = item["request"]
                req = Request(
                    m.get(r.principal, r.principal), r.action, m.get(r.resource, r.resource), r.context
                )
                label = evaluate(pol, req)["decision"]
                key = (w["world_id"], assignment, f"q{qi}")
                pair_key = f"{req.action}|{req.resource.key()}|{_context(req.context)}"
                if want.get(key) != (label, pair_key):
                    raise SystemExit(f"{key}: re-derived request does not match the frozen benchmark")
                checked += 1
                for g in renderings:
                    episodes.append(
                        Episode(
                            episode_id=f"{w['world_id']}/{assignment}/q{qi}/{g}",
                            world_id=w["world_id"],
                            assignment=assignment,
                            instance=f"q{qi}",
                            rendering=g,
                            policy=pol,
                            request=req,
                            label=label,
                            principal_types=pt,
                            resource_types=rt,
                            rendering_text=texts[g],
                            meta={
                                "source": w["source"],
                                "source_kind": w["source_kind"],
                                "pair_key": pair_key,
                            },
                        )
                    )
    return episodes, {
        "instances_checked": checked,
        "episodes": len(episodes),
        "dataset_sha256": json.loads((benchmark / "dataset_manifest.json").read_text())["sha256"],
    }
