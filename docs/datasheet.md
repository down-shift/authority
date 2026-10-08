# Datasheet: authinv Phase-1 benchmark

Format after Gebru et al., *Datasheets for Datasets*. Version: prompt-v1 and
renderers nl-v1 / owner-v1 / table-v1 / json-v1 / cedar-v1 / rego-v1.
Frozen dataset sha256 `255c9ba2ed4b6483f3023a0bfb4a610ad7075e0292a0c1218b9378e3343b9380`.
Build: `docs/experiments/p1.10-benchmark.md`.

## Motivation

The benchmark measures whether LLMs make the same authorization decision when
one policy is written in different but **engine-certified equivalent** forms.
The forms are a natural-language statement, decision-owner statements, a
permission table, a JSON policy, Cedar, and Rego. The measures are worst-case
accuracy over renderings and the direction of errors: deny→allow vs
allow→deny (`docs/RESEARCH_PLAN.md` §0).

Created by the authority project (down-shift/authority) for a NeurIPS 2027
submission.

## Composition

- **1,984 worlds.** A world is one canonical policy with its entities and a
  request universe.
  - 544 real: CedarBench 494, Quacky 50.
  - 1,440 synthetic: MCP tool allowlists 720, GitHub-style repo permissions 720.
  - Tiers: conditioned 1,571, multi_rule 329, single 84.
- **172,296 rows.** Each row is one prompt.
  - Application: 95,232 rows, labelled allow or deny, exactly balanced.
  - Interpretation: 77,064 rows; the label is the set of permitted principals.
  - Each world contributes 4 requests (2 allow, 2 deny, boundary-first).
  - Each request is asked under 2 name assignments and 6 renderings.
- **Labels** come from the canonical reference evaluator, and Cedar 4.12.1 and
  OPA 1.21.1 agree with it on every request of every universe. No labels are
  human-annotated or LLM-generated.
- **No personal data.** Identifiers are neutral codes (synthetic), synthesized
  ids (CedarBench), or mapped AWS-style ids (Quacky).
- **Fields:** `source`, `source_kind`, `tier`, `assignment`, `swap_axis`,
  `option_order`, `boundary`, and `rendering_chars` are on every row, so every
  analysis can be split by them.

## Collection process

| source | origin | license | version pin | what we take |
|---|---|---|---|---|
| CedarBench | github.com/neselab/cedar-synthesis-engine (AutoCedar, arXiv 2607.03656) | Apache-2.0 | commit `0201565`, tree sha256 pinned in `configs/authinv/sources/cedarbench.yaml` | `references/*.cedar` + schemas; entities synthesized (importer v2) |
| Quacky | github.com/vlab-cs-ucsb/quacky (ASE '22) | BSD-2-Clause | commit `31c13ee`, `configs/authinv/sources/quacky.yaml` | 41 original + 546 mutated AWS IAM policies (importer v2) |
| synthetic | `src/authinv/sources/synthetic.py` | project's own | `configs/authinv/synthetic.yaml` seed 20261009 | 24-cell balanced grid × 30 replicates × 2 domains |
| ACRE | — | **no license found** | — | **dropped** (Jerzy, 2026-10-08) |

## Preprocessing and equivalence

- **Exact translation only.** Policies are translated into the canonical
  model with no approximation. Constructs outside it are excluded and counted
  per construct: Cedar `has`, sets, extension types, nested `||`; IAM
  `StringLike`, `IpAddress`, policy variables. See the P1.7 and P1.8 reports.
- **Faithfulness.**
  - Each CedarBench world agrees with its original Cedar text on the enlarged
    universe, checked with cedarpy.
  - Each Quacky world agrees with its original IAM JSON under an independent
    IAM evaluator.
- **Certification** (`src/authinv/equivalence/check.py`). A world ships only
  if every rendering:
  - decodes back to the canonical rules;
  - matches the reference decision on the whole universe;
  - for Cedar and Rego, the real engine agrees with zero errors;
  - for Cedar, strict typechecking passes.
- **Name swap.** The `swap` assignment renames principals (or resources, in
  single-caller worlds). All 1,984 swapped policies were re-certified
  separately.
- **Universe enlargement** is exact; it only adds requests the policy already
  decides:
  - CedarBench: schema-action probes and replicated entities;
  - Quacky: wildcard witnesses and closed-world Not* complements.

  Affected worlds are flagged in their meta.

## Uses

- **Intended:** measuring the representation sensitivity of authorization
  decisions in LLMs and agents (tiers T1–T3), and evaluating mitigations
  (canonicalization, fine-tuning).
- **Not intended:** training a production authorization system, or certifying
  any real system. The worlds are small closed universes.

## Distribution

Not yet released. Release is step D1.1, and publishing is Jerzy's decision.
A release must carry:
- CedarBench's Apache-2.0 license and NOTICE for derived policies;
- Quacky's BSD-2-Clause license;
- the project's license for synthetic content.

## Maintenance

- Any renderer or prompt change gets a version bump, a `docs/CHANGELOG.md`
  entry, and a rebuild.
- Raw artifacts are git-ignored. The long-term copy is on V100 storage
  `…/authinv/phase1/`; the archive was pending at build time (cluster
  unreachable).

## Known limitations

- Real policies are 27% of worlds. Report real and synthetic separately.
- CedarBench entities are synthesized, and its non-degeneracy relies on
  entity replication and action probes. The 5 stress scenarios are capped at
  25 worlds each.
- Quacky worlds depend on wildcard witnesses (most shipped files) and on
  closed-world complements. Identities are synthetic callers.
- Rendering lengths differ widely: median 204 characters for natural language
  vs 1,083 for Rego. Length is a covariate, not a control.
- The renderings are one fixed template per format. Results are a lower bound
  on sensitivity to arbitrary equivalent rewrites (`docs/threat_model.md`).
