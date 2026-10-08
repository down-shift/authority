# Related work (verified)

**Last re-searched: 2026-10-08.** Re-run the arXiv check monthly (cs.CR / cs.CL / cs.SE / cs.AI),
as `docs/RESEARCH_PLAN.md` §6.5 requires. Most close neighbours appeared May–Oct 2026, so the
next check is due 2026-11-08. Watch especially for follow-ups from the Eiers group (Stevens:
PolicySummarizer, AutoCedar, RAISE), from Gupta & Sreenivasamurthy (Prose2Policy, Structured
Decomposition), and for a re-submission of the withdrawn PolicyGuard DLP paper.

This file expands the table in `docs/RESEARCH_PLAN.md` §2 (step R0.1). It positions the paper
against the thesis in §0: one authorization policy, rendered five engine-certified-equivalent ways
(NL statement, decision-owner statement, permission table, JSON policy, Cedar), yields different
model decisions and agent actions, and we measure worst-case accuracy over renderings,
worst-case unauthorized-action rate, and deny→allow flips.

## How this was checked

- For every paper with an arXiv ID, the abstract page `https://arxiv.org/abs/<id>` was downloaded
  on 2026-10-08 and parsed for title, author list, submission history (latest version and date),
  the Comments field, and withdrawal status. Where the latest version had been retitled or
  revised, the earlier versions (`<id>v1`, `v2`) were also fetched to see where the description
  in `docs/novelty_check.md` came from.
- Quacky and ACRE have no arXiv ID; they were checked against the publisher / conference page
  (ASE '22 paper PDF on par.nsf.gov; ACSAC 2014 program page).
- Only the **abstract** was read, not full text. Numbers that `docs/novelty_check.md` reports but
  the abstract does not contain are marked *not in abstract*. That is not a contradiction; it
  means the claim still needs a full-text read before it goes into the paper.
- **Mismatches are flagged, not silently fixed.** `docs/novelty_check.md` and
  `docs/RESEARCH_PLAN.md` are left unchanged; the corrected values are in this file and in
  `paper/references.bib`.

Verification column legend: **verified** = ID, title, authors and status match what the repo
docs say (or the docs gave no conflicting value); **mismatch** = the arXiv page contradicts the repo
docs (what differs is stated); **not verified** = no page could be checked.

## Summary

- **77 papers verified** on their arXiv abstract page or publisher page (75 arXiv + Quacky +
  ACRE). None of the arXiv IDs was wrong: every ID resolves to the paper the docs name.
- **7 mismatches flagged** (details in the per-paper sections):
  1. **PolicySummarizer (2510.20692)**: the current (v3) abstract reports **546 AWS + 100 Azure +
     100 GCP** policies, not "587 AWS IAM policies from Quacky". The 59.1–67.9% decision accuracy
     and 90.7–96.0% explanation consistency are in no version's abstract, and the user-study
     figure is "39% → 93%" on the hardest sub-task, not "39–44% correct". The paper was
     retitled in v3 (v1/v2: "Exploring Large Language Models for Access Control Policy Synthesis
     and Summarization"). This affects step P1.8, whose done-when says "Quacky's 587 policies".
  2. **Ghost in the Context (2605.12535)**: the title on arXiv is "Policy-Carriage Integrity in LLM
     Agents" (v3), not "... in LLM Agent Context Assembly" (v1/v2: "Measuring Policy-Carriage
     Failures in Decision-Time Assembly"). The "Policy IR" of subject/action/object/effect/
     condition/evidence records and the 512-token budget are in no version's abstract. Author
     (not retrieved in novelty_check): Igor Santos-Grueiro.
  3. **APPA (2607.24625)**: retitled in v2 to "APPA: Recoverable Information-Flow Control for
     Real-World LLM Agents" (v1: "Agentic Permissions Policy Algebra for Taint Confinement in LLM
     Agents"). It is an information-flow-control framework; the novelty_check description
     "puts tool calls into canonical form" is not supported by either abstract.
  4. **Prose2Policy (2603.15799)**: arXiv lists **two** authors (Vatsal Gupta, Darshan
     Sreenivasamurthy). novelty_check lists "Gupta, Sreenivasamurthy, Apple"; "Apple" is not an
     author on the arXiv page (probably an affiliation).
  5. **Phase transitions (2608.12426)**: novelty_check quotes the abstract as "weakly correlated
     through shared output features (mean φ=+0.067), not pairwise interference". The abstract
     says failures are "nearly independent" and the residual coupling "tracks shared output
     features rather than pairwise interference"; φ=+0.067 is not in it. Substance agrees, quote
     does not.
  6. **GrantBox (2603.28166)**: novelty_check quotes "only foundational security awareness"; the
     abstract says "basic security awareness" and reports 84.80% average attack success.
  7. **TsuGO (2608.13221)**: novelty_check groups it with "chess and Go stability papers"; its
     abstract is about search efficiency in reasoning on Go life-and-death problems and does not
     describe an invariance or stability metric.
- **Withdrawn:** only **PolicyGuard DLP (2608.02687)**, confirmed withdrawn at v3 (6 Aug 2026). The
  stated reason is "submitted prior to completion of a required institutional review process";
  the authors intend to resubmit.
- **Not verified (8 items):** sources that novelty_check names without an arXiv ID and that are
  not in RESEARCH_PLAN §2: HiddenLayer "Policy Puppetry" (blog), Trend Micro "Sockpuppeting"
  (blog), Palla et al. Policy-as-Prompt (FAccT 2025), the EMNLP 2025 Industry company-policy
  enforcement paper, "Probing prompt-leakage intent" (EMNLP 2025), the clinical-NLI
  knowledge–reasoning dissociation paper, the tabular-prediction robustness paper (PMC), and
  "When Thinking Fails" (NeurIPS 2025). No BibTeX entries were added for them. Verify before citing.

## Expanded §2 table

| Role | Paper | arXiv ID | Title on arXiv | Authors (arXiv) | Latest version | Withdrawn | Verification | BibTeX key |
|---|---|---|---|---|---|---|---|---|
| Generic format sensitivity | Sclar et al. (ICLR 2024) | 2310.11324 | Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting | Sclar, Choi, Tsvetkov, Suhr | v2, 1 Jul 2024 (ICLR 2024 camera-ready) | no | verified | `sclar2024quantifying` |
| | He et al. | 2411.10541 | Does Prompt Formatting Have Any Impact on LLM Performance? | He, Rungta, Koleczek, Sekhon, Wang, Hasan | v1, 15 Nov 2024 ("Submitted to NAACL 2025") | no | verified | `he2024prompt` |
| | Not as Sweet by Another Name | 2607.27648 | Not as Sweet by Another Name: An Empirical Study of Format Robustness in LLM Document Workflows | Zhang, Cheng, Li, Zheng, Yang, Liu | v1, 30 Jul 2026 (ASE 2026; DOI 10.1145/3832783.3837456) | no | verified | `zhang2026notassweet` |
| | Multi-Format Training | 2606.11643 | Improving Cross-Format Robustness in Language Models with Multi-Format Training | Liu, Zheng, Cao, Jin, Cui, Zhou | v1, 10 Jun 2026 | no | verified | `liu2026crossformat` |
| Closest policy-format precedent | PolicyGuard DLP | 2608.02687 | PolicyGuard: Prompt-Configurable Semantic DLP for LLM Coding Agents | Park, Kim, Shim | v3, 6 Aug 2026 | **yes** | verified (withdrawn confirmed) | `park2026policyguard` |
| LLMs misapply IAM policies | PolicySummarizer | 2510.20692 | Neurosymbolic Characterization for Reliable Access Control Policy Analysis | Vatsa, Hall, Eiers | v3, 3 Jul 2026 (ISSRE 2026; retitled) | no | **mismatch** (corpus size, numbers; see §PolicySummarizer) | `vatsa2025neurosymbolic` |
| UIR metric | Prompts Don't Protect | 2605.18414 | Prompts Don't Protect: Architectural Enforcement via MCP Proxy for LLM Tool Access Control | Uppala | v3, 27 Aug 2026 (EMNLP 2026 Industry, camera-ready) | no | verified | `uppala2026prompts` |
| Representation sensitivity of attacks | Threat-Preserving Representation Sensitivity | 2610.03585 | Threat-Preserving Representation Sensitivity in Agent-Security Benchmarks | Karamchandani, Nagasubramaniam, Xie, Zhu, Wu | v1, 2 Oct 2026 | no | verified | `karamchandani2026threat` |
| Canonicalize-then-decide | Policy-as-Logic | 2608.11905 | Policy-as-logic for robust reasoning over rules | Nair, Lipka, Daly | v1, 12 Aug 2026 (RobustifAI Workshop, IJCAI-ECAI '26) | no | verified | `nair2026policyaslogic` |
| | Ghost in the Context | 2605.12535 | Ghost in the Context: Policy-Carriage Integrity in LLM Agents | Santos-Grueiro | v3, 1 Jul 2026 | no | **mismatch** (title; "Policy IR" not in abstract) | `santosgrueiro2026ghost` |
| | MetaPermit | 2609.31039 | MetaPermit: Scalable and Auditable Access Control for AI Agents via LLM-Inferred Meta-Attributes | Ma, Hariri, Shen, Zou, Zheng, Wang, Yi, Jia, Liu, Chen, Wang, Roy | v1, 25 Sep 2026 | no | verified | `ma2026metapermit` |
| | Structured Decomposition | 2609.24036 | Structured Decomposition for Reliable LLM-Generated Access Control Policies | Gupta, Sreenivasamurthy | v1, 21 Sep 2026 | no | verified | `gupta2026structured` |
| Policy sources | CedarBench / AutoCedar | 2607.03656 | AutoCedar: An Agentic Framework for Verifier-Guided Access Control Policy Synthesis | Vatsa, Shome, Zhou, Eiers | v1, 4 Jul 2026 | no | verified | `vatsa2026autocedar` |
| | Quacky | — (ASE '22) | Quacky: Quantitative Access Control Permissiveness Analyzer | Eiers, Sankaran, Li, O'Mahony, Prince, Bultan | ASE '22, DOI 10.1145/3551349.3559530 | n/a | verified (tool paper); "587 policies" not confirmed | `eiers2022quacky` |
| | ACRE | — (ACSAC 2014) | Relation Extraction for Inferring Access Control Rules from Natural Language Artifacts | Slankas, Xiao, Williams, Xie | ACSAC 2014 | n/a | verified (paper); "ACRE" name from Prose2Policy / Structured Decomposition | `slankas2014relation` |
| | Prose2Policy | 2603.15799 | Prose2Policy (P2P): A Practical LLM Pipeline for Translating Natural-Language Access Policies into Executable Rego | Gupta, Sreenivasamurthy | v1, 16 Mar 2026 | no | **mismatch** (author list) | `gupta2026prose2policy` |
| Probing precedents | How Language Models Choose Sides | 2608.28648 | How Language Models Choose Sides: Internal Representations of Instruction Hierarchy | Balp-Straffon, Hsu, Gadhvi, Dev, McDougall, Mujumdar | v1, 16 Aug 2026 (ICML 2026 MI workshop) | no | verified | `balpstraffon2026choosing` |
| | System Prompt Illusion | 2609.38205 | The System Prompt Illusion: How Instruction Preambles Modify Computation in Language Models | Usama, Chang | v1, 23 Sep 2026 | no | verified | `usama2026system` |
| Multi-constraint counterpoint | Phase transitions | 2608.12426 | Large Language Models Can Follow Instructions, But Not Many at Once: Phase Transitions in Compositional Constraint Satisfaction | Vasileva | v1, 12 Aug 2026 | no | **mismatch** (quoted wording) | `vasileva2026phase` |
| Agent benchmarks | AgentDojo | 2406.13352 | AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents | Debenedetti, Zhang, Balunović, Beurer-Kellner, Fischer, Tramèr | v3, 24 Nov 2024 | no | verified | `debenedetti2024agentdojo` |
| | Agent Security Bench | 2410.02644 | Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents | Zhang, Huang, Mei, Yao, Wang, Zhan, Wang, Zhang | v4, 30 May 2025 (ICLR 2025) | no | verified | `zhang2024asb` |
| | GrantBox | 2603.28166 | Evaluating Privilege Usage of Agents with Real-World Tools | Zhang, Fu, Lian, Go, Wang, Zhou, Jiang, Pu | v2, 20 Apr 2026 (FSE 2026 IVR) | no | **mismatch** (quoted wording) | `zhang2026grantbox` |
| | AuthBench | 2605.14859 | Do Coding Agents Understand Least-Privilege Authorization? | Yan, Weng, Chen, Peng, Qin, Guan, Liu, Yu, Yuan, Meng, Che, Hu | v2, 15 May 2026 | no | verified | `yan2026authbench` |

## Overlap and differentiator: §2 papers

**Sclar et al., ICLR 2024 (2310.11324).** Meaning-preserving prompt-format changes (separators,
casing, spacing) move few-shot accuracy by up to 76 points on LLaMA-2-13B. The spread persists
with scale and instruction tuning, and they propose FormatSpread to report a range of
performance instead of a single number. *Overlap:* establishes that format is a confound and
that a range over formats should be reported. That is the basis for our worst-case metric.
*Differentiator:* their formats are cosmetic templates around a fixed task. Ours are different
*languages* for the same authorization policy (prose, table, JSON, Cedar), with equivalence proved
by an engine, and the outcome is security-relevant (deny→allow), not task accuracy.

**He et al. (2411.10541).** The same contexts as plain text, Markdown, JSON and YAML change GPT-3.5
performance by up to 40% on code translation; GPT-4 is more robust. *Overlap:* the closest generic
precedent for "JSON vs prose vs table" renderings. *Differentiator:* OpenAI models only, generic
tasks, no equivalence check and no notion of a policy being obeyed. We use open weights, an engine
as ground truth, and an agentic outcome. Note that the arXiv record is v1 "Submitted to NAACL
2025"; a published venue was not confirmed.

**Not as Sweet by Another Name (2607.27648, ASE 2026).** A metamorphic-testing framework for
LLM document workflows. Switching the document format of the same content (4 workflows × 4 tasks ×
4 formats, 48,000 runs) drops accuracy by up to 53.63% and causes decision drift in over 41% of
instances; user-side mitigations recover up to 44.21% of the drift. *Overlap:* methodologically
the closest. It formalizes decision-outcome invariance across formats and per-instance drift,
which is our per-world rendering-disagreement metric. *Differentiator:* the inputs are documents,
not policies that govern the model's own actions, and equivalence is by construction rather than
engine-certified. There is no authorization or agent action. novelty_check's quote "proximity in
average accuracy does not guarantee behavioral consistency" is *not in abstract*.

**Multi-Format Training (2606.11643).** Defines cross-format robustness for semantically equivalent
answer formats and compares full multi-format SFT with FormatMix (expanding a subset). Multi-format
supervision improves both accuracy and consistency on GLM4 and Llama-3.1; expanding ~30% of the
data recovers most of the gain. *Overlap:* the direct recipe for Phase 4 (P4.x). *Differentiator:*
it varies *answer/question* format on generic tasks. We vary *policy* rendering, hold out a whole
policy language (Cedar) to test unseen-format generalization, and judge by the worst-case gap. The
GLM4-9B pass^4 numbers (12.08% → 19.61%) are *not in abstract*.

**PolicyGuard DLP (2608.02687), withdrawn.** An LLM pre-filter for coding-agent prompts,
driven by a plaintext DLP policy file. It reports 96.5% effective block rate at 3.0% FPR on 927
frozen prompts. An information-matched comparison finds the natural-language policy significantly
better than the same content in JSON (McNemar χ²=31.58, p<0.001). *Overlap:* the only paper found
that holds policy content fixed and varies its format, and finds a decision difference. *Differentiator:*
two formats, DLP classification rather than authorization, no engine ground truth, no worst-case or
flip-direction analysis, no agent actions. **Withdrawn at v3 (6 Aug 2026)** pending institutional
review, so cite it only as motivating evidence. The "6.1 pp" gap, "h=0.260" and the 1.3% vs
3.0% FPR comparison are *not in abstract*. The abstract's only Cohen's h (0.915) is against zero-shot
classification, not against JSON.

**PolicySummarizer / Neurosymbolic Characterization (2510.20692, ISSRE 2026). Mismatch.** The v3
abstract says LLMs "fluently explain policy behavior but cannot reason about policy semantics with
reliability-grade precision", and calls this the Verifiable Synthesis Paradox. It introduces
PolicySummarizer (automata + LLM simplification + model counting), evaluated on 546 AWS, 100 Azure
and 100 GCP policies (similarity 0.93, 2.7× over an SMT baseline). A user study raises
policy-change-review accuracy from 39% to 93% on the hardest sub-task. *Overlap:* the closest
"LLMs misapply real access-control policies, with formal ground truth" result, and its
explain-vs-reason gap anticipates part of our interpretation/application dissociation (P2.2, P2.9).
*Differentiator:* one format (native cloud JSON), no invariance across renderings, no agents.
*Flags:* novelty_check's "587 AWS IAM policies from the Quacky dataset", "59.1–67.9% accurate",
"90.7–96.0% consistent" and "39–44% correct without the tool" are in **no** abstract version
(v1/v2 are an earlier synthesis-and-summarization paper with a different title). The policy count
on the current abstract is 546 AWS. **P1.8 assumes "Quacky's 587 policies"**; the source count must
be confirmed from the dataset itself when P1.8 runs.

**Prompts Don't Protect (2605.18414, EMNLP 2026 Industry).** With unauthorized tools visible,
models select them in 48–68% of adversarial scenarios; role escalation reaches 96%. Explicit
per-tool allowlists cut UIR to 4.0–37.0% depending on the model, never to zero. An ABAC MCP proxy
that filters the registry gives 0% UIR by design. *Overlap:* defines UIR, which we adopt for T3
(P2.7, P2.8), and the external-enforcement upper bound we use as a baseline (P3.2). *Differentiator:*
the allowlist appears in one rendering. We ask how much of the model-to-model spread comes from how
the allowlist is written. The v1 abstract names the three models (Qwen 2.5 7B, Llama 3.1 8B, Claude
Haiku 3.5); "Qwen 2.5 7B is worst" is *not in abstract*.

**Threat-Preserving Representation Sensitivity (2610.03585).** Introduces TPRS: hold task, harmful
action, *security policy*, ground truth and evaluation fixed, change only the agent-visible
representation (e.g., tool names). Neutral tool names raise ASB attack success by 11.67 pp
(GPT-5-mini) and 13.21 pp (Claude Haiku 4.5); on AgentDojo the ASR shift is 0.50 pp but utility
falls 5.36 pp. *Overlap:* the closest "representation changes agent security outcomes" result, and
it argues for reporting over a controlled set of representations, as we do. *Differentiator:* it
varies the *threat/tool* representation and holds the policy fixed; we vary the *policy*
representation and hold the threat/task fixed. The two axes are complementary.

**Policy-as-Logic (2608.11905, workshop).** Policies are written in formal logic; at inference the
LLM extracts facts to ground predicates and an answer-set solver decides. This beats policy-as-prompt
and policy-as-code "in most cases" with ~10× fewer tokens, and stays robust under input
perturbations. *Overlap:* the strongest canonicalize-then-decide baseline, implemented as the
solver arm in P3.2. *Differentiator:* perturbations are of the *query*, not the policy's
rendering, and the domain is rules-in-general (tax, baggage), not authorization. RuleArena airline,
the 0.94–1.00 accuracy, the 0.38/0.40 baselines, Tax 0.00 and the subjective-HR boundary are
*not in abstract*.

**Ghost in the Context (2605.12535). Mismatch.** v3 studies *policy-carriage integrity*: whether
trusted policies stay present, sound and bound in the decision state assembled before an agent
acts. On AutoGen/tau3 and OpenHands/SWE-bench traces, protected placement preserves policy and
task-local placement loses it. A behavioural calibration shows 0/90 unsafe proposals, so policy
absence alone did not cause unsafe behaviour. It proposes ControlCapsule. *Overlap:* a systems view
of how a policy reaches the model, and "enforce structured policies at the action boundary".
*Differentiator:* it asks whether the policy *survives* context assembly, not whether equivalent
renderings yield the same decision. *Flags:* the title differs from novelty_check, and the
"Policy IR (subject/action/object/effect/condition/evidence)" and 512-token budget are in no
abstract version, so do not cite it as a "Policy IR" precedent without a full-text check.

**MetaPermit (2609.31039).** An LLM infers a fixed set of task-independent meta-attributes per
proposed tool call, and a fixed policy decides. It reports 31% more consistent decisions than
LLM-driven authorization and up to 109% better task completion than CaMeL / IPIGuard on AgentDojo
and AgentDyn with two open-weight LLMs. *Overlap:* separates semantic inference from a deterministic
decision, as our IR arm does (P3.1). *Differentiator:* it canonicalizes the *request*; we
canonicalize the *policy* from any rendering. Its "consistency" is not invariance across policy formats.

**Structured Decomposition (2609.24036).** NL access-control statements → extracted components →
schema-validated → Rego, with linting and generated tests. On 372 ACRE-complete statements it
reaches 50.3% end-to-end correctness vs 15.3% for a single-prompt baseline (3.3×), and correct deny
semantics on 87.5% of deny policies vs 37.5%. *Overlap:* an existing intermediate schema for
access-control statements that P3.1's IR should be compared against, and a consumer of ACRE (P1.9).
*Differentiator:* policy *generation*, not deciding under a given policy, and no invariance
measurement. The exact JSON fields {decision, subject, action, resource, condition, purpose} are
*not in abstract*. Same authors as Prose2Policy.

**AutoCedar / CedarBench (2607.03656).** A verifier-guided agent that turns NL requirements into
reviewed "intent atoms" and synthesizes Cedar against them; it converges on all 221 CedarBench
tasks ("authorization tasks paired with executable semantic boundaries"). *Overlap:* CedarBench is a
planned real-policy source (P1.7) and Cedar is our executable rendering. *Differentiator:*
synthesis, not decision-making under equivalent renderings. novelty_check's "explicitly about
cross-policy interference" is *not in abstract*.

**Quacky (ASE '22 tool paper).** Translates AWS IAM / Azure / GCP policies to SMT and uses a model
counter to quantify (relative) permissiveness. The paper analyzes 41 AWS, 5 Azure and 5 GCP
policies. *Overlap:* the SMT/model-counting route to equivalence for IAM, the backend P1.8 may
reuse. *Differentiator:* a verification tool, no LLMs. The "587 policies" dataset attributed to
Quacky is not described in this paper; its provenance must be pinned in P1.8.

**ACRE (Slankas, Xiao, Williams, Xie, ACSAC 2014).** Extracts access-control rules
(subject–action–resource) from NL requirements documents in conference management, education
and healthcare. *Overlap:* the NL policy corpus for P1.9; later LLM work (Prose2Policy, Structured
Decomposition) evaluates on an ACRE statement set (372 "ACRE-complete" statements per the
Structured Decomposition abstract; the full set size was not confirmed from an abstract). *Differentiator:* extraction, no LLM decision-making. The name "ACRE"
is not on the ACSAC program page; the dataset's exact release and license must be pinned in P1.9.

**Prose2Policy (2603.15799). Mismatch (authors).** NL access-control policies → Rego via a
detect / extract / validate / lint / compile / test pipeline; on ACRE, 95.3% compile rate, 82.2%
positive-test and 98.9% negative-test pass rates. *Overlap:* NL→Rego is one of our rendering
directions (P1.5 if Rego is adopted), and it supplies the ACRE pointer. *Differentiator:*
generation, not decision invariance. *Flag:* arXiv lists two authors, not three.

**How Language Models Choose Sides (2608.28648, ICML 2026 MI workshop).** Benchmark of 41 paired
system/user constraint conflicts over eight models. Llama-3.1-8B follows the system prompt in only
0.10 of conflicts, yet the outcome is linearly decodable at 0.97 balanced accuracy (17 pp above a
metadata-only baseline), and steering raises compliance from 0.132 to 0.530. *Overlap:* the probing
template for Phase 5 (P5.1–P5.4), including metadata-only baselines and a steering check.
*Differentiator:* it probes channel arbitration. We probe owner identity, allow/deny and the
governing field, and test whether probes transfer *across policy renderings*.

**The System Prompt Illusion (2609.38205).** CKA over 17 models (1.5B–72B) and 20 system prompts:
persona/format instructions restructure intermediate layers, safety instructions barely move them
(restrictive vs permissive CKA correlation 0.997). Prompt category is linearly decodable at every
layer, but restructuring happens at few layers, and depth predicts behavioural effect (ρ=0.761).
*Overlap:* the per-layer CKA + probe + patching design P5.1/P5.3/P5.4 adopt. *Differentiator:*
compares different instructions; we compare *equivalent* renderings of one policy and ask whether
divergence predicts decision disagreement. The ">85% probe accuracy" is *not in abstract*.

**Phase transitions in constraint satisfaction (2608.12426). Mismatch (quote).** CSE benchmark,
15 models, k=1–12 constraints with deterministic verifiers: all-k success collapses while
per-constraint pass decays smoothly, and reliability breaks beyond 5–6 constraints. Failures are
nearly independent; residual coupling tracks shared output features "rather than pairwise
interference". *Overlap:* the direct counterpoint to cross-scope interference (A1.1).
*Differentiator:* generic output constraints; A1.1 holds constraint count fixed and varies semantic
independence of authorization fields. *Flag:* the φ=+0.067 figure and novelty_check's quoted
wording are not in the abstract.

**AgentDojo (2406.13352).** Extensible prompt-injection environment: 97 tasks, 629 security cases,
attacks and defenses. *Overlap:* candidate harness for T3 (H0.1 question 3, P2.7) and the
benchmark MetaPermit and TPRS evaluate on. *Differentiator:* injection-centric; the policy is not
a varied factor.

**Agent Security Bench (2410.02644, ICLR 2025).** 10 scenarios, >400 tools, 27 attack/defense
methods, 13 backbones; highest average ASR 84.30%. *Overlap:* the benchmark TPRS perturbs. It
shows the community already scores agent security per configuration. *Differentiator:* attacks
vary, the policy does not.

**GrantBox (2603.28166, FSE 2026 IVR). Mismatch (quote).** A sandbox that wires real-world tools
with genuine privileges into agents and measures privilege usage under prompt injection: basic
security awareness but 84.80% average ASR in crafted scenarios. *Overlap:* privilege misuse as the
outcome, like our UIR. *Differentiator:* attack-driven, single policy representation.

**AuthBench (2605.14859).** Permission-boundary inference: a model maps a terminal task to a
file-level rwx policy; 120 tasks with validators. Models both omit needed permissions and grant
unneeded ones, and more reasoning pushes each toward a model-specific "authorization attractor".
*Overlap:* LLM authorization competence on realistic tasks. *Differentiator:* the model *writes*
the policy; we give it one (in varying renderings) and test whether it *obeys* it.

## Overlap and differentiator: other papers named in novelty_check.md

All verified on their arXiv abstract page (2026-10-08); no withdrawals. One paragraph each.

**Format and representation sensitivity**
- **Tam et al., "Let Me Speak Freely?" (2408.02442).** Format *restrictions on output* (JSON/XML
  generation) degrade reasoning. *Overlap:* a format effect our strict-JSON T2 answers could
  inherit, so T1 log-probs are a useful cross-check. *Differentiator:* output format, not input
  policy format.
- **Mixture of Formats, Ngweta et al. (2504.06969, NAACL SRW 2025; arXiv title "Towards LLMs
  Robustness to Changes in Prompt Format Styles").** Diversifying few-shot example styles reduces
  prompt brittleness. *Overlap:* a prompting-time alternative to P4 SFT. *Differentiator:* generic
  tasks, few-shot style, not policy renderings.
- **Lyu et al., "Keeping LLMs Aligned After Fine-tuning" (2402.18540, NeurIPS 2024).** Template
  choice at fine-tune vs test time governs safety retention (Pure Tuning, Safe Testing). *Overlap:*
  P4.4 must report safety/utility regressions and control templates. *Differentiator:* about
  safety prompts, not policy formats.
- **Schema First Tool APIs (2603.13404).** Free-form docs vs JSON Schema vs schema + diagnostics
  with identical tool semantics; schemas cut interface misuse but not semantic misuse (one local
  model, pilot). *Overlap:* holds content fixed and varies representation of an agent-facing spec.
  *Differentiator:* tool specs, not permissions. "Generated from one canonical contract" is *not in
  abstract*.
- **Representation Robustness in math (2607.20520, HCII 2026).** Story / word-equation / symbolic /
  isomorphic variants flip correctness; code scaffolds do not remove the sensitivity. *Overlap:*
  representation flips as a metric. *Differentiator:* math. "Max–min representation gap" is *not in
  abstract*.
- **GeoRepEval (2604.16421).** Invariance@3 is provably bounded by the weakest representation; up to
  14 pp gaps across Euclidean / coordinate / vector forms. *Overlap:* the closest formal worst-case
  invariance metric. We adopt the idea (P0.2, P2.2) and cite it as such. *Differentiator:* geometry,
  no security consequence.
- **Same Quantity, Different Answer (2609.25009).** 3,600 exact-rational problems, five open-weight
  systems: canonical accuracy 0.969–0.996, orbit invariance 0.851–0.981. It warns that strict
  parsers can masquerade as reasoning failures. *Overlap:* orbit invariance ≈ worst-case over
  renderings, and the parser warning applies to our strict T2 parser (parse failures as their own
  category). *Differentiator:* numerals. (arXiv shows v1 dated 27 Jul 2026 under a 2609 ID; recorded
  as displayed.)
- **Chess geometric stability (2512.15033).** Consistency under board rotation, mirroring, colour
  inversion and format conversion; high accuracy with low stability. *Overlap:* accuracy–stability
  gap. *Differentiator:* games.
- **TsuGO (2608.13221). Mismatch (characterization).** Process-level search-efficiency benchmark on
  Go life-and-death problems. *Overlap:* little; it is not an invariance paper. Drop it from the
  "worst-case metric" lineage unless full text shows otherwise.
- **Policy-as-Prompt for agents (2509.23994, NeurIPS 2025 workshop).** Compiles design documents
  into a policy tree and prompt-based runtime classifiers. *Overlap:* policies delivered as prompts.
  *Differentiator:* no rendering comparison.

**Authorization and policy-compliance benchmarks**
- **Role-Conditioned Refusals (2510.07642).** Text-to-SQL with PostgreSQL RBAC policies; compares
  prompting, generator–verifier and LoRA; longer policies reduce reliability. *Overlap:* open models
  applying role policies. *Differentiator:* policy length, not format. Model names (Llama 3.1,
  Mistral-7B) *not in abstract*.
- **Can LLMs Make (Personalized) Access Control Decisions? (2511.20284).** 307 user privacy
  statements, 14,682 smartphone permission decisions vs LLM decisions. *Overlap:* LLM allow/deny.
  *Differentiator:* preference-based, no formal policy.
- **LLMAC (2602.09392, CCNC 2026).** Mistral 7B trained on synthetic RBAC/ABAC/DAC scenarios, 98.5%
  accuracy. *Overlap:* LLM as policy decision point. *Differentiator:* one representation.
- **MiniScope (2512.11147).** Task-centric hierarchical permission model with runtime least
  privilege; 43.4–89.4% fewer confirmations. *Overlap:* least privilege for agents. *Flag:*
  novelty_check's "computes scopes with a solver and finds LLM-chosen scopes weaker" is *not in
  abstract*.
- **ClawsBench (2604.05172).** Five mock productivity services, 44 tasks including safety-critical
  ones; varies scaffolding levers. *Overlap:* agent safety harness. *Differentiator:* policy
  representation not varied.
- **AgentSecBench (2605.26269).** Formal games (instruction integrity, retrieval confidentiality,
  capability integrity); separates prompt annotations from enforcing projections. *Overlap:* the
  "prompt-level policy vs enforcement" distinction behind our proxy baseline. *Differentiator:* no
  rendering axis.
- **ToolPrivacyBench (2606.28061).** 2,150 cases auditing whether private atoms reach only
  authorized tools. *Overlap:* authorized-sink checks resemble UIR. *Differentiator:* privacy flow.
- **Capability Gates Are Not Authorization (2606.28679).** Agent frameworks gate capabilities but
  do not re-authorize each call; ScopeGate PDP/PEP. *Overlap:* motivates action-boundary
  enforcement (P3.2 upper bound). *Differentiator:* framework audit, no model-side rendering study.
- **The Claws in Plain Sight (2608.20658).** Authority-pressure attack causes disclosure in tool
  arguments; stronger privacy prompts reduce but do not eliminate it (20.8–75.0%). *Overlap:*
  prompt-level policy is not a portable boundary. *Differentiator:* policy *strength* levels, not
  equivalent renderings.
- **Influence Is Not Authority (2608.29942).** An authorization-equivalence audit (96 conditions):
  relocating a value from user to tool shifts causal guardrail signals toward "attack" under Llama
  and Gemma scorers. *Overlap:* the "authorization-equivalent inputs should get equal treatment"
  logic, applied to guardrail scorers. *Differentiator:* varies value provenance, not policy
  rendering. "62.5–75% missed at zero false alarms" is *not in abstract*.
- **τ-bench (2406.12045).** Tool-agent-user benchmark with domain policies; pass^k reliability.
  *Overlap:* policy-following agents. *Differentiator:* policies in one NL rendering.
- **RuleArena (2412.08972, ACL 2025).** Airline, NBA, tax rules; models confuse similar rules.
  *Overlap:* the same domains Policy-as-Logic's abstract names (tax, airline baggage); novelty_check
  says Policy-as-Logic evaluates on RuleArena, which its abstract does not state. *Differentiator:* NL rules only.
- **DeonticBench (2604.04443).** 6,232 deontic tasks (tax, airline, immigration, housing) with
  optional NL→Prolog solving. *Overlap:* rules with an executable form. *Differentiator:* the
  Prolog is a solving aid, not a rendering whose equivalence to NL is tested for decision invariance.
- **RGDT-Bench (2609.34455).** Rule-governed decisions and checkable justifications, failures
  attributed to rule use / condition / evidence / aggregation. *Overlap:* diagnosis of where
  rule application fails. *Differentiator:* no format axis.
- **PolicyBank (2604.15505).** Agents refine their understanding of NL authorization policies with
  ambiguities and gaps via memory. *Overlap:* NL authorization policies are ambiguous, which is why
  Gate 1 includes a human ambiguity spot-check. *Differentiator:* gaps in the spec, not equivalent
  renderings.
- **PolicyGuard dialogue verifier (2606.29225).** A sub-agent verifier for policy adherence on
  τ²-bench airline. *Overlap:* name collision with the DLP PolicyGuard; cite with the distinct key
  `kang2026policyguard`. *Differentiator:* verification, no rendering axis.
- **Policy-as-Skill (2609.27087).** Versioned executable policy capabilities with audit; deterministic
  control helps selectively. *Overlap:* deterministic control as a mitigation. *Differentiator:*
  governance decision support.
- **RuLES (2311.04235), Instruction Hierarchy (2404.13208), IHEval (2502.08745).** Rule-following
  and channel-priority benchmarks / training. *Overlap:* the instruction-compliance background.
  *Differentiator:* none varies the representation of the rule.

**LLMs and formal access-control languages (generation side)**
- **Synthesizing Access Control Policies using LLMs (2503.11573, NLBSE@ICSE 2025).** Zero-shot IAM
  policy synthesis from request lists or NL; argues for structured, syntax-based prompts.
  *Overlap:* prompt-specification *style* affects policy quality. *Differentiator:* generation.
- **RAISE / CedarInstruct (2609.33796).** 5,800 Cedar synthesis scenarios with verified targets;
  SFT + verifier-signal RL. The abstract confirms "untrained models rarely write valid Cedar but
  often reason correctly when they do". *Overlap:* Cedar competence of open models and a generation-
  side interpretation/application split. *Differentiator:* generation. ">97% valid / about a third
  verified" is *not in abstract*.
- **NLAC / NLACBench (2606.06726).** NL help-desk requests → network access policies; accuracy falls
  with network size. *Differentiator:* generation.
- **DePLOI / IBAC-DB (2402.07332).** LLM synthesis and auditing of database access control from
  intent-based abstractions; IBACBench. *Differentiator:* generation/audit. The "synonym-replace
  column names" perturbation is *not in abstract*.
- **ARPaCCino (2507.10584).** Agentic RAG producing Rego for IaC compliance. *Differentiator:*
  generation.
- **ABAC policy mining with LLMs (2511.18098).** LLMs mine compact ABAC policies for small
  scenarios. *Differentiator:* mining.

**Canonicalization and formal enforcement**
- **LACE, "Say What You Mean" (2505.23835).** NL IoT access policies → structured rules with
  retrieval and formal validation. *Overlap:* NL→structured policy before deciding. JSON rules,
  SMT conflict check and OPA verification are *not in abstract*.
- **APPA (2607.24625). Mismatch (title, characterization).** IFC with recoverable trajectory
  confinement and a dual-phase reference monitor at tool dispatch / MCP gateways. *Overlap:*
  enforcement at the action boundary. *Differentiator:* information flow, not policy rendering, and
  not canonicalization of tool calls.
- **Formal Policy Enforcement for Real-World Agentic Systems (2602.16708).** Datalog policies over
  abstract predicates with a reference monitor; argues that NL policies in system prompts give no
  guarantee. *Overlap:* the "prompts are not enforcement" premise. *Differentiator:* enforcement,
  not measurement of rendering effects.
- **Progent (2504.11703).** Symbolic privilege policies over tool names/arguments, LLM-generated and
  updated, with SMT-checked narrowing. "JSON-Schema policies" and "Z3" are *not in abstract* (it
  says "symbolic rules" and "SMT solver").
- **VeriGuard (2510.05156).** Verified behavioural policies plus online action monitoring.
  *Differentiator:* enforcement.
- **AudAgent (2511.07441, PETS'26).** Cross-LLM voting to formalize privacy policies; runtime
  compliance auditing. "Into JSON" is *not in abstract* ("formal models").
- **Executable Governance / P2T (2512.04408, AAAI-26 workshop).** NL policy → compact DSL, a
  canonical rule representation. *Overlap:* an alternative IR. *Differentiator:* governance rules.
- **Logic-LM (2305.12295, EMNLP 2023 Findings) and LINC (2310.15164, EMNLP 2023).** LLM translates
  NL to a symbolic form, a solver/prover decides. *Overlap:* the general pattern behind P3.1/P3.2.
  *Differentiator:* logic puzzles, not policies, and no rendering invariance.

**Format attacks on guardrails**
- **Prompt Injection as Role Confusion (2603.12277, ICML 2026).** Injected text that *sounds* like a
  role is represented like that role; CoT Forgery attack. *Overlap:* format manufactures authority,
  the mirror image of our "legitimate policy loses force when re-rendered". Its role probes are a
  probing precedent for P5.
- **From Shield to Target (2606.14517).** DoS on LLM guardrails via reasoning-length payloads and
  structural mutations. *Differentiator:* availability attack.

**Knowing vs applying**
- **Know It, Act on It (2607.29433).** Paired Know/Act tests show agents recall preferences but do
  not act on them. *Overlap:* the interpretation/application dissociation (P2.2, P2.9) in another
  domain.
- **When Do LLMs Apply the Wrong Law? (2608.14610).** Models know historical statutes but apply the
  newest one. *Overlap:* knowing the governing rule ≠ applying it.
- **Correct Chains, Wrong Answers (2604.13065, ICLR 2026 workshop).** Correct reasoning steps with
  wrong declared answers. *Overlap:* dissociation; reminds us to score the final structured answer.
- **RuleBench / inferential rule following (2407.08440).** Separates inferential rule following
  from instruction following. *Overlap:* background.

**Multi-constraint interference**
- **ComplexBench (2407.03978), DeCRIM / RealInstruct (2410.06458), RECAST (2505.19030).** Multiple
  simultaneous constraints degrade compliance. *Overlap:* background for A1.1. *Differentiator:*
  constraint count is the variable; A1.1 holds it fixed.

## Novelty claims mapped to planned results

Rows are the Novelty Assessment table of `docs/novelty_check.md`; step ids are from `docs/PLAN.md`.
"Must cite" lists the verified papers the paper must engage with for that claim.

| Planned contribution | Status (novelty_check) | Planned result that carries it | Steps | Must cite |
|---|---|---|---|---|
| Decisions vary with policy rendering, five equivalent formats | Open (partial precedent) | Phase-0 kill test (Gate 0), then per-rendering accuracy and rendering disagreement on the engine-certified benchmark | P0.2–P0.4, G0; P1.2–P1.4; P2.2, P2.4–P2.6, P2.9 | PolicyGuard DLP (withdrawn), Not as Sweet, Sclar, He |
| Exploitable through format choice | Open but needs a threat model | Direction of errors: deny→allow vs allow→deny flip rates per rendering; T3 UIR by rendering. **No step writes the threat model itself**; it currently falls to the paper draft | P0.4, P2.2, P2.9, P2.8; W1.1 | Role Confusion, TPRS, (Policy Puppetry, not verified) |
| Worst-case accuracy over renderings | Metric taken generically; new for authorization | Worst-case accuracy = min over renderings, with world-clustered CIs; deny→allow as the security-relevant worst case | P0.2, P0.3, P0.4; P2.2, P2.9 | GeoRepEval, Same Quantity Different Answer, Not as Sweet, Sclar |
| Benchmark of real IAM/Cedar/OPA/GitHub/MCP policies with engine ground truth | Partly open | ≥1,000 engine-checked worlds × ≥5 renderings, frozen with hashes; datasheet; release | P1.1–P1.11, G1; P1.7 (CedarBench), P1.8 (Quacky; see 587-vs-546 flag), P1.9 (ACRE), P1.6 (MCP/repo synthetic); D1.1 | AutoCedar/CedarBench, Quacky, PolicySummarizer, ACRE, Prose2Policy, Structured Decomposition |
| Agentic harness: UIR vs representation | Open (metric exists) | T3 UIR, authorized-action refusal and completion with rendering as the only varied factor; T2→T3 transfer | P2.7, P2.8 | Prompts Don't Protect, AgentDojo, ASB, GrantBox, TPRS |
| Interpretation vs application dissociation | Partly taken | Dissociation rate (interpretation correct ∧ application wrong) by rendering | P1.10 (both query types), P2.2 (metric), P2.9 | PolicySummarizer, Know It Act on It, Wrong Law, RAISE |
| Interference between policy fields | Crowded | Appendix: interference with constraint count fixed, semantic independence varied | A1.1 (only if time) | Phase transitions (2608.12426), ComplexBench, DeCRIM, RECAST |
| Per-format probing of owner/permission representations | Open (methods exist) | Cross-rendering probe transfer, per-layer CKA, steering/patching check; metadata-only and shuffled-label baselines | P5.1–P5.4 | Choosing Sides, System Prompt Illusion, Role Confusion |
| IR canonicalization mitigation | Partly taken | Closure of the worst-case gap and T3 UIR under the IR arm vs baselines (CoT, restate-as-JSON, Policy-as-Logic solver, enforcement proxy) | P3.1, P3.2, P3.3, P3.4, G3 | Policy-as-Logic, MetaPermit, Structured Decomposition, Logic-LM, LINC, Prompts Don't Protect; Ghost in the Context only after full-text check |
| Format-diverse fine-tuning | Method taken | Worst-case gap on held-in vs held-out (Cedar) renderings; single-format matched-token and no-SFT controls; BFCL regression | P4.1–P4.4 | Multi-Format Training, Mixture of Formats, Lyu et al. |

## Open items for a human

- **P1.8 source count.** The plan and P1.8 say "Quacky (AWS IAM, 587 policies)"; the verified
  PolicySummarizer abstract says 546 AWS policies (plus 100 Azure, 100 GCP), and the Quacky tool
  paper analyzes 41 AWS policies. Pin the actual dataset and its size when P1.8 runs; no plan text
  was changed here.
- **Threat-model step.** The "exploitable through format choice" claim has no dedicated step; it
  falls to W1.1. Raise it if the paper should state the threat model earlier.
- **Full-text checks before citing numbers.** Every *not in abstract* claim above needs a full-text
  read before it appears in `paper/`.
