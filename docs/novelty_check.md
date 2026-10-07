# Novelty Check: Representation Invariance of Authorization Policies for LLM Agents

No paper we found makes the headline claim. We found no 2023–2026 work showing that one authorization policy, rendered as semantically equivalent natural language, decision-owner statements, permission tables, JSON, and Rego/Cedar, produces different LLM decisions and agent actions, measured as worst-case accuracy across renderings. The parts around that claim are crowded, though, and several 2026 preprints come close enough that reviewers will cite them.

## TL;DR
- **No scoop found.** The closest direct evidence is PolicyGuard (arXiv 2608.02687), which compared only two formats (natural language vs JSON). It covered data-loss-prevention (DLP) classification, not authorization, and reported a 6.1 pp advantage for natural language. The arXiv abstract page now marks it as withdrawn. No paper compares five equivalent renderings of an authorization policy, scores them against an authorization engine's ground truth, and reports worst-case accuracy.
- **Much of the surrounding ground is taken.** Prior work already covers: LLMs failing to apply real AWS policies (59–68% accuracy on allow/deny predictions); agents calling forbidden tools despite allowlists; format robustness of document workflows (decisions drift in over 41% of instances); translating policies into logic before answering (policy-as-logic, Policy IR, MetaPermit); multi-format fine-tuning; linear probes of instruction-hierarchy conflicts; and failures from stacking many constraints.
- **The most defensible framing** is a controlled, engine-checked measurement of format-induced *authorization* errors, with worst-case-over-renderings as the metric. Run it in an agentic harness that measures unauthorized-action rate, and add canonicalization and format-diverse fine-tuning as mitigations evaluated against that metric. The interpretation-vs-application dissociation and per-format probing are secondary novelties. Cross-scope interference has the most prior art and the weakest novelty.

## Key Findings

1. **Headline claim: open, with one partial precedent.** PolicyGuard (Park, Kim, Shim; arXiv 2608.02687, Aug 2026) is the only paper we found that tests the same policy content in two formats and reports a significant decision difference: McNemar χ²=31.58, p<0.001, effect size h=0.260. But it is a DLP prompt classifier, compares only two formats, does not measure worst-case behaviour, and the arXiv abstract page lists it as withdrawn (v3, 6 Aug 2026). A planned subagent search found no paper testing tables, JSON, Rego/Cedar, or IAM versions of the same authorization policy against each other.
2. **The general phenomenon (format sensitivity) is established**, so reviewers will treat it as expected. Sclar et al. 2024, He et al. 2024, and Tam et al. 2024 (all known to the user) cover generic tasks. Newer 2026 work extends the finding to document workflows, math, geometry, numerical representations, and tabular prediction, and several papers already use "worst-case" or "invariance" metrics. The contribution therefore has to be the *security consequence* (unauthorized actions), not the existence of sensitivity.
3. **Canonicalization as a mitigation is partly taken.** Policy-as-Logic (IBM, arXiv 2608.11905), Ghost in the Context's "Policy IR" (arXiv 2605.12535), MetaPermit (arXiv 2609.31039), and Logic-LM/LINC already separate "extract into a structured form" from "decide deterministically". None of them measures invariance across policy *renderings*. The novel element is canonicalizing *the policy*, starting from any of several formats, as an invariance fix, with the gain quantified as the closed worst-case gap.
4. **Format-diverse fine-tuning has a direct generic precedent** in "Improving Cross-Format Robustness in Language Models with Multi-Format Training" (arXiv 2606.11643), and Mixture of Formats prompting is related. Applying either to authorization is new; the method itself is not.
5. **The agentic unauthorized-action measurement is adjacent to a crowded area.** Prompts Don't Protect (arXiv 2605.18414), GrantBox (arXiv 2603.28166), AuthBench (arXiv 2605.14859), ClawsBench, AgentSecBench, and MiniScope all measure unauthorized tool use or least privilege. None varies the *representation of the policy*. The closest is Threat-Preserving Representation Sensitivity (arXiv 2610.03585), which varies the representation of the *attack and tools* while holding the security policy fixed.
6. **Mechanistic probing has close precedents** in "How Language Models Choose Sides" (arXiv 2608.28648), which decodes system-vs-user conflict outcomes at 0.97 balanced accuracy, and "Prompt Injection as Role Confusion" (arXiv 2603.12277). Neither probes who owns a decision or which permission applies, and neither compares across policy formats.

## Details

### 1. Format and representation sensitivity for policies and rules

**PolicyGuard: Prompt-Configurable Semantic DLP for LLM Coding Agents**
Kyutae Park, Jungwon Kim, Daeyeol Shim. arXiv 2608.02687 (cs.CR), Aug 2026; withdrawn per the arXiv abstract page (v3).
An LLM filters prompts before they reach a coding agent, following a natural-language policy file, and decides whether each prompt contains credentials, personal data, or proprietary content. On a frozen 927-prompt test set it reports a 96.5% effective block rate with a 3.0% false-positive rate, and 86.4–96.5% across four LLMs. In an information-matched comparison, natural-language policies beat the same content written as JSON by 6.1 pp. JSON had a lower false-positive rate (1.3% vs 3.0%); the authors call JSON "more conservative (ALLOW-biased)", which is internally inconsistent wording.
**Overlap: partial, and the most direct precedent.** It shows a decision difference across two formats, but in DLP rather than authorization, with no worst-case metric, no agent actions, no ground truth from an authorization engine, and no mitigation. Because it is withdrawn, it is weak as a citation. Cite it as motivating evidence, not as established prior work.

**Not as Sweet by Another Name: An Empirical Study of Format Robustness in LLM Document Workflows**
Xiaoyu Zhang, Xianyun Cheng, Tianlin Li, Yuwei Zheng, Yue Yang, Yang Liu. arXiv 2607.27648, Jul 2026 (ACM DOI 10.1145/3832783.3837456).
The paper builds a metamorphic testing framework with three invariance relations (same decision, same supporting evidence, same execution stability). It runs 4 workflows × 4 tasks × 4 document formats, 48,000 executions in total. Switching formats lowers accuracy by up to 53.63%, and over 41% of instances change their decision under at least one format pair. Lightweight mitigations recover up to 44.21% of the drift.
**Overlap: adjacent, but methodologically very close.** It formalizes a "decision outcome invariance" relation and observes that "proximity in average accuracy does not guarantee behavioral consistency", which is essentially the motivation for a worst-case metric. Its inputs are documents, not policies, and nothing involves authorization. Expect reviewers to cite it.

**Threat-Preserving Representation Sensitivity in Agent-Security Benchmarks**
Neeraj Karamchandani, Piyush Nagasubramaniam, Xinhong Xie, Sencun Zhu, Dinghao Wu. arXiv 2610.03585, Oct 2026.
The paper keeps the security policy fixed and changes how threats and tools are presented. Renaming threat-related tools neutrally raises attack success on Agent Security Bench by 11.67 pp (GPT-5-mini) and 13.21 pp (Claude Haiku 4.5). On MCPTox, an explicit threat-related name lowers it by 11.00 pp and 4.11 pp respectively. On AgentDojo the shift is 0.50 pp, but benign utility drops 5.36 pp.
**Overlap: adjacent, and the closest "representation changes agent security outcomes" paper.** It changes the attack's surface form, not the policy's. Your paper is the complementary axis, and this paper shows reviewers already accept representation sensitivity as a security problem.

**Schema First Tool APIs for LLM Agents** (arXiv 2603.13404, 2026). Holds the environment fixed and compares free-form natural-language tool documentation, strict JSON Schema, and JSON Schema with structured diagnostics, all generated from one canonical contract. **Overlap: adjacent.** It varies the format of tool specifications, not permission policies, and measures misuse and recovery rather than authorization.

**Generic invariance studies with worst-case or invariance metrics** (all 2025–2026 arXiv): "Representation Robustness Under Executable Reasoning Constraints" in math (2607.20520, max–min representation gap); GeoRepEval (2604.16421, "Invariance@3", which is bounded by the weakest format); "Same Quantity, Different Answer" (Atta-Duncan, arXiv 2609.25009; 3,600 exact-rational problems across five open-weight systems, where canonical accuracy is 0.969–0.996 but orbit invariance falls to 0.851–0.981); the tabular-prediction robustness paper (PMC, NL vs JSON row formats); and the chess and Go stability papers (2512.15033, 2608.13221). **Overlap: adjacent.** Together they mean that "worst-case accuracy over equivalent renderings" is no longer a new *metric idea*. Present it as an adopted metric applied to a new risk, not as an invention.

**Policy-as-Prompt: Turning AI Governance Rules into Guardrails** (arXiv 2509.23994) and Policy-as-Prompt for content moderation (Palla et al., FAccT 2025). **Overlap: adjacent.** They convert governance rules into prompts but do not test whether different renderings agree.

### 2. Benchmarks for authorization, permissions, and policy compliance

**Prompts Don't Protect: Architectural Enforcement via MCP Proxy for LLM Tool Access Control**
Rohith Uppala. arXiv 2605.18414 (v3, Aug 2026).
When unauthorized tools are visible in context, models pick them in 48–68% of adversarial scenarios. Role-escalation prompts reach 96% unauthorized invocation on frontier models. Explicit per-tool allowlists cut the unauthorized invocation rate (UIR) to between 4.0% and 37.0% depending on the model (Qwen 2.5 7B is worst), "with no reliable relationship to general capability". An attribute-based access control (ABAC) proxy that filters the tool registry makes UIR 0% by design.
**Overlap: partial, and your strongest baseline for the agentic harness.** It establishes UIR as a metric and uses open-weight models, but presents the allowlist in only one format. Your paper could reuse its scenarios and ask how much of the 4–37% spread comes from how the allowlist is written.

**Neurosymbolic Characterization for Reliable Access Control Policy Analysis (PolicySummarizer)**
Adarsh Vatsa, Bethel Hall, William Eiers (Stevens). arXiv 2510.20692, ISSRE 2026.
On 587 AWS IAM policies from the Quacky dataset, LLMs predicted allow/deny for 20 requests per policy. The best reasoning models were only 59.1–67.9% accurate, even though their explanations of the policies were highly consistent (90.7–96.0%). The authors call this the "Verifiable Synthesis Paradox". A user study found people misjudging two syntactically different but semantically equivalent JSON policies: only 39–44% were correct without the tool.
**Overlap: partial, and important.** It is the closest work on LLMs *applying* real IAM policies with formal ground truth, from an SMT solver and automata. It also already reports a gap between explaining a policy and deciding under it, which pre-empts part of your interpretation-vs-application claim. It uses one format (raw JSON) and does not look at agents or invariance.

**Role-Conditioned Refusals: Evaluating Access Control Reasoning in LLMs** (arXiv 2510.07642). Text-to-SQL generation conditioned on a role and an access policy, with LoRA-tuned Llama 3.1 and Mistral-7B and an ablation over policy length. **Overlap: partial.** It covers open-weight models applying role policies and varies policy length, but not policy format.

**Can LLMs Make (Personalized) Access Control Decisions?** (arXiv 2511.20284). Compares LLM and user decisions on smartphone permission requests. **Overlap: adjacent.** These are preference-based decisions with no formal policy.

**LLMAC** (arXiv 2602.09392). A model trained on synthetic RBAC, ABAC, and DAC scenarios to make allow/deny decisions with explanations. **Overlap: adjacent.** It uses one representation.

**Agent privilege and least-privilege benchmarks**:
- GrantBox, "Evaluating Privilege Usage of Agents with Real-World Tools" (arXiv 2603.28166), finds "only foundational security awareness".
- AuthBench, "Do Coding Agents Understand Least-Privilege Authorization?" (arXiv 2605.14859), tests whether agents can generate permission policies for a task.
- MiniScope (arXiv 2512.11147) computes least-privilege scopes with a solver and finds LLM-chosen scopes weaker.
- ClawsBench (arXiv 2604.05172), AgentSecBench (arXiv 2605.26269), ToolPrivacyBench (arXiv 2606.28061), Capability Gates Are Not Authorization (arXiv 2606.28679), The Claws in Plain Sight (arXiv 2608.20658), and Influence Is Not Authority (arXiv 2608.29942, Llama/Gemma scorers miss 62.5–75% of unauthorized actions at a zero-false-alarm threshold).

**Overlap: adjacent.** All of these measure unauthorized or over-privileged actions. None varies how the policy is represented.

**Policy-following agent benchmarks**: τ-bench and related work, RuleArena (ACL 2025), DeonticBench (Dou et al., arXiv 2604.04443, 6,232 tasks covering tax, airline, immigration, and housing, with reference Prolog), RGDT-Bench (arXiv 2609.34455), PolicyBank (arXiv 2604.15505, refining agents' understanding of authorization policies), the PolicyGuard dialogue verifier (arXiv 2606.29225), Policy-as-Skill (arXiv 2609.27087), and the EMNLP 2025 Industry track work on enforcing company policy in agentic workflows. RuLES (Mu et al. 2023), the Instruction Hierarchy (Wallace et al. 2024), IHEval, AgentDojo, and Agent Security Bench are already known to the user. **Overlap: adjacent.** The rules are given as natural-language text (DeonticBench also allows translation to Prolog), and none tests whether the decision stays the same across renderings of the rules.

### 3. LLMs and formal access-control languages

**Mostly generating policies, which is off your axis**:
- Synthesizing Access Control Policies using LLMs (Vatsa, Patel, Eiers, arXiv 2503.11573). Compares prompt-specification styles for *generating* IAM policies.
- Prose2Policy (Gupta, Sreenivasamurthy, Apple, arXiv 2603.15799). Natural language to Rego; 95.3% compile rate, 82.2% positive and 98.9% negative test pass rates on ACRE.
- Structured Decomposition for Reliable LLM-Generated Access Control Policies (arXiv 2609.24036). Natural language to an intermediate JSON {decision, subject, action, resource, condition, purpose} to Rego. Effective correct coverage is 3.3× the single-shot baseline (50.3% vs 15.3%), which writes semantically wrong deny-Rego for 62.5% of deny policies.
- AutoCedar / CedarBench (arXiv 2607.03656, 221 Cedar tasks, explicitly about "cross-policy interference").
- RAISE / CedarInstruct (Zhou, Vatsa, Eiers, arXiv 2609.33796). Frontier models write valid Cedar for over 97% of tasks but satisfy the full verification plan for only about a third.
- NLAC/NLACBench (arXiv 2606.06726), DePLOI/IBAC-DB (arXiv 2402.07332), ARPaCCino (arXiv 2507.10584), Lawal et al. (TPS-ISA 2024), and ABAC policy mining (arXiv 2511.18098).

**Overlap: adjacent.** These papers are about *producing* policies. They do matter for your work in three ways:
- CedarBench and Quacky are candidate sources of real policies and test requests.
- The intermediate JSON in Structured Decomposition is an existing IR schema you could adopt or argue against.
- The RAISE observation that "untrained models rarely write valid Cedar but often reason correctly when they do" is a generation-side form of an interpretation/application split.

DePLOI tests "syntactically different but semantically equivalent" perturbations, but only to synonym-replace column names, and only for policy diffing.

**Interpreting or applying existing policies**: only PolicySummarizer (above) and Role-Conditioned Refusals test whether LLMs apply given formal policies correctly. Neither compares formats. **This is the clearest open slot.**

### 4. Canonicalization and translating policies into a formal form

**Policy-as-Logic for Robust Reasoning over Rules**
Rahul Nair, Bastian Lipka, Elizabeth Daly (IBM Research). arXiv 2608.11905, Aug 2026.
Policies are translated once into answer-set programs (ASP). An LLM extracts JSON facts from each query against a schema, and the Clingo solver decides. On RuleArena's airline domain, the method reaches 0.94–1.00 accuracy on GPT-OSS-120B, Qwen-2.5-72B, and Llama-3.3-70B, versus at most 0.38 for policy-as-prompt and 0.40 for policy-as-code. Robustness under six query perturbations stays close to accuracy, while baselines fall to 0.00 on Tax. The gain disappears on a subjective HR policy.
**Overlap: partial, and the closest precedent for the mitigation.** It is canonicalize-then-decide, measured for robustness, on open-weight models. But the perturbations are of the *query* (paraphrase, sentiment), not of the *policy rendering*, and the domain is not authorization. Its finding that symbolic decision-making gains nothing on subjective policies is a useful boundary for your IR claim.

**Ghost in the Context: Policy-Carriage Integrity in LLM Agent Context Assembly** (arXiv 2605.12535, v3). Defines a "Policy IR" of subject/action/object/effect/condition/evidence records that a deterministic monitor evaluates, "against the canonical policy representation, not against the model's prose". It also shows that truncation and summarization drop the trusted policy below a 512-token context budget. **Overlap: partial.** It is a policy IR plus a deterministic checker for agents, but it addresses whether the policy survives context assembly, not whether decisions are invariant across formats.

**MetaPermit: Scalable and Auditable Access Control for AI Agents via LLM-Inferred Meta-Attributes** (Ma et al., arXiv 2609.31039, Sep 2026). An LLM infers a fixed set of meta-attributes for each tool call, and a fixed policy decides. The authors report "31% more consistent authorization decisions than LLM-driven authorization" and up to 109% better task completion than CaMeL and IPIGuard on AgentDojo and AgentDyn with two open-weight LLMs. **Overlap: partial.** It puts the *request* rather than the *policy* into canonical form, and its "consistency" is not invariance across formats.

**Other neighbours**: LACE, "Say What You Mean" (arXiv 2505.23835: natural language to JSON rules, an SMT conflict check, then an LLM decision verified by OPA); Agentic Permissions Policy Algebra (arXiv 2607.24625, which puts tool calls into canonical form); Formal Policy Enforcement for Real-World Agentic Systems (arXiv 2602.16708, Datalog); Progent (arXiv 2504.11703, JSON-Schema privilege policies plus Z3); VeriGuard (arXiv 2510.05156); AudAgent (arXiv 2511.07441, which formalizes privacy policies into JSON with cross-LLM voting); Executable Governance (arXiv 2512.04408); and Logic-LM and LINC (EMNLP 2023). **Overlap: adjacent.** The idea of enforcing a policy through a formal intermediate representation is common. The question "does canonicalizing *any* rendering into an IR close the worst-case gap?" is open.

**Format-diverse fine-tuning**: "Improving Cross-Format Robustness in Language Models with Multi-Format Training" (Liu, Zheng, Cao, Jin, Cui, Zhou; arXiv 2606.11643; tested on GLM4 and Llama-3.1) finds that multi-format supervised fine-tuning, not extra single-format data, drives cross-format consistency: on GLM4-9B, full multi-format training raised pass^4 from 12.08% to 19.61%. It also finds that "expanding only about 30% of the training set into multiple formats often recovers most of the gain from full-format training". Mixture of Formats prompting (Ngweta, Kate, Tsay, Rizk; arXiv 2504.06969) varies the styles of the few-shot examples to reduce prompt brittleness. "Keeping LLMs Aligned After Fine-tuning" (Lyu et al., NeurIPS 2024, arXiv 2402.18540) shows that safety after fine-tuning depends on whether training and test templates match, and proposes "Pure Tuning, Safe Testing": fine-tune without a safety prompt, then include it at test time. **Overlap: partial (method), adjacent (domain).** Your fine-tuning arm should be framed as applying and testing this recipe on authorization, ideally with a held-out format such as Cedar when training on Rego.

### 5. Worst-case robustness and format attacks on guardrails

- **HiddenLayer's "Policy Puppetry"** ("Novel Universal Bypass for All Major LLMs", April 2025) is an industry blog post, not peer-reviewed. Prompts disguised as XML, INI, or JSON policy files bypassed system prompts and safety alignment in models from OpenAI, Google, Microsoft, Anthropic, Meta, DeepSeek, Qwen and Mistral, and HiddenLayer notes that "instructions do not need to be in any particular policy language." This is the attack mirror of your "exploited through formatting" claim: text that *looks* like a policy gains authority.
- **"Prompt Injection as Role Confusion"** (arXiv 2603.12277) finds that text which stylistically imitates a high-privilege role is internally represented as high-privilege.
- **Sockpuppeting** (Trend Micro blog, April 2026) inserts a fake acceptance into the assistant-prefill role rather than relying on JSON or code specifically. Of 11 models tested, "every model that accepted the prefill was at least partially vulnerable", including GPT-4o, Claude 4 Sonnet and Gemini 2.5 Flash.
- The **Shield-to-Target guardrail DoS** paper (arXiv 2606.14517).

**Overlap: adjacent.** These attacks use format to manufacture authority. None tests whether a *legitimate* policy loses force when re-rendered. Your "exploitable" claim should state this distinction explicitly: an attacker who chooses the policy's format, such as a tenant writing their own MCP allowlist or a policy file synced from a repository, versus one who injects fake policy text.

### 6. Knowing a rule vs applying it, and probing

- **PolicySummarizer** (above): consistent explanations alongside 59–68% request accuracy. This is the closest dissociation result that involves policies.
- **Know It, Act on It** (arXiv 2607.29433): personalization agents that recall a preference but fail to act on it. Failures are attributed to retrieval, comprehension, or application.
- **When Do LLMs Apply the Wrong Law?** (arXiv 2608.14610): models can recite the old statute but apply the new one.
- **Knowledge-Reasoning Dissociation in clinical NLI**: 92% on fact probes vs 25% on reasoning.
- **Correct Chains, Wrong Answers** (arXiv 2604.13065).
- **Inferential rule-following** (arXiv 2407.08440), which separates triggering errors from execution errors.

**Overlap: adjacent for the general dissociation, partial for the policy setting via PolicySummarizer.** Separating *who owns a decision* (interpretation) from *which action is taken* (application), compared across formats, is not covered.

**Mechanistic work**:
- **"How Language Models Choose Sides"** (Balp-Straffon et al., arXiv 2608.28648). On 41 paired constraints across eight models, Llama-3.1-8B is "anti-hierarchy", following the system prompt in only 0.10 of conflicts. Conflict outcomes are linearly decodable at 0.97 balanced accuracy, and steering raises system compliance from 0.132 to 0.530.
- **"The System Prompt Illusion"** (arXiv 2609.38205): prompts are encoded at every layer (probe accuracy over 85%) but restructure computation only at certain layers (ρ=0.761 with behavioural effect).
- **Probing prompt-leakage intent** (EMNLP 2025).

**Overlap: partial.** The methods exist and use the same open-weight families. Probes for *owner and permission* concepts, and their consistency across formats, are new.

### 7. Interference between multiple constraints

- **"Large Language Models Can Follow Instructions, But Not Many at Once"** (arXiv 2608.12426), 15 models: reliability breaks down beyond 5–6 simultaneous constraints. Notably, its abstract says failures are "weakly correlated through shared output features (mean φ=+0.067), not pairwise interference."
- **ComplexBench** (arXiv 2407.03978), **DeCRIM/RealInstruct** (arXiv 2410.06458), **RECAST** (arXiv 2505.19030), and **"When Thinking Fails"** (NeurIPS 2025: chain-of-thought hurts IFEval and ComplexBench).
- **CedarBench's** explicit framing of "cross-policy interference" on the policy-generation side.

**Overlap: partial and crowded.** A finding of interference between scopes would need to be framed as interference between *semantically independent authorization fields* (for example, a repository-scope rule changing a decision about an MCP tool). The phase-transitions paper's claim that failures are *not* pairwise is a direct counterpoint you would need to address.

## Novelty Assessment

| Planned contribution | Status | Closest prior work | Differentiator |
|---|---|---|---|
| Decisions vary with policy rendering, across five equivalent formats | **Open** (partial precedent) | PolicyGuard (2 formats, DLP, withdrawn); Not as Sweet by Another Name (documents) | Authorization domain, 5 renderings including Rego/Cedar, engine-checked equivalence |
| Exploitable through format choice | **Open but needs a threat model** | Policy Puppetry; Role Confusion; Threat-Preserving Representation Sensitivity | A legitimate policy loses force when re-rendered, rather than fake policy text gaining force |
| Worst-case accuracy over renderings | **Metric taken generically**; new for authorization | GeoRepEval Invariance@3; max–min gap; metamorphic invariance relations | Present it as an adopted metric; report the denials that become allows as the security-relevant worst case |
| Benchmark of real IAM/Cedar/OPA/GitHub/MCP policies with engine ground truth | **Partly open** | Quacky (AWS, 587), CedarBench (221), ACRE, PolicySummarizer | Multi-ecosystem and *multi-rendering*, with equivalence proved by the engine |
| Agentic harness: unauthorized-action rate vs representation | **Open** (metric exists) | Prompts Don't Protect (UIR), GrantBox, AgentDojo | Policy rendering as the independent variable |
| Interpretation vs application dissociation | **Partly taken** | PolicySummarizer (explain vs decide); Know It, Act on It | Split by owner vs action, and show the split *varies by format* |
| Interference between policy fields | **Crowded** | Phase transitions (2608.12426), ComplexBench, CedarBench | Must show causal interference between independent fields, controlled for count |
| Per-format probing of owner/permission representations | **Open** (methods exist) | Choosing Sides (2608.28648); System Prompt Illusion | Show whether representations converge across formats in middle layers and where they diverge |
| IR canonicalization mitigation | **Partly taken** | Policy-as-Logic, Policy IR (Ghost in the Context), MetaPermit, Structured Decomposition | Canonicalize the *policy* from any rendering; measure closure of the worst-case gap |
| Format-diverse fine-tuning | **Method taken** | Multi-Format Training (2606.11643); Mixture of Formats | Held-out-format generalization (e.g., train without Cedar, test on Cedar) |

**Scoop verdict:** no paper makes essentially the same headline claim. The papers most likely to be raised as "already done" are PolicyGuard (format effect on a policy), Not as Sweet by Another Name (format invariance as a metamorphic relation), PolicySummarizer (LLMs misapply IAM policies), and Policy-as-Logic (an IR fixes robustness).

## Recommendations

1. **Lead with security consequences, not sensitivity.** Headline numbers should be worst-case unauthorized-action rate and denials that become allows across engine-equivalent renderings, not average accuracy changes. Use the observation from Not as Sweet by Another Name that similar averages can hide inconsistency to justify the worst-case metric.
2. **Make equivalence airtight.** Prove every rendering equivalent with the Cedar analyzer or OPA tests, and Zelkova/Quacky-style SMT checks for IAM. That is your moat against the generic format-sensitivity papers, none of which can certify equivalence formally.
3. **Report the direction of errors.** PolicyGuard hints that format shifts the allow/deny bias. Whether rendering a policy as code makes models more permissive is the most reviewer-relevant result you could produce.
4. **Benchmark the IR against the strongest baselines.** Compare against Policy-as-Logic-style solver pipelines, the intermediate JSON from Structured Decomposition, and an external enforcement proxy as in Prompts Don't Protect, and show what the IR adds: invariance at lower cost, without a separate enforcement stack.
5. **Reuse existing assets.** Seed the benchmark from Quacky (AWS), CedarBench, and ACRE, and adopt the UIR metric so results can be compared with Prompts Don't Protect.
6. **Reduce or reframe cross-scope interference.** Present it as a controlled causal test with constraint count held fixed, and engage directly with the "not pairwise" finding in arXiv 2608.12426.
7. **Scope the probing claim carefully.** Report probe baselines (metadata-only and control tasks, as in Choosing Sides) and add a causal intervention such as steering or patching. Otherwise reviewers will discount decodability.
8. **Watch the space.** Most close neighbours appeared between May and October 2026. Recheck arXiv cs.CR/cs.CL before submission, especially for follow-ups from the Eiers group (Stevens) and the PolicyGuard authors.

## Caveats

- PolicyGuard is withdrawn on arXiv, and its "ALLOW-biased" wording conflicts with its own false-positive numbers. Do not rely on its 6.1 pp figure as settled.
- Several summaries (Ghost in the Context, Agentic Permissions Policy Algebra, LACE) come from abstracts or search snippets rather than full-text reads. The Ghost in the Context author list was not retrieved.
- The search covered arXiv broadly but did not query OpenReview or the ACL Anthology separately, so workshop papers indexed only there could have been missed.
- Many cited papers are 2026 preprints that have not been peer-reviewed. Some report results for model names that cannot be checked independently (e.g., RAISE's frontier-model labels). Treat their numbers as provisional.
- Classic baselines already known to the user (Sclar, He, Tam, RuLES, IHEval, AgentDojo, τ-bench, Logic-LM, LINC) were not re-verified here.
