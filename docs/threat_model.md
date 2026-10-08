# Threat model: choosing among equivalent renderings of a legitimate policy

Step W0.1 (2026-10-08). This file scopes the paper's "exploitable through format choice" claim,
which `docs/novelty_check.md` (Novelty Assessment) rates as *open but needs a threat model*. It says
who the attacker is, what they can and cannot do, which experiment can support each claim, and
what the paper must not claim. Citations use BibTeX keys from `paper/references.bib`; verification
status is from `docs/related_work.md`. The thesis and gates are frozen in `docs/RESEARCH_PLAN.md`;
nothing here changes them.

## 1. System model

- **Agent.** An LLM agent gets an authorization policy in its context (system prompt or tool
  preamble): a tool allowlist, resource permissions, or a decision-owner assignment. It decides
  requests (T1/T2) or acts through tools (T3, an MCP-style sandbox).
- **Ground truth.** An authorization engine (Cedar, plus OPA/Rego) and a reference evaluator over
  the canonical policy object define allow/deny for every request (P1.1, P1.3).
- **Renderings.** A policy is shown in one of five renderings (NL statement, decision-owner
  statement, Markdown table, JSON policy, Cedar), with Rego as a sixth (P1.5). A world ships only
  if every rendering decodes to the canonical object and the engine agrees with the reference on
  every request (P1.4). "Equivalent" in this paper always means *engine-certified* equivalent.
- **No external enforcement in the threat setting.** The model's own decision gates the action.
  An enforcement proxy that filters tools before the model sees them makes UIR 0% by
  construction (`uppala2026prompts`). We use it as an upper-bound baseline (P3.2), not as the
  deployment under attack.

## 2. Threat model A (in scope): the rendering chooser

**Who.** A party who controls *how* a legitimate policy is written but not *what* it allows.
Examples:
- a tenant who writes their own MCP tool allowlist on a shared agent platform;
- a contributor whose policy file is synced from a repository into the agent's context, e.g. a
  PR that "converts the allowlist to a table" and passes a semantic-equivalence check in CI;
- whoever owns a format-conversion step in a pipeline (NL → JSON, JSON → Cedar).

**Assumptions.**
1. The deployment checks a policy's *semantics*: engine equivalence, a diff review, or an approval
   step. It does not check how the model *behaves* under each format. This is the gap we measure.
2. The policy text is in a trusted channel and is not adversarial instruction text. It contains no
   imperatives beyond the policy itself, and no role markers or fake system text.
3. The model, decoding (temperature 0) and task are fixed. Only the rendering varies.

**Capabilities.** The attacker can choose any rendering from the certified-equivalent set, either
once for all policies or separately per policy. With query access to the deployed model, they can
find the worst rendering for that model.

**Cannot.** They cannot change the policy's semantics; the engine certifies equivalence, so a
semantic change is detected. They cannot inject instructions or forge authority (that is threat
model B). They cannot change the model, the tools, or the user's task.

**Goals.** Both directions are security-relevant:
- **Fail-open (deny→allow):** the agent grants or performs what the policy denies. This harms
  confidentiality and integrity. It is the headline worst case.
- **Fail-closed (allow→deny, refusal of an authorized action):** the agent refuses what the policy
  allows. This harms availability and utility: denial of service with no injected payload,
  related to guardrail DoS (`zhou2026shield`) but obtained by formatting a legitimate policy.

**How metrics map to attackers.** An attacker who fixes one format for all policies achieves at
most the **worst-case accuracy over renderings** (min over renderings of mean accuracy). An
attacker who picks per policy succeeds on the worlds where *some* rendering errs, i.e.
1 − all-correct rate, bounded below by **per-world rendering disagreement**. Our rendering set is a
small, fixed subset of all equivalent texts. A real attacker can also paraphrase or reorder within
a format, so our numbers are a **lower bound** on adversarial success, not an estimate of it.

## 3. Threat model B (out of scope, contrast)

- **Injected fake policy text that manufactures authority.** HiddenLayer's "Policy Puppetry"
  (industry blog, April 2025, not peer-reviewed, **not verified and not in `references.bib`**)
  disguises attack prompts as XML/INI/JSON policy files to bypass system prompts. Prompt Injection
  as Role Confusion (`ye2026roleconfusion`) finds that text which imitates a privileged role is
  represented as privileged. Both concern *untrusted* text gaining force, a violation of the
  instruction hierarchy (`wallace2024instruction`). Our case is the mirror image: a *trusted,
  legitimate* policy losing (or over-gaining) force when re-rendered, with no injected content.
- **Representation attacks with the policy fixed.** Threat-Preserving Representation Sensitivity
  (`karamchandani2026threat`) holds the security policy fixed and varies the *tool/threat*
  representation (e.g. tool names); ASB attack success moves by 11.67–13.21 pp. We hold the task
  and threat fixed and vary the *policy* representation. The two axes are complementary.
- **Prompt-injection benchmarks** (`debenedetti2024agentdojo`, `zhang2024asb`) assume an attacker
  who injects content through tools. That is out of scope here, and T3 contains no injections.

## 4. Non-adversarial framing: benign format drift

The same failures need no attacker. Teams write policies in whatever format they use (prose in a
README, a table in a wiki, JSON in config, Cedar in a policy store), and converters move policies
between them. A deployment that accepts "any equivalent format" gets the worst rendering's
behaviour on some of its policies. Worst-case over renderings is therefore the right reliability
metric even with no adversary. This framing is weaker than threat model A (drift is not chosen to
be worst), and it is the one the evidence most directly supports.

## 5. Claim → evidence map

Status as of 2026-10-08. Phase 0 used the pilot's **toy single-scope worlds** (180 worlds; two
actors, one field, two candidates). Their renderings are equivalent by construction, not
engine-certified, and Phase 0 is T2 only. Sources: `docs/experiments/p0.4-phase0-kill-test.md`,
`docs/experiments/p0.5-phase0-addendum.md`.

| # | Intended claim | Experiment / step that can support it | Current evidence |
|---|---|---|---|
| C1 | Renderings are semantically identical (the attacker cannot change semantics) | P1.2 renderers + decoders, P1.3 Cedar, P1.4 equivalence proofs (`equivalence.jsonl`), P1.5 Rego, G1 human spot-check | None yet; Phase 1 not built. Toy worlds: equivalent by construction only. |
| C2 | Model decisions depend on rendering (T2 worst-case gap, disagreement) | P2.4–P2.6 sweeps; P2.9 tables; Holm across models × renderings | Toy worlds: Qwen3-8B gap 8.6 pp [5.6, 11.9], disagreement 13.9%. ≥27B dense ≤1.1 pp (Qwen3-32B, Gemma-3-27B), 0.6 pp (Qwen3.8-27B), 0.0 (Gemma-4-31B, Llama-3.3-70B W8A8). gpt-oss-120b 7.5 pp [5.0, 10.3] on NL. Gate-0 rule: FAIL; G0 waived. |
| C3 | Format choice raises **deny→allow** (fail-open) | P2.2 flip metric on engine-labelled requests; P2.4–P2.6 per-rendering deny→allow rates | Toy: Qwen3-8B all 50 flips deny→allow, none allow→deny (worst rendering NL). ≥27B: 4 / 8 / 0 deny→allow instances (Qwen3-32B / Gemma-3-27B / Llama-70B). |
| C4 | Format choice causes **fail-closed** refusals of authorized actions (availability) | P2.0 strict-v2 parser; P2.4–P2.6 allow→deny and refusal categories; P2.8 authorized-action refusal rate | Toy: gpt-oss-120b NL errors are mostly refusals (26 of 32 strict-v1 parse failures). **Post-hoc, not preregistered**; to be re-scored under strict-v2 and tested prospectively in Phase 2. |
| C5 | The effect reaches agent **actions**: rendering raises T3 UIR | P2.7 sandbox; P2.8 T3 runs (UIR, refusal, completion), T2→T3 transfer | **None.** No T3 run exists. |
| C6 | A per-policy chooser does better than a fixed-format one | P2.9 all-correct rate and disagreement vs worst-case accuracy | Toy only (e.g. Qwen3-8B all-correct 86.1%). |
| C7 | Scale does not remove the effect on realistic policies | P2.4–P2.6 on the Phase-1 benchmark; P2.9 gap vs parameter count | Toy worlds point the other way for dense models (near ceiling at ≥27B). Open. |
| C8 | Mitigations close the attack surface | P3.1 IR canonicalization, P3.2 baselines (incl. enforcement proxy), P3.4 gap closure on T2 and T3, G3; P4.x SFT with Cedar held out | None. |
| C9 | Where renderings diverge internally | P5.1–P5.4 probes, CKA, and the causal check | None. Without P5.4 there is no causal claim. |

## 6. What the paper must not claim

1. **That format choice is a practical exploit** without T3 evidence (C5). Until P2.8 shows a UIR
   difference, say "decision-level vulnerability" and "lower bound on a rendering chooser's
   success", not "attack".
2. **That large models are invariant in general.** The ≥27B invariance holds on toy single-scope
   worlds that sit near ceiling. It says nothing about multi-rule, deny-overrides or conditioned
   policies, and gpt-oss-120b is already an exception.
3. **That errors are always fail-open.** Direction depends on the model: fail-open for Qwen3-8B,
   fail-closed for gpt-oss-120b. The gpt-oss direction is a post-hoc reading until strict-v2.
4. **Mechanism from behaviour.** Accuracy, flips and log-prob margins do not show why models
   differ. Internal claims need P5, and causal claims need P5.4 (CLAUDE.md invariant 9).
5. **Generalisation from the toy worlds**, or from our five/six renderings to "all formats". The
   rendering set is a sample. Worst-case over it bounds the attacker from below only.
6. **That renderings are equivalent** for any world not certified by the engine (P1.4). Phase-0
   worlds are not certified.
7. **That an attacker can change what a policy allows**, or that this paper studies injection.
   Threat model B is cited for contrast, not measured.
8. **Unverified prior-work figures** flagged in `docs/related_work.md`:
   - PolicySummarizer (`vatsa2025neurosymbolic`): its "587 Quacky policies", "59.1–67.9%
     accuracy", "90.7–96.0% consistency" and "39–44% correct" are in no abstract version. The
     abstract says 546 AWS + 100 Azure + 100 GCP and "39% → 93%".
   - PolicyGuard DLP (`park2026policyguard`) is **withdrawn**. Cite it only as motivating
     evidence. Its "6.1 pp" NL-vs-JSON gap and "h=0.260" are not in the abstract.
   - Ghost in the Context (`santosgrueiro2026ghost`): the "Policy IR" and its 512-token budget are
     not in any abstract, and the current title is "Policy-Carriage Integrity in LLM Agents".
   - Policy Puppetry and Sockpuppeting are blogs with no `references.bib` entry. If the paper
     cites them, label them as non-peer-reviewed industry reports and add a verified entry first.
   - Any other *not in abstract* number needs a full-text read before it appears in `paper/`.
9. **Pooled numbers** across models, tiers or renderings without the per-rendering values
   (CLAUDE.md invariant 7), or any claim not traceable to a committed `docs/experiments/` report.
