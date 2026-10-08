# Gate 1 spot-check, round 1 (human review by Jerzy, 2026-10-08)
- status: done
- result: **G1 not passed.** All 20 sampled worlds have at least one rendering flagged as ambiguous. The rule (RESEARCH_PLAN §3 Gate 1) needs zero.
- packet: `scripts/authinv_spotcheck.py`, seed 20261012, benchmark dataset sha256 `255c9ba2…` (`docs/experiments/p1.11-benchmark-audit.md`)

## Flag counts by rendering (worlds flagged, of 20)

| rendering | flagged |
|---|---:|
| owner_statement | 20 |
| nl_statement | 15 |
| json_policy | 14 |
| table | 9 |
| executable (Cedar) | 0 |
| rego | 0 |

## Themes

1. **"Only when"** (natural language, owner): it reads as a necessary
   condition but not a sufficient one. Worlds 1, 3, 5, 10, 11, 14, 16, 19.
2. **"Decides whether"** (owner): it reads as decision authority or choice,
   not as permission. All 20 worlds. "Never decides" reads differently from
   a prohibition (worlds 11, 14, 17).
3. **Rule and condition combination unstated in JSON** (AND/OR over a rule's
   conditions; how separate grants combine). Worlds 1, 3, 7, 8, 11, 12, 14,
   15, 16, 17, 19, 20.
4. **Deny precedence unstated in the table and JSON**. Worlds 11, 14, 16, 17.
5. **Group membership**: "anyone in Team X" doesn't clearly include X itself,
   which Cedar's `in` does. Worlds 12, 13, 15, 16, 17, 18, 20.
6. **Source artifacts**: a redundant conditional rule after an unconditional
   one reads as a restriction (world 7), and a duplicate condition (world 14).

## Packet limitation (agent's note)

The packet showed the renderings without the shared prompt block that every
benchmark prompt includes. That block (prompt-v1) states the combining rule:
"allowed only if at least one rule allows it and no rule forbids it; a
forbidding rule always wins".

That addresses theme 4 and the rule-combination half of theme 3 *in the
prompts*. It doesn't address condition conjunction (theme 3), themes 1, 2, 5,
or 6. Round 2 must show reviewers exactly what the model sees.

## Per-world verdicts

| World | Flagged renderings | Note |
|---|---|---|
| 1 | Natural language, Decision owner, JSON policy | "Only when" may state necessary conditions without guaranteeing permission. "Decides whether" suggests decision authority. JSON leaves AND/OR combination unspecified. |
| 2 | Decision owner | "Anyone decides whether to Search" can describe choice or authority rather than permission. |
| 3 | Natural language, Decision owner, JSON policy | "Only when" may leave additional requirements open. Decision authority differs from permission. JSON leaves conjunction of the four conditions unspecified. |
| 4 | Decision owner | "Decides whether to login" can describe the user's choice or authority rather than permission. |
| 5 | Natural language, Decision owner | Consent can read as necessary without being sufficient. "Decides whether" can imply decision authority. |
| 6 | Decision owner | "Decides whether to report" can describe choice or authority rather than permission. |
| 7 | Natural language, Decision owner, JSON policy | The conditional second rule may read as restricting the unconditional first rule, although it is formally redundant. Decision owner implies authority; JSON leaves rule combination unspecified. |
| 8 | Decision owner, JSON policy | Decision owner may imply the named role controls authorization. JSON leaves combination of the independent grants unspecified. |
| 9 | Decision owner | "Anyone decides whether" can assign decision authority rather than grant permission. |
| 10 | Natural language, Decision owner | "Only when" may express necessity without sufficiency. "Decides whether" can assign decision authority. |
| 11 | Natural language, Decision owner, Permission table, JSON policy | "Only when" may leave further requirements open. "Never decides" differs from prohibiting an action. Table and JSON leave deny precedence unstated for `u_d7` deleting `r_wren`. |
| 12 | Natural language, Decision owner, Permission table, JSON policy | Membership wording does not clearly include the Team itself as a principal. Decision owner implies authority. JSON also leaves rule combination unspecified. |
| 13 | Natural language, Decision owner, Permission table, JSON policy | "In Team g_onyx" does not clearly include `g_onyx` itself, as the formal rules do. Decision owner confuses authority with permission. |
| 14 | Natural language, Decision owner, Permission table, JSON policy | "Only when" may express necessity alone. "Never decides" differs from prohibition. Table/JSON leave deny precedence unstated for merging `r_lynx`; JSON also leaves conjunction unspecified. The duplicate MFA check is redundant. |
| 15 | Natural language, Decision owner, Permission table, JSON policy | Rule r2's membership wording does not clearly include `g_amber` itself. Decision owner implies authority. JSON leaves grant combination unspecified. |
| 16 | Natural language, Decision owner, Permission table, JSON policy | Membership wording does not clearly include the Role itself. "Only when" may express necessity alone. Decision owner implies authority. Table/JSON leave deny precedence unstated for Agents listing `t_maple`. |
| 17 | Natural language, Decision owner, Permission table, JSON policy | Membership wording does not clearly include `g_basalt` itself. "Never decides" differs from prohibition. Table/JSON leave deny precedence unstated for Agents calling `t_maple`. |
| 18 | Natural language, Decision owner, Permission table, JSON policy | Membership wording does not clearly include `g_ivory` itself, which the formal rules permit to configure/list Tools. Decision owner implies authority. |
| 19 | Natural language, Decision owner, JSON policy | "Only when" may express necessity alone. Decision owner implies authority. JSON leaves condition conjunction and rule combination unspecified: r2 requires both conditions, while either matching grant suffices. |
| 20 | Natural language, Decision owner, Permission table, JSON policy | Rule r3's membership wording does not clearly include `g_basalt` itself. Decision owner implies authority. JSON leaves combination of independent grants unspecified. |

## Consequence

No model has been run on the authinv renderings yet: Phase 0 used the pilot's
prompts, and T2/T3 have only had smoke tests with a non-sweep model.
Revising the four flagged renderers is therefore allowed before any results.
Each revision gets a version bump, a benchmark rebuild, and a second
spot-check round. The proposed changes need Jerzy's approval.
