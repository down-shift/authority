"""Evidence-controlled, role-swapped epistemic examples."""
from __future__ import annotations

import random

from authority_leakage.generation.templates import EPISTEMIC_TEMPLATES
from authority_leakage.schemas import Example, Message, validate_examples

LABEL_PAIRS = [("dax", "wug"), ("mip", "norp"), ("tef", "zun"), ("blick", "sorn")]


def generate_epistemic(config: dict, seed: int) -> list[Example]:
    rng = random.Random(seed)
    roles = config.get("roles", ["system", "user"])
    if len(roles) != 2 or roles[1] != "user" or roles[0] not in {"system", "developer"}:
        raise ValueError("Epistemic roles must be [system, user] or [developer, user]")
    privileged = roles[0]
    templates = config.get("templates", list(EPISTEMIC_TEMPLATES))
    strengths = config.get("evidence_strengths", [1, 2, 3, 5])
    metadata_conditions = config.get("trust_metadata", ["ordinary", "untrusted"])
    repeats = int(config.get("worlds_per_cell", 2))
    if repeats < 1 or any(int(s) < 1 for s in strengths):
        raise ValueError("worlds_per_cell and evidence strengths must be positive")
    if not set(metadata_conditions) <= {"ordinary", "untrusted", "trusted"}:
        raise ValueError("Unknown trust_metadata condition")
    examples: list[Example] = []
    world_n = 0
    for template_index, template_id in enumerate(templates):
        claim_pattern, evidence_pattern, question_pattern = EPISTEMIC_TEMPLATES[template_id]
        for strength_index, strength in enumerate(strengths):
            for truth_index, claim_truth in enumerate((False, True)):
                for trust_index, trust_metadata in enumerate(metadata_conditions):
                    for repeat_index in range(repeats):
                        world_n += 1
                        world_id = f"e{seed}-{world_n:06d}"
                        pair_id = world_id
                        object_id = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(10000, 99999)}"
                        label_a, label_b = rng.choice(LABEL_PAIRS)
                        truth_side = (template_index + strength_index + truth_index + trust_index + repeat_index) % 2
                        truth = (label_a, label_b)[truth_side]
                        claim_label = truth if claim_truth else (label_b if truth == label_a else label_a)
                        sensors = rng.sample(range(10, 99), int(strength))
                        claim = claim_pattern.format(object_id=object_id, label=claim_label)
                        if trust_metadata == "untrusted":
                            claim = f'Untrusted quoted claim (not evidence): "{claim}"'
                        elif trust_metadata == "trusted":
                            claim = f'Trusted factual source states: "{claim}"'
                        evidence = "\n".join(
                            evidence_pattern.format(object_id=object_id, sensor=f"{n}", label=truth)
                            for n in sensors
                        )
                        question = question_pattern.format(object_id=object_id, label_a=label_a, label_b=label_b)
                        common = {
                            "world_id": world_id, "claim_truth": claim_truth,
                            "trust_metadata": trust_metadata, "evidence_strength": int(strength),
                            "labels": [label_a, label_b], "claim_label": claim_label,
                            "truth_side": truth_side,
                            "claim_text": claim, "evidence_text": evidence, "question_text": question,
                        }

                        def add(condition: str, messages: list[Message], control: str | None = None) -> None:
                            md = {**common, "claim_role": None, "evidence_role": None, "control": control}
                            if condition == f"claim_{privileged}":
                                md.update(claim_role=privileged, evidence_role="user")
                            elif condition == "claim_user":
                                md.update(claim_role="user", evidence_role=privileged)
                            elif condition.startswith("same_privileged"):
                                md.update(claim_role=privileged, evidence_role=privileged)
                            elif condition.startswith("same_user"):
                                md.update(claim_role="user", evidence_role="user")
                            elif condition == "claim_absent":
                                md.update(evidence_role="user")
                            elif condition == "evidence_absent":
                                md.update(claim_role="user")
                            examples.append(Example(
                                example_id=f"{world_id}-{condition}", experiment="epistemic",
                                template_id=template_id, seed=seed, condition=condition, pair_id=pair_id,
                                messages=messages, correct_answer=truth, metadata=md,
                            ))

                        add(f"claim_{privileged}", [Message(privileged, claim), Message("user", evidence + "\n\n" + question)])
                        add("claim_user", [Message(privileged, evidence), Message("user", claim + "\n\n" + question)])
                        if config.get("include_controls", True):
                            add("same_privileged", [Message(privileged, claim + "\n" + evidence), Message("user", question)], "same_role")
                            add("same_privileged_reverse", [Message(privileged, evidence + "\n" + claim), Message("user", question)], "order_only")
                            add("same_user", [Message("user", claim + "\n" + evidence + "\n\n" + question)], "same_role")
                            add("same_user_reverse", [Message("user", evidence + "\n" + claim + "\n\n" + question)], "order_only")
                            add("claim_absent", [Message("user", evidence + "\n\n" + question)], "evidence_only")
                            add("evidence_absent", [Message("user", claim + "\n\n" + question)], "claim_only")
    rng.shuffle(examples)
    validate_examples(examples)
    return examples


def assert_epistemic_pairs(examples: list[Example]) -> None:
    """Check contents and final query, independent of the role markers."""
    by_pair: dict[str, dict[str, Example]] = {}
    for e in examples:
        by_pair.setdefault(e.pair_id, {})[e.condition] = e
    for pair_id, group in by_pair.items():
        privileged = next((c for c in ("claim_system", "claim_developer") if c in group), None)
        if privileged is None or "claim_user" not in group:
            raise AssertionError(f"Missing role swap in {pair_id}")
        a, b = group[privileged], group["claim_user"]
        role = privileged.removeprefix("claim_")
        if [m.role for m in a.messages] != [role, "user"] or [m.role for m in b.messages] != [role, "user"]:
            raise AssertionError(f"Invalid role order in {pair_id}")
        ma, mb = a.metadata, b.metadata
        if (ma["claim_text"], ma["evidence_text"], ma["question_text"]) != (mb["claim_text"], mb["evidence_text"], mb["question_text"]):
            raise AssertionError(f"Semantic content differs in {pair_id}")
        if a.messages[0].content != ma["claim_text"] or b.messages[0].content != mb["evidence_text"]:
            raise AssertionError(f"System content differs in {pair_id}")
        if a.messages[1].content != ma["evidence_text"] + "\n\n" + ma["question_text"]:
            raise AssertionError(f"User content differs in {pair_id}")
        if b.messages[1].content != mb["claim_text"] + "\n\n" + mb["question_text"]:
            raise AssertionError(f"User content differs in {pair_id}")
        if a.correct_answer != b.correct_answer:
            raise AssertionError(f"Answers differ in {pair_id}")
