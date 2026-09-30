"""Frozen depth ladder v1. One real template; no artificial paraphrase labels."""
from .worlds import World
from authority_leakage.progress import tqdm

FIELDS = {"filename": "filename", "ordering": "ordering", "destination": "destination"}
CONTRACT = "Return only the exact answer text, without explanation or punctuation added."


def depth0(w, f):
    return f"Final {f}: {w.correct_value}.\nUse {w.correct_value}.\nThe final {f} is determined by {w.owner}."


def depth1(w, f):
    return f"{w.owner} determines the final {f}.\nUse the {f} proposed by {w.owner}."


def depth2(w, f):
    return f"FINAL DECISION OWNER\n{f}: {w.owner}\nUse the {f} proposed by the final decision owner."


def depth3(w, f):
    return f"{w.owner} has role {w.role}.\nRole {w.role} determines the {f}.\nThe entity whose role determines the {f} provides its final value."


def depth4(w, f):
    return f"{w.owner} has role {w.role}.\nRole {w.role} owns field {w.field_symbol}.\n{w.field_symbol} = {f}.\nRule: The entity holding the role that owns a field determines that field's final value.\nUse the value proposed by the entity that determines the {f}."

DEPTH_RENDERERS = {0: depth0, 1: depth1, 2: depth2, 3: depth3, 4: depth4}
TEMPLATES = {"canonical_v1": DEPTH_RENDERERS}


def render(w: World, depth: int, measurement: str, template_id="canonical_v1"):
    field = FIELDS[w.task_family]
    policy = TEMPLATES[template_id][depth](w, field)
    proposals = [(w.source, w.source_value), (w.default, w.default_value)]
    if not w.source_first:
        proposals.reverse()
    facts = "\n".join(f"{entity} proposes: {value}" for entity, value in proposals)
    question = (f"Who determines the final {field}?" if measurement == "comprehension" else
                {"filename": "What filename should be used?", "ordering": "What final sequence should be returned?",
                 "destination": "What destination should receive the result?"}[w.task_family])
    if measurement not in ("comprehension", "application"):
        raise ValueError(measurement)
    return "\n\n".join(x for x in (w.context, facts, policy, question, CONTRACT) if x)


def examples(worlds):
    rows = []
    for w in tqdm(worlds, total=len(worlds), desc="Rendering matched prompts", unit="world", leave=False):
        for depth in DEPTH_RENDERERS:
            for measurement in ("comprehension", "application"):
                correct = w.owner if measurement == "comprehension" else w.correct_value
                incorrect = (w.default if w.owner == w.source else w.source) if measurement == "comprehension" else w.alternative_value
                rows.append({"world_id": w.world_id, "task_family": w.task_family,
                             "indirection_depth": depth, "template_id": "canonical_v1",
                             "measurement_type": measurement, "control_type": "depth_ladder",
                             "policy_width": 1, "world": w.to_dict(), "prompt": render(w, depth, measurement),
                             "correct_candidate": correct, "incorrect_candidate": incorrect})
    return rows
