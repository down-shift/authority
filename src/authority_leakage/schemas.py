"""Minimal, explicit records persisted as JSONL."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Example:
    example_id: str
    experiment: str
    template_id: str
    seed: int
    condition: str
    pair_id: str
    messages: list[Message]
    correct_answer: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Example":
        return cls(**{**data, "messages": [Message(**m) for m in data["messages"]]})


def validate_examples(examples: list[Example]) -> None:
    ids = [e.example_id for e in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example_id")
    for e in examples:
        if not e.messages or not e.pair_id or e.experiment not in {"epistemic", "delegation", "scope", "scope_pilot", "instrument_validation", "instrument_validation_v2", "two_scope_pilot"}:
            raise ValueError(f"Invalid example {e.example_id}")
