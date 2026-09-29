from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from authority_leakage.schemas import Message


@dataclass
class Generation:
    text: str
    rendered_prompt: str
    generated_token_logprobs: list[float] | None


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], max_new_tokens: int) -> Generation: ...

    @abstractmethod
    def score_candidates(self, messages: list[Message], candidates: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def candidate_token_ids(self, candidates: list[str]) -> dict[str, list[int]]: ...

    @abstractmethod
    def provenance(self) -> dict: ...
