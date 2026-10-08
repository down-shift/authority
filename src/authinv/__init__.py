"""authinv: representation invariance of authorization policies for LLM agents.

The paper's self-contained code (docs/RESEARCH_PLAN.md). Pilot packages
(authority_leakage, authorization_*, indirection*) are copied and adapted here,
never imported.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Packages authinv must never import (RESEARCH_PLAN §1 [DECISION]).
FORBIDDEN_IMPORTS = (
    "authority_leakage",
    "authorization_competence",
    "authorization_invariance",
    "indirection",
    "indirection_v2",
    "indirection_v3",
)
