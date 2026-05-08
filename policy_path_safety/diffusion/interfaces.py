from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class CandidateProposal:
    state: Tuple[str, ...]
    family: str
    score: float
    filled_positions: Tuple[int, ...]
    source_text: str


@dataclass
class DenoisingStepLog:
    step: int
    state_before: str
    num_candidates: int
    num_allowed: int
    selected_family: str
    selected_score: float
    state_after: str


@dataclass
class DenoisingTrace:
    initial_state: str
    final_state: str
    steps: List[DenoisingStepLog]
