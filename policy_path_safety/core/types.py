from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

MASK = "[MASK]"
PAD = "<PAD>"
OTHER = "other"


@dataclass(frozen=True)
class ResponseCandidate:
    family: str
    text: str
    gold_trace: Tuple[str, ...]
    weight: float

    def tokens(self) -> Tuple[str, ...]:
        return tuple(self.text.lower().split())


@dataclass(frozen=True)
class RestrictionSpec:
    state: str
    forbidden: Tuple[str, ...]
    prereq: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class PolicyClause:
    clause_id: str
    domain: str
    clause_type: str
    text: str
    condition_state: Optional[str]
    target_predicate: Optional[str]
    prereq_label: Optional[str]


@dataclass
class Scenario:
    name: str
    domain: str
    phase: str
    target_predicate: str
    prereq_label: str
    context: str
    latent_states: Tuple[str, ...]
    true_restrictions: Dict[str, RestrictionSpec]
    responses: Tuple[ResponseCandidate, ...]

    @property
    def max_len(self) -> int:
        return max(len(response.tokens()) for response in self.responses)

    def padded_tokens(self, response: ResponseCandidate) -> Tuple[str, ...]:
        toks = list(response.tokens())
        toks.extend([PAD] * (self.max_len - len(toks)))
        return tuple(toks)


@dataclass
class PlanResult:
    plan_name: str
    drop_rate: float
    retrieved_ids: Tuple[str, ...]
    retrieved_texts: Tuple[str, ...]
    has_forbid: bool
    has_prereq: bool
    retrieval_complete: bool
    attempts: int
    retrieved_count: int
    stage_log: Tuple[str, ...]


@dataclass
class DynamicResult:
    seed: int
    domain: str
    extractor: str
    plan_name: str
    drop_rate: float
    method: str

    retrieval_complete: bool
    has_forbid: bool
    has_prereq: bool
    retrieval_attempts: int
    retrieved_count: int

    turn1_text: str
    turn1_family: Optional[str]
    turn1_true_safe: bool
    turn1_true_violation: bool
    turn1_true_intent: bool
    turn1_monitor_intent: bool
    turn1_clarification: bool
    turn1_deferral: bool
    turn1_dead_end: bool

    context_updated: bool

    turn2_text: Optional[str]
    turn2_family: Optional[str]
    turn2_true_safe: Optional[bool]
    turn2_true_violation: Optional[bool]
    turn2_true_intent: Optional[bool]

    end2end_safe: bool
    end2end_intent: bool
    safe_but_no_intent: bool
