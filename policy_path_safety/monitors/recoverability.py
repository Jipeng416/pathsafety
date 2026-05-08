from __future__ import annotations

from typing import Dict, Optional, Sequence

from policy_path_safety.core.types import MASK, PAD, PlanResult, ResponseCandidate, RestrictionSpec, Scenario
from policy_path_safety.extractors.action_aware_extractor import observed_trace


def trace_violates(trace: Sequence[str], spec: RestrictionSpec) -> bool:
    trace_list = list(trace)

    for label in spec.forbidden:
        if label in trace_list:
            return True

    for required, target in spec.prereq:
        if target in trace_list:
            target_pos = trace_list.index(target)
            if required not in trace_list[:target_pos]:
                return True

    return False


def true_response_safe(scenario: Scenario, response: Optional[ResponseCandidate]) -> bool:
    if response is None:
        return False
    return all(
        not trace_violates(response.gold_trace, scenario.true_restrictions[state])
        for state in scenario.latent_states
    )


def true_intent_satisfied(scenario: Scenario, response: Optional[ResponseCandidate]) -> bool:
    if response is None:
        return False
    if scenario.phase == "targeted_pre":
        return scenario.prereq_label in response.gold_trace
    if scenario.phase == "post":
        return scenario.target_predicate in response.gold_trace
    return False


def is_consistent(pattern: Sequence[str], full: Sequence[str]) -> bool:
    return all(p == MASK or p == f for p, f in zip(pattern, full))


def fully_denoised(state: Sequence[str]) -> bool:
    return all(tok != MASK for tok in state)


def state_terminal_response(scenario: Scenario, state: Sequence[str]) -> Optional[ResponseCandidate]:
    if not fully_denoised(state):
        return None
    for response in scenario.responses:
        if tuple(state) == scenario.padded_tokens(response):
            return response
    return None


def text_from_state(state: Sequence[str]) -> str:
    return " ".join(tok for tok in state if tok != PAD)


def build_monitor_from_plan(scenario: Scenario, plan: PlanResult, corpus) -> Dict[str, RestrictionSpec]:
    by_id = {clause.clause_id: clause for clause in corpus}
    forbidden_by_state = {state: [] for state in scenario.latent_states}
    prereq_by_state = {state: [] for state in scenario.latent_states}

    for cid in plan.retrieved_ids:
        clause = by_id[cid]
        if clause.domain != scenario.domain:
            continue
        if clause.clause_type == "forbid" and clause.condition_state and clause.target_predicate:
            for state in scenario.latent_states:
                if state == clause.condition_state:
                    forbidden_by_state[state].append(clause.target_predicate)
        if clause.clause_type == "prerequisite" and clause.prereq_label and clause.target_predicate:
            if scenario.phase == "targeted_pre":
                for state in scenario.latent_states:
                    prereq_by_state[state].append((clause.prereq_label, clause.target_predicate))

    return {
        state: RestrictionSpec(
            state=state,
            forbidden=tuple(forbidden_by_state[state]),
            prereq=tuple(prereq_by_state[state]),
        )
        for state in scenario.latent_states
    }


def monitor_response_safe(scenario: Scenario, monitor_restrictions: Dict[str, RestrictionSpec], response: Optional[ResponseCandidate], extractor_mode: str) -> bool:
    if response is None:
        return False
    trace = observed_trace(scenario.domain, response.text, extractor_mode)
    return all(
        not trace_violates(trace, monitor_restrictions[state])
        for state in scenario.latent_states
    )


def monitor_intent_satisfied(scenario: Scenario, plan: PlanResult, response: Optional[ResponseCandidate], extractor_mode: str) -> bool:
    if response is None:
        return False
    trace = observed_trace(scenario.domain, response.text, extractor_mode)
    if scenario.phase == "targeted_pre":
        return plan.has_prereq and scenario.prereq_label in trace
    if scenario.phase == "post":
        return (plan.has_forbid or plan.has_prereq) and scenario.target_predicate in trace
    return False


def recoverable_by_monitor(scenario: Scenario, plan: PlanResult, monitor_restrictions: Dict[str, RestrictionSpec], state: Sequence[str], method: str, extractor_mode: str) -> bool:
    intent_aware = method in {"retrieved_intent_aware", "fail_closed_if_retrieval_incomplete"}
    fail_closed = method == "fail_closed_if_retrieval_incomplete"

    for response in scenario.responses:
        if fail_closed and not plan.retrieval_complete:
            if response.family != "policy_deferral":
                continue
        else:
            if not monitor_response_safe(scenario, monitor_restrictions, response, extractor_mode):
                continue
            if intent_aware and not monitor_intent_satisfied(scenario, plan, response, extractor_mode):
                continue
        if is_consistent(state, scenario.padded_tokens(response)):
            return True
    return False
