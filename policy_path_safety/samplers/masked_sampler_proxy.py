from __future__ import annotations

import math
import random
from typing import Sequence

from policy_path_safety.core.types import MASK, PlanResult, Scenario
from policy_path_safety.monitors.recoverability import (
    is_consistent,
    monitor_intent_satisfied,
    monitor_response_safe,
    recoverable_by_monitor,
    state_terminal_response,
    text_from_state,
    true_intent_satisfied,
    true_response_safe,
)


def weighted_choice(items, rng: random.Random):
    total = sum(max(item.weight, 0.0) for item in items)
    if total <= 0:
        return rng.choice(list(items))
    r = rng.random() * total
    acc = 0.0
    for item in items:
        acc += max(item.weight, 0.0)
        if r <= acc:
            return item
    return items[-1]


def softmax_sample(items, rng: random.Random, temperature: float):
    if not items:
        raise ValueError("empty item list")
    scores = [x[2] for x in items]
    m = max(scores)
    weights = [math.exp((s - m) / temperature) for s in scores]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += weight
        if r <= acc:
            return item
    return items[-1]


def propose_candidate_pool(scenario: Scenario, state: Sequence[str], rng: random.Random, block_size: int, num_candidates: int, prior_strength: float):
    consistent = [
        response for response in scenario.responses
        if is_consistent(state, scenario.padded_tokens(response))
    ]

    if not consistent:
        return []

    masked_positions = [i for i, tok in enumerate(state) if tok == MASK]

    if not masked_positions:
        return [(list(state), "terminal", 0.0)]

    k = max(1, min(block_size, len(masked_positions)))
    chosen_positions = rng.sample(masked_positions, k=k)

    proposals = []
    seen = set()

    for _ in range(num_candidates):
        response = weighted_choice(consistent, rng)
        full = scenario.padded_tokens(response)
        new_state = list(state)
        score = prior_strength * math.log(max(response.weight, 1e-12))

        for pos in chosen_positions:
            new_state[pos] = full[pos]

        key = tuple(new_state)
        if key not in seen:
            seen.add(key)
            proposals.append((new_state, response.family, score))

    return proposals


def true_recoverable(scenario: Scenario, state: Sequence[str]) -> bool:
    for response in scenario.responses:
        if true_response_safe(scenario, response) and is_consistent(state, scenario.padded_tokens(response)):
            return True
    return False


def add_method_bonus(scenario: Scenario, plan: PlanResult, proposals, method: str, intent_bonus: float):
    adjusted = []
    for state, family, score in proposals:
        bonus = 0.0
        if method == "retrieved_intent_aware":
            if scenario.phase == "targeted_pre" and family == "clarification" and plan.has_prereq:
                bonus += intent_bonus
            if scenario.phase == "post" and family == "restricted_commitment" and (plan.has_forbid or plan.has_prereq):
                bonus += intent_bonus
        if method == "fail_closed_if_retrieval_incomplete":
            if not plan.retrieval_complete and family == "policy_deferral":
                bonus += intent_bonus
            if plan.retrieval_complete and scenario.phase == "targeted_pre" and family == "clarification":
                bonus += intent_bonus
            if plan.retrieval_complete and scenario.phase == "post" and family == "restricted_commitment":
                bonus += intent_bonus
        adjusted.append((state, family, score + bonus))
    return adjusted


def fallback_candidate(scenario: Scenario, plan: PlanResult, monitor_restrictions, state, method: str, extractor_mode: str, block_size: int):
    valid = []

    for response in scenario.responses:
        if method == "fail_closed_if_retrieval_incomplete" and not plan.retrieval_complete:
            if response.family != "policy_deferral":
                continue
        else:
            if not monitor_response_safe(scenario, monitor_restrictions, response, extractor_mode):
                continue
            if method in {"retrieved_intent_aware", "fail_closed_if_retrieval_incomplete"}:
                if not monitor_intent_satisfied(scenario, plan, response, extractor_mode):
                    continue
        if is_consistent(state, scenario.padded_tokens(response)):
            valid.append(response)

    if not valid:
        return None

    order = {
        "policy_deferral": 0 if not plan.retrieval_complete else 3,
        "clarification": 0 if scenario.phase == "targeted_pre" else 2,
        "restricted_commitment": 0 if scenario.phase == "post" else 4,
        "safe_pass_through": 2,
    }
    valid.sort(key=lambda r: (order.get(r.family, 9), -r.weight))
    target = valid[0]
    full = scenario.padded_tokens(target)

    masked_positions = [i for i, tok in enumerate(state) if tok == MASK]
    if not masked_positions:
        return list(state), target.family, math.log(max(target.weight, 1e-12))

    new_state = list(state)
    for pos in masked_positions[: max(1, min(block_size, len(masked_positions)))]:
        new_state[pos] = full[pos]
    return new_state, target.family, math.log(max(target.weight, 1e-12))


def run_one_trajectory(scenario: Scenario, plan: PlanResult, monitor_restrictions, rng: random.Random, method: str, extractor_mode: str, sampler_config: dict):
    state = [MASK] * scenario.max_len
    dead_end = False
    premature_true = False
    steps = 0

    block_size = int(sampler_config["block_size"])
    num_candidates = int(sampler_config["num_candidates"])
    resample_rounds = int(sampler_config["resample_rounds"])
    temperature = float(sampler_config["temperature"])
    prior_strength = float(sampler_config["prior_strength"])
    intent_bonus = float(sampler_config["intent_bonus"])

    while MASK in state and steps < scenario.max_len + 6:
        steps += 1
        proposals = propose_candidate_pool(scenario, state, rng, block_size, num_candidates, prior_strength)

        if not proposals:
            dead_end = True
            break

        if method == "baseline_model_scored":
            chosen = softmax_sample(proposals, rng, temperature)
            state = list(chosen[0])

        elif method == "terminal_only_filter":
            allowed = []
            for candidate_state, family, score in proposals:
                terminal_response = state_terminal_response(scenario, candidate_state)
                if terminal_response is not None and not true_response_safe(scenario, terminal_response):
                    continue
                allowed.append((candidate_state, family, score))
            if not allowed:
                dead_end = True
                break
            chosen = softmax_sample(allowed, rng, temperature)
            state = list(chosen[0])

        else:
            allowed = [
                p for p in proposals
                if recoverable_by_monitor(scenario, plan, monitor_restrictions, p[0], method, extractor_mode)
            ]

            should_resample = method in {
                "recoverability_resample",
                "retrieved_intent_aware",
                "fail_closed_if_retrieval_incomplete",
            }

            round_id = 0
            while not allowed and should_resample and round_id < resample_rounds:
                round_id += 1
                proposals = propose_candidate_pool(scenario, state, rng, block_size, num_candidates, prior_strength)
                allowed = [
                    p for p in proposals
                    if recoverable_by_monitor(scenario, plan, monitor_restrictions, p[0], method, extractor_mode)
                ]

            if not allowed:
                if method == "recoverability_no_resample":
                    dead_end = True
                    break
                fallback = fallback_candidate(scenario, plan, monitor_restrictions, state, method, extractor_mode, block_size)
                if fallback is None:
                    dead_end = True
                    break
                state = list(fallback[0])
            else:
                allowed = add_method_bonus(scenario, plan, allowed, method, intent_bonus)
                chosen = softmax_sample(allowed, rng, temperature)
                state = list(chosen[0])

        if not true_recoverable(scenario, state) and MASK in state:
            premature_true = True

    response = state_terminal_response(scenario, state)
    true_safe = true_response_safe(scenario, response)
    true_violate = bool(response and not true_safe)
    true_intent = true_intent_satisfied(scenario, response)
    monitor_intent = monitor_intent_satisfied(scenario, plan, response, extractor_mode)

    if true_violate:
        premature_true = True

    return {
        "text": text_from_state(state),
        "family": response.family if response else None,
        "safe": true_safe,
        "violate": true_violate,
        "intent": true_intent,
        "monitor_intent": monitor_intent,
        "clarify": bool(response and response.family == "clarification"),
        "deferral": bool(response and response.family == "policy_deferral"),
        "dead_end": dead_end,
        "premature": premature_true,
    }
