from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from policy_path_safety.core.types import MASK, DynamicResult
from policy_path_safety.data.scenarios import build_policy_corpus, make_scenarios
from policy_path_safety.diffusion.interfaces import DenoisingStepLog
from policy_path_safety.diffusion.mlm_denoising import HFMaskedLMDenoisingSampler
from policy_path_safety.monitors.recoverability import (
    build_monitor_from_plan,
    monitor_intent_satisfied,
    recoverable_by_monitor,
    state_terminal_response,
    text_from_state,
    true_intent_satisfied,
    true_response_safe,
)
from policy_path_safety.retrieval.adaptive_retrieval import RetrievalEngine, merge_plans, run_retrieval_plan
from policy_path_safety.samplers.masked_sampler_proxy import add_method_bonus, fallback_candidate, softmax_sample


@dataclass
class Phase2Trajectory:
    seed: int
    domain: str
    extractor: str
    plan_name: str
    drop_rate: float
    method: str

    model_name: str
    retrieval_complete: bool
    has_forbid: bool
    has_prereq: bool
    retrieval_attempts: int
    retrieved_count: int

    terminal_text: str
    terminal_family: Optional[str]
    terminal_safe: bool
    terminal_violation: bool
    terminal_intent: bool
    monitor_intent: bool

    clarification: bool
    deferral: bool
    dead_end: bool
    premature_true: bool

    num_steps: int
    avg_candidates: float
    avg_allowed: float
    rejection_rate: float


@dataclass
class Phase2DynamicResult:
    seed: int
    domain: str
    extractor: str
    plan_name: str
    drop_rate: float
    method: str
    model_name: str

    retrieval_complete: bool
    has_forbid: bool
    has_prereq: bool
    retrieval_attempts: int
    retrieved_count: int

    turn1_text: str
    turn1_family: Optional[str]
    turn1_safe: bool
    turn1_violation: bool
    turn1_intent: bool
    turn1_clarification: bool
    turn1_deferral: bool
    turn1_dead_end: bool
    turn1_premature: bool

    context_updated: bool

    turn2_text: Optional[str]
    turn2_family: Optional[str]
    turn2_safe: Optional[bool]
    turn2_violation: Optional[bool]
    turn2_intent: Optional[bool]

    end2end_safe: bool
    end2end_intent: bool
    safe_but_no_intent: bool

    avg_candidates: float
    avg_allowed: float
    rejection_rate: float


def _softmax_sample_candidates(candidates, rng: random.Random, temperature: float):
    tuples = [(list(c.state), c.family, c.score) for c in candidates]
    state, family, score = softmax_sample(tuples, rng, temperature)
    return tuple(state), family, score


def _candidate_allowed(
    scenario,
    plan,
    monitor_restrictions,
    candidate_state,
    method: str,
    extractor_mode: str,
):
    if method == "base_mlm_diffusion":
        return True

    if method == "terminal_only_filter":
        terminal_response = state_terminal_response(scenario, candidate_state)
        if terminal_response is not None and not true_response_safe(scenario, terminal_response):
            return False
        return True

    if method == "recoverability_guided":
        return recoverable_by_monitor(
            scenario=scenario,
            plan=plan,
            monitor_restrictions=monitor_restrictions,
            state=candidate_state,
            method="recoverability_resample",
            extractor_mode=extractor_mode,
        )

    return recoverable_by_monitor(
        scenario=scenario,
        plan=plan,
        monitor_restrictions=monitor_restrictions,
        state=candidate_state,
        method=method,
        extractor_mode=extractor_mode,
    )


def _true_recoverable(scenario, state):
    from policy_path_safety.monitors.recoverability import is_consistent

    for response in scenario.responses:
        if true_response_safe(scenario, response) and is_consistent(state, scenario.padded_tokens(response)):
            return True
    return False


def run_phase2_trajectory(
    scenario,
    plan,
    monitor_restrictions,
    sampler: HFMaskedLMDenoisingSampler,
    rng: random.Random,
    method: str,
    extractor_mode: str,
    sampler_config: Dict,
    seed: int,
):
    state = tuple([MASK] * scenario.max_len)
    dead_end = False
    premature_true = False
    logs: List[DenoisingStepLog] = []
    total_candidates = 0
    total_allowed = 0

    block_size = int(sampler_config["block_size"])
    num_candidates = int(sampler_config["num_candidates"])
    resample_rounds = int(sampler_config["resample_rounds"])
    temperature = float(sampler_config["temperature"])
    prior_strength = float(sampler_config["prior_strength"])
    model_score_weight = float(sampler_config["model_score_weight"])
    intent_bonus = float(sampler_config["intent_bonus"])

    steps = 0

    while MASK in state and steps < scenario.max_len + 6:
        steps += 1
        state_before = text_from_state(state)

        proposals = sampler.propose(
            scenario=scenario,
            state=state,
            rng=rng,
            block_size=block_size,
            num_candidates=num_candidates,
            prior_strength=prior_strength,
            model_score_weight=model_score_weight,
        )

        total_candidates += len(proposals)

        if not proposals:
            dead_end = True
            break

        allowed = [
            p for p in proposals
            if _candidate_allowed(
                scenario=scenario,
                plan=plan,
                monitor_restrictions=monitor_restrictions,
                candidate_state=p.state,
                method=method,
                extractor_mode=extractor_mode,
            )
        ]

        total_allowed += len(allowed)

        round_id = 0
        should_resample = method in {
            "recoverability_guided",
            "retrieved_intent_aware",
            "fail_closed_if_retrieval_incomplete",
        }

        while not allowed and should_resample and round_id < resample_rounds:
            round_id += 1

            proposals = sampler.propose(
                scenario=scenario,
                state=state,
                rng=rng,
                block_size=block_size,
                num_candidates=num_candidates,
                prior_strength=prior_strength,
                model_score_weight=model_score_weight,
            )
            total_candidates += len(proposals)

            allowed = [
                p for p in proposals
                if _candidate_allowed(
                    scenario=scenario,
                    plan=plan,
                    monitor_restrictions=monitor_restrictions,
                    candidate_state=p.state,
                    method=method,
                    extractor_mode=extractor_mode,
                )
            ]
            total_allowed += len(allowed)

        if not allowed:
            if method == "recoverability_no_resample":
                dead_end = True
                break

            fallback_method = "recoverability_resample" if method == "recoverability_guided" else method
            fallback = fallback_candidate(
                scenario=scenario,
                plan=plan,
                monitor_restrictions=monitor_restrictions,
                state=state,
                method=fallback_method,
                extractor_mode=extractor_mode,
                block_size=block_size,
            )

            if fallback is None:
                dead_end = True
                break

            selected_state = tuple(fallback[0])
            selected_family = fallback[1]
            selected_score = float(fallback[2])

        else:
            proposal_tuples = [(list(p.state), p.family, p.score) for p in allowed]
            bonus_method = "recoverability_resample" if method == "recoverability_guided" else method
            proposal_tuples = add_method_bonus(scenario, plan, proposal_tuples, bonus_method, intent_bonus)
            selected_state_list, selected_family, selected_score = softmax_sample(proposal_tuples, rng, temperature)
            selected_state = tuple(selected_state_list)

        state = selected_state

        if not _true_recoverable(scenario, state) and MASK in state:
            premature_true = True

        logs.append(
            DenoisingStepLog(
                step=steps,
                state_before=state_before,
                num_candidates=len(proposals),
                num_allowed=len(allowed),
                selected_family=str(selected_family),
                selected_score=float(selected_score),
                state_after=text_from_state(state),
            )
        )

    response = state_terminal_response(scenario, state)

    terminal_safe = true_response_safe(scenario, response)
    terminal_violation = bool(response and not terminal_safe)
    terminal_intent = true_intent_satisfied(scenario, response)
    monitor_intent = monitor_intent_satisfied(scenario, plan, response, extractor_mode)

    if terminal_violation:
        premature_true = True

    avg_candidates = total_candidates / max(1, len(logs))
    avg_allowed = total_allowed / max(1, len(logs))
    rejection_rate = 1.0 - (total_allowed / total_candidates) if total_candidates else 0.0

    return Phase2Trajectory(
        seed=seed,
        domain=scenario.domain,
        extractor=extractor_mode,
        plan_name=plan.plan_name,
        drop_rate=plan.drop_rate,
        method=method,
        model_name=sampler.model_name,
        retrieval_complete=plan.retrieval_complete,
        has_forbid=plan.has_forbid,
        has_prereq=plan.has_prereq,
        retrieval_attempts=plan.attempts,
        retrieved_count=plan.retrieved_count,
        terminal_text=text_from_state(state),
        terminal_family=response.family if response else None,
        terminal_safe=terminal_safe,
        terminal_violation=terminal_violation,
        terminal_intent=terminal_intent,
        monitor_intent=monitor_intent,
        clarification=bool(response and response.family == "clarification"),
        deferral=bool(response and response.family == "policy_deferral"),
        dead_end=dead_end,
        premature_true=premature_true,
        num_steps=len(logs),
        avg_candidates=avg_candidates,
        avg_allowed=avg_allowed,
        rejection_rate=rejection_rate,
    ), logs


def run_phase2_dynamic_dialogue(
    domain,
    pre,
    post,
    corpus,
    engine,
    sampler,
    plan_name,
    drop_rate,
    method,
    extractor_mode,
    rng,
    seed,
    sampler_config,
):
    pre_plan = run_retrieval_plan(pre, corpus, engine, plan_name, drop_rate, rng)
    pre_monitor = build_monitor_from_plan(pre, pre_plan, corpus)

    turn1, logs1 = run_phase2_trajectory(
        scenario=pre,
        plan=pre_plan,
        monitor_restrictions=pre_monitor,
        sampler=sampler,
        rng=rng,
        method=method,
        extractor_mode=extractor_mode,
        sampler_config=sampler_config,
        seed=seed,
    )

    context_updated = False
    turn2 = None
    logs2 = []
    total_attempts = pre_plan.attempts
    total_retrieved = pre_plan.retrieved_count

    if turn1.terminal_safe and turn1.clarification and not turn1.dead_end:
        context_updated = True
        post_plan_raw = run_retrieval_plan(post, corpus, engine, plan_name, drop_rate, rng)
        post_plan = merge_plans(pre_plan, post_plan_raw, post, corpus)
        post_monitor = build_monitor_from_plan(post, post_plan, corpus)

        total_attempts = post_plan.attempts
        total_retrieved = post_plan.retrieved_count

        turn2, logs2 = run_phase2_trajectory(
            scenario=post,
            plan=post_plan,
            monitor_restrictions=post_monitor,
            sampler=sampler,
            rng=rng,
            method=method,
            extractor_mode=extractor_mode,
            sampler_config=sampler_config,
            seed=seed,
        )

    if turn2 is not None:
        end2end_safe = bool(turn1.terminal_safe and turn2.terminal_safe)
        end2end_intent = bool(turn2.terminal_intent)
    else:
        end2end_safe = bool(turn1.terminal_safe)
        end2end_intent = False

    avg_candidates = turn1.avg_candidates
    avg_allowed = turn1.avg_allowed
    rejection_rate = turn1.rejection_rate

    if turn2 is not None:
        avg_candidates = (turn1.avg_candidates + turn2.avg_candidates) / 2.0
        avg_allowed = (turn1.avg_allowed + turn2.avg_allowed) / 2.0
        rejection_rate = (turn1.rejection_rate + turn2.rejection_rate) / 2.0

    result = Phase2DynamicResult(
        seed=seed,
        domain=domain,
        extractor=extractor_mode,
        plan_name=plan_name,
        drop_rate=drop_rate,
        method=method,
        model_name=sampler.model_name,
        retrieval_complete=pre_plan.retrieval_complete,
        has_forbid=pre_plan.has_forbid,
        has_prereq=pre_plan.has_prereq,
        retrieval_attempts=total_attempts,
        retrieved_count=total_retrieved,
        turn1_text=turn1.terminal_text,
        turn1_family=turn1.terminal_family,
        turn1_safe=turn1.terminal_safe,
        turn1_violation=turn1.terminal_violation,
        turn1_intent=turn1.terminal_intent,
        turn1_clarification=turn1.clarification,
        turn1_deferral=turn1.deferral,
        turn1_dead_end=turn1.dead_end,
        turn1_premature=turn1.premature_true,
        context_updated=context_updated,
        turn2_text=turn2.terminal_text if turn2 else None,
        turn2_family=turn2.terminal_family if turn2 else None,
        turn2_safe=turn2.terminal_safe if turn2 else None,
        turn2_violation=turn2.terminal_violation if turn2 else None,
        turn2_intent=turn2.terminal_intent if turn2 else None,
        end2end_safe=end2end_safe,
        end2end_intent=end2end_intent,
        safe_but_no_intent=end2end_safe and not end2end_intent,
        avg_candidates=avg_candidates,
        avg_allowed=avg_allowed,
        rejection_rate=rejection_rate,
    )

    trace_logs = {
        "turn1": [asdict(x) for x in logs1],
        "turn2": [asdict(x) for x in logs2],
    }

    return result, trace_logs


def summarize_phase2(results: List[Phase2DynamicResult]) -> pd.DataFrame:
    groups = defaultdict(list)
    for result in results:
        groups[(result.method, result.drop_rate, result.extractor, result.plan_name)].append(result)

    rows = []
    for (method, drop_rate, extractor, plan_name), items in sorted(groups.items()):
        n = len(items)
        rows.append({
            "method": method,
            "drop": drop_rate,
            "extractor": extractor,
            "plan": plan_name,
            "n": n,
            "retrieval_complete": sum(x.retrieval_complete for x in items) / n,
            "turn1_violate": sum(x.turn1_violation for x in items) / n,
            "turn1_premature": sum(x.turn1_premature for x in items) / n,
            "turn1_dead_end": sum(x.turn1_dead_end for x in items) / n,
            "clarify": sum(x.turn1_clarification for x in items) / n,
            "deferral": sum(x.turn1_deferral for x in items) / n,
            "context_update": sum(x.context_updated for x in items) / n,
            "end2end_safe": sum(x.end2end_safe for x in items) / n,
            "end2end_intent": sum(x.end2end_intent for x in items) / n,
            "safe_but_no_intent": sum(x.safe_but_no_intent for x in items) / n,
            "avg_candidates": sum(x.avg_candidates for x in items) / n,
            "avg_allowed": sum(x.avg_allowed for x in items) / n,
            "rejection_rate": sum(x.rejection_rate for x in items) / n,
        })

    return pd.DataFrame(rows)


def markdown_table(df, cols):
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_phase2_report(summary_df):
    cols = [
        "method",
        "drop",
        "retrieval_complete",
        "turn1_violate",
        "turn1_premature",
        "clarify",
        "deferral",
        "end2end_safe",
        "end2end_intent",
        "safe_but_no_intent",
        "avg_candidates",
        "rejection_rate",
    ]

    lines = []
    lines.append("=== PHASE 2 MASKED DENOISING INTERFACE SUMMARY ===")
    lines.append("")
    lines.append(markdown_table(summary_df, cols))
    lines.append("")
    lines.append("=== INTERPRETATION ===")
    lines.append("- Phase 2 replaces the pure random controlled proposal proxy with a real Hugging Face masked-LM denoising scorer.")
    lines.append("- The sampler exposes intermediate masked states and candidate denoising states.")
    lines.append("- The recoverability monitor filters candidate states before committing to a denoising step.")
    lines.append("- This is still not the final large MDLM or LLaDA experiment. It is the working masked denoising interface needed before that integration.")
    lines.append("- The next step is to swap HFMaskedLMDenoisingSampler with an MDLM or LLaDA backend while keeping the same monitor interface.")
    return "\n".join(lines)


def run_phase2_experiment(config: Dict):
    corpus = build_policy_corpus()
    scenarios = make_scenarios()
    engine = RetrievalEngine(corpus, stop_words=config["retrieval"]["tfidf_stop_words"])

    sampler = HFMaskedLMDenoisingSampler(
        model_name=config["model"]["model_name"],
        device=config["model"]["device"],
        max_wordpiece_length=int(config["model"]["max_wordpiece_length"]),
    )

    seeds = config["experiment"]["seeds"]
    trials_per_seed = int(config["experiment"]["trials_per_seed"])
    domains = list(config["experiment"]["domains"])
    drop_rates = [float(x) for x in config["experiment"]["drop_rates"]]
    extractor_modes = list(config["experiment"]["extractor_modes"])
    plan_names = list(config["experiment"]["plan_names"])
    methods = list(config["experiment"]["methods"])
    sampler_config = dict(config["sampler"])

    results: List[Phase2DynamicResult] = []
    examples = {}

    for seed in seeds:
        rng = random.Random(int(seed))

        for extractor in extractor_modes:
            for plan_name in plan_names:
                for drop_rate in drop_rates:
                    for domain in domains:
                        pre, post = scenarios[domain]
                        for method in methods:
                            for trial in range(trials_per_seed):
                                result, logs = run_phase2_dynamic_dialogue(
                                    domain=domain,
                                    pre=pre,
                                    post=post,
                                    corpus=corpus,
                                    engine=engine,
                                    sampler=sampler,
                                    plan_name=plan_name,
                                    drop_rate=drop_rate,
                                    method=method,
                                    extractor_mode=extractor,
                                    rng=rng,
                                    seed=int(seed),
                                    sampler_config=sampler_config,
                                )
                                results.append(result)

                                if trial == 0 and int(seed) == int(seeds[0]):
                                    key = f"{extractor}::{plan_name}::drop{drop_rate}::{domain}::{method}"
                                    examples[key] = {
                                        "result": asdict(result),
                                        "trace_logs": logs,
                                    }

    summary_df = summarize_phase2(results)
    report_text = build_phase2_report(summary_df)

    return {
        "summary_df": summary_df,
        "results": results,
        "examples": examples,
        "report_text": report_text,
    }
