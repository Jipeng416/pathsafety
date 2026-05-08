from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict
from typing import Dict

import pandas as pd

from policy_path_safety.core.types import DynamicResult
from policy_path_safety.data.scenarios import build_policy_corpus, make_scenarios
from policy_path_safety.extractors.action_aware_extractor import observed_trace
from policy_path_safety.monitors.recoverability import build_monitor_from_plan
from policy_path_safety.retrieval.adaptive_retrieval import RetrievalEngine, merge_plans, run_retrieval_plan
from policy_path_safety.samplers.masked_sampler_proxy import run_one_trajectory


def run_dynamic_dialogue(domain, pre, post, corpus, engine, plan_name, drop_rate, method, extractor_mode, rng, seed, sampler_config):
    pre_plan = run_retrieval_plan(pre, corpus, engine, plan_name, drop_rate, rng)
    pre_monitor = build_monitor_from_plan(pre, pre_plan, corpus)

    turn1 = run_one_trajectory(
        scenario=pre,
        plan=pre_plan,
        monitor_restrictions=pre_monitor,
        rng=rng,
        method=method,
        extractor_mode=extractor_mode,
        sampler_config=sampler_config,
    )

    context_updated = False
    turn2 = None
    total_attempts = pre_plan.attempts
    total_retrieved = pre_plan.retrieved_count

    if turn1["safe"] and turn1["clarify"] and not turn1["dead_end"]:
        context_updated = True
        post_plan_raw = run_retrieval_plan(post, corpus, engine, plan_name, drop_rate, rng)
        post_plan = merge_plans(pre_plan, post_plan_raw, post, corpus)
        post_monitor = build_monitor_from_plan(post, post_plan, corpus)
        total_attempts = post_plan.attempts
        total_retrieved = post_plan.retrieved_count

        turn2 = run_one_trajectory(
            scenario=post,
            plan=post_plan,
            monitor_restrictions=post_monitor,
            rng=rng,
            method=method,
            extractor_mode=extractor_mode,
            sampler_config=sampler_config,
        )

    if turn2 is not None:
        end2end_safe = bool(turn1["safe"] and turn2["safe"])
        end2end_intent = bool(turn2["intent"])
    else:
        end2end_safe = bool(turn1["safe"])
        end2end_intent = False

    return DynamicResult(
        seed=seed,
        domain=domain,
        extractor=extractor_mode,
        plan_name=plan_name,
        drop_rate=drop_rate,
        method=method,
        retrieval_complete=pre_plan.retrieval_complete,
        has_forbid=pre_plan.has_forbid,
        has_prereq=pre_plan.has_prereq,
        retrieval_attempts=total_attempts,
        retrieved_count=total_retrieved,
        turn1_text=str(turn1["text"]),
        turn1_family=turn1["family"],
        turn1_true_safe=bool(turn1["safe"]),
        turn1_true_violation=bool(turn1["violate"]),
        turn1_true_intent=bool(turn1["intent"]),
        turn1_monitor_intent=bool(turn1["monitor_intent"]),
        turn1_clarification=bool(turn1["clarify"]),
        turn1_deferral=bool(turn1["deferral"]),
        turn1_dead_end=bool(turn1["dead_end"]),
        context_updated=context_updated,
        turn2_text=str(turn2["text"]) if turn2 else None,
        turn2_family=turn2["family"] if turn2 else None,
        turn2_true_safe=bool(turn2["safe"]) if turn2 else None,
        turn2_true_violation=bool(turn2["violate"]) if turn2 else None,
        turn2_true_intent=bool(turn2["intent"]) if turn2 else None,
        end2end_safe=end2end_safe,
        end2end_intent=end2end_intent,
        safe_but_no_intent=end2end_safe and not end2end_intent,
    )


def extractor_audit(scenarios):
    rows = []
    extractor_modes = ["loose_keyword", "strict_phrase", "action_aware"]

    for domain, (pre, post) in scenarios.items():
        for extractor in extractor_modes:
            for scenario in (pre, post):
                for response in scenario.responses:
                    obs = observed_trace(domain, response.text, extractor)
                    gold = response.gold_trace
                    prereq = scenario.prereq_label
                    target = scenario.target_predicate
                    rows.append({
                        "domain": domain,
                        "scenario": scenario.name,
                        "extractor": extractor,
                        "family": response.family,
                        "text": response.text,
                        "gold_trace": ",".join(gold),
                        "observed_trace": ",".join(obs),
                        "prereq_fp": (prereq in obs and prereq not in gold),
                        "restricted_fp": (target in obs and target not in gold),
                        "prereq_fn": (prereq in gold and prereq not in obs),
                        "restricted_fn": (target in gold and target not in obs),
                    })

    return pd.DataFrame(rows)


def summarize_audit(audit_df):
    return (
        audit_df
        .groupby(["extractor", "family"], as_index=False)
        .agg(
            n=("text", "count"),
            prereq_fp=("prereq_fp", "mean"),
            restricted_fp=("restricted_fp", "mean"),
            prereq_fn=("prereq_fn", "mean"),
            restricted_fn=("restricted_fn", "mean"),
        )
    )


def summarize_results(results):
    groups = defaultdict(list)
    for result in results:
        groups[(result.extractor, result.plan_name, result.drop_rate, result.method)].append(result)

    rows = []
    for (extractor, plan_name, drop_rate, method), items in sorted(groups.items()):
        n = len(items)
        rows.append({
            "extractor": extractor,
            "plan": plan_name,
            "drop": drop_rate,
            "method": method,
            "n": n,
            "retrieval_complete": sum(r.retrieval_complete for r in items) / n,
            "avg_attempts": sum(r.retrieval_attempts for r in items) / n,
            "turn1_violate": sum(r.turn1_true_violation for r in items) / n,
            "turn1_dead_end": sum(r.turn1_dead_end for r in items) / n,
            "clarify": sum(r.turn1_clarification for r in items) / n,
            "deferral": sum(r.turn1_deferral for r in items) / n,
            "context_update": sum(r.context_updated for r in items) / n,
            "end2end_safe": sum(r.end2end_safe for r in items) / n,
            "end2end_intent": sum(r.end2end_intent for r in items) / n,
            "safe_but_no_intent": sum(r.safe_but_no_intent for r in items) / n,
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


def build_report_text(summary_df, audit_summary):
    audit_cols = ["extractor", "family", "n", "prereq_fp", "restricted_fp", "prereq_fn", "restricted_fn"]
    result_cols = [
        "extractor",
        "plan",
        "drop",
        "method",
        "retrieval_complete",
        "turn1_violate",
        "turn1_dead_end",
        "clarify",
        "deferral",
        "end2end_safe",
        "end2end_intent",
        "safe_but_no_intent",
    ]

    lines = []
    lines.append("=== PHASE 1 CONTROLLED PROTOTYPE SUMMARY ===")
    lines.append("")
    lines.append("This suite refactors the V1-V14 controlled experiments into a reusable project structure.")
    lines.append("The sampler is a lightweight controlled masked-state proposal proxy. It is not yet a real masked diffusion model.")
    lines.append("")
    lines.append("=== EXTRACTOR AUDIT ===")
    lines.append(markdown_table(audit_summary, audit_cols))
    lines.append("")
    lines.append("=== MAIN CONTROLLED RESULTS ===")
    lines.append(markdown_table(summary_df, result_cols))
    lines.append("")
    lines.append("=== INTERPRETATION ===")
    lines.append("- Terminal-only filtering is represented by terminal_only_filter.")
    lines.append("- Recoverability variants check candidate masked states before final completion.")
    lines.append("- Adaptive retrieval is represented by adaptive_escalating.")
    lines.append("- Action-aware extraction separates restricted-object mention from restricted-action commitment.")
    lines.append("- Real masked diffusion model integration is explicitly left for Phase 2.")
    return "\n".join(lines)


def run_controlled_suite(config: Dict):
    corpus = build_policy_corpus()
    scenarios = make_scenarios()
    engine = RetrievalEngine(corpus, stop_words=config["retrieval"]["tfidf_stop_words"])

    seeds = config["experiment"]["seeds"]
    trials_per_seed = int(config["experiment"]["trials_per_seed"])
    drop_rates = [float(x) for x in config["experiment"]["drop_rates"]]
    extractor_modes = list(config["experiment"]["extractor_modes"])
    plan_names = list(config["experiment"]["plan_names"])
    methods = list(config["experiment"]["methods"])
    sampler_config = dict(config["sampler"])

    results = []
    examples = {}

    for seed in seeds:
        rng = random.Random(int(seed))
        for extractor in extractor_modes:
            for plan_name in plan_names:
                for drop_rate in drop_rates:
                    for domain, (pre, post) in scenarios.items():
                        for method in methods:
                            for trial in range(trials_per_seed):
                                result = run_dynamic_dialogue(
                                    domain=domain,
                                    pre=pre,
                                    post=post,
                                    corpus=corpus,
                                    engine=engine,
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
                                    examples[key] = result.__dict__

    summary_df = summarize_results(results)
    audit_df = extractor_audit(scenarios)
    audit_summary = summarize_audit(audit_df)
    report_text = build_report_text(summary_df, audit_summary)

    return {
        "summary_df": summary_df,
        "audit_df": audit_df,
        "audit_summary": audit_summary,
        "results": results,
        "examples": examples,
        "report_text": report_text,
    }
