from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from policy_path_safety.core.types import PlanResult, PolicyClause, Scenario


def label_to_words(label: str) -> str:
    return label.replace("_", " ")


def build_query(scenario: Scenario, query_mode: str) -> str:
    if query_mode == "raw_context":
        return scenario.context
    if query_mode == "intent_enriched":
        return scenario.context + " " + label_to_words(scenario.target_predicate) + " " + label_to_words(scenario.prereq_label)
    if query_mode == "policy_oriented":
        return "policy restriction prerequisite condition " + scenario.domain + " " + label_to_words(scenario.target_predicate) + " " + label_to_words(scenario.prereq_label)
    raise ValueError(query_mode)


class RetrievalEngine:
    def __init__(self, corpus: List[PolicyClause], stop_words: str = "english"):
        self.corpus = corpus
        self.texts = [clause.text for clause in corpus]
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def score(self, query: str, strategy: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.tfidf_matrix).flatten()
        if strategy in {"tfidf", "dense", "hybrid"}:
            return scores
        raise ValueError(strategy)

    def retrieve(self, query: str, strategy: str, top_k: int, drop_rate: float, rng: random.Random):
        scores = self.score(query, strategy)
        ranked = sorted(range(len(self.corpus)), key=lambda i: float(scores[i]), reverse=True)
        selected = [self.corpus[i] for i in ranked[:top_k]]
        return [clause for clause in selected if rng.random() >= drop_rate]


def plan_stages(plan_name: str) -> List[Tuple[str, str, int]]:
    if plan_name == "fixed_raw_tfidf_top2":
        return [("raw_context", "tfidf", 2)]
    if plan_name == "adaptive_escalating":
        return [
            ("raw_context", "tfidf", 2),
            ("intent_enriched", "hybrid", 3),
            ("policy_oriented", "dense", 5),
            ("policy_oriented", "hybrid", 5),
        ]
    raise ValueError(plan_name)


def summarize_clause_set(scenario: Scenario, clause_ids: Sequence[str], corpus: List[PolicyClause]):
    by_id = {clause.clause_id: clause for clause in corpus}
    has_forbid = False
    has_prereq = False
    for cid in clause_ids:
        clause = by_id[cid]
        if clause.domain != scenario.domain:
            continue
        if clause.clause_type == "forbid":
            has_forbid = True
        if clause.clause_type == "prerequisite":
            has_prereq = True
    return has_forbid, has_prereq, has_forbid and has_prereq


def run_retrieval_plan(scenario: Scenario, corpus: List[PolicyClause], engine: RetrievalEngine, plan_name: str, drop_rate: float, rng: random.Random) -> PlanResult:
    all_ids = []
    all_texts = []
    stage_log = []
    attempts = 0
    retrieved_count = 0

    for query_mode, strategy, top_k in plan_stages(plan_name):
        attempts += 1
        query = build_query(scenario, query_mode)
        kept = engine.retrieve(query, strategy, top_k, drop_rate, rng)
        kept_ids = [clause.clause_id for clause in kept]
        kept_texts = [clause.text for clause in kept]
        retrieved_count += len(kept_ids)

        for cid, text in zip(kept_ids, kept_texts):
            if cid not in all_ids:
                all_ids.append(cid)
                all_texts.append(text)

        has_forbid, has_prereq, complete = summarize_clause_set(scenario, all_ids, corpus)
        stage_log.append(f"{query_mode}+{strategy}+top{top_k}: kept={kept_ids}, complete={complete}")
        if complete:
            break

    has_forbid, has_prereq, complete = summarize_clause_set(scenario, all_ids, corpus)

    return PlanResult(
        plan_name=plan_name,
        drop_rate=drop_rate,
        retrieved_ids=tuple(all_ids),
        retrieved_texts=tuple(all_texts),
        has_forbid=has_forbid,
        has_prereq=has_prereq,
        retrieval_complete=complete,
        attempts=attempts,
        retrieved_count=retrieved_count,
        stage_log=tuple(stage_log),
    )


def merge_plans(pre_plan: PlanResult, post_plan: PlanResult, scenario: Scenario, corpus: List[PolicyClause]) -> PlanResult:
    ids = list(pre_plan.retrieved_ids)
    texts = list(pre_plan.retrieved_texts)

    for cid, text in zip(post_plan.retrieved_ids, post_plan.retrieved_texts):
        if cid not in ids:
            ids.append(cid)
            texts.append(text)

    has_forbid, has_prereq, complete = summarize_clause_set(scenario, ids, corpus)

    return PlanResult(
        plan_name=post_plan.plan_name,
        drop_rate=post_plan.drop_rate,
        retrieved_ids=tuple(ids),
        retrieved_texts=tuple(texts),
        has_forbid=has_forbid,
        has_prereq=has_prereq,
        retrieval_complete=complete,
        attempts=pre_plan.attempts + post_plan.attempts,
        retrieved_count=pre_plan.retrieved_count + post_plan.retrieved_count,
        stage_log=pre_plan.stage_log + post_plan.stage_log,
    )
