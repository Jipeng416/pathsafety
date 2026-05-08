from pathlib import Path

import yaml

from policy_path_safety.data.scenarios import build_policy_corpus, make_scenarios
from policy_path_safety.experiments.run_controlled import run_controlled_suite


def test_scenarios_and_corpus_exist():
    scenarios = make_scenarios()
    corpus = build_policy_corpus()
    assert "alcohol" in scenarios
    assert "mature" in scenarios
    assert "medical" in scenarios
    assert "finance" in scenarios
    assert len(corpus) >= 8


def test_fast_suite_runs(tmp_path):
    config = yaml.safe_load(Path("configs/main.yaml").read_text())
    config["experiment"]["seeds"] = [1]
    config["experiment"]["trials_per_seed"] = 1
    config["experiment"]["drop_rates"] = [0.0]
    config["experiment"]["extractor_modes"] = ["action_aware"]
    config["experiment"]["plan_names"] = ["adaptive_escalating"]
    config["experiment"]["methods"] = ["baseline_model_scored", "retrieved_intent_aware"]
    config["outputs"]["results_dir"] = str(tmp_path)

    bundle = run_controlled_suite(config)
    assert len(bundle["summary_df"]) > 0
    assert len(bundle["audit_summary"]) > 0
    assert "report_text" in bundle
