from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import yaml

from policy_path_safety.experiments.run_phase2_diffusion import run_phase2_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase2.yaml")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.fast:
        config["experiment"]["seeds"] = [1]
        config["experiment"]["trials_per_seed"] = 1
        config["experiment"]["domains"] = ["alcohol", "mature"]
        config["experiment"]["drop_rates"] = [0.0]
        config["experiment"]["extractor_modes"] = ["action_aware"]
        config["experiment"]["plan_names"] = ["adaptive_escalating"]
        config["experiment"]["methods"] = [
            "base_mlm_diffusion",
            "terminal_only_filter",
            "recoverability_guided",
            "retrieved_intent_aware",
            "fail_closed_if_retrieval_incomplete",
        ]

    results_dir = Path(config["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = run_phase2_experiment(config)

    summary_df = bundle["summary_df"]
    results = bundle["results"]
    examples = bundle["examples"]
    report_text = bundle["report_text"]

    summary_df.to_csv(results_dir / "phase2_results.csv", index=False)
    pd.DataFrame([asdict(x) for x in results]).to_csv(results_dir / "phase2_per_trial.csv", index=False)

    with (results_dir / "phase2_examples.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    (results_dir / "phase2_report_ready_text.txt").write_text(report_text, encoding="utf-8")

    with (results_dir / "phase2_tables.tex").open("w", encoding="utf-8") as f:
        f.write("% Phase 2 masked denoising interface results\n")
        f.write(summary_df.to_latex(index=False, float_format="%.3f"))

    print(report_text)
    print("\nSaved outputs:")
    for name in [
        "phase2_results.csv",
        "phase2_per_trial.csv",
        "phase2_examples.json",
        "phase2_report_ready_text.txt",
        "phase2_tables.tex",
    ]:
        print(results_dir / name)


if __name__ == "__main__":
    main()
