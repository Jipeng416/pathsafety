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

from policy_path_safety.experiments.run_controlled import run_controlled_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main.yaml")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.fast:
        config["experiment"]["seeds"] = [1]
        config["experiment"]["trials_per_seed"] = 2
        config["experiment"]["drop_rates"] = [0.0, 0.3]
        config["experiment"]["extractor_modes"] = ["action_aware"]
        config["experiment"]["plan_names"] = ["fixed_raw_tfidf_top2", "adaptive_escalating"]

    results_dir = Path(config["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = run_controlled_suite(config)

    summary_df = bundle["summary_df"]
    audit_df = bundle["audit_df"]
    audit_summary = bundle["audit_summary"]
    results = bundle["results"]
    examples = bundle["examples"]
    report_text = bundle["report_text"]

    summary_df.to_csv(results_dir / "main_results.csv", index=False)
    audit_df.to_csv(results_dir / "extractor_audit_full.csv", index=False)
    audit_summary.to_csv(results_dir / "extractor_audit_summary.csv", index=False)
    pd.DataFrame([asdict(r) for r in results]).to_csv(results_dir / "per_trial_results.csv", index=False)

    with (results_dir / "examples.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    (results_dir / "report_ready_text.txt").write_text(report_text, encoding="utf-8")

    with (results_dir / "main_tables.tex").open("w", encoding="utf-8") as f:
        f.write("% Main controlled results\n")
        f.write(summary_df.to_latex(index=False, float_format="%.3f"))
        f.write("\n\n% Extractor audit summary\n")
        f.write(audit_summary.to_latex(index=False, float_format="%.3f"))

    print(report_text)
    print("\nSaved outputs:")
    for name in [
        "main_results.csv",
        "per_trial_results.csv",
        "extractor_audit_full.csv",
        "extractor_audit_summary.csv",
        "examples.json",
        "report_ready_text.txt",
        "main_tables.tex",
    ]:
        print(results_dir / name)


if __name__ == "__main__":
    main()
