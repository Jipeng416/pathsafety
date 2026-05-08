# PathSafety

This repository contains the Phase 1 clean reproducible prototype for belief-dependent path safety in policy-constrained masked language generation.

## Core idea

Final-response checking is too late when policy-relevant facts are missing. A partial masked state can already lose all policy-valid completions before the final answer is produced. The monitor therefore checks whether a candidate partial state is still recoverable.

## What Phase 1 provides

Phase 1 turns the earlier V1-V14 exploratory notebooks into a reproducible controlled prototype.

It includes:

- modular code under policy_path_safety/
- YAML configuration under configs/
- a controlled response universe for four domains
- action-aware semantic predicate extraction
- adaptive policy retrieval
- recoverability-guided sampling over masked states
- terminal-only and fail-closed baselines
- automatic result tables under results/
- smoke tests under tests/

## Important boundary

The current sampler is a controlled masked-state proposal proxy. It is not yet a real MDLM or LLaDA sampler. This is intentional for Phase 1. Phase 2 should replace this proxy with a real masked diffusion language model.

## One-command reproduction

Run:

    python scripts/run_controlled_suite.py --config configs/main.yaml

Fast smoke run:

    python scripts/run_controlled_suite.py --config configs/main.yaml --fast

## Main outputs

After running the suite, the following files are produced:

    results/main_results.csv
    results/main_tables.tex
    results/report_ready_text.txt
    results/examples.json
    results/extractor_audit_summary.csv

## Main metrics

- turn1_violate: whether turn 1 violates the policy.
- turn1_dead_end: whether turn 1 reaches a dead end.
- clarify: whether turn 1 asks the needed clarification.
- deferral: whether the system uses policy deferral.
- context_update: whether clarification leads to a second turn.
- end2end_safe: whether the full two-turn interaction is safe.
- end2end_intent: whether the original targeted request is completed after clarification.
- safe_but_no_intent: safe but not useful for the user's original request.
- retrieval_complete: whether both forbid and prerequisite clauses are retrieved.


## Phase 2: Masked denoising interface

Phase 2 adds a Hugging Face masked-LM-backed denoising sampler.

Fast demo:

    python scripts/run_phase2_diffusion_demo.py --config configs/phase2.yaml --fast

Full demo:

    python scripts/run_phase2_diffusion_demo.py --config configs/phase2.yaml

Phase 2 outputs:

    results/phase2_results.csv
    results/phase2_per_trial.csv
    results/phase2_examples.json
    results/phase2_report_ready_text.txt
    results/phase2_tables.tex

Boundary:

The current Phase 2 backend is a real masked-LM denoising backend, not yet the final large MDLM or LLaDA backend. The main contribution of this phase is the sampler-monitor interface.
