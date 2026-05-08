# Phase 1 Controlled Experiment Report

## Purpose

This report records the clean reproducible controlled prototype for path-level policy safety under evolving context.

## Main claim tested in Phase 1

Final-response checking is too late. A candidate masked state can become unrecoverable before the final response is produced. The recoverability monitor checks candidate partial states during generation.

## Current boundary

The sampler in Phase 1 is a controlled masked-state proposal proxy. It is not a real MDLM or LLaDA sampler. Phase 2 replaces the proxy with a real masked diffusion language model.

## Reproduction

Main run:

    python scripts/run_controlled_suite.py --config configs/main.yaml

Fast smoke test:

    python scripts/run_controlled_suite.py --config configs/main.yaml --fast

## Outputs

- results/main_results.csv
- results/main_tables.tex
- results/report_ready_text.txt
- results/examples.json
- results/extractor_audit_summary.csv
