# Phase 2 Masked Denoising Interface Report

## Purpose

Phase 2 adds a real Hugging Face masked-LM denoising backend and connects it to the Phase 1 recoverability monitor.

## What changed from Phase 1

Phase 1 used a controlled masked-state proposal proxy. Phase 2 adds HFMaskedLMDenoisingSampler, which uses a real masked language model to score candidate denoising states.

## Boundary

This is still not the final large MDLM or LLaDA experiment. The current backend is a masked-LM denoising backend with the same interface needed for MDLM/LLaDA integration.

## Reproduction

Fast demo:

    python scripts/run_phase2_diffusion_demo.py --config configs/phase2.yaml --fast

Full Phase 2 demo:

    python scripts/run_phase2_diffusion_demo.py --config configs/phase2.yaml

## Outputs

- results/phase2_results.csv
- results/phase2_per_trial.csv
- results/phase2_examples.json
- results/phase2_report_ready_text.txt
- results/phase2_tables.tex

## Next step

Replace HFMaskedLMDenoisingSampler with a real MDLM or LLaDA backend while preserving the candidate-state interface.
