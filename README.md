# language-interface-experiments

Experiments for the paper "Language as the Interface of Power" (proposition P1, safety gradient across languages, dialects and code-switching).

Start with `docs/EXPERIMENTS_LANGUAGE_INTERFACE.md`. The rendering and generation pipeline is reused from an earlier health-domain study (see `docs/source_EXPERIMENTS.md`, `docs/source_RENDERING_METHODS.md`); that study's aggregate behavioral results are in `results/aggregate/` and its summary in `results/source_paper/SUMMARY.md`.

## What is new here
- `src/dlsd/mechanism.py`, `src/dlsd/activations.py`, `scripts/10_extract_activations.py`, `scripts/11_directions.py`: E1 pipeline (activations, English-only layer choice, per-condition retention against a random control). Tested offline; no real model run yet.
- `docs/EXPERIMENTS_LANGUAGE_INTERFACE.md`: E1 mechanism, E2 behavioral pilot, E3 simulation.

## Status
Nothing has been run for this paper. E1 extraction and analysis are written and tested on a tiny random model; the rendered prompts must be supplied (see the spec).

## Setup
```
pip install -r requirements.txt
pytest
```
Keep this repository private while the paper and the earlier study are under double-blind review. Row-level generations stay local.
