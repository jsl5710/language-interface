# language-interface-experiments

Experiments for the paper "Language as the Interface of Power" (proposition P1, safety gradient across languages, dialects and code-switching).

Start with `docs/EXPERIMENTS_LANGUAGE_INTERFACE.md`. The rendering and generation pipeline is reused from an earlier health-domain study (see `docs/source_EXPERIMENTS.md`, `docs/source_RENDERING_METHODS.md`); that study's aggregate behavioral results are in `results/aggregate/` and its summary in `results/source_paper/SUMMARY.md`.

## What is new here
- `src/dlsd/mechanism.py`: refusal and harmfulness direction arithmetic (difference in means, projection, retention, controls, ablation), tested in `tests/test_mechanism.py`.
- `docs/EXPERIMENTS_LANGUAGE_INTERFACE.md`: E1 mechanism, E2 behavioral pilot, E3 simulation.

## Status
Nothing has been run for this paper. Activation extraction is not yet written.

## Setup
```
pip install -r requirements.txt
pytest
```
Keep this repository private while the paper and the earlier study are under double-blind review. Row-level generations stay local.
