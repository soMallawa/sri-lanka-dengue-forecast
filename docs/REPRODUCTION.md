# Reproduction and development

This repository contains acquisition, feature preparation, training and evaluation source code. The actual case-containing dataset is withheld pending redistribution permission. Source availability is not a claim that a fresh download will reproduce every frozen result: source revisions, permissions, missing snapshots and historical availability matter.

## Install and test

Use Python 3.11 for the verified release environment.

```bash
python -m pip install '.[dev]'
ruff check --config pyproject.toml src tests
python -m pytest tests
```

The default tests are offline. Optional integration tests need `DENGUE_FORECAST_SOURCE_FITS` pointing to a trusted original fit-directory layout; ordinary contributors do not need the private research archive. The release verification separately exercises all four downloadable models.

To test an installed wheel outside the checkout:

```bash
uv build --wheel
uv venv .release-work/wheel-venv
uv pip install --python .release-work/wheel-venv/bin/python dist/*.whl
(cd .release-work && wheel-venv/bin/python -I -m dengue_forecast.public_inference list-models)
```

## Executable synthetic training example

The following file is supplied and **entirely synthetic**. Its arbitrary features and targets test the training interface, not epidemiological performance.

```bash
dengue-forecast-train \
  --input examples/SYNTHETIC/public_training_h1.csv \
  --horizon 1 \
  --train-end 2024-12-14 \
  --output artifacts/new-run/synthetic-h1
```

The output directory must not already exist. The wrapper uses the original fitted-model family/configuration, feature schema and `train_fold_model` implementation; this example fits a separate tiny Ridge model, never changes the released weights and never evaluates the frozen 2025 benchmark.

For real work, replace the input with authorised model-ready rows, choose the horizon and declare the completed target-week cutoff. The command requires `week_start_date`, the model's feature columns and `target_hN`. It excludes target weeks ending after the cutoff. Actual publication delays require additional filtering using verified availability timestamps; the built-in retrospective date boundary is not operational validation.

## Acquisition and features

Install the optional acquisition/geospatial dependencies:

```bash
python -m pip install '.[train,geospatial]'
dengue-forecast --help
dengue-forecast dengue discover --help
dengue-forecast weather download --help
dengue-forecast features build --help
```

Use `configs/sources.yaml` and the commands' documented arguments to obtain and validate source inputs under their own terms. Weather acquisition requires compatible geometry and weighting; feature generation requires the canonical case/weather inputs. These are pipeline stages, not independent substitutes for missing prerequisites. Quarantined or unavailable source material must not become fabricated zero counts.

The preserved `dengue_forecast.milestone3.core.build_direct_tasks` constructs direct horizon targets by district and calendar week. Use it on the canonical feature table to obtain `target_h1` through `target_h4`; do not replace missing calendar weeks with positional shifts, and do not confuse direct weekly targets with cumulative targets from other experiments. The preserved milestone3 development/evaluation modules implement the historical comparison protocol; their frozen final-evaluation orchestration additionally requires authorisation/provenance assets not shipped here.

## Evaluation discipline

Create new artifact directories and use chronological development/validation splits with target-availability boundaries and the documented embargo rules. Fit preprocessing only on training rows. Retain the persistence baseline and compare MAE/RMSE and district-level errors, not a generic 'accuracy' score.

The qualified partial-year 2025 benchmark has already been consumed. This public release copies its saved aggregate results; it does not rerun it. New tuning or deployment claims require a new evaluation period/protocol. The `--allow-2025-research-scope` flag explicitly labels use of 2025-or-later rows as a new research scope; it does not make the old test set fresh again.
