# Sri Lanka Dengue Forecast

Research code and guarded inference utilities for a Sri Lanka district-level dengue forecasting model release.

This repository contains the original source modules for acquisition/parsing, preprocessing/features, model training, evaluation, and prediction. It does not redistribute raw provider data, patient-level rows, GeoJSON files, or full outcome tables. Full scientific reproduction requires the original data sources, their licensing terms, and the frozen artifacts described in the release manifests.

## Quickstart: inference

Install the package, then download the frozen model release from Hugging Face:

```bash
git clone https://github.com/soMallawa/sri-lanka-dengue-forecast.git
cd sri-lanka-dengue-forecast
python -m pip install .
dengue-forecast-predict download \
  --revision 3c40226aa97579624794d91a10fbabe5b3443acb \
  --destination .models
dengue-forecast-predict verify --model-root .models
dengue-forecast-predict predict \
  --model-root .models \
  --model-id h1__cases_only__ridge__final \
  --input examples/SYNTHETIC/h1__cases_only__ridge__final_input.csv \
  --trust-pickle
```

`--revision` must be the full 40-character Hugging Face commit SHA. The downloader streams only the eight files listed in the trusted manifest, verifies each SHA-256 hash before placing it under `.models`, and does not unzip archives. `--trust-pickle` is required because the released model format is a scikit-learn/LightGBM joblib bundle. The command verifies SHA-256 hashes from the packaged `model_manifest_trusted.json` before loading.

Inputs must already be model-ready weekly feature rows with the exact columns in the manifest. This CLI does not build features from raw PDFs or weather APIs.

## Models

Model repository: <https://huggingface.co/manthilaffs/sri-lanka-dengue-forecast>

Interactive demo: <https://manthilaffs-sri-lanka-dengue-demo.hf.space> — historical predictions and a clearly labelled synthetic model sandbox; not live forecasting.

Selected final models:

| Horizon | Model | MAE | Persistence MAE |
| --- | --- | ---: | ---: |
| 1 week | `h1__cases_only__ridge__final` | 7.435948 | 7.901667 |
| 2 weeks | `h2__cases_full_weather__lightgbm_trial010__final` | 7.960736 | 9.180000 |
| 3 weeks | `h3__cases_full_weather__lightgbm_trial010__final` | 8.439516 | 10.473333 |
| 4 weeks | `h4__cases_full_weather__lightgbm_trial010__final` | 9.966845 | 12.720000 |

The frozen evaluation used 25 districts, 24 eligible 2025 forecast origins, and 600 district-origin pairs. The 2025 period is now a historical benchmark, not a fresh holdout. These models are post hoc research artifacts trained through late 2024; they are not a live operational surveillance system.

Predictions are one value per future target week. They are not cumulative multi-week totals.

## Public training interface

The public training command is for users who already have a model-ready weekly feature table. It does not download raw dengue bulletins, build weather features, rerun the consumed 2025 benchmark, or tune new hyperparameters.

Required columns:

- `week_start_date`
- `district_id`
- The exact feature columns for the selected horizon in `model_manifest_trusted.json`
- `target_h1`, `target_h2`, `target_h3`, or `target_h4`

Example:

```bash
dengue-forecast-train \
  --input model_ready_rows.parquet \
  --horizon 1 \
  --train-end 2024-12-14 \
  --output artifacts/new-run/h1
```

`--train-end` is the latest completed target reporting-week end permitted in training. Rows are kept only when `week_start_date + horizon_weeks * 7 days + 6 days <= train_end`. This retrospective boundary does not establish when a bulletin was actually published; operational work must additionally enforce real publication/availability timestamps. Dates in 2025 are rejected by default because the frozen 2025 evaluation is now a historical benchmark. Use `--allow-2025-research-scope` only for a clearly labelled new experiment. Output directories must not already exist.

The wrapper reuses the selected frozen model family, hyperparameters, feature schema, preprocessing code, and `train_fold_model` implementation. Numeric feature `NaN` values are allowed for the trained imputer; non-finite values, negative targets, missing required columns, and unknown districts are rejected.

## Source pipeline

The repository also keeps the original acquisition, feature, and evaluation entry points:

```bash
python -m pip install -e '.[train,geospatial]'
dengue-forecast dengue discover --source epid --start-year 2010 --end-year 2024 --data-root data
dengue-forecast dengue download --data-root data
dengue-forecast dengue parse --start-year 2010 --end-year 2024 --data-root data
dengue-forecast weather download --weather-start-date 2010-01-01 --weather-end-date 2024-12-31 --data-root data
dengue-forecast features build --data-root data
```

These commands require source data under the applicable third-party terms and optional training/geospatial dependencies. Model stages are guarded by explicit roots. Use a new artifact directory for any rerun:

```bash
dengue-forecast model train --root . --artifact-root artifacts/new-run --reports-root reports/new-run
```

Do not treat this as a one-command paper reproduction. The public package intentionally omits raw data and frozen private artifacts until rights review is complete.

## Dataset

Dataset documentation (metadata only): <https://huggingface.co/datasets/manthilaffs/sri-lanka-dengue-forecast-data>

The initial dataset release is metadata-only. Actual research rows are withheld pending redistribution-rights review. Synthetic examples under `examples/SYNTHETIC/` are only for API smoke tests and are not real dengue or weather observations.

## Repository status

Source repository: <https://github.com/soMallawa/sri-lanka-dengue-forecast>

Version: `v1.0.1`

## Citation

Please cite the software when using the source code or training pipeline, and the model release when using the frozen fitted models.

### APA

**Software**

> Mallawa, M. (2026). *Sri Lanka Dengue Forecast* (Version 1.0.1) [Computer software]. GitHub. https://github.com/soMallawa/sri-lanka-dengue-forecast

**Models**

> Mallawa, M. (2026). *Sri Lanka district-level dengue forecasting* (Model release v1.0.0) [Trained models]. Hugging Face. https://huggingface.co/manthilaffs/sri-lanka-dengue-forecast

### BibTeX

```bibtex
@misc{mallawa_dengue_software_2026,
  author       = {Mallawa, Manthila},
  title        = {Sri Lanka Dengue Forecast},
  year         = {2026},
  howpublished = {\url{https://github.com/soMallawa/sri-lanka-dengue-forecast}},
  note         = {Software version 1.0.1}
}

@misc{mallawa_dengue_models_2026,
  author       = {Mallawa, Manthila},
  title        = {Sri Lanka district-level dengue forecasting},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/manthilaffs/sri-lanka-dengue-forecast}},
  note         = {Model release v1.0.0; revision 3c40226aa97579624794d91a10fbabe5b3443acb}
}
```
