from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dengue_forecast.milestone3 import core, development
from dengue_forecast.modeling.train import TrainedModel, load_champion
from dengue_forecast.utils.hashing import sha256_file

APPROVED_DEVELOPMENT_RUN = (
    core.REPO_ROOT / "artifacts" / "milestone3" / "development" / "approved-development-002"
)
APPROVED_SOURCE_SHA256 = "750ca89f93ba83d005df48271cdd0f10700fa64ce67043da0034f140ecd02793"
DEFAULT_SOURCE = core.REPO_ROOT / development.PRODUCTION_SOURCE_RELATIVE
DEFAULT_OUTPUT_ROOT = core.REPO_ROOT / "artifacts" / "milestone3" / "importance"
DEFAULT_RUN_ID = "saved-development-2020-importance-001"
DEFAULT_SAVED_IMPORTANCE_RUN = DEFAULT_OUTPUT_ROOT / DEFAULT_RUN_ID
CORRECTED_PLOTS_RUN_ID = "saved-development-2020-importance-plots-corrected-001"
LATEST_VALIDATION_FOLD = 2020
SAMPLE_SEED = 42
MAX_SAMPLE_ROWS = 200
PERMUTATION_REPEATS = 3
COEFFICIENT_TOP_N = 12
PERMUTATION_TOP_N = 8
KEY_COLUMNS = ["district_id", "week_start_date"]


class ImportanceError(ValueError):
    """Raised when M3 importance cannot be produced under the fixed contract."""


@dataclass(frozen=True)
class ImportanceResult:
    output_dir: Path
    manifest_path: Path
    coefficient_path: Path
    permutation_path: Path
    sampled_keys_path: Path
    run_id: str
    config_count: int
    sample_count_by_horizon: dict[int, int]


@dataclass(frozen=True)
class ExistingPlotResult:
    output_dir: Path
    manifest_path: Path
    source_dir: Path
    plot_count: int
    contact_sheet_path: Path | None


def run_importance(
    *,
    development_run: str | Path = APPROVED_DEVELOPMENT_RUN,
    source_path: str | Path = DEFAULT_SOURCE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    approved_source_sha256: str = APPROVED_SOURCE_SHA256,
    max_sample_rows: int = MAX_SAMPLE_ROWS,
    seed: int = SAMPLE_SEED,
    repeats: int = PERMUTATION_REPEATS,
    command: str | None = None,
    model_loader: Callable[[str | Path], TrainedModel] = load_champion,
) -> ImportanceResult:
    source_dir = _require_approved_development_path(Path(development_run))
    out_dir = _exclusive_output_dir(Path(output_root), run_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = _run_importance_inner(
            source_dir=source_dir,
            source_path=Path(source_path),
            out_dir=out_dir,
            run_id=run_id,
            approved_source_sha256=approved_source_sha256,
            max_sample_rows=max_sample_rows,
            seed=seed,
            repeats=repeats,
            command=command,
            model_loader=model_loader,
        )
    except Exception:
        (out_dir / "failed.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "timestamp_utc": _now(),
                    "contract": "no importance fallback; inspect exception from caller",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    return result


def render_existing_importance_plots(
    *,
    source_dir: str | Path = DEFAULT_SAVED_IMPORTANCE_RUN,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = CORRECTED_PLOTS_RUN_ID,
    command: str | None = None,
) -> ExistingPlotResult:
    saved_dir = Path(source_dir).resolve()
    out_dir = _exclusive_output_dir(Path(output_root), run_id)
    manifest = _read_json(saved_dir / "manifest.json")
    table_hashes = _verify_saved_importance_tables(saved_dir, manifest)
    coefficients = pd.read_csv(saved_dir / "ridge_coefficients.csv")
    permutation = pd.read_csv(saved_dir / "permutation_importance.csv")
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        plot_paths = _write_plots(out_dir, coefficients, permutation)
        captions_path = _write_captions(out_dir, plot_paths)
        plot_hashes = {path.name: sha256_file(path) for path in plot_paths}
        result_manifest = {
            "status": "complete",
            "run_id": run_id,
            "timestamp_utc": _now(),
            "scope": "corrected plots from authenticated saved M3 importance CSVs only",
            "source_importance_run": str(saved_dir),
            "source_manifest_sha256": sha256_file(saved_dir / "manifest.json"),
            "source_table_hashes": table_hashes,
            "source_tables_immutable": True,
            "rendering_contract": {
                "no_model_loading": True,
                "no_rawdata_reads": True,
                "no_fitting": True,
                "no_permutation_recomputation": True,
                "top_n_grouping": {
                    "coefficients": [
                        "horizon",
                        "feature_set",
                        "model_family",
                    ],
                    "permutation": [
                        "horizon",
                        "feature_set",
                        "model_family",
                    ],
                },
                "coefficient_top_n_per_configuration": COEFFICIENT_TOP_N,
                "permutation_top_n_per_configuration": PERMUTATION_TOP_N,
            },
            "plot_file_hashes": plot_hashes,
            "captions_path": str(captions_path.relative_to(out_dir)),
            "plot_paths": [str(path.relative_to(out_dir)) for path in plot_paths],
            "code_hashes": {
                "importance.py": sha256_file(Path(__file__)),
                "milestone3_importance.py": sha256_file(
                    core.REPO_ROOT / "scripts" / "milestone3_importance.py"
                )
                if (core.REPO_ROOT / "scripts" / "milestone3_importance.py").exists()
                else None,
                "milestone3_config": sha256_file(core.FIXED_PROTOCOL_CONFIG),
            },
            "command": command,
            "interpretation_limits": [
                "observational_noncausal",
                "not_operational_backtest",
                "no_feature_or_model_selection_changes",
                "no_2025_outcomes_or_values_used",
                "2025_still_locked_24_week_partial_year",
                "full_m3_not_complete",
            ],
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(result_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        (out_dir / "failed.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "timestamp_utc": _now(),
                    "contract": "CSV-only corrected plot rendering failed",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    contact = next((path for path in plot_paths if path.name == "contact-sheet.png"), None)
    return ExistingPlotResult(
        output_dir=out_dir,
        manifest_path=manifest_path,
        source_dir=saved_dir,
        plot_count=len(plot_paths),
        contact_sheet_path=contact,
    )


def _run_importance_inner(
    *,
    source_dir: Path,
    source_path: Path,
    out_dir: Path,
    run_id: str,
    approved_source_sha256: str,
    max_sample_rows: int,
    seed: int,
    repeats: int,
    command: str | None,
    model_loader: Callable[[str | Path], TrainedModel],
) -> ImportanceResult:
    if max_sample_rows < 1 or max_sample_rows > MAX_SAMPLE_ROWS:
        raise ImportanceError("max_sample_rows must be in [1, 200]")
    if seed != SAMPLE_SEED or repeats != PERMUTATION_REPEATS:
        raise ImportanceError("importance is fixed to seed42 and permutation3repeats")

    validation_receipt = development.validate_development_run(source_dir)
    registry = pd.read_parquet(source_dir / "model_registry.parquet")
    plan = _read_json(source_dir / "cohort_plan.json")["fits"]
    selected = _fold2020_registry(registry)
    protocol = development.load_protocol()
    source_before = _file_identity(source_path)
    observations, read_audit = development.load_development_data(
        source_path,
        protocol=protocol,
        approved_source_sha256=approved_source_sha256,
        fixture_mode=False,
    )
    source_after = _file_identity(source_path)
    if source_before != source_after:
        raise ImportanceError("source identity changed during importance reconstruction")
    tasks = core.build_development_direct_tasks(
        observations,
        test_boundary=development.CALENDAR_2025_CUTOFF,
    )
    schedule = [
        spec for spec in development.expected_schedule() if spec.fold == LATEST_VALIDATION_FOLD
    ]
    frozen = development._precompute_frozen_cohorts(tasks, protocol, schedule)
    _assert_reconstructed_against_saved(frozen, plan, selected)

    sample_by_horizon = _sample_evaluation_by_horizon(frozen, max_rows=max_sample_rows, seed=seed)
    sample_records = _sample_key_records(sample_by_horizon)
    sample_path = out_dir / "sampled_keys.csv"
    sample_records.to_csv(sample_path, index=False)
    sample_sha = sha256_file(sample_path)

    model_identities_before = _model_identities(source_dir, selected)
    coefficients: list[pd.DataFrame] = []
    permutations: list[pd.DataFrame] = []
    config_receipts: list[dict[str, Any]] = []
    for row in selected.sort_values(["horizon", "feature_set", "model_family"]).itertuples(
        index=False
    ):
        fit_id = str(row.fit_id)
        cohort = frozen[fit_id]
        model_dir = source_dir / str(row.model_path)
        trained = _load_trusted_model(model_dir, row, cohort, model_loader=model_loader)
        sample = sample_by_horizon[int(row.horizon)]
        _assert_sample_matches_frozen(sample, cohort, fit_id)
        coefficients.append(_ridge_coefficients(trained, row, cohort))
        permutations.append(
            _permutation_importance(
                trained,
                row,
                sample,
                cohort.feature_columns,
                target_column=cohort.spec.target_column,
                repeats=repeats,
                seed=seed,
            )
        )
        config_receipts.append(_config_receipt(row, cohort, trained, sample))
    model_identities_after = _model_identities(source_dir, selected)
    if model_identities_after != model_identities_before:
        raise ImportanceError("model identities changed during importance run")

    coef_table = _concat_or_empty(coefficients)
    perm_table = _concat_or_empty(permutations)
    coeff_path = out_dir / "ridge_coefficients.csv"
    perm_path = out_dir / "permutation_importance.csv"
    coef_table.to_csv(coeff_path, index=False)
    perm_table.to_csv(perm_path, index=False)
    table_hashes = {
        "ridge_coefficients.csv": sha256_file(coeff_path),
        "permutation_importance.csv": sha256_file(perm_path),
        "sampled_keys.csv": sample_sha,
    }
    plot_paths = _write_plots(out_dir, coef_table, perm_table)
    captions_path = _write_captions(out_dir, plot_paths)
    manifest = {
        "status": "complete",
        "run_id": run_id,
        "timestamp_utc": _now(),
        "scope": "fixed M3 development fold-2020 feature importance only",
        "development_run": str(source_dir),
        "development_validation": validation_receipt,
        "source_identity_before": source_before,
        "source_identity_after": source_after,
        "approved_source_sha256": approved_source_sha256,
        "read_audit_projected_content_digest": read_audit["projected_content_digest"],
        "fold": LATEST_VALIDATION_FOLD,
        "config_count": int(len(selected)),
        "expected_config_count": 24,
        "horizons": list(core.HORIZONS),
        "sample_seed": seed,
        "sample_max_rows": max_sample_rows,
        "permutation_repeats": repeats,
        "sample_count_by_horizon": {
            str(horizon): int(len(frame)) for horizon, frame in sample_by_horizon.items()
        },
        "model_identities_before": model_identities_before,
        "model_identities_after": model_identities_after,
        "config_receipts": config_receipts,
        "table_hashes": table_hashes,
        "plot_paths": [str(path.relative_to(out_dir)) for path in plot_paths],
        "captions_path": str(captions_path.relative_to(out_dir)),
        "code_hashes": {
            "importance.py": sha256_file(Path(__file__)),
            "milestone3_importance.py": sha256_file(
                core.REPO_ROOT / "scripts" / "milestone3_importance.py"
            )
            if (core.REPO_ROOT / "scripts" / "milestone3_importance.py").exists()
            else None,
            "milestone3_config": sha256_file(core.FIXED_PROTOCOL_CONFIG),
        },
        "command": command,
        "interpretation_limits": [
            "observational_noncausal",
            "not_operational_backtest",
            "no_feature_or_model_selection_changes",
            "no_2025_outcomes_or_values_used",
            "2025_still_locked_24_week_partial_year",
            "full_m3_not_complete",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ImportanceResult(
        output_dir=out_dir,
        manifest_path=manifest_path,
        coefficient_path=coeff_path,
        permutation_path=perm_path,
        sampled_keys_path=sample_path,
        run_id=run_id,
        config_count=int(len(selected)),
        sample_count_by_horizon={h: int(len(f)) for h, f in sample_by_horizon.items()},
    )


def _require_approved_development_path(path: Path) -> Path:
    resolved = path.resolve()
    approved = APPROVED_DEVELOPMENT_RUN.resolve()
    if resolved != approved:
        raise ImportanceError("importance may only authenticate approved-development-002")
    return resolved


def _exclusive_output_dir(output_root: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ImportanceError("invalid run_id")
    out = (output_root / run_id).resolve()
    root = output_root.resolve()
    if root not in out.parents:
        raise ImportanceError("importance output path escapes output root")
    if out.exists():
        raise ImportanceError(f"importance output already exists: {out}")
    return out


def _verify_saved_importance_tables(saved_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    expected = manifest.get("table_hashes")
    if not isinstance(expected, dict):
        raise ImportanceError("saved importance manifest does not contain table_hashes")
    required = ["ridge_coefficients.csv", "permutation_importance.csv", "sampled_keys.csv"]
    observed: dict[str, str] = {}
    for name in required:
        path = saved_dir / name
        if not path.is_file():
            raise ImportanceError(f"saved importance table missing: {name}")
        actual = sha256_file(path)
        if expected.get(name) != actual:
            raise ImportanceError(f"saved importance table hash mismatch: {name}")
        observed[name] = actual
    return observed


def _fold2020_registry(registry: pd.DataFrame) -> pd.DataFrame:
    selected = registry.loc[registry["fold"].astype(int).eq(LATEST_VALIDATION_FOLD)].copy()
    expected_ids = {
        spec.fit_id
        for spec in development.expected_schedule()
        if spec.fold == LATEST_VALIDATION_FOLD
    }
    if len(selected) != 24 or set(selected["fit_id"].astype(str)) != expected_ids:
        raise ImportanceError("fold2020 registry slice must contain exactly 24 fixed configs")
    if set(selected["status"].astype(str)) != {"complete"}:
        raise ImportanceError("fold2020 registry contains non-complete configs")
    return selected


def _assert_reconstructed_against_saved(
    frozen: Mapping[str, development.FrozenCohort],
    plan: Mapping[str, Any],
    selected: pd.DataFrame,
) -> None:
    for row in selected.itertuples(index=False):
        fit_id = str(row.fit_id)
        cohort = frozen[fit_id]
        saved = plan[fit_id]
        observed_keys = core.freeze_common_keys(cohort.evaluation)
        if _canonical(observed_keys) != _canonical(saved["evaluation_keys"]):
            raise ImportanceError(f"{fit_id} reconstructed evaluation key mismatch")
        observed_digest = core.cohort_digests(
            cohort.evaluation,
            target_column=cohort.spec.target_column,
            feature_columns=cohort.feature_columns,
        )
        if _canonical(observed_digest) != _canonical(saved["evaluation_digests"]):
            raise ImportanceError(f"{fit_id} reconstructed evaluation content digest mismatch")
        if str(row.evaluation_key_digest) != saved["evaluation_keys"]["row_key_digest"]:
            raise ImportanceError(f"{fit_id} registry frozen key mismatch")
        if str(row.evaluation_feature_content_digest) != saved["evaluation_digests"][
            "feature_content_digest"
        ]:
            raise ImportanceError(f"{fit_id} registry frozen feature digest mismatch")
        if str(row.evaluation_target_content_digest) != saved["evaluation_digests"][
            "target_content_digest"
        ]:
            raise ImportanceError(f"{fit_id} registry frozen target digest mismatch")


def _sample_evaluation_by_horizon(
    frozen: Mapping[str, development.FrozenCohort],
    *,
    max_rows: int,
    seed: int,
) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for horizon in core.HORIZONS:
        exemplar = frozen[f"h{horizon}__cases_only__ridge__fold{LATEST_VALIDATION_FOLD}"]
        frame = exemplar.evaluation.sort_values(KEY_COLUMNS).reset_index(drop=True)
        if len(frame) > max_rows:
            frame = frame.sample(n=max_rows, random_state=seed).sort_values(KEY_COLUMNS)
            frame = frame.reset_index(drop=True)
        out[horizon] = frame.copy(deep=True)
    return out


def _sample_key_records(sample_by_horizon: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, frame in sorted(sample_by_horizon.items()):
        for row in frame[KEY_COLUMNS].itertuples(index=False):
            rows.append(
                {
                    "horizon": horizon,
                    "district_id": str(row.district_id),
                    "week_start_date": pd.Timestamp(row.week_start_date).date().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def _assert_sample_matches_frozen(
    sample: pd.DataFrame,
    cohort: development.FrozenCohort,
    fit_id: str,
) -> None:
    full_keys = {
        (row["district_id"], row["week_start_date"]) for row in cohort.evaluation_keys["keys"]
    }
    sample_keys = {
        (str(row.district_id), pd.Timestamp(row.week_start_date).date().isoformat())
        for row in sample[KEY_COLUMNS].itertuples(index=False)
    }
    if not sample_keys <= full_keys:
        raise ImportanceError(f"{fit_id} sample keys are not a subset of frozen evaluation")
    digest = core.cohort_digests(
        cohort.evaluation,
        target_column=cohort.spec.target_column,
        feature_columns=cohort.feature_columns,
    )
    if _canonical(digest) != _canonical(cohort.evaluation_digests):
        raise ImportanceError(f"{fit_id} frozen evaluation changed before importance")


def _load_trusted_model(
    model_dir: Path,
    row: Any,
    cohort: development.FrozenCohort,
    *,
    model_loader: Callable[[str | Path], TrainedModel],
) -> TrainedModel:
    if not model_dir.is_dir():
        raise ImportanceError(f"{row.fit_id} trusted model directory missing")
    if sha256_file(model_dir / "model.joblib") != str(row.model_sha256):
        raise ImportanceError(f"{row.fit_id} model byte hash mismatch before load")
    if sha256_file(model_dir / "metadata.json") != str(row.model_metadata_sha256):
        raise ImportanceError(f"{row.fit_id} model metadata hash mismatch before load")
    trained = model_loader(model_dir)
    if list(trained.feature_columns) != list(cohort.feature_columns):
        raise ImportanceError(f"{row.fit_id} loaded model feature schema mismatch")
    if str(trained.target_column) != cohort.spec.target_column:
        raise ImportanceError(f"{row.fit_id} loaded model target mismatch")
    if trained.metadata.get("feature_columns") != list(cohort.feature_columns):
        raise ImportanceError(f"{row.fit_id} metadata feature schema mismatch")
    if trained.metadata.get("target_column") != cohort.spec.target_column:
        raise ImportanceError(f"{row.fit_id} metadata target mismatch")
    config = development._model_config(development.load_protocol(), str(row.model_family))
    if _canonical(trained.metadata.get("config")) != _canonical(config.serializable()):
        raise ImportanceError(f"{row.fit_id} metadata config mismatch")
    return trained


def _ridge_coefficients(
    trained: TrainedModel,
    row: Any,
    cohort: development.FrozenCohort,
) -> pd.DataFrame:
    if str(row.model_family) != "ridge":
        return pd.DataFrame()
    if not hasattr(trained.model, "coef_"):
        raise ImportanceError(f"{row.fit_id} ridge model has no coefficients")
    transformed = list(trained.preprocessor.output_feature_columns)
    coef = np.asarray(trained.model.coef_, dtype="float64")
    if len(coef) != len(transformed):
        raise ImportanceError(f"{row.fit_id} coefficient length mismatch")
    rows = []
    for feature, value in zip(transformed, coef, strict=True):
        rows.append(
            {
                **_fit_columns(row),
                "transformed_feature": feature,
                "raw_feature": _raw_feature(feature),
                "feature_unit": _transformed_unit(feature, trained),
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
                "sign_orientation": (
                    "positive coefficient increases predicted dengue cases after preprocessing"
                    if value >= 0
                    else "negative coefficient decreases predicted dengue cases after preprocessing"
                ),
                "coefficient_scale": "ridge_linear_coefficient_transformed_feature_units",
                "feature_list_sha256": core.feature_list_digest(cohort.feature_columns),
            }
        )
    return pd.DataFrame(rows)


def _permutation_importance(
    trained: TrainedModel,
    row: Any,
    sample: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    x = sample.loc[:, feature_columns].copy(deep=True)
    y = pd.to_numeric(sample[target_column], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(y).all():
        raise ImportanceError(f"{row.fit_id} sample target contains nonfinite values")
    baseline = _mae(y, trained.predict_next_week(x))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        repeat_values: list[float] = []
        for repeat in range(repeats):
            permuted = x.copy(deep=True)
            order = rng.permutation(len(permuted))
            permuted.loc[:, feature] = permuted[feature].to_numpy()[order]
            mae = _mae(y, trained.predict_next_week(permuted))
            repeat_values.append(float(mae - baseline))
            rows.append(
                {
                    **_fit_columns(row),
                    "feature": feature,
                    "repeat": repeat,
                    "baseline_mae": float(baseline),
                    "permuted_mae": float(mae),
                    "mae_increase": float(mae - baseline),
                    "importance_unit": "increase_in_MAE_cases_on_fixed_fold2020_sample",
                }
            )
        rows.append(
            {
                **_fit_columns(row),
                "feature": feature,
                "repeat": "mean",
                "baseline_mae": float(baseline),
                "permuted_mae": float(baseline + np.mean(repeat_values)),
                "mae_increase": float(np.mean(repeat_values)),
                "importance_unit": "increase_in_MAE_cases_on_fixed_fold2020_sample",
            }
        )
    return pd.DataFrame(rows)


def _fit_columns(row: Any) -> dict[str, Any]:
    return {
        "fit_id": str(row.fit_id),
        "horizon": int(row.horizon),
        "feature_set": str(row.feature_set),
        "model_family": str(row.model_family),
        "fold": int(row.fold),
    }


def _raw_feature(feature: str) -> str:
    if feature.endswith("__missing"):
        return feature.removesuffix("__missing")
    if feature.startswith("district_id__"):
        return "district_id"
    return feature.split("__", 1)[0] if "__" in feature else feature


def _transformed_unit(feature: str, trained: TrainedModel) -> str:
    raw = _raw_feature(feature)
    if feature.endswith("__missing"):
        return "binary_missing_indicator_0_or_1"
    if raw in trained.preprocessor.categories_:
        return "one_hot_encoded_indicator_0_or_1"
    if raw in trained.preprocessor.numeric_columns and trained.preprocessor.scale_numeric:
        return "standardized_numeric_z_score_after_training_fold_imputation"
    if raw in trained.preprocessor.numeric_columns:
        return "numeric_after_training_fold_imputation"
    return "transformed_feature_unit"


def _model_identities(source_dir: Path, selected: pd.DataFrame) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for row in selected.itertuples(index=False):
        model_dir = source_dir / str(row.model_path)
        identities[str(row.fit_id)] = {
            "model_path": str(Path(str(row.model_path))),
            "model_sha256": sha256_file(model_dir / "model.joblib"),
            "metadata_sha256": sha256_file(model_dir / "metadata.json"),
        }
    return identities


def _config_receipt(
    row: Any,
    cohort: development.FrozenCohort,
    trained: TrainedModel,
    sample: pd.DataFrame,
) -> dict[str, Any]:
    sample_keys = core.freeze_common_keys(sample)
    sample_digest = core.cohort_digests(
        sample,
        target_column=cohort.spec.target_column,
        feature_columns=cohort.feature_columns,
    )
    return {
        **_fit_columns(row),
        "sample_row_count": sample_keys["row_count"],
        "sample_key_digest": sample_keys["row_key_digest"],
        "sample_feature_content_digest": sample_digest["feature_content_digest"],
        "sample_target_content_digest": sample_digest["target_content_digest"],
        "frozen_evaluation_key_digest": cohort.evaluation_keys["row_key_digest"],
        "frozen_evaluation_feature_content_digest": cohort.evaluation_digests[
            "feature_content_digest"
        ],
        "frozen_evaluation_target_content_digest": cohort.evaluation_digests[
            "target_content_digest"
        ],
        "input_feature_count": len(cohort.feature_columns),
        "transformed_feature_count": len(trained.preprocessor.output_feature_columns),
        "target_column": cohort.spec.target_column,
    }


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def _write_plots(
    out_dir: Path, coefficients: pd.DataFrame, permutation: pd.DataFrame
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(_plot_coefficients(out_dir, coefficients))
    paths.extend(_plot_permutation(out_dir, permutation))
    pngs = [path for path in paths if path.suffix == ".png"]
    paths.extend(_plot_contact_sheet(out_dir, pngs))
    return paths


def _full_coefficient_identity(row: pd.Series) -> str:
    return (
        f"H{int(row['horizon'])} | {row['feature_set']} | {row['model_family']} | "
        f"{row['transformed_feature']}"
    )


def _full_permutation_identity(row: pd.Series) -> str:
    return (
        f"H{int(row['horizon'])} | {row['feature_set']} | {row['model_family']} | "
        f"{row['feature']}"
    )


def _select_top_coefficients(
    coefficients: pd.DataFrame, *, top_n: int = COEFFICIENT_TOP_N
) -> pd.DataFrame:
    top = (
        coefficients.assign(
            rank_value=pd.to_numeric(coefficients["abs_coefficient"], errors="raise"),
            plot_value=pd.to_numeric(coefficients["coefficient"], errors="raise"),
        )
        .sort_values(
            [
                "horizon",
                "feature_set",
                "model_family",
                "rank_value",
                "transformed_feature",
            ],
            ascending=[True, True, True, False, True],
        )
        .groupby(["horizon", "feature_set", "model_family"], group_keys=False)
        .head(top_n)
        .copy()
    )
    top["plot_identity"] = top.apply(_full_coefficient_identity, axis=1)
    return top


def _select_top_permutation(
    permutation: pd.DataFrame, *, top_n: int = PERMUTATION_TOP_N
) -> pd.DataFrame:
    means = permutation.loc[permutation["repeat"].astype(str).eq("mean")].copy()
    top = (
        means.assign(plot_value=pd.to_numeric(means["mae_increase"], errors="raise"))
        .sort_values(
            ["horizon", "feature_set", "model_family", "plot_value", "feature"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["horizon", "feature_set", "model_family"], group_keys=False)
        .head(top_n)
        .copy()
    )
    top["plot_identity"] = top.apply(_full_permutation_identity, axis=1)
    return top


def _unit_short_label(unit: str) -> str:
    labels = {
        "standardized_numeric_z_score_after_training_fold_imputation": "std numeric",
        "binary_missing_indicator_0_or_1": "missing binary",
        "one_hot_encoded_indicator_0_or_1": "one-hot binary",
        "numeric_after_training_fold_imputation": "numeric",
    }
    return labels.get(unit, unit)


def _format_axis_label(label: str, max_chars: int = 82) -> str:
    if len(label) <= max_chars:
        return label
    head = label[: max_chars - 1]
    cut = head.rfind("_")
    if cut < 40:
        cut = max_chars - 1
    return f"{label[:cut]}\n{label[cut:]}"


def _plot_coefficients(out_dir: Path, coefficients: pd.DataFrame) -> list[Path]:
    if coefficients.empty:
        return []
    paths: list[Path] = []
    top = _select_top_coefficients(coefficients)
    feature_sets = sorted(top["feature_set"].astype(str).unique())
    for horizon in core.HORIZONS:
        fig, axes = plt.subplots(
            1,
            len(feature_sets),
            figsize=(23, 8.5),
            constrained_layout=True,
            sharex=True,
        )
        if len(feature_sets) == 1:
            axes = [axes]
        for ax, feature_set in zip(axes, feature_sets, strict=True):
            sub = top.loc[
                top["horizon"].eq(horizon) & top["feature_set"].astype(str).eq(feature_set)
            ].copy()
            sub = sub.sort_values(["plot_value", "plot_identity"], ascending=[True, True])
            y_positions = np.arange(len(sub), dtype="float64")
            colors = np.where(sub["plot_value"].to_numpy() >= 0, "#287c71", "#a33f3f")
            ax.barh(y_positions, sub["plot_value"], color=colors)
            labels = [
                _format_axis_label(
                    f"{identity} ({_unit_short_label(str(unit))})",
                )
                for identity, unit in zip(
                    sub["plot_identity"].astype(str),
                    sub["feature_unit"].astype(str),
                    strict=True,
                )
            ]
            ax.set_yticks(y_positions, labels)
            ax.axvline(0, color="#1f2933", linewidth=0.8)
            ax.set_title(f"H{horizon} | ridge | {feature_set}")
            ax.set_xlabel("Coefficient: cases per transformed unit, linear prediction")
            ax.tick_params(axis="y", labelsize=6.5)
            ax.grid(axis="x", color="#d8dee4", linewidth=0.6)
        fig.suptitle(
            f"H{horizon} Ridge coefficients, top {COEFFICIENT_TOP_N} per feature set",
            fontsize=13,
        )
        paths.extend(_save_figure(fig, out_dir / f"ridge-coefficients-h{horizon}"))
    return paths


def _plot_permutation(out_dir: Path, permutation: pd.DataFrame) -> list[Path]:
    if permutation.empty:
        return []
    paths: list[Path] = []
    top = _select_top_permutation(permutation)
    feature_sets = sorted(top["feature_set"].astype(str).unique())
    families = sorted(top["model_family"].astype(str).unique())
    for horizon in core.HORIZONS:
        for family in families:
            fig, axes = plt.subplots(
                1,
                len(feature_sets),
                figsize=(23, 6.8),
                constrained_layout=True,
                sharex=True,
            )
            if len(feature_sets) == 1:
                axes = [axes]
            for ax, feature_set in zip(axes, feature_sets, strict=True):
                sub = top.loc[
                    top["horizon"].eq(horizon)
                    & top["model_family"].astype(str).eq(family)
                    & top["feature_set"].astype(str).eq(feature_set)
                ].copy()
                sub = sub.sort_values(["plot_value", "plot_identity"], ascending=[True, True])
                y_positions = np.arange(len(sub), dtype="float64")
                ax.barh(y_positions, sub["plot_value"], color="#3b6670")
                labels = [_format_axis_label(label) for label in sub["plot_identity"].astype(str)]
                ax.set_yticks(y_positions, labels)
                ax.axvline(0, color="#1f2933", linewidth=0.8)
                ax.set_title(f"H{horizon} | {family} | {feature_set}")
                ax.set_xlabel("Mean increase in MAE (cases)")
                ax.tick_params(axis="y", labelsize=6.5)
                ax.grid(axis="x", color="#d8dee4", linewidth=0.6)
            fig.suptitle(
                f"H{horizon} {family} permutation importance, "
                f"top {PERMUTATION_TOP_N} per feature set",
                fontsize=13,
            )
            paths.extend(_save_figure(fig, out_dir / f"permutation-importance-h{horizon}-{family}"))
    return paths


def _plot_contact_sheet(out_dir: Path, png_paths: list[Path]) -> list[Path]:
    if not png_paths:
        return []
    columns = min(3, len(png_paths))
    rows = int(np.ceil(len(png_paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 5 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, path in zip(axes_array, png_paths, strict=False):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(path.name, fontsize=9)
        ax.axis("off")
    for ax in axes_array[len(png_paths) :]:
        ax.axis("off")
    fig.tight_layout()
    return _save_figure(fig, out_dir / "contact-sheet")


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def _write_captions(out_dir: Path, plot_paths: list[Path]) -> Path:
    lines = [
        "# M3 importance captions",
        "",
        "- `ridge-coefficients-h*.{png,svg}`: Ridge coefficients only, faceted by horizon "
        "and feature set. Each bar is one full configuration-feature identity. Values are "
        "cases per transformed-unit change on the Ridge linear prediction before nonnegative "
        "prediction clipping. Standardized numeric features are z-score units after "
        "training-fold imputation; missingness and one-hot district features are binary "
        "0/1 indicators.",
        "- `permutation-importance-h*-*.{png,svg}`: Mean increase in MAE after permuting one "
        "original input feature on the fixed fold-2020 sample. Each mean averages only the "
        "three repeats for the same horizon, feature set, model family, and feature.",
        f"- Top-N omission: coefficient plots show top {COEFFICIENT_TOP_N} transformed "
        "features within each horizon/feature-set/Ridge configuration by absolute "
        f"coefficient. Permutation plots show top {PERMUTATION_TOP_N} original features "
        "within each horizon/feature-set/model-family configuration by mean MAE increase. "
        "Unshown rows remain in the authenticated CSVs.",
        "- Coefficient and permutation panels use distinct axes and should not be compared "
        "as common effect-size scales.",
        "- `contact-sheet.*`: Visual index of the corrected coefficient and permutation panels; "
        "the individual panel files are the legible review artifacts.",
        "",
        "These panels are observational/noncausal and are not an operational backtest. They do "
        "not change feature sets, model families, hyperparameters, selection, or 2025 lock status.",
        "",
        "Generated files:",
    ]
    lines.extend(f"- `{path.name}`" for path in plot_paths)
    path = out_dir / "captions.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - np.asarray(y_pred, dtype="float64"))))


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", default=str(APPROVED_DEVELOPMENT_RUN))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--approved-source-sha256", default=APPROVED_SOURCE_SHA256)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="render corrected plots from saved CSVs only; no model/source loading",
    )
    parser.add_argument(
        "--existing-importance-dir",
        default=str(DEFAULT_SAVED_IMPORTANCE_RUN),
        help="saved importance directory containing manifest.json and authenticated CSVs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = " ".join([Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])])
    if args.render_existing:
        run_id = (
            CORRECTED_PLOTS_RUN_ID if args.run_id == DEFAULT_RUN_ID else str(args.run_id)
        )
        result = render_existing_importance_plots(
            source_dir=args.existing_importance_dir,
            output_root=args.output_root,
            run_id=run_id,
            command=command,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(result.output_dir),
                    "manifest_path": str(result.manifest_path),
                    "plot_count": result.plot_count,
                    "contact_sheet_path": (
                        str(result.contact_sheet_path) if result.contact_sheet_path else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_importance(
        development_run=args.development_run,
        source_path=args.source,
        output_root=args.output_root,
        run_id=args.run_id,
        approved_source_sha256=args.approved_source_sha256,
        command=command,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(result.output_dir),
                "manifest_path": str(result.manifest_path),
                "config_count": result.config_count,
                "sample_count_by_horizon": result.sample_count_by_horizon,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "APPROVED_DEVELOPMENT_RUN",
    "APPROVED_SOURCE_SHA256",
    "CORRECTED_PLOTS_RUN_ID",
    "ExistingPlotResult",
    "ImportanceError",
    "ImportanceResult",
    "PERMUTATION_REPEATS",
    "SAMPLE_SEED",
    "render_existing_importance_plots",
    "run_importance",
]
