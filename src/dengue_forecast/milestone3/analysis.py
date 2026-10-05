from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.milestone3 import core, development

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATIONS = 1000
BOOTSTRAP_BLOCK_WEEKS = 4
DEFAULT_OUTPUT_ROOT = core.REPO_ROOT / "artifacts" / "milestone3" / "analysis"
FEATURE_SETS = development.FEATURE_SET_ORDER
MODEL_FAMILIES = development.MODEL_FAMILIES
FOLDS = development.FOLDS
HORIZONS = development.HORIZONS


class AnalysisError(ValueError):
    """Raised when saved-prediction analysis cannot be completed faithfully."""


@dataclass(frozen=True)
class AnalysisResult:
    output_dir: Path
    table_paths: dict[str, Path]
    summary_path: Path
    manifest_path: Path


def run_analysis(
    development_run: str | Path,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str,
    bootstrap_reps: int = BOOTSTRAP_REPLICATIONS,
    authenticate: bool = True,
) -> AnalysisResult:
    source_dir = Path(development_run).resolve()
    output_dir = Path(output_root).resolve() / run_id
    if output_dir.exists():
        raise AnalysisError(f"analysis output already exists: {output_dir}")
    if _is_relative_to(output_dir, source_dir):
        raise AnalysisError("analysis output must not be inside the completed development run")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = _run_analysis_inner(
            source_dir,
            output_dir,
            bootstrap_reps=bootstrap_reps,
            authenticate=authenticate,
        )
    except Exception:
        failed = output_dir / "failed.json"
        failed.write_text(
            json.dumps({"status": "failed"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    return result


def _run_analysis_inner(
    source_dir: Path,
    output_dir: Path,
    *,
    bootstrap_reps: int,
    authenticate: bool,
) -> AnalysisResult:
    validation = (
        development.validate_development_run(source_dir)
        if authenticate
        else {"status": "not_authenticated_test_fixture", "run_dir": str(source_dir)}
    )
    registry = pd.read_parquet(source_dir / "model_registry.parquet")
    selection = _read_json(source_dir / "selection.json")
    predictions = _load_predictions(source_dir, registry)
    thresholds = _load_thresholds(source_dir, registry)
    config_metrics, fold_metrics = config_metric_tables(registry, selection)
    weather = weather_effect_table(fold_metrics)
    district = district_table(predictions)
    high = high_incidence_table(predictions, thresholds)
    change = change_category_table(predictions, thresholds)
    ci_model = bootstrap_model_vs_persistence(predictions, bootstrap_reps=bootstrap_reps)
    ci_weather = bootstrap_weather_effects(predictions, bootstrap_reps=bootstrap_reps)

    tables = {
        "config_metrics": config_metrics,
        "fold_metrics": fold_metrics,
        "weather_effects": weather,
        "district_diagnostics": district,
        "high_incidence_thresholds": high,
        "change_categories": change,
        "bootstrap_model_vs_persistence": ci_model,
        "bootstrap_weather_effects": ci_weather,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        csv_path = output_dir / f"{name}.csv"
        json_path = output_dir / f"{name}.json"
        _write_csv(table, csv_path)
        _write_json(json_path, _json_records(table))
        paths[name] = csv_path
        paths[f"{name}_json"] = json_path

    manifest = _manifest(source_dir, output_dir, validation, paths, bootstrap_reps)
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    summary_path = output_dir / "development_summary.md"
    summary_path.write_text(
        render_summary(config_metrics, weather, ci_model, manifest),
        encoding="utf-8",
    )
    return AnalysisResult(
        output_dir=output_dir,
        table_paths=paths,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )


def config_metric_tables(
    registry: pd.DataFrame,
    selection: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    selected = selection.get("selected", {})
    for row in registry.sort_values(
        ["horizon", "feature_set", "model_family", "fold"]
    ).itertuples():
        diff = float(row.mae_model - row.mae_persistence)
        rel = (
            None
            if row.mae_persistence == 0
            else 100.0 * (row.mae_persistence - row.mae_model) / row.mae_persistence
        )
        fold_rows.append(
            {
                "horizon": int(row.horizon),
                "feature_set": str(row.feature_set),
                "model_family": str(row.model_family),
                "fold": int(row.fold),
                "fit_id": str(row.fit_id),
                "mae_model": float(row.mae_model),
                "mae_persistence": float(row.mae_persistence),
                "mae_model_minus_persistence": diff,
                "relative_improvement_pct": rel,
                "rmse_model": float(row.rmse_model),
                "rmse_persistence": float(row.rmse_persistence),
                "strict_mae_win_vs_persistence": bool(row.strict_mae_win),
                "strict_rmse_win_vs_persistence": bool(row.strict_rmse_win),
            }
        )
    fold = pd.DataFrame(fold_rows)
    config_rows = []
    for keys, group in fold.groupby(["horizon", "feature_set", "model_family"], sort=True):
        horizon, feature_set, model_family = keys
        if len(group) != len(FOLDS):
            raise AnalysisError(f"expected six folds for h{horizon} {feature_set} {model_family}")
        selected_row = selected.get(f"h{horizon}", {})
        role = "secondary"
        if feature_set == "cases_only" and model_family == "ridge":
            role = "primary_cases_only_ridge"
        if (
            selected_row.get("feature_set") == feature_set
            and selected_row.get("model_family") == model_family
        ):
            role = (
                "preselected_development_champion"
                if role == "secondary"
                else f"{role};preselected_development_champion"
            )
        mean_model = float(group["mae_model"].mean())
        mean_persistence = float(group["mae_persistence"].mean())
        rel = (
            None
            if mean_persistence == 0
            else 100.0 * (mean_persistence - mean_model) / mean_persistence
        )
        config_rows.append(
            {
                "horizon": int(horizon),
                "feature_set": str(feature_set),
                "model_family": str(model_family),
                "role": role,
                "fold_count": int(len(group)),
                "mean_fold_mae_model": mean_model,
                "mean_fold_mae_persistence": mean_persistence,
                "mean_fold_mae_model_minus_persistence": float(
                    (group["mae_model"] - group["mae_persistence"]).mean()
                ),
                "relative_improvement_pct_from_mean_fold_mae": rel,
                "mean_fold_rmse_model": float(group["rmse_model"].mean()),
                "mean_fold_rmse_persistence": float(group["rmse_persistence"].mean()),
                "mae_fold_wins_vs_persistence": int(group["strict_mae_win_vs_persistence"].sum()),
                "rmse_fold_wins_vs_persistence": int(group["strict_rmse_win_vs_persistence"].sum()),
                "aggregation": "unweighted_mean_of_fold_metrics_not_pooled_rows",
            }
        )
    return pd.DataFrame(config_rows), fold


def weather_effect_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("B_minus_A", "cases_rainfall", "cases_only"),
        ("C_minus_B", "cases_full_weather", "cases_rainfall"),
    ]
    for horizon in HORIZONS:
        for family in MODEL_FAMILIES:
            for label, added, base in comparisons:
                left = fold_metrics[
                    (fold_metrics["horizon"] == horizon)
                    & (fold_metrics["model_family"] == family)
                    & (fold_metrics["feature_set"] == added)
                ].set_index("fold")
                right = fold_metrics[
                    (fold_metrics["horizon"] == horizon)
                    & (fold_metrics["model_family"] == family)
                    & (fold_metrics["feature_set"] == base)
                ].set_index("fold")
                joined = left[["mae_model"]].join(
                    right[["mae_model"]],
                    how="inner",
                    lsuffix="_added",
                    rsuffix="_base",
                )
                if len(joined) != len(FOLDS):
                    raise AnalysisError(
                        f"weather comparison {label} h{horizon} {family} lacks matched folds"
                    )
                diffs = joined["mae_model_added"] - joined["mae_model_base"]
                rows.append(
                    {
                        "horizon": horizon,
                        "model_family": family,
                        "comparison": label,
                        "added_feature_set": added,
                        "baseline_feature_set": base,
                        "mean_fold_mae_difference_added_minus_baseline": float(diffs.mean()),
                        "sign_definition": "negative means added weather set has lower MAE",
                        "folds_added_weather_lower_mae": int((diffs < 0).sum()),
                        "folds_added_weather_higher_mae": int((diffs > 0).sum()),
                        "folds_tied": int((diffs == 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def district_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["horizon", "feature_set", "model_family", "district_id"], sort=True
    ):
        horizon, feature_set, model_family, district = keys
        observed = group["observed_target"].astype(float)
        mae = float((group["prediction_model"] - observed).abs().mean())
        mean_observed = float(observed.mean())
        rows.append(
            {
                "horizon": int(horizon),
                "feature_set": str(feature_set),
                "model_family": str(model_family),
                "district_id": str(district),
                "row_count": int(len(group)),
                "observed_target_total": float(observed.sum()),
                "observed_target_mean": mean_observed,
                "model_mae": mae,
                "persistence_mae": float((group["prediction_persistence"] - observed).abs().mean()),
                "normalized_model_mae_by_mean_observed": None
                if mean_observed == 0
                else mae / mean_observed,
                "normalized_error_semantics": (
                    "descriptive_only_null_when_mean_observed_target_is_zero"
                ),
            }
        )
    return pd.DataFrame(rows)


def high_incidence_table(predictions: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in thresholds.itertuples(index=False):
        group = predictions[
            (predictions["horizon"] == row.horizon)
            & (predictions["fold"] == row.fold)
            & (predictions["feature_set"] == row.feature_set)
            & (predictions["model_family"] == row.model_family)
        ]
        observed = group["observed_target"].astype(float)
        q90 = float(row.q90_incidence)
        q95 = float(row.q95_incidence)
        rows.append(
            {
                "horizon": int(row.horizon),
                "fold": int(row.fold),
                "feature_set": str(row.feature_set),
                "model_family": str(row.model_family),
                "fit_id": str(row.fit_id),
                "q90_incidence": q90,
                "q95_incidence": q95,
                "q90_boundary_rule": "observed_target >= q90_incidence",
                "q95_boundary_rule": "observed_target >= q95_incidence",
                "q90_count": int((observed >= q90).sum()),
                "q95_count": int((observed >= q95).sum()),
                "threshold_binding_sha256": str(row.threshold_binding_sha256),
            }
        )
    return pd.DataFrame(rows)


def change_category_table(predictions: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in thresholds.itertuples(index=False):
        group = predictions[
            (predictions["horizon"] == row.horizon)
            & (predictions["fold"] == row.fold)
            & (predictions["feature_set"] == row.feature_set)
            & (predictions["model_family"] == row.model_family)
        ].copy()
        group["delta"] = group["observed_target"].astype(float) - group[
            "observed_current_cases"
        ].astype(float)
        categories = _change_categories(
            group["delta"],
            row.stable_abs_delta_q25,
            row.large_up_q90,
            row.large_down_abs_q90,
        )
        counts = categories.value_counts(dropna=False).to_dict()
        for category in [
            "stable",
            "directional_up",
            "large_up",
            "directional_down",
            "large_down",
            "uncategorized",
        ]:
            rows.append(
                {
                    "horizon": int(row.horizon),
                    "fold": int(row.fold),
                    "feature_set": str(row.feature_set),
                    "model_family": str(row.model_family),
                    "fit_id": str(row.fit_id),
                    "category": category,
                    "count": int(counts.get(category, 0)),
                    "stable_abs_delta_q25": _none_if_nan(row.stable_abs_delta_q25),
                    "large_up_q90": _none_if_nan(row.large_up_q90),
                    "large_down_abs_q90": _none_if_nan(row.large_down_abs_q90),
                    "null_threshold_groups": _null_groups(row.large_up_q90, row.large_down_abs_q90),
                    "category_source": (
                        "observed evaluation delta categorized by training-bound cutoffs "
                        "from saved threshold receipt"
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_model_vs_persistence(
    predictions: pd.DataFrame,
    *,
    bootstrap_reps: int = BOOTSTRAP_REPLICATIONS,
) -> pd.DataFrame:
    predictions = _with_datetime_origin(predictions)
    rows = []
    draws = _fold_index_draws(predictions, bootstrap_reps)
    for keys, group in predictions.groupby(["horizon", "feature_set", "model_family"], sort=True):
        horizon, feature_set, family = keys
        point = _mean_fold_stat(group, _mae_model_minus_persistence)
        support = _support_summary(group)
        reason = _insufficient_reason(support)
        if reason is None:
            arrays = _model_error_arrays(group)
            samples = [_bootstrap_model_stat(arrays, draws, rep) for rep in range(bootstrap_reps)]
            lo, hi = np.quantile(samples, [0.025, 0.975])
        else:
            lo = hi = None
        rows.append(
            {
                "horizon": int(horizon),
                "feature_set": str(feature_set),
                "model_family": str(family),
                "statistic": "paired_mae_model_minus_persistence_mean_of_folds",
                "point_estimate": point,
                "ci_low": None if lo is None else float(lo),
                "ci_high": None if hi is None else float(hi),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_reps": int(bootstrap_reps),
                "block_weeks": BOOTSTRAP_BLOCK_WEEKS,
                "insufficient_support_reason": reason,
                **support,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_weather_effects(
    predictions: pd.DataFrame,
    *,
    bootstrap_reps: int = BOOTSTRAP_REPLICATIONS,
) -> pd.DataFrame:
    predictions = _with_datetime_origin(predictions)
    rows = []
    draws = _fold_index_draws(predictions, bootstrap_reps)
    comparisons = [
        ("B_minus_A", "cases_rainfall", "cases_only"),
        ("C_minus_B", "cases_full_weather", "cases_rainfall"),
    ]
    for horizon in HORIZONS:
        for family in MODEL_FAMILIES:
            for label, added, base in comparisons:
                group = predictions[
                    (predictions["horizon"] == horizon)
                    & (predictions["model_family"] == family)
                    & (predictions["feature_set"].isin([added, base]))
                ]
                point = _paired_config_mae_diff(group, added, base)
                support = _support_summary(group)
                reason = _insufficient_reason(support)
                if reason is None:
                    arrays = _weather_error_arrays(group, added, base)
                    samples = [
                        _bootstrap_weather_stat(arrays, draws, rep) for rep in range(bootstrap_reps)
                    ]
                    lo, hi = np.quantile(samples, [0.025, 0.975])
                else:
                    lo = hi = None
                rows.append(
                    {
                        "horizon": horizon,
                        "model_family": family,
                        "comparison": label,
                        "statistic": "matched_mae_added_minus_baseline_mean_of_folds",
                        "point_estimate": point,
                        "ci_low": None if lo is None else float(lo),
                        "ci_high": None if hi is None else float(hi),
                        "sign_definition": "negative means added weather set has lower MAE",
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_reps": int(bootstrap_reps),
                        "block_weeks": BOOTSTRAP_BLOCK_WEEKS,
                        "insufficient_support_reason": reason,
                        **support,
                    }
                )
    return pd.DataFrame(rows)


def render_summary(
    config_metrics: pd.DataFrame,
    weather: pd.DataFrame,
    bootstrap_model: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    champion_lines = []
    for row in (
        config_metrics[config_metrics["role"].str.contains("preselected_development_champion")]
        .sort_values("horizon")
        .itertuples()
    ):
        champion_lines.append(
            f"- H{row.horizon}: {row.model_family} / {row.feature_set}, "
            f"mean-fold MAE {row.mean_fold_mae_model:.6f} vs persistence "
            f"{row.mean_fold_mae_persistence:.6f}; model-minus-persistence "
            f"{row.mean_fold_mae_model_minus_persistence:.6f}."
        )
    weather_lines = []
    for row in weather.sort_values(["horizon", "model_family", "comparison"]).itertuples():
        weather_lines.append(
            f"- H{row.horizon} {row.model_family} {row.comparison}: "
            f"{row.mean_fold_mae_difference_added_minus_baseline:.6f} "
            f"(negative means lower MAE with added weather), "
            f"{row.folds_added_weather_lower_mae}/6 folds lower."
        )
    source = manifest["source_development_run"]
    authentication_line = (
        "- Authenticated saved development predictions were consumed read-only."
        if source["validation_status"] == "completed_read_only_verified"
        else (
            "- Saved prediction fixture inputs were consumed without production "
            f"authentication (`{source['validation_status']}`)."
        )
    )
    return "\n".join(
        [
            "# Milestone 3 Development Analysis Result",
            "",
            (
                "This is a development-only saved-prediction diagnostic report. It is a "
                "qualified retrospective observation-time analysis; historical "
                "publication/vintage availability is unestablished, so this is not "
                "operational backtesting."
            ),
            "",
            (
                "The 2025 cohort remains locked for this slice. Future 2025 evaluation, "
                "once authorized, is a 24-week, 25-district, 600-origin partial-year "
                "cohort across March and July-November and is not representative annual "
                "coverage."
            ),
            "",
            "## Source Binding",
            "",
            f"- Source run: `{source['path']}`",
            f"- Validation status: `{source['validation_status']}`",
            f"- Selection SHA256: `{source['selection_sha256']}`",
            f"- Registry SHA256: `{source['registry_sha256']}`",
            f"- Run identity SHA256: `{source['run_identity_sha256']}`",
            f"- Analysis source SHA256: `{manifest['analysis_source']['analysis_py_sha256']}`",
            f"- Analysis policy SHA256: `{manifest['analysis_source']['dispatch_policy_sha256']}`",
            f"- Reproducibility command: `{manifest['reproducibility_command']}`",
            "- Reproducibility requires a fresh `--run-id` because analysis outputs are immutable.",
            "",
            "## Preselected Development Champions",
            "",
            *champion_lines,
            "",
            "## Weather Diagnostics",
            "",
            *weather_lines,
            "",
            "## Bootstrap Policy",
            "",
            (
                f"Seed {BOOTSTRAP_SEED}, {manifest['bootstrap_reps']} draws, "
                "four-contiguous-origin-week moving blocks retaining all districts. "
                "Resampling is within each contiguous run and fold, never bridging gaps "
                "or crossing folds. CV statistics preserve equal-fold weighting. Paired "
                "model-minus-persistence and matched weather-difference intervals are "
                "separate; marginal intervals are not used to infer difference "
                "significance."
            ),
            "",
            "## Finished",
            "",
            authentication_line,
            (
                "- Machine-readable metric, weather, district, high-incidence, "
                "change-category, and bootstrap tables were written."
            ),
            (
                "- Cases-only Ridge and preselected development champions are explicitly "
                "labeled; no test winner selection was performed."
            ),
            "",
            "## Pending Later Slices",
            "",
            "- Plots are incomplete in this slice.",
            "- Feature importance is incomplete in this slice.",
            "- Portable package and final test-gate materials are incomplete.",
            (
                "- R2, Poisson deviance, bias, high-incidence error, change-error, "
                "and district-improvement deliverables remain pending."
            ),
            "",
            "## Limitations",
            "",
            (
                "Development results are exploratory, noncausal, and subject to multiple "
                "comparisons and limited effective origin-week/block support."
            ),
            "",
        ]
    )


def _load_predictions(source_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in registry.sort_values(
        ["horizon", "feature_set", "model_family", "fold"]
    ).itertuples():
        path = source_dir / str(row.prediction_path)
        frame = pd.read_parquet(path)
        expected = {
            "fit_id": row.fit_id,
            "fold": row.fold,
            "horizon": row.horizon,
            "feature_set": row.feature_set,
            "model_family": row.model_family,
        }
        for column, value in expected.items():
            if set(frame[column].astype(str if isinstance(value, str) else int)) != {value}:
                raise AnalysisError(f"prediction identity mismatch for {row.fit_id}: {column}")
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    return _with_datetime_origin(predictions)


def _with_datetime_origin(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["origin_start"] = pd.to_datetime(frame["origin_start"])
    return frame


def _load_thresholds(source_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in registry.itertuples():
        payload = _read_json(source_dir / str(row.threshold_path))
        rows.append(
            {
                "horizon": int(row.horizon),
                "fold": int(row.fold),
                "feature_set": str(row.feature_set),
                "model_family": str(row.model_family),
                "fit_id": str(row.fit_id),
                "q90_incidence": float(payload["q90_incidence"]),
                "q95_incidence": float(payload["q95_incidence"]),
                "stable_abs_delta_q25": float(payload["stable_abs_delta_q25"]),
                "large_up_q90": _float_or_nan(payload.get("large_up_q90")),
                "large_down_abs_q90": _float_or_nan(payload.get("large_down_abs_q90")),
                "threshold_binding_sha256": str(row.threshold_binding_sha256),
            }
        )
    return pd.DataFrame(rows)


def _change_categories(
    deltas: pd.Series,
    stable_abs_delta_q25: float,
    large_up_q90: float,
    large_down_abs_q90: float,
) -> pd.Series:
    out = pd.Series("uncategorized", index=deltas.index, dtype="object")
    stable = deltas.abs() <= stable_abs_delta_q25
    out[stable] = "stable"
    up = deltas > stable_abs_delta_q25
    down = deltas < -stable_abs_delta_q25
    out[up] = "directional_up"
    out[down] = "directional_down"
    if np.isfinite(large_up_q90):
        out[up & (deltas >= large_up_q90)] = "large_up"
    if np.isfinite(large_down_abs_q90):
        out[down & (-deltas >= large_down_abs_q90)] = "large_down"
    return out


def _fold_index_draws(predictions: pd.DataFrame, reps: int) -> dict[int, list[np.ndarray]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: dict[int, list[np.ndarray]] = {}
    for fold, group in predictions.groupby("fold", sort=True):
        weeks = sorted(pd.to_datetime(group["origin_start"]).drop_duplicates())
        positions = {week: index for index, week in enumerate(weeks)}
        runs = _contiguous_runs(weeks)
        run_blocks = [
            [
                np.array([positions[week] for week in block], dtype=int)
                for block in _contiguous_blocks(run)
            ]
            for run in runs
        ]
        fold_draws = []
        for _ in range(reps):
            sampled: list[int] = []
            for run, block_indexes in zip(runs, run_blocks, strict=True):
                run_sampled: list[int] = []
                if block_indexes:
                    while len(run_sampled) < len(run):
                        run_sampled.extend(
                            block_indexes[int(rng.integers(0, len(block_indexes)))]
                        )
                sampled.extend(run_sampled[: len(run)])
            fold_draws.append(np.array(sampled[: len(weeks)], dtype=int))
        draws[int(fold)] = fold_draws
    return draws


def _contiguous_runs(weeks: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    if not weeks:
        return []
    runs = [[weeks[0]]]
    for week in weeks[1:]:
        if (week - runs[-1][-1]).days == 7:
            runs[-1].append(week)
        else:
            runs.append([week])
    return runs


def _contiguous_blocks(weeks: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    blocks = []
    for start in range(0, len(weeks) - BOOTSTRAP_BLOCK_WEEKS + 1):
        block = weeks[start : start + BOOTSTRAP_BLOCK_WEEKS]
        if all((block[i] - block[i - 1]).days == 7 for i in range(1, len(block))):
            blocks.append(block)
    return blocks


def _mean_fold_stat(group: pd.DataFrame, fn: Any) -> float:
    return float(np.mean([fn(fold_group) for _, fold_group in group.groupby("fold", sort=True)]))


def _mae_model_minus_persistence(group: pd.DataFrame) -> float:
    observed = group["observed_target"].astype(float)
    return float(
        (group["prediction_model"] - observed).abs().mean()
        - (group["prediction_persistence"] - observed).abs().mean()
    )


def _paired_config_mae_diff(group: pd.DataFrame, added: str, base: str) -> float:
    diffs = []
    for _, fold_group in group.groupby("fold", sort=True):
        added_group = fold_group[fold_group["feature_set"] == added]
        base_group = fold_group[fold_group["feature_set"] == base]
        observed_added = added_group["observed_target"].astype(float)
        observed_base = base_group["observed_target"].astype(float)
        diffs.append(
            float(
                (added_group["prediction_model"] - observed_added).abs().mean()
                - (base_group["prediction_model"] - observed_base).abs().mean()
            )
        )
    return float(np.mean(diffs))


def _model_error_arrays(group: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    arrays = {}
    working = group.copy()
    observed = working["observed_target"].astype(float)
    working["model_abs_error"] = (working["prediction_model"] - observed).abs()
    working["persistence_abs_error"] = (working["prediction_persistence"] - observed).abs()
    for fold, fold_group in working.groupby("fold", sort=True):
        arrays[int(fold)] = _weekly_error_sums(fold_group)
    return arrays


def _weather_error_arrays(
    group: pd.DataFrame,
    added: str,
    base: str,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    arrays = {}
    working = group.copy()
    working["model_abs_error"] = (
        working["prediction_model"] - working["observed_target"].astype(float)
    ).abs()
    for fold, fold_group in working.groupby("fold", sort=True):
        arrays[int(fold)] = {
            "added": _weekly_error_sums(fold_group[fold_group["feature_set"] == added]),
            "base": _weekly_error_sums(fold_group[fold_group["feature_set"] == base]),
        }
    return arrays


def _weekly_error_sums(group: pd.DataFrame) -> dict[str, np.ndarray]:
    grouped = group.groupby("origin_start", sort=True)
    result = pd.DataFrame(
        {
            "model_abs_error_sum": grouped["model_abs_error"].sum(),
            "row_count": grouped.size(),
        }
    )
    if "persistence_abs_error" in group.columns:
        result["persistence_abs_error_sum"] = grouped["persistence_abs_error"].sum()
    return {column: result[column].to_numpy(dtype=float) for column in result.columns}


def _bootstrap_model_stat(
    arrays: dict[int, dict[str, np.ndarray]],
    draws: dict[int, list[np.ndarray]],
    rep: int,
) -> float:
    fold_stats = []
    for fold, fold_arrays in arrays.items():
        indexes = draws[fold][rep]
        count = fold_arrays["row_count"][indexes].sum()
        model = fold_arrays["model_abs_error_sum"][indexes].sum() / count
        persistence = fold_arrays["persistence_abs_error_sum"][indexes].sum() / count
        fold_stats.append(model - persistence)
    return float(np.mean(fold_stats))


def _bootstrap_weather_stat(
    arrays: dict[int, dict[str, dict[str, np.ndarray]]],
    draws: dict[int, list[np.ndarray]],
    rep: int,
) -> float:
    fold_stats = []
    for fold, fold_arrays in arrays.items():
        indexes = draws[fold][rep]
        added = fold_arrays["added"]
        base = fold_arrays["base"]
        added_mae = added["model_abs_error_sum"][indexes].sum() / added["row_count"][
            indexes
        ].sum()
        base_mae = base["model_abs_error_sum"][indexes].sum() / base["row_count"][
            indexes
        ].sum()
        fold_stats.append(added_mae - base_mae)
    return float(np.mean(fold_stats))


def _support_summary(group: pd.DataFrame) -> dict[str, Any]:
    counts = group.groupby("fold")["origin_start"].nunique().to_dict()
    run_lengths: dict[int, list[int]] = {}
    block_counts: dict[int, list[int]] = {}
    for fold, fold_group in group.groupby("fold", sort=True):
        weeks = sorted(pd.to_datetime(fold_group["origin_start"]).drop_duplicates())
        runs = _contiguous_runs(weeks)
        run_lengths[int(fold)] = [len(run) for run in runs]
        block_counts[int(fold)] = [len(_contiguous_blocks(run)) for run in runs]
    min_run_weeks = min(
        (length for lengths in run_lengths.values() for length in lengths),
        default=0,
    )
    min_run_blocks = min(
        (count for counts_per_fold in block_counts.values() for count in counts_per_fold),
        default=0,
    )
    return {
        "fold_origin_week_counts": json.dumps(
            {int(k): int(v) for k, v in counts.items()}, sort_keys=True
        ),
        "fold_contiguous_run_lengths": json.dumps(run_lengths, sort_keys=True),
        "fold_run_moving_block_counts": json.dumps(block_counts, sort_keys=True),
        "min_origin_weeks_per_fold": int(min(counts.values())) if counts else 0,
        "min_origin_weeks_per_contiguous_run": int(min_run_weeks),
        "min_blocks_per_contiguous_run": int(min_run_blocks),
    }


def _insufficient_reason(support: dict[str, Any]) -> str | None:
    if support["min_origin_weeks_per_contiguous_run"] < BOOTSTRAP_BLOCK_WEEKS:
        return "one_or_more_required_contiguous_runs_has_fewer_than_4_origin_weeks"
    if support["min_blocks_per_contiguous_run"] < 1:
        return "one_or_more_required_contiguous_runs_has_no_four_week_contiguous_block"
    return None


def _manifest(
    source_dir: Path,
    output_dir: Path,
    validation: dict[str, Any],
    paths: dict[str, Path],
    bootstrap_reps: int,
) -> dict[str, Any]:
    analysis_source = {
        "analysis_py_sha256": development.sha256_file(Path(__file__)),
        "standalone_script_sha256": development.sha256_file(
            core.REPO_ROOT / "scripts" / "milestone3_analysis.py"
        ),
        "analysis_tests_sha256": development.sha256_file(
            core.REPO_ROOT / "tests" / "unit" / "test_milestone3_analysis.py"
        ),
        "dispatch_policy_sha256": development.sha256_file(
            core.REPO_ROOT / "docs" / "m3-analysis-dispatch.md"
        ),
    }
    source = {
        "path": str(source_dir),
        "validation_status": validation.get("status"),
        "selection_sha256": development.sha256_file(source_dir / "selection.json"),
        "registry_sha256": development.sha256_file(source_dir / "model_registry.parquet"),
        "run_identity_sha256": development.sha256_file(source_dir / "run_identity.json"),
        "complete_sha256": development.sha256_file(source_dir / "complete.json"),
    }
    return {
        "status": "complete",
        "analysis_output_dir": str(output_dir),
        "source_development_run": source,
        "analysis_source": analysis_source,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_block_weeks": BOOTSTRAP_BLOCK_WEEKS,
        "tables": {
            name: {"path": str(path), "sha256": development.sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "reproducibility_command": (
            f"TMPDIR=/tmp/dengue-forecast-pytest-temp "
            f".venv/bin/python scripts/milestone3_analysis.py "
            f"--development-run {source_dir} --output-root {output_dir.parent} "
            f"--run-id <fresh-run-id> --bootstrap-reps {bootstrap_reps}"
        ),
        "pending_later_slices": ["plots", "feature_importance", "portable_package"],
        "qualification": (
            "development_only_qualified_retrospective_observation_time_not_operational_backtesting"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze authenticated M3 saved development predictions."
    )
    parser.add_argument("--development-run", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPLICATIONS)
    args = parser.parse_args(argv)
    result = run_analysis(
        args.development_run,
        output_root=args.output_root,
        run_id=args.run_id,
        bootstrap_reps=args.bootstrap_reps,
        authenticate=True,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(result.output_dir),
                "manifest_path": str(result.manifest_path),
                "summary_path": str(result.summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict(orient="records")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise AnalysisError(f"refusing to overwrite {path}")
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise AnalysisError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _none_if_nan(value: Any) -> float | None:
    numeric = float(value)
    return None if not np.isfinite(numeric) else numeric


def _null_groups(large_up_q90: float, large_down_abs_q90: float) -> str:
    groups = []
    if not np.isfinite(float(large_up_q90)):
        groups.append("large_up")
    if not np.isfinite(float(large_down_abs_q90)):
        groups.append("large_down")
    return ",".join(groups)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
