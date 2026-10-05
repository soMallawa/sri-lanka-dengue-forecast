from __future__ import annotations

# ruff: noqa: I001

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {
    "district_id",
    "week_start_date",
    "week_end_date",
    "target",
    "prediction",
    "model",
    "feature_set",
    "fold",
    "experiment_id",
    "config_id",
    "threshold90",
    "threshold95",
}
IDENTITY_COLUMNS = ["model", "feature_set", "experiment_id", "config_id"]
MISSINGNESS_GROUPS = (
    "complete_recent_case_history",
    "partial_historical_case_context",
    "rows_requiring_feature_imputation",
    "rows_with_quality_warnings",
    "m1_not_trainable",
)


class ModelingReportError(ValueError):
    """Raised when modeling report inputs violate the development-only contract."""


@dataclass(frozen=True)
class ModelingReportResult:
    district_error_analysis: pd.DataFrame
    temporal_error_analysis: pd.DataFrame
    outbreak_error_analysis: pd.DataFrame
    largest_error_cases: pd.DataFrame
    missingness_sensitivity: pd.DataFrame
    summary_markdown: str
    paths: dict[str, Path]


def _resolve_report_dir(output_dir: str | Path | None, root: str | Path | None) -> Path | None:
    if output_dir is not None:
        return Path(output_dir)
    if root is not None:
        return Path(root) / "data" / "reports"
    return None


def _artifact_name(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if prefix else name


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ModelingReportError(f"Predictions missing required columns: {missing}")
    out = predictions.copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"], errors="coerce")
    out["week_end_date"] = pd.to_datetime(out["week_end_date"], errors="coerce")
    for column in ["target", "prediction", "threshold90", "threshold95"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["week_start_date", "week_end_date"]].isna().any().any():
        raise ModelingReportError("Predictions contain invalid forecast origin dates")
    if out[["target", "prediction"]].isna().any().any():
        raise ModelingReportError("Predictions contain missing target or prediction values")
    out["actual_date"] = out["week_start_date"] + pd.Timedelta(days=7)
    out["cutoff_date"] = out["week_end_date"]
    out["error"] = out["prediction"] - out["target"]
    out["absolute_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    return out


def _filter_selected_config(
    predictions: pd.DataFrame,
    selected_config: dict[str, Any] | None,
) -> pd.DataFrame:
    if not selected_config:
        return predictions
    out = predictions
    for column in IDENTITY_COLUMNS:
        if column in selected_config and column in out.columns:
            out = out[out[column].astype("string").eq(str(selected_config[column]))]
    if out.empty:
        raise ModelingReportError("Selected config filtered predictions to zero rows")
    return out.copy()


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "n": 0,
            "mean_target": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "median_absolute_error": np.nan,
            "mae_relative_to_mean_target": np.nan,
        }
    mean_target = float(frame["target"].mean())
    mae = float(frame["absolute_error"].mean())
    return {
        "n": int(len(frame)),
        "mean_target": mean_target,
        "mae": mae,
        "rmse": float(np.sqrt(frame["squared_error"].mean())),
        "bias": float(frame["error"].mean()),
        "median_absolute_error": float(frame["absolute_error"].median()),
        "mae_relative_to_mean_target": mae / mean_target if mean_target > 0 else np.nan,
    }


def _grouped_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def district_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_predictions(predictions)
    return _grouped_metrics(frame, [*IDENTITY_COLUMNS, "district_id"])


def temporal_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_predictions(predictions)
    iso_week = frame["actual_date"].dt.isocalendar().week.astype("int64")
    enriched = frame.assign(
        actual_year=frame["actual_date"].dt.year,
        actual_month=frame["actual_date"].dt.month,
        actual_quarter=frame["actual_date"].dt.quarter,
        actual_week_band=pd.cut(
            iso_week,
            bins=[0, 13, 26, 39, 53],
            labels=["weeks_01_13", "weeks_14_26", "weeks_27_39", "weeks_40_53"],
            include_lowest=True,
        ).astype("string"),
    )
    pieces: list[pd.DataFrame] = []
    for grouping, column in [
        ("year", "actual_year"),
        ("month", "actual_month"),
        ("quarter", "actual_quarter"),
        ("week_band", "actual_week_band"),
    ]:
        piece = _grouped_metrics(enriched, [*IDENTITY_COLUMNS, column])
        piece = piece.rename(columns={column: "group_value"})
        piece.insert(len(IDENTITY_COLUMNS), "grouping", grouping)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def outbreak_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_predictions(predictions)
    rows: list[dict[str, Any]] = []
    group_columns = [*IDENTITY_COLUMNS, "fold"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        base = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=True))
        for subset_name, threshold_column in [
            ("top_10pct", "threshold90"),
            ("top_5pct", "threshold95"),
        ]:
            subset = group[group["target"].ge(group[threshold_column])]
            metrics = _metrics(subset)
            rows.append(
                {
                    **base,
                    "subset": subset_name,
                    "threshold_column": threshold_column,
                    "threshold_min": float(group[threshold_column].min())
                    if group[threshold_column].notna().any()
                    else np.nan,
                    "threshold_max": float(group[threshold_column].max())
                    if group[threshold_column].notna().any()
                    else np.nan,
                    "subset_n": int(len(subset)),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "bias": metrics["bias"],
                    "underprediction_rate": float((subset["prediction"] < subset["target"]).mean())
                    if len(subset)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _merge_context(
    predictions: pd.DataFrame,
    canonical_features: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = _prepare_predictions(predictions)
    if canonical_features is None:
        return frame
    context = canonical_features.copy()
    if "week_start_date" not in context.columns or "district_id" not in context.columns:
        raise ModelingReportError(
            "Canonical feature frame requires district_id and week_start_date"
        )
    context["week_start_date"] = pd.to_datetime(context["week_start_date"], errors="coerce")
    overlap = [
        column for column in context.columns if column in frame.columns and column != "district_id"
    ]
    overlap = [column for column in overlap if column != "week_start_date"]
    context = context.drop(columns=overlap)
    return frame.merge(context, on=["district_id", "week_start_date"], how="left", validate="m:1")


def largest_error_cases(
    predictions: pd.DataFrame,
    canonical_features: pd.DataFrame | None = None,
    *,
    limit: int = 50,
) -> pd.DataFrame:
    merged = _merge_context(predictions, canonical_features)
    preferred = [
        "district_id",
        "week_start_date",
        "actual_date",
        "cutoff_date",
        "target",
        "prediction",
        "absolute_error",
        "error",
        "dengue_cases",
    ]
    context_patterns = (
        "cases_lag_",
        "cases_roll_",
        "rainfall",
        "rain_days",
        "temp_",
        "humidity",
        "quality",
        "missing",
        "is_trainable",
    )
    context_columns = [
        column
        for column in merged.columns
        if column not in preferred and any(pattern in column for pattern in context_patterns)
    ]
    columns = [column for column in [*preferred, *context_columns] if column in merged.columns]
    return (
        merged.sort_values("absolute_error", ascending=False)
        .head(limit)
        .loc[:, columns]
        .reset_index(drop=True)
    )


def _missingness_masks(merged: pd.DataFrame) -> dict[str, pd.Series]:
    case_lag_columns = [column for column in merged.columns if column.startswith("cases_lag_")]
    imputation_columns = [
        column
        for column in merged.columns
        if column.endswith("__missing") or column.endswith("_missing_flag")
    ]
    quality_columns = [column for column in merged.columns if "quality" in column.lower()]
    if case_lag_columns:
        complete_history = merged[case_lag_columns].notna().all(axis=1)
        partial_history = ~complete_history
    else:
        complete_history = pd.Series(False, index=merged.index)
        partial_history = pd.Series(False, index=merged.index)
    if imputation_columns:
        requires_imputation = merged[imputation_columns].fillna(False).astype(bool).any(axis=1)
    else:
        requires_imputation = pd.Series(False, index=merged.index)
    if quality_columns:
        quality_warning = pd.Series(False, index=merged.index)
        for column in quality_columns:
            values = merged[column]
            if pd.api.types.is_bool_dtype(values):
                quality_warning = quality_warning | values.fillna(False)
            else:
                clean = values.astype("string").fillna("OK").str.upper()
                quality_warning = quality_warning | ~clean.isin(
                    ["OK", "PASS", "GOOD", "FALSE", "0"]
                )
    else:
        quality_warning = pd.Series(False, index=merged.index)
    if "is_trainable" in merged.columns:
        not_trainable = ~merged["is_trainable"].fillna(False).astype(bool)
    else:
        not_trainable = pd.Series(False, index=merged.index)
    return {
        "complete_recent_case_history": complete_history,
        "partial_historical_case_context": partial_history,
        "rows_requiring_feature_imputation": requires_imputation,
        "rows_with_quality_warnings": quality_warning,
        "m1_not_trainable": not_trainable,
    }


def missingness_sensitivity(
    predictions: pd.DataFrame,
    canonical_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = _merge_context(predictions, canonical_features)
    masks = _missingness_masks(merged)
    rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(IDENTITY_COLUMNS, key_values, strict=True))
        for group_name in MISSINGNESS_GROUPS:
            subset = group[masks[group_name].loc[group.index]]
            rows.append({**base, "missingness_group": group_name, **_metrics(subset)})
    return pd.DataFrame(rows)


def build_pretest_error_summary(
    *,
    district_errors: pd.DataFrame,
    temporal_errors: pd.DataFrame,
    outbreak_errors: pd.DataFrame,
    missingness: pd.DataFrame,
) -> str:
    worst = district_errors.sort_values("mae", ascending=False).head(5)
    best = district_errors.sort_values("mae", ascending=True).head(5)
    outbreak_nonempty = outbreak_errors[outbreak_errors["subset_n"] > 0]
    lines = [
        "# Development Validation Error Summary",
        "",
        "This is a pre-test validation summary. It must not include locked holdout results or "
        "drive post-test tuning.",
        "",
        "## District Patterns",
    ]
    lines.extend(
        f"- High MAE: {row.district_id} MAE={row.mae:.3f}, n={int(row.n)}"
        for row in worst.itertuples(index=False)
    )
    lines.extend(
        ["", "## Stronger Districts"]
        + [
            f"- Low MAE: {row.district_id} MAE={row.mae:.3f}, n={int(row.n)}"
            for row in best.itertuples(index=False)
        ]
    )
    if not outbreak_nonempty.empty:
        mean_outbreak_mae = outbreak_nonempty.groupby("subset")["mae"].mean()
        lines.extend(["", "## Outbreak-Like Rows"])
        for subset, value in mean_outbreak_mae.items():
            lines.append(f"- {subset}: mean fold MAE={value:.3f}")
    empty_groups = missingness.loc[missingness["n"].eq(0), "missingness_group"].unique().tolist()
    if empty_groups:
        lines.extend(["", "## Missingness Coverage"])
        lines.append(
            "- Absent groups are reported with n=0 and NA metrics: " + ", ".join(empty_groups)
        )
    temporal_groups = ", ".join(sorted(temporal_errors["grouping"].dropna().unique()))
    lines.extend(
        ["", "## Temporal Grouping", f"- Actual target dates used for: {temporal_groups}."]
    )
    return "\n".join(lines) + "\n"


def write_error_analysis_reports(
    predictions: pd.DataFrame,
    canonical_features: pd.DataFrame | None = None,
    *,
    selected_config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    root: str | Path | None = None,
    prefix: str = "",
    largest_errors_limit: int = 50,
) -> ModelingReportResult:
    report_dir = _resolve_report_dir(output_dir, root)
    selected_predictions = _filter_selected_config(predictions, selected_config)
    district = district_error_analysis(selected_predictions)
    temporal = temporal_error_analysis(selected_predictions)
    outbreak = outbreak_error_analysis(selected_predictions)
    largest = largest_error_cases(
        selected_predictions,
        canonical_features=canonical_features,
        limit=largest_errors_limit,
    )
    missingness = missingness_sensitivity(
        selected_predictions,
        canonical_features=canonical_features,
    )
    summary = build_pretest_error_summary(
        district_errors=district,
        temporal_errors=temporal,
        outbreak_errors=outbreak,
        missingness=missingness,
    )
    paths: dict[str, Path] = {}
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "district_error_analysis": (district, "district_error_analysis.csv"),
            "temporal_error_analysis": (temporal, "temporal_error_analysis.csv"),
            "outbreak_error_analysis": (outbreak, "outbreak_error_analysis.csv"),
            "largest_error_cases": (largest, "largest_error_cases.csv"),
            "missingness_sensitivity": (missingness, "missingness_sensitivity.csv"),
        }
        for key, (frame, filename) in outputs.items():
            path = report_dir / _artifact_name(prefix, filename)
            frame.to_csv(path, index=False)
            paths[key] = path
        summary_path = report_dir / _artifact_name(prefix, "pretest_error_summary.md")
        summary_path.write_text(summary, encoding="utf-8")
        paths["pretest_error_summary"] = summary_path
    return ModelingReportResult(
        district_error_analysis=district,
        temporal_error_analysis=temporal,
        outbreak_error_analysis=outbreak,
        largest_error_cases=largest,
        missingness_sensitivity=missingness,
        summary_markdown=summary,
        paths=paths,
    )


__all__ = [
    "ModelingReportError",
    "ModelingReportResult",
    "district_error_analysis",
    "largest_error_cases",
    "missingness_sensitivity",
    "outbreak_error_analysis",
    "temporal_error_analysis",
    "write_error_analysis_reports",
]
