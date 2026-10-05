from __future__ import annotations

# ruff: noqa: E402, I001

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dengue_forecast.reports.modeling import ModelingReportError


class PlotReportError(ValueError):
    """Raised when development plot inputs are invalid."""


def _resolve_plot_dir(output_dir: str | Path | None, root: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if root is not None:
        return Path(root) / "data" / "reports" / "plots"
    return Path("data") / "reports" / "plots"


def _artifact_name(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if prefix else name


def _prepare(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"district_id", "week_start_date", "target", "prediction", "model"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise PlotReportError(f"Predictions missing required columns: {missing}")
    out = predictions.copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"], errors="coerce")
    out["actual_date"] = out["week_start_date"] + pd.Timedelta(days=7)
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    if out[["actual_date", "target", "prediction"]].isna().any().any():
        raise PlotReportError("Predictions contain invalid dates, targets, or predictions")
    out["residual"] = out["prediction"] - out["target"]
    out["absolute_error"] = out["residual"].abs()
    return out


def _filter_selected_config(
    predictions: pd.DataFrame,
    selected_config: dict[str, object] | None,
) -> pd.DataFrame:
    if not selected_config:
        return predictions
    out = predictions
    for column in ["model", "feature_set", "experiment_id", "config_id"]:
        if column in selected_config and column in out.columns:
            out = out[out[column].astype("string").eq(str(selected_config[column]))]
    if out.empty:
        raise PlotReportError("Selected config filtered predictions to zero rows")
    return out.copy()


def _identity_columns(frame: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    columns = ["district_id", "week_start_date"]
    if "fold" in frame.columns and "fold" in baseline.columns:
        columns.append("fold")
    return columns


def _model_label(frame: pd.DataFrame, fallback: str) -> str:
    parts = []
    for column in ["model", "objective", "feature_set", "experiment_id", "config_id"]:
        if column in frame.columns:
            values = sorted(frame[column].dropna().astype(str).unique())
            if len(values) == 1:
                parts.append(values[0])
    return " / ".join(dict.fromkeys(parts)) or fallback


def _select_baseline_for_primary_chart(
    baseline_predictions: pd.DataFrame,
    champion: pd.DataFrame,
    *,
    baseline_name: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    baseline = _prepare(baseline_predictions)
    if "model" not in baseline.columns:
        raise PlotReportError("Baseline predictions require a model column")
    key_columns = _identity_columns(champion, baseline)
    champion_keys = champion[key_columns].drop_duplicates()
    expected_rows = len(champion_keys)
    if expected_rows == 0:
        raise PlotReportError("Champion predictions contain no rows for baseline comparison")

    candidates: list[dict[str, Any]] = []
    selected_names = [baseline_name] if baseline_name is not None else sorted(
        baseline["model"].dropna().astype(str).unique()
    )
    for name in selected_names:
        subset = baseline[baseline["model"].astype(str).eq(str(name))].copy()
        if subset.empty:
            continue
        duplicate_count = int(subset.duplicated(key_columns).sum())
        if duplicate_count:
            raise PlotReportError(
                f"Baseline {name} has duplicate matched cohort keys: {duplicate_count}"
            )
        matched = champion_keys.merge(subset, on=key_columns, how="left", validate="1:1")
        available = matched["prediction"].notna() & matched["target"].notna()
        full_coverage = bool(available.all() and len(matched) == expected_rows)
        if not full_coverage:
            candidates.append(
                {
                    "baseline_name": str(name),
                    "available_rows": int(available.sum()),
                    "expected_rows": expected_rows,
                    "full_matched_coverage": False,
                    "mean_fold_mae": np.nan,
                }
            )
            continue
        matched["absolute_error"] = (
            pd.to_numeric(matched["prediction"], errors="coerce")
            - pd.to_numeric(matched["target"], errors="coerce")
        ).abs()
        fold_mae = matched.groupby("fold", dropna=False)["absolute_error"].mean()
        candidates.append(
            {
                "baseline_name": str(name),
                "available_rows": int(available.sum()),
                "expected_rows": expected_rows,
                "full_matched_coverage": True,
                "mean_fold_mae": float(fold_mae.mean()),
                "matched": matched,
            }
        )

    availability = pd.DataFrame(
        [
            {key: value for key, value in candidate.items() if key != "matched"}
            for candidate in candidates
        ]
    )
    full = [candidate for candidate in candidates if candidate.get("full_matched_coverage")]
    if not full:
        requested = f" named {baseline_name}" if baseline_name is not None else ""
        raise PlotReportError(
            "No full-coverage baseline"
            f"{requested} matches the champion district/week/fold cohort"
        )
    selected = min(full, key=lambda candidate: float(candidate["mean_fold_mae"]))
    return selected["matched"].copy(), availability, str(selected["baseline_name"])


def _fold_mae_rows(
    frame: pd.DataFrame,
    *,
    role: str,
    label: str,
) -> pd.DataFrame:
    rows = []
    for fold, group in frame.groupby("fold", dropna=False):
        rows.append(
            {
                "series_role": role,
                "series_label": label,
                "fold": fold,
                "n": int(len(group)),
                "fold_mae": float(group["absolute_error"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mean_fold_mae"] = float(out["fold_mae"].mean())
    out["weighting"] = "unweighted_mean_of_matched_fold_mae"
    return out


def _series_with_breaks(
    group: pd.DataFrame,
    *,
    value_column: str,
) -> tuple[list[pd.Timestamp], list[float]]:
    ordered = group.sort_values(["actual_date", "fold", "model"], kind="mergesort")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    previous: pd.Series | None = None
    for _, row in ordered.iterrows():
        if previous is not None:
            consecutive = pd.Timestamp(row["actual_date"]) - pd.Timestamp(
                previous["actual_date"]
            ) == pd.Timedelta(days=7)
            same_fold = str(row.get("fold", "")) == str(previous.get("fold", ""))
            same_model = str(row.get("model", "")) == str(previous.get("model", ""))
            if not (consecutive and same_fold and same_model):
                dates.append(pd.Timestamp(previous["actual_date"]) + pd.Timedelta(days=1))
                values.append(np.nan)
        dates.append(pd.Timestamp(row["actual_date"]))
        values.append(float(row[value_column]))
        previous = row
    return dates, values


def _select_representative_districts(
    frame: pd.DataFrame,
    selected_districts: dict[str, str] | None,
) -> dict[str, str]:
    if selected_districts:
        return dict(selected_districts)
    by_district = frame.groupby("district_id").agg(
        mean_target=("target", "mean"),
        mae=("absolute_error", "mean"),
    )
    if by_district.empty:
        raise ModelingReportError("No districts available for representative plots")
    ordered_volume = by_district.sort_values("mean_target")
    selected = {
        "low_volume": str(ordered_volume.index[0]),
        "medium_volume": str(ordered_volume.index[len(ordered_volume) // 2]),
        "high_volume": str(ordered_volume.index[-1]),
        "best_cv_mae": str(by_district["mae"].idxmin()),
        "worst_cv_mae": str(by_district["mae"].idxmax()),
    }
    return selected


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_development_plots(
    predictions: pd.DataFrame,
    *,
    baseline_predictions: pd.DataFrame | None = None,
    baseline_name: str | None = None,
    selected_districts: dict[str, str] | None = None,
    selected_config: dict[str, object] | None = None,
    output_dir: str | Path | None = None,
    root: str | Path | None = None,
    prefix: str = "",
    title_prefix: str = "Development validation",
) -> dict[str, Path]:
    frame = _prepare(_filter_selected_config(predictions, selected_config))
    plot_dir = _resolve_plot_dir(output_dir, root)
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    representatives = _select_representative_districts(frame, selected_districts)
    for label, district_id in representatives.items():
        group = frame[frame["district_id"].eq(district_id)].sort_values("actual_date")
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.8))
        actual_dates, actual_values = _series_with_breaks(group, value_column="target")
        forecast_dates, forecast_values = _series_with_breaks(group, value_column="prediction")
        ax.plot(actual_dates, actual_values, marker="o", label="Actual target count")
        ax.plot(
            forecast_dates,
            forecast_values,
            marker="o",
            label=f"Out-of-fold forecast: {_model_label(group, 'selected config')}",
        )
        ax.set_title(f"{title_prefix}: actual vs forecast counts ({label}: {district_id})")
        ax.set_xlabel("Target date (week start + 7 days)")
        ax.set_ylabel("Dengue case count target")
        ax.text(
            0.01,
            0.98,
            "Blank gaps mark missing/nonconsecutive target weeks or model/fold boundaries; "
            "no interpolation is drawn.",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.legend()
        ax.grid(alpha=0.25)
        paths[f"actual_vs_forecast_{label}"] = _save(
            fig,
            plot_dir / _artifact_name(prefix, f"actual_vs_forecast_{label}_{district_id}.png"),
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame["target"], frame["prediction"], alpha=0.7, s=24)
    limit = max(float(frame[["target", "prediction"]].max().max()), 1.0)
    ax.plot([0, limit], [0, limit], color="black", linewidth=1)
    ax.set_title(f"{title_prefix}: predicted vs actual target counts")
    ax.set_xlabel("Actual dengue case count target")
    ax.set_ylabel("Predicted dengue case count target")
    ax.grid(alpha=0.25)
    paths["predicted_vs_actual_scatter"] = _save(
        fig,
        plot_dir / _artifact_name(prefix, "predicted_vs_actual_scatter.png"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(frame["residual"], bins=min(30, max(5, int(np.sqrt(len(frame))))), color="#52616b")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"{title_prefix}: residual distribution")
    ax.set_xlabel("Prediction minus actual target count")
    ax.set_ylabel("Row count")
    paths["residual_distribution"] = _save(
        fig,
        plot_dir / _artifact_name(prefix, "residual_distribution.png"),
    )

    mae_by_district = frame.groupby("district_id")["absolute_error"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    mae_by_district.plot(kind="bar", ax=ax, color="#2f6f73")
    ax.set_title(f"{title_prefix}: MAE by district (all {len(mae_by_district)} districts)")
    ax.set_xlabel("District")
    ax.set_ylabel("Mean absolute error in target counts")
    paths["mae_by_district"] = _save(fig, plot_dir / _artifact_name(prefix, "mae_by_district.png"))

    by_month = frame.assign(actual_month=frame["actual_date"].dt.month).groupby("actual_month")[
        "absolute_error"
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    by_month.reindex(range(1, 13)).plot(kind="bar", ax=ax, color="#7a4d2b")
    ax.set_title(f"{title_prefix}: MAE by actual target month")
    ax.set_xlabel("Actual target month")
    ax.set_ylabel("Mean absolute error in target counts")
    paths["mae_by_month"] = _save(fig, plot_dir / _artifact_name(prefix, "mae_by_month.png"))

    if baseline_predictions is not None:
        matched_baseline, availability, selected_baseline = _select_baseline_for_primary_chart(
            baseline_predictions,
            frame,
            baseline_name=baseline_name,
        )
        champion_label = _model_label(frame, "champion")
        baseline_label = selected_baseline
        paired_input = pd.concat(
            [
                _fold_mae_rows(matched_baseline, role="baseline", label=baseline_label),
                _fold_mae_rows(frame, role="champion", label=champion_label),
            ],
            ignore_index=True,
        )
        paths["baseline_vs_champion_input"] = plot_dir / _artifact_name(
            prefix, "baseline_vs_champion_input.csv"
        )
        paired_input.to_csv(paths["baseline_vs_champion_input"], index=False)
        paths["baseline_availability"] = plot_dir / _artifact_name(
            prefix, "baseline_availability.csv"
        )
        availability.to_csv(paths["baseline_availability"], index=False)
        chart = (
            paired_input[["series_role", "series_label", "mean_fold_mae", "weighting"]]
            .drop_duplicates()
            .sort_values("series_role")
        )
        fig, ax = plt.subplots(figsize=(8, 4.8))
        colors = ["#7a4d2b" if role == "baseline" else "#2f6f73" for role in chart["series_role"]]
        ax.bar(chart["series_label"], chart["mean_fold_mae"], color=colors)
        for index, row in enumerate(chart.itertuples(index=False)):
            ax.text(index, float(row.mean_fold_mae), f"{float(row.mean_fold_mae):.3f}", ha="center")
        ax.set_title(
            f"{title_prefix}: strongest full-coverage baseline vs champion\n"
            "MAE is unweighted mean of matched fold MAE on identical district/week/fold rows"
        )
        ax.set_xlabel("Series")
        ax.set_ylabel("Mean fold MAE in target case counts")
        paths["baseline_vs_champion"] = _save(
            fig,
            plot_dir / _artifact_name(prefix, "baseline_vs_champion.png"),
        )

    return paths


__all__ = ["PlotReportError", "generate_development_plots"]
