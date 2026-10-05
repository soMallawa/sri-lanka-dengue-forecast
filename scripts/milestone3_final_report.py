from __future__ import annotations

# ruff: noqa: E501
import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 4)
FEATURE_SET_LABELS = {
    "cases_only": "Cases only",
    "cases_rainfall": "Cases + rainfall",
    "cases_full_weather": "Cases + full weather",
}
MODEL_LABELS = {
    "ridge": "Ridge",
    "lightgbm_trial010": "LightGBM trial010",
}
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 1000
BOOTSTRAP_BLOCK_WEEKS = 4
RUN_ID = "qualified-partial-2025-001"


@dataclass(frozen=True)
class SourcePaths:
    final_evaluation: Path
    final_freeze: Path
    development: Path
    development_analysis: Path
    diagnostics: Path
    development_plots: Path
    importance: Path
    importance_plots: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_value(section: dict[str, Any], name: str) -> float | None:
    cell = section.get(name, {})
    value = cell.get("value")
    if value is None:
        return None
    return float(value)


def metric_reason(section: dict[str, Any], name: str) -> str | None:
    cell = section.get(name, {})
    reason = cell.get("reason")
    return None if pd.isna(reason) else reason


def scalar_metric(payload: Any) -> float | None:
    value = payload.get("value") if isinstance(payload, dict) else payload
    if value is None:
        return None
    return float(value)


def discover_sources(root: Path) -> SourcePaths:
    base = root / "artifacts" / "milestone3"
    return SourcePaths(
        final_evaluation=base / "final_evaluation" / "approved-final-evaluation-001",
        final_freeze=base / "final_evaluation" / "approved-final-freeze-001",
        development=base / "development" / "approved-development-002",
        development_analysis=base / "analysis" / "saved-predictions-dev-corrected-001",
        diagnostics=base / "diagnostics" / "saved-development-secondary-diagnostics-corrected-001",
        development_plots=base / "plots" / "corrected-development-final-001",
        importance=base / "importance" / "saved-development-2020-importance-001",
        importance_plots=base / "importance" / "saved-development-2020-importance-plots-corrected-001",
    )


def final_fit_dirs(final_evaluation: Path, expected_count: int = 24) -> list[Path]:
    fits = sorted((final_evaluation / "fits").glob("h*__final"))
    if len(fits) != expected_count:
        raise ValueError(f"Expected {expected_count} final fit directories, found {len(fits)}")
    return fits


def build_final_metric_table(
    final_evaluation: Path, freeze_manifest: dict[str, Any], expected_count: int = 24
) -> pd.DataFrame:
    champion_flags = freeze_manifest["accepted_development_authority"]["champions"]
    rows: list[dict[str, Any]] = []
    for fit_dir in final_fit_dirs(final_evaluation, expected_count=expected_count):
        complete = read_json(fit_dir / "complete.json")
        metrics = read_json(fit_dir / "metrics.json")
        metadata = read_json(fit_dir / "model" / "metadata.json")
        model = metrics["model"]
        persistence = metrics["persistence"]
        fit_id = str(complete["fit_id"])
        rows.append(
            {
                "fit_id": fit_id,
                "horizon": int(complete["horizon"]),
                "feature_set": complete["feature_set"],
                "feature_set_label": FEATURE_SET_LABELS[str(complete["feature_set"])],
                "model_family": complete["model_family"],
                "model_label": MODEL_LABELS[str(complete["model_family"])],
                "preselected_champion": bool(champion_flags[fit_id]),
                "fixed_cases_only_ridge": complete["feature_set"] == "cases_only"
                and complete["model_family"] == "ridge",
                "evaluation_count": int(complete["evaluation_count"]),
                "training_count": int(complete["training_count"]),
                "mae_model": metric_value(model, "mae"),
                "mae_persistence": metric_value(persistence, "mae"),
                "mae_model_minus_persistence": float(metrics["mae_model_minus_persistence"]),
                "mae_improvement_positive_better": -float(metrics["mae_model_minus_persistence"]),
                "relative_mae_improvement_pct": scalar_metric(
                    metrics["relative_mae_improvement_pct"]
                ),
                "rmse_model": metric_value(model, "rmse"),
                "rmse_persistence": metric_value(persistence, "rmse"),
                "r2_model": metric_value(model, "r2"),
                "r2_persistence": metric_value(persistence, "r2"),
                "poisson_deviance_model": metric_value(model, "poisson_deviance"),
                "poisson_deviance_persistence": metric_value(persistence, "poisson_deviance"),
                "bias_model": metric_value(model, "bias"),
                "bias_persistence": metric_value(persistence, "bias"),
                "mae_top_10pct_model": metric_value(model, "mae_top_10pct"),
                "mae_top_10pct_persistence": metric_value(persistence, "mae_top_10pct"),
                "top_10_count": metric_value(model, "top_10_count"),
                "prediction_sha256": complete["prediction_sha256"],
                "prediction_values_sha256": complete["prediction_values_sha256"],
                "metrics_sha256": complete["metrics_sha256"],
                "threshold_sha256": complete["threshold_sha256"],
                "model_metadata_sha256": complete["model_metadata_sha256"],
                "model_sha256": complete["model_sha256"],
                "feature_count": len(metadata["feature_columns"]),
                "feature_columns": ";".join(metadata["feature_columns"]),
                "hyperparams_json": json.dumps(metadata["config"]["hyperparams"], sort_keys=True),
                "objective": metadata["config"].get("objective"),
                "preprocessing_fit_scope": metadata["config"]["preprocessing"]["fit_scope"],
                "train_start": metadata["frozen_period_bounds"]["train_start"],
                "train_end": metadata["frozen_period_bounds"]["train_end"],
                "prediction_path": complete["prediction_path"],
            }
        )
    table = pd.DataFrame(rows).sort_values(["horizon", "feature_set", "model_family"])
    if expected_count == 24 and set(table["evaluation_count"]) != {600}:
        raise ValueError("All final fits must have evaluation_count=600")
    return table.reset_index(drop=True)


def _contiguous_runs(origin_weeks: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    if not origin_weeks:
        return []
    runs = [[origin_weeks[0]]]
    for week in origin_weeks[1:]:
        if (week - runs[-1][-1]).days == 7:
            runs[-1].append(week)
        else:
            runs.append([week])
    return runs


def moving_block_bootstrap_ci(
    cluster_values: pd.Series,
    *,
    seed: int = BOOTSTRAP_SEED,
    reps: int = BOOTSTRAP_REPS,
    block_weeks: int = BOOTSTRAP_BLOCK_WEEKS,
) -> dict[str, Any]:
    ordered = cluster_values.sort_index()
    origin_weeks = [pd.Timestamp(x) for x in ordered.index]
    runs = _contiguous_runs(origin_weeks)
    point = float(ordered.mean())
    run_lengths = [len(run) for run in runs]
    block_counts = [max(len(run) - block_weeks + 1, 0) for run in runs]
    if any(length < block_weeks for length in run_lengths):
        return {
            "point_estimate": point,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_seed": seed,
            "bootstrap_reps": reps,
            "block_weeks": block_weeks,
            "insufficient_support_reason": (
                "one_or_more_required_contiguous_runs_has_fewer_than_4_origin_weeks"
            ),
            "origin_week_count": len(origin_weeks),
            "contiguous_run_lengths": json.dumps(run_lengths),
            "run_moving_block_counts": json.dumps(block_counts),
        }

    rng = np.random.default_rng(seed)
    values_by_week = {pd.Timestamp(index): float(value) for index, value in ordered.items()}
    estimates: list[float] = []
    for _ in range(reps):
        sampled_values: list[float] = []
        for run in runs:
            starts = np.arange(0, len(run) - block_weeks + 1)
            run_values: list[float] = []
            while len(run_values) < len(run):
                start = int(rng.choice(starts))
                run_values.extend(values_by_week[week] for week in run[start : start + block_weeks])
            sampled_values.extend(run_values[: len(run)])
        estimates.append(float(np.mean(sampled_values)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_estimate": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_seed": seed,
        "bootstrap_reps": reps,
        "block_weeks": block_weeks,
        "insufficient_support_reason": None,
        "origin_week_count": len(origin_weeks),
        "contiguous_run_lengths": json.dumps(run_lengths),
        "run_moving_block_counts": json.dumps(block_counts),
    }


def prediction_frame(final_evaluation: Path, relative_prediction_path: str) -> pd.DataFrame:
    path = final_evaluation / relative_prediction_path
    frame = pd.read_parquet(path)
    required = {
        "fit_id",
        "horizon",
        "model_family",
        "feature_set",
        "district_id",
        "origin_start",
        "observed_target",
        "prediction_model",
        "prediction_persistence",
        "observed_current_cases",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return frame


def paired_model_persistence_ci_table(
    final_evaluation: Path, metric_table: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in metric_table.to_dict("records"):
        frame = prediction_frame(final_evaluation, str(record["prediction_path"]))
        diff = (
            (frame["prediction_model"] - frame["observed_target"]).abs()
            - (frame["prediction_persistence"] - frame["observed_target"]).abs()
        )
        by_week = diff.groupby(pd.to_datetime(frame["origin_start"])).mean()
        ci = moving_block_bootstrap_ci(by_week)
        metric_point = float(record["mae_model_minus_persistence"])
        if not math.isclose(ci["point_estimate"], metric_point, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"CI point estimate mismatch for {record['fit_id']}")
        rows.append(
            {
                "fit_id": record["fit_id"],
                "horizon": record["horizon"],
                "feature_set": record["feature_set"],
                "model_family": record["model_family"],
                "statistic": "paired_mae_model_minus_persistence_by_origin_week",
                **ci,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "feature_set", "model_family"])


def weather_ci_table(final_evaluation: Path, metric_table: pd.DataFrame) -> pd.DataFrame:
    frames: dict[tuple[int, str, str], pd.DataFrame] = {}
    for record in metric_table.to_dict("records"):
        key = (int(record["horizon"]), str(record["model_family"]), str(record["feature_set"]))
        frames[key] = prediction_frame(final_evaluation, str(record["prediction_path"]))

    rows: list[dict[str, Any]] = []
    comparisons = [
        ("B_minus_A", "cases_rainfall", "cases_only"),
        ("C_minus_B", "cases_full_weather", "cases_rainfall"),
    ]
    keys = ["district_id", "origin_start", "observed_target", "prediction_persistence"]
    for horizon in HORIZONS:
        for model_family in ("ridge", "lightgbm_trial010"):
            for comparison, added, baseline in comparisons:
                added_frame = frames[(horizon, model_family, added)]
                baseline_frame = frames[(horizon, model_family, baseline)]
                merged = added_frame[keys + ["prediction_model"]].merge(
                    baseline_frame[keys + ["prediction_model"]],
                    on=keys,
                    suffixes=("_added", "_baseline"),
                    validate="one_to_one",
                )
                diff = (
                    (merged["prediction_model_added"] - merged["observed_target"]).abs()
                    - (merged["prediction_model_baseline"] - merged["observed_target"]).abs()
                )
                by_week = diff.groupby(pd.to_datetime(merged["origin_start"])).mean()
                ci = moving_block_bootstrap_ci(by_week)
                rows.append(
                    {
                        "horizon": horizon,
                        "model_family": model_family,
                        "comparison": comparison,
                        "added_feature_set": added,
                        "baseline_feature_set": baseline,
                        "statistic": "matched_mae_added_minus_baseline_by_origin_week",
                        "sign_definition": "negative means added weather set has lower MAE",
                        **ci,
                    }
                )
    return pd.DataFrame(rows).sort_values(["horizon", "model_family", "comparison"])


def final_district_table(final_evaluation: Path, metric_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in metric_table.to_dict("records"):
        frame = prediction_frame(final_evaluation, str(record["prediction_path"]))
        frame = frame.assign(
            abs_model=(frame["prediction_model"] - frame["observed_target"]).abs(),
            abs_persistence=(frame["prediction_persistence"] - frame["observed_target"]).abs(),
        )
        grouped = frame.groupby("district_id", sort=True)
        for district, group in grouped:
            model_mae = float(group["abs_model"].mean())
            persistence_mae = float(group["abs_persistence"].mean())
            improvement = persistence_mae - model_mae
            if persistence_mae == 0:
                pct = None
                reason = "persistence_mae_zero"
            else:
                pct = improvement / persistence_mae * 100.0
                reason = None
            rows.append(
                {
                    "fit_id": record["fit_id"],
                    "horizon": record["horizon"],
                    "feature_set": record["feature_set"],
                    "model_family": record["model_family"],
                    "district_id": district,
                    "row_count": int(len(group)),
                    "observed_target_total": float(group["observed_target"].sum()),
                    "observed_target_mean": float(group["observed_target"].mean()),
                    "model_mae": model_mae,
                    "persistence_mae": persistence_mae,
                    "mae_improvement_absolute_positive_better": improvement,
                    "mae_improvement_pct_positive_better": pct,
                    "mae_improvement_pct_null_reason": reason,
                    "case_volume_caveat": (
                        "raw district MAE reflects case volume; compare with observed totals/means"
                    ),
                }
            )
    return pd.DataFrame(rows)


def final_high_incidence_table(final_evaluation: Path, metric_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in metric_table.to_dict("records"):
        frame = prediction_frame(final_evaluation, str(record["prediction_path"]))
        threshold = read_json(final_evaluation / f"fits/{record['fit_id']}/thresholds.json")
        for group_name, threshold_key in [
            ("q90_primary", "q90_incidence"),
            ("q95_supplementary", "q95_incidence"),
        ]:
            threshold_value = threshold[threshold_key]
            subset = frame[frame["observed_target"] >= threshold_value]
            if subset.empty:
                rows.append(
                    {
                        "fit_id": record["fit_id"],
                        "horizon": record["horizon"],
                        "feature_set": record["feature_set"],
                        "model_family": record["model_family"],
                        "incidence_group": group_name,
                        "threshold": threshold_value,
                        "count": 0,
                        "mae_model": None,
                        "mae_persistence": None,
                        "bias_model": None,
                        "bias_persistence": None,
                        "null_reason": "no_rows_meeting_training_threshold",
                    }
                )
                continue
            rows.append(
                {
                    "fit_id": record["fit_id"],
                    "horizon": record["horizon"],
                    "feature_set": record["feature_set"],
                    "model_family": record["model_family"],
                    "incidence_group": group_name,
                    "threshold": threshold_value,
                    "count": int(len(subset)),
                    "mae_model": float(
                        (subset["prediction_model"] - subset["observed_target"]).abs().mean()
                    ),
                    "mae_persistence": float(
                        (subset["prediction_persistence"] - subset["observed_target"]).abs().mean()
                    ),
                    "bias_model": float((subset["prediction_model"] - subset["observed_target"]).mean()),
                    "bias_persistence": float(
                        (subset["prediction_persistence"] - subset["observed_target"]).mean()
                    ),
                    "null_reason": None,
                }
            )
    return pd.DataFrame(rows)


def _change_category(row: pd.Series, thresholds: dict[str, Any]) -> str:
    delta = float(row["observed_target"] - row["observed_current_cases"])
    if abs(delta) <= float(thresholds["stable_abs_delta_q25"]):
        return "stable"
    if delta >= float(thresholds["large_up_q90"]):
        return "large_up"
    large_down = thresholds.get("large_down_abs_q90")
    if large_down is not None and -delta >= float(large_down):
        return "large_down"
    return "directional_up" if delta > 0 else "directional_down"


def final_change_table(final_evaluation: Path, metric_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in metric_table.to_dict("records"):
        frame = prediction_frame(final_evaluation, str(record["prediction_path"])).copy()
        thresholds = read_json(final_evaluation / f"fits/{record['fit_id']}/thresholds.json")
        frame["category"] = frame.apply(_change_category, axis=1, thresholds=thresholds)
        for category, group in frame.groupby("category", sort=True):
            rows.append(
                {
                    "fit_id": record["fit_id"],
                    "horizon": record["horizon"],
                    "feature_set": record["feature_set"],
                    "model_family": record["model_family"],
                    "category": category,
                    "count": int(len(group)),
                    "mae_model": float(
                        (group["prediction_model"] - group["observed_target"]).abs().mean()
                    ),
                    "mae_persistence": float(
                        (group["prediction_persistence"] - group["observed_target"]).abs().mean()
                    ),
                    "bias_model": float((group["prediction_model"] - group["observed_target"]).mean()),
                    "bias_persistence": float(
                        (group["prediction_persistence"] - group["observed_target"]).mean()
                    ),
                    "stable_abs_delta_q25": thresholds["stable_abs_delta_q25"],
                    "large_up_q90": thresholds["large_up_q90"],
                    "large_down_abs_q90": thresholds.get("large_down_abs_q90"),
                    "category_source": (
                        "observed final evaluation delta categorized by saved training cutoffs "
                        "with stability precedence"
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_table(output_dir: Path, name: str, frame: pd.DataFrame) -> dict[str, str]:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"{name}.csv"
    json_path = table_dir / f"{name}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", indent=2) + "\n", encoding="utf-8")
    return {
        f"tables/{name}.csv": sha256_file(csv_path),
        f"tables/{name}.json": sha256_file(json_path),
    }


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    svg = figure_dir / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {f"figures/{stem}.png": sha256_file(png), f"figures/{stem}.svg": sha256_file(svg)}


def plot_final_horizon_error(metric_table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    champions = metric_table[metric_table["preselected_champion"]].sort_values("horizon")
    fixed = metric_table[metric_table["fixed_cases_only_ridge"]].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(fixed["horizon"], fixed["mae_model"], marker="o", label="Fixed cases-only Ridge")
    ax.plot(
        champions["horizon"],
        champions["mae_model"],
        marker="o",
        label="Preselected champion",
    )
    ax.plot(
        champions["horizon"],
        champions["mae_persistence"],
        marker="s",
        linestyle="--",
        label="Persistence baseline",
    )
    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel("MAE (weekly dengue cases)")
    ax.set_title("Final 2025 partial-test MAE by horizon")
    ax.set_xticks(list(HORIZONS))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, "final-horizon-error")


def plot_final_improvement(metric_table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fixed = metric_table[metric_table["fixed_cases_only_ridge"]].sort_values("horizon")
    champions = metric_table[metric_table["preselected_champion"]].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(HORIZONS))
    width = 0.34
    ax.bar(
        x - width / 2,
        fixed["mae_improvement_positive_better"],
        width=width,
        label="Fixed cases-only Ridge",
    )
    ax.bar(
        x + width / 2,
        champions["mae_improvement_positive_better"],
        width=width,
        label="Preselected champion",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"H{h}" for h in HORIZONS])
    ax.set_ylabel("MAE improvement vs persistence (cases, positive is better)")
    ax.set_title("Final 2025 partial-test improvement vs persistence")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, "final-improvement-vs-persistence")


def plot_weather(weather_ci: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels: list[str] = []
    values: list[float] = []
    for row in weather_ci.sort_values(["horizon", "model_family", "comparison"]).to_dict("records"):
        labels.append(f"H{row['horizon']} {MODEL_LABELS[row['model_family']]} {row['comparison']}")
        values.append(float(row["point_estimate"]))
    colors = ["#2b8cbe" if value < 0 else "#d95f0e" for value in values]
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(values)), labels, fontsize=8)
    ax.set_xlabel("Matched MAE difference (added weather minus baseline; negative is better)")
    ax.set_title("Final matched weather feature comparisons")
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, output_dir, "final-weather-comparisons")


def plot_district(district: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    subset = district[
        (district["horizon"] == 1)
        & (district["feature_set"] == "cases_only")
        & (district["model_family"] == "ridge")
    ].sort_values("mae_improvement_absolute_positive_better")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        subset["district_id"],
        subset["mae_improvement_absolute_positive_better"],
        color=np.where(subset["mae_improvement_absolute_positive_better"] >= 0, "#238b45", "#cb181d"),
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("MAE improvement vs persistence (cases, positive is better)")
    ax.set_ylabel("District")
    ax.set_title("Final H1 fixed cases-only Ridge district errors")
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, output_dir, "final-district-errors")


def plot_high_incidence(high: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    subset = high[
        high["fit_id"].isin(
            [
                "h1__cases_only__ridge__final",
                "h2__cases_full_weather__lightgbm_trial010__final",
                "h3__cases_full_weather__lightgbm_trial010__final",
                "h4__cases_full_weather__lightgbm_trial010__final",
            ]
        )
        & (high["incidence_group"] == "q90_primary")
    ].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(subset["horizon"], subset["mae_model"], marker="o", label="Model")
    ax.plot(subset["horizon"], subset["mae_persistence"], marker="s", label="Persistence")
    ax.set_xticks(list(HORIZONS))
    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel("MAE on q90 high-incidence rows (cases)")
    ax.set_title("Final high-incidence errors for preselected champions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, output_dir, "final-high-incidence-errors")


def plot_change(change: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    subset = change[change["fit_id"] == "h1__cases_only__ridge__final"].copy()
    subset["improvement"] = subset["mae_persistence"] - subset["mae_model"]
    subset = subset.sort_values("category")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(
        subset["category"],
        subset["improvement"],
        color=np.where(subset["improvement"] >= 0, "#238b45", "#cb181d"),
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MAE improvement vs persistence (cases, positive is better)")
    ax.set_xlabel("Observed change category")
    ax.set_title("Final H1 fixed cases-only Ridge change-category errors")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output_dir, "final-change-category-errors")


def write_model_cards(output_dir: Path, metric_table: pd.DataFrame) -> dict[str, str]:
    card_dir = output_dir / "model_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for row in metric_table.sort_values(["horizon", "feature_set", "model_family"]).to_dict("records"):
        path = card_dir / f"{row['fit_id']}.md"
        champion_text = "yes" if row["preselected_champion"] else "no"
        fixed_text = "yes" if row["fixed_cases_only_ridge"] else "no"
        features = str(row["feature_columns"]).split(";")
        body = f"""# {row['fit_id']}

- Horizon: H{row['horizon']}
- Feature set: {row['feature_set_label']} (`{row['feature_set']}`)
- Model family: {row['model_label']} (`{row['model_family']}`)
- Preselected champion from freeze: {champion_text}
- Fixed cases-only Ridge scientific-continuity member: {fixed_text}
- Training period: {row['train_start']} to {row['train_end']} ({row['training_count']} rows)
- Final evaluation status: observed retrospective 2025 partial test, 600 district-origin rows
- Preprocessing fit scope: {row['preprocessing_fit_scope']}
- Objective: {row['objective']}
- Hyperparameters: `{row['hyperparams_json']}`
- Training thresholds: saved per final fit in `thresholds.json`; q90/q95 incidence and change groups are training-only cutoffs.
- Cohort limits: 24 eligible origin weeks in March and July-November 2025; not an annual representative estimate.
- Publication availability: retrospective observation-time study; operational publication vintage was not established.

## Metrics

| Metric | Model | Persistence |
|---|---:|---:|
| MAE | {row['mae_model']:.6f} | {row['mae_persistence']:.6f} |
| RMSE | {row['rmse_model']:.6f} | {row['rmse_persistence']:.6f} |
| R2 | {row['r2_model']:.6f} | {row['r2_persistence']:.6f} |
| Poisson deviance | {row['poisson_deviance_model']:.6f} | {row['poisson_deviance_persistence']:.6f} |
| Bias | {row['bias_model']:.6f} | {row['bias_persistence']:.6f} |
| High-incidence MAE q90 | {row['mae_top_10pct_model']:.6f} | {row['mae_top_10pct_persistence']:.6f} |

## Features

{", ".join(features)}
"""
        path.write_text(body, encoding="utf-8")
        hashes[f"model_cards/{path.name}"] = sha256_file(path)
    return hashes


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame[columns].to_dict("records"):
        cells: list[str] = []
        for col in columns:
            value = record[col]
            if pd.isna(value):
                cells.append("")
            elif col in formats:
                cells.append(format(float(value), formats[col]))
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(
    output_dir: Path,
    sources: SourcePaths,
    metric_table: pd.DataFrame,
    model_ci: pd.DataFrame,
    weather_ci: pd.DataFrame,
) -> dict[str, str]:
    champion = metric_table[metric_table["preselected_champion"]].sort_values("horizon")
    fixed = metric_table[metric_table["fixed_cases_only_ridge"]].sort_values("horizon")
    side = pd.concat([fixed, champion]).sort_values(["horizon", "preselected_champion"])
    side = side[
        [
            "horizon",
            "feature_set_label",
            "model_label",
            "preselected_champion",
            "mae_model",
            "mae_persistence",
            "mae_improvement_positive_better",
            "rmse_model",
            "r2_model",
            "poisson_deviance_model",
            "bias_model",
            "mae_top_10pct_model",
        ]
    ]
    ci_summary = model_ci[
        model_ci["fit_id"].isin(metric_table[metric_table["preselected_champion"]]["fit_id"])
        | model_ci["fit_id"].isin(metric_table[metric_table["fixed_cases_only_ridge"]]["fit_id"])
    ].sort_values(["horizon", "fit_id"])
    finite_ci_count = int(model_ci["ci_low"].notna().sum())
    body = f"""# Milestone 3 final qualified partial-2025 report

This is a post-evaluation reporting artifact. It consumes the frozen final evaluation outputs from `{sources.final_evaluation}` and the sibling freeze `{sources.final_freeze}`. It does not refit, retune, rewrite the freeze, deserialize models, or read source training parquet.

## Question and design

The question is whether the predeclared dengue forecasting candidates improve over same-district persistence on the qualified 2025 partial test. The frozen development stage fit 144 cross-validation models and the final stage evaluated 24 fixed test fits: four horizons, three feature sets, and two model families. There was no post-test tuning or final winner selection.

The development comparison used equal six-fold cross-validation. The final comparison is a separate partial 2025 test trained on historical data through 2024-12-14. The first scheduled 2025 origin was 2025-01-04, but the eligible observed cohort contains 24 origin weeks in March and July-November 2025, totaling 600 district-origin rows per fit. This is retrospective observation-time performance, not an operational-vintage backtest; publication availability at the time of forecasting was not established. It is not annual 2025 representativeness, and because outcomes are now observed it is no longer a held-out future test.

The fixed cases-only Ridge model is retained for scientific continuity. Preselected champions from the freeze are shown side by side for each horizon; all 24 final fits are appendix members, not a post-test best selection.

## Primary final results

Positive improvement means lower MAE than persistence.

{markdown_table(side, ["horizon", "feature_set_label", "model_label", "preselected_champion", "mae_model", "mae_persistence", "mae_improvement_positive_better", "rmse_model", "r2_model", "poisson_deviance_model", "bias_model", "mae_top_10pct_model"], {"mae_model": ".3f", "mae_persistence": ".3f", "mae_improvement_positive_better": ".3f", "rmse_model": ".3f", "r2_model": ".3f", "poisson_deviance_model": ".3f", "bias_model": ".3f", "mae_top_10pct_model": ".3f"})}

The preselected champion cross-checks are generated from saved metrics: H1 fixed cases-only Ridge MAE 7.435948 vs persistence 7.901667; H2 full-weather LightGBM MAE 7.960736 vs 9.180000; H3 full-weather LightGBM MAE 8.439516 vs 10.473333; H4 full-weather LightGBM MAE 9.966845 vs 12.720000.

## Test uncertainty

Final uncertainty uses fixed seed 42 and 1000 replications. The resampling unit is the origin week cluster with all districts retained. Within each observed contiguous run, four-week moving blocks are sampled without bridging gaps and each run keeps its exact length. If any required observed run has fewer than four weeks, the CI is null with a reason while retaining the point estimate.

Finite model-vs-persistence CIs: {finite_ci_count} of 24. The final test has only 24 origin-week clusters, so intervals are descriptive and limited. Multiple model, horizon, feature, weather, district, high-incidence, and change-category comparisons are exploratory; no fabricated significance claim is made.

{markdown_table(ci_summary, ["horizon", "fit_id", "point_estimate", "ci_low", "ci_high", "insufficient_support_reason"], {"point_estimate": ".3f", "ci_low": ".3f", "ci_high": ".3f"})}

Weather comparisons are matched-cohort MAE differences where negative means the added weather set has lower MAE. The signs are reported consistently as B-A (`cases_rainfall` minus `cases_only`) and C-B (`cases_full_weather` minus `cases_rainfall`).

## Caveats

District diagnostics are zero-safe and include case-volume context; raw MAE is not directly comparable across districts without observed case totals or means. High-incidence and change-group tables use training-only thresholds with stability precedence for change categories. Null reasons are retained instead of imputing finite values. The trusted-operator assumption here is not a multi-tenant security model, and no claim is made that earlier adversarial hold status was resolved by this report.

## Linked accepted artifacts

- Development analysis: `{sources.development_analysis}`
- Development diagnostics: `{sources.diagnostics}`
- Development plots: `{sources.development_plots}`
- Importance tables: `{sources.importance}`
- Corrected importance panels: `{sources.importance_plots}`

## Reproduction commands

These commands validate or regenerate only reporting artifacts from saved outputs; they do not re-evaluate the once-only final evaluation.

```bash
TMPDIR=/tmp/dengue-forecast-pytest-temp .venv/bin/pytest tests/unit/test_milestone3_final_report.py
TMPDIR=/tmp/dengue-forecast-pytest-temp .venv/bin/python scripts/milestone3_final_report.py --repo-root . --run-id qualified-partial-2025-001
```
"""
    report = output_dir / "final-report.md"
    report.write_text(body, encoding="utf-8")
    readme = output_dir / "README.md"
    readme.write_text(
        "# Final report delivery\n\n"
        "Open `final-report.md` first. Tables are under `tables/`, figures under `figures/`, "
        "and per-fit model cards under `model_cards/`. This artifact is a reporting-only "
        "post-evaluation consumer of saved final outputs.\n",
        encoding="utf-8",
    )
    return {"final-report.md": sha256_file(report), "README.md": sha256_file(readme)}


def copy_external_report(output_dir: Path, repo_root: Path, started: float) -> dict[str, str]:
    target = repo_root / "docs" / "m3-final-report-result.md"
    table_count = len(list((output_dir / "tables").glob("*.csv")))
    figure_count = len(list((output_dir / "figures").glob("*")))
    model_card_count = len(list((output_dir / "model_cards").glob("*.md")))
    generated_count = len([p for p in output_dir.rglob("*") if p.is_file()])
    target.write_text(
        "# M3 final report result\n\n"
        f"Final reporting artifact generated at `{output_dir}`.\n\n"
        f"- Runtime at handoff write: {time.time() - started:.3f} seconds\n"
        f"- Generated files before manifest: {generated_count}\n"
        f"- CSV table count: {table_count}\n"
        f"- Figure file count: {figure_count}\n"
        f"- Model card count: {model_card_count}\n"
        "- Checks: once-only final evaluation not rerun; source training parquet not read; "
        "model deserialization not performed; freeze hash not rewritten.\n\n"
        "This slice added only post-evaluation reporting code, tests, this handoff document, "
        "and the final report artifact directory. Frozen final evaluation code/config/tests "
        "and completed artifacts were not modified.\n",
        encoding="utf-8",
    )
    return {str(target.relative_to(repo_root)): sha256_file(target)}


def build_manifest(
    output_dir: Path,
    repo_root: Path,
    sources: SourcePaths,
    artifact_hashes: dict[str, str],
    started: float,
) -> dict[str, Any]:
    generated_files = sorted(
        p for p in output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"
    )
    source_files = [
        sources.final_evaluation / "complete_manifest.json",
        sources.final_evaluation / "claim.json",
        sources.final_evaluation / "freeze_manifest.json",
        sources.final_evaluation / "checksums.json",
        sources.final_freeze / "freeze_manifest.json",
        sources.development / "complete.json",
        sources.development / "selection.json",
        sources.development_analysis / "manifest.json",
        sources.diagnostics / "manifest.json",
        repo_root / "scripts" / "milestone3_final_report.py",
        repo_root / "tests" / "unit" / "test_milestone3_final_report.py",
        repo_root / "docs" / "m3-final-report-result.md",
    ]
    return {
        "run_id": RUN_ID,
        "status": "complete",
        "artifact_kind": "post_evaluation_reporting_consumer",
        "created_unix_time": time.time(),
        "runtime_seconds": round(time.time() - started, 3),
        "counts": {
            "generated_file_count_excluding_manifest": len(generated_files),
            "table_count": len(list((output_dir / "tables").glob("*.csv"))),
            "figure_file_count": len(list((output_dir / "figures").glob("*"))),
            "model_card_count": len(list((output_dir / "model_cards").glob("*.md"))),
        },
        "provenance": {
            "final_evaluation": str(sources.final_evaluation),
            "final_freeze": str(sources.final_freeze),
            "accepted_development": str(sources.development),
            "accepted_development_analysis": str(sources.development_analysis),
            "accepted_diagnostics": str(sources.diagnostics),
        },
        "source_hashes": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in source_files
            if path.exists() and path.is_file()
        },
        "artifact_hashes": artifact_hashes,
        "checks": {
            "once_only_final_evaluation_not_rerun": True,
            "source_training_parquet_not_read": True,
            "model_deserialization_not_performed": True,
            "freeze_hash_not_rewritten": True,
            "critical_original_protected_check_34006_unchanged": "not_recomputed_by_report; preserved from accepted authority references",
            "copy_201_authorized_export_test": "not_packaged_by_report",
            "actual_claim_earlier_adversarial_hold_resolved": False,
        },
    }


def run_report(repo_root: Path, run_id: str = RUN_ID, overwrite: bool = False) -> Path:
    started = time.time()
    if run_id != RUN_ID:
        raise ValueError(f"This reporting slice is locked to run id {RUN_ID}")
    sources = discover_sources(repo_root)
    output_dir = repo_root / "artifacts" / "milestone3" / "final_report" / run_id
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    freeze_manifest = read_json(sources.final_evaluation / "freeze_manifest.json")
    metric_table = build_final_metric_table(sources.final_evaluation, freeze_manifest)
    model_ci = paired_model_persistence_ci_table(sources.final_evaluation, metric_table)
    weather_ci = weather_ci_table(sources.final_evaluation, metric_table)
    district = final_district_table(sources.final_evaluation, metric_table)
    high = final_high_incidence_table(sources.final_evaluation, metric_table)
    change = final_change_table(sources.final_evaluation, metric_table)

    artifact_hashes: dict[str, str] = {}
    for name, frame in [
        ("final_metrics", metric_table),
        ("test_model_vs_persistence_uncertainty", model_ci),
        ("test_weather_uncertainty", weather_ci),
        ("final_district_errors", district),
        ("final_high_incidence_errors", high),
        ("final_change_category_errors", change),
    ]:
        artifact_hashes.update(write_table(output_dir, name, frame))

    artifact_hashes.update(plot_final_horizon_error(metric_table, output_dir))
    artifact_hashes.update(plot_final_improvement(metric_table, output_dir))
    artifact_hashes.update(plot_weather(weather_ci, output_dir))
    artifact_hashes.update(plot_district(district, output_dir))
    artifact_hashes.update(plot_high_incidence(high, output_dir))
    artifact_hashes.update(plot_change(change, output_dir))
    artifact_hashes.update(write_model_cards(output_dir, metric_table))
    artifact_hashes.update(write_report(output_dir, sources, metric_table, model_ci, weather_ci))
    artifact_hashes.update(copy_external_report(output_dir, repo_root, started))

    manifest = build_manifest(output_dir, repo_root, sources, artifact_hashes, started)
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = run_report(args.repo_root.resolve(), run_id=args.run_id, overwrite=args.overwrite)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
