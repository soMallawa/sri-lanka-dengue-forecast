from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.modeling.evaluate import metric_bundle
from dengue_forecast.modeling.holdout import BASELINE_NAMES

KEY_COLUMNS = ["fold", "district_id", "week_start_date"]


class ComparisonError(ValueError):
    """Raised when validation and baseline evidence cannot be paired exactly."""


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ComparisonError(f"{label} missing required columns: {missing}")


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fold"] = out["fold"].astype(str)
    out["district_id"] = out["district_id"].astype(str)
    out["week_start_date"] = pd.to_datetime(out["week_start_date"]).dt.date.astype(str)
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    return out


def _thresholds_by_fold(model: pd.DataFrame) -> dict[str, tuple[float, float]]:
    thresholds: dict[str, tuple[float, float]] = {}
    for fold, group in model.groupby("fold", sort=True):
        q90 = pd.to_numeric(group["threshold90"], errors="coerce").dropna().unique()
        q95 = pd.to_numeric(group["threshold95"], errors="coerce").dropna().unique()
        if len(q90) != 1 or len(q95) != 1:
            raise ComparisonError(f"Model thresholds are not unique for fold {fold}")
        thresholds[str(fold)] = (float(q90[0]), float(q95[0]))
    return thresholds


def _metric_row(
    *,
    support_type: str,
    fold: str,
    series_role: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    q90: float,
    q95: float,
    total: int,
    districts: int,
    baseline_name: str | None = None,
) -> dict[str, Any]:
    metrics = metric_bundle(y_true, y_pred, threshold90=q90, threshold95=q95)
    return {
        "support_type": support_type,
        "fold": fold,
        "series_role": series_role,
        "model": model_name,
        "baseline": baseline_name or "",
        "total": int(total),
        "n": int(metrics["n"]),
        "districts": int(districts),
        "coverage": float(metrics["n"]) / float(total) if total else 0.0,
        **metrics,
    }


def build_shared_metric_comparison(
    *,
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    selected_config_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare one selected CV config with every baseline on shared row support."""
    _require_columns(
        model_predictions,
        {
            "config_id",
            "fold",
            "district_id",
            "week_start_date",
            "target",
            "prediction",
            "threshold90",
            "threshold95",
        },
        "model predictions",
    )
    _require_columns(
        baseline_predictions,
        {"config_id", "fold", "district_id", "week_start_date", "target", "prediction"},
        "baseline predictions",
    )
    model = _normalise(
        model_predictions[
            model_predictions["config_id"].astype(str).eq(str(selected_config_id))
        ].copy()
    )
    baselines = _normalise(baseline_predictions.copy())
    if model.empty:
        raise ComparisonError(f"No predictions found for selected config {selected_config_id}")
    if model.duplicated(KEY_COLUMNS).any():
        raise ComparisonError("Selected model predictions contain duplicate row keys")
    if baselines.duplicated(["config_id", *KEY_COLUMNS]).any():
        raise ComparisonError("Baseline predictions contain duplicate baseline row keys")
    observed = set(baselines["config_id"].dropna().astype(str))
    if observed != set(BASELINE_NAMES):
        raise ComparisonError(
            "Baseline predictions must cover exactly all five canonical baselines"
        )
    thresholds = _thresholds_by_fold(model)
    model_keys = set(map(tuple, model[KEY_COLUMNS].to_numpy()))
    rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for baseline_name in BASELINE_NAMES:
        baseline = baselines[baselines["config_id"].astype(str).eq(baseline_name)].copy()
        baseline_keys = set(map(tuple, baseline[KEY_COLUMNS].to_numpy()))
        if baseline_keys != model_keys:
            raise ComparisonError(f"Baseline {baseline_name} row keys do not match selected model")
        merged = model.merge(
            baseline,
            on=KEY_COLUMNS,
            how="inner",
            validate="1:1",
            suffixes=("_model", "_baseline"),
        )
        target_delta = (
            merged["target_model"].astype("float64") - merged["target_baseline"].astype("float64")
        ).abs()
        if bool(target_delta.gt(1e-12).any()):
            raise ComparisonError(f"Baseline {baseline_name} target values do not match model")
        for fold, group in merged.groupby("fold", sort=True):
            q90, q95 = thresholds[str(fold)]
            target = group["target_model"].astype("float64")
            model_pred = group["prediction_model"].astype("float64")
            baseline_pred = group["prediction_baseline"].astype("float64")
            available = (
                target.notna()
                & model_pred.notna()
                & baseline_pred.notna()
                & np.isfinite(target)
                & np.isfinite(model_pred)
                & np.isfinite(baseline_pred)
            )
            pair = group.loc[available]
            total = len(group)
            districts = int(pair["district_id"].nunique())
            model_name = str(
                group["model_model"].iloc[0] if "model_model" in group else group["model"].iloc[0]
            )
            rows.append(
                _metric_row(
                    support_type="pairwise_baseline_available",
                    fold=str(fold),
                    series_role="model",
                    model_name=model_name,
                    baseline_name=baseline_name,
                    y_true=pair["target_model"],
                    y_pred=pair["prediction_model"],
                    q90=q90,
                    q95=q95,
                    total=total,
                    districts=districts,
                )
            )
            rows.append(
                _metric_row(
                    support_type="pairwise_baseline_available",
                    fold=str(fold),
                    series_role="baseline",
                    model_name=baseline_name,
                    baseline_name=baseline_name,
                    y_true=pair["target_model"],
                    y_pred=pair["prediction_baseline"],
                    q90=q90,
                    q95=q95,
                    total=total,
                    districts=districts,
                )
            )
    paired = pd.DataFrame(rows)
    all_joined = model.rename(
        columns={"target": "target_model", "prediction": "prediction_model", "model": "model_model"}
    )
    for baseline_name in BASELINE_NAMES:
        baseline = baselines[baselines["config_id"].astype(str).eq(baseline_name)]
        all_joined = all_joined.merge(
            baseline[KEY_COLUMNS + ["target", "prediction"]].rename(
                columns={
                    "target": f"target_{baseline_name}",
                    "prediction": f"prediction_{baseline_name}",
                }
            ),
            on=KEY_COLUMNS,
            how="inner",
            validate="1:1",
        )
    for fold, group in all_joined.groupby("fold", sort=True):
        q90, q95 = thresholds[str(fold)]
        common = (
            group["target_model"].notna()
            & group["prediction_model"].notna()
            & np.isfinite(group["target_model"].astype("float64"))
            & np.isfinite(group["prediction_model"].astype("float64"))
        )
        for baseline_name in BASELINE_NAMES:
            pred_col = f"prediction_{baseline_name}"
            target_col = f"target_{baseline_name}"
            target_delta = (
                group["target_model"].astype("float64") - group[target_col].astype("float64")
            ).abs()
            if bool(target_delta.gt(1e-12).any()):
                raise ComparisonError(f"Baseline {baseline_name} target values do not match model")
            common &= group[pred_col].notna() & np.isfinite(group[pred_col].astype("float64"))
        support = group.loc[common]
        total = len(group)
        districts = int(support["district_id"].nunique())
        common_rows.append(
            _metric_row(
                support_type="secondary_all_five_common_intersection",
                fold=str(fold),
                series_role="model",
                model_name=str(group["model_model"].iloc[0]),
                y_true=support["target_model"],
                y_pred=support["prediction_model"],
                q90=q90,
                q95=q95,
                total=total,
                districts=districts,
            )
        )
        for baseline_name in BASELINE_NAMES:
            common_rows.append(
                _metric_row(
                    support_type="secondary_all_five_common_intersection",
                    fold=str(fold),
                    series_role="baseline",
                    model_name=baseline_name,
                    baseline_name=baseline_name,
                    y_true=support["target_model"],
                    y_pred=support[f"prediction_{baseline_name}"],
                    q90=q90,
                    q95=q95,
                    total=total,
                    districts=districts,
                )
            )
    return paired, pd.DataFrame(common_rows)


def write_shared_metric_comparison(
    *,
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    selected_config_id: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    paired, common = build_shared_metric_comparison(
        model_predictions=model_predictions,
        baseline_predictions=baseline_predictions,
        selected_config_id=selected_config_id,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired_path = out / "paired_baseline_comparison.csv"
    common_path = out / "common_support_comparison.csv"
    paired.to_csv(paired_path, index=False)
    common.to_csv(common_path, index=False)
    return {"paired_baseline_comparison": paired_path, "common_support_comparison": common_path}


__all__ = [
    "ComparisonError",
    "build_shared_metric_comparison",
    "write_shared_metric_comparison",
]
