from __future__ import annotations

import json
from typing import Any

import pandas as pd

CONTROLLED_FEATURE_SETS = (
    "cases_only",
    "cases_rainfall",
    "cases_full_weather",
    "full_context",
)
FEATURE_SET_LABELS = {
    "cases_only": "A_cases_only",
    "cases_rainfall": "B_cases_rainfall",
    "cases_full_weather": "C_cases_full_weather",
    "full_context": "D_full_context",
}
PROVENANCE_COLUMNS = (
    "model_config",
    "hyperparams",
    "params",
    "model_hyperparams",
    "seed",
)
AGGREGATED_ABLATION_COLUMNS = (
    "model",
    "objective",
    "config_variant",
    "config_id",
    "ablation_step",
    "feature_set",
    "fold_count",
    "mean_mae",
    "median_mae",
    "std_mae",
    "fold_wins_vs_cases_only",
    "rainfall_incremental_vs_A",
    "full_weather_incremental_vs_B",
    "area_context_incremental_vs_C",
    "relative_improvement_pct_vs_cases_only",
    "relative_effect_pct_vs_cases_only",
    "interpretation_policy",
)


def aggregate_ablation_results(
    metrics: pd.DataFrame,
    *,
    required_feature_sets: tuple[str, ...] | None = None,
    expected_fold_count: int | None = None,
) -> pd.DataFrame:
    required = {"model", "feature_set", "fold", "mae"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Ablation metrics missing columns: {sorted(missing)}")
    screen = controlled_ablation_fold_metrics(
        metrics,
        required_feature_sets=required_feature_sets,
        expected_fold_count=expected_fold_count,
    )
    rows: list[dict[str, Any]] = []
    group_columns = ["model", "objective", "config_variant"]
    for keys, group in screen.groupby(group_columns, dropna=False, sort=True):
        model, objective, config_variant = keys
        baseline = group[group["feature_set"].eq("cases_only")].set_index("fold")["mae"]
        baseline_mean = float(baseline.mean()) if not baseline.empty else float("nan")
        incremental_deltas = _variant_incremental_deltas(group)
        for feature_set, feature_group in group.groupby("feature_set", dropna=False, sort=True):
            maes = feature_group.set_index("fold")["mae"].astype("float64")
            common = maes.index.intersection(baseline.index)
            fold_wins = int((maes.loc[common] < baseline.loc[common]).sum()) if len(common) else 0
            mean_mae = float(maes.mean())
            rows.append(
                {
                    "model": model,
                    "objective": objective,
                    "config_variant": config_variant,
                    "config_id": _single_value(feature_group, "config_id"),
                    "ablation_step": FEATURE_SET_LABELS.get(str(feature_set), str(feature_set)),
                    "feature_set": feature_set,
                    "fold_count": int(maes.index.nunique()),
                    "mean_mae": mean_mae,
                    "median_mae": float(maes.median()),
                    "std_mae": float(maes.std(ddof=0)),
                    "fold_wins_vs_cases_only": fold_wins,
                    "rainfall_incremental_vs_A": incremental_deltas[
                        "rainfall_incremental_vs_A"
                    ],
                    "full_weather_incremental_vs_B": incremental_deltas[
                        "full_weather_incremental_vs_B"
                    ],
                    "area_context_incremental_vs_C": incremental_deltas[
                        "area_context_incremental_vs_C"
                    ],
                    "relative_improvement_pct_vs_cases_only": _relative_improvement_pct(
                        baseline_mean, mean_mae
                    ),
                    "relative_effect_pct_vs_cases_only": _relative_improvement_pct(
                        baseline_mean, mean_mae
                    ),
                    "interpretation_policy": "associational_validation_only_no_causality",
                }
            )
    return pd.DataFrame(rows, columns=AGGREGATED_ABLATION_COLUMNS)


def controlled_ablation_fold_metrics(
    metrics: pd.DataFrame,
    *,
    required_feature_sets: tuple[str, ...] | None = None,
    expected_fold_count: int | None = None,
) -> pd.DataFrame:
    screen = _screen_metrics(metrics).copy()
    if screen.empty:
        raise ValueError("Ablation metrics contain no screen-stage rows")
    if "objective" not in screen.columns:
        screen["objective"] = "unspecified"
    if "config_id" not in screen.columns:
        screen["config_id"] = screen.apply(
            lambda row: f"{row['model']}__{row['objective']}__{row['feature_set']}", axis=1
        )
    screen["config_variant"] = screen.apply(_config_variant, axis=1)
    _validate_ablation_identities(
        screen,
        required_feature_sets=required_feature_sets,
        expected_fold_count=expected_fold_count,
    )
    screen["ablation_step"] = screen["feature_set"].astype(str).map(
        lambda feature_set: FEATURE_SET_LABELS.get(feature_set, feature_set)
    )
    return screen.sort_values(
        ["model", "objective", "config_variant", "feature_set", "fold"],
        kind="stable",
    ).reset_index(drop=True)


def _screen_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if "stage" not in metrics.columns:
        return metrics.copy()
    return metrics.loc[metrics["stage"].astype(str).eq("screen")].copy()


def _config_variant(row: pd.Series) -> str:
    config_id = str(row["config_id"])
    feature_set = str(row["feature_set"])
    suffix = f"__{feature_set}"
    if config_id.endswith(suffix):
        return config_id[: -len(suffix)]
    return config_id


def _validate_ablation_identities(
    metrics: pd.DataFrame,
    *,
    required_feature_sets: tuple[str, ...] | None,
    expected_fold_count: int | None,
) -> None:
    identity = ["model", "objective", "config_variant", "feature_set", "fold"]
    duplicate = metrics.loc[metrics.duplicated(identity, keep=False), identity + ["config_id"]]
    if not duplicate.empty:
        raise ValueError(
            "Ablation metrics contain duplicate variant-feature-fold identities: "
            f"{duplicate.drop_duplicates().to_dict('records')}"
        )

    for keys, group in metrics.groupby(["model", "objective", "config_variant"], dropna=False):
        model, objective, config_variant = keys
        _validate_provenance_columns(group, model, objective, config_variant)
        observed_features = set(group["feature_set"].astype(str))
        if required_feature_sets is not None:
            expected_features = set(required_feature_sets)
            if observed_features != expected_features:
                missing = sorted(expected_features - observed_features)
                extra = sorted(observed_features - expected_features)
                raise ValueError(
                    "Ablation variant does not contain the required controlled feature sets: "
                    f"model={model}, objective={objective}, config_variant={config_variant}, "
                    f"missing={missing}, extra={extra}"
                )
        fold_sets = {
            str(feature_set): set(feature_group["fold"].astype(str))
            for feature_set, feature_group in group.groupby("feature_set", dropna=False)
        }
        if not fold_sets:
            continue
        first_feature, first_folds = next(iter(fold_sets.items()))
        for feature_set, folds in fold_sets.items():
            if folds != first_folds:
                raise ValueError(
                    "Ablation variant feature sets must have identical paired fold coverage: "
                    f"model={model}, objective={objective}, config_variant={config_variant}, "
                    f"{first_feature}={sorted(first_folds)}, {feature_set}={sorted(folds)}"
                )
        if expected_fold_count is not None and len(first_folds) != expected_fold_count:
            raise ValueError(
                "Ablation variant has incomplete fold coverage: "
                f"model={model}, objective={objective}, config_variant={config_variant}, "
                f"fold_count={len(first_folds)}, expected={expected_fold_count}"
            )


def _validate_provenance_columns(
    group: pd.DataFrame, model: Any, objective: Any, config_variant: Any
) -> None:
    for column in PROVENANCE_COLUMNS:
        if column not in group.columns:
            continue
        values = {_stable_value(value) for value in group[column].dropna()}
        if len(values) > 1:
            raise ValueError(
                "Ablation variant mixes different model provenance across feature sets: "
                f"model={model}, objective={objective}, config_variant={config_variant}, "
                f"column={column}"
            )


def _stable_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _single_value(group: pd.DataFrame, column: str) -> Any:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Ablation group has mixed {column} values: {values.astype(str).tolist()}"
        )
    return values.iloc[0]


def _variant_incremental_deltas(group: pd.DataFrame) -> dict[str, float]:
    return {
        "rainfall_incremental_vs_A": _paired_feature_delta(
            group, "cases_rainfall", "cases_only"
        ),
        "full_weather_incremental_vs_B": _paired_feature_delta(
            group, "cases_full_weather", "cases_rainfall"
        ),
        "area_context_incremental_vs_C": _paired_feature_delta(
            group, "full_context", "cases_full_weather"
        ),
    }


def _paired_feature_delta(group: pd.DataFrame, feature_set: str, comparator: str) -> float:
    current = group[group["feature_set"].eq(feature_set)].set_index("fold")["mae"]
    other = group[group["feature_set"].eq(comparator)].set_index("fold")["mae"]
    common = current.index.intersection(other.index)
    if len(common) == 0:
        return float("nan")
    return float((current.loc[common] - other.loc[common]).mean())


def _relative_improvement_pct(baseline_mean: float, mean_mae: float) -> float:
    if pd.isna(baseline_mean):
        return float("nan")
    if baseline_mean == 0.0:
        return 0.0 if mean_mae == 0.0 else float("-inf")
    return 100.0 * (baseline_mean - mean_mae) / baseline_mean


def write_ablation_artifacts(metrics: pd.DataFrame, *, reports_dir: str) -> pd.DataFrame:
    from pathlib import Path

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    fold_metrics = controlled_ablation_fold_metrics(
        metrics,
        required_feature_sets=CONTROLLED_FEATURE_SETS,
        expected_fold_count=6,
    )
    summary = aggregate_ablation_results(
        fold_metrics,
        required_feature_sets=CONTROLLED_FEATURE_SETS,
        expected_fold_count=6,
    )
    summary.to_csv(reports / "ablation_results.csv", index=False)
    summary.to_csv(reports / "ablation_summary.csv", index=False)
    fold_metrics.to_csv(reports / "ablation_fold_metrics.csv", index=False)
    lines = [
        "# Ablation Summary",
        "",
        "Validation-only comparison. These effects are predictive associations, not causal claims "
        "and do not establish decisive benefits from tiny differences.",
        "",
        "Cases-only is the reference baseline for each fixed screen-stage model variant; "
        "no-advantage or worse ML/weather extensions are valid outcomes.",
        "",
        "Incremental columns are fixed variant-level paired deltas repeated on every row: "
        "rainfall is B-A, full weather is C-B, and area context is D-C.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.model} / {row.objective} / {row.feature_set}: mean MAE {row.mean_mae:.4f}, "
            f"relative improvement vs cases_only {row.relative_improvement_pct_vs_cases_only:.2f}%."
        )
    (reports / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "AGGREGATED_ABLATION_COLUMNS",
    "aggregate_ablation_results",
    "controlled_ablation_fold_metrics",
    "write_ablation_artifacts",
]
