from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_forecast.contracts import ContractError, validate_ml_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_training_dataset.parquet"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "reports" / "feature_registry.csv"

TARGET_COLUMN = "cases_next_week"
FUTURE_TARGET_COLUMNS = {
    "cases_next_week",
    "cases_next_2w",
    "cases_next_4w",
    "incidence_next_week_per_100k",
}
MODELING_ONLY_ELIGIBLE = {"district_id", "area_km2"}
FORBIDDEN_CONTEXT_COLUMNS = {
    "population_reference",
    "population_reference_year",
    "population_method",
    "population_density_per_km2",
    "incidence_per_100k_using_2024_population",
    "centroid_lat",
    "centroid_lon",
    "province_name",
}
FEATURE_SET_NAMES = (
    "cases_only",
    "cases_rainfall",
    "cases_full_weather",
    "full_context",
)

_REQUIRED_REGISTRY_COLUMNS = {
    "feature_name",
    "feature_group",
    "dtype",
    "uses_future_information",
    "eligible_for_training",
}
_KNOWN_FEATURE_GROUPS = {
    "epidemiology",
    "identifier",
    "metadata",
    "quality",
    "reference",
    "seasonality",
    "target",
    "weather",
}


def get_target_column() -> str:
    """Return the explicit one-week-ahead modeling target."""
    return TARGET_COLUMN


def _parse_bool(value: object, *, column: str, feature: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ContractError(
        f"Feature registry column {column} for {feature} must contain typed boolean values"
    )


def _normalize_registry(registry: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing:
        raise ContractError(f"Feature registry missing columns: {sorted(missing)}")
    out = registry.copy()
    out["feature_name"] = out["feature_name"].astype(str)
    if out["feature_name"].duplicated().any():
        duplicates = sorted(out.loc[out["feature_name"].duplicated(), "feature_name"].unique())
        raise ContractError(f"Feature registry has duplicate feature names: {duplicates}")
    for column in ["uses_future_information", "eligible_for_training"]:
        out[column] = [
            _parse_bool(value, column=column, feature=feature)
            for value, feature in zip(out[column], out["feature_name"], strict=True)
        ]

    selected = out[out["eligible_for_training"]]
    future_selected = selected[selected["uses_future_information"]]["feature_name"].tolist()
    if future_selected:
        raise ContractError(
            f"Registry marks future-information columns trainable: {future_selected}"
        )
    future_targets = sorted(set(FUTURE_TARGET_COLUMNS) & set(selected["feature_name"]))
    if future_targets:
        raise ContractError(f"Registry marks target columns trainable: {future_targets}")
    unknown_trainable_groups = selected[
        ~selected["feature_group"].astype(str).isin(_KNOWN_FEATURE_GROUPS)
    ]["feature_name"].tolist()
    if unknown_trainable_groups:
        raise ContractError(
            f"Registry marks unknown feature groups trainable: {unknown_trainable_groups}"
        )
    return out


def load_modeling_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> pd.DataFrame:
    registry = pd.read_csv(path)
    return _normalize_registry(registry)


def _registry_lookup(registry: pd.DataFrame) -> dict[str, dict[str, Any]]:
    normalized = _normalize_registry(registry)
    return normalized.set_index("feature_name").to_dict("index")


def _modeling_eligible_features(registry: pd.DataFrame) -> set[str]:
    lookup = _registry_lookup(registry)
    eligible = {
        name
        for name, row in lookup.items()
        if bool(row["eligible_for_training"]) and not bool(row["uses_future_information"])
    }
    for name in MODELING_ONLY_ELIGIBLE:
        if name not in lookup:
            raise ContractError(f"Modeling-only eligible feature {name} is absent from registry")
        if bool(lookup[name]["uses_future_information"]):
            raise ContractError(f"Modeling-only eligible feature {name} uses future information")
        eligible.add(name)

    forbidden = sorted((eligible & FORBIDDEN_CONTEXT_COLUMNS) | (eligible & FUTURE_TARGET_COLUMNS))
    if forbidden:
        raise ContractError(f"Forbidden modeling features selected: {forbidden}")
    return eligible


def get_feature_columns(registry: pd.DataFrame | None = None) -> list[str]:
    """Return all modeling-eligible features after the approved Stage 1 overlay."""
    registry = load_modeling_registry() if registry is None else registry
    eligible = _modeling_eligible_features(registry)
    return [name for name in registry["feature_name"].astype(str).tolist() if name in eligible]


def get_feature_set(name: str, registry: pd.DataFrame | None = None) -> list[str]:
    registry = load_modeling_registry() if registry is None else _normalize_registry(registry)
    if name not in FEATURE_SET_NAMES:
        raise ContractError(f"Unknown feature set {name!r}; expected one of {FEATURE_SET_NAMES}")

    lookup = _registry_lookup(registry)
    eligible = _modeling_eligible_features(registry)
    selected: list[str] = []
    for feature in registry["feature_name"].astype(str).tolist():
        group = str(lookup[feature]["feature_group"])
        if feature not in eligible:
            continue
        include = False
        if name in FEATURE_SET_NAMES:
            include = group in {"epidemiology", "seasonality"} or feature == "district_id"
        if name in {"cases_rainfall", "cases_full_weather", "full_context"}:
            include = include or _is_rainfall_feature(feature)
        if name in {"cases_full_weather", "full_context"}:
            include = include or (group == "weather")
        if name == "full_context":
            include = include or feature == "area_km2"
        if include:
            selected.append(feature)

    _assert_features_allowed(selected, registry)
    return selected


def _is_rainfall_feature(feature: str) -> bool:
    return feature.startswith("rainfall_") or feature.startswith("rain_days_")


def _assert_features_allowed(features: Iterable[str], registry: pd.DataFrame) -> None:
    lookup = _registry_lookup(registry)
    feature_set = set(features)
    unknown = sorted(feature_set - set(lookup))
    if unknown:
        raise ContractError(f"Unknown feature columns requested: {unknown}")
    future = sorted(
        feature
        for feature in feature_set
        if feature in FUTURE_TARGET_COLUMNS or bool(lookup[feature]["uses_future_information"])
    )
    if future:
        raise ContractError(f"Future or target columns are not allowed in X: {future}")
    eligible = _modeling_eligible_features(registry)
    ineligible = sorted(feature_set - eligible)
    if ineligible:
        raise ContractError(f"Feature columns are not modeling-eligible: {ineligible}")
    forbidden = sorted(feature_set & FORBIDDEN_CONTEXT_COLUMNS)
    if forbidden:
        raise ContractError(f"Forbidden context columns are not allowed in X: {forbidden}")


def load_modeling_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
    registry: pd.DataFrame | None = None,
    *,
    production: bool = True,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    registry = load_modeling_registry() if registry is None else registry
    return validate_modeling_dataset(df, registry=registry, production=production)


def validate_modeling_dataset(
    df: pd.DataFrame,
    registry: pd.DataFrame | None = None,
    *,
    production: bool = False,
) -> pd.DataFrame:
    registry = load_modeling_registry() if registry is None else _normalize_registry(registry)
    out = validate_ml_dataset(df).copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"])
    out["week_end_date"] = pd.to_datetime(out["week_end_date"])
    out = out.sort_values(["district_id", "week_start_date"]).reset_index(drop=True)

    if production and out["district_id"].nunique() != 25:
        district_count = out["district_id"].nunique()
        raise ContractError(
            f"Production modeling dataset must contain 25 districts, got {district_count}"
        )
    if out.duplicated(["district_id", "week_start_date"]).any():
        raise ContractError("Modeling dataset has duplicate district-week rows")

    _validate_exact_week_grids(out)
    _validate_target_alignment(out)
    _validate_lag_alignment(out, prefix="cases_lag_", source_col="dengue_cases")
    _validate_lag_alignment(out, prefix="rainfall_lag_", source_col="rainfall_sum_mm")
    _validate_lag_alignment(out, prefix="temp_mean_lag_", source_col="temp_mean_c")
    _validate_lag_alignment(out, prefix="humidity_lag_", source_col="humidity_mean_pct")

    missing_registry = sorted(set(out.columns) - set(registry["feature_name"].astype(str)))
    if missing_registry:
        raise ContractError(f"Dataset columns missing from feature registry: {missing_registry}")
    return out


def _validate_exact_week_grids(df: pd.DataFrame) -> None:
    for district_id, group in df.groupby("district_id", sort=False):
        starts = group["week_start_date"].sort_values()
        if starts.duplicated().any() or not starts.is_monotonic_increasing:
            raise ContractError(f"District {district_id} week grid is not sorted and unique")
        deltas = starts.diff().dropna().dt.days
        if not deltas.eq(7).all():
            bad = deltas[deltas.ne(7)].head().tolist()
            raise ContractError(f"District {district_id} does not have an exact 7-day grid: {bad}")


def _values_equal_or_both_missing(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def _validate_target_alignment(df: pd.DataFrame) -> None:
    lookup = df.set_index(["district_id", "week_start_date"])["dengue_cases"].to_dict()
    for row in df.itertuples(index=False):
        target_date = row.week_start_date + pd.Timedelta(days=7)
        expected = lookup.get((row.district_id, target_date), pd.NA)
        actual = getattr(row, TARGET_COLUMN)
        if not _values_equal_or_both_missing(actual, expected):
            raise ContractError(
                "Target alignment failed: cases_next_week must equal same-district cases "
                f"for the week beginning {target_date.date()}"
            )


def _validate_lag_alignment(df: pd.DataFrame, *, prefix: str, source_col: str) -> None:
    if source_col not in df.columns:
        return
    lag_columns = [column for column in df.columns if column.startswith(prefix)]
    if not lag_columns:
        return
    lookup = df.set_index(["district_id", "week_start_date"])[source_col].to_dict()
    for column in lag_columns:
        lag_weeks = int(column.removeprefix(prefix))
        for row in df.itertuples(index=False):
            lag_date = row.week_start_date - pd.Timedelta(days=7 * lag_weeks)
            expected = lookup.get((row.district_id, lag_date), pd.NA)
            actual = getattr(row, column)
            if not _values_equal_or_both_missing(actual, expected):
                raise ContractError(
                    f"Lag alignment failed for {column}: expected same-district value "
                    f"from {lag_date.date()}"
                )


def make_modeling_matrix(
    df: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    feature_set: str = "full_context",
    trainable_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features = get_feature_set(feature_set, registry)
    _assert_features_allowed(features, registry)
    required = set(features) | {
        TARGET_COLUMN,
        "district_id",
        "week_start_date",
        "week_end_date",
        "is_trainable",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ContractError(f"Modeling dataset missing required columns: {missing}")
    frame = df.copy()
    if trainable_only:
        frame = frame[frame["is_trainable"] & frame[TARGET_COLUMN].notna()].copy()
    X = frame[features].copy()
    if set(FUTURE_TARGET_COLUMNS) & set(X.columns):
        raise ContractError("Target columns are not allowed in X")
    y = frame[TARGET_COLUMN].copy()
    meta = frame[["district_id", "week_start_date", "week_end_date", "year", "week"]].copy()
    return X, y, meta
