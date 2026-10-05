from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.features.merge import _add_lags, _add_rolling, _add_trends
from dengue_forecast.features.seasonality import add_seasonality_features

AUTHORIZATION_SHA256 = "01ec9af9b409bbb67f688f28da21703ff40f84e584e03fb9e962c50110d2e1a7"
FEATURE_REGISTRY_SHA256 = "7b8092ebeea00a7392ba2fbfdb5462ae49faf52ce28bb6b5ca7cd824b3c81475"
FIXED_PROTOCOL_CONFIG_SHA256 = "f75ca410a41d4914db658006966925b3fc725d5858922554269d28a22c83e613"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXED_PROTOCOL_CONFIG = REPO_ROOT / "configs" / "milestone3.json"
FIXED_FEATURE_REGISTRY = REPO_ROOT / "data" / "reports" / "feature_registry.csv"
FIXED_AUTHORIZATION = REPO_ROOT / "docs" / "m3-protocol-deviation-authorization.md"
DEFAULT_TRIAL010_METADATA = (
    REPO_ROOT
    / "artifacts"
    / "models"
    / "tune_lightgbm_trial_010__val_2014"
    / "metadata.json"
)
DEFAULT_TRIAL010_METADATA_SHA256 = (
    "c7c5cb8b60942d978203cf177076014fc5c6fbd816b322c42eb46a16da38862a"
)
HORIZONS = (1, 2, 3, 4)
EVALUATION_REQUIRED_FEATURES = (
    "dengue_cases",
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
)
M3_FEATURE_SETS = ("cases_only", "cases_rainfall", "cases_full_weather")
DEVELOPMENT_FOLDS = (2014, 2015, 2016, 2018, 2019, 2020)
CALENDAR_2025_BOUNDARY = pd.Timestamp("2025-01-01")
FORBIDDEN_FEATURES = {
    "area_km2",
    "population_reference",
    "population_reference_year",
    "population_method",
    "population_density_per_km2",
    "incidence_per_100k_using_2024_population",
    "incidence_next_week_per_100k",
    "centroid_lat",
    "centroid_lon",
    "province_name",
    "cases_next_week",
    "cases_next_2w",
    "cases_next_4w",
    "target_h1",
    "target_h2",
    "target_h3",
    "target_h4",
}

PINNED_FEATURE_SETS: dict[str, list[str]] = {
    "cases_only": [
        "cases_change_1w",
        "cases_change_2w",
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",
        "cases_lag_6",
        "cases_lag_8",
        "cases_pct_change_1w",
        "cases_pct_change_4w",
        "cases_roll_max_4",
        "cases_roll_mean_2",
        "cases_roll_mean_4",
        "cases_roll_mean_8",
        "cases_roll_min_4",
        "cases_roll_std_4",
        "cases_roll_std_8",
        "cases_slope_4w",
        "dengue_cases",
        "district_id",
        "log1p_cases",
        "month",
        "month_cos",
        "month_sin",
        "quarter",
        "week_cos",
        "week_of_year",
        "week_sin",
    ],
    "cases_rainfall": [
        "cases_change_1w",
        "cases_change_2w",
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",
        "cases_lag_6",
        "cases_lag_8",
        "cases_pct_change_1w",
        "cases_pct_change_4w",
        "cases_roll_max_4",
        "cases_roll_mean_2",
        "cases_roll_mean_4",
        "cases_roll_mean_8",
        "cases_roll_min_4",
        "cases_roll_std_4",
        "cases_roll_std_8",
        "cases_slope_4w",
        "dengue_cases",
        "district_id",
        "log1p_cases",
        "month",
        "month_cos",
        "month_sin",
        "quarter",
        "rain_days_10mm",
        "rain_days_1mm",
        "rain_days_roll_sum_4",
        "rainfall_lag_1",
        "rainfall_lag_2",
        "rainfall_lag_3",
        "rainfall_lag_4",
        "rainfall_lag_6",
        "rainfall_lag_8",
        "rainfall_max_daily_mm",
        "rainfall_mean_daily_mm",
        "rainfall_roll_max_4",
        "rainfall_roll_mean_4",
        "rainfall_roll_sum_2",
        "rainfall_roll_sum_4",
        "rainfall_roll_sum_8",
        "rainfall_sum_mm",
        "week_cos",
        "week_of_year",
        "week_sin",
    ],
    "cases_full_weather": [
        "cases_change_1w",
        "cases_change_2w",
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",
        "cases_lag_6",
        "cases_lag_8",
        "cases_pct_change_1w",
        "cases_pct_change_4w",
        "cases_roll_max_4",
        "cases_roll_mean_2",
        "cases_roll_mean_4",
        "cases_roll_mean_8",
        "cases_roll_min_4",
        "cases_roll_std_4",
        "cases_roll_std_8",
        "cases_slope_4w",
        "dengue_cases",
        "district_id",
        "humidity_lag_1",
        "humidity_lag_2",
        "humidity_lag_4",
        "humidity_mean_pct",
        "humidity_roll_mean_2",
        "humidity_roll_mean_4",
        "log1p_cases",
        "month",
        "month_cos",
        "month_sin",
        "quarter",
        "rain_days_10mm",
        "rain_days_1mm",
        "rain_days_roll_sum_4",
        "rainfall_lag_1",
        "rainfall_lag_2",
        "rainfall_lag_3",
        "rainfall_lag_4",
        "rainfall_lag_6",
        "rainfall_lag_8",
        "rainfall_max_daily_mm",
        "rainfall_mean_daily_mm",
        "rainfall_roll_max_4",
        "rainfall_roll_mean_4",
        "rainfall_roll_sum_2",
        "rainfall_roll_sum_4",
        "rainfall_roll_sum_8",
        "rainfall_sum_mm",
        "temp_max_c",
        "temp_mean_c",
        "temp_mean_lag_1",
        "temp_mean_lag_2",
        "temp_mean_lag_4",
        "temp_min_c",
        "temp_roll_mean_2",
        "temp_roll_mean_4",
        "temp_roll_std_4",
        "week_cos",
        "week_of_year",
        "week_sin",
    ],
}


class Milestone3ProtocolError(ValueError):
    """Raised when Milestone 3 protocol invariants are violated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _coerce_calendar_timestamp(series: pd.Series, label: str) -> pd.Series:
    raw = pd.Series(series, copy=False)
    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.isna().any():
        raise Milestone3ProtocolError(f"frame contains invalid {label} values")
    if getattr(parsed.dt, "tz", None) is not None:
        raise Milestone3ProtocolError(f"{label} requires timezone-naive calendar timestamps")
    if parsed.dt.normalize().ne(parsed).any():
        raise Milestone3ProtocolError(f"{label} requires midnight calendar timestamps")
    return parsed


def _dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "week_start_date" not in out.columns or "week_end_date" not in out.columns:
        raise Milestone3ProtocolError("frame requires week_start_date and week_end_date")
    for column in ("week_start_date", "week_end_date", "feature_through"):
        if column in out.columns:
            out[column] = _coerce_calendar_timestamp(out[column], column)
    if "district_id" not in out.columns or out["district_id"].isna().any():
        raise Milestone3ProtocolError("frame requires non-null district_id")
    invalid = out["week_end_date"].ne(out["week_start_date"] + pd.Timedelta(days=6))
    if invalid.any():
        raise Milestone3ProtocolError("invalid weekly interval: end must equal start plus 6 days")
    return out.sort_values(["district_id", "week_start_date"]).reset_index(drop=True)


def _date_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "district_id" not in out.columns or "week_start_date" not in out.columns:
        raise Milestone3ProtocolError("frame requires district_id and week_start_date")
    if out["district_id"].isna().any():
        raise Milestone3ProtocolError("frame contains null district_id values")
    out["week_start_date"] = _coerce_calendar_timestamp(
        out["week_start_date"], "week_start_date"
    )
    if out.duplicated(["district_id", "week_start_date"]).any():
        raise Milestone3ProtocolError("duplicate district/week_start_date keys")
    return out.sort_values(["district_id", "week_start_date"]).reset_index(drop=True)


def _validate_per_district_weekly_grid(frame: pd.DataFrame) -> None:
    for district_id, group in frame.groupby("district_id", sort=False):
        starts = group["week_start_date"].sort_values()
        day_steps = starts.diff().dropna().dt.days
        if day_steps.mod(7).ne(0).any():
            raise Milestone3ProtocolError(
                f"district {district_id} contains off-grid week_start_date values"
            )


def _origin_key_tuples(frame: pd.DataFrame) -> set[tuple[Any, pd.Timestamp]]:
    return set(frame[["district_id", "week_start_date"]].itertuples(index=False, name=None))


def _complete_weekly_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.duplicated(["district_id", "week_start_date"]).any():
        raise Milestone3ProtocolError("duplicate district/week_start_date keys")
    _validate_per_district_weekly_grid(frame)
    completed: list[pd.DataFrame] = []
    for district_id, group in frame.groupby("district_id", sort=False):
        group = group.sort_values("week_start_date").copy()
        full_weeks = pd.date_range(
            group["week_start_date"].min(),
            group["week_start_date"].max(),
            freq="7D",
        )
        group = group.set_index("week_start_date")
        expanded = group.reindex(full_weeks)
        expanded.index.name = "week_start_date"
        expanded["district_id"] = district_id
        expanded["week_end_date"] = expanded.index + pd.Timedelta(days=6)
        expanded["_m3_original_origin"] = expanded.index.isin(group.index)
        completed.append(expanded.reset_index())
    return pd.concat(completed, ignore_index=True)


def add_origin_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute causal M1 origin-time features on caller-provided observations."""
    dated = _dates(frame)
    original_keys = _origin_key_tuples(dated)
    out = _complete_weekly_calendar(dated)
    out = _add_lags(out)
    out = _add_rolling(out)
    out = _add_trends(out)
    out = add_seasonality_features(out)
    out = out.loc[out.pop("_m3_original_origin")].reset_index(drop=True)
    if len(out) != len(dated) or _origin_key_tuples(out) != original_keys:
        raise Milestone3ProtocolError("origin feature construction did not preserve input keys")
    return out


def build_direct_tasks(frame: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """Attach exact-date direct H1-H4 labels by district and target week start."""
    out = _dates(frame)
    if out.duplicated(["district_id", "week_start_date"]).any():
        raise Milestone3ProtocolError("duplicate district/week_start_date keys")
    if "dengue_cases" not in out.columns:
        raise Milestone3ProtocolError("dengue_cases is required for direct targets")
    _validate_numeric_series(out["dengue_cases"], "dengue_cases", allow_na=True)
    lookup = out.set_index(["district_id", "week_start_date"])["dengue_cases"]
    for horizon in horizons:
        if horizon not in HORIZONS:
            raise Milestone3ProtocolError(f"unsupported horizon: {horizon}")
        values: list[Any] = []
        for row in out[["district_id", "week_start_date"]].itertuples(index=False):
            target_start = row.week_start_date + pd.Timedelta(days=7 * horizon)
            key = (row.district_id, target_start)
            if key in lookup.index:
                values.append(lookup.loc[key])
            else:
                values.append(np.nan)
        out[f"target_h{horizon}"] = values
    return out


def _validate_numeric_series(series: pd.Series, label: str, *, allow_na: bool = False) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    bad_na = numeric.isna() if not allow_na else numeric.isna() & series.notna()
    if (
        bad_na.any()
        or np.isinf(numeric.dropna().to_numpy(dtype="float64")).any()
        or numeric.dropna().lt(0).any()
    ):
        raise Milestone3ProtocolError(f"{label} requires finite nonnegative values")


def origin_eligible_mask(frame: pd.DataFrame, *, horizon: int | None = None) -> pd.Series:
    if horizon is not None and horizon not in HORIZONS:
        raise Milestone3ProtocolError(f"unsupported horizon: {horizon}")
    required = list(EVALUATION_REQUIRED_FEATURES)
    if horizon is None:
        required.extend(f"target_h{h}" for h in HORIZONS)
    else:
        required.append(f"target_h{horizon}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise Milestone3ProtocolError(f"origin eligibility frame missing columns: {missing}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    return (
        numeric.notna().all(axis=1)
        & np.isfinite(numeric).all(axis=1)
        & numeric.ge(0).all(axis=1)
    )


def common_evaluation_mask(
    frame: pd.DataFrame, *, boundary: pd.Timestamp | str | None = None
) -> pd.Series:
    mask = origin_eligible_mask(frame)
    if boundary is not None:
        cutoff = pd.Timestamp(boundary)
        if "week_start_date" not in frame.columns or "week_end_date" not in frame.columns:
            raise Milestone3ProtocolError("frame requires week_start_date and week_end_date")
        week_start = _coerce_calendar_timestamp(frame["week_start_date"], "week_start_date")
        for horizon in HORIZONS:
            target_end = _target_end(week_start, horizon)
            mask &= target_end.lt(cutoff)
    return mask


def _target_end(origin_start: pd.Series, horizon: int) -> pd.Series:
    return pd.to_datetime(origin_start) + pd.to_timedelta(7 * horizon + 6, unit="D")


def training_frame(
    frame: pd.DataFrame,
    *,
    horizon: int,
    first_origin: pd.Timestamp | str,
    test_boundary: pd.Timestamp | str = "2025-01-01",
) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise Milestone3ProtocolError(f"unsupported horizon: {horizon}")
    target_column = f"target_h{horizon}"
    if target_column not in frame.columns:
        raise Milestone3ProtocolError(f"training frame missing {target_column}")
    out = _dates(frame)
    first = pd.Timestamp(first_origin)
    boundary = pd.Timestamp(test_boundary)
    if boundary > CALENDAR_2025_BOUNDARY:
        boundary = CALENDAR_2025_BOUNDARY
    target_end = _target_end(out["week_start_date"], horizon)
    embargo_limit = target_end + pd.Timedelta(days=7)
    mask = (
        target_end.lt(boundary)
        & embargo_limit.lt(first)
        & out["week_end_date"].lt(CALENDAR_2025_BOUNDARY)
        & origin_eligible_mask(out, horizon=horizon)
    )
    selected = out.loc[mask].reset_index(drop=True)
    if not selected.empty:
        retained_target_end = _target_end(selected["week_start_date"], horizon)
        if not retained_target_end.lt(CALENDAR_2025_BOUNDARY).all():
            raise Milestone3ProtocolError("retained training labels cross calendar 2025")
        if not (retained_target_end + pd.Timedelta(days=7)).lt(first).all():
            raise Milestone3ProtocolError("retained training rows violate strict embargo")
    return selected


def build_development_direct_tasks(
    raw_frame: pd.DataFrame,
    *,
    test_boundary: pd.Timestamp | str,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    boundary = pd.Timestamp(test_boundary)
    raw = _dates(raw_frame)
    cutoff = min(boundary, CALENDAR_2025_BOUNDARY)
    pre_boundary = raw.loc[raw["week_end_date"].lt(cutoff)].copy()
    return build_direct_tasks(add_origin_features(pre_boundary), horizons=horizons)


def resolve_feature_sets(registry: pd.DataFrame) -> dict[str, list[str]]:
    """Return pinned M3 feature sets; reject injected trainable candidates."""
    if not registry.empty:
        names = set(registry.get("feature_name", pd.Series(dtype=str)).astype(str))
        allowed = set().union(*(set(columns) for columns in PINNED_FEATURE_SETS.values()))
        unknown_trainable = set()
        if "eligible_for_training" in registry.columns:
            eligible = registry["eligible_for_training"].astype(bool)
            unknown_trainable = set(registry.loc[eligible, "feature_name"].astype(str)) - allowed
        forbidden_trainable = names & FORBIDDEN_FEATURES
        if unknown_trainable or forbidden_trainable:
            raise Milestone3ProtocolError(
                f"unpinned or forbidden trainable feature candidates: "
                f"{sorted(unknown_trainable | forbidden_trainable)}"
            )
    return {name: list(columns) for name, columns in PINNED_FEATURE_SETS.items()}


def feature_list_digest(columns: Iterable[str]) -> str:
    return _sha256_json(list(columns))


def selected_feature_columns(name: str, *, frame: pd.DataFrame | None = None) -> list[str]:
    if frame is not None:
        validate_feature_timestamps(frame)
    if name not in PINNED_FEATURE_SETS:
        raise Milestone3ProtocolError(f"unknown pinned feature set: {name}")
    return list(PINNED_FEATURE_SETS[name])


def validate_requested_features(columns: Iterable[str]) -> list[str]:
    requested = list(columns)
    allowed = set().union(*(set(values) for values in PINNED_FEATURE_SETS.values()))
    forbidden = [
        column
        for column in requested
        if column in FORBIDDEN_FEATURES or column.startswith("target_h")
    ]
    if forbidden:
        raise Milestone3ProtocolError(f"forbidden requested features: {forbidden}")
    unknown = [column for column in requested if column not in allowed]
    if unknown:
        raise Milestone3ProtocolError(f"unpinned requested features: {unknown}")
    return requested


def validate_feature_timestamps(frame: pd.DataFrame) -> None:
    out = _dates(frame)
    if "feature_through" not in out.columns:
        return
    if out["feature_through"].isna().any():
        raise Milestone3ProtocolError("invalid feature_through timestamp")
    invalid = out["feature_through"].gt(out["week_end_date"])
    if invalid.any():
        raise Milestone3ProtocolError("feature_through exceeds origin week_end_date")


def audit_causal_features(
    frame: pd.DataFrame,
    *,
    origin_end: pd.Timestamp | str,
    expected_master: pd.DataFrame | None = None,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Recompute features from the prefix available by origin_end only."""
    cutoff = pd.Timestamp(origin_end)
    raw = _dates(frame)
    prefix = raw.loc[raw["week_end_date"].le(cutoff)].copy()
    audited = add_origin_features(prefix)
    audited["feature_through"] = audited["week_end_date"]
    validate_feature_timestamps(audited)
    if expected_master is not None:
        columns = list(feature_columns or selected_feature_columns("cases_full_weather"))
        compare_causal_feature_audit(audited, expected_master, feature_columns=columns)
        audited.attrs["causal_feature_audit"] = {"compared_columns": columns}
    return audited


def compare_causal_feature_audit(
    recomputed: pd.DataFrame,
    provided_master: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
) -> bool:
    columns = list(feature_columns)
    for column in columns:
        if column not in recomputed.columns or column not in provided_master.columns:
            raise Milestone3ProtocolError(f"causal feature audit missing column: {column}")
    value_columns = [
        column for column in columns if column not in {"district_id", "week_start_date"}
    ]
    left = _date_keys(recomputed)[["district_id", "week_start_date", *value_columns]]
    right = _date_keys(provided_master)[["district_id", "week_start_date", *value_columns]]
    right = right.assign(_m3_key_present=True)
    merged = left.merge(
        right,
        on=["district_id", "week_start_date"],
        how="left",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    missing_right = merged["_m3_key_present"].isna()
    if len(merged) != len(left) or missing_right.any():
        raise Milestone3ProtocolError("causal feature audit key mismatch")
    for column in value_columns:
        left_values = merged[f"{column}_left"]
        right_values = merged[f"{column}_right"]
        equal = left_values.eq(right_values) | (left_values.isna() & right_values.isna())
        if not equal.all():
            raise Milestone3ProtocolError(f"causal feature audit mismatch: {column}")
    return True


def _key_rows(frame: pd.DataFrame) -> list[dict[str, str]]:
    keys = _date_keys(frame)[["district_id", "week_start_date"]].copy()
    rows = keys.sort_values(["district_id", "week_start_date"]).itertuples(index=False)
    return [
        {
            "district_id": str(row.district_id),
            "week_start_date": pd.Timestamp(row.week_start_date).date().isoformat(),
        }
        for row in rows
    ]


def freeze_common_keys(frame: pd.DataFrame) -> dict[str, Any]:
    keys = _key_rows(frame)
    return {
        "keys": keys,
        "row_count": len(keys),
        "row_key_digest": _sha256_json(keys),
        "labels_included": False,
    }


def verify_common_keys(frame: pd.DataFrame, frozen: Mapping[str, Any]) -> bool:
    observed = freeze_common_keys(frame)
    if observed["row_count"] != frozen.get("row_count"):
        raise Milestone3ProtocolError("common key row count mismatch")
    if observed["row_key_digest"] != frozen.get("row_key_digest"):
        raise Milestone3ProtocolError("common key digest mismatch")
    if "keys" in frozen and observed["keys"] != frozen["keys"]:
        raise Milestone3ProtocolError("common key list mismatch")
    return True


def cohort_digests(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Iterable[str],
) -> dict[str, Any]:
    out = _dates(frame) if "week_end_date" in frame.columns else _date_keys(frame)
    feature_columns = list(feature_columns)
    missing = [column for column in [target_column, *feature_columns] if column not in out.columns]
    if missing:
        raise Milestone3ProtocolError(f"digest frame missing columns: {missing}")
    ordered = out.sort_values(["district_id", "week_start_date"]).reset_index(drop=True)
    if ordered.duplicated(["district_id", "week_start_date"]).any():
        raise Milestone3ProtocolError("duplicate district/week_start_date keys")
    keys = _key_rows(ordered)
    target_payload = [
        {**key, "target": None if pd.isna(value) else float(value)}
        for key, value in zip(keys, ordered[target_column], strict=True)
    ]
    feature_payload = []
    for key, (_, row) in zip(keys, ordered[feature_columns].iterrows(), strict=True):
        feature_payload.append(
            {
                **key,
                "features": {
                    column: None if pd.isna(row[column]) else _json_scalar(row[column])
                    for column in feature_columns
                },
            }
        )
    return {
        "row_count": int(len(ordered)),
        "row_key_digest": _sha256_json(keys),
        "target_content_digest": _sha256_json(target_payload),
        "feature_content_digest": _sha256_json(feature_payload),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def threshold_binding(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Iterable[str] = (),
) -> dict[str, Any]:
    if "dengue_cases" not in frame.columns or target_column not in frame.columns:
        raise Milestone3ProtocolError("threshold frame requires dengue_cases and target column")
    if frame.empty:
        raise Milestone3ProtocolError("threshold frame is empty")
    feature_columns = validate_requested_features(feature_columns)
    digests = cohort_digests(frame, target_column=target_column, feature_columns=feature_columns)
    ordered = (
        _date_keys(frame)
        .sort_values(["district_id", "week_start_date"])
        .reset_index(drop=True)
    )
    if ordered.duplicated(["district_id", "week_start_date"]).any():
        raise Milestone3ProtocolError("duplicate district/week_start_date keys")
    current = pd.to_numeric(ordered["dengue_cases"], errors="coerce")
    target = pd.to_numeric(ordered[target_column], errors="coerce")
    if (
        current.isna().any()
        or np.isinf(current.to_numpy(dtype="float64")).any()
        or current.lt(0).any()
    ):
        raise Milestone3ProtocolError(
            "threshold frame requires finite nonnegative current dengue_cases"
        )
    if (
        target.isna().any()
        or np.isinf(target.to_numpy(dtype="float64")).any()
        or target.lt(0).any()
    ):
        raise Milestone3ProtocolError("threshold frame requires finite nonnegative targets")
    delta = target - current
    positive = delta[delta.gt(0)]
    negative_abs = delta[delta.lt(0)].abs()
    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "sorted_training_keys": _key_rows(ordered),
        "digests": {
            **digests,
            "current_dengue_content_digest": _sha256_json(
                [
                    {**key, "dengue_cases": float(value)}
                    for key, value in zip(_key_rows(ordered), current, strict=True)
                ]
            ),
            "requested_feature_digest": feature_list_digest(feature_columns),
        },
        "q90_incidence": float(target.quantile(0.90, interpolation="linear")),
        "q95_incidence": float(target.quantile(0.95, interpolation="linear")),
        "stable_abs_delta_q25": float(delta.abs().quantile(0.25, interpolation="linear")),
        "large_up_q90": None
        if positive.empty
        else float(positive.quantile(0.90, interpolation="linear")),
        "large_down_abs_q90": None
        if negative_abs.empty
        else float(negative_abs.quantile(0.90, interpolation="linear")),
        "direction_counts": {
            "up": int(delta.gt(0).sum()),
            "down": int(delta.lt(0).sum()),
            "stable": int(delta.eq(0).sum()),
        },
        "category_boundaries": {
            "high_incidence": "target >= q90_incidence",
            "very_high_incidence": "target >= q95_incidence",
            "stable": "abs(target-current) <= stable_abs_delta_q25",
            "large_up": "target-current >= large_up_q90 when non-null",
            "large_down": "current-target >= large_down_abs_q90 when non-null",
        },
        "quantile_method": "linear",
        "empty_direction_null_semantics": True,
    }


def verify_threshold_binding(
    frame: pd.DataFrame,
    binding: Mapping[str, Any],
    *,
    target_column: str,
    feature_columns: Iterable[str] = (),
) -> bool:
    observed = threshold_binding(
        frame,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    observed_json = json.loads(_canonical_json(observed))
    binding_json = json.loads(_canonical_json(dict(binding)))
    if observed_json.get("sorted_training_keys") != binding_json.get("sorted_training_keys"):
        raise Milestone3ProtocolError("threshold binding key digest mismatch")
    if observed_json.get("digests") != binding_json.get("digests"):
        raise Milestone3ProtocolError("threshold binding digest mismatch")
    if observed_json != binding_json:
        raise Milestone3ProtocolError("threshold binding value mismatch")
    return True


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_metrics() -> dict[str, str]:
    return {
        "mae": "mean absolute error on dengue case counts",
        "rmse": "root mean squared error on dengue case counts",
        "r2": "coefficient of determination; undefined denominators reported as null",
        "poisson_deviance": (
            "mean Poisson deviance on nonnegative case counts; undefined values reported as null"
        ),
        "high_incidence_mae": "MAE on rows whose observed target is at or above q90_incidence",
        "bias": "mean prediction minus observed dengue case count",
        "fold_wins": (
            "count of validation folds where model MAE is strictly lower than same-origin "
            "persistence MAE; ties are not wins"
        ),
    }


def build_default_protocol_config(
    *,
    trial_metadata_path: str | Path = DEFAULT_TRIAL010_METADATA,
    feature_sets: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build the fixed protocol document for creating a new config before fits."""
    path = Path(trial_metadata_path)
    if path == DEFAULT_TRIAL010_METADATA and _sha256_file(path) != DEFAULT_TRIAL010_METADATA_SHA256:
        raise Milestone3ProtocolError("trial010 metadata hash mismatch")
    if path != DEFAULT_TRIAL010_METADATA:
        raise Milestone3ProtocolError("trial010 metadata hash mismatch")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise Milestone3ProtocolError("trial010 metadata missing config")
    if config.get("family") != "lightgbm" or config.get("objective") != "poisson":
        raise Milestone3ProtocolError("trial010 config must be LightGBM Poisson")
    fixed_feature_sets = {name: list(columns) for name, columns in PINNED_FEATURE_SETS.items()}
    if feature_sets is not None and dict(feature_sets) != fixed_feature_sets:
        raise Milestone3ProtocolError("provided feature sets do not match pinned protocol")
    try:
        lightgbm_version = package_metadata.version("lightgbm")
    except package_metadata.PackageNotFoundError:
        lightgbm_version = None
    registry_sha256 = _sha256_file(FIXED_FEATURE_REGISTRY)
    protocol = {
        "authorization_sha256": AUTHORIZATION_SHA256,
        "authorization_source": "docs/m3-protocol-deviation-authorization.md",
        "study_designation": "qualified_retrospective_observation_time",
        "historical_publication_vintage": {
            "status": "future_strict_required",
            "present_availability": "unestablished",
            "strict_future_validation_required": True,
            "prohibited_claims": [
                "no_operational_backtesting_claim",
                "no_representative_annual_2025_claim",
                "no_actual_2025_outcome_claim_until_gate_engine_exists",
            ],
        },
        "claim_limits": [
            "no_operational_backtesting_claim",
            "no_representative_annual_2025_claim",
        ],
        "test_cohort": {
            "district_origins": 600,
            "districts": 25,
            "weeks": 24,
            "coverage_months": ["March", "July", "August", "September", "October", "November"],
            "coverage": "partial_year_availability_informed",
        },
        "targets": {
            "horizons": list(HORIZONS),
            "join": "exact_district_date_origin_start_plus_7h_days",
            "common_evaluation_requires": [
                *EVALUATION_REQUIRED_FEATURES,
                *(f"target_h{h}" for h in HORIZONS),
            ],
            "calendar_2025_boundary": CALENDAR_2025_BOUNDARY.date().isoformat(),
            "cutoff_origin_week_end": "origin week_end_date < 2025-01-01 for development",
            "predictor_observation_cutoff": (
                "all predictor observation/reference and aggregation end timestamps must be <= "
                "origin week_end_date"
            ),
            "aggregation_end": "target week end must be < 2025-01-01 for development",
            "label_embargo": "target_week_end + 7 days < first scheduled test origin",
            "no2025_boundary": "calendar 2025-01-01 independent of scheduled test boundary",
        },
        "features": {
            "sets": fixed_feature_sets,
            "digests": {
                name: feature_list_digest(columns)
                for name, columns in fixed_feature_sets.items()
            },
            "registry_source": str(FIXED_FEATURE_REGISTRY.relative_to(REPO_ROOT)),
            "registry_sha256": registry_sha256,
            "forbidden": sorted(FORBIDDEN_FEATURES),
            "origin_calendar": "unchanged_from_m2",
            "area_population": "excluded",
            "availability_informed_partial_year": {
                "district_origins": 600,
                "districts": 25,
                "weeks": 24,
                "coverage_months": ["March", "July", "August", "September", "October", "November"],
            },
        },
        "models": {
            "ridge": {
                "family": "ridge",
                "seed": 42,
                "hyperparams": {"alpha": 1.0},
                "preprocessing": {"fit_scope": "fold_training_rows_only"},
            },
            "lightgbm_trial010": {
                "source_metadata_path": str(path.relative_to(REPO_ROOT)),
                "source_metadata_sha256": _sha256_file(path),
                "model_sha256": metadata.get("model_sha256"),
                "family": config["family"],
                "objective": config["objective"],
                "seed": config.get("seed", 42),
                "hyperparams": dict(config.get("hyperparams", {})),
                "effective_estimator_defaults": {
                    "objective": "poisson",
                    "random_state": config.get("seed", 42),
                    "n_jobs": 1,
                    "verbose": -1,
                    "force_col_wise": None,
                },
                "installed_lightgbm_version": lightgbm_version,
                "preprocessing": dict(config.get("preprocessing", {})),
                "postprocessing": dict(config.get("postprocessing", {})),
            },
        },
        "selection": {
            "folds": list(DEVELOPMENT_FOLDS),
            "champion_metric": "mean_fold_mae",
            "primary": "ridge__cases_only",
            "max_additional_tuning_trials": 0,
            "final_2024_train_allowed": True,
            "model_count": 144,
            "candidate_count": 24,
            "tie_order": [
                "mean_fold_mae",
                "ridge_family",
                "fewer_features",
                "lexicographic_id",
            ],
        },
        "determinism": {
            "seed": 42,
            "lightgbm_n_jobs": config.get("hyperparams", {}).get("n_jobs"),
            "preprocessing_fit_scope": config.get("preprocessing", {}).get("fit_scope"),
            "m2_conventions": {
                "numeric_median_imputation": True,
                "categorical_one_hot": True,
                "all_missing_features_dropped": True,
                "clip_negative_predictions": True,
            },
        },
        "metrics": _protocol_metrics(),
        "thresholds": {
            "quantiles": [0.90, 0.95],
            "delta_quantile": 0.25,
            "method": "linear",
            "boundaries": (
                "frozen from sorted training keys, target, current dengue, "
                "and requested feature digests"
            ),
        },
        "bootstrap": {
            "policy": "frozen moving4 observed-week blocks; no gaps; no fold crossing",
            "short_run_handling": (
                "use available complete moving4 blocks only; report insufficiency"
            ),
            "fixed_length": True,
            "percentile_method": "linear",
            "replicates": 1000,
            "seed": 42,
            "statistic": "mean fold stats",
        },
        "explainability": {
            "fold": 2020,
            "rows": 200,
            "permutation_repeats": 3,
        },
        "freeze_status": {
            "real_2025_exact_key_freeze": "pending_authorized_metadata_only_seam",
            "once_only_freeze_gate": "pending_not_implemented",
            "no_real_cohort_keys_fabricated": True,
        },
    }
    return protocol


def protocol_load(
    *,
    trial_metadata_path: str | Path = DEFAULT_TRIAL010_METADATA,
    feature_sets: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    if _sha256_file(FIXED_AUTHORIZATION) != AUTHORIZATION_SHA256:
        raise Milestone3ProtocolError("authorization hash mismatch")
    if _sha256_file(FIXED_FEATURE_REGISTRY) != FEATURE_REGISTRY_SHA256:
        raise Milestone3ProtocolError("feature registry hash mismatch")
    if _sha256_file(FIXED_PROTOCOL_CONFIG) != FIXED_PROTOCOL_CONFIG_SHA256:
        raise Milestone3ProtocolError("fixed protocol config hash mismatch")
    protocol = build_default_protocol_config(
        trial_metadata_path=trial_metadata_path,
        feature_sets=feature_sets,
    )
    fixed_config = json.loads(FIXED_PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    if fixed_config != protocol:
        raise Milestone3ProtocolError("fixed protocol config content mismatch")
    return protocol


def require_2025_outcome_gate(capability: Any) -> None:
    raise Milestone3ProtocolError(
        "2025 outcome gate engine is not implemented; real 2025 outcomes are prohibited"
    )
