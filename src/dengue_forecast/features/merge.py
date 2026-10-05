from __future__ import annotations

from dengue_forecast.config import DISTRICTS
from dengue_forecast.contracts import (
    ContractError,
    assert_unique_key,
    get_target_column,
    get_training_columns,
    validate_dengue_weekly,
    validate_district_reference,
    validate_ml_dataset,
    validate_weather_weekly,
)
from dengue_forecast.features.calendar import build_district_week_calendar
from dengue_forecast.features.registry import build_feature_registry
from dengue_forecast.features.seasonality import add_seasonality_features
from dengue_forecast.features.target import add_targets

CASE_LAGS = [1, 2, 3, 4, 6, 8]
RAINFALL_LAGS = [1, 2, 3, 4, 6, 8]
TEMP_LAGS = [1, 2, 4]
HUMIDITY_LAGS = [1, 2, 4]


def _coerce_dates(df, columns=("week_start_date", "week_end_date")):  # type: ignore[no-untyped-def]
    import pandas as pd

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column]).dt.date
    return out


def _prepare_inputs(dengue, weather, district_reference):  # type: ignore[no-untyped-def]
    dengue = validate_dengue_weekly(dengue)
    weather = validate_weather_weekly(weather)
    district_reference = validate_district_reference(district_reference.copy())
    return dengue, weather, district_reference


def _add_lags(out):  # type: ignore[no-untyped-def]
    grouped = out.groupby("district_id", sort=False, group_keys=False)
    for lag in CASE_LAGS:
        out[f"cases_lag_{lag}"] = grouped["dengue_cases"].shift(lag)
    for lag in RAINFALL_LAGS:
        out[f"rainfall_lag_{lag}"] = grouped["rainfall_sum_mm"].shift(lag)
    for lag in TEMP_LAGS:
        out[f"temp_mean_lag_{lag}"] = grouped["temp_mean_c"].shift(lag)
    for lag in HUMIDITY_LAGS:
        out[f"humidity_lag_{lag}"] = grouped["humidity_mean_pct"].shift(lag)
    return out


def _add_rolling(out):  # type: ignore[no-untyped-def]
    grouped = out.groupby("district_id", sort=False)
    cases = grouped["dengue_cases"]
    for window in [2, 4, 8]:
        roll = cases.rolling(window=window, min_periods=window)
        out[f"cases_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
    for window in [4, 8]:
        roll = cases.rolling(window=window, min_periods=window)
        out[f"cases_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)
    out["cases_roll_min_4"] = (
        cases.rolling(window=4, min_periods=4).min().reset_index(level=0, drop=True)
    )
    out["cases_roll_max_4"] = (
        cases.rolling(window=4, min_periods=4).max().reset_index(level=0, drop=True)
    )

    rain = grouped["rainfall_sum_mm"]
    for window in [2, 4, 8]:
        out[f"rainfall_roll_sum_{window}"] = (
            rain.rolling(window=window, min_periods=window).sum().reset_index(level=0, drop=True)
        )
    out["rainfall_roll_mean_4"] = (
        rain.rolling(window=4, min_periods=4).mean().reset_index(level=0, drop=True)
    )
    out["rainfall_roll_max_4"] = (
        rain.rolling(window=4, min_periods=4).max().reset_index(level=0, drop=True)
    )
    out["rain_days_roll_sum_4"] = (
        grouped["rain_days_1mm"]
        .rolling(window=4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    temp = grouped["temp_mean_c"]
    out["temp_roll_mean_2"] = (
        temp.rolling(window=2, min_periods=2).mean().reset_index(level=0, drop=True)
    )
    out["temp_roll_mean_4"] = (
        temp.rolling(window=4, min_periods=4).mean().reset_index(level=0, drop=True)
    )
    out["temp_roll_std_4"] = (
        temp.rolling(window=4, min_periods=4).std().reset_index(level=0, drop=True)
    )

    humidity = grouped["humidity_mean_pct"]
    out["humidity_roll_mean_2"] = (
        humidity.rolling(window=2, min_periods=2).mean().reset_index(level=0, drop=True)
    )
    out["humidity_roll_mean_4"] = (
        humidity.rolling(window=4, min_periods=4).mean().reset_index(level=0, drop=True)
    )
    return out


def _add_trends(out):  # type: ignore[no-untyped-def]
    import numpy as np

    grouped = out.groupby("district_id", sort=False, group_keys=False)
    out["cases_change_1w"] = out["dengue_cases"] - grouped["dengue_cases"].shift(1)
    out["cases_change_2w"] = out["dengue_cases"] - grouped["dengue_cases"].shift(2)
    denom_1 = grouped["dengue_cases"].shift(1)
    denom_4 = grouped["dengue_cases"].shift(4)
    safe_denom_1 = denom_1.where(denom_1.notna() & denom_1.ne(0))
    safe_denom_4 = denom_4.where(denom_4.notna() & denom_4.ne(0))
    out["cases_pct_change_1w"] = out["cases_change_1w"] / safe_denom_1
    out["cases_pct_change_4w"] = (out["dengue_cases"] - denom_4) / safe_denom_4
    out["log1p_cases"] = np.log1p(out["dengue_cases"])

    def slope(values):  # type: ignore[no-untyped-def]
        if values.isna().any() or len(values) < 4:
            return np.nan
        x = np.arange(len(values), dtype=float)
        return float(np.polyfit(x, values.to_numpy(dtype=float), 1)[0])

    out["cases_slope_4w"] = (
        grouped["dengue_cases"]
        .rolling(window=4, min_periods=4)
        .apply(slope, raw=False)
        .reset_index(level=0, drop=True)
    )
    return out


def _add_missingness_and_quality(
    out,
    *,
    min_weather_temporal_coverage_pct: float,
    min_weather_spatial_coverage_pct: float,
    required_lags: list[int],
):  # type: ignore[no-untyped-def]
    import pandas as pd

    out["case_missing_flag"] = out["dengue_cases"].isna().astype(bool)
    weather_cols = ["rainfall_sum_mm", "temp_mean_c", "humidity_mean_pct"]
    out["weather_missing_flag"] = out[weather_cols].isna().any(axis=1).astype(bool)
    lag_cols = [f"cases_lag_{lag}" for lag in required_lags]
    out["case_history_missing_flag"] = out[lag_cols].isna().any(axis=1).astype(bool)

    low_weather = (
        out["weather_temporal_coverage_pct"].fillna(-1).lt(min_weather_temporal_coverage_pct)
        | out["weather_spatial_coverage_pct"].fillna(-1).lt(min_weather_spatial_coverage_pct)
        | out.get("weather_quality_flag", pd.Series(index=out.index, dtype=object))
        .fillna("UNKNOWN")
        .ne("OK")
    )

    missing_population = out["population_reference"].isna()
    issue_count = (
        out["case_history_missing_flag"].astype(int)
        + out["weather_missing_flag"].astype(int)
        + low_weather.astype(int)
        + missing_population.astype(int)
    )
    out["row_quality_flag"] = "MULTIPLE_ISSUES"
    out.loc[issue_count == 0, "row_quality_flag"] = "OK"
    out.loc[(issue_count == 1) & out["case_history_missing_flag"], "row_quality_flag"] = (
        "MISSING_CASE_HISTORY"
    )
    out.loc[(issue_count == 1) & out["weather_missing_flag"], "row_quality_flag"] = (
        "MISSING_WEATHER"
    )
    out.loc[(issue_count == 1) & low_weather, "row_quality_flag"] = "LOW_WEATHER_COVERAGE"
    out.loc[(issue_count == 1) & missing_population, "row_quality_flag"] = "MISSING_POPULATION"
    out["is_trainable"] = (
        out[get_target_column()].notna()
        & out["dengue_cases"].notna()
        & ~out["case_history_missing_flag"]
        & ~out["weather_missing_flag"]
        & ~low_weather
        & ~missing_population
    )
    out["is_trainable"] = out["is_trainable"].astype(bool)
    return out


def _add_continuity(out):  # type: ignore[no-untyped-def]
    import pandas as pd

    observed = out["dengue_cases"].notna()
    observed_starts = pd.to_datetime(out["week_start_date"]).where(observed)
    previous_observed = (
        observed_starts.groupby(out["district_id"], sort=False)
        .ffill()
        .groupby(out["district_id"])
        .shift(1)
    )
    current_start = pd.to_datetime(out["week_start_date"])
    out["days_since_previous_observation"] = (current_start - previous_observed).dt.days
    out["is_consecutive_week"] = out["days_since_previous_observation"].eq(7)
    return out


def build_ml_dataset(
    dengue,
    weather,
    district_reference,
    *,
    calendar_district_ids: list[str] | None = None,
    min_weather_coverage_pct: float = 85.0,
    min_weather_temporal_coverage_pct: float | None = None,
    min_weather_spatial_coverage_pct: float = 90.0,
    required_lags: list[int] | None = None,
):  # type: ignore[no-untyped-def]
    """Build the canonical Stage 5 district-week master and exhaustive registry."""

    import pandas as pd

    required_lags = required_lags or [1, 2, 3, 4]
    min_weather_temporal_coverage_pct = (
        min_weather_coverage_pct
        if min_weather_temporal_coverage_pct is None
        else min_weather_temporal_coverage_pct
    )
    dengue, weather, district_reference = _prepare_inputs(dengue, weather, district_reference)

    district_ids = calendar_district_ids or [district.district_id for district in DISTRICTS]
    calendar = build_district_week_calendar(dengue, district_ids=district_ids)
    dengue_payload = dengue.drop(columns=["district_name"], errors="ignore")
    weather_payload = weather.drop(columns=["district_name"], errors="ignore")
    reference_payload = district_reference.drop(columns=["geometry"], errors="ignore")

    assert_unique_key(calendar, ["district_id", "week_start_date"])
    assert_unique_key(dengue_payload, ["district_id", "week_start_date"])
    assert_unique_key(weather_payload, ["district_id", "week_start_date"])
    assert_unique_key(reference_payload, ["district_id"])
    _assert_source_dates_fit_calendar(dengue_payload, calendar, "dengue")
    _assert_source_dates_fit_calendar(weather_payload, calendar, "weather")

    out = calendar.merge(
        dengue_payload, on=["district_id", "week_start_date"], how="left", validate="one_to_one"
    )
    out = out.merge(
        weather_payload,
        on=["district_id", "week_start_date"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_weather"),
    )
    weather_interval = out["week_end_date_weather"].notna()
    bad_weather_interval = weather_interval & (
        out["week_end_date_weather"] != out["week_end_date_calendar"]
    )
    if bad_weather_interval.any():
        sample = (
            out.loc[
                bad_weather_interval,
                [
                    "district_id",
                    "week_start_date",
                    "week_end_date_calendar",
                    "week_end_date_weather",
                ],
            ]
            .head(5)
            .to_dict("records")
        )
        raise ContractError(
            f"Weather interval does not match canonical calendar interval: {sample}"
        )
    out = out.drop(columns=["week_end_date_weather"], errors="ignore")
    out = out.merge(reference_payload, on="district_id", how="left", validate="many_to_one")
    out["district_name"] = out["district_name"].fillna(out["district_name_calendar"])
    out["week_end_date"] = out["week_end_date"].fillna(out["week_end_date_calendar"])
    out["year"] = out["year"].fillna(out["year_calendar"])
    out["week"] = out["week"].fillna(out["week_calendar"])
    out["year_week"] = out["year_week"].fillna(out["year_week_calendar"])
    out = _fill_generated_case_rows(out)
    out = out.drop(
        columns=[
            "district_name_calendar",
            "week_end_date_calendar",
            "year_calendar",
            "week_calendar",
            "year_week_calendar",
        ],
        errors="ignore",
    )

    out = out.sort_values(["district_id", "week_start_date"]).reset_index(drop=True)
    out = _add_continuity(out)
    out = _add_lags(out)
    out = _add_rolling(out)
    out = _add_trends(out)
    out = add_seasonality_features(out)

    if "population_reference" in out.columns:
        population = out["population_reference"].replace({0: pd.NA})
        out["incidence_per_100k_using_2024_population"] = (
            out["dengue_cases"] / population
        ) * 100_000

    out = add_targets(out)
    out = _add_missingness_and_quality(
        out,
        min_weather_temporal_coverage_pct=min_weather_temporal_coverage_pct,
        min_weather_spatial_coverage_pct=min_weather_spatial_coverage_pct,
        required_lags=required_lags,
    )
    out["week_start_date"] = pd.to_datetime(out["week_start_date"]).dt.date
    out["week_end_date"] = pd.to_datetime(out["week_end_date"]).dt.date
    assert_unique_key(out, ["district_id", "week_start_date"])
    validate_ml_dataset(out)

    registry = build_feature_registry(out.columns.tolist())
    get_training_columns(registry)
    return out, registry


def build_training_view(dataset, registry, *, target_column: str | None = None):  # type: ignore[no-untyped-def]
    target = target_column or get_target_column()
    training_columns = get_training_columns(registry)
    selected = [column for column in training_columns if column in dataset.columns]
    missing = sorted(set(training_columns) - set(selected))
    if missing:
        raise ContractError(f"Training columns missing from dataset: {missing}")
    return dataset.loc[
        dataset["is_trainable"].astype(bool), selected + [target, "is_trainable"]
    ].copy()


def _assert_source_dates_fit_calendar(source, calendar, label: str) -> None:  # type: ignore[no-untyped-def]
    valid = set(zip(calendar["district_id"], calendar["week_start_date"], strict=False))
    observed = set(zip(source["district_id"], source["week_start_date"], strict=False))
    off_grid = sorted(observed - valid)
    if off_grid:
        raise ContractError(
            f"{label} rows do not fit the single seven-day calendar anchor: {off_grid[:5]}"
        )


def _fill_generated_case_rows(out):  # type: ignore[no-untyped-def]
    import pandas as pd

    generated = out["case_status"].isna()
    if not generated.any():
        return out
    out.loc[generated, "case_status"] = "missing"
    out.loc[generated, "source_name"] = "calendar_gap"
    out.loc[generated, "source_document"] = "generated calendar row"
    out.loc[generated, "source_url"] = "generated:calendar-gap"
    out.loc[generated, "source_retrieved_at"] = pd.Timestamp("1970-01-01T00:00:00Z")
    out.loc[generated, "parser_version"] = "calendar-gap-v1"
    return out
