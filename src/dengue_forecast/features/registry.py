from __future__ import annotations

from dataclasses import dataclass

from dengue_forecast.contracts import ContractError


@dataclass(frozen=True)
class FeatureDefinition:
    feature_group: str
    dtype: str
    description: str
    source: str
    transformation: str
    lag_weeks: str = ""
    uses_future_information: bool = False
    eligible_for_training: bool = False


TARGET_COLUMNS = {
    "cases_next_week",
    "cases_next_2w",
    "cases_next_4w",
    "incidence_next_week_per_100k",
}


def _known_definitions() -> dict[str, FeatureDefinition]:
    definitions: dict[str, FeatureDefinition] = {
        "district_id": FeatureDefinition(
            "identifier", "string", "Canonical district ID", "contract", "identity"
        ),
        "district_name": FeatureDefinition(
            "identifier", "string", "Canonical district name", "contract", "identity"
        ),
        "week_start_date": FeatureDefinition(
            "identifier", "date", "Week start date", "contract", "identity"
        ),
        "week_end_date": FeatureDefinition(
            "identifier", "date", "Week end date", "contract", "identity"
        ),
        "year": FeatureDefinition(
            "identifier",
            "integer",
            "Source-reported epidemiological year label",
            "dengue_cases_weekly",
            "calendar mapping",
        ),
        "week": FeatureDefinition(
            "identifier",
            "integer",
            "Source-reported epidemiological week label",
            "dengue_cases_weekly",
            "calendar mapping",
        ),
        "year_week": FeatureDefinition(
            "identifier",
            "string",
            "Source-reported year-week label",
            "dengue_cases_weekly",
            "calendar mapping",
        ),
        "month": FeatureDefinition(
            "seasonality",
            "integer",
            "Calendar month from week_start_date",
            "calendar",
            "derived",
            eligible_for_training=True,
        ),
        "quarter": FeatureDefinition(
            "seasonality",
            "integer",
            "Calendar quarter from week_start_date",
            "calendar",
            "derived",
            eligible_for_training=True,
        ),
        "week_of_year": FeatureDefinition(
            "seasonality",
            "integer",
            "Seasonality week label using source week when available",
            "calendar",
            "derived",
            eligible_for_training=True,
        ),
        "week_sin": FeatureDefinition(
            "seasonality",
            "float",
            "Cyclical week sine",
            "calendar",
            "sin(2*pi*week/52.1775)",
            eligible_for_training=True,
        ),
        "week_cos": FeatureDefinition(
            "seasonality",
            "float",
            "Cyclical week cosine",
            "calendar",
            "cos(2*pi*week/52.1775)",
            eligible_for_training=True,
        ),
        "month_sin": FeatureDefinition(
            "seasonality",
            "float",
            "Cyclical month sine",
            "calendar",
            "sin(2*pi*month/12)",
            eligible_for_training=True,
        ),
        "month_cos": FeatureDefinition(
            "seasonality",
            "float",
            "Cyclical month cosine",
            "calendar",
            "cos(2*pi*month/12)",
            eligible_for_training=True,
        ),
        "dengue_cases": FeatureDefinition(
            "epidemiology",
            "integer",
            "Reported weekly dengue cases",
            "dengue_cases_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "case_status": FeatureDefinition(
            "quality",
            "string",
            "Case observation status",
            "dengue_cases_weekly/calendar",
            "identity",
        ),
        "source_name": FeatureDefinition(
            "metadata",
            "string",
            "Dengue source or generated calendar-gap marker",
            "dengue_cases_weekly/calendar",
            "identity",
        ),
        "source_document": FeatureDefinition(
            "metadata",
            "string",
            "Dengue source document or generated marker",
            "dengue_cases_weekly/calendar",
            "identity",
        ),
        "source_url": FeatureDefinition(
            "metadata",
            "string",
            "Dengue source URL or generated marker",
            "dengue_cases_weekly/calendar",
            "identity",
        ),
        "source_retrieved_at": FeatureDefinition(
            "metadata",
            "datetime",
            "Dengue source retrieval timestamp",
            "dengue_cases_weekly/calendar",
            "identity",
        ),
        "parser_version": FeatureDefinition(
            "metadata",
            "string",
            "Parser or calendar derivation version",
            "dengue_cases_weekly/calendar",
            "identity",
        ),
        "rainfall_sum_mm": FeatureDefinition(
            "weather",
            "float",
            "Weekly district rainfall total depth",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "rainfall_mean_daily_mm": FeatureDefinition(
            "weather",
            "float",
            "Mean daily rainfall",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "rainfall_max_daily_mm": FeatureDefinition(
            "weather",
            "float",
            "Maximum daily rainfall",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "rain_days_1mm": FeatureDefinition(
            "weather",
            "integer",
            "Days with rainfall >= 1mm",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "rain_days_10mm": FeatureDefinition(
            "weather",
            "integer",
            "Days with rainfall >= 10mm",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "temp_mean_c": FeatureDefinition(
            "weather",
            "float",
            "Mean air temperature C",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "temp_min_c": FeatureDefinition(
            "weather",
            "float",
            "Minimum air temperature C",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "temp_max_c": FeatureDefinition(
            "weather",
            "float",
            "Maximum air temperature C",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "humidity_mean_pct": FeatureDefinition(
            "weather",
            "float",
            "Mean relative humidity percent",
            "district_weather_weekly",
            "identity",
            eligible_for_training=True,
        ),
        "weather_expected_days": FeatureDefinition(
            "quality",
            "integer",
            "Expected weather days in interval",
            "district_weather_weekly",
            "identity",
        ),
        "weather_available_days": FeatureDefinition(
            "quality",
            "integer",
            "Available weather days in interval",
            "district_weather_weekly",
            "identity",
        ),
        "weather_temporal_coverage_pct": FeatureDefinition(
            "quality",
            "float",
            "Minimum temporal weather coverage percent",
            "district_weather_weekly",
            "identity",
        ),
        "weather_grid_cells": FeatureDefinition(
            "quality", "integer", "Weather grid cells used", "district_weather_weekly", "identity"
        ),
        "weather_spatial_coverage_pct": FeatureDefinition(
            "quality",
            "float",
            "Minimum spatial weather coverage percent",
            "district_weather_weekly",
            "identity",
        ),
        "weather_quality_flag": FeatureDefinition(
            "quality", "string", "Weather quality flag", "district_weather_weekly", "identity"
        ),
        "rain_source": FeatureDefinition(
            "metadata", "string", "Rainfall source name", "district_weather_weekly", "identity"
        ),
        "rain_source_version": FeatureDefinition(
            "metadata", "string", "Rainfall source version", "district_weather_weekly", "identity"
        ),
        "climate_source": FeatureDefinition(
            "metadata", "string", "Climate source name", "district_weather_weekly", "identity"
        ),
        "climate_source_version": FeatureDefinition(
            "metadata", "string", "Climate source version", "district_weather_weekly", "identity"
        ),
        "province_name": FeatureDefinition(
            "reference", "string", "Province name", "district_reference", "identity"
        ),
        "population_reference": FeatureDefinition(
            "reference",
            "integer",
            "Retrospective population reference denominator",
            "district_reference",
            "identity",
        ),
        "population_reference_year": FeatureDefinition(
            "reference", "integer", "Population reference year", "district_reference", "identity"
        ),
        "population_method": FeatureDefinition(
            "reference", "string", "Population method", "district_reference", "identity"
        ),
        "population_density_per_km2": FeatureDefinition(
            "reference",
            "float",
            "Retrospective population density reference",
            "district_reference",
            "identity",
        ),
        "area_km2": FeatureDefinition(
            "reference",
            "float",
            "District area in square kilometres",
            "district_reference",
            "identity",
        ),
        "centroid_lat": FeatureDefinition(
            "reference", "float", "District centroid latitude", "district_reference", "identity"
        ),
        "centroid_lon": FeatureDefinition(
            "reference", "float", "District centroid longitude", "district_reference", "identity"
        ),
        "geometry_source": FeatureDefinition(
            "metadata", "string", "Geometry source", "district_reference", "identity"
        ),
        "geometry_source_version": FeatureDefinition(
            "metadata", "string", "Geometry source version", "district_reference", "identity"
        ),
        "population_source": FeatureDefinition(
            "metadata", "string", "Population source", "district_reference", "identity"
        ),
        "population_source_version": FeatureDefinition(
            "metadata", "string", "Population source version", "district_reference", "identity"
        ),
        "incidence_per_100k_using_2024_population": FeatureDefinition(
            "reference",
            "float",
            "Retrospective incidence using 2024 reference denominator; not point-in-time eligible",
            "dengue/population_reference_2024",
            "dengue_cases / population_reference * 100000",
        ),
        "case_missing_flag": FeatureDefinition(
            "quality", "boolean", "Dengue case value is missing", "derived", "isna(dengue_cases)"
        ),
        "weather_missing_flag": FeatureDefinition(
            "quality",
            "boolean",
            "Any core weather value is missing",
            "derived",
            "weather core isna",
        ),
        "case_history_missing_flag": FeatureDefinition(
            "quality", "boolean", "Required case lags are missing", "derived", "required lag isna"
        ),
        "row_quality_flag": FeatureDefinition(
            "quality", "string", "Row quality classification", "derived", "quality rule"
        ),
        "is_trainable": FeatureDefinition(
            "quality", "boolean", "Baseline training eligibility", "derived", "quality rule"
        ),
        "days_since_previous_observation": FeatureDefinition(
            "quality",
            "float",
            "Days since previous non-null case observation",
            "derived",
            "observed-date delta",
        ),
        "is_consecutive_week": FeatureDefinition(
            "quality",
            "boolean",
            "Previous observation is exactly seven days earlier",
            "derived",
            "observed-date delta == 7",
        ),
        "cases_next_week": FeatureDefinition(
            "target",
            "integer",
            "Cases exactly one week ahead",
            "dengue_cases_weekly",
            "exact-date future join",
            "-1",
            True,
            False,
        ),
        "cases_next_2w": FeatureDefinition(
            "target",
            "integer",
            "Cases exactly two weeks ahead",
            "dengue_cases_weekly",
            "exact-date future join",
            "-2",
            True,
            False,
        ),
        "cases_next_4w": FeatureDefinition(
            "target",
            "integer",
            "Cases exactly four weeks ahead",
            "dengue_cases_weekly",
            "exact-date future join",
            "-4",
            True,
            False,
        ),
        "incidence_next_week_per_100k": FeatureDefinition(
            "target",
            "float",
            "Next-week incidence with retrospective denominator",
            "dengue/population_reference_2024",
            "future target / population_reference",
            "-1",
            True,
            False,
        ),
    }
    for lag in [1, 2, 3, 4, 6, 8]:
        definitions[f"cases_lag_{lag}"] = FeatureDefinition(
            "epidemiology",
            "integer",
            f"Cases lagged {lag} week(s)",
            "dengue_cases_weekly",
            "district-isolated exact calendar lag",
            str(lag),
            False,
            True,
        )
        definitions[f"rainfall_lag_{lag}"] = FeatureDefinition(
            "weather",
            "float",
            f"Rainfall lagged {lag} week(s)",
            "district_weather_weekly",
            "district-isolated exact calendar lag",
            str(lag),
            False,
            True,
        )
    for lag in [1, 2, 4]:
        definitions[f"temp_mean_lag_{lag}"] = FeatureDefinition(
            "weather",
            "float",
            f"Temperature lagged {lag} week(s)",
            "district_weather_weekly",
            "district-isolated exact calendar lag",
            str(lag),
            False,
            True,
        )
        definitions[f"humidity_lag_{lag}"] = FeatureDefinition(
            "weather",
            "float",
            f"Humidity lagged {lag} week(s)",
            "district_weather_weekly",
            "district-isolated exact calendar lag",
            str(lag),
            False,
            True,
        )
    for window in [2, 4, 8]:
        definitions[f"cases_roll_mean_{window}"] = FeatureDefinition(
            "epidemiology",
            "float",
            f"Cases rolling mean over {window} weeks",
            "dengue_cases_weekly",
            "rolling window, full observations required",
            f"0-{window - 1}",
            False,
            True,
        )
        definitions[f"rainfall_roll_sum_{window}"] = FeatureDefinition(
            "weather",
            "float",
            f"Rainfall rolling sum over {window} weeks",
            "district_weather_weekly",
            "rolling window, full observations required",
            f"0-{window - 1}",
            False,
            True,
        )
    for name in ["cases_roll_std_4", "cases_roll_std_8", "cases_roll_min_4", "cases_roll_max_4"]:
        definitions[name] = FeatureDefinition(
            "epidemiology",
            "float",
            name.replace("_", " "),
            "dengue_cases_weekly",
            "rolling window, full observations required",
            eligible_for_training=True,
        )
    for name in [
        "cases_change_1w",
        "cases_change_2w",
        "cases_pct_change_1w",
        "cases_pct_change_4w",
        "cases_slope_4w",
        "log1p_cases",
    ]:
        definitions[name] = FeatureDefinition(
            "epidemiology",
            "float",
            name.replace("_", " "),
            "dengue_cases_weekly",
            "district-isolated trend",
            eligible_for_training=True,
        )
    for name in [
        "rainfall_roll_mean_4",
        "rainfall_roll_max_4",
        "rain_days_roll_sum_4",
        "temp_roll_mean_2",
        "temp_roll_mean_4",
        "temp_roll_std_4",
        "humidity_roll_mean_2",
        "humidity_roll_mean_4",
    ]:
        definitions[name] = FeatureDefinition(
            "weather",
            "float",
            name.replace("_", " "),
            "district_weather_weekly",
            "rolling window, full observations required",
            eligible_for_training=True,
        )
    return definitions


def _is_future_like(name: str) -> bool:
    lowered = name.casefold()
    return (
        name in TARGET_COLUMNS
        or "next" in lowered
        or "future" in lowered
        or "actual_next" in lowered
    )


def _metadata_for(name: str) -> dict[str, object]:
    definitions = _known_definitions()
    definition = definitions.get(name)
    if definition is None:
        future = _is_future_like(name)
        definition = FeatureDefinition(
            feature_group="unknown_future" if future else "unknown",
            dtype="unknown",
            description=f"Unrecognized column {name}; fail-closed and not trainable",
            source="unknown",
            transformation="unknown",
            uses_future_information=future,
            eligible_for_training=False,
        )
    return {
        "feature_name": name,
        "feature_group": definition.feature_group,
        "dtype": definition.dtype,
        "description": definition.description,
        "source": definition.source,
        "transformation": definition.transformation,
        "lag_weeks": definition.lag_weeks,
        "uses_future_information": definition.uses_future_information,
        "eligible_for_training": definition.eligible_for_training,
    }


def build_feature_registry(columns: list[str]):  # type: ignore[no-untyped-def]
    import pandas as pd

    rows = [_metadata_for(column) for column in columns]
    registry = pd.DataFrame(rows)
    duplicate = registry["feature_name"].duplicated()
    if duplicate.any():
        duplicates = registry.loc[duplicate, "feature_name"].tolist()
        raise ContractError(f"Feature registry has duplicate names: {duplicates}")
    for column in ["uses_future_information", "eligible_for_training"]:
        if not registry[column].map(lambda value: isinstance(value, bool)).all():
            raise ContractError(f"Feature registry column {column} must contain boolean values")
    future_trainable = registry[
        registry["uses_future_information"] & registry["eligible_for_training"]
    ]
    if not future_trainable.empty:
        future_names = future_trainable["feature_name"].tolist()
        raise ContractError(
            f"Feature registry marks future information as trainable: {future_names}"
        )
    return registry
