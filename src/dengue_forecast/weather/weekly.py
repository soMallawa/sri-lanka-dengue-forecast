from __future__ import annotations

import pandas as pd

from dengue_forecast.contracts import ContractError, validate_weather_weekly
from dengue_forecast.weather.chirps import CHIRPS_SOURCE, CHIRPS_SOURCE_VERSION
from dengue_forecast.weather.climate import ERA5_SOURCE, ERA5_SOURCE_VERSION


def _empty_climate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "district_id",
            "district_name",
            "date",
            "temp_mean_c",
            "temp_min_c",
            "temp_max_c",
            "humidity_mean_pct",
            "spatial_coverage_pct",
            "grid_cells",
        ]
    )


def _validate_daily_keys(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        return
    required = {"district_id", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"{label} daily frame missing columns: {sorted(missing)}")
    duplicates = frame.duplicated(["district_id", "date"], keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, ["district_id", "date"]].head(5).to_dict("records")
        raise ContractError(f"Duplicate weather daily key in {label}: {sample}")


def _partition_by_district(frame: pd.DataFrame) -> dict[object, pd.DataFrame]:
    if frame.empty:
        return {}
    return {
        district_id: district_frame
        for district_id, district_frame in frame.groupby("district_id", sort=False)
    }


def _slice_dates(frame: pd.DataFrame | None, start, end) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame[(frame["date"] >= start) & (frame["date"] <= end)]


def combine_weekly_weather(
    weeks: pd.DataFrame,
    *,
    rainfall_daily: pd.DataFrame | None,
    climate_daily: pd.DataFrame | None,
    min_temporal_coverage_pct: float = 85.0,
    min_spatial_coverage_pct: float = 90.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rainfall = rainfall_daily.copy() if rainfall_daily is not None else pd.DataFrame()
    climate = climate_daily.copy() if climate_daily is not None else _empty_climate_frame()
    if not rainfall.empty:
        rainfall["date"] = pd.to_datetime(rainfall["date"]).dt.date
    if not climate.empty:
        climate["date"] = pd.to_datetime(climate["date"]).dt.date
    _validate_daily_keys(rainfall, "rainfall")
    _validate_daily_keys(climate, "climate")
    rainfall_by_district = _partition_by_district(rainfall)
    climate_by_district = _partition_by_district(climate)

    for _, week in weeks.iterrows():
        start = pd.Timestamp(week["week_start_date"]).date()
        end = pd.Timestamp(week["week_end_date"]).date()
        district_id = week["district_id"]
        expected_days = (end - start).days + 1

        rain_slice = _slice_dates(rainfall_by_district.get(district_id), start, end)
        climate_slice = _slice_dates(climate_by_district.get(district_id), start, end)

        rain_valid_dates = set()
        if not rain_slice.empty and "rainfall_mm" in rain_slice:
            rain_valid_dates = set(rain_slice.loc[rain_slice["rainfall_mm"].notna(), "date"])

        climate_vars = ["temp_mean_c", "temp_min_c", "temp_max_c", "humidity_mean_pct"]
        climate_valid_dates = set()
        if not climate_slice.empty and set(climate_vars).issubset(climate_slice.columns):
            complete_climate = climate_slice[climate_vars].notna().all(axis=1)
            climate_valid_dates = set(climate_slice.loc[complete_climate, "date"])

        available_dates = rain_valid_dates & climate_valid_dates
        available_days = len(available_dates)
        temporal_coverage = 100.0 * available_days / expected_days if expected_days else None
        spatial_values = []
        if not rain_slice.empty:
            valid_rain_spatial = rain_slice[rain_slice["date"].isin(rain_valid_dates)]
            spatial_values.extend(
                valid_rain_spatial["spatial_coverage_pct"].dropna().astype(float).tolist()
            )
        if not climate_slice.empty and "spatial_coverage_pct" in climate_slice:
            valid_climate_spatial = climate_slice[climate_slice["date"].isin(climate_valid_dates)]
            spatial_values.extend(
                valid_climate_spatial["spatial_coverage_pct"].dropna().astype(float).tolist()
            )
        spatial_coverage = min(spatial_values) if spatial_values else None

        flags = []
        if temporal_coverage is None or temporal_coverage < min_temporal_coverage_pct:
            flags.append("low_temporal_coverage")
        if spatial_coverage is not None and spatial_coverage < min_spatial_coverage_pct:
            flags.append("low_spatial_coverage")
        if rainfall_daily is None or rain_slice.empty or not rain_valid_dates:
            flags.append("rain_missing")
        if climate_daily is None or climate_slice.empty or not climate_valid_dates:
            flags.append("climate_missing")

        rain_values = (
            rain_slice.loc[rain_slice["date"].isin(rain_valid_dates), "rainfall_mm"].dropna()
            if not rain_slice.empty
            else pd.Series(dtype=float)
        )
        rainfall_sum = float(rain_values.sum()) if not rain_values.empty else None
        rainfall_mean = float(rain_values.mean()) if not rain_values.empty else None
        rainfall_max = float(rain_values.max()) if not rain_values.empty else None
        rain_days_1mm = int((rain_values >= 1.0).sum()) if not rain_values.empty else None
        rain_days_10mm = int((rain_values >= 10.0).sum()) if not rain_values.empty else None
        valid_climate_slice = (
            climate_slice[climate_slice["date"].isin(climate_valid_dates)]
            if not climate_slice.empty
            else pd.DataFrame()
        )
        temp_mean = (
            float(valid_climate_slice["temp_mean_c"].mean())
            if not valid_climate_slice.empty
            else None
        )
        temp_min = (
            float(valid_climate_slice["temp_min_c"].min())
            if not valid_climate_slice.empty
            else None
        )
        temp_max = (
            float(valid_climate_slice["temp_max_c"].max())
            if not valid_climate_slice.empty
            else None
        )
        humidity_mean = (
            float(valid_climate_slice["humidity_mean_pct"].mean())
            if not valid_climate_slice.empty
            else None
        )
        rows.append(
            {
                "district_id": district_id,
                "district_name": week["district_name"],
                "week_start_date": start,
                "week_end_date": end,
                "rainfall_sum_mm": rainfall_sum,
                "rainfall_mean_daily_mm": rainfall_mean,
                "rainfall_max_daily_mm": rainfall_max,
                "rain_days_1mm": rain_days_1mm,
                "rain_days_10mm": rain_days_10mm,
                "temp_mean_c": temp_mean,
                "temp_min_c": temp_min,
                "temp_max_c": temp_max,
                "humidity_mean_pct": humidity_mean,
                "weather_expected_days": expected_days,
                "weather_available_days": available_days,
                "weather_temporal_coverage_pct": temporal_coverage,
                "weather_grid_cells": int(
                    max(
                        rain_slice["grid_cells"].max() if not rain_slice.empty else 0,
                        climate_slice["grid_cells"].max()
                        if not climate_slice.empty and "grid_cells" in climate_slice
                        else 0,
                    )
                ),
                "weather_spatial_coverage_pct": spatial_coverage,
                "weather_quality_flag": ";".join(flags) if flags else "ok",
                "rain_source": CHIRPS_SOURCE,
                "rain_source_version": CHIRPS_SOURCE_VERSION,
                "climate_source": ERA5_SOURCE,
                "climate_source_version": ERA5_SOURCE_VERSION,
            }
        )

    out = pd.DataFrame(rows)
    validate_weather_weekly(out)
    return out
