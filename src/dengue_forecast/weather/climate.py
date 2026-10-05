from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from dengue_forecast.contracts import ContractError
from dengue_forecast.sources.config import source_url
from dengue_forecast.weather.grid import build_area_weights

OPEN_METEO_ARCHIVE_URL = source_url("era5_archive")
ERA5_SOURCE = "Open-Meteo Historical Weather API ERA5"
ERA5_SOURCE_VERSION = "era5; native 0.25deg grid nodes via archive-api.open-meteo.com"
PERCENT_EPSILON = 1e-8


def kelvin_to_celsius(value: float) -> float:
    return float(value) - 273.15


def relative_humidity_from_dewpoint_c(temperature_c: float, dewpoint_c: float) -> float:
    numerator = math.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = math.exp((17.625 * temperature_c) / (243.04 + temperature_c))
    return max(0.0, min(100.0, 100.0 * numerator / denominator))


class ClimateProvider:
    source_name: str
    source_version: str

    def fetch_daily_grid(self, grid_nodes, start_date: date, end_date: date) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        raise NotImplementedError


class GLDASProvider(ClimateProvider):
    source_name = "GLDAS"
    source_version = "unsupported_without_authenticated_input_files"

    def fetch_daily_grid(
        self, grid_nodes, start_date: date, end_date: date
    ) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "GLDAS requires authenticated NASA Earthdata/file inputs; "
            "no fake public provider is implemented"
        )


class ERA5Provider(ClimateProvider):
    source_name = ERA5_SOURCE
    source_version = ERA5_SOURCE_VERSION

    def __init__(
        self,
        *,
        http_get: Callable[..., object] | None = None,
        batch_size: int = 20,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        self.http_get = http_get or requests.get
        self.batch_size = batch_size
        self.rate_limit_seconds = rate_limit_seconds

    def fetch_daily_grid(
        self, grid_nodes, start_date: date, end_date: date
    ) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        nodes = sorted({(float(lat), float(lon)) for lat, lon in grid_nodes})
        rows: list[dict[str, object]] = []
        for start in range(0, len(nodes), self.batch_size):
            batch = nodes[start : start + self.batch_size]
            params = {
                "latitude": ",".join(f"{lat:g}" for lat, _ in batch),
                "longitude": ",".join(f"{lon:g}" for _, lon in batch),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": "temperature_2m,relative_humidity_2m",
                "models": "era5",
                "timezone": "Asia/Colombo",
                "elevation": ",".join(["nan"] * len(batch)),
                "cell_selection": "nearest",
            }
            response = self.http_get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
            if getattr(response, "status_code", 200) != 200:
                status_code = getattr(response, "status_code", "unknown")
                raise ContractError(f"Open-Meteo returned HTTP {status_code}")
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            payload = normalize_era5_payload(response.json())
            if len(payload) != len(batch):
                raise ContractError("Open-Meteo response count did not match requested grid nodes")
            validate_era5_payload_hour_set(payload, batch, start_date, end_date)
            for requested, item in zip(batch, payload, strict=True):
                returned = (round(float(item["latitude"]), 6), round(float(item["longitude"]), 6))
                expected = (round(requested[0], 6), round(requested[1], 6))
                if returned != expected:
                    raise ContractError(f"Open-Meteo snapped grid node {expected} to {returned}")
                rows.extend(self._daily_rows(item, requested))
            if start + self.batch_size < len(nodes):
                time.sleep(self.rate_limit_seconds)
        return pd.DataFrame(rows)

    def _daily_rows(self, item: dict, node: tuple[float, float]) -> list[dict[str, object]]:
        _validate_era5_timezone(item)
        units = item.get("hourly_units") or {}
        if units.get("temperature_2m") != "°C" or units.get("relative_humidity_2m") != "%":
            raise ContractError(f"Unexpected Open-Meteo units: {units}")
        hourly = item.get("hourly") or {}
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(hourly.get("time", []), errors="coerce"),
                "temperature_2m": pd.to_numeric(
                    pd.Series(hourly.get("temperature_2m", [])), errors="coerce"
                ),
                "relative_humidity_2m": pd.to_numeric(
                    pd.Series(hourly.get("relative_humidity_2m", [])), errors="coerce"
                ),
            }
        )
        if frame.empty:
            return []
        if frame["time"].isna().any():
            raise ContractError("Open-Meteo response contains invalid timestamps")
        values = frame[["temperature_2m", "relative_humidity_2m"]].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ContractError("Open-Meteo contains non-finite values")
        frame["date"] = frame["time"].dt.date
        rows = []
        for day, group in frame.groupby("date"):
            complete = group["temperature_2m"].notna() & group["relative_humidity_2m"].notna()
            if int(complete.sum()) != 24 or group["time"].nunique(dropna=True) != 24:
                raise ContractError("Open-Meteo day does not contain 24 unique hourly observations")
            expected = pd.Series(pd.date_range(start=pd.Timestamp(day), periods=24, freq="h"))
            actual = group["time"].sort_values().reset_index(drop=True)
            if actual.tolist() != expected.tolist():
                raise ContractError("Open-Meteo day does not match expected local hourly grid")
            humidity = group["relative_humidity_2m"]
            if humidity.lt(0).any() or humidity.gt(100).any():
                raise ContractError("Open-Meteo relative humidity outside 0..100")
            temps = group["temperature_2m"]
            rows.append(
                {
                    "latitude": node[0],
                    "longitude": node[1],
                    "cell_id": f"{node[0]:.4f},{node[1]:.4f}",
                    "date": day,
                    "temp_mean_c": float(temps.mean()),
                    "temp_min_c": float(temps.min()),
                    "temp_max_c": float(temps.max()),
                    "humidity_mean_pct": float(humidity.mean()),
                    "valid_hours": 24,
                }
            )
        return rows


def normalize_era5_payload(payload: object) -> list[dict]:
    if isinstance(payload, bytes):
        payload = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ContractError("Open-Meteo response must be a JSON object or list")


def _validate_era5_timezone(item: dict) -> None:
    if item.get("timezone") != "Asia/Colombo" or item.get("utc_offset_seconds") != 19800:
        raise ContractError(
            "Open-Meteo timezone metadata must be Asia/Colombo with utc_offset_seconds=19800"
        )


def validate_era5_payload_hour_set(
    payload: list[dict],
    nodes: list[tuple[float, float]],
    start_date: date,
    end_date: date,
) -> None:
    expected_hours = pd.Series(
        pd.date_range(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date) + pd.Timedelta(hours=23),
            freq="h",
        )
    )
    if len(payload) != len(nodes):
        raise ContractError("Open-Meteo response count did not match requested grid nodes")
    for node, item in zip(nodes, payload, strict=True):
        _validate_era5_timezone(item)
        hourly = item.get("hourly") or {}
        times = pd.to_datetime(pd.Series(hourly.get("time", [])), errors="coerce")
        if times.isna().any():
            raise ContractError("Open-Meteo response contains invalid timestamps")
        actual_hours = times.sort_values().reset_index(drop=True)
        if actual_hours.tolist() != expected_hours.tolist():
            raise ContractError(
                f"Open-Meteo cached response for {node} does not cover requested hourly set"
            )


def aggregate_daily_climate(grid_daily: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if grid_daily.empty or weights.empty:
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
    merged = weights.merge(grid_daily, on="cell_id", how="left", validate="many_to_many")
    rows: list[dict[str, object]] = []
    variables = ["temp_mean_c", "temp_min_c", "temp_max_c", "humidity_mean_pct"]
    grouped = merged.groupby(["district_id", "date"], dropna=True, sort=False)
    for (district_id, day), group in grouped:
        valid = group[variables].notna().all(axis=1)
        valid_weight = float(group.loc[valid, "weight"].sum())
        values: dict[str, float | None] = {}
        for variable in variables:
            if valid_weight > 0:
                weighted_sum = (group.loc[valid, variable] * group.loc[valid, "weight"]).sum()
                values[variable] = float(weighted_sum / valid_weight)
            else:
                values[variable] = None
        rows.append(
            {
                "district_id": district_id,
                "district_name": group["district_name"].iloc[0],
                "date": day,
                **values,
                "spatial_coverage_pct": _coverage_percent(valid_weight),
                "grid_cells": int(group["cell_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _coverage_percent(weight: float) -> float:
    pct = float(weight) * 100.0
    if pct < -PERCENT_EPSILON or pct > 100.0 + PERCENT_EPSILON:
        raise ContractError(f"Weather spatial coverage out of bounds: {pct}")
    return min(100.0, max(0.0, pct))


def climate_weights_for_districts(
    districts, resolution: float = 0.25
) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    from dengue_forecast.weather.grid import native_grid_cells_for_bounds

    if hasattr(districts, "total_bounds"):
        bounds = tuple(districts.total_bounds)
    else:
        bounds = tuple(gpd.GeoSeries(districts["geometry"], crs="EPSG:4326").total_bounds)
    cells = native_grid_cells_for_bounds(bounds, resolution)
    return build_area_weights(districts, cells)
