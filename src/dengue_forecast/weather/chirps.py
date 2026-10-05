from __future__ import annotations

import math
import os
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box

from dengue_forecast.contracts import ContractError
from dengue_forecast.sources.config import source_url
from dengue_forecast.weather.grid import build_area_weights

CHIRPS_NODATA_SENTINEL = -9999.0
CHIRPS_SOURCE = "CHIRPS v3.0 daily final rnl COG"
CHIRPS_SOURCE_VERSION = "v3.0-final-rnl; 0.05deg; retrospective"
CHIRPS_COG_TEMPLATE = source_url("chirps_daily_template")
PERCENT_EPSILON = 1e-8


class CHIRPSProvider:
    source_name = CHIRPS_SOURCE
    source_version = CHIRPS_SOURCE_VERSION

    def __init__(self, *, resolver: Callable[[str, date], str] | None = None) -> None:
        self.resolver = resolver or (lambda url, _day: url)

    def open_daily(self, day: date):  # type: ignore[no-untyped-def]
        url = chirps_cog_url(day)
        target = self.resolver(url, day)
        env = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".cog",
            "GDAL_HTTP_TIMEOUT": "30",
        }
        for key, value in env.items():
            os.environ.setdefault(key, value)
        return rasterio.open(target)

    def aggregate_daily(self, districts, day: date) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        with self.open_daily(day) as dataset:
            return aggregate_daily_rainfall_subset(dataset, districts, day)


def chirps_cog_url(day: date) -> str:
    return CHIRPS_COG_TEMPLATE.format(year=day.year, date=day.strftime("%Y.%m.%d"))


def outward_snapped_window(dataset, bounds: tuple[float, float, float, float]) -> Window:  # type: ignore[no-untyped-def]
    minx, miny, maxx, maxy = bounds
    inv = ~dataset.transform
    col0, row0 = inv * (minx, maxy)
    col1, row1 = inv * (maxx, miny)
    col_off = max(0, math.floor(min(col0, col1)))
    row_off = max(0, math.floor(min(row0, row1)))
    col_end = min(dataset.width, math.ceil(max(col0, col1)))
    row_end = min(dataset.height, math.ceil(max(row0, row1)))
    return Window(col_off, row_off, max(0, col_end - col_off), max(0, row_end - row_off))


def _raster_cells(dataset) -> tuple[np.ndarray, list]:  # type: ignore[no-untyped-def]
    data = dataset.read(1).astype(float)
    if dataset.nodata is not None:
        data[data == float(dataset.nodata)] = np.nan
    data[data == CHIRPS_NODATA_SENTINEL] = np.nan
    finite = data[np.isfinite(data)]
    if finite.size and (finite < 0).any():
        raise ContractError(
            "CHIRPS rainfall contains invalid negative values besides nodata sentinel"
        )

    cell_geometries = []
    cell_values = []
    for row in range(dataset.height):
        for col in range(dataset.width):
            left, top = dataset.transform * (col, row)
            right, bottom = dataset.transform * (col + 1, row + 1)
            cell_geometries.append(box(left, bottom, right, top))
            value = data[row, col]
            cell_values.append(float(value) if np.isfinite(value) else np.nan)
    return np.asarray(cell_values, dtype=float), cell_geometries


def read_subset_to_memory(dataset, bounds: tuple[float, float, float, float]) -> MemoryFile:  # type: ignore[no-untyped-def]
    window = outward_snapped_window(dataset, bounds)
    data = dataset.read(1, window=window)
    transform = dataset.window_transform(window)
    profile = dataset.profile.copy()
    profile.update(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        transform=transform,
        count=1,
    )
    memory = MemoryFile()
    with memory.open(**profile) as subset:
        subset.write(data, 1)
    return memory


def aggregate_daily_rainfall_subset(dataset, districts, day: date) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    bounds = tuple(districts.total_bounds)
    memory = read_subset_to_memory(dataset, bounds)
    try:
        with memory.open() as subset:
            return aggregate_daily_rainfall(subset, districts, day)
    finally:
        memory.close()


def raster_values(dataset) -> np.ndarray:  # type: ignore[no-untyped-def]
    values, _ = _raster_cells(dataset)
    return values


def raster_cell_geometries(dataset) -> list:  # type: ignore[no-untyped-def]
    _, cells = _raster_cells(dataset)
    return cells


def aggregate_rainfall_values(
    values: np.ndarray, weights: pd.DataFrame, day: date
) -> pd.DataFrame:
    cell_values = pd.DataFrame(
        {"cell_id": [str(i) for i in range(len(values))], "rainfall_mm": values}
    )
    merged = weights.merge(cell_values, on="cell_id", how="left", validate="many_to_one")
    merged["valid_weight"] = np.where(merged["rainfall_mm"].notna(), merged["weight"], 0.0)
    merged["weighted_rain"] = merged["rainfall_mm"].fillna(0.0) * merged["weight"]
    rows: list[dict[str, object]] = []
    for district_id, group in merged.groupby("district_id", sort=False):
        valid_weight = float(group["valid_weight"].sum())
        rainfall = None
        if valid_weight > 0:
            rainfall = float(group["weighted_rain"].sum() / valid_weight)
        spatial_coverage = _coverage_percent(valid_weight)
        rows.append(
            {
                "district_id": district_id,
                "district_name": group["district_name"].iloc[0],
                "date": day,
                "rainfall_mm": rainfall,
                "spatial_coverage_pct": spatial_coverage,
                "grid_cells": int(group["cell_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _coverage_percent(weight: float) -> float:
    pct = float(weight) * 100.0
    if pct < -PERCENT_EPSILON or pct > 100.0 + PERCENT_EPSILON:
        raise ContractError(f"Weather spatial coverage out of bounds: {pct}")
    return min(100.0, max(0.0, pct))


def aggregate_daily_rainfall(dataset, districts, day: date) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    values, cells = _raster_cells(dataset)
    weights = build_area_weights(districts, cells)
    if weights.empty:
        raise ContractError("No CHIRPS raster cells intersect district geometry")
    return aggregate_rainfall_values(values, weights, day)


def raster_bounds_from_array(
    shape: tuple[int, int], transform
) -> tuple[float, float, float, float]:  # type: ignore[no-untyped-def]
    return tuple(array_bounds(shape[0], shape[1], transform))  # type: ignore[return-value]
