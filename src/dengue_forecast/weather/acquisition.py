from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterio.io import MemoryFile

from dengue_forecast.contracts import ContractError
from dengue_forecast.weather.chirps import (
    CHIRPS_SOURCE_VERSION,
    aggregate_rainfall_values,
    chirps_cog_url,
    outward_snapped_window,
    raster_cell_geometries,
    raster_values,
)
from dengue_forecast.weather.climate import (
    ERA5_SOURCE_VERSION,
    OPEN_METEO_ARCHIVE_URL,
    ERA5Provider,
    aggregate_daily_climate,
    climate_weights_for_districts,
    normalize_era5_payload,
    validate_era5_payload_hour_set,
)
from dengue_forecast.weather.grid import build_area_weights


@dataclass(frozen=True)
class CachedWeatherSource:
    path: Path
    metadata_path: Path
    sha256: str


class WeatherQuotaBlocked(ContractError):
    """Open-Meteo quota blocked this run."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int | None = None,
        quota_window: str = "unknown",
        provider_error: dict[str, Any] | None = None,
        provider_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.quota_window = quota_window
        self.provider_error = provider_error
        self.provider_body = provider_body


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _date_range(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        raise ContractError("Weather end_date must be on or after start_date")
    days = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_cached_source(metadata_path: Path) -> tuple[CachedWeatherSource, dict[str, Any]]:
    metadata = _read_json(metadata_path)
    filename = str(metadata.get("filename", ""))
    if Path(filename).name != filename:
        raise ContractError(f"Unsafe cached weather filename in {metadata_path}: {filename}")
    path = metadata_path.with_name(filename)
    if not path.exists():
        raise ContractError(f"Cached weather payload missing for {metadata_path}")
    content = path.read_bytes()
    expected_length = int(metadata.get("byte_length", -1))
    if len(content) != expected_length:
        raise ContractError(f"Cached weather byte_length mismatch for {path}")
    digest = _sha256(content)
    if digest != metadata.get("sha256"):
        raise ContractError(f"Cached weather SHA256 mismatch for {path}")
    return CachedWeatherSource(path=path, metadata_path=metadata_path, sha256=digest), metadata


def _retrieval_sort_key(item: tuple[CachedWeatherSource, dict[str, Any]]) -> tuple[str, str]:
    cached, metadata = item
    return str(metadata.get("retrieved_at", "")), cached.sha256


def _bounds_cover(
    covering: tuple[float, float, float, float],
    requested: tuple[float, float, float, float],
) -> bool:
    tol = 1e-9
    return (
        covering[0] <= requested[0] + tol
        and covering[1] <= requested[1] + tol
        and covering[2] >= requested[2] - tol
        and covering[3] >= requested[3] - tol
    )


def _raster_fingerprint(dataset) -> tuple[str, tuple[float, ...], int, int]:  # type: ignore[no-untyped-def]
    return (
        str(dataset.crs),
        tuple(round(value, 12) for value in dataset.transform.to_gdal()),
        int(dataset.width),
        int(dataset.height),
    )


def _chirps_root(raw_dir: Path) -> Path:
    return Path(raw_dir) / "chirps_v3_rnl"


def _era5_root(raw_dir: Path) -> Path:
    return Path(raw_dir) / "era5_openmeteo"


def _find_chirps_subset(
    raw_dir: Path, day: date, bounds: tuple[float, float, float, float] | None = None
) -> CachedWeatherSource | None:
    root = _chirps_root(raw_dir) / f"{day.year:04d}"
    candidates: list[tuple[CachedWeatherSource, dict[str, Any]]] = []
    for metadata_path in sorted(root.glob(f"chirps_v3_rnl_{day:%Y%m%d}_*.metadata.json")):
        cached, metadata = _verified_cached_source(metadata_path)
        if metadata.get("date") != day.isoformat():
            continue
        if bounds is not None:
            request_bounds = tuple(float(value) for value in metadata.get("request_bounds", ()))
            if len(request_bounds) != 4 or not _bounds_cover(request_bounds, bounds):
                continue
            with rasterio.open(cached.path) as dataset:
                raster_bounds = tuple(float(value) for value in dataset.bounds)
            if not _bounds_cover(raster_bounds, bounds):
                continue
        candidates.append((cached, metadata))
    if not candidates:
        return None
    return max(candidates, key=_retrieval_sort_key)[0]


def cache_chirps_subset(
    dataset,
    *,
    bounds: tuple[float, float, float, float],
    day: date,
    raw_dir: Path,
    upstream_url: str,
    force: bool = False,
) -> CachedWeatherSource:
    if not force:
        cached = _find_chirps_subset(raw_dir, day, bounds=bounds)
        if cached is not None:
            return cached

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

    with MemoryFile() as memory:
        with memory.open(**profile) as subset:
            subset.write(data, 1)
        content = memory.read()

    digest = _sha256(content)
    root = _chirps_root(raw_dir) / f"{day.year:04d}"
    root.mkdir(parents=True, exist_ok=True)
    filename = f"chirps_v3_rnl_{day:%Y%m%d}_{digest[:12]}.tif"
    path = root / filename
    if not path.exists():
        path.write_bytes(content)
    metadata_path = root / f"{path.stem}.metadata.json"
    metadata = {
        "filename": filename,
        "sha256": digest,
        "byte_length": path.stat().st_size,
        "date": day.isoformat(),
        "source": "CHIRPS v3.0 daily final rnl COG",
        "source_version": CHIRPS_SOURCE_VERSION,
        "upstream_url": upstream_url,
        "utc_day_semantic": "daily rainfall total for UTC source day",
        "retrieved_at": _now(),
        "request_bounds": list(bounds),
        "request_window": {
            "col_off": int(window.col_off),
            "row_off": int(window.row_off),
            "width": int(window.width),
            "height": int(window.height),
        },
        "subset_transform": list(transform.to_gdal()),
    }
    if not metadata_path.exists():
        _write_json(metadata_path, metadata)
    return CachedWeatherSource(path=path, metadata_path=metadata_path, sha256=digest)


def _find_era5_response(
    raw_dir: Path,
    start_date: date,
    end_date: date,
    batch_index: int,
    expected_params: dict[str, str] | None = None,
) -> CachedWeatherSource | None:
    root = _era5_root(raw_dir)
    pattern = f"era5_openmeteo_*_batch{batch_index:03d}_*.metadata.json"
    candidates: list[tuple[CachedWeatherSource, dict[str, Any]]] = []
    for metadata_path in sorted(root.glob(pattern)):
        cached, metadata = _verified_cached_source(metadata_path)
        cached_start = date.fromisoformat(str(metadata.get("start_date")))
        cached_end = date.fromisoformat(str(metadata.get("end_date")))
        if cached_start > start_date or cached_end < end_date:
            continue
        if expected_params is not None and not _same_era5_identity(
            metadata.get("request_params", {}), expected_params
        ):
            continue
        candidates.append((cached, metadata))
    if not candidates:
        return None
    return max(candidates, key=_retrieval_sort_key)[0]


def _same_era5_identity(actual: dict[str, Any], expected: dict[str, str]) -> bool:
    identity_keys = [
        "latitude",
        "longitude",
        "hourly",
        "models",
        "timezone",
        "elevation",
        "cell_selection",
    ]
    return all(str(actual.get(key)) == str(expected.get(key)) for key in identity_keys)


def cache_era5_response(
    content: bytes,
    *,
    raw_dir: Path,
    request_params: dict[str, str],
    start_date: date,
    end_date: date,
    batch_index: int = 0,
    force: bool = False,
) -> CachedWeatherSource:
    if not force:
        cached = _find_era5_response(
            raw_dir,
            start_date,
            end_date,
            batch_index,
            expected_params=request_params,
        )
        if cached is not None and cached.path.read_bytes() == content:
            return cached
    digest = _sha256(content)
    root = _era5_root(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    filename = (
        f"era5_openmeteo_{start_date:%Y%m%d}_{end_date:%Y%m%d}_"
        f"batch{batch_index:03d}_{digest[:12]}.json"
    )
    path = root / filename
    if not path.exists():
        path.write_bytes(content)
    metadata_path = root / f"{path.stem}.metadata.json"
    metadata = {
        "filename": filename,
        "sha256": digest,
        "byte_length": path.stat().st_size,
        "source": "Open-Meteo Historical Weather API ERA5",
        "source_version": ERA5_SOURCE_VERSION,
        "request_url": OPEN_METEO_ARCHIVE_URL,
        "request_params": request_params,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "retrieved_at": _now(),
    }
    if not metadata_path.exists():
        _write_json(metadata_path, metadata)
    return CachedWeatherSource(path=path, metadata_path=metadata_path, sha256=digest)


def _era5_request_params(
    nodes: list[tuple[float, float]], start_date: date, end_date: date
) -> dict[str, str]:
    return {
        "latitude": ",".join(f"{lat:g}" for lat, _ in nodes),
        "longitude": ",".join(f"{lon:g}" for _, lon in nodes),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m",
        "models": "era5",
        "timezone": "Asia/Colombo",
        "elevation": ",".join(["nan"] * len(nodes)),
        "cell_selection": "nearest",
    }


def _default_chirps_open(url: str):
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".cog",
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_CACHEMAX": "32",
    }
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return rasterio.open(url)


def _fetch_era5_with_retry(
    params: dict[str, str], *, max_retry_after_seconds: int | None = None
) -> bytes:
    minute_quota_retries = 0
    for attempt in range(4):
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)
        if response.status_code == 429:
            provider_error, provider_body = _response_error_payload(response)
            quota_window = _classify_quota_window(provider_error, provider_body)
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            if retry_after is None:
                if quota_window != "minutely":
                    raise WeatherQuotaBlocked(
                        _quota_message(
                            429, quota_window, retry_after, provider_error, provider_body
                        ),
                        retry_after_seconds=None,
                        quota_window=quota_window,
                        provider_error=provider_error,
                        provider_body=provider_body,
                    )
                retry_after = 61
                minute_quota_retries += 1
                if minute_quota_retries > 2:
                    raise WeatherQuotaBlocked(
                        _quota_message(
                            429, quota_window, retry_after, provider_error, provider_body
                        ),
                        retry_after_seconds=retry_after,
                        quota_window=quota_window,
                        provider_error=provider_error,
                        provider_body=provider_body,
                    )
            if max_retry_after_seconds is not None and retry_after > max_retry_after_seconds:
                raise WeatherQuotaBlocked(
                    _quota_message(429, quota_window, retry_after, provider_error, provider_body),
                    retry_after_seconds=retry_after,
                    quota_window=quota_window,
                    provider_error=provider_error,
                    provider_body=provider_body,
                )
            time.sleep(retry_after)
            continue
        if response.status_code >= 500 and attempt < 3:
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code != 200:
            raise ContractError(f"Open-Meteo returned HTTP {response.status_code}")
        return response.content
    raise ContractError("Open-Meteo request failed after retries")


def _response_error_payload(
    response: requests.Response,
) -> tuple[dict[str, Any] | None, str | None]:
    body = getattr(response, "text", None)
    if body is None:
        content = getattr(response, "content", b"")
        if isinstance(content, bytes):
            body = content.decode("utf-8", errors="replace")
        elif content:
            body = str(content)
    try:
        payload = response.json()
    except (TypeError, ValueError, AttributeError):
        payload = None
    if isinstance(payload, dict):
        return payload, body
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None, body
        if isinstance(parsed, dict):
            return parsed, body
    return None, body


def _classify_quota_window(
    provider_error: dict[str, Any] | None, provider_body: str | None
) -> str:
    text = " ".join(
        str(value)
        for value in [
            provider_error.get("reason") if provider_error else None,
            provider_error.get("error") if provider_error else None,
            provider_error.get("message") if provider_error else None,
            provider_body,
        ]
        if value
    ).lower()
    if "minute" in text or "minutely" in text:
        return "minutely"
    if "hour" in text or "hourly" in text:
        return "hourly"
    if "day" in text or "daily" in text:
        return "daily"
    return "unknown"


def _quota_message(
    status_code: int,
    quota_window: str,
    retry_after_seconds: int | None,
    provider_error: dict[str, Any] | None,
    provider_body: str | None,
) -> str:
    reason = None
    if provider_error:
        reason = provider_error.get("reason") or provider_error.get("message") or provider_error
    if reason is None:
        reason = provider_body or "no provider error body"
    retry = "none" if retry_after_seconds is None else f"{retry_after_seconds}s"
    return (
        f"Open-Meteo returned HTTP {status_code} quota_window={quota_window} "
        f"Retry-After={retry}: {reason}"
    )


def _retry_after_seconds(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    value = value.strip()
    try:
        return max(0, int(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0, int((retry_at - datetime.now(UTC)).total_seconds()))


def _nodes_from_boundaries(boundaries: gpd.GeoDataFrame) -> list[tuple[float, float]]:
    weights = climate_weights_for_districts(boundaries, resolution=0.25)
    nodes = weights[["latitude", "longitude"]].dropna().drop_duplicates()
    return sorted((float(row.latitude), float(row.longitude)) for row in nodes.itertuples())


def download_weather_sources(
    boundaries: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    *,
    raw_dir: Path,
    offline: bool = False,
    force: bool = False,
    chirps_opener=None,
    era5_fetcher=None,
    era5_grid_nodes: list[tuple[float, float]] | None = None,
    era5_batch_size: int = 8,
) -> dict[str, int]:
    raw_dir = Path(raw_dir)
    days = _date_range(start_date, end_date)
    bounds = tuple(boundaries.total_bounds)
    status = {"chirps_downloaded": 0, "chirps_cached": 0, "era5_downloaded": 0, "era5_cached": 0}

    opener = chirps_opener or _default_chirps_open
    for day in days:
        url = chirps_cog_url(day)
        if not force and _find_chirps_subset(raw_dir, day, bounds=bounds) is not None:
            status["chirps_cached"] += 1
            continue
        if offline:
            raise ContractError(f"Offline CHIRPS cache miss for {day}")
        print(f"weather acquisition: CHIRPS {day} regional subset")
        with opener(url) as dataset:
            cache_chirps_subset(
                dataset,
                bounds=bounds,
                day=day,
                raw_dir=raw_dir,
                upstream_url=url,
                force=force,
            )
        status["chirps_downloaded"] += 1

    nodes = era5_grid_nodes or _nodes_from_boundaries(boundaries)
    for batch_index, offset in enumerate(range(0, len(nodes), era5_batch_size)):
        batch = nodes[offset : offset + era5_batch_size]
        params = _era5_request_params(batch, start_date, end_date)
        cached_era5 = _find_era5_response(
            raw_dir,
            start_date,
            end_date,
            batch_index,
            expected_params=params,
        )
        if not force and cached_era5 is not None:
            status["era5_cached"] += 1
            continue
        if offline:
            raise ContractError(f"Offline ERA5 cache miss for batch {batch_index}")
        print(f"weather acquisition: ERA5 batch {batch_index + 1} nodes={len(batch)}")
        if era5_fetcher is not None:
            payload = era5_fetcher(batch, start_date, end_date)
            content = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        else:
            content = _fetch_era5_with_retry(params)
        cache_era5_response(
            content,
            raw_dir=raw_dir,
            request_params=params,
            start_date=start_date,
            end_date=end_date,
            batch_index=batch_index,
            force=force,
        )
        status["era5_downloaded"] += 1
        time.sleep(0.2)
    return status


def _load_rainfall_daily(
    boundaries: gpd.GeoDataFrame, start_date: date, end_date: date, raw_dir: Path
) -> pd.DataFrame:
    rows = []
    weights = None
    expected_fingerprint = None
    bounds = tuple(boundaries.total_bounds)
    for day in _date_range(start_date, end_date):
        cached = _find_chirps_subset(raw_dir, day, bounds=bounds)
        if cached is None:
            raise ContractError(f"Missing CHIRPS raw subset for {day}")
        with rasterio.open(cached.path) as dataset:
            fingerprint = _raster_fingerprint(dataset)
            if expected_fingerprint is None:
                expected_fingerprint = fingerprint
            elif fingerprint != expected_fingerprint:
                raise ContractError(f"CHIRPS grid fingerprint mismatch for {day}")
            if weights is None:
                cells = raster_cell_geometries(dataset)
                weights = build_area_weights(boundaries, cells)
            values = raster_values(dataset)
        rows.append(aggregate_rainfall_values(values, weights, day))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_climate_daily(
    boundaries: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    era5_grid_nodes: list[tuple[float, float]] | None,
    era5_batch_size: int,
) -> pd.DataFrame:
    nodes = era5_grid_nodes or _nodes_from_boundaries(boundaries)
    provider = ERA5Provider()
    grid_rows = []
    for batch_index, offset in enumerate(range(0, len(nodes), era5_batch_size)):
        batch = nodes[offset : offset + era5_batch_size]
        params = _era5_request_params(batch, start_date, end_date)
        cached = _find_era5_response(
            raw_dir,
            start_date,
            end_date,
            batch_index,
            expected_params=params,
        )
        if cached is None:
            raise ContractError(f"Missing ERA5 raw response for batch {batch_index}")
        payload = normalize_era5_payload(cached.path.read_bytes())
        metadata = _read_json(cached.metadata_path)
        source_start = date.fromisoformat(str(metadata["start_date"]))
        source_end = date.fromisoformat(str(metadata["end_date"]))
        validate_era5_payload_hour_set(payload, batch, source_start, source_end)
        if len(payload) != len(batch):
            raise ContractError(f"ERA5 cached batch {batch_index} response count mismatch")
        for node, item in zip(batch, payload, strict=True):
            returned = (round(float(item["latitude"]), 6), round(float(item["longitude"]), 6))
            expected = (round(node[0], 6), round(node[1], 6))
            if returned != expected:
                raise ContractError(f"ERA5 cached response snapped {expected} to {returned}")
            grid_rows.extend(provider._daily_rows(item, node))
    grid_daily = pd.DataFrame(grid_rows)
    if grid_daily.empty:
        return pd.DataFrame()
    grid_daily["date"] = pd.to_datetime(grid_daily["date"]).dt.date
    grid_daily = grid_daily[(grid_daily["date"] >= start_date) & (grid_daily["date"] <= end_date)]
    if grid_daily.empty:
        return pd.DataFrame()
    weights = climate_weights_for_districts(boundaries, resolution=0.25)
    return aggregate_daily_climate(grid_daily, weights)


def build_weather_daily_from_raw(
    boundaries: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    *,
    raw_dir: Path,
    interim_dir: Path,
    offline: bool = True,
    force: bool = False,
    era5_grid_nodes: list[tuple[float, float]] | None = None,
    era5_batch_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not offline and force:
        pass
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    rainfall = _load_rainfall_daily(boundaries, start_date, end_date, raw_dir)
    climate = _load_climate_daily(
        boundaries, start_date, end_date, raw_dir, era5_grid_nodes, era5_batch_size
    )
    rainfall.to_parquet(interim_dir / "rainfall_daily.parquet", index=False)
    climate.to_parquet(interim_dir / "climate_daily.parquet", index=False)
    return rainfall, climate
