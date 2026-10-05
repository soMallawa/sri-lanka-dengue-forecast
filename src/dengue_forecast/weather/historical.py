from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
import requests

from dengue_forecast.contracts import ContractError
from dengue_forecast.weather.acquisition import (
    WeatherQuotaBlocked,
    _date_range,
    _default_chirps_open,
    _era5_request_params,
    _fetch_era5_with_retry,
    _find_chirps_subset,
    _find_era5_response,
    _load_rainfall_daily,
    _nodes_from_boundaries,
    _read_json,
    _retrieval_sort_key,
    _sha256,
    _verified_cached_source,
    _write_json,
    cache_chirps_subset,
    cache_era5_response,
)
from dengue_forecast.weather.chirps import chirps_cog_url
from dengue_forecast.weather.climate import (
    ERA5Provider,
    aggregate_daily_climate,
    climate_weights_for_districts,
    normalize_era5_payload,
    validate_era5_payload_hour_set,
)

DEFAULT_HISTORICAL_START = date(2010, 1, 1)
DEFAULT_HISTORICAL_END = date(2026, 1, 2)
DEFAULT_RAW_DIR = Path("data/raw/weather")
DEFAULT_INTERIM_DIR = Path("data/historical/interim/weather")
DEFAULT_ERA5_BATCH_SIZE = 8
DEFAULT_CHIRPS_WORKERS = 4
DEFAULT_MAX_429_SLEEP_SECONDS = 900
DEFAULT_ERA5_REQUEST_SPACING_SECONDS = 5
WeatherProvider = str


@dataclass(frozen=True)
class WeatherYearChunk:
    year: int
    start_date: date
    end_date: date


def annual_weather_chunks(start_date: date, end_date: date) -> list[WeatherYearChunk]:
    if end_date < start_date:
        raise ContractError("Historical weather end date must be on or after start date")
    chunks: list[WeatherYearChunk] = []
    year = start_date.year
    while year <= end_date.year:
        chunk_start = max(start_date, date(year, 1, 1))
        if year == 2025 and end_date >= date(2026, 1, 2):
            chunk_end = date(2026, 1, 2)
        else:
            chunk_end = min(end_date, date(year, 12, 31))
        if chunk_start <= chunk_end:
            chunks.append(WeatherYearChunk(year, chunk_start, chunk_end))
        if chunk_end >= end_date:
            break
        year += 1
    return chunks


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _flush_log(message: str) -> None:
    print(message, flush=True)


def _source_record(path: Path, metadata_path: Path) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    return {
        "path": str(path),
        "metadata_path": str(metadata_path),
        "sha256": metadata["sha256"],
        "byte_length": metadata["byte_length"],
        "source": metadata.get("source"),
        "source_version": metadata.get("source_version"),
        "start_date": metadata.get("start_date") or metadata.get("date"),
        "end_date": metadata.get("end_date") or metadata.get("date"),
        "request_params": metadata.get("request_params"),
        "request_bounds": metadata.get("request_bounds"),
    }


def _write_machine_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, data)


def _year_status_path(interim_dir: Path, year: int) -> Path:
    return interim_dir / "manifests" / f"weather_{year}.json"


def _empty_year_status(chunk: WeatherYearChunk) -> dict[str, Any]:
    return {
        "year": chunk.year,
        "start_date": chunk.start_date.isoformat(),
        "end_date": chunk.end_date.isoformat(),
        "status": "pending",
        "created_at": _now(),
        "updated_at": _now(),
        "request_counts": {
            "chirps_days_planned": 0,
            "chirps_provider_calls": 0,
            "era5_batches_planned": 0,
            "era5_provider_calls": 0,
        },
        "request_spacing_seconds": {"era5_uncached": DEFAULT_ERA5_REQUEST_SPACING_SECONDS},
        "reused_files": {"chirps": 0, "era5": 0},
        "retrieved_files": {"chirps": 0, "era5": 0},
        "quality_exceptions": [],
    }


def _append_exception(
    status: dict[str, Any], *, code: str, message: str, stage: str, **details: Any
) -> None:
    issue = {"code": code, "message": message, "stage": stage, "recorded_at": _now()}
    issue.update({key: value for key, value in details.items() if value is not None})
    status["quality_exceptions"].append(issue)


def _quota_exception_details(exc: WeatherQuotaBlocked) -> dict[str, Any]:
    details: dict[str, Any] = {
        "quota_window": exc.quota_window,
        "retry_after_seconds": exc.retry_after_seconds,
        "provider_error": exc.provider_error,
        "provider_body": exc.provider_body,
        "severity": f"provider_quota_{exc.quota_window}",
    }
    return {key: value for key, value in details.items() if value is not None}


def _bounded_chirps_workers(value: int) -> int:
    return min(4, max(1, int(value)))


def _provider_enabled(provider: WeatherProvider, target: WeatherProvider) -> bool:
    return provider == "both" or provider == target


def _same_era5_covering_request_identity(
    actual: dict[str, Any], expected: dict[str, str]
) -> bool:
    actual_identity = {str(key): str(value) for key, value in actual.items()}
    expected_identity = {str(key): str(value) for key, value in expected.items()}
    for key in ("start_date", "end_date"):
        actual_identity.pop(key, None)
        expected_identity.pop(key, None)
    return actual_identity == expected_identity


def _expected_era5_hourly_variables(expected_params: dict[str, str]) -> list[str]:
    return [name for name in expected_params.get("hourly", "").split(",") if name]


def _era5_payload_matches_identity(content: bytes, expected_params: dict[str, str]) -> bool:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ContractError(f"Cached ERA5 payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ContractError("Cached ERA5 payload must be a non-empty list")

    expected_timezone = expected_params.get("timezone")
    expected_hourly = _expected_era5_hourly_variables(expected_params)
    expected_units: dict[str, str] | None = None
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ContractError(f"Cached ERA5 payload item {index} is not an object")
        if expected_timezone is not None and str(item.get("timezone")) != expected_timezone:
            return False
        hourly_units = item.get("hourly_units")
        if not isinstance(hourly_units, dict):
            raise ContractError(f"Cached ERA5 payload item {index} missing hourly_units")
        candidate_units = {name: str(hourly_units.get(name)) for name in expected_hourly}
        if any(unit == "None" for unit in candidate_units.values()):
            return False
        if expected_units is None:
            expected_units = candidate_units
        elif candidate_units != expected_units:
            return False
    return True


def _find_historical_era5_covering_response(
    raw_dir: Path,
    start_date: date,
    end_date: date,
    batch_index: int,
    *,
    expected_params: dict[str, str],
) -> Any | None:
    exact = _find_era5_response(
        raw_dir,
        start_date,
        end_date,
        batch_index,
        expected_params=expected_params,
    )
    if exact is not None and _era5_payload_matches_identity(
        exact.path.read_bytes(), expected_params
    ):
        return exact

    root = Path(raw_dir) / "era5_openmeteo"
    pattern = f"era5_openmeteo_*_batch{batch_index:03d}_*.metadata.json"
    candidates = []
    for metadata_path in sorted(root.glob(pattern)):
        cached, metadata = _verified_cached_source(metadata_path)
        cached_start = date.fromisoformat(str(metadata.get("start_date")))
        cached_end = date.fromisoformat(str(metadata.get("end_date")))
        if cached_start > start_date or cached_end < end_date:
            continue
        if not _same_era5_covering_request_identity(
            metadata.get("request_params", {}), expected_params
        ):
            continue
        if not _era5_payload_matches_identity(cached.path.read_bytes(), expected_params):
            continue
        candidates.append((cached, metadata))
    if not candidates:
        return None
    return max(candidates, key=_retrieval_sort_key)[0]


def _load_historical_climate_daily(
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
        cached = _find_historical_era5_covering_response(
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


def _fetch_chirps_day(
    *,
    day: date,
    bounds: tuple[float, float, float, float],
    raw_dir: Path,
    force: bool,
    opener: Callable[[str], Any],
) -> None:
    url = chirps_cog_url(day)
    with opener(url) as dataset:
        cache_chirps_subset(
            dataset,
            bounds=bounds,
            day=day,
            raw_dir=raw_dir,
            upstream_url=url,
            force=force,
        )


def acquire_weather_chunk(
    boundaries: gpd.GeoDataFrame,
    chunk: WeatherYearChunk,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    offline: bool = False,
    force: bool = False,
    chirps_opener: Callable[[str], Any] | None = None,
    era5_fetcher: Callable[[dict[str, str]], bytes] | None = None,
    era5_grid_nodes: list[tuple[float, float]] | None = None,
    era5_batch_size: int = DEFAULT_ERA5_BATCH_SIZE,
    chirps_workers: int = DEFAULT_CHIRPS_WORKERS,
    max_429_sleep_seconds: int = DEFAULT_MAX_429_SLEEP_SECONDS,
    era5_request_spacing_seconds: int = DEFAULT_ERA5_REQUEST_SPACING_SECONDS,
    provider: WeatherProvider = "both",
    defer_era5: bool = False,
    defer_quota_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    status = _empty_year_status(chunk)
    status_path = _year_status_path(interim_dir, chunk.year)
    status["request_spacing_seconds"]["era5_uncached"] = int(era5_request_spacing_seconds)
    bounds = tuple(float(value) for value in boundaries.total_bounds)
    opener = chirps_opener or _default_chirps_open
    days = _date_range(chunk.start_date, chunk.end_date)
    chirps_workers = _bounded_chirps_workers(chirps_workers)

    _flush_log(
        f"historical weather {chunk.year}: acquire {chunk.start_date}..{chunk.end_date}"
    )
    if _provider_enabled(provider, "rainfall"):
        status["request_counts"]["chirps_days_planned"] = len(days)
        chirps_misses: list[date] = []
        for day in days:
            try:
                cached = None if force else _find_chirps_subset(raw_dir, day, bounds=bounds)
                if cached is not None:
                    status["reused_files"]["chirps"] += 1
                    continue
                if offline:
                    raise ContractError(f"Offline CHIRPS cache miss for {day}")
                chirps_misses.append(day)
            except ContractError as exc:
                _append_exception(status, code="CHIRPS_ERROR", message=str(exc), stage="chirps")
        status["request_counts"]["chirps_provider_calls"] = len(chirps_misses)
        with ThreadPoolExecutor(max_workers=chirps_workers) as executor:
            futures = {
                executor.submit(
                    _fetch_chirps_day,
                    day=day,
                    bounds=bounds,
                    raw_dir=raw_dir,
                    force=force,
                    opener=opener,
                ): day
                for day in chirps_misses
            }
            for future in as_completed(futures):
                day = futures[future]
                try:
                    future.result()
                    status["retrieved_files"]["chirps"] += 1
                except rasterio.errors.RasterioIOError as exc:
                    _append_exception(
                        status,
                        code="NETWORK_UNAVAILABLE",
                        message=f"{day}: {exc}",
                        stage="chirps",
                    )
                except ContractError as exc:
                    _append_exception(
                        status,
                        code="CHIRPS_ERROR",
                        message=f"{day}: {exc}",
                        stage="chirps",
                    )
                except requests.RequestException as exc:
                    _append_exception(
                        status,
                        code="NETWORK_UNAVAILABLE",
                        message=f"{day}: {exc}",
                        stage="chirps",
                    )

    nodes = era5_grid_nodes or _nodes_from_boundaries(boundaries) if _provider_enabled(
        provider, "climate"
    ) else []
    era5_offsets = list(range(0, len(nodes), era5_batch_size))
    if _provider_enabled(provider, "climate"):
        status["request_counts"]["era5_batches_planned"] = len(era5_offsets)
    if defer_era5 and _provider_enabled(provider, "climate"):
        _append_exception(
            status,
            code="QUOTA_DEFERRED",
            message="ERA5 acquisition deferred after earlier Open-Meteo provider quota block",
            stage="era5",
            **(defer_quota_details or {}),
        )
    elif _provider_enabled(provider, "climate"):
        for batch_index, offset in enumerate(era5_offsets):
            batch = nodes[offset : offset + era5_batch_size]
            params = _era5_request_params(batch, chunk.start_date, chunk.end_date)
            try:
                cached_era5 = None if force else _find_historical_era5_covering_response(
                    raw_dir,
                    chunk.start_date,
                    chunk.end_date,
                    batch_index,
                    expected_params=params,
                )
                if cached_era5 is not None:
                    status["reused_files"]["era5"] += 1
                    continue
                if offline:
                    raise ContractError(f"Offline ERA5 cache miss for batch {batch_index}")
                status["request_counts"]["era5_provider_calls"] += 1
                content = (
                    era5_fetcher(params)
                    if era5_fetcher is not None
                    else _fetch_era5_with_retry(
                        params, max_retry_after_seconds=max_429_sleep_seconds
                    )
                )
                cache_era5_response(
                    content,
                    raw_dir=raw_dir,
                    request_params=params,
                    start_date=chunk.start_date,
                    end_date=chunk.end_date,
                    batch_index=batch_index,
                    force=force,
                )
                status["retrieved_files"]["era5"] += 1
            except WeatherQuotaBlocked as exc:
                _append_exception(
                    status,
                    code="QUOTA_BLOCKED",
                    message=str(exc),
                    stage="era5",
                    batch_index=batch_index,
                    **_quota_exception_details(exc),
                )
                status["status"] = "blocked_quota"
                break
            except ContractError as exc:
                _append_exception(status, code="ERA5_ERROR", message=str(exc), stage="era5")
            except requests.RequestException as exc:
                _append_exception(
                    status,
                    code="NETWORK_UNAVAILABLE",
                    message=str(exc),
                    stage="era5",
                )
            else:
                if (
                    era5_fetcher is None
                    and era5_request_spacing_seconds > 0
                    and batch_index < len(era5_offsets) - 1
                ):
                    time.sleep(era5_request_spacing_seconds)

    if status["status"] != "blocked_quota":
        status["status"] = "complete" if not status["quality_exceptions"] else "partial"
        if defer_era5 and _provider_enabled(provider, "climate"):
            status["status"] = "deferred_quota"
    status["updated_at"] = _now()
    _write_machine_json(status_path, status)
    _flush_log(
        "historical weather "
        f"{chunk.year}: {status['status']} reused={status['reused_files']} "
        f"retrieved={status['retrieved_files']}"
    )
    return status


def acquire_historical_weather(
    boundaries: gpd.GeoDataFrame,
    *,
    start_date: date = DEFAULT_HISTORICAL_START,
    end_date: date = DEFAULT_HISTORICAL_END,
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    offline: bool = False,
    force: bool = False,
    chirps_opener: Callable[[str], Any] | None = None,
    era5_fetcher: Callable[[dict[str, str]], bytes] | None = None,
    era5_grid_nodes: list[tuple[float, float]] | None = None,
    era5_batch_size: int = DEFAULT_ERA5_BATCH_SIZE,
    chirps_workers: int = DEFAULT_CHIRPS_WORKERS,
    max_429_sleep_seconds: int = DEFAULT_MAX_429_SLEEP_SECONDS,
    era5_request_spacing_seconds: int = DEFAULT_ERA5_REQUEST_SPACING_SECONDS,
    provider: WeatherProvider = "both",
) -> dict[str, Any]:
    statuses = []
    era5_quota_blocked = False
    era5_quota_details: dict[str, Any] | None = None
    for chunk in annual_weather_chunks(start_date, end_date):
        status = acquire_weather_chunk(
            boundaries,
            chunk,
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            offline=offline,
            force=force,
            chirps_opener=chirps_opener,
            era5_fetcher=era5_fetcher,
            era5_grid_nodes=era5_grid_nodes,
            era5_batch_size=era5_batch_size,
            chirps_workers=chirps_workers,
            max_429_sleep_seconds=max_429_sleep_seconds,
            era5_request_spacing_seconds=era5_request_spacing_seconds,
            provider=provider,
            defer_era5=era5_quota_blocked,
            defer_quota_details=era5_quota_details,
        )
        statuses.append(status)
        if status["status"] == "blocked_quota":
            era5_quota_blocked = True
            quota_issue = next(
                (
                    issue
                    for issue in status["quality_exceptions"]
                    if issue.get("code") == "QUOTA_BLOCKED"
                ),
                {},
            )
            era5_quota_details = {
                key: quota_issue[key]
                for key in (
                    "quota_window",
                    "retry_after_seconds",
                    "provider_error",
                    "provider_body",
                    "severity",
                )
                if key in quota_issue
            }
    summary = _summarize_statuses(start_date, end_date, statuses, provider=provider)
    _write_machine_json(Path(interim_dir) / "historical_weather_acquisition.json", summary)
    return summary


def _sort_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.drop_duplicates(["district_id", "date"], keep="last")
    return out.sort_values(["district_id", "date"]).reset_index(drop=True)


def _write_source_index(
    boundaries: gpd.GeoDataFrame,
    chunks: list[WeatherYearChunk],
    *,
    raw_dir: Path,
    interim_dir: Path,
    era5_grid_nodes: list[tuple[float, float]] | None,
    era5_batch_size: int,
) -> dict[str, Any]:
    bounds = tuple(float(value) for value in boundaries.total_bounds)
    nodes = era5_grid_nodes or _nodes_from_boundaries(boundaries)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for chunk in chunks:
        for day in _date_range(chunk.start_date, chunk.end_date):
            cached = _find_chirps_subset(raw_dir, day, bounds=bounds)
            if cached is None:
                missing.append({"provider": "chirps", "date": day.isoformat(), "year": chunk.year})
            else:
                records.append(
                    {
                        "provider": "chirps",
                        "year": chunk.year,
                        **_source_record(cached.path, cached.metadata_path),
                    }
                )
        for batch_index, offset in enumerate(range(0, len(nodes), era5_batch_size)):
            batch = nodes[offset : offset + era5_batch_size]
            params = _era5_request_params(batch, chunk.start_date, chunk.end_date)
            cached = _find_historical_era5_covering_response(
                raw_dir,
                chunk.start_date,
                chunk.end_date,
                batch_index,
                expected_params=params,
            )
            if cached is None:
                missing.append(
                    {"provider": "era5", "year": chunk.year, "batch_index": batch_index}
                )
            else:
                records.append(
                    {
                        "provider": "era5",
                        "year": chunk.year,
                        "batch_index": batch_index,
                        **_source_record(cached.path, cached.metadata_path),
                    }
                )
    source_index = {
        "generated_at": _now(),
        "source_index_hash": _sha256(
            json.dumps(records, sort_keys=True, default=str).encode("utf-8")
        ),
        "records": records,
        "missing": missing,
    }
    _write_machine_json(Path(interim_dir) / "historical_weather_source_index.json", source_index)
    return source_index


def build_historical_weather_daily(
    boundaries: gpd.GeoDataFrame,
    *,
    start_date: date = DEFAULT_HISTORICAL_START,
    end_date: date = DEFAULT_HISTORICAL_END,
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    era5_grid_nodes: list[tuple[float, float]] | None = None,
    era5_batch_size: int = DEFAULT_ERA5_BATCH_SIZE,
    allow_partial: bool = False,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    chunks = annual_weather_chunks(start_date, end_date)
    year_dir = interim_dir / "by_year"
    rainfall_frames = []
    climate_frames = []
    exceptions: list[dict[str, Any]] = []
    expected_dates = {day for day in _date_range(start_date, end_date)}
    expected_districts = set(boundaries["district_id"].astype(str))

    for chunk in chunks:
        _flush_log(f"historical weather {chunk.year}: build from raw")
        try:
            rainfall = _load_rainfall_daily(boundaries, chunk.start_date, chunk.end_date, raw_dir)
            climate = _load_historical_climate_daily(
                boundaries,
                chunk.start_date,
                chunk.end_date,
                raw_dir,
                era5_grid_nodes,
                era5_batch_size,
            )
        except ContractError as exc:
            issue = {
                "year": chunk.year,
                "status": "missing_year",
                "message": str(exc),
                "recorded_at": _now(),
            }
            exceptions.append(issue)
            if not allow_partial:
                _write_machine_json(
                    interim_dir / "historical_weather_quality_exceptions.json",
                    {"status": "failed", "exceptions": exceptions},
                )
                raise
            continue

        chunk_dir = year_dir / f"{chunk.year:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        rainfall = _sort_daily(rainfall)
        climate = _sort_daily(climate)
        rainfall.to_parquet(chunk_dir / "rainfall_daily.parquet", index=False)
        climate.to_parquet(chunk_dir / "climate_daily.parquet", index=False)
        rainfall_frames.append(rainfall)
        climate_frames.append(climate)

    if not rainfall_frames or not climate_frames:
        raise ContractError("No historical weather daily frames were built")

    rainfall_all = _sort_daily(pd.concat(rainfall_frames, ignore_index=True))
    climate_all = _sort_daily(pd.concat(climate_frames, ignore_index=True))
    _check_complete_daily_grid(
        rainfall_all, expected_dates, expected_districts, "CHIRPS rainfall", exceptions
    )
    _check_complete_daily_grid(
        climate_all, expected_dates, expected_districts, "ERA5 climate", exceptions
    )
    if exceptions and not allow_partial:
        _write_machine_json(
            interim_dir / "historical_weather_quality_exceptions.json",
            {"status": "failed", "exceptions": exceptions},
        )
        raise ContractError("Historical weather build incomplete; see quality exceptions JSON")

    combined = rainfall_all.merge(
        climate_all,
        on=["district_id", "district_name", "date"],
        how="outer",
        suffixes=("_rain", "_climate"),
        validate="one_to_one",
    )
    combined = combined.sort_values(["district_id", "date"]).reset_index(drop=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    rainfall_all.to_parquet(interim_dir / "rainfall_daily.parquet", index=False)
    climate_all.to_parquet(interim_dir / "climate_daily.parquet", index=False)
    combined.to_parquet(interim_dir / "weather_daily.parquet", index=False)
    source_index = _write_source_index(
        boundaries,
        chunks,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        era5_grid_nodes=era5_grid_nodes,
        era5_batch_size=era5_batch_size,
    )
    quality = {
        "status": "partial" if exceptions else "complete",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "rainfall_rows": int(len(rainfall_all)),
        "climate_rows": int(len(climate_all)),
        "combined_rows": int(len(combined)),
        "source_index_hash": source_index["source_index_hash"],
        "exceptions": exceptions,
    }
    _write_machine_json(interim_dir / "historical_weather_quality_exceptions.json", quality)
    return quality


def _check_complete_daily_grid(
    frame: pd.DataFrame,
    expected_dates: set[date],
    expected_districts: set[str],
    label: str,
    exceptions: list[dict[str, Any]],
) -> None:
    if frame.empty:
        exceptions.append({"provider": label, "status": "missing_all"})
        return
    observed = set(
        zip(
            frame["district_id"].astype(str),
            pd.to_datetime(frame["date"]).dt.date,
            strict=False,
        )
    )
    missing = [
        {"district_id": district_id, "date": day.isoformat()}
        for district_id in sorted(expected_districts)
        for day in sorted(expected_dates)
        if (district_id, day) not in observed
    ]
    if missing:
        exceptions.append(
            {
                "provider": label,
                "status": "missing_days",
                "missing_count": len(missing),
                "sample": missing[:20],
            }
        )


def _summarize_statuses(
    start_date: date, end_date: date, statuses: list[dict[str, Any]], *, provider: WeatherProvider
) -> dict[str, Any]:
    return {
        "generated_at": _now(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "requested_provider": provider,
        "status": "complete"
        if all(item["status"] == "complete" for item in statuses)
        else "partial",
        "years": statuses,
        "totals": {
            "chirps_reused": sum(int(item["reused_files"]["chirps"]) for item in statuses),
            "chirps_retrieved": sum(int(item["retrieved_files"]["chirps"]) for item in statuses),
            "era5_reused": sum(int(item["reused_files"]["era5"]) for item in statuses),
            "era5_retrieved": sum(int(item["retrieved_files"]["era5"]) for item in statuses),
            "quality_exceptions": sum(len(item["quality_exceptions"]) for item in statuses),
        },
    }


def run_historical_weather(
    boundaries: gpd.GeoDataFrame,
    *,
    mode: str,
    start_date: date = DEFAULT_HISTORICAL_START,
    end_date: date = DEFAULT_HISTORICAL_END,
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    offline: bool = False,
    force: bool = False,
    allow_partial: bool = False,
    era5_batch_size: int = DEFAULT_ERA5_BATCH_SIZE,
    chirps_workers: int = DEFAULT_CHIRPS_WORKERS,
    provider: WeatherProvider = "both",
    era5_request_spacing_seconds: int = DEFAULT_ERA5_REQUEST_SPACING_SECONDS,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if mode in {"download", "combined"}:
        result["acquisition"] = acquire_historical_weather(
            boundaries,
            start_date=start_date,
            end_date=end_date,
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            offline=offline,
            force=force,
            era5_batch_size=era5_batch_size,
            chirps_workers=chirps_workers,
            era5_request_spacing_seconds=era5_request_spacing_seconds,
            provider=provider,
        )
    if mode in {"build", "combined"} and provider == "both":
        result["build"] = build_historical_weather_daily(
            boundaries,
            start_date=start_date,
            end_date=end_date,
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            era5_batch_size=era5_batch_size,
            allow_partial=allow_partial,
        )
    elif mode in {"build", "combined"}:
        result["build"] = {
            "status": "skipped",
            "reason": "build requires --provider both cached rainfall and climate inputs",
        }
    return result
