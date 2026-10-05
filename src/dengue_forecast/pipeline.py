from __future__ import annotations

# ruff: noqa: E501, I001

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dengue_forecast.config import DISTRICTS, REPO_ROOT
from dengue_forecast.contracts import (
    ContractError,
    validate_dengue_weekly,
    validate_district_reference,
)
from dengue_forecast.dengue.build import build_dengue_dataset
from dengue_forecast.dengue.discover import discover_dengue_reports
from dengue_forecast.dengue.download import download_dengue_report
from dengue_forecast.dengue.models import PARSER_VERSION, ReportDescriptor, SourceDocument
from dengue_forecast.features import build_ml_dataset
from dengue_forecast.geography.boundaries import load_district_boundaries, write_boundary_artifacts
from dengue_forecast.geography.population import load_census_population
from dengue_forecast.geography.reference import build_district_reference
from dengue_forecast.sources.cache import SourceCache, SourceCacheError
from dengue_forecast.sources.config import source_url
from dengue_forecast.weather.weekly import combine_weekly_weather


@dataclass(frozen=True)
class PipelinePaths:
    root: Path = REPO_ROOT
    data_root: Path = REPO_ROOT / "data"

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed(self) -> Path:
        return self.data_root / "processed"

    @property
    def reports(self) -> Path:
        return self.data_root / "reports"


@dataclass(frozen=True)
class Milestone1Config:
    start_year: int = 2024
    end_year: int = 2024
    weather_start_date: date = date(2024, 1, 1)
    weather_end_date: date = date(2025, 1, 3)


def load_milestone1_config(path: Path | None = None) -> Milestone1Config:
    config_path = path or (REPO_ROOT / "configs" / "pipeline.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    section = raw.get("milestone1", {}) if isinstance(raw, dict) else {}
    start_year = int(section.get("start_year", 2024))
    end_year = int(section.get("end_year", start_year))
    weather_start = date.fromisoformat(str(section.get("weather_start_date", "2024-01-01")))
    weather_end = date.fromisoformat(str(section.get("weather_end_date", "2025-01-03")))
    return Milestone1Config(
        start_year=start_year,
        end_year=end_year,
        weather_start_date=weather_start,
        weather_end_date=weather_end,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _read_cache_metadata(raw_dir: Path) -> list[dict[str, Any]]:
    records = []
    for metadata_path in sorted(raw_dir.glob("**/*.metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        file_path = metadata_path.with_name(str(metadata["filename"]))
        if not file_path.exists():
            raise SourceCacheError(f"Missing cached raw file for metadata {metadata_path}")
        actual = _sha256(file_path)
        if actual != metadata.get("sha256"):
            raise SourceCacheError(f"Hash mismatch for cached raw file {file_path}")
        metadata["_metadata_path"] = str(metadata_path)
        metadata["_path"] = str(file_path)
        records.append(metadata)
    return records


def _documents_from_cache(raw_dir: Path, *, source_name: str = "epid") -> list[SourceDocument]:
    docs = []
    for metadata in _read_cache_metadata(raw_dir):
        if metadata.get("source_name") != source_name:
            continue
        path = Path(metadata["_path"])
        docs.append(
            SourceDocument(
                path=path,
                source_name=str(metadata["source_name"]),
                source_url=str(metadata["source_url"]),
                retrieved_at=str(metadata["retrieved_at"]),
                sha256=str(metadata["sha256"]),
                media_type=str(metadata.get("media_type", "application/pdf")),
                descriptor={"period": metadata.get("period", {})},
            )
        )
    return docs


def _filter_actual_observation_years(
    df: pd.DataFrame, *, start_year: int, end_year: int
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    starts = pd.to_datetime(out["week_start_date"]).dt.date
    mask = [(start_year <= value.year <= end_year) for value in starts]
    return out.loc[mask].sort_values(["week_start_date", "district_id"]).reset_index(drop=True)


def _drop_off_anchor_weeks(df: pd.DataFrame, *, paths: PipelinePaths) -> pd.DataFrame:
    if df.empty:
        return df
    starts = pd.to_datetime(df["week_start_date"])
    weekday_counts = starts.dt.weekday.value_counts()
    if weekday_counts.empty:
        return df
    dominant_weekday = int(weekday_counts.idxmax())
    off_grid = df.loc[starts.dt.weekday.ne(dominant_weekday)].copy()
    if off_grid.empty:
        return df

    quarantine_dir = paths.interim / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    records = (
        off_grid[
            ["source_document", "source_url", "source_name", "week_start_date", "week_end_date"]
        ]
        .drop_duplicates()
        .assign(reason="off_anchor_week_start_not_matching_dominant_wer_calendar")
        .sort_values(["week_start_date", "source_document"])
    )
    quarantine_path = quarantine_dir / "dengue_off_anchor_periods.jsonl"
    quarantine_path.write_text(
        "\n".join(
            json.dumps(record, default=str, sort_keys=True) for record in records.to_dict("records")
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = paths.reports / "dengue_quarantine.csv"
    if report_path.exists():
        existing = pd.read_csv(report_path)
    else:
        existing = pd.DataFrame(columns=["source_document", "source_url", "source_name", "reason"])
    appended = records[["source_document", "source_url", "source_name", "reason"]]
    pd.concat([existing, appended], ignore_index=True).drop_duplicates().to_csv(
        report_path, index=False
    )
    return df.loc[starts.dt.weekday.eq(dominant_weekday)].reset_index(drop=True)


def _canonical_week_frame(dengue: pd.DataFrame) -> pd.DataFrame:
    if dengue.empty:
        raise ContractError("Cannot build weather weeks from empty dengue dataset")
    starts = pd.to_datetime(dengue["week_start_date"]).dt.date
    min_start = starts.min()
    max_start = starts.max()
    week_starts = pd.date_range(min_start, max_start, freq="7D")
    rows: list[dict[str, object]] = []
    for start_ts in week_starts:
        start = start_ts.date()
        for district in DISTRICTS:
            rows.append(
                {
                    "district_id": district.district_id,
                    "district_name": district.district_name,
                    "week_start_date": start,
                    "week_end_date": start + timedelta(days=6),
                }
            )
    return pd.DataFrame(rows)


def discover_dengue(
    *,
    source: str,
    start_year: int,
    end_year: int | None,
    offline: bool,
    paths: PipelinePaths,
    force: bool = False,
) -> int:
    source = source.casefold()
    if source not in {"epid", "ndcu"}:
        raise ValueError(f"Unsupported dengue source: {source}")
    index_config_key = "ndcu_index" if source == "ndcu" else "wer_index"
    archive_url = source_url(index_config_key)
    archive_name = "ndcu_archive_index" if source == "ndcu" else "epid_archive_index"
    try:
        cached = SourceCache(paths.raw / "dengue" / "indexes").download(
            archive_url,
            source_name=archive_name,
            source_type="archive_html",
            period={},
            parser_version=PARSER_VERSION,
            offline=offline,
            force=force and not offline,
        )
    except SourceCacheError as exc:
        raise ContractError(str(exc)) from exc
    output_dir = paths.interim / "dengue"
    source_output_path = output_dir / f"{source}_source_index.parquet"
    descriptors = discover_dengue_reports(
        source,
        start_year,
        end_year,
        archive_url=archive_url,
        html_path=cached.path,
        output_path=source_output_path,
        offline=True,
    )
    if source == "epid":
        report_index_path = output_dir / "report_index.parquet"
        report_index_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([d.__dict__ for d in descriptors]).to_parquet(report_index_path, index=False)
    return len(descriptors)


def download_dengue(*, offline: bool, paths: PipelinePaths) -> int:
    index_path = paths.interim / "dengue" / "report_index.parquet"
    if not index_path.exists():
        raise ContractError(f"Missing dengue report index: {index_path}")
    records = pd.read_parquet(index_path).to_dict("records")
    from dengue_forecast.dengue.models import ReportDescriptor

    count = 0
    for record in records:
        descriptor = ReportDescriptor(
            **{k: v for k, v in record.items() if k in ReportDescriptor.__dataclass_fields__}
        )
        download_dengue_report(descriptor, raw_dir=paths.raw / "dengue", offline=offline)
        count += 1
    return count


def _descriptor_from_record(record: dict[str, Any]) -> ReportDescriptor:
    values = {k: v for k, v in record.items() if k in ReportDescriptor.__dataclass_fields__}
    for key in ("issue_year", "issue_week", "observation_week", "volume"):
        if pd.isna(values.get(key)):
            values[key] = None
        elif values.get(key) is not None:
            values[key] = int(values[key])
    return ReportDescriptor(**values)


def _load_report_descriptors(path: Path) -> list[ReportDescriptor]:
    if not path.exists():
        return []
    return [_descriptor_from_record(record) for record in pd.read_parquet(path).to_dict("records")]


def _wanted_issue_window(descriptor: ReportDescriptor, *, start_year: int, end_year: int) -> bool:
    if descriptor.source != "epid" or descriptor.issue_year is None:
        return False
    if start_year <= descriptor.issue_year <= end_year:
        return True
    return (
        descriptor.issue_year == end_year + 1
        and descriptor.issue_week is not None
        and 1 <= descriptor.issue_week <= 2
    )


def _select_requested_epid_descriptors(
    descriptors: list[ReportDescriptor], *, start_year: int, end_year: int
) -> list[ReportDescriptor]:
    selected = [
        descriptor
        for descriptor in descriptors
        if _wanted_issue_window(descriptor, start_year=start_year, end_year=end_year)
    ]
    return sorted(selected, key=lambda d: (d.issue_year or 0, d.issue_week or 0, d.url))


def _load_or_discover_epid_descriptors(
    *, start_year: int, end_year: int, offline: bool, paths: PipelinePaths, force: bool
) -> list[ReportDescriptor]:
    index_path = paths.interim / "dengue" / "epid_source_index.parquet"
    descriptors = _load_report_descriptors(index_path)
    if force and not offline:
        descriptors = []
    if not descriptors:
        if offline:
            fallback_path = paths.interim / "dengue" / "report_index.parquet"
            descriptors = _load_report_descriptors(fallback_path)
        else:
            discover_dengue(
                source="epid",
                start_year=start_year,
                end_year=end_year + 1,
                offline=False,
                paths=paths,
                force=force,
            )
            descriptors = _load_report_descriptors(index_path)
    selected = _select_requested_epid_descriptors(
        descriptors, start_year=start_year, end_year=end_year
    )
    if not selected:
        raise ContractError(f"No WER descriptors found for issue window {start_year}-{end_year}")
    return selected


def _filter_documents_for_issue_window(
    docs: list[SourceDocument], *, start_year: int, end_year: int
) -> list[SourceDocument]:
    selected = []
    for doc in docs:
        period = doc.descriptor.get("period", {})
        issue_year = period.get("issue_year")
        issue_week = period.get("issue_week")
        try:
            descriptor = ReportDescriptor(
                source=doc.source_name,
                title=doc.path.name,
                url=doc.source_url,
                document_type="pdf",
                discovered_at=doc.retrieved_at,
                issue_year=int(issue_year) if issue_year is not None else None,
                issue_week=int(issue_week) if issue_week is not None else None,
            )
        except (TypeError, ValueError):
            continue
        if _wanted_issue_window(descriptor, start_year=start_year, end_year=end_year):
            selected.append(doc)
    return selected


def build_dengue(
    *, start_year: int, end_year: int, offline: bool, paths: PipelinePaths, force: bool = False
) -> pd.DataFrame:
    raw_dir = paths.raw / "dengue"
    docs = _filter_documents_for_issue_window(
        _documents_from_cache(raw_dir), start_year=start_year, end_year=end_year
    )
    if not offline:
        descriptors = _load_or_discover_epid_descriptors(
            start_year=start_year,
            end_year=end_year,
            offline=False,
            paths=paths,
            force=force,
        )
        df = build_dengue_dataset(
            descriptors=descriptors,
            raw_dir=raw_dir,
            processed_path=paths.processed / "dengue_cases_weekly.parquet",
            reports_dir=paths.reports,
            offline=False,
        )
    else:
        if not docs:
            raise ContractError("No cached dengue PDFs available for offline build")
        df = build_dengue_dataset(
            documents=docs,
            raw_dir=raw_dir,
            processed_path=paths.processed / "dengue_cases_weekly.parquet",
            reports_dir=paths.reports,
            offline=offline,
        )
    filtered = _filter_actual_observation_years(df, start_year=start_year, end_year=end_year)
    if filtered.empty:
        processed = paths.processed / "dengue_cases_weekly.parquet"
        if processed.exists():
            processed.unlink()
        raise ContractError(
            f"No parsed dengue rows in actual observation years {start_year}-{end_year}"
        )
    filtered = _drop_off_anchor_weeks(filtered, paths=paths)
    if filtered.empty:
        raise ContractError("No parsed dengue rows remain after off-anchor quarantine")
    validate_dengue_weekly(filtered)
    paths.processed.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(paths.processed / "dengue_cases_weekly.parquet", index=False)
    return filtered


def _cache_source_if_needed(
    *,
    cache: SourceCache,
    source_url: str,
    source_name: str,
    source_type: str,
    offline: bool,
    force: bool = False,
) -> Path:
    cached = cache.find_by_url(source_url)
    if cached is not None and (offline or not force):
        return cached.path
    if offline:
        raise ContractError(f"Offline cache miss for required raw source {source_url}")
    return cache.download(
        source_url,
        source_name=source_name,
        source_type=source_type,
        period={},
        parser_version="milestone1-pipeline",
        force=force,
    ).path


def build_geography(*, offline: bool, paths: PipelinePaths, force: bool = False) -> pd.DataFrame:
    boundary_path = _cache_source_if_needed(
        cache=SourceCache(paths.raw / "geography"),
        source_url=source_url("boundaries_geojson"),
        source_name="geoboundaries_lka_adm2",
        source_type="district_boundaries_geojson",
        offline=offline,
        force=force,
    )
    population_path = _cache_source_if_needed(
        cache=SourceCache(paths.raw / "population"),
        source_url=source_url("population_workbook"),
        source_name="dcs_cph2024_population_tables",
        source_type="population_workbook_xlsx",
        offline=offline,
        force=force,
    )
    boundaries = load_district_boundaries(boundary_path)
    population = load_census_population(population_path)
    reference = build_district_reference(boundaries, population)
    validate_district_reference(reference)
    write_boundary_artifacts(
        reference,
        paths.processed / "sri_lanka_districts.geojson",
        paths.processed / "district_reference.parquet",
    )
    return pd.DataFrame(reference.drop(columns=["geometry"]))


def _load_boundaries_for_weather(paths: PipelinePaths, *, offline: bool) -> Any:
    import geopandas as gpd

    geojson_path = paths.processed / "sri_lanka_districts.geojson"
    if not geojson_path.exists():
        build_geography(offline=offline, paths=paths)
    return gpd.read_file(geojson_path)


def download_weather(
    *,
    start_date: date,
    end_date: date,
    offline: bool,
    force: bool,
    paths: PipelinePaths,
) -> dict[str, int]:
    from dengue_forecast.weather.acquisition import download_weather_sources

    boundaries = _load_boundaries_for_weather(paths, offline=offline)
    return download_weather_sources(
        boundaries,
        start_date,
        end_date,
        raw_dir=paths.raw / "weather",
        offline=offline,
        force=force,
    )


def build_weather_daily(
    *,
    start_date: date,
    end_date: date,
    offline: bool,
    force: bool,
    paths: PipelinePaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from dengue_forecast.weather.acquisition import build_weather_daily_from_raw

    boundaries = _load_boundaries_for_weather(paths, offline=offline)
    return build_weather_daily_from_raw(
        boundaries,
        start_date,
        end_date,
        raw_dir=paths.raw / "weather",
        interim_dir=paths.interim / "weather",
        offline=offline,
        force=force,
    )


def build_weather(
    *,
    paths: PipelinePaths,
    start_date: date | None = None,
    end_date: date | None = None,
    offline: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    dengue_path = paths.processed / "dengue_cases_weekly.parquet"
    if not dengue_path.exists():
        raise ContractError(f"Missing dengue artifact: {dengue_path}")
    dengue = pd.read_parquet(dengue_path)
    weeks = _canonical_week_frame(dengue)

    config = load_milestone1_config()
    daily_start = start_date or config.weather_start_date
    daily_end = end_date or config.weather_end_date
    rainfall, climate = build_weather_daily(
        start_date=daily_start,
        end_date=daily_end,
        offline=offline,
        force=force,
        paths=paths,
    )
    weather = combine_weekly_weather(weeks, rainfall_daily=rainfall, climate_daily=climate)
    weather.to_parquet(paths.processed / "district_weather_weekly.parquet", index=False)
    return weather


def build_features(*, paths: PipelinePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    dengue = pd.read_parquet(paths.processed / "dengue_cases_weekly.parquet")
    weather = pd.read_parquet(paths.processed / "district_weather_weekly.parquet")
    reference = pd.read_parquet(paths.processed / "district_reference.parquet")
    dataset, registry = build_ml_dataset(dengue, weather, reference)
    dataset = dataset.sort_values(["week_start_date", "district_id"]).reset_index(drop=True)
    registry = registry.sort_values("feature_name").reset_index(drop=True)
    dataset.to_parquet(paths.processed / "ml_training_dataset.parquet", index=False)
    registry.to_csv(paths.reports / "feature_registry.csv", index=False)
    from dengue_forecast.contracts import get_training_columns

    training_columns = get_training_columns(registry)
    _write_json(paths.reports / "training_feature_columns.json", training_columns)
    return dataset, registry


def build_reports(*, paths: PipelinePaths) -> None:
    from dengue_forecast.reports.quality import build_quality_reports

    build_quality_reports(paths)


def validate_all(*, paths: PipelinePaths) -> dict[str, Any]:
    from dengue_forecast.reports.quality import validate_artifacts

    return validate_artifacts(paths)


def run_all(
    *,
    start_year: int,
    end_year: int,
    offline: bool,
    paths: PipelinePaths,
    force: bool = False,
    weather_start_date: date | None = None,
    weather_end_date: date | None = None,
) -> dict[str, Any]:
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    build_dengue(
        start_year=start_year,
        end_year=end_year,
        offline=offline,
        paths=paths,
        force=force,
    )
    build_geography(offline=offline, paths=paths, force=force)
    if not offline:
        config = load_milestone1_config()
        download_weather(
            start_date=weather_start_date or config.weather_start_date,
            end_date=weather_end_date or config.weather_end_date,
            offline=False,
            force=force,
            paths=paths,
        )
    build_weather(
        paths=paths,
        start_date=weather_start_date,
        end_date=weather_end_date,
        offline=offline,
        force=force,
    )
    try:
        build_features(paths=paths)
    except TypeError as exc:
        raise ContractError(
            f"Feature build API mismatch, likely pending parent foundation fix: {exc}"
        ) from exc
    build_reports(paths=paths)
    return validate_all(paths=paths)


def rebuild_check(
    *, source_data_root: Path, output_root: Path, force: bool = False
) -> dict[str, Any]:
    marker = output_root / ".dengue_forecast_rebuild"
    if output_root.exists():
        if not marker.exists():
            raise ContractError(f"Refusing to remove unowned rebuild output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    marker.write_text("owned by dengue_forecast rebuild_check\n", encoding="utf-8")
    (output_root / "data" / "raw").mkdir(parents=True)
    shutil.copytree(source_data_root / "raw", output_root / "data" / "raw", dirs_exist_ok=True)
    paths = PipelinePaths(root=REPO_ROOT, data_root=output_root / "data")
    config = load_milestone1_config()
    result = run_all(
        start_year=config.start_year,
        end_year=config.end_year,
        offline=True,
        paths=paths,
        force=force,
        weather_start_date=config.weather_start_date,
        weather_end_date=config.weather_end_date,
    )
    checksums = {}
    for artifact in sorted((output_root / "data" / "processed").glob("*")):
        if artifact.is_file():
            checksums[artifact.name] = _sha256(artifact)
    result["processed_checksums"] = checksums
    return result


def weeks_inclusive(start: date, end: date) -> list[date]:
    starts = []
    current = start
    while current <= end:
        starts.append(current)
        current += timedelta(days=7)
    return starts
