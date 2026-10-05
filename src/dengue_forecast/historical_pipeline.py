from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_forecast.config import DISTRICTS, REPO_ROOT
from dengue_forecast.contracts import ContractError
from dengue_forecast.dengue.historical import _save_index, parse_historical_reports
from dengue_forecast.historical_reporting import write_historical_reporting_artifacts
from dengue_forecast.pipeline import (
    PipelinePaths,
    build_features,
    build_geography,
    build_reports,
    validate_all,
)
from dengue_forecast.weather.historical import build_historical_weather_daily
from dengue_forecast.weather.weekly import combine_weekly_weather

DEFAULT_OUTPUT_DATA_ROOT = Path("data/historical")
DEFAULT_SHARED_DATA_ROOT = Path("data")
DEFAULT_BASELINE_JSON = Path("docs/baselines/phase-a-accepted-dengue.json")
CANONICAL_ARTIFACTS = [
    "dengue_cases_weekly.parquet",
    "district_weather_weekly.parquet",
    "district_reference.parquet",
    "sri_lanka_districts.geojson",
    "ml_training_dataset.parquet",
]


@dataclass(frozen=True)
class HistoricalPipelineResult:
    output_data_root: Path
    summary: dict[str, Any]


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_raw_cache_dir(source: Path, target: Path) -> None:
    if not source.exists():
        raise ContractError(f"Missing retained raw cache directory: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def prepare_raw_caches(*, shared_data_root: Path, output_data_root: Path) -> None:
    """Copy retained raw inputs only, never processed/interim artifacts."""

    for name in ["dengue", "weather", "geography", "population"]:
        source = shared_data_root / "raw" / name
        target = output_data_root / "raw" / name
        if source.resolve() == target.resolve():
            continue
        _copy_raw_cache_dir(source, target)


def _rebase_historical_dengue_index(raw_dengue_dir: Path) -> None:
    index_path = raw_dengue_dir / "historical_report_index.parquet"
    if not index_path.exists():
        raise ContractError(f"Missing historical dengue raw index: {index_path}")
    index = pd.read_parquet(index_path)
    if "document_filename" not in index.columns or "raw_path" not in index.columns:
        raise ContractError("Historical dengue index missing document_filename/raw_path columns")
    has_file = index["document_filename"].notna()
    index.loc[has_file, "raw_path"] = index.loc[has_file, "document_filename"].map(
        lambda name: Path(str(name)).name
    )
    _save_index(index, index_path)


def _candidate_paths(output_data_root: Path) -> PipelinePaths:
    return PipelinePaths(root=REPO_ROOT, data_root=output_data_root)


def canonical_week_frame(dengue: pd.DataFrame) -> pd.DataFrame:
    if dengue.empty:
        raise ContractError("Cannot build historical calendar from empty dengue candidate")
    starts = pd.to_datetime(dengue["week_start_date"]).dt.date
    first = starts.min()
    last = starts.max()
    weeks = pd.date_range(first, last, freq="7D")
    rows: list[dict[str, object]] = []
    for start_ts in weeks:
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


def _read_weather_daily(interim_weather_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rainfall_path = interim_weather_dir / "rainfall_daily.parquet"
    climate_path = interim_weather_dir / "climate_daily.parquet"
    missing = [str(path) for path in [rainfall_path, climate_path] if not path.exists()]
    if missing:
        raise ContractError(f"Missing candidate historical weather daily artifact(s): {missing}")
    return pd.read_parquet(rainfall_path), pd.read_parquet(climate_path)


def build_historical_weather_weekly(*, output_data_root: Path) -> pd.DataFrame:
    paths = _candidate_paths(output_data_root)
    dengue_path = paths.processed / "dengue_cases_weekly.parquet"
    if not dengue_path.exists():
        raise ContractError(f"Missing candidate historical dengue artifact: {dengue_path}")
    dengue = pd.read_parquet(dengue_path)
    weeks = canonical_week_frame(dengue)
    rainfall, climate = _read_weather_daily(paths.interim / "weather")
    weather = combine_weekly_weather(weeks, rainfall_daily=rainfall, climate_daily=climate)
    paths.processed.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(paths.processed / "district_weather_weekly.parquet", index=False)
    return weather


def _load_baseline_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ContractError(f"Missing portable Phase A dengue baseline JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if len(rows) != 1250:
        raise ContractError(f"Phase A dengue baseline JSON must contain 1250 rows, got {len(rows)}")
    baseline = pd.DataFrame(rows)
    baseline["week_start_date"] = pd.to_datetime(baseline["week_start_date"]).dt.date
    baseline["week_end_date"] = pd.to_datetime(baseline["week_end_date"]).dt.date
    return baseline


def check_phase_a_dengue_baseline(
    candidate: pd.DataFrame, *, baseline_json: Path
) -> dict[str, Any]:
    baseline = _load_baseline_rows(baseline_json)
    cand = candidate.copy()
    cand["week_start_date"] = pd.to_datetime(cand["week_start_date"]).dt.date
    cand["week_end_date"] = pd.to_datetime(cand["week_end_date"]).dt.date
    cand_2024 = cand[
        pd.Series([value.year == 2024 for value in cand["week_start_date"]], index=cand.index)
    ]
    if cand_2024.empty:
        raise ContractError("Historical candidate is missing all Phase A 2024 baseline rows")

    keys = ["district_id", "week_start_date"]
    baseline_keyed = baseline.set_index(keys).sort_index()
    cand_keyed = cand.set_index(keys).sort_index()
    missing = sorted(set(baseline_keyed.index) - set(cand_keyed.index))
    if missing:
        raise ContractError(f"Historical candidate is missing Phase A baseline rows: {missing[:5]}")

    changed: list[dict[str, Any]] = []
    for key, base_row in baseline_keyed.iterrows():
        cand_row = cand_keyed.loc[key]
        if isinstance(cand_row, pd.DataFrame):
            raise ContractError(f"Duplicate candidate baseline key: {key}")
        for column in ["week_end_date", "dengue_cases", "case_status"]:
            left = base_row[column]
            right = cand_row[column]
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                changed.append(
                    {
                        "district_id": key[0],
                        "week_start_date": key[1].isoformat(),
                        "column": column,
                        "baseline": left,
                        "candidate": right,
                    }
                )
    if changed:
        raise ContractError(f"Historical candidate changed Phase A baseline rows: {changed[:5]}")
    return {
        "status": "passed",
        "baseline_rows": int(len(baseline)),
        "matched_rows": int(len(baseline)),
    }


def _ensure_quality_report_alias(paths: PipelinePaths) -> None:
    target = paths.reports / "dengue_extraction_quality.csv"
    source = paths.reports / "historical_dengue_extraction_quality.csv"
    if not source.exists():
        raise ContractError(
            "Missing historical_dengue_extraction_quality.csv required for candidate validation"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


def _available_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return int(series.fillna(False).sum())
    return int(series.notna().sum())


def _coverage_by_year(dataset: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    years = sorted(pd.to_datetime(dataset["week_start_date"]).dt.year.unique().tolist())
    rows = []
    for year in years:
        frame = dataset[pd.to_datetime(dataset["week_start_date"]).dt.year.eq(year)]
        total = len(frame)
        available = _available_count(frame, column)
        rows.append(
            {
                "year": int(year),
                "rows": int(total),
                "available": available,
                "coverage_pct": round(100.0 * available / total, 2) if total else 0.0,
                "trainable_rows": int(frame.get("is_trainable", pd.Series(dtype=bool)).sum()),
            }
        )
    return rows


def _annual_not_trainable_reason_buckets(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    years = sorted(pd.to_datetime(dataset["week_start_date"]).dt.year.unique().tolist())
    return [
        {
            "year": int(year),
            "untrainable_rows": int(len(frame := dataset[
                pd.to_datetime(dataset["week_start_date"]).dt.year.eq(year)
                & ~dataset["is_trainable"].fillna(False).astype(bool)
            ])),
            "reason_counts": _primary_not_trainable_reason_counts(frame),
        }
        for year in years
    ]


def _coverage_by_district(dataset: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for district_id, frame in dataset.groupby("district_id", sort=True):
        total = len(frame)
        available = _available_count(frame, column)
        rows.append(
            {
                "district_id": district_id,
                "rows": int(total),
                "available": available,
                "coverage_pct": round(100.0 * available / total, 2) if total else 0.0,
            }
        )
    return rows


def _missing_week_stats(dataset: pd.DataFrame) -> dict[str, Any]:
    observed = dataset[dataset["dengue_cases"].notna()].copy()
    all_weeks = sorted(pd.to_datetime(dataset["week_start_date"]).dt.date.unique().tolist())
    observed_counts = observed.groupby("week_start_date")["district_id"].nunique()
    missing_whole = [week for week in all_weeks if int(observed_counts.get(week, 0)) == 0]
    longest = 0
    current = 0
    last: date | None = None
    for week in missing_whole:
        if last is not None and week == last + timedelta(days=7):
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        last = week
    year_counts = Counter(week.year for week in missing_whole)
    return {
        "missing_whole_weeks": len(missing_whole),
        "longest_missing_run_weeks": longest,
        "sample_missing_whole_weeks": [week.isoformat() for week in missing_whole[:20]],
        "years_with_major_gaps": [
            {"year": int(year), "missing_whole_weeks": int(count)}
            for year, count in sorted(year_counts.items())
            if count >= 4
        ],
    }


def _target_available_counts(dataset: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(dataset[column].notna().sum()) if column in dataset else 0
        for column in ["cases_next_week", "cases_next_2w", "cases_next_4w"]
    }


def _low_weather_mask(dataset: pd.DataFrame) -> pd.Series:
    return (
        dataset["weather_temporal_coverage_pct"].fillna(-1).lt(85.0)
        | dataset["weather_spatial_coverage_pct"].fillna(-1).lt(90.0)
        | dataset["weather_quality_flag"].fillna("UNKNOWN").ne("OK")
    )


def _secondary_not_trainable_flags(dataset: pd.DataFrame) -> dict[str, int]:
    if dataset.empty:
        return {
            "missing_current_cases": 0,
            "missing_target_next_week": 0,
            "insufficient_case_history": 0,
            "missing_weather": 0,
            "low_weather_coverage": 0,
            "missing_population": 0,
        }
    target_available = (
        dataset["cases_next_week"].notna()
        if "cases_next_week" in dataset
        else pd.Series(False, index=dataset.index)
    )
    return {
        "missing_current_cases": int(dataset["dengue_cases"].isna().sum()),
        "missing_target_next_week": int((~target_available).sum()),
        "insufficient_case_history": int(dataset["case_history_missing_flag"].fillna(False).sum()),
        "missing_weather": int(dataset["weather_missing_flag"].fillna(False).sum()),
        "low_weather_coverage": int(_low_weather_mask(dataset).sum()),
        "missing_population": int(
            (
                dataset["row_quality_flag"].eq("MISSING_POPULATION")
                | dataset["population_reference"].isna()
            ).sum()
        ),
    }


def _primary_not_trainable_reason(row: pd.Series) -> str:
    if pd.isna(row.get("dengue_cases")):
        return "missing_current_cases"
    if pd.isna(row.get("cases_next_week")):
        return "missing_target_next_week"
    if bool(row.get("case_history_missing_flag", False)):
        return "insufficient_case_history"
    if bool(row.get("weather_missing_flag", False)):
        return "missing_weather"
    weather_quality = row.get("weather_quality_flag")
    temporal = row.get("weather_temporal_coverage_pct")
    spatial = row.get("weather_spatial_coverage_pct")
    low_weather = (
        pd.isna(temporal)
        or pd.isna(spatial)
        or float(temporal) < 85.0
        or float(spatial) < 90.0
        or str(weather_quality) != "OK"
    )
    if low_weather:
        return "low_weather_coverage"
    if str(row.get("row_quality_flag")) == "MISSING_POPULATION" or pd.isna(
        row.get("population_reference")
    ):
        return "missing_population"
    return "other_not_trainable"


def _primary_not_trainable_reason_counts(dataset: pd.DataFrame) -> dict[str, int]:
    buckets = [
        "missing_current_cases",
        "missing_target_next_week",
        "insufficient_case_history",
        "missing_weather",
        "low_weather_coverage",
        "missing_population",
        "other_not_trainable",
    ]
    if dataset.empty:
        return {bucket: 0 for bucket in buckets}
    reasons = dataset.apply(_primary_not_trainable_reason, axis=1)
    counts = Counter(reasons)
    return {bucket: int(counts.get(bucket, 0)) for bucket in buckets}


def _trainability_reasons(dataset: pd.DataFrame) -> dict[str, Any]:
    not_trainable = dataset[~dataset["is_trainable"].fillna(False).astype(bool)].copy()
    primary = _primary_not_trainable_reason_counts(not_trainable)
    return {
        "primary_not_trainable_reason_counts": primary,
        "primary_not_trainable_reason_counts_by_year": (
            _annual_not_trainable_reason_buckets(dataset)
        ),
        "secondary_not_trainable_flags_overlapping": _secondary_not_trainable_flags(
            not_trainable
        ),
    }


def _maybe_count_csv(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path)))


def summarize_historical_pipeline(
    *,
    output_data_root: Path,
    baseline_status: dict[str, Any],
    validation_status: str,
    clean_rebuild_status: str = "NOT RUN",
    hash_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _candidate_paths(output_data_root)
    dengue = pd.read_parquet(paths.processed / "dengue_cases_weekly.parquet")
    reference = pd.read_parquet(paths.processed / "district_reference.parquet")
    dataset = pd.read_parquet(paths.processed / "ml_training_dataset.parquet")
    weather_path = paths.processed / "district_weather_weekly.parquet"
    weather = pd.read_parquet(weather_path) if weather_path.exists() else dataset
    reporting = write_historical_reporting_artifacts(
        reports_dir=paths.reports,
        dengue=dengue,
        weather=weather,
        dataset=dataset,
    )
    starts = pd.to_datetime(dataset["week_start_date"]).dt.date
    observed_starts = pd.to_datetime(
        dataset.loc[dataset["dengue_cases"].notna(), "week_start_date"]
    ).dt.date
    target_counts = _target_available_counts(dataset)
    weather_available = (
        dataset[["rainfall_sum_mm", "temp_mean_c", "humidity_mean_pct"]].notna().all(axis=1)
    )
    years_represented = sorted(
        int(year) for year in pd.to_datetime(dataset["week_start_date"]).dt.year.unique()
    )
    weather_dataset = dataset.assign(_weather_available=weather_available)
    overall_dengue_pct = (
        round(100.0 * dataset["dengue_cases"].notna().sum() / len(dataset), 2)
        if len(dataset)
        else 0.0
    )
    overall_weather_pct = (
        round(100.0 * weather_available.sum() / len(dataset), 2) if len(dataset) else 0.0
    )
    summary = {
        "generated_at": _now(),
        "date_range": {
            "earliest_calendar_week": starts.min().isoformat() if len(starts) else None,
            "latest_calendar_week": starts.max().isoformat() if len(starts) else None,
            "earliest_observed_week": (
                observed_starts.min().isoformat() if len(observed_starts) else None
            ),
            "latest_observed_week": (
                observed_starts.max().isoformat() if len(observed_starts) else None
            ),
            "years_represented": years_represented,
        },
        "districts": int(reference["district_id"].nunique()),
        "rows": {
            "district_week_rows": int(len(dataset)),
            "dengue_rows": int(len(dengue)),
            "observed_dengue_rows": int(dataset["dengue_cases"].notna().sum()),
            "training_eligible_rows": int(dataset["is_trainable"].sum()),
            "target_available_counts": target_counts,
        },
        "coverage": {
            "overall_dengue_pct": overall_dengue_pct,
            "overall_weather_pct": overall_weather_pct,
            "dengue_by_year": _coverage_by_year(dataset, "dengue_cases"),
            "weather_by_year": _coverage_by_year(weather_dataset, "_weather_available"),
            "dengue_by_district": _coverage_by_district(dataset, "dengue_cases"),
            "weather_by_district": _coverage_by_district(
                weather_dataset, "_weather_available"
            ),
        },
        "missing_weeks": _missing_week_stats(dataset),
        "trainability": _trainability_reasons(dataset),
        "quality": {
            "source_conflicts": reporting["source_issues"]["total_documented_source_issues"],
            "source_issue_reporting": reporting["source_issues"],
            "quarantined_reports": _maybe_count_csv(
                paths.reports / "historical_dengue_quarantine.csv"
            ),
            "row_quality_breakdown": dict(Counter(dataset["row_quality_flag"].fillna("UNKNOWN"))),
        },
        "source_acquisition": _source_acquisition_summary(paths),
        "baseline_regression": baseline_status,
        "reproducibility": {
            "tests": "NOT RUN",
            "ruff": "NOT RUN",
            "dataset_validation": validation_status,
            "clean_rebuild": clean_rebuild_status,
            "hash_verification": hash_verification or {},
        },
        "readiness": "NOT READY",
    }
    _write_json(paths.reports / "historical_backfill_summary.json", summary)
    return summary


def _source_acquisition_summary(paths: PipelinePaths) -> dict[str, Any]:
    dengue_summary_path = paths.reports / "historical_dengue_summary.json"
    weather_acq_path = paths.interim / "weather" / "historical_weather_acquisition.json"
    weather_quality_path = paths.interim / "weather" / "historical_weather_quality_exceptions.json"
    dengue_summary = (
        json.loads(dengue_summary_path.read_text(encoding="utf-8"))
        if dengue_summary_path.exists()
        else {}
    )
    weather_acq = (
        json.loads(weather_acq_path.read_text(encoding="utf-8"))
        if weather_acq_path.exists()
        else {}
    )
    weather_quality = (
        json.loads(weather_quality_path.read_text(encoding="utf-8"))
        if weather_quality_path.exists()
        else {}
    )
    return {
        "reports_discovered": int(dengue_summary.get("index_rows", 0)),
        "reports_parsed": int(dengue_summary.get("documents_parsed", 0)),
        "reports_quarantined": int(dengue_summary.get("documents_quarantined", 0)),
        "weather_provider_quota_acquisition": weather_acq,
        "weather_quality_acquisition": weather_quality,
    }


def write_historical_report(*, output_data_root: Path, summary: dict[str, Any]) -> Path:
    paths = _candidate_paths(output_data_root)
    lines = [
        "# Historical Backfill Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Source acquisition",
        "",
        f"- Reports discovered: {summary['source_acquisition']['reports_discovered']}",
        f"- Reports successfully parsed: {summary['source_acquisition']['reports_parsed']}",
        f"- Reports quarantined: {summary['source_acquisition']['reports_quarantined']}",
        f"- Documented source issues: {summary['quality']['source_conflicts']}",
        f"- Source issue reporting: {summary['quality']['source_issue_reporting']}",
        "- Weather provider quota/acquisition stats: see "
        "`interim/weather/historical_weather_acquisition.json` when present.",
        "",
        "## Temporal coverage",
        "",
        f"- Earliest accepted week: {summary['date_range']['earliest_observed_week']}",
        f"- Latest accepted week: {summary['date_range']['latest_observed_week']}",
        f"- Earliest calendar week: {summary['date_range']['earliest_calendar_week']}",
        f"- Latest calendar week: {summary['date_range']['latest_calendar_week']}",
        f"- Years represented: {summary['date_range']['years_represented']}",
        f"- Missing whole weeks: {summary['missing_weeks']['missing_whole_weeks']}",
        f"- Longest missing run: {summary['missing_weeks']['longest_missing_run_weeks']} week(s)",
        f"- Years with major gaps: {summary['missing_weeks']['years_with_major_gaps']}",
        "",
        "## Dataset size",
        "",
        f"- District-week rows: {summary['rows']['district_week_rows']}",
        f"- Observed dengue rows: {summary['rows']['observed_dengue_rows']}",
        f"- Training-eligible rows: {summary['rows']['training_eligible_rows']}",
        f"- Target-available counts: {summary['rows']['target_available_counts']}",
        "",
        "## Coverage",
        "",
        f"- Overall dengue coverage: {summary['coverage']['overall_dengue_pct']}%",
        f"- Overall weather coverage: {summary['coverage']['overall_weather_pct']}%",
        f"- Dengue coverage by year: {summary['coverage']['dengue_by_year']}",
        f"- Weather coverage by year: {summary['coverage']['weather_by_year']}",
        f"- Dengue observed/district stats: {summary['coverage']['dengue_by_district']}",
        f"- Weather observed/district stats: {summary['coverage']['weather_by_district']}",
        "- Requested-year coverage supplement: "
        "`historical_requested_year_coverage.csv` uses expected Saturday weeks from "
        "2010-01-01 through 2025-12-31, separately from the accepted cohort calendar.",
        "",
        "## Quality",
        "",
        "- Primary non-trainable reason buckets: "
        f"{summary['trainability']['primary_not_trainable_reason_counts']}",
        "- Annual primary non-trainable reason buckets: "
        f"{summary['trainability']['primary_not_trainable_reason_counts_by_year']}",
        "- Secondary non-trainable flags are overlapping diagnostics: "
        f"{summary['trainability']['secondary_not_trainable_flags_overlapping']}",
        f"- Row quality breakdown: {summary['quality']['row_quality_breakdown']}",
        f"- Quarantine/source-conflict summary: {summary['quality']}",
        "- Reporting definitions caveat: historical WER/NDCU reporting periods are only "
        "accepted when parser modules map them to the canonical seven-day calendar.",
        "- Quarantine issue-year caveat: annual quarantine counts are by source issue "
        "year/week metadata, not inferred observed week; failed reports have unknown "
        "observation periods.",
        "- NDCU caveat: WER/NDCU diagnostic comparisons are not direct same-temporal "
        "counts unless the separate NDCU comparison module says so.",
        "- Population caveat: population fields use the existing 2024 reference population "
        "strategy; this is not historical annual population.",
        "- Weather overlap caveat: weather columns and provenance reuse existing CHIRPS/ERA5 "
        "weekly definitions; 2024 overlap should be compared by parent after candidate data "
        "are built.",
        "- Feature eligibility caveat: feature values and trainability can improve with longer "
        "history because lag/rolling windows have more prior observations; feature definitions "
        "are unchanged.",
        "",
        "## Reproducibility",
        "",
        f"- Tests: {summary['reproducibility']['tests']}",
        f"- Ruff: {summary['reproducibility']['ruff']}",
        f"- Dataset validation: {summary['reproducibility']['dataset_validation']}",
        f"- Clean rebuild: {summary['reproducibility']['clean_rebuild']}",
        f"- Hash verification: {summary['reproducibility']['hash_verification']}",
        "",
        "## Readiness",
        "",
        "Recommendation for Milestone 2: NOT READY",
        "",
    ]
    report_path = paths.reports / "historical_backfill_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_from_candidate_inputs(
    *,
    output_data_root: Path = DEFAULT_OUTPUT_DATA_ROOT,
    shared_data_root: Path = DEFAULT_SHARED_DATA_ROOT,
    baseline_json: Path = DEFAULT_BASELINE_JSON,
    run_validation: bool = True,
) -> HistoricalPipelineResult:
    output_data_root = Path(output_data_root)
    shared_data_root = Path(shared_data_root)
    prepare_raw_caches(shared_data_root=shared_data_root, output_data_root=output_data_root)
    paths = _candidate_paths(output_data_root)
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

    dengue_path = paths.processed / "dengue_cases_weekly.parquet"
    if not dengue_path.exists():
        raise ContractError(f"Missing candidate dengue cases: {dengue_path}")
    build_geography(offline=True, paths=paths)
    build_historical_weather_weekly(output_data_root=output_data_root)
    dengue = pd.read_parquet(dengue_path)
    baseline_status = check_phase_a_dengue_baseline(dengue, baseline_json=baseline_json)
    _ensure_quality_report_alias(paths)
    build_features(paths=paths)
    build_reports(paths=paths)
    validation_status = "NOT RUN"
    if run_validation:
        validate_all(paths=paths)
        validation_status = "PASS"
    summary = summarize_historical_pipeline(
        output_data_root=output_data_root,
        baseline_status=baseline_status,
        validation_status=validation_status,
    )
    write_historical_report(output_data_root=output_data_root, summary=summary)
    return HistoricalPipelineResult(output_data_root=output_data_root, summary=summary)


def _owned_rebuild_root(output_data_root: Path) -> None:
    marker = output_data_root / ".dengue_forecast_historical_rebuild"
    if output_data_root.exists():
        if not marker.exists():
            raise ContractError(
                f"Refusing to reuse unowned historical rebuild root: {output_data_root}"
            )
        shutil.rmtree(output_data_root)
    output_data_root.mkdir(parents=True)
    marker.write_text("owned by dengue_forecast historical raw rebuild\n", encoding="utf-8")


def rebuild_from_raw(
    *,
    shared_data_root: Path = DEFAULT_SHARED_DATA_ROOT,
    output_data_root: Path,
    baseline_json: Path = DEFAULT_BASELINE_JSON,
    run_validation: bool = True,
) -> HistoricalPipelineResult:
    output_data_root = Path(output_data_root)
    _owned_rebuild_root(output_data_root)
    prepare_raw_caches(shared_data_root=Path(shared_data_root), output_data_root=output_data_root)
    _rebase_historical_dengue_index(output_data_root / "raw" / "dengue")
    paths = _candidate_paths(output_data_root)
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

    parse_historical_reports(
        index_path=paths.raw / "dengue" / "historical_report_index.parquet",
        raw_dir=paths.raw / "dengue",
        processed_path=paths.processed / "dengue_cases_weekly.parquet",
        reports_dir=paths.reports,
        baseline_path=baseline_json,
    )
    build_geography(offline=True, paths=paths)
    boundaries = pd.read_parquet(paths.processed / "district_reference.parquet")
    import geopandas as gpd

    boundaries_gdf = gpd.read_file(paths.processed / "sri_lanka_districts.geojson")
    dengue = pd.read_parquet(paths.processed / "dengue_cases_weekly.parquet")
    starts = pd.to_datetime(dengue["week_start_date"]).dt.date
    build_historical_weather_daily(
        boundaries_gdf,
        start_date=starts.min(),
        end_date=starts.max() + timedelta(days=6),
        raw_dir=paths.raw / "weather",
        interim_dir=paths.interim / "weather",
        allow_partial=False,
    )
    baseline_status = check_phase_a_dengue_baseline(dengue, baseline_json=baseline_json)
    _ensure_quality_report_alias(paths)
    build_historical_weather_weekly(output_data_root=output_data_root)
    build_features(paths=paths)
    build_reports(paths=paths)
    validation_status = "NOT RUN"
    if run_validation:
        validate_all(paths=paths)
        validation_status = "PASS"
    hash_verification = {}
    for name in CANONICAL_ARTIFACTS:
        artifact = paths.processed / name
        if artifact.exists():
            hash_verification[name] = _sha256(artifact)
    summary = summarize_historical_pipeline(
        output_data_root=output_data_root,
        baseline_status=baseline_status,
        validation_status=validation_status,
        clean_rebuild_status="PASS",
        hash_verification=hash_verification,
    )
    write_historical_report(output_data_root=output_data_root, summary=summary)
    # Keep the symbol used above from looking accidental to static readers.
    _ = boundaries
    return HistoricalPipelineResult(output_data_root=output_data_root, summary=summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase B historical integration pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--build", action="store_true", help="Build from candidate processed/interim inputs"
    )
    mode.add_argument(
        "--rebuild", action="store_true", help="Clean rebuild from retained raw inputs only"
    )
    parser.add_argument("--shared-data-root", type=Path, default=DEFAULT_SHARED_DATA_ROOT)
    parser.add_argument("--output-data-root", type=Path, default=DEFAULT_OUTPUT_DATA_ROOT)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Generate artifacts and reports without running validate all",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.build:
        result = build_from_candidate_inputs(
            output_data_root=args.output_data_root,
            shared_data_root=args.shared_data_root,
            baseline_json=args.baseline_json,
            run_validation=not args.skip_validation,
        )
    else:
        result = rebuild_from_raw(
            output_data_root=args.output_data_root,
            shared_data_root=args.shared_data_root,
            baseline_json=args.baseline_json,
            run_validation=not args.skip_validation,
        )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
