from __future__ import annotations

import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_forecast.config import DISTRICTS

CONFLICT_COLUMNS = [
    "conflict_type",
    "district",
    "week",
    "issue_year",
    "issue_week",
    "observation_period",
    "source_document",
    "source_url",
    "source_a",
    "value_a",
    "source_b",
    "value_b",
    "chosen_value",
    "reason",
]
NATIONAL_DISTRICT_LABEL = "NATIONAL (diagnostic, not a district observation)"
REQUESTED_START = date(2010, 1, 1)
REQUESTED_END = date(2025, 12, 31)
FIRST_WEEK_CORROBORATION_METHOD = (
    "first_week_all_rdhs_weekly_a_equals_cumulative_b_national_b_corroboration"
)


_NATIONAL_RECONCILIATION_RE = re.compile(
    r"national reconciliation failed:\s*RDHS sum\s+(?P<rdhs>\d+)\s*!=\s*"
    r"national\s+(?P<national>\d+)",
    flags=re.IGNORECASE,
)


def _empty_conflicts() -> pd.DataFrame:
    return pd.DataFrame(columns=CONFLICT_COLUMNS)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _string_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _nullable_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def within_report_national_conflicts(quarantine: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if quarantine.empty:
        return _empty_conflicts()
    for _, row in quarantine.iterrows():
        if _string_value(row.get("reason")) != "SOURCE_CONFLICT":
            continue
        details = _string_value(row.get("details"))
        match = _NATIONAL_RECONCILIATION_RE.search(details)
        if match is None:
            continue
        issue_year = _nullable_int(row.get("year"))
        issue_week = _nullable_int(row.get("week"))
        week = f"issue {issue_year}-W{issue_week:02d}" if issue_year and issue_week else pd.NA
        rows.append(
            {
                "conflict_type": "within_report_national",
                "district": NATIONAL_DISTRICT_LABEL,
                "week": week,
                "issue_year": issue_year,
                "issue_week": issue_week,
                "observation_period": "unknown; report failed national reconciliation",
                "source_document": row.get("source_document"),
                "source_url": row.get("source_url"),
                "source_a": "WER printed national weekly A",
                "value_a": int(match.group("national")),
                "source_b": "sum of WER RDHS weekly A",
                "value_b": int(match.group("rdhs")),
                "chosen_value": pd.NA,
                "reason": "quarantined national inconsistency",
            }
        )
    if not rows:
        return _empty_conflicts()
    return pd.DataFrame(rows).reindex(columns=CONFLICT_COLUMNS)


def augment_source_conflicts(conflicts: pd.DataFrame, quarantine: pd.DataFrame) -> pd.DataFrame:
    if conflicts.empty:
        cross_source = _empty_conflicts()
    else:
        cross_source = conflicts.copy()
        if "conflict_type" not in cross_source.columns:
            cross_source["conflict_type"] = "accepted_cross_source"
        cross_source = cross_source[
            cross_source["conflict_type"].fillna("accepted_cross_source").ne(
                "within_report_national"
            )
        ].copy()
        cross_source["conflict_type"] = cross_source["conflict_type"].fillna(
            "accepted_cross_source"
        )
        for column in CONFLICT_COLUMNS:
            if column not in cross_source.columns:
                cross_source[column] = pd.NA
        cross_source = cross_source.reindex(columns=CONFLICT_COLUMNS)

    within = within_report_national_conflicts(quarantine)
    out = pd.concat([cross_source, within], ignore_index=True)
    if out.empty:
        return _empty_conflicts()
    return out.drop_duplicates(
        subset=[
            "conflict_type",
            "district",
            "week",
            "source_document",
            "source_url",
            "issue_year",
            "issue_week",
            "source_a",
            "source_b",
            "value_a",
            "value_b",
            "reason",
        ],
        keep="first",
    ).reset_index(drop=True)


def accepted_first_week_corroboration_warnings(quality: pd.DataFrame) -> pd.DataFrame:
    if quality.empty or "national_reconciliation_method" not in quality.columns:
        return pd.DataFrame(
            columns=[
                "warning_type",
                "source_document",
                "source_url",
                "issue_year",
                "issue_week",
                "raw_reported_national_weekly_a",
                "calculated_rdhs_weekly_a",
                "corroborating_national_total",
                "chosen_value",
                "district_quality_rows",
                "explanation",
            ]
        )
    rows = quality[
        quality["national_reconciliation_method"].eq(FIRST_WEEK_CORROBORATION_METHOD)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    group_cols = [
        column
        for column in ["source_document", "source_url", "year", "week"]
        if column in rows.columns
    ]
    warnings: list[dict[str, Any]] = []
    for key, group in rows.groupby(group_cols, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        keyed = dict(zip(group_cols, values, strict=True))
        calculated = int(group["calculated_rdhs_total"].dropna().iloc[0])
        raw_reported = int(group["reported_national_total"].dropna().iloc[0])
        corroborating = int(group["corroborating_national_total"].dropna().iloc[0])
        warnings.append(
            {
                "warning_type": "accepted_first_week_corroboration",
                "source_document": keyed.get("source_document"),
                "source_url": keyed.get("source_url"),
                "issue_year": _nullable_int(keyed.get("year")),
                "issue_week": _nullable_int(keyed.get("week")),
                "raw_reported_national_weekly_a": raw_reported,
                "calculated_rdhs_weekly_a": calculated,
                "corroborating_national_total": corroborating,
                "chosen_value": calculated,
                "district_quality_rows": int(len(group)),
                "explanation": (
                    "Accepted explained warning: chosen_value is the aggregate of retained "
                    "district weekly A rows corroborated by national B, not a replacement of "
                    "the raw printed national weekly A value."
                ),
            }
        )
    return pd.DataFrame(warnings)


def source_issue_summary(conflicts: pd.DataFrame, warnings: pd.DataFrame) -> dict[str, Any]:
    conflict_types = Counter(
        conflicts.get("conflict_type", pd.Series(dtype=object)).fillna("accepted_cross_source")
    )
    return {
        "accepted_cross_source_conflict_rows": int(
            conflict_types.get("accepted_cross_source", 0)
        ),
        "quarantined_within_report_national_conflicts": int(
            conflict_types.get("within_report_national", 0)
        ),
        "accepted_explained_warning_source_issues": int(len(warnings)),
        "total_documented_source_issues": int(len(conflicts) + len(warnings)),
        "cross_source_pair_detail": (
            "available in historical_source_conflicts.csv when accepted cross-source rows exist"
            if int(conflict_types.get("accepted_cross_source", 0))
            else "no accepted cross-source pair rows available; current issues are within-report "
            "or explained warnings"
        ),
        "status": (
            "unresolved_or_quarantined_conflicts_present"
            if int(conflict_types.get("within_report_national", 0))
            else "no_unresolved_or_quarantined_conflicts_reported"
        ),
    }


def _requested_saturday_weeks() -> pd.DatetimeIndex:
    return pd.date_range(REQUESTED_START, REQUESTED_END, freq="W-SAT")


def _available_weather_count(frame: pd.DataFrame) -> int:
    required = ["rainfall_sum_mm", "temp_mean_c", "humidity_mean_pct"]
    if frame.empty or any(column not in frame.columns for column in required):
        return 0
    return int(frame[required].notna().all(axis=1).sum())


def requested_year_coverage(
    dengue: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    cohort_calendar_rows: int,
) -> pd.DataFrame:
    weeks = _requested_saturday_weeks()
    week_dates = set(weeks.date)
    dengue_rows = dengue.copy()
    weather_rows = weather.copy()
    dengue_rows["week_start_date"] = pd.to_datetime(dengue_rows["week_start_date"]).dt.date
    weather_rows["week_start_date"] = pd.to_datetime(weather_rows["week_start_date"]).dt.date
    dengue_rows = dengue_rows[dengue_rows["week_start_date"].isin(week_dates)]
    weather_rows = weather_rows[weather_rows["week_start_date"].isin(week_dates)]
    rows: list[dict[str, Any]] = []
    for year in range(REQUESTED_START.year, REQUESTED_END.year + 1):
        year_weeks = [week.date() for week in weeks if week.year == year]
        expected_rows = len(year_weeks) * len(DISTRICTS)
        dengue_year = dengue_rows[dengue_rows["week_start_date"].isin(year_weeks)]
        weather_year = weather_rows[weather_rows["week_start_date"].isin(year_weeks)]
        observed_dengue = int(dengue_year["dengue_cases"].notna().sum())
        observed_weather = _available_weather_count(weather_year)
        full = observed_dengue == expected_rows and observed_weather == expected_rows
        rows.append(
            {
                "year": year,
                "requested_period_start": REQUESTED_START.isoformat(),
                "requested_period_end": REQUESTED_END.isoformat(),
                "expected_saturday_weeks": len(year_weeks),
                "expected_rows": expected_rows,
                "observed_dengue_rows": observed_dengue,
                "dengue_coverage_pct": round(100.0 * observed_dengue / expected_rows, 2)
                if expected_rows
                else 0.0,
                "observed_weather_rows": observed_weather,
                "weather_coverage_pct": round(100.0 * observed_weather / expected_rows, 2)
                if expected_rows
                else 0.0,
                "cohort_calendar_rows": int(cohort_calendar_rows),
                "status": "complete" if full else "partial_boundary_or_gaps",
            }
        )
    return pd.DataFrame(rows)


def write_historical_reporting_artifacts(
    *,
    reports_dir: Path,
    dengue: pd.DataFrame,
    weather: pd.DataFrame,
    dataset: pd.DataFrame,
) -> dict[str, Any]:
    conflicts_path = reports_dir / "historical_source_conflicts.csv"
    quarantine_path = reports_dir / "historical_dengue_quarantine.csv"
    quality_path = reports_dir / "historical_dengue_extraction_quality.csv"

    conflicts = augment_source_conflicts(
        _read_csv_or_empty(conflicts_path),
        _read_csv_or_empty(quarantine_path),
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(conflicts_path, index=False)

    warnings = accepted_first_week_corroboration_warnings(_read_csv_or_empty(quality_path))
    warnings_path = reports_dir / "historical_accepted_source_warnings.csv"
    warnings.to_csv(warnings_path, index=False)

    requested = requested_year_coverage(
        dengue,
        weather,
        cohort_calendar_rows=int(len(dataset)),
    )
    requested_path = reports_dir / "historical_requested_year_coverage.csv"
    requested.to_csv(requested_path, index=False)

    return {
        "source_issues": source_issue_summary(conflicts, warnings),
        "requested_year_coverage_csv": requested_path.name,
        "accepted_source_warnings_csv": warnings_path.name,
        "source_conflicts_csv": conflicts_path.name,
    }
