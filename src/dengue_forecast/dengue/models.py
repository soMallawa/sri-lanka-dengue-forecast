from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

PARSER_VERSION = "dengue-stage2-phase-b-corroboration-2026-10-04"
RECONCILIATION_REPORTED_WEEKLY_MATCH = "reported_weekly_total_match"
RECONCILIATION_FIRST_WEEK_CORROBORATION = (
    "first_week_all_rdhs_weekly_a_equals_cumulative_b_national_b_corroboration"
)
FIRST_WEEK_CORROBORATION_WARNING = (
    "Printed national weekly A differs from summed RDHS weekly A; accepted only because "
    "source week is 1, all 26 RDHS weekly A values equal their cumulative B values, "
    "and their sum equals the printed national cumulative B."
)


@dataclass(frozen=True)
class ReportDescriptor:
    source: str
    title: str
    url: str
    document_type: str
    discovered_at: str
    issue_year: int | None = None
    issue_week: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    observation_week: int | None = None
    volume: int | None = None

    @property
    def year(self) -> int | None:
        return self.issue_year

    @property
    def week(self) -> int | None:
        return self.issue_week


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    source_name: str
    source_url: str
    retrieved_at: str
    sha256: str
    media_type: str
    descriptor: dict[str, Any]


@dataclass(frozen=True)
class ParsedRegionCase:
    region_name: str
    current_week_cases: int | None
    cumulative_cases: int | None
    reporting_returns_pct: int | None
    page_number: int
    bbox: tuple[float, float, float, float]
    raw_values: dict[str, Any]
    case_status: str = "observed"
    completeness_flag: str | None = None


@dataclass(frozen=True)
class ParsedDengueReport:
    source_name: str
    source_document: str
    source_url: str
    source_retrieved_at: str
    parser_version: str
    source_year: int
    source_week: int
    week_start_date: date
    week_end_date: date
    issue_year: int | None
    issue_week: int | None
    reported_national_total: int | None
    calculated_rdhs_total: int
    rows: list[ParsedRegionCase]
    parser_name: str
    authentic_snippet: str
    corroborating_national_total: int | None = None
    national_reconciliation_method: str = RECONCILIATION_REPORTED_WEEKLY_MATCH
    national_reconciliation_warning: str | None = None
    quarantine_reason: str | None = None

    def with_replaced_case(
        self, region_name: str, value: int, *, source_document_suffix: str = ""
    ) -> ParsedDengueReport:
        rows = [
            replace(row, current_week_cases=value) if row.region_name == region_name else row
            for row in self.rows
        ]
        return replace(
            self,
            rows=rows,
            source_document=f"{self.source_document}{source_document_suffix}",
        )
