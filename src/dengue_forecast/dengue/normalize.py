from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from dengue_forecast.config import DISTRICTS, normalize_district_name
from dengue_forecast.contracts import ContractError, validate_dengue_weekly
from dengue_forecast.dengue.models import (
    RECONCILIATION_FIRST_WEEK_CORROBORATION,
    RECONCILIATION_REPORTED_WEEKLY_MATCH,
    ParsedDengueReport,
    ParsedRegionCase,
)
from dengue_forecast.utils.dates import make_week_fields

KALMUNAI_CROSSWALK_NOTE = (
    "WER reports 26 RDHS regions. Kalmunai is a health-region component of the "
    "administrative Ampara district for this Stage2 25-district dataset; Ampara "
    "district counts are summed from Ampara + Kalmunai components with provenance retained."
)

EXPECTED_COMPONENTS = {
    "Ampara": ("Ampara", "Kalmunai"),
}


def _expected_components(district_name: str) -> tuple[str, ...]:
    return EXPECTED_COMPONENTS.get(district_name, (district_name,))


def _validate_national_reconciliation(report: ParsedDengueReport) -> None:
    method = report.national_reconciliation_method
    if method == RECONCILIATION_REPORTED_WEEKLY_MATCH:
        if report.reported_national_total != report.calculated_rdhs_total:
            raise ContractError(
                "Reported national reconciliation method requires RDHS sum to equal "
                "reported national weekly total."
            )
        if report.corroborating_national_total not in (None, report.calculated_rdhs_total):
            raise ContractError(
                "Reported national reconciliation cannot carry a conflicting "
                "corroborating total."
            )
        return

    if method != RECONCILIATION_FIRST_WEEK_CORROBORATION:
        raise ContractError(f"Unknown national reconciliation method: {method}")

    current_values = [row.current_week_cases for row in report.rows]
    cumulative_values = [row.cumulative_cases for row in report.rows]
    if (
        report.source_week != 1
        or len(report.rows) != 26
        or report.reported_national_total is None
        or report.reported_national_total == report.calculated_rdhs_total
        or report.corroborating_national_total != report.calculated_rdhs_total
        or any(value is None for value in current_values + cumulative_values)
        or any(row.current_week_cases != row.cumulative_cases for row in report.rows)
        or sum(int(value) for value in current_values if value is not None)
        != report.calculated_rdhs_total
    ):
        raise ContractError(
            "First-week national corroboration method lacks required evidence: "
            "source week 1, 26 complete RDHS rows, every weekly A equal to its "
            "cumulative B, RDHS sum equal to corroborating national B, and printed "
            "national weekly A different."
        )


def _district_rows(report: ParsedDengueReport) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_national_reconciliation(report)
    grouped: dict[str, list[ParsedRegionCase]] = defaultdict(list)
    for row in report.rows:
        district_name = "Ampara" if row.region_name == "Kalmunai" else row.region_name
        grouped[district_name].append(row)

    week_fields = make_week_fields(
        report.week_start_date,
        report.week_end_date,
        source_year=report.source_year,
        source_week=report.source_week,
    )
    canonical_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for district in DISTRICTS:
        components = grouped.get(district.district_name, [])
        component_names = [component.region_name for component in components]
        duplicate_components = sorted(
            {name for name in component_names if component_names.count(name) > 1}
        )
        if duplicate_components:
            raise ContractError(
                f"Duplicate source components for {district.district_name}: {duplicate_components}"
            )

        expected_components = _expected_components(district.district_name)
        missing_components = [name for name in expected_components if name not in component_names]
        observed = [
            component
            for component in components
            if component.current_week_cases is not None and component.case_status == "observed"
        ]
        missing_observed_components = [
            name
            for name in expected_components
            if not any(
                component.region_name == name
                and component.current_week_cases is not None
                and component.case_status == "observed"
                for component in components
            )
        ]
        complete_components = not missing_components and not missing_observed_components
        dengue_cases = (
            sum(component.current_week_cases or 0 for component in observed)
            if complete_components
            else None
        )
        case_status = "observed" if complete_components else "missing"
        integrity_issues: list[str] = []
        if missing_components:
            integrity_issues.append(f"missing expected components: {', '.join(missing_components)}")
        observed_missing_only = [
            name for name in missing_observed_components if name not in missing_components
        ]
        if observed_missing_only:
            integrity_issues.append(
                f"components not observed/non-null: {', '.join(observed_missing_only)}"
            )
        common = {
            "district_id": district.district_id,
            "district_name": district.district_name,
            **week_fields,
            "source_name": report.source_name,
            "source_document": report.source_document,
            "source_url": report.source_url,
            "source_retrieved_at": report.source_retrieved_at,
            "parser_version": report.parser_version,
        }
        canonical_rows.append({**common, "dengue_cases": dengue_cases, "case_status": case_status})
        quality_rows.append(
            {
                **common,
                "reported_national_total": report.reported_national_total,
                "calculated_rdhs_total": report.calculated_rdhs_total,
                "corroborating_national_total": report.corroborating_national_total,
                "national_reconciliation_method": report.national_reconciliation_method,
                "national_reconciliation_warning": report.national_reconciliation_warning or "",
                "component_regions": "|".join(component.region_name for component in components),
                "component_current_week_cases": "|".join(
                    ""
                    if component.current_week_cases is None
                    else str(component.current_week_cases)
                    for component in components
                ),
                "component_cumulative_cases": "|".join(
                    "" if component.cumulative_cases is None else str(component.cumulative_cases)
                    for component in components
                ),
                "component_page_bboxes": "|".join(
                    f"p{component.page_number}:"
                    f"{','.join(f'{value:.1f}' for value in component.bbox)}"
                    for component in components
                ),
                "zero_with_zero_returns": any(
                    component.completeness_flag == "zero_with_zero_returns"
                    for component in components
                ),
                "component_completeness_status": "complete"
                if complete_components
                else "incomplete",
                "component_integrity_issue": "; ".join(integrity_issues),
                "crosswalk_note": KALMUNAI_CROSSWALK_NOTE
                if district.district_name == "Ampara"
                else "",
                "parser_name": report.parser_name,
            }
        )
    return canonical_rows, quality_rows


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "source_name",
        "source_document",
        "source_url",
        "source_retrieved_at",
        "parser_version",
    ]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _conflict_records(group: pd.DataFrame, *, resolution: str, reason: str) -> list[dict[str, Any]]:
    observed = group[group["dengue_cases"].notna()].copy()
    observed = _stable_sort(observed)
    records: list[dict[str, Any]] = []
    if len(observed) < 2:
        return records
    baseline = observed.iloc[0]
    for _, other in observed.iloc[1:].iterrows():
        if other["dengue_cases"] == baseline["dengue_cases"]:
            continue
        source_a = str(baseline["source_document"])
        source_b = str(other["source_document"])
        value_a = baseline["dengue_cases"]
        value_b = other["dengue_cases"]
        if (source_b, str(value_b)) < (source_a, str(value_a)):
            source_a, source_b = source_b, source_a
            value_a, value_b = value_b, value_a
        records.append(
            {
                "district": baseline["district_name"],
                "week": baseline["year_week"],
                "source_a": source_a,
                "value_a": value_a,
                "source_b": source_b,
                "value_b": value_b,
                "resolution": resolution,
                "reason": reason,
            }
        )
    return records


def normalize_parsed_reports(
    reports: list[ParsedDengueReport],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for report in reports:
        report_rows, quality_rows = _district_rows(report)
        rows.extend(report_rows)
        quality.extend(quality_rows)
    df = pd.DataFrame(rows)
    quality_df = pd.DataFrame(quality)
    if df.empty:
        return df, quality_df, pd.DataFrame()

    for name in df["district_name"].dropna().unique():
        normalize_district_name(str(name))

    df = df.drop_duplicates().reset_index(drop=True)
    conflicts: list[dict[str, Any]] = []
    resolved_rows: list[pd.Series] = []
    for _, group in df.groupby(["district_id", "week_start_date"], sort=False, dropna=False):
        stable = _stable_sort(group)
        observed = stable[stable["dengue_cases"].notna()].copy()
        observed_values = observed["dengue_cases"].dropna().unique().tolist()
        if len(observed_values) == 0:
            resolved_rows.append(stable.iloc[0].copy())
            continue
        if len(observed_values) == 1:
            resolved_rows.append(_stable_sort(observed).iloc[0].copy())
            continue

        conflicts.extend(
            _conflict_records(
                stable,
                resolution="unresolved_official_conflict",
                reason=(
                    "Official sources disagree for the same district-period and no structured "
                    "revision evidence establishes precedence; canonical value set missing."
                ),
            )
        )
        unresolved = stable.iloc[0].copy()
        unresolved["dengue_cases"] = pd.NA
        unresolved["case_status"] = "missing"
        resolved_rows.append(unresolved)

    out = pd.DataFrame(resolved_rows)
    out = out.sort_values(["district_id", "week_start_date"]).reset_index(drop=True)
    out = validate_dengue_weekly(out)
    return out, quality_df, pd.DataFrame(conflicts)
