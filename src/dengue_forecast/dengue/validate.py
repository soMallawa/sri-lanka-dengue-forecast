from __future__ import annotations

import pandas as pd


def dengue_coverage_report(
    df: pd.DataFrame, *, conflict_count: int, quarantined_reports: int
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "expected_district_weeks",
                "observed",
                "missing",
                "coverage_pct",
                "number_of_source_conflicts",
                "number_of_quarantined_reports",
            ]
        )
    rows = []
    for year, group in df.groupby("year"):
        expected = group["week_start_date"].nunique() * group["district_id"].nunique()
        observed = int(group["dengue_cases"].notna().sum())
        missing = int(expected - observed)
        rows.append(
            {
                "year": int(year),
                "expected_district_weeks": int(expected),
                "observed": observed,
                "missing": missing,
                "coverage_pct": round(observed / expected * 100, 2) if expected else 0.0,
                "number_of_source_conflicts": conflict_count,
                "number_of_quarantined_reports": quarantined_reports,
            }
        )
    return pd.DataFrame(rows)
