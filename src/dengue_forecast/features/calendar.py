from __future__ import annotations

from datetime import timedelta

from dengue_forecast.config import DISTRICT_BY_ID, DISTRICTS
from dengue_forecast.contracts import ContractError


def build_district_week_calendar(
    dengue,
    *,
    district_ids: list[str] | None = None,
):  # type: ignore[no-untyped-def]
    import pandas as pd

    ids = district_ids or [district.district_id for district in DISTRICTS]
    unknown = sorted(set(ids) - set(DISTRICT_BY_ID))
    if unknown:
        raise ContractError(f"Unknown calendar district IDs: {unknown}")

    starts = pd.to_datetime(dengue["week_start_date"]).dt.date
    if starts.empty:
        raise ContractError("Cannot build calendar from an empty dengue frame")
    min_start = starts.min()
    max_start = starts.max()
    week_starts = pd.date_range(min_start, max_start, freq="7D")
    source_starts = set(starts)
    off_grid = sorted(source_starts - {week_start.date() for week_start in week_starts})
    if off_grid:
        raise ContractError(
            f"Input rows do not fit a single seven-day calendar anchor: {off_grid[:5]}"
        )

    label_columns = ["week_start_date", "week_end_date", "year", "week", "year_week"]
    labels = dengue[label_columns].drop_duplicates()
    conflicting = labels.duplicated(["week_start_date"], keep=False)
    if conflicting.any():
        sample = labels.loc[conflicting].sort_values("week_start_date").head(5).to_dict("records")
        raise ContractError(f"Contradictory source week labels for the same interval: {sample}")
    labels = labels.set_index("week_start_date")

    records: list[dict[str, object]] = []
    for district_id in ids:
        district_name = DISTRICT_BY_ID[district_id].district_name
        for week_start in week_starts:
            start_date = week_start.date()
            label = labels.loc[start_date] if start_date in labels.index else None
            end_date = (
                label["week_end_date"] if label is not None else start_date + timedelta(days=6)
            )
            records.append(
                {
                    "district_id": district_id,
                    "district_name_calendar": district_name,
                    "week_start_date": start_date,
                    "week_end_date_calendar": end_date,
                    "year_calendar": int(label["year"]) if label is not None else None,
                    "week_calendar": int(label["week"]) if label is not None else None,
                    "year_week_calendar": str(label["year_week"]) if label is not None else None,
                }
            )
    return pd.DataFrame.from_records(records)
