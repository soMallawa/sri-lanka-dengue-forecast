from __future__ import annotations

from datetime import date, timedelta

from dengue_forecast.contracts import ContractError


def validate_week_boundary(week_start_date: date, week_end_date: date) -> None:
    days = (week_end_date - week_start_date).days + 1
    if days != 7:
        raise ContractError(
            "Week boundaries must represent exactly 7 calendar days; "
            f"got {week_start_date} to {week_end_date}"
        )


def make_week_fields(
    week_start_date: date,
    week_end_date: date,
    *,
    source_year: int,
    source_week: int,
) -> dict[str, object]:
    validate_week_boundary(week_start_date, week_end_date)
    if not 1 <= source_week <= 53:
        raise ContractError(f"Source week must be 1..53, got {source_week}")
    return {
        "week_start_date": week_start_date,
        "week_end_date": week_end_date,
        "year": int(source_year),
        "week": int(source_week),
        "year_week": f"{int(source_year)}-W{int(source_week):02d}",
    }


def complete_week_starts(start: date, end: date) -> list[date]:
    if end < start:
        raise ContractError("Calendar end must be on or after start")
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=7)
    return dates
