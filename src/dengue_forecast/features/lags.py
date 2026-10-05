from __future__ import annotations


def add_lag_features(df, *, group_col: str, value_col: str, lags: list[int], prefix: str):  # type: ignore[no-untyped-def]
    import pandas as pd

    out = df.copy()
    if "week_start_date" not in out.columns:
        raise ValueError("add_lag_features requires week_start_date for exact-date lag joins")
    original_dates = out["week_start_date"].copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"])
    key = [group_col, "week_start_date"]
    lookup = out[key + [value_col]].copy()
    for lag in lags:
        past = lookup.copy()
        past["week_start_date"] = past["week_start_date"] + pd.Timedelta(days=7 * lag)
        past = past.rename(columns={value_col: f"{prefix}_lag_{lag}"})
        out = out.merge(
            past[key + [f"{prefix}_lag_{lag}"]], on=key, how="left", validate="one_to_one"
        )
    out["week_start_date"] = original_dates.to_numpy()
    return out
