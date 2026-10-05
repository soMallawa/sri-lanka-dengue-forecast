from __future__ import annotations


def add_targets(df, *, group_col: str = "district_id", value_col: str = "dengue_cases"):  # type: ignore[no-untyped-def]
    import pandas as pd

    out = df.copy()
    if "week_start_date" not in out.columns:
        raise ValueError("add_targets requires week_start_date for exact-date target joins")
    original_dates = out["week_start_date"].copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"])
    key = [group_col, "week_start_date"]
    lookup = out[key + [value_col]].copy()
    lookup["week_start_date"] = pd.to_datetime(lookup["week_start_date"])
    for horizon, column in [(1, "cases_next_week"), (2, "cases_next_2w"), (4, "cases_next_4w")]:
        future = lookup.copy()
        future["week_start_date"] = future["week_start_date"] - pd.Timedelta(days=7 * horizon)
        future = future.rename(columns={value_col: column})
        out = out.merge(future[key + [column]], on=key, how="left", validate="one_to_one")
    out["week_start_date"] = original_dates.to_numpy()
    if "population_reference" in out.columns:
        population = out["population_reference"].replace({0: None})
        out["incidence_next_week_per_100k"] = (out["cases_next_week"] / population) * 100_000
    return out
