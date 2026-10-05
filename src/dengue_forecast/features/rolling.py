from __future__ import annotations


def add_group_rolling_features(
    df, *, group_col: str, value_col: str, windows: list[int], prefix: str
):  # type: ignore[no-untyped-def]
    out = df.copy()
    grouped = out.groupby(group_col, sort=False)[value_col]
    for window in windows:
        rolling = grouped.rolling(window=window, min_periods=window)
        out[f"{prefix}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
    return out
