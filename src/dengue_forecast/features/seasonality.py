from __future__ import annotations

import math


def add_seasonality_features(df):  # type: ignore[no-untyped-def]
    import numpy as np
    import pandas as pd

    out = df.copy()
    week_start = pd.to_datetime(out["week_start_date"])
    out["month"] = week_start.dt.month
    out["quarter"] = week_start.dt.quarter
    if "week" in out.columns:
        out["week_of_year"] = out["week"]
    else:
        out["week_of_year"] = week_start.dt.isocalendar().week.astype(int)
    out["week_sin"] = np.sin(2 * math.pi * out["week_of_year"] / 52.1775)
    out["week_cos"] = np.cos(2 * math.pi * out["week_of_year"] / 52.1775)
    out["month_sin"] = np.sin(2 * math.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * math.pi * out["month"] / 12)
    return out
