from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DISTRICT_COL = "district_id"
DATE_COL = "week_start_date"
OBSERVED_CASES_COL = "dengue_cases"

BASELINE_NAMES = (
    "naive_last_week",
    "naive_4w_mean",
    "naive_8w_mean",
    "seasonal_52w",
    "district_seasonal_mean",
)


def naive_last_week(validation_frame: pd.DataFrame, full_history: pd.DataFrame) -> pd.Series:
    """Predict next week with the exactly aligned observed count at forecast cutoff week t."""
    return _exact_history_lookup(validation_frame, full_history, week_offsets=[0], reducer="single")


def naive_4w_mean(validation_frame: pd.DataFrame, full_history: pd.DataFrame) -> pd.Series:
    """Predict next week with mean(cases[t], cases[t-1], cases[t-2], cases[t-3]).

    All four canonical weeks must be present as observed values. Missing weeks or missing
    observations return NaN; nearest prior observations are never substituted.
    """
    return _exact_history_lookup(validation_frame, full_history, week_offsets=[0, -1, -2, -3])


def naive_8w_mean(validation_frame: pd.DataFrame, full_history: pd.DataFrame) -> pd.Series:
    """Predict next week with the exact eight-week observed mean ending at cutoff week t."""
    return _exact_history_lookup(
        validation_frame,
        full_history,
        week_offsets=[0, -1, -2, -3, -4, -5, -6, -7],
    )


def seasonal_52w(validation_frame: pd.DataFrame, full_history: pd.DataFrame) -> pd.Series:
    """Predict next week from the observation 52 canonical weeks before the target date.

    For a one-week-ahead forecast made at cutoff week t, the target date is t+1 week.
    Therefore target(t+1) minus 52 weeks is cutoff t minus 51 weeks, not t minus 52
    weeks. The lookup is exact and never falls back to nearest prior-year observations.
    """
    return _exact_history_lookup(
        validation_frame,
        full_history,
        week_offsets=[-51],
        reducer="single",
    )


def district_seasonal_mean(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    full_history: pd.DataFrame | None = None,
) -> pd.Series:
    """Predict from training-only district by ISO calendar-week observed means.

    The prediction calendar week is computed from the target date, i.e. validation
    ``week_start_date + 7 days``. Only rows in ``training_frame`` at or before each
    forecast cutoff contribute to fitted district-season groups. Missing trained groups
    return NaN to preserve unavailable coverage, and no full-history or cross-district
    fallback is used.
    """
    _ = full_history
    train = _validate_frame(training_frame, "training_frame", require_cases=True)
    validation = _validate_frame(validation_frame, "validation_frame", require_cases=False)

    train = train.dropna(subset=[OBSERVED_CASES_COL]).copy()
    train["_season_week"] = _iso_week(train[DATE_COL])

    result: list[float] = []
    for _, row in validation.iterrows():
        target_week = int((row[DATE_COL] + pd.Timedelta(days=7)).isocalendar().week)
        eligible = train[
            train[DISTRICT_COL].eq(row[DISTRICT_COL])
            & train["_season_week"].eq(target_week)
            & train[DATE_COL].le(row[DATE_COL])
        ]
        result.append(float(eligible[OBSERVED_CASES_COL].mean()) if not eligible.empty else np.nan)
    return pd.Series(
        result,
        index=validation_frame.index,
        dtype="float64",
        name="district_seasonal_mean",
    )


def baseline_predict(
    *,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    full_history: pd.DataFrame,
) -> pd.DataFrame:
    """Return all required baseline predictions as columns aligned to validation index.

    ``full_history`` is used only for exact historical observed lookups at dates that
    are at or before each row's forecast cutoff. ``training_frame`` is used only to fit
    the district seasonal mean. The caller remains responsible for masking NaN baseline
    predictions before metric evaluation and for evaluating validation/test folds.
    """
    predictions = pd.DataFrame(index=validation_frame.index)
    predictions["naive_last_week"] = naive_last_week(validation_frame, full_history)
    predictions["naive_4w_mean"] = naive_4w_mean(validation_frame, full_history)
    predictions["naive_8w_mean"] = naive_8w_mean(validation_frame, full_history)
    predictions["seasonal_52w"] = seasonal_52w(validation_frame, full_history)
    predictions["district_seasonal_mean"] = district_seasonal_mean(
        training_frame,
        validation_frame,
        full_history,
    )
    return predictions.loc[:, list(BASELINE_NAMES)]


def baseline_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize available prediction counts and coverage for each baseline column."""
    missing = [name for name in BASELINE_NAMES if name not in predictions.columns]
    if missing:
        raise ValueError(f"Missing baseline prediction columns: {missing}")

    rows: list[dict[str, float | int | str]] = []
    total = len(predictions)
    for name in BASELINE_NAMES:
        available = int(predictions[name].notna().sum())
        rows.append(
            {
                "baseline_name": name,
                "available": available,
                "total": total,
                "coverage": float(available / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("baseline_name")


def _exact_history_lookup(
    validation_frame: pd.DataFrame,
    full_history: pd.DataFrame,
    *,
    week_offsets: Iterable[int],
    reducer: str = "mean",
) -> pd.Series:
    validation = _validate_frame(validation_frame, "validation_frame", require_cases=False)
    history = _validate_frame(full_history, "full_history", require_cases=True)
    lookup = history.set_index([DISTRICT_COL, DATE_COL])[OBSERVED_CASES_COL]

    predictions: list[float] = []
    for _, row in validation.iterrows():
        cutoff = row[DATE_COL]
        values: list[float] = []
        for offset in week_offsets:
            lookup_date = cutoff + pd.Timedelta(weeks=offset)
            if lookup_date > cutoff:
                raise ValueError("Baseline lookup attempted to use data after forecast cutoff")
            value = lookup.get((row[DISTRICT_COL], lookup_date), np.nan)
            if pd.isna(value):
                values = []
                break
            values.append(float(value))
        if not values:
            predictions.append(np.nan)
        elif reducer == "single":
            predictions.append(values[0])
        else:
            predictions.append(float(np.mean(values)))
    return pd.Series(predictions, index=validation_frame.index, dtype="float64")


def _validate_frame(frame: pd.DataFrame, frame_name: str, *, require_cases: bool) -> pd.DataFrame:
    required = {DISTRICT_COL, DATE_COL}
    if require_cases:
        required.add(OBSERVED_CASES_COL)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")

    result = frame.copy()
    result[DATE_COL] = pd.to_datetime(result[DATE_COL], errors="coerce")
    if result[DATE_COL].isna().any():
        raise ValueError(f"{frame_name}.{DATE_COL} contains missing or invalid dates")
    if result[DISTRICT_COL].isna().any():
        raise ValueError(f"{frame_name}.{DISTRICT_COL} contains missing values")
    if result.duplicated([DISTRICT_COL, DATE_COL]).any():
        raise ValueError(f"{frame_name} contains duplicate district/week rows")
    if require_cases:
        result[OBSERVED_CASES_COL] = pd.to_numeric(result[OBSERVED_CASES_COL], errors="coerce")
        negative = result[OBSERVED_CASES_COL].dropna().lt(0).any()
        if negative:
            raise ValueError(f"{frame_name}.{OBSERVED_CASES_COL} must be non-negative")
    return result


def _iso_week(values: pd.Series) -> pd.Series:
    return values.dt.isocalendar().week.astype("int64")
