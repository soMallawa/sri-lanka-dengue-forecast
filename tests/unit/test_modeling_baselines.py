from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dengue_forecast.modeling.baselines import (
    BASELINE_NAMES,
    baseline_coverage,
    baseline_predict,
    district_seasonal_mean,
    naive_4w_mean,
    naive_8w_mean,
    naive_last_week,
    seasonal_52w,
)


def _history() -> pd.DataFrame:
    dates = pd.date_range("2023-01-07", periods=62, freq="7D")
    rows: list[dict[str, object]] = []
    for district_id, offset in [("A", 0), ("B", 1000)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "district_id": district_id,
                    "week_start_date": date,
                    "dengue_cases": float(i + offset),
                }
            )
    frame = pd.DataFrame(rows)
    # Preserve the calendar row but make one observed value unavailable for district A.
    frame.loc[
        frame["district_id"].eq("A") & frame["week_start_date"].eq(pd.Timestamp("2023-02-04")),
        "dengue_cases",
    ] = np.nan
    return frame


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district_id": ["A", "A", "B"],
            "week_start_date": pd.to_datetime(["2024-03-09", "2023-02-11", "2024-03-09"]),
            "cases_next_week": [62.0, 6.0, 1062.0],
        },
        index=[10, 11, 12],
    )


def test_required_baseline_names_are_stable() -> None:
    assert BASELINE_NAMES == (
        "naive_last_week",
        "naive_4w_mean",
        "naive_8w_mean",
        "seasonal_52w",
        "district_seasonal_mean",
    )


def test_current_and_rolling_baselines_require_exact_observed_canonical_weeks() -> None:
    history = _history()
    validation = _validation()

    assert naive_last_week(validation.iloc[[0]], history).loc[10] == 61.0
    assert naive_4w_mean(validation.iloc[[0]], history).loc[10] == pytest.approx(
        np.mean([61.0, 60.0, 59.0, 58.0])
    )
    assert naive_8w_mean(validation.iloc[[0]], history).loc[10] == pytest.approx(
        np.mean([61.0, 60.0, 59.0, 58.0, 57.0, 56.0, 55.0, 54.0])
    )

    # The current week exists, but t-1/t-2/t-3 include a missing observed value.
    assert naive_last_week(validation.iloc[[1]], history).loc[11] == 5.0
    assert pd.isna(naive_4w_mean(validation.iloc[[1]], history).loc[11])
    assert pd.isna(naive_8w_mean(validation.iloc[[1]], history).loc[11])


def test_seasonal_52w_uses_target_date_minus_52_weeks_not_cutoff_minus_52() -> None:
    history = _history()
    validation = _validation().iloc[[0]]

    prediction = seasonal_52w(validation, history).loc[10]

    # Forecast cutoff is 2024-03-09. Target date is 2024-03-16, and target minus
    # 52 weeks is 2023-03-18. This is t-51 weeks, not t-52 weeks.
    target_minus_52w = history.loc[
        history["district_id"].eq("A")
        & history["week_start_date"].eq(pd.Timestamp("2023-03-18")),
        "dengue_cases",
    ].item()
    cutoff_minus_52w = history.loc[
        history["district_id"].eq("A")
        & history["week_start_date"].eq(pd.Timestamp("2023-03-11")),
        "dengue_cases",
    ].item()
    assert prediction == target_minus_52w
    assert prediction != cutoff_minus_52w


def test_district_seasonal_mean_uses_training_only_without_cross_district_leakage() -> None:
    train = pd.DataFrame(
        {
            "district_id": ["A", "A", "B", "B", "A"],
            "week_start_date": pd.to_datetime(
                ["2022-03-19", "2023-03-18", "2022-03-19", "2023-03-18", "2025-03-15"]
            ),
            "dengue_cases": [10.0, 30.0, 500.0, 700.0, 9999.0],
        }
    )
    validation = pd.DataFrame(
        {
            "district_id": ["A", "B", "A"],
            "week_start_date": pd.to_datetime(["2024-03-09", "2024-03-09", "2024-12-21"]),
        },
        index=[0, 1, 2],
    )
    full_history = pd.concat(
        [
            train,
            pd.DataFrame(
                {
                    "district_id": ["A"],
                    "week_start_date": pd.to_datetime(["2024-03-16"]),
                    "dengue_cases": [9999.0],
                }
            ),
        ],
        ignore_index=True,
    )

    predictions = district_seasonal_mean(train, validation, full_history)

    assert predictions.loc[0] == 20.0
    assert predictions.loc[1] == 600.0
    assert pd.isna(predictions.loc[2])


def test_baseline_predict_returns_aligned_wide_predictions_and_coverage() -> None:
    predictions = baseline_predict(
        training_frame=_history().loc[lambda x: x["week_start_date"].dt.year == 2023],
        validation_frame=_validation(),
        full_history=_history(),
    )

    assert predictions.index.tolist() == [10, 11, 12]
    assert tuple(predictions.columns) == BASELINE_NAMES
    assert predictions.loc[10, "naive_last_week"] == 61.0
    assert predictions.loc[12, "naive_last_week"] == 1061.0
    assert predictions.loc[10, "district_seasonal_mean"] == 10.0
    assert predictions.loc[12, "district_seasonal_mean"] == 1010.0

    coverage = baseline_coverage(predictions)
    assert coverage.loc["naive_last_week", "available"] == 3
    assert coverage.loc["naive_4w_mean", "available"] == 2
    assert coverage.loc["naive_4w_mean", "coverage"] == pytest.approx(2 / 3)


def test_baseline_inputs_reject_duplicate_district_weeks_and_bad_dates() -> None:
    history = _history()
    duplicate = pd.concat([history, history.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        naive_last_week(_validation(), duplicate)

    bad_validation = _validation()
    bad_validation.loc[10, "week_start_date"] = pd.NaT
    with pytest.raises(ValueError, match="week_start_date"):
        naive_last_week(bad_validation, history)
