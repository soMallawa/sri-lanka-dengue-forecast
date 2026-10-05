from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dengue_forecast.modeling.preprocessing import FoldPreprocessor, PreprocessingError


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district_id": ["LK-A", "LK-B", "LK-A", "LK-Z"],
            "week_start_date": pd.to_datetime(
                ["2020-01-04", "2020-01-04", "2020-01-11", "2020-01-18"]
            ),
            "cases_lag_1": pd.Series([1, None, 3, 99], dtype="Int64"),
            "rainfall_sum_mm": pd.Series([0.5, 1.5, None, 99.0], dtype="Float64"),
            "all_missing_train": pd.Series([None, None, None, 5.0], dtype="Float64"),
            "cases_next_week": [2, 3, 4, 5],
        }
    )


def test_fit_state_uses_only_provided_training_rows_and_freezes_columns() -> None:
    train = _frame().iloc[:3].copy()
    validation = _frame().iloc[[3]].copy()
    pre = FoldPreprocessor(model_family="ridge").fit(
        train,
        feature_columns=["district_id", "cases_lag_1", "rainfall_sum_mm", "all_missing_train"],
    )

    state = pre.fit_state
    assert state["training_row_count"] == 3
    assert state["training_row_keys"] == [
        {"district_id": "LK-A", "week_start_date": "2020-01-04"},
        {"district_id": "LK-B", "week_start_date": "2020-01-04"},
        {"district_id": "LK-A", "week_start_date": "2020-01-11"},
    ]
    assert state["dropped_all_missing_features"] == ["all_missing_train"]
    assert "all_missing_train" not in pre.output_feature_columns
    assert pre.output_feature_columns == state["output_feature_columns"]

    transformed = pre.transform(validation[pre.feature_columns])
    assert transformed.shape == (1, len(pre.output_feature_columns))
    assert np.isfinite(transformed.to_numpy()).all()
    assert transformed.loc[validation.index[0], "district_id__LK-A"] == 0
    assert transformed.loc[validation.index[0], "district_id__LK-B"] == 0
    assert "cases_lag_1__missing" in transformed.columns
    assert "rainfall_sum_mm__missing" in transformed.columns


def test_transform_rejects_schema_mismatch_forbidden_fields_and_unfitted_use() -> None:
    with pytest.raises(PreprocessingError, match="not fitted"):
        FoldPreprocessor().transform(_frame())

    pre = FoldPreprocessor().fit(
        _frame().iloc[:3],
        feature_columns=["district_id", "cases_lag_1", "rainfall_sum_mm"],
    )
    extra = _frame().assign(population_reference=1)
    with pytest.raises(PreprocessingError, match="Unexpected feature columns"):
        pre.transform(
            extra[["district_id", "cases_lag_1", "rainfall_sum_mm", "population_reference"]]
        )

    with pytest.raises(PreprocessingError, match="Forbidden"):
        FoldPreprocessor().fit(
            _frame().iloc[:3],
            feature_columns=["district_id", "cases_lag_1", "cases_next_week"],
        )


def test_ridge_scales_numeric_but_tree_families_keep_native_numeric_scale() -> None:
    train = _frame().iloc[:3].copy()
    features = ["district_id", "cases_lag_1", "rainfall_sum_mm"]
    ridge = FoldPreprocessor(model_family="ridge").fit(train, feature_columns=features)
    unscaled_preprocessors = [
        FoldPreprocessor(model_family=family).fit(train, feature_columns=features)
        for family in [
            "random_forest",
            "hist_gradient_boosting_poisson",
            "xgboost",
            "lightgbm",
        ]
    ]

    ridge_x = ridge.transform(train[features])

    for pre in unscaled_preprocessors:
        unscaled_x = pre.transform(train[features])
        assert not np.allclose(
            ridge_x["rainfall_sum_mm"].to_numpy(),
            unscaled_x["rainfall_sum_mm"].to_numpy(),
        )
        assert unscaled_x["rainfall_sum_mm"].tolist() == [0.5, 1.5, 1.0]


def test_poisson_scales_numeric_using_training_rows_only() -> None:
    train = _frame().iloc[:3].copy()
    validation = _frame().iloc[[3]].copy()
    features = ["district_id", "cases_lag_1", "rainfall_sum_mm"]
    pre = FoldPreprocessor(model_family="poisson").fit(train, feature_columns=features)

    assert pre.fit_state["scale_numeric"] is True
    assert pre.scaler_ is not None
    assert pre.scaler_.n_samples_seen_ == 3

    train_x = pre.transform(train[features])
    validation_x = pre.transform(validation[features])

    assert np.allclose(train_x[["cases_lag_1", "rainfall_sum_mm"]].mean().to_numpy(), 0.0)
    assert np.allclose(train_x[["cases_lag_1", "rainfall_sum_mm"]].std(ddof=0).to_numpy(), 1.0)
    assert validation_x.loc[validation.index[0], "rainfall_sum_mm"] > 100.0
    assert pre.scaler_.n_samples_seen_ == 3
