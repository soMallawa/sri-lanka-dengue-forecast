from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dengue_forecast.public_inference import list_model_specs
from dengue_forecast.public_training import PublicTrainingError, train_public_model


def _synthetic_training_frame() -> pd.DataFrame:
    spec = list_model_specs()[0]
    rows = []
    dates = pd.date_range("2024-01-06", periods=8, freq="7D")
    districts = ["LK-COL", "LK-GAM"] * 4
    for idx, (week_start_date, district_id) in enumerate(zip(dates, districts, strict=True)):
        row = {"week_start_date": week_start_date, "district_id": district_id}
        for column in spec.feature_columns:
            if column == "district_id":
                continue
            row[column] = float(idx + 1)
        row["month"] = int(week_start_date.month)
        row["quarter"] = int(week_start_date.quarter)
        row["week_of_year"] = int(week_start_date.isocalendar().week)
        row["target_h1"] = float(idx + 2)
        rows.append(row)
    return pd.DataFrame(rows)


def test_public_training_wrapper_trains_synthetic_model_ready_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "model_ready.parquet"
    _synthetic_training_frame().to_parquet(input_path, index=False)
    output_dir = tmp_path / "trained" / "h1"

    result = train_public_model(
        input_path=input_path,
        horizon=1,
        train_end="2024-03-09",
        output_dir=output_dir,
        provenance={"data_scope": "synthetic unit test"},
    )

    assert result.model_dir == output_dir
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["target_column"] == "target_h1"
    assert metadata["public_training"]["horizon_weeks"] == 1
    assert metadata["public_training"]["label_cutoff_date"] == "2024-03-09"
    assert (output_dir / "model.joblib").exists()


def test_public_training_rejects_2025_dates_by_default(tmp_path: Path) -> None:
    frame = _synthetic_training_frame()
    frame["week_start_date"] = pd.date_range("2025-01-04", periods=len(frame), freq="7D")
    input_path = tmp_path / "model_ready.parquet"
    frame.to_parquet(input_path, index=False)

    with pytest.raises(PublicTrainingError, match="2025"):
        train_public_model(
            input_path=input_path,
            horizon=1,
            train_end="2025-03-01",
            output_dir=tmp_path / "trained",
        )


def test_public_training_rejects_nonfinite_features(tmp_path: Path) -> None:
    frame = _synthetic_training_frame()
    frame.loc[0, "cases_lag_1"] = np.inf
    input_path = tmp_path / "model_ready.parquet"
    frame.to_parquet(input_path, index=False)

    with pytest.raises(PublicTrainingError, match="non-finite"):
        train_public_model(
            input_path=input_path,
            horizon=1,
            train_end="2024-03-09",
            output_dir=tmp_path / "trained",
        )
