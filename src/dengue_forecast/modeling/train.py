from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge

from dengue_forecast.modeling.dataset import get_target_column
from dengue_forecast.modeling.preprocessing import FoldPreprocessor, PreprocessingError
from dengue_forecast.utils.hashing import sha256_file


class TrainingError(ValueError):
    """Raised when fold training or local prediction violates the model contract."""


DEFAULT_SEED = 42
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
POISSON_OBJECTIVES = {"poisson", "count:poisson"}
AUTHORITATIVE_ROW_IDENTITY_KEYS = {
    "row_count",
    "row_key_digest",
    "first_week_start_date",
    "last_week_start_date",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    family: str
    objective: str = "regression"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    seed: int = DEFAULT_SEED
    postprocessing: dict[str, Any] = field(
        default_factory=lambda: {"clip_negative_predictions": True}
    )

    def serializable(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "objective": self.objective,
            "hyperparams": self.hyperparams,
            "seed": self.seed,
            "preprocessing": {"fit_scope": "fold_training_rows_only"},
            "postprocessing": self.postprocessing,
        }


@dataclass
class TrainedModel:
    model: Any
    preprocessor: FoldPreprocessor
    config: ModelConfig
    feature_columns: list[str]
    target_column: str
    metadata: dict[str, Any]
    model_dir: Path

    def predict_next_week(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != self.feature_columns:
            if set(features.columns) == set(self.feature_columns):
                raise TrainingError("Feature column order mismatch")
            raise TrainingError("Feature columns do not match trained schema")
        try:
            x = self.preprocessor.transform(features)
        except PreprocessingError as exc:
            raise TrainingError(str(exc)) from exc
        predictions = np.asarray(self.model.predict(x), dtype="float64")
        if bool(self.config.postprocessing.get("clip_negative_predictions", True)):
            predictions = np.clip(predictions, 0.0, None)
        return predictions


def _make_estimator(config: ModelConfig) -> Any:
    params = dict(config.hyperparams)
    family = config.family
    objective = config.objective
    if family == "ridge":
        return Ridge(random_state=config.seed, **params)
    if family == "poisson":
        params.setdefault("max_iter", 300)
        return PoissonRegressor(**params)
    if family in {"hist_gradient_boosting", "hist_gradient_boosting_poisson"}:
        loss = "poisson" if objective in POISSON_OBJECTIVES else "squared_error"
        if family == "hist_gradient_boosting_poisson":
            loss = "poisson"
        return HistGradientBoostingRegressor(loss=loss, random_state=config.seed, **params)
    if family == "random_forest":
        defaults = {
            "n_estimators": 128,
            "max_depth": 10,
            "min_samples_leaf": 3,
            "random_state": config.seed,
            "n_jobs": 1,
        }
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if family == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - availability tested in environment
            raise TrainingError("XGBoost is required for family='xgboost'") from exc
        defaults = {
            "objective": "count:poisson" if objective in POISSON_OBJECTIVES else "reg:squarederror",
            "n_estimators": 32,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": config.seed,
            "n_jobs": 1,
            "verbosity": 0,
        }
        defaults.update(params)
        return XGBRegressor(**defaults)
    if family == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - availability tested in environment
            raise TrainingError("LightGBM is required for family='lightgbm'") from exc
        defaults = {
            "objective": "poisson" if objective in POISSON_OBJECTIVES else "regression",
            "n_estimators": 32,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": config.seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        defaults.update(params)
        return LGBMRegressor(**defaults)
    raise TrainingError(f"Unknown model family: {family}")


def _validate_training_dates(
    frame: pd.DataFrame,
    frozen_period_bounds: dict[str, Any] | None,
) -> None:
    if "week_start_date" not in frame.columns:
        raise TrainingError("Training frame requires week_start_date for temporal provenance")
    dates = pd.to_datetime(frame["week_start_date"], errors="coerce")
    if dates.isna().any():
        raise TrainingError("Training frame contains invalid week_start_date values")
    bounds = frozen_period_bounds or {}
    train_start = bounds.get("train_start")
    train_end = bounds.get("train_end")
    if train_start is not None and dates.lt(pd.Timestamp(train_start)).any():
        raise TrainingError("Training rows outside frozen training period")
    if train_end is not None and dates.gt(pd.Timestamp(train_end)).any():
        raise TrainingError("Training rows outside frozen training period")


def _row_identity(frame: pd.DataFrame, provenance: dict[str, Any] | None) -> dict[str, Any]:
    rows = [
        f"{row.district_id}|{pd.Timestamp(row.week_start_date).date().isoformat()}"
        for row in frame[["district_id", "week_start_date"]].itertuples(index=False)
    ]
    identity = {
        "row_count": int(len(frame)),
        "row_key_digest": _sha256_text("\n".join(rows)),
        "first_week_start_date": pd.to_datetime(frame["week_start_date"]).min().date().isoformat(),
        "last_week_start_date": pd.to_datetime(frame["week_start_date"]).max().date().isoformat(),
    }
    if provenance:
        collisions = sorted(AUTHORITATIVE_ROW_IDENTITY_KEYS & set(provenance))
        if collisions:
            raise TrainingError(
                "Training row provenance cannot override authoritative identity keys: "
                f"{collisions}"
            )
        identity.update(provenance)
    return identity


def train_fold_model(
    training_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str | None = None,
    config: ModelConfig,
    output_dir: str | Path,
    frozen_period_bounds: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> TrainedModel:
    target = target_column or get_target_column()
    required = set(feature_columns) | {target, "district_id", "week_start_date"}
    missing = sorted(required - set(training_frame.columns))
    if missing:
        raise TrainingError(f"Training frame missing required columns: {missing}")
    y = pd.to_numeric(training_frame[target], errors="coerce").astype("float64")
    if y.isna().any():
        raise TrainingError(f"Training target {target} contains missing or non-numeric values")
    if (config.objective in POISSON_OBJECTIVES or config.family == "poisson") and y.lt(0).any():
        raise TrainingError("Count-aware model targets must be non-negative")
    if y.lt(0).any():
        raise TrainingError("Training target must be non-negative for dengue count forecasts")
    _validate_training_dates(training_frame, frozen_period_bounds)

    preprocessor = FoldPreprocessor(model_family=config.family).fit(
        training_frame,
        feature_columns=feature_columns,
    )
    x = preprocessor.transform(training_frame.loc[:, feature_columns])
    model = _make_estimator(config)
    model.fit(x, y.to_numpy(dtype="float64"))

    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=False)
    trained = TrainedModel(
        model=model,
        preprocessor=preprocessor,
        config=config,
        feature_columns=list(feature_columns),
        target_column=target,
        metadata={},
        model_dir=model_dir,
    )
    metadata = {
        "config": config.serializable(),
        "feature_columns": list(feature_columns),
        "target_column": target,
        "preprocessing_fit_state": preprocessor.fit_state,
        "frozen_period_bounds": frozen_period_bounds or {},
        "row_identity": _row_identity(training_frame, provenance),
    }
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "config": config,
        "feature_columns": list(feature_columns),
        "target_column": target,
        "metadata": metadata,
    }
    model_path = model_dir / MODEL_FILENAME
    joblib.dump(bundle, model_path)
    metadata["model_sha256"] = sha256_file(model_path)
    (model_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    trained.metadata = metadata
    return trained


def load_champion(directory: str | Path) -> TrainedModel:
    """Load a trusted local model directory created by this module."""
    model_dir = Path(directory)
    metadata_path = model_dir / METADATA_FILENAME
    model_path = model_dir / MODEL_FILENAME
    if not metadata_path.exists() or not model_path.exists():
        raise TrainingError("Champion directory must contain metadata.json and model.joblib")
    metadata = json.loads(metadata_path.read_text())
    expected_sha = metadata.get("model_sha256")
    actual_sha = sha256_file(model_path)
    if expected_sha != actual_sha:
        raise TrainingError("Stored model hash does not match metadata")
    bundle = joblib.load(model_path)
    config = bundle["config"]
    if not isinstance(config, ModelConfig):
        config = ModelConfig(**asdict(config))
    return TrainedModel(
        model=bundle["model"],
        preprocessor=bundle["preprocessor"],
        config=config,
        feature_columns=list(bundle["feature_columns"]),
        target_column=str(bundle["target_column"]),
        metadata=metadata,
        model_dir=model_dir,
    )


def predict_next_week(features: pd.DataFrame, *, champion_dir: str | Path) -> np.ndarray:
    return load_champion(champion_dir).predict_next_week(features)


__all__ = [
    "ModelConfig",
    "TrainedModel",
    "TrainingError",
    "load_champion",
    "predict_next_week",
    "train_fold_model",
]
