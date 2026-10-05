from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from dengue_forecast.modeling.dataset import FORBIDDEN_CONTEXT_COLUMNS, FUTURE_TARGET_COLUMNS


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class PreprocessingError(ValueError):
    """Raised when fold preprocessing would violate the modeling contract."""


ROW_KEY_COLUMNS = ("district_id", "week_start_date")
DEFAULT_CATEGORICAL_COLUMNS = ("district_id",)


def _normalise_feature_columns(feature_columns: list[str] | tuple[str, ...]) -> list[str]:
    columns = [str(column) for column in feature_columns]
    if len(columns) != len(set(columns)):
        raise PreprocessingError("Feature columns must be unique")
    forbidden = sorted(
        (set(columns) & FUTURE_TARGET_COLUMNS) | (set(columns) & FORBIDDEN_CONTEXT_COLUMNS)
    )
    if forbidden:
        raise PreprocessingError(f"Forbidden feature columns requested: {forbidden}")
    return columns


def _row_keys(frame: pd.DataFrame) -> list[dict[str, str]]:
    missing = [column for column in ROW_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise PreprocessingError(f"Training frame missing row identity columns: {missing}")
    keys = frame.loc[:, ROW_KEY_COLUMNS].copy()
    keys["week_start_date"] = pd.to_datetime(keys["week_start_date"], errors="coerce")
    if keys["week_start_date"].isna().any():
        raise PreprocessingError("week_start_date contains missing or invalid dates")
    if keys.duplicated(list(ROW_KEY_COLUMNS)).any():
        raise PreprocessingError("Training frame contains duplicate district/week row keys")
    return [
        {
            "district_id": str(row.district_id),
            "week_start_date": row.week_start_date.date().isoformat(),
        }
        for row in keys.itertuples(index=False)
    ]


def _row_key_digest(keys: list[dict[str, str]]) -> str:
    text = "\n".join(f"{row['district_id']}|{row['week_start_date']}" for row in keys)
    return _sha256_text(text)


def _numeric_series(values: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bad_mask = numeric.isna().to_numpy() & values.notna().to_numpy()
    if bad_mask.any():
        bad = values.loc[bad_mask].head(3).tolist()
        raise PreprocessingError(f"Numeric feature {column} contains non-numeric values: {bad}")
    return numeric.astype("float64")


@dataclass
class FoldPreprocessor:
    """Fold-local preprocessing fitted only on caller-provided training rows."""

    model_family: str = "ridge"
    categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS
    feature_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    medians_: dict[str, float] = field(default_factory=dict)
    dropped_all_missing_features: list[str] = field(default_factory=list)
    output_feature_columns: list[str] = field(default_factory=list)
    scaler_: StandardScaler | None = None
    fit_state: dict[str, Any] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return bool(self.fit_state)

    @property
    def scale_numeric(self) -> bool:
        return self.model_family in {"ridge", "poisson"}

    def fit(
        self,
        training_frame: pd.DataFrame,
        *,
        feature_columns: list[str] | tuple[str, ...],
    ) -> FoldPreprocessor:
        columns = _normalise_feature_columns(feature_columns)
        missing = [column for column in columns if column not in training_frame.columns]
        if missing:
            raise PreprocessingError(f"Training frame missing feature columns: {missing}")
        keys = _row_keys(training_frame)

        categorical = [column for column in columns if column in set(self.categorical_columns)]
        numeric = [column for column in columns if column not in set(categorical)]

        self.feature_columns = columns
        self.numeric_columns = []
        self.categories_ = {}
        self.medians_ = {}
        self.dropped_all_missing_features = []

        for column in numeric:
            values = _numeric_series(training_frame[column], column=column)
            if values.notna().sum() == 0:
                self.dropped_all_missing_features.append(column)
                continue
            self.numeric_columns.append(column)
            self.medians_[column] = float(values.median())

        for column in categorical:
            series = training_frame[column].astype("string")
            if series.notna().sum() == 0:
                self.dropped_all_missing_features.append(column)
                continue
            self.categories_[column] = sorted(str(value) for value in series.dropna().unique())

        unscaled = self._transform(training_frame, require_exact_columns=False)
        self.output_feature_columns = list(unscaled.columns)
        if self.scale_numeric and self.numeric_columns:
            numeric_output = self.numeric_columns
            self.scaler_ = StandardScaler()
            self.scaler_.fit(unscaled[numeric_output].to_numpy(dtype="float64"))
        else:
            self.scaler_ = None
        self.fit_state = {
            "model_family": self.model_family,
            "input_feature_columns": list(self.feature_columns),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categories_),
            "category_levels": {key: list(value) for key, value in self.categories_.items()},
            "numeric_medians": dict(self.medians_),
            "numeric_missing_indicators": [f"{column}__missing" for column in self.numeric_columns],
            "dropped_all_missing_features": list(self.dropped_all_missing_features),
            "all_missing_convention": (
                "Features entirely unobserved in the training fold are dropped and recorded; "
                "no constant-zero placeholder is emitted."
            ),
            "scale_numeric": self.scale_numeric,
            "output_feature_columns": list(self.output_feature_columns),
            "training_row_count": int(len(training_frame)),
            "training_row_keys": keys,
            "training_row_key_digest": _row_key_digest(keys),
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise PreprocessingError("FoldPreprocessor is not fitted")
        transformed = self._transform(frame, require_exact_columns=True)
        if list(transformed.columns) != self.output_feature_columns:
            raise PreprocessingError("Transformed feature column order changed")
        if self.scaler_ is not None and self.numeric_columns:
            transformed.loc[:, self.numeric_columns] = self.scaler_.transform(
                transformed[self.numeric_columns].to_numpy(dtype="float64")
            )
        return transformed

    def _transform(self, frame: pd.DataFrame, *, require_exact_columns: bool) -> pd.DataFrame:
        if require_exact_columns:
            actual = list(frame.columns)
            if actual != self.feature_columns:
                extra = [column for column in actual if column not in self.feature_columns]
                missing = [column for column in self.feature_columns if column not in actual]
                if extra:
                    raise PreprocessingError(f"Unexpected feature columns: {extra}")
                if missing:
                    raise PreprocessingError(f"Missing feature columns: {missing}")
                raise PreprocessingError("Feature column order mismatch")
        else:
            missing = [column for column in self.feature_columns if column not in frame.columns]
            if missing:
                raise PreprocessingError(f"Frame missing feature columns: {missing}")

        parts: list[pd.DataFrame] = []
        index = frame.index
        for column in self.numeric_columns:
            values = _numeric_series(frame[column], column=column)
            missing = values.isna()
            imputed = values.fillna(self.medians_[column]).astype("float64")
            parts.append(
                pd.DataFrame({column: imputed, f"{column}__missing": missing.astype("float64")})
            )

        for column, categories in self.categories_.items():
            series = frame[column].astype("string")
            encoded = {
                f"{column}__{category}": series.eq(category).fillna(False).astype("float64")
                for category in categories
            }
            parts.append(pd.DataFrame(encoded, index=index))

        if not parts:
            return pd.DataFrame(index=index)
        result = pd.concat(parts, axis=1)
        return result.astype("float64")


__all__ = ["FoldPreprocessor", "PreprocessingError"]
