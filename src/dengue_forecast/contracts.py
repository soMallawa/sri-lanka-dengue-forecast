from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class ContractError(ValueError):
    """Raised when a canonical data contract is violated."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    required: bool = True
    nullable: bool = True
    allowed: tuple[Any, ...] = ()
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    primary_key: tuple[str, ...]
    columns: dict[str, ColumnSpec]


def _schema_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "schemas"


def load_schema(name: str) -> SchemaSpec:
    path = _schema_dir() / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    columns = {
        col_name: ColumnSpec(
            name=col_name,
            dtype=str(spec["dtype"]),
            required=bool(spec.get("required", True)),
            nullable=bool(spec.get("nullable", True)),
            allowed=tuple(spec.get("allowed", ())),
            min=spec.get("min"),
            max=spec.get("max"),
        )
        for col_name, spec in raw["columns"].items()
    }
    return SchemaSpec(
        name=raw["name"], primary_key=tuple(raw.get("primary_key", ())), columns=columns
    )


class SchemaValidator:
    def __init__(self, schema: SchemaSpec):
        self.schema = schema

    def validate(self, df):  # type: ignore[no-untyped-def]
        import numpy as np
        import pandas as pd

        out = df.copy()
        required = [name for name, spec in self.schema.columns.items() if spec.required]
        missing = [name for name in required if name not in out.columns]
        if missing:
            raise ContractError(f"Missing required columns for {self.schema.name}: {missing}")

        for name, spec in self.schema.columns.items():
            if name not in out.columns:
                continue
            series = out[name]
            if not spec.nullable and series.isna().any():
                raise ContractError(f"{name} contains null values")

            non_null = series.dropna()
            if spec.dtype in {"integer", "float"}:
                numeric = pd.to_numeric(series, errors="coerce")
                numeric_non_null = numeric[series.notna()]
                if spec.dtype == "integer":
                    if not non_null.empty and numeric_non_null.isna().any():
                        raise ContractError(f"{name} contains non-numeric values")
                    if not non_null.empty and not np.isfinite(numeric_non_null.astype(float)).all():
                        raise ContractError(f"{name} contains non-finite values")
                    rounded = numeric_non_null.astype(float).round()
                    if (
                        not non_null.empty
                        and not np.equal(numeric_non_null.astype(float), rounded).all()
                    ):
                        raise ContractError(f"{name} must contain integer values")
                    out[name] = numeric.round().astype("Int64")
                else:
                    if not non_null.empty and numeric_non_null.isna().any():
                        raise ContractError(f"{name} contains non-numeric values")
                    if not non_null.empty and not np.isfinite(numeric_non_null.astype(float)).all():
                        raise ContractError(f"{name} contains non-finite values")
                    out[name] = numeric.astype("Float64")
                checked = out[name].dropna()
                if spec.min is not None and (checked < spec.min).any():
                    raise ContractError(f"{name} contains values below {spec.min}")
                if spec.max is not None and (checked > spec.max).any():
                    raise ContractError(f"{name} contains values above {spec.max}")
            elif non_null.empty:
                continue
            elif spec.dtype == "date":
                parsed = _normalize_date_series(series, name)
                if parsed.isna().any():
                    raise ContractError(f"{name} contains invalid dates")
                out[name] = parsed
            elif spec.dtype == "datetime":
                parsed = pd.to_datetime(non_null, errors="coerce", utc=True)
                if parsed.isna().any():
                    raise ContractError(f"{name} contains invalid datetimes")
                out[name] = pd.to_datetime(series, errors="coerce", utc=True)
            elif spec.dtype == "enum":
                invalid = sorted(set(non_null) - set(spec.allowed))
                if invalid:
                    raise ContractError(f"{name} contains invalid enum values: {invalid}")
                out[name] = series.astype("string")
            elif spec.dtype == "string":
                invalid = non_null.map(lambda value: not isinstance(value, str))
                if invalid.any():
                    raise ContractError(f"{name} contains non-string values")
                out[name] = series.astype("string")
            elif spec.dtype == "boolean":
                if not non_null.map(lambda value: isinstance(value, bool)).all():
                    raise ContractError(f"{name} must contain boolean values")
                out[name] = series.astype(bool)

        if self.schema.primary_key:
            assert_unique_key(out, self.schema.primary_key)
        return out


def _normalize_date_series(series, name: str):  # type: ignore[no-untyped-def]
    import pandas as pd

    def normalize(value):  # type: ignore[no-untyped-def]
        if pd.isna(value):
            return pd.NA
        if isinstance(value, date) and not hasattr(value, "hour"):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if len(stripped) != 10:
                raise ContractError(f"{name} must contain calendar dates, not timestamps")
            parsed = pd.to_datetime(stripped, errors="coerce")
        else:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return pd.NA
            if parsed.tzinfo is not None:
                raise ContractError(f"{name} must contain calendar dates, not timestamps")
            if parsed.time().isoformat() != "00:00:00":
                raise ContractError(f"{name} must contain calendar dates, not timestamps")
        if pd.isna(parsed):
            return pd.NA
        return parsed.date()

    return series.map(normalize)


def assert_unique_key(df, keys: Iterable[str]) -> None:  # type: ignore[no-untyped-def]
    key_list = list(keys)
    missing = [key for key in key_list if key not in df.columns]
    if missing:
        raise ContractError(f"Missing key columns: {missing}")
    duplicate_mask = df.duplicated(key_list, keep=False)
    if duplicate_mask.any():
        sample = df.loc[duplicate_mask, key_list].head(5).to_dict("records")
        raise ContractError(f"Duplicate key for {key_list}: {sample}")


def validate_dengue_weekly(df):  # type: ignore[no-untyped-def]
    out = SchemaValidator(load_schema("dengue_weekly")).validate(df)
    _validate_district_columns(out)
    _validate_week_intervals(out)
    if ((out["case_status"] == "missing") & out["dengue_cases"].notna()).any():
        raise ContractError("case_status=missing requires dengue_cases to be null")
    if ((out["case_status"].isin(["observed", "imputed"])) & out["dengue_cases"].isna()).any():
        raise ContractError("case_status=observed/imputed requires dengue_cases to be non-null")
    return out


def validate_weather_weekly(df):  # type: ignore[no-untyped-def]
    out = SchemaValidator(load_schema("weather_weekly")).validate(df)
    _validate_district_columns(out)
    _validate_week_intervals(out)
    if "weather_quality_flag" in out.columns:
        out["weather_quality_flag"] = out["weather_quality_flag"].map(
            normalize_weather_quality_flag
        )
    return out


def normalize_weather_quality_flag(value: object) -> str | None:
    import pandas as pd

    if pd.isna(value):
        return None
    if not isinstance(value, str):
        raise ContractError("weather_quality_flag must be a string or null")
    raw_tokens = [token.strip().casefold() for token in value.split(";") if token.strip()]
    if not raw_tokens:
        return None
    if raw_tokens == ["ok"]:
        return "OK"
    allowed = {
        "low_temporal_coverage": "LOW_TEMPORAL_COVERAGE",
        "low_spatial_coverage": "LOW_SPATIAL_COVERAGE",
        "rain_missing": "RAIN_MISSING",
        "climate_missing": "CLIMATE_MISSING",
        "low_weather_coverage": "LOW_WEATHER_COVERAGE",
    }
    unknown = [token for token in raw_tokens if token not in allowed]
    if unknown:
        raise ContractError(f"Unrecognized weather_quality_flag token(s): {unknown}")
    return ";".join(allowed[token] for token in raw_tokens)


def validate_district_reference(df):  # type: ignore[no-untyped-def]
    validated = SchemaValidator(load_schema("district_reference")).validate(df)
    _validate_district_columns(validated)
    if "district_id" in validated.columns and validated["district_id"].nunique(dropna=True) != len(
        validated
    ):
        raise ContractError("District reference must contain one row per district_id")
    return validated


def validate_ml_dataset(df):  # type: ignore[no-untyped-def]
    out = SchemaValidator(load_schema("ml_dataset")).validate(df)
    _validate_district_columns(out)
    _validate_week_intervals(out)
    return out


def _validate_district_columns(df) -> None:  # type: ignore[no-untyped-def]
    if "district_id" not in df.columns or "district_name" not in df.columns:
        return
    from dengue_forecast.config import DISTRICT_BY_ID

    unknown = sorted(set(df["district_id"].dropna()) - set(DISTRICT_BY_ID))
    if unknown:
        raise ContractError(f"Unknown district_id values: {unknown}")
    expected = df["district_id"].map(
        lambda value: DISTRICT_BY_ID[str(value)].district_name if value in DISTRICT_BY_ID else None
    )
    mismatch = (
        df["district_name"].notna()
        & expected.notna()
        & (df["district_name"].astype(str) != expected)
    )
    if mismatch.any():
        sample = df.loc[mismatch, ["district_id", "district_name"]].head(5).to_dict("records")
        raise ContractError(f"district_name does not match district_id: {sample}")


def _validate_week_intervals(df) -> None:  # type: ignore[no-untyped-def]
    if "week_start_date" not in df.columns or "week_end_date" not in df.columns:
        return
    import pandas as pd

    starts = pd.to_datetime(df["week_start_date"])
    ends = pd.to_datetime(df["week_end_date"])
    invalid = ((ends - starts).dt.days != 6) | starts.isna() | ends.isna()
    if invalid.any():
        sample = df.loc[invalid, ["week_start_date", "week_end_date"]].head(5).to_dict("records")
        raise ContractError(f"Week intervals must be exactly 7 days inclusive: {sample}")


def get_target_column() -> str:
    return "cases_next_week"


def get_training_columns(registry) -> list[str]:  # type: ignore[no-untyped-def]
    required = {"feature_name", "uses_future_information", "eligible_for_training"}
    missing = required - set(registry.columns)
    if missing:
        raise ContractError(f"Feature registry missing columns: {sorted(missing)}")

    for column in ["uses_future_information", "eligible_for_training"]:
        invalid = registry[column].dropna().map(lambda value: not isinstance(value, bool))
        if invalid.any():
            raise ContractError(f"Feature registry column {column} must contain boolean values")

    selected = registry[registry["eligible_for_training"]].copy()
    future = selected[selected["uses_future_information"]]["feature_name"].tolist()
    if future:
        raise ContractError(f"Training columns include future information: {future}")
    target_like = [name for name in selected["feature_name"] if str(name).startswith("cases_next")]
    if target_like:
        raise ContractError(f"Target columns are not eligible training features: {target_like}")
    return selected["feature_name"].astype(str).tolist()
