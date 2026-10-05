from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_forecast.modeling.dataset import get_feature_set, load_modeling_registry

ROOT = Path(__file__).resolve().parents[1]

INTENDED_TEST_YEAR = 2025
DISTRICT_COUNT = 25
HORIZONS = (1, 2, 3, 4)
HISTORY_LAGS = ("cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4")
FEATURE_SETS = ("cases_only", "cases_rainfall", "cases_full_weather")
MILESTONE3_ARTIFACT_DIR = ROOT / "artifacts" / "milestone3"
MILESTONE3_REPORT_DIR = ROOT / "data" / "reports" / "milestone3"


class PreflightError(RuntimeError):
    """Raised when the Milestone 3 preflight cannot certify metadata-only feasibility."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise PreflightError(f"Unsupported table format: {path}")


def _required_columns(df: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise PreflightError(f"{label} missing required columns: {missing}")


def build_horizon_availability(
    dataset: pd.DataFrame, *, intended_year: int = INTENDED_TEST_YEAR
) -> pd.DataFrame:
    """Return boolean metadata masks for exact direct H1..H4 target availability."""
    required = ("district_id", "week_start_date", "dengue_cases", *HISTORY_LAGS)
    _required_columns(dataset, required, label="modeling dataset")
    df = dataset.loc[:, required].copy()
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    if df.duplicated(["district_id", "week_start_date"]).any():
        raise PreflightError("modeling dataset has duplicate district/week rows")

    origins = df[df["week_start_date"].dt.year.eq(intended_year)].copy()
    origins = origins.rename(columns={"week_start_date": "origin_date"})
    origins["origin_observed"] = origins["dengue_cases"].notna()
    origins["origin_history_available"] = origins[list(HISTORY_LAGS)].notna().all(axis=1)
    availability = origins.loc[
        :, ["district_id", "origin_date", "origin_observed", "origin_history_available"]
    ].copy()

    target_lookup = df.loc[:, ["district_id", "week_start_date", "dengue_cases"]].copy()
    for horizon in HORIZONS:
        target_date_column = f"h{horizon}_target_date"
        availability[target_date_column] = availability["origin_date"] + pd.to_timedelta(
            7 * horizon, unit="D"
        )
        target = target_lookup.rename(
            columns={
                "week_start_date": target_date_column,
                "dengue_cases": f"h{horizon}_target_observed",
            }
        )
        availability = availability.merge(
            target,
            on=["district_id", target_date_column],
            how="left",
            validate="m:1",
        )
        availability[f"h{horizon}_available"] = availability[f"h{horizon}_target_observed"].notna()
        availability = availability.drop(columns=[f"h{horizon}_target_observed"])

    horizon_columns = [f"h{horizon}_available" for horizon in HORIZONS]
    availability["all_targets_available"] = availability[horizon_columns].all(axis=1)
    availability["common_cohort"] = (
        availability["origin_observed"]
        & availability["origin_history_available"]
        & availability["all_targets_available"]
    )
    return availability.sort_values(["district_id", "origin_date"]).reset_index(drop=True)


def summarize_coverage(availability: pd.DataFrame) -> dict[str, Any]:
    horizon_columns = [f"h{horizon}_available" for horizon in HORIZONS]
    common = availability[availability["common_cohort"]]
    return {
        "intended_test_year": INTENDED_TEST_YEAR,
        "test_case_values_exported": False,
        "origin_rows": int(len(availability)),
        "origin_week_count": int(availability["origin_date"].nunique()),
        "origin_district_count": int(availability["district_id"].nunique()),
        "all_25_districts_present_in_origins": bool(
            availability["district_id"].nunique() == DISTRICT_COUNT
        ),
        "common_cohort_rows": int(len(common)),
        "common_cohort_origin_week_count": int(common["origin_date"].nunique()),
        "common_cohort_district_count": int(common["district_id"].nunique()),
        "all_25_districts_present_in_common_cohort": bool(
            common["district_id"].nunique() == DISTRICT_COUNT
        ),
        "origin_history_available_rows": int(availability["origin_history_available"].sum()),
        "all_targets_available_rows": int(availability["all_targets_available"].sum()),
        "per_horizon_available_rows": {
            f"H{horizon}": int(availability[f"h{horizon}_available"].sum())
            for horizon in HORIZONS
        },
        "per_horizon_available_origin_weeks": {
            f"H{horizon}": int(
                availability.loc[availability[f"h{horizon}_available"], "origin_date"].nunique()
            )
            for horizon in HORIZONS
        },
        "per_horizon_available_districts": {
            f"H{horizon}": int(
                availability.loc[availability[f"h{horizon}_available"], "district_id"].nunique()
            )
            for horizon in HORIZONS
        },
        "common_cohort_rows_by_district": {
            str(district): int(count)
            for district, count in common.groupby("district_id").size().sort_index().items()
        },
        "availability_mask_columns": horizon_columns + ["common_cohort"],
    }


def _date_columns(df: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in {
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "origin_date",
            "week_start_date",
            "target_date",
        } or lowered.endswith("_date"):
            names.append(str(column))
    return names


def _parse_dates_fail_closed(series: pd.Series, *, path: Path, column: str) -> pd.Series:
    nonempty = series.dropna()
    nonempty = nonempty[nonempty.astype(str).str.strip().ne("")]
    parsed = pd.to_datetime(nonempty, errors="coerce", utc=False)
    bad = nonempty[parsed.isna()]
    if not bad.empty:
        raise PreflightError(
            f"Unparseable date in {path} column {column}; failing closed before 2025 audit"
        )
    return parsed


def _summarize_dates(
    df: pd.DataFrame, path: Path, *, date_columns: list[str], cutoff_year: int
) -> dict[str, Any]:
    if not date_columns:
        raise PreflightError(f"No auditable date columns found in {path}")
    max_dates: dict[str, str | None] = {}
    min_dates: dict[str, str | None] = {}
    row_hits: list[dict[str, Any]] = []
    for column in date_columns:
        parsed = _parse_dates_fail_closed(df[column], path=path, column=column)
        if parsed.empty:
            max_dates[column] = None
            min_dates[column] = None
            continue
        max_dates[column] = str(parsed.max().date())
        min_dates[column] = str(parsed.min().date())
        violating = parsed[parsed.dt.year.ge(cutoff_year)]
        if not violating.empty:
            row_hits.append(
                {
                    "column": column,
                    "first_date_at_or_after_cutoff": str(violating.min().date()),
                    "rows": int(len(violating)),
                }
            )
    if row_hits:
        raise PreflightError(
            f"{path} contains dates at or after {cutoff_year}: {row_hits}"
        )
    all_max = [value for value in max_dates.values() if value is not None]
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path) if path.exists() else None,
        "rows": int(len(df)),
        "date_columns": date_columns,
        "min_dates": min_dates,
        "max_dates": max_dates,
        "max_date": max(all_max) if all_max else None,
        "cutoff_year": cutoff_year,
        "dates_at_or_after_cutoff": 0,
    }


def audit_registry_periods(path: Path, *, cutoff_year: int = INTENDED_TEST_YEAR) -> dict[str, Any]:
    df = _read_table(path)
    required = ["train_start", "train_end", "validation_start", "validation_end"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise PreflightError(f"M2 registry missing required period columns {missing}: {path}")
    return _summarize_dates(df, path, date_columns=required, cutoff_year=cutoff_year)


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return list(pd.read_parquet(path).columns)


def audit_prediction_dates(path: Path, *, cutoff_year: int = INTENDED_TEST_YEAR) -> dict[str, Any]:
    dates = _date_columns(pd.DataFrame(columns=_parquet_columns(path)))
    if not dates:
        raise PreflightError(f"No auditable date columns found in prediction file {path}")
    df = pd.read_parquet(path, columns=dates)
    return _summarize_dates(df, path, date_columns=dates, cutoff_year=cutoff_year)


def audit_sqlite_dates(path: Path, *, cutoff_year: int = INTENDED_TEST_YEAR) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    with sqlite3.connect(path) as connection:
        table_names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        for table in table_names:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
            dates = _date_columns(df)
            if dates:
                tables.append(
                    _summarize_dates(
                        df,
                        Path(f"{path.as_posix()}::{table}"),
                        date_columns=dates,
                        cutoff_year=cutoff_year,
                    )
                )
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "tables_with_date_columns": tables,
        "table_count": len(tables),
    }


def audit_tuning_trials(path: Path, *, cutoff_year: int = INTENDED_TEST_YEAR) -> dict[str, Any]:
    df = _read_table(path)
    dates = _date_columns(df)
    summary = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "date_columns": dates,
        "date_evidence": None,
    }
    if dates:
        summary["date_evidence"] = _summarize_dates(
            df, path, date_columns=dates, cutoff_year=cutoff_year
        )
    return summary


def audit_tuning_training_periods(
    root: Path, *, cutoff_year: int = INTENDED_TEST_YEAR
) -> dict[str, Any]:
    paths = sorted(root.glob("tune_*/training_period.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        rows.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "train_start": payload.get("train_start"),
                "train_end": payload.get("train_end"),
                "validation_start": payload.get("validation_start"),
                "validation_end": payload.get("validation_end"),
            }
        )
    if not rows:
        return {"paths": 0, "date_evidence": None}
    df = pd.DataFrame(rows)
    evidence = _summarize_dates(
        df,
        root / "tune_*/training_period.json",
        date_columns=["train_start", "train_end", "validation_start", "validation_end"],
        cutoff_year=cutoff_year,
    )
    return {"paths": len(paths), "date_evidence": evidence}


def find_m2_prediction_files(artifacts_root: Path) -> list[Path]:
    excluded_parts = {"milestone3"}
    return [
        path
        for path in sorted(artifacts_root.rglob("*predictions*.parquet"))
        if excluded_parts.isdisjoint(path.relative_to(artifacts_root).parts)
    ]


def summarize_feature_sets(registry_path: Path, dataset: pd.DataFrame) -> dict[str, Any]:
    registry = load_modeling_registry(registry_path)
    out: dict[str, Any] = {
        "registry_path": registry_path.as_posix(),
        "registry_sha256": sha256_file(registry_path),
        "feature_sets": {},
    }
    origin_rows = dataset[pd.to_datetime(dataset["week_start_date"]).dt.year.eq(INTENDED_TEST_YEAR)]
    for name in FEATURE_SETS:
        features = get_feature_set(name, registry)
        missing_features = [feature for feature in features if feature not in origin_rows.columns]
        present = [feature for feature in features if feature in origin_rows.columns]
        missing_counts = origin_rows[present].isna().sum().astype(int).to_dict() if present else {}
        out["feature_sets"][name] = {
            "features": features,
            "feature_count": len(features),
            "missing_features_in_dataset": missing_features,
            "rows_with_any_missing_feature": int(origin_rows[present].isna().any(axis=1).sum())
            if present
            else 0,
            "missing_cell_count": int(origin_rows[present].isna().sum().sum()) if present else 0,
            "missing_counts_by_feature": {str(k): int(v) for k, v in missing_counts.items()},
        }
    return out


def run_preflight(
    *,
    dataset_path: Path,
    registry_path: Path,
    experiments_registry_path: Path,
    artifacts_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    dataset = pd.read_parquet(dataset_path)
    dataset["week_start_date"] = pd.to_datetime(dataset["week_start_date"])
    availability = build_horizon_availability(dataset, intended_year=INTENDED_TEST_YEAR)
    prediction_files = find_m2_prediction_files(artifacts_root)
    locked_test_root = artifacts_root / "locked_test"
    locked_test_predictions = [
        audit_prediction_dates(path, cutoff_year=INTENDED_TEST_YEAR)
        for path in sorted((artifacts_root / "locked_test").glob("*.parquet"))
    ]
    sqlite_evidence = [
        audit_sqlite_dates(path, cutoff_year=INTENDED_TEST_YEAR)
        for path in sorted((artifacts_root / "tuning").glob("*.sqlite"))
        + sorted((artifacts_root / "tuning").glob("*.db"))
    ]
    report = {
        "audit_timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "M3 feasibility/provenance metadata-only audit for intended 2025 origin test",
        "test_values_opened": False,
        "test_case_values_printed_or_exported": False,
        "no_model_training_or_prediction_scoring": True,
        "source_hashes": {
            "dataset": {"path": dataset_path.as_posix(), "sha256": sha256_file(dataset_path)},
            "feature_registry": {
                "path": registry_path.as_posix(),
                "sha256": sha256_file(registry_path),
            },
            "protected_m1_m2_before": {
                "path": (ROOT / "docs" / "protected-m1-m2-before.json").as_posix(),
                "sha256": sha256_file(ROOT / "docs" / "protected-m1-m2-before.json"),
            },
        },
        "feature_sets": summarize_feature_sets(registry_path, dataset),
        "coverage": summarize_coverage(availability),
        "provenance_audit": {
            "m2_experiment_registry": audit_registry_periods(
                experiments_registry_path, cutoff_year=INTENDED_TEST_YEAR
            ),
            "m2_prediction_origin_dates": [
                audit_prediction_dates(path, cutoff_year=INTENDED_TEST_YEAR)
                for path in prediction_files
                if locked_test_root not in path.parents
            ],
            "m2_locked_2024_predictions_historical_observed_not_rescored": locked_test_predictions,
            "m2_tuning_trials": [
                audit_tuning_trials(path, cutoff_year=INTENDED_TEST_YEAR)
                for path in sorted((artifacts_root / "tuning").glob("trials.*"))
                if path.suffix in {".csv", ".parquet"}
            ],
            "m2_tuning_sqlite": sqlite_evidence,
            "m2_tuning_training_period_files": audit_tuning_training_periods(
                artifacts_root / "models", cutoff_year=INTENDED_TEST_YEAR
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, default=_json_default, sort_keys=True) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "processed" / "ml_training_dataset.parquet",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "data" / "reports" / "feature_registry.csv",
    )
    parser.add_argument(
        "--experiments-registry",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "registry.parquet",
    )
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--output",
        type=Path,
        default=MILESTONE3_ARTIFACT_DIR / "preflight_metadata_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_preflight(
        dataset_path=args.dataset,
        registry_path=args.registry,
        experiments_registry_path=args.experiments_registry,
        artifacts_root=args.artifacts_root,
        output_path=args.output,
    )
    coverage = report["coverage"]
    print(
        "M3 preflight metadata audit complete: "
        f"origins={coverage['origin_rows']}, "
        f"common_cohort_rows={coverage['common_cohort_rows']}, "
        f"all_25_common={coverage['all_25_districts_present_in_common_cohort']}, "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
