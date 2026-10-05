from __future__ import annotations

# ruff: noqa: E501, I001

import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from dengue_forecast.config import DISTRICTS
from dengue_forecast.contracts import (
    ContractError,
    get_target_column,
    get_training_columns,
    validate_dengue_weekly,
    validate_district_reference,
    validate_weather_weekly,
)
from dengue_forecast.pipeline import PipelinePaths, _read_cache_metadata, _sha256

REQUIRED_PROCESSED = {
    "dengue": "dengue_cases_weekly.parquet",
    "weather": "district_weather_weekly.parquet",
    "reference": "district_reference.parquet",
    "ml": "ml_training_dataset.parquet",
    "geojson": "sri_lanka_districts.geojson",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
    raise ContractError(f"Invalid registry boolean value: {value!r}")


def _read_processed(
    paths: PipelinePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dengue = pd.read_parquet(paths.processed / REQUIRED_PROCESSED["dengue"])
    weather = pd.read_parquet(paths.processed / REQUIRED_PROCESSED["weather"])
    reference = pd.read_parquet(paths.processed / REQUIRED_PROCESSED["reference"])
    dataset = pd.read_parquet(paths.processed / REQUIRED_PROCESSED["ml"])
    return dengue, weather, reference, dataset


def _source_inventory(paths: PipelinePaths) -> dict[str, Any]:
    raw_records = []
    for raw_dir in [
        paths.raw / "dengue",
        paths.raw / "weather",
        paths.raw / "geography",
        paths.raw / "population",
    ]:
        if not raw_dir.exists():
            continue
        raw_records.extend(_read_cache_metadata(raw_dir))
    processed = []
    for path in sorted(paths.processed.glob("*")):
        if path.is_file():
            processed.append(
                {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            )
    return {
        "raw_sources": [_normalize_raw_source_inventory_record(record) for record in raw_records],
        "processed_artifacts": processed,
        "source_notes": {
            "dengue": "Sri Lanka Epidemiology Unit WER weekly suspected dengue tables; NDCU indexed only because Monday-Sunday NaDSys weeks are incompatible with WER Saturday-Friday semantics.",
            "crosswalk": "Official NDCU 2026 Week 37 states Ampara district is divided into Ampara and Kalmunai RDHS; canonical Ampara sums both when present.",
            "rainfall": "CHIRPS v3.0 final retrospective rainfall. This is a retrospective baseline source, not an operational as-of forecast source.",
            "climate": "ERA5 via Open-Meteo historical archive. Open-Meteo free API is non-commercial with CCBY4.0 attribution and documented rate limits.",
            "population": "DCS Census 2024 reference population is retained for context; 2024-derived incidence/population fields are excluded from baseline training features.",
        },
    }


def _normalize_raw_source_inventory_record(record: dict[str, Any]) -> dict[str, Any]:
    request_url = record.get("request_url")
    request_params = record.get("request_params")
    if request_url and isinstance(request_params, dict):
        source_url = f"{request_url}?{urlencode(request_params, doseq=True)}"
    else:
        source_url = record.get("source_url") or record.get("upstream_url") or request_url

    return {
        "path": record["_path"],
        "metadata_path": record["_metadata_path"],
        "source_name": record.get("source_name") or record.get("source"),
        "source_url": source_url,
        "upstream_url": record.get("upstream_url"),
        "request_url": request_url,
        "request_params": request_params,
        "sha256": record.get("sha256"),
        "bytes": record.get("byte_length", record.get("content_length")),
        "retrieved_at": record.get("retrieved_at"),
        "media_type": record.get("media_type"),
        "source_type": record.get("source_type"),
        "source_version": record.get("source_version"),
        "licenses": record.get("licenses", record.get("license")),
        "period": record.get("period", {}),
        "parser_version": record.get("parser_version"),
        "original_metadata": {
            key: value
            for key, value in record.items()
            if not key.startswith("_") and key != "request_params"
        },
    }


def _schema_report(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    report = {}
    for name, df in frames.items():
        report[name] = {
            "rows": len(df),
            "columns": [
                {
                    "name": column,
                    "dtype": str(df[column].dtype),
                    "nullable": bool(df[column].isna().any()),
                    "nulls": int(df[column].isna().sum()),
                }
                for column in df.columns
            ],
        }
    return report


def _missingness_report(dataset: pd.DataFrame, path: Path) -> None:
    rows = []
    for column in sorted(dataset.columns):
        nulls = int(dataset[column].isna().sum())
        rows.append(
            {
                "column": column,
                "rows": len(dataset),
                "missing_rows": nulls,
                "missing_pct": 100.0 * nulls / len(dataset) if len(dataset) else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _coverage_report(
    dengue: pd.DataFrame, weather: pd.DataFrame, dataset: pd.DataFrame, path: Path
) -> None:
    rows = []
    for district in DISTRICTS:
        d = dengue[dengue["district_id"] == district.district_id]
        w = weather[weather["district_id"] == district.district_id]
        m = dataset[dataset["district_id"] == district.district_id]
        rows.append(
            {
                "district_id": district.district_id,
                "district_name": district.district_name,
                "dengue_weeks": len(d),
                "observed_case_weeks": int(d["dengue_cases"].notna().sum()) if not d.empty else 0,
                "weather_weeks": len(w),
                "min_weather_temporal_coverage_pct": w["weather_temporal_coverage_pct"].min()
                if not w.empty
                else None,
                "min_weather_spatial_coverage_pct": w["weather_spatial_coverage_pct"].min()
                if not w.empty
                else None,
                "ml_rows": len(m),
                "trainable_rows": int(m["is_trainable"].sum()) if "is_trainable" in m else 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _column_dictionary(dataset: pd.DataFrame, registry: pd.DataFrame | None, path: Path) -> None:
    registry_map = {}
    if registry is not None and not registry.empty:
        registry = registry.copy()
        for column in ["eligible_for_training", "uses_future_information"]:
            if column in registry:
                registry[column] = registry[column].map(_strict_bool)
        registry_map = registry.set_index("feature_name").to_dict("index")
    rows = []
    for column in dataset.columns:
        info = registry_map.get(column, {})
        rows.append(
            {
                "column": column,
                "dtype": str(dataset[column].dtype),
                "eligible_for_training": info.get("eligible_for_training", False),
                "uses_future_information": info.get("uses_future_information", False),
                "meaning": info.get("description") or _default_column_description(column),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _default_column_description(column: str) -> str:
    if column == "cases_next_week":
        return "Primary target: same-district dengue cases in the next weekly period."
    if column.startswith("cases_lag_"):
        return "Lagged dengue case count from prior observed weekly periods."
    if column.startswith("rainfall"):
        return "Weekly or lagged CHIRPS rainfall aggregate."
    if column.startswith("temp"):
        return "Weekly or lagged ERA5 temperature aggregate."
    if column.startswith("humidity"):
        return "Weekly or lagged ERA5 relative humidity aggregate."
    return "Milestone 1 dataset column."


def _sample_rows(dataset: pd.DataFrame, path: Path) -> None:
    sample = dataset.sort_values(["week_start_date", "district_id"]).head(
        max(20, min(20, len(dataset)))
    )
    sample.to_csv(path, index=False)


def _source_crosscheck(paths: PipelinePaths, dengue: pd.DataFrame, dataset: pd.DataFrame) -> None:
    quality_path = paths.reports / "dengue_extraction_quality.csv"
    if not quality_path.exists():
        raise ContractError(f"Missing dengue extraction quality report: {quality_path}")

    accepted = dengue.copy()
    accepted["week_start_date"] = pd.to_datetime(accepted["week_start_date"]).dt.date
    accepted = accepted[
        accepted["case_status"].eq("observed")
        & accepted["source_url"].astype(str).str.startswith(("http://", "https://"))
        & accepted["source_document"].notna()
    ].copy()

    quality = pd.read_csv(quality_path)
    quality["week_start_date"] = pd.to_datetime(quality["week_start_date"]).dt.date
    join_keys = ["district_id", "week_start_date", "source_document"]
    keep = [
        "district_id",
        "week_start_date",
        "source_document",
        "component_regions",
        "component_current_week_cases",
        "component_page_bboxes",
    ]
    quality = quality[[column for column in keep if column in quality.columns]]
    joined = accepted[
        [
            "district_id",
            "district_name",
            "week_start_date",
            "week_end_date",
            "source_document",
            "source_url",
            "source_retrieved_at",
            "parser_version",
            "dengue_cases",
        ]
    ].merge(quality, on=join_keys, how="inner", validate="one_to_one")
    joined = joined.dropna(
        subset=["component_regions", "component_current_week_cases", "component_page_bboxes"]
    )
    if len(joined) != len(accepted):
        raise ContractError(
            "Dengue quality provenance does not match accepted observed cohort: "
            f"accepted={len(accepted)} matched={len(joined)}"
        )
    if len(joined) < min(20, len(dataset)):
        raise ContractError(f"Only {len(joined)} dengue provenance rows available for crosscheck")

    metadata_by_filename: dict[str, dict[str, Any]] = {}
    metadata_by_url: dict[str, dict[str, Any]] = {}
    for record in _read_cache_metadata(paths.raw / "dengue"):
        metadata_by_filename[str(record.get("filename"))] = record
        metadata_by_url[str(record.get("source_url"))] = record

    def metadata_for(row: pd.Series) -> pd.Series:
        metadata = metadata_by_filename.get(str(row["source_document"])) or metadata_by_url.get(
            str(row["source_url"])
        )
        if metadata is None:
            return pd.Series({"source_sha256": pd.NA, "source_bytes": pd.NA})
        return pd.Series(
            {
                "source_sha256": metadata.get("sha256"),
                "source_bytes": metadata.get("byte_length", metadata.get("content_length")),
            }
        )

    joined = pd.concat([joined, joined.apply(metadata_for, axis=1)], axis=1)
    joined["component_case_total"] = joined["component_current_week_cases"].map(
        _component_case_total
    )
    joined["component_cases_match_dengue_cases"] = joined["component_case_total"].eq(
        joined["dengue_cases"]
    )
    candidates = joined.sort_values(["week_start_date", "district_id", "source_document"])
    sample = candidates.sample(n=min(25, len(candidates)), random_state=20241004)
    ampara_components = candidates[
        candidates["district_id"].eq("LK-AMP")
        & candidates["component_regions"].astype(str).str.contains("Ampara|Kalmunai", regex=False)
    ]
    if not ampara_components.empty and not sample.index.isin(ampara_components.index).any():
        sample = pd.concat([sample.iloc[:-1], ampara_components.head(1)])
    sample = sample.sort_values(["week_start_date", "district_id", "source_document"])
    sample.to_csv(paths.reports / "source_crosscheck_rows.csv", index=False)


def _component_case_total(value: object) -> object:
    try:
        return sum(int(part) for part in str(value).split("|") if part != "")
    except ValueError:
        return pd.NA


def _assert_source_crosscheck(paths: PipelinePaths, dengue: pd.DataFrame) -> None:
    sample_path = paths.reports / "source_crosscheck_rows.csv"
    sample = pd.read_csv(sample_path)
    accepted_keys = set(
        zip(
            dengue.loc[dengue["case_status"].eq("observed"), "district_id"],
            pd.to_datetime(
                dengue.loc[dengue["case_status"].eq("observed"), "week_start_date"]
            ).dt.date,
            dengue.loc[dengue["case_status"].eq("observed"), "source_document"],
            strict=True,
        )
    )
    sample_keys = set(
        zip(
            sample["district_id"],
            pd.to_datetime(sample["week_start_date"]).dt.date,
            sample["source_document"],
            strict=True,
        )
    )
    if not sample_keys <= accepted_keys:
        raise ContractError("Source crosscheck sample includes rows outside accepted cohort")
    if len(sample) < min(20, len(accepted_keys)):
        raise ContractError("Source crosscheck sample has fewer rows than required")
    if sample["source_sha256"].isna().any():
        raise ContractError("Source crosscheck sample is missing raw source SHA256 provenance")


def _missing_weeks_by_district(
    dengue: pd.DataFrame, dataset: pd.DataFrame
) -> list[dict[str, object]]:
    rows = []
    dengue_keys = set(
        zip(dengue["district_id"], pd.to_datetime(dengue["week_start_date"]).dt.date, strict=True)
    )
    for district in DISTRICTS:
        frame = dataset[dataset["district_id"].eq(district.district_id)]
        missing = [
            str(pd.Timestamp(value).date())
            for value in frame["week_start_date"]
            if (district.district_id, pd.Timestamp(value).date()) not in dengue_keys
        ]
        rows.append(
            {
                "district_id": district.district_id,
                "district_name": district.district_name,
                "missing_case_weeks": len(missing),
                "missing_week_start_dates": missing,
            }
        )
    return rows


def _weather_summary(weather: pd.DataFrame) -> dict[str, object]:
    variables = ["rainfall_sum_mm", "temp_mean_c", "temp_min_c", "temp_max_c", "humidity_mean_pct"]
    ranges = {}
    for variable in variables:
        ranges[variable] = {
            "non_null_rows": int(weather[variable].notna().sum()) if variable in weather else 0,
            "min": float(weather[variable].min())
            if variable in weather and weather[variable].notna().any()
            else None,
            "max": float(weather[variable].max())
            if variable in weather and weather[variable].notna().any()
            else None,
        }
    monthly = pd.DataFrame()
    if not weather.empty:
        temp = weather.copy()
        temp["month"] = pd.to_datetime(temp["week_start_date"]).dt.month
        monthly = temp.groupby("month", dropna=False)[variables].agg(["mean", "min", "max"])
    return {
        "ranges": ranges,
        "low_temporal_rows": int(weather["weather_temporal_coverage_pct"].fillna(-1).lt(85).sum()),
        "low_spatial_rows": int(weather["weather_spatial_coverage_pct"].fillna(-1).lt(90).sum()),
        "monthly_summary": json.loads(monthly.to_json(default_handler=str))
        if not monthly.empty
        else {},
    }


def _data_quality(
    dengue: pd.DataFrame, weather: pd.DataFrame, reference: pd.DataFrame, dataset: pd.DataFrame
) -> dict[str, Any]:
    quality: dict[str, Any] = {
        "dengue_rows": len(dengue),
        "weather_rows": len(weather),
        "district_reference_rows": len(reference),
        "ml_rows": len(dataset),
        "districts_in_reference": int(reference["district_id"].nunique()),
        "districts_in_dengue": int(dengue["district_id"].nunique()),
        "districts_in_ml": int(dataset["district_id"].nunique()),
        "date_range": {
            "min_week_start_date": str(pd.to_datetime(dataset["week_start_date"]).min().date())
            if len(dataset)
            else None,
            "max_week_start_date": str(pd.to_datetime(dataset["week_start_date"]).max().date())
            if len(dataset)
            else None,
        },
        "trainable_rows": int(dataset["is_trainable"].sum()) if "is_trainable" in dataset else 0,
        "row_quality_breakdown": dict(
            Counter(dataset.get("row_quality_flag", pd.Series(dtype=str)).fillna("UNKNOWN"))
        ),
        "missing_weeks_by_district": _missing_weeks_by_district(dengue, dataset),
        "duplicate_counts": {
            "dengue": int(dengue.duplicated(["district_id", "week_start_date"]).sum()),
            "weather": int(weather.duplicated(["district_id", "week_start_date"]).sum()),
            "ml_dataset": int(dataset.duplicated(["district_id", "week_start_date"]).sum()),
        },
        "weather": _weather_summary(weather),
        "licenses_and_limits": {
            "open_meteo": "Free API non-commercial only; CCBY4.0 attribution; <600 calls/min, <5000/hour, <10000/day.",
            "chirps": "CHIRPS v3.0 final retrospective gridded rainfall.",
            "geoboundaries": "geoBoundaries gbOpen: OpenStreetMap/Wambacher 2017, ODbL-1.0.",
        },
        "retrospective_vintage_assumptions": [
            "CHIRPS final and ERA5 reanalysis are retrospective baseline inputs.",
            "Late WER report availability is not modeled as an operational as-of backtest in Milestone 1.",
            "DCS 2024 population context is not eligible as a baseline training feature.",
        ],
    }
    return quality


def build_quality_reports(paths: PipelinePaths) -> None:
    dengue, weather, reference, dataset = _read_processed(paths)
    registry_path = paths.reports / "feature_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else None
    frames = {
        "dengue": dengue,
        "weather": weather,
        "district_reference": reference,
        "ml_dataset": dataset,
    }

    _write_json(paths.reports / "source_inventory.json", _source_inventory(paths))
    _write_json(paths.reports / "schema_report.json", _schema_report(frames))
    _write_json(
        paths.reports / "data_quality_report.json",
        _data_quality(dengue, weather, reference, dataset),
    )
    _missingness_report(dataset, paths.reports / "missingness_report.csv")
    _coverage_report(dengue, weather, dataset, paths.reports / "coverage_report.csv")
    _column_dictionary(dataset, registry, paths.reports / "column_dictionary.csv")
    _sample_rows(dataset, paths.reports / "sample_rows_for_manual_review.csv")
    _source_crosscheck(paths, dengue, dataset)
    summary = _summary_markdown(dengue, weather, reference, dataset)
    (paths.reports / "milestone_1_summary.md").write_text(summary, encoding="utf-8")


def _summary_markdown(
    dengue: pd.DataFrame, weather: pd.DataFrame, reference: pd.DataFrame, dataset: pd.DataFrame
) -> str:
    min_date = pd.to_datetime(dataset["week_start_date"]).min().date() if len(dataset) else "NA"
    max_date = pd.to_datetime(dataset["week_start_date"]).max().date() if len(dataset) else "NA"
    trainable = int(dataset["is_trainable"].sum()) if "is_trainable" in dataset else 0
    return f"""# Milestone 1 Summary

Date range: {min_date} to {max_date}

- Dengue rows: {len(dengue)}
- Weather rows: {len(weather)}
- District reference rows: {len(reference)}
- ML dataset rows: {len(dataset)}
- Trainable rows: {trainable}
- District coverage: {reference["district_id"].nunique()} of 25 districts

Sources: WER dengue surveillance, geoBoundaries ADM2, DCS CPH2024 population, CHIRPS rainfall, and ERA5/Open-Meteo climate when cached daily weather inputs are available.

Limitations: CHIRPS final and ERA5 are retrospective baseline sources. NDCU Monday-Sunday NaDSys weekly reports are indexed but not merged into the WER Saturday-Friday canonical series. DCS 2024 population-derived fields are context only and excluded from baseline training.

Repro commands:

```bash
.venv/bin/python -m dengue_forecast.cli pipeline run-all --offline --start-year 2024 --end-year 2024
.venv/bin/python -m dengue_forecast.cli validate all
.venv/bin/python scripts/rebuild_check.py
```
"""


def _assert_seven_day_intervals(df: pd.DataFrame, name: str) -> None:
    starts = pd.to_datetime(df["week_start_date"])
    ends = pd.to_datetime(df["week_end_date"])
    if not ((ends - starts).dt.days == 6).all():
        raise ContractError(f"{name} contains non-7-day intervals")


def _assert_next_week_target(dataset: pd.DataFrame) -> None:
    target = get_target_column()
    frame = dataset.copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"]).dt.date
    lookup = frame.set_index(["district_id", "week_start_date"])["dengue_cases"]
    for district_id, group in frame.sort_values(["district_id", "week_start_date"]).groupby(
        "district_id"
    ):
        max_start = group["week_start_date"].max()
        for row in group.itertuples():
            expected_date = row.week_start_date + timedelta(days=7)
            expected = lookup.get((district_id, expected_date), pd.NA)
            actual = getattr(row, target)
            if pd.isna(expected) and pd.isna(actual):
                continue
            if pd.isna(expected) != pd.isna(actual) or expected != actual:
                raise ContractError(
                    f"Next-week target mismatch for {district_id} {row.week_start_date}"
                )
        if pd.notna(group.loc[group["week_start_date"].eq(max_start), target].iloc[0]):
            raise ContractError(f"Last week target must be null for {district_id}")


def _assert_weather_coverage(weather: pd.DataFrame) -> None:
    bad_temporal = weather["weather_temporal_coverage_pct"].fillna(-1).lt(85)
    bad_spatial = weather["weather_spatial_coverage_pct"].fillna(-1).lt(90)
    if bad_temporal.all() or bad_spatial.all():
        raise ContractError(
            "Weather coverage below milestone thresholds: "
            f"temporal_bad={int(bad_temporal.sum())}, spatial_bad={int(bad_spatial.sum())}"
        )
    weather_vars = [
        "rainfall_sum_mm",
        "temp_mean_c",
        "temp_min_c",
        "temp_max_c",
        "humidity_mean_pct",
    ]
    all_missing = [
        name for name in weather_vars if name in weather and weather[name].notna().sum() == 0
    ]
    if all_missing:
        raise ContractError(f"Weather variables are entirely missing: {all_missing}")
    missing_districts = []
    for district in DISTRICTS:
        district_weather = weather[weather["district_id"].eq(district.district_id)]
        if district_weather.empty or district_weather[weather_vars].notna().sum().sum() == 0:
            missing_districts.append(district.district_id)
    if missing_districts:
        raise ContractError(f"Weather entirely missing for districts: {missing_districts}")


def validate_artifacts(paths: PipelinePaths) -> dict[str, Any]:
    missing = [
        filename
        for filename in REQUIRED_PROCESSED.values()
        if not (paths.processed / filename).exists()
    ]
    if missing:
        raise ContractError(f"Missing processed artifacts: {missing}")
    required_reports = [
        "source_inventory.json",
        "data_quality_report.json",
        "missingness_report.csv",
        "coverage_report.csv",
        "schema_report.json",
        "milestone_1_summary.md",
        "feature_registry.csv",
        "training_feature_columns.json",
        "column_dictionary.csv",
        "sample_rows_for_manual_review.csv",
        "source_crosscheck_rows.csv",
    ]
    missing_reports = [name for name in required_reports if not (paths.reports / name).exists()]
    if missing_reports:
        raise ContractError(f"Missing reports: {missing_reports}")

    dengue, weather, reference, dataset = _read_processed(paths)
    validate_dengue_weekly(dengue)
    validate_weather_weekly(weather)
    validate_district_reference(reference)
    _assert_seven_day_intervals(dengue, "dengue")
    _assert_seven_day_intervals(weather, "weather")
    if reference["district_id"].nunique() != 25:
        raise ContractError("District reference must contain exactly 25 districts")
    for name, frame in {"dengue": dengue, "weather": weather, "ml": dataset}.items():
        if frame.duplicated(["district_id", "week_start_date"]).any():
            raise ContractError(f"{name} contains duplicate district-week rows")
    if (dengue["dengue_cases"].dropna() < 0).any():
        raise ContractError("Dengue contains negative counts")
    _assert_next_week_target(dataset)
    _assert_weather_coverage(weather)
    registry_path = paths.reports / "feature_registry.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        for column in ["eligible_for_training", "uses_future_information"]:
            registry[column] = registry[column].map(_strict_bool)
        get_training_columns(registry)
    if "is_trainable" not in dataset or int(dataset["is_trainable"].sum()) <= 0:
        raise ContractError("Validation failed: zero trainable ML rows")
    sample_path = paths.reports / "sample_rows_for_manual_review.csv"
    if len(pd.read_csv(sample_path)) < min(20, len(dataset)):
        raise ContractError("Manual review sample has fewer rows than required")
    _assert_source_crosscheck(paths, dengue)
    return {
        "status": "passed",
        "dengue_rows": len(dengue),
        "weather_rows": len(weather),
        "district_reference_rows": len(reference),
        "ml_rows": len(dataset),
        "trainable_rows": int(dataset["is_trainable"].sum()),
    }
