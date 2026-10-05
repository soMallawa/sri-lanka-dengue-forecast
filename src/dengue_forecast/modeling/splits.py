from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dengue_forecast.contracts import ContractError
from dengue_forecast.modeling.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REGISTRY_PATH,
    TARGET_COLUMN,
    load_modeling_dataset,
    load_modeling_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "modeling.yaml"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "docs" / "modeling-feature-policy.md"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "split_definition.json"
DEFAULT_READINESS_REPORT = PROJECT_ROOT / "data" / "reports" / "modeling_readiness_report.md"
DEFAULT_SPLIT_REPORT = PROJECT_ROOT / "data" / "reports" / "temporal_split_report.md"


@dataclass(frozen=True)
class SplitPolicy:
    min_training_years: int = 4
    min_validation_rows: int = 300
    min_validation_districts: int = 20
    min_target_coverage_pct: float = 60.0
    embargo_weeks: int = 1
    horizon_weeks: int = 1
    preferred_holdout_year: int = 2024
    candidate_holdout_years: tuple[int, ...] = (2023, 2024, 2025)
    min_holdout_rows: int = 300
    min_holdout_districts: int = 20
    min_holdout_target_coverage_pct: float = 60.0


def load_split_policy(path: str | Path = DEFAULT_CONFIG_PATH) -> SplitPolicy:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    splits = raw.get("splits", {})
    holdout = raw.get("holdout", {})
    return SplitPolicy(
        min_training_years=int(splits.get("min_training_years", 4)),
        min_validation_rows=int(splits.get("min_validation_rows", 300)),
        min_validation_districts=int(splits.get("min_validation_districts", 20)),
        min_target_coverage_pct=float(splits.get("min_target_coverage_pct", 60.0)),
        embargo_weeks=int(splits.get("embargo_weeks", 1)),
        horizon_weeks=int(splits.get("horizon_weeks", 1)),
        preferred_holdout_year=int(holdout.get("preferred_year", 2024)),
        candidate_holdout_years=tuple(
            int(year) for year in holdout.get("candidate_years", [2023, 2024, 2025])
        ),
        min_holdout_rows=int(holdout.get("min_rows", splits.get("min_validation_rows", 300))),
        min_holdout_districts=int(
            holdout.get("min_districts", splits.get("min_validation_districts", 20))
        ),
        min_holdout_target_coverage_pct=float(
            holdout.get("min_target_coverage_pct", splits.get("min_target_coverage_pct", 60.0))
        ),
    )


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_hash(policy: SplitPolicy) -> str:
    payload = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _actual_year(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["week_start_date"]).dt.year


def _year_window(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[_actual_year(df).eq(year)]


def _target_end_date(df: pd.DataFrame, horizon_weeks: int) -> pd.Series:
    return pd.to_datetime(df["week_end_date"]) + pd.Timedelta(days=horizon_weeks * 7)


def _complete_calendar_rows(df: pd.DataFrame, year: int) -> int:
    return int(_actual_year(df).eq(year).sum())


def _longest_trainable_gap(group: pd.DataFrame) -> int:
    longest = 0
    current = 0
    for value in group.sort_values("week_start_date")["is_trainable"].astype(bool):
        if value:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


def _year_metrics(
    df: pd.DataFrame, year: int, *, high_threshold: float | None = None
) -> dict[str, Any]:
    year_df = _year_window(df, year).copy()
    trainable = year_df[year_df["is_trainable"] & year_df[TARGET_COLUMN].notna()]
    target_available = int(year_df[TARGET_COLUMN].notna().sum())
    complete_rows = _complete_calendar_rows(df, year)
    target_coverage = 100.0 * target_available / complete_rows if complete_rows else 0.0
    case_missing = 100.0 * year_df["case_missing_flag"].mean() if len(year_df) else 0.0
    weather_missing = 100.0 * year_df["weather_missing_flag"].mean() if len(year_df) else 0.0
    high_count = 0
    if high_threshold is not None:
        high_count = int((year_df[TARGET_COLUMN].dropna() >= high_threshold).sum())
    return {
        "year": year,
        "complete_rows": int(complete_rows),
        "trainable_rows": int(len(trainable)),
        "target_available_rows": target_available,
        "target_coverage_pct": round(target_coverage, 2),
        "districts": int(year_df["district_id"].nunique()),
        "trainable_districts": int(trainable["district_id"].nunique()),
        "seasonal_span": _seasonal_span(year_df),
        "case_missing_pct": round(case_missing, 2),
        "weather_missing_pct": round(weather_missing, 2),
        "longest_trainable_gap_weeks": int(
            year_df.groupby("district_id").apply(_longest_trainable_gap, include_groups=False).max()
            if len(year_df)
            else 0
        ),
        "high_incidence_rows": high_count,
    }


def _seasonal_span(df: pd.DataFrame) -> str:
    if df.empty:
        return "none"
    return f"{df['week_start_date'].min().date()} to {df['week_end_date'].max().date()}"


def readiness_summary(df: pd.DataFrame) -> dict[str, Any]:
    actual_year = _actual_year(df)
    pre_2023 = df[(actual_year < 2023) & df[TARGET_COLUMN].notna()][TARGET_COLUMN]
    high_threshold = float(pre_2023.quantile(0.9)) if not pre_2023.empty else None
    years = sorted(int(year) for year in actual_year.dropna().unique())
    return {
        "rows": int(len(df)),
        "districts": int(df["district_id"].nunique()),
        "date_start": df["week_start_date"].min().date().isoformat(),
        "date_end": df["week_end_date"].max().date().isoformat(),
        "unique_district_week": bool(not df.duplicated(["district_id", "week_start_date"]).any()),
        "target_column": TARGET_COLUMN,
        "high_incidence_threshold_pre_2023": high_threshold,
        "years": [_year_metrics(df, year, high_threshold=high_threshold) for year in years],
        "trainable_by_district": _trainable_by_district(df),
    }


def _trainable_by_district(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for district_id, group in df.groupby("district_id"):
        rows.append(
            {
                "district_id": district_id,
                "complete_rows": int(len(group)),
                "trainable_rows": int((group["is_trainable"] & group[TARGET_COLUMN].notna()).sum()),
                "target_available_rows": int(group[TARGET_COLUMN].notna().sum()),
                "longest_trainable_gap_weeks": int(_longest_trainable_gap(group)),
            }
        )
    return rows


def _holdout_audit(df: pd.DataFrame, policy: SplitPolicy) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for year in policy.candidate_holdout_years:
        candidate = _year_window(df, year)
        candidate_start = candidate["week_start_date"].min() if not candidate.empty else pd.NaT
        target_end = _target_end_date(df, policy.horizon_weeks)
        previous_mask = (
            target_end.lt(candidate_start) & df[TARGET_COLUMN].notna()
            if not pd.isna(candidate_start)
            else pd.Series(False, index=df.index)
        )
        previous = df[previous_mask][TARGET_COLUMN]
        threshold = float(previous.quantile(0.9)) if not previous.empty else None
        metrics = _year_metrics(df, year, high_threshold=threshold)
        metrics["high_incidence_threshold_source"] = (
            f"target intervals ending before actual {year} holdout start"
        )
        metrics["high_incidence_threshold"] = threshold
        metrics["adequate"] = _candidate_adequate(metrics, policy)
        metrics["rejection_reason"] = _candidate_rejection_reason(metrics, policy)
        audits.append(metrics)
    return audits


def _candidate_adequate(metrics: dict[str, Any], policy: SplitPolicy) -> bool:
    return (
        metrics["trainable_rows"] >= policy.min_holdout_rows
        and metrics["trainable_districts"] >= policy.min_holdout_districts
        and metrics["target_coverage_pct"] >= policy.min_holdout_target_coverage_pct
    )


def _candidate_rejection_reason(metrics: dict[str, Any], policy: SplitPolicy) -> str:
    reasons: list[str] = []
    if metrics["trainable_rows"] < policy.min_holdout_rows:
        reasons.append(f"trainable rows {metrics['trainable_rows']} < {policy.min_holdout_rows}")
    if metrics["trainable_districts"] < policy.min_holdout_districts:
        reasons.append(
            f"districts {metrics['trainable_districts']} < {policy.min_holdout_districts}"
        )
    if metrics["target_coverage_pct"] < policy.min_holdout_target_coverage_pct:
        reasons.append(
            "target coverage "
            f"{metrics['target_coverage_pct']}% < {policy.min_holdout_target_coverage_pct}%"
        )
    return "; ".join(reasons) if reasons else "accepted"


def _select_holdout(audits: list[dict[str, Any]], policy: SplitPolicy) -> int:
    by_year = {audit["year"]: audit for audit in audits}
    preferred = by_year.get(policy.preferred_holdout_year)
    if preferred and preferred["adequate"]:
        return policy.preferred_holdout_year
    adequate = [audit["year"] for audit in audits if audit["adequate"] and audit["year"] != 2025]
    if adequate:
        return max(adequate)
    raise ContractError("No adequate locked holdout year found")


def build_temporal_splits(df: pd.DataFrame, policy: SplitPolicy | None = None) -> dict[str, Any]:
    policy = policy or SplitPolicy()
    audits = _holdout_audit(df, policy)
    holdout_year = _select_holdout(audits, policy)
    locked = _year_window(df, holdout_year)
    locked_start = locked["week_start_date"].min()
    locked_end = locked["week_end_date"].max()
    development = df[df["week_start_date"] < locked_start]
    development_actual_year = _actual_year(development)
    years = sorted(int(year) for year in development_actual_year.dropna().unique())

    folds: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for validation_year in years:
        training_years = [year for year in years if year < validation_year]
        if len(training_years) < policy.min_training_years:
            rejected.append(
                {"year": validation_year, "reason": "fewer than four prior training years"}
            )
            continue
        validation = _year_window(development, validation_year)
        validation_target_end = _target_end_date(validation, policy.horizon_weeks)
        validation_scored = validation[validation_target_end.lt(locked_start)]
        metrics = _year_metrics(validation_scored, validation_year)
        if not _validation_year_adequate(metrics, policy):
            rejected.append(
                {
                    "year": validation_year,
                    "reason": _validation_rejection_reason(metrics, policy),
                }
            )
            continue

        validation_start = validation_scored["week_start_date"].min()
        validation_end = validation_scored["week_end_date"].max()
        validation_target_end_max = (
            validation_scored["week_end_date"].max()
            + pd.Timedelta(days=policy.horizon_weeks * 7)
        )
        train_forecast_end = validation_start - pd.Timedelta(
            days=(policy.horizon_weeks + policy.embargo_weeks) * 7 + 1
        )
        horizon_delta = pd.Timedelta(days=policy.horizon_weeks * 7)
        embargo_delta = pd.Timedelta(days=(policy.horizon_weeks + policy.embargo_weeks) * 7)
        train = development[
            (_actual_year(development).isin(training_years))
            & (development["week_end_date"] <= train_forecast_end)
            & (development["week_end_date"] + horizon_delta < validation_start)
            & (development["week_end_date"] + embargo_delta < validation_start)
        ]
        train = train[train["is_trainable"] & train[TARGET_COLUMN].notna()]
        val_rows = validation_scored[
            validation_scored["is_trainable"] & validation_scored[TARGET_COLUMN].notna()
        ]
        if train.empty:
            rejected.append(
                {"year": validation_year, "reason": "embargo leaves no eligible training rows"}
            )
            continue

        train_target_end_max = train["week_end_date"].max() + pd.Timedelta(
            days=policy.horizon_weeks * 7
        )
        train_embargo_end_max = train["week_end_date"].max() + pd.Timedelta(
            days=(policy.horizon_weeks + policy.embargo_weeks) * 7
        )
        folds.append(
            {
                "fold_id": f"val_{validation_year}",
                "train_start": train["week_start_date"].min().date().isoformat(),
                "train_end": train["week_end_date"].max().date().isoformat(),
                "train_target_end_max": train_target_end_max.date().isoformat(),
                "train_embargo_end_max": train_embargo_end_max.date().isoformat(),
                "validation_start": validation_start.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "validation_target_end_max": validation_target_end_max.date().isoformat(),
                "locked_test_start": locked_start.date().isoformat(),
                "locked_test_end": locked_end.date().isoformat(),
                "embargo_weeks": policy.embargo_weeks,
                "horizon_weeks": policy.horizon_weeks,
                "rows_train": int(len(train)),
                "rows_validation": int(len(val_rows)),
                "rows_test": int((locked["is_trainable"] & locked[TARGET_COLUMN].notna()).sum()),
                "districts_train": int(train["district_id"].nunique()),
                "districts_validation": int(val_rows["district_id"].nunique()),
                "districts_test": int(
                    locked.loc[
                        locked["is_trainable"] & locked[TARGET_COLUMN].notna(),
                        "district_id",
                    ].nunique()
                ),
                "target_coverage_pct_validation_complete_calendar": metrics["target_coverage_pct"],
                "date_inequality": (
                    f"max train embargo end {train_embargo_end_max.date().isoformat()} "
                    f"< validation start {validation_start.date().isoformat()}; "
                    f"max validation target end {validation_target_end_max.date().isoformat()} "
                    f"< locked test start {locked_start.date().isoformat()}"
                ),
            }
        )

    if not folds:
        raise ContractError("No temporal validation folds satisfied the split policy")

    return {
        "schema_version": 1,
        "policy": asdict(policy),
        "locked_holdout_year": holdout_year,
        "locked_test_start": locked_start.date().isoformat(),
        "locked_test_end": locked_end.date().isoformat(),
        "post_holdout_excluded_start": (
            df[_actual_year(df) > holdout_year]["week_start_date"].min().date().isoformat()
            if not df[_actual_year(df) > holdout_year].empty
            else None
        ),
        "post_holdout_excluded_end": (
            df[_actual_year(df) > holdout_year]["week_end_date"].max().date().isoformat()
            if not df[_actual_year(df) > holdout_year].empty
            else None
        ),
        "holdout_audit": audits,
        "rejected_validation_years": rejected,
        "folds": folds,
    }


def _validation_year_adequate(metrics: dict[str, Any], policy: SplitPolicy) -> bool:
    return (
        metrics["trainable_rows"] >= policy.min_validation_rows
        and metrics["trainable_districts"] >= policy.min_validation_districts
        and metrics["target_coverage_pct"] >= policy.min_target_coverage_pct
    )


def _validation_rejection_reason(metrics: dict[str, Any], policy: SplitPolicy) -> str:
    reasons: list[str] = []
    if metrics["trainable_rows"] < policy.min_validation_rows:
        reasons.append(
            f"eligible validation rows {metrics['trainable_rows']} "
            f"< {policy.min_validation_rows}"
        )
    if metrics["trainable_districts"] < policy.min_validation_districts:
        reasons.append(
            f"districts {metrics['trainable_districts']} "
            f"< {policy.min_validation_districts}"
        )
    if metrics["target_coverage_pct"] < policy.min_target_coverage_pct:
        reasons.append(
            f"target coverage {metrics['target_coverage_pct']}% "
            f"< {policy.min_target_coverage_pct}%"
        )
    return "; ".join(reasons)


def attach_hashes(
    split_definition: dict[str, Any],
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    out = json.loads(json.dumps(split_definition, default=_json_safe))
    out["hashes"] = {
        "dataset": _sha256_path(dataset_path),
        "registry": _sha256_path(registry_path),
        "modeling_overlay": _sha256_path(config_path),
        "feature_policy": _sha256_path(policy_path),
        "split_policy": _policy_hash(SplitPolicy(**out["policy"])),
    }
    return out


def load_or_create_split_definition(
    df: pd.DataFrame,
    policy: SplitPolicy,
    *,
    output_path: str | Path = DEFAULT_SPLIT_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    candidate = attach_hashes(
        build_temporal_splits(df, policy),
        dataset_path=dataset_path,
        registry_path=registry_path,
        config_path=config_path,
        policy_path=policy_path,
    )
    path = Path(output_path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != candidate:
            raise ContractError(
                f"Frozen split definition differs from current inputs/policy: {path}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return candidate


def write_readiness_report(
    summary: dict[str, Any], path: str | Path = DEFAULT_READINESS_REPORT
) -> None:
    lines = [
        "# Modeling Readiness Report",
        "",
        "Stage 1 audit only. No fitting, predictions, tuning, or forecast scores are reported.",
        "",
        f"- Rows: {summary['rows']}",
        f"- Districts: {summary['districts']}",
        f"- Date span: {summary['date_start']} to {summary['date_end']}",
        f"- Unique district-week key: {summary['unique_district_week']}",
        f"- Target column: `{summary['target_column']}`",
        "- High-incidence threshold source: pre-2023 development targets, "
        f"threshold={summary['high_incidence_threshold_pre_2023']}",
        "",
        "## Year Metrics",
        "",
        "| Year | Complete rows | Trainable rows | Target rows | Target coverage % | "
        "Trainable districts | Case missing % | Weather missing % | "
        "Longest trainable gap | High-incidence rows |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["years"]:
        lines.append(
            f"| {row['year']} | {row['complete_rows']} | {row['trainable_rows']} | "
            f"{row['target_available_rows']} | {row['target_coverage_pct']} | "
            f"{row['trainable_districts']} | {row['case_missing_pct']} | "
            f"{row['weather_missing_pct']} | {row['longest_trainable_gap_weeks']} | "
            f"{row['high_incidence_rows']} |"
        )
    lines.extend(
        [
            "",
            "## District Trainability",
            "",
            "| District | Complete rows | Trainable rows | Target rows | Longest trainable gap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["trainable_by_district"]:
        lines.append(
            f"| {row['district_id']} | {row['complete_rows']} | {row['trainable_rows']} | "
            f"{row['target_available_rows']} | {row['longest_trainable_gap_weeks']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_report(split: dict[str, Any], path: str | Path = DEFAULT_SPLIT_REPORT) -> None:
    lines = [
        "# Temporal Split Report",
        "",
        "Stage 1 split design only. The locked holdout is inspected for coverage, "
        "not forecast performance.",
        "",
        f"- Locked holdout year: {split['locked_holdout_year']}",
        f"- Locked test period: {split['locked_test_start']} to {split['locked_test_end']}",
        "- Post-holdout excluded from development: "
        f"{split['post_holdout_excluded_start']} to {split['post_holdout_excluded_end']}",
        f"- Embargo weeks: {split['policy']['embargo_weeks']}",
        f"- Horizon weeks: {split['policy']['horizon_weeks']}",
        "- Forecast cutoff is `week_end_date`; the target week begins the next day.",
        "- Conservative inequality: `train.week_end_date + horizon + embargo < validation_start`.",
        "",
        "## Holdout Candidate Audit",
        "",
        "| Year | Adequate | Complete rows | Trainable rows | Target coverage % | "
        "Districts | Seasonal span | Longest gap | High rows | Reason |",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in split["holdout_audit"]:
        reason = row["rejection_reason"]
        if row["year"] == split["locked_holdout_year"]:
            reason = "selected; preferred 2024 if adequate" if row["year"] == 2024 else "selected"
        lines.append(
            f"| {row['year']} | {row['adequate']} | {row['complete_rows']} | "
            f"{row['trainable_rows']} | {row['target_coverage_pct']} | "
            f"{row['trainable_districts']} | {row['seasonal_span']} | "
            f"{row['longest_trainable_gap_weeks']} | {row['high_incidence_rows']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Rejected Validation Years",
            "",
            "| Year | Reason |",
            "|---:|---|",
        ]
    )
    for row in split["rejected_validation_years"]:
        lines.append(f"| {row['year']} | {row['reason']} |")
    lines.extend(
        [
            "",
            "Validation-year rejection uses actual `week_start_date` calendar denominators "
            "and scored validation masks after locked-holdout target clipping.",
            "",
            "## Expanding Folds",
            "",
            "| Fold | Train | Validation | Train rows | Validation rows | "
            "Train districts | Validation districts | Inequality |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for fold in split["folds"]:
        lines.append(
            f"| {fold['fold_id']} | {fold['train_start']} to {fold['train_end']} | "
            f"{fold['validation_start']} to {fold['validation_end']} | {fold['rows_train']} | "
            f"{fold['rows_validation']} | {fold['districts_train']} | "
            f"{fold['districts_validation']} | {fold['date_inequality']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_stage1_artifacts(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
) -> dict[str, Any]:
    registry = load_modeling_registry(registry_path)
    df = load_modeling_dataset(dataset_path, registry=registry, production=True)
    policy = load_split_policy(config_path)
    summary = readiness_summary(df)
    split = load_or_create_split_definition(
        df,
        policy,
        output_path=split_path,
        dataset_path=dataset_path,
        registry_path=registry_path,
        config_path=config_path,
        policy_path=policy_path,
    )
    write_readiness_report(summary)
    write_split_report(split)
    return {"readiness": summary, "split": split}
