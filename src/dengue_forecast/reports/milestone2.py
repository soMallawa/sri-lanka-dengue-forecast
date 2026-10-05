# ruff: noqa: E501

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_forecast.modeling.baselines import BASELINE_NAMES
from dengue_forecast.utils.hashing import sha256_file


class Milestone2ReportError(ValueError):
    """Raised when final Milestone 2 reports cannot be assembled truthfully."""


REQUIRED_SUMMARY_SECTIONS = [
    "Executive summary",
    "Dataset/version used",
    "Modeling readiness",
    "Temporal split methodology",
    "Baselines",
    "Candidate models",
    "Hyperparameter tuning",
    "Feature ablation",
    "Temporal CV results",
    "Locked test results",
    "District error analysis",
    "High-incidence error analysis",
    "Explainability",
    "Champion model",
    "Baseline comparison",
    "Known limitations",
    "Reproducibility",
    "Recommendation for Milestone 3",
]

METRIC_COLUMNS = ["mae", "rmse", "r2", "poisson_deviance", "bias", "mae_top_decile"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.exists() else None


def _artifact_root(root: str | Path, artifact_root: str | Path | None) -> Path:
    base = Path(root)
    if artifact_root is None:
        return base / "artifacts"
    path = Path(artifact_root)
    return path if path.is_absolute() else base / path


def _reports_root(root: str | Path, reports_root: str | Path | None) -> Path:
    base = Path(root)
    if reports_root is None:
        return base / "data" / "reports"
    path = Path(reports_root)
    return path if path.is_absolute() else base / path


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or value is pd.NA:
        return "pending"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "pending"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _hash_line(path: Path, root: Path) -> str:
    portable = _portable(path, root)
    return (
        f"- `{portable}` SHA256 `{sha256_file(path)}`"
        if path.exists()
        else f"- `{portable}` missing"
    )


def _load_metrics(artifacts: Path) -> pd.DataFrame:
    path = artifacts / "experiments" / "fold_metrics.parquet"
    if not path.exists():
        raise Milestone2ReportError(f"Fold metrics missing: {path}")
    metrics = pd.read_parquet(path)
    required = {"config_id", "model", "objective", "feature_set", "fold", *METRIC_COLUMNS}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise Milestone2ReportError(f"Metrics missing columns: {missing}")
    return metrics


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _mean_table(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    config_id_default: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, dropna=False):
        values_tuple = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values_tuple, strict=True))
        if config_id_default is not None:
            row["config_id"] = config_id_default
        row["fold_count"] = int(group["fold"].nunique()) if "fold" in group else pd.NA
        row["validation_n"] = int(pd.to_numeric(group.get("n", pd.Series(dtype=float))).sum())
        row["validation_total"] = int(
            pd.to_numeric(group.get("total", pd.Series(dtype=float))).sum()
        )
        if "coverage" in group:
            row["min_coverage"] = float(pd.to_numeric(group["coverage"], errors="coerce").min())
        for metric in METRIC_COLUMNS:
            if metric in group:
                row[f"mean_cv_{metric}"] = float(
                    pd.to_numeric(group[metric], errors="coerce").mean()
                )
        row["cv_mae_stability_std"] = float(
            pd.to_numeric(group["mae"], errors="coerce").std(ddof=0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _best_rows(candidates: pd.DataFrame, baselines: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if candidates.empty:
        raise Milestone2ReportError("No model candidates available for final report")
    best_ml = candidates.sort_values("mean_cv_mae", kind="mergesort").iloc[0]
    full = (
        baselines[baselines["min_coverage"].fillna(0).ge(1.0)] if not baselines.empty else baselines
    )
    if full.empty:
        raise Milestone2ReportError("No full-coverage baseline available for final report")
    best_baseline = full.sort_values("mean_cv_mae", kind="mergesort").iloc[0]
    return best_ml, best_baseline


def _load_holdout(artifacts: Path) -> dict[str, Any] | None:
    holdout = artifacts / "locked_test"
    state = _maybe_json(holdout / "holdout_state.json")
    metrics = _maybe_json(holdout / "locked_test_metrics.json")
    if not state or state.get("status") != "completed" or not metrics:
        return None
    predictions = holdout / "locked_test_predictions.parquet"
    baseline_predictions = holdout / "locked_test_baseline_predictions.parquet"
    if not predictions.exists() or not baseline_predictions.exists():
        raise Milestone2ReportError(
            "Completed holdout state exists but prediction receipts are missing"
        )
    return {
        "state": state,
        "metrics": metrics,
        "predictions_path": predictions,
        "baseline_predictions_path": baseline_predictions,
    }


def _candidate_table(
    metrics: pd.DataFrame,
    freeze: dict[str, Any] | None,
    holdout: dict[str, Any] | None,
) -> pd.DataFrame:
    table = _mean_table(metrics, ["config_id", "model", "objective", "feature_set"])
    table.insert(0, "row_type", "candidate")
    table["locked_test_status"] = ""
    table["locked_test_mae"] = pd.NA
    table["locked_test_rmse"] = pd.NA
    table["locked_test_r2"] = pd.NA
    table["locked_test_poisson_deviance"] = pd.NA
    table["locked_test_mae_top_decile"] = pd.NA
    selected = (freeze or {}).get("champion_config", {}).get("stable_id")
    if selected is not None:
        table.loc[table["config_id"].astype(str).eq(str(selected)), "locked_test_status"] = (
            "frozen_champion_pending_locked_test" if holdout is None else "frozen_champion_scored"
        )
    if holdout is not None and selected is not None:
        model_metrics = holdout["metrics"].get("model_metrics") or holdout["metrics"]
        mask = table["config_id"].astype(str).eq(str(selected))
        for source, target in [
            ("mae", "locked_test_mae"),
            ("rmse", "locked_test_rmse"),
            ("r2", "locked_test_r2"),
            ("poisson_deviance", "locked_test_poisson_deviance"),
            ("mae_top_decile", "locked_test_mae_top_decile"),
        ]:
            table.loc[mask, target] = model_metrics.get(source, pd.NA)
    return table.sort_values("mean_cv_mae", kind="mergesort").reset_index(drop=True)


def _baseline_table(artifacts: Path, reports: Path, holdout: dict[str, Any] | None) -> pd.DataFrame:
    source = artifacts / "reports" / "baseline_results.csv"
    if not source.exists():
        source = reports / "baseline_results.csv"
    baseline = _read_csv(source)
    if baseline.empty:
        return pd.DataFrame()
    table = _mean_table(baseline, ["model", "feature_set"], config_id_default="baseline")
    table.insert(0, "row_type", "baseline")
    table["objective"] = "baseline"
    table["locked_test_status"] = "pending_locked_test" if holdout is None else "locked_test_scored"
    table["locked_test_mae"] = pd.NA
    table["locked_test_rmse"] = pd.NA
    table["locked_test_r2"] = pd.NA
    table["locked_test_poisson_deviance"] = pd.NA
    table["locked_test_mae_top_decile"] = pd.NA
    table["locked_test_availability_n"] = pd.NA
    table["locked_test_availability_coverage"] = pd.NA
    table["locked_test_paired_model_mae"] = pd.NA
    if holdout is not None:
        baselines = holdout["metrics"].get("baselines", {})
        native = baselines.get("native", {})
        availability = baselines.get("availability", {})
        paired = baselines.get("paired_with_model", {})
        for name in BASELINE_NAMES:
            mask = table["model"].astype(str).eq(name)
            for source_key, target_key in [
                ("mae", "locked_test_mae"),
                ("rmse", "locked_test_rmse"),
                ("r2", "locked_test_r2"),
                ("poisson_deviance", "locked_test_poisson_deviance"),
                ("mae_top_decile", "locked_test_mae_top_decile"),
            ]:
                table.loc[mask, target_key] = native.get(name, {}).get(source_key, pd.NA)
            table.loc[mask, "locked_test_availability_n"] = availability.get(name, {}).get(
                "n", pd.NA
            )
            table.loc[mask, "locked_test_availability_coverage"] = availability.get(name, {}).get(
                "coverage", pd.NA
            )
            table.loc[mask, "locked_test_paired_model_mae"] = (
                paired.get(name, {}).get("model", {}).get("mae", pd.NA)
            )
    return table.sort_values("mean_cv_mae", kind="mergesort").reset_index(drop=True)


def _primary_table(candidates: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_type",
        "model",
        "objective",
        "feature_set",
        "config_id",
        "fold_count",
        "validation_n",
        "validation_total",
        "min_coverage",
        "mean_cv_mae",
        "mean_cv_rmse",
        "mean_cv_r2",
        "mean_cv_poisson_deviance",
        "mean_cv_bias",
        "mean_cv_mae_top_decile",
        "cv_mae_stability_std",
        "locked_test_status",
        "locked_test_mae",
        "locked_test_rmse",
        "locked_test_r2",
        "locked_test_poisson_deviance",
        "locked_test_mae_top_decile",
        "locked_test_availability_n",
        "locked_test_availability_coverage",
        "locked_test_paired_model_mae",
    ]
    table = pd.concat([baselines, candidates], ignore_index=True, sort=False)
    for column in columns:
        if column not in table:
            table[column] = pd.NA
    return table[columns]


def _read_inputs_hashes(root: Path) -> dict[str, str]:
    payload = _maybe_json(root / "docs" / "baselines" / "milestone-2-inputs.json") or {}
    hashes = payload.get("immutable_inputs") or {}
    return {str(key): str(value) for key, value in hashes.items()}


def _top_bottom(
    frame: pd.DataFrame, *, metric: str = "mae", n: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or metric not in frame:
        return pd.DataFrame(), pd.DataFrame()
    valid = frame[pd.to_numeric(frame[metric], errors="coerce").notna()].copy()
    return valid.nsmallest(n, metric), valid.nlargest(n, metric)


def _ablation_lines(ablation: pd.DataFrame) -> list[str]:
    if ablation.empty:
        return ["- Pending: `data/reports/ablation_summary.csv` is not available."]
    lines = [f"- Controlled ablation rows: {len(ablation)}."]
    for variant, group in ablation.groupby("config_variant", dropna=False):
        by_step = {str(row.ablation_step): row for row in group.itertuples(index=False)}
        a = by_step.get("A_cases_only")
        b = by_step.get("B_cases_rainfall")
        c = by_step.get("C_cases_full_weather")
        d = by_step.get("D_full_context")
        pieces = []
        if a is not None and b is not None:
            pieces.append(
                f"rainfall B vs A {float(b.mean_mae) - float(a.mean_mae):+.6f} MAE, "
                f"fold wins {int(b.fold_wins_vs_cases_only)}/6"
            )
        if b is not None and c is not None:
            pieces.append(f"weather C vs B {float(c.mean_mae) - float(b.mean_mae):+.6f} MAE")
        if c is not None and d is not None:
            pieces.append(f"area D vs C {float(d.mean_mae) - float(c.mean_mae):+.6f} MAE")
        if pieces:
            lines.append(f"- `{variant}`: " + "; ".join(pieces) + ".")
    return lines


def _read_final_model_metadata(artifacts: Path, freeze: dict[str, Any] | None) -> dict[str, Any]:
    if freeze is None:
        return {}
    final_id = (
        f"final__{freeze['champion_config']['stable_id']}__{freeze['configuration_hash'][:12]}"
    )
    model_dir = artifacts / "models" / final_id
    champion_dir = artifacts / "models" / "champion"
    for directory in [champion_dir, model_dir]:
        metadata = _maybe_json(directory / "metadata.json")
        if metadata is not None:
            return {"directory": directory, "metadata": metadata}
    return {"directory": model_dir}


def _read_features_for_config(artifacts: Path, config_id: str) -> list[str]:
    matches = sorted((artifacts / "experiments").glob(f"{config_id}__val_*/features.json"))
    if not matches:
        return []
    raw = _maybe_json(matches[0]) or {}
    return list(raw.get("feature_columns") or [])


def _recommendation(
    best_ml: pd.Series,
    best_baseline: pd.Series,
    holdout: dict[str, Any] | None,
    freeze: dict[str, Any] | None,
) -> str:
    cv_delta = float(best_ml["mean_cv_mae"]) - float(best_baseline["mean_cv_mae"])
    if cv_delta >= 0:
        return (
            "NOT READY: the best ML validation candidate does not beat the strongest "
            "full-coverage naive baseline. This is a valid negative study, not an "
            "engineering failure."
        )
    if holdout is None:
        return (
            "NOT READY: validation suggests possible ML benefit, but locked-test evidence is "
            "pending and cannot be inferred."
        )
    if freeze is None:
        return (
            "NOT READY: locked-test metrics exist but no frozen champion configuration was found."
        )
    model_metrics = holdout["metrics"].get("model_metrics") or holdout["metrics"]
    best_cv_baseline_name = str(best_baseline["model"])
    locked_baseline = (
        holdout["metrics"].get("baselines", {}).get("native", {}).get(best_cv_baseline_name, {})
    )
    locked_delta = float(model_metrics["mae"]) - float(locked_baseline.get("mae", math.nan))
    if math.isnan(locked_delta) or locked_delta >= 0:
        return (
            "NOT READY: locked-test MAE does not beat the strongest validation baseline on "
            "its recorded locked-test metric."
        )
    return (
        "REVIEW REQUIRED: ML beats the strongest baseline on MAE, but Milestone 3 still "
        "requires explicit stability, high-incidence, non-dominance, and degradation review."
    )


def generate_milestone2_summary(
    root: str | Path,
    *,
    artifact_root: str | Path | None = None,
    reports_root: str | Path | None = None,
) -> Path:
    base = Path(root)
    artifacts = _artifact_root(base, artifact_root)
    reports = _reports_root(base, reports_root)
    reports.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(artifacts)
    freeze = _maybe_json(artifacts / "experiments" / "freeze.json")
    split = _maybe_json(artifacts / "experiments" / "split_definition.json") or {}
    holdout = _load_holdout(artifacts)
    candidates = _candidate_table(metrics, freeze, holdout)
    baselines = _baseline_table(artifacts, reports, holdout)
    primary = _primary_table(candidates, baselines)
    primary_path = reports / "primary_evaluation_table.csv"
    baseline_path = reports / "baseline_evaluation_table.csv"
    primary.to_csv(primary_path, index=False)
    baselines.to_csv(baseline_path, index=False)

    best_ml, best_baseline = _best_rows(candidates, baselines)
    rec = _recommendation(best_ml, best_baseline, holdout, freeze)
    hashes = _read_inputs_hashes(base)
    validation_rows = _read_csv(reports / "primary_validation_table.csv")
    ablation = _read_csv(reports / "ablation_summary.csv")
    trials = _read_csv(artifacts / "tuning" / "trials.csv")
    district = _read_csv(reports / "district_error_analysis.csv")
    outbreak = _read_csv(reports / "outbreak_error_analysis.csv")
    missingness = _read_csv(reports / "missingness_report.csv")
    missing_sensitivity = _read_csv(reports / "missingness_sensitivity.csv")
    errors = _read_csv(reports / "largest_error_cases.csv")
    inputs = [
        "data/processed/ml_training_dataset.parquet",
        "data/reports/feature_registry.csv",
        "artifacts/experiments/split_definition.json",
    ]
    rejected = split.get("rejected_validation_years") or []
    holdout_audit = {int(row["year"]): row for row in split.get("holdout_audit") or []}
    locked_2024 = holdout_audit.get(2024, {})
    family_counts = metrics.groupby("model")["config_id"].nunique().sort_index()
    trial_status = (
        trials.groupby(["family", "status"]).size() if not trials.empty else pd.Series(dtype=int)
    )
    best_districts, worst_districts = _top_bottom(district)
    _, worst_errors = _top_bottom(errors, metric="absolute_error")
    locked_status = (
        "completed immutable holdout loaded"
        if holdout
        else "pending; no locked-test score is reported"
    )
    champion_label = (
        str(freeze.get("champion_config", {}).get("stable_id"))
        if freeze
        else str(best_ml["config_id"])
    )

    lines = [
        "# Milestone 2 Summary",
        "",
        "## Executive summary",
        (
            f"Best ML by equal-fold validation MAE is `{best_ml['config_id']}` "
            f"({_fmt(best_ml['mean_cv_mae'])}); the strongest full-coverage baseline is "
            f"`{best_baseline['model']}` ({_fmt(best_baseline['mean_cv_mae'])}). "
            f"Delta ML minus baseline is {_fmt(float(best_ml['mean_cv_mae']) - float(best_baseline['mean_cv_mae']))} MAE."
        ),
        "The result is reported as a validation-negative ML study unless the frozen champion beats the best baseline under the required criteria.",
        "",
        "## Dataset/version used",
        *(_hash_line(base / item, base) for item in inputs),
        *[f"- `{key}` baseline SHA256 `{value}`" for key, value in hashes.items() if key in inputs],
        f"- Completed CV artifact: `{_portable(artifacts / 'experiments' / 'fold_metrics.parquet', base)}` with {len(metrics)} fold rows.",
        "",
        "## Modeling readiness",
        f"- Actual OOF validation cohort: {len(validation_rows) if not validation_rows.empty else 'pending'} rows across {validation_rows['district_id'].nunique() if not validation_rows.empty else 'pending'} districts.",
        "- Structured missingness is retained; target imputation is not permitted.",
        "- Reported counts are official notified dengue cases, not true infection counts.",
        "",
        "## Temporal split methodology",
        f"- Validation folds: {', '.join(str(fold.get('fold_id')) for fold in split.get('folds', []))}.",
        f"- Locked test period: {split.get('locked_test_start', 'pending')} to {split.get('locked_test_end', 'pending')}.",
        "- Rejected validation years before performance evaluation: "
        + ", ".join(f"{row['year']} ({row['reason']})" for row in rejected),
        (
            "- 2024 locked cohort: "
            f"{locked_2024.get('trainable_rows', 'pending')} eligible trainable rows, "
            f"{locked_2024.get('districts', 'pending')} districts; origins are 2024 only and "
            "the final one-week target may spill into early 2025, which is not 2025 development."
        ),
        "",
        "## Baselines",
        *[
            f"- `{row.model}`: mean CV MAE {_fmt(row.mean_cv_mae)}, RMSE {_fmt(row.mean_cv_rmse)}, "
            f"high-incidence MAE {_fmt(row.mean_cv_mae_top_decile)}, min coverage {_fmt(row.min_coverage)}."
            for row in baselines.itertuples(index=False)
        ],
        "",
        "## Candidate models",
        "- Families/configs: "
        + ", ".join(f"{model}={count}" for model, count in family_counts.items())
        + ".",
        f"- Candidate fold fits: {len(metrics)} ({metrics['config_id'].nunique()} configs x {metrics['fold'].nunique()} folds).",
        f"- Best ML candidate: `{best_ml['config_id']}` with MAE {_fmt(best_ml['mean_cv_mae'])}, RMSE {_fmt(best_ml['mean_cv_rmse'])}, high-incidence MAE {_fmt(best_ml['mean_cv_mae_top_decile'])}.",
        "",
        "## Hyperparameter tuning",
        *(
            [
                f"- Trial status counts: {', '.join(f'{idx[0]}/{idx[1]}={value}' for idx, value in trial_status.items())}."
            ]
            if not trial_status.empty
            else ["- Pending: tuning trial CSV is not available."]
        ),
        "",
        "## Feature ablation",
        *_ablation_lines(ablation),
        "",
        "## Temporal CV results",
        "- All mean CV metrics are equal-weight fold means; pooled row metrics are not substituted for model selection.",
        f"- Best ML fold stability std(MAE): {_fmt(best_ml['cv_mae_stability_std'])}.",
        f"- Primary evaluation table: `{_portable(primary_path, base)}`.",
        "",
        "## Locked test results",
        f"- Status: {locked_status}.",
        (
            f"- Frozen champion locked-test MAE {_fmt((holdout or {}).get('metrics', {}).get('model_metrics', {}).get('mae'))}; "
            f"validation mean-fold MAE for selected ML candidate {_fmt(candidates[candidates['config_id'].astype(str).eq(champion_label)]['mean_cv_mae'].iloc[0] if champion_label in set(candidates['config_id'].astype(str)) else best_ml['mean_cv_mae'])}."
            if holdout
            else "- Locked-test metrics are intentionally blank until a completed immutable holdout run exists."
        ),
        "",
        "## District error analysis",
        *[
            f"- Stronger district `{row.district_id}`: MAE {_fmt(row.mae)}, n={int(row.n)}."
            for row in best_districts.itertuples(index=False)
        ],
        *[
            f"- Weak district `{row.district_id}`: MAE {_fmt(row.mae)}, n={int(row.n)}."
            for row in worst_districts.itertuples(index=False)
        ],
        "",
        "## High-incidence error analysis",
        *[
            f"- {row.fold} {row.subset}: MAE {_fmt(row.mae)}, bias {_fmt(row.bias)}, underprediction rate {_fmt(row.underprediction_rate)}."
            for row in outbreak.head(8).itertuples(index=False)
        ],
        "",
        "## Explainability",
        "- Feature importance and explainability artifacts are validation/pretest interpretation only; they are not causal weather claims.",
        "- Importance scope is the frozen/selected validation artifact where available; pending explainability is reported as pending, not absent.",
        "",
        "## Champion model",
        f"- Champion among ML candidates: `{champion_label}`.",
        f"- Honest label: ML does not materially beat `{best_baseline['model']}` on validation if delta is non-negative.",
        "- Final champion model hash is reported from `artifacts/models/champion/model.joblib` or the frozen final model only; CV fold binaries are not final hashes.",
        "",
        "## Baseline comparison",
        f"- Best validation baseline `{best_baseline['model']}` MAE {_fmt(best_baseline['mean_cv_mae'])}; best ML MAE {_fmt(best_ml['mean_cv_mae'])}.",
        f"- Relative ML change vs best baseline: {_fmt((float(best_ml['mean_cv_mae']) - float(best_baseline['mean_cv_mae'])) / float(best_baseline['mean_cv_mae']) * 100)}%.",
        "",
        "## Known limitations",
        "- Registry override population was not tested; historical incidence uses a static denominator where present.",
        "- 2024 static denominator and 2017 district geography do not reconstruct historical administrative changes.",
        "- Retrospective observed weather is not an operational weather forecast feed.",
        "- Target is reported/notified dengue counts, not true infections.",
        "- Operational availability is not proven by retrospective artifacts.",
        "- Overall structured dengue missingness is 74.82%; missing historical periods cannot be generalized.",
        *[
            f"- Missingness `{row.column}`: {int(row.missing_rows)}/{int(row.rows)} rows missing ({_fmt(row.missing_pct)}%)."
            for row in missingness.head(5).itertuples(index=False)
        ],
        *[
            f"- Missingness group `{row.missingness_group}`: n={int(row.n) if not pd.isna(row.n) else 0}, MAE {_fmt(row.mae)}."
            for row in missing_sensitivity.head(6).itertuples(index=False)
        ],
        "",
        "## Reproducibility",
        "- CV source archive: `.hermes/m2-source-before-first-cv.tar.gz` is the immutable old-version source for the completed CV run; current report/gate glue may be newer.",
        "- Commands: `UV_CACHE_DIR=/tmp/uv-cache uv run python -m dengue_forecast.cli milestone2 validate` for read-only completed-run validation.",
        "- Local prediction uses the saved champion bundle only after freeze; reports never rescore the locked test.",
        *[
            f"- Large error example `{row.district_id}` {row.week_start_date}: target {row.target}, prediction {_fmt(row.prediction)}, absolute error {_fmt(row.absolute_error)}."
            for row in worst_errors.head(3).itertuples(index=False)
        ],
        "",
        "## Recommendation for Milestone 3",
        rec,
        "",
    ]
    path = reports / "milestone_2_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_champion_model_card(
    root: str | Path,
    *,
    artifact_root: str | Path | None = None,
    reports_root: str | Path | None = None,
) -> Path:
    base = Path(root)
    artifacts = _artifact_root(base, artifact_root)
    reports = _reports_root(base, reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    freeze = _maybe_json(artifacts / "experiments" / "freeze.json")
    if freeze is None:
        raise Milestone2ReportError("Cannot write champion model card before freeze.json exists")
    config = freeze["champion_config"]
    metrics = _load_metrics(artifacts)
    subset = metrics[metrics["config_id"].astype(str).eq(str(config["stable_id"]))]
    holdout = _load_holdout(artifacts)
    final = _read_final_model_metadata(artifacts, freeze)
    model_dir = final.get("directory")
    model_path = model_dir / "model.joblib" if isinstance(model_dir, Path) else Path()
    metadata = final.get("metadata") or {}
    feature_columns = list(
        config.get("feature_columns")
        or _read_features_for_config(artifacts, str(config["stable_id"]))
    )
    schema = metadata.get("schema") or metadata.get("feature_schema") or {}
    model_hash = (
        sha256_file(model_path) if model_path.exists() else "pending_final_champion_model_joblib"
    )
    holdout_metrics = (holdout or {}).get("metrics", {})
    model_metrics = holdout_metrics.get("model_metrics", {})
    strongest = str(config.get("strongest_full_cohort_baseline", "pending"))
    baseline_metric = (
        holdout_metrics.get("baselines", {}).get("native", {}).get(strongest, {})
        if holdout_metrics
        else {}
    )
    cv_means = {
        metric: float(pd.to_numeric(subset[metric], errors="coerce").mean())
        for metric in METRIC_COLUMNS
        if metric in subset and not subset.empty
    }
    lines = [
        "# Champion Model Card",
        "",
        "## Intended Use",
        "Retrospective one-week-ahead district-level dengue reported-count forecasting for Milestone 2 evaluation. Prohibited interpretation: do not read predictions as true infection incidence, operational forecast readiness, or causal weather effect.",
        "",
        "## Model",
        f"- Stable ID: `{config['stable_id']}`",
        f"- Family: `{config['family']}`",
        f"- Objective: `{config['objective']}`",
        f"- Feature set: `{config.get('feature_set', 'frozen ordered feature list')}`",
        "- Target: next-week reported dengue case count.",
        f"- Training range: {config.get('allowed_train_bound', {}).get('first_origin', 'pending')} to {config.get('allowed_train_bound', {}).get('last_origin', 'pending')}",
        f"- Training rows/districts: {config.get('allowed_train_bound', {}).get('rows', 'pending')}/{config.get('allowed_train_bound', {}).get('districts', 'pending')}",
        f"- Validation folds: {', '.join(sorted(subset['fold'].astype(str).unique())) if not subset.empty else 'pending'}",
        "- Locked test range: 2024-01-06 to 2025-01-03 feature interval; origins are 2024 only.",
        f"- Hyperparameters: `{json.dumps(config.get('hyperparams', {}), sort_keys=True)}`",
        f"- Preprocessing/seed/postprocessing: `{json.dumps({'preprocessing': config.get('preprocessing', {}), 'seed': config.get('seed'), 'postprocessing': config.get('postprocessing', {})}, sort_keys=True)}`",
        "",
        "## Metrics",
        *[f"- CV mean {key}: {_fmt(value)}" for key, value in cv_means.items()],
        f"- Locked-test MAE: {_fmt(model_metrics.get('mae'))}",
        f"- Locked-test RMSE: {_fmt(model_metrics.get('rmse'))}",
        f"- Locked-test R2: {_fmt(model_metrics.get('r2'))}",
        f"- Locked-test Poisson deviance: {_fmt(model_metrics.get('poisson_deviance'))}",
        f"- Locked-test high-incidence MAE: {_fmt(model_metrics.get('mae_top_decile'))}",
        f"- Incumbent baseline `{strongest}` locked-test MAE: {_fmt(baseline_metric.get('mae'))}",
        "",
        "## Artifacts",
        f"- Configuration hash: `{freeze['configuration_hash']}`",
        f"- Final model path: `{_portable(model_path, base) if model_path else 'pending'}`",
        f"- Final model SHA256: `{model_hash}`",
        f"- Freeze path: `{_portable(artifacts / 'experiments' / 'freeze.json', base)}`",
        f"- Holdout metrics path: `{_portable(artifacts / 'locked_test' / 'locked_test_metrics.json', base)}`",
        "",
        "## Feature Schema",
        f"- Ordered feature count: {len(feature_columns)}",
        *[f"- `{name}`" for name in feature_columns],
        f"- Schema: `{json.dumps(schema, sort_keys=True)}`",
        "",
        "## Known Limits",
        "- Champion among ML candidates is not necessarily better than the overall strongest baseline.",
        "- Weak districts and high-incidence underprediction require operational review before Milestone 3.",
        "- Registry override population was not tested.",
        "- Static 2024 denominator and 2017 geography do not reconstruct historical populations or boundaries.",
        "- Target is reported/notified counts, not true infections.",
        "- Retrospective observed weather does not prove operational weather-forecast availability.",
        "- Structured missingness is 74.82%; no target imputation is permitted.",
        "- Importance/explainability is validation-scope interpretation, not causality.",
        "",
        "## Reproduction",
        "`UV_CACHE_DIR=/tmp/uv-cache uv run python -m dengue_forecast.cli milestone2 validate` validates the completed frozen run read-only. Local prediction must load the saved champion bundle and exact ordered feature list; reports must not rerun or rescore holdout models.",
        "",
    ]
    path = reports / "champion_model_card.md"
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    champion_dir = artifacts / "models" / "champion"
    if champion_dir.exists():
        (champion_dir / "champion_model_card.md").write_text(text, encoding="utf-8")
    return path


__all__ = [
    "Milestone2ReportError",
    "REQUIRED_SUMMARY_SECTIONS",
    "generate_champion_model_card",
    "generate_milestone2_summary",
]
