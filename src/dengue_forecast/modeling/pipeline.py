from __future__ import annotations

# ruff: noqa: I001

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.reports.comparison import write_shared_metric_comparison
from dengue_forecast.modeling.ablation import write_ablation_artifacts
from dengue_forecast.modeling.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REGISTRY_PATH,
    FEATURE_SET_NAMES,
    get_feature_set,
    load_modeling_dataset,
    load_modeling_registry,
)
from dengue_forecast.modeling.evaluate import (
    PREDICTION_KEY_COLUMNS,
    build_primary_validation_table,
    ensure_disk_budget,
    evaluate_baselines,
    load_development_context,
    summarize_model_results,
)
from dengue_forecast.modeling.explain import explain_validation_model
from dengue_forecast.modeling.holdout import (
    BASELINE_NAMES,
    permitted_development_training_frame,
    score_locked_holdout_once,
    validate_model_binary_load,
    validate_holdout_run,
)
from dengue_forecast.modeling.registry import ExperimentRegistry, RegistryError
from dengue_forecast.modeling.selection import (
    SelectionError,
    build_freeze_payload,
    freeze_champion_config,
    select_champion,
    validate_selection_prerequisites,
)
from dengue_forecast.modeling.splits import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_POLICY_PATH,
    SplitPolicy,
    attach_hashes,
    build_temporal_splits,
    load_or_create_split_definition,
    load_split_policy,
    readiness_summary,
    write_readiness_report,
    write_split_report,
)
from dengue_forecast.modeling.train import ModelConfig, load_champion, train_fold_model
from dengue_forecast.modeling.tune import (
    default_tuning_plan,
    run_optuna_tuning,
    run_screen,
    select_feature_sets_for_tuning,
)
from dengue_forecast.reports.milestone2 import (
    generate_champion_model_card,
    generate_milestone2_summary,
)
from dengue_forecast.reports.modeling import write_error_analysis_reports
from dengue_forecast.reports.plots import generate_development_plots
from dengue_forecast.utils.hashing import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTECTED_INPUTS = PROJECT_ROOT / "docs" / "baselines" / "milestone-2-inputs.json"


class Milestone2Error(RuntimeError):
    """Raised when the Milestone 2 workflow would violate the evaluation policy."""


@dataclass(frozen=True)
class ModelingPaths:
    root: Path = PROJECT_ROOT
    dataset_path: Path = DEFAULT_DATASET_PATH
    registry_path: Path = DEFAULT_REGISTRY_PATH
    config_path: Path = DEFAULT_CONFIG_PATH
    policy_path: Path = DEFAULT_POLICY_PATH
    artifact_root: Path = PROJECT_ROOT / "artifacts"
    reports_root: Path = PROJECT_ROOT / "data" / "reports"

    @classmethod
    def resolve(
        cls,
        *,
        root: str | Path | None = None,
        dataset: str | Path | None = None,
        registry: str | Path | None = None,
        config: str | Path | None = None,
        artifact_root: str | Path | None = None,
        reports_root: str | Path | None = None,
    ) -> ModelingPaths:
        base = Path(root).resolve() if root is not None else PROJECT_ROOT

        def under_base(value: str | Path | None, default: Path) -> Path:
            if value is None:
                return default if default.is_absolute() else base / default
            path = Path(value)
            return path if path.is_absolute() else base / path

        return cls(
            root=base,
            dataset_path=under_base(dataset, Path("data/processed/ml_training_dataset.parquet")),
            registry_path=under_base(registry, Path("data/reports/feature_registry.csv")),
            config_path=under_base(config, Path("configs/modeling.yaml")),
            policy_path=under_base(None, Path("docs/modeling-feature-policy.md")),
            artifact_root=under_base(artifact_root, Path("artifacts")),
            reports_root=under_base(reports_root, Path("data/reports")),
        )

    @property
    def split_path(self) -> Path:
        return self.artifact_root / "experiments" / "split_definition.json"

    @property
    def freeze_path(self) -> Path:
        return self.artifact_root / "experiments" / "freeze.json"

    @property
    def review_receipt_path(self) -> Path:
        return self.artifact_root / "independent_review_receipt.json"

    @property
    def holdout_dir(self) -> Path:
        return self.artifact_root / "locked_test"

    @property
    def explain_dir(self) -> Path:
        return self.artifact_root / "explainability"

    @property
    def plot_dir(self) -> Path:
        return self.reports_root / "plots"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_protected_inputs(paths: ModelingPaths) -> dict[str, Any]:
    manifest_path = paths.root / "docs" / "baselines" / "milestone-2-inputs.json"
    if not manifest_path.exists():
        raise Milestone2Error(f"Protected input manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    failures: list[str] = []
    for relative, expected in (manifest.get("immutable_inputs") or {}).items():
        path = paths.root / relative
        if not path.exists():
            failures.append(f"{relative}: missing")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, got {actual}")
    if failures:
        raise Milestone2Error("Protected Milestone 1 input hashes changed: " + "; ".join(failures))
    return {"manifest": str(manifest_path), "checked": len(manifest.get("immutable_inputs") or {})}


def _context(paths: ModelingPaths, *, production: bool = True):
    return load_development_context(
        dataset_path=paths.dataset_path,
        registry_path=paths.registry_path,
        split_path=paths.split_path,
        production=production,
    )


def stage_readiness(paths: ModelingPaths) -> dict[str, Any]:
    verify_protected_inputs(paths)
    registry = load_modeling_registry(paths.registry_path)
    frame = load_modeling_dataset(paths.dataset_path, registry=registry, production=True)
    summary = readiness_summary(frame)
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    write_readiness_report(summary, paths.reports_root / "modeling_readiness_report.md")
    return {"readiness": summary}


def stage_splits(paths: ModelingPaths) -> dict[str, Any]:
    verify_protected_inputs(paths)
    registry = load_modeling_registry(paths.registry_path)
    frame = load_modeling_dataset(paths.dataset_path, registry=registry, production=True)
    policy = load_split_policy(paths.config_path)
    split = load_or_create_split_definition(
        frame,
        policy,
        output_path=paths.split_path,
        dataset_path=paths.dataset_path,
        registry_path=paths.registry_path,
        config_path=paths.config_path,
        policy_path=paths.policy_path,
    )
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    write_split_report(split, paths.reports_root / "temporal_split_report.md")
    return {"split": split}


def stage_baselines(paths: ModelingPaths, *, production: bool = True) -> dict[str, Any]:
    verify_protected_inputs(paths)
    context = _context(paths, production=production)
    result = evaluate_baselines(context, output_root=paths.artifact_root)
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        paths.artifact_root / "reports" / "baseline_results.csv",
        paths.reports_root / "baseline_results.csv",
    )
    _write_json(
        paths.artifact_root / "experiments" / "baseline_stage.json",
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "metrics_rows": len(result.metrics),
            "prediction_rows": len(result.predictions),
        },
    )
    return {"metrics_rows": len(result.metrics), "prediction_rows": len(result.predictions)}


def stage_train(paths: ModelingPaths, *, production: bool = True) -> dict[str, Any]:
    verify_protected_inputs(paths)
    ensure_disk_budget(paths.artifact_root, minimum_gb=1.0)
    context = _context(paths, production=production)
    metrics = run_screen(context, output_root=paths.artifact_root)
    return {"stage": "screen", "metrics_rows": len(metrics)}


def _load_fold_metrics(paths: ModelingPaths) -> pd.DataFrame:
    metrics_path = paths.artifact_root / "experiments" / "fold_metrics.parquet"
    if not metrics_path.exists():
        raise Milestone2Error(f"Fold metrics missing; run model train first: {metrics_path}")
    metrics = pd.read_parquet(metrics_path)
    required = {"config_id", "fold", "mae", "model", "feature_set", "experiment_id"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise Milestone2Error(f"Fold metrics missing columns: {missing}")
    return metrics


def stage_ablation(paths: ModelingPaths) -> dict[str, Any]:
    metrics = _load_fold_metrics(paths)
    observed_sets = set(metrics["feature_set"].dropna().astype(str))
    missing_sets = sorted(set(FEATURE_SET_NAMES) - observed_sets)
    if missing_sets:
        raise Milestone2Error(f"Ablation requires all four feature sets; missing {missing_sets}")
    summary = write_ablation_artifacts(metrics, reports_dir=str(paths.reports_root))
    return {"rows": len(summary), "feature_sets": sorted(observed_sets)}


def _screen_summary(paths: ModelingPaths) -> pd.DataFrame:
    metrics = _load_fold_metrics(paths)
    summary = summarize_model_results(metrics)
    if summary.empty:
        raise Milestone2Error("No completed screen metrics are available")
    path = paths.artifact_root / "reports" / "model_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def stage_tune(paths: ModelingPaths, *, production: bool = True) -> dict[str, Any]:
    verify_protected_inputs(paths)
    context = _context(paths, production=production)
    selected = select_feature_sets_for_tuning(_screen_summary(paths))
    trials = run_optuna_tuning(
        context,
        output_root=paths.artifact_root,
        selected_feature_sets=selected,
        plan=default_tuning_plan(),
    )
    return {"selected_feature_sets": selected, "trial_rows": len(trials)}


def _validation_row_keys(predictions: pd.DataFrame) -> list[str]:
    return [
        f"{row.district_id}|{pd.Timestamp(row.week_start_date).date().isoformat()}"
        for row in predictions[["district_id", "week_start_date"]].itertuples(index=False)
    ]


def _candidate_from_group(paths: ModelingPaths, group: pd.DataFrame) -> dict[str, Any]:
    first = group.sort_values("fold").iloc[0]
    experiment_id = str(first["experiment_id"])
    experiment_dir = paths.artifact_root / "experiments" / experiment_id
    config = _read_json(experiment_dir / "config.json")
    features = _read_json(experiment_dir / "features.json")
    rows: list[dict[str, Any]] = []
    for fold_id, fold_group in group.groupby("fold", sort=True):
        pred_path = paths.artifact_root / "experiments" / str(
            fold_group.iloc[0]["experiment_id"]
        ) / "predictions.parquet"
        predictions = pd.read_parquet(pred_path)
        rows.append(
            {
                "fold_id": str(fold_id),
                "mae": float(fold_group.iloc[0]["mae"]),
                "mae_top_decile": float(fold_group.iloc[0].get("mae_top_decile", float("nan"))),
                "validation_row_keys": _validation_row_keys(predictions),
            }
        )
    return {
        "stable_id": str(first["config_id"]),
        "split_id": sha256_file(paths.split_path),
        "dataset_sha256": sha256_file(paths.dataset_path),
        "family": config["family"],
        "objective": config.get("objective", "regression"),
        "hyperparams": config.get("hyperparams", {}),
        "feature_columns": features["feature_columns"],
        "preprocessing": config.get("preprocessing", {"fit_scope": "fold_training_rows_only"}),
        "postprocessing": config.get("postprocessing", {"clip_negative_predictions": True}),
        "seed": int(config.get("seed", 42)),
        "allowed_train_bound": {
            "locked_first_origin": _read_json(paths.split_path)["locked_test_start"]
        },
        "thresholds": {
            "q90": float(group["threshold90"].dropna().mean()),
            "q95": float(group["threshold95"].dropna().mean()),
        },
        "strongest_full_cohort_baseline": _strongest_full_coverage_baseline(paths),
        "simplicity": len(features["feature_columns"]),
        "folds": rows,
    }


def _selection_candidates(paths: ModelingPaths) -> list[dict[str, Any]]:
    metrics = _load_fold_metrics(paths)
    return [
        _candidate_from_group(paths, group)
        for _, group in metrics.groupby("config_id", sort=True)
        if group["fold"].nunique() == metrics["fold"].nunique()
    ]


def _strongest_full_coverage_baseline(paths: ModelingPaths) -> str | None:
    path = paths.artifact_root / "reports" / "baseline_results.csv"
    if not path.exists():
        return None
    metrics = pd.read_csv(path)
    if metrics.empty or "coverage" not in metrics.columns:
        return None
    full = metrics.groupby("model")["coverage"].min()
    full = full[full.eq(1.0)].index.tolist()
    if not full:
        return None
    return str(metrics[metrics["model"].isin(full)].groupby("model")["mae"].mean().idxmin())


def _best_candidate(paths: ModelingPaths) -> dict[str, Any]:
    selection = select_champion(_selection_candidates(paths))
    return selection.champion


def _final_training_binding(
    paths: ModelingPaths, *, production: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = load_modeling_registry(paths.registry_path)
    frame = load_modeling_dataset(paths.dataset_path, registry=registry, production=production)
    split = _read_json(paths.split_path)
    training_frame, training_metadata = permitted_development_training_frame(frame, split)
    training_metadata = {
        **training_metadata,
        "fit_row_digest": _training_fit_row_digest(training_frame),
    }
    return frame, training_metadata


def _bind_selection_to_final_training_frame(
    selection,
    training_metadata: dict[str, Any],
):
    champion = {
        **selection.champion,
        "allowed_train_bound": {
            "locked_first_origin": training_metadata["locked_first_origin"],
            "train_start": training_metadata["first_origin"],
            "train_end": training_metadata["last_origin"],
            "rows": training_metadata["rows"],
            "districts": training_metadata["districts"],
            "row_key_digest": training_metadata["row_key_digest"],
            "fit_row_digest": training_metadata["fit_row_digest"],
            "max_target_end": training_metadata["max_target_end"],
            "horizon_weeks": training_metadata["horizon_weeks"],
            "embargo_weeks": training_metadata["embargo_weeks"],
        },
        "thresholds": {
            "q90": float(training_metadata["thresholds"]["q90"]),
            "q95": float(training_metadata["thresholds"]["q95"]),
        },
    }
    return type(selection)(
        champion=champion,
        ranking=selection.ranking,
        close_models=selection.close_models,
        comparable_folds=selection.comparable_folds,
        validation_row_key_digest=selection.validation_row_key_digest,
        selected_by=selection.selected_by,
    )


def _all_model_predictions(paths: ModelingPaths) -> pd.DataFrame:
    path = paths.artifact_root / "predictions" / "models.parquet"
    if not path.exists():
        raise Milestone2Error(f"Model predictions missing: {path}")
    predictions = pd.read_parquet(path)
    if predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise Milestone2Error("Model predictions contain duplicate config/fold row keys")
    return predictions


def _expected_validation_row_count(paths: ModelingPaths) -> int:
    split = _read_json(paths.split_path)
    folds = split.get("folds") or []
    if not folds:
        raise Milestone2Error("Split definition contains no validation folds")
    return int(sum(int(fold["rows_validation"]) for fold in folds))


def _expected_validation_rows(paths: ModelingPaths) -> pd.DataFrame:
    context = _context(paths, production=True)
    rows: list[pd.DataFrame] = []
    for fold in context.folds:
        validation = context.fold_frames(fold).validation.copy()
        item = validation[["district_id", "week_start_date", "cases_next_week"]].rename(
            columns={"cases_next_week": "expected_target"}
        )
        item["fold"] = str(fold["fold_id"])
        rows.append(item)
    if not rows:
        raise Milestone2Error("Frozen split contains no validation rows")
    expected = pd.concat(rows, ignore_index=True)
    expected["district_id"] = expected["district_id"].astype(str)
    expected["fold"] = expected["fold"].astype(str)
    expected["week_start_date"] = pd.to_datetime(expected["week_start_date"]).dt.date.astype(str)
    return expected[["fold", "district_id", "week_start_date", "expected_target"]]


def stage_analyze(paths: ModelingPaths) -> dict[str, Any]:
    champion = _best_candidate(paths)
    predictions = _all_model_predictions(paths)
    expected_validation_rows = _expected_validation_row_count(paths)
    selected_predictions = predictions[
        predictions["config_id"].astype(str).eq(str(champion["stable_id"]))
    ]
    if len(selected_predictions) != expected_validation_rows:
        raise Milestone2Error(
            "Selected champion predictions do not preserve the full validation cohort: "
            f"expected {expected_validation_rows}, got {len(selected_predictions)}"
        )
    selected = {"config_id": champion["stable_id"]}
    registry = load_modeling_registry(paths.registry_path)
    frame = load_modeling_dataset(paths.dataset_path, registry=registry, production=True)
    report = write_error_analysis_reports(
        predictions,
        frame,
        selected_config=selected,
        output_dir=paths.reports_root,
        largest_errors_limit=50,
    )
    baseline_path = paths.artifact_root / "predictions" / "baselines.parquet"
    baseline_predictions = None
    comparison_paths: dict[str, Path] = {}
    if baseline_path.exists():
        raw_baselines = pd.read_parquet(baseline_path)
        required = {"week_start_date", "target", "prediction"}
        missing = sorted(required - set(raw_baselines.columns))
        if missing:
            raise Milestone2Error(f"Baseline predictions missing plot columns: {missing}")
        comparison_paths = write_shared_metric_comparison(
            model_predictions=predictions,
            baseline_predictions=raw_baselines,
            selected_config_id=str(champion["stable_id"]),
            output_dir=paths.reports_root,
        )
        finite = (
            pd.to_datetime(raw_baselines["week_start_date"], errors="coerce").notna()
            & pd.to_numeric(raw_baselines["target"], errors="coerce").notna()
            & pd.to_numeric(raw_baselines["prediction"], errors="coerce").notna()
        )
        baseline_predictions = raw_baselines.loc[finite].copy()
    plots = generate_development_plots(
        predictions,
        baseline_predictions=baseline_predictions,
        selected_config=selected,
        output_dir=paths.plot_dir,
    )
    primary = build_primary_validation_table(_context(paths, production=True))
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    primary.to_csv(paths.reports_root / "primary_validation_table.csv", index=False)
    return {
        "selected_config": champion["stable_id"],
        "reports": {key: str(value) for key, value in report.paths.items()},
        "comparisons": {key: str(value) for key, value in comparison_paths.items()},
        "plots": {key: str(value) for key, value in plots.items()},
    }


def _latest_fold_model_dir(paths: ModelingPaths, config_id: str) -> tuple[Path, pd.DataFrame]:
    metrics = _load_fold_metrics(paths)
    subset = metrics[metrics["config_id"].astype(str).eq(config_id)].sort_values("fold")
    if subset.empty:
        raise Milestone2Error(f"No metrics found for config {config_id}")
    row = subset.iloc[-1]
    model_dir = paths.artifact_root / "models" / str(row["experiment_id"])
    predictions = pd.read_parquet(
        paths.artifact_root / "experiments" / str(row["experiment_id"]) / "predictions.parquet"
    )
    return model_dir, predictions


def _latest_admissible_boosted_model_dir(
    paths: ModelingPaths, champion: dict[str, Any]
) -> tuple[Path, pd.DataFrame, dict[str, Any]] | None:
    if str(champion.get("family")) != "ridge":
        return None
    metrics = _load_fold_metrics(paths).copy()
    boosted = metrics[
        metrics["model"].astype(str).isin({"xgboost", "lightgbm", "random_forest"})
    ].copy()
    if boosted.empty:
        return None
    complete_folds = set(metrics["fold"].dropna().astype(str))
    complete_configs = boosted.groupby("config_id")["fold"].nunique()
    complete_configs = complete_configs[complete_configs.eq(len(complete_folds))]
    if complete_configs.empty:
        return None
    candidates = boosted[boosted["config_id"].isin(complete_configs.index)].copy()
    summary = candidates.groupby("config_id", as_index=False)["mae"].mean()
    chosen_config = str(summary.sort_values("mae", kind="mergesort").iloc[0]["config_id"])
    subset = candidates[candidates["config_id"].astype(str).eq(chosen_config)].sort_values("fold")
    row = subset.iloc[-1]
    experiment_id = str(row["experiment_id"])
    model_dir = paths.artifact_root / "models" / experiment_id
    predictions = pd.read_parquet(
        paths.artifact_root / "experiments" / experiment_id / "predictions.parquet"
    )
    selected_config = {
        "stable_id": chosen_config,
        "model_id": chosen_config,
        "family": str(row["model"]),
        "fold_id": str(row["fold"]),
        "validation_id": "latest admissible boosted validation fold",
        "provisional": True,
    }
    return model_dir, predictions, selected_config


def stage_explain(paths: ModelingPaths) -> dict[str, Any]:
    champion = _best_candidate(paths)
    model_dir, fold_predictions = _latest_fold_model_dir(paths, str(champion["stable_id"]))
    registry = load_modeling_registry(paths.registry_path)
    frame = load_modeling_dataset(paths.dataset_path, registry=registry, production=True)
    validation = frame.merge(
        fold_predictions[["district_id", "week_start_date"]],
        on=["district_id", "week_start_date"],
        how="inner",
        validate="1:1",
    )
    boosted = _latest_admissible_boosted_model_dir(paths, champion)
    boosted_model = None
    boosted_validation = None
    if boosted is not None:
        boosted_model, boosted_predictions, boosted_selected = boosted
        boosted_validation = frame.merge(
            boosted_predictions[["district_id", "week_start_date"]],
            on=["district_id", "week_start_date"],
            how="inner",
            validate="1:1",
        )
        champion = {**champion, "shortlisted_boosted": boosted_selected}
    result = explain_validation_model(
        model_dir,
        validation,
        output_dir=paths.explain_dir,
        feature_registry=registry,
        selected_config=champion,
        max_sample=200,
        permutation_repeats=3,
        seed=42,
        shortlisted_boosted_model=boosted_model,
        shortlisted_boosted_validation_frame=boosted_validation,
    )
    feature_copy = paths.reports_root / "feature_importance.csv"
    summary_copy = paths.reports_root / "explainability_summary.md"
    shutil.copy2(result.paths["feature_importance"], feature_copy)
    shutil.copy2(result.paths["summary"], summary_copy)
    return {
        "selected_config": champion["stable_id"],
        "paths": {key: str(value) for key, value in result.paths.items()},
    }


def _review_receipt(paths: ModelingPaths) -> dict[str, Any]:
    if not paths.review_receipt_path.exists():
        raise SelectionError(
            "Independent review receipt missing; freeze is intentionally blocked until "
            f"{paths.review_receipt_path} is supplied"
        )
    receipt = _read_json(paths.review_receipt_path)
    required = {"approval_hash", "source_sha256", "split_sha256", "dataset_sha256"}
    missing = sorted(required - set(receipt))
    if missing:
        raise SelectionError(f"Independent review receipt missing fields: {missing}")
    expected = {
        "source_sha256": _source_content_hash(paths.root),
        "split_sha256": sha256_file(paths.split_path),
        "dataset_sha256": sha256_file(paths.dataset_path),
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise SelectionError(f"Independent review receipt hash mismatch: {mismatches}")
    return receipt


def _source_identity(source_root: str | Path | None = None) -> dict[str, Any]:
    from dengue_forecast.modeling.registry import _source_identity as registry_source_identity

    return registry_source_identity(Path(source_root) if source_root is not None else PROJECT_ROOT)


def _source_content_hash(source_root: str | Path | None = None) -> str:
    identity = _source_identity(source_root)
    source_hash = identity.get("source_hash")
    if not isinstance(source_hash, str) or not source_hash:
        raise Milestone2Error("Source identity missing content-based source_hash")
    return source_hash


def _assert_freeze_matches_current_inputs(paths: ModelingPaths, freeze: dict[str, Any]) -> None:
    provenance = freeze.get("provenance") or {}
    expected = {
        "dataset_sha256": sha256_file(paths.dataset_path),
        "registry_sha256": sha256_file(paths.registry_path),
        "modeling_overlay_sha256": sha256_file(paths.config_path),
        "split_sha256": sha256_file(paths.split_path),
    }
    mismatches = [key for key, value in expected.items() if provenance.get(key) != value]
    current_source_hash = _source_content_hash(paths.root)
    try:
        receipt = _review_receipt(paths)
    except SelectionError as exc:
        raise Milestone2Error(f"Frozen champion identity mismatch: {exc}") from exc
    if receipt.get("source_sha256") != current_source_hash:
        mismatches.append("review_receipt.source_sha256")
    if receipt.get("split_sha256") != expected["split_sha256"]:
        mismatches.append("review_receipt.split_sha256")
    if receipt.get("dataset_sha256") != expected["dataset_sha256"]:
        mismatches.append("review_receipt.dataset_sha256")
    frozen_source = provenance.get("source_identity")
    if (
        not isinstance(frozen_source, dict)
        or frozen_source.get("source_hash") != current_source_hash
    ):
        mismatches.append("freeze.provenance.source_identity")
    if mismatches:
        raise Milestone2Error(f"Frozen champion identity mismatch: {sorted(set(mismatches))}")


def _final_experiment_id(freeze: dict[str, Any]) -> str:
    stable_id = str(freeze["champion_config"]["stable_id"])
    digest = str(freeze["configuration_hash"])[:16]
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stable_id)
    return f"final__{safe}__{digest}"


def _model_config_from_freeze(freeze: dict[str, Any]) -> ModelConfig:
    config = freeze["champion_config"]
    return ModelConfig(
        family=str(config["family"]),
        objective=str(config.get("objective", "regression")),
        hyperparams=dict(config.get("hyperparams") or {}),
        seed=int(config.get("seed", 42)),
        postprocessing=dict(config.get("postprocessing") or {"clip_negative_predictions": True}),
    )


def _feature_schema(frame: pd.DataFrame, features: list[str]) -> list[dict[str, str]]:
    return [{"name": name, "dtype": str(frame[name].dtype)} for name in features]


def _training_fit_row_digest(frame: pd.DataFrame) -> str:
    text = "\n".join(
        f"{row.district_id}|{pd.Timestamp(row.week_start_date).date().isoformat()}"
        for row in frame[["district_id", "week_start_date"]].itertuples(index=False)
    )
    return _hash_text(text)


def _verify_final_model(
    *,
    model_dir: Path,
    freeze: dict[str, Any],
    training_frame: pd.DataFrame,
    training_metadata: dict[str, Any],
    input_hashes: dict[str, str],
    frame: pd.DataFrame,
) -> None:
    config = freeze["champion_config"]
    feature_columns = list(config["feature_columns"])
    model = load_champion(model_dir)
    if model.config.serializable() != _model_config_from_freeze(freeze).serializable():
        raise Milestone2Error(f"Existing final model config does not match freeze: {model_dir}")
    if model.feature_columns != feature_columns:
        raise Milestone2Error(
            f"Existing final model feature order does not match freeze: {model_dir}"
        )
    row_identity = model.metadata.get("row_identity") or {}
    if row_identity.get("row_key_digest") != _training_fit_row_digest(training_frame):
        raise Milestone2Error(
            f"Existing final model fit row digest does not match freeze: {model_dir}"
        )
    for key, expected in input_hashes.items():
        if row_identity.get(key) != expected:
            raise Milestone2Error(
                f"Existing final model input hash mismatch for {key}: {model_dir}"
            )
    features_path = model_dir / "features.json"
    if not features_path.exists():
        raise Milestone2Error(f"Existing final model missing features.json: {model_dir}")
    features = _read_json(features_path)
    if features.get("feature_columns") != feature_columns:
        raise Milestone2Error(
            f"Existing final model feature schema does not match freeze: {model_dir}"
        )
    if features.get("schema") != _feature_schema(frame, feature_columns):
        raise Milestone2Error(f"Existing final model dataframe schema changed: {model_dir}")


def _verify_final_registry_record(paths: ModelingPaths, final_experiment_id: str) -> None:
    experiment_dir = paths.artifact_root / "experiments" / final_experiment_id
    metadata_path = experiment_dir / "metadata.json"
    registry_path = paths.artifact_root / "experiments" / "registry.parquet"
    model_path = paths.artifact_root / "models" / final_experiment_id / "model.joblib"
    if not metadata_path.exists():
        raise Milestone2Error(
            f"Final model registry metadata missing for {final_experiment_id}; refusing holdout"
        )
    if not registry_path.exists():
        raise Milestone2Error("Final model registry.parquet missing; refusing holdout")
    if not model_path.exists():
        raise Milestone2Error(f"Final model binary missing: {model_path}")
    metadata = _read_json(metadata_path)
    model_hash = sha256_file(model_path)
    if (metadata.get("artifact_sha256") or {}).get("model") != model_hash:
        raise Milestone2Error("Final model registry metadata artifact hash does not verify")
    registry = pd.read_parquet(registry_path)
    if "experiment_id" not in registry.columns:
        raise Milestone2Error("Experiment registry missing experiment_id column")
    rows = registry[registry["experiment_id"].astype(str).eq(final_experiment_id)]
    if len(rows) != 1:
        raise Milestone2Error(
            "Experiment registry must contain exactly one final model row for "
            f"{final_experiment_id}"
        )
    row = rows.iloc[0]
    if "source_hash" in row and str(row["source_hash"]) != str(metadata.get("source_hash")):
        raise Milestone2Error("Final model registry source hash does not match metadata")
    if "artifact_paths" in row:
        try:
            artifact_paths = json.loads(str(row["artifact_paths"]))
        except json.JSONDecodeError as exc:
            raise Milestone2Error("Final model registry artifact_paths is not JSON") from exc
        if str(artifact_paths.get("model")) != str(model_path):
            raise Milestone2Error("Final model registry artifact path does not match model binary")


def _write_final_model_metadata(
    *,
    model_dir: Path,
    final_experiment_id: str,
    freeze: dict[str, Any],
    frame: pd.DataFrame,
    training_metadata: dict[str, Any],
    input_hashes: dict[str, str],
) -> None:
    config = freeze["champion_config"]
    feature_columns = list(config["feature_columns"])
    _write_json(
        model_dir / "config.json",
        {
            **_model_config_from_freeze(freeze).serializable(),
            "model": config["family"],
            "config_id": config["stable_id"],
            "experiment_id": final_experiment_id,
            "configuration_hash": freeze["configuration_hash"],
            "input_hashes": input_hashes,
            "paths": {
                "dataset": "data/processed/ml_training_dataset.parquet",
                "registry": "data/reports/feature_registry.csv",
                "split": "artifacts/experiments/split_definition.json",
            },
        },
    )
    _write_json(
        model_dir / "features.json",
        {"feature_columns": feature_columns, "schema": _feature_schema(frame, feature_columns)},
    )
    _write_json(
        model_dir / "training_period.json",
        {
            "train_start": training_metadata["first_origin"],
            "train_end": training_metadata["last_origin"],
            "locked_first_origin": training_metadata["locked_first_origin"],
            "rows": training_metadata["rows"],
            "districts": training_metadata["districts"],
            "row_key_digest": training_metadata["row_key_digest"],
        },
    )
    _write_json(
        model_dir / "metrics.json",
        {
            "stage": "final_training_pre_holdout",
            "score_metrics_available": False,
            "configuration_hash": freeze["configuration_hash"],
            "training_rows": training_metadata["rows"],
        },
    )
    from dengue_forecast.modeling.registry import _library_versions

    _write_json(model_dir / "environment.json", _library_versions())
    (model_dir / "README.md").write_text(
        f"# Final model {final_experiment_id}\n\n"
        "Frozen Milestone 2 final-training artifact. `model.joblib` contains both the "
        "estimator and its fitted preprocessor; holdout scoring must call the saved bundle "
        "with the exact feature order recorded in `features.json`.\n",
        encoding="utf-8",
    )


def _ensure_final_model(
    paths: ModelingPaths, freeze: dict[str, Any], *, production: bool = True
) -> tuple[Path, dict[str, Any]]:
    registry = load_modeling_registry(paths.registry_path)
    frame = load_modeling_dataset(paths.dataset_path, registry=registry, production=production)
    split = _read_json(paths.split_path)
    training_frame, training_metadata = permitted_development_training_frame(frame, split)
    training_metadata = {
        **training_metadata,
        "fit_row_digest": _training_fit_row_digest(training_frame),
    }
    input_hashes = {
        "dataset_sha256": sha256_file(paths.dataset_path),
        "registry_sha256": sha256_file(paths.registry_path),
        "split_sha256": sha256_file(paths.split_path),
    }
    config = freeze["champion_config"]
    if config.get("allowed_train_bound", {}).get("locked_first_origin") != training_metadata[
        "locked_first_origin"
    ]:
        raise Milestone2Error("Frozen train bound does not match current split")
    frozen_bound = config.get("allowed_train_bound", {})
    for frozen_key, metadata_key in [
        ("rows", "rows"),
        ("train_start", "first_origin"),
        ("train_end", "last_origin"),
        ("row_key_digest", "row_key_digest"),
        ("fit_row_digest", "fit_row_digest"),
    ]:
        if (
            frozen_key in frozen_bound
            and frozen_bound[frozen_key] != training_metadata[metadata_key]
        ):
            raise Milestone2Error(
                f"Frozen train bound {frozen_key} does not match current training frame"
            )
    for key in ["q90", "q95"]:
        actual = float(training_metadata["thresholds"][key])
        frozen = float(config["thresholds"][key])
        if abs(actual - frozen) > 1e-12:
            raise Milestone2Error(f"Frozen threshold {key} does not match current training frame")
    final_experiment_id = _final_experiment_id(freeze)
    model_dir = paths.artifact_root / "models" / final_experiment_id
    if model_dir.exists():
        _verify_final_model(
            model_dir=model_dir,
            freeze=freeze,
            training_frame=training_frame,
            training_metadata=training_metadata,
            input_hashes=input_hashes,
            frame=frame,
        )
        _verify_final_registry_record(paths, final_experiment_id)
        _refresh_champion_copy(paths, model_dir, final_experiment_id, freeze)
        return model_dir, {"frame": frame, "split": split, "training": training_metadata}

    feature_columns = list(config["feature_columns"])
    trained = train_fold_model(
        training_frame,
        feature_columns=feature_columns,
        target_column="cases_next_week",
        config=_model_config_from_freeze(freeze),
        output_dir=model_dir,
        frozen_period_bounds={
            "train_start": training_metadata["first_origin"],
            "train_end": training_metadata["last_origin"],
        },
        provenance={
            **input_hashes,
            "configuration_hash": freeze["configuration_hash"],
            "final_experiment_id": final_experiment_id,
        },
    )
    _write_final_model_metadata(
        model_dir=model_dir,
        final_experiment_id=final_experiment_id,
        freeze=freeze,
        frame=frame,
        training_metadata=training_metadata,
        input_hashes=input_hashes,
    )
    try:
        ExperimentRegistry(paths.artifact_root, strict=False).create_experiment(
            experiment_id=final_experiment_id,
            config=_read_json(model_dir / "config.json"),
            features=_read_json(model_dir / "features.json"),
            metrics=_read_json(model_dir / "metrics.json"),
            training_period=_read_json(model_dir / "training_period.json"),
            artifact_paths={"model": trained.model_dir / "model.joblib"},
            row_identity=trained.metadata.get("row_identity"),
            runtime={"seed": trained.config.seed},
        )
    except RegistryError as exc:
        raise Milestone2Error(f"Could not persist final model registry record: {exc}") from exc
    _verify_final_registry_record(paths, final_experiment_id)
    _refresh_champion_copy(paths, model_dir, final_experiment_id, freeze)
    return model_dir, {"frame": frame, "split": split, "training": training_metadata}


def _refresh_champion_copy(
    paths: ModelingPaths,
    model_dir: Path,
    final_experiment_id: str,
    freeze: dict[str, Any],
) -> None:
    champion_dir = paths.artifact_root / "models" / "champion"
    if champion_dir.exists():
        _verify_champion_copy(champion_dir, freeze)
        return
    tmp = champion_dir.with_name(".champion.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(model_dir, tmp)
    _write_json(
        tmp / "champion_manifest.json",
        {
            "original_experiment_id": final_experiment_id,
            "configuration_hash": freeze["configuration_hash"],
            "original_model_dir": f"models/{final_experiment_id}",
        },
    )
    tmp.replace(champion_dir)


def _verify_champion_copy(champion_dir: Path, freeze: dict[str, Any]) -> None:
    manifest_path = champion_dir / "champion_manifest.json"
    if not manifest_path.exists():
        raise Milestone2Error(f"Existing champion copy missing manifest: {champion_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("configuration_hash") != freeze.get("configuration_hash"):
        raise Milestone2Error("Existing champion copy does not match frozen configuration")
    validate_model_binary_load(
        champion_dir=champion_dir,
        expected_feature_columns=list(freeze["champion_config"]["feature_columns"]),
    )


def _publish_holdout_prediction_copies(paths: ModelingPaths) -> None:
    copies = {
        paths.holdout_dir / "locked_test_predictions.parquet": (
            paths.artifact_root / "predictions" / "locked_test_model.parquet"
        ),
        paths.holdout_dir / "locked_test_baseline_predictions.parquet": (
            paths.artifact_root / "predictions" / "locked_test_baselines.parquet"
        ),
    }
    for source, target in copies.items():
        if not source.exists():
            raise Milestone2Error(f"Completed holdout prediction file missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise Milestone2Error(
                    f"Existing prediction copy differs from holdout receipt: {target}"
                )
            continue
        shutil.copy2(source, target)


def _scored_final_experiment_id(final_experiment_id: str, metrics_hash: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in final_experiment_id
    )
    return f"scored__{safe}__{metrics_hash[:16]}"


def _validation_cv_metrics_for_config(paths: ModelingPaths, config_id: str) -> dict[str, Any]:
    metrics_path = paths.artifact_root / "experiments" / "fold_metrics.parquet"
    if not metrics_path.exists() and paths.freeze_path.exists():
        freeze = _read_json(paths.freeze_path)
        frozen_config = freeze.get("champion_config") or {}
        if str(frozen_config.get("stable_id")) == str(config_id):
            folds = frozen_config.get("folds") or []
            maes = [
                float(fold["mae"])
                for fold in folds
                if fold.get("mae") is not None and pd.notna(fold.get("mae"))
            ]
            top = [
                float(fold["mae_top_decile"])
                for fold in folds
                if fold.get("mae_top_decile") is not None and pd.notna(fold.get("mae_top_decile"))
            ]
            return {
                "source": "freeze_champion_config",
                "folds": len(folds),
                "mae": float(pd.Series(maes).mean()) if maes else None,
                "mae_top_decile": float(pd.Series(top).mean()) if top else None,
            }
    metrics = _load_fold_metrics(paths)
    subset = metrics[metrics["config_id"].astype(str).eq(str(config_id))]
    if subset.empty:
        raise Milestone2Error(f"No CV metrics found for final scored config {config_id}")
    return {
        "folds": int(subset["fold"].nunique()),
        "n": int(pd.to_numeric(subset.get("n", pd.Series(dtype=float)), errors="coerce").sum()),
        "mae": float(pd.to_numeric(subset["mae"], errors="coerce").mean()),
        "rmse": float(pd.to_numeric(subset["rmse"], errors="coerce").mean()),
    }


def _read_final_training_period(paths: ModelingPaths, final_experiment_id: str) -> dict[str, Any]:
    path = paths.artifact_root / "models" / final_experiment_id / "training_period.json"
    if not path.exists():
        raise Milestone2Error(f"Final model training period missing: {path}")
    return _read_json(path)


def _publish_scored_final_artifacts(
    paths: ModelingPaths,
    *,
    freeze: dict[str, Any],
    final_experiment_id: str,
    champion_dir: Path,
    holdout_result: Any,
) -> None:
    metrics_path = paths.holdout_dir / "locked_test_metrics.json"
    predictions_path = paths.holdout_dir / "locked_test_predictions.parquet"
    baseline_predictions_path = paths.holdout_dir / "locked_test_baseline_predictions.parquet"
    for path, label in [
        (metrics_path, "holdout metrics"),
        (predictions_path, "holdout predictions"),
        (baseline_predictions_path, "holdout baseline predictions"),
    ]:
        _require_existing_file(path, label)
    holdout_metrics = _read_json(metrics_path)
    metrics_hash = sha256_file(metrics_path)
    predictions_hash = sha256_file(predictions_path)
    baseline_predictions_hash = sha256_file(baseline_predictions_path)
    original_model_path = paths.artifact_root / "models" / final_experiment_id / "model.joblib"
    original_metadata_path = (
        paths.artifact_root / "experiments" / final_experiment_id / "metadata.json"
    )
    _require_existing_file(original_model_path, "original final model binary")
    _require_existing_file(original_metadata_path, "original final model metadata")
    original_model_hash = sha256_file(original_model_path)
    original_metadata_hash = sha256_file(original_metadata_path)
    scored_id = _scored_final_experiment_id(final_experiment_id, metrics_hash)
    registry_path = paths.artifact_root / "experiments" / "registry.parquet"
    if registry_path.exists():
        registry = pd.read_parquet(registry_path)
        existing = registry[registry["experiment_id"].astype(str).eq(scored_id)]
        if len(existing) > 1:
            raise Milestone2Error(f"Duplicate scored final registry rows for {scored_id}")
        if len(existing) == 1:
            metadata_path = paths.artifact_root / "experiments" / scored_id / "metadata.json"
            if not metadata_path.exists():
                raise Milestone2Error(f"Scored final metadata missing for registry row {scored_id}")
            metadata = _read_json(metadata_path)
            artifacts = metadata.get("artifact_sha256") or {}
            expected = {
                "original_model": original_model_hash,
                "holdout_metrics": metrics_hash,
                "holdout_predictions": predictions_hash,
                "holdout_baseline_predictions": baseline_predictions_hash,
            }
            if any(artifacts.get(key) != value for key, value in expected.items()):
                raise Milestone2Error("Existing scored final registry row conflicts with receipts")
        else:
            ExperimentRegistry(paths.artifact_root, strict=False).create_experiment(
                experiment_id=scored_id,
                config={
                    "stage": "final_scored_receipt",
                    "original_prefit_experiment_id": final_experiment_id,
                    "configuration_hash": freeze["configuration_hash"],
                    "champion_config_id": freeze["champion_config"]["stable_id"],
                    "input_hashes": {
                        "dataset_sha256": sha256_file(paths.dataset_path),
                        "registry_sha256": sha256_file(paths.registry_path),
                        "split_sha256": sha256_file(paths.split_path),
                        "config_sha256": sha256_file(paths.config_path),
                    },
                },
                features={
                    "feature_columns": list(freeze["champion_config"]["feature_columns"]),
                    "schema": _read_json(champion_dir / "features.json").get("schema", []),
                },
                metrics={
                    "stage": "final_scored",
                    "score_metrics_available": True,
                    "validation": _validation_cv_metrics_for_config(
                        paths, str(freeze["champion_config"]["stable_id"])
                    ),
                    "locked_test": holdout_metrics.get("model_metrics", {}),
                    "baselines": holdout_metrics.get("baselines", {}),
                    "hash_refs": {
                        "holdout_metrics_sha256": metrics_hash,
                        "holdout_predictions_sha256": predictions_hash,
                        "holdout_baseline_predictions_sha256": baseline_predictions_hash,
                        "original_model_sha256": original_model_hash,
                        "original_prefit_metadata_sha256": original_metadata_hash,
                    },
                },
                training_period=_read_final_training_period(paths, final_experiment_id),
                artifact_paths={
                    "original_model": original_model_path,
                    "holdout_metrics": metrics_path,
                    "holdout_predictions": predictions_path,
                    "holdout_baseline_predictions": baseline_predictions_path,
                },
                row_identity={
                    "score_stage": "locked_holdout_completed",
                    "holdout_row_key_digest": holdout_result.metrics.get("row_key_digest"),
                    "state_completed_count": (holdout_result.state.get("counts") or {}).get(
                        "completed"
                    ),
                },
                runtime={"score_reuse_only": True},
            )
    stable_metrics = {
        "stage": "stable_champion_scored_copy",
        "score_metrics_available": True,
        "original_prefit_experiment_id": final_experiment_id,
        "scored_experiment_id": scored_id,
        "configuration_hash": freeze["configuration_hash"],
        "validation": _validation_cv_metrics_for_config(
            paths, str(freeze["champion_config"]["stable_id"])
        ),
        "locked_test": holdout_metrics.get("model_metrics", {}),
        "baselines": holdout_metrics.get("baselines", {}),
        "hash_refs": {
            "holdout_metrics_sha256": metrics_hash,
            "holdout_predictions_sha256": predictions_hash,
            "holdout_baseline_predictions_sha256": baseline_predictions_hash,
            "original_model_sha256": original_model_hash,
            "original_prefit_metadata_sha256": original_metadata_hash,
        },
    }
    _write_json(champion_dir / "metrics.json", stable_metrics)
    _write_json(
        champion_dir / "hashes.json",
        {
            "schema_version": 1,
            "original_prefit_experiment_id": final_experiment_id,
            **stable_metrics["hash_refs"],
        },
    )
    features = _read_json(champion_dir / "features.json")
    _write_json(
        champion_dir / "schema.json",
        {
            "feature_columns": list(freeze["champion_config"]["feature_columns"]),
            "schema": features.get("schema", []),
        },
    )


def stage_select(paths: ModelingPaths) -> dict[str, Any]:
    receipt = _review_receipt(paths)
    if paths.freeze_path.exists():
        freeze = _read_json(paths.freeze_path)
        _assert_freeze_matches_current_inputs(paths, freeze)
        selection = select_champion(_selection_candidates(paths))
        if freeze["champion_config"]["stable_id"] != selection.champion["stable_id"]:
            raise Milestone2Error("Frozen champion does not match current CV ranking")
        return {
            "champion": freeze["champion_config"]["stable_id"],
            "configuration_hash": freeze["configuration_hash"],
        }
    validate_selection_prerequisites(
        development_analysis_path=paths.reports_root / "pretest_error_summary.md",
        explainability_path=paths.explain_dir / "explainability_summary.md",
        leakage_review_path=paths.explain_dir / "leakage_audit.csv",
        cohort_coverage={"actual_cohort_coverages_clear": True},
    )
    selection = select_champion(_selection_candidates(paths))
    _frame, training_metadata = _final_training_binding(paths, production=False)
    selection = _bind_selection_to_final_training_frame(selection, training_metadata)
    payload = build_freeze_payload(
        selection=selection,
        artifact_context={
            "dataset_sha256": sha256_file(paths.dataset_path),
            "registry_sha256": sha256_file(paths.registry_path),
            "modeling_overlay_sha256": sha256_file(paths.config_path),
            "split_sha256": sha256_file(paths.split_path),
            "source_identity": _source_identity(paths.root),
        },
        approval_hash=str(receipt["approval_hash"]),
    )
    freeze = freeze_champion_config(payload, paths.freeze_path)
    _write_json(
        paths.artifact_root / "experiments" / "selection.json",
        {**selection.__dict__, "final_training_binding": training_metadata},
    )
    return {
        "champion": selection.champion["stable_id"],
        "configuration_hash": freeze["configuration_hash"],
    }


def stage_test(paths: ModelingPaths, *, production: bool = True) -> dict[str, Any]:
    if not paths.freeze_path.exists():
        raise Milestone2Error("Frozen champion config missing; model test refuses to score")
    verify_protected_inputs(paths)
    freeze = _read_json(paths.freeze_path)
    _assert_freeze_matches_current_inputs(paths, freeze)
    champion_dir, context = _ensure_final_model(paths, freeze, production=production)
    final_experiment_id = _final_experiment_id(freeze)

    def scorer(request):
        model = load_champion(champion_dir)
        if list(request.test_features.columns) != model.feature_columns:
            raise Milestone2Error("Holdout scorer received feature order different from champion")
        return model.predict_next_week(request.test_features)

    result = score_locked_holdout_once(
        freeze_path=paths.freeze_path,
        frame=context["frame"],
        split=context["split"],
        output_dir=paths.holdout_dir,
        approval_hash=str(freeze["approval_hash"]),
        scorer=scorer,
        champion_dir=champion_dir,
    )
    _publish_holdout_prediction_copies(paths)
    champion_copy = paths.artifact_root / "models" / "champion"
    validate_model_binary_load(
        champion_dir=champion_copy,
        expected_feature_columns=list(freeze["champion_config"]["feature_columns"]),
    )
    _publish_scored_final_artifacts(
        paths,
        freeze=freeze,
        final_experiment_id=final_experiment_id,
        champion_dir=champion_copy,
        holdout_result=result,
    )
    return {
        "status": result.state["status"],
        "metrics_path": str(result.metrics_path),
        "champion_dir": str(champion_copy),
    }


def stage_final_reports(paths: ModelingPaths) -> dict[str, Any]:
    summary = generate_milestone2_summary(
        paths.root,
        artifact_root=paths.artifact_root,
        reports_root=paths.reports_root,
    )
    card = generate_champion_model_card(
        paths.root,
        artifact_root=paths.artifact_root,
        reports_root=paths.reports_root,
    )
    return {"summary": str(summary), "model_card": str(card)}


def _require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise Milestone2Error(f"Required {label} missing: {path}")


def _validate_cv_metrics(paths: ModelingPaths, metrics: pd.DataFrame) -> None:
    required = {
        "config_id",
        "fold",
        "mae",
        "rmse",
        "model",
        "feature_set",
        "experiment_id",
        "stage",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise Milestone2Error(f"Fold metrics missing required columns: {missing}")
    split = _read_json(paths.split_path)
    expected_folds = {str(fold["fold_id"]) for fold in split.get("folds") or []}
    if len(expected_folds) != 6:
        raise Milestone2Error(f"Expected 6 completed CV folds, found {len(expected_folds)}")
    observed_folds = set(metrics["fold"].dropna().astype(str))
    if observed_folds != expected_folds:
        raise Milestone2Error("Fold metrics do not cover the frozen 6-fold split")
    nonfinite = pd.to_numeric(metrics["mae"], errors="coerce").isna() | pd.to_numeric(
        metrics["rmse"], errors="coerce"
    ).isna()
    if bool(nonfinite.any()):
        raise Milestone2Error("Fold metrics contain missing/non-numeric MAE or RMSE")
    completed = metrics.groupby("config_id")["fold"].nunique()
    if completed.empty or int(completed.max()) != 6:
        raise Milestone2Error("No model configuration has complete 6-fold metrics")
    families = set(metrics["model"].dropna().astype(str))
    required_families = {"ridge", "poisson", "random_forest", "xgboost", "lightgbm"}
    missing_families = sorted(required_families - families)
    if missing_families:
        raise Milestone2Error(f"Required model family attempts missing: {missing_families}")
    feature_sets = set(metrics["feature_set"].dropna().astype(str))
    missing_sets = sorted(set(FEATURE_SET_NAMES) - feature_sets)
    if missing_sets:
        raise Milestone2Error(f"Fixed four-set ablation coverage missing: {missing_sets}")


def _validate_model_prediction_contract(paths: ModelingPaths, predictions: pd.DataFrame) -> None:
    required = set(PREDICTION_KEY_COLUMNS) | {
        "week_end_date",
        "target",
        "prediction",
        "model",
        "feature_set",
        "experiment_id",
        "threshold90",
        "threshold95",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise Milestone2Error(f"Model predictions missing required columns: {missing}")
    if predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise Milestone2Error("Model predictions are not unique by config/fold/row")
    dates = pd.to_datetime(predictions["week_start_date"], errors="coerce")
    targets = pd.to_numeric(predictions["target"], errors="coerce")
    preds = pd.to_numeric(predictions["prediction"], errors="coerce")
    if dates.isna().any() or targets.isna().any() or preds.isna().any():
        raise Milestone2Error("Model predictions contain invalid dates, targets, or predictions")
    if bool(targets.lt(0).any()) or bool(preds.lt(0).any()):
        raise Milestone2Error("Model predictions contain negative targets or predictions")
    if not bool(np.isfinite(targets.to_numpy(dtype="float64")).all()) or not bool(
        np.isfinite(preds.to_numpy(dtype="float64")).all()
    ):
        raise Milestone2Error("Model predictions contain non-finite targets or predictions")
    expected_rows = _expected_validation_row_count(paths)
    per_config = predictions.groupby("config_id").size()
    incomplete = per_config[per_config.ne(expected_rows)]
    if not incomplete.empty:
        raise Milestone2Error(
            "Model predictions must preserve every validation row for each config; "
            f"expected {expected_rows}, incomplete={incomplete.head().to_dict()}"
        )
    metrics = _load_fold_metrics(paths)
    expected_config_count = int(metrics["config_id"].nunique())
    if int(per_config.size) != expected_config_count:
        raise Milestone2Error("Model prediction config coverage does not match fold metrics")
    expected = _expected_validation_rows(paths)
    expected_keys = expected[["fold", "district_id", "week_start_date"]].copy()
    expected_key_set = set(map(tuple, expected_keys.to_numpy()))
    pred = predictions.copy()
    pred["config_id"] = pred["config_id"].astype(str)
    pred["fold"] = pred["fold"].astype(str)
    pred["district_id"] = pred["district_id"].astype(str)
    pred["week_start_date"] = pd.to_datetime(pred["week_start_date"]).dt.date.astype(str)
    observed_folds = set(pred["fold"].dropna().astype(str))
    expected_folds = set(expected["fold"].dropna().astype(str))
    for config_id, group in pred.groupby("config_id", sort=True):
        if set(group["fold"].astype(str)) != expected_folds:
            raise Milestone2Error(f"Config {config_id} does not contain all 6 validation folds")
        observed_key_set = set(
            map(tuple, group[["fold", "district_id", "week_start_date"]].to_numpy())
        )
        if observed_key_set != expected_key_set:
            raise Milestone2Error(f"Config {config_id} validation row keys do not match split")
        merged = group.merge(
            expected,
            on=["fold", "district_id", "week_start_date"],
            how="left",
            validate="1:1",
        )
        if merged["expected_target"].isna().any():
            raise Milestone2Error(f"Config {config_id} has row keys outside expected cohort")
        delta = (
            pd.to_numeric(merged["target"], errors="coerce").astype("float64")
            - pd.to_numeric(merged["expected_target"], errors="coerce").astype("float64")
        ).abs()
        if bool(delta.gt(1e-12).any()):
            raise Milestone2Error(f"Config {config_id} target values differ from dataset")
    if observed_folds != expected_folds:
        raise Milestone2Error("Model prediction folds do not match frozen split")


def _validate_ablation_report_artifacts(paths: ModelingPaths) -> None:
    results_path = paths.reports_root / "ablation_results.csv"
    fold_path = paths.reports_root / "ablation_fold_metrics.csv"
    summary_path = paths.reports_root / "ablation_summary.csv"
    for path, label in [
        (results_path, "aggregate ablation results"),
        (fold_path, "screen ablation fold metrics"),
        (summary_path, "ablation summary"),
    ]:
        _require_existing_file(path, label)
    results = pd.read_csv(results_path)
    fold_metrics = pd.read_csv(fold_path)
    summary = pd.read_csv(summary_path)
    if len(results) != 28:
        raise Milestone2Error(
            f"ablation_results.csv must contain 28 aggregate rows, got {len(results)}"
        )
    if len(fold_metrics) != 168:
        raise Milestone2Error(
            f"ablation_fold_metrics.csv must contain 168 screen rows, got {len(fold_metrics)}"
        )
    if summary.empty:
        raise Milestone2Error("ablation_summary.csv is empty")
    if set(fold_metrics.get("stage", pd.Series(dtype=str)).astype(str)) != {"screen"}:
        raise Milestone2Error("ablation_fold_metrics.csv must contain only screen rows")
    grouped = fold_metrics.groupby(["model", "objective", "config_variant", "feature_set"])[
        "fold"
    ].nunique()
    if not grouped.eq(6).all():
        raise Milestone2Error("Each ablation variant-feature group must contain six folds")
    set_counts = fold_metrics.groupby(["model", "objective", "config_variant"])[
        "feature_set"
    ].nunique()
    if not set_counts.eq(4).all():
        raise Milestone2Error("Each ablation variant must contain all four feature sets")


def _validate_analysis_report_artifacts(paths: ModelingPaths) -> None:
    required_reports = [
        paths.reports_root / "district_error_analysis.csv",
        paths.reports_root / "temporal_error_analysis.csv",
        paths.reports_root / "outbreak_error_analysis.csv",
        paths.reports_root / "largest_error_cases.csv",
        paths.reports_root / "missingness_sensitivity.csv",
        paths.reports_root / "model_results.csv",
        paths.reports_root / "primary_validation_table.csv",
        paths.reports_root / "feature_importance.csv",
        paths.reports_root / "explainability_summary.md",
        paths.explain_dir / "shap_mean_abs_values.csv",
        paths.reports_root / "paired_baseline_comparison.csv",
        paths.reports_root / "common_support_comparison.csv",
        paths.root / "docs" / "evidence" / "milestone2-cv-source.tar.gz",
    ]
    plot_names = [
        "predicted_vs_actual_scatter.png",
        "residual_distribution.png",
        "mae_by_district.png",
        "mae_by_month.png",
        "baseline_vs_champion.png",
        "baseline_vs_champion_input.csv",
        "baseline_availability.csv",
    ]
    required_reports.extend(paths.plot_dir / name for name in plot_names)
    missing = [str(path) for path in required_reports if not path.exists()]
    if missing:
        raise Milestone2Error(f"Required analysis/report artifacts missing: {missing}")
    paired = pd.read_csv(paths.reports_root / "paired_baseline_comparison.csv")
    common = pd.read_csv(paths.reports_root / "common_support_comparison.csv")
    if paired.empty or common.empty:
        raise Milestone2Error("Shared-support baseline comparison CSVs must be non-empty")
    if set(common["support_type"].astype(str)) != {"secondary_all_five_common_intersection"}:
        raise Milestone2Error("Common-support comparison must be labelled secondary")


def _validate_experiment_registry_integrity(paths: ModelingPaths, metrics: pd.DataFrame) -> None:
    registry_path = paths.artifact_root / "experiments" / "registry.parquet"
    _require_existing_file(registry_path, "experiment registry")
    registry = pd.read_parquet(registry_path)
    if "experiment_id" not in registry.columns:
        raise Milestone2Error("Experiment registry missing experiment_id")
    expected_ids = set(metrics["experiment_id"].dropna().astype(str))
    registry_ids = set(registry["experiment_id"].dropna().astype(str))
    missing_registry = sorted(expected_ids - registry_ids)
    if missing_registry:
        raise Milestone2Error(f"Experiment registry missing CV rows: {missing_registry[:5]}")
    feature_names = set(load_modeling_registry(paths.registry_path)["feature_name"].astype(str))
    for experiment_id in sorted(expected_ids):
        experiment_dir = paths.artifact_root / "experiments" / experiment_id
        model_dir = paths.artifact_root / "models" / experiment_id
        metadata_path = experiment_dir / "metadata.json"
        features_path = experiment_dir / "features.json"
        model_path = model_dir / "model.joblib"
        for path, label in [
            (metadata_path, "experiment metadata"),
            (features_path, "experiment features"),
            (model_path, "experiment model binary"),
        ]:
            _require_existing_file(path, f"{label} for {experiment_id}")
        metadata = _read_json(metadata_path)
        artifact_hashes = metadata.get("artifact_sha256") or {}
        if artifact_hashes.get("model") != sha256_file(model_path):
            raise Milestone2Error(f"Model hash mismatch for experiment {experiment_id}")
        if not metadata.get("source_hash"):
            raise Milestone2Error(f"Experiment {experiment_id} missing recorded source hash")
        features = _read_json(features_path)
        declared = set(map(str, features.get("feature_columns") or []))
        if not declared or not declared <= feature_names:
            raise Milestone2Error(f"Experiment {experiment_id} declares unknown features")


def _validate_baseline_evidence(paths: ModelingPaths) -> None:
    metrics_path = paths.artifact_root / "reports" / "baseline_results.csv"
    predictions_path = paths.artifact_root / "predictions" / "baselines.parquet"
    canonical_copy = paths.reports_root / "baseline_results.csv"
    for path, label in [
        (metrics_path, "baseline metrics"),
        (predictions_path, "baseline predictions"),
        (canonical_copy, "canonical baseline report"),
    ]:
        _require_existing_file(path, label)
    if sha256_file(metrics_path) != sha256_file(canonical_copy):
        raise Milestone2Error("Canonical baseline report copy differs from artifact report")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_parquet(predictions_path)
    observed_metrics = set(metrics.get("model", pd.Series(dtype=str)).dropna().astype(str))
    observed_predictions = set(
        predictions.get("model", pd.Series(dtype=str)).dropna().astype(str)
    )
    if observed_metrics != set(BASELINE_NAMES) or observed_predictions != set(BASELINE_NAMES):
        raise Milestone2Error("Baseline evidence must cover all five canonical baselines")
    required_metric_cols = {"model", "fold", "total", "n", "coverage", "mae", "rmse"}
    missing_metric_cols = sorted(required_metric_cols - set(metrics.columns))
    if missing_metric_cols:
        raise Milestone2Error(f"Baseline metrics missing columns: {missing_metric_cols}")
    expected_rows = _expected_validation_row_count(paths)
    for name in BASELINE_NAMES:
        subset = predictions[predictions["model"].astype(str).eq(name)]
        if len(subset) != expected_rows:
            raise Milestone2Error(
                f"Baseline {name} predictions must retain full validation cohort rows"
            )
        metric_subset = metrics[metrics["model"].astype(str).eq(name)]
        if metric_subset["fold"].nunique() != 6:
            raise Milestone2Error(f"Baseline {name} does not cover all 6 folds")
        if not metric_subset["coverage"].between(0, 1).all():
            raise Milestone2Error(f"Baseline {name} coverage outside [0, 1]")


def _validate_holdout_report_artifacts(paths: ModelingPaths, freeze: dict[str, Any]) -> None:
    required = [
        paths.holdout_dir / "locked_test_metrics.json",
        paths.holdout_dir / "locked_test_predictions.parquet",
        paths.holdout_dir / "locked_test_baseline_predictions.parquet",
        paths.artifact_root / "predictions" / "locked_test_model.parquet",
        paths.artifact_root / "predictions" / "locked_test_baselines.parquet",
        paths.reports_root / "champion_model_card.md",
        paths.reports_root / "milestone_2_summary.md",
    ]
    for path in required:
        _require_existing_file(path, "post-freeze artifact")
    if sha256_file(paths.holdout_dir / "locked_test_predictions.parquet") != sha256_file(
        paths.artifact_root / "predictions" / "locked_test_model.parquet"
    ):
        raise Milestone2Error("Published model prediction copy differs from holdout receipt")
    if sha256_file(
        paths.holdout_dir / "locked_test_baseline_predictions.parquet"
    ) != sha256_file(paths.artifact_root / "predictions" / "locked_test_baselines.parquet"):
        raise Milestone2Error("Published baseline prediction copy differs from holdout receipt")
    _validate_model_card(paths.reports_root / "champion_model_card.md", freeze)
    _validate_final_summary(paths.reports_root / "milestone_2_summary.md")


def _markdown_sections(path: Path) -> set[str]:
    sections: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sections.add(line.removeprefix("## ").strip())
    return sections


def _validate_model_card(path: Path, freeze: dict[str, Any]) -> None:
    sections = _markdown_sections(path)
    required = {"Intended Use", "Model", "Metrics", "Artifacts", "Feature Schema", "Known Limits"}
    missing = sorted(required - sections)
    if missing:
        raise Milestone2Error(f"Champion model card missing required sections: {missing}")
    text = path.read_text(encoding="utf-8")
    config = freeze["champion_config"]
    for value in [config["stable_id"], config["family"], freeze["configuration_hash"]]:
        if str(value) not in text:
            raise Milestone2Error("Champion model card missing frozen config/hash details")


def _validate_final_summary(path: Path) -> None:
    from dengue_forecast.reports.milestone2 import REQUIRED_SUMMARY_SECTIONS

    sections = _markdown_sections(path)
    missing = sorted(set(REQUIRED_SUMMARY_SECTIONS) - sections)
    if missing:
        raise Milestone2Error(f"Milestone 2 summary missing required sections: {missing}")
    text = path.read_text(encoding="utf-8")
    required_tokens = [
        "SHA256",
        "Locked Test",
        "Known",
        "configuration",
        "metric",
    ]
    absent = [token for token in required_tokens if token not in text]
    if absent:
        raise Milestone2Error(f"Milestone 2 summary lacks meaningful content: {absent}")


def validate_milestone2(paths: ModelingPaths) -> dict[str, Any]:
    protected = verify_protected_inputs(paths)
    _require_existing_file(paths.review_receipt_path, "independent review receipt")
    _require_existing_file(paths.freeze_path, "frozen champion config")
    split = _read_json(paths.split_path)
    expected_split = attach_hashes(
        build_temporal_splits(
            load_modeling_dataset(
                paths.dataset_path,
                registry=load_modeling_registry(paths.registry_path),
                production=True,
            ),
            SplitPolicy(**split["policy"]),
        ),
        dataset_path=paths.dataset_path,
        registry_path=paths.registry_path,
        config_path=paths.config_path,
        policy_path=paths.policy_path,
    )
    if split != expected_split:
        raise Milestone2Error("Frozen split does not match current actual-date inputs and policy")
    metrics = _load_fold_metrics(paths)
    if metrics.empty:
        raise Milestone2Error("No fold metrics recorded")
    _validate_cv_metrics(paths, metrics)
    predictions = _all_model_predictions(paths)
    _validate_model_prediction_contract(paths, predictions)
    _validate_baseline_evidence(paths)
    _validate_ablation_report_artifacts(paths)
    _validate_analysis_report_artifacts(paths)
    _validate_experiment_registry_integrity(paths, metrics)
    registry = load_modeling_registry(paths.registry_path)
    for feature_set in FEATURE_SET_NAMES:
        get_feature_set(feature_set, registry)
    selection = select_champion(_selection_candidates(paths))
    freeze = _read_json(paths.freeze_path)
    _assert_freeze_matches_current_inputs(paths, freeze)
    if freeze["champion_config"]["stable_id"] != selection.champion["stable_id"]:
        raise Milestone2Error("Frozen champion does not match current CV ranking")
    champion_dir = paths.artifact_root / "models" / "champion"
    validate_model_binary_load(
        champion_dir=champion_dir,
        expected_feature_columns=list(freeze["champion_config"]["feature_columns"]),
    )
    validate_holdout_run(freeze_path=paths.freeze_path, output_dir=paths.holdout_dir)
    _validate_holdout_report_artifacts(paths, freeze)
    return {
        "protected_inputs": protected,
        "fold_metric_rows": len(metrics),
        "prediction_rows": len(predictions),
    }


def run_milestone2(paths: ModelingPaths, *, production: bool = True) -> dict[str, Any]:
    if paths.freeze_path.exists() and (paths.holdout_dir / "holdout_state.json").exists():
        stage_test(paths, production=production)
        stage_final_reports(paths)
        return {"status": "completed_release_reused"}
    stage_readiness(paths)
    stage_splits(paths)
    stage_baselines(paths, production=production)
    stage_train(paths, production=production)
    stage_ablation(paths)
    stage_tune(paths, production=production)
    stage_analyze(paths)
    stage_explain(paths)
    try:
        stage_select(paths)
    except SelectionError as exc:
        raise Milestone2Error(
            "Development artifacts complete; fail-closed awaiting independent review receipt. "
            f"{exc}"
        ) from exc
    return {"status": "frozen_awaiting_authorized_holdout"}


__all__ = [
    "Milestone2Error",
    "ModelingPaths",
    "run_milestone2",
    "stage_ablation",
    "stage_analyze",
    "stage_baselines",
    "stage_explain",
    "stage_final_reports",
    "stage_readiness",
    "stage_select",
    "stage_splits",
    "stage_test",
    "stage_train",
    "stage_tune",
    "validate_milestone2",
    "verify_protected_inputs",
]
