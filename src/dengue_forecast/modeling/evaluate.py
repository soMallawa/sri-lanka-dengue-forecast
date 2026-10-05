from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.modeling.baselines import BASELINE_NAMES, baseline_predict
from dengue_forecast.modeling.dataset import (
    FEATURE_SET_NAMES,
    TARGET_COLUMN,
    get_feature_set,
    load_modeling_dataset,
    load_modeling_registry,
)
from dengue_forecast.modeling.metrics import (
    high_incidence_metrics,
    top_quantile_threshold,
)
from dengue_forecast.modeling.metrics import metric_bundle as canonical_metric_bundle
from dengue_forecast.modeling.registry import ExperimentRegistry, RegistryError, _source_identity
from dengue_forecast.modeling.train import ModelConfig, load_champion, train_fold_model
from dengue_forecast.utils.hashing import sha256_file

ROW_KEY_COLUMNS = ["district_id", "week_start_date"]
PREDICTION_KEY_COLUMNS = ["config_id", "fold", "district_id", "week_start_date"]


class EvaluationError(ValueError):
    """Raised when development-only CV execution would violate policy."""


@dataclass(frozen=True)
class FoldFrames:
    fold: dict[str, Any]
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class EvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    artifacts: dict[str, Any]


@dataclass
class DevelopmentCVContext:
    frame: pd.DataFrame
    registry: pd.DataFrame
    split: dict[str, Any]
    dataset_path: Path
    registry_path: Path
    split_path: Path
    dataset_sha256: str
    registry_sha256: str
    split_sha256: str

    @property
    def folds(self) -> list[dict[str, Any]]:
        return list(self.split["folds"])

    @property
    def locked_test_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.split["locked_test_start"])

    def fold_frames(self, fold: dict[str, Any]) -> FoldFrames:
        return _fold_frames(self.frame, fold, locked_test_start=self.locked_test_start)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_frame(
    dataset_path: Path, registry: pd.DataFrame, *, production: bool
) -> pd.DataFrame:
    if production:
        return load_modeling_dataset(dataset_path, registry=registry, production=True)
    frame = pd.read_parquet(dataset_path).copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    frame["week_end_date"] = pd.to_datetime(frame["week_end_date"])
    return frame.sort_values(ROW_KEY_COLUMNS).reset_index(drop=True)


def load_development_context(
    *,
    dataset_path: str | Path,
    registry_path: str | Path,
    split_path: str | Path,
    production: bool = True,
) -> DevelopmentCVContext:
    dataset = Path(dataset_path)
    registry_file = Path(registry_path)
    split_file = Path(split_path)
    registry = load_modeling_registry(registry_file)
    frame = _load_frame(dataset, registry, production=production)
    split = json.loads(split_file.read_text(encoding="utf-8"))
    context = DevelopmentCVContext(
        frame=frame,
        registry=registry,
        split=split,
        dataset_path=dataset,
        registry_path=registry_file,
        split_path=split_file,
        dataset_sha256=sha256_file(dataset),
        registry_sha256=sha256_file(registry_file),
        split_sha256=sha256_file(split_file),
    )
    _validate_split_hashes(context)
    _validate_context(context)
    return context


def _validate_split_hashes(context: DevelopmentCVContext) -> None:
    hashes = context.split.get("hashes") or {}
    expected = {
        "dataset": context.dataset_sha256,
        "registry": context.registry_sha256,
    }
    for key, actual in expected.items():
        recorded = hashes.get(key)
        if recorded and recorded != actual:
            raise EvaluationError(f"Frozen split {key} hash does not match current input")


def _validate_context(context: DevelopmentCVContext) -> None:
    frame = context.frame
    missing = {
        "district_id",
        "week_start_date",
        "week_end_date",
        "is_trainable",
        TARGET_COLUMN,
        "dengue_cases",
    } - set(frame.columns)
    if missing:
        raise EvaluationError(f"Dataset missing required evaluation columns: {sorted(missing)}")
    if frame.duplicated(ROW_KEY_COLUMNS).any():
        raise EvaluationError("Dataset contains duplicate primary row keys")
    locked_start = context.locked_test_start
    for fold in context.folds:
        frames = context.fold_frames(fold)
        _assert_count(fold, "rows_train", len(frames.train))
        _assert_count(fold, "rows_validation", len(frames.validation))
        if not frames.train.empty and frames.train["week_start_date"].ge(locked_start).any():
            raise EvaluationError("Locked rows appear in training")
        if (
            not frames.validation.empty
            and frames.validation["week_start_date"].ge(locked_start).any()
        ):
            raise EvaluationError("Locked rows appear in validation")


def _assert_count(fold: dict[str, Any], key: str, actual: int) -> None:
    expected = int(fold[key])
    if expected != actual:
        raise EvaluationError(
            f"Fold {fold['fold_id']} {key} mismatch: serialized={expected}, actual={actual}"
        )


def _target_information_end(frame: pd.DataFrame, horizon_weeks: int) -> pd.Series:
    return pd.to_datetime(frame["week_end_date"]) + pd.Timedelta(days=7 * horizon_weeks)


def _fold_frames(
    frame: pd.DataFrame, fold: dict[str, Any], *, locked_test_start: pd.Timestamp
) -> FoldFrames:
    horizon_weeks = int(fold.get("horizon_weeks", 1))
    train_start = pd.Timestamp(fold["train_start"])
    train_end = pd.Timestamp(fold["train_end"])
    validation_start = pd.Timestamp(fold["validation_start"])
    validation_end = pd.Timestamp(fold["validation_end"])
    train_target_limit = pd.Timestamp(fold["train_target_end_max"])
    validation_target_limit = pd.Timestamp(fold["validation_target_end_max"])
    target_end = _target_information_end(frame, horizon_weeks)
    eligible = frame["is_trainable"].astype(bool) & frame[TARGET_COLUMN].notna()
    train_mask = (
        eligible
        & pd.to_datetime(frame["week_start_date"]).ge(train_start)
        & pd.to_datetime(frame["week_end_date"]).le(train_end)
        & target_end.le(train_target_limit)
        & target_end.lt(validation_start)
    )
    validation_mask = (
        eligible
        & pd.to_datetime(frame["week_start_date"]).ge(validation_start)
        & pd.to_datetime(frame["week_end_date"]).le(validation_end)
        & target_end.le(validation_target_limit)
        & target_end.lt(locked_test_start)
    )
    return FoldFrames(
        fold=fold,
        train=frame.loc[train_mask].copy(),
        validation=frame.loc[validation_mask].copy(),
    )


def metric_bundle(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    threshold90: float | None = None,
    threshold95: float | None = None,
) -> dict[str, float | int]:
    try:
        out = dict(canonical_metric_bundle(y_true, y_pred, top_decile_threshold=threshold90))
        if threshold90 is None:
            out["mae_top_90"] = float("nan")
        else:
            out["mae_top_90"] = out["mae_top_decile"]
        if threshold95 is None:
            out["mae_top_95"] = float("nan")
        else:
            high = high_incidence_metrics(
                y_true,
                y_pred,
                top_10_threshold=threshold90 if threshold90 is not None else threshold95,
                top_5_threshold=threshold95,
            )
            out["mae_top_95"] = high["mae_top_5pct"]
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    return out


def evaluate_baselines(
    context: DevelopmentCVContext, *, output_root: str | Path
) -> EvaluationResult:
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in context.folds:
        fold_frames = context.fold_frames(fold)
        observed_train = context.frame[
            pd.to_datetime(context.frame["week_start_date"]).ge(pd.Timestamp(fold["train_start"]))
            & pd.to_datetime(context.frame["week_end_date"]).le(pd.Timestamp(fold["train_end"]))
            & pd.to_datetime(context.frame["week_start_date"]).lt(
                pd.Timestamp(fold["validation_start"])
            )
            & context.frame["dengue_cases"].notna()
        ].copy()
        history = context.frame[
            pd.to_datetime(context.frame["week_start_date"]) <= pd.Timestamp(fold["validation_end"])
        ].copy()
        wide = baseline_predict(
            training_frame=observed_train,
            validation_frame=fold_frames.validation,
            full_history=history,
        )
        threshold90 = top_quantile_threshold(fold_frames.train[TARGET_COLUMN], quantile=0.9)
        threshold95 = top_quantile_threshold(fold_frames.train[TARGET_COLUMN], quantile=0.95)
        for name in BASELINE_NAMES:
            pred = wide[name]
            pred_frame = _prediction_frame(
                fold_frames.validation,
                prediction=pred,
                model=name,
                feature_set="baseline",
                fold_id=fold["fold_id"],
                experiment_id=f"baseline__{name}__{fold['fold_id']}",
                config_id=name,
                threshold90=threshold90,
                threshold95=threshold95,
                context=context,
            )
            rows.append(pred_frame)
            available = pred.notna() & np.isfinite(pred.astype("float64"))
            metrics = {
                "model": name,
                "feature_set": "baseline",
                "fold": fold["fold_id"],
                "total": int(len(pred)),
                "n": int(available.sum()),
                "coverage": float(available.mean()) if len(pred) else 0.0,
            }
            if available.any():
                metrics.update(
                    metric_bundle(
                        fold_frames.validation.loc[available, TARGET_COLUMN],
                        pred.loc[available],
                        threshold90=threshold90,
                        threshold95=threshold95,
                    )
                )
            else:
                metrics.update(_empty_metrics())
            metric_rows.append(metrics)
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    output = Path(output_root)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions" / "baselines.parquet", index=False)
    metrics.to_csv(output / "reports" / "baseline_results.csv", index=False)
    return EvaluationResult(metrics=metrics, predictions=predictions, artifacts={})


def _prediction_frame(
    validation: pd.DataFrame,
    *,
    prediction: pd.Series | np.ndarray,
    model: str,
    feature_set: str,
    fold_id: str,
    experiment_id: str,
    config_id: str,
    threshold90: float,
    threshold95: float,
    context: DevelopmentCVContext,
) -> pd.DataFrame:
    out = validation[
        ["district_id", "week_start_date", "week_end_date", TARGET_COLUMN]
    ].rename(columns={TARGET_COLUMN: "target"})
    out = out.copy()
    out["prediction"] = np.asarray(prediction, dtype="float64")
    out["model"] = model
    out["feature_set"] = feature_set
    out["fold"] = fold_id
    out["experiment_id"] = experiment_id
    out["config_id"] = config_id
    out["threshold90"] = threshold90
    out["threshold95"] = threshold95
    out["dataset_sha256"] = context.dataset_sha256
    out["registry_sha256"] = context.registry_sha256
    out["split_sha256"] = context.split_sha256
    return out


def rank_full_coverage_baseline(metrics: pd.DataFrame) -> str | None:
    if metrics.empty:
        return None
    totals = metrics.groupby("model")["coverage"].min()
    full = totals[totals.eq(1.0)].index
    if len(full) == 0:
        return None
    fold_mean = metrics[metrics["model"].isin(full)].groupby("model")["mae"].mean()
    return str(fold_mean.sort_values(kind="mergesort").index[0])


def evaluate_model_config(
    context: DevelopmentCVContext,
    *,
    config_id: str,
    model_config: ModelConfig,
    feature_set: str,
    experiment_id: str,
    output_root: str | Path,
    stage: str = "screen",
) -> EvaluationResult:
    if feature_set not in FEATURE_SET_NAMES:
        raise EvaluationError(f"Unknown feature set: {feature_set}")
    output = Path(output_root)
    refuse_development_after_holdout(output)
    model_root = output / "models"
    registry = ExperimentRegistry(output)
    features = get_feature_set(feature_set, context.registry)
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    artifact_rows: dict[str, Any] = {}
    for fold in context.folds:
        started = time.monotonic()
        fold_id = str(fold["fold_id"])
        fold_frames = context.fold_frames(fold)
        threshold90 = float(fold_frames.train[TARGET_COLUMN].quantile(0.9))
        threshold95 = float(fold_frames.train[TARGET_COLUMN].quantile(0.95))
        fold_experiment_id = f"{experiment_id}__{fold_id}"
        model_dir = model_root / fold_experiment_id
        existing = _load_completed_experiment(
            output=output,
            experiment_id=fold_experiment_id,
            context=context,
            config_id=config_id,
            model_config=model_config,
            feature_set=feature_set,
            feature_columns=features,
            fold=fold,
        )
        if existing is not None:
            pred_frame, metrics = existing
            rows.append(pred_frame)
            metric_rows.append(metrics)
            artifact_rows[fold_id] = {"model_dir": str(model_dir)}
            continue

        if model_dir.exists():
            raise EvaluationError(
                f"Incomplete or incompatible model directory exists before training: {model_dir}"
            )

        try:
            trained = train_fold_model(
                fold_frames.train,
                feature_columns=features,
                target_column=TARGET_COLUMN,
                config=model_config,
                output_dir=model_dir,
                frozen_period_bounds={
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                },
                provenance={
                    "fold_id": fold_id,
                    "dataset_sha256": context.dataset_sha256,
                    "registry_sha256": context.registry_sha256,
                    "split_sha256": context.split_sha256,
                },
            )
            predictions = trained.predict_next_week(fold_frames.validation.loc[:, features])
            pred_frame = _prediction_frame(
                fold_frames.validation,
                prediction=predictions,
                model=model_config.family,
                feature_set=feature_set,
                fold_id=fold_id,
                experiment_id=fold_experiment_id,
                config_id=config_id,
                threshold90=threshold90,
                threshold95=threshold95,
                context=context,
            )
        except Exception as exc:
            if model_dir.exists() and not (model_dir / "model.joblib").exists():
                raise EvaluationError(
                    f"Training failed before complete model artifact: {model_dir}"
                ) from exc
            raise
        rows.append(pred_frame)
        metrics = {
            "model": model_config.family,
            "objective": model_config.objective,
            "feature_set": feature_set,
            "fold": fold_id,
            "config_id": config_id,
            "experiment_id": fold_experiment_id,
            "stage": stage,
            "total": int(len(pred_frame)),
            "threshold90": threshold90,
            "threshold95": threshold95,
        }
        metrics.update(
            metric_bundle(
                pred_frame["target"],
                pred_frame["prediction"],
                threshold90=threshold90,
                threshold95=threshold95,
            )
        )
        metrics["poisson"] = metrics["poisson_deviance"]
        metrics["topdecile"] = metrics["mae_top_decile"]
        metric_rows.append(metrics)
        runtime = {"seconds": round(time.monotonic() - started, 6)}
        try:
            record = registry.create_experiment(
                experiment_id=fold_experiment_id,
                config={
                    **model_config.serializable(),
                    "model": model_config.family,
                    "config_id": config_id,
                    "feature_set": feature_set,
                    "dataset_sha256": context.dataset_sha256,
                    "dataset_path": str(context.dataset_path),
                    "input_hashes": _input_hashes(context),
                    "registry_sha256": context.registry_sha256,
                    "split_sha256": context.split_sha256,
                    "split": _fold_contract(fold),
                },
                features={
                    "feature_set": feature_set,
                    "feature_columns": features,
                    "schema": _feature_schema(context.frame, features),
                    "prediction_schema": _prediction_schema(pred_frame),
                },
                metrics=metrics,
                training_period={
                    "fold": fold_id,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "validation_start": fold["validation_start"],
                    "validation_end": fold["validation_end"],
                    "locked_test_start": fold["locked_test_start"],
                },
                artifact_paths={"model": model_dir / "model.joblib"},
                row_identity={
                    "fold_id": fold_id,
                    "validation_row_count": len(pred_frame),
                    "validation_row_key_digest": _row_key_digest(pred_frame),
                },
                runtime={**runtime, "seed": model_config.seed},
                predictions=pred_frame,
            )
            _mirror_registry_metadata_to_model_dir(record.experiment_dir, model_dir)
            artifact_rows[fold_id] = {"experiment_dir": str(record.experiment_dir)}
        except RegistryError as exc:
            if not _is_completed_experiment(output, fold_experiment_id, context, config_id):
                raise
            raise EvaluationError(
                f"Registry conflict for {fold_experiment_id}; existing experiment was not "
                "validated before fit and will not be overwritten"
            ) from exc
        artifact_rows[fold_id] = {**artifact_rows.get(fold_id, {}), "model_dir": str(model_dir)}
    predictions_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics_out = pd.DataFrame(metric_rows)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    existing_predictions = output / "predictions" / "models.parquet"
    _append_parquet(predictions_out, existing_predictions)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "experiments").mkdir(parents=True, exist_ok=True)
    metrics_path = output / "experiments" / "fold_metrics.parquet"
    _append_parquet(metrics_out, metrics_path, key_columns=["experiment_id"])
    summary_path = output / "reports" / "model_results.csv"
    all_metrics = pd.read_parquet(metrics_path) if metrics_path.exists() else metrics_out
    summarize_model_results(all_metrics).to_csv(summary_path, index=False)
    return EvaluationResult(
        metrics=metrics_out, predictions=predictions_out, artifacts=artifact_rows
    )


def _is_completed_experiment(
    output: Path, experiment_id: str, context: DevelopmentCVContext, config_id: str
) -> bool:
    metadata = output / "experiments" / experiment_id / "metadata.json"
    if not metadata.exists():
        return False
    data = json.loads(metadata.read_text(encoding="utf-8"))
    return (
        data.get("dataset_sha256") == context.dataset_sha256
        and data.get("registry_sha256") == context.registry_sha256
        and data.get("config_id", config_id) == config_id
    )


def _input_hashes(context: DevelopmentCVContext) -> dict[str, str]:
    return {
        "dataset_sha256": context.dataset_sha256,
        "registry_sha256": context.registry_sha256,
        "split_sha256": context.split_sha256,
    }


def _fold_contract(fold: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "fold_id",
        "train_start",
        "train_end",
        "train_target_end_max",
        "validation_start",
        "validation_end",
        "validation_target_end_max",
        "locked_test_start",
        "locked_test_end",
        "horizon_weeks",
        "embargo_weeks",
        "rows_train",
        "rows_validation",
    ]
    return {key: fold.get(key) for key in keys if key in fold}


def _feature_schema(frame: pd.DataFrame, features: list[str]) -> list[dict[str, str]]:
    return [{"name": name, "dtype": str(frame[name].dtype)} for name in features]


def _prediction_schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": name, "dtype": str(frame[name].dtype)} for name in frame.columns]


def _load_completed_experiment(
    *,
    output: Path,
    experiment_id: str,
    context: DevelopmentCVContext,
    config_id: str,
    model_config: ModelConfig,
    feature_set: str,
    feature_columns: list[str],
    fold: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    experiment_dir = output / "experiments" / experiment_id
    metadata_path = experiment_dir / "metadata.json"
    predictions_path = experiment_dir / "predictions.parquet"
    metrics_path = experiment_dir / "metrics.json"
    config_path = experiment_dir / "config.json"
    features_path = experiment_dir / "features.json"
    if not experiment_dir.exists():
        return None
    required = [metadata_path, predictions_path, metrics_path, config_path, features_path]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise EvaluationError(
            f"Existing experiment {experiment_id} is incomplete; missing {missing}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    features = json.loads(features_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_config = {
        **model_config.serializable(),
        "model": model_config.family,
        "config_id": config_id,
        "feature_set": feature_set,
        "dataset_sha256": context.dataset_sha256,
        "dataset_path": str(context.dataset_path),
        "input_hashes": _input_hashes(context),
        "registry_sha256": context.registry_sha256,
        "split_sha256": context.split_sha256,
        "split": _fold_contract(fold),
    }
    expected_features = {
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "schema": _feature_schema(context.frame, feature_columns),
    }
    if config != expected_config:
        raise EvaluationError(f"Existing experiment {experiment_id} config does not match request")
    if features.get("feature_set") != expected_features["feature_set"]:
        raise EvaluationError(f"Existing experiment {experiment_id} feature set does not match")
    if features.get("feature_columns") != expected_features["feature_columns"]:
        raise EvaluationError(f"Existing experiment {experiment_id} feature columns do not match")
    if features.get("schema") != expected_features["schema"]:
        raise EvaluationError(f"Existing experiment {experiment_id} feature schema does not match")
    current_source = _source_identity()
    if metadata.get("dirty_source_digest") != current_source.get("dirty_source_digest"):
        raise EvaluationError(f"Existing experiment {experiment_id} source hash does not match")
    pred_frame = pd.read_parquet(predictions_path)
    if pred_frame.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise EvaluationError(f"Existing experiment {experiment_id} has duplicate predictions")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_dir = output / "models" / experiment_id
    if not model_dir.exists():
        raise EvaluationError(f"Existing experiment {experiment_id} missing model directory")
    load_champion(model_dir)
    return pred_frame, metrics


def _row_key_digest(frame: pd.DataFrame) -> str:
    text = "\n".join(
        f"{row.district_id}|{pd.Timestamp(row.week_start_date).date().isoformat()}"
        for row in frame[ROW_KEY_COLUMNS].itertuples(index=False)
    )
    return _sha256_text(text)


def _append_parquet(
    frame: pd.DataFrame, path: Path, *, key_columns: list[str] | None = None
) -> None:
    if path.exists():
        current = pd.read_parquet(path)
        if key_columns is None and set(PREDICTION_KEY_COLUMNS).issubset(current.columns) and set(
            PREDICTION_KEY_COLUMNS
        ).issubset(frame.columns):
            key_columns = PREDICTION_KEY_COLUMNS
        if key_columns:
            old_keys = current.loc[:, key_columns].astype(str).agg("\0".join, axis=1)
            new_keys = frame.loc[:, key_columns].astype(str).agg("\0".join, axis=1)
            current = current.loc[~old_keys.isin(set(new_keys))].copy()
        combined = pd.concat([current, frame], ignore_index=True)
    else:
        combined = frame
    if key_columns and combined.duplicated(key_columns).any():
        raise EvaluationError(f"Duplicate rows for unique key {key_columns}")
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)


def _mirror_registry_metadata_to_model_dir(experiment_dir: Path, model_dir: Path) -> None:
    for name in [
        "config.json",
        "features.json",
        "metrics.json",
        "training_period.json",
        "environment.json",
        "README.md",
    ]:
        source = experiment_dir / name
        if source.exists():
            shutil.copy2(source, model_dir / name)


def _empty_metrics() -> dict[str, float | int]:
    return {
        "n": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "r2": float("nan"),
        "poisson_deviance": float("nan"),
        "bias": float("nan"),
        "mae_top_decile": float("nan"),
        "mae_top_90": float("nan"),
        "mae_top_95": float("nan"),
    }


def summarize_model_results(
    metrics: pd.DataFrame, *, required_fold_count: int | None = None
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    grouped = metrics.groupby(["model", "objective", "feature_set", "config_id"], dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        model, objective, feature_set, config_id = keys
        fold_count = int(group["fold"].nunique())
        if required_fold_count is not None and fold_count != required_fold_count:
            continue
        rows.append(
            {
                "model": model,
                "objective": objective,
                "feature_set": feature_set,
                "config_id": config_id,
                "fold_count": fold_count,
                "mean_fold_mae": float(group["mae"].mean()),
                "median_fold_mae": float(group["mae"].median()),
                "std_fold_mae": float(group["mae"].std(ddof=0)),
                "metric_weighting": "equal_weight_fold_means",
            }
        )
    return pd.DataFrame(rows)


def build_primary_validation_table(context: DevelopmentCVContext) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in context.folds:
        validation = context.fold_frames(fold).validation
        item = validation[["district_id", "week_start_date", "week_end_date", TARGET_COLUMN]].copy()
        item["fold"] = fold["fold_id"]
        item["locked_test"] = pd.NA
        rows.append(item.rename(columns={TARGET_COLUMN: "target"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ensure_disk_budget(path: str | Path, *, minimum_gb: float = 2.0) -> None:
    usage = shutil.disk_usage(Path(path).resolve())
    free_gb = usage.free / (1024**3)
    if free_gb < minimum_gb:
        raise EvaluationError(f"Free disk {free_gb:.2f}GB is below required {minimum_gb:.2f}GB")


def refuse_development_after_holdout(artifact_root: str | Path) -> None:
    root = Path(artifact_root)
    candidates = [
        root / "locked_test_receipt.json",
        root / "champion_holdout_receipt.json",
        root / "holdout_requested.json",
        root / "holdout_running.json",
        root / "holdout_completed.json",
        root / "champion_frozen",
        root / "experiments" / "champion_frozen",
        root / "experiments" / "freeze.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise EvaluationError(f"Development training/tuning is closed by holdout state: {joined}")


__all__ = [
    "DevelopmentCVContext",
    "EvaluationError",
    "EvaluationResult",
    "FoldFrames",
    "build_primary_validation_table",
    "ensure_disk_budget",
    "evaluate_baselines",
    "evaluate_model_config",
    "load_development_context",
    "rank_full_coverage_baseline",
    "summarize_model_results",
]
