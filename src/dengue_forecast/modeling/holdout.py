from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.modeling.baselines import BASELINE_NAMES, baseline_predict
from dengue_forecast.modeling.dataset import FUTURE_TARGET_COLUMNS, TARGET_COLUMN
from dengue_forecast.modeling.evaluate import metric_bundle
from dengue_forecast.modeling.selection import sha256_json
from dengue_forecast.modeling.train import load_champion
from dengue_forecast.utils.hashing import sha256_file


class HoldoutError(ValueError):
    """Raised when locked-test scoring would violate one-time holdout policy."""


ROW_KEY_COLUMNS = ["district_id", "week_start_date"]
STATE_FILENAME = "holdout_state.json"
PREDICTIONS_FILENAME = "locked_test_predictions.parquet"
BASELINE_PREDICTIONS_FILENAME = "locked_test_baseline_predictions.parquet"
METRICS_FILENAME = "locked_test_metrics.json"


@dataclass(frozen=True)
class HoldoutScoringRequest:
    freeze: dict[str, Any]
    test_features: pd.DataFrame
    test_row_keys: pd.DataFrame
    champion_dir: Path | None = None


@dataclass(frozen=True)
class HoldoutResult:
    state: dict[str, Any]
    metrics: dict[str, Any]
    predictions_path: Path
    metrics_path: Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _row_key_digest(frame: pd.DataFrame) -> str:
    rows = [
        f"{row.district_id}|{pd.Timestamp(row.week_start_date).date().isoformat()}"
        for row in frame[ROW_KEY_COLUMNS].itertuples(index=False)
    ]
    return sha256_json(rows)


def _target_information_end(frame: pd.DataFrame, horizon_weeks: int = 1) -> pd.Series:
    return pd.to_datetime(frame["week_end_date"]) + pd.Timedelta(days=7 * horizon_weeks)


def permitted_development_training_frame(
    frame: pd.DataFrame,
    split: dict[str, Any],
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return final-training rows allowed before frozen holdout, plus q90/q95 metadata."""
    locked_start = pd.Timestamp(split["locked_test_start"])
    horizon_weeks = int(split.get("policy", {}).get("horizon_weeks", 1))
    embargo_weeks = int(split.get("policy", {}).get("embargo_weeks", 1))
    required = {"district_id", "week_start_date", "week_end_date", "is_trainable", target_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HoldoutError(f"Development training frame missing required columns: {missing}")
    origins = pd.to_datetime(frame["week_start_date"], errors="coerce")
    ends = pd.to_datetime(frame["week_end_date"], errors="coerce")
    target_end = ends + pd.Timedelta(days=7 * horizon_weeks)
    embargo_end = target_end + pd.Timedelta(days=7 * embargo_weeks)
    eligible = (
        frame["is_trainable"].astype(bool)
        & frame[target_column].notna()
        & origins.lt(locked_start)
        & embargo_end.lt(locked_start)
        & origins.dt.year.lt(2025)
    )
    out = frame.loc[eligible].copy().sort_values(ROW_KEY_COLUMNS).reset_index(drop=True)
    if out.empty:
        raise HoldoutError("No eligible pre-holdout development training rows")
    targets = pd.to_numeric(out[target_column], errors="coerce").astype("float64")
    if targets.isna().any() or not np.isfinite(targets).all():
        raise HoldoutError("Development training targets must be finite")
    metadata = {
        "policy": (
            "origin_before_locked_start_and_target_end_plus_one_week_embargo_"
            "before_locked_start"
        ),
        "locked_first_origin": locked_start.date().isoformat(),
        "horizon_weeks": horizon_weeks,
        "embargo_weeks": embargo_weeks,
        "rows": int(len(out)),
        "districts": int(out["district_id"].nunique()),
        "first_origin": pd.to_datetime(out["week_start_date"]).min().date().isoformat(),
        "last_origin": pd.to_datetime(out["week_start_date"]).max().date().isoformat(),
        "max_target_end": _target_information_end(out, horizon_weeks).max().date().isoformat(),
        "thresholds": {
            "q90": float(targets.quantile(0.9)),
            "q95": float(targets.quantile(0.95)),
        },
        "row_key_digest": _row_key_digest(out),
    }
    return out, metadata


def locked_holdout_frame(
    frame: pd.DataFrame,
    split: dict[str, Any],
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the locked rolling-origin holdout rows: 2024 origins only, Jan 6-Dec 28."""
    start = pd.Timestamp(split["locked_test_start"])
    feature_interval_end = pd.Timestamp(split["locked_test_end"])
    if start.date().isoformat() != "2024-01-06":
        raise HoldoutError("This holdout gate is locked to actual 2024 origin start 2024-01-06")
    origin_end = pd.Timestamp("2024-12-28")
    required = {"district_id", "week_start_date", "week_end_date", "is_trainable", target_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HoldoutError(f"Holdout frame missing required columns: {missing}")
    origins = pd.to_datetime(frame["week_start_date"], errors="coerce")
    ends = pd.to_datetime(frame["week_end_date"], errors="coerce")
    horizon_weeks = int(split.get("policy", {}).get("horizon_weeks", 1))
    mask = (
        frame["is_trainable"].astype(bool)
        & frame[target_column].notna()
        & origins.ge(start)
        & origins.le(origin_end)
        & origins.dt.year.eq(2024)
        & ends.le(feature_interval_end)
    )
    out = frame.loc[mask].copy().sort_values(ROW_KEY_COLUMNS).reset_index(drop=True)
    if out.empty:
        raise HoldoutError("Locked holdout contains no scoreable rows")
    if pd.to_datetime(out["week_start_date"]).dt.year.ne(2024).any():
        raise HoldoutError("No 2025 origins may be scored")
    metadata = {
        "origin_start": start.date().isoformat(),
        "origin_end": origin_end.date().isoformat(),
        "feature_interval_end": feature_interval_end.date().isoformat(),
        "max_target_interval_end": (
            _target_information_end(out, horizon_weeks).max().date().isoformat()
        ),
        "target_boundary_note": (
            "Feature interval ends 2025-01-03; final target interval may end 2025-01-10."
        ),
        "rows": int(len(out)),
        "districts": int(out["district_id"].nunique()),
        "row_key_digest": _row_key_digest(out),
    }
    return out, metadata


def validate_holdout_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if list(feature_columns) != list(dict.fromkeys(feature_columns)):
        raise HoldoutError("Feature columns must be unique and ordered")
    forbidden = set(feature_columns) & FUTURE_TARGET_COLUMNS
    if forbidden:
        raise HoldoutError(f"Target/future columns are forbidden in holdout X: {sorted(forbidden)}")
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise HoldoutError(f"Holdout frame missing authorized features: {missing}")
    extras = sorted(set(frame.columns) & FUTURE_TARGET_COLUMNS)
    if extras and any(col in feature_columns for col in extras):
        raise HoldoutError(f"Holdout features include future targets: {extras}")
    return frame.loc[:, feature_columns].copy()


def _load_freeze(freeze_path: str | Path) -> dict[str, Any]:
    freeze = _read_json(Path(freeze_path))
    expected = freeze.get("configuration_hash")
    actual = sha256_json(
        {key: value for key, value in freeze.items() if key != "configuration_hash"}
    )
    if expected != actual:
        raise HoldoutError("freeze.json configuration_hash does not verify")
    return freeze


def _claim_running(state_path: Path, run_hash: str, approval_hash: str) -> dict[str, Any]:
    requested = {"status": "requested", "timestamp_utc": _now(), "run_hash": run_hash}
    event = {"status": "running", "timestamp_utc": _now(), "run_hash": run_hash}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with state_path.open("x", encoding="utf-8") as handle:
            state = {
                "schema_version": 1,
                "status": "running",
                "configuration_hash": run_hash,
                "approval_hash": approval_hash,
                "counts": {"requested": 1, "running": 1, "completed": 0, "failed": 0},
                "events": [requested, event],
            }
            handle.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
            return state
    except FileExistsError:
        existing = _read_json(state_path)
        if existing.get("configuration_hash") != run_hash:
            raise HoldoutError(
                "Holdout state exists for a different frozen configuration"
            ) from None
        status = existing.get("status")
        if status == "completed":
            return existing
        raise HoldoutError(
            f"Holdout state is {status}; interrupted or failed locked-test runs do not "
            "rerun automatically"
        ) from None


def _complete_state(
    state_path: Path, state: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    state = {**state, **updates, "status": "completed"}
    counts = dict(state.get("counts") or {})
    if int(counts.get("completed", 0)) != 0:
        raise HoldoutError("Holdout state already has a completed event")
    counts["completed"] = int(counts.get("completed", 0)) + 1
    state["counts"] = counts
    state.setdefault("events", []).append({"status": "completed", "timestamp_utc": _now()})
    _write_json_atomic(state_path, state)
    return state


def _running_timestamp(state: dict[str, Any]) -> str | None:
    for event in state.get("events") or []:
        if event.get("status") == "running":
            return str(event.get("timestamp_utc"))
    return None


def _fail_state(state_path: Path, state: dict[str, Any], error: Exception) -> None:
    state = {**state, "status": "failed", "error": str(error)}
    counts = dict(state.get("counts") or {})
    counts["failed"] = int(counts.get("failed", 0)) + 1
    state["counts"] = counts
    state.setdefault("events", []).append({"status": "failed", "timestamp_utc": _now()})
    _write_json_atomic(state_path, state)


def _default_scorer(request: HoldoutScoringRequest) -> np.ndarray:
    if request.champion_dir is None:
        raise HoldoutError("champion_dir is required when no explicit scorer is supplied")
    return load_champion(request.champion_dir).predict_next_week(request.test_features)


def _baseline_frames_and_metrics(
    *,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame,
    holdout: pd.DataFrame,
    model_prediction: np.ndarray,
    thresholds: dict[str, Any],
    config_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = frame[
        pd.to_datetime(frame["week_start_date"]).le(pd.to_datetime(holdout["week_start_date"]).max())
    ].copy()
    wide = baseline_predict(
        training_frame=training_frame,
        validation_frame=holdout,
        full_history=history,
    )
    threshold90 = float(thresholds["q90"])
    threshold95 = float(thresholds["q95"])
    rows: list[pd.DataFrame] = []
    native: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    availability: dict[str, dict[str, Any]] = {}
    target = pd.to_numeric(holdout[TARGET_COLUMN], errors="coerce").astype("float64")
    model_series = pd.Series(model_prediction, index=holdout.index, dtype="float64")
    common_mask = pd.Series(True, index=holdout.index)
    for name in BASELINE_NAMES:
        pred = pd.to_numeric(wide[name], errors="coerce").astype("float64")
        available = pred.notna() & np.isfinite(pred)
        common_mask &= available
        availability[name] = {
            "available": bool(available.all()),
            "n": int(available.sum()),
            "total": int(len(pred)),
            "coverage": float(available.mean()) if len(pred) else 0.0,
        }
        item = holdout[["district_id", "week_start_date", "week_end_date", TARGET_COLUMN]].rename(
            columns={TARGET_COLUMN: "target"}
        )
        item = item.copy()
        item["baseline"] = name
        item["prediction"] = pred.to_numpy(dtype="float64")
        item["available"] = available.to_numpy(dtype=bool)
        item["configuration_hash"] = config_hash
        rows.append(item)
        if available.any():
            native[name] = {
                "coverage": availability[name]["coverage"],
                **metric_bundle(
                    target.loc[available],
                    pred.loc[available],
                    threshold90=threshold90,
                    threshold95=threshold95,
                ),
            }
            paired[name] = {
                "coverage": availability[name]["coverage"],
                "model": metric_bundle(
                    target.loc[available],
                    model_series.loc[available],
                    threshold90=threshold90,
                    threshold95=threshold95,
                ),
                "baseline": native[name],
            }
        else:
            native[name] = {"coverage": 0.0, **_empty_metric_bundle()}
            paired[name] = {
                "coverage": 0.0,
                "model": _empty_metric_bundle(),
                "baseline": native[name],
            }
    if common_mask.any():
        common_support = {
            "n": int(common_mask.sum()),
            "coverage": float(common_mask.mean()),
            "model": metric_bundle(
                target.loc[common_mask],
                model_series.loc[common_mask],
                threshold90=threshold90,
                threshold95=threshold95,
            ),
            "baselines": {
                name: metric_bundle(
                    target.loc[common_mask],
                    pd.to_numeric(wide.loc[common_mask, name], errors="coerce"),
                    threshold90=threshold90,
                    threshold95=threshold95,
                )
                for name in BASELINE_NAMES
            },
        }
    else:
        common_support = {"n": 0, "coverage": 0.0, "model": _empty_metric_bundle(), "baselines": {}}
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return predictions, {
        "availability": availability,
        "native": native,
        "paired_with_model": paired,
        "common_support": common_support,
    }


def _empty_metric_bundle() -> dict[str, float | int]:
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


def _validate_completed_event_count(state: dict[str, Any]) -> None:
    events = [event for event in state.get("events") or [] if event.get("status") == "completed"]
    if len(events) != 1 or int((state.get("counts") or {}).get("completed", 0)) != 1:
        raise HoldoutError("Holdout state must contain exactly one completed event")


def _recalculate_saved_metrics(
    *,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> None:
    thresholds = {
        "q90": float(metrics["thresholds"]["q90"]),
        "q95": float(metrics["thresholds"]["q95"]),
    }
    recalculated = metric_bundle(
        predictions["target"],
        predictions["prediction"],
        threshold90=thresholds["q90"],
        threshold95=thresholds["q95"],
    )
    for key, value in recalculated.items():
        recorded = float(metrics["model_metrics"][key])
        if np.isnan(value) and np.isnan(recorded):
            continue
        if not np.isclose(float(value), recorded, rtol=1e-12, atol=1e-12):
            raise HoldoutError(f"Saved model metric {key} does not recalculate")
    for name in BASELINE_NAMES:
        subset = baseline_predictions[baseline_predictions["baseline"].astype(str).eq(name)]
        if subset.empty:
            raise HoldoutError(f"Saved baseline predictions missing {name}")
        available = subset["available"].astype(bool)
        recorded = metrics["baselines"]["availability"][name]
        if int(available.sum()) != int(recorded["n"]):
            raise HoldoutError(f"Saved baseline availability does not recalculate for {name}")
        if available.any():
            recalculated_baseline = metric_bundle(
                subset.loc[available, "target"],
                subset.loc[available, "prediction"],
                threshold90=thresholds["q90"],
                threshold95=thresholds["q95"],
            )
            for key, value in recalculated_baseline.items():
                recorded_value = float(metrics["baselines"]["native"][name][key])
                if np.isnan(value) and np.isnan(recorded_value):
                    continue
                if not np.isclose(float(value), recorded_value, rtol=1e-12, atol=1e-12):
                    raise HoldoutError(f"Saved baseline metric {name}.{key} does not recalculate")


def score_locked_holdout_once(
    *,
    freeze_path: str | Path,
    frame: pd.DataFrame,
    split: dict[str, Any],
    output_dir: str | Path,
    approval_hash: str,
    scorer: Callable[[HoldoutScoringRequest], np.ndarray] | None = None,
    champion_dir: str | Path | None = None,
) -> HoldoutResult:
    """Score the locked test exactly once; completed identical runs validate stored artifacts."""
    freeze = _load_freeze(freeze_path)
    if freeze.get("approval_hash") != approval_hash:
        raise HoldoutError("Caller approval hash does not match frozen champion approval")
    config_hash = str(freeze["configuration_hash"])
    out_dir = Path(output_dir)
    state_path = out_dir / STATE_FILENAME
    metrics_path = out_dir / METRICS_FILENAME
    predictions_path = out_dir / PREDICTIONS_FILENAME
    baseline_predictions_path = out_dir / BASELINE_PREDICTIONS_FILENAME
    state = _claim_running(state_path, config_hash, approval_hash)
    if state.get("status") == "completed":
        return validate_holdout_run(freeze_path=freeze_path, output_dir=out_dir)

    try:
        freeze_timestamp = pd.Timestamp(freeze["timestamp_utc"])
        running_timestamp = _running_timestamp(state)
        if running_timestamp is None or pd.Timestamp(running_timestamp) <= freeze_timestamp:
            raise HoldoutError("Holdout running timestamp must be after freeze creation timestamp")
        holdout, holdout_metadata = locked_holdout_frame(frame, split)
        training_frame, training_metadata = permitted_development_training_frame(frame, split)
        config = freeze["champion_config"]
        feature_columns = list(config["feature_columns"])
        test_x = validate_holdout_features(holdout, feature_columns)
        request = HoldoutScoringRequest(
            freeze=freeze,
            test_features=test_x,
            test_row_keys=holdout[ROW_KEY_COLUMNS].copy(),
            champion_dir=Path(champion_dir) if champion_dir is not None else None,
        )
        prediction = np.asarray((scorer or _default_scorer)(request), dtype="float64")
        if len(prediction) != len(holdout) or not np.isfinite(prediction).all():
            raise HoldoutError("Scorer must return one finite prediction per holdout row")
        prediction = np.clip(prediction, 0.0, None)
        thresholds = config["thresholds"]
        model_metrics = metric_bundle(
            holdout[TARGET_COLUMN],
            prediction,
            threshold90=float(thresholds["q90"]),
            threshold95=float(thresholds["q95"]),
        )
        baseline_predictions, baseline_metrics = _baseline_frames_and_metrics(
            frame=frame,
            training_frame=training_frame,
            holdout=holdout,
            model_prediction=prediction,
            thresholds=thresholds,
            config_hash=config_hash,
        )
        metrics = dict(model_metrics)
        metrics.update(
            {
                "configuration_hash": config_hash,
                "row_key_digest": _row_key_digest(holdout),
                "valid_original_count_units": True,
                "holdout": holdout_metadata,
                "training": training_metadata,
                "thresholds": {"q90": float(thresholds["q90"]), "q95": float(thresholds["q95"])},
                "model_metrics": model_metrics,
                "baselines": baseline_metrics,
                "best_full_cohort_baseline_from_cv": config.get("strongest_full_cohort_baseline"),
                "baseline_availability": baseline_metrics["availability"],
            }
        )
        predictions = holdout[
            ["district_id", "week_start_date", "week_end_date", TARGET_COLUMN]
        ].rename(columns={TARGET_COLUMN: "target"})
        predictions = predictions.copy()
        predictions["prediction"] = prediction
        predictions["configuration_hash"] = config_hash
        predictions["threshold90"] = float(thresholds["q90"])
        predictions["threshold95"] = float(thresholds["q95"])
        predictions.to_parquet(predictions_path, index=False)
        baseline_predictions.to_parquet(baseline_predictions_path, index=False)
        _write_json_atomic(metrics_path, metrics)
        completed = _complete_state(
            state_path,
            state,
            {
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(predictions_path),
                "baseline_predictions_sha256": sha256_file(baseline_predictions_path),
                "row_key_digest": metrics["row_key_digest"],
                "metrics_path": METRICS_FILENAME,
                "predictions_path": PREDICTIONS_FILENAME,
                "baseline_predictions_path": BASELINE_PREDICTIONS_FILENAME,
            },
        )
        return HoldoutResult(completed, metrics, predictions_path, metrics_path)
    except Exception as exc:
        _fail_state(state_path, state, exc)
        raise


def validate_holdout_run(*, freeze_path: str | Path, output_dir: str | Path) -> HoldoutResult:
    """Pure validation of an already completed holdout run; performs no scoring or retraining."""
    freeze = _load_freeze(freeze_path)
    out_dir = Path(output_dir)
    state_path = out_dir / STATE_FILENAME
    metrics_path = out_dir / METRICS_FILENAME
    predictions_path = out_dir / PREDICTIONS_FILENAME
    baseline_predictions_path = out_dir / BASELINE_PREDICTIONS_FILENAME
    if (
        not state_path.exists()
        or not metrics_path.exists()
        or not predictions_path.exists()
        or not baseline_predictions_path.exists()
    ):
        raise HoldoutError("Completed holdout artifacts are missing")
    state = _read_json(state_path)
    if state.get("status") != "completed":
        raise HoldoutError("Holdout state is not completed")
    _validate_completed_event_count(state)
    if state.get("configuration_hash") != freeze.get("configuration_hash"):
        raise HoldoutError("Holdout state configuration does not match freeze")
    if state.get("metrics_sha256") != sha256_file(metrics_path):
        raise HoldoutError("Recorded metrics hash does not verify")
    if state.get("predictions_sha256") != sha256_file(predictions_path):
        raise HoldoutError("Recorded predictions hash does not verify")
    if state.get("baseline_predictions_sha256") != sha256_file(baseline_predictions_path):
        raise HoldoutError("Recorded baseline predictions hash does not verify")
    metrics = _read_json(metrics_path)
    if metrics.get("configuration_hash") != freeze.get("configuration_hash"):
        raise HoldoutError("Metrics configuration hash does not match freeze")
    predictions = pd.read_parquet(predictions_path)
    if _row_key_digest(predictions) != state.get("row_key_digest"):
        raise HoldoutError("Prediction row keys do not match recorded holdout state")
    baseline_predictions = pd.read_parquet(baseline_predictions_path)
    observed_baselines = set(baseline_predictions.get("baseline", pd.Series(dtype=str)).astype(str))
    if observed_baselines != set(BASELINE_NAMES):
        raise HoldoutError("Saved baseline predictions do not include all canonical baselines")
    _recalculate_saved_metrics(
        metrics=metrics,
        predictions=predictions,
        baseline_predictions=baseline_predictions,
    )
    return HoldoutResult(state, metrics, predictions_path, metrics_path)


def validate_model_binary_load(
    *,
    champion_dir: str | Path,
    expected_feature_columns: list[str],
) -> dict[str, Any]:
    model = load_champion(champion_dir)
    if model.feature_columns != expected_feature_columns:
        raise HoldoutError("Loaded champion feature schema does not match freeze")
    metadata_path = Path(champion_dir) / "metadata.json"
    return {
        "champion_dir": str(champion_dir),
        "metadata_sha256": sha256_file(metadata_path),
        "feature_columns": model.feature_columns,
    }


__all__ = [
    "HoldoutError",
    "HoldoutResult",
    "HoldoutScoringRequest",
    "locked_holdout_frame",
    "permitted_development_training_frame",
    "score_locked_holdout_once",
    "validate_holdout_features",
    "validate_holdout_run",
    "validate_model_binary_load",
]
