from __future__ import annotations

import numpy as np
import pandas as pd

POISSON_EPSILON = 1.0e-12


def clip_predictions_nonnegative(predictions: pd.Series | np.ndarray) -> pd.Series:
    """Clip predictions to the nonnegative count domain before metric evaluation."""
    values = pd.Series(predictions, copy=True, dtype="float64")
    return values.clip(lower=0.0)


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean absolute error for finite, nonnegative count targets and predictions."""
    truth, pred = _validated_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred - truth)))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root mean squared error for finite, nonnegative count targets and predictions."""
    truth, pred = _validated_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(pred - truth))))


def r2_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Coefficient of determination; returns NaN for n=1 or zero target variance."""
    truth, pred = _validated_arrays(y_true, y_pred)
    if len(truth) < 2:
        return float("nan")
    denominator = float(np.sum(np.square(truth - np.mean(truth))))
    if denominator == 0.0:
        return float("nan")
    numerator = float(np.sum(np.square(truth - pred)))
    return float(1.0 - numerator / denominator)


def poisson_deviance(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    epsilon: float = POISSON_EPSILON,
) -> float:
    """Mean Poisson deviance with a common positive epsilon for zero predictions.

    The same epsilon is applied to every model prediction before the logarithmic term,
    making zero predictions valid but consistently treated across model families.
    """
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    truth, pred = _validated_arrays(y_true, y_pred)
    pred = np.maximum(pred, epsilon)
    terms = pred.copy()
    positive_truth = truth > 0.0
    terms[positive_truth] = (
        truth[positive_truth] * np.log(truth[positive_truth] / pred[positive_truth])
        - (truth[positive_truth] - pred[positive_truth])
    )
    return float(2.0 * np.mean(terms))


def bias(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean signed prediction error, prediction minus actual."""
    truth, pred = _validated_arrays(y_true, y_pred)
    return float(np.mean(pred - truth))


def top_quantile_threshold(y_train: pd.Series | np.ndarray, *, quantile: float = 0.9) -> float:
    """Estimate a high-incidence threshold from training/development targets only."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    values = _validated_target(y_train, name="y_train")
    if len(values) == 0:
        raise ValueError("y_train must contain at least one value")
    return float(np.quantile(values, quantile))


def metric_bundle(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    top_decile_threshold: float | None = None,
) -> dict[str, float | int]:
    """Return the standard Milestone 2 metric bundle.

    ``mae_top_decile`` is computed using a caller-supplied threshold derived from
    training/development data. If no validation rows meet the threshold, the value is
    NaN rather than a fabricated zero.
    """
    truth, pred = _validated_arrays(y_true, y_pred)
    result: dict[str, float | int] = {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(pred - truth))),
        "rmse": float(np.sqrt(np.mean(np.square(pred - truth)))),
        "r2": _r2_from_arrays(truth, pred),
        "poisson_deviance": poisson_deviance(truth, pred),
        "bias": float(np.mean(pred - truth)),
    }
    if top_decile_threshold is None:
        result["mae_top_decile"] = float("nan")
    else:
        mask = truth >= _validate_threshold(top_decile_threshold, "top_decile_threshold")
        result["mae_top_decile"] = _masked_mae(truth, pred, mask)
    return result


def high_incidence_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    top_10_threshold: float,
    top_5_threshold: float,
) -> dict[str, float]:
    """Return top-10% and top-5% high-incidence metrics using passed thresholds."""
    truth, pred = _validated_arrays(y_true, y_pred)
    mask10 = truth >= _validate_threshold(top_10_threshold, "top_10_threshold")
    mask5 = truth >= _validate_threshold(top_5_threshold, "top_5_threshold")
    return {
        "mae_top_10pct": _masked_mae(truth, pred, mask10),
        "rmse_top_10pct": _masked_rmse(truth, pred, mask10),
        "bias_top_10pct": _masked_bias(truth, pred, mask10),
        "underprediction_rate_top_10pct": _masked_underprediction_rate(truth, pred, mask10),
        "mae_top_5pct": _masked_mae(truth, pred, mask5),
        "rmse_top_5pct": _masked_rmse(truth, pred, mask5),
        "bias_top_5pct": _masked_bias(truth, pred, mask5),
        "underprediction_rate_top_5pct": _masked_underprediction_rate(truth, pred, mask5),
    }


def _validated_arrays(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = _validated_target(y_true, name="y_true")
    pred = np.asarray(pd.Series(y_pred, dtype="float64"), dtype="float64")
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(truth) == 0:
        raise ValueError("metrics require at least one row")
    if not np.isfinite(pred).all():
        raise ValueError("y_pred must contain only finite values")
    if (pred < 0).any():
        raise ValueError("y_pred must not contain negative values")
    return truth, pred


def _validated_target(y_true: pd.Series | np.ndarray, *, name: str) -> np.ndarray:
    truth = np.asarray(pd.Series(y_true, dtype="float64"), dtype="float64")
    if not np.isfinite(truth).all():
        raise ValueError(f"{name} must contain only finite values")
    if (truth < 0).any():
        raise ValueError(f"{name} must be non-negative")
    return truth


def _validate_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return threshold


def _r2_from_arrays(truth: np.ndarray, pred: np.ndarray) -> float:
    if len(truth) < 2:
        return float("nan")
    denominator = float(np.sum(np.square(truth - np.mean(truth))))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - float(np.sum(np.square(truth - pred))) / denominator)


def _masked_mae(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - truth[mask])))


def _masked_rmse(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred[mask] - truth[mask]))))


def _masked_bias(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean(pred[mask] - truth[mask]))


def _masked_underprediction_rate(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean(pred[mask] < truth[mask]))
