from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SelectionError(ValueError):
    """Raised when validation-only champion selection would violate policy."""


TIE_TOLERANCE = 1e-9
CLOSE_RELATIVE_MAE = 0.01


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(data: Any) -> str:
    return json.dumps(data, default=_json_default, sort_keys=True, separators=(",", ":"))


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChampionSelection:
    champion: dict[str, Any]
    ranking: list[dict[str, Any]]
    close_models: list[dict[str, Any]]
    comparable_folds: list[str]
    validation_row_key_digest: str
    selected_by: str = "validation_only_equal_fold_mean_mae"


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise SelectionError(f"{name} must be finite")
    return result


def _row_digest(row_keys: dict[str, list[str]]) -> str:
    normalized = {
        fold_id: [str(key) for key in keys] for fold_id, keys in sorted(row_keys.items())
    }
    return sha256_json(normalized)


def _candidate_fold_map(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    folds = candidate.get("folds")
    if not isinstance(folds, list) or not folds:
        raise SelectionError(f"Candidate {candidate.get('stable_id')} has no fold metrics")
    out: dict[str, dict[str, Any]] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            raise SelectionError("Fold metric entries must be objects")
        fold_id = str(fold.get("fold_id") or fold.get("fold"))
        if not fold_id or fold_id == "None":
            raise SelectionError("Fold metric entry missing fold_id")
        if fold_id in out:
            raise SelectionError(f"Candidate {candidate.get('stable_id')} duplicates {fold_id}")
        mae = _finite_float(fold.get("mae"), name=f"{candidate.get('stable_id')} {fold_id} mae")
        row_keys = fold.get("validation_row_keys")
        if not isinstance(row_keys, list) or not row_keys:
            raise SelectionError(
                f"Candidate {candidate.get('stable_id')} {fold_id} missing row keys"
            )
        out[fold_id] = {**fold, "mae": mae, "validation_row_keys": [str(k) for k in row_keys]}
    return out


def select_champion(candidates: list[dict[str, Any]]) -> ChampionSelection:
    """Select a champion from complete comparable validation artifacts only.

    The input is intentionally plain dictionaries so later orchestration can adapt real
    registry/evaluation artifacts without this module loading real data or scores itself.
    """
    if not candidates:
        raise SelectionError("At least one candidate is required")

    fold_ids: list[str] | None = None
    row_keys_by_fold: dict[str, list[str]] | None = None
    ranking: list[dict[str, Any]] = []

    for candidate in candidates:
        stable_id = str(candidate.get("stable_id") or candidate.get("config_id") or "")
        if not stable_id:
            raise SelectionError("Every candidate requires a stable_id")
        split_id = candidate.get("split_id")
        dataset_hash = candidate.get("dataset_sha256") or candidate.get("dataset_hash")
        if not split_id or not dataset_hash:
            raise SelectionError(f"Candidate {stable_id} missing split_id or dataset hash")
        fold_map = _candidate_fold_map(candidate)
        current_fold_ids = sorted(fold_map)
        current_row_keys = {
            fold_id: fold_map[fold_id]["validation_row_keys"] for fold_id in current_fold_ids
        }
        if fold_ids is None:
            fold_ids = current_fold_ids
            row_keys_by_fold = current_row_keys
        elif current_fold_ids != fold_ids:
            raise SelectionError("Candidate configurations have incomplete or mixed folds")
        elif current_row_keys != row_keys_by_fold:
            raise SelectionError("Candidate configurations use mixed validation row keys")

        maes = [fold_map[fold_id]["mae"] for fold_id in current_fold_ids]
        high_values = [
            _finite_float(
                fold_map[fold_id].get("mae_top_decile"),
                name=f"{stable_id} high incidence",
            )
            for fold_id in current_fold_ids
            if fold_map[fold_id].get("mae_top_decile") is not None
            and not math.isnan(float(fold_map[fold_id].get("mae_top_decile")))
        ]
        if len(high_values) != len(current_fold_ids):
            candidate_high = _finite_float(
                candidate.get("high_incidence_mae"), name=f"{stable_id} high_incidence_mae"
            )
        else:
            candidate_high = float(sum(high_values) / len(high_values))
        mean_mae = float(sum(maes) / len(maes))
        fold_std = float(0.0 if len(maes) == 1 else _population_std(maes))
        simplicity = _finite_float(
            candidate.get("simplicity", 1_000_000),
            name=f"{stable_id} simplicity",
        )
        ranking.append(
            {
                "stable_id": stable_id,
                "mean_validation_mae": mean_mae,
                "high_incidence_mae": candidate_high,
                "fold_std": fold_std,
                "simplicity": simplicity,
                "fold_mae": {fold_id: fold_map[fold_id]["mae"] for fold_id in current_fold_ids},
                "source": candidate,
            }
        )

    assert fold_ids is not None and row_keys_by_fold is not None
    best_before_ties = min(float(row["mean_validation_mae"]) for row in ranking)
    ranking.sort(
        key=lambda row: (
            _tie_bucket(row["mean_validation_mae"], best_before_ties),
            row["high_incidence_mae"],
            row["fold_std"],
            row["simplicity"],
            row["stable_id"],
        )
    )
    best = ranking[0]
    best_mae = float(best["mean_validation_mae"])
    close = [
        {
            "stable_id": row["stable_id"],
            "mean_validation_mae": row["mean_validation_mae"],
            "relative_mae_gap": (
                0.0
                if best_mae == 0
                else (row["mean_validation_mae"] - best_mae) / best_mae
            ),
            "decisive": False,
        }
        for row in ranking
        if best_mae == 0
        or (row["mean_validation_mae"] - best_mae) / best_mae <= CLOSE_RELATIVE_MAE + TIE_TOLERANCE
    ]
    return ChampionSelection(
        champion=best["source"],
        ranking=[{key: value for key, value in row.items() if key != "source"} for row in ranking],
        close_models=close,
        comparable_folds=fold_ids,
        validation_row_key_digest=_row_digest(row_keys_by_fold),
    )


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _tie_bucket(value: float, minimum: float) -> float:
    return minimum if abs(value - minimum) <= TIE_TOLERANCE else value


def validate_selection_prerequisites(
    *,
    development_analysis_path: str | Path,
    explainability_path: str | Path,
    leakage_review_path: str | Path,
    cohort_coverage: dict[str, Any],
) -> None:
    missing = [
        str(path)
        for path in [development_analysis_path, explainability_path, leakage_review_path]
        if not Path(path).is_file()
    ]
    if missing:
        raise SelectionError(f"Selection prerequisites missing: {missing}")
    if not cohort_coverage or not bool(cohort_coverage.get("actual_cohort_coverages_clear")):
        raise SelectionError("Actual cohort coverage clearance is required before freeze")


def build_freeze_payload(
    *,
    selection: ChampionSelection,
    artifact_context: dict[str, Any],
    approval_hash: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    if not approval_hash:
        raise SelectionError("Caller explicit approval hash is required")
    champion = selection.champion
    required_config_keys = {
        "family",
        "objective",
        "hyperparams",
        "feature_columns",
        "preprocessing",
        "postprocessing",
        "seed",
        "allowed_train_bound",
        "thresholds",
        "strongest_full_cohort_baseline",
    }
    missing = sorted(required_config_keys - set(champion))
    if missing:
        raise SelectionError(f"Champion configuration missing freeze fields: {missing}")
    thresholds = champion.get("thresholds") or {}
    if "q90" not in thresholds or "q95" not in thresholds:
        raise SelectionError("Champion freeze requires q90 and q95 thresholds")
    feature_columns = champion.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise SelectionError("Champion freeze requires exact ordered feature_columns")

    payload = {
        "schema_version": 1,
        "timestamp_utc": timestamp_utc or datetime.now(UTC).isoformat(),
        "approval_hash": approval_hash,
        "selection_policy": selection.selected_by,
        "validation_only_selection_evidence": {
            "ranking": selection.ranking,
            "close_models_within_1pct": selection.close_models,
            "folds": selection.comparable_folds,
            "validation_row_key_digest": selection.validation_row_key_digest,
        },
        "champion_config": {
            key: champion[key]
            for key in [
                "stable_id",
                "family",
                "objective",
                "hyperparams",
                "feature_columns",
                "preprocessing",
                "postprocessing",
                "seed",
                "allowed_train_bound",
                "thresholds",
                "strongest_full_cohort_baseline",
            ]
            if key in champion
        },
        "provenance": {
            key: artifact_context.get(key)
            for key in [
                "dataset_sha256",
                "registry_sha256",
                "modeling_overlay_sha256",
                "split_sha256",
                "source_identity",
            ]
        },
    }
    payload["configuration_hash"] = sha256_json(
        {key: value for key, value in payload.items() if key != "configuration_hash"}
    )
    return payload


def freeze_champion_config(payload: dict[str, Any], freeze_path: str | Path) -> dict[str, Any]:
    """Persist `freeze.json` immutably; identical reuse is allowed, changes are rejected."""
    path = Path(freeze_path)
    if path.suffix != ".json":
        path = path / "freeze.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        return payload
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SelectionError(
                "Existing freeze.json differs from requested champion freeze"
            ) from None
        if existing.get("configuration_hash") != payload.get("configuration_hash"):
            raise SelectionError(
                "Existing freeze hash does not match requested payload"
            ) from None
        return existing


__all__ = [
    "ChampionSelection",
    "SelectionError",
    "build_freeze_payload",
    "canonical_json",
    "freeze_champion_config",
    "select_champion",
    "sha256_json",
    "validate_selection_prerequisites",
]
