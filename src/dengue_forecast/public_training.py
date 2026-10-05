from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.modeling.train import ModelConfig, TrainedModel, TrainingError, train_fold_model
from dengue_forecast.public_inference import (
    PublicInferenceError,
    load_trusted_manifest,
    validate_features,
)


class PublicTrainingError(ValueError):
    """Raised when public model-ready training inputs fail validation."""


def _entry_for_horizon(horizon: int, manifest_path: str | Path | None) -> dict[str, Any]:
    manifest = load_trusted_manifest(manifest_path)
    for entry in manifest["models"]:
        if int(entry["horizon_weeks"]) == horizon:
            return entry
    raise PublicTrainingError("horizon must be one of 1, 2, 3, or 4")


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise PublicTrainingError("Training input must be a .parquet or .csv file")


def _validate_target(frame: pd.DataFrame, target_column: str) -> pd.Series:
    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any():
        raise PublicTrainingError(f"Training target {target_column} contains missing values")
    if np.isinf(target).any():
        raise PublicTrainingError(f"Training target {target_column} contains non-finite values")
    if target.lt(0).any():
        raise PublicTrainingError("Training target must be non-negative")
    return target.astype("float64")


def _training_rows_available_by_cutoff(
    frame: pd.DataFrame,
    *,
    horizon: int,
    label_cutoff: pd.Timestamp,
    allow_2025_research_scope: bool,
) -> pd.DataFrame:
    if "week_start_date" not in frame.columns:
        raise PublicTrainingError("Training input requires week_start_date")
    dates = pd.to_datetime(frame["week_start_date"], errors="coerce")
    if dates.isna().any():
        raise PublicTrainingError("week_start_date contains invalid dates")
    # A weekly total cannot be available before the target reporting week ends.
    # This is a retrospective boundary, not proof of historical publication time.
    target_dates = dates + pd.to_timedelta(7 * horizon + 6, unit="D")
    if not allow_2025_research_scope:
        if dates.ge(pd.Timestamp("2025-01-01")).any() or target_dates.ge(
            pd.Timestamp("2025-01-01")
        ).any() or label_cutoff >= pd.Timestamp("2025-01-01"):
            raise PublicTrainingError(
                "2025 dates are rejected by default because the consumed 2025 benchmark "
                "is historical; pass --allow-2025-research-scope for a new experiment."
            )
    out = frame.copy()
    out["week_start_date"] = dates
    eligible = target_dates.le(label_cutoff)
    out = out.loc[eligible].copy()
    if out.empty:
        raise PublicTrainingError("No rows have labels available at the requested train-end cutoff")
    return out


def _config_from_entry(entry: dict[str, Any]) -> ModelConfig:
    metadata = entry["metadata"]
    config = metadata["config"]
    return ModelConfig(
        family=str(config["family"]),
        objective=str(config.get("objective", "regression")),
        hyperparams=dict(config.get("hyperparams") or {}),
        seed=int(config.get("seed", 42)),
        postprocessing=dict(config.get("postprocessing") or {"clip_negative_predictions": True}),
    )


def train_public_model(
    *,
    input_path: str | Path,
    horizon: int,
    train_end: str,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    allow_2025_research_scope: bool = False,
    provenance: dict[str, Any] | None = None,
) -> TrainedModel:
    horizon = int(horizon)
    entry = _entry_for_horizon(horizon, manifest_path)
    metadata = entry["metadata"]
    feature_columns = list(metadata["feature_columns"])
    target_column = str(metadata.get("target_column") or f"target_h{horizon}")
    label_cutoff = pd.Timestamp(train_end)
    out_dir = Path(output_dir)
    if out_dir.exists():
        raise PublicTrainingError(f"Output directory already exists: {out_dir}")

    frame = _read_frame(Path(input_path))
    missing = sorted({"week_start_date", target_column, *feature_columns} - set(frame.columns))
    if missing:
        raise PublicTrainingError(f"Training input missing required columns: {missing}")
    frame = _training_rows_available_by_cutoff(
        frame,
        horizon=horizon,
        label_cutoff=label_cutoff,
        allow_2025_research_scope=allow_2025_research_scope,
    )
    try:
        features = validate_features(frame.loc[:, feature_columns], _spec_like(entry))
    except PublicInferenceError as exc:
        raise PublicTrainingError(str(exc)) from exc
    training_frame = pd.concat(
        [
            frame.loc[:, ["week_start_date"]].reset_index(drop=True),
            features.reset_index(drop=True),
            _validate_target(frame, target_column).rename(target_column).reset_index(drop=True),
        ],
        axis=1,
    )
    trained = train_fold_model(
        training_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        config=_config_from_entry(entry),
        output_dir=out_dir,
        frozen_period_bounds={
            "train_start": pd.to_datetime(training_frame["week_start_date"]).min().date().isoformat(),
            "train_end": label_cutoff.date().isoformat(),
        },
        provenance={
            "public_training": True,
            "source_model_id": entry["model_id"],
            "data_scope": "user supplied model-ready rows",
            **(provenance or {}),
        },
    )
    metadata_path = out_dir / "metadata.json"
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_metadata["public_training"] = {
        "horizon_weeks": horizon,
        "source_model_id": entry["model_id"],
        "label_cutoff_date": label_cutoff.date().isoformat(),
        "rows_used": int(len(training_frame)),
        "feature_schema": "model_manifest_trusted.json",
    }
    metadata_path.write_text(json.dumps(saved_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trained.metadata = saved_metadata
    return trained


def _spec_like(entry: dict[str, Any]):
    from dengue_forecast.public_inference import _spec_from_entry

    return _spec_from_entry(entry)


def _cmd_train(args: argparse.Namespace) -> int:
    trained = train_public_model(
        input_path=args.input,
        horizon=args.horizon,
        train_end=args.train_end,
        output_dir=args.output,
        manifest_path=args.manifest,
        allow_2025_research_scope=args.allow_2025_research_scope,
    )
    print(
        json.dumps(
            {
                "model_dir": str(trained.model_dir),
                "rows_used": trained.metadata["public_training"]["rows_used"],
                "target_column": trained.target_column,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dengue-forecast-train")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True, help="Model-ready .parquet or .csv rows.")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument(
        "--train-end",
        required=True,
        help="Last target reporting-week end allowed in training labels, e.g. 2024-12-14.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory.")
    parser.add_argument("--allow-2025-research-scope", action="store_true")
    parser.set_defaults(func=_cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (PublicTrainingError, TrainingError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
