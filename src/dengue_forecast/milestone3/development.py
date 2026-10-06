from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.milestone3 import core
from dengue_forecast.modeling import metrics, preprocessing
from dengue_forecast.modeling.train import ModelConfig, TrainedModel, train_fold_model

MODEL_FAMILIES = ("ridge", "lightgbm_trial010")
FEATURE_SET_ORDER = ("cases_only", "cases_rainfall", "cases_full_weather")
HORIZONS = core.HORIZONS
FOLDS = core.DEVELOPMENT_FOLDS
CALENDAR_2025_CUTOFF = core.CALENDAR_2025_BOUNDARY
KEY_COLUMNS = ["district_id", "week_start_date"]
BASE_COLUMNS = ["district_id", "week_start_date", "week_end_date", "dengue_cases"]
ORIGIN_CALENDAR_PROVENANCE_COLUMNS = ["week"]
PRODUCTION_SOURCE_RELATIVE = Path("data/processed/ml_training_dataset.parquet")
PRODUCTION_OUTPUT_ROOT = core.REPO_ROOT / "artifacts" / "milestone3" / "development"
FIXTURE_OUTPUT_ROOTS = (
    core.REPO_ROOT / ".research" / "qa-temp",
    Path("/tmp/dengue-forecast-m3-qa-temp"),
)
PROTECTED_M1_M2_METADATA = core.REPO_ROOT / "docs" / "protected-m1-m2-before.json"
RAW_FEATURE_SUPPORT_COLUMNS = [
    "rainfall_sum_mm",
    "rain_days_1mm",
    "rain_days_10mm",
    "rainfall_mean_daily_mm",
    "rainfall_max_daily_mm",
    "temp_mean_c",
    "temp_min_c",
    "temp_max_c",
    "humidity_mean_pct",
]
REQUIRED_INTERVAL_BOUND_MEMBERS = (
    "train_start",
    "train_end",
    "evaluation_start",
    "evaluation_end",
    "first_scheduled_validation_origin",
    "max_training_target_end",
    "strict_embargo_limit_lt_first_origin",
    "embargo_days",
    "training_evaluation_disjoint",
)


class DevelopmentError(ValueError):
    """Raised when the bounded M3 development engine fails closed."""


@dataclass(frozen=True)
class FitSpec:
    horizon: int
    feature_set: str
    model_family: str
    fold: int

    @property
    def target_column(self) -> str:
        return f"target_h{self.horizon}"

    @property
    def fit_id(self) -> str:
        return (
            f"h{self.horizon}__{self.feature_set}__"
            f"{self.model_family}__fold{self.fold}"
        )

    def record(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "feature_set": self.feature_set,
            "model_family": self.model_family,
            "fold": self.fold,
            "fit_id": self.fit_id,
        }


@dataclass(frozen=True)
class DevelopmentResult:
    run_dir: Path
    run_id: str
    fit_count: int
    registry_count: int
    selection_path: Path
    registry_path: Path
    read_audit_path: Path


Reader = Callable[[Path, list[str], Any, dict[str, Any]], pd.DataFrame]
Fitter = Callable[..., TrainedModel]


@dataclass(frozen=True)
class SourceApproval:
    mode: str
    approved_path: Path
    approved_sha256: str


@dataclass(frozen=True)
class FrozenCohort:
    spec: FitSpec
    training: pd.DataFrame
    evaluation: pd.DataFrame
    feature_columns: list[str]
    thresholds: dict[str, Any]
    training_keys: dict[str, Any]
    evaluation_keys: dict[str, Any]
    training_digests: dict[str, Any]
    evaluation_digests: dict[str, Any]
    threshold_digest: str
    interval_bounds: dict[str, Any]


def expected_schedule() -> list[FitSpec]:
    specs = [
        FitSpec(horizon, feature_set, model_family, fold)
        for horizon in HORIZONS
        for feature_set in FEATURE_SET_ORDER
        for model_family in MODEL_FAMILIES
        for fold in FOLDS
    ]
    if len(specs) != 144 or len({spec.fit_id for spec in specs}) != 144:
        raise DevelopmentError("development schedule must contain exactly 144 unique fits")
    return specs


def projected_columns(
    feature_sets: Mapping[str, list[str]] | None = None,
    *,
    include_origin_provenance: bool = False,
) -> list[str]:
    sets = feature_sets or {name: core.selected_feature_columns(name) for name in FEATURE_SET_ORDER}
    ordered = [*BASE_COLUMNS]
    if include_origin_provenance:
        ordered.extend(ORIGIN_CALENDAR_PROVENANCE_COLUMNS)
    ordered.extend(RAW_FEATURE_SUPPORT_COLUMNS)
    for name in FEATURE_SET_ORDER:
        for column in sets[name]:
            if column not in ordered:
                ordered.append(column)
    return ordered


def load_protocol(protocol_config: str | Path = core.FIXED_PROTOCOL_CONFIG) -> dict[str, Any]:
    path = Path(protocol_config)
    config_bytes = path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("authorization_sha256") != core.AUTHORIZATION_SHA256:
        raise DevelopmentError("protocol config authorization hash mismatch")
    loaded = core.protocol_load()
    if _canonicalize(config) != _canonicalize(loaded):
        raise DevelopmentError("protocol config does not exactly match repaired core protocol")
    loaded["machine_config_path"] = str(path)
    loaded["machine_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    loaded["machine_config"] = config
    return loaded


def load_development_data(
    source_path: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    approved_source_sha256: str | None = None,
    fixture_mode: bool = False,
    reader: Reader | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = protocol or load_protocol()
    approval = _approve_source(
        source_path,
        fixture_mode=fixture_mode,
        approved_sha256=approved_source_sha256,
    )
    path = _contained_existing_file(approval.approved_path)
    source_schema = _source_schema_audit(path)
    origin_calendar_provenance_columns = [
        column
        for column in ORIGIN_CALENDAR_PROVENANCE_COLUMNS
        if column in source_schema["column_names"]
    ]
    columns = projected_columns(
        protocol["features"]["sets"],
        include_origin_provenance=bool(origin_calendar_provenance_columns),
    )
    cutoff = pd.Timestamp(
        protocol["targets"].get("calendar_2025_boundary", CALENDAR_2025_CUTOFF)
    )
    if cutoff != CALENDAR_2025_CUTOFF:
        raise DevelopmentError("development loader only accepts calendar 2025 cutoff")
    predicates = {
        "week_start_date_lt": cutoff.date().isoformat(),
        "week_end_date_lt": cutoff.date().isoformat(),
        "scheduled_boundary_status": (
            "pending_final_2025_boundary; development read uses calendar cutoff only"
        ),
    }
    before = _file_identity(path)
    if before["sha256"] != approval.approved_sha256:
        raise DevelopmentError("source sha256 does not match approved input identity")
    read_identity = {
        "source_mode": approval.mode,
        "source_path": str(path),
        "approved_source_sha256": approval.approved_sha256,
        "source_sha256": before["sha256"],
        "full_file_byte_sha256": before["sha256"],
        "full_file_byte_hash_purpose": (
            "identity audit over the entire approved source file before projected "
            "observation reads"
        ),
        "source_identity_before_read": before,
        "projection": columns,
        "origin_calendar_provenance_columns": origin_calendar_provenance_columns,
        "origin_calendar_provenance_policy": (
            "non-model source metadata projected only to preserve M2 origin "
            "calendar feature semantics during independent recomputation"
        ),
        "source_schema": source_schema,
        "predicates": predicates,
        "purpose": (
            "m3_development_synthetic_fixture_pre2025"
            if approval.mode == "synthetic_fixture"
            else "m3_development_production_pre2025_approved_source"
        ),
    }
    active_reader = reader or _pyarrow_projected_reader
    frame = active_reader(path, columns, _pyarrow_filter_expression(cutoff), read_identity)
    after = _file_identity(path)
    if after != before:
        raise DevelopmentError("source identity changed during read")
    validated = validate_loaded_observations(frame, expected_columns=columns, cutoff=cutoff)
    read_identity.update(_frame_receipt(validated))
    read_identity["source_identity_after_read"] = after
    read_identity["schema_audit"] = _schema_audit(validated)
    return validated, read_identity


def validate_loaded_observations(
    frame: pd.DataFrame,
    *,
    expected_columns: list[str],
    cutoff: pd.Timestamp = CALENDAR_2025_CUTOFF,
) -> pd.DataFrame:
    actual = list(frame.columns)
    if actual != expected_columns:
        raise DevelopmentError("development read returned unexpected projection")
    out = frame.copy()
    for column in ("week_start_date", "week_end_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce")
    if out.empty:
        raise DevelopmentError("development read returned no rows")
    if out[["week_start_date", "week_end_date"]].isna().any().any():
        raise DevelopmentError("development read returned invalid dates")
    if out["week_end_date"].ne(out["week_start_date"] + pd.Timedelta(days=6)).any():
        raise DevelopmentError("development read returned non-weekly intervals")
    if out.duplicated(KEY_COLUMNS).any():
        raise DevelopmentError("development read returned duplicate district/week keys")
    if out["week_start_date"].ge(cutoff).any() or out["week_end_date"].ge(cutoff).any():
        raise DevelopmentError("development read returned 2025-or-later observations")
    if out["district_id"].isna().any():
        raise DevelopmentError("development read returned null district_id")
    for column in ["dengue_cases", "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4"]:
        if column in out.columns:
            _require_finite_nonnegative(out[column], column, allow_na=True)
    return out.sort_values(KEY_COLUMNS).reset_index(drop=True)


def run_development(
    *,
    source_path: str | Path,
    output_root: str | Path,
    run_id: str,
    protocol_config: str | Path = core.FIXED_PROTOCOL_CONFIG,
    approved_source_sha256: str | None = None,
    fixture_mode: bool = False,
    reader: Reader | None = None,
    fitter: Fitter = train_fold_model,
) -> DevelopmentResult:
    protocol = load_protocol(protocol_config)
    run_dir = _claim_run_dir(output_root, run_id, fixture_mode=fixture_mode)
    try:
        observations, read_audit = load_development_data(
            source_path,
            protocol=protocol,
            approved_source_sha256=approved_source_sha256,
            fixture_mode=fixture_mode,
            reader=reader,
        )
        observation_content_before_audit = _sha256_obj(_canonical_frame_records(observations))
        causal_audit = _audit_master_features(observations, protocol)
        if _sha256_obj(_canonical_frame_records(observations)) != observation_content_before_audit:
            raise DevelopmentError("observations changed during causal feature audit")
        tasks = core.build_development_direct_tasks(
            observations,
            test_boundary=CALENDAR_2025_CUTOFF,
            horizons=HORIZONS,
        )
        if _sha256_obj(_canonical_frame_records(observations)) != observation_content_before_audit:
            raise DevelopmentError("observations changed during task construction")
        _assert_no_crossing_targets(tasks)
        schedule = expected_schedule()
        frozen = _precompute_frozen_cohorts(tasks, protocol, schedule)
        _write_json_new(
            run_dir / "run_identity.json",
            _run_identity(run_id, protocol, read_audit, causal_audit),
        )
        _write_json_new(run_dir / "read_audit.json", read_audit)
        _write_json_new(run_dir / "causal_feature_audit.json", causal_audit)
        _write_json_new(run_dir / "schedule.json", {"fits": [spec.record() for spec in schedule]})
        _write_json_new(run_dir / "cohort_plan.json", _cohort_plan_receipt(frozen))
        registry = _run_all_fits(run_dir, tasks, protocol, schedule, frozen, fitter=fitter)
        registry_path = run_dir / "model_registry.parquet"
        _write_parquet_new(pd.DataFrame(registry), registry_path)
        selection = select_champions(pd.DataFrame(registry))
        selection_path = run_dir / "selection.json"
        _write_json_new(selection_path, selection)
        _strict_validate_completed_run(run_dir, pd.DataFrame(registry), selection)
        _verify_source_identity_at_completion(read_audit)
        checksums = _checksums(run_dir)
        _write_json_new(run_dir / "checksums.json", checksums)
        _write_json_new(
            run_dir / "complete.json",
            {
                "run_id": run_id,
                "fit_count": len(registry),
                "registry_count": int(len(registry)),
                "selection_path": str(selection_path.relative_to(run_dir)),
                "study_designation": protocol["study_designation"],
            },
        )
    except Exception:
        _write_failure_marker(run_dir)
        raise
    return DevelopmentResult(
        run_dir=run_dir,
        run_id=run_id,
        fit_count=len(registry),
        registry_count=len(registry),
        selection_path=selection_path,
        registry_path=registry_path,
        read_audit_path=run_dir / "read_audit.json",
    )


def validate_development_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir).resolve()
    if (path / "failed.json").exists():
        raise DevelopmentError("completed run contains failure marker")
    complete = _read_json(path / "complete.json")
    registry_path = path / "model_registry.parquet"
    selection_path = path / "selection.json"
    checksums = _read_json(path / "checksums.json")
    registry = pd.read_parquet(registry_path)
    _validate_registry(registry)
    if int(complete.get("fit_count", -1)) != 144 or len(registry) != 144:
        raise DevelopmentError("completed run does not contain 144 registry rows")
    selection = _read_json(selection_path)
    _strict_validate_completed_run(path, registry, selection)
    expected_manifest = _checksums(path)
    if set(checksums) != set(expected_manifest):
        raise DevelopmentError(
            "completed run checksum manifest does not contain exact artifact set"
        )
    for rel_path, expected in checksums.items():
        target = path / rel_path
        if rel_path == "checksums.json":
            continue
        if rel_path not in expected_manifest or expected_manifest[rel_path] != expected:
            raise DevelopmentError(f"completed run checksum mismatch: {rel_path}")
        if (
            not _is_relative_to(target.resolve(), path)
            or not target.exists()
            or sha256_file(target) != expected
        ):
            raise DevelopmentError(f"completed run checksum mismatch: {rel_path}")
    observed_selection = select_champions(registry)
    if _canonicalize(observed_selection) != _canonicalize(selection):
        raise DevelopmentError("completed run selection is not derived from registry")
    return {
        "run_dir": str(path),
        "fit_count": int(len(registry)),
        "registry_count": int(len(registry)),
        "selection_sha256": sha256_file(selection_path),
        "status": "completed_read_only_verified",
    }


def select_champions(registry: pd.DataFrame) -> dict[str, Any]:
    _validate_registry(registry)
    rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_rows = registry.loc[registry["horizon"].eq(horizon)]
        for feature_set in FEATURE_SET_ORDER:
            for model_family in MODEL_FAMILIES:
                candidate = horizon_rows.loc[
                    horizon_rows["feature_set"].eq(feature_set)
                    & horizon_rows["model_family"].eq(model_family)
                ].sort_values("fold")
                if len(candidate) != 6:
                    raise DevelopmentError("selection requires six folds per candidate")
                fold_maes = [float(value) for value in candidate["mae_model"]]
                fold_rmses = [float(value) for value in candidate["rmse_model"]]
                rows.append(
                    {
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "model_family": model_family,
                        "fold_maes": fold_maes,
                        "mean_fold_mae": float(np.mean(fold_maes)),
                        "mean_fold_rmse": float(np.mean(fold_rmses)),
                        "family_tie_order": 0 if model_family == "ridge" else 1,
                        "feature_tie_order": FEATURE_SET_ORDER.index(feature_set),
                        "declared_feature_count": len(core.selected_feature_columns(feature_set)),
                    }
                )
        ranked = sorted(
            [row for row in rows if row["horizon"] == horizon],
            key=lambda row: (
                row["mean_fold_mae"],
                row["family_tie_order"],
                row["declared_feature_count"],
                row["feature_tie_order"],
            ),
        )
        selections[f"h{horizon}"] = ranked[0]
    return {
        "champion_metric": "unweighted_mean_of_six_fold_mae",
        "tie_policy": [
            "lowest_mean_fold_mae",
            "ridge_before_lightgbm",
            "fewer_declared_input_features",
            "cases_only_cases_rainfall_cases_full_weather",
        ],
        "candidate_table": rows,
        "selected": selections,
        "development_selection_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the bounded Milestone 3 development engine."
    )
    parser.add_argument("--source-parquet", type=Path, help="Projected pre-2025 source parquet")
    parser.add_argument("--output-root", type=Path, help="Directory that will receive run_id")
    parser.add_argument("--run-id", help="New run id; existing output is rejected")
    parser.add_argument("--approved-source-sha256", help="Required production approved source hash")
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Explicit synthetic fixture mode for tests; never production",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=core.FIXED_PROTOCOL_CONFIG,
        help="Machine-readable M3 protocol config",
    )
    parser.add_argument(
        "--validate-existing",
        type=Path,
        help="Read-only verification of a completed development run",
    )
    args = parser.parse_args(argv)
    if args.validate_existing is not None:
        receipt = validate_development_run(args.validate_existing)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    missing = [
        name
        for name, value in {
            "--source-parquet": args.source_parquet,
            "--output-root": args.output_root,
            "--run-id": args.run_id,
        }.items()
        if value is None
    ]
    if missing:
        parser.error(f"missing required arguments for a new run: {', '.join(missing)}")
    result = run_development(
        source_path=args.source_parquet,
        output_root=args.output_root,
        run_id=args.run_id,
        protocol_config=args.protocol_config,
        approved_source_sha256=args.approved_source_sha256,
        fixture_mode=args.fixture_mode,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "fit_count": result.fit_count,
                "registry_count": result.registry_count,
                "selection_path": str(result.selection_path),
                "registry_path": str(result.registry_path),
                "read_audit_path": str(result.read_audit_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_all_fits(
    run_dir: Path,
    tasks: pd.DataFrame,
    protocol: Mapping[str, Any],
    schedule: list[FitSpec],
    frozen: Mapping[str, FrozenCohort],
    *,
    fitter: Fitter,
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    written_eval: set[int] = set()
    for spec in schedule:
        cohort = frozen[spec.fit_id]
        fold_dir = run_dir / "cohorts" / f"fold{spec.fold}"
        if spec.fold not in written_eval:
            fold_dir.mkdir(parents=True, exist_ok=False)
            _write_parquet_new(cohort.evaluation[KEY_COLUMNS], fold_dir / "evaluation_keys.parquet")
            written_eval.add(spec.fold)
        registry.append(
            _run_one_fit(
                run_dir,
                protocol,
                cohort,
                fitter=fitter,
            )
        )
    _validate_registry(pd.DataFrame(registry))
    return registry


def _precompute_frozen_cohorts(
    tasks: pd.DataFrame,
    protocol: Mapping[str, Any],
    schedule: list[FitSpec],
) -> dict[str, FrozenCohort]:
    frozen: dict[str, FrozenCohort] = {}
    eval_by_fold: dict[int, pd.DataFrame] = {}
    first_origin_by_fold: dict[int, pd.Timestamp] = {}
    for fold in FOLDS:
        fold_rows = tasks.loc[pd.to_datetime(tasks["week_start_date"]).dt.year.eq(fold)].copy()
        fold_rows = fold_rows.reset_index(drop=True)
        eval_rows = fold_rows.loc[
            core.common_evaluation_mask(fold_rows, boundary=CALENDAR_2025_CUTOFF)
        ]
        if eval_rows.empty:
            raise DevelopmentError(f"fold {fold} has empty common evaluation cohort")
        eval_rows = _canonical(eval_rows)
        first_origin = pd.to_datetime(fold_rows["week_start_date"]).min()
        if pd.isna(first_origin):
            raise DevelopmentError(f"fold {fold} has no scheduled validation origins")
        eval_by_fold[fold] = eval_rows
        first_origin_by_fold[fold] = pd.Timestamp(first_origin)
    for spec in schedule:
        feature_columns = core.selected_feature_columns(spec.feature_set, frame=tasks)
        training = core.training_frame(
            tasks,
            horizon=spec.horizon,
            first_origin=first_origin_by_fold[spec.fold],
            test_boundary=CALENDAR_2025_CUTOFF,
        )
        if training.empty:
            raise DevelopmentError(f"{spec.fit_id} has empty training cohort")
        training = _canonical(training)
        eval_rows = eval_by_fold[spec.fold]
        _assert_strict_embargo_and_disjoint(
            training,
            eval_rows,
            spec,
            first_origin_by_fold[spec.fold],
        )
        thresholds = core.threshold_binding(
            training,
            target_column=spec.target_column,
            feature_columns=(),
        )
        training_digests = core.cohort_digests(
            training,
            target_column=spec.target_column,
            feature_columns=feature_columns,
        )
        evaluation_digests = core.cohort_digests(
            eval_rows,
            target_column=spec.target_column,
            feature_columns=feature_columns,
        )
        frozen[spec.fit_id] = FrozenCohort(
            spec=spec,
            training=training,
            evaluation=eval_rows,
            feature_columns=feature_columns,
            thresholds=thresholds,
            training_keys=core.freeze_common_keys(training),
            evaluation_keys=core.freeze_common_keys(eval_rows),
            training_digests=training_digests,
            evaluation_digests=evaluation_digests,
            threshold_digest=_sha256_obj(thresholds),
            interval_bounds=_interval_bounds(
                training,
                eval_rows,
                spec,
                first_origin_by_fold[spec.fold],
            ),
        )
    _validate_shared_cohorts(frozen)
    # Touch protocol so config drift cannot be hidden by callers that skip strict loading.
    if protocol["features"]["sets"] != {
        name: core.selected_feature_columns(name) for name in FEATURE_SET_ORDER
    }:
        raise DevelopmentError("effective protocol feature sets changed before fit")
    return frozen


def _validate_shared_cohorts(frozen: Mapping[str, FrozenCohort]) -> None:
    eval_digest_by_fold: dict[int, str] = {}
    train_digest_by_hf: dict[tuple[int, int], str] = {}
    threshold_by_hf: dict[tuple[int, int], str] = {}
    for cohort in frozen.values():
        spec = cohort.spec
        eval_digest = cohort.evaluation_keys["row_key_digest"]
        existing_eval = eval_digest_by_fold.setdefault(spec.fold, eval_digest)
        if existing_eval != eval_digest:
            raise DevelopmentError(f"fold {spec.fold} evaluation cohort is not shared")
        key = (spec.horizon, spec.fold)
        train_digest = cohort.training_keys["row_key_digest"]
        threshold_digest = cohort.threshold_digest
        existing_train = train_digest_by_hf.setdefault(key, train_digest)
        existing_threshold = threshold_by_hf.setdefault(key, threshold_digest)
        if existing_train != train_digest or existing_threshold != threshold_digest:
            raise DevelopmentError(f"h{spec.horizon}/fold{spec.fold} training cohort is not shared")


def _cohort_plan_receipt(frozen: Mapping[str, FrozenCohort]) -> dict[str, Any]:
    return {
        "fit_count": len(frozen),
        "fits": {
            fit_id: {
                **cohort.spec.record(),
                "training_keys": cohort.training_keys,
                "evaluation_keys": cohort.evaluation_keys,
                "training_digests": cohort.training_digests,
                "evaluation_digests": cohort.evaluation_digests,
                "threshold_binding_sha256": cohort.threshold_digest,
                "feature_list_sha256": core.feature_list_digest(cohort.feature_columns),
                "interval_bounds": cohort.interval_bounds,
            }
            for fit_id, cohort in sorted(frozen.items())
        },
    }


def _run_one_fit(
    run_dir: Path,
    protocol: Mapping[str, Any],
    cohort: FrozenCohort,
    *,
    fitter: Fitter,
) -> dict[str, Any]:
    spec = cohort.spec
    feature_columns = list(cohort.feature_columns)
    training = cohort.training.copy(deep=True)
    eval_rows = cohort.evaluation.copy(deep=True)
    train_dir = run_dir / "cohorts" / f"fold{spec.fold}" / f"h{spec.horizon}"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_keys_path = train_dir / "training_keys.parquet"
    if not train_keys_path.exists():
        _write_parquet_new(training[KEY_COLUMNS], train_keys_path)
    else:
        saved_train = pd.read_parquet(train_keys_path)
        core.verify_common_keys(saved_train, cohort.training_keys)
    thresholds = dict(cohort.thresholds)
    threshold_path = train_dir / "thresholds.json"
    if not threshold_path.exists():
        _write_json_new(threshold_path, thresholds)
    else:
        if _read_json(threshold_path) != thresholds:
            raise DevelopmentError(f"{spec.fit_id} threshold binding mismatch")
    fit_dir = run_dir / "fits" / spec.fit_id
    fit_dir.mkdir(parents=True, exist_ok=False)
    _write_json_new(fit_dir / "claim.json", {"fit": spec.record(), "status": "claimed"})
    model_dir = fit_dir / "model"
    if model_dir.exists():
        raise DevelopmentError(f"{spec.fit_id} model path exists before fitter entry")
    before_training_digest = core.cohort_digests(
        training,
        target_column=spec.target_column,
        feature_columns=feature_columns,
    )
    if _canonicalize(before_training_digest) != _canonicalize(cohort.training_digests):
        raise DevelopmentError(f"{spec.fit_id} training cohort differs from frozen plan")
    before_evaluation_digest = core.cohort_digests(
        eval_rows,
        target_column=spec.target_column,
        feature_columns=feature_columns,
    )
    if _canonicalize(before_evaluation_digest) != _canonicalize(cohort.evaluation_digests):
        raise DevelopmentError(f"{spec.fit_id} evaluation cohort differs from frozen plan")
    model = fitter(
        training,
        feature_columns=feature_columns,
        target_column=spec.target_column,
        config=_model_config(protocol, spec.model_family),
        output_dir=model_dir,
        frozen_period_bounds={
            "train_start": training["week_start_date"].min().date().isoformat(),
            "train_end": training["week_start_date"].max().date().isoformat(),
        },
        provenance={
            "m3_fit_id": spec.fit_id,
            "m3_horizon": spec.horizon,
            "m3_fold": spec.fold,
            "m3_feature_set": spec.feature_set,
        },
    )
    after_training_digest = core.cohort_digests(
        training,
        target_column=spec.target_column,
        feature_columns=feature_columns,
    )
    if _canonicalize(before_training_digest) != _canonicalize(after_training_digest):
        raise DevelopmentError(f"{spec.fit_id} fitter mutated frozen training cohort")
    _verify_training_identity(training, model)
    _verify_returned_model(spec, model, _model_config(protocol, spec.model_family), feature_columns)
    _verify_persisted_metadata(
        spec,
        model,
        model_dir / "metadata.json",
        _model_config(protocol, spec.model_family),
        feature_columns,
    )
    persisted_thresholds = _read_json(threshold_path)
    core.verify_threshold_binding(training, persisted_thresholds, target_column=spec.target_column)
    pre_prediction_eval_digest = core.cohort_digests(
        eval_rows,
        target_column=spec.target_column,
        feature_columns=feature_columns,
    )
    if _canonicalize(pre_prediction_eval_digest) != _canonicalize(cohort.evaluation_digests):
        raise DevelopmentError(f"{spec.fit_id} evaluation cohort changed before prediction")
    predictions = _prediction_frame(spec, model, eval_rows, feature_columns, thresholds)
    post_prediction_eval_digest = core.cohort_digests(
        eval_rows,
        target_column=spec.target_column,
        feature_columns=feature_columns,
    )
    if _canonicalize(post_prediction_eval_digest) != _canonicalize(cohort.evaluation_digests):
        raise DevelopmentError(f"{spec.fit_id} prediction mutated frozen evaluation cohort")
    pred_path = fit_dir / "predictions.parquet"
    _write_parquet_new(predictions, pred_path)
    saved_predictions = pd.read_parquet(pred_path)
    _verify_prediction_frame(saved_predictions, predictions)
    prediction_sha = sha256_file(pred_path)
    metric_doc = _metric_document(saved_predictions, persisted_thresholds)
    metrics_path = fit_dir / "metrics.json"
    _write_json_new(metrics_path, metric_doc)
    complete_path = fit_dir / "complete.json"
    record = {
        **spec.record(),
        "status": "complete",
        "training_count": int(len(training)),
        "evaluation_count": int(len(eval_rows)),
        "training_key_digest": core.freeze_common_keys(training)["row_key_digest"],
        "evaluation_key_digest": core.freeze_common_keys(eval_rows)["row_key_digest"],
        "training_feature_content_digest": cohort.training_digests["feature_content_digest"],
        "training_target_content_digest": cohort.training_digests["target_content_digest"],
        "evaluation_feature_content_digest": cohort.evaluation_digests["feature_content_digest"],
        "evaluation_target_content_digest": cohort.evaluation_digests["target_content_digest"],
        "threshold_path": str(threshold_path.relative_to(run_dir)),
        "threshold_sha256": sha256_file(threshold_path),
        "threshold_binding_sha256": cohort.threshold_digest,
        "model_path": str(model_dir.relative_to(run_dir)),
        "model_sha256": sha256_file(model_dir / "model.joblib"),
        "model_metadata_sha256": sha256_file(model_dir / "metadata.json"),
        "prediction_path": str(pred_path.relative_to(run_dir)),
        "prediction_sha256": prediction_sha,
        "metrics_path": str(metrics_path.relative_to(run_dir)),
        "metrics_sha256": sha256_file(metrics_path),
        "config_sha256": _sha256_obj(_model_config(protocol, spec.model_family).serializable()),
        "feature_list_sha256": core.feature_list_digest(feature_columns),
        "interval_bounds": cohort.interval_bounds,
        "mae_model": metric_doc["model"]["mae"]["value"],
        "rmse_model": metric_doc["model"]["rmse"]["value"],
        "mae_persistence": metric_doc["persistence"]["mae"]["value"],
        "rmse_persistence": metric_doc["persistence"]["rmse"]["value"],
        "strict_mae_win": bool(
            metric_doc["model"]["mae"]["value"] < metric_doc["persistence"]["mae"]["value"]
        ),
        "strict_rmse_win": bool(
            metric_doc["model"]["rmse"]["value"] < metric_doc["persistence"]["rmse"]["value"]
        ),
    }
    _write_json_new(complete_path, record)
    return record


def _prediction_frame(
    spec: FitSpec,
    model: TrainedModel,
    eval_rows: pd.DataFrame,
    feature_columns: list[str],
    thresholds: Mapping[str, Any],
) -> pd.DataFrame:
    features = eval_rows.loc[:, feature_columns]
    values = np.asarray(model.predict_next_week(features), dtype="float64")
    if len(values) != len(eval_rows) or not np.isfinite(values).all() or (values < 0).any():
        raise DevelopmentError(f"{spec.fit_id} produced invalid predictions")
    out = pd.DataFrame(
        {
            "fit_id": spec.fit_id,
            "fold": spec.fold,
            "horizon": spec.horizon,
            "model_family": spec.model_family,
            "feature_set": spec.feature_set,
            "district_id": eval_rows["district_id"].astype(str).to_numpy(),
            "origin_start": pd.to_datetime(eval_rows["week_start_date"]).dt.date.astype(str),
            "origin_end": pd.to_datetime(eval_rows["week_end_date"]).dt.date.astype(str),
            "target_start": (
                pd.to_datetime(eval_rows["week_start_date"])
                + pd.to_timedelta(7 * spec.horizon, unit="D")
            ).dt.date.astype(str),
            "target_end": (
                pd.to_datetime(eval_rows["week_start_date"])
                + pd.to_timedelta(7 * spec.horizon + 6, unit="D")
            ).dt.date.astype(str),
            "observed_current_cases": pd.to_numeric(eval_rows["dengue_cases"]).to_numpy(
                dtype="float64"
            ),
            "observed_target": pd.to_numeric(eval_rows[spec.target_column]).to_numpy(
                dtype="float64"
            ),
            "prediction_model": values,
            "prediction_persistence": pd.to_numeric(eval_rows["dengue_cases"]).to_numpy(
                dtype="float64"
            ),
            "training_binding_id": thresholds["digests"]["row_key_digest"],
            "evaluation_binding_id": core.freeze_common_keys(eval_rows)["row_key_digest"],
            "threshold_binding_id": _sha256_obj(thresholds),
        }
    )
    if out.duplicated(["fit_id", "district_id", "origin_start"]).any():
        raise DevelopmentError(f"{spec.fit_id} prediction rows are not unique")
    return out


def _metric_document(predictions: pd.DataFrame, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    y = predictions["observed_target"]
    model = predictions["prediction_model"]
    persistence = predictions["prediction_persistence"]
    model_metrics = _clean_metrics(y, model, thresholds)
    persistence_metrics = _clean_metrics(y, persistence, thresholds)
    baseline_mae = persistence_metrics["mae"]["value"]
    if baseline_mae in (None, 0.0):
        relative = {"value": None, "reason": "zero_or_undefined_persistence_mae"}
    else:
        relative = {
            "value": 100.0 * (baseline_mae - model_metrics["mae"]["value"]) / baseline_mae,
            "reason": None,
        }
    return {
        "prediction_first_source": "metrics computed only after predictions parquet readback",
        "prediction_input_sha256": _sha256_obj(_canonical_frame_records(predictions)),
        "n": int(len(predictions)),
        "model": model_metrics,
        "persistence": persistence_metrics,
        "mae_model_minus_persistence": model_metrics["mae"]["value"] - baseline_mae,
        "relative_mae_improvement_pct": relative,
    }


def _clean_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    thresholds: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    bundle = metrics.metric_bundle(y_true, y_pred, top_decile_threshold=thresholds["q90_incidence"])
    high = metrics.high_incidence_metrics(
        y_true,
        y_pred,
        top_10_threshold=thresholds["q90_incidence"],
        top_5_threshold=thresholds["q95_incidence"],
    )
    result = {name: _metric_value(value) for name, value in {**bundle, **high}.items()}
    result["top_10_count"] = {
        "value": int((pd.Series(y_true) >= thresholds["q90_incidence"]).sum()),
        "reason": None,
    }
    result["top_5_count"] = {
        "value": int((pd.Series(y_true) >= thresholds["q95_incidence"]).sum()),
        "reason": None,
    }
    return result


def _metric_value(value: Any) -> dict[str, Any]:
    numeric = float(value)
    if not np.isfinite(numeric):
        return {"value": None, "reason": "undefined_for_cohort"}
    return {"value": numeric, "reason": None}


def _validate_registry(registry: pd.DataFrame) -> None:
    required = {
        "horizon",
        "feature_set",
        "model_family",
        "fold",
        "fit_id",
        "status",
        "training_count",
        "evaluation_count",
        "training_key_digest",
        "evaluation_key_digest",
        "training_feature_content_digest",
        "training_target_content_digest",
        "evaluation_feature_content_digest",
        "evaluation_target_content_digest",
        "threshold_path",
        "threshold_sha256",
        "threshold_binding_sha256",
        "model_path",
        "model_sha256",
        "model_metadata_sha256",
        "prediction_path",
        "prediction_sha256",
        "metrics_path",
        "metrics_sha256",
        "config_sha256",
        "feature_list_sha256",
        "interval_bounds",
        "mae_model",
        "rmse_model",
        "mae_persistence",
        "rmse_persistence",
        "strict_mae_win",
        "strict_rmse_win",
    }
    missing = required - set(registry.columns)
    if missing:
        raise DevelopmentError(f"registry missing columns: {sorted(missing)}")
    expected = {(s.horizon, s.feature_set, s.model_family, s.fold) for s in expected_schedule()}
    observed = {
        (int(row.horizon), str(row.feature_set), str(row.model_family), int(row.fold))
        for row in registry.itertuples(index=False)
    }
    if observed != expected or len(registry) != 144:
        raise DevelopmentError("registry tuple set does not match expected 144 fits")
    if registry.duplicated(["horizon", "feature_set", "model_family", "fold"]).any():
        raise DevelopmentError("registry contains duplicate fit tuple")
    if not registry["fit_id"].is_unique:
        raise DevelopmentError("registry contains duplicate fit ids")
    if set(registry["status"].astype(str)) != {"complete"}:
        raise DevelopmentError("registry contains non-complete fit status")
    for column in ("prediction_path", "metrics_path", "model_path", "threshold_path"):
        if not registry[column].astype(str).is_unique and column != "threshold_path":
            raise DevelopmentError(f"registry contains duplicate {column}")
    for column in ("mae_model", "rmse_model", "mae_persistence", "rmse_persistence"):
        if not np.isfinite(pd.to_numeric(registry[column], errors="coerce")).all():
            raise DevelopmentError(f"registry contains missing/nonfinite {column}")


def _pyarrow_projected_reader(
    path: Path,
    columns: list[str],
    filter_expression: Any,
    identity: Mapping[str, Any],
) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    schema = dataset.schema
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise DevelopmentError(f"source parquet missing projected columns: {missing}")
    for column in ("week_start_date", "week_end_date"):
        field = schema.field(column)
        if not (pa.types.is_date(field.type) or pa.types.is_timestamp(field.type)):
            raise DevelopmentError(f"{column} must be a physical date/timestamp type")
    table = dataset.to_table(columns=columns, filter=filter_expression)
    return table.to_pandas()


def _pyarrow_filter_expression(cutoff: pd.Timestamp) -> Any:
    import pyarrow.dataset as ds

    return (ds.field("week_start_date") < cutoff.date()) & (
        ds.field("week_end_date") < cutoff.date()
    )


def _model_config(protocol: Mapping[str, Any], family: str) -> ModelConfig:
    if family == "ridge":
        ridge = protocol["models"]["ridge"]
        return ModelConfig(
            family="ridge",
            objective="regression",
            hyperparams=dict(ridge["hyperparams"]),
            seed=int(ridge["seed"]),
        )
    if family == "lightgbm_trial010":
        lightgbm = protocol["models"]["lightgbm_trial010"]
        return ModelConfig(
            family="lightgbm",
            objective="poisson",
            hyperparams=dict(lightgbm["hyperparams"]),
            seed=int(lightgbm["seed"]),
        )
    raise DevelopmentError(f"unknown model family: {family}")


def _verify_training_identity(training: pd.DataFrame, model: TrainedModel) -> None:
    expected = [
        {
            "district_id": str(row.district_id),
            "week_start_date": pd.Timestamp(row.week_start_date).date().isoformat(),
        }
        for row in training[KEY_COLUMNS].itertuples(index=False)
    ]
    observed = model.metadata["preprocessing_fit_state"]["training_row_keys"]
    if observed != expected:
        raise DevelopmentError("trained preprocessor row identity mismatch")


def _verify_prediction_frame(saved: pd.DataFrame, expected: pd.DataFrame) -> None:
    if list(saved.columns) != list(expected.columns) or len(saved) != len(expected):
        raise DevelopmentError("prediction readback schema/count mismatch")
    saved_norm = pd.DataFrame(_canonical_frame_records(saved))
    expected_norm = pd.DataFrame(_canonical_frame_records(expected))
    if not saved_norm.equals(expected_norm):
        raise DevelopmentError("prediction readback value mismatch")
    if not saved["observed_current_cases"].equals(saved["prediction_persistence"]):
        raise DevelopmentError("prediction persistence does not equal current observed cases")


def _assert_no_crossing_targets(frame: pd.DataFrame) -> None:
    dates = pd.to_datetime(frame["week_start_date"])
    for horizon in HORIZONS:
        has_label = frame[f"target_h{horizon}"].notna()
        target_end = dates + pd.to_timedelta(7 * horizon + 6, unit="D")
        if target_end.loc[has_label].ge(CALENDAR_2025_CUTOFF).any():
            raise DevelopmentError("development target crosses calendar 2025 cutoff")


def _assert_strict_embargo_and_disjoint(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    spec: FitSpec,
    first_origin: pd.Timestamp,
) -> None:
    train_keys = set(
        zip(
            training["district_id"].astype(str),
            pd.to_datetime(training["week_start_date"]).dt.date.astype(str),
            strict=False,
        )
    )
    eval_keys = set(
        zip(
            evaluation["district_id"].astype(str),
            pd.to_datetime(evaluation["week_start_date"]).dt.date.astype(str),
            strict=False,
        )
    )
    if train_keys & eval_keys:
        raise DevelopmentError(f"{spec.fit_id} training and evaluation cohorts overlap")
    target_end = pd.to_datetime(training["week_start_date"]) + pd.to_timedelta(
        7 * spec.horizon + 6, unit="D"
    )
    if (target_end + pd.Timedelta(days=7)).ge(first_origin).any():
        raise DevelopmentError(f"{spec.fit_id} strict embargo violation")


def _interval_bounds(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    spec: FitSpec,
    first_origin: pd.Timestamp,
) -> dict[str, Any]:
    train_start = pd.to_datetime(training["week_start_date"]).min()
    train_end = pd.to_datetime(training["week_start_date"]).max()
    eval_start = pd.to_datetime(evaluation["week_start_date"]).min()
    eval_end = pd.to_datetime(evaluation["week_start_date"]).max()
    max_training_target_end = train_end + pd.Timedelta(days=7 * spec.horizon + 6)
    return {
        "train_start": train_start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "evaluation_start": eval_start.date().isoformat(),
        "evaluation_end": eval_end.date().isoformat(),
        "first_scheduled_validation_origin": first_origin.date().isoformat(),
        "max_training_target_end": max_training_target_end.date().isoformat(),
        "strict_embargo_limit_lt_first_origin": (
            max_training_target_end + pd.Timedelta(days=7)
        ).date().isoformat(),
        "embargo_days": 7,
        "training_evaluation_disjoint": True,
    }


def _audit_master_features(
    observations: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    union = projected_columns(protocol["features"]["sets"])
    audit_columns = [
        column
        for column in union
        if column not in {"week_end_date"}
        and column in set(core.PINNED_FEATURE_SETS["cases_full_weather"])
    ]
    before = _sha256_obj(_canonical_frame_records(observations))
    recomputed = core.add_origin_features(observations)
    result = core.compare_causal_feature_audit(
        recomputed,
        observations,
        feature_columns=audit_columns,
    )
    after = _sha256_obj(_canonical_frame_records(observations))
    if after != before:
        raise DevelopmentError("causal feature audit mutated observations")
    return {
        "audit_name": "m3_causal_origin_feature_audit",
        "ordered_feature_columns": audit_columns,
        "ordered_feature_list_sha256": core.feature_list_digest(audit_columns),
        "input_projected_content_digest": before,
        "recomputed_projected_content_digest": _sha256_obj(
            _canonical_frame_records(recomputed.loc[:, observations.columns])
        ),
        "audit_result": bool(result),
        "observation_content_unchanged": True,
    }


def _verify_returned_model(
    spec: FitSpec,
    model: TrainedModel,
    expected_config: ModelConfig,
    feature_columns: list[str],
) -> None:
    if model.target_column != spec.target_column:
        raise DevelopmentError(f"{spec.fit_id} returned model target mismatch")
    if model.feature_columns != feature_columns:
        raise DevelopmentError(f"{spec.fit_id} returned model feature mismatch")
    if _canonicalize(model.config.serializable()) != _canonicalize(expected_config.serializable()):
        raise DevelopmentError(f"{spec.fit_id} returned model config mismatch")
    metadata = model.metadata
    if metadata.get("target_column") != spec.target_column:
        raise DevelopmentError(f"{spec.fit_id} metadata target mismatch")
    if metadata.get("feature_columns") != feature_columns:
        raise DevelopmentError(f"{spec.fit_id} metadata feature mismatch")
    provenance = metadata.get("row_identity", {})
    for key, value in {
        "m3_fit_id": spec.fit_id,
        "m3_horizon": spec.horizon,
        "m3_fold": spec.fold,
        "m3_feature_set": spec.feature_set,
    }.items():
        if provenance.get(key) != value:
            raise DevelopmentError(f"{spec.fit_id} metadata provenance mismatch")


def _verify_persisted_metadata(
    spec: FitSpec,
    model: TrainedModel,
    metadata_path: Path,
    expected_config: ModelConfig,
    feature_columns: list[str],
) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    if _canonicalize(metadata) != _canonicalize(model.metadata):
        raise DevelopmentError(f"{spec.fit_id} persisted metadata differs from returned model")
    _verify_metadata_doc(spec, metadata, expected_config, feature_columns)
    return metadata


def _verify_metadata_doc(
    spec: FitSpec,
    metadata: Mapping[str, Any],
    expected_config: ModelConfig,
    feature_columns: list[str],
    expected_training_keys: Mapping[str, Any] | None = None,
) -> None:
    if metadata.get("target_column") != spec.target_column:
        raise DevelopmentError(f"{spec.fit_id} persisted metadata target mismatch")
    if metadata.get("feature_columns") != feature_columns:
        raise DevelopmentError(f"{spec.fit_id} persisted metadata feature mismatch")
    if _canonicalize(metadata.get("config")) != _canonicalize(expected_config.serializable()):
        raise DevelopmentError(f"{spec.fit_id} persisted metadata config mismatch")
    fit_state = metadata.get("preprocessing_fit_state", {})
    if fit_state.get("input_feature_columns") != feature_columns:
        raise DevelopmentError(f"{spec.fit_id} persisted preprocessing feature mismatch")
    if expected_training_keys is not None:
        training_row_keys = fit_state.get("training_row_keys")
        if not isinstance(training_row_keys, list) or not training_row_keys:
            raise DevelopmentError(
                f"{spec.fit_id} persisted preprocessing training keys missing"
            )
        if _canonicalize(training_row_keys) != _canonicalize(expected_training_keys.get("keys")):
            raise DevelopmentError(
                f"{spec.fit_id} persisted preprocessing training keys mismatch"
            )
    row_identity = metadata.get("row_identity", {})
    if row_identity.get("m3_fit_id") != spec.fit_id:
        raise DevelopmentError(f"{spec.fit_id} persisted metadata fit id mismatch")
    if row_identity.get("m3_horizon") != spec.horizon:
        raise DevelopmentError(f"{spec.fit_id} persisted metadata horizon mismatch")
    if row_identity.get("m3_fold") != spec.fold:
        raise DevelopmentError(f"{spec.fit_id} persisted metadata fold mismatch")
    if row_identity.get("m3_feature_set") != spec.feature_set:
        raise DevelopmentError(f"{spec.fit_id} persisted metadata feature-set mismatch")


def _claim_run_dir(output_root: str | Path, run_id: str, *, fixture_mode: bool = False) -> Path:
    if not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise DevelopmentError("run_id must be a simple path segment")
    root_input = Path(output_root)
    _reject_symlink_components(root_input)
    root = root_input.resolve()
    allowed_roots = tuple(path.resolve() for path in FIXTURE_OUTPUT_ROOTS) if fixture_mode else (
        PRODUCTION_OUTPUT_ROOT.resolve(),
    )
    if not any(_is_relative_to(root, allowed) for allowed in allowed_roots):
        raise DevelopmentError("development output root is outside allowed M3 development roots")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    try:
        run_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise DevelopmentError(
            "run output already exists; default policy rejects rerun/resume"
        ) from exc
    if run_dir.is_symlink():
        raise DevelopmentError("run output cannot be a symlink")
    return run_dir


def _contained_existing_file(path: str | Path) -> Path:
    candidate = Path(path)
    _reject_symlink_components(candidate)
    resolved = candidate.resolve()
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise DevelopmentError("source path must be an existing non-symlink file")
    return resolved


def _approve_source(
    source_path: str | Path,
    *,
    fixture_mode: bool,
    approved_sha256: str | None,
) -> SourceApproval:
    path = Path(source_path)
    if fixture_mode:
        if not approved_sha256:
            raise DevelopmentError("fixture mode requires explicit approved source sha256")
        approved_path = _contained_existing_file(path)
        allowed_roots = tuple(root.resolve() for root in FIXTURE_OUTPUT_ROOTS)
        if not any(_is_relative_to(approved_path, root) for root in allowed_roots):
            raise DevelopmentError("fixture source is outside allowed M3 QA roots")
        return SourceApproval(
            mode="synthetic_fixture",
            approved_path=approved_path,
            approved_sha256=approved_sha256,
        )
    _reject_symlink_components(path)
    production_path = (core.REPO_ROOT / PRODUCTION_SOURCE_RELATIVE).resolve()
    resolved = path.resolve()
    if resolved != production_path:
        raise DevelopmentError("production development source must be the approved M3 source path")
    protected = _read_json(PROTECTED_M1_M2_METADATA)
    protected_hash = protected.get("sha256", {}).get(str(PRODUCTION_SOURCE_RELATIVE))
    if not protected_hash:
        raise DevelopmentError("protected M1/M2 metadata lacks production source hash")
    if approved_sha256 != protected_hash:
        raise DevelopmentError("production source approved hash does not match protected metadata")
    return SourceApproval(
        mode="production_approved",
        approved_path=production_path,
        approved_sha256=str(protected_hash),
    )


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
    }


def _verify_source_identity_at_completion(read_audit: Mapping[str, Any]) -> None:
    source_path = Path(str(read_audit["source_path"]))
    expected = read_audit.get("source_identity_after_read")
    if not isinstance(expected, Mapping):
        raise DevelopmentError("read audit lacks source identity receipt")
    if _canonicalize(_file_identity(source_path)) != _canonicalize(expected):
        raise DevelopmentError("source identity changed before completion")


def _schema_audit(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
        ],
        "column_count": int(len(frame.columns)),
        "row_count": int(len(frame)),
    }


def _source_schema_audit(path: Path) -> dict[str, Any]:
    import pyarrow.dataset as ds

    schema = ds.dataset(path, format="parquet").schema
    return {
        "column_names": list(schema.names),
        "columns": [
            {"name": field.name, "type": str(field.type)}
            for field in schema
        ],
        "column_count": int(len(schema.names)),
    }


def _reject_symlink_components(path: Path) -> None:
    probe = path if path.is_absolute() else core.REPO_ROOT / path
    current = Path(probe.anchor) if probe.is_absolute() else Path()
    for part in probe.parts:
        if part in ("", current.anchor):
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise DevelopmentError("path contains a symlink component")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _strict_validate_completed_run(
    run_dir: Path,
    registry: pd.DataFrame,
    selection: Mapping[str, Any],
) -> None:
    if (run_dir / "failed.json").exists():
        raise DevelopmentError("completed run contains failure marker")
    _validate_registry(registry)
    run_identity = _read_json(run_dir / "run_identity.json")
    read_audit = _read_json(run_dir / "read_audit.json")
    causal_audit = _read_json(run_dir / "causal_feature_audit.json")
    schedule_doc = _read_json(run_dir / "schedule.json")
    cohort_plan = _read_json(run_dir / "cohort_plan.json")
    expected_records = [spec.record() for spec in expected_schedule()]
    if schedule_doc.get("fits") != expected_records:
        raise DevelopmentError("completed run schedule does not match fixed schedule")
    if int(cohort_plan.get("fit_count", -1)) != 144:
        raise DevelopmentError("completed run cohort plan does not contain 144 fits")
    plan_fits = cohort_plan.get("fits", {})
    if set(plan_fits) != {spec.fit_id for spec in expected_schedule()}:
        raise DevelopmentError("completed run cohort plan fit set mismatch")
    if read_audit.get("projected_content_digest") != run_identity.get(
        "read_audit_projected_content_digest"
    ):
        raise DevelopmentError("run identity/read audit content digest mismatch")
    if _sha256_obj(causal_audit) != run_identity.get("causal_feature_audit_sha256"):
        raise DevelopmentError("run identity causal audit digest mismatch")
    _validate_causal_audit_receipt(causal_audit, read_audit)
    _validate_shared_cohort_plan(plan_fits)
    for row in registry.itertuples(index=False):
        fit_id = str(row.fit_id)
        expected = next(spec for spec in expected_schedule() if spec.fit_id == fit_id)
        plan = plan_fits.get(fit_id)
        if not isinstance(plan, Mapping):
            raise DevelopmentError(f"{fit_id} missing frozen cohort plan")
        if (
            int(row.horizon),
            str(row.feature_set),
            str(row.model_family),
            int(row.fold),
        ) != (expected.horizon, expected.feature_set, expected.model_family, expected.fold):
            raise DevelopmentError(f"{fit_id} registry tuple does not match fit id")
        for attr in (
            "threshold_path",
            "model_path",
            "prediction_path",
            "metrics_path",
        ):
            rel = Path(str(getattr(row, attr)))
            target = (run_dir / rel).resolve()
            if rel.is_absolute() or not _is_relative_to(target, run_dir):
                raise DevelopmentError(f"{fit_id} registry path escapes run directory")
            if not target.exists():
                raise DevelopmentError(f"{fit_id} registry path missing: {attr}")
        threshold_path = run_dir / str(row.threshold_path)
        prediction_path = run_dir / str(row.prediction_path)
        metrics_path = run_dir / str(row.metrics_path)
        model_path = run_dir / str(row.model_path) / "model.joblib"
        metadata_path = run_dir / str(row.model_path) / "metadata.json"
        claim_path = run_dir / "fits" / fit_id / "claim.json"
        complete_path = run_dir / "fits" / fit_id / "complete.json"
        if not claim_path.exists() or not complete_path.exists():
            raise DevelopmentError(f"{fit_id} missing mandatory fit receipt")
        claim_doc = _read_json(claim_path)
        if claim_doc.get("fit") != expected.record() or claim_doc.get("status") != "claimed":
            raise DevelopmentError(f"{fit_id} claim receipt mismatch")
        complete_doc = _read_json(complete_path)
        if complete_doc.get("status") != "complete":
            raise DevelopmentError(f"{fit_id} complete receipt status mismatch")
        expected_hashes = {
            "threshold_sha256": sha256_file(threshold_path),
            "prediction_sha256": sha256_file(prediction_path),
            "metrics_sha256": sha256_file(metrics_path),
            "model_sha256": sha256_file(model_path),
            "model_metadata_sha256": sha256_file(metadata_path),
        }
        for attr, actual in expected_hashes.items():
            if str(getattr(row, attr)) != actual:
                raise DevelopmentError(f"{fit_id} registry {attr} mismatch")
            if str(complete_doc.get(attr)) != actual:
                raise DevelopmentError(f"{fit_id} complete receipt {attr} mismatch")
        _reconcile_fit_evidence(run_dir, row, expected, plan, complete_doc)
        predictions = pd.read_parquet(prediction_path)
        if predictions.empty:
            raise DevelopmentError(f"{fit_id} has empty prediction artifact")
        if (
            predictions["fit_id"].astype(str).nunique() != 1
            or str(predictions["fit_id"].iloc[0]) != fit_id
        ):
            raise DevelopmentError(f"{fit_id} prediction artifact is not bound to registry fit")
        if not predictions["observed_current_cases"].equals(predictions["prediction_persistence"]):
            raise DevelopmentError(f"{fit_id} persisted prediction baseline mismatch")
        metrics_doc = _read_json(metrics_path)
        threshold_doc = _read_json(threshold_path)
        recomputed_metrics = _metric_document(predictions, threshold_doc)
        if _canonicalize(metrics_doc) != _canonicalize(recomputed_metrics):
            raise DevelopmentError(f"{fit_id} metrics are not derived from predictions")
        if row.mae_model != metrics_doc["model"]["mae"]["value"]:
            raise DevelopmentError(f"{fit_id} registry MAE does not match metrics")
        if row.rmse_model != metrics_doc["model"]["rmse"]["value"]:
            raise DevelopmentError(f"{fit_id} registry RMSE does not match metrics")
        if row.mae_persistence != metrics_doc["persistence"]["mae"]["value"]:
            raise DevelopmentError(f"{fit_id} registry persistence MAE does not match metrics")
        if row.rmse_persistence != metrics_doc["persistence"]["rmse"]["value"]:
            raise DevelopmentError(f"{fit_id} registry persistence RMSE does not match metrics")
        if bool(row.strict_mae_win) != (
            metrics_doc["model"]["mae"]["value"] < metrics_doc["persistence"]["mae"]["value"]
        ):
            raise DevelopmentError(f"{fit_id} registry MAE win does not match metrics")
        if bool(row.strict_rmse_win) != (
            metrics_doc["model"]["rmse"]["value"] < metrics_doc["persistence"]["rmse"]["value"]
        ):
            raise DevelopmentError(f"{fit_id} registry RMSE win does not match metrics")
        if str(row.threshold_binding_sha256) != _sha256_obj(threshold_doc):
            raise DevelopmentError(f"{fit_id} threshold binding hash mismatch")
        metadata = _read_json(metadata_path)
        _verify_metadata_doc(
            expected,
            metadata,
            _model_config(_read_json(core.FIXED_PROTOCOL_CONFIG), expected.model_family),
            core.selected_feature_columns(expected.feature_set),
            plan["training_keys"],
        )
    selected = select_champions(registry)
    if _canonicalize(selected) != _canonicalize(selection):
        raise DevelopmentError("selection receipt does not match authenticated registry")


def _reconcile_fit_evidence(
    run_dir: Path,
    row: Any,
    spec: FitSpec,
    plan: Mapping[str, Any],
    complete_doc: Mapping[str, Any],
) -> None:
    for key, expected_value in spec.record().items():
        if complete_doc.get(key) != expected_value:
            raise DevelopmentError(f"{spec.fit_id} complete receipt fit identity mismatch")
        if plan.get(key) != expected_value:
            raise DevelopmentError(f"{spec.fit_id} cohort plan fit identity mismatch")
    for key in (
        "training_count",
        "evaluation_count",
        "training_key_digest",
        "evaluation_key_digest",
        "training_feature_content_digest",
        "training_target_content_digest",
        "evaluation_feature_content_digest",
        "evaluation_target_content_digest",
        "threshold_binding_sha256",
        "feature_list_sha256",
        "config_sha256",
        "mae_model",
        "rmse_model",
        "mae_persistence",
        "rmse_persistence",
        "strict_mae_win",
        "strict_rmse_win",
    ):
        if key in complete_doc and getattr(row, key) != complete_doc[key]:
            raise DevelopmentError(f"{spec.fit_id} registry/complete {key} mismatch")
    training_keys = plan["training_keys"]
    evaluation_keys = plan["evaluation_keys"]
    training_digests = plan["training_digests"]
    evaluation_digests = plan["evaluation_digests"]
    for label, evidence in (
        ("training keys", training_keys),
        ("evaluation keys", evaluation_keys),
        ("training digests", training_digests),
        ("evaluation digests", evaluation_digests),
    ):
        if not isinstance(evidence, Mapping):
            raise DevelopmentError(f"{spec.fit_id} frozen {label} missing")
    if int(row.training_count) != training_keys["row_count"]:
        raise DevelopmentError(f"{spec.fit_id} registry training count mismatch")
    if int(row.evaluation_count) != evaluation_keys["row_count"]:
        raise DevelopmentError(f"{spec.fit_id} registry evaluation count mismatch")
    if row.training_key_digest != training_keys["row_key_digest"]:
        raise DevelopmentError(f"{spec.fit_id} registry training key digest mismatch")
    if row.evaluation_key_digest != evaluation_keys["row_key_digest"]:
        raise DevelopmentError(f"{spec.fit_id} registry evaluation key digest mismatch")
    if row.training_key_digest != training_digests["row_key_digest"]:
        raise DevelopmentError(f"{spec.fit_id} frozen training digest key mismatch")
    if row.evaluation_key_digest != evaluation_digests["row_key_digest"]:
        raise DevelopmentError(f"{spec.fit_id} frozen evaluation digest key mismatch")
    if row.training_feature_content_digest != training_digests["feature_content_digest"]:
        raise DevelopmentError(f"{spec.fit_id} frozen training feature digest mismatch")
    if row.training_target_content_digest != training_digests["target_content_digest"]:
        raise DevelopmentError(f"{spec.fit_id} frozen training target digest mismatch")
    if row.evaluation_feature_content_digest != evaluation_digests["feature_content_digest"]:
        raise DevelopmentError(f"{spec.fit_id} frozen evaluation feature digest mismatch")
    if row.evaluation_target_content_digest != evaluation_digests["target_content_digest"]:
        raise DevelopmentError(f"{spec.fit_id} frozen evaluation target digest mismatch")
    if row.threshold_binding_sha256 != plan["threshold_binding_sha256"]:
        raise DevelopmentError(f"{spec.fit_id} frozen threshold digest mismatch")
    if row.feature_list_sha256 != plan["feature_list_sha256"]:
        raise DevelopmentError(f"{spec.fit_id} frozen feature digest mismatch")
    expected_config_sha = _sha256_obj(
        _model_config(_read_json(core.FIXED_PROTOCOL_CONFIG), spec.model_family).serializable()
    )
    if row.config_sha256 != expected_config_sha:
        raise DevelopmentError(f"{spec.fit_id} effective config digest mismatch")
    training_keys_path = (
        run_dir
        / "cohorts"
        / f"fold{spec.fold}"
        / f"h{spec.horizon}"
        / "training_keys.parquet"
    )
    evaluation_keys_path = run_dir / "cohorts" / f"fold{spec.fold}" / "evaluation_keys.parquet"
    if not training_keys_path.exists() or not evaluation_keys_path.exists():
        raise DevelopmentError(f"{spec.fit_id} missing mandatory cohort key artifact")
    core.verify_common_keys(pd.read_parquet(training_keys_path), training_keys)
    core.verify_common_keys(pd.read_parquet(evaluation_keys_path), evaluation_keys)
    predictions = pd.read_parquet(run_dir / str(row.prediction_path))
    if len(predictions) != evaluation_keys["row_count"]:
        raise DevelopmentError(f"{spec.fit_id} prediction row count mismatch")
    if set(predictions["training_binding_id"].astype(str)) != {training_keys["row_key_digest"]}:
        raise DevelopmentError(f"{spec.fit_id} prediction training binding mismatch")
    if set(predictions["evaluation_binding_id"].astype(str)) != {evaluation_keys["row_key_digest"]}:
        raise DevelopmentError(f"{spec.fit_id} prediction evaluation binding mismatch")
    if set(predictions["threshold_binding_id"].astype(str)) != {plan["threshold_binding_sha256"]}:
        raise DevelopmentError(f"{spec.fit_id} prediction threshold binding mismatch")
    prediction_keys = [
        {"district_id": str(row.district_id), "week_start_date": str(row.origin_start)}
        for row in predictions[["district_id", "origin_start"]].itertuples(index=False)
    ]
    if prediction_keys != evaluation_keys["keys"]:
        raise DevelopmentError(f"{spec.fit_id} prediction keys do not match frozen evaluation")
    registry_interval = row.interval_bounds
    plan_interval = plan.get("interval_bounds")
    complete_interval = complete_doc.get("interval_bounds")
    if (
        not isinstance(registry_interval, Mapping)
        or not isinstance(plan_interval, Mapping)
        or not isinstance(complete_interval, Mapping)
        or not registry_interval
        or not plan_interval
        or not complete_interval
    ):
        raise DevelopmentError(f"{spec.fit_id} interval evidence missing")
    for label, interval in (
        ("registry", registry_interval),
        ("frozen cohort plan", plan_interval),
        ("complete receipt", complete_interval),
    ):
        _require_interval_bound_members(spec.fit_id, label, interval)
    if (
        _canonicalize(registry_interval) != _canonicalize(plan_interval)
        or _canonicalize(complete_interval) != _canonicalize(plan_interval)
    ):
        raise DevelopmentError(f"{spec.fit_id} interval evidence mismatch")


def _require_interval_bound_members(
    fit_id: str,
    label: str,
    interval: Mapping[str, Any],
) -> None:
    missing = [member for member in REQUIRED_INTERVAL_BOUND_MEMBERS if member not in interval]
    if missing:
        raise DevelopmentError(
            f"{fit_id} {label} interval evidence missing members: {missing}"
        )
    null_members = [
        member
        for member in REQUIRED_INTERVAL_BOUND_MEMBERS
        if _is_null_interval_value(interval[member])
    ]
    if null_members:
        raise DevelopmentError(
            f"{fit_id} {label} interval evidence null members: {null_members}"
        )


def _is_null_interval_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bool, int)):
        return False
    try:
        result = pd.isna(value)
    except TypeError:
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _validate_shared_cohort_plan(plan_fits: Mapping[str, Any]) -> None:
    eval_digest_by_fold: dict[int, str] = {}
    train_digest_by_hf: dict[tuple[int, int], str] = {}
    threshold_by_hf: dict[tuple[int, int], str] = {}
    train_target_by_hf: dict[tuple[int, int], str] = {}
    eval_target_by_hf: dict[tuple[int, int], str] = {}
    train_feature_by_hff: dict[tuple[int, int, str], str] = {}
    eval_feature_by_hff: dict[tuple[int, int, str], str] = {}
    for fit_id, plan in plan_fits.items():
        fold = int(plan["fold"])
        horizon = int(plan["horizon"])
        feature_set = str(plan["feature_set"])
        eval_digest = plan["evaluation_keys"]["row_key_digest"]
        train_digest = plan["training_keys"]["row_key_digest"]
        threshold_digest = plan["threshold_binding_sha256"]
        existing_eval = eval_digest_by_fold.setdefault(fold, eval_digest)
        if existing_eval != eval_digest:
            raise DevelopmentError(f"{fit_id} evaluation cohort is not shared")
        key = (horizon, fold)
        existing_train = train_digest_by_hf.setdefault(key, train_digest)
        existing_threshold = threshold_by_hf.setdefault(key, threshold_digest)
        if existing_train != train_digest or existing_threshold != threshold_digest:
            raise DevelopmentError(f"{fit_id} training cohort is not shared")
        training_digests = plan["training_digests"]
        evaluation_digests = plan["evaluation_digests"]
        train_target = training_digests["target_content_digest"]
        eval_target = evaluation_digests["target_content_digest"]
        existing_train_target = train_target_by_hf.setdefault(key, train_target)
        existing_eval_target = eval_target_by_hf.setdefault(key, eval_target)
        if existing_train_target != train_target or existing_eval_target != eval_target:
            raise DevelopmentError(f"{fit_id} target content is not shared")
        feature_key = (horizon, fold, feature_set)
        train_feature = training_digests["feature_content_digest"]
        eval_feature = evaluation_digests["feature_content_digest"]
        existing_train_feature = train_feature_by_hff.setdefault(feature_key, train_feature)
        existing_eval_feature = eval_feature_by_hff.setdefault(feature_key, eval_feature)
        if existing_train_feature != train_feature or existing_eval_feature != eval_feature:
            raise DevelopmentError(f"{fit_id} feature content is not shared")


def _validate_causal_audit_receipt(
    causal_audit: Mapping[str, Any],
    read_audit: Mapping[str, Any],
) -> None:
    required = {
        "audit_name",
        "ordered_feature_columns",
        "ordered_feature_list_sha256",
        "input_projected_content_digest",
        "recomputed_projected_content_digest",
        "audit_result",
        "observation_content_unchanged",
    }
    missing = required - set(causal_audit)
    if missing:
        raise DevelopmentError(f"causal feature audit missing fields: {sorted(missing)}")
    if causal_audit.get("audit_name") != "m3_causal_origin_feature_audit":
        raise DevelopmentError("causal feature audit name mismatch")
    if causal_audit.get("audit_result") is not True:
        raise DevelopmentError("causal feature audit did not pass")
    if causal_audit.get("observation_content_unchanged") is not True:
        raise DevelopmentError("causal feature audit observation preservation failed")
    ordered_features = causal_audit.get("ordered_feature_columns")
    if not isinstance(ordered_features, list) or not ordered_features:
        raise DevelopmentError("causal feature audit ordered features missing")
    expected_features = [
        column
        for column in projected_columns(_read_json(core.FIXED_PROTOCOL_CONFIG)["features"]["sets"])
        if column not in {"week_end_date"}
        and column in set(core.PINNED_FEATURE_SETS["cases_full_weather"])
    ]
    if ordered_features != expected_features:
        raise DevelopmentError("causal feature audit ordered features mismatch")
    if causal_audit.get("ordered_feature_list_sha256") != core.feature_list_digest(
        ordered_features
    ):
        raise DevelopmentError("causal feature audit ordered feature digest mismatch")
    if causal_audit.get("input_projected_content_digest") != read_audit.get(
        "projected_content_digest"
    ):
        raise DevelopmentError("causal feature audit input digest mismatch")
    if not isinstance(causal_audit.get("recomputed_projected_content_digest"), str):
        raise DevelopmentError("causal feature audit recomputed digest missing")


def _frame_receipt(frame: pd.DataFrame) -> dict[str, Any]:
    keys = core.freeze_common_keys(frame)
    payload = _canonical_frame_records(frame)
    return {
        "returned_count": int(len(frame)),
        "returned_min_week_start": (
            pd.to_datetime(frame["week_start_date"]).min().date().isoformat()
        ),
        "returned_max_week_end": pd.to_datetime(frame["week_end_date"]).max().date().isoformat(),
        "returned_key_digest": keys["row_key_digest"],
        "projected_content_digest": _sha256_obj(payload),
    }


def _canonical_frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    content = frame.copy()
    for column in content.columns:
        if pd.api.types.is_datetime64_any_dtype(content[column]):
            content[column] = pd.to_datetime(content[column]).map(
                lambda value: None if pd.isna(value) else pd.Timestamp(value).isoformat()
            )
    records: list[dict[str, Any]] = []
    for record in content.to_dict(orient="records"):
        records.append({str(key): _canonical_scalar(value) for key, value in record.items()})
    return records


def _canonical_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float):
        return float(value)
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return _canonical_scalar(value)


def _sha256_obj(value: Any) -> str:
    text = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_identity(
    run_id: str,
    protocol: Mapping[str, Any],
    read_audit: Mapping[str, Any],
    causal_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "study_designation": protocol["study_designation"],
        "authorization_sha256": protocol["authorization_sha256"],
        "protocol_config_path": protocol["machine_config_path"],
        "protocol_config_sha256": protocol["machine_config_sha256"],
        "source_sha256": read_audit["source_sha256"],
        "read_audit_projected_content_digest": read_audit["projected_content_digest"],
        "causal_feature_audit_sha256": _sha256_obj(causal_audit),
        "causal_feature_audit_result": causal_audit["audit_result"],
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "code_sha256": {
            "development.py": sha256_file(Path(__file__)),
            "core.py": sha256_file(Path(core.__file__)),
            "modeling/train.py": sha256_file(
                Path(sys.modules[train_fold_model.__module__].__file__)
            ),
            "modeling/metrics.py": sha256_file(Path(metrics.__file__)),
            "modeling/preprocessing.py": sha256_file(Path(preprocessing.__file__)),
        },
        "scheduled_2025_boundary": {
            "status": "pending_later_gate",
            "development_policy": (
                "calendar_2025_cutoff_and_each_fold_first_scheduled_validation_origin"
            ),
        },
    }


def _dependency_versions() -> dict[str, str | None]:
    names = ["pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "joblib"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _require_finite_nonnegative(series: pd.Series, label: str, *, allow_na: bool = False) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    bad = numeric.isna() if not allow_na else numeric.isna() & series.notna()
    if bad.any() or np.isinf(numeric.dropna()).any() or numeric.dropna().lt(0).any():
        raise DevelopmentError(f"{label} must be finite nonnegative")


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_new(path, text)


def _write_text_new(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(path.parent)


def _write_parquet_new(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise DevelopmentError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        frame.to_parquet(tmp, index=False)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_failure_marker(run_dir: Path) -> None:
    path = run_dir / "failed.json"
    if not path.exists():
        _write_json_new(path, {"status": "failed", "complete": False})


def _checksums(run_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        rel_path = path.relative_to(run_dir)
        if path.is_file() and str(rel_path) not in {"checksums.json", "complete.json"}:
            checksums[str(path.relative_to(run_dir))] = sha256_file(path)
    return checksums


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
