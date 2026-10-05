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

from dengue_forecast.milestone3 import core, development
from dengue_forecast.modeling import metrics, preprocessing
from dengue_forecast.modeling import train as modeling_train
from dengue_forecast.modeling.train import ModelConfig, TrainedModel, train_fold_model

MODEL_FAMILIES = development.MODEL_FAMILIES
FEATURE_SET_ORDER = development.FEATURE_SET_ORDER
HORIZONS = core.HORIZONS
KEY_COLUMNS = development.KEY_COLUMNS
BASE_COLUMNS = development.BASE_COLUMNS
PRODUCTION_SOURCE_RELATIVE = development.PRODUCTION_SOURCE_RELATIVE
PRODUCTION_OUTPUT_ROOT = core.REPO_ROOT / "artifacts" / "milestone3" / "final_evaluation"
FIXTURE_OUTPUT_ROOTS = development.FIXTURE_OUTPUT_ROOTS
TRUSTED_REGISTRY_ROOT = core.REPO_ROOT / ".hermes" / "m3-final-trusted-registry"
_FIXTURE_TRUSTED_REGISTRY_ROOT: Path | None = None
SYNTHETIC_GATE_PURPOSE = "m3_final_synthetic_outcome_fixture_gate"
PRODUCTION_OUTCOME_GATE_PURPOSE = "m3_final_authorized_outcome_gate"
NO_FIT_SYNTHETIC_MODEL_BYTES = "INERT SYNTHETIC BYTES; NOT A SERIALIZED MODEL\n"
REQUIRED_FREEZE_GATES = (
    "development_run_verified",
    "independent_review_passed",
    "portable_installation_passed",
    "protected_hashes_verified",
    "final_access_authorized_by_parent",
)


class FinalEvaluationError(ValueError):
    """Raised when the once-only M3 final evaluation engine fails closed."""


@dataclass(frozen=True)
class FinalFitSpec:
    horizon: int
    feature_set: str
    model_family: str

    @property
    def target_column(self) -> str:
        return f"target_h{self.horizon}"

    @property
    def fit_id(self) -> str:
        return f"h{self.horizon}__{self.feature_set}__{self.model_family}__final"

    def record(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "feature_set": self.feature_set,
            "model_family": self.model_family,
            "fit_id": self.fit_id,
        }


@dataclass(frozen=True)
class FinalFreezeResult:
    freeze_dir: Path
    freeze_id: str
    freeze_manifest_path: Path
    freeze_sha256: str
    fit_count: int


@dataclass(frozen=True)
class FinalEvaluationResult:
    run_dir: Path
    run_id: str
    fit_count: int
    registry_path: Path
    manifest_path: Path


Reader = Callable[[Path, list[str], Any, dict[str, Any]], pd.DataFrame]
Fitter = Callable[..., TrainedModel]


def final_schedule() -> list[FinalFitSpec]:
    specs = [
        FinalFitSpec(horizon, feature_set, model_family)
        for horizon in HORIZONS
        for feature_set in FEATURE_SET_ORDER
        for model_family in MODEL_FAMILIES
    ]
    if len(specs) != 24 or len({spec.fit_id for spec in specs}) != 24:
        raise FinalEvaluationError("final schedule must contain exactly 24 unique fits")
    return specs


def fixture_development_authority(
    development_receipt: Mapping[str, Any],
    development_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the synthetic accepted-development authority used by final fixtures."""
    champions = {
        spec.fit_id: _is_selected_champion(spec, development_selection)
        for spec in final_schedule()
    }
    payload = {
        "schema": "m3_final_accepted_development_authority_v1",
        "development_receipt_sha256": _sha256_obj(development_receipt),
        "development_selection_sha256": _sha256_obj(development_selection),
        "selection_sha256": development_receipt.get("selection_sha256"),
        "fit_count": 24,
        "champions": champions,
    }
    payload["authority_sha256"] = _sha256_obj(payload)
    return payload


def provision_synthetic_trusted_registry(
    *,
    registry_root: str | Path,
    experiment_id: str,
    authorized_source_sha256: str,
    development_receipt: Mapping[str, Any],
    development_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Provision an external synthetic trust record before freeze preparation."""
    _set_fixture_trusted_registry_root(registry_root)
    _validate_experiment_id(experiment_id)
    record = {
        "schema": "m3_final_trusted_setup_v1",
        "experiment_id": experiment_id,
        "authorized_source_sha256": authorized_source_sha256,
        "accepted_development_receipt": dict(development_receipt),
        "accepted_development_selection": _canonicalize(development_selection),
        "accepted_development_authority": fixture_development_authority(
            development_receipt,
            development_selection,
        ),
        "status": "trusted_setup_provisioned",
    }
    path = _trusted_setup_path(experiment_id, fixture_mode=True)
    _write_json_new(path, record)
    return {
        "experiment_id": experiment_id,
        "setup_path": str(path),
        "setup_sha256": sha256_file(path),
    }


def prepare_freeze(
    *,
    source_path: str | Path,
    output_root: str | Path,
    freeze_id: str,
    development_run_dir: str | Path,
    first_scheduled_origin: str | pd.Timestamp,
    gate_receipts: Mapping[str, Any],
    approved_source_sha256: str | None = None,
    fixture_mode: bool = False,
    metadata_authority: Mapping[str, Any] | None = None,
    reader: Reader | None = None,
) -> FinalFreezeResult:
    _verify_prepare_gates(gate_receipts, fixture_mode=fixture_mode)
    protocol = development.load_protocol()
    dev_receipt = development.validate_development_run(development_run_dir)
    source = _approve_source(
        source_path,
        approved_source_sha256=approved_source_sha256,
        fixture_mode=fixture_mode,
    )
    first_origin = _coerce_midnight(first_scheduled_origin, "first_scheduled_origin")
    freeze_dir = _claim_dir(output_root, freeze_id, fixture_mode=fixture_mode, kind="freeze")
    try:
        schedule_authority = _verify_metadata_authority(
            metadata_authority,
            supplied_first_origin=first_origin,
            fixture_mode=fixture_mode,
        )
        observations, read_audit = _load_prepare_training_observations(
            source,
            protocol=protocol,
            first_origin=first_origin,
            reader=reader,
        )
        features = core.add_origin_features(observations)
        tasks = core.build_direct_tasks(features, horizons=HORIZONS)
        training_by_horizon = {
            horizon: _final_training_frame(tasks, horizon=horizon, first_origin=first_origin)
            for horizon in HORIZONS
        }
        metadata_rows, metadata_read_audit = _load_prepare_metadata_rows(
            source,
            first_origin=first_origin,
            reader=reader,
        )
        test_metadata = _test_metadata_rows(metadata_rows, schedule_authority=schedule_authority)
        test_keys = core.freeze_common_keys(test_metadata)
        _validate_test_metadata_common_cohort(test_metadata)
        schedule = final_schedule()
        selection = development._read_json(Path(development_run_dir) / "selection.json")
        accepted_development_authority = gate_receipts.get("accepted_development_authority")
        expected_development_authority = fixture_development_authority(dev_receipt, selection)
        if accepted_development_authority is None:
            accepted_development_authority = expected_development_authority
        elif _canonicalize(accepted_development_authority) != _canonicalize(
            expected_development_authority
        ):
            raise FinalEvaluationError("prepare development authority receipt mismatch")
        freeze = _freeze_manifest(
            freeze_id=freeze_id,
            fixture_mode=fixture_mode,
            protocol=protocol,
            gate_receipts=gate_receipts,
            development_receipt=dev_receipt,
            development_selection=selection,
            accepted_development_authority=accepted_development_authority,
            read_audit=read_audit,
            first_origin=first_origin,
            test_metadata=test_metadata,
            test_keys=test_keys,
            metadata_authority=schedule_authority,
            metadata_read_audit=metadata_read_audit,
            training_by_horizon=training_by_horizon,
            schedule=schedule,
        )
        freeze_path = freeze_dir / "freeze_manifest.json"
        _write_json_new(freeze_path, freeze)
        freeze_sha = sha256_file(freeze_path)
        complete = {
            "freeze_id": freeze_id,
            "freeze_sha256": freeze_sha,
            "fit_count": len(schedule),
            "status": "freeze_prepared_no_outcomes_read",
            "freeze_manifest": "freeze_manifest.json",
        }
        _write_json_new(freeze_dir / "complete.json", complete)
    except Exception:
        _write_failure_marker(freeze_dir)
        raise
    return FinalFreezeResult(
        freeze_dir=freeze_dir,
        freeze_id=freeze_id,
        freeze_manifest_path=freeze_path,
        freeze_sha256=freeze_sha,
        fit_count=len(schedule),
    )


def evaluate_freeze(
    *,
    freeze_manifest: str | Path,
    source_path: str | Path,
    output_root: str | Path,
    run_id: str,
    outcome_gate: Mapping[str, Any],
    approved_source_sha256: str | None = None,
    fixture_mode: bool = False,
    reader: Reader | None = None,
    fitter: Fitter = train_fold_model,
) -> FinalEvaluationResult:
    freeze_path = Path(freeze_manifest).resolve()
    freeze = _read_json(freeze_path)
    freeze_sha = sha256_file(freeze_path)
    _verify_outcome_gate(outcome_gate, freeze_sha=freeze_sha, fixture_mode=fixture_mode)
    setup_record = _trusted_setup_record(outcome_gate, fixture_mode=fixture_mode)
    _verify_freeze_static(
        freeze,
        freeze_sha=freeze_sha,
        fixture_mode=fixture_mode,
        trusted_setup=setup_record,
    )
    source = _approve_source(
        source_path,
        approved_source_sha256=approved_source_sha256,
        fixture_mode=fixture_mode,
    )
    _verify_freeze_source_identity(freeze, source)
    _verify_trusted_setup_source(setup_record, source)
    consumed_claim = _claim_experiment_authority(
        outcome_gate,
        freeze=freeze,
        freeze_sha=freeze_sha,
        source=source,
        trusted_setup=setup_record,
        fixture_mode=fixture_mode,
    )
    historical, historical_read_audit = _load_historical_training_observations(
        source,
        freeze=freeze,
        reader=reader,
    )
    historical_tasks = core.build_direct_tasks(
        core.add_origin_features(historical),
        horizons=HORIZONS,
    )
    training_by_horizon = {
        horizon: _final_training_frame(
            historical_tasks,
            horizon=horizon,
            first_origin=pd.Timestamp(freeze["first_scheduled_origin"]),
        )
        for horizon in HORIZONS
    }
    _verify_training_against_freeze(training_by_horizon, freeze)
    run_dir = _claim_dir(output_root, run_id, fixture_mode=fixture_mode, kind="evaluation")
    try:
        _write_text_new(
            run_dir / "freeze_manifest.json",
            freeze_path.read_text(encoding="utf-8"),
        )
        _write_json_new(
            run_dir / "claim.json",
            {
                "run_id": run_id,
                "freeze_sha256": freeze_sha,
                "consumed_authorization_claim": consumed_claim,
                "status": "claimed_once_only",
            },
        )
        observations, read_audit = _load_outcome_observations(
            source,
            freeze=freeze,
            reader=reader,
        )
        tasks = core.build_direct_tasks(core.add_origin_features(observations), horizons=HORIZONS)
        evaluation = _evaluation_outcome_rows(tasks, freeze=freeze)
        _verify_evaluation_against_freeze(evaluation, freeze)
        trusted_outcome_content = _trusted_outcome_content(
            evaluation,
            freeze=freeze,
            read_audit=read_audit,
        )
        outcome_receipt = _persist_trusted_outcome_content(
            outcome_gate,
            freeze=freeze,
            freeze_sha=freeze_sha,
            source=source,
            trusted_outcome_content=trusted_outcome_content,
            fixture_mode=fixture_mode,
        )
        registry_pre_metrics = _fit_and_predict_all(
            run_dir,
            freeze,
            training_by_horizon,
            evaluation,
            trusted_outcome_content=trusted_outcome_content,
            fitter=fitter,
        )
        registry = _score_saved_predictions(run_dir, freeze, registry_pre_metrics)
        registry_path = run_dir / "final_model_registry.parquet"
        _write_parquet_new(pd.DataFrame(registry), registry_path)
        manifest = _complete_manifest(
            run_id=run_id,
            freeze=freeze,
            freeze_sha=freeze_sha,
            read_audit={**read_audit, "historical_training_read_audit": historical_read_audit},
            trusted_outcome_content=trusted_outcome_content,
            trusted_outcome_receipt=outcome_receipt,
            registry=registry,
        )
        manifest_path = run_dir / "complete_manifest.json"
        _write_json_new(manifest_path, manifest)
        _write_json_new(run_dir / "checksums.json", _checksums(run_dir))
    except Exception:
        _write_failure_marker(run_dir)
        raise
    return FinalEvaluationResult(
        run_dir=run_dir,
        run_id=run_id,
        fit_count=len(registry),
        registry_path=registry_path,
        manifest_path=manifest_path,
    )


def validate_final_evaluation_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir).resolve()
    development._reject_symlink_components(path)
    if (path / "failed.json").exists():
        raise FinalEvaluationError("final evaluation run contains failure marker")
    manifest = _read_json(path / "complete_manifest.json")
    if manifest.get("schema") != "m3_final_evaluation_complete_v1":
        raise FinalEvaluationError("completion manifest schema mismatch")
    if manifest.get("fit_count") != 24:
        raise FinalEvaluationError("completion manifest fit count mismatch")
    if manifest.get("prediction_vectors_saved_before_metrics") is not True:
        raise FinalEvaluationError("completion manifest lacks prediction-before-metrics assertion")
    freeze = _read_json(_checked_artifact_path(path, "freeze_manifest.json"))
    freeze_sha = sha256_file(path / "freeze_manifest.json")
    if manifest.get("freeze_sha256") != freeze_sha:
        raise FinalEvaluationError("completion manifest freeze hash mismatch")
    claim = _read_json(path / "claim.json")
    consumed = claim.get("consumed_authorization_claim")
    if not isinstance(consumed, Mapping):
        raise FinalEvaluationError("run claim lacks consumed experiment authority")
    setup_record = _trusted_setup_record(consumed, fixture_mode=bool(freeze["fixture_mode"]))
    _verify_freeze_static(
        freeze,
        freeze_sha=freeze_sha,
        fixture_mode=bool(freeze["fixture_mode"]),
        trusted_setup=setup_record,
    )
    checksums = _read_json(path / "checksums.json")
    _verify_artifact_paths(path, checksums)
    required_artifacts = _required_completed_artifacts()
    if set(checksums) != required_artifacts:
        raise FinalEvaluationError("completed run artifact set is not exact")
    expected = _checksums(path)
    if set(checksums) != set(expected):
        raise FinalEvaluationError("checksum manifest does not contain exact artifact set")
    for rel_path, digest in checksums.items():
        if rel_path == "checksums.json":
            continue
        target = _checked_artifact_path(path, rel_path)
        if not target.exists() or expected.get(rel_path) != digest or sha256_file(target) != digest:
            raise FinalEvaluationError(f"checksum mismatch: {rel_path}")
    _verify_completed_claim(claim, manifest=manifest, freeze=freeze, freeze_sha=freeze_sha)
    registry = pd.read_parquet(path / "final_model_registry.parquet")
    _validate_final_registry(registry)
    registry_records = _registry_records(registry)
    if manifest.get("registry_sha256") != _sha256_obj(registry_records):
        raise FinalEvaluationError("completion manifest registry hash mismatch")
    if manifest.get("fit_count") != len(registry_records):
        raise FinalEvaluationError("completion manifest registry count mismatch")
    _verify_prediction_barrier(path, freeze, registry_records)
    trusted_outcome_content = manifest.get("trusted_outcome_content")
    if not isinstance(trusted_outcome_content, Mapping):
        raise FinalEvaluationError("completion manifest lacks trusted outcome content")
    _verify_completed_external_outcome_receipt(
        consumed,
        manifest=manifest,
        freeze=freeze,
        freeze_sha=freeze_sha,
        trusted_outcome_content=trusted_outcome_content,
    )
    for row in registry.itertuples(index=False):
        fit_id = str(row.fit_id)
        fit_freeze = freeze["fits"][fit_id]
        pred_path = _checked_artifact_path(path, str(row.prediction_path))
        metrics_path = _checked_artifact_path(path, str(row.metrics_path))
        threshold_path = _checked_artifact_path(path, str(row.threshold_path))
        model_path = _checked_artifact_path(path, f"{row.model_path}/model.joblib")
        model_metadata_path = _checked_artifact_path(path, f"{row.model_path}/metadata.json")
        pred = pd.read_parquet(pred_path)
        metrics_doc = _read_json(metrics_path)
        threshold_doc = _read_json(threshold_path)
        fit_claim = _read_json(path / "fits" / fit_id / "claim.json")
        pending = _read_json(path / "fits" / fit_id / "prediction_receipt_pending_metrics.json")
        complete = _read_json(path / "fits" / fit_id / "complete.json")
        model_metadata = _read_json(model_metadata_path)
        _verify_completed_fit_claim(row, fit_claim)
        _verify_completed_fit_receipts(
            row,
            pending=pending,
            complete=complete,
            fit_freeze=fit_freeze,
            freeze=freeze,
        )
        _verify_completed_model_metadata(
            row,
            metadata=model_metadata,
            fit_freeze=fit_freeze,
        )
        _verify_completed_predictions(
            row,
            pred,
            fit_freeze=fit_freeze,
            freeze=freeze,
            trusted_outcome_content=trusted_outcome_content,
        )
        recomputed = development._metric_document(pred, threshold_doc)
        if _canonicalize(metrics_doc) != _canonicalize(recomputed):
            raise FinalEvaluationError(f"{fit_id} metrics are not saved-prediction derived")
        _verify_registry_metrics(row, recomputed)
        if str(row.prediction_sha256) != sha256_file(pred_path):
            raise FinalEvaluationError(f"{fit_id} prediction hash mismatch")
        if str(row.metrics_sha256) != sha256_file(metrics_path):
            raise FinalEvaluationError(f"{fit_id} metrics hash mismatch")
        if str(row.model_sha256) != sha256_file(model_path):
            raise FinalEvaluationError(f"{fit_id} model hash mismatch")
        if str(row.model_metadata_sha256) != sha256_file(model_metadata_path):
            raise FinalEvaluationError(f"{fit_id} model metadata hash mismatch")
    return {
        "run_dir": str(path),
        "fit_count": int(len(registry)),
        "freeze_sha256": manifest["freeze_sha256"],
        "status": "final_evaluation_completed_read_only_verified",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the once-only Milestone 3 final evaluation engine."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-freeze")
    prep.add_argument("--source-parquet", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, required=True)
    prep.add_argument("--freeze-id", required=True)
    prep.add_argument("--development-run-dir", type=Path, required=True)
    prep.add_argument("--first-scheduled-origin", required=True)
    prep.add_argument("--gate-receipts-json", type=Path, required=True)
    prep.add_argument("--metadata-authority-json", type=Path)
    prep.add_argument("--approved-source-sha256")
    prep.add_argument("--fixture-mode", action="store_true")

    evalp = sub.add_parser("evaluate")
    evalp.add_argument("--freeze-manifest", type=Path, required=True)
    evalp.add_argument("--source-parquet", type=Path, required=True)
    evalp.add_argument("--output-root", type=Path, required=True)
    evalp.add_argument("--run-id", required=True)
    evalp.add_argument("--outcome-gate-json", type=Path, required=True)
    evalp.add_argument("--approved-source-sha256")
    evalp.add_argument("--fixture-mode", action="store_true")

    valid = sub.add_parser("validate-existing")
    valid.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare-freeze":
        result = prepare_freeze(
            source_path=args.source_parquet,
            output_root=args.output_root,
            freeze_id=args.freeze_id,
            development_run_dir=args.development_run_dir,
            first_scheduled_origin=args.first_scheduled_origin,
            gate_receipts=_read_json(args.gate_receipts_json),
            approved_source_sha256=args.approved_source_sha256,
            fixture_mode=args.fixture_mode,
            metadata_authority=(
                _read_json(args.metadata_authority_json)
                if args.metadata_authority_json is not None
                else None
            ),
        )
        print(json.dumps(_dataclass_result(result), indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        result = evaluate_freeze(
            freeze_manifest=args.freeze_manifest,
            source_path=args.source_parquet,
            output_root=args.output_root,
            run_id=args.run_id,
            outcome_gate=_read_json(args.outcome_gate_json),
            approved_source_sha256=args.approved_source_sha256,
            fixture_mode=args.fixture_mode,
        )
        print(json.dumps(_dataclass_result(result), indent=2, sort_keys=True))
        return 0
    receipt = validate_final_evaluation_run(args.run_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _dataclass_result(value: Any) -> dict[str, Any]:
    return {
        key: str(item) if isinstance(item, Path) else item
        for key, item in value.__dict__.items()
    }


def _verify_prepare_gates(gates: Mapping[str, Any], *, fixture_mode: bool) -> None:
    if not isinstance(gates, Mapping):
        raise FinalEvaluationError("prepare freeze requires gate receipt mapping")
    if fixture_mode:
        if gates.get("mode") != "synthetic_fixture":
            raise FinalEvaluationError("synthetic prepare requires synthetic fixture gate mode")
    else:
        if gates.get("schema") != "m3_final_prepare_gate_receipts_v1":
            raise FinalEvaluationError("production prepare requires parent gate receipt schema")
        if gates.get("authorize_freeze_preparation") is not True:
            raise FinalEvaluationError("production prepare requires explicit freeze authorization")
        missing = [name for name in REQUIRED_FREEZE_GATES if gates.get(name) is not True]
        if missing:
            raise FinalEvaluationError(f"missing verified final prepare gates: {missing}")
        authority = gates.get("accepted_development_authority")
        if not isinstance(authority, Mapping):
            raise FinalEvaluationError(
                "production prepare requires accepted development authority receipt"
            )


def _verify_outcome_gate(
    gate: Mapping[str, Any],
    *,
    freeze_sha: str,
    fixture_mode: bool,
) -> None:
    if fixture_mode:
        if gate.get("mode") != "synthetic_fixture":
            raise FinalEvaluationError("synthetic outcomes require explicit synthetic fixture gate")
        if gate.get("purpose") != SYNTHETIC_GATE_PURPOSE:
            raise FinalEvaluationError("synthetic outcome gate purpose mismatch")
    else:
        if gate.get("schema") != "m3_final_outcome_gate_v1":
            raise FinalEvaluationError("production outcomes require parent outcome gate schema")
        if gate.get("mode") != "production_authorized":
            raise FinalEvaluationError("production outcomes require production_authorized mode")
        if gate.get("purpose") != PRODUCTION_OUTCOME_GATE_PURPOSE:
            raise FinalEvaluationError("production outcome gate purpose mismatch")
        if gate.get("allow_production_outcomes") is not True:
            raise FinalEvaluationError("production outcome gate does not authorize outcome read")
    if gate.get("freeze_sha256") != freeze_sha:
        raise FinalEvaluationError("outcome gate is not bound to freeze hash")
    if fixture_mode and gate.get("allow_synthetic_outcomes") is not True:
        raise FinalEvaluationError("synthetic outcome gate does not authorize outcome read")
    if not fixture_mode and gate.get("authorized_freeze_sha256") != freeze_sha:
        raise FinalEvaluationError("production outcome gate lacks explicit authorized freeze")
    if "experiment_authority_context" in gate or "experiment_authority_dir" in gate:
        raise FinalEvaluationError("outcome gate cannot choose experiment authority location")


def _verify_freeze_static(
    freeze: Mapping[str, Any],
    *,
    freeze_sha: str,
    fixture_mode: bool,
    trusted_setup: Mapping[str, Any] | None = None,
) -> None:
    protocol = development.load_protocol()
    if freeze.get("freeze_sha256") not in (None, freeze_sha):
        raise FinalEvaluationError("freeze contains stale self hash")
    if freeze.get("schema") != "m3_final_freeze_v1":
        raise FinalEvaluationError("freeze schema mismatch")
    if freeze.get("fixture_mode") is not fixture_mode:
        raise FinalEvaluationError("freeze fixture mode mismatch")
    if freeze.get("fit_count") != 24:
        raise FinalEvaluationError("freeze does not bind 24 final fits")
    schedule = final_schedule()
    if freeze.get("schedule") != [spec.record() for spec in schedule]:
        raise FinalEvaluationError("freeze schedule mismatch")
    if freeze.get("outcome_access") != "unavailable_until_evaluate_gate":
        raise FinalEvaluationError("freeze outcome access policy mismatch")
    if freeze.get("authorization_sha256") != core.AUTHORIZATION_SHA256:
        raise FinalEvaluationError("freeze authorization hash mismatch")
    if freeze.get("feature_registry_sha256") != core.FEATURE_REGISTRY_SHA256:
        raise FinalEvaluationError("freeze feature registry hash mismatch")
    if freeze.get("protocol_config_sha256") != sha256_file(core.FIXED_PROTOCOL_CONFIG):
        raise FinalEvaluationError("freeze protocol config hash mismatch")
    if freeze.get("protocol_config_path") != str(
        core.FIXED_PROTOCOL_CONFIG.relative_to(core.REPO_ROOT)
    ):
        raise FinalEvaluationError("freeze protocol config path mismatch")
    if freeze.get("study_designation") != protocol["study_designation"]:
        raise FinalEvaluationError("freeze study designation mismatch")
    if _canonicalize(freeze.get("claim_limits")) != _canonicalize(protocol["claim_limits"]):
        raise FinalEvaluationError("freeze claim limits mismatch")
    _verify_metadata_authority_binding(freeze)
    _verify_development_selection_binding(freeze, trusted_setup=trusted_setup)
    _verify_fits_static(freeze, protocol, schedule)
    _verify_code_hashes(freeze["code_sha256"])


def _verify_development_selection_binding(
    freeze: Mapping[str, Any],
    *,
    trusted_setup: Mapping[str, Any] | None = None,
) -> None:
    selection = freeze.get("development_selection")
    receipt = freeze.get("development_receipt")
    authority = freeze.get("accepted_development_authority")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(authority, Mapping)
    ):
        raise FinalEvaluationError("freeze development binding missing")
    selection_sha = selection.get("selection_sha256")
    if selection_sha is not None and selection_sha != receipt.get("selection_sha256"):
        raise FinalEvaluationError("freeze development selection receipt mismatch")
    if trusted_setup is None:
        expected = fixture_development_authority(receipt, selection)
    else:
        if _canonicalize(receipt) != _canonicalize(
            trusted_setup.get("accepted_development_receipt")
        ):
            raise FinalEvaluationError(
                "freeze development authority receipt differs from trusted registry"
            )
        if _canonicalize(selection) != _canonicalize(
            trusted_setup.get("accepted_development_selection")
        ):
            raise FinalEvaluationError(
                "freeze development authority selection differs from trusted registry"
            )
        expected = trusted_setup.get("accepted_development_authority")
        if not isinstance(expected, Mapping):
            raise FinalEvaluationError("trusted registry lacks accepted development authority")
    if _canonicalize(authority) != _canonicalize(expected):
        raise FinalEvaluationError("freeze development authority mismatch")


def _verify_fits_static(
    freeze: Mapping[str, Any],
    protocol: Mapping[str, Any],
    schedule: list[FinalFitSpec],
) -> None:
    fits = freeze.get("fits")
    if not isinstance(fits, Mapping) or set(fits) != {spec.fit_id for spec in schedule}:
        raise FinalEvaluationError("freeze fit binding set mismatch")
    for spec in schedule:
        fit = fits[spec.fit_id]
        if not isinstance(fit, Mapping):
            raise FinalEvaluationError(f"{spec.fit_id} freeze fit binding missing")
        for key, value in spec.record().items():
            if fit.get(key) != value:
                raise FinalEvaluationError(f"{spec.fit_id} freeze fit identity mismatch")
        feature_columns = core.selected_feature_columns(spec.feature_set)
        if fit.get("feature_columns") != feature_columns:
            raise FinalEvaluationError(f"{spec.fit_id} freeze feature list mismatch")
        if fit.get("feature_list_sha256") != core.feature_list_digest(feature_columns):
            raise FinalEvaluationError(f"{spec.fit_id} freeze feature hash mismatch")
        config = _model_config(protocol, spec.model_family).serializable()
        if _canonicalize(fit.get("model_config")) != _canonicalize(config):
            raise FinalEvaluationError(f"{spec.fit_id} freeze model config mismatch")
        if fit.get("model_config_sha256") != _sha256_obj(config):
            raise FinalEvaluationError(f"{spec.fit_id} freeze model config hash mismatch")
        if fit.get("primary_cases_only_ridge") is not (
            spec.feature_set == "cases_only" and spec.model_family == "ridge"
        ):
            raise FinalEvaluationError(f"{spec.fit_id} primary model flag mismatch")
        trusted_selection = freeze["development_selection"]
        if fit.get("selected_development_champion") is not _is_selected_champion(
            spec,
            trusted_selection,
        ):
            raise FinalEvaluationError(f"{spec.fit_id} development champion flag mismatch")
        for name in ("training_keys", "training_digests", "thresholds"):
            if not isinstance(fit.get(name), Mapping):
                raise FinalEvaluationError(f"{spec.fit_id} freeze {name} missing")
        if not isinstance(fit.get("expected_preprocessing_fit_state"), Mapping):
            raise FinalEvaluationError(f"{spec.fit_id} expected preprocessing fit state missing")
        if fit["training_keys"].get("row_count") != fit["training_digests"].get("row_count"):
            raise FinalEvaluationError(f"{spec.fit_id} training row count binding mismatch")
        if fit["training_keys"].get("row_key_digest") != fit["training_digests"].get(
            "row_key_digest"
        ):
            raise FinalEvaluationError(f"{spec.fit_id} training key digest binding mismatch")
        if fit.get("threshold_binding_sha256") != _sha256_obj(fit["thresholds"]):
            raise FinalEvaluationError(f"{spec.fit_id} threshold binding hash mismatch")


def _verify_metadata_authority_binding(freeze: Mapping[str, Any]) -> None:
    authority = freeze.get("metadata_authority")
    if not isinstance(authority, Mapping):
        raise FinalEvaluationError("freeze metadata authority missing")
    expected = freeze.get("test_keys")
    if not isinstance(expected, Mapping):
        raise FinalEvaluationError("freeze test key binding missing")
    schedule_keys = _metadata_schedule_frame(authority, label="freeze metadata authority")
    full_scheduled_keys = core.freeze_common_keys(schedule_keys)
    if authority.get("full_scheduled_keys") != full_scheduled_keys:
        raise FinalEvaluationError("freeze metadata authority full schedule mismatch")
    first_scheduled = pd.Timestamp(schedule_keys["week_start_date"].min()).date().isoformat()
    if first_scheduled != freeze.get("first_scheduled_origin"):
        raise FinalEvaluationError("freeze metadata first scheduled origin mismatch")
    if authority.get("expected_keys") != expected:
        raise FinalEvaluationError("freeze metadata authority key mismatch")
    _verify_expected_key_subset(
        full_schedule=schedule_keys,
        expected_keys=expected,
        first_scheduled=pd.Timestamp(first_scheduled),
        label="freeze metadata authority",
    )
    if freeze.get("test_metadata_availability_digest") != _sha256_obj(
        {
            "first_scheduled_origin": freeze.get("first_scheduled_origin"),
            "expected_keys": expected,
        }
    ):
        raise FinalEvaluationError("freeze metadata availability digest mismatch")
    if _sha256_obj(authority) != freeze.get("metadata_authority_sha256"):
        raise FinalEvaluationError("freeze metadata authority hash mismatch")


def _approve_source(
    source_path: str | Path,
    *,
    approved_source_sha256: str | None,
    fixture_mode: bool,
) -> development.SourceApproval:
    return development._approve_source(
        source_path,
        fixture_mode=fixture_mode,
        approved_sha256=approved_source_sha256,
    )


def _load_prepare_training_observations(
    source: development.SourceApproval,
    *,
    protocol: Mapping[str, Any],
    first_origin: pd.Timestamp,
    reader: Reader | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = development.projected_columns(
        protocol["features"]["sets"],
        include_origin_provenance=True,
    )
    columns = _append_optional_source_columns(source.approved_path, columns)
    identity = _read_identity(source, columns, "prepare_freeze_metadata_and_training")
    identity["read_contract"] = "historical_training_values_only_before_first_scheduled_origin"
    cutoff = min(first_origin, core.CALENDAR_2025_BOUNDARY)
    frame = _read_frame(
        source.approved_path,
        columns,
        development._pyarrow_filter_expression(cutoff),
        identity,
        reader,
    )
    _assert_source_stable(source, identity)
    out = _validate_source_frame(frame)
    if pd.to_datetime(out["week_start_date"]).ge(first_origin).any():
        raise FinalEvaluationError("prepare historical training read crossed scheduled boundary")
    if pd.to_datetime(out["week_end_date"]).ge(core.CALENDAR_2025_BOUNDARY).any():
        raise FinalEvaluationError("prepare historical training read crossed 2025 boundary")
    identity.update(development._frame_receipt(out))
    identity["first_scheduled_origin"] = first_origin.date().isoformat()
    identity["future_outcome_values_projected"] = False
    return out, identity


def _load_prepare_metadata_rows(
    source: development.SourceApproval,
    *,
    first_origin: pd.Timestamp,
    reader: Reader | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["district_id", "week_start_date", "week_end_date"]
    columns = _append_optional_source_columns(source.approved_path, columns)
    identity = _read_identity(source, columns, "prepare_freeze_metadata_schedule_keys")
    identity["read_contract"] = "metadata_keys_and_schedule_flags_only_no_numeric_future_values"
    frame = _read_frame(source.approved_path, columns, None, identity, reader)
    _assert_source_stable(source, identity)
    out = _validate_source_frame(frame)
    if pd.to_datetime(out["week_start_date"]).max() < first_origin:
        raise FinalEvaluationError("prepare source lacks scheduled test metadata origins")
    identity.update(development._frame_receipt(out))
    identity["first_scheduled_origin"] = first_origin.date().isoformat()
    identity["future_numeric_values_projected"] = False
    return out, identity


def _load_historical_training_observations(
    source: development.SourceApproval,
    *,
    freeze: Mapping[str, Any],
    reader: Reader | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = development.load_protocol()
    first_origin = pd.Timestamp(freeze["first_scheduled_origin"])
    columns = development.projected_columns(
        protocol["features"]["sets"],
        include_origin_provenance=True,
    )
    columns = _append_optional_source_columns(source.approved_path, columns)
    identity = _read_identity(source, columns, "evaluate_freeze_historical_training_binding")
    identity["read_contract"] = "historical_training_values_only_before_outcome_unlock"
    cutoff = min(first_origin, core.CALENDAR_2025_BOUNDARY)
    frame = _read_frame(
        source.approved_path,
        columns,
        development._pyarrow_filter_expression(cutoff),
        identity,
        reader,
    )
    _assert_source_stable(source, identity)
    out = _validate_source_frame(frame)
    if pd.to_datetime(out["week_start_date"]).ge(first_origin).any():
        raise FinalEvaluationError("historical training read crossed scheduled boundary")
    if pd.to_datetime(out["week_end_date"]).ge(core.CALENDAR_2025_BOUNDARY).any():
        raise FinalEvaluationError("historical training read crossed 2025 boundary")
    identity.update(development._frame_receipt(out))
    identity["first_scheduled_origin"] = first_origin.date().isoformat()
    identity["outcomes_unlocked"] = False
    return out, identity


def _load_outcome_observations(
    source: development.SourceApproval,
    *,
    freeze: Mapping[str, Any],
    reader: Reader | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = development.load_protocol()
    columns = development.projected_columns(
        protocol["features"]["sets"],
        include_origin_provenance=True,
    )
    columns = _append_optional_source_columns(source.approved_path, columns)
    identity = _read_identity(source, columns, "evaluate_freeze_synthetic_outcomes")
    frame = _read_frame(source.approved_path, columns, None, identity, reader)
    _assert_source_stable(source, identity)
    out = _validate_source_frame(frame)
    identity.update(development._frame_receipt(out))
    identity["freeze_test_key_digest"] = freeze["test_keys"]["row_key_digest"]
    identity["synthetic_outcomes_unlocked"] = True
    return out, identity


def _verify_freeze_source_identity(
    freeze: Mapping[str, Any],
    source: development.SourceApproval,
) -> None:
    audit = freeze.get("read_audit")
    if not isinstance(audit, Mapping):
        raise FinalEvaluationError("freeze source audit missing")
    if audit.get("source_sha256") != source.approved_sha256:
        raise FinalEvaluationError("approved source does not match frozen source identity")
    if audit.get("approved_source_sha256") != source.approved_sha256:
        raise FinalEvaluationError("frozen approved source identity mismatch")


def _read_identity(
    source: development.SourceApproval,
    columns: list[str],
    purpose: str,
) -> dict[str, Any]:
    before = development._file_identity(source.approved_path)
    if before["sha256"] != source.approved_sha256:
        raise FinalEvaluationError("source sha256 does not match approved input identity")
    return {
        "source_mode": source.mode,
        "source_path": str(source.approved_path),
        "source_sha256": before["sha256"],
        "approved_source_sha256": source.approved_sha256,
        "source_identity_before_read": before,
        "projection": columns,
        "purpose": purpose,
    }


def _append_optional_source_columns(path: Path, columns: list[str]) -> list[str]:
    available = development._source_schema_audit(path)["column_names"]
    out = list(columns)
    for column in ("m3_final_evaluation_origin",):
        if column in available and column not in out:
            out.append(column)
    return out


def _read_frame(
    path: Path,
    columns: list[str],
    filter_expression: Any,
    identity: Mapping[str, Any],
    reader: Reader | None,
) -> pd.DataFrame:
    active = reader or development._pyarrow_projected_reader
    available = development._source_schema_audit(path)["column_names"]
    effective = [column for column in columns if column in available]
    return active(path, effective, filter_expression, dict(identity))


def _assert_source_stable(
    source: development.SourceApproval,
    identity: dict[str, Any],
) -> None:
    after = development._file_identity(source.approved_path)
    if after != identity["source_identity_before_read"]:
        raise FinalEvaluationError("source identity changed during read")
    identity["source_identity_after_read"] = after


def _validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("week_start_date", "week_end_date"):
        if column not in out.columns:
            raise FinalEvaluationError(f"source frame missing {column}")
        out[column] = pd.to_datetime(out[column], errors="coerce")
    if out.empty:
        raise FinalEvaluationError("source read returned no rows")
    if out[["week_start_date", "week_end_date"]].isna().any().any():
        raise FinalEvaluationError("source read returned invalid dates")
    if out["week_end_date"].ne(out["week_start_date"] + pd.Timedelta(days=6)).any():
        raise FinalEvaluationError("source read returned non-weekly intervals")
    if out.duplicated(KEY_COLUMNS).any():
        raise FinalEvaluationError("source read returned duplicate district/week keys")
    if out["district_id"].isna().any():
        raise FinalEvaluationError("source read returned null district_id")
    return out.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _final_training_frame(
    tasks: pd.DataFrame,
    *,
    horizon: int,
    first_origin: pd.Timestamp,
) -> pd.DataFrame:
    out = tasks.copy()
    target_end = pd.to_datetime(out["week_start_date"]) + pd.to_timedelta(
        7 * horizon + 6,
        unit="D",
    )
    embargo = target_end + pd.Timedelta(days=7)
    mask = (
        pd.to_datetime(out["week_end_date"]).lt(core.CALENDAR_2025_BOUNDARY)
        & target_end.lt(first_origin)
        & embargo.lt(first_origin)
        & core.origin_eligible_mask(out, horizon=horizon)
    )
    selected = out.loc[mask].sort_values(KEY_COLUMNS).reset_index(drop=True)
    if selected.empty:
        raise FinalEvaluationError(f"h{horizon} final training cohort is empty")
    if pd.to_datetime(selected["week_end_date"]).ge(core.CALENDAR_2025_BOUNDARY).any():
        raise FinalEvaluationError("final training origins must be pre-2025 only")
    return selected


def _verify_metadata_authority(
    authority: Mapping[str, Any] | None,
    *,
    supplied_first_origin: pd.Timestamp,
    fixture_mode: bool,
) -> dict[str, Any]:
    if authority is None or not isinstance(authority, Mapping):
        raise FinalEvaluationError("prepare requires explicit metadata schedule authority")
    if fixture_mode and authority.get("mode") != "synthetic_fixture_metadata_authority":
        raise FinalEvaluationError("synthetic metadata authority mode mismatch")
    if not fixture_mode:
        if authority.get("schema") != "m3_final_metadata_authority_v1":
            raise FinalEvaluationError("production metadata authority schema mismatch")
        if authority.get("mode") != "production_metadata_authority":
            raise FinalEvaluationError("production metadata authority mode mismatch")
        if authority.get("source") != "parent_metadata_only_availability_extraction":
            raise FinalEvaluationError("production metadata authority source mismatch")
    rows = _metadata_schedule_frame(authority, label="metadata authority")
    rows = rows.sort_values(KEY_COLUMNS).reset_index(drop=True)
    first_scheduled = pd.Timestamp(rows["week_start_date"].min())
    if first_scheduled != supplied_first_origin:
        raise FinalEvaluationError("supplied first scheduled origin differs from authority")
    expected_keys = authority.get("expected_keys")
    if not isinstance(expected_keys, Mapping):
        raise FinalEvaluationError("metadata authority expected keys missing")
    _verify_expected_key_subset(
        full_schedule=rows,
        expected_keys=expected_keys,
        first_scheduled=first_scheduled,
        label="metadata authority",
    )
    payload = {
        **(
            {
                "schema": authority.get("schema"),
                "source": authority.get("source"),
            }
            if not fixture_mode
            else {}
        ),
        "mode": authority.get("mode"),
        "schedule": core.freeze_common_keys(rows)["keys"],
        "full_scheduled_keys": core.freeze_common_keys(rows),
        "expected_keys": expected_keys,
    }
    return {
        "first_scheduled_origin": first_scheduled.date().isoformat(),
        "payload": payload,
        "authority_sha256": _sha256_obj(payload),
        "expected_keys": expected_keys,
    }


def _metadata_schedule_frame(authority: Mapping[str, Any], *, label: str) -> pd.DataFrame:
    schedule = authority.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise FinalEvaluationError(f"{label} schedule is empty")
    rows = pd.DataFrame(schedule)
    missing = set(KEY_COLUMNS) - set(rows.columns)
    if missing:
        raise FinalEvaluationError(f"{label} schedule missing keys: {sorted(missing)}")
    rows = rows[KEY_COLUMNS].copy()
    rows["week_start_date"] = pd.to_datetime(rows["week_start_date"], errors="coerce")
    if rows["week_start_date"].isna().any() or rows["district_id"].isna().any():
        raise FinalEvaluationError(f"{label} schedule has invalid keys")
    if rows.duplicated(KEY_COLUMNS).any():
        raise FinalEvaluationError(f"{label} schedule has duplicate keys")
    return rows.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _verify_expected_key_subset(
    *,
    full_schedule: pd.DataFrame,
    expected_keys: Mapping[str, Any],
    first_scheduled: pd.Timestamp,
    label: str,
) -> None:
    keys = pd.DataFrame(expected_keys.get("keys", []))
    missing = set(KEY_COLUMNS) - set(keys.columns)
    if missing:
        raise FinalEvaluationError(f"{label} expected keys missing columns: {sorted(missing)}")
    keys = keys[KEY_COLUMNS].copy()
    keys["week_start_date"] = pd.to_datetime(keys["week_start_date"], errors="coerce")
    if keys["week_start_date"].isna().any() or keys["district_id"].isna().any():
        raise FinalEvaluationError(f"{label} expected keys are invalid")
    if keys.duplicated(KEY_COLUMNS).any():
        raise FinalEvaluationError(f"{label} expected keys contain duplicates")
    if keys.empty:
        raise FinalEvaluationError(f"{label} expected keys are empty")
    if keys["week_start_date"].lt(first_scheduled).any():
        raise FinalEvaluationError(f"{label} expected keys precede scheduled boundary")
    try:
        core.verify_common_keys(keys, expected_keys)
    except core.Milestone3ProtocolError as exc:
        raise FinalEvaluationError(f"{label} expected key set mismatch") from exc
    schedule_keys = core.freeze_common_keys(full_schedule)["keys"]
    schedule_set = {
        (row["district_id"], row["week_start_date"])
        for row in schedule_keys
    }
    expected_set = {
        (str(row.district_id), pd.Timestamp(row.week_start_date).date().isoformat())
        for row in keys.itertuples(index=False)
    }
    if not expected_set.issubset(schedule_set):
        raise FinalEvaluationError(f"{label} expected keys are not a subset of schedule")


def _test_metadata_rows(
    tasks: pd.DataFrame,
    *,
    schedule_authority: Mapping[str, Any],
) -> pd.DataFrame:
    first_origin = pd.Timestamp(schedule_authority["first_scheduled_origin"])
    _verify_optional_origin_marker(
        tasks,
        expected_keys=schedule_authority["expected_keys"],
        first_origin=first_origin,
        label="test metadata",
    )
    rows = _select_expected_key_rows(
        tasks,
        schedule_authority["expected_keys"],
        label="test metadata",
    )
    if pd.to_datetime(rows["week_start_date"]).lt(first_origin).any():
        raise FinalEvaluationError("test metadata authority includes pre-boundary origin")
    if rows.empty:
        raise FinalEvaluationError("test metadata cohort is empty")
    rows = rows.sort_values(KEY_COLUMNS).reset_index(drop=True)
    expected = schedule_authority["expected_keys"]
    try:
        core.verify_common_keys(rows, expected)
    except core.Milestone3ProtocolError as exc:
        raise FinalEvaluationError("test metadata key set differs from authority") from exc
    return rows


def _evaluation_outcome_rows(tasks: pd.DataFrame, *, freeze: Mapping[str, Any]) -> pd.DataFrame:
    first_origin = pd.Timestamp(freeze["first_scheduled_origin"])
    _verify_optional_origin_marker(
        tasks,
        expected_keys=freeze["test_keys"],
        first_origin=first_origin,
        label="evaluation outcome",
    )
    rows = _select_expected_key_rows(tasks, freeze["test_keys"], label="evaluation outcome")
    rows = (
        rows.loc[core.common_evaluation_mask(rows)]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    if rows.empty:
        raise FinalEvaluationError("evaluation outcome cohort is empty")
    core.verify_common_keys(rows, freeze["test_keys"])
    return rows


def _select_expected_key_rows(
    frame: pd.DataFrame,
    expected_keys: Mapping[str, Any],
    *,
    label: str,
) -> pd.DataFrame:
    keys = pd.DataFrame(expected_keys.get("keys", []))
    if set(KEY_COLUMNS) - set(keys.columns):
        raise FinalEvaluationError(f"{label} expected keys are missing columns")
    keys = keys[KEY_COLUMNS].copy()
    keys["week_start_date"] = pd.to_datetime(keys["week_start_date"], errors="coerce")
    if keys["week_start_date"].isna().any() or keys["district_id"].isna().any():
        raise FinalEvaluationError(f"{label} expected keys are invalid")
    if keys.duplicated(KEY_COLUMNS).any():
        raise FinalEvaluationError(f"{label} expected keys contain duplicates")
    source = frame.copy()
    source["week_start_date"] = pd.to_datetime(source["week_start_date"], errors="coerce")
    selected = keys.merge(source, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if len(selected) != int(expected_keys.get("row_count", -1)):
        raise FinalEvaluationError(f"{label} expected key row count mismatch")
    if selected["week_end_date"].isna().any():
        raise FinalEvaluationError(f"{label} source is missing expected authority keys")
    core.verify_common_keys(selected, expected_keys)
    return selected.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _verify_optional_origin_marker(
    frame: pd.DataFrame,
    *,
    expected_keys: Mapping[str, Any],
    first_origin: pd.Timestamp,
    label: str,
) -> None:
    if "m3_final_evaluation_origin" not in frame.columns:
        return
    origins = pd.to_datetime(frame["week_start_date"])
    marked = frame.loc[
        origins.ge(first_origin) & frame["m3_final_evaluation_origin"].astype(bool)
    ].copy()
    try:
        core.verify_common_keys(marked, expected_keys)
    except core.Milestone3ProtocolError as exc:
        raise FinalEvaluationError(f"{label} marker key set differs from authority") from exc


def _validate_test_metadata_common_cohort(test_metadata: pd.DataFrame) -> None:
    base_keys = core.freeze_common_keys(test_metadata)
    core.verify_common_keys(test_metadata, base_keys)


def _freeze_manifest(
    *,
    freeze_id: str,
    fixture_mode: bool,
    protocol: Mapping[str, Any],
    gate_receipts: Mapping[str, Any],
    development_receipt: Mapping[str, Any],
    development_selection: Mapping[str, Any],
    accepted_development_authority: Mapping[str, Any],
    read_audit: Mapping[str, Any],
    first_origin: pd.Timestamp,
    test_metadata: pd.DataFrame,
    test_keys: Mapping[str, Any],
    metadata_authority: Mapping[str, Any],
    metadata_read_audit: Mapping[str, Any],
    training_by_horizon: Mapping[int, pd.DataFrame],
    schedule: list[FinalFitSpec],
) -> dict[str, Any]:
    protocol_hash = sha256_file(core.FIXED_PROTOCOL_CONFIG)
    fits = {}
    for spec in schedule:
        feature_columns = core.selected_feature_columns(spec.feature_set)
        training = training_by_horizon[spec.horizon]
        thresholds = core.threshold_binding(
            training,
            target_column=spec.target_column,
            feature_columns=(),
        )
        fits[spec.fit_id] = {
            **spec.record(),
            "primary_cases_only_ridge": (
                spec.feature_set == "cases_only" and spec.model_family == "ridge"
            ),
            "selected_development_champion": _is_selected_champion(
                spec,
                development_selection,
            ),
            "feature_columns": feature_columns,
            "feature_list_sha256": core.feature_list_digest(feature_columns),
            "model_config": _model_config(protocol, spec.model_family).serializable(),
            "model_config_sha256": _sha256_obj(
                _model_config(protocol, spec.model_family).serializable()
            ),
            "training_keys": core.freeze_common_keys(training),
            "training_digests": core.cohort_digests(
                training,
                target_column=spec.target_column,
                feature_columns=feature_columns,
            ),
            "expected_preprocessing_fit_state": _expected_preprocessing_fit_state(
                training,
                feature_columns=feature_columns,
                model_family=_model_config(protocol, spec.model_family).family,
            ),
            "thresholds": thresholds,
            "threshold_binding_sha256": _sha256_obj(thresholds),
        }
    return {
        "schema": "m3_final_freeze_v1",
        "freeze_id": freeze_id,
        "freeze_sha256": None,
        "fixture_mode": fixture_mode,
        "fit_count": len(schedule),
        "schedule": [spec.record() for spec in schedule],
        "authorization_sha256": core.AUTHORIZATION_SHA256,
        "protocol_config_path": str(core.FIXED_PROTOCOL_CONFIG.relative_to(core.REPO_ROOT)),
        "protocol_config_sha256": protocol_hash,
        "feature_registry_sha256": core.FEATURE_REGISTRY_SHA256,
        "study_designation": protocol["study_designation"],
        "claim_limits": protocol["claim_limits"],
        "first_scheduled_origin": first_origin.date().isoformat(),
        "first_scheduled_origin_policy": (
            "explicit audited metadata boundary; not derived from first surviving evaluation row"
        ),
        "test_keys": dict(test_keys),
        "metadata_authority": dict(metadata_authority["payload"]),
        "metadata_authority_sha256": str(metadata_authority["authority_sha256"]),
        "test_metadata_availability_digest": _metadata_availability_digest(
            first_origin=first_origin,
            expected_keys=test_keys,
        ),
        "test_key_count": int(test_keys["row_count"]),
        "outcome_access": "unavailable_until_evaluate_gate",
        "training_policy": {
            "origins": "pre_2025_only",
            "embargo": "target_end_plus_7_days_lt_first_scheduled_origin",
            "preprocessing": "fit_training_rows_only",
            "tuning": "zero_final_tuning_or_early_stopping",
        },
        "development_receipt": dict(development_receipt),
        "development_selection": development_selection,
        "accepted_development_authority": dict(accepted_development_authority),
        "prepare_gate_receipts": dict(gate_receipts),
        "read_audit": read_audit,
        "metadata_read_audit": metadata_read_audit,
        "code_sha256": _code_hashes(),
        "environment": _environment(),
        "fits": fits,
        "final_winner_selection": "prohibited",
        "operational_claim": "prohibited",
        "partial_year_warning": "2025 cohort is not representative annual performance",
    }


def _is_selected_champion(
    spec: FinalFitSpec,
    development_selection: Mapping[str, Any],
) -> bool:
    selected = development_selection.get("selected", {}).get(f"h{spec.horizon}", {})
    return (
        selected.get("feature_set") == spec.feature_set
        and selected.get("model_family") == spec.model_family
    )


def _fit_and_predict_all(
    run_dir: Path,
    freeze: Mapping[str, Any],
    training_by_horizon: Mapping[int, pd.DataFrame],
    evaluation: pd.DataFrame,
    *,
    trusted_outcome_content: Mapping[str, Any],
    fitter: Fitter,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in final_schedule():
        fit_freeze = freeze["fits"][spec.fit_id]
        fit_dir = run_dir / "fits" / spec.fit_id
        fit_dir.mkdir(parents=True, exist_ok=False)
        _write_json_new(fit_dir / "claim.json", {"fit": spec.record(), "status": "claimed"})
        training = training_by_horizon[spec.horizon].copy(deep=True)
        feature_columns = list(fit_freeze["feature_columns"])
        _verify_fit_inputs(spec, training, evaluation, feature_columns, fit_freeze)
        model_dir = fit_dir / "model"
        model = fitter(
            training,
            feature_columns=feature_columns,
            target_column=spec.target_column,
            config=_model_config(development.load_protocol(), spec.model_family),
            output_dir=model_dir,
            frozen_period_bounds={
                "train_start": pd.to_datetime(training["week_start_date"]).min().date().isoformat(),
                "train_end": pd.to_datetime(training["week_start_date"]).max().date().isoformat(),
            },
            provenance={
                "m3_fit_id": spec.fit_id,
                "m3_horizon": spec.horizon,
                "m3_feature_set": spec.feature_set,
                "m3_stage": "final_evaluation",
            },
        )
        _verify_returned_final_model(
            spec,
            model,
            _model_config(development.load_protocol(), spec.model_family),
            feature_columns,
        )
        predictions = development._prediction_frame(
            development.FitSpec(spec.horizon, spec.feature_set, spec.model_family, 2025),
            model,
            evaluation,
            feature_columns,
            fit_freeze["thresholds"],
        )
        predictions["fit_id"] = spec.fit_id
        pred_path = fit_dir / "predictions.parquet"
        _write_parquet_new(predictions, pred_path)
        saved = pd.read_parquet(pred_path)
        development._verify_prediction_frame(saved, predictions)
        threshold_path = fit_dir / "thresholds.json"
        _write_json_new(threshold_path, fit_freeze["thresholds"])
        pending = {
            **spec.record(),
            "status": "predictions_saved_pending_metrics",
            "training_count": fit_freeze["training_keys"]["row_count"],
            "evaluation_count": freeze["test_keys"]["row_count"],
            "training_key_digest": fit_freeze["training_keys"]["row_key_digest"],
            "evaluation_key_digest": freeze["test_keys"]["row_key_digest"],
            "training_feature_content_digest": fit_freeze["training_digests"][
                "feature_content_digest"
            ],
            "training_target_content_digest": fit_freeze["training_digests"][
                "target_content_digest"
            ],
            "threshold_path": str(threshold_path.relative_to(run_dir)),
            "threshold_sha256": sha256_file(threshold_path),
            "threshold_binding_sha256": fit_freeze["threshold_binding_sha256"],
            "model_path": str(model_dir.relative_to(run_dir)),
            "model_sha256": sha256_file(model_dir / "model.joblib"),
            "model_metadata_sha256": sha256_file(model_dir / "metadata.json"),
            "prediction_path": str(pred_path.relative_to(run_dir)),
            "prediction_sha256": sha256_file(pred_path),
            "prediction_values_sha256": _sha256_obj(
                development._canonical_frame_records(saved)
            ),
            "trusted_outcome_content_sha256": _outcome_digest_for_horizon(
                trusted_outcome_content,
                spec.horizon,
            ),
            "metrics_path": str((fit_dir / "metrics.json").relative_to(run_dir)),
            "model_config_sha256": fit_freeze["model_config_sha256"],
            "feature_list_sha256": fit_freeze["feature_list_sha256"],
        }
        _write_json_new(fit_dir / "prediction_receipt_pending_metrics.json", pending)
        rows.append(pending)
    return rows


def _score_saved_predictions(
    run_dir: Path,
    freeze: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _verify_prediction_barrier(run_dir, freeze, rows)
    complete_rows: list[dict[str, Any]] = []
    for row in rows:
        fit_id = str(row["fit_id"])
        predictions = pd.read_parquet(run_dir / row["prediction_path"])
        threshold_doc = _read_json(run_dir / row["threshold_path"])
        metric_doc = development._metric_document(predictions, threshold_doc)
        metrics_path = run_dir / row["metrics_path"]
        _write_json_new(metrics_path, metric_doc)
        complete = {
            **row,
            "status": "complete",
            "metrics_sha256": sha256_file(metrics_path),
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
        _write_json_new(run_dir / "fits" / fit_id / "complete.json", complete)
        complete_rows.append(complete)
    _validate_final_registry(pd.DataFrame(complete_rows))
    if freeze["fit_count"] != len(complete_rows):
        raise FinalEvaluationError("final registry count differs from freeze")
    return complete_rows


def _verify_prediction_barrier(
    run_dir: Path,
    freeze: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) != 24:
        raise FinalEvaluationError("all 24 prediction vectors must be saved before metrics")
    expected = {(s.horizon, s.feature_set, s.model_family, s.fit_id) for s in final_schedule()}
    observed = {
        (int(row["horizon"]), str(row["feature_set"]), str(row["model_family"]), str(row["fit_id"]))
        for row in rows
    }
    if observed != expected:
        raise FinalEvaluationError("prediction barrier does not contain exact 24 scheduled fits")
    for path_column in ("prediction_path", "threshold_path", "model_path"):
        values = [str(row[path_column]) for row in rows]
        if len(values) != len(set(values)):
            raise FinalEvaluationError(f"prediction barrier contains duplicate {path_column}")
    for row in rows:
        fit_id = str(row["fit_id"])
        fit_freeze = freeze["fits"][fit_id]
        pred_path = run_dir / row["prediction_path"]
        threshold_path = run_dir / row["threshold_path"]
        pending_path = run_dir / "fits" / fit_id / "prediction_receipt_pending_metrics.json"
        if not pending_path.exists():
            raise FinalEvaluationError(f"{fit_id} pending receipt missing before metrics")
        pending = _read_json(pending_path)
        for key in _pending_barrier_keys():
            if _canonicalize(pending.get(key)) != _canonicalize(row.get(key)):
                raise FinalEvaluationError(f"{fit_id} pending receipt mismatch before metrics")
        if not pred_path.exists():
            raise FinalEvaluationError("missing prediction vector before metrics")
        if not threshold_path.exists():
            raise FinalEvaluationError("missing threshold receipt before metrics")
        if str(row["prediction_sha256"]) != sha256_file(pred_path):
            raise FinalEvaluationError(f"{fit_id} prediction hash mismatch before metrics")
        if str(row["threshold_sha256"]) != sha256_file(threshold_path):
            raise FinalEvaluationError(f"{fit_id} threshold hash mismatch before metrics")
        threshold_doc = _read_json(threshold_path)
        if _canonicalize(threshold_doc) != _canonicalize(fit_freeze["thresholds"]):
            raise FinalEvaluationError(f"{fit_id} threshold receipt mismatch before metrics")
        predictions = pd.read_parquet(pred_path)
        if str(row.get("prediction_values_sha256")) != _sha256_obj(
            development._canonical_frame_records(predictions)
        ):
            raise FinalEvaluationError(f"{fit_id} prediction value digest mismatch before metrics")
        trusted_digest = row.get("trusted_outcome_content_sha256")
        if not isinstance(trusted_digest, str):
            raise FinalEvaluationError(f"{fit_id} trusted outcome receipt missing before metrics")
        _verify_prediction_vector_semantics(
            row,
            predictions,
            fit_freeze=fit_freeze,
            freeze=freeze,
            expected_outcome_digest=trusted_digest,
            stage="before metrics",
        )


def _pending_barrier_keys() -> tuple[str, ...]:
    return (
        "horizon",
        "feature_set",
        "model_family",
        "fit_id",
        "training_count",
        "evaluation_count",
        "training_key_digest",
        "evaluation_key_digest",
        "training_feature_content_digest",
        "training_target_content_digest",
        "threshold_path",
        "threshold_sha256",
        "threshold_binding_sha256",
        "model_path",
        "model_sha256",
        "model_metadata_sha256",
        "prediction_path",
        "prediction_sha256",
        "prediction_values_sha256",
        "trusted_outcome_content_sha256",
        "metrics_path",
        "model_config_sha256",
        "feature_list_sha256",
    )


def _verify_fit_inputs(
    spec: FinalFitSpec,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: list[str],
    fit_freeze: Mapping[str, Any],
) -> None:
    core.verify_common_keys(training, fit_freeze["training_keys"])
    core.verify_threshold_binding(
        training,
        fit_freeze["thresholds"],
        target_column=spec.target_column,
    )
    observed = core.cohort_digests(
        training,
        target_column=spec.target_column,
        feature_columns=feature_columns,
    )
    if _canonicalize(observed) != _canonicalize(fit_freeze["training_digests"]):
        raise FinalEvaluationError(f"{spec.fit_id} training digest mismatch")
    if not core.origin_eligible_mask(evaluation, horizon=spec.horizon).all():
        raise FinalEvaluationError(f"{spec.fit_id} evaluation availability mismatch")


def _verify_returned_final_model(
    spec: FinalFitSpec,
    model: TrainedModel,
    expected_config: ModelConfig,
    feature_columns: list[str],
) -> None:
    if model.target_column != spec.target_column:
        raise FinalEvaluationError(f"{spec.fit_id} returned model target mismatch")
    if model.feature_columns != feature_columns:
        raise FinalEvaluationError(f"{spec.fit_id} returned model feature mismatch")
    if _canonicalize(model.config.serializable()) != _canonicalize(expected_config.serializable()):
        raise FinalEvaluationError(f"{spec.fit_id} returned model config mismatch")
    metadata = model.metadata
    if metadata.get("target_column") != spec.target_column:
        raise FinalEvaluationError(f"{spec.fit_id} metadata target mismatch")
    if metadata.get("feature_columns") != feature_columns:
        raise FinalEvaluationError(f"{spec.fit_id} metadata feature mismatch")
    if _canonicalize(metadata.get("config")) != _canonicalize(expected_config.serializable()):
        raise FinalEvaluationError(f"{spec.fit_id} metadata config mismatch")
    provenance = metadata.get("row_identity", {})
    for key, value in {
        "m3_fit_id": spec.fit_id,
        "m3_horizon": spec.horizon,
        "m3_feature_set": spec.feature_set,
        "m3_stage": "final_evaluation",
    }.items():
        if provenance.get(key) != value:
            raise FinalEvaluationError(f"{spec.fit_id} metadata provenance mismatch")


def _verify_training_against_freeze(
    training_by_horizon: Mapping[int, pd.DataFrame],
    freeze: Mapping[str, Any],
) -> None:
    for spec in final_schedule():
        fit = freeze["fits"][spec.fit_id]
        training = training_by_horizon[spec.horizon]
        feature_columns = list(fit["feature_columns"])
        core.verify_common_keys(training, fit["training_keys"])
        core.verify_threshold_binding(
            training,
            fit["thresholds"],
            target_column=spec.target_column,
        )
        observed = core.cohort_digests(
            training,
            target_column=spec.target_column,
            feature_columns=feature_columns,
        )
        if _canonicalize(observed) != _canonicalize(fit["training_digests"]):
            raise FinalEvaluationError(f"{spec.fit_id} training digest mismatch")
        expected_state = fit.get("expected_preprocessing_fit_state")
        if not isinstance(expected_state, Mapping):
            raise FinalEvaluationError(f"{spec.fit_id} expected preprocessing fit state missing")
        actual_state = _expected_preprocessing_fit_state(
            training,
            feature_columns=feature_columns,
            model_family=_model_config(development.load_protocol(), spec.model_family).family,
        )
        if _canonicalize(actual_state) != _canonicalize(expected_state):
            raise FinalEvaluationError(f"{spec.fit_id} expected preprocessing fit state mismatch")


def _verify_evaluation_against_freeze(evaluation: pd.DataFrame, freeze: Mapping[str, Any]) -> None:
    core.verify_common_keys(evaluation, freeze["test_keys"])
    for horizon in HORIZONS:
        if not core.origin_eligible_mask(evaluation, horizon=horizon).all():
            raise FinalEvaluationError(f"h{horizon} evaluation cohort mismatch")


def _trusted_outcome_content(
    evaluation: pd.DataFrame,
    *,
    freeze: Mapping[str, Any],
    read_audit: Mapping[str, Any],
) -> dict[str, Any]:
    core.verify_common_keys(evaluation, freeze["test_keys"])
    by_horizon = {
        f"h{horizon}": _evaluation_outcome_digest(evaluation, horizon)
        for horizon in HORIZONS
    }
    return {
        "schema": "m3_final_trusted_outcome_content_v1",
        "source_sha256": read_audit.get("source_sha256"),
        "approved_source_sha256": read_audit.get("approved_source_sha256"),
        "read_contract": read_audit.get("purpose"),
        "test_key_digest": freeze["test_keys"]["row_key_digest"],
        "test_key_count": freeze["test_keys"]["row_count"],
        "by_horizon": by_horizon,
    }


def _evaluation_outcome_digest(evaluation: pd.DataFrame, horizon: int) -> str:
    target_column = f"target_h{horizon}"
    records = []
    for row in evaluation.sort_values(KEY_COLUMNS).itertuples(index=False):
        data = row._asdict()
        records.append(
            {
                "district_id": str(data["district_id"]),
                "week_start_date": pd.Timestamp(data["week_start_date"]).date().isoformat(),
                "observed_current_cases": float(data["dengue_cases"]),
                "observed_target": float(data[target_column]),
            }
        )
    return _sha256_obj(records)


def _prediction_outcome_digest(predictions: pd.DataFrame, horizon: int) -> str:
    expected_target_start = pd.to_datetime(predictions["origin_start"]) + pd.to_timedelta(
        7 * horizon,
        unit="D",
    )
    if not pd.to_datetime(predictions["target_start"]).equals(expected_target_start):
        raise FinalEvaluationError("prediction target calendar mismatch for outcome digest")
    frame = predictions.copy()
    frame["week_start_date"] = pd.to_datetime(frame["origin_start"]).dt.date.astype(str)
    records = []
    for row in frame.sort_values(["district_id", "week_start_date"]).itertuples(index=False):
        data = row._asdict()
        records.append(
            {
                "district_id": str(data["district_id"]),
                "week_start_date": str(data["week_start_date"]),
                "observed_current_cases": float(data["observed_current_cases"]),
                "observed_target": float(data["observed_target"]),
            }
        )
    return _sha256_obj(records)


def _outcome_digest_for_horizon(authority: Mapping[str, Any], horizon: int) -> str:
    by_horizon = authority.get("by_horizon")
    if not isinstance(by_horizon, Mapping):
        raise FinalEvaluationError("trusted outcome content lacks horizon digests")
    digest = by_horizon.get(f"h{horizon}")
    if not isinstance(digest, str):
        raise FinalEvaluationError(f"h{horizon} trusted outcome digest missing")
    return digest


def _validate_final_registry(registry: pd.DataFrame) -> None:
    required = {
        "horizon",
        "feature_set",
        "model_family",
        "fit_id",
        "status",
        "training_count",
        "evaluation_count",
        "training_key_digest",
        "evaluation_key_digest",
        "threshold_path",
        "threshold_sha256",
        "threshold_binding_sha256",
        "model_path",
        "model_sha256",
        "model_metadata_sha256",
        "prediction_path",
        "prediction_sha256",
        "prediction_values_sha256",
        "trusted_outcome_content_sha256",
        "metrics_path",
        "metrics_sha256",
        "mae_model",
        "rmse_model",
        "mae_persistence",
        "rmse_persistence",
    }
    missing = required - set(registry.columns)
    if missing:
        raise FinalEvaluationError(f"final registry missing columns: {sorted(missing)}")
    expected = {(s.horizon, s.feature_set, s.model_family, s.fit_id) for s in final_schedule()}
    observed = {
        (int(row.horizon), str(row.feature_set), str(row.model_family), str(row.fit_id))
        for row in registry.itertuples(index=False)
    }
    if observed != expected or len(registry) != 24:
        raise FinalEvaluationError("final registry tuple set does not match expected 24 fits")
    if set(registry["status"].astype(str)) != {"complete"}:
        raise FinalEvaluationError("final registry contains non-complete rows")
    for column in ("prediction_path", "metrics_path", "model_path"):
        if not registry[column].astype(str).is_unique:
            raise FinalEvaluationError(f"final registry contains duplicate {column}")


def _complete_manifest(
    *,
    run_id: str,
    freeze: Mapping[str, Any],
    freeze_sha: str,
    read_audit: Mapping[str, Any],
    trusted_outcome_content: Mapping[str, Any],
    trusted_outcome_receipt: Mapping[str, Any],
    registry: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "m3_final_evaluation_complete_v1",
        "run_id": run_id,
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": freeze_sha,
        "fit_count": len(registry),
        "prediction_vectors_saved_before_metrics": True,
        "final_winner_selection": "not_performed",
        "operational_claim": "not_performed",
        "partial_year_warning": freeze["partial_year_warning"],
        "read_audit": read_audit,
        "trusted_outcome_content": dict(trusted_outcome_content),
        "trusted_outcome_receipt": dict(trusted_outcome_receipt),
        "registry_sha256": _sha256_obj(registry),
    }


def _required_completed_artifacts() -> set[str]:
    required = {
        "claim.json",
        "complete_manifest.json",
        "final_model_registry.parquet",
        "freeze_manifest.json",
    }
    for spec in final_schedule():
        prefix = f"fits/{spec.fit_id}"
        required.update(
            {
                f"{prefix}/claim.json",
                f"{prefix}/complete.json",
                f"{prefix}/metrics.json",
                f"{prefix}/prediction_receipt_pending_metrics.json",
                f"{prefix}/predictions.parquet",
                f"{prefix}/thresholds.json",
                f"{prefix}/model/metadata.json",
                f"{prefix}/model/model.joblib",
            }
        )
    return required


def _verify_artifact_paths(root: Path, checksums: Mapping[str, Any]) -> None:
    if not isinstance(checksums, Mapping) or not checksums:
        raise FinalEvaluationError("checksum manifest is empty")
    seen: set[Path] = set()
    for rel_path in checksums:
        target = _checked_artifact_path(root, str(rel_path))
        if target in seen:
            raise FinalEvaluationError("checksum manifest contains duplicate artifact path")
        seen.add(target)


def _checked_artifact_path(root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        raise FinalEvaluationError(f"invalid artifact path: {rel_path}")
    target = root / rel
    development._reject_symlink_components(target)
    resolved = target.resolve()
    if not development._is_relative_to(resolved, root):
        raise FinalEvaluationError(f"artifact path escapes run directory: {rel_path}")
    if target.is_symlink():
        raise FinalEvaluationError(f"artifact path cannot be a symlink: {rel_path}")
    return target


def _verify_completed_claim(
    claim: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    freeze_sha: str,
) -> None:
    if claim.get("run_id") != manifest.get("run_id"):
        raise FinalEvaluationError("run claim id mismatch")
    if claim.get("status") != "claimed_once_only":
        raise FinalEvaluationError("run claim status mismatch")
    if claim.get("freeze_sha256") != freeze_sha:
        raise FinalEvaluationError("run claim freeze hash mismatch")
    consumed = claim.get("consumed_authorization_claim")
    if not isinstance(consumed, Mapping):
        raise FinalEvaluationError("run claim lacks consumed experiment authority")
    experiment_id = _validate_experiment_id(consumed.get("experiment_id"))
    authority_path = _trusted_consumed_path(
        experiment_id,
        fixture_mode=bool(freeze["fixture_mode"]),
    )
    if not authority_path.exists() or not authority_path.is_file():
        raise FinalEvaluationError("consumed experiment authority is missing")
    if consumed.get("claim_sha256") != sha256_file(authority_path):
        raise FinalEvaluationError("consumed experiment authority hash mismatch")
    authority = _read_json(authority_path)
    setup = _trusted_setup_record(consumed, fixture_mode=bool(freeze["fixture_mode"]))
    expected = {
        "schema": "m3_final_consumed_experiment_claim_v1",
        "experiment_id": experiment_id,
        "freeze_id": freeze.get("freeze_id"),
        "freeze_sha256": freeze_sha,
        "authorized_source_sha256": setup["authorized_source_sha256"],
        "freeze_source_sha256": freeze["read_audit"]["source_sha256"],
        "status": "consumed_before_outcome_access",
    }
    if _canonicalize(authority) != _canonicalize(expected):
        raise FinalEvaluationError("consumed experiment authority content mismatch")
    if manifest.get("freeze_id") != freeze.get("freeze_id"):
        raise FinalEvaluationError("completion manifest freeze id mismatch")


def _verify_completed_external_outcome_receipt(
    consumed: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    freeze_sha: str,
    trusted_outcome_content: Mapping[str, Any],
) -> None:
    experiment_id = _validate_experiment_id(consumed.get("experiment_id"))
    path = _trusted_outcome_path(experiment_id, fixture_mode=bool(freeze["fixture_mode"]))
    if not path.exists() or not path.is_file():
        raise FinalEvaluationError("external trusted outcome receipt missing")
    external = _read_json(path)
    expected = {
        "schema": "m3_final_trusted_outcome_content_receipt_v1",
        "experiment_id": experiment_id,
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": freeze_sha,
        "source_sha256": freeze["read_audit"]["source_sha256"],
        "test_keys": freeze["test_keys"],
        "trusted_outcome_content": _canonicalize(trusted_outcome_content),
        "trusted_outcome_content_sha256": _sha256_obj(trusted_outcome_content),
        "status": "outcome_content_persisted_before_fit",
    }
    if _canonicalize(external) != _canonicalize(expected):
        raise FinalEvaluationError("external trusted outcome receipt content mismatch")
    local = manifest.get("trusted_outcome_receipt")
    if not isinstance(local, Mapping):
        raise FinalEvaluationError("completion manifest lacks external outcome receipt copy")
    if (
        local.get("experiment_id") != experiment_id
        or local.get("outcome_sha256") != sha256_file(path)
    ):
        raise FinalEvaluationError("completion manifest outcome receipt mismatch")


def _registry_records(registry: pd.DataFrame) -> list[dict[str, Any]]:
    return [_canonicalize(record) for record in registry.to_dict(orient="records")]


def _verify_completed_fit_receipts(
    row: Any,
    *,
    pending: Mapping[str, Any],
    complete: Mapping[str, Any],
    fit_freeze: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> None:
    fit_id = str(row.fit_id)
    if pending.get("status") != "predictions_saved_pending_metrics":
        raise FinalEvaluationError(f"{fit_id} pending receipt status mismatch")
    if complete.get("status") != "complete":
        raise FinalEvaluationError(f"{fit_id} complete receipt status mismatch")
    row_record = _canonicalize(row._asdict())
    complete_record = _canonicalize(dict(complete))
    if complete_record != row_record:
        raise FinalEvaluationError(f"{fit_id} complete receipt does not match registry")
    for key in (
        "horizon",
        "feature_set",
        "model_family",
        "fit_id",
        "training_count",
        "evaluation_count",
        "training_key_digest",
        "evaluation_key_digest",
        "training_feature_content_digest",
        "training_target_content_digest",
        "threshold_path",
        "threshold_sha256",
        "threshold_binding_sha256",
        "model_path",
        "model_sha256",
        "model_metadata_sha256",
        "prediction_path",
        "prediction_sha256",
        "prediction_values_sha256",
        "trusted_outcome_content_sha256",
        "metrics_path",
        "model_config_sha256",
        "feature_list_sha256",
    ):
        if _canonicalize(pending.get(key)) != _canonicalize(getattr(row, key)):
            raise FinalEvaluationError(f"{fit_id} pending receipt {key} mismatch")
    if int(row.training_count) != int(fit_freeze["training_keys"]["row_count"]):
        raise FinalEvaluationError(f"{fit_id} training count mismatch")
    if int(row.evaluation_count) != int(freeze["test_keys"]["row_count"]):
        raise FinalEvaluationError(f"{fit_id} evaluation count mismatch")
    expected_bindings = {
        "training_key_digest": fit_freeze["training_keys"]["row_key_digest"],
        "evaluation_key_digest": freeze["test_keys"]["row_key_digest"],
        "training_feature_content_digest": fit_freeze["training_digests"][
            "feature_content_digest"
        ],
        "training_target_content_digest": fit_freeze["training_digests"][
            "target_content_digest"
        ],
        "threshold_binding_sha256": fit_freeze["threshold_binding_sha256"],
        "model_config_sha256": fit_freeze["model_config_sha256"],
        "feature_list_sha256": fit_freeze["feature_list_sha256"],
    }
    for key, expected in expected_bindings.items():
        if getattr(row, key) != expected:
            raise FinalEvaluationError(f"{fit_id} {key} does not match freeze")


def _verify_completed_fit_claim(row: Any, claim: Mapping[str, Any]) -> None:
    fit_id = str(row.fit_id)
    expected = {
        "horizon": int(row.horizon),
        "feature_set": str(row.feature_set),
        "model_family": str(row.model_family),
        "fit_id": fit_id,
    }
    if claim.get("status") != "claimed":
        raise FinalEvaluationError(f"{fit_id} fit claim status mismatch")
    if _canonicalize(claim.get("fit")) != _canonicalize(expected):
        raise FinalEvaluationError(f"{fit_id} fit claim identity mismatch")


def _verify_completed_model_metadata(
    row: Any,
    *,
    metadata: Mapping[str, Any],
    fit_freeze: Mapping[str, Any],
) -> None:
    fit_id = str(row.fit_id)
    if metadata.get("target_column") != f"target_h{int(row.horizon)}":
        raise FinalEvaluationError(f"{fit_id} model metadata target mismatch")
    if metadata.get("feature_columns") != fit_freeze["feature_columns"]:
        raise FinalEvaluationError(f"{fit_id} model metadata feature mismatch")
    if _canonicalize(metadata.get("config")) != _canonicalize(fit_freeze["model_config"]):
        raise FinalEvaluationError(f"{fit_id} model metadata config mismatch")
    state = metadata.get("preprocessing_fit_state")
    if not isinstance(state, Mapping):
        raise FinalEvaluationError(f"{fit_id} model metadata preprocessing state missing")
    expected_state = fit_freeze.get("expected_preprocessing_fit_state")
    if not isinstance(expected_state, Mapping):
        raise FinalEvaluationError(f"{fit_id} expected preprocessing fit state missing")
    if _canonicalize(state) != _canonicalize(expected_state):
        raise FinalEvaluationError(f"{fit_id} preprocessing fit state mismatch")
    provenance = metadata.get("row_identity")
    if not isinstance(provenance, Mapping):
        raise FinalEvaluationError(f"{fit_id} model metadata provenance missing")
    expected = {
        "m3_fit_id": fit_id,
        "m3_horizon": int(row.horizon),
        "m3_feature_set": str(row.feature_set),
        "m3_stage": "final_evaluation",
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise FinalEvaluationError(f"{fit_id} model metadata provenance mismatch")


def _verify_completed_predictions(
    row: Any,
    predictions: pd.DataFrame,
    *,
    fit_freeze: Mapping[str, Any],
    freeze: Mapping[str, Any],
    trusted_outcome_content: Mapping[str, Any],
) -> None:
    fit_id = str(row.fit_id)
    if str(row.prediction_values_sha256) != _sha256_obj(
        development._canonical_frame_records(predictions)
    ):
        raise FinalEvaluationError(f"{fit_id} prediction value digest mismatch")
    expected_outcome_digest = _outcome_digest_for_horizon(
        trusted_outcome_content,
        int(row.horizon),
    )
    if row.trusted_outcome_content_sha256 != expected_outcome_digest:
        raise FinalEvaluationError(f"{fit_id} trusted outcome content receipt mismatch")
    _verify_prediction_vector_semantics(
        row,
        predictions,
        fit_freeze=fit_freeze,
        freeze=freeze,
        expected_outcome_digest=expected_outcome_digest,
        stage="",
    )


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key)


def _verify_prediction_vector_semantics(
    row: Any,
    predictions: pd.DataFrame,
    *,
    fit_freeze: Mapping[str, Any],
    freeze: Mapping[str, Any],
    expected_outcome_digest: str,
    stage: str,
) -> None:
    fit_id = str(_row_value(row, "fit_id"))
    suffix = f" {stage}" if stage else ""
    required = {
        "fit_id",
        "horizon",
        "model_family",
        "feature_set",
        "district_id",
        "origin_start",
        "origin_end",
        "target_start",
        "target_end",
        "observed_current_cases",
        "observed_target",
        "prediction_model",
        "prediction_persistence",
        "training_binding_id",
        "evaluation_binding_id",
        "threshold_binding_id",
    }
    if required - set(predictions.columns):
        raise FinalEvaluationError(f"{fit_id} prediction columns missing{suffix}")
    if int(len(predictions)) != int(_row_value(row, "evaluation_count")):
        raise FinalEvaluationError(f"{fit_id} prediction count mismatch{suffix}")
    horizon = int(_row_value(row, "horizon"))
    if _prediction_outcome_digest(predictions, horizon) != expected_outcome_digest:
        raise FinalEvaluationError(f"{fit_id} prediction trusted outcome content mismatch{suffix}")
    if set(predictions["fit_id"].astype(str)) != {fit_id}:
        raise FinalEvaluationError(f"{fit_id} prediction fit identity mismatch{suffix}")
    if set(predictions["horizon"].astype(int)) != {horizon}:
        raise FinalEvaluationError(f"{fit_id} prediction horizon mismatch{suffix}")
    if set(predictions["model_family"].astype(str)) != {str(_row_value(row, "model_family"))}:
        raise FinalEvaluationError(f"{fit_id} prediction model family mismatch{suffix}")
    if set(predictions["feature_set"].astype(str)) != {str(_row_value(row, "feature_set"))}:
        raise FinalEvaluationError(f"{fit_id} prediction feature set mismatch{suffix}")
    if predictions.duplicated(["fit_id", "district_id", "origin_start"]).any():
        raise FinalEvaluationError(f"{fit_id} prediction keys are duplicated{suffix}")
    keys = predictions[["district_id", "origin_start"]].rename(
        columns={"origin_start": "week_start_date"}
    )
    core.verify_common_keys(keys, freeze["test_keys"])
    origin_start = pd.to_datetime(predictions["origin_start"], errors="coerce")
    origin_end = pd.to_datetime(predictions["origin_end"], errors="coerce")
    target_start = pd.to_datetime(predictions["target_start"], errors="coerce")
    target_end = pd.to_datetime(predictions["target_end"], errors="coerce")
    if (
        origin_start.isna().any()
        or origin_end.isna().any()
        or target_start.isna().any()
        or target_end.isna().any()
    ):
        raise FinalEvaluationError(f"{fit_id} prediction dates invalid{suffix}")
    if origin_end.ne(origin_start + pd.Timedelta(days=6)).any():
        raise FinalEvaluationError(f"{fit_id} prediction origin calendar mismatch{suffix}")
    if target_start.ne(origin_start + pd.to_timedelta(7 * horizon, unit="D")).any():
        raise FinalEvaluationError(f"{fit_id} prediction target start mismatch{suffix}")
    if target_end.ne(origin_start + pd.to_timedelta(7 * horizon + 6, unit="D")).any():
        raise FinalEvaluationError(f"{fit_id} prediction target end mismatch{suffix}")
    for column in (
        "observed_current_cases",
        "observed_target",
        "prediction_model",
        "prediction_persistence",
    ):
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values < 0).any():
            raise FinalEvaluationError(
                f"{fit_id} prediction {column} contains invalid values{suffix}"
            )
    current = pd.to_numeric(
        predictions["observed_current_cases"],
        errors="coerce",
    ).to_numpy(dtype="float64")
    persistence = pd.to_numeric(
        predictions["prediction_persistence"],
        errors="coerce",
    ).to_numpy(dtype="float64")
    if not np.array_equal(current, persistence):
        raise FinalEvaluationError(f"{fit_id} prediction persistence mismatch{suffix}")
    if set(predictions["training_binding_id"].astype(str)) != {
        fit_freeze["thresholds"]["digests"]["row_key_digest"]
    }:
        raise FinalEvaluationError(f"{fit_id} prediction training binding mismatch{suffix}")
    if set(predictions["evaluation_binding_id"].astype(str)) != {
        freeze["test_keys"]["row_key_digest"]
    }:
        raise FinalEvaluationError(f"{fit_id} prediction evaluation binding mismatch{suffix}")
    if set(predictions["threshold_binding_id"].astype(str)) != {
        fit_freeze["threshold_binding_sha256"]
    }:
        raise FinalEvaluationError(f"{fit_id} prediction threshold binding mismatch{suffix}")


def _verify_registry_metrics(row: Any, recomputed: Mapping[str, Any]) -> None:
    fit_id = str(row.fit_id)
    expected = {
        "mae_model": recomputed["model"]["mae"]["value"],
        "rmse_model": recomputed["model"]["rmse"]["value"],
        "mae_persistence": recomputed["persistence"]["mae"]["value"],
        "rmse_persistence": recomputed["persistence"]["rmse"]["value"],
    }
    for key, value in expected.items():
        observed = getattr(row, key)
        if value is None:
            if not pd.isna(observed):
                raise FinalEvaluationError(f"{fit_id} registry {key} mismatch")
            continue
        observed_float = float(observed)
        if not np.isfinite(observed_float) or observed_float < 0:
            raise FinalEvaluationError(f"{fit_id} registry {key} invalid")
        if not np.isclose(observed_float, float(value), rtol=0.0, atol=1e-12):
            raise FinalEvaluationError(f"{fit_id} registry {key} mismatch")


def _metadata_availability_digest(
    *,
    first_origin: pd.Timestamp,
    expected_keys: Mapping[str, Any],
) -> str:
    return _sha256_obj(
        {
            "first_scheduled_origin": first_origin.date().isoformat(),
            "expected_keys": expected_keys,
        }
    )


def _model_config(protocol: Mapping[str, Any], family: str) -> ModelConfig:
    return development._model_config(protocol, family)


def _expected_preprocessing_fit_state(
    training: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_family: str,
) -> dict[str, Any]:
    preprocessor = preprocessing.FoldPreprocessor(model_family=model_family).fit(
        training,
        feature_columns=feature_columns,
    )
    return dict(preprocessor.fit_state)


def _code_hashes() -> dict[str, str]:
    return {
        "milestone3/final_evaluation.py": sha256_file(Path(__file__)),
        "milestone3/core.py": sha256_file(Path(core.__file__)),
        "milestone3/development.py": sha256_file(Path(development.__file__)),
        "modeling/metrics.py": sha256_file(Path(metrics.__file__)),
        "modeling/preprocessing.py": sha256_file(Path(preprocessing.__file__)),
        "modeling/train.py": sha256_file(Path(modeling_train.__file__)),
    }


def _verify_code_hashes(expected: Mapping[str, str]) -> None:
    observed = _code_hashes()
    if dict(expected) != observed:
        raise FinalEvaluationError("code hash mismatch")


def _environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "joblib"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "dependencies": versions}


def _claim_dir(
    output_root: str | Path,
    name: str,
    *,
    fixture_mode: bool,
    kind: str,
) -> Path:
    if not name or any(part in name for part in ("/", "\\", "..")):
        raise FinalEvaluationError(f"{kind} id must be a simple path segment")
    root_input = Path(output_root)
    development._reject_symlink_components(root_input)
    root = root_input.resolve()
    allowed = (
        tuple(path.resolve() for path in FIXTURE_OUTPUT_ROOTS)
        if fixture_mode
        else (PRODUCTION_OUTPUT_ROOT.resolve(),)
    )
    if not any(development._is_relative_to(root, candidate) for candidate in allowed):
        raise FinalEvaluationError(f"final {kind} output root is outside allowed roots")
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    try:
        target.mkdir(mode=0o755, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise FinalEvaluationError(
            f"final {kind} output already exists; rerun/resume is prohibited"
        ) from exc
    if target.is_symlink():
        raise FinalEvaluationError(f"final {kind} output cannot be a symlink")
    return target


def _set_fixture_trusted_registry_root(registry_root: str | Path) -> None:
    global _FIXTURE_TRUSTED_REGISTRY_ROOT
    root_input = Path(registry_root)
    development._reject_symlink_components(root_input)
    root = root_input.resolve()
    allowed = tuple(path.resolve() for path in FIXTURE_OUTPUT_ROOTS)
    if not any(development._is_relative_to(root, candidate) for candidate in allowed):
        raise FinalEvaluationError("synthetic trusted registry root is outside allowed roots")
    _FIXTURE_TRUSTED_REGISTRY_ROOT = root


def _trusted_registry_root(*, fixture_mode: bool) -> Path:
    if fixture_mode and _FIXTURE_TRUSTED_REGISTRY_ROOT is not None:
        root = _FIXTURE_TRUSTED_REGISTRY_ROOT
    else:
        root = TRUSTED_REGISTRY_ROOT
    development._reject_symlink_components(root)
    resolved = root.resolve()
    if fixture_mode:
        allowed = tuple(path.resolve() for path in FIXTURE_OUTPUT_ROOTS)
        if not any(development._is_relative_to(resolved, candidate) for candidate in allowed):
            raise FinalEvaluationError("synthetic trusted registry root is outside allowed roots")
    elif resolved != TRUSTED_REGISTRY_ROOT.resolve():
        raise FinalEvaluationError("production trusted registry root is not fixed")
    return resolved


def _validate_experiment_id(experiment_id: Any) -> str:
    if not isinstance(experiment_id, str) or not experiment_id:
        raise FinalEvaluationError("outcome gate requires canonical experiment_id")
    if any(part in experiment_id for part in ("/", "\\", "..")):
        raise FinalEvaluationError("canonical experiment_id must be a simple path segment")
    return experiment_id


def _trusted_experiment_dir(experiment_id: str, *, fixture_mode: bool) -> Path:
    experiment_id = _validate_experiment_id(experiment_id)
    return _trusted_registry_root(fixture_mode=fixture_mode) / "experiments" / experiment_id


def _trusted_setup_path(experiment_id: str, *, fixture_mode: bool) -> Path:
    return _trusted_experiment_dir(experiment_id, fixture_mode=fixture_mode) / "setup.json"


def _trusted_consumed_path(experiment_id: str, *, fixture_mode: bool) -> Path:
    return _trusted_experiment_dir(experiment_id, fixture_mode=fixture_mode) / "consumed.json"


def _trusted_outcome_path(experiment_id: str, *, fixture_mode: bool) -> Path:
    return (
        _trusted_experiment_dir(experiment_id, fixture_mode=fixture_mode)
        / "outcome_content.json"
    )


def _trusted_setup_record(
    gate_or_claim: Mapping[str, Any],
    *,
    fixture_mode: bool,
) -> dict[str, Any]:
    experiment_id = _validate_experiment_id(gate_or_claim.get("experiment_id"))
    path = _trusted_setup_path(experiment_id, fixture_mode=fixture_mode)
    if not path.exists():
        raise FinalEvaluationError("trusted experiment setup record is missing")
    record = _read_json(path)
    if record.get("schema") != "m3_final_trusted_setup_v1":
        raise FinalEvaluationError("trusted experiment setup schema mismatch")
    if record.get("experiment_id") != experiment_id:
        raise FinalEvaluationError("trusted experiment setup id mismatch")
    if record.get("status") != "trusted_setup_provisioned":
        raise FinalEvaluationError("trusted experiment setup status mismatch")
    return record


def _verify_trusted_setup_source(
    setup_record: Mapping[str, Any],
    source: development.SourceApproval,
) -> None:
    if setup_record.get("authorized_source_sha256") != source.approved_sha256:
        raise FinalEvaluationError("trusted registry source identity mismatch")


def _claim_experiment_authority(
    gate: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    freeze_sha: str,
    source: development.SourceApproval,
    trusted_setup: Mapping[str, Any],
    fixture_mode: bool,
) -> dict[str, Any]:
    experiment_id = _validate_experiment_id(gate.get("experiment_id"))
    _verify_trusted_setup_source(trusted_setup, source)
    claim_path = _trusted_consumed_path(experiment_id, fixture_mode=fixture_mode)
    claim = {
        "schema": "m3_final_consumed_experiment_claim_v1",
        "experiment_id": experiment_id,
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": freeze_sha,
        "authorized_source_sha256": trusted_setup["authorized_source_sha256"],
        "freeze_source_sha256": freeze["read_audit"]["source_sha256"],
        "status": "consumed_before_outcome_access",
    }
    try:
        _write_json_new(claim_path, claim)
    except FileExistsError as exc:
        raise FinalEvaluationError("final experiment authorization already consumed") from exc
    return {
        "experiment_id": experiment_id,
        "claim_sha256": sha256_file(claim_path),
    }


def _persist_trusted_outcome_content(
    gate: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    freeze_sha: str,
    source: development.SourceApproval,
    trusted_outcome_content: Mapping[str, Any],
    fixture_mode: bool,
) -> dict[str, Any]:
    experiment_id = _validate_experiment_id(gate.get("experiment_id"))
    if trusted_outcome_content.get("source_sha256") != source.approved_sha256:
        raise FinalEvaluationError("trusted outcome source identity mismatch")
    path = _trusted_outcome_path(experiment_id, fixture_mode=fixture_mode)
    receipt = {
        "schema": "m3_final_trusted_outcome_content_receipt_v1",
        "experiment_id": experiment_id,
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": freeze_sha,
        "source_sha256": source.approved_sha256,
        "test_keys": freeze["test_keys"],
        "trusted_outcome_content": _canonicalize(trusted_outcome_content),
        "trusted_outcome_content_sha256": _sha256_obj(trusted_outcome_content),
        "status": "outcome_content_persisted_before_fit",
    }
    _write_json_new(path, receipt)
    return {"experiment_id": experiment_id, "outcome_sha256": sha256_file(path)}


def _coerce_midnight(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.normalize() != parsed:
        raise FinalEvaluationError(f"{label} must be a midnight calendar date")
    return parsed


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FinalEvaluationError(f"required json artifact is missing: {target.name}")
    return json.loads(target.read_text(encoding="utf-8"))


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
        raise FinalEvaluationError(f"refusing to overwrite existing artifact: {path}")
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


def _checksums(root: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in files
        if path.name != "checksums.json"
    }


def _sha256_obj(value: Any) -> str:
    return development._sha256_obj(value)


def _canonicalize(value: Any) -> Any:
    return development._canonicalize(value)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
