from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dengue_forecast.milestone3 import core, development
from dengue_forecast.modeling import metrics

DEFAULT_DEVELOPMENT_RUN = (
    core.REPO_ROOT / "artifacts" / "milestone3" / "development" / "approved-development-002"
)
DEFAULT_ANALYSIS_RUN = (
    core.REPO_ROOT / "artifacts" / "milestone3" / "analysis" / "saved-predictions-dev-corrected-001"
)
DEFAULT_OUTPUT_ROOT = core.REPO_ROOT / "artifacts" / "milestone3" / "diagnostics"
DEFAULT_RESULT_DOC = core.REPO_ROOT / "docs" / "m3-secondary-diagnostics-result.md"

FOLDS = development.FOLDS
HORIZONS = development.HORIZONS
FEATURE_SETS = development.FEATURE_SET_ORDER
MODEL_FAMILIES = development.MODEL_FAMILIES
REQUIRED_FOLD_COUNT = len(FOLDS)

SECONDARY_METRICS = (
    "r2",
    "poisson_deviance",
    "bias",
    "mae_top_10pct",
    "mae_top_5pct",
)
CONFIG_AGGREGATE_METRICS = (
    "mae",
    "rmse",
    "r2",
    "poisson_deviance",
    "bias",
    "mae_top_10pct",
)
CHANGE_CATEGORIES = (
    "stable",
    "directional_up",
    "large_up",
    "directional_down",
    "large_down",
)
INCIDENCE_GROUPS = (
    ("q90_primary", "q90_incidence"),
    ("q95_supplementary", "q95_incidence"),
)

POLICY_TEXT = """# Milestone 3 secondary diagnostics aggregation policy

This run is a saved-development diagnostic supplement only. It uses authenticated
saved predictions, saved fit metric documents, and saved training threshold
receipts from `approved-development-002`, plus the corrected analysis tables in
`saved-predictions-dev-corrected-001`.

Fold-level values are explicit. Config-level averages equally weight the six
predeclared folds: 2014, 2015, 2016, 2018, 2019, and 2020. If any required fold
is missing, or a metric is undefined or a required subset is empty in any fold,
the sixfold aggregate is reported as null with a reason and valid-fold count.
Available-fold-only averaging is not used.

Conditional high-incidence and change-category tables retain all required
folds/categories, including empty cells and zero counts. High-incidence
membership uses saved training q90/q95 thresholds with inclusive boundaries.
Change categories use saved training cutoffs with stability precedence.

District tables are pooled descriptive summaries across saved development rows,
not equal-fold comparisons. Percentage MAE improvement is null when the
persistence denominator is zero. No uncertainty intervals or significance claims
are produced here. The 2025 partial-year 24-week cohort remains locked and is not
an annual evaluation.
"""


class DiagnosticsError(ValueError):
    """Raised when secondary diagnostics cannot be produced faithfully."""


@dataclass(frozen=True)
class DiagnosticsResult:
    output_dir: Path
    table_paths: dict[str, Path]
    manifest_path: Path
    policy_path: Path
    result_doc_path: Path


def compute_metric_values(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    truth = frame["observed_target"].astype(float)
    model = metrics.clip_predictions_nonnegative(frame["prediction_model"].astype(float))
    persistence = metrics.clip_predictions_nonnegative(
        frame["prediction_persistence"].astype(float)
    )
    return {
        "model": _metric_values_for(truth, model),
        "persistence": _metric_values_for(truth, persistence),
    }


def categorize_change_deltas(
    deltas: pd.Series,
    *,
    stable_abs_delta_q25: float,
    large_up_q90: float | None,
    large_down_abs_q90: float | None,
) -> pd.Series:
    stable_cutoff = float(stable_abs_delta_q25)
    out = pd.Series("uncategorized", index=deltas.index, dtype="object")
    stable = deltas.abs() <= stable_cutoff
    up = deltas > stable_cutoff
    down = deltas < -stable_cutoff
    out[stable] = "stable"
    out[up] = "directional_up"
    out[down] = "directional_down"
    if large_up_q90 is not None and np.isfinite(float(large_up_q90)):
        out[up & (deltas >= float(large_up_q90))] = "large_up"
    if large_down_abs_q90 is not None and np.isfinite(float(large_down_abs_q90)):
        out[down & (-deltas >= float(large_down_abs_q90))] = "large_down"
    return out


def high_incidence_error_table(predictions: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keyed = predictions.groupby(["horizon", "fold", "feature_set", "model_family"], sort=True)
    for threshold in thresholds.sort_values(
        ["horizon", "feature_set", "model_family", "fold"]
    ).itertuples(index=False):
        key = (
            int(threshold.horizon),
            int(threshold.fold),
            str(threshold.feature_set),
            str(threshold.model_family),
        )
        if key not in keyed.groups:
            raise DiagnosticsError(f"missing predictions for threshold key: {key}")
        group = keyed.get_group(key)
        for label, threshold_column in INCIDENCE_GROUPS:
            cutoff = float(getattr(threshold, threshold_column))
            subset = group[group["observed_target"].astype(float) >= cutoff]
            rows.append(
                _error_row(
                    threshold,
                    subset,
                    {
                        "incidence_group": label,
                        "threshold": cutoff,
                        "boundary_rule": f"observed_target >= {threshold_column}",
                    },
                )
            )
    return pd.DataFrame(rows)


def change_category_error_table(
    predictions: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keyed = predictions.groupby(["horizon", "fold", "feature_set", "model_family"], sort=True)
    for threshold in thresholds.sort_values(
        ["horizon", "feature_set", "model_family", "fold"]
    ).itertuples(index=False):
        key = (
            int(threshold.horizon),
            int(threshold.fold),
            str(threshold.feature_set),
            str(threshold.model_family),
        )
        if key not in keyed.groups:
            raise DiagnosticsError(f"missing predictions for threshold key: {key}")
        group = keyed.get_group(key).copy()
        delta = group["observed_target"].astype(float) - group["observed_current_cases"].astype(
            float
        )
        group["category"] = categorize_change_deltas(
            delta,
            stable_abs_delta_q25=float(threshold.stable_abs_delta_q25),
            large_up_q90=_none_if_nan(threshold.large_up_q90),
            large_down_abs_q90=_none_if_nan(threshold.large_down_abs_q90),
        )
        large_up_q90 = _none_if_nan(threshold.large_up_q90)
        large_down_abs_q90 = _none_if_nan(threshold.large_down_abs_q90)
        null_groups = []
        if large_up_q90 is None:
            null_groups.append("large_up")
        if large_down_abs_q90 is None:
            null_groups.append("large_down")
        for category in CHANGE_CATEGORIES:
            subset = group[group["category"].eq(category)]
            rows.append(
                _error_row(
                    threshold,
                    subset,
                    {
                        "category": category,
                        "stable_abs_delta_q25": float(threshold.stable_abs_delta_q25),
                        "large_up_q90": large_up_q90,
                        "large_down_abs_q90": large_down_abs_q90,
                        "null_threshold_groups": ",".join(null_groups) or None,
                        "category_source": (
                            "observed evaluation delta categorized by saved training cutoffs "
                            "with stability precedence"
                        ),
                    },
                )
            )
    return pd.DataFrame(rows)


def district_improvement_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(
        ["horizon", "feature_set", "model_family", "district_id"], sort=True
    )
    for (horizon, feature_set, model_family, district_id), group in grouped:
        truth = group["observed_target"].astype(float)
        model_abs = (metrics.clip_predictions_nonnegative(group["prediction_model"]) - truth).abs()
        persistence_abs = (
            metrics.clip_predictions_nonnegative(group["prediction_persistence"]) - truth
        ).abs()
        model_mae = float(model_abs.mean())
        persistence_mae = float(persistence_abs.mean())
        improvement = persistence_mae - model_mae
        pct = None if persistence_mae == 0.0 else 100.0 * improvement / persistence_mae
        rows.append(
            {
                "horizon": int(horizon),
                "feature_set": str(feature_set),
                "model_family": str(model_family),
                "district_id": str(district_id),
                "row_count": int(len(group)),
                "observed_target_total": float(truth.sum()),
                "observed_target_mean": float(truth.mean()),
                "model_mae": model_mae,
                "persistence_mae": persistence_mae,
                "mae_improvement_absolute": improvement,
                "mae_improvement_pct": pct,
                "mae_improvement_pct_null_reason": (
                    "zero_persistence_mae_denominator" if pct is None else None
                ),
                "case_volume_caveat": (
                    "raw district MAE and absolute improvement reflect case volume; compare "
                    "with observed_target_total/mean"
                ),
                "semantics": "pooled_descriptive_district_table_not_equalfold_comparison",
            }
        )
    return pd.DataFrame(rows)


def saved_metric_fold_table(source_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in registry.sort_values(
        ["horizon", "feature_set", "model_family", "fold"]
    ).itertuples():
        metrics_path = source_dir / str(row.metrics_path)
        expected = str(row.metrics_sha256)
        actual = sha256_file(metrics_path)
        if actual != expected:
            raise DiagnosticsError(f"metrics hash mismatch for {row.fit_id}")
        payload = _read_json(metrics_path)
        record: dict[str, Any] = {
            "horizon": int(row.horizon),
            "feature_set": str(row.feature_set),
            "model_family": str(row.model_family),
            "fold": int(row.fold),
            "fit_id": str(row.fit_id),
        }
        for family in ("model", "persistence"):
            for metric_name in SECONDARY_METRICS:
                value, reason = _metric_doc_value(payload[family][metric_name])
                record[f"{metric_name}_{family}"] = value
                record[f"{metric_name}_{family}_null_reason"] = reason
            for count_name in ("top_10_count", "top_5_count"):
                if count_name in payload[family]:
                    value, reason = _metric_doc_value(payload[family][count_name])
                    record[f"{count_name}_{family}"] = value
                    record[f"{count_name}_{family}_null_reason"] = reason
        rows.append(record)
    return pd.DataFrame(rows)


def aggregate_config_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["horizon", "feature_set", "model_family"]
    for keys, group in fold_metrics.groupby(group_cols, sort=True, dropna=False):
        horizon, feature_set, model_family = keys
        row: dict[str, Any] = {
            "horizon": int(horizon),
            "feature_set": str(feature_set),
            "model_family": str(model_family),
            "fold_count": int(group["fold"].nunique()),
            "required_fold_count": REQUIRED_FOLD_COUNT,
            "aggregation": "unweighted_mean_of_six_predeclared_fold_metrics",
        }
        for metric_name in CONFIG_AGGREGATE_METRICS:
            for family in ("model", "persistence"):
                column = f"{metric_name}_{family}"
                if column not in group.columns:
                    continue
                value, reason, valid_count = _sixfold_mean_or_null(group, column)
                out = f"mean_fold_{metric_name}_{family}"
                row[out] = value
                row[f"{out}_null_reason"] = reason
                row[f"{out}_valid_fold_count"] = valid_count
        rows.append(row)
    return pd.DataFrame(rows)


def run_diagnostics(
    *,
    development_run: str | Path = DEFAULT_DEVELOPMENT_RUN,
    analysis_run: str | Path = DEFAULT_ANALYSIS_RUN,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str,
    result_doc: str | Path = DEFAULT_RESULT_DOC,
    authenticate: bool = True,
) -> DiagnosticsResult:
    source_dir = Path(development_run).resolve()
    analysis_dir = Path(analysis_run).resolve()
    output_dir = Path(output_root).resolve() / run_id
    if output_dir.exists():
        raise DiagnosticsError(f"diagnostics output already exists: {output_dir}")
    if _is_relative_to(output_dir, source_dir) or _is_relative_to(output_dir, analysis_dir):
        raise DiagnosticsError("diagnostics output must not be nested inside consumed artifacts")
    output_dir.mkdir(parents=True, exist_ok=False)
    policy_path = output_dir / "aggregation_policy.md"
    policy_path.write_text(POLICY_TEXT, encoding="utf-8")
    try:
        result = _run_diagnostics_inner(
            source_dir=source_dir,
            analysis_dir=analysis_dir,
            output_dir=output_dir,
            policy_path=policy_path,
            result_doc=Path(result_doc).resolve(),
            authenticate=authenticate,
        )
    except Exception:
        (output_dir / "failed.json").write_text(
            json.dumps({"status": "failed"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    return result


def _run_diagnostics_inner(
    *,
    source_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    policy_path: Path,
    result_doc: Path,
    authenticate: bool,
) -> DiagnosticsResult:
    validation = (
        development.validate_development_run(source_dir)
        if authenticate
        else {"status": "not_authenticated_test_fixture", "run_dir": str(source_dir)}
    )
    registry = pd.read_parquet(source_dir / "model_registry.parquet")
    _verify_development_hashes(source_dir, registry)
    analysis_manifest = _verify_analysis_manifest(analysis_dir)
    thresholds = load_thresholds(source_dir, registry)
    predictions = load_predictions(source_dir, registry)

    saved_fold_secondary = saved_metric_fold_table(source_dir, registry)
    recomputed_fold_secondary = recomputed_fold_metric_table(predictions)
    fold_secondary = _merge_saved_and_recomputed(saved_fold_secondary, recomputed_fold_secondary)
    existing_fold = _existing_fold_metrics(analysis_dir)
    fold_secondary = fold_secondary.merge(
        existing_fold,
        on=["horizon", "feature_set", "model_family", "fold", "fit_id"],
        how="left",
        validate="one_to_one",
    )
    config_secondary = aggregate_config_metrics(fold_secondary)
    high_incidence = high_incidence_error_table(predictions, thresholds)
    change_categories = change_category_error_table(predictions, thresholds)
    district = district_improvement_table(predictions)
    consistency = consistency_checks(
        source_dir=source_dir,
        analysis_dir=analysis_dir,
        analysis_manifest=analysis_manifest,
        registry=registry,
        predictions=predictions,
        fold_secondary=fold_secondary,
        high_incidence=high_incidence,
    )
    _assert_consistency_passes(consistency)

    tables = {
        "secondary_fold_metrics": fold_secondary,
        "secondary_config_metrics": config_secondary,
        "high_incidence_conditional_errors": high_incidence,
        "change_category_errors": change_categories,
        "district_improvements": district,
        "consistency_checks": consistency,
    }
    table_paths: dict[str, Path] = {}
    for name, table in tables.items():
        csv_path = output_dir / f"{name}.csv"
        json_path = output_dir / f"{name}.json"
        _write_csv(table, csv_path)
        _write_json(json_path, _json_records(table))
        table_paths[name] = csv_path
        table_paths[f"{name}_json"] = json_path

    manifest = _manifest(
        source_dir=source_dir,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        policy_path=policy_path,
        validation=validation,
        analysis_manifest=analysis_manifest,
        table_paths=table_paths,
    )
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    result_doc.write_text(
        render_result_doc(manifest, table_paths, config_secondary), encoding="utf-8"
    )
    return DiagnosticsResult(
        output_dir=output_dir,
        table_paths=table_paths,
        manifest_path=manifest_path,
        policy_path=policy_path,
        result_doc_path=result_doc,
    )


def load_predictions(source_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in registry.sort_values(
        ["horizon", "feature_set", "model_family", "fold"]
    ).itertuples():
        path = source_dir / str(row.prediction_path)
        if sha256_file(path) != str(row.prediction_sha256):
            raise DiagnosticsError(f"prediction hash mismatch for {row.fit_id}")
        frame = pd.read_parquet(path)
        for column, expected in {
            "fit_id": str(row.fit_id),
            "fold": int(row.fold),
            "horizon": int(row.horizon),
            "feature_set": str(row.feature_set),
            "model_family": str(row.model_family),
        }.items():
            actual = set(frame[column].astype(type(expected)))
            if actual != {expected}:
                raise DiagnosticsError(f"prediction identity mismatch for {row.fit_id}: {column}")
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["origin_start"] = pd.to_datetime(predictions["origin_start"])
    return predictions


def load_thresholds(source_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in registry.sort_values(
        ["horizon", "feature_set", "model_family", "fold"]
    ).itertuples():
        path = source_dir / str(row.threshold_path)
        if sha256_file(path) != str(row.threshold_sha256):
            raise DiagnosticsError(f"threshold hash mismatch for {row.fit_id}")
        payload = _read_json(path)
        binding = _sha256_obj(payload)
        if binding != str(row.threshold_binding_sha256):
            raise DiagnosticsError(f"threshold binding hash mismatch for {row.fit_id}")
        rows.append(
            {
                "horizon": int(row.horizon),
                "fold": int(row.fold),
                "feature_set": str(row.feature_set),
                "model_family": str(row.model_family),
                "fit_id": str(row.fit_id),
                "q90_incidence": float(payload["q90_incidence"]),
                "q95_incidence": float(payload["q95_incidence"]),
                "stable_abs_delta_q25": float(payload["stable_abs_delta_q25"]),
                "large_up_q90": _none_if_nan(payload.get("large_up_q90")),
                "large_down_abs_q90": _none_if_nan(payload.get("large_down_abs_q90")),
                "threshold_binding_sha256": str(row.threshold_binding_sha256),
            }
        )
    return pd.DataFrame(rows)


def recomputed_fold_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(
        ["horizon", "feature_set", "model_family", "fold", "fit_id"], sort=True
    )
    for (horizon, feature_set, model_family, fold, fit_id), group in grouped:
        computed = compute_metric_values(group)
        row: dict[str, Any] = {
            "horizon": int(horizon),
            "feature_set": str(feature_set),
            "model_family": str(model_family),
            "fold": int(fold),
            "fit_id": str(fit_id),
        }
        for family in ("model", "persistence"):
            for metric_name, value in computed[family].items():
                row[f"recomputed_{metric_name}_{family}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def consistency_checks(
    *,
    source_dir: Path,
    analysis_dir: Path,
    analysis_manifest: dict[str, Any],
    registry: pd.DataFrame,
    predictions: pd.DataFrame,
    fold_secondary: pd.DataFrame,
    high_incidence: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    existing = _existing_fold_metrics(analysis_dir)
    merged = existing.merge(
        fold_secondary,
        on=["horizon", "feature_set", "model_family", "fold", "fit_id"],
        how="outer",
        indicator=True,
    )
    checks.append(
        {
            "check": "corrected_analysis_fold_metrics_key_match",
            "status": "pass" if merged["_merge"].eq("both").all() else "fail",
            "details": "corrected analysis fold metrics align with secondary fold rows",
        }
    )
    for metric_name in ("mae_model", "mae_persistence", "rmse_model", "rmse_persistence"):
        status, diff = _compare_nullable_numeric_columns(
            merged, f"{metric_name}_x", f"{metric_name}_y"
        )
        checks.append(
            {
                "check": f"existing_{metric_name}_unchanged",
                "status": status,
                "max_abs_difference": diff,
                "details": "point estimates agree with corrected analysis fold table",
            }
        )
        status, diff = _compare_nullable_numeric_columns(
            merged, f"{metric_name}_x", f"recomputed_{metric_name}"
        )
        checks.append(
            {
                "check": f"recomputed_{metric_name}_matches_corrected_analysis",
                "status": status,
                "max_abs_difference": diff,
                "details": (
                    "independently recomputed MAE/RMSE agrees with corrected analysis "
                    "with exact null parity"
                ),
            }
        )
    checks.extend(
        _conditional_consistency_checks(
            analysis_dir=analysis_dir,
            fold_secondary=fold_secondary,
            high_incidence=high_incidence,
        )
    )
    checks.append(
        {
            "check": "development_registry_count",
            "status": "pass" if len(registry) == 144 else "fail",
            "observed": int(len(registry)),
            "details": "144 saved fit rows required",
        }
    )
    checks.append(
        {
            "check": "prediction_rows_loaded",
            "status": "pass" if len(predictions) > 0 else "fail",
            "observed": int(len(predictions)),
            "details": "readback saved predictions only; no model loads or fits",
        }
    )
    checks.append(
        {
            "check": "source_run_hashes_verified",
            "status": "pass",
            "details": (
                f"registry={sha256_file(source_dir / 'model_registry.parquet')}; "
                f"analysis={sha256_file(analysis_dir / 'manifest.json')}"
            ),
        }
    )
    checks.append(_source_identity_check(source_dir, analysis_manifest))
    return pd.DataFrame(checks)


def render_result_doc(
    manifest: dict[str, Any],
    table_paths: dict[str, Path],
    config_secondary: pd.DataFrame,
) -> str:
    source = manifest["source_development_run"]
    analysis = manifest["source_corrected_analysis_run"]
    diagnostics_source = manifest["diagnostics_source"]
    null_r2 = int(config_secondary["mean_fold_r2_model"].isna().sum())
    authenticated = source.get("validation_status") != "not_authenticated_test_fixture"
    scope_input = (
        "authenticated saved development predictions and saved metric/threshold documents"
        if authenticated
        else "synthetic unauthenticated fixture inputs"
    )
    return "\n".join(
        [
            "# Milestone 3 secondary diagnostics result",
            "",
            "Status: saved-development diagnostic supplement complete. This is not complete M3.",
            "",
            (
                f"Scope: {scope_input} only. No fits, model loads, raw dataset "
                "reads, 2025 outcome access, installs, commits, existing plots, or "
                "completed-artifact edits were used."
            ),
            "",
            (
                "Interpretation: qualified retrospective observation-time diagnostics "
                "only; no operational claim, no causal claim, and no significance "
                "claim. The 2025 partial-year 24-week cohort remains locked and is "
                "not an annual evaluation."
            ),
            "",
            "## Aggregation policy",
            "",
            (
                "Fold-level values are explicit. Config-level values equally weight "
                "the six predeclared folds. If any required fold is missing or "
                "undefined, the sixfold aggregate is null with a reason and "
                "valid-fold count. Conditional tables retain empty cells and zero "
                "counts. District tables are pooled descriptive summaries, not "
                "equalfold comparisons."
            ),
            "",
            "## Machine-readable outputs",
            "",
            _output_line(
                "Secondary fold metrics",
                table_paths["secondary_fold_metrics"],
                table_paths["secondary_fold_metrics_json"],
            ),
            _output_line(
                "Secondary config metrics",
                table_paths["secondary_config_metrics"],
                table_paths["secondary_config_metrics_json"],
            ),
            _output_line(
                "q90/q95 conditional errors",
                table_paths["high_incidence_conditional_errors"],
                table_paths["high_incidence_conditional_errors_json"],
            ),
            _output_line(
                "Change-category errors",
                table_paths["change_category_errors"],
                table_paths["change_category_errors_json"],
            ),
            _output_line(
                "District improvements",
                table_paths["district_improvements"],
                table_paths["district_improvements_json"],
            ),
            _output_line(
                "Consistency checks",
                table_paths["consistency_checks"],
                table_paths["consistency_checks_json"],
            ),
            f"- Run manifest: `{_rel(Path(manifest['manifest_path']))}`",
            f"- Frozen aggregation policy: `{_rel(Path(manifest['aggregation_policy']['path']))}`",
            "",
            "## Source and code hashes",
            "",
            f"- Development run: `{source['path']}`",
            f"- Development registry SHA256: `{source['registry_sha256']}`",
            f"- Development selection SHA256: `{source['selection_sha256']}`",
            f"- Development complete SHA256: `{source['complete_sha256']}`",
            f"- Corrected analysis run: `{analysis['path']}`",
            f"- Corrected analysis manifest SHA256: `{analysis['manifest_sha256']}`",
            f"- Corrected analysis table verification: `{analysis['table_hash_status']}`",
            f"- Diagnostics code SHA256: `{diagnostics_source['diagnostics_py_sha256']}`",
            f"- Diagnostics script SHA256: `{diagnostics_source['standalone_script_sha256']}`",
            f"- Diagnostics tests SHA256: `{diagnostics_source['tests_sha256']}`",
            f"- Policy SHA256: `{manifest['aggregation_policy']['sha256']}`",
            "",
            "## Consistency notes",
            "",
            (
                "- Existing corrected-analysis MAE/RMSE point estimates are carried "
                "through unchanged in the consistency checks."
            ),
            (
                "- Saved fit metric documents are the source of R2, Poisson "
                "deviance, signed bias, and high-incidence MAE fold values."
            ),
            (
                "- Poisson deviance handling follows the existing metric "
                "implementation with its saved outputs; clipping policy was not "
                "altered."
            ),
            (
                "- Sixfold config R2 model aggregates that are null under the "
                f"policy: `{null_r2}`."
            ),
            "",
            "## Pending outside this slice",
            "",
            (
                "Feature importance, portable packaging, and the test gate remain "
                "pending. Corrected analysis CV intervals remain null where marked "
                "insufficient-support; no new intervals were invented here."
            ),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate saved-development M3 secondary diagnostics."
    )
    parser.add_argument("--development-run", default=str(DEFAULT_DEVELOPMENT_RUN))
    parser.add_argument("--analysis-run", default=str(DEFAULT_ANALYSIS_RUN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-doc", default=str(DEFAULT_RESULT_DOC))
    args = parser.parse_args(argv)
    result = run_diagnostics(
        development_run=args.development_run,
        analysis_run=args.analysis_run,
        output_root=args.output_root,
        run_id=args.run_id,
        result_doc=args.result_doc,
        authenticate=True,
    )
    print(f"diagnostics output: {result.output_dir}")
    print(f"manifest: {result.manifest_path}")
    print(f"result doc: {result.result_doc_path}")
    return 0


def _metric_values_for(truth: pd.Series, pred: pd.Series) -> dict[str, float | None]:
    bundle = metrics.metric_bundle(truth, pred)
    return {
        "mae": _finite_or_none(bundle["mae"]),
        "rmse": _finite_or_none(bundle["rmse"]),
        "r2": _finite_or_none(bundle["r2"]),
        "poisson_deviance": _finite_or_none(bundle["poisson_deviance"]),
        "bias": _finite_or_none(bundle["bias"]),
    }


def _error_row(
    threshold: Any,
    subset: pd.DataFrame,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "horizon": int(threshold.horizon),
        "fold": int(threshold.fold),
        "feature_set": str(threshold.feature_set),
        "model_family": str(threshold.model_family),
        "fit_id": str(threshold.fit_id),
        "threshold_binding_sha256": str(threshold.threshold_binding_sha256),
        "count": int(len(subset)),
        "mae_model": None,
        "mae_persistence": None,
        "bias_model": None,
        "bias_persistence": None,
        "null_reason": "empty_subset" if subset.empty else None,
    }
    if not subset.empty:
        truth = subset["observed_target"].astype(float)
        model = metrics.clip_predictions_nonnegative(subset["prediction_model"])
        persistence = metrics.clip_predictions_nonnegative(subset["prediction_persistence"])
        row.update(
            {
                "mae_model": metrics.mae(truth, model),
                "mae_persistence": metrics.mae(truth, persistence),
                "bias_model": metrics.bias(truth, model),
                "bias_persistence": metrics.bias(truth, persistence),
            }
        )
    row.update(extra)
    return row


def _sixfold_mean_or_null(group: pd.DataFrame, column: str) -> tuple[float | None, str | None, int]:
    folds = set(int(value) for value in group["fold"])
    if folds != set(FOLDS):
        valid = int(group[column].notna().sum()) if column in group else 0
        return None, "missing_required_folds", valid
    values = pd.to_numeric(group[column], errors="coerce")
    valid_count = int(values.notna().sum())
    if valid_count != REQUIRED_FOLD_COUNT:
        return None, "one_or_more_required_folds_undefined", valid_count
    return float(values.mean()), None, valid_count


def _merge_saved_and_recomputed(saved: pd.DataFrame, recomputed: pd.DataFrame) -> pd.DataFrame:
    merged = saved.merge(
        recomputed,
        on=["horizon", "feature_set", "model_family", "fold", "fit_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise DiagnosticsError("saved metric doc key mismatch against recomputed fold metrics")
    merged = merged.drop(columns=["_merge"])
    for metric_name in ("r2", "poisson_deviance", "bias"):
        for family in ("model", "persistence"):
            saved_col = f"{metric_name}_{family}"
            recomputed_col = f"recomputed_{metric_name}_{family}"
            status, _ = _compare_nullable_numeric_columns(merged, saved_col, recomputed_col)
            if status != "pass":
                raise DiagnosticsError(f"saved metric doc mismatch: {saved_col}")
    return merged


def _existing_fold_metrics(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "fold_metrics.csv"
    table = pd.read_csv(path)
    return table[
        [
            "horizon",
            "feature_set",
            "model_family",
            "fold",
            "fit_id",
            "mae_model",
            "mae_persistence",
            "rmse_model",
            "rmse_persistence",
        ]
    ].copy()


def _conditional_consistency_checks(
    *,
    analysis_dir: Path,
    fold_secondary: pd.DataFrame,
    high_incidence: pd.DataFrame,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    analysis = pd.read_csv(analysis_dir / "high_incidence_thresholds.csv")
    key_cols = ["horizon", "fold", "feature_set", "model_family", "fit_id"]
    for label, analysis_count_col, saved_count_col, saved_mae_col in (
        ("q90_primary", "q90_count", "top_10_count", "mae_top_10pct"),
        ("q95_supplementary", "q95_count", "top_5_count", "mae_top_5pct"),
    ):
        conditional = high_incidence[high_incidence["incidence_group"].eq(label)].copy()
        merged = analysis[key_cols + [analysis_count_col]].merge(
            conditional[key_cols + ["count", "mae_model", "mae_persistence"]],
            on=key_cols,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        key_status = "pass" if merged["_merge"].eq("both").all() else "fail"
        checks.append(
            {
                "check": f"{label}_corrected_analysis_key_match",
                "status": key_status,
                "details": "conditional incidence rows align with corrected analysis thresholds",
            }
        )
        count_status, count_diff = _compare_nullable_numeric_columns(
            merged, analysis_count_col, "count"
        )
        checks.append(
            {
                "check": f"{label}_count_matches_corrected_analysis",
                "status": count_status,
                "max_abs_difference": count_diff,
                "details": "conditional row counts match corrected analysis q90/q95 counts",
            }
        )
        saved = fold_secondary[
            key_cols
            + [
                f"{saved_count_col}_model",
                f"{saved_count_col}_persistence",
                f"{saved_mae_col}_model",
                f"{saved_mae_col}_persistence",
            ]
        ]
        saved_merged = conditional[key_cols + ["count", "mae_model", "mae_persistence"]].merge(
            saved,
            on=key_cols,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        checks.append(
            {
                "check": f"{label}_saved_metric_doc_key_match",
                "status": "pass" if saved_merged["_merge"].eq("both").all() else "fail",
                "details": "conditional incidence rows align with saved fit metric documents",
            }
        )
        for family in ("model", "persistence"):
            count_status, count_diff = _compare_nullable_numeric_columns(
                saved_merged, "count", f"{saved_count_col}_{family}"
            )
            checks.append(
                {
                    "check": f"{label}_{family}_count_matches_saved_metric_doc",
                    "status": count_status,
                    "max_abs_difference": count_diff,
                    "details": "saved conditional count agrees with recomputed conditional table",
                }
            )
            mae_status, mae_diff = _compare_nullable_numeric_columns(
                saved_merged, f"mae_{family}", f"{saved_mae_col}_{family}"
            )
            checks.append(
                {
                    "check": f"{label}_{family}_mae_matches_saved_metric_doc",
                    "status": mae_status,
                    "max_abs_difference": mae_diff,
                    "details": "saved conditional MAE agrees with recomputed conditional table",
                }
            )
    return checks


def _source_identity_check(source_dir: Path, analysis_manifest: dict[str, Any]) -> dict[str, Any]:
    expected = analysis_manifest.get("source_development_run", {})
    observed = {
        "registry_sha256": sha256_file(source_dir / "model_registry.parquet"),
        "selection_sha256": sha256_file(source_dir / "selection.json"),
        "run_identity_sha256": sha256_file(source_dir / "run_identity.json"),
        "complete_sha256": sha256_file(source_dir / "complete.json"),
    }
    mismatches = [
        name
        for name, digest in observed.items()
        if str(expected.get(name)) != str(digest)
    ]
    return {
        "check": "corrected_analysis_source_development_identity_match",
        "status": "pass" if not mismatches else "fail",
        "details": (
            "corrected analysis source binding matches selected development "
            f"registry/selection/run identity/completion byte hashes; mismatches={mismatches}"
        ),
    }


def _compare_nullable_numeric_columns(
    table: pd.DataFrame,
    left: str,
    right: str,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-9,
) -> tuple[str, float | None]:
    left_values = pd.to_numeric(table[left], errors="coerce")
    right_values = pd.to_numeric(table[right], errors="coerce")
    left_null = left_values.isna()
    right_null = right_values.isna()
    if not left_null.equals(right_null):
        return "fail", None
    finite = ~(left_null | right_null)
    if not finite.any():
        return "pass", None
    diff = (left_values[finite] - right_values[finite]).abs()
    max_diff = float(diff.max())
    close = np.isclose(left_values[finite], right_values[finite], rtol=rtol, atol=atol)
    return ("pass" if bool(close.all()) else "fail"), max_diff


def _assert_consistency_passes(consistency: pd.DataFrame) -> None:
    failed = consistency[~consistency["status"].eq("pass")]
    if not failed.empty:
        names = ", ".join(str(value) for value in failed["check"].tolist())
        raise DiagnosticsError(f"diagnostics consistency checks failed: {names}")


def _verify_development_hashes(source_dir: Path, registry: pd.DataFrame) -> None:
    checksums = _read_json(source_dir / "checksums.json")
    for rel in ("model_registry.parquet", "selection.json", "complete.json", "run_identity.json"):
        expected = checksums.get(rel)
        if expected is not None and sha256_file(source_dir / rel) != expected:
            raise DiagnosticsError(f"development checksum mismatch: {rel}")
    for row in registry.itertuples():
        for attr in ("prediction_path", "metrics_path", "threshold_path"):
            rel = str(getattr(row, attr))
            hash_attr = attr.replace("_path", "_sha256")
            if sha256_file(source_dir / rel) != str(getattr(row, hash_attr)):
                raise DiagnosticsError(f"development registry hash mismatch: {rel}")


def _verify_analysis_manifest(analysis_dir: Path) -> dict[str, Any]:
    manifest_path = analysis_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    for name, entry in manifest.get("tables", {}).items():
        path = Path(entry["path"])
        if not path.is_absolute():
            path = analysis_dir / path
        if not _is_relative_to(path.resolve(), analysis_dir):
            raise DiagnosticsError(f"analysis table path escapes run directory: {name}")
        if sha256_file(path) != str(entry["sha256"]):
            raise DiagnosticsError(f"corrected analysis table hash mismatch: {name}")
    return manifest


def _manifest(
    *,
    source_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    policy_path: Path,
    validation: dict[str, Any],
    analysis_manifest: dict[str, Any],
    table_paths: dict[str, Path],
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "complete",
        "manifest_path": str(manifest_path),
        "diagnostics_output_dir": str(output_dir),
        "qualification": (
            "development_only_qualified_retrospective_observation_time_"
            "not_operational_backtesting"
        ),
        "no_significance_claims": True,
        "cohort_2025_status": "locked_partial_year_24_weeks_not_annual_no_outcome_access",
        "source_development_run": {
            "path": str(source_dir),
            "validation_status": validation.get("status"),
            "registry_sha256": sha256_file(source_dir / "model_registry.parquet"),
            "selection_sha256": sha256_file(source_dir / "selection.json"),
            "complete_sha256": sha256_file(source_dir / "complete.json"),
            "run_identity_sha256": sha256_file(source_dir / "run_identity.json"),
        },
        "source_corrected_analysis_run": {
            "path": str(analysis_dir),
            "manifest_sha256": sha256_file(analysis_dir / "manifest.json"),
            "table_hash_status": "verified_before_consumption",
            "source_manifest_status": analysis_manifest.get("status"),
        },
        "aggregation_policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "frozen_before_analysis": True,
        },
        "diagnostics_source": {
            "diagnostics_py_sha256": sha256_file(Path(__file__)),
            "standalone_script_sha256": sha256_file(
                core.REPO_ROOT / "scripts" / "milestone3_diagnostics.py"
            ),
            "tests_sha256": sha256_file(
                core.REPO_ROOT / "tests" / "unit" / "test_milestone3_diagnostics.py"
            ),
            "task_policy_sha256": sha256_file(
                core.REPO_ROOT / "docs" / "m3-secondary-diagnostics-task.md"
            ),
        },
        "tables": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(table_paths.items())
        },
        "reproducibility_command": (
            "TMPDIR=/tmp/dengue-forecast-pytest-temp "
            ".venv/bin/python scripts/milestone3_diagnostics.py "
            "--development-run artifacts/milestone3/development/approved-development-002 "
            "--analysis-run artifacts/milestone3/analysis/saved-predictions-dev-corrected-001 "
            "--output-root artifacts/milestone3/diagnostics --run-id <fresh-run-id>"
        ),
    }
    return manifest


def _metric_doc_value(entry: Any) -> tuple[float | None, str | None]:
    if not isinstance(entry, dict):
        value = entry
        reason = None
    else:
        value = entry.get("value")
        reason = entry.get("reason")
    value = _finite_or_none(value)
    if value is None and reason is None:
        reason = "undefined"
    return value, reason


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    candidate = float(value)
    if not np.isfinite(candidate):
        return None
    return candidate


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    candidate = float(value)
    if not np.isfinite(candidate):
        return None
    return candidate


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)


def _json_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    clean = table.astype(object).where(pd.notna(table), None)
    return clean.to_dict(orient="records")


def _sha256_obj(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(core.REPO_ROOT))
    except ValueError:
        return str(path)


def _output_line(label: str, csv_path: Path, json_path: Path) -> str:
    return f"- {label}: `{_rel(csv_path)}` and `{_rel(json_path)}`"
