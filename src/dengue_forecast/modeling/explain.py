from __future__ import annotations

# ruff: noqa: E402, I001

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from dengue_forecast.modeling.dataset import FORBIDDEN_CONTEXT_COLUMNS, FUTURE_TARGET_COLUMNS
from dengue_forecast.modeling.train import TrainedModel, load_champion


TREE_FAMILIES = {
    "random_forest",
    "hist_gradient_boosting",
    "hist_gradient_boosting_poisson",
    "xgboost",
    "lightgbm",
}
SUSPICIOUS_TOKENS = ("future", "next", "lead", "target", "incidence")


class ExplainabilityError(ValueError):
    """Raised when explainability inputs violate the validation-only contract."""


@dataclass(frozen=True)
class ExplainabilityResult:
    feature_importance: pd.DataFrame
    shap_values: pd.DataFrame
    transformed_sample: pd.DataFrame
    summary_markdown: str
    paths: dict[str, Path]


class _PredictWrapper:
    def __init__(self, trained_model: TrainedModel) -> None:
        self.trained_model = trained_model

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> _PredictWrapper:
        del x, y
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if list(x.columns) != self.trained_model.feature_columns:
            raise ExplainabilityError("Permutation wrapper received feature column order mismatch")
        return self.trained_model.predict_next_week(x)


def _load_model(model: TrainedModel | str | Path) -> TrainedModel:
    if isinstance(model, TrainedModel):
        return model
    return load_champion(model)


def _sample_validation(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    max_sample: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    missing = [
        column for column in [*feature_columns, target_column] if column not in frame.columns
    ]
    if missing:
        raise ExplainabilityError(f"Validation frame missing required columns: {missing}")
    clean = frame.dropna(subset=[target_column]).copy()
    if clean.empty:
        raise ExplainabilityError("Validation frame has no observed targets")
    if len(clean) > max_sample:
        clean = clean.sample(n=max_sample, random_state=seed).sort_index()
    return clean.loc[:, feature_columns], pd.to_numeric(clean[target_column], errors="coerce")


def _feature_metadata(
    transformed_features: list[str],
    feature_registry: pd.DataFrame | None,
) -> pd.DataFrame:
    registry_lookup: dict[str, dict[str, Any]] = {}
    if feature_registry is not None and "feature_name" in feature_registry.columns:
        registry_lookup = feature_registry.set_index("feature_name").to_dict("index")
    rows: list[dict[str, Any]] = []
    for feature in transformed_features:
        raw_feature = feature
        derived_type = "raw"
        if feature.endswith("__missing"):
            raw_feature = feature.removesuffix("__missing")
            derived_type = "missing_indicator"
        elif feature.startswith("district_id__"):
            raw_feature = "district_id"
            derived_type = "district_onehot"
        elif "__" in feature:
            raw_feature = feature.split("__", 1)[0]
            derived_type = "onehot"
        registry_row = registry_lookup.get(raw_feature, {})
        rows.append(
            {
                "feature": feature,
                "raw_feature": raw_feature,
                "derived_type": derived_type,
                "feature_group": registry_row.get("feature_group", "unknown"),
                "uses_future_information": bool(registry_row.get("uses_future_information", False)),
                "eligible_for_training": registry_row.get("eligible_for_training", pd.NA),
            }
        )
    return pd.DataFrame(rows)


def _registry_row(feature_registry: pd.DataFrame | None, raw_feature: str) -> dict[str, Any]:
    if feature_registry is None or "feature_name" not in feature_registry.columns:
        return {}
    matches = feature_registry[feature_registry["feature_name"].eq(raw_feature)]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def _fallback_group(raw_feature: str) -> str:
    name = raw_feature.lower()
    if raw_feature == "district_id" or name.startswith("district_id"):
        return "identifier"
    if "case" in name or "dengue" in name:
        return "epidemiology"
    if any(token in name for token in ["rain", "precip", "temp", "humidity", "weather"]):
        return "weather"
    if any(token in name for token in ["week", "month", "season", "sin", "cos"]):
        return "seasonality"
    if raw_feature == "area_km2":
        return "reference"
    if "population" in name:
        return "population"
    return "unknown"


def _metadata_for_raw_feature(
    raw_feature: str,
    feature_registry: pd.DataFrame | None,
) -> dict[str, Any]:
    registry_row = _registry_row(feature_registry, raw_feature)
    return {
        "raw_feature": raw_feature,
        "derived_type": "raw",
        "feature_group": registry_row.get("feature_group", _fallback_group(raw_feature)),
        "uses_future_information": bool(registry_row.get("uses_future_information", False)),
        "eligible_for_training": registry_row.get("eligible_for_training", pd.NA),
    }


def _attach_importance_metadata(
    importance: pd.DataFrame,
    transformed_metadata: pd.DataFrame,
    feature_registry: pd.DataFrame | None,
) -> pd.DataFrame:
    merged = importance.merge(transformed_metadata, on="feature", how="left")
    missing = merged["raw_feature"].isna()
    if missing.any():
        for index, feature in merged.loc[missing, "feature"].items():
            metadata = _metadata_for_raw_feature(str(feature), feature_registry)
            for column, value in metadata.items():
                merged.at[index, column] = value
    return merged


def _native_importance(trained: TrainedModel, transformed_columns: list[str]) -> pd.DataFrame:
    model = trained.model
    if hasattr(model, "booster_"):
        booster = model.booster_
        gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype="float64")
        split = np.asarray(booster.feature_importance(importance_type="split"), dtype="float64")
        return pd.DataFrame(
            [
                {"feature": feature, "importance_type": "native_gain", "importance": float(g)}
                for feature, g in zip(transformed_columns, gain, strict=True)
            ]
            + [
                {"feature": feature, "importance_type": "native_split", "importance": float(s)}
                for feature, s in zip(transformed_columns, split, strict=True)
            ]
        )
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype="float64")
        return pd.DataFrame(
            {
                "feature": transformed_columns,
                "importance_type": "native_feature_importance",
                "importance": values,
            }
        )
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        rows: list[dict[str, Any]] = []
        for importance_type in ["gain", "weight"]:
            scores = booster.get_score(importance_type=importance_type)
            for index, feature in enumerate(transformed_columns):
                rows.append(
                    {
                        "feature": feature,
                        "importance_type": f"native_{importance_type}",
                        "importance": float(scores.get(f"f{index}", 0.0)),
                    }
                )
        return pd.DataFrame(rows)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype="float64")
        return pd.DataFrame(
            {
                "feature": transformed_columns,
                "importance_type": "linear_coefficient",
                "importance": coef,
            }
        )
    return pd.DataFrame(columns=["feature", "importance_type", "importance"])


def _permutation_importance(
    trained: TrainedModel,
    raw_x: pd.DataFrame,
    y: pd.Series,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    repeats = max(1, min(3, int(repeats)))

    def scorer(estimator: _PredictWrapper, x: pd.DataFrame, target: pd.Series) -> float:
        return -float(mean_absolute_error(target, estimator.predict(x)))

    result = permutation_importance(
        _PredictWrapper(trained),
        raw_x,
        y,
        scoring=scorer,
        n_repeats=repeats,
        random_state=seed,
    )
    return pd.DataFrame(
        {
            "feature": raw_x.columns,
            "importance_type": "permutation_mae_increase",
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )


def _output_units(trained: TrainedModel) -> str:
    objective = trained.config.objective
    family = trained.config.family
    if objective in {"poisson", "count:poisson"} or family.endswith("_poisson"):
        return (
            "Poisson objective raw model margin/log expected case count; "
            "SHAP values are not additive case counts."
        )
    return "Predicted dengue case count target."


def _selected_value(selected_config: dict[str, Any] | None, *keys: str) -> Any:
    if not selected_config:
        return None
    current: Any = selected_config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _provenance_labels(
    trained: TrainedModel,
    selected_config: dict[str, Any] | None,
) -> tuple[str, str]:
    selected_name = (
        _selected_value(selected_config, "model_id")
        or _selected_value(selected_config, "config_id")
        or _selected_value(selected_config, "stable_id")
        or _selected_value(selected_config, "id")
        or trained.config.family
    )
    fold_id = (
        _selected_value(selected_config, "fold_id")
        or _selected_value(selected_config, "fold", "fold_id")
        or _selected_value(selected_config, "fold")
        or trained.metadata.get("row_identity", {}).get("fold_id")
        or "not supplied"
    )
    return str(selected_name), str(fold_id)


def _scope_label(
    trained: TrainedModel,
    *,
    selected_config: dict[str, Any] | None,
    sample_size: int | None,
) -> str:
    selected_name, fold_id = _provenance_labels(trained, selected_config)
    if fold_id == "not supplied":
        fold_id = ""
    parts = [str(value) for value in [selected_name, fold_id] if value]
    if sample_size is not None:
        parts.append(f"n={sample_size}")
    return "; ".join(parts)


def _shap_axis_label(trained: TrainedModel) -> str:
    objective = trained.config.objective
    if objective in {"poisson", "count:poisson"} or trained.config.family.endswith("_poisson"):
        return "SHAP value on raw log-mean-count margin for Poisson model"
    return "SHAP value on raw model output"


def _shap_title(
    trained: TrainedModel,
    *,
    selected_config: dict[str, Any] | None,
    sample_size: int | None,
    prefix: str,
) -> str:
    family = trained.config.family
    objective = trained.config.objective
    if "shortlisted_boosted_" in prefix and family == "lightgbm" and objective == "poisson":
        base = "LightGBM Poisson shortlist vs Ridge champion SHAP"
    elif family == "lightgbm" and objective == "poisson":
        base = "LightGBM Poisson validation SHAP"
    else:
        base = f"{family} validation SHAP"
    scope = _scope_label(trained, selected_config=selected_config, sample_size=sample_size)
    return f"{base}: {scope}" if scope else base


def _shap_or_equivalent(
    trained: TrainedModel,
    transformed_x: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str,
    selected_config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, Path | None, str]:
    family = trained.config.family
    if family == "ridge" and hasattr(trained.model, "coef_"):
        coef = np.asarray(trained.model.coef_, dtype="float64")
        values = transformed_x.to_numpy(dtype="float64") * coef
        shap_frame = pd.DataFrame(values, columns=transformed_x.columns, index=transformed_x.index)
        mean_abs = shap_frame.abs().mean().rename("mean_abs_value").reset_index()
        mean_abs = mean_abs.rename(columns={"index": "feature"})
        mean_abs["method"] = "linear_equivalent_contribution"
        return (
            mean_abs,
            None,
            "Linear equivalent contributions were used for Ridge; these are not SHAP.",
        )
    if family not in TREE_FAMILIES:
        return (
            pd.DataFrame(columns=["feature", "mean_abs_value", "method"]),
            None,
            f"No SHAP-compatible path implemented for family {family}.",
        )
    try:
        import shap

        explainer = shap.TreeExplainer(trained.model)
        values = explainer.shap_values(transformed_x)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values, dtype="float64")
        if values.ndim == 3:
            values = values[:, :, 0]
        mean_abs = pd.DataFrame(
            {
                "feature": transformed_x.columns,
                "mean_abs_value": np.abs(values).mean(axis=0),
                "method": "shap_tree_explainer",
            }
        )
        plot_path = output_dir / f"{prefix}shap_summary_beeswarm.png"
        shap.summary_plot(
            values,
            transformed_x,
            show=False,
            max_display=min(20, transformed_x.shape[1]),
        )
        plt.title(
            _shap_title(
                trained,
                selected_config=selected_config,
                sample_size=len(transformed_x),
                prefix=prefix,
            )
        )
        plt.gca().set_xlabel(_shap_axis_label(trained))
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
        return (
            mean_abs,
            plot_path,
            "TreeExplainer SHAP values computed on transformed validation features.",
        )
    except Exception as exc:  # pragma: no cover - compatibility depends on external backends
        note = (
            "SHAP TreeExplainer failed for this estimator/library combination. "
            f"No fake SHAP values were produced. Error: {type(exc).__name__}: {exc}"
        )
        return pd.DataFrame(columns=["feature", "mean_abs_value", "method"]), None, note


def _raw_permutation_plot(
    importance: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str,
    trained: TrainedModel,
    selected_config: dict[str, Any] | None,
    sample_size: int,
    max_display: int = 20,
) -> Path | None:
    rows = importance[
        importance["importance_type"].eq("permutation_mae_increase")
        & importance["method_scope"].eq("raw_feature")
    ].copy()
    rows["importance"] = pd.to_numeric(rows["importance"], errors="coerce")
    rows = rows.dropna(subset=["importance"])
    if rows.empty:
        return None
    rows["abs_importance"] = rows["importance"].abs()
    rows = rows.sort_values("abs_importance", ascending=False).head(max_display)
    rows = rows.sort_values("importance")
    colors = np.where(rows["importance"].to_numpy(dtype="float64") >= 0, "#2f6f73", "#9b4d4d")
    height = max(4.8, 0.32 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(rows["feature"].astype(str), rows["importance"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    scope = _scope_label(trained, selected_config=selected_config, sample_size=sample_size)
    title = "Raw permutation importance: validation MAE delta"
    if trained.config.family == "ridge":
        title = "Lead Ridge raw permutation importance: validation MAE delta"
    ax.set_title(f"{title}\n{scope}" if scope else title)
    ax.set_xlabel("Permutation MAE delta in case-count target units")
    ax.set_ylabel("Raw feature")
    ax.grid(axis="x", alpha=0.25)
    path = output_dir / f"{prefix}raw_permutation_mae_delta.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _leakage_audit(feature_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in feature_metadata.itertuples(index=False):
        raw = str(row.raw_feature)
        suspicious = (
            raw in FUTURE_TARGET_COLUMNS
            or raw in FORBIDDEN_CONTEXT_COLUMNS
            or any(token in raw.lower() for token in SUSPICIOUS_TOKENS)
        )
        authorized_exception = raw in {"district_id", "area_km2"}
        rows.append(
            {
                "feature": row.feature,
                "raw_feature": raw,
                "suspicious": suspicious,
                "authorized_modeling_exception": authorized_exception,
                "status": "review_required" if suspicious and not authorized_exception else "ok",
            }
        )
    return pd.DataFrame(rows)


def _summary(
    importance: pd.DataFrame,
    leakage: pd.DataFrame,
    *,
    shap_note: str,
    output_units: str,
    trained: TrainedModel,
    sample_size: int,
    selected_config: dict[str, Any] | None,
) -> str:
    def top_rows(
        frame: pd.DataFrame,
        *,
        importance_type: str,
        scope: str | None = None,
        n: int = 5,
        positive_only: bool = False,
    ) -> pd.DataFrame:
        rows = frame[frame["importance_type"].eq(importance_type)].copy()
        if scope is not None:
            rows = rows[rows["method_scope"].eq(scope)]
        rows["importance"] = pd.to_numeric(rows["importance"], errors="coerce")
        rows = rows.dropna(subset=["importance"])
        if positive_only:
            rows = rows[rows["importance"].gt(0)]
        return rows.sort_values("importance", ascending=False).head(n)

    def group_rows(
        frame: pd.DataFrame,
        *,
        importance_type: str,
        scope: str | None = None,
        positive_only: bool = False,
    ) -> pd.Series:
        rows = frame[frame["importance_type"].eq(importance_type)].copy()
        if scope is not None:
            rows = rows[rows["method_scope"].eq(scope)]
        rows["importance"] = pd.to_numeric(rows["importance"], errors="coerce")
        rows = rows.dropna(subset=["importance"])
        if positive_only:
            rows = rows[rows["importance"].gt(0)]
        if rows.empty:
            return pd.Series(dtype="float64")
        return rows.groupby("feature_group", dropna=False)["importance"].sum().sort_values(
            ascending=False
        )

    def format_feature_list(rows: pd.DataFrame, value_label: str) -> str:
        if rows.empty:
            return "none observed"
        parts = []
        for rank, row in enumerate(rows.itertuples(index=False), start=1):
            parts.append(f"{rank}. {row.feature} ({value_label} {float(row.importance):.6g})")
        return "; ".join(parts)

    def permutation_for(predicate: Any) -> pd.DataFrame:
        rows = importance[
            importance["importance_type"].eq("permutation_mae_increase")
            & importance["method_scope"].eq("raw_feature")
        ].copy()
        rows["importance"] = pd.to_numeric(rows["importance"], errors="coerce")
        rows = rows.dropna(subset=["importance"])
        mask = rows["raw_feature"].astype(str).map(predicate)
        return rows[mask].sort_values("importance", ascending=False)

    def raw_feature_present(predicate: Any) -> bool:
        raw_features = set(importance["raw_feature"].dropna().astype(str))
        return any(predicate(feature) for feature in raw_features)

    def raw_feature_names(predicate: Any) -> list[str]:
        raw_features = sorted(set(importance["raw_feature"].dropna().astype(str)))
        return [feature for feature in raw_features if predicate(feature)]

    def feature_rank(feature: str) -> str:
        rows = top_rows(
            importance,
            importance_type="permutation_mae_increase",
            scope="raw_feature",
            n=len(importance),
        )
        for rank, row in enumerate(rows.itertuples(index=False), start=1):
            if str(row.raw_feature) == feature:
                return f"rank {rank}, MAE delta {float(row.importance):.6g}"
        return "not ranked by positive permutation contribution"

    def selected_value(*keys: str) -> Any:
        if not selected_config:
            return None
        current: Any = selected_config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    permutation_groups = group_rows(
        importance,
        importance_type="permutation_mae_increase",
        scope="raw_feature",
        positive_only=True,
    )
    native_types = [
        str(value)
        for value in importance["importance_type"].dropna().unique()
        if str(value).startswith("native_") or str(value) == "linear_coefficient"
    ]
    shap_rows = top_rows(
        importance,
        importance_type="mean_abs_shap_or_equivalent",
        scope="transformed_feature",
        n=5,
        positive_only=True,
    )
    suspicious = leakage[leakage["status"].eq("review_required")]
    case_rows = permutation_for(
        lambda feature: ("case" in feature.lower()) or ("dengue" in feature.lower())
    )
    lag_rows = permutation_for(
        lambda feature: ("lag" in feature.lower())
        or ("roll" in feature.lower())
        or ("change" in feature.lower())
        or feature in {"dengue_cases", "log1p_cases"}
    )
    rainfall_rows = permutation_for(
        lambda feature: "rain" in feature.lower() or "precip" in feature.lower()
    )
    temp_humidity_rows = permutation_for(
        lambda feature: "temp" in feature.lower() or "humidity" in feature.lower()
    )
    district_rows = permutation_for(
        lambda feature: feature == "district_id" or feature.startswith("district")
    )
    season_rows = permutation_for(
        lambda feature: any(
            token in feature.lower() for token in ["week", "month", "season", "sin", "cos"]
        )
    )
    area_rows = permutation_for(lambda feature: feature == "area_km2")
    population_present = raw_feature_present(lambda feature: "population" in feature.lower())
    all_feature_names = set(trained.feature_columns)
    has_weather_features = any(
        any(
            token in feature.lower()
            for token in ["rain", "precip", "temp", "humidity", "weather"]
        )
        for feature in all_feature_names
    )
    selected_name, fold_id = _provenance_labels(trained, selected_config)
    validation_id = (
        selected_value("validation_id")
        or selected_value("validation", "id")
        or selected_value("validation_fold_id")
        or "latest admissible validation sample"
    )
    target_scale = (
        selected_value("target_scale")
        or selected_value("target", "scale")
        or trained.target_column
    )
    provisional = selected_value("provisional")
    provisional_text = "not supplied" if provisional is None else str(bool(provisional)).lower()

    if rainfall_rows.empty and not has_weather_features:
        rainfall_answer = (
            "Rainfall is not in this case-only model; use the weather/full-context ablation "
            "for incremental weather value rather than inferring a zero causal effect."
        )
    elif rainfall_rows.empty:
        rainfall_names = raw_feature_names(
            lambda feature: "rain" in feature.lower() or "precip" in feature.lower()
        )
        rainfall_answer = (
            "Rainfall features are present but have no positive raw permutation MAE increase "
            "in this sample "
            f"({', '.join(rainfall_names) if rainfall_names else 'no named rainfall rows'})."
        )
    else:
        rainfall_answer = (
            "Rainfall raw permutation ranks: "
            f"{format_feature_list(rainfall_rows.head(5), 'MAE delta')}."
        )

    if temp_humidity_rows.empty and not has_weather_features:
        temp_humidity_answer = (
            "Temperature and humidity are not in this case-only model; interpret them through "
            "the weather/full-context ablation and shortlisted boosted artifacts only."
        )
    elif temp_humidity_rows.empty:
        names = raw_feature_names(
            lambda feature: "temp" in feature.lower() or "humidity" in feature.lower()
        )
        temp_humidity_answer = (
            "Temperature/humidity features are present but have no positive raw permutation "
            f"MAE increase in this sample ({', '.join(names) if names else 'no named rows'})."
        )
    else:
        temp_humidity_answer = (
            "Temperature/humidity raw permutation ranks: "
            f"{format_feature_list(temp_humidity_rows.head(5), 'MAE delta')}."
        )

    if district_rows.empty:
        district_answer = (
            "District identity has no positive raw permutation MAE increase in this sample."
        )
    else:
        district_answer = (
            f"District identity: {format_feature_list(district_rows.head(3), 'MAE delta')}; "
            "top recent case-history features are "
            f"{format_feature_list(lag_rows.head(3), 'MAE delta')}."
        )

    if season_rows.empty:
        season_answer = (
            "Seasonality has no positive raw permutation MAE increase here; near-zero or "
            "negative values should be read as weak/no observed contribution under this "
            "small correlated validation sample."
        )
    else:
        non_positive = season_rows[season_rows["importance"].le(0)]
        season_answer = (
            "Seasonality raw permutation ranks: "
            f"{format_feature_list(season_rows.head(5), 'MAE delta')}."
        )
        if not non_positive.empty:
            season_answer += (
                " Some seasonality rows are zero/negative and should not be over-interpreted."
            )

    suspicious_power = suspicious
    if suspicious_power.empty:
        suspicious_answer = (
            "No suspicious future/proxy feature names were detected by the declared "
            "name-scope audit."
        )
    else:
        names = ", ".join(suspicious_power["raw_feature"].drop_duplicates().astype(str).head(8))
        suspicious_answer = (
            f"{len(suspicious_power)} transformed features require review by name-scope "
            f"leakage audit: {names}."
        )

    lines = [
        "# Validation Explainability Summary",
        "",
        "Explainability is model interpretation, not causal epidemiological evidence. "
        "Effects are not causal.",
        "",
        "## Provenance",
        f"- Config interpreted: {selected_name}",
        f"- Family/objective: {trained.config.family}/{trained.config.objective}",
        f"- Validation fold/sample: {fold_id}; {validation_id}; n={sample_size}",
        f"- Target scale: {target_scale}",
        f"- Provisional config flag: {provisional_text}",
        "- Sample source: validation rows supplied to this function, capped by max_sample "
        "before any locked test use.",
        "",
        f"Output units: {output_units}",
        "",
        f"SHAP/equivalent status: {shap_note}",
        "",
        "## Dominant Feature Groups By Method",
        "",
        "Raw-feature permutation importance ranks observed contribution as validation "
        "MAE increase. "
        "Groups below are summed only within this one method and unit.",
    ]
    if permutation_groups.empty:
        lines.append("- permutation_mae_increase: no positive raw-feature MAE increases observed.")
    else:
        lines.extend(
            f"- permutation_mae_increase group {group}: {value:.6g} MAE delta"
            for group, value in permutation_groups.items()
        )
    lines.extend(
        [
            "",
            "Native model importances are reported in their estimator-specific units and are not "
            "added to permutation or SHAP/equivalent values.",
        ]
    )
    for importance_type in native_types:
        native_group = group_rows(
            importance,
            importance_type=importance_type,
            scope="transformed_feature",
        )
        if native_group.empty:
            continue
        lines.extend(
            f"- {importance_type} group {group}: {value:.6g}"
            for group, value in native_group.items()
        )
    lines.extend(
        [
            "",
            "Transformed-feature SHAP/equivalent magnitudes are separate. Ridge uses linear "
            "equivalent contributions, not SHAP; Poisson tree SHAP is on the raw margin/log "
            "expected count scale and is not an absolute case-count effect.",
            "- Top transformed SHAP/equivalent features: "
            f"{format_feature_list(shap_rows, 'mean abs')}",
            "",
            "## Required Questions",
            "- Recent cases/lags: top raw permutation case-history features are "
            f"{format_feature_list(case_rows.head(5), 'MAE delta')}. "
            "Lag/rolling/change features specifically rank as "
            f"{format_feature_list(lag_rows.head(5), 'MAE delta')}.",
            f"- Rainfall: {rainfall_answer}",
            f"- Temperature and humidity: {temp_humidity_answer}",
            f"- District identity vs recent cases: {district_answer}",
            f"- Seasonality: {season_answer}",
            f"- Unexpected or suspicious power: {suspicious_answer} Large positive known "
            "current-case features are not future leakage by name, but they require "
            "as-of semantics review in the data audit.",
            "- Future/proxy audit: scope is feature-name and registry metadata review of "
            "the transformed validation design matrix; it is a numerical review of the "
            "actual supplied features, not a claim that every point-in-time join is correct.",
            "",
            "## Scope And Limitations",
            "- Comparisons are associational interpretation, not causal claims.",
            "- Permutation importances can be noisy on a small 200-row correlated validation "
            "sample and may understate interchangeable lag features.",
            "- Do not compare absolute SHAP/equivalent magnitudes across Ridge and "
            "Poisson/tree models.",
        ]
    )
    if not area_rows.empty:
        lines.append(
            "- Static area_km2 is present as approved 2017 reference geography: "
            f"{feature_rank('area_km2')}."
        )
    else:
        lines.append("- Static area_km2 is approved only as 2017 reference geography when present.")
    if population_present:
        lines.append(
            "- Population-like features are present and should remain blocked unless "
            "explicitly approved."
        )
    else:
        lines.append("- Population is not present in the interpreted feature set.")
    lines.extend(
        [
            "- Read shortlisted boosted SHAP separately from the Ridge-equivalent contribution "
            "summary; the two use different model families and scales.",
        ]
    )
    if suspicious.empty:
        lines.append(
            "- Leakage audit: no suspicious feature names were detected in transformed features."
        )
    else:
        lines.append(
            f"- Leakage audit: {len(suspicious)} suspicious transformed features require review."
        )
    return "\n".join(lines) + "\n"


def explain_validation_model(
    model: TrainedModel | str | Path,
    validation_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    feature_registry: pd.DataFrame | None = None,
    selected_config: dict[str, Any] | None = None,
    prefix: str = "",
    max_sample: int = 200,
    permutation_repeats: int = 3,
    seed: int = 42,
    shortlisted_boosted_model: TrainedModel | str | Path | None = None,
    shortlisted_boosted_validation_frame: pd.DataFrame | None = None,
) -> ExplainabilityResult:
    trained = _load_model(model)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_x, y = _sample_validation(
        validation_frame,
        feature_columns=trained.feature_columns,
        target_column=trained.target_column,
        max_sample=max_sample,
        seed=seed,
    )
    transformed_x = trained.preprocessor.transform(raw_x)
    transformed_x.to_csv(
        output_path / f"{prefix}representative_validation_features.csv",
        index=False,
    )
    metadata = _feature_metadata(list(transformed_x.columns), feature_registry)
    native = _native_importance(trained, list(transformed_x.columns))
    permutation = _permutation_importance(
        trained,
        raw_x,
        y,
        repeats=permutation_repeats,
        seed=seed,
    )
    shap_values, shap_plot, shap_note = _shap_or_equivalent(
        trained,
        transformed_x,
        output_path,
        prefix=prefix,
        selected_config=selected_config,
    )
    importance = pd.concat(
        [
            native.assign(method_scope="transformed_feature"),
            permutation.assign(method_scope="raw_feature"),
            shap_values.rename(columns={"mean_abs_value": "importance"}).assign(
                importance_type="mean_abs_shap_or_equivalent",
                method_scope="transformed_feature",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    importance = _attach_importance_metadata(importance, metadata, feature_registry)
    leakage = _leakage_audit(metadata)
    raw_permutation_plot = _raw_permutation_plot(
        importance,
        output_path,
        prefix=prefix,
        trained=trained,
        selected_config=selected_config,
        sample_size=len(raw_x),
    )
    summary = _summary(
        importance,
        leakage,
        shap_note=shap_note,
        output_units=_output_units(trained),
        trained=trained,
        sample_size=len(raw_x),
        selected_config=selected_config,
    )
    paths = {
        "feature_importance": output_path / f"{prefix}feature_importance.csv",
        "shap_mean_abs": output_path / f"{prefix}shap_mean_abs_values.csv",
        "feature_metadata": output_path / f"{prefix}feature_metadata.csv",
        "leakage_audit": output_path / f"{prefix}leakage_audit.csv",
        "summary": output_path / f"{prefix}explainability_summary.md",
        "representative_validation_features": output_path
        / f"{prefix}representative_validation_features.csv",
    }
    if shap_plot is not None:
        paths["shap_summary_beeswarm"] = shap_plot
    if raw_permutation_plot is not None:
        paths["raw_permutation_mae_delta_plot"] = raw_permutation_plot
    importance.to_csv(paths["feature_importance"], index=False)
    shap_values.to_csv(paths["shap_mean_abs"], index=False)
    metadata.to_csv(paths["feature_metadata"], index=False)
    leakage.to_csv(paths["leakage_audit"], index=False)
    paths["summary"].write_text(summary, encoding="utf-8")
    if trained.config.family == "ridge" and shortlisted_boosted_model is not None:
        boosted_frame = (
            shortlisted_boosted_validation_frame
            if shortlisted_boosted_validation_frame is not None
            else validation_frame
        )
        boosted = explain_validation_model(
            shortlisted_boosted_model,
            boosted_frame,
            output_dir=output_path,
            feature_registry=feature_registry,
            selected_config=selected_config,
            prefix=f"{prefix}shortlisted_boosted_",
            max_sample=max_sample,
            permutation_repeats=permutation_repeats,
            seed=seed,
        )
        for key, path in boosted.paths.items():
            paths[f"shortlisted_boosted_{key}"] = path
        summary += (
            "\nRidge champion note: shortlisted boosted candidate SHAP artifacts were also "
            "generated with prefix `shortlisted_boosted_`.\n"
        )
        paths["summary"].write_text(summary, encoding="utf-8")
    return ExplainabilityResult(
        feature_importance=importance,
        shap_values=shap_values,
        transformed_sample=transformed_x,
        summary_markdown=summary,
        paths=paths,
    )


__all__ = ["ExplainabilityError", "ExplainabilityResult", "explain_validation_model"]
