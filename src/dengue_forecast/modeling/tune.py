from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dengue_forecast.modeling.dataset import FEATURE_SET_NAMES
from dengue_forecast.modeling.evaluate import (
    DevelopmentCVContext,
    EvaluationError,
    evaluate_model_config,
    refuse_development_after_holdout,
)
from dengue_forecast.modeling.train import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"


def _model_entry(
    name: str,
    family: str,
    objective: str = "regression",
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "objective": objective,
        "hyperparams": hyperparams or {},
        "seed": 42,
    }


def default_model_entries() -> list[dict[str, Any]]:
    configured = _development_config().get("screen", {}).get("models")
    if configured:
        return [
            _model_entry(
                name,
                str(raw["family"]),
                str(raw.get("objective", "regression")),
                dict(raw.get("hyperparams") or {}),
            )
            for name, raw in configured.items()
        ]
    safe_boosted = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.08, "n_jobs": 1}
    return [
        _model_entry("ridge", "ridge", hyperparams={"alpha": 1.0}),
        _model_entry("poisson", "poisson", "poisson", {"alpha": 1.0, "max_iter": 300}),
        _model_entry(
            "random_forest",
            "random_forest",
            hyperparams={
                "n_estimators": 128,
                "max_depth": 10,
                "min_samples_leaf": 3,
                "n_jobs": 1,
            },
        ),
        _model_entry("xgboost_square", "xgboost", "reg:squarederror", safe_boosted),
        _model_entry("xgboost_poisson", "xgboost", "count:poisson", safe_boosted),
        _model_entry("lightgbm_regression", "lightgbm", "regression", safe_boosted),
        _model_entry("lightgbm_poisson", "lightgbm", "poisson", safe_boosted),
    ]


def default_screen_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    configured_features = _development_config().get("feature_sets") or list(FEATURE_SET_NAMES)
    _validate_configured_feature_sets(configured_features)
    for feature_set in configured_features:
        for model in default_model_entries():
            configs.append(
                {
                    "config_id": f"{model['name']}__{feature_set}",
                    "feature_set": feature_set,
                    "model": {
                        "family": model["family"],
                        "objective": model["objective"],
                        "hyperparams": dict(model["hyperparams"]),
                        "seed": model["seed"],
                    },
                }
            )
    return configs


def model_config_from_dict(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        family=str(raw["family"]),
        objective=str(raw.get("objective", "regression")),
        hyperparams=dict(raw.get("hyperparams") or {}),
        seed=int(raw.get("seed", 42)),
    )


def default_tuning_plan() -> dict[str, Any]:
    configured = _development_config().get("tuning") or {}
    plan = {
        "seed": 42,
        "sampler": "TPESampler",
        "storage": "sqlite",
        "timeout_minutes": 90,
        "trials": {"xgboost": 12, "lightgbm": 12, "random_forest": 8},
        "search_spaces": {
            "random_forest": tuning_search_space("random_forest"),
            "xgboost": tuning_search_space("xgboost"),
            "lightgbm": tuning_search_space("lightgbm"),
        },
    }
    for key in ["seed", "sampler", "timeout_minutes", "trials"]:
        if key in configured:
            plan[key] = configured[key]
    plan["search_spaces"] = {
        "random_forest": tuning_search_space("random_forest"),
        "xgboost": tuning_search_space("xgboost"),
        "lightgbm": tuning_search_space("lightgbm"),
    }
    return plan


def _development_config(path: str | Path = DEFAULT_EXPERIMENTS_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    development = raw.get("development") or {}
    if not isinstance(development, dict):
        raise EvaluationError("configs/experiments.yaml development section must be a mapping")
    return development


def _validate_configured_feature_sets(feature_sets: list[str]) -> None:
    unknown = sorted(set(map(str, feature_sets)) - set(FEATURE_SET_NAMES))
    if unknown:
        raise EvaluationError(f"Configured feature sets are unknown: {unknown}")


def tuning_search_space(family: str) -> dict[str, Any]:
    if family == "random_forest":
        return {
            "n_estimators": [64, 256],
            "max_depth": [4, 10],
            "min_samples_leaf": [2, 12],
            "max_features": ["sqrt", 1.0],
        }
    if family == "xgboost":
        return {
            "objective": ["reg:squarederror", "count:poisson"],
            "n_estimators": [100, 400],
            "max_depth": [2, 6],
            "learning_rate": [0.02, 0.2],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "reg_alpha": [0.0, 2.0],
            "reg_lambda": [0.1, 10.0],
            "min_child_weight": [1.0, 10.0],
            "n_jobs": 1,
        }
    if family == "lightgbm":
        return {
            "objective": ["regression", "poisson"],
            "n_estimators": [100, 400],
            "max_depth": [2, 6],
            "num_leaves": [7, 31],
            "learning_rate": [0.02, 0.2],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "reg_alpha": [0.0, 2.0],
            "reg_lambda": [0.1, 10.0],
            "min_child_samples": [5, 30],
            "n_jobs": 1,
        }
    raise ValueError(f"Unknown tunable family: {family}")


def select_feature_sets_for_tuning(screen_summary: pd.DataFrame) -> dict[str, str]:
    required = {"model", "feature_set", "mean_fold_mae"}
    missing = required - set(screen_summary.columns)
    if missing:
        raise ValueError(f"Screen summary missing columns: {sorted(missing)}")
    selections: dict[str, str] = {}
    family_order = ["random_forest", "xgboost", "lightgbm"]
    for family in family_order:
        mask = screen_summary["model"].astype(str).str.contains(family, regex=False)
        candidates = screen_summary[mask].sort_values(
            ["mean_fold_mae", "feature_set"], kind="mergesort"
        )
        if not candidates.empty:
            selections[family] = str(candidates.iloc[0]["feature_set"])
    return selections


def run_screen(
    context: DevelopmentCVContext,
    *,
    output_root: str | Path,
    configs: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    output = Path(output_root)
    refuse_development_after_holdout(output)
    _persist_resolved_configuration(
        output,
        stage="screen",
        payload={"configs": configs or default_screen_configs()},
    )
    rows: list[pd.DataFrame] = []
    for config in configs or default_screen_configs():
        result = evaluate_model_config(
            context,
            config_id=str(config["config_id"]),
            model_config=model_config_from_dict(config["model"]),
            feature_set=str(config["feature_set"]),
            experiment_id=str(config["config_id"]),
            output_root=output_root,
        )
        if result.metrics["fold"].nunique() == len(context.folds):
            rows.append(result.metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_optuna_tuning(
    context: DevelopmentCVContext,
    *,
    output_root: str | Path,
    selected_feature_sets: dict[str, str],
    plan: dict[str, Any] | None = None,
) -> pd.DataFrame:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency verified in target env
        raise EvaluationError("Optuna is required for tuning") from exc

    tuning_plan = plan or default_tuning_plan()
    output = Path(output_root)
    refuse_development_after_holdout(output)
    trials_dir = output / "tuning"
    trials_dir.mkdir(parents=True, exist_ok=True)
    _persist_resolved_configuration(
        output,
        stage="tune",
        payload={"selected_feature_sets": selected_feature_sets, "plan": tuning_plan},
    )
    for family, feature_set in selected_feature_sets.items():
        storage = f"sqlite:///{trials_dir / f'{family}.sqlite3'}"
        sampler = optuna.samplers.TPESampler(seed=int(tuning_plan["seed"]))
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=storage,
            study_name=f"m2_{family}_{feature_set}",
            load_if_exists=True,
        )
        max_trials = int(tuning_plan["trials"][family])
        existing_trials = len(study.trials)
        remaining_trials = max(0, max_trials - existing_trials)

        current_family = family
        current_feature_set = feature_set

        def objective(
            trial: Any,
            *,
            family_name: str = current_family,
            tuned_feature_set: str = current_feature_set,
        ) -> float:
            model_config = _suggest_model_config(trial, family_name)
            config_id = f"tune_{family_name}_trial_{trial.number:03d}"
            result = evaluate_model_config(
                context,
                config_id=config_id,
                model_config=model_config,
                feature_set=tuned_feature_set,
                experiment_id=config_id,
                output_root=output,
                stage="tune",
            )
            if result.metrics["fold"].nunique() != len(context.folds):
                raise EvaluationError(
                    f"Trial {trial.number} completed partial folds only; excluded"
                )
            return float(result.metrics["mae"].mean())

        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=int(tuning_plan["timeout_minutes"]) * 60,
            catch=(Exception,),
        )
    trials = _collect_optuna_trials(trials_dir, selected_feature_sets)
    trials.to_csv(trials_dir / "trials.csv", index=False)
    (trials_dir / "search_space.json").write_text(
        json.dumps(tuning_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trials


def _collect_optuna_trials(trials_dir: Path, selected_feature_sets: dict[str, str]) -> pd.DataFrame:
    import optuna

    rows: list[dict[str, Any]] = []
    for family, feature_set in selected_feature_sets.items():
        storage = f"sqlite:///{trials_dir / f'{family}.sqlite3'}"
        study = optuna.load_study(
            study_name=f"m2_{family}_{feature_set}",
            storage=storage,
        )
        for trial in study.trials:
            rows.append(
                {
                    "family": family,
                    "feature_set": feature_set,
                    "trial_number": trial.number,
                    "status": trial.state.name.lower(),
                    "value": trial.value,
                    "params": json.dumps(trial.params, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def _persist_resolved_configuration(
    output_root: Path, *, stage: str, payload: dict[str, Any]
) -> None:
    config_dir = output_root / "experiments" / "resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{stage}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _suggest_model_config(trial: Any, family: str) -> ModelConfig:
    if family == "random_forest":
        return ModelConfig(
            family="random_forest",
            hyperparams={
                "n_estimators": trial.suggest_int("n_estimators", 64, 256),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 1.0]),
                "n_jobs": 1,
            },
        )
    if family == "xgboost":
        objective = trial.suggest_categorical("objective", ["reg:squarederror", "count:poisson"])
        return ModelConfig(
            family="xgboost",
            objective=objective,
            hyperparams={
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "n_jobs": 1,
            },
        )
    if family == "lightgbm":
        objective = trial.suggest_categorical("objective", ["regression", "poisson"])
        return ModelConfig(
            family="lightgbm",
            objective=objective,
            hyperparams={
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "n_jobs": 1,
            },
        )
    raise ValueError(f"Unknown tunable family: {family}")


__all__ = [
    "default_screen_configs",
    "default_tuning_plan",
    "model_config_from_dict",
    "run_optuna_tuning",
    "run_screen",
    "select_feature_sets_for_tuning",
    "tuning_search_space",
]
