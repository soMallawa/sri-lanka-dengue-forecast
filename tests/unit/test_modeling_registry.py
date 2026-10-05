from __future__ import annotations

import json

import pandas as pd
import pytest

from dengue_forecast.modeling.registry import ExperimentRegistry, RegistryError, _source_identity


def _valid_inputs(tmp_path):
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"trusted local model artifact")
    return {
        "config": {
            "family": "ridge",
            "seed": 42,
            "dataset_hash": "dataset-abc",
            "hyperparams": {"alpha": 0.1},
        },
        "features": {"feature_set": "lagged-baseline", "columns": ["district_id", "cases_lag_1"]},
        "metrics": {
            "mae": 1.25,
            "rmse": 1.5,
            "r2": 0.2,
            "poisson": 0.8,
            "bias": -0.1,
            "topdecile": 0.7,
        },
        "training_period": {
            "train_start": "2020-01-04",
            "train_end": "2020-02-01",
            "validation_start": "2020-02-08",
            "validation_end": "2020-02-15",
            "fold": "fold-1",
        },
        "artifact_paths": {"model": model_file},
        "row_identity": {"fold_id": "fold-1", "row_key_digest": "abc"},
        "runtime": {"seconds": 0.1, "seed": 42},
    }


def test_registry_creates_unique_immutable_experiment_and_appends_records(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path)
    valid = _valid_inputs(tmp_path)

    first = registry.create_experiment(
        experiment_id="exp-a",
        **valid,
    )

    assert first.experiment_dir.exists()
    assert (first.experiment_dir / "README.md").exists()
    assert json.loads((first.experiment_dir / "config.json").read_text())["seed"] == 42
    assert first.metadata["artifact_sha256"]["model"] == first.artifact_sha256["model"]
    assert first.metadata["git_commit"]
    assert "dirty_source_digest" in first.metadata

    with pytest.raises(RegistryError, match="already exists"):
        registry.create_experiment(
            experiment_id="exp-a",
            **valid,
        )

    second_valid = _valid_inputs(tmp_path)
    second = registry.create_experiment(
        experiment_id="exp-b",
        **second_valid,
    )
    assert second.experiment_id == "exp-b"

    records = pd.read_parquet(tmp_path / "experiments" / "registry.parquet")
    assert records["experiment_id"].tolist() == ["exp-a", "exp-b"]
    assert records["experimentid"].tolist() == ["exp-a", "exp-b"]
    assert records["artifact_dir"].is_unique
    assert set(
        [
            "git_commit",
            "dataset_hash",
            "feature_set",
            "model",
            "params_hash",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "fold",
            "seed",
            "mae",
            "rmse",
            "r2",
            "poisson",
            "bias",
            "topdecile",
            "artifact_path",
            "created_at",
            "source_hash",
            "source_manifest_version",
        ]
    ).issubset(records.columns)


def test_registry_rejects_duplicate_id_even_if_record_exists_without_directory(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path)
    (tmp_path / "experiments").mkdir()
    pd.DataFrame([{"experiment_id": "exp-a", "artifact_dir": "missing"}]).to_parquet(
        tmp_path / "experiments" / "registry.parquet", index=False
    )

    with pytest.raises(RegistryError, match="already exists"):
        registry.create_experiment(
            experiment_id="exp-a",
            **_valid_inputs(tmp_path),
        )


def test_registry_rejects_missing_and_non_file_artifacts(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path)
    valid = _valid_inputs(tmp_path)
    valid["artifact_paths"] = {"model": tmp_path / "missing.joblib"}
    with pytest.raises(RegistryError, match="does not exist"):
        registry.create_experiment(experiment_id="exp-a", **valid)

    valid = _valid_inputs(tmp_path)
    directory = tmp_path / "artifact-dir"
    directory.mkdir()
    valid["artifact_paths"] = {"model": directory}
    with pytest.raises(RegistryError, match="not a file"):
        registry.create_experiment(experiment_id="exp-b", **valid)


def test_registry_strict_mode_rejects_missing_required_metadata(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path)
    valid = _valid_inputs(tmp_path)
    valid["metrics"] = {"mae": 1.0}
    with pytest.raises(RegistryError, match="Missing required registry metadata"):
        registry.create_experiment(experiment_id="exp-a", **valid)


def test_source_identity_changes_when_same_path_content_changes(tmp_path, monkeypatch) -> None:
    source = tmp_path / "src" / "pkg"
    source.mkdir(parents=True)
    module = source / "modeling.py"
    module.write_text("VALUE = 1\n")
    monkeypatch.chdir(tmp_path)

    first = _source_identity()
    module.write_text("VALUE = 2\n")
    second = _source_identity()

    assert first["source_hash"] != second["source_hash"]
    assert first["source_manifest"]["files"][0]["path"] == "src/pkg/modeling.py"


def test_source_identity_is_deterministic_without_git(tmp_path, monkeypatch) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "train.py").write_text("print('train')\n")
        (root / "configs").mkdir()
        (root / "configs" / "model.yaml").write_text("seed: 42\n")
        (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setenv("PATH", "")

    monkeypatch.chdir(first_root)
    first = _source_identity()
    monkeypatch.chdir(second_root)
    second = _source_identity()

    assert first["source_hash"] == second["source_hash"]
    assert first["source_manifest"] == second["source_manifest"]
    assert first["git_commit"] == "unavailable"
    assert [item["path"] for item in first["source_manifest"]["files"]] == [
        "configs/model.yaml",
        "pyproject.toml",
        "scripts/train.py",
    ]
