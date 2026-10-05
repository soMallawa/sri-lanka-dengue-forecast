from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from dengue_forecast.utils.hashing import sha256_file


class RegistryError(RuntimeError):
    """Raised when an immutable experiment registry operation cannot be completed."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    experiment_dir: Path
    metadata: dict[str, Any]
    artifact_sha256: dict[str, str]


class _FileLock:
    def __init__(self, path: Path, *, timeout_seconds: float = 10.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.fd: int | None = None

    def __enter__(self) -> _FileLock:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise RegistryError(f"Timed out acquiring registry lock {self.path}") from None
                time.sleep(0.05)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        with suppress(FileNotFoundError):
            self.path.unlink()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, default=_json_default, sort_keys=True, separators=(",", ":"))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _metric_value(metrics: dict[str, Any], name: str) -> Any:
    return _first_present(
        metrics.get(name),
        metrics.get(name.upper()),
        metrics.get(name.lower()),
        _nested_get(metrics, "validation", name),
        _nested_get(metrics, "val", name),
        _nested_get(metrics, "test", name),
    )


def _require_registry_metadata(row: dict[str, Any]) -> None:
    missing = [
        key
        for key, value in row.items()
        if value is None or value == "" or (isinstance(value, float) and pd.isna(value))
    ]
    if missing:
        raise RegistryError(
            "Missing required registry metadata: "
            + ", ".join(sorted(missing))
            + ". Pass complete production metadata or create the registry in non-strict mode."
        )


SOURCE_MANIFEST_VERSION = 1
SOURCE_MANIFEST_PATTERNS = (
    ("src", "**/*.py"),
    ("scripts", "*.py"),
    ("configs", "*.yaml"),
)
SOURCE_MANIFEST_FILES = ("pyproject.toml", "uv.lock")


def _git_output(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _git_tracked_source_paths(source_root: Path) -> list[Path] | None:
    pathspecs = [
        ":(glob)src/**/*.py",
        ":(glob)scripts/*.py",
        ":(glob)configs/*.yaml",
        *SOURCE_MANIFEST_FILES,
    ]
    output = _git_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *pathspecs],
        cwd=source_root,
    )
    if output == "unavailable":
        return None
    return [source_root / line for line in output.splitlines() if line]


def _fallback_source_paths(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, pattern in SOURCE_MANIFEST_PATTERNS:
        base = source_root / directory
        if base.exists():
            paths.extend(base.glob(pattern))
    paths.extend(source_root / name for name in SOURCE_MANIFEST_FILES)
    return paths


def _source_manifest(source_root: Path | None = None) -> dict[str, Any]:
    root = (source_root or Path.cwd()).resolve()
    paths = _git_tracked_source_paths(root)
    if paths is None:
        paths = _fallback_source_paths(root)

    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    files.sort(key=lambda item: item["path"])
    return {
        "version": SOURCE_MANIFEST_VERSION,
        "include": [
            "src/**/*.py",
            "scripts/*.py",
            "configs/*.yaml",
            *SOURCE_MANIFEST_FILES,
        ],
        "files": files,
    }


def _source_identity(source_root: Path | None = None) -> dict[str, Any]:
    root = (source_root or Path.cwd()).resolve()
    commit = _git_output(["git", "rev-parse", "HEAD"], cwd=root)
    manifest = _source_manifest(root)
    source_hash = _sha256_text(_canonical_json(manifest))
    return {
        "git_commit": commit,
        "source_hash": source_hash,
        "source_manifest": manifest,
        "source_root": str(root),
        "source_manifest_version": SOURCE_MANIFEST_VERSION,
        "source_manifest_file_count": len(manifest["files"]),
        "dirty_source_digest": source_hash,
    }


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    for name in ("sklearn", "xgboost", "lightgbm"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "unavailable"
    return versions


class ExperimentRegistry:
    def __init__(self, root: str | Path, *, strict: bool = True) -> None:
        self.root = Path(root)
        self.experiments_dir = self.root / "experiments"
        self.registry_path = self.experiments_dir / "registry.parquet"
        self.lock_path = self.experiments_dir / ".registry.lock"
        self.strict = strict

    def create_experiment(
        self,
        *,
        experiment_id: str,
        config: dict[str, Any],
        features: dict[str, Any],
        metrics: dict[str, Any],
        training_period: dict[str, Any],
        artifact_paths: dict[str, str | Path],
        row_identity: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        predictions: pd.DataFrame | None = None,
        environment: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        if not experiment_id or "/" in experiment_id or "\\" in experiment_id:
            raise RegistryError("experiment_id must be a non-empty path-safe identifier")
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        with _FileLock(self.lock_path):
            existing = self._read_registry()
            if not existing.empty and experiment_id in set(existing["experiment_id"].astype(str)):
                raise RegistryError(f"Experiment {experiment_id} already exists")
            experiment_dir = self.experiments_dir / experiment_id
            try:
                experiment_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError as exc:
                raise RegistryError(f"Experiment {experiment_id} already exists") from exc

            if self.strict and not artifact_paths:
                raise RegistryError("At least one artifact path is required in strict mode")

            artifact_sha = {}
            resolved_artifact_paths: dict[str, str] = {}
            for name, path in artifact_paths.items():
                artifact_path = Path(path)
                if not artifact_path.exists():
                    raise RegistryError(f"Artifact path does not exist: {artifact_path}")
                if not artifact_path.is_file():
                    raise RegistryError(f"Artifact path is not a file: {artifact_path}")
                artifact_sha[name] = sha256_file(artifact_path)
                resolved_artifact_paths[name] = str(artifact_path)
            source = _source_identity()
            feature_set = _first_present(
                features.get("feature_set"),
                features.get("name"),
                config.get("feature_set"),
                _canonical_json(features.get("columns")) if features.get("columns") else None,
            )
            model_name = _first_present(
                config.get("model"),
                config.get("model_name"),
                config.get("family"),
                _nested_get(config, "model", "name"),
            )
            dataset_hash = _first_present(
                config.get("dataset_hash"),
                config.get("dataset_sha256"),
                config.get("registry_sha256"),
                _nested_get(config, "data", "dataset_hash"),
                _nested_get(config, "dataset", "hash"),
            )
            train_start = _first_present(
                training_period.get("train_start"),
                training_period.get("training_start"),
                _nested_get(training_period, "train", "start"),
            )
            train_end = _first_present(
                training_period.get("train_end"),
                training_period.get("training_end"),
                _nested_get(training_period, "train", "end"),
            )
            validation_start = _first_present(
                training_period.get("validation_start"),
                training_period.get("val_start"),
                _nested_get(training_period, "validation", "start"),
                _nested_get(training_period, "val", "start"),
            )
            validation_end = _first_present(
                training_period.get("validation_end"),
                training_period.get("val_end"),
                _nested_get(training_period, "validation", "end"),
                _nested_get(training_period, "val", "end"),
            )
            fold = _first_present(
                training_period.get("fold"),
                training_period.get("fold_id"),
                (row_identity or {}).get("fold"),
                (row_identity or {}).get("fold_id"),
                config.get("fold"),
            )
            seed = _first_present((runtime or {}).get("seed"), config.get("seed"))
            artifact_path = _first_present(*resolved_artifact_paths.values())
            created_at = datetime.now(UTC).isoformat()
            required_row = {
                "experimentid": experiment_id,
                "git_commit": source["git_commit"],
                "dataset_hash": dataset_hash,
                "feature_set": feature_set,
                "model": model_name,
                "params_hash": _sha256_text(_canonical_json(config.get("hyperparams", config))),
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "fold": fold,
                "seed": seed,
                "mae": _metric_value(metrics, "mae"),
                "rmse": _metric_value(metrics, "rmse"),
                "r2": _metric_value(metrics, "r2"),
                "poisson": _metric_value(metrics, "poisson"),
                "bias": _metric_value(metrics, "bias"),
                "topdecile": _metric_value(metrics, "topdecile"),
                "artifact_path": artifact_path,
                "created_at": created_at,
            }
            if self.strict:
                _require_registry_metadata(required_row)
            metadata = {
                "experiment_id": experiment_id,
                "timestamp_utc": created_at,
                "created_at": created_at,
                **source,
                "config_hash": _sha256_text(_canonical_json(config)),
                "params_hash": required_row["params_hash"],
                "features_hash": _sha256_text(_canonical_json(features)),
                "metrics_hash": _sha256_text(_canonical_json(metrics)),
                "dataset_hash": str(dataset_hash or ""),
                "dataset_sha256": str(config.get("dataset_sha256", "")),
                "registry_sha256": str(config.get("registry_sha256", "")),
                "artifact_sha256": artifact_sha,
                "artifact_paths": resolved_artifact_paths,
                "row_identity": row_identity or {},
                "training_period": training_period,
                "runtime": runtime or {},
                "library_versions": _library_versions(),
            }
            if environment:
                metadata["environment_overrides"] = environment

            self._write_json_atomic(experiment_dir / "config.json", config)
            self._write_json_atomic(experiment_dir / "features.json", features)
            self._write_json_atomic(experiment_dir / "metrics.json", metrics)
            self._write_json_atomic(experiment_dir / "training_period.json", training_period)
            self._write_json_atomic(
                experiment_dir / "environment.json", metadata["library_versions"]
            )
            self._write_json_atomic(experiment_dir / "metadata.json", metadata)
            if predictions is not None:
                predictions.to_parquet(experiment_dir / "predictions.parquet", index=False)
            (experiment_dir / "README.md").write_text(
                f"# Experiment {experiment_id}\n\n"
                "Immutable Milestone 2 model experiment artifact.\n"
            )

            record = pd.DataFrame(
                [
                    {
                        **required_row,
                        "experiment_id": experiment_id,
                        "timestamp_utc": created_at,
                        "artifact_dir": str(experiment_dir),
                        "git_commit": metadata["git_commit"],
                        "dirty_source_digest": metadata["dirty_source_digest"],
                        "source_hash": metadata["source_hash"],
                        "source_manifest_version": metadata["source_manifest_version"],
                        "source_manifest_file_count": metadata["source_manifest_file_count"],
                        "config_hash": metadata["config_hash"],
                        "params_hash": metadata["params_hash"],
                        "features_hash": metadata["features_hash"],
                        "metrics_hash": metadata["metrics_hash"],
                        "dataset_sha256": metadata["dataset_sha256"],
                        "artifact_paths": _canonical_json(resolved_artifact_paths),
                    }
                ]
            )
            updated = pd.concat([existing, record], ignore_index=True)
            tmp = self.registry_path.with_suffix(".parquet.tmp")
            updated.to_parquet(tmp, index=False)
            tmp.replace(self.registry_path)
        return ExperimentRecord(experiment_id, experiment_dir, metadata, artifact_sha)

    def _read_registry(self) -> pd.DataFrame:
        if not self.registry_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.registry_path)

    @staticmethod
    def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, default=_json_default, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)


__all__ = ["ExperimentRecord", "ExperimentRegistry", "RegistryError"]
