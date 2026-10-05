from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd

from dengue_forecast.modeling.train import TrainingError, load_champion
from dengue_forecast.utils.hashing import sha256_file


HF_MODEL_REPO = "manthilaffs/sri-lanka-dengue-forecast"
TRUSTED_MANIFEST_FILENAME = "model_manifest_trusted.json"
TRUSTED_MANIFEST = Path(__file__).resolve().with_name(TRUSTED_MANIFEST_FILENAME)
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
MAX_HF_FILE_BYTES = 16 * 1024 * 1024


class PublicInferenceError(ValueError):
    """Raised when public inference inputs or model files fail validation."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    horizon_weeks: int
    feature_set: str
    model_sha256: str
    metadata_sha256: str
    feature_columns: tuple[str, ...]
    district_levels: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise PublicInferenceError(f"{path} must contain a JSON object")
    return data


def trusted_manifest_path() -> Path:
    resource = resources.files("dengue_forecast").joinpath(TRUSTED_MANIFEST_FILENAME)
    if isinstance(resource, Path):
        return resource
    return TRUSTED_MANIFEST


def _read_default_manifest() -> dict[str, Any]:
    resource = resources.files("dengue_forecast").joinpath(TRUSTED_MANIFEST_FILENAME)
    if not resource.is_file():
        raise PublicInferenceError(f"Packaged trusted manifest is missing: {TRUSTED_MANIFEST_FILENAME}")
    data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PublicInferenceError("Trusted manifest must contain a JSON object")
    return data


def load_trusted_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest = _read_default_manifest() if path is None else _read_json(Path(path))
    if manifest.get("schema") != "sri_lanka_dengue_forecast_model_manifest_v1":
        raise PublicInferenceError("Unsupported model manifest schema")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise PublicInferenceError("Trusted manifest must list at least one model")
    return manifest


def _safe_relative_file(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise PublicInferenceError("Manifest file path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PublicInferenceError(f"Unsafe manifest file path: {value!r}")
    if len(path.parts) < 2 or path.name not in {MODEL_FILENAME, METADATA_FILENAME}:
        raise PublicInferenceError(f"Unexpected manifest file path: {value!r}")
    return path


def _spec_from_entry(entry: dict[str, Any]) -> ModelSpec:
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PublicInferenceError("Model entry metadata must be an object")
    feature_columns = metadata.get("feature_columns")
    levels = (
        metadata.get("preprocessing_fit_state", {})
        .get("category_levels", {})
        .get("district_id", [])
    )
    if not isinstance(feature_columns, list) or not all(isinstance(x, str) for x in feature_columns):
        raise PublicInferenceError("Model metadata is missing feature_columns")
    if not isinstance(levels, list) or not all(isinstance(x, str) for x in levels):
        raise PublicInferenceError("Model metadata is missing district_id category levels")
    return ModelSpec(
        model_id=str(entry["model_id"]),
        horizon_weeks=int(entry["horizon_weeks"]),
        feature_set=str(entry["feature_set"]),
        model_sha256=str(entry["model_sha256"]),
        metadata_sha256=str(entry["metadata_sha256"]),
        feature_columns=tuple(feature_columns),
        district_levels=tuple(levels),
    )


def list_model_specs(manifest_path: str | Path | None = None) -> list[ModelSpec]:
    manifest = load_trusted_manifest(manifest_path)
    return [_spec_from_entry(entry) for entry in manifest["models"]]


def _model_entry(model_id: str, manifest_path: str | Path | None = None) -> dict[str, Any]:
    manifest = load_trusted_manifest(manifest_path)
    for entry in manifest["models"]:
        if entry.get("model_id") == model_id:
            return entry
    available = ", ".join(str(entry.get("model_id")) for entry in manifest["models"])
    raise PublicInferenceError(f"Unknown model_id {model_id!r}; available: {available}")


def verify_model_directory(model_root: str | Path, model_id: str, *, manifest_path: str | Path | None = None) -> ModelSpec:
    root = Path(model_root).resolve()
    entry = _model_entry(model_id, manifest_path)
    model_rel = _safe_relative_file(entry["files"]["model"])
    metadata_rel = _safe_relative_file(entry["files"]["metadata"])
    model_path = (root / model_rel).resolve()
    metadata_path = (root / metadata_rel).resolve()
    if root not in model_path.parents or root not in metadata_path.parents:
        raise PublicInferenceError("Resolved model files escaped the model root")
    if not model_path.exists() or not metadata_path.exists():
        raise PublicInferenceError(f"Model files for {model_id} are missing under {root}")
    if sha256_file(model_path) != entry["model_sha256"]:
        raise PublicInferenceError(f"Hash mismatch for {model_rel}")
    if sha256_file(metadata_path) != entry["metadata_sha256"]:
        raise PublicInferenceError(f"Hash mismatch for {metadata_rel}")
    return _spec_from_entry(entry)


def validate_features(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    expected = list(spec.feature_columns)
    missing = [column for column in expected if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    if missing:
        raise PublicInferenceError(f"Input is missing required feature columns: {missing}")
    if extra:
        raise PublicInferenceError(f"Input has unexpected columns: {extra}")
    ordered = frame.loc[:, expected].copy()
    if "district_id" in ordered:
        districts = ordered["district_id"].astype("string")
        unknown = sorted(set(districts.dropna()) - set(spec.district_levels))
        if districts.isna().any():
            raise PublicInferenceError("district_id contains missing values")
        if unknown:
            raise PublicInferenceError(f"district_id contains unknown values: {unknown}")
    numeric_columns = [column for column in expected if column != "district_id"]
    for column in numeric_columns:
        numeric = pd.to_numeric(ordered[column], errors="coerce")
        bad = numeric.isna() & ordered[column].notna()
        if bad.any():
            examples = ordered.loc[bad, column].head(3).tolist()
            raise PublicInferenceError(f"Numeric feature {column} contains non-numeric values: {examples}")
        if np.isinf(numeric).any():
            raise PublicInferenceError(f"Numeric feature {column} contains non-finite values")
        ordered[column] = numeric.astype("float64")
    return ordered


def predict_dataframe(
    frame: pd.DataFrame,
    *,
    model_root: str | Path,
    model_id: str,
    manifest_path: str | Path | None = None,
    trust_pickle: bool = False,
) -> pd.DataFrame:
    if not trust_pickle:
        raise PublicInferenceError(
            "Refusing to load pickle model without trust_pickle=True. "
            "Only use the published model files after verifying their hashes."
        )
    spec = verify_model_directory(model_root, model_id, manifest_path=manifest_path)
    features = validate_features(frame, spec)
    model_dir = Path(model_root) / model_id
    model = load_champion(model_dir)
    predictions = model.predict_next_week(features)
    if not np.isfinite(predictions).all():
        raise PublicInferenceError("Model returned non-finite predictions")
    return pd.DataFrame(
        {
            "model_id": model_id,
            "horizon_weeks": spec.horizon_weeks,
            "prediction_cases": predictions.astype("float64"),
        }
    )


def download_hf_snapshot(
    *,
    revision: str,
    destination: str | Path,
    repo_id: str = HF_MODEL_REPO,
    manifest_path: str | Path | None = None,
    max_file_bytes: int = MAX_HF_FILE_BYTES,
) -> Path:
    if not isinstance(revision, str) or len(revision) != 40 or not all(
        char in "0123456789abcdefABCDEF" for char in revision
    ):
        raise PublicInferenceError("Pass a 40-character hexadecimal Hugging Face commit SHA in --revision")
    if not _valid_repo_id(repo_id):
        raise PublicInferenceError(f"Unsafe Hugging Face repo id: {repo_id!r}")
    if max_file_bytes <= 0:
        raise PublicInferenceError("max_file_bytes must be positive")
    manifest = load_trusted_manifest(manifest_path)
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    files = _trusted_download_files(manifest)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        temp_root = Path(tmp)
        for rel, expected_sha in files:
            target = _safe_destination(dest, rel)
            if target.exists() and sha256_file(target) == expected_sha:
                continue
            url = _hf_resolve_url(repo_id=repo_id, revision=revision, relative_path=rel)
            temp_file = temp_root / hashlib.sha256(str(rel).encode("utf-8")).hexdigest()
            _download_one(url, temp_file, max_file_bytes=max_file_bytes)
            actual_sha = sha256_file(temp_file)
            if actual_sha != expected_sha:
                raise PublicInferenceError(
                    f"Hash mismatch for {rel}: expected {expected_sha}, got {actual_sha}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            _reject_symlink_path(target)
            temp_file.replace(target)
    return dest


def _valid_repo_id(repo_id: str) -> bool:
    parts = repo_id.split("/")
    return (
        len(parts) == 2
        and all(parts)
        and all(part not in {".", ".."} for part in parts)
        and all("/" not in part and "\\" not in part for part in parts)
    )


def _trusted_download_files(manifest: dict[str, Any]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for entry in manifest["models"]:
        entry_files = entry.get("files")
        if not isinstance(entry_files, dict):
            raise PublicInferenceError("Model entry files must be an object")
        for key, sha_key in [("model", "model_sha256"), ("metadata", "metadata_sha256")]:
            rel = _safe_relative_file(entry_files.get(key))
            expected_sha = str(entry.get(sha_key, ""))
            if len(expected_sha) != 64 or not all(
                char in "0123456789abcdefABCDEF" for char in expected_sha
            ):
                raise PublicInferenceError(f"Invalid SHA-256 for {rel}")
            if rel not in seen:
                files.append((rel, expected_sha.lower()))
                seen.add(rel)
    return files


def _safe_destination(root: Path, rel: Path) -> Path:
    target = (root / rel).resolve()
    if root not in target.parents:
        raise PublicInferenceError(f"Resolved download path escaped destination: {rel}")
    return target


def _reject_symlink_path(path: Path) -> None:
    for candidate in [path, *path.parents]:
        if candidate.exists() and candidate.is_symlink():
            raise PublicInferenceError(f"Refusing to write through symlink: {candidate}")


def _hf_resolve_url(*, repo_id: str, revision: str, relative_path: Path) -> str:
    safe_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    safe_path = quote(relative_path.as_posix(), safe="/")
    return f"https://huggingface.co/{safe_repo}/resolve/{revision}/{safe_path}"


def _download_one(url: str, output_path: Path, *, max_file_bytes: int) -> None:
    with urlopen(url, timeout=60) as response, output_path.open("wb") as file:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_file_bytes:
            raise PublicInferenceError(f"Remote file exceeds {max_file_bytes} byte limit")
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > max_file_bytes:
                raise PublicInferenceError(f"Remote file exceeds {max_file_bytes} byte limit")
            file.write(chunk)


def _cmd_list(args: argparse.Namespace) -> int:
    writer = csv.DictWriter(sys.stdout, fieldnames=["model_id", "horizon_weeks", "feature_set"])
    writer.writeheader()
    for spec in list_model_specs(args.manifest):
        writer.writerow(
            {
                "model_id": spec.model_id,
                "horizon_weeks": spec.horizon_weeks,
                "feature_set": spec.feature_set,
            }
        )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    specs = list_model_specs(args.manifest)
    for spec in specs:
        verify_model_directory(args.model_root, spec.model_id, manifest_path=args.manifest)
    print(json.dumps({"verified_models": [spec.model_id for spec in specs]}, indent=2))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.input)
    result = predict_dataframe(
        frame,
        model_root=args.model_root,
        model_id=args.model_id,
        manifest_path=args.manifest,
        trust_pickle=args.trust_pickle,
    )
    if args.output:
        result.to_csv(args.output, index=False)
    else:
        result.to_csv(sys.stdout, index=False)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    path = download_hf_snapshot(revision=args.revision, destination=args.destination, repo_id=args.repo_id)
    print(json.dumps({"downloaded_to": str(path), "revision": args.revision}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dengue-forecast-predict")
    parser.add_argument("--manifest", type=Path, default=TRUSTED_MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list-models")
    list_parser.set_defaults(func=_cmd_list)
    verify = sub.add_parser("verify")
    verify.add_argument("--model-root", type=Path, required=True)
    verify.set_defaults(func=_cmd_verify)
    predict = sub.add_parser("predict")
    predict.add_argument("--model-root", type=Path, required=True)
    predict.add_argument("--model-id", required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=None)
    predict.add_argument("--trust-pickle", action="store_true")
    predict.set_defaults(func=_cmd_predict)
    download = sub.add_parser("download")
    download.add_argument("--revision", required=True)
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument("--repo-id", default=HF_MODEL_REPO)
    download.set_defaults(func=_cmd_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (PublicInferenceError, TrainingError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
