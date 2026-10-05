from __future__ import annotations

import json
import os
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dengue_forecast.public_inference as public_inference
from dengue_forecast.public_inference import (
    PublicInferenceError,
    download_hf_snapshot,
    list_model_specs,
    predict_dataframe,
    trusted_manifest_path,
    validate_features,
    verify_model_directory,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FITS = os.environ.get("DENGUE_FORECAST_SOURCE_FITS")


def _copy_model_root(tmp_path: Path, model_id: str) -> Path:
    if not SOURCE_FITS:
        pytest.skip("DENGUE_FORECAST_SOURCE_FITS is not set")
    root = tmp_path / "models"
    model_dir = root / model_id
    model_dir.mkdir(parents=True)
    source = Path(SOURCE_FITS)
    shutil.copy2(source / model_id / "model" / "model.joblib", model_dir / "model.joblib")
    shutil.copy2(source / model_id / "model" / "metadata.json", model_dir / "metadata.json")
    return root


def test_manifest_lists_four_selected_models() -> None:
    specs = list_model_specs()
    assert [spec.horizon_weeks for spec in specs] == [1, 2, 3, 4]
    assert specs[0].model_id == "h1__cases_only__ridge__final"
    assert trusted_manifest_path().name == "model_manifest_trusted.json"


def test_numeric_infinity_is_rejected() -> None:
    spec = list_model_specs()[0]
    frame = pd.read_csv(ROOT / "examples/SYNTHETIC/h1__cases_only__ridge__final_input.csv")
    numeric_column = next(column for column in spec.feature_columns if column != "district_id")
    frame.loc[0, numeric_column] = np.inf
    with pytest.raises(PublicInferenceError, match="non-finite"):
        validate_features(frame, spec)


def test_refuses_pickle_without_explicit_trust(tmp_path: Path) -> None:
    model_id = "h1__cases_only__ridge__final"
    model_root = _copy_model_root(tmp_path, model_id)
    frame = pd.read_csv(ROOT / "examples/SYNTHETIC/h1__cases_only__ridge__final_input.csv")
    with pytest.raises(PublicInferenceError, match="trust_pickle=True"):
        predict_dataframe(
            frame,
            model_root=model_root,
            model_id=model_id,
            manifest_path=ROOT / "model_manifest_trusted.json",
        )


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    model_id = "h1__cases_only__ridge__final"
    model_root = _copy_model_root(tmp_path, model_id)
    (model_root / model_id / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PublicInferenceError, match="Hash mismatch"):
        verify_model_directory(
            model_root,
            model_id,
            manifest_path=ROOT / "model_manifest_trusted.json",
        )


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "model_manifest_trusted.json").read_text(encoding="utf-8"))
    manifest["models"][0]["files"]["model"] = "../outside.joblib"
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PublicInferenceError, match="Unsafe manifest file path"):
        verify_model_directory(tmp_path, manifest["models"][0]["model_id"], manifest_path=bad_manifest)


def test_download_requires_full_immutable_revision(tmp_path: Path) -> None:
    for revision in ["main", "master", "abc123", "g" * 40]:
        with pytest.raises(PublicInferenceError, match="40-character hexadecimal"):
            download_hf_snapshot(revision=revision, destination=tmp_path)


def test_download_uses_resolve_urls_and_rejects_bad_hash(tmp_path: Path, monkeypatch) -> None:
    manifest = json.loads((ROOT / "model_manifest_trusted.json").read_text(encoding="utf-8"))
    entry = manifest["models"][0]
    entry["model_sha256"] = "0" * 64
    entry["metadata_sha256"] = "1" * 64
    manifest["models"] = [entry]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    seen_urls: list[str] = []

    class Response(BytesIO):
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(url: str, timeout: int = 0):
        seen_urls.append(url)
        return Response(b"nope")

    monkeypatch.setattr(public_inference, "urlopen", fake_urlopen)

    revision = "a" * 40
    with pytest.raises(PublicInferenceError, match="Hash mismatch"):
        download_hf_snapshot(
            revision=revision,
            destination=tmp_path / "models",
            repo_id="owner/repo",
            manifest_path=manifest_path,
        )
    assert seen_urls
    assert all(f"/resolve/{revision}/" in url for url in seen_urls)
    assert not any("/archive/" in url for url in seen_urls)


def test_missing_columns_are_rejected(tmp_path: Path) -> None:
    model_id = "h1__cases_only__ridge__final"
    model_root = _copy_model_root(tmp_path, model_id)
    frame = pd.read_csv(ROOT / "examples/SYNTHETIC/h1__cases_only__ridge__final_input.csv")
    frame = frame.drop(columns=[frame.columns[0]])
    with pytest.raises(PublicInferenceError, match="missing required feature columns"):
        predict_dataframe(
            frame,
            model_root=model_root,
            model_id=model_id,
            manifest_path=ROOT / "model_manifest_trusted.json",
            trust_pickle=True,
        )


def test_actual_inference_returns_finite_predictions(tmp_path: Path) -> None:
    model_id = "h1__cases_only__ridge__final"
    model_root = _copy_model_root(tmp_path, model_id)
    frame = pd.read_csv(ROOT / "examples/SYNTHETIC/h1__cases_only__ridge__final_input.csv")
    out = predict_dataframe(
        frame,
        model_root=model_root,
        model_id=model_id,
        manifest_path=ROOT / "model_manifest_trusted.json",
        trust_pickle=True,
    )
    assert list(out.columns) == ["model_id", "horizon_weeks", "prediction_cases"]
    assert out["prediction_cases"].notna().all()
    assert out["prediction_cases"].ge(0).all()
