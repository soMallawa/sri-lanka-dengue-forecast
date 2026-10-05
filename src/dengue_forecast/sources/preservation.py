from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

MANIFEST_NAME = "raw-cache-manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class RawPreservationError(RuntimeError):
    """Raised when raw-cache preservation cannot prove byte identity."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_bytes(manifest).decode("utf-8"), encoding="utf-8")


def build_raw_manifest(raw_root: Path) -> dict[str, Any]:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise RawPreservationError(f"Raw root does not exist: {raw_root}")
    if not raw_root.is_dir():
        raise RawPreservationError(f"Raw root is not a directory: {raw_root}")

    payload_paths = _payload_paths(raw_root)
    metadata_paths = _metadata_paths(raw_root)
    payload_paths = _include_payloads_created_during_scan(payload_paths, metadata_paths)

    entries: list[dict[str, Any]] = []
    expected_metadata_paths = set()
    for payload_path in payload_paths:
        metadata_path = _metadata_path_for_payload(payload_path)
        if not metadata_path.exists():
            raise RawPreservationError(f"Missing metadata sidecar for {payload_path}")
        expected_metadata_paths.add(metadata_path)
        metadata = _load_metadata(metadata_path)
        _validate_payload_against_metadata(payload_path, metadata_path, metadata)
        entries.append(_payload_entry(raw_root, payload_path, metadata_path, metadata))

    orphan_metadata = sorted(set(metadata_paths) - expected_metadata_paths)
    if orphan_metadata:
        paths = ", ".join(_relative_path(raw_root, path) for path in orphan_metadata[:5])
        raise RawPreservationError(f"Metadata sidecar without payload: {paths}")

    for metadata_path in metadata_paths:
        metadata = _load_metadata(metadata_path)
        payload_name = str(metadata.get("filename", ""))
        payload_path = metadata_path.with_name(payload_name)
        entries.append(_metadata_entry(raw_root, metadata_path, payload_path, metadata))

    entries.sort(key=lambda item: str(item["relative_path"]))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": "dengue_forecast.raw_cache",
        "raw_root": "raw",
        "entry_count": len(entries),
        "payload_count": len(payload_paths),
        "metadata_count": len(metadata_paths),
        "entries": entries,
    }


def validate_raw_manifest(raw_root: Path, manifest: dict[str, Any]) -> dict[str, int]:
    raw_root = Path(raw_root)
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise RawPreservationError("Unsupported raw manifest schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RawPreservationError("Raw manifest entries must be a list")

    seen: set[str] = set()
    payload_count = 0
    metadata_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise RawPreservationError("Raw manifest entry must be an object")
        relative = _safe_manifest_path(str(entry.get("relative_path", "")))
        if relative in seen:
            raise RawPreservationError(f"Duplicate manifest entry: {relative}")
        seen.add(relative)
        path = raw_root / relative
        if not path.exists():
            raise RawPreservationError(f"Manifest file missing: {relative}")
        size = path.stat().st_size
        if size != int(entry.get("file_size", -1)):
            raise RawPreservationError(
                f"Size mismatch for {relative}: "
                f"manifest={entry.get('file_size')} actual={size}"
            )
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            raise RawPreservationError(
                f"SHA256 mismatch for {relative}: "
                f"manifest={entry.get('sha256')} actual={digest}"
            )
        role = entry.get("role")
        if role == "payload":
            payload_count += 1
            metadata_info = entry.get("metadata")
            if not isinstance(metadata_info, dict):
                raise RawPreservationError(f"Payload missing metadata reference: {relative}")
            metadata_relative = _safe_manifest_path(str(metadata_info.get("relative_path", "")))
            metadata_path = raw_root / metadata_relative
            if not metadata_path.exists():
                raise RawPreservationError(f"Payload metadata missing: {metadata_relative}")
            if sha256_file(metadata_path) != metadata_info.get("sha256"):
                raise RawPreservationError(f"Payload metadata SHA256 mismatch: {metadata_relative}")
        elif role == "metadata":
            metadata_count += 1
        else:
            raise RawPreservationError(f"Unsupported manifest role for {relative}: {role}")

    raw_paths = _payload_paths(raw_root) + _metadata_paths(raw_root)
    actual_paths = {_relative_path(raw_root, path) for path in raw_paths}
    missing_from_manifest = sorted(actual_paths - seen)
    if missing_from_manifest:
        raise RawPreservationError(f"Raw files missing from manifest: {missing_from_manifest[0]}")
    extra_in_manifest = sorted(seen - actual_paths)
    if extra_in_manifest:
        raise RawPreservationError(
            f"Manifest entry not present under raw root: {extra_in_manifest[0]}"
        )

    return {"entries": len(entries), "payloads": payload_count, "metadata": metadata_count}


def create_raw_archive(
    raw_root: Path,
    archive_path: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    archive_path = Path(archive_path)
    manifest = manifest or build_raw_manifest(raw_root)
    validate_raw_manifest(raw_root, manifest)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        archive_path.open("wb") as raw_file,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file,
        tarfile.open(mode="w", fileobj=gzip_file, format=tarfile.PAX_FORMAT) as archive,
    ):
        manifest_bytes = _json_bytes(manifest)
        _add_bytes(archive, MANIFEST_NAME, manifest_bytes)
        for entry in manifest["entries"]:
            relative = str(entry["relative_path"])
            _add_file(archive, raw_root / relative, f"raw/{relative}")

    archive_sha256 = sha256_file(archive_path)
    artifact = {
        "schema_version": 1,
        "archive": {
            "relative_path": archive_path.name,
            "sha256": archive_sha256,
            "file_size": archive_path.stat().st_size,
        },
        "manifest": {
            "name": MANIFEST_NAME,
            "sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest(),
            "entry_count": manifest["entry_count"],
            "payload_count": manifest["payload_count"],
            "metadata_count": manifest["metadata_count"],
        },
    }
    return artifact


def restore_archive(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    members: dict[str, bytes | None] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            _safe_archive_member(member.name)
            if member.isfile() and member.name == MANIFEST_NAME:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RawPreservationError("Archive manifest could not be read")
                members[MANIFEST_NAME] = extracted.read()
        if MANIFEST_NAME not in members:
            raise RawPreservationError(f"Archive missing {MANIFEST_NAME}")
        manifest = json.loads(members[MANIFEST_NAME].decode("utf-8"))
        expected = {f"raw/{entry['relative_path']}" for entry in manifest.get("entries", [])}
        actual = {
            member.name
            for member in archive.getmembers()
            if member.isfile() and member.name != MANIFEST_NAME
        }
        missing = sorted(expected - actual)
        if missing:
            raise RawPreservationError(f"Archive missing raw member: {missing[0]}")
        extra = sorted(actual - expected)
        if extra:
            raise RawPreservationError(f"Archive contains unmanifested member: {extra[0]}")
        archive.extractall(output_dir)

    validate_raw_manifest(output_dir / "raw", manifest)
    return manifest


def _payload_paths(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.name != ".gitkeep" and not path.name.endswith(".metadata.json")
    )


def _metadata_paths(raw_root: Path) -> list[Path]:
    return sorted(path for path in raw_root.rglob("*.metadata.json") if path.is_file())


def _metadata_path_for_payload(payload_path: Path) -> Path:
    stem_metadata = payload_path.with_name(f"{payload_path.stem}.metadata.json")
    explicit_metadata = payload_path.with_name(f"{payload_path.name}.metadata.json")
    if explicit_metadata.exists():
        return explicit_metadata
    return stem_metadata


def _include_payloads_created_during_scan(
    payload_paths: list[Path], metadata_paths: list[Path]
) -> list[Path]:
    payload_set = set(payload_paths)
    for metadata_path in metadata_paths:
        metadata = _load_metadata(metadata_path)
        filename = metadata.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            continue
        payload_path = metadata_path.with_name(filename)
        if payload_path.exists() and payload_path.is_file():
            payload_set.add(payload_path)
    return sorted(payload_set)


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RawPreservationError(f"Invalid metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise RawPreservationError(f"Metadata must be an object: {metadata_path}")
    return metadata


def _validate_payload_against_metadata(
    payload_path: Path, metadata_path: Path, metadata: dict[str, Any]
) -> None:
    expected_filename = metadata.get("filename")
    if expected_filename != payload_path.name:
        raise RawPreservationError(
            f"Metadata filename mismatch for {metadata_path}: "
            f"expected {payload_path.name}, got {expected_filename}"
        )
    actual_size = payload_path.stat().st_size
    expected_size = metadata.get("byte_length", metadata.get("content_length"))
    if expected_size is None:
        raise RawPreservationError(f"Metadata missing byte length for {payload_path}")
    if actual_size != int(expected_size):
        raise RawPreservationError(
            f"Size mismatch for {payload_path}: metadata={expected_size} actual={actual_size}"
        )
    actual_sha256 = sha256_file(payload_path)
    if actual_sha256 != metadata.get("sha256"):
        raise RawPreservationError(
            f"SHA256 mismatch for {payload_path}: "
            f"metadata={metadata.get('sha256')} actual={actual_sha256}"
        )


def _payload_entry(
    raw_root: Path, payload_path: Path, metadata_path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "relative_path": _relative_path(raw_root, payload_path),
        "role": "payload",
        "source": _source(metadata),
        "source_url": _source_url(metadata),
        "retrieved_at": str(metadata.get("retrieved_at", "")),
        "sha256": sha256_file(payload_path),
        "file_size": payload_path.stat().st_size,
        "data_domain": _data_domain(raw_root, payload_path),
        "source_period": _source_period(metadata),
        "metadata": {
            "relative_path": _relative_path(raw_root, metadata_path),
            "sha256": sha256_file(metadata_path),
            "file_size": metadata_path.stat().st_size,
        },
        "acquisition_parameters": _acquisition_parameters(metadata),
    }


def _metadata_entry(
    raw_root: Path, metadata_path: Path, payload_path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "relative_path": _relative_path(raw_root, metadata_path),
        "role": "metadata",
        "source": _source(metadata),
        "source_url": _source_url(metadata),
        "retrieved_at": str(metadata.get("retrieved_at", "")),
        "sha256": sha256_file(metadata_path),
        "file_size": metadata_path.stat().st_size,
        "data_domain": _data_domain(raw_root, metadata_path),
        "source_period": _source_period(metadata),
        "metadata_for": {
            "relative_path": _relative_path(raw_root, payload_path),
            "sha256": sha256_file(payload_path),
            "file_size": payload_path.stat().st_size,
        },
        "acquisition_parameters": _acquisition_parameters(metadata),
    }


def _source(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("source")
        or metadata.get("source_name")
        or metadata.get("source_type")
        or ""
    )


def _source_url(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("source_url")
        or metadata.get("upstream_url")
        or metadata.get("request_url")
        or ""
    )


def _source_period(metadata: dict[str, Any]) -> dict[str, Any]:
    if "date" in metadata:
        return {"date": metadata["date"]}
    if "start_date" in metadata or "end_date" in metadata:
        return {
            key: metadata[key]
            for key in ("start_date", "end_date")
            if key in metadata
        }
    period = metadata.get("period")
    if isinstance(period, dict):
        return {key: period[key] for key in sorted(period)}
    return {}


def _acquisition_parameters(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_type",
        "source_version",
        "source_url",
        "upstream_url",
        "request_url",
        "request_params",
        "request_bounds",
        "request_window",
        "subset_transform",
        "utc_day_semantic",
        "media_type",
        "parser_version",
        "period",
    ]
    return {key: metadata[key] for key in keys if key in metadata}


def _data_domain(raw_root: Path, path: Path) -> str:
    relative = Path(_relative_path(raw_root, path))
    return relative.parts[0] if relative.parts else ""


def _relative_path(root: Path, path: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _safe_manifest_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise RawPreservationError(f"Unsafe manifest path: {value}")
    return path.as_posix()


def _safe_archive_member(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise RawPreservationError(f"Unsafe archive member: {name}")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _add_file(archive: tarfile.TarFile, source_path: Path, archive_name: str) -> None:
    info = archive.gettarinfo(str(source_path), arcname=archive_name)
    _normalize_tarinfo(info)
    with source_path.open("rb") as file:
        archive.addfile(info, file)


def _add_bytes(archive: tarfile.TarFile, archive_name: str, content: bytes) -> None:
    info = tarfile.TarInfo(archive_name)
    info.size = len(content)
    _normalize_tarinfo(info)
    archive.addfile(info, io.BytesIO(content))


def _normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644 if info.isfile() else 0o755
    return info
