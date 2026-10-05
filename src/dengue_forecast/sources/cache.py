from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; dengue-forecast/0.1; +https://www.epid.gov.lk/)"


class SourceCacheError(RuntimeError):
    """Raised when a source cannot be cached safely."""


@dataclass(frozen=True)
class CachedSource:
    path: Path
    metadata_path: Path
    sha256: str
    source_url: str
    source_name: str
    media_type: str
    retrieved_at: str


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_component(value: str, *, fallback: str = "source") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip(".-_")
    return cleaned[:140] or fallback


def _extension_from_url(url: str, media_type: str) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(media_type.split(";")[0].strip())
    return guessed or ".bin"


def _looks_like_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF-") and b"%%EOF" in content[-4096:]


class SourceCache:
    """Content-addressed immutable cache for official source files."""

    def __init__(
        self,
        root: Path,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        rate_limit_seconds: float = 0.5,
        timeout_seconds: float = 30.0,
        retries: int = 2,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def adopt_file(
        self,
        source_path: Path,
        *,
        source_url: str,
        source_name: str,
        source_type: str,
        period: dict[str, Any],
        parser_version: str,
        retrieved_at: str | None = None,
    ) -> CachedSource:
        content = Path(source_path).read_bytes()
        media_type = mimetypes.guess_type(str(source_path))[0] or "application/octet-stream"
        return self._write_content(
            content,
            source_url=source_url,
            source_name=source_name,
            source_type=source_type,
            period=period,
            parser_version=parser_version,
            media_type=media_type,
            retrieved_at=retrieved_at or _utc_now(),
        )

    def download(
        self,
        url: str,
        *,
        source_name: str,
        source_type: str,
        period: dict[str, Any],
        parser_version: str,
        offline: bool = False,
        referer: str | None = None,
        force: bool = False,
    ) -> CachedSource:
        if not force:
            cached = self.find_by_url(url)
            if cached is not None:
                return cached

        if offline:
            cached = self.find_by_url(url)
            if cached is None:
                raise SourceCacheError(f"Offline cache miss for {url}")
            return cached

        headers = {"User-Agent": self.user_agent}
        if referer:
            headers["Referer"] = referer

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.rate_limit_seconds * attempt)
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
                if response.status_code >= 500 and attempt < self.retries:
                    last_error = SourceCacheError(
                        f"transient HTTP {response.status_code} for {url}"
                    )
                    continue
                if response.status_code != 200:
                    raise SourceCacheError(f"HTTP {response.status_code} for {url}")
                media_type = response.headers.get("Content-Type", "application/octet-stream")
                return self._write_content(
                    response.content,
                    source_url=url,
                    source_name=source_name,
                    source_type=source_type,
                    period=period,
                    parser_version=parser_version,
                    media_type=media_type,
                    retrieved_at=_utc_now(),
                )
            except (requests.RequestException, SourceCacheError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
        raise SourceCacheError(str(last_error or f"failed to download {url}"))

    def find_by_url(self, url: str) -> CachedSource | None:
        for metadata_path in self.root.glob("*.metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("source_url") == url:
                path = self.root / str(metadata["filename"])
                if path.exists():
                    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                    expected_sha256 = str(metadata["sha256"])
                    if actual_sha256 != expected_sha256:
                        raise SourceCacheError(
                            f"Cached source hash mismatch for {path}: "
                            f"metadata={expected_sha256} actual={actual_sha256}"
                        )
                    return CachedSource(
                        path=path,
                        metadata_path=metadata_path,
                        sha256=expected_sha256,
                        source_url=url,
                        source_name=str(metadata["source_name"]),
                        media_type=str(metadata["media_type"]),
                        retrieved_at=str(metadata["retrieved_at"]),
                    )
        return None

    def _write_content(
        self,
        content: bytes,
        *,
        source_url: str,
        source_name: str,
        source_type: str,
        period: dict[str, Any],
        parser_version: str,
        media_type: str,
        retrieved_at: str,
    ) -> CachedSource:
        ext = _extension_from_url(source_url, media_type)
        is_pdf = ext == ".pdf" or "pdf" in media_type.lower()
        if is_pdf and not _looks_like_pdf(content):
            raise SourceCacheError(f"PDF media check failed for {source_url}")

        sha256 = hashlib.sha256(content).hexdigest()
        stem_parts = [source_name]
        if period.get("source_year") is not None and period.get("source_week") is not None:
            stem_parts.append(f"{period['source_year']}_w{int(period['source_week']):02d}")
        elif period.get("issue_year") is not None and period.get("issue_week") is not None:
            stem_parts.append(f"issue_{period['issue_year']}_w{int(period['issue_week']):02d}")
        else:
            stem_parts.append(Path(unquote(urlparse(source_url).path)).stem)
        stem = _safe_component("_".join(str(part) for part in stem_parts))
        filename = f"{stem}_{sha256[:12]}{ext}"
        path = self.root / filename
        metadata_path = self.root / f"{path.stem}.metadata.json"

        if not path.exists():
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_bytes(content)
            os.replace(tmp_path, path)
        else:
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != sha256:
                raise SourceCacheError(
                    f"Existing cache file hash mismatch for {path}: "
                    f"expected {sha256}, found {actual_sha256}"
                )
        metadata = {
            "filename": filename,
            "sha256": sha256,
            "source_url": source_url,
            "source_name": source_name,
            "source_type": source_type,
            "retrieved_at": retrieved_at,
            "media_type": media_type.split(";")[0],
            "content_length": len(content),
            "period": period,
            "parser_version": parser_version,
        }
        if not metadata_path.exists():
            tmp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            tmp_metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(tmp_metadata_path, metadata_path)
        else:
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            if stored.get("sha256") != sha256 or stored.get("filename") != filename:
                raise SourceCacheError(
                    f"Existing cache metadata does not match immutable file {path}"
                )
        return CachedSource(
            path=path,
            metadata_path=metadata_path,
            sha256=sha256,
            source_url=source_url,
            source_name=source_name,
            media_type=str(metadata["media_type"]),
            retrieved_at=retrieved_at,
        )

    @staticmethod
    def metadata(cached: CachedSource) -> dict[str, Any]:
        return asdict(cached)
