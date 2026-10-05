from __future__ import annotations

import os
from pathlib import Path
from string import Formatter
from urllib.parse import urlparse

import yaml

REQUIRED_SOURCE_KEYS = frozenset(
    {
        "wer_index",
        "ndcu_index",
        "boundaries_geojson",
        "population_workbook",
        "chirps_daily_template",
        "era5_archive",
    }
)
TEMPLATE_KEYS = {"chirps_daily_template": frozenset({"year", "date"})}
CONFIG_ENV_VAR = "DENGUE_SOURCES_CONFIG"


class SourceConfigError(ValueError):
    """Raised when configured public source URLs are missing or invalid."""


def default_config_path() -> Path:
    checkout = Path(__file__).resolve().parents[3] / "configs" / "sources.yaml"
    if checkout.is_file():
        return checkout
    return Path(__file__).resolve().parents[1] / "_configs" / "sources.yaml"


def _config_path() -> Path:
    override = os.environ.get(CONFIG_ENV_VAR)
    return Path(override).expanduser() if override else default_config_path()


def _load_sources() -> dict[str, str]:
    path = _config_path()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceConfigError(f"Source config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SourceConfigError(f"Source config is not valid YAML: {path}") from exc

    if not isinstance(payload, dict):
        raise SourceConfigError("Source config must be a mapping of source keys to URLs")

    keys = set(payload)
    missing = REQUIRED_SOURCE_KEYS - keys
    extra = keys - REQUIRED_SOURCE_KEYS
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing required keys: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"unknown keys: {', '.join(sorted(extra))}")
        raise SourceConfigError("; ".join(parts))

    sources: dict[str, str] = {}
    for key in sorted(REQUIRED_SOURCE_KEYS):
        value = payload[key]
        if not isinstance(value, str):
            raise SourceConfigError(f"Source URL for {key} must be a string")
        sources[key] = _validate_source_url(key, value)
    return sources


def _validate_source_url(key: str, value: str) -> str:
    if not value:
        raise SourceConfigError(f"Source URL for {key} must not be empty")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SourceConfigError(f"Source URL for {key} must be an HTTP(S) URL")

    expected_template_fields = TEMPLATE_KEYS.get(key)
    if expected_template_fields is None:
        fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(value)
            if field_name is not None
        }
        if fields:
            raise SourceConfigError(f"Source URL for {key} must not be a template")
        return value

    fields = {
        field_name for _, field_name, _, _ in Formatter().parse(value) if field_name is not None
    }
    if fields != expected_template_fields:
        expected = ", ".join(sorted(expected_template_fields))
        raise SourceConfigError(f"Source template for {key} must contain exactly: {expected}")
    try:
        value.format(**{field: "test" for field in expected_template_fields})
    except (KeyError, IndexError, ValueError) as exc:
        raise SourceConfigError(f"Source template for {key} is not format-compatible") from exc
    return value


def source_url(key: str) -> str:
    if key not in REQUIRED_SOURCE_KEYS:
        raise SourceConfigError(f"Unknown source URL key: {key}")
    return _load_sources()[key]
