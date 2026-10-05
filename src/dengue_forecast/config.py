from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from dengue_forecast.contracts import ContractError

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
if not CONFIG_DIR.is_dir():
    CONFIG_DIR = Path(__file__).resolve().parent / "_configs"
SCHEMA_DIR = CONFIG_DIR / "schemas"
TIMEZONE = "Asia/Colombo"


@dataclass(frozen=True)
class District:
    district_id: str
    district_name: str


DISTRICTS: tuple[District, ...] = (
    District("LK-AMP", "Ampara"),
    District("LK-ANU", "Anuradhapura"),
    District("LK-BAD", "Badulla"),
    District("LK-BAT", "Batticaloa"),
    District("LK-COL", "Colombo"),
    District("LK-GAL", "Galle"),
    District("LK-GAM", "Gampaha"),
    District("LK-HAM", "Hambantota"),
    District("LK-JAF", "Jaffna"),
    District("LK-KAL", "Kalutara"),
    District("LK-KAN", "Kandy"),
    District("LK-KEG", "Kegalle"),
    District("LK-KIL", "Kilinochchi"),
    District("LK-KUR", "Kurunegala"),
    District("LK-MAN", "Mannar"),
    District("LK-MAT", "Matale"),
    District("LK-MAR", "Matara"),
    District("LK-MON", "Monaragala"),
    District("LK-MUL", "Mullaitivu"),
    District("LK-NUW", "Nuwara Eliya"),
    District("LK-POL", "Polonnaruwa"),
    District("LK-PUT", "Puttalam"),
    District("LK-RAT", "Ratnapura"),
    District("LK-TRI", "Trincomalee"),
    District("LK-VAV", "Vavuniya"),
)

DISTRICT_BY_ID = {district.district_id: district for district in DISTRICTS}
DISTRICT_ID_BY_NAME = {district.district_name: district.district_id for district in DISTRICTS}


def normalize_alias_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.casefold()


@lru_cache
def load_district_aliases(path: Path | None = None) -> dict[str, tuple[str, str]]:
    alias_path = path or CONFIG_DIR / "district_aliases.yaml"
    with alias_path.open("r", encoding="utf-8") as file:
        raw: dict[str, dict[str, Any]] = yaml.safe_load(file)

    if len(raw) != 25:
        raise ContractError(f"Expected 25 canonical districts in alias map, found {len(raw)}")

    aliases: dict[str, tuple[str, str]] = {}
    expected_names = {district.district_name for district in DISTRICTS}
    for canonical_name, spec in raw.items():
        if canonical_name not in expected_names:
            raise ContractError(f"Unknown canonical district in aliases: {canonical_name}")
        district_id = spec["district_id"]
        if DISTRICT_ID_BY_NAME[canonical_name] != district_id:
            raise ContractError(f"District ID mismatch for {canonical_name}: {district_id}")
        for alias in spec.get("aliases", []):
            key = normalize_alias_key(str(alias))
            if key in aliases and aliases[key] != (district_id, canonical_name):
                raise ContractError(f"Alias maps to multiple districts: {alias}")
            aliases[key] = (district_id, canonical_name)
        aliases[normalize_alias_key(canonical_name)] = (district_id, canonical_name)
    return aliases


def normalize_district_name(value: str) -> tuple[str, str]:
    key = normalize_alias_key(value)
    try:
        return load_district_aliases()[key]
    except KeyError as exc:
        raise ContractError(f"Unknown district name: {value!r}") from exc


def canonical_district_frame():  # type: ignore[no-untyped-def]
    import pandas as pd

    return pd.DataFrame(
        {
            "district_id": [d.district_id for d in DISTRICTS],
            "district_name": [d.district_name for d in DISTRICTS],
        }
    )
