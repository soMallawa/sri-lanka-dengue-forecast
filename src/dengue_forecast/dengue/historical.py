from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from dengue_forecast.contracts import ContractError
from dengue_forecast.dengue.discover import (
    EPID_ARCHIVE_URL,
    NDCU_WEEKLY_URL,
    discover_dengue_reports,
    discover_dengue_reports_from_html,
)
from dengue_forecast.dengue.download import download_dengue_report
from dengue_forecast.dengue.models import PARSER_VERSION, ReportDescriptor, SourceDocument
from dengue_forecast.dengue.normalize import normalize_parsed_reports
from dengue_forecast.dengue.parse import parse_dengue_document, select_parser
from dengue_forecast.sources.cache import DEFAULT_USER_AGENT, SourceCache, SourceCacheError

DISCOVERY_COLUMNS = [
    "source",
    "title",
    "year",
    "week",
    "publication_date",
    "source_url",
    "document_filename",
    "document_sha256",
    "document_type",
    "discovery_status",
    "parser_family",
    "status",
    "download_status",
    "source_retrieved_at",
    "raw_path",
    "quarantine_reason",
    "quarantine_details",
    "attempted_parser",
    "possible_remediation",
    "parsed_source_year",
    "parsed_source_week",
    "parsed_week_start_date",
    "parsed_week_end_date",
    "accepted_rows",
    "observed_rows",
]

QUARANTINE_REASONS = {
    "SOURCE_ERROR",
    "DOWNLOAD_ERROR",
    "PARSER_ERROR",
    "REPORTING_PERIOD_AMBIGUITY",
    "CUMULATIVE_WEEKLY_AMBIGUITY",
    "DUPLICATE_OR_REVISION",
    "DISTRICT_TABLE_INCOMPLETE",
    "SOURCE_CONFLICT",
    "UNKNOWN",
}

CONFLICT_COLUMNS = [
    "district",
    "week",
    "source_a",
    "value_a",
    "source_b",
    "value_b",
    "chosen_value",
    "reason",
]

DEFAULT_BASELINE_PATH = Path("docs/baselines/phase-a-accepted-dengue.json")
HISTORICAL_END_YEAR = 2025
EPID_BOUNDARY_ISSUES = {1, 2}


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.metadata.json")


def _empty_index() -> pd.DataFrame:
    return pd.DataFrame(columns=DISCOVERY_COLUMNS)


def _cohort_start() -> date:
    return date(2010, 1, 1)


def _cohort_end() -> date:
    return date(2025, 12, 31)


def _record_from_descriptor(descriptor: ReportDescriptor) -> dict[str, Any]:
    return {
        "source": descriptor.source,
        "title": descriptor.title,
        "year": descriptor.issue_year,
        "week": descriptor.issue_week,
        "publication_date": None,
        "source_url": descriptor.url,
        "document_filename": None,
        "document_sha256": None,
        "document_type": descriptor.document_type,
        "discovery_status": "discovered",
        "parser_family": None,
        "status": "discovered",
        "download_status": None,
        "source_retrieved_at": None,
        "raw_path": None,
        "quarantine_reason": None,
        "quarantine_details": None,
        "attempted_parser": None,
        "possible_remediation": None,
        "parsed_source_year": None,
        "parsed_source_week": None,
        "parsed_week_start_date": None,
        "parsed_week_end_date": None,
        "accepted_rows": 0,
        "observed_rows": 0,
    }


def _descriptor_from_record(record: dict[str, Any]) -> ReportDescriptor:
    def maybe_int(value: object) -> int | None:
        if pd.isna(value):
            return None
        return int(value)

    return ReportDescriptor(
        source=str(record["source"]),
        title=str(record.get("title") or ""),
        url=str(record["source_url"]),
        document_type=str(record.get("document_type") or "pdf"),
        discovered_at=_now(),
        issue_year=maybe_int(record.get("year")),
        issue_week=maybe_int(record.get("week")),
    )


def _save_index(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reindex(columns=DISCOVERY_COLUMNS).sort_values(
        ["source", "year", "week", "source_url"], na_position="last", kind="mergesort"
    )
    written_paths: list[Path] = []
    if path.suffix == ".parquet":
        out.to_parquet(path, index=False)
        written_paths.append(path)
        json_path = path.with_suffix(".json")
        out.to_json(json_path, orient="records", indent=2, date_format="iso")
        written_paths.append(json_path)
    elif path.suffix == ".json":
        out.to_json(path, orient="records", indent=2, date_format="iso")
        written_paths.append(path)
    else:
        out.to_csv(path, index=False)
        written_paths.append(path)
    generated_at = _now()
    for written_path in written_paths:
        _write_index_metadata(written_path, out, generated_at=generated_at)


def _write_index_metadata(path: Path, df: pd.DataFrame, *, generated_at: str) -> None:
    media_type = {
        ".parquet": "application/vnd.apache.parquet",
        ".json": "application/json",
        ".csv": "text/csv",
    }.get(path.suffix, "application/octet-stream")
    source_urls = sorted(
        {
            str(value)
            for value in df["source_url"].dropna().unique()
            if str(value).startswith(("http://", "https://"))
        }
    )
    metadata = {
        "filename": path.name,
        "sha256": _sha256(path),
        "content_length": path.stat().st_size,
        "source_name": "historical_report_index",
        "source_type": "derived_discovery_index",
        "media_type": media_type,
        "generated_at": generated_at,
        "retrieved_at": generated_at,
        "parser_version": PARSER_VERSION,
        "period": {
            "start_year": int(df["year"].dropna().min()) if df["year"].notna().any() else None,
            "end_year": int(df["year"].dropna().max()) if df["year"].notna().any() else None,
        },
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "derived_from": "historical dengue discovery/acquisition/parse index state",
        "official_archive_urls": {
            "epid": EPID_ARCHIVE_URL,
            "ndcu": NDCU_WEEKLY_URL,
        },
        "underlying_official_document_urls": source_urls,
    }
    metadata_path = _index_metadata_path(path)
    tmp_metadata = metadata_path.with_name(f"{metadata_path.name}.tmp")
    tmp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_metadata, metadata_path)


def load_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_index()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    return df.reindex(columns=DISCOVERY_COLUMNS)


def _retain_archive_html(url: str, *, source_name: str, raw_dir: Path) -> Path:
    response = requests.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=30)
    response.raise_for_status()
    content = response.content
    sha = hashlib.sha256(content).hexdigest()
    index_dir = raw_dir / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source_name}_archive_index_{Path(url.rstrip('/')).name or 'index'}_{sha[:12]}"
    path = index_dir / f"{stem}.html"
    metadata_path = index_dir / f"{stem}.metadata.json"
    if not path.exists():
        tmp_path = path.with_suffix(".html.tmp")
        tmp_path.write_bytes(content)
        os.replace(tmp_path, path)
    metadata = {
        "filename": path.name,
        "sha256": sha,
        "source_url": url,
        "source_name": f"{source_name}_archive_index",
        "source_type": "archive_html",
        "retrieved_at": _now(),
        "media_type": response.headers.get("Content-Type", "text/html").split(";")[0],
        "content_length": len(content),
        "period": {},
        "parser_version": PARSER_VERSION,
    }
    if not metadata_path.exists():
        tmp_metadata = metadata_path.with_suffix(".metadata.json.tmp")
        tmp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_metadata, metadata_path)
    return path


def discover_historical_reports(
    *,
    start_year: int = 2010,
    end_year: int = HISTORICAL_END_YEAR,
    raw_dir: Path = Path("data/raw/dengue"),
    index_path: Path = Path("data/raw/dengue/historical_report_index.parquet"),
    epid_html: Path | None = None,
    ndcu_html: Path | None = None,
    refresh: bool = False,
    offline: bool = False,
    include_ndcu: bool = True,
) -> pd.DataFrame:
    descriptors: list[ReportDescriptor] = []
    source_inputs = [
        ("epid", EPID_ARCHIVE_URL, epid_html),
        ("ndcu", NDCU_WEEKLY_URL, ndcu_html),
    ]
    if not include_ndcu:
        source_inputs = source_inputs[:1]

    for source, url, html_path in source_inputs:
        if refresh and html_path is None and not offline:
            html_path = _retain_archive_html(url, source_name=source, raw_dir=raw_dir)
        if html_path is not None:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            descriptors.extend(
                discover_dengue_reports_from_html(
                    html,
                    source=source,
                    start_year=start_year,
                    end_year=end_year + 1,
                    base_url=url,
                )
            )
        else:
            descriptors.extend(
                discover_dengue_reports(
                    source,
                    start_year,
                    end_year + 1,
                    archive_url=url,
                    offline=offline,
                )
            )

    descriptors = [
        descriptor
        for descriptor in descriptors
        if _descriptor_in_historical_scope(descriptor, start_year=start_year, end_year=end_year)
    ]
    df = pd.DataFrame([_record_from_descriptor(d) for d in descriptors])
    if df.empty:
        df = _empty_index()
    df = df.drop_duplicates(["source", "source_url"], keep="first").reset_index(drop=True)
    _save_index(df, index_path)
    return df


def _descriptor_in_historical_scope(
    descriptor: ReportDescriptor, *, start_year: int, end_year: int
) -> bool:
    issue_year = descriptor.issue_year
    issue_week = descriptor.issue_week
    if issue_year is None:
        return False
    if start_year <= issue_year <= end_year:
        return True
    return (
        descriptor.source.casefold() == "epid"
        and issue_year == end_year + 1
        and issue_week in EPID_BOUNDARY_ISSUES
    )


def _resolve_raw_path(record: dict[str, Any], raw_dir: Path) -> Path:
    candidates: list[Path] = []
    raw_value = record.get("raw_path")
    if raw_value and not pd.isna(raw_value):
        raw_path = Path(str(raw_value))
        candidates.append(raw_path if raw_path.is_absolute() else raw_dir / raw_path)
        candidates.append(raw_path)
    filename = record.get("document_filename")
    if filename and not pd.isna(filename):
        candidates.append(raw_dir / Path(str(filename)).name)

    expected_sha = str(record.get("document_sha256") or "")
    source_url = str(record.get("source_url") or "")
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        actual_sha = _sha256(candidate)
        if expected_sha and actual_sha != expected_sha:
            continue
        metadata_path = candidate.with_name(f"{candidate.stem}.metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("filename") != candidate.name:
                continue
            if expected_sha and str(metadata.get("sha256")) != expected_sha:
                continue
            if source_url and str(metadata.get("source_url")) != source_url:
                continue
        return candidate
    raise SourceCacheError(
        "Retained raw file not found or failed metadata validation for "
        f"{filename or raw_value or source_url}"
    )


def _document_from_record(record: dict[str, Any], *, raw_dir: Path) -> SourceDocument:
    raw_path = _resolve_raw_path(record, raw_dir)
    return SourceDocument(
        path=raw_path,
        source_name=str(record["source"]),
        source_url=str(record["source_url"]),
        retrieved_at=str(record["source_retrieved_at"]),
        sha256=str(record["document_sha256"]),
        media_type="application/pdf",
        descriptor={"issue_year": record.get("year"), "issue_week": record.get("week")},
    )


def acquire_historical_reports(
    *,
    index_path: Path = Path("data/raw/dengue/historical_report_index.parquet"),
    raw_dir: Path = Path("data/raw/dengue"),
    offline: bool = False,
    max_workers: int = 2,
) -> tuple[pd.DataFrame, Counter[str]]:
    df = load_index(index_path)
    if df.empty:
        return df, Counter()
    cache = SourceCache(raw_dir)

    def acquire_one(record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record)
        descriptor = _descriptor_from_record(out)
        try:
            cached_before = cache.find_by_url(descriptor.url)
            document = download_dengue_report(descriptor, raw_dir=raw_dir, offline=offline)
            actual_sha = _sha256(document.path)
            if actual_sha != document.sha256:
                raise SourceCacheError(
                    "Downloaded cache hash mismatch: "
                    f"metadata={document.sha256} actual={actual_sha}"
                )
            out.update(
                {
                    "document_filename": document.path.name,
                    "document_sha256": document.sha256,
                    "source_retrieved_at": document.retrieved_at,
                    "raw_path": str(document.path),
                    "status": "downloaded",
                    "download_status": "reused_hash_verified"
                    if cached_before is not None
                    else "downloaded",
                    "quarantine_reason": None,
                    "quarantine_details": None,
                }
            )
        except Exception as exc:
            out.update(
                _quarantine_fields(
                    "DOWNLOAD_ERROR",
                    str(exc),
                    parser=None,
                    remediation=(
                        "Retain the archive index, fetch the missing PDF, then rerun offline."
                    ),
                )
            )
            out["status"] = "quarantined"
            out["download_status"] = "failed"
        return out

    workers = max(1, min(2, max_workers))
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(acquire_one, row) for row in df.to_dict("records")]
        for future in as_completed(futures):
            records.append(future.result())
    out = pd.DataFrame(records).reindex(columns=DISCOVERY_COLUMNS)
    _save_index(out, index_path)
    return out, Counter(out["download_status"].dropna())


def _parser_family(parser_name: str | None) -> str:
    if parser_name is None:
        return "unknown"
    lowered = parser_name.casefold()
    if "modern" in lowered:
        return "wer_modern"
    if "legacy" in lowered:
        return "wer_legacy"
    if "ndcu" in lowered:
        return "ndcu_weekly"
    return parser_name


def _classify_exception(exc: Exception) -> str:
    text = str(exc).casefold()
    if "inconsistent with source" in text or "period" in text or "week boundary" in text:
        return "REPORTING_PERIOD_AMBIGUITY"
    if "cumulative" in text:
        return "CUMULATIVE_WEEKLY_AMBIGUITY"
    if "expected 26" in text or "duplicate source components" in text:
        return "DISTRICT_TABLE_INCOMPLETE"
    if "national reconciliation failed" in text:
        return "SOURCE_CONFLICT"
    if "unsupported" in text or "no supported" in text or "no dengue parser" in text:
        return "PARSER_ERROR"
    if isinstance(exc, SourceCacheError):
        return "DOWNLOAD_ERROR"
    return "UNKNOWN"


def _quarantine_fields(
    reason: str,
    details: str,
    *,
    parser: str | None,
    remediation: str,
) -> dict[str, Any]:
    if reason not in QUARANTINE_REASONS:
        reason = "UNKNOWN"
    return {
        "quarantine_reason": reason,
        "quarantine_details": details,
        "attempted_parser": parser,
        "possible_remediation": remediation,
    }


def _within_target_cohort(value: object) -> bool:
    if pd.isna(value):
        return False
    parsed = pd.to_datetime(value).date()
    return _cohort_start() <= parsed <= _cohort_end()


def _parse_one(record: dict[str, Any], *, raw_dir: Path) -> tuple[dict[str, Any], Any | None]:
    out = dict(record)
    if not out.get("raw_path") or pd.isna(out.get("raw_path")):
        out["status"] = "quarantined"
        out.update(
            _quarantine_fields(
                "DOWNLOAD_ERROR",
                "No retained raw file path in historical index.",
                parser=None,
                remediation="Run acquisition online once or provide retained raw files and index.",
            )
        )
        return out, None

    document = _document_from_record(out, raw_dir=raw_dir)
    parser_name: str | None = None
    try:
        actual_sha = _sha256(document.path)
        if actual_sha != document.sha256:
            raise SourceCacheError(
                f"Raw file hash mismatch for {document.path}: "
                f"index={document.sha256} actual={actual_sha}"
            )
        parser = select_parser(document)
        parser_name = parser.name
        report = parse_dengue_document(document)
        normalized, _, _ = normalize_parsed_reports([report])
        if not _within_target_cohort(report.week_start_date):
            out.update(
                {
                    "parser_family": _parser_family(report.parser_name),
                    "status": "excluded_outside_cohort",
                    "attempted_parser": report.parser_name,
                    "parsed_source_year": report.source_year,
                    "parsed_source_week": report.source_week,
                    "parsed_week_start_date": report.week_start_date.isoformat(),
                    "parsed_week_end_date": report.week_end_date.isoformat(),
                    "accepted_rows": 0,
                    "observed_rows": int(normalized["dengue_cases"].notna().sum()),
                    "quarantine_reason": None,
                    "quarantine_details": None,
                    "possible_remediation": None,
                }
            )
            return out, None
        out.update(
            {
                "parser_family": _parser_family(report.parser_name),
                "status": "parsed",
                "attempted_parser": report.parser_name,
                "parsed_source_year": report.source_year,
                "parsed_source_week": report.source_week,
                "parsed_week_start_date": report.week_start_date.isoformat(),
                "parsed_week_end_date": report.week_end_date.isoformat(),
                "accepted_rows": len(normalized),
                "observed_rows": int(normalized["dengue_cases"].notna().sum()),
                "quarantine_reason": None,
                "quarantine_details": None,
                "possible_remediation": None,
            }
        )
        return out, report
    except Exception as exc:
        reason = _classify_exception(exc)
        out["parser_family"] = _parser_family(parser_name)
        out["status"] = "quarantined"
        out.update(
            _quarantine_fields(
                reason,
                str(exc),
                parser=parser_name,
                remediation=(
                    "Review the retained PDF against the parser family; "
                    "add a fixture before accepting."
                ),
            )
        )
        return out, None


def _quarantine_frame(index: pd.DataFrame) -> pd.DataFrame:
    rows = index[index["status"].eq("quarantined")].copy()
    columns = {
        "source_document": rows["document_filename"],
        "source_url": rows["source_url"],
        "source_name": rows["source"],
        "source_file": rows["document_filename"],
        "year": rows["year"],
        "week": rows["week"],
        "reason": rows["quarantine_reason"],
        "details": rows["quarantine_details"],
        "parser_attempted": rows["attempted_parser"],
        "possible_remediation": rows["possible_remediation"],
    }
    return pd.DataFrame(columns)


def _conflicts_with_chosen_value(conflicts: pd.DataFrame) -> pd.DataFrame:
    if conflicts.empty:
        return pd.DataFrame(columns=CONFLICT_COLUMNS)
    out = conflicts.copy()
    out["chosen_value"] = pd.NA
    if "reason" not in out.columns and "resolution" in out.columns:
        out["reason"] = out["resolution"]
    return out.reindex(columns=CONFLICT_COLUMNS)


def _load_phase_a_baseline(baseline_path: Path) -> pd.DataFrame:
    if not baseline_path.exists():
        raise ContractError(
            f"Phase A baseline guard file is required but missing: {baseline_path}"
        )
    if baseline_path.suffix == ".parquet":
        return pd.read_parquet(baseline_path)
    if baseline_path.suffix == ".json":
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ContractError(f"Phase A baseline JSON has no row list: {baseline_path}")
        return pd.DataFrame(rows)
    raise ContractError(f"Unsupported Phase A baseline format: {baseline_path}")


def _guard_phase_a_baseline(
    candidate: pd.DataFrame,
    *,
    baseline_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    baseline = _load_phase_a_baseline(baseline_path)
    keys = ["district_id", "week_start_date"]
    compare_cols = keys + ["dengue_cases", "case_status"]
    base = baseline[compare_cols].copy()
    cand = candidate.copy()
    cand["week_start_date"] = pd.to_datetime(cand["week_start_date"]).dt.date
    base["week_start_date"] = pd.to_datetime(base["week_start_date"]).dt.date

    base_keyed = base.set_index(keys).sort_index()
    cand_keyed = cand.set_index(keys).sort_index()
    missing = base_keyed.index.difference(cand_keyed.index)
    if len(missing):
        raise ContractError(
            f"Historical candidate deleted Phase A baseline keys: {list(missing[:5])}"
        )

    changed: list[dict[str, Any]] = []
    for key, base_row in base_keyed.iterrows():
        cand_row = cand_keyed.loc[key]
        if isinstance(cand_row, pd.DataFrame):
            raise ContractError(f"Historical candidate has duplicate baseline key: {key}")
        base_cases = base_row["dengue_cases"]
        cand_cases = cand_row["dengue_cases"]
        same_cases = (pd.isna(base_cases) and pd.isna(cand_cases)) or base_cases == cand_cases
        if not same_cases or str(base_row["case_status"]) != str(cand_row["case_status"]):
            changed.append(
                {
                    "district_id": key[0],
                    "week_start_date": key[1].isoformat(),
                    "baseline_cases": None if pd.isna(base_cases) else int(base_cases),
                    "candidate_cases": None if pd.isna(cand_cases) else int(cand_cases),
                    "baseline_status": str(base_row["case_status"]),
                    "candidate_status": str(cand_row["case_status"]),
                }
            )
    if changed:
        raise ContractError(f"Historical candidate changed Phase A baseline rows: {changed[:5]}")

    base_index = set(base_keyed.index)
    extra_2024 = [
        key for key in cand_keyed.index if key[1].year == 2024 and key not in base_index
    ]
    review_rows: list[dict[str, Any]] = []
    if extra_2024:
        mask = cand.set_index(keys).index.isin(extra_2024)
        extras = cand.loc[mask].copy()
        review_rows = [
            {
                "source_document": row.get("source_document"),
                "source_url": row.get("source_url"),
                "source_name": row.get("source_name"),
                "source_file": row.get("source_document"),
                "year": 2024,
                "week": row.get("week"),
                "reason": "DUPLICATE_OR_REVISION",
                "details": (
                    "Parsed 2024 district-week was not part of the Phase A accepted baseline; "
                    "kept out of candidate pending explicit review."
                ),
                "parser_attempted": row.get("parser_version"),
                "possible_remediation": (
                    "Review official correction evidence before changing baseline."
                ),
            }
            for _, row in extras.iterrows()
        ]
        cand = cand.loc[~mask].copy()
    return cand, review_rows


def _normalization_quarantine_reason(exc: Exception) -> str:
    text = str(exc).casefold()
    if "duplicate" in text or "revision" in text:
        return "DUPLICATE_OR_REVISION"
    if "conflict" in text or "reconciliation" in text or "disagree" in text:
        return "SOURCE_CONFLICT"
    return _classify_exception(exc)


def _normalize_reports_preserving_good_groups(
    reports: list[Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[str, str]]]:
    canonical_parts: list[pd.DataFrame] = []
    quality_parts: list[pd.DataFrame] = []
    conflict_parts: list[pd.DataFrame] = []
    quarantined_documents: dict[str, tuple[str, str]] = {}
    groups: dict[tuple[date, date], list[Any]] = {}
    for report in reports:
        groups.setdefault((report.week_start_date, report.week_end_date), []).append(report)

    for group_reports in groups.values():
        try:
            canonical, quality, conflicts = normalize_parsed_reports(group_reports)
        except Exception as exc:
            reason = _normalization_quarantine_reason(exc)
            for report in group_reports:
                quarantined_documents[report.source_document] = (reason, str(exc))
            continue
        canonical_parts.append(canonical)
        quality_parts.append(quality)
        conflict_parts.append(conflicts)

    canonical = pd.concat(canonical_parts, ignore_index=True) if canonical_parts else pd.DataFrame()
    quality = pd.concat(quality_parts, ignore_index=True) if quality_parts else pd.DataFrame()
    conflicts = pd.concat(conflict_parts, ignore_index=True) if conflict_parts else pd.DataFrame()
    return canonical, quality, conflicts, quarantined_documents


def parse_historical_reports(
    *,
    index_path: Path = Path("data/raw/dengue/historical_report_index.parquet"),
    processed_path: Path = Path("data/historical/processed/dengue_cases_weekly.parquet"),
    reports_dir: Path = Path("data/reports"),
    baseline_path: Path | None = DEFAULT_BASELINE_PATH,
    raw_dir: Path = Path("data/raw/dengue"),
    max_workers: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    index = load_index(index_path)
    parseable = index[index["document_sha256"].notna()].copy()
    workers = max(1, min(2, max_workers))
    records: list[dict[str, Any]] = []
    reports = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_parse_one, row, raw_dir=raw_dir)
            for row in parseable.to_dict("records")
        ]
        for future in as_completed(futures):
            record, report = future.result()
            records.append(record)
            if report is not None:
                reports.append(report)

    untouched = index[index["document_sha256"].isna()].to_dict("records")
    out_index = pd.DataFrame([*untouched, *records]).reindex(columns=DISCOVERY_COLUMNS)

    review_rows: list[dict[str, Any]] = []
    if reports:
        canonical, quality, conflicts, normalization_quarantines = (
            _normalize_reports_preserving_good_groups(reports)
        )
        if normalization_quarantines:
            for record in records:
                filename = str(record.get("document_filename") or "")
                if filename not in normalization_quarantines:
                    continue
                reason, details = normalization_quarantines[filename]
                record["status"] = "quarantined"
                record.update(
                    _quarantine_fields(
                        reason,
                        details,
                        parser=record.get("attempted_parser"),
                        remediation=(
                            "Review duplicate or conflicting official source group before "
                            "accepting this report."
                        ),
                    )
                )
            out_index = pd.DataFrame([*untouched, *records]).reindex(columns=DISCOVERY_COLUMNS)
        canonical = canonical[
            canonical["week_start_date"].map(_within_target_cohort)
        ].sort_values(
            ["district_id", "week_start_date", "source_name", "source_document"],
            kind="mergesort",
        )
    else:
        canonical = pd.DataFrame()
        quality = pd.DataFrame()
        conflicts = pd.DataFrame()

    quarantine = _quarantine_frame(out_index)
    baseline_guard = "disabled_explicitly"
    if baseline_path is not None:
        baseline_guard = "passed"
        canonical, review_rows = _guard_phase_a_baseline(canonical, baseline_path=baseline_path)
        if review_rows:
            quarantine = pd.concat([quarantine, pd.DataFrame(review_rows)], ignore_index=True)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not canonical.empty:
        canonical.to_parquet(processed_path, index=False)
        canonical.to_csv(processed_path.with_suffix(".csv"), index=False)
    else:
        pd.DataFrame().to_parquet(processed_path, index=False)

    conflicts_out = _conflicts_with_chosen_value(conflicts)
    conflicts_out.to_csv(reports_dir / "historical_source_conflicts.csv", index=False)
    quarantine.to_csv(reports_dir / "historical_dengue_quarantine.csv", index=False)
    quality.to_csv(reports_dir / "historical_dengue_extraction_quality.csv", index=False)
    _save_index(out_index, index_path)

    summary = {
        "generated_at": _now(),
        "index_rows": int(len(out_index)),
        "documents_parsed": int(out_index["status"].eq("parsed").sum()),
        "documents_quarantined": int(out_index["status"].eq("quarantined").sum()),
        "documents_excluded_outside_cohort": int(
            out_index["status"].eq("excluded_outside_cohort").sum()
        ),
        "candidate_rows": int(len(canonical)),
        "candidate_observed_rows": int(canonical["dengue_cases"].notna().sum())
        if not canonical.empty
        else 0,
        "conflicts": int(len(conflicts_out)),
        "document_quarantine_rows": int(out_index["status"].eq("quarantined").sum()),
        "review_quarantine_rows": int(len(review_rows)),
        "quarantine_rows": int(len(quarantine)),
        "baseline_guard": baseline_guard,
        "national_reconciliation_methods": quality[
            "national_reconciliation_method"
        ].value_counts(dropna=False).to_dict()
        if not quality.empty and "national_reconciliation_method" in quality.columns
        else {},
        "ndcu_cross_source_comparison": (
            "not_available_for_historical_cohort; current NDCU 2026 weekly PDFs are outside "
            "the 2010-2025 historical acceptance scope"
        ),
    }
    (reports_dir / "historical_dengue_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(reports_dir / "historical_dengue_summary.csv", index=False)
    return canonical, out_index, summary


def run_historical_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.discover or args.all:
        discover_historical_reports(
            start_year=args.start_year,
            end_year=args.end_year,
            raw_dir=args.raw_dir,
            index_path=args.index,
            epid_html=args.epid_html,
            ndcu_html=args.ndcu_html,
            refresh=args.refresh,
            offline=args.offline,
            include_ndcu=not args.no_ndcu,
        )
    if args.acquire or args.all:
        acquire_historical_reports(
            index_path=args.index,
            raw_dir=args.raw_dir,
            offline=args.offline,
            max_workers=args.workers,
        )
    summary: dict[str, Any] = {}
    if args.parse or args.all:
        _, _, summary = parse_historical_reports(
            index_path=args.index,
            processed_path=args.processed,
            reports_dir=args.reports_dir,
            baseline_path=args.baseline,
            raw_dir=args.raw_dir,
            max_workers=args.workers,
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Historical dengue discovery/acquisition/parser")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Run discover, acquire, and parse")
    mode.add_argument("--discover", action="store_true", help="Refresh/build retained source index")
    mode.add_argument("--acquire", action="store_true", help="Download or reuse indexed PDFs")
    mode.add_argument(
        "--parse", action="store_true", help="Parse retained raw PDFs into candidate data"
    )
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=HISTORICAL_END_YEAR)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/dengue"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/raw/dengue/historical_report_index.parquet")
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/historical/processed/dengue_cases_weekly.parquet"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
    )
    parser.add_argument("--epid-html", type=Path)
    parser.add_argument("--ndcu-html", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-ndcu", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not (args.all or args.discover or args.acquire or args.parse):
        args.all = True
    args.workers = max(1, min(2, args.workers))
    summary = run_historical_pipeline(args)
    if summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_BASELINE_PATH",
    "DISCOVERY_COLUMNS",
    "QUARANTINE_REASONS",
    "acquire_historical_reports",
    "discover_historical_reports",
    "load_index",
    "main",
    "parse_historical_reports",
]
