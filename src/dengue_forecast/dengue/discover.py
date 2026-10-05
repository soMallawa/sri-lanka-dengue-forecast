from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from dengue_forecast.dengue.models import ReportDescriptor
from dengue_forecast.sources.cache import DEFAULT_USER_AGENT
from dengue_forecast.sources.config import source_url

EPID_ARCHIVE_URL = source_url("wer_index")
NDCU_WEEKLY_URL = source_url("ndcu_index")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _issue_from_title_url(text: str) -> tuple[int | None, int | None, int | None]:
    volume = None
    vol_match = re.search(r"Vol[_\s.-]*(\d+)", text, flags=re.I)
    if vol_match:
        volume = int(vol_match.group(1))
    week_match = re.search(r"(?:no|No)[_\s.-]*(\d{1,2})", text, flags=re.I)
    issue_week = int(week_match.group(1)) if week_match else None
    # Verified current WER volume convention: Vol 37=2010, Vol 51=2024.
    issue_year = volume + 1973 if volume is not None else None
    if issue_week is not None and not 1 <= issue_week <= 53:
        issue_week = None
    return issue_year, issue_week, volume


def _ndcu_issue_from_title_url(text: str) -> tuple[int | None, int | None, int | None]:
    year_match = re.search(r"\b(20\d{2})\b", text)
    week_match = re.search(r"\bWeek[_\s.-]*(\d{1,2})\b", text, flags=re.I)
    issue_year = int(year_match.group(1)) if year_match else None
    issue_week = int(week_match.group(1)) if week_match else None
    if issue_week is not None and not 1 <= issue_week <= 53:
        issue_week = None
    return issue_year, issue_week, None


def discover_dengue_reports_from_html(
    html: str,
    *,
    source: str,
    start_year: int,
    end_year: int | None = None,
    base_url: str,
) -> list[ReportDescriptor]:
    soup = BeautifulSoup(html, "lxml")
    end_year = end_year or datetime.now(UTC).year
    descriptors: list[ReportDescriptor] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        title = " ".join(anchor.get_text(" ", strip=True).split()) or Path(href).name
        url = urljoin(base_url, href)
        if ".pdf" not in url.lower():
            continue
        haystack = f"{title} {url}"
        if source.casefold() == "ndcu":
            issue_year, issue_week, volume = _ndcu_issue_from_title_url(haystack)
        else:
            issue_year, issue_week, volume = _issue_from_title_url(haystack)
        if issue_year is None or not start_year <= issue_year <= end_year:
            continue
        if url in seen:
            continue
        seen.add(url)
        descriptors.append(
            ReportDescriptor(
                source=source,
                title=title,
                url=url,
                document_type="pdf",
                discovered_at=_now(),
                issue_year=issue_year,
                issue_week=issue_week,
                volume=volume,
            )
        )
    return sorted(descriptors, key=lambda d: (d.issue_year or 0, d.issue_week or 0, d.url))


def discover_dengue_reports(
    source: str,
    start_year: int,
    end_year: int | None = None,
    *,
    archive_url: str | None = None,
    html_path: Path | None = None,
    output_path: Path | None = None,
    offline: bool = False,
) -> list[ReportDescriptor]:
    source = source.casefold()
    if source not in {"epid", "ndcu"}:
        raise ValueError(f"Unsupported dengue source: {source}")

    url = archive_url or (EPID_ARCHIVE_URL if source == "epid" else NDCU_WEEKLY_URL)
    if html_path is not None:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    elif offline:
        raise RuntimeError("offline discovery requires html_path")
    else:
        response = requests.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=30)
        response.raise_for_status()
        html = response.text

    descriptors = discover_dengue_reports_from_html(
        html, source=source, start_year=start_year, end_year=end_year, base_url=url
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([d.__dict__ for d in descriptors]).to_parquet(output_path, index=False)
    return descriptors
