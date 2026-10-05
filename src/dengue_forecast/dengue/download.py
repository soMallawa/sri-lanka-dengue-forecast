from __future__ import annotations

from pathlib import Path

from dengue_forecast.dengue.models import PARSER_VERSION, ReportDescriptor, SourceDocument
from dengue_forecast.sources.cache import SourceCache


def download_dengue_report(
    descriptor: ReportDescriptor,
    *,
    raw_dir: Path = Path("data/raw/dengue"),
    offline: bool = False,
) -> SourceDocument:
    cache = SourceCache(raw_dir)
    cached = cache.download(
        descriptor.url,
        source_name=descriptor.source,
        source_type="weekly_dengue_report",
        period={
            "issue_year": descriptor.issue_year,
            "issue_week": descriptor.issue_week,
            "source_year": descriptor.start_date.year if descriptor.start_date else None,
            "source_week": descriptor.observation_week,
        },
        parser_version=PARSER_VERSION,
        offline=offline,
        referer="https://www.epid.gov.lk/weekly-epidemiological-report",
    )
    return SourceDocument(
        path=cached.path,
        source_name=descriptor.source,
        source_url=descriptor.url,
        retrieved_at=cached.retrieved_at,
        sha256=cached.sha256,
        media_type=cached.media_type,
        descriptor=descriptor.__dict__,
    )
