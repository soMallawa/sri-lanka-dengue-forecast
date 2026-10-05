from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from dengue_forecast.dengue.download import download_dengue_report
from dengue_forecast.dengue.models import ReportDescriptor, SourceDocument
from dengue_forecast.dengue.normalize import normalize_parsed_reports
from dengue_forecast.dengue.parse import parse_dengue_document
from dengue_forecast.dengue.validate import dengue_coverage_report


def build_dengue_dataset(
    *,
    descriptors: Iterable[ReportDescriptor] | None = None,
    documents: Iterable[SourceDocument] | None = None,
    raw_dir: Path = Path("data/raw/dengue"),
    processed_path: Path = Path("data/processed/dengue_cases_weekly.parquet"),
    reports_dir: Path = Path("data/reports"),
    offline: bool = False,
) -> pd.DataFrame:
    docs = list(documents or [])
    if descriptors is not None:
        docs.extend(
            download_dengue_report(d, raw_dir=raw_dir, offline=offline) for d in descriptors
        )

    parsed = []
    quarantined = []
    for doc in docs:
        try:
            parsed.append(parse_dengue_document(doc))
        except Exception as exc:  # quarantine is an output, not a fake zero-count record.
            quarantined.append(
                {
                    "source_document": doc.path.name,
                    "source_url": doc.source_url,
                    "source_name": doc.source_name,
                    "reason": str(exc),
                }
            )

    df, quality, conflicts = normalize_parsed_reports(parsed)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(processed_path, index=False)
    quality.to_csv(reports_dir / "dengue_extraction_quality.csv", index=False)
    conflict_columns = [
        "district",
        "week",
        "source_a",
        "value_a",
        "source_b",
        "value_b",
        "resolution",
        "reason",
    ]
    if conflicts.empty:
        conflicts = pd.DataFrame(columns=conflict_columns)
    conflicts.to_csv(reports_dir / "dengue_conflicts.csv", index=False)
    quarantine_columns = ["source_document", "source_url", "source_name", "reason"]
    pd.DataFrame(quarantined, columns=quarantine_columns).to_csv(
        reports_dir / "dengue_quarantine.csv", index=False
    )
    dengue_coverage_report(
        df, conflict_count=len(conflicts), quarantined_reports=len(quarantined)
    ).to_csv(reports_dir / "dengue_coverage.csv", index=False)
    (reports_dir / "dengue_source_summary.json").write_text(
        json.dumps(
            {
                "documents_attempted": len(docs),
                "documents_parsed": len(parsed),
                "documents_quarantined": len(quarantined),
                "district_week_rows": len(df),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return df
