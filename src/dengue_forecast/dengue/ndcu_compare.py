from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_forecast.config import DISTRICTS, normalize_district_name
from dengue_forecast.contracts import ContractError
from dengue_forecast.dengue.models import PARSER_VERSION

CURRENT_NDCU_SOURCE_URL = (
    "https://www.dengue.health.gov.lk/wp-content/uploads/2026/06/"
    "Weekly-Dengue-Update-2026-Week-01.pdf"
)
CURRENT_NDCU_SHA256 = "342a26e3b5eb0f52a90d8165bbac2fd8c370cde4a6aef4632d1232574923eea2"

HEADER_ROWS = 8
EXPECTED_RDHS_ROWS = 26
NDCU_WEEKLY_SOURCE = "ndcu_weekly"
NOT_COMPARABLE_INTERVAL = "NOT_COMPARABLE_INTERVAL"
EXACT_INTERVAL = "EXACT_INTERVAL"

NDCU_COMPONENT_NOTE = (
    "NDCU reports Ampara and Kalmunai as RDHS components. This comparison reconciles "
    "the 26 RDHS rows first, then aggregates Ampara + Kalmunai to the canonical "
    "25-district district frame."
)

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True)
class NdcuWeeklyColumn:
    column_index: int
    source_year: int
    source_week: int
    week_start_date: date
    week_end_date: date
    header_text: str
    printed_date_text: str


@dataclass(frozen=True)
class NdcuComponentRow:
    source_year: int
    source_week: int
    year_week: str
    week_start_date: date
    week_end_date: date
    rdhs_name: str
    current_week_cases: int
    revision_flag: bool
    raw_value: str
    source_name: str
    source_document: str
    source_url: str
    source_sha256: str
    source_retrieved_at: str | None
    parser_version: str
    page_number: int
    table_bbox: tuple[float, float, float, float] | None
    cell_bbox: tuple[float, float, float, float] | None
    printed_date_text: str
    header_text: str
    provenance_note: str


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _header_for_column(rows: list[list[Any]], column_index: int) -> str:
    parts = []
    for row in rows[:HEADER_ROWS]:
        if column_index < len(row):
            text = _clean_text(row[column_index])
            if text:
                parts.append(text)
    return " ".join(parts)


def _left_filled_header_cell(rows: list[list[Any]], row_index: int, column_index: int) -> str:
    row = rows[row_index]
    for index in range(column_index, -1, -1):
        text = _clean_text(row[index] if index < len(row) else "")
        if text:
            return text
    return ""


def _parse_int_cell(value: object) -> tuple[int, bool, str]:
    raw = _clean_text(value)
    if not raw:
        raise ContractError("NDCU weekly case cell is blank")
    if raw.casefold() == "nil":
        return 0, False, raw
    revision_flag = "*" in raw
    digits = re.sub(r"[^0-9]", "", raw)
    if not digits:
        raise ContractError(f"NDCU weekly case cell is not numeric: {raw!r}")
    return int(digits), revision_flag, raw


def _strip_footnote(value: str) -> str:
    return re.sub(r"\*+$", "", _clean_text(value)).strip()


def _parse_printed_week_interval(
    text: str, *, expected_year: int, expected_week: int
) -> tuple[str, date, date]:
    normalized = re.sub(r"\s+", " ", text.replace("\u2013", "-").replace("\u2014", "-"))
    pattern = (
        rf"Week\s+0*{expected_week}\s*\(\s*"
        r"(?P<sd>\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(?P<sm>[A-Za-z]+)\s+(?P<sy>\d{4})\s*-\s*"
        r"(?P<ed>\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(?P<em>[A-Za-z]+)\s+(?P<ey>\d{4})"
    )
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if not match:
        raise ContractError("NDCU printed current-week date range was not found")
    start = date(
        int(match.group("sy")),
        MONTHS[match.group("sm").casefold()],
        int(match.group("sd")),
    )
    end = date(
        int(match.group("ey")),
        MONTHS[match.group("em").casefold()],
        int(match.group("ed")),
    )
    if expected_year != end.year or (end - start).days != 6:
        raise ContractError(
            f"NDCU printed interval is unexpected for {expected_year}-W{expected_week:02d}: "
            f"{start} to {end}"
        )
    return match.group(0), start, end


def _find_weekly_columns(rows: list[list[Any]], printed_text: str) -> list[NdcuWeeklyColumn]:
    if len(rows) < HEADER_ROWS + EXPECTED_RDHS_ROWS + 1:
        raise ContractError("NDCU table does not contain the expected header and RDHS rows")
    max_cols = max(len(row) for row in rows)
    if max_cols != 7:
        raise ContractError(f"NDCU table expected 7 columns, found {max_cols}")

    printed_date_text, current_start, current_end = _parse_printed_week_interval(
        printed_text, expected_year=2026, expected_week=1
    )
    expected = {
        (2025, 52): (current_start - timedelta(days=7), current_end - timedelta(days=7)),
        (2026, 1): (current_start, current_end),
    }

    found: dict[tuple[int, int], NdcuWeeklyColumn] = {}
    for column_index in range(max_cols):
        header_text = _header_for_column(rows, column_index)
        year_band = _left_filled_header_cell(rows, 3, column_index)
        folded = header_text.casefold()
        if "up to" in folded or year_band != "2025/2026" or "week" not in folded:
            continue
        week_match = re.search(r"(?<!\d)(52|0?1)\*?\*?(?!\d)", header_text)
        if not week_match:
            continue
        week = int(week_match.group(1))
        source_year = 2026 if week == 1 else 2025
        key = (source_year, week)
        if key not in expected:
            continue
        if key in found:
            raise ContractError(f"NDCU weekly column is duplicated for {source_year}-W{week:02d}")
        start, end = expected[key]
        found[key] = NdcuWeeklyColumn(
            column_index=column_index,
            source_year=source_year,
            source_week=week,
            week_start_date=start,
            week_end_date=end,
            header_text=header_text,
            printed_date_text=printed_date_text
            if key == (2026, 1)
            else f"inferred from {printed_date_text}",
        )

    if set(found) != set(expected):
        missing = sorted(set(expected) - set(found))
        raise ContractError(f"NDCU table missing expected weekly columns: {missing}")
    return [found[(2025, 52)], found[(2026, 1)]]


def _cell_bbox(
    cells: list[tuple[float, float, float, float]] | None,
    row_index: int,
    column_index: int,
    column_count: int,
) -> tuple[float, float, float, float] | None:
    if cells is None:
        return None
    index = row_index * column_count + column_index
    if index >= len(cells):
        return None
    return cells[index]


def component_rows_from_ndcu_table(
    rows: list[list[Any]],
    *,
    printed_text: str,
    source_document: str,
    source_url: str,
    source_sha256: str,
    source_retrieved_at: str | None = None,
    page_number: int = 1,
    table_bbox: tuple[float, float, float, float] | None = None,
    cells: list[tuple[float, float, float, float]] | None = None,
) -> list[NdcuComponentRow]:
    columns = _find_weekly_columns(rows, printed_text)
    component_rows = rows[HEADER_ROWS : HEADER_ROWS + EXPECTED_RDHS_ROWS]
    total_row = rows[HEADER_ROWS + EXPECTED_RDHS_ROWS]
    total_label = _strip_footnote(_clean_text(total_row[0]))
    if total_label.casefold() != "total":
        raise ContractError(
            f"NDCU table expected total row after 26 RDHS rows, found {total_label!r}"
        )

    output: list[NdcuComponentRow] = []
    seen_names: set[str] = set()
    for row_offset, row in enumerate(component_rows, start=HEADER_ROWS):
        raw_name = _clean_text(row[0])
        rdhs_name = _strip_footnote(raw_name)
        if not rdhs_name:
            raise ContractError("NDCU RDHS row has no district/RDHS name")
        if rdhs_name in seen_names:
            raise ContractError(f"NDCU RDHS row is duplicated: {rdhs_name}")
        seen_names.add(rdhs_name)
        if rdhs_name != "Kalmunai":
            normalize_district_name(rdhs_name)
        for column in columns:
            value, value_revision, raw_value = _parse_int_cell(row[column.column_index])
            output.append(
                NdcuComponentRow(
                    source_year=column.source_year,
                    source_week=column.source_week,
                    year_week=f"{column.source_year}-W{column.source_week:02d}",
                    week_start_date=column.week_start_date,
                    week_end_date=column.week_end_date,
                    rdhs_name=rdhs_name,
                    current_week_cases=value,
                    revision_flag=value_revision or "*" in raw_name,
                    raw_value=raw_value,
                    source_name=NDCU_WEEKLY_SOURCE,
                    source_document=source_document,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    source_retrieved_at=source_retrieved_at,
                    parser_version=PARSER_VERSION,
                    page_number=page_number,
                    table_bbox=table_bbox,
                    cell_bbox=_cell_bbox(cells, row_offset, column.column_index, 7),
                    printed_date_text=column.printed_date_text,
                    header_text=column.header_text,
                    provenance_note=NDCU_COMPONENT_NOTE,
                )
            )

    expected_names = {district.district_name for district in DISTRICTS} | {"Kalmunai"}
    if seen_names != expected_names:
        raise ContractError(
            "NDCU RDHS rows do not match the expected 26 components: "
            f"missing={sorted(expected_names - seen_names)} "
            f"extra={sorted(seen_names - expected_names)}"
        )

    for column in columns:
        calculated = sum(
            row.current_week_cases
            for row in output
            if row.source_year == column.source_year and row.source_week == column.source_week
        )
        reported, _, raw_total = _parse_int_cell(total_row[column.column_index])
        if calculated != reported:
            year_week = f"{column.source_year}-W{column.source_week:02d}"
            raise ContractError(
                f"NDCU national reconciliation failed for {year_week}: "
                f"RDHS sum {calculated} != reported total {raw_total}"
            )
    return output


def extract_ndcu_component_rows(pdf_path: Path) -> list[NdcuComponentRow]:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise ContractError("PyMuPDF is required to parse NDCU PDFs") from exc

    metadata_path = pdf_path.with_suffix(".metadata.json")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with pymupdf.open(pdf_path) as document:
        if document.page_count < 1:
            raise ContractError("NDCU PDF has no pages")
        page = document[0]
        tables = page.find_tables().tables
        if len(tables) != 1:
            raise ContractError(f"NDCU page 1 expected exactly one table, found {len(tables)}")
        table = tables[0]
        rows = table.extract()
        return component_rows_from_ndcu_table(
            rows,
            printed_text=page.get_text(),
            source_document=metadata.get("filename", pdf_path.name),
            source_url=metadata.get("source_url", CURRENT_NDCU_SOURCE_URL),
            source_sha256=metadata.get("sha256", CURRENT_NDCU_SHA256),
            source_retrieved_at=metadata.get("retrieved_at"),
            page_number=1,
            table_bbox=tuple(table.bbox),
            cells=list(table.cells),
        )


def ndcu_component_rows_frame(rows: list[NdcuComponentRow]) -> pd.DataFrame:
    data = [asdict(row) for row in rows]
    df = pd.DataFrame(data)
    if df.empty:
        return df
    for column in ["week_start_date", "week_end_date"]:
        df[column] = df[column].astype(str)
    for column in ["table_bbox", "cell_bbox"]:
        df[column] = df[column].map(
            lambda value: "" if value is None else ",".join(f"{v:.1f}" for v in value)
        )
    return df.sort_values(["week_start_date", "rdhs_name"]).reset_index(drop=True)


def aggregate_ndcu_to_districts(rows: list[NdcuComponentRow]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    by_period: dict[tuple[int, int], list[NdcuComponentRow]] = {}
    for row in rows:
        by_period.setdefault((row.source_year, row.source_week), []).append(row)

    for (_, _), components in sorted(by_period.items()):
        by_name = {row.rdhs_name: row for row in components}
        for district in DISTRICTS:
            names = (
                ["Ampara", "Kalmunai"]
                if district.district_name == "Ampara"
                else [district.district_name]
            )
            selected = [by_name[name] for name in names]
            first = selected[0]
            component_page_bboxes = "|".join(
                ""
                if row.cell_bbox is None
                else f"p{row.page_number}:{','.join(f'{value:.1f}' for value in row.cell_bbox)}"
                for row in selected
            )
            records.append(
                {
                    "district_id": district.district_id,
                    "district_name": district.district_name,
                    "week_start_date": first.week_start_date,
                    "week_end_date": first.week_end_date,
                    "year": first.source_year,
                    "week": first.source_week,
                    "year_week": first.year_week,
                    "source_name": NDCU_WEEKLY_SOURCE,
                    "source_document": first.source_document,
                    "source_url": first.source_url,
                    "source_sha256": first.source_sha256,
                    "source_retrieved_at": first.source_retrieved_at,
                    "parser_version": first.parser_version,
                    "dengue_cases": sum(row.current_week_cases for row in selected),
                    "case_status": "observed",
                    "component_regions": "|".join(row.rdhs_name for row in selected),
                    "component_current_week_cases": "|".join(
                        str(row.current_week_cases) for row in selected
                    ),
                    "component_revision_flags": "|".join(
                        "true" if row.revision_flag else "false" for row in selected
                    ),
                    "component_page_bboxes": component_page_bboxes,
                    "printed_date_text": first.printed_date_text,
                    "comparison_note": (
                        NDCU_COMPONENT_NOTE if district.district_name == "Ampara" else ""
                    ),
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values(["week_start_date", "district_id"])
        .reset_index(drop=True)
    )


def _load_wer_rows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "district_id",
        "district_name",
        "week_start_date",
        "week_end_date",
        "year_week",
        "source_name",
        "source_document",
        "dengue_cases",
    }
    missing = required - set(df.columns)
    if missing:
        raise ContractError(f"WER candidate cases missing required columns: {sorted(missing)}")
    out = df.copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"]).dt.date
    out["week_end_date"] = pd.to_datetime(out["week_end_date"]).dt.date
    return out


def build_source_comparison(ndcu_rows: pd.DataFrame, wer_rows: pd.DataFrame) -> pd.DataFrame:
    ndcu = ndcu_rows.copy()
    wer = wer_rows.copy()
    ndcu["week_start_date"] = pd.to_datetime(ndcu["week_start_date"]).dt.date
    ndcu["week_end_date"] = pd.to_datetime(ndcu["week_end_date"]).dt.date

    records: list[dict[str, Any]] = []
    for _, ndcu_row in ndcu[ndcu["dengue_cases"].notna()].iterrows():
        candidates = wer[
            wer["district_id"].eq(ndcu_row["district_id"])
            & wer["dengue_cases"].notna()
            & (wer["week_start_date"] <= ndcu_row["week_end_date"])
            & (wer["week_end_date"] >= ndcu_row["week_start_date"])
        ]
        for _, wer_row in candidates.iterrows():
            overlap_start = max(ndcu_row["week_start_date"], wer_row["week_start_date"])
            overlap_end = min(ndcu_row["week_end_date"], wer_row["week_end_date"])
            overlap_days = (overlap_end - overlap_start).days + 1
            exact_interval = (
                ndcu_row["week_start_date"] == wer_row["week_start_date"]
                and ndcu_row["week_end_date"] == wer_row["week_end_date"]
            )
            reason = EXACT_INTERVAL if exact_interval else NOT_COMPARABLE_INTERVAL
            records.append(
                {
                    "district_id": ndcu_row["district_id"],
                    "district_name": ndcu_row["district_name"],
                    "wer_source_name": wer_row["source_name"],
                    "wer_year_week": wer_row["year_week"],
                    "wer_week_start_date": wer_row["week_start_date"].isoformat(),
                    "wer_week_end_date": wer_row["week_end_date"].isoformat(),
                    "wer_dengue_cases": int(wer_row["dengue_cases"]),
                    "wer_source_document": wer_row["source_document"],
                    "wer_source_url": wer_row.get("source_url", ""),
                    "ndcu_source_name": ndcu_row["source_name"],
                    "ndcu_year_week": ndcu_row["year_week"],
                    "ndcu_week_start_date": ndcu_row["week_start_date"].isoformat(),
                    "ndcu_week_end_date": ndcu_row["week_end_date"].isoformat(),
                    "ndcu_dengue_cases": int(ndcu_row["dengue_cases"]),
                    "ndcu_source_document": ndcu_row["source_document"],
                    "ndcu_source_url": ndcu_row["source_url"],
                    "overlap_start_date": overlap_start.isoformat(),
                    "overlap_end_date": overlap_end.isoformat(),
                    "overlap_days": overlap_days,
                    "comparable_exact_interval": exact_interval,
                    "reason": reason,
                    "chosen_source_for_canonical_period": wer_row["source_name"],
                    "chosen_value_for_canonical_period": int(wer_row["dengue_cases"]),
                }
            )
    columns = [
        "district_id",
        "district_name",
        "wer_source_name",
        "wer_year_week",
        "wer_week_start_date",
        "wer_week_end_date",
        "wer_dengue_cases",
        "wer_source_document",
        "wer_source_url",
        "ndcu_source_name",
        "ndcu_year_week",
        "ndcu_week_start_date",
        "ndcu_week_end_date",
        "ndcu_dengue_cases",
        "ndcu_source_document",
        "ndcu_source_url",
        "overlap_start_date",
        "overlap_end_date",
        "overlap_days",
        "comparable_exact_interval",
        "reason",
        "chosen_source_for_canonical_period",
        "chosen_value_for_canonical_period",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        ["district_id", "ndcu_week_start_date", "wer_week_start_date"], kind="mergesort"
    )


def comparison_summary(
    comparison: pd.DataFrame, ndcu_district_rows: pd.DataFrame
) -> dict[str, Any]:
    if comparison.empty:
        reason_counts: dict[str, int] = {}
        exact_pairs = 0
    else:
        reason_counts = {
            str(key): int(value)
            for key, value in comparison["reason"].value_counts().to_dict().items()
        }
        exact_pairs = int(comparison["comparable_exact_interval"].sum())
    return {
        "ndcu_source": NDCU_WEEKLY_SOURCE,
        "ndcu_periods": sorted(str(value) for value in ndcu_district_rows["year_week"].unique()),
        "ndcu_district_rows": int(len(ndcu_district_rows)),
        "overlapping_pairs": int(len(comparison)),
        "exact_comparable_pairs": exact_pairs,
        "reason_counts": reason_counts,
        "canonical_note": (
            "This comparison is diagnostic only. NDCU Monday-Sunday rows are not promoted "
            "onto the WER Saturday-Friday canonical grid."
        ),
    }


def run_comparison(
    *,
    ndcu_pdf: Path,
    wer_cases: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    components = extract_ndcu_component_rows(ndcu_pdf)
    component_df = ndcu_component_rows_frame(components)
    ndcu_districts = aggregate_ndcu_to_districts(components)
    wer = _load_wer_rows(wer_cases)
    comparison = build_source_comparison(ndcu_districts, wer)
    summary = comparison_summary(comparison, ndcu_districts)

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "historical_source_comparison.csv", index=False)
    ndcu_districts.assign(
        week_start_date=ndcu_districts["week_start_date"].astype(str),
        week_end_date=ndcu_districts["week_end_date"].astype(str),
    ).to_json(
        reports_dir / "historical_ndcu_derived_rows.json",
        orient="records",
        indent=2,
        date_format="iso",
    )
    component_df.to_json(
        reports_dir / "historical_ndcu_component_rows.json",
        orient="records",
        indent=2,
    )
    (reports_dir / "historical_source_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare retained NDCU weekly rows against WER rows."
    )
    parser.add_argument(
        "--ndcu-pdf",
        type=Path,
        default=Path("data/raw/dengue/ndcu_issue_2026_w01_342a26e3b5eb.pdf"),
    )
    parser.add_argument(
        "--wer-cases",
        type=Path,
        default=Path("data/processed/dengue_cases_weekly.parquet"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_comparison(
        ndcu_pdf=args.ndcu_pdf,
        wer_cases=args.wer_cases,
        reports_dir=args.reports_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
