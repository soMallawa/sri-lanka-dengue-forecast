from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import date, timedelta

import pymupdf as fitz

from dengue_forecast.contracts import ContractError
from dengue_forecast.dengue.models import (
    FIRST_WEEK_CORROBORATION_WARNING,
    PARSER_VERSION,
    RECONCILIATION_FIRST_WEEK_CORROBORATION,
    RECONCILIATION_REPORTED_WEEKLY_MATCH,
    ParsedDengueReport,
    ParsedRegionCase,
    SourceDocument,
)
from dengue_forecast.utils.dates import validate_week_boundary

MONTHS = {
    "jan": 1,
    "janu": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "apri": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "augu": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "nove": 11,
    "november": 11,
    "dec": 12,
    "dece": 12,
    "december": 12,
}

RDHS_TO_CANONICAL = {
    "Colombo": "Colombo",
    "Gampaha": "Gampaha",
    "paha": "Gampaha",
    "Kalutara": "Kalutara",
    "Kandy": "Kandy",
    "Matale": "Matale",
    "NuwaraEliya": "Nuwara Eliya",
    "Nuwara": "Nuwara Eliya",
    "Galle": "Galle",
    "Hambantota": "Hambantota",
    "Hambant": "Hambantota",
    "Hambanto": "Hambantota",
    "Matara": "Matara",
    "0Matara": "Matara",
    "Jaffna": "Jaffna",
    "Kilinochchi": "Kilinochchi",
    "Kili-": "Kilinochchi",
    "Mannar": "Mannar",
    "Vavuniya": "Vavuniya",
    "Mullaitivu": "Mullaitivu",
    "Batticaloa": "Batticaloa",
    "Ampara": "Ampara",
    "Trincomalee": "Trincomalee",
    "Trincomal": "Trincomalee",
    "Kurunegala": "Kurunegala",
    "Kurunega": "Kurunegala",
    "Puttalam": "Puttalam",
    "Anuradhapura": "Anuradhapura",
    "Anuradhapur": "Anuradhapura",
    "Anuradhapu": "Anuradhapura",
    "Anuradha": "Anuradhapura",
    "Polonnaruwa": "Polonnaruwa",
    "Polonnar": "Polonnaruwa",
    "Polonnaruw": "Polonnaruwa",
    "Polonnaru": "Polonnaruwa",
    "Badulla": "Badulla",
    "Monaragala": "Monaragala",
    "Monaraga": "Monaragala",
    "Moneragala": "Monaragala",
    "Ratnapura": "Ratnapura",
    "Ratnapur": "Ratnapura",
    "Kegalle": "Kegalle",
    "Kalmunai": "Kalmunai",
    "Kalmune": "Kalmunai",
    "Kalmunei": "Kalmunai",
}


# Source-year week-1 anchors evidenced from retained WER early/mid/late reports.
# These are source observation calendars, not issue/publication calendars.
SOURCE_YEAR_WEEK1_SATURDAY = {
    2009: date(2008, 12, 27),
    2010: date(2010, 1, 2),
    2011: date(2011, 1, 1),
    2012: date(2011, 12, 31),
    2013: date(2012, 12, 29),
    2014: date(2013, 12, 28),
    2015: date(2014, 12, 27),
    2016: date(2015, 12, 26),
    2017: date(2016, 12, 31),
    2018: date(2017, 12, 30),
    2019: date(2018, 12, 29),
    2020: date(2019, 12, 28),
    2021: date(2020, 12, 26),
    2022: date(2022, 1, 1),
    2023: date(2022, 12, 31),
    2024: date(2023, 12, 30),
    2025: date(2024, 12, 28),
}

MODERN_ORDER = [
    "Colombo",
    "Gampaha",
    "Kalutara",
    "Kandy",
    "Matale",
    "NuwaraEliya",
    "Galle",
    "Hambantota",
    "Matara",
    "Jaffna",
    "Kilinochchi",
    "Mannar",
    "Vavuniya",
    "Mullaitivu",
    "Batticaloa",
    "Ampara",
    "Trincomalee",
    "Kurunegala",
    "Puttalam",
    "Anuradhapur",
    "Polonnaruwa",
    "Badulla",
    "Monaragala",
    "Ratnapura",
    "Kegalle",
    "Kalmune",
]

MODERN_TABLE_TITLE_MARKERS = (
    "Selected notifiable diseases",
    "Distribution of Notified Diseases",
)

def _clean_int(value: str) -> int | None:
    text = value.replace(",", "").strip()
    if not text or text in {"-", "NA", "Not"}:
        return None
    if not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def _page_text(document: SourceDocument) -> str:
    with fitz.open(document.path) as pdf:
        return "\n".join(page.get_text() for page in pdf)


def _ord_day(value: str) -> int:
    return int(re.sub(r"\D", "", value))


_DAY_TOKEN = r"\d{1,2}\s*(?:st|nd|rd|th)?"
_MONTH_TOKEN = r"[A-Za-z]+"
_PERIOD_PATTERNS = (
    re.compile(
        rf"(?P<sd>{_DAY_TOKEN})\s*"
        rf"(?:(?P<sm>{_MONTH_TOKEN})\s*)?[-–—]\s*"
        rf"(?P<ed>{_DAY_TOKEN})\s*"
        rf"(?P<em>{_MONTH_TOKEN})\s*-?,?\s*(?P<year>\d{{4}})\s*"
        r"\((?P<week>\d{1,2})(?:st|nd|rd|th)?\s*Week\)",
        flags=re.I,
    ),
    re.compile(
        rf"(?P<sd>{_DAY_TOKEN})\s+"
        rf"(?P<sm>{_MONTH_TOKEN})\s+"
        rf"(?P<ed>{_DAY_TOKEN})\s*"
        rf"(?P<em>{_MONTH_TOKEN})\s*-?,?\s*(?P<year>\d{{4}})\s*"
        r"\((?P<week>\d{1,2})(?:st|nd|rd|th)?\s*Week\)",
        flags=re.I,
    ),
)


def _month_number(token: str) -> int:
    key = token.casefold()
    try:
        return MONTHS[key]
    except KeyError as exc:
        raise ContractError(f"Unrecognized printed month token {token!r}") from exc


def _parse_period(text: str, *, issue_year: int | None) -> tuple[date, date, int, int]:
    normalized = re.sub(r"\s+", " ", text.replace("–", "-").replace("—", "-"))
    matches = [
        match
        for pattern in _PERIOD_PATTERNS
        for match in pattern.finditer(normalized)
    ]
    if not matches:
        raise ContractError("Could not extract report period/week")
    match = next(
        (
            m
            for m in matches
            if any(
                marker in normalized[max(0, m.start() - 140) : m.start()]
                for marker in (
                    "Selected notifiable diseases reported by Medical Officers of Health",
                    "Distribution of Notified Diseases reported by Medical Officers of Health",
                )
            )
        ),
        next(
            (m for m in matches if "Table" in normalized[max(0, m.start() - 90) : m.start()]),
            matches[0],
        ),
    )
    source_year = int(match.group("year"))
    source_week = int(match.group("week"))
    end_month = _month_number(match.group("em"))
    explicit_start_month = match.group("sm")
    start_month = _month_number(explicit_start_month) if explicit_start_month else None
    start_day = _ord_day(match.group("sd"))
    end_day = _ord_day(match.group("ed"))
    start, end = _infer_seven_day_period(
        start_day=start_day,
        start_month=start_month,
        end_day=end_day,
        end_month=end_month,
        printed_year=source_year,
        issue_year=issue_year,
    )
    validate_week_boundary(start, end)
    expected_start = _expected_saturday_week_start(source_year, source_week)
    if start != expected_start:
        raise ContractError(
            f"Printed period {start} to {end} is inconsistent with source "
            f"{source_year}-W{source_week:02d}; "
            f"expected start {expected_start}"
        )
    return start, end, source_year, source_week


def _expected_saturday_week_start(source_year: int, source_week: int) -> date:
    if not 1 <= source_week <= 53:
        raise ContractError(f"Source week must be 1..53, got {source_week}")
    try:
        first_week_start = SOURCE_YEAR_WEEK1_SATURDAY[source_year]
    except KeyError as exc:
        raise ContractError(
            f"No evidenced Saturday-Friday source calendar anchor for {source_year}"
        ) from exc
    return first_week_start + timedelta(days=7 * (source_week - 1))


def _infer_seven_day_period(
    *,
    start_day: int,
    start_month: int | None,
    end_day: int,
    end_month: int,
    printed_year: int,
    issue_year: int | None,
) -> tuple[date, date]:
    candidate_years = {printed_year - 1, printed_year, printed_year + 1}
    if issue_year is not None:
        candidate_years.update({issue_year - 1, issue_year, issue_year + 1})

    def rank(year: int) -> tuple[int, int]:
        crosses_into_january_issue = (
            issue_year is not None
            and issue_year == printed_year + 1
            and end_month == 1
            and (start_month is None or start_month == 12)
        )
        anchor = issue_year if crosses_into_january_issue else printed_year
        return (abs(year - anchor), abs(year - printed_year))

    for end_year in sorted(candidate_years, key=rank):
        try:
            end = date(end_year, end_month, end_day)
        except ValueError:
            continue
        start = end - timedelta(days=6)
        if start.day != start_day:
            continue
        if start_month is not None and start.month != start_month:
            continue
        return start, end

    start_label = f"{start_day}" if start_month is None else f"{start_day}/{start_month}"
    raise ContractError(
        "Could not reconcile printed period to exactly 7 days: "
        f"{start_label} to {end_day}/{end_month}/{printed_year}"
    )


def _bbox_union(
    words: Iterable[tuple[float, float, float, float, str]],
) -> tuple[float, float, float, float]:
    seq = list(words)
    return (
        min(w[0] for w in seq),
        min(w[1] for w in seq),
        max(w[2] for w in seq),
        max(w[3] for w in seq),
    )


def _first_week_corroboration_total(
    *,
    source_week: int,
    rows: list[ParsedRegionCase],
    reported_national_total: int | None,
    national_cumulative_total: int | None,
) -> int | None:
    if source_week != 1:
        return None
    if reported_national_total is None or national_cumulative_total is None:
        return None
    if len(rows) != 26:
        return None
    current_values = [row.current_week_cases for row in rows]
    cumulative_values = [row.cumulative_cases for row in rows]
    if any(value is None for value in current_values + cumulative_values):
        return None
    if any(
        row.current_week_cases != row.cumulative_cases
        for row in rows
    ):
        return None
    calculated = sum(int(value) for value in current_values if value is not None)
    if calculated != national_cumulative_total:
        return None
    if reported_national_total == calculated:
        return None
    return calculated


class BaseDengueReportParser(ABC):
    name = "base"

    @abstractmethod
    def can_parse(self, document: SourceDocument) -> float:
        raise NotImplementedError

    @abstractmethod
    def parse(self, document: SourceDocument) -> ParsedDengueReport:
        raise NotImplementedError


class WerModernTableParser(BaseDengueReportParser):
    name = "wer_modern_rotated"

    def can_parse(self, document: SourceDocument) -> float:
        with fitz.open(document.path) as pdf:
            for page in pdf:
                if self._has_rotated_table_headers(page):
                    return 0.95
        return 0.0

    def parse(self, document: SourceDocument) -> ParsedDengueReport:
        full_text = _page_text(document)
        issue_year = document.descriptor.get("issue_year")
        start, end, source_year, source_week = _parse_period(full_text, issue_year=issue_year)

        rows: list[ParsedRegionCase] = []
        reported_national_total = None
        national_cumulative_total = None
        with fitz.open(document.path) as pdf:
            page_index = self._find_table_page(pdf)
            page = pdf[page_index]
            words = [w[:5] for w in page.get_text("words")]
            table_words = [w for w in words if w[0] < 545 and 70 <= w[1] <= 805]
            rdhs_columns = self._rdhs_columns(table_words, page_height=float(page.rect.height))
            if len(rdhs_columns) != 26:
                found = [column[0] for column in rdhs_columns]
                raise ContractError(
                    f"Modern WER expected 26 RDHS rows, found {len(rdhs_columns)}: {found}"
                )
            national_x = self._national_column_x(table_words, page_height=float(page.rect.height))
            national_col = [w for w in table_words if abs(float(w[0]) - national_x) <= 2.5]
            current_y, cumulative_y = self._dengue_value_ys(table_words, national_col)
            completeness_y = self._completeness_value_y(table_words)
            for raw_name, region, x0, label_words in rdhs_columns:
                same_col = [w for w in table_words if abs(float(w[0]) - x0) <= 3.0]
                current = self._value_near_y(same_col, current_y)
                cumulative = self._value_near_y(same_col, cumulative_y)
                completeness = (
                    self._value_near_y(same_col, completeness_y)
                    if completeness_y is not None
                    else None
                )
                rows.append(
                    ParsedRegionCase(
                        region_name=region,
                        current_week_cases=current,
                        cumulative_cases=cumulative,
                        reporting_returns_pct=completeness,
                        page_number=page_index + 1,
                        bbox=_bbox_union(same_col + label_words),
                        raw_values={
                            "raw_region_name": raw_name,
                            "orientation": "rotated_90_ccw",
                            "quality_metric": "completeness_c_pct"
                            if completeness_y is not None
                            else "unavailable",
                        },
                        case_status="observed" if current is not None else "missing",
                        completeness_flag=None
                        if completeness_y is not None
                        else "quality_unavailable",
                    )
                )
            reported_national_total = self._value_near_y(national_col, current_y)
            national_cumulative_total = self._value_near_y(national_col, cumulative_y)
        missing_current = [row.region_name for row in rows if row.current_week_cases is None]
        if missing_current:
            raise ContractError(
                "Modern WER dengue weekly A cells missing for RDHS rows: "
                f"{', '.join(missing_current)}"
            )
        calculated = sum(row.current_week_cases or 0 for row in rows)
        if reported_national_total is None:
            raise ContractError("Modern WER national total not found")
        corroborating_total = None
        reconciliation_method = RECONCILIATION_REPORTED_WEEKLY_MATCH
        reconciliation_warning = None
        if calculated != reported_national_total:
            corroborating_total = _first_week_corroboration_total(
                source_week=source_week,
                rows=rows,
                reported_national_total=reported_national_total,
                national_cumulative_total=national_cumulative_total,
            )
        if corroborating_total is not None:
            reconciliation_method = RECONCILIATION_FIRST_WEEK_CORROBORATION
            reconciliation_warning = FIRST_WEEK_CORROBORATION_WARNING
        elif calculated != reported_national_total:
            raise ContractError(
                "Modern WER national reconciliation failed: "
                f"RDHS sum {calculated} != national {reported_national_total}"
            )
        return self._report(
            document,
            start,
            end,
            source_year,
            source_week,
            rows,
            reported_national_total,
            corroborating_total,
            reconciliation_method,
            reconciliation_warning,
            full_text,
        )

    @staticmethod
    def _find_table_page(pdf: fitz.Document) -> int:
        parser = WerModernTableParser()
        for idx, page in enumerate(pdf):
            if parser._has_rotated_table_headers(page):
                return idx
        raise ContractError("Modern WER Table 1 page not found")

    @staticmethod
    def _has_rotated_table_headers(page: fitz.Page) -> bool:
        text = page.get_text()
        if not any(marker in text for marker in MODERN_TABLE_TITLE_MARKERS):
            return False
        directions = [
            line["dir"]
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
        ]
        if not any(
            WerModernTableParser._is_near_vertical_up(direction) for direction in directions
        ):
            return False
        words = [w[:5] for w in page.get_text("words")]
        rdhs = [
            w
            for w in words
            if str(w[4]).strip().casefold() == "rdhs"
            and 35 <= float(w[0]) <= 65
            and 650 <= float(w[1]) <= 810
        ]
        dengue = [
            w
            for w in words
            if str(w[4]).strip().casefold().startswith("dengue")
            and 35 <= float(w[0]) <= 65
            and 650 <= float(w[1]) <= 760
        ]
        ab = [
            w
            for w in words
            if str(w[4]).strip() in {"A", "B"}
            and 55 <= float(w[0]) <= 90
            and 650 <= float(w[1]) <= 760
        ]
        return bool(rdhs and dengue and {str(w[4]).strip() for w in ab} == {"A", "B"})

    @staticmethod
    def _is_near_vertical_up(direction: tuple[float, float]) -> bool:
        dx, dy = direction
        return abs(float(dx)) < 1e-3 and abs(float(dy) + 1.0) < 1e-3

    @staticmethod
    def _value_near_y(
        words: list[tuple[float, float, float, float, str]], target_y: float
    ) -> int | None:
        candidates = [
            (abs(float(w[1]) - target_y), _clean_int(str(w[4])))
            for w in words
            if abs(float(w[1]) - target_y) < 10.0
        ]
        candidates = [(distance, value) for distance, value in candidates if value is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _rdhs_columns(
        table_words: list[tuple[float, float, float, float, str]], *, page_height: float
    ) -> list[tuple[str, str, float, list[tuple[float, float, float, float, str]]]]:
        label_floor = page_height * 0.86
        label_words = [
            w
            for w in table_words
            if label_floor <= float(w[1]) <= page_height - 35
            and not str(w[4]).replace(",", "").isdigit()
            and str(w[4]).strip() != "%"
        ]
        grouped: list[list[tuple[float, float, float, float, str]]] = []
        for word in sorted(label_words, key=lambda w: (float(w[0]), float(w[1]))):
            token = str(word[4]).strip()
            if token.casefold() in {"rdhs", "division", "srilanka"}:
                continue
            if grouped and abs(float(grouped[-1][0][0]) - float(word[0])) <= 1.0:
                grouped[-1].append(word)
            else:
                grouped.append([word])

        columns: list[tuple[str, str, float, list[tuple[float, float, float, float, str]]]] = []
        for group in grouped:
            parts = [
                str(w[4]).strip() for w in sorted(group, key=lambda w: float(w[1]), reverse=True)
            ]
            raw_name = "".join(parts)
            region = RDHS_TO_CANONICAL.get(raw_name)
            if region is None:
                spaced = " ".join(parts)
                region = RDHS_TO_CANONICAL.get(spaced)
                raw_name = spaced if region is not None else raw_name
            if region is None:
                continue
            columns.append((raw_name, region, float(group[0][0]), group))
        return sorted(columns, key=lambda item: item[2])

    @staticmethod
    def _national_column_x(
        table_words: list[tuple[float, float, float, float, str]], *, page_height: float
    ) -> float:
        label_floor = page_height * 0.86
        matches = [
            w
            for w in table_words
            if label_floor <= float(w[1]) <= page_height - 35
            and str(w[4]).strip().casefold() in {"srilanka", "sri", "lanka"}
        ]
        if not matches:
            raise ContractError("Modern WER SRILANKA national column label not found")
        return max(float(w[0]) for w in matches)

    @staticmethod
    def _dengue_value_ys(
        table_words: list[tuple[float, float, float, float, str]],
        national_words: list[tuple[float, float, float, float, str]],
    ) -> tuple[float, float]:
        dengue_headers = [
            w
            for w in table_words
            if str(w[4]).strip().casefold().startswith("dengue")
            and 35 <= float(w[0]) <= 65
            and 650 <= float(w[1]) <= 760
        ]
        if not dengue_headers:
            synthetic_numeric = [
                w
                for w in national_words
                if _clean_int(str(w[4])) is not None and 600 <= float(w[1]) <= 760
            ]
            by_y = sorted(synthetic_numeric, key=lambda w: float(w[1]), reverse=True)
            if len(by_y) >= 2:
                return float(by_y[0][1]), float(by_y[1][1])
            raise ContractError("Modern WER dengue header not found")
        header_top = min(float(w[1]) for w in dengue_headers)
        header_bottom = max(float(w[3]) for w in dengue_headers)
        ab_headers = [
            w
            for w in table_words
            if str(w[4]).strip() in {"A", "B"}
            and 55 <= float(w[0]) <= 85
            and header_top - 45 <= float(w[1]) <= header_bottom + 8
        ]
        a_headers = [w for w in ab_headers if str(w[4]).strip() == "A"]
        b_headers = [w for w in ab_headers if str(w[4]).strip() == "B"]
        if not a_headers or not b_headers:
            raise ContractError("Modern WER dengue A/B headers not found")

        a_label_y = max(float(w[1]) for w in a_headers)
        b_label_y = max(float(w[1]) for w in b_headers)
        numeric = [
            w
            for w in national_words
            if _clean_int(str(w[4])) is not None
            and header_top - 45 <= float(w[1]) <= header_bottom + 8
        ]
        current = [w for w in numeric if float(w[1]) <= a_label_y + 2]
        cumulative = [w for w in numeric if float(w[1]) <= b_label_y + 2]
        if not current or not cumulative:
            raise ContractError("Modern WER dengue A/B rows not found in national column")
        current_y = float(min(current, key=lambda w: abs(float(w[1]) - a_label_y))[1])
        cumulative_y = float(min(cumulative, key=lambda w: abs(float(w[1]) - b_label_y))[1])
        if abs(current_y - cumulative_y) < 8:
            raise ContractError("Modern WER dengue A/B rows are not distinct")
        return current_y, cumulative_y

    @staticmethod
    def _completeness_value_y(
        table_words: list[tuple[float, float, float, float, str]],
    ) -> float | None:
        headers = [
            w
            for w in table_words
            if str(w[4]).strip().casefold() in {"c**", "completeness"}
            and 55 <= float(w[0]) <= 90
            and 70 <= float(w[1]) <= 130
        ]
        if not headers:
            return None
        return max(float(w[1]) for w in headers)

    def _report(
        self,
        document: SourceDocument,
        start: date,
        end: date,
        source_year: int,
        source_week: int,
        rows: list[ParsedRegionCase],
        reported_national_total: int | None,
        corroborating_national_total: int | None,
        national_reconciliation_method: str,
        national_reconciliation_warning: str | None,
        full_text: str,
    ) -> ParsedDengueReport:
        return ParsedDengueReport(
            source_name=document.source_name,
            source_document=document.path.name,
            source_url=document.source_url,
            source_retrieved_at=document.retrieved_at,
            parser_version=PARSER_VERSION,
            source_year=source_year,
            source_week=source_week,
            week_start_date=start,
            week_end_date=end,
            issue_year=document.descriptor.get("issue_year"),
            issue_week=document.descriptor.get("issue_week"),
            reported_national_total=reported_national_total,
            calculated_rdhs_total=sum(row.current_week_cases or 0 for row in rows),
            rows=rows,
            parser_name=self.name,
            authentic_snippet=self._snippet(full_text),
            corroborating_national_total=corroborating_national_total,
            national_reconciliation_method=national_reconciliation_method,
            national_reconciliation_warning=national_reconciliation_warning,
        )

    @staticmethod
    def _snippet(text: str) -> str:
        idx = text.find("A=current week")
        if idx < 0:
            idx = text.find("Table 1: Selected notifiable diseases")
        return text[idx : idx + 600]


class WerLegacyTableParser(BaseDengueReportParser):
    name = "wer_legacy_horizontal"

    def can_parse(self, document: SourceDocument) -> float:
        text = _page_text(document)
        return (
            0.9
            if "Selected notifiable diseases" in text
            and "Dengue" in text
            and "A = Cases reported during the current week" in text
            else 0.0
        )

    def parse(self, document: SourceDocument) -> ParsedDengueReport:
        full_text = _page_text(document)
        issue_year = document.descriptor.get("issue_year")
        issue_week = document.descriptor.get("issue_week")
        start, end, source_year, source_week = _parse_period(full_text, issue_year=issue_year)
        rows: list[ParsedRegionCase] = []
        reported_national_total = None
        with fitz.open(document.path) as pdf:
            page_index = self._find_table_page(pdf)
            page = pdf[page_index]
            words = [w[:5] for w in page.get_text("words")]
            current_x, cumulative_x = self._dengue_column_xs(words)
            returns_x = self._returns_column_x(words, cumulative_x)
            row_heads = self._row_heads(words)
            for raw_name, y in row_heads:
                row_words = sorted(
                    [w for w in words if abs(float(w[1]) - y) < 4 and float(w[0]) > 75],
                    key=lambda w: float(w[0]),
                )
                current = self._value_near_x(row_words, current_x)
                cumulative = self._value_near_x(row_words, cumulative_x)
                if raw_name == "SRI LANKA":
                    reported_national_total = current
                    continue
                region = RDHS_TO_CANONICAL.get(raw_name)
                if region is None:
                    continue
                if current is None:
                    raise ContractError(f"Legacy WER dengue weekly A cell missing for {region}")
                if cumulative is None:
                    raise ContractError(f"Legacy WER dengue cumulative B cell missing for {region}")
                reporting_returns = self._reporting_returns_pct(
                    row_words, cumulative_x, returns_x
                )
                line_words = [w for w in words if abs(float(w[1]) - y) < 4]
                bbox = _bbox_union(line_words) if line_words else (0.0, 0.0, 0.0, 0.0)
                canonical_current, case_status, completeness_flag = (
                    self._legacy_case_observation(current, reporting_returns)
                )
                rows.append(
                    ParsedRegionCase(
                        region_name=region,
                        current_week_cases=canonical_current,
                        cumulative_cases=cumulative,
                        reporting_returns_pct=reporting_returns,
                        page_number=page_index + 1,
                        bbox=bbox,
                        raw_values={
                            "raw_region_name": raw_name,
                            "raw_current_week_cases": current,
                            "raw_reporting_returns_pct": reporting_returns,
                            "quality_metric": "reporting_returns_pct"
                            if reporting_returns is not None
                            else "unavailable",
                        },
                        case_status=case_status,
                        completeness_flag=completeness_flag,
                    )
                )
        if len(rows) != 26:
            raise ContractError(f"Legacy WER expected 26 RDHS rows, found {len(rows)}")
        calculated = sum(
            int(row.raw_values.get("raw_current_week_cases") or row.current_week_cases or 0)
            for row in rows
        )
        if reported_national_total is None:
            raise ContractError("Legacy WER national total not found")
        if calculated != reported_national_total:
            raise ContractError(
                "Legacy WER national reconciliation failed: "
                f"RDHS sum {calculated} != national {reported_national_total}"
            )
        return ParsedDengueReport(
            source_name=document.source_name,
            source_document=document.path.name,
            source_url=document.source_url,
            source_retrieved_at=document.retrieved_at,
            parser_version=PARSER_VERSION,
            source_year=source_year,
            source_week=source_week,
            week_start_date=start,
            week_end_date=end,
            issue_year=issue_year,
            issue_week=issue_week,
            reported_national_total=reported_national_total,
            calculated_rdhs_total=calculated,
            rows=rows,
            parser_name=self.name,
            authentic_snippet=self._snippet(full_text),
        )

    @staticmethod
    def _find_table_page(pdf: fitz.Document) -> int:
        for idx, page in enumerate(pdf):
            text = page.get_text()
            if "Selected notifiable diseases" in text and "Dengue" in text and "SRI LANKA" in text:
                return idx
        raise ContractError("Legacy dengue table page not found")

    @staticmethod
    def _parse_line(line: str) -> tuple[str, int, int | None, int | None] | None:
        names = sorted(RDHS_TO_CANONICAL, key=len, reverse=True) + ["SRI LANKA"]
        stripped = line.strip()
        raw_name = next((name for name in names if stripped.startswith(name)), None)
        if raw_name is None:
            return None
        nums = [int(value) for value in re.findall(r"\b\d+\b", stripped[len(raw_name) :])]
        if len(nums) < 2:
            return None
        returns = nums[-1] if nums else None
        return raw_name, nums[0], nums[1], returns

    @staticmethod
    def _dengue_column_xs(
        words: list[tuple[float, float, float, float, str]],
    ) -> tuple[float, float]:
        dengue_words = [
            w
            for w in words
            if str(w[4]).strip().casefold().startswith("dengue")
            and 75 <= float(w[0]) <= 520
            and 70 <= float(w[1]) <= 130
        ]
        if not dengue_words:
            raise ContractError("Legacy WER dengue header not found")
        header_left = min(float(w[0]) for w in dengue_words) - 8
        header_right = max(float(w[2]) for w in dengue_words) + 40
        ab_headers = sorted(
            [
                w
                for w in words
                if str(w[4]).strip() in {"A", "B"}
                and header_left <= float(w[0]) <= header_right
                and 120 <= float(w[1]) <= 160
            ],
            key=lambda w: float(w[0]),
        )
        a_headers = [w for w in ab_headers if str(w[4]).strip() == "A"]
        b_headers = [w for w in ab_headers if str(w[4]).strip() == "B"]
        if not a_headers or not b_headers:
            raise ContractError("Legacy WER dengue A/B headers not found")
        a_x = (float(a_headers[0][0]) + float(a_headers[0][2])) / 2
        b_x = (float(b_headers[0][0]) + float(b_headers[0][2])) / 2
        if b_x <= a_x or abs(b_x - a_x) < 8:
            raise ContractError("Legacy WER dengue A/B columns are not distinct")
        return a_x, b_x

    @staticmethod
    def _value_near_x(
        words: list[tuple[float, float, float, float, str]], target_x: float
    ) -> int | None:
        candidates = [
            (abs(((float(w[0]) + float(w[2])) / 2) - target_x), _clean_int(str(w[4])))
            for w in words
            if abs(((float(w[0]) + float(w[2])) / 2) - target_x) < 13.0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _reporting_returns_pct(
        row_words: list[tuple[float, float, float, float, str]],
        cumulative_x: float,
        returns_x: float | None,
    ) -> int | None:
        if returns_x is None:
            return None
        candidates = [
            (abs(((float(w[0]) + float(w[2])) / 2) - returns_x), _clean_int(str(w[4])))
            for w in row_words
            if ((float(w[0]) + float(w[2])) / 2) > cumulative_x + 20.0
            and abs(((float(w[0]) + float(w[2])) / 2) - returns_x) < 18.0
        ]
        candidates = [(distance, value) for distance, value in candidates if value is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _returns_column_x(
        words: list[tuple[float, float, float, float, str]], cumulative_x: float
    ) -> float | None:
        header_words = sorted(
            [
                w
                for w in words
                if ((float(w[0]) + float(w[2])) / 2) > cumulative_x + 20.0
                and 90 <= float(w[1]) <= 155
            ],
            key=lambda w: (float(w[0]), float(w[1])),
        )
        for word in header_words:
            token = str(word[4]).strip().casefold().replace("-", "")
            if token in {"returns", "returns%"}:
                return (float(word[0]) + float(word[2])) / 2

        tagged = [
            (str(w[4]).strip().casefold().replace("-", ""), w) for w in header_words
        ]
        group = [
            w
            for token, w in tagged
            if token == "%"
            or token == "turns"
            or (token == "re" and str(w[4]).strip().endswith("-"))
        ]
        if (
            len([w for token, w in tagged if token == "re"]) >= 2
            and any(token == "turns" for token, _ in tagged)
            and any(token == "%" for token, _ in tagged)
            and max(float(w[2]) for w in group) - min(float(w[0]) for w in group) <= 35
        ):
            return (min(float(w[0]) for w in group) + max(float(w[2]) for w in group)) / 2
        return None

    @staticmethod
    def _legacy_case_observation(
        current_week_cases: int, reporting_returns_pct: int | None
    ) -> tuple[int | None, str, str | None]:
        if current_week_cases == 0 and reporting_returns_pct == 0:
            return None, "missing", "zero_with_zero_returns"
        return (
            current_week_cases,
            "observed",
            None if reporting_returns_pct is not None else "quality_unavailable",
        )

    @staticmethod
    def _row_heads(words: list[tuple[float, float, float, float, str]]) -> list[tuple[str, float]]:
        heads: list[tuple[str, float]] = []
        names = sorted(RDHS_TO_CANONICAL, key=len, reverse=True)
        for word in sorted(words, key=lambda w: (float(w[1]), float(w[0]))):
            text = str(word[4]).strip()
            y = float(word[1])
            if not 140 <= y <= 600:
                continue
            if text == "SRI":
                next_words = [
                    w for w in words if abs(float(w[1]) - y) < 4 and 45 <= float(w[0]) <= 80
                ]
                if any(str(w[4]).strip() == "LANKA" for w in next_words):
                    heads.append(("SRI LANKA", y))
                continue
            if text in names:
                heads.append((text, y))
        return heads

    @staticmethod
    def _snippet(text: str) -> str:
        idx = text.find("A = Cases reported during the current week")
        start = max(0, idx - 200)
        return text[start : idx + 300]


class NdcuWeeklyParser(BaseDengueReportParser):
    name = "ndcu_weekly_unsupported"

    def can_parse(self, document: SourceDocument) -> float:
        return 0.2 if "dengue.health.gov.lk" in document.source_url else 0.0

    def parse(self, document: SourceDocument) -> ParsedDengueReport:
        raise ContractError(
            "NDCU weekly PDF parser is intentionally unsupported for Stage2 canonical counts"
        )


class GenericFallbackParser(BaseDengueReportParser):
    name = "generic_quarantine"

    def can_parse(self, document: SourceDocument) -> float:
        return 0.01

    def parse(self, document: SourceDocument) -> ParsedDengueReport:
        raise ContractError("No supported dengue parser matched this document")


PARSERS: tuple[BaseDengueReportParser, ...] = (
    WerModernTableParser(),
    WerLegacyTableParser(),
    NdcuWeeklyParser(),
    GenericFallbackParser(),
)


def select_parser(document: SourceDocument) -> BaseDengueReportParser:
    scored = sorted(
        ((parser.can_parse(document), parser) for parser in PARSERS),
        key=lambda item: item[0],
        reverse=True,
    )
    if not scored or scored[0][0] <= 0:
        raise ContractError("No dengue parser matched document")
    return scored[0][1]


def parse_dengue_document(document: SourceDocument) -> ParsedDengueReport:
    return select_parser(document).parse(document)
