from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = (
    REPO_ROOT
    / "artifacts"
    / "milestone3"
    / "analysis"
    / "saved-predictions-dev-20261005-optimized"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "milestone3" / "plots"
DEFAULT_RUN_ID = "saved-predictions-dev-20261005-optimized-plots"
QUALIFICATION = (
    "development_only_qualified_retrospective_observation_time_not_operational_backtesting"
)
QUALIFICATION_LABEL = (
    "Development-only qualified retrospective observation-time; "
    "not operational backtesting; 2025 still locked."
)
REQUIRED_TABLES = {
    "config_metrics",
    "fold_metrics",
    "weather_effects",
    "district_diagnostics",
    "high_incidence_thresholds",
    "change_categories",
    "bootstrap_model_vs_persistence",
    "bootstrap_weather_effects",
}
CSV_TABLES = {name for name in REQUIRED_TABLES}
PRIMARY_ROLE = "primary_cases_only_ridge"
CHAMPION_ROLE = "preselected_development_champion"


class PlotInputError(RuntimeError):
    """Raised when saved analysis tables are incomplete or unauthenticated."""


@dataclass(frozen=True)
class PlotResult:
    output_dir: Path
    manifest_path: Path
    index_markdown_path: Path
    index_html_path: Path
    contact_sheet_path: Path
    png_count: int
    svg_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PlotInputError(f"JSON root must be an object: {path}")
    return payload


def _relative_to(path: Path, parent: Path) -> str:
    try:
        return path.relative_to(parent).as_posix()
    except ValueError as exc:
        raise PlotInputError(f"path is outside required container: {path}") from exc


def verify_analysis_dir(analysis_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    analysis_dir = analysis_dir.resolve()
    manifest_path = analysis_dir / "manifest.json"
    if not manifest_path.exists():
        raise PlotInputError(f"missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise PlotInputError("analysis manifest status is not complete")
    if manifest.get("qualification") != QUALIFICATION:
        raise PlotInputError("analysis manifest qualification is unexpected")
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise PlotInputError("analysis manifest has no tables object")
    missing = sorted(REQUIRED_TABLES - set(tables))
    if missing:
        raise PlotInputError(f"analysis manifest is missing required tables: {missing}")

    verified: dict[str, Path] = {}
    for table_name, entry in tables.items():
        if not isinstance(entry, dict):
            raise PlotInputError(f"manifest table entry is not an object: {table_name}")
        raw_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise PlotInputError(f"manifest table entry is incomplete: {table_name}")
        table_path = Path(raw_path).resolve()
        _relative_to(table_path, analysis_dir)
        if not table_path.exists():
            raise PlotInputError(f"manifest table is missing: {table_path}")
        actual_hash = sha256_file(table_path)
        if actual_hash != expected_hash:
            raise PlotInputError(
                f"manifest table hash mismatch for {table_name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        if table_name in CSV_TABLES:
            verified[table_name] = table_path
    return manifest, verified


def _read_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path, keep_default_na=True) for name, path in paths.items()}


def _require_columns(table: pd.DataFrame, table_name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise PlotInputError(f"{table_name} is missing required columns: {missing}")


def _validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    _require_columns(
        tables["config_metrics"],
        "config_metrics",
        {
            "horizon",
            "feature_set",
            "model_family",
            "role",
            "mean_fold_mae_model",
            "mean_fold_mae_persistence",
            "relative_improvement_pct_from_mean_fold_mae",
            "mean_fold_rmse_model",
            "mean_fold_rmse_persistence",
            "aggregation",
        },
    )
    _require_columns(
        tables["fold_metrics"],
        "fold_metrics",
        {
            "horizon",
            "feature_set",
            "model_family",
            "fold",
            "mae_model",
            "mae_persistence",
            "relative_improvement_pct",
            "rmse_model",
            "rmse_persistence",
        },
    )
    _require_columns(
        tables["weather_effects"],
        "weather_effects",
        {
            "horizon",
            "model_family",
            "comparison",
            "mean_fold_mae_difference_added_minus_baseline",
            "sign_definition",
        },
    )
    _require_columns(
        tables["district_diagnostics"],
        "district_diagnostics",
        {
            "horizon",
            "feature_set",
            "model_family",
            "district_id",
            "observed_target_total",
            "model_mae",
            "persistence_mae",
            "normalized_model_mae_by_mean_observed",
        },
    )
    _require_columns(
        tables["high_incidence_thresholds"],
        "high_incidence_thresholds",
        {"horizon", "fold", "feature_set", "model_family", "q90_count", "q95_count"},
    )
    _require_columns(
        tables["change_categories"],
        "change_categories",
        {"horizon", "fold", "feature_set", "model_family", "category", "count"},
    )


def _prepare_output_dir(output_root: Path, run_id: str, *, repo_root: Path = REPO_ROOT) -> Path:
    expected_root = (repo_root / "artifacts" / "milestone3" / "plots").resolve()
    output_root = output_root.resolve()
    if output_root != expected_root:
        raise PlotInputError(f"output root must be the claimed plots root: {expected_root}")
    if "/" in run_id or "\\" in run_id or run_id in {"", ".", ".."}:
        raise PlotInputError("run id must be a single directory name")
    output_dir = output_root / run_id
    try:
        output_dir.relative_to(expected_root)
    except ValueError as exc:
        raise PlotInputError(f"output directory is outside plots root: {output_dir}") from exc
    if output_dir.exists():
        raise PlotInputError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def _is_primary(row: pd.Series) -> bool:
    role = str(row.get("role", ""))
    return (
        row.get("feature_set") == "cases_only" and row.get("model_family") == "ridge"
    ) or (PRIMARY_ROLE in role)


def _is_champion(row: pd.Series) -> bool:
    return CHAMPION_ROLE in str(row.get("role", ""))


def _label(row: pd.Series) -> str:
    return f"h{int(row['horizon'])} {row['feature_set']} {row['model_family']}"


def _format_nulls(ax: plt.Axes, x: list[float], values: pd.Series, labels: list[str]) -> None:
    for xpos, value, label in zip(x, values, labels, strict=True):
        if pd.isna(value):
            ax.text(xpos, 0.0, f"{label}\nnull", ha="center", va="bottom", fontsize=7, rotation=90)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, Any]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    image = plt.imread(png)
    height, width = image.shape[:2]
    return {
        "stem": stem,
        "png": png.name,
        "svg": svg.name,
        "png_sha256": sha256_file(png),
        "svg_sha256": sha256_file(svg),
        "dimensions_px": {"width": int(width), "height": int(height)},
    }


def _style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d7dee8", linewidth=0.7)
    ax.set_axisbelow(True)


def _plot_horizon_metric(
    config: pd.DataFrame,
    output_dir: Path,
    metric: str,
    persistence_metric: str,
    stem: str,
    ylabel: str,
) -> dict[str, Any]:
    frame = config.sort_values(["horizon", "feature_set", "model_family"]).copy()
    labels = [_label(row) for _, row in frame.iterrows()]
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [
        "#c94936" if _is_primary(row) else "#2d5b8a" if _is_champion(row) else "#7d8896"
        for _, row in frame.iterrows()
    ]
    ax.bar(x - 0.18, frame[metric], width=0.36, color=colors, label="model")
    ax.bar(x + 0.18, frame[persistence_metric], width=0.36, color="#c7ced8", label="persistence")
    _format_nulls(ax, list(x - 0.18), frame[metric], labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.legend(frameon=False, ncols=2)
    _style_axis(
        ax,
        f"Horizon {ylabel} from saved development tables",
        ylabel,
    )
    ax.text(0, 1.02, QUALIFICATION_LABEL, transform=ax.transAxes, fontsize=8, color="#5d6875")
    return _save_figure(fig, output_dir, stem)


def _plot_relative_improvement(config: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    frame = config.sort_values(["horizon", "feature_set", "model_family"]).copy()
    labels = [_label(row) for _, row in frame.iterrows()]
    x = np.arange(len(frame))
    values = frame["relative_improvement_pct_from_mean_fold_mae"]
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [
        "#c94936" if _is_primary(row) else "#2d5b8a" if _is_champion(row) else "#7d8896"
        for _, row in frame.iterrows()
    ]
    ax.axhline(0, color="#1f2933", linewidth=1)
    ax.bar(x, values, color=colors)
    _format_nulls(ax, list(x), values, labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    _style_axis(ax, "Relative improvement vs persistence", "MAE improvement (%)")
    ax.text(
        0,
        1.02,
        "Positive means lower MAE than persistence. " + QUALIFICATION_LABEL,
        transform=ax.transAxes,
        fontsize=8,
        color="#5d6875",
    )
    return _save_figure(fig, output_dir, "relative-improvement-vs-persistence")


def _plot_fold_cases_only(fold: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    frame = fold[(fold["feature_set"] == "cases_only") & (fold["model_family"] == "ridge")].copy()
    frame = frame.sort_values(["horizon", "fold"])
    labels = [f"h{int(row.horizon)} {int(row.fold)}" for row in frame.itertuples()]
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(
        x - 0.18, frame["mae_model"], width=0.36, color="#c94936", label="cases-only Ridge"
    )
    axes[0].bar(
        x + 0.18, frame["mae_persistence"], width=0.36, color="#c7ced8", label="persistence"
    )
    axes[0].legend(frameon=False, ncols=2)
    _style_axis(axes[0], "Cases-only Ridge vs persistence by fold and horizon", "MAE")
    axes[1].axhline(0, color="#1f2933", linewidth=1)
    axes[1].bar(x, frame["relative_improvement_pct"], color="#c94936")
    _style_axis(axes[1], "Fold-level relative improvement", "MAE improvement (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    axes[0].text(
        0,
        1.04,
        QUALIFICATION_LABEL,
        transform=axes[0].transAxes,
        fontsize=8,
        color="#5d6875",
    )
    return _save_figure(fig, output_dir, "cases-only-ridge-fold-horizon")


def _plot_weather_effects(
    weather: pd.DataFrame, bootstrap: pd.DataFrame, output_dir: Path
) -> dict[str, Any]:
    frame = weather.merge(
        bootstrap[["horizon", "model_family", "comparison", "ci_low", "ci_high"]],
        on=["horizon", "model_family", "comparison"],
        how="left",
    ).sort_values(["horizon", "model_family", "comparison"])
    labels = [
        f"h{int(row.horizon)} {row.model_family} {row.comparison.replace('_', '-')}"
        for row in frame.itertuples()
    ]
    x = np.arange(len(frame))
    values = frame["mean_fold_mae_difference_added_minus_baseline"]
    lower = values - frame["ci_low"]
    upper = frame["ci_high"] - values
    yerr = np.vstack([lower.where(lower.notna(), 0), upper.where(upper.notna(), 0)])
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = np.where(values < 0, "#28745d", "#a64b3f")
    ax.axhline(0, color="#1f2933", linewidth=1)
    ax.bar(x, values, color=colors)
    ax.errorbar(x, values, yerr=yerr, fmt="none", ecolor="#26313f", capsize=2, linewidth=0.8)
    _format_nulls(ax, list(x), values, labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    _style_axis(ax, "Signed weather B-A and C-B effects", "Added minus baseline MAE")
    ax.text(
        0,
        1.02,
        "Negative means added weather set has lower MAE. " + QUALIFICATION_LABEL,
        transform=ax.transAxes,
        fontsize=8,
        color="#5d6875",
    )
    return _save_figure(fig, output_dir, "signed-weather-effects")


def _plot_district_errors(district: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    frame = district[
        (district["feature_set"] == "cases_only") & (district["model_family"] == "ridge")
    ]
    frame = frame.groupby("district_id", as_index=False).agg(
        observed_target_total=("observed_target_total", "sum"),
        model_mae=("model_mae", "mean"),
        persistence_mae=("persistence_mae", "mean"),
        normalized_model_mae_by_mean_observed=("normalized_model_mae_by_mean_observed", "mean"),
    )
    frame = frame.sort_values("observed_target_total", ascending=False).head(25)
    labels = frame["district_id"].tolist()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(x - 0.18, frame["model_mae"], width=0.36, color="#c94936", label="model MAE")
    axes[0].bar(
        x + 0.18,
        frame["persistence_mae"],
        width=0.36,
        color="#c7ced8",
        label="persistence MAE",
    )
    axes[0].legend(frameon=False, ncols=2)
    _style_axis(axes[0], "District descriptive errors, top observed volumes", "MAE")
    sizes = np.clip(frame["observed_target_total"].to_numpy(dtype=float) / 35, 20, 260)
    values = frame["normalized_model_mae_by_mean_observed"]
    axes[1].scatter(x, values, s=sizes, color="#2d5b8a", alpha=0.8)
    _format_nulls(axes[1], list(x), values, labels)
    _style_axis(axes[1], "Normalized model MAE with volume cue", "Model MAE / mean observed")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axes[0].text(
        0,
        1.04,
        "Descriptive only; higher-volume districts dominate visual attention. "
        + QUALIFICATION_LABEL,
        transform=axes[0].transAxes,
        fontsize=8,
        color="#5d6875",
    )
    return _save_figure(fig, output_dir, "district-descriptive-errors")


def _plot_high_incidence(thresholds: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    frame = thresholds[
        (thresholds["feature_set"] == "cases_only") & (thresholds["model_family"] == "ridge")
    ]
    frame = frame.sort_values(["horizon", "fold"])
    labels = [f"h{int(row.horizon)} {int(row.fold)}" for row in frame.itertuples()]
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.18, frame["q90_count"], width=0.36, color="#b4772b", label="q90 count")
    ax.bar(x + 0.18, frame["q95_count"], width=0.36, color="#7d3f8c", label="q95 count")
    _format_nulls(ax, list(x - 0.18), frame["q90_count"], labels)
    _format_nulls(ax, list(x + 0.18), frame["q95_count"], labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.legend(frameon=False, ncols=2)
    _style_axis(ax, "Training-threshold high-incidence membership", "Evaluation row count")
    ax.text(
        0,
        1.02,
        "q90/q95 thresholds are training-bound; zero counts are observed zeros, not nulls. "
        + QUALIFICATION_LABEL,
        transform=ax.transAxes,
        fontsize=8,
        color="#5d6875",
    )
    return _save_figure(fig, output_dir, "training-threshold-high-incidence")


def _plot_change_categories(change: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    frame = change[(change["feature_set"] == "cases_only") & (change["model_family"] == "ridge")]
    category_order = [
        "stable",
        "directional_up",
        "large_up",
        "directional_down",
        "large_down",
        "uncategorized",
    ]
    pivot = (
        frame.pivot_table(
            index=["horizon", "fold"],
            columns="category",
            values="count",
            aggfunc="sum",
            dropna=False,
        )
        .reindex(columns=category_order)
        .sort_index()
    )
    labels = [f"h{int(h)} {int(f)}" for h, f in pivot.index]
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(pivot))
    colors = {
        "stable": "#7d8896",
        "directional_up": "#d59a34",
        "large_up": "#a64b3f",
        "directional_down": "#4f83b8",
        "large_down": "#2d5b8a",
        "uncategorized": "#9b8f74",
    }
    for category in category_order:
        values = pivot[category]
        draw_values = values.fillna(0)
        ax.bar(x, draw_values, bottom=bottom, color=colors[category], label=category)
        _format_nulls(ax, list(x), values, labels)
        bottom += draw_values.to_numpy(dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    _style_axis(ax, "Postprediction change categories with counts", "Evaluation row count")
    ax.text(
        0,
        1.02,
        "Categories use saved threshold receipts; absent cells are labeled null. "
        + QUALIFICATION_LABEL,
        transform=ax.transAxes,
        fontsize=8,
        color="#5d6875",
    )
    return _save_figure(fig, output_dir, "postprediction-change-categories")


def _make_contact_sheet(output_dir: Path, plot_records: list[dict[str, Any]]) -> dict[str, Any]:
    pngs = [output_dir / record["png"] for record in plot_records]
    images = [plt.imread(path) for path in pngs]
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    axes_flat = axes.ravel()
    for ax, image, record in zip(axes_flat, images, plot_records, strict=False):
        ax.imshow(image)
        ax.set_title(record["stem"], fontsize=10, loc="left")
        ax.axis("off")
    for ax in axes_flat[len(images) :]:
        ax.axis("off")
    fig.suptitle("Milestone 3 development plots contact sheet", x=0.03, ha="left", fontsize=14)
    return _save_figure(fig, output_dir, "contact-sheet")


def _source_hashes(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _write_indexes(
    output_dir: Path,
    plot_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    captions = {
        "horizon-mae": "Horizon MAE from saved unweighted mean-of-fold metrics.",
        "relative-improvement-vs-persistence": (
            "Relative MAE improvement vs persistence; positive is better than persistence."
        ),
        "horizon-rmse": "Horizon RMSE from saved unweighted mean-of-fold metrics.",
        "cases-only-ridge-fold-horizon": "Cases-only Ridge versus persistence by fold and horizon.",
        "signed-weather-effects": (
            "Signed weather B-A/C-B effects; negative means added weather lower MAE."
        ),
        "district-descriptive-errors": "District descriptive errors with observed-volume caveat.",
        "training-threshold-high-incidence": "Training-threshold q90/q95 high-incidence counts.",
        "postprediction-change-categories": "Postprediction change categories with counts.",
        "contact-sheet": "Visual QA contact sheet.",
    }
    lines = [
        "# Milestone 3 Development Plots",
        "",
        QUALIFICATION_LABEL,
        "",
        "Feature importance is pending in a later slice.",
        "",
        f"Source analysis: `{manifest.get('analysis_output_dir', '')}`",
        "",
    ]
    for record in plot_records:
        lines.extend(
            [
                f"## {record['stem']}",
                "",
                captions.get(record["stem"], ""),
                "",
                f"![{record['stem']}]({record['png']})",
                "",
            ]
        )
    md_path = output_dir / "index.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    body = "\n".join(
        (
            f"<section><h2>{html.escape(record['stem'])}</h2>"
            f"<p>{html.escape(captions.get(record['stem'], ''))}</p>"
            f"<img src=\"{html.escape(record['png'])}\" "
            f"alt=\"{html.escape(record['stem'])}\"></section>"
        )
        for record in plot_records
    )
    html_path = output_dir / "index.html"
    html_path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Milestone 3 development plots</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1200px}"
        "img{max-width:100%;height:auto;border:1px solid #d7dee8}"
        "section{margin:0 0 32px}</style>"
        f"<h1>Milestone 3 Development Plots</h1><p>{html.escape(QUALIFICATION_LABEL)}</p>"
        "<p>Feature importance is pending in a later slice.</p>"
        + body,
        encoding="utf-8",
    )
    return md_path, html_path


def run_plots(
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    repo_root: Path = REPO_ROOT,
) -> PlotResult:
    manifest, verified_paths = verify_analysis_dir(analysis_dir)
    tables = _read_tables(verified_paths)
    _validate_tables(tables)
    output_dir = _prepare_output_dir(output_root, run_id, repo_root=repo_root)

    plot_records = [
        _plot_horizon_metric(
            tables["config_metrics"],
            output_dir,
            "mean_fold_mae_model",
            "mean_fold_mae_persistence",
            "horizon-mae",
            "MAE",
        ),
        _plot_relative_improvement(tables["config_metrics"], output_dir),
        _plot_horizon_metric(
            tables["config_metrics"],
            output_dir,
            "mean_fold_rmse_model",
            "mean_fold_rmse_persistence",
            "horizon-rmse",
            "RMSE",
        ),
        _plot_fold_cases_only(tables["fold_metrics"], output_dir),
        _plot_weather_effects(
            tables["weather_effects"], tables["bootstrap_weather_effects"], output_dir
        ),
        _plot_district_errors(tables["district_diagnostics"], output_dir),
        _plot_high_incidence(tables["high_incidence_thresholds"], output_dir),
        _plot_change_categories(tables["change_categories"], output_dir),
    ]
    contact = _make_contact_sheet(output_dir, plot_records)
    all_records = [*plot_records, contact]

    plotting_source = Path(__file__).resolve()
    provenance = {
        "status": "complete",
        "qualification": QUALIFICATION,
        "qualification_label": QUALIFICATION_LABEL,
        "analysis_manifest_sha256": sha256_file(Path(analysis_dir).resolve() / "manifest.json"),
        "source_tables": _source_hashes(verified_paths),
        "plotting_source": {
            "path": str(plotting_source),
            "sha256": sha256_file(plotting_source),
        },
        "plots": {
            record["stem"]: {
                "files": {
                    "png": record["png"],
                    "svg": record["svg"],
                    "png_sha256": record["png_sha256"],
                    "svg_sha256": record["svg_sha256"],
                },
                "dimensions_px": record["dimensions_px"],
                "source_tables": sorted(REQUIRED_TABLES),
                "qualification": QUALIFICATION,
            }
            for record in all_records
        },
        "pending_requirements": ["feature_importance"],
    }
    manifest_path = output_dir / "provenance-manifest.json"
    manifest_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path, html_path = _write_indexes(output_dir, all_records, manifest)
    return PlotResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        index_markdown_path=md_path,
        index_html_path=html_path,
        contact_sheet_path=output_dir / contact["png"],
        png_count=len(list(output_dir.glob("*.png"))),
        svg_count=len(list(output_dir.glob("*.svg"))),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Milestone 3 development plots.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_plots(args.analysis_dir, output_root=args.output_root, run_id=args.run_id)
    print(f"plots_dir={result.output_dir}")
    print(f"contact_sheet={result.contact_sheet_path}")
    print(f"png_count={result.png_count}")
    print(f"svg_count={result.svg_count}")
    return 0
