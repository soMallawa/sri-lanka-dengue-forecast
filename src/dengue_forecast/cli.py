from __future__ import annotations

# ruff: noqa: E501, I001

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

from dengue_forecast.contracts import ContractError
from dengue_forecast.pipeline import (
    PipelinePaths,
    build_dengue,
    build_features,
    build_geography,
    build_reports,
    build_weather,
    discover_dengue,
    download_dengue,
    download_weather,
    load_milestone1_config,
    rebuild_check,
    run_all,
    validate_all,
)

LOG = logging.getLogger("dengue_forecast.cli")


def _paths(args: argparse.Namespace) -> PipelinePaths:
    data_root = Path(args.data_root).resolve() if getattr(args, "data_root", None) else None
    return PipelinePaths(data_root=data_root) if data_root else PipelinePaths()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root containing raw/interim/processed/reports.",
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    config = load_milestone1_config()
    _add_common(parser)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--start-year", type=int, default=config.start_year)
    parser.add_argument("--end-year", type=int, default=config.end_year)
    parser.add_argument(
        "--weather-start-date", type=date.fromisoformat, default=config.weather_start_date
    )
    parser.add_argument(
        "--weather-end-date", type=date.fromisoformat, default=config.weather_end_date
    )


def _cmd_pipeline_run_all(args: argparse.Namespace) -> int:
    result = run_all(
        start_year=args.start_year,
        end_year=args.end_year,
        offline=args.offline,
        paths=_paths(args),
        force=args.force,
        weather_start_date=args.weather_start_date,
        weather_end_date=args.weather_end_date,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_validate_all(args: argparse.Namespace) -> int:
    result = validate_all(paths=_paths(args))
    print("Milestone 1 validation: PASS")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_report_build(args: argparse.Namespace) -> int:
    build_reports(paths=_paths(args))
    return 0


def _cmd_dengue_discover(args: argparse.Namespace) -> int:
    count = discover_dengue(
        source=args.source,
        start_year=args.start_year,
        end_year=args.end_year,
        offline=args.offline,
        paths=_paths(args),
        force=getattr(args, "force", False),
    )
    print(f"discovered={count}")
    return 0


def _cmd_dengue_download(args: argparse.Namespace) -> int:
    count = download_dengue(offline=args.offline, paths=_paths(args))
    print(f"downloaded={count}")
    return 0


def _cmd_dengue_parse(args: argparse.Namespace) -> int:
    df = build_dengue(
        start_year=args.start_year,
        end_year=args.end_year,
        offline=args.offline,
        paths=_paths(args),
        force=getattr(args, "force", False),
    )
    print(f"rows={len(df)}")
    return 0


def _cmd_dengue_build(args: argparse.Namespace) -> int:
    return _cmd_dengue_parse(args)


def _cmd_dengue_validate(args: argparse.Namespace) -> int:
    import pandas as pd

    from dengue_forecast.contracts import validate_dengue_weekly

    path = _paths(args).processed / "dengue_cases_weekly.parquet"
    validate_dengue_weekly(pd.read_parquet(path))
    print("dengue=valid")
    return 0


def _cmd_geography_build(args: argparse.Namespace) -> int:
    df = build_geography(offline=args.offline, paths=_paths(args), force=args.force)
    print(f"rows={len(df)}")
    return 0


def _cmd_weather_download(args: argparse.Namespace) -> int:
    result = download_weather(
        start_date=args.weather_start_date,
        end_date=args.weather_end_date,
        offline=args.offline,
        force=args.force,
        paths=_paths(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_weather_build(args: argparse.Namespace) -> int:
    df = build_weather(
        paths=_paths(args),
        start_date=args.weather_start_date,
        end_date=args.weather_end_date,
        offline=args.offline,
        force=args.force,
    )
    print(f"rows={len(df)}")
    return 0


def _cmd_features_build(args: argparse.Namespace) -> int:
    dataset, registry = build_features(paths=_paths(args))
    print(f"rows={len(dataset)} registry={len(registry)}")
    return 0


def _cmd_rebuild_check(args: argparse.Namespace) -> int:
    result = rebuild_check(
        source_data_root=Path(args.source_data_root),
        output_root=Path(args.output_root),
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _modeling_paths(args: argparse.Namespace):
    from dengue_forecast.modeling.pipeline import ModelingPaths

    return ModelingPaths.resolve(
        root=getattr(args, "root", None),
        dataset=getattr(args, "dataset", None),
        registry=getattr(args, "registry", None),
        config=getattr(args, "config", None),
        artifact_root=getattr(args, "artifact_root", None),
        reports_root=getattr(args, "reports_root", None),
    )


def _add_model_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=None, help="Project root; defaults to cwd repo.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Allow non-production synthetic datasets for unit/integration tests.",
    )


def _cmd_model_stage(args: argparse.Namespace) -> int:
    from dengue_forecast.modeling import pipeline as m2

    paths = _modeling_paths(args)
    production = not getattr(args, "synthetic", False)
    commands = {
        "readiness": lambda: m2.stage_readiness(paths),
        "splits": lambda: m2.stage_splits(paths),
        "baselines": lambda: m2.stage_baselines(paths, production=production),
        "train": lambda: m2.stage_train(paths, production=production),
        "ablation": lambda: m2.stage_ablation(paths),
        "tune": lambda: m2.stage_tune(paths, production=production),
        "analyze": lambda: m2.stage_analyze(paths),
        "explain": lambda: m2.stage_explain(paths),
        "select": lambda: m2.stage_select(paths),
        "test": lambda: m2.stage_test(paths, production=production),
    }
    result = commands[args.model_command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_milestone2_run(args: argparse.Namespace) -> int:
    from dengue_forecast.modeling.pipeline import run_milestone2

    result = run_milestone2(_modeling_paths(args), production=not args.synthetic)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_milestone2_validate(args: argparse.Namespace) -> int:
    from dengue_forecast.modeling.pipeline import validate_milestone2

    result = validate_milestone2(_modeling_paths(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    print("MILESTONE 2 VALIDATION: PASS")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dengue_forecast.cli")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command")

    milestone1 = sub.add_parser("milestone1")
    milestone1_sub = milestone1.add_subparsers(dest="milestone1_command")
    milestone1_run = milestone1_sub.add_parser("run")
    _add_run_options(milestone1_run)
    milestone1_run.set_defaults(func=_cmd_pipeline_run_all)

    pipeline = sub.add_parser("pipeline")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command")
    run_all_parser = pipeline_sub.add_parser("run-all")
    _add_run_options(run_all_parser)
    run_all_parser.set_defaults(func=_cmd_pipeline_run_all)

    validate = sub.add_parser("validate")
    validate_sub = validate.add_subparsers(dest="validate_command")
    validate_all_parser = validate_sub.add_parser("all")
    _add_common(validate_all_parser)
    validate_all_parser.set_defaults(func=_cmd_validate_all)

    report = sub.add_parser("report")
    report_sub = report.add_subparsers(dest="report_command")
    report_build = report_sub.add_parser("build")
    _add_common(report_build)
    report_build.set_defaults(func=_cmd_report_build)
    report_milestone1 = report_sub.add_parser("milestone1")
    _add_common(report_milestone1)
    report_milestone1.set_defaults(func=_cmd_report_build)

    dengue = sub.add_parser("dengue")
    dengue_sub = dengue.add_subparsers(dest="dengue_command")
    dengue_discover = dengue_sub.add_parser("discover")
    _add_common(dengue_discover)
    dengue_discover.add_argument("--source", choices=["epid", "ndcu"], default="epid")
    dengue_discover.add_argument("--start-year", type=int, required=True)
    dengue_discover.add_argument("--end-year", type=int, default=None)
    dengue_discover.add_argument("--offline", action="store_true")
    dengue_discover.add_argument("--force", action="store_true")
    dengue_discover.set_defaults(func=_cmd_dengue_discover)

    dengue_download = dengue_sub.add_parser("download")
    _add_common(dengue_download)
    dengue_download.add_argument("--offline", action="store_true")
    dengue_download.set_defaults(func=_cmd_dengue_download)

    dengue_parse = dengue_sub.add_parser("parse")
    _add_common(dengue_parse)
    dengue_parse.add_argument("--offline", action="store_true")
    dengue_parse.add_argument("--start-year", type=int, default=2024)
    dengue_parse.add_argument("--end-year", type=int, default=2024)
    dengue_parse.set_defaults(func=_cmd_dengue_parse)

    dengue_build = dengue_sub.add_parser("build")
    _add_common(dengue_build)
    dengue_build.add_argument("--offline", action="store_true")
    dengue_build.add_argument("--force", action="store_true")
    dengue_build.add_argument("--verbose", action="store_true")
    config = load_milestone1_config()
    dengue_build.add_argument("--start-year", type=int, default=config.start_year)
    dengue_build.add_argument("--end-year", type=int, default=config.end_year)
    dengue_build.set_defaults(func=_cmd_dengue_build)

    dengue_validate = dengue_sub.add_parser("validate")
    _add_common(dengue_validate)
    dengue_validate.set_defaults(func=_cmd_dengue_validate)

    geography = sub.add_parser("geography")
    geography_sub = geography.add_subparsers(dest="geography_command")
    geography_build = geography_sub.add_parser("build")
    _add_common(geography_build)
    geography_build.add_argument("--offline", action="store_true")
    geography_build.add_argument("--force", action="store_true")
    geography_build.set_defaults(func=_cmd_geography_build)

    weather = sub.add_parser("weather")
    weather_sub = weather.add_subparsers(dest="weather_command")
    weather_download = weather_sub.add_parser("download")
    _add_run_options(weather_download)
    weather_download.set_defaults(func=_cmd_weather_download)
    weather_build = weather_sub.add_parser("build")
    _add_run_options(weather_build)
    weather_build.set_defaults(func=_cmd_weather_build)

    features = sub.add_parser("features")
    features_sub = features.add_subparsers(dest="features_command")
    features_build = features_sub.add_parser("build")
    _add_common(features_build)
    features_build.set_defaults(func=_cmd_features_build)

    rebuild = sub.add_parser("rebuild-check")
    rebuild.add_argument("--source-data-root", type=Path, default=Path("data"))
    rebuild.add_argument("--output-root", type=Path, default=Path("/tmp/dengue-forecast-rebuild"))
    rebuild.add_argument("--force", action="store_true")
    rebuild.set_defaults(func=_cmd_rebuild_check)

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="model_command")
    for name in [
        "readiness",
        "splits",
        "baselines",
        "train",
        "ablation",
        "tune",
        "analyze",
        "explain",
        "select",
        "test",
    ]:
        stage = model_sub.add_parser(name)
        _add_model_common(stage)
        stage.set_defaults(func=_cmd_model_stage)

    milestone2 = sub.add_parser("milestone2")
    milestone2_sub = milestone2.add_subparsers(dest="milestone2_command")
    milestone2_run = milestone2_sub.add_parser("run")
    _add_model_common(milestone2_run)
    milestone2_run.set_defaults(func=_cmd_milestone2_run)
    milestone2_validate = milestone2_sub.add_parser("validate")
    _add_model_common(milestone2_validate)
    milestone2_validate.set_defaults(func=_cmd_milestone2_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    func: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    try:
        return func(args)
    except (ContractError, FileNotFoundError, RuntimeError, ValueError, Exception) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
