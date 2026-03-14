from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from model_readiness import (
    DEFAULT_CANDIDATE_MODEL_KEY,
    DEFAULT_READINESS_MODEL_KEY,
    MODEL_CANDIDATE_COMPARE_SUITE_TABLE,
    MODEL_CANDIDATE_COMPARE_TABLE,
    MODEL_CANDIDATE_COMPARE_WINDOW_TABLE,
    build_fact_model_candidate_compare_suite,
    build_fact_model_candidate_compare_window,
    materialize_model_readiness_review,
)
from opportunity_backtest import (
    BACKTEST_PREDICTION_TABLE,
    BACKTEST_SUMMARY_SLICE_TABLE,
    BACKTEST_TOP_ERROR_TABLE,
    DRIFT_WINDOW_TABLE,
    MODEL_POTENTIAL_RATIO_V2,
    load_curtailment_opportunity_input,
    materialize_opportunity_backtest,
)


BENCHMARK_WINDOW_TABLE = "dim_model_benchmark_window"
MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE = "fact_model_candidate_compare_window_daily"

_REQUIRED_MANIFEST_COLUMNS = [
    "benchmark_suite_name",
    "benchmark_window_key",
    "benchmark_window_label",
    "opportunity_input_path",
    "readiness_start",
    "readiness_end",
]


@dataclass(frozen=True)
class BenchmarkWindowSpec:
    benchmark_suite_name: str
    benchmark_window_key: str
    benchmark_window_label: str
    opportunity_input_path: str
    readiness_start: dt.date
    readiness_end: dt.date
    benchmark_window_family: str = "acceptance"
    benchmark_role: str = "acceptance"
    promotion_window_flag: bool = True
    display_order: int = 0
    window_notes: str = ""


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value '{value}'")


def _parse_market_day(value: object, *, column_name: str) -> dt.date:
    text = str(value).strip()
    if not text:
        raise ValueError(f"benchmark suite manifest column '{column_name}' cannot be blank")
    try:
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid {column_name} '{text}', expected YYYY-MM-DD") from exc


def _empty_benchmark_window_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "benchmark_suite_name",
            "benchmark_window_key",
            "benchmark_window_label",
            "opportunity_input_path",
            "readiness_start",
            "readiness_end",
            "benchmark_window_family",
            "benchmark_role",
            "promotion_window_flag",
            "display_order",
            "window_notes",
            "source_lineage",
        ]
    )


def _empty_model_candidate_compare_window_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "benchmark_suite_name",
            "benchmark_window_key",
            "benchmark_window_label",
            "benchmark_window_start_date",
            "benchmark_window_end_date",
            "benchmark_window_family",
            "benchmark_role",
            "promotion_window_flag",
            "display_order",
            "window_date",
            "baseline_model_key",
            "candidate_model_key",
            "overall_t_plus_1h_deliverable_mae_delta_mwh",
            "gb_nl_t_plus_1h_deliverable_mae_delta_mwh",
            "gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh",
            "blocker_row_delta",
            "severe_focus_area_delta",
            "candidate_scope_row_count",
            "promotion_state",
            "source_lineage",
        ]
    )


def load_benchmark_suite_manifest(
    path: str | Path,
    *,
    suite_name: str | None = None,
) -> list[BenchmarkWindowSpec]:
    manifest_path = Path(path)
    manifest = pd.read_csv(manifest_path)
    missing_columns = [column for column in _REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"benchmark suite manifest missing required columns: {missing}")

    if suite_name:
        manifest = manifest[manifest["benchmark_suite_name"].astype(str).eq(suite_name)].copy()
    else:
        suite_names = sorted({str(value).strip() for value in manifest["benchmark_suite_name"].dropna() if str(value).strip()})
        if len(suite_names) > 1:
            raise ValueError(
                f"benchmark suite manifest contains multiple suites ({', '.join(suite_names)}); "
                "pass suite_name to select one"
            )
    if manifest.empty:
        raise ValueError("benchmark suite manifest did not yield any rows")

    if "display_order" not in manifest.columns:
        manifest["display_order"] = range(1, len(manifest) + 1)

    specs: list[BenchmarkWindowSpec] = []
    for _, row in manifest.iterrows():
        role = str(row.get("benchmark_role", "")).strip() or "acceptance"
        family = str(row.get("benchmark_window_family", "")).strip() or role
        specs.append(
            BenchmarkWindowSpec(
                benchmark_suite_name=str(row["benchmark_suite_name"]).strip(),
                benchmark_window_key=str(row["benchmark_window_key"]).strip(),
                benchmark_window_label=str(row["benchmark_window_label"]).strip(),
                opportunity_input_path=str(row["opportunity_input_path"]).strip(),
                readiness_start=_parse_market_day(row["readiness_start"], column_name="readiness_start"),
                readiness_end=_parse_market_day(row["readiness_end"], column_name="readiness_end"),
                benchmark_window_family=family,
                benchmark_role=role,
                promotion_window_flag=_coerce_bool(
                    row.get("promotion_window_flag"),
                    default=role == "acceptance",
                ),
                display_order=int(row.get("display_order") or 0),
                window_notes=str(row.get("window_notes", "")).strip(),
            )
        )

    specs.sort(key=lambda spec: (spec.display_order, spec.benchmark_window_key))
    return specs


def build_dim_model_benchmark_window(
    benchmark_windows: Iterable[BenchmarkWindowSpec],
    *,
    manifest_source: str,
) -> pd.DataFrame:
    rows = [
        {
            "benchmark_suite_name": spec.benchmark_suite_name,
            "benchmark_window_key": spec.benchmark_window_key,
            "benchmark_window_label": spec.benchmark_window_label,
            "opportunity_input_path": spec.opportunity_input_path,
            "readiness_start": spec.readiness_start.isoformat(),
            "readiness_end": spec.readiness_end.isoformat(),
            "benchmark_window_family": spec.benchmark_window_family,
            "benchmark_role": spec.benchmark_role,
            "promotion_window_flag": spec.promotion_window_flag,
            "display_order": spec.display_order,
            "window_notes": spec.window_notes,
            "source_lineage": manifest_source,
        }
        for spec in benchmark_windows
    ]
    return pd.DataFrame(rows, columns=_empty_benchmark_window_frame().columns)


def _filter_opportunity_input_to_window(
    opportunity_input: pd.DataFrame,
    *,
    readiness_start: dt.date,
    readiness_end: dt.date,
) -> pd.DataFrame:
    frame = opportunity_input.copy()
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    window_start = pd.Timestamp(readiness_start, tz="Europe/London").tz_convert("UTC")
    window_end = pd.Timestamp(readiness_end + dt.timedelta(days=1), tz="Europe/London").tz_convert("UTC")
    return frame[frame["interval_start_utc"].ge(window_start) & frame["interval_start_utc"].lt(window_end)].copy()


def _annotate_window_daily_compare(
    compare_daily: pd.DataFrame,
    *,
    spec: BenchmarkWindowSpec,
) -> pd.DataFrame:
    if compare_daily is None or compare_daily.empty:
        return _empty_model_candidate_compare_window_daily_frame()
    annotated = compare_daily.copy()
    annotated.insert(0, "display_order", int(spec.display_order))
    annotated.insert(0, "promotion_window_flag", bool(spec.promotion_window_flag))
    annotated.insert(0, "benchmark_role", spec.benchmark_role)
    annotated.insert(0, "benchmark_window_family", spec.benchmark_window_family)
    annotated.insert(0, "benchmark_window_end_date", spec.readiness_end.isoformat())
    annotated.insert(0, "benchmark_window_start_date", spec.readiness_start.isoformat())
    annotated.insert(0, "benchmark_window_label", spec.benchmark_window_label)
    annotated.insert(0, "benchmark_window_key", spec.benchmark_window_key)
    annotated.insert(0, "benchmark_suite_name", spec.benchmark_suite_name)
    return annotated[_empty_model_candidate_compare_window_daily_frame().columns]


def materialize_model_benchmark_suite(
    output_dir: str | Path,
    *,
    benchmark_windows: list[BenchmarkWindowSpec],
    model_key: str,
    forecast_horizons: Iterable[int],
    baseline_model_key: str = DEFAULT_READINESS_MODEL_KEY,
    candidate_model_key: str = DEFAULT_CANDIDATE_MODEL_KEY,
    manifest_source: str = "benchmark_suite_manifest",
) -> Dict[str, pd.DataFrame]:
    if not benchmark_windows:
        raise ValueError("benchmark suite requires at least one benchmark window")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    windows_path = output_path / "windows"
    windows_path.mkdir(parents=True, exist_ok=True)

    benchmark_window_frame = build_dim_model_benchmark_window(
        benchmark_windows,
        manifest_source=manifest_source,
    )
    window_daily_frames: list[pd.DataFrame] = []
    window_compare_frames: list[pd.DataFrame] = []

    for spec in benchmark_windows:
        opportunity_input = load_curtailment_opportunity_input(spec.opportunity_input_path)
        opportunity_input = _filter_opportunity_input_to_window(
            opportunity_input,
            readiness_start=spec.readiness_start,
            readiness_end=spec.readiness_end,
        )
        window_output_dir = windows_path / spec.benchmark_window_key
        backtest_frames = materialize_opportunity_backtest(
            output_dir=window_output_dir,
            fact_curtailment_opportunity_hourly=opportunity_input,
            model_key=model_key,
            forecast_horizons=forecast_horizons,
        )
        selected_model_keys = set(backtest_frames[BACKTEST_PREDICTION_TABLE]["model_key"].dropna())
        readiness_model_key = MODEL_POTENTIAL_RATIO_V2 if MODEL_POTENTIAL_RATIO_V2 in selected_model_keys else model_key
        readiness_frames = materialize_model_readiness_review(
            output_dir=window_output_dir,
            fact_backtest_prediction_hourly=backtest_frames[BACKTEST_PREDICTION_TABLE],
            fact_backtest_summary_slice=backtest_frames[BACKTEST_SUMMARY_SLICE_TABLE],
            fact_backtest_top_error_hourly=backtest_frames[BACKTEST_TOP_ERROR_TABLE],
            fact_drift_window=backtest_frames[DRIFT_WINDOW_TABLE],
            model_key=readiness_model_key,
            baseline_model_key=baseline_model_key,
            candidate_model_key=candidate_model_key,
        )

        window_daily_frames.append(
            _annotate_window_daily_compare(
                readiness_frames.get(MODEL_CANDIDATE_COMPARE_TABLE, pd.DataFrame()),
                spec=spec,
            )
        )
        window_compare_frames.append(
            build_fact_model_candidate_compare_window(
                fact_backtest_prediction_hourly=backtest_frames[BACKTEST_PREDICTION_TABLE],
                fact_backtest_summary_slice=backtest_frames[BACKTEST_SUMMARY_SLICE_TABLE],
                fact_backtest_top_error_hourly=backtest_frames[BACKTEST_TOP_ERROR_TABLE],
                fact_drift_window=backtest_frames[DRIFT_WINDOW_TABLE],
                benchmark_suite_name=spec.benchmark_suite_name,
                benchmark_window_key=spec.benchmark_window_key,
                benchmark_window_label=spec.benchmark_window_label,
                benchmark_window_start_date=spec.readiness_start.isoformat(),
                benchmark_window_end_date=spec.readiness_end.isoformat(),
                benchmark_window_family=spec.benchmark_window_family,
                benchmark_role=spec.benchmark_role,
                promotion_window_flag=spec.promotion_window_flag,
                display_order=spec.display_order,
                baseline_model_key=baseline_model_key,
                candidate_model_key=candidate_model_key,
            )
        )

    window_daily_compare = (
        pd.concat(window_daily_frames, ignore_index=True)
        if any(not frame.empty for frame in window_daily_frames)
        else _empty_model_candidate_compare_window_daily_frame()
    )
    window_compare = pd.concat(window_compare_frames, ignore_index=True)
    window_compare = window_compare.sort_values(["display_order", "benchmark_window_key"]).reset_index(drop=True)
    suite_compare = build_fact_model_candidate_compare_suite(window_compare)

    benchmark_window_frame.to_csv(output_path / f"{BENCHMARK_WINDOW_TABLE}.csv", index=False)
    window_daily_compare.to_csv(output_path / f"{MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE}.csv", index=False)
    window_compare.to_csv(output_path / f"{MODEL_CANDIDATE_COMPARE_WINDOW_TABLE}.csv", index=False)
    suite_compare.to_csv(output_path / f"{MODEL_CANDIDATE_COMPARE_SUITE_TABLE}.csv", index=False)

    return {
        BENCHMARK_WINDOW_TABLE: benchmark_window_frame,
        MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE: window_daily_compare,
        MODEL_CANDIDATE_COMPARE_WINDOW_TABLE: window_compare,
        MODEL_CANDIDATE_COMPARE_SUITE_TABLE: suite_compare,
    }
