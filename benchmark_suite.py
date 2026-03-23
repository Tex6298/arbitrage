from __future__ import annotations

import datetime as dt
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from model_readiness import (
    DEFAULT_CANDIDATE_MODEL_KEY,
    DEFAULT_READINESS_MODEL_KEY,
    INFORMATIVE_WINDOW_SIGNAL_EPSILON_MWH,
    MODEL_BLOCKER_PRIORITY_TABLE,
    MODEL_CANDIDATE_COMPARE_SUITE_TABLE,
    MODEL_CANDIDATE_COMPARE_TABLE,
    MODEL_CANDIDATE_COMPARE_WINDOW_TABLE,
    MODEL_READINESS_TABLE,
    READINESS_HORIZON_HOURS,
    build_fact_model_candidate_compare_suite,
    build_fact_model_candidate_compare_window,
    materialize_model_readiness_review,
)
from opportunity_backtest import (
    BACKTEST_PREDICTION_TABLE,
    BACKTEST_SUMMARY_SLICE_TABLE,
    BACKTEST_TOP_ERROR_TABLE,
    CURTAILMENT_OPPORTUNITY_TABLE,
    DRIFT_WINDOW_TABLE,
    MODEL_POTENTIAL_RATIO_V2,
    SPECIALIST_SCOPE_HUB_KEY,
    SPECIALIST_SCOPE_INTERNAL_TIER,
    SPECIALIST_SCOPE_ROUTE_NAME,
    load_curtailment_opportunity_input,
    materialize_opportunity_backtest,
)


BENCHMARK_WINDOW_TABLE = "dim_model_benchmark_window"
MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE = "fact_model_candidate_compare_window_daily"
MODEL_BENCHMARK_WINDOW_SCOUT_TABLE = "fact_model_benchmark_window_scout"
REVIEWED_BUNDLE_BATCH_WINDOW_TABLE = "dim_reviewed_bundle_batch_window"
REVIEWED_BUNDLE_BATCH_SCOUT_TABLE = "fact_reviewed_bundle_batch_window_scout"
REVIEWED_BUNDLE_BATCH_READINESS_DAILY_TABLE = "fact_reviewed_bundle_batch_readiness_daily"
REVIEWED_BUNDLE_BATCH_WINDOW_SUMMARY_TABLE = "fact_reviewed_bundle_batch_window_summary"
REVIEWED_BUNDLE_BATCH_BLOCKER_SUMMARY_TABLE = "fact_reviewed_bundle_batch_blocker_summary"
DEFAULT_REVIEWED_BUNDLE_BATCH_GLOB = "curtailment_opportunity_live_britned_reviewed_*"
REVIEWED_BUNDLE_BATCH_NAME = "reviewed_bundle_batch"

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


def _empty_model_benchmark_window_scout_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "benchmark_window_key",
            "benchmark_window_label",
            "benchmark_window_start_date",
            "benchmark_window_end_date",
            "opportunity_input_path",
            "baseline_model_key",
            "scope_route_name",
            "scope_hub_key",
            "scope_internal_transfer_evidence_tier",
            "window_day_count",
            "specialist_scope_row_count",
            "specialist_scope_actual_opportunity_deliverable_mwh_sum",
            "specialist_scope_nonzero_actual_row_count",
            "specialist_scope_positive_route_price_row_count",
            "specialist_scope_positive_deliverable_proxy_row_count",
            "baseline_specialist_t_plus_1h_deliverable_abs_error_mwh_sum",
            "informative_window_flag",
            "informative_signal_basis",
            "source_lineage",
        ]
    )


def _empty_reviewed_bundle_batch_window_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "benchmark_window_key",
            "benchmark_window_label",
            "opportunity_input_path",
            "readiness_start",
            "readiness_end",
            "display_order",
            "window_notes",
            "source_lineage",
        ]
    )


def _empty_reviewed_bundle_batch_scout_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "display_order",
            *_empty_model_benchmark_window_scout_frame().columns.tolist(),
        ]
    )


def _empty_reviewed_bundle_batch_readiness_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "display_order",
            "benchmark_window_key",
            "benchmark_window_label",
            "benchmark_window_start_date",
            "benchmark_window_end_date",
            "opportunity_input_path",
            "window_date",
            "model_key",
            "overall_t_plus_1h_deliverable_mae_mwh",
            "overall_t_plus_6h_deliverable_mae_mwh",
            "gb_nl_t_plus_1h_deliverable_mae_mwh",
            "proxy_internal_transfer_share_t_plus_1h",
            "reviewed_internal_transfer_share_t_plus_1h",
            "capacity_unknown_route_share_t_plus_1h",
            "route_warn_count_t_plus_1h",
            "cluster_warn_count_t_plus_1h",
            "severe_unresolved_focus_area_count_t_plus_1h",
            "model_ready_flag",
            "model_readiness_state",
            "blocking_reasons",
            "source_lineage",
        ]
    )


def _empty_reviewed_bundle_batch_window_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "display_order",
            "benchmark_window_key",
            "benchmark_window_label",
            "benchmark_window_start_date",
            "benchmark_window_end_date",
            "opportunity_input_path",
            "model_key",
            "window_day_count",
            "ready_day_count",
            "not_ready_day_count",
            "ready_day_share",
            "blocking_day_count",
            "mean_overall_t_plus_1h_deliverable_mae_mwh",
            "max_overall_t_plus_1h_deliverable_mae_mwh",
            "mean_overall_t_plus_6h_deliverable_mae_mwh",
            "max_overall_t_plus_6h_deliverable_mae_mwh",
            "mean_gb_nl_t_plus_1h_deliverable_mae_mwh",
            "max_gb_nl_t_plus_1h_deliverable_mae_mwh",
            "mean_proxy_internal_transfer_share_t_plus_1h",
            "max_proxy_internal_transfer_share_t_plus_1h",
            "route_warn_day_count",
            "cluster_warn_day_count",
            "severe_focus_day_count",
            "informative_window_flag",
            "informative_signal_basis",
            "specialist_scope_row_count",
            "specialist_scope_actual_opportunity_deliverable_mwh_sum",
            "baseline_specialist_t_plus_1h_deliverable_abs_error_mwh_sum",
            "source_lineage",
        ]
    )


def _empty_reviewed_bundle_batch_blocker_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "display_order",
            "benchmark_window_key",
            "benchmark_window_label",
            "benchmark_window_start_date",
            "benchmark_window_end_date",
            "opportunity_input_path",
            "model_key",
            "blocker_type",
            "blocker_scope",
            "blocker_day_count",
            "blocker_row_count",
            "top_blocker_slice_key",
            "top_slice_dimension",
            "top_slice_value",
            "top_route_name",
            "top_cluster_key",
            "mean_blocker_priority_score",
            "max_blocker_priority_score",
            "eligible_row_count_sum",
            "actual_volume_mwh_sum",
            "mean_deliverable_abs_error_mwh_mean",
            "max_deliverable_abs_error_mwh_max",
            "recommended_next_step",
            "source_lineage",
        ]
    )


def _parse_market_day_range_from_bundle_name(bundle_name: str) -> tuple[dt.date, dt.date] | None:
    matches = re.findall(r"\d{4}-\d{2}-\d{2}", bundle_name)
    if len(matches) < 2:
        return None
    try:
        return dt.date.fromisoformat(matches[-2]), dt.date.fromisoformat(matches[-1])
    except ValueError:
        return None


def _infer_readiness_window_from_opportunity_input(opportunity_input: pd.DataFrame) -> tuple[dt.date, dt.date]:
    timestamps = pd.to_datetime(
        opportunity_input.get("interval_start_utc", pd.Series(dtype=object)),
        utc=True,
        errors="coerce",
    ).dropna()
    if timestamps.empty:
        raise ValueError("opportunity input is missing valid interval_start_utc values")
    market_dates = timestamps.dt.tz_convert("Europe/London").dt.date
    return min(market_dates), max(market_dates)


def _reviewed_bundle_alias_priority(bundle_path: Path) -> tuple[int, int, str]:
    name = bundle_path.name.lower()
    return (
        0 if "rerun" in name else 1 if "refresh" not in name else 2,
        len(bundle_path.name),
        bundle_path.name,
    )


def _reviewed_bundle_has_opportunity_output(bundle_path: Path) -> bool:
    if bundle_path.is_dir():
        return (bundle_path / f"{CURTAILMENT_OPPORTUNITY_TABLE}.csv").exists()
    return bundle_path.exists()


def discover_reviewed_bundle_batch_windows(
    root_dir: str | Path,
    *,
    bundle_glob: str = DEFAULT_REVIEWED_BUNDLE_BATCH_GLOB,
) -> list[BenchmarkWindowSpec]:
    root_path = Path(root_dir)
    bundle_paths = sorted(
        path
        for path in root_path.glob(bundle_glob)
        if path.is_dir() and _reviewed_bundle_has_opportunity_output(path)
    )
    if not bundle_paths:
        raise ValueError(
            f"no complete reviewed opportunity bundles found under {root_path} matching '{bundle_glob}'"
        )

    canonical_paths_by_range: dict[tuple[dt.date, dt.date] | tuple[str, str], Path] = {}
    for bundle_path in bundle_paths:
        parsed_range = _parse_market_day_range_from_bundle_name(bundle_path.name)
        dedupe_key: tuple[dt.date, dt.date] | tuple[str, str]
        if parsed_range is not None:
            dedupe_key = parsed_range
        else:
            dedupe_key = ("name", bundle_path.name)
        incumbent = canonical_paths_by_range.get(dedupe_key)
        if incumbent is None or _reviewed_bundle_alias_priority(bundle_path) < _reviewed_bundle_alias_priority(incumbent):
            canonical_paths_by_range[dedupe_key] = bundle_path

    def _sort_key(path: Path) -> tuple[dt.date, str]:
        parsed_range = _parse_market_day_range_from_bundle_name(path.name)
        if parsed_range is not None:
            return parsed_range[0], path.name
        return dt.date.max, path.name

    canonical_bundle_paths = sorted(canonical_paths_by_range.values(), key=_sort_key)

    specs: list[BenchmarkWindowSpec] = []
    for display_order, bundle_path in enumerate(canonical_bundle_paths, start=1):
        parsed_range = _parse_market_day_range_from_bundle_name(bundle_path.name)
        if parsed_range is not None:
            readiness_start, readiness_end = parsed_range
        else:
            opportunity_input = load_curtailment_opportunity_input(bundle_path)
            readiness_start, readiness_end = _infer_readiness_window_from_opportunity_input(opportunity_input)
        specs.append(
            BenchmarkWindowSpec(
                benchmark_suite_name=REVIEWED_BUNDLE_BATCH_NAME,
                benchmark_window_key=bundle_path.name,
                benchmark_window_label=bundle_path.name,
                opportunity_input_path=str(bundle_path),
                readiness_start=readiness_start,
                readiness_end=readiness_end,
                benchmark_window_family="reviewed_bundle",
                benchmark_role="reviewed_bundle",
                promotion_window_flag=False,
                display_order=display_order,
                window_notes="autodiscovered reviewed opportunity bundle",
            )
        )
    return specs


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


def build_dim_reviewed_bundle_batch_window(
    benchmark_windows: Iterable[BenchmarkWindowSpec],
    *,
    discovery_source: str,
) -> pd.DataFrame:
    rows = [
        {
            "benchmark_window_key": spec.benchmark_window_key,
            "benchmark_window_label": spec.benchmark_window_label,
            "opportunity_input_path": spec.opportunity_input_path,
            "readiness_start": spec.readiness_start.isoformat(),
            "readiness_end": spec.readiness_end.isoformat(),
            "display_order": spec.display_order,
            "window_notes": spec.window_notes,
            "source_lineage": discovery_source,
        }
        for spec in benchmark_windows
    ]
    return pd.DataFrame(rows, columns=_empty_reviewed_bundle_batch_window_frame().columns)


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


def _annotate_reviewed_bundle_scout(
    scout: pd.DataFrame,
    *,
    spec: BenchmarkWindowSpec,
) -> pd.DataFrame:
    if scout is None or scout.empty:
        return _empty_reviewed_bundle_batch_scout_frame()
    annotated = scout.copy()
    annotated.insert(0, "display_order", int(spec.display_order))
    return annotated[_empty_reviewed_bundle_batch_scout_frame().columns]


def _annotate_reviewed_bundle_readiness_daily(
    readiness: pd.DataFrame,
    *,
    spec: BenchmarkWindowSpec,
) -> pd.DataFrame:
    if readiness is None or readiness.empty:
        return _empty_reviewed_bundle_batch_readiness_daily_frame()
    annotated = readiness.copy()
    annotated.insert(0, "opportunity_input_path", spec.opportunity_input_path)
    annotated.insert(0, "benchmark_window_end_date", spec.readiness_end.isoformat())
    annotated.insert(0, "benchmark_window_start_date", spec.readiness_start.isoformat())
    annotated.insert(0, "benchmark_window_label", spec.benchmark_window_label)
    annotated.insert(0, "benchmark_window_key", spec.benchmark_window_key)
    annotated.insert(0, "display_order", int(spec.display_order))
    return annotated[_empty_reviewed_bundle_batch_readiness_daily_frame().columns]


def _annotate_reviewed_bundle_blocker_rows(
    blocker: pd.DataFrame,
    *,
    spec: BenchmarkWindowSpec,
) -> pd.DataFrame:
    if blocker is None or blocker.empty:
        return blocker.copy() if blocker is not None else pd.DataFrame()
    annotated = blocker.copy()
    annotated.insert(0, "opportunity_input_path", spec.opportunity_input_path)
    annotated.insert(0, "benchmark_window_end_date", spec.readiness_end.isoformat())
    annotated.insert(0, "benchmark_window_start_date", spec.readiness_start.isoformat())
    annotated.insert(0, "benchmark_window_label", spec.benchmark_window_label)
    annotated.insert(0, "benchmark_window_key", spec.benchmark_window_key)
    annotated.insert(0, "display_order", int(spec.display_order))
    return annotated


def _build_reviewed_bundle_window_summary(
    readiness: pd.DataFrame,
    scout: pd.DataFrame,
    *,
    spec: BenchmarkWindowSpec,
) -> pd.DataFrame:
    readiness = readiness.copy() if readiness is not None else pd.DataFrame()
    scout_row = scout.iloc[0] if scout is not None and not scout.empty else pd.Series(dtype=object)
    overall_t1 = pd.to_numeric(
        readiness.get("overall_t_plus_1h_deliverable_mae_mwh", pd.Series(dtype=float, index=readiness.index)),
        errors="coerce",
    )
    overall_t6 = pd.to_numeric(
        readiness.get("overall_t_plus_6h_deliverable_mae_mwh", pd.Series(dtype=float, index=readiness.index)),
        errors="coerce",
    )
    gb_nl_t1 = pd.to_numeric(
        readiness.get("gb_nl_t_plus_1h_deliverable_mae_mwh", pd.Series(dtype=float, index=readiness.index)),
        errors="coerce",
    )
    proxy_share = pd.to_numeric(
        readiness.get(
            "proxy_internal_transfer_share_t_plus_1h",
            pd.Series(dtype=float, index=readiness.index),
        ),
        errors="coerce",
    )
    window_day_count = int(len(readiness))
    ready_day_count = int(readiness.get("model_ready_flag", pd.Series(False, index=readiness.index)).fillna(False).sum())
    not_ready_day_count = int(
        readiness.get("model_readiness_state", pd.Series("", index=readiness.index)).eq("not_ready").sum()
    )
    blocking_day_count = int(
        readiness.get("blocking_reasons", pd.Series("", index=readiness.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    route_warn_day_count = int(
        pd.to_numeric(
            readiness.get("route_warn_count_t_plus_1h", pd.Series(0.0, index=readiness.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
        .sum()
    )
    cluster_warn_day_count = int(
        pd.to_numeric(
            readiness.get("cluster_warn_count_t_plus_1h", pd.Series(0.0, index=readiness.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
        .sum()
    )
    severe_focus_day_count = int(
        pd.to_numeric(
            readiness.get("severe_unresolved_focus_area_count_t_plus_1h", pd.Series(0.0, index=readiness.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
        .sum()
    )
    model_key = str(readiness["model_key"].iloc[0]) if not readiness.empty and "model_key" in readiness.columns else DEFAULT_READINESS_MODEL_KEY
    row = {
        "display_order": int(spec.display_order),
        "benchmark_window_key": spec.benchmark_window_key,
        "benchmark_window_label": spec.benchmark_window_label,
        "benchmark_window_start_date": spec.readiness_start.isoformat(),
        "benchmark_window_end_date": spec.readiness_end.isoformat(),
        "opportunity_input_path": spec.opportunity_input_path,
        "model_key": model_key,
        "window_day_count": window_day_count,
        "ready_day_count": ready_day_count,
        "not_ready_day_count": not_ready_day_count,
        "ready_day_share": (float(ready_day_count) / float(window_day_count)) if window_day_count > 0 else pd.NA,
        "blocking_day_count": blocking_day_count,
        "mean_overall_t_plus_1h_deliverable_mae_mwh": overall_t1.mean(),
        "max_overall_t_plus_1h_deliverable_mae_mwh": overall_t1.max(),
        "mean_overall_t_plus_6h_deliverable_mae_mwh": overall_t6.mean(),
        "max_overall_t_plus_6h_deliverable_mae_mwh": overall_t6.max(),
        "mean_gb_nl_t_plus_1h_deliverable_mae_mwh": gb_nl_t1.mean(),
        "max_gb_nl_t_plus_1h_deliverable_mae_mwh": gb_nl_t1.max(),
        "mean_proxy_internal_transfer_share_t_plus_1h": proxy_share.mean(),
        "max_proxy_internal_transfer_share_t_plus_1h": proxy_share.max(),
        "route_warn_day_count": route_warn_day_count,
        "cluster_warn_day_count": cluster_warn_day_count,
        "severe_focus_day_count": severe_focus_day_count,
        "informative_window_flag": bool(scout_row.get("informative_window_flag", False)),
        "informative_signal_basis": str(scout_row.get("informative_signal_basis", "")),
        "specialist_scope_row_count": int(scout_row.get("specialist_scope_row_count", 0) or 0),
        "specialist_scope_actual_opportunity_deliverable_mwh_sum": float(
            scout_row.get("specialist_scope_actual_opportunity_deliverable_mwh_sum", 0.0) or 0.0
        ),
        "baseline_specialist_t_plus_1h_deliverable_abs_error_mwh_sum": float(
            scout_row.get("baseline_specialist_t_plus_1h_deliverable_abs_error_mwh_sum", 0.0) or 0.0
        ),
        "source_lineage": "fact_model_readiness_daily|fact_model_benchmark_window_scout",
    }
    return pd.DataFrame([row], columns=_empty_reviewed_bundle_batch_window_summary_frame().columns)


def _build_reviewed_bundle_blocker_summary(blocker_rows: pd.DataFrame) -> pd.DataFrame:
    if blocker_rows is None or blocker_rows.empty:
        return _empty_reviewed_bundle_batch_blocker_summary_frame()

    blocker = blocker_rows.copy()
    blocker["blocker_priority_score"] = pd.to_numeric(blocker.get("blocker_priority_score"), errors="coerce")
    blocker["eligible_row_count"] = pd.to_numeric(blocker.get("eligible_row_count"), errors="coerce")
    blocker["actual_volume_mwh"] = pd.to_numeric(blocker.get("actual_volume_mwh"), errors="coerce")
    blocker["mean_deliverable_abs_error_mwh"] = pd.to_numeric(
        blocker.get("mean_deliverable_abs_error_mwh"),
        errors="coerce",
    )
    blocker["max_deliverable_abs_error_mwh"] = pd.to_numeric(
        blocker.get("max_deliverable_abs_error_mwh"),
        errors="coerce",
    )
    blocker["window_date"] = pd.to_datetime(blocker.get("window_date"), utc=True, errors="coerce")

    rows: list[dict[str, object]] = []
    group_columns = [
        "display_order",
        "benchmark_window_key",
        "benchmark_window_label",
        "benchmark_window_start_date",
        "benchmark_window_end_date",
        "opportunity_input_path",
        "model_key",
        "blocker_type",
        "blocker_scope",
    ]
    for group_key, group in blocker.groupby(group_columns, dropna=False, sort=False):
        ranked = group.assign(_priority_order=group["blocker_priority_score"].fillna(float("-inf"))).sort_values(
            ["_priority_order", "actual_volume_mwh", "mean_deliverable_abs_error_mwh"],
            ascending=[False, False, False],
        )
        top = ranked.iloc[0]
        (
            display_order,
            benchmark_window_key,
            benchmark_window_label,
            benchmark_window_start_date,
            benchmark_window_end_date,
            opportunity_input_path,
            model_key,
            blocker_type,
            blocker_scope,
        ) = group_key
        rows.append(
            {
                "display_order": int(display_order),
                "benchmark_window_key": benchmark_window_key,
                "benchmark_window_label": benchmark_window_label,
                "benchmark_window_start_date": benchmark_window_start_date,
                "benchmark_window_end_date": benchmark_window_end_date,
                "opportunity_input_path": opportunity_input_path,
                "model_key": model_key,
                "blocker_type": blocker_type,
                "blocker_scope": blocker_scope,
                "blocker_day_count": int(group["window_date"].nunique()),
                "blocker_row_count": int(len(group)),
                "top_blocker_slice_key": top.get("blocker_slice_key", pd.NA),
                "top_slice_dimension": top.get("slice_dimension", pd.NA),
                "top_slice_value": top.get("slice_value", pd.NA),
                "top_route_name": top.get("route_name", pd.NA),
                "top_cluster_key": top.get("cluster_key", pd.NA),
                "mean_blocker_priority_score": float(group["blocker_priority_score"].mean()),
                "max_blocker_priority_score": float(group["blocker_priority_score"].max()),
                "eligible_row_count_sum": int(group["eligible_row_count"].fillna(0.0).sum()),
                "actual_volume_mwh_sum": float(group["actual_volume_mwh"].fillna(0.0).sum()),
                "mean_deliverable_abs_error_mwh_mean": float(group["mean_deliverable_abs_error_mwh"].mean()),
                "max_deliverable_abs_error_mwh_max": float(group["max_deliverable_abs_error_mwh"].max()),
                "recommended_next_step": top.get("recommended_next_step", pd.NA),
                "source_lineage": "fact_model_blocker_priority",
            }
        )
    summary = pd.DataFrame(rows, columns=_empty_reviewed_bundle_batch_blocker_summary_frame().columns)
    return summary.sort_values(
        ["display_order", "max_blocker_priority_score", "blocker_type"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_fact_model_benchmark_window_scout(
    fact_curtailment_opportunity_hourly: pd.DataFrame,
    fact_backtest_prediction_hourly: pd.DataFrame,
    *,
    benchmark_window_key: str,
    benchmark_window_label: str,
    benchmark_window_start_date: str,
    benchmark_window_end_date: str,
    opportunity_input_path: str,
    baseline_model_key: str = MODEL_POTENTIAL_RATIO_V2,
) -> pd.DataFrame:
    opportunity = fact_curtailment_opportunity_hourly.copy() if fact_curtailment_opportunity_hourly is not None else pd.DataFrame()
    opportunity["interval_start_utc"] = pd.to_datetime(opportunity.get("interval_start_utc"), utc=True, errors="coerce")
    opportunity["date"] = pd.to_datetime(
        opportunity.get("date", pd.Series(pd.NaT, index=opportunity.index)),
        errors="coerce",
    )
    route_hub_mask = (
        opportunity.get("route_name", pd.Series(index=opportunity.index)).eq(SPECIALIST_SCOPE_ROUTE_NAME)
        & opportunity.get("hub_key", pd.Series(index=opportunity.index)).eq(SPECIALIST_SCOPE_HUB_KEY)
    )
    specialist_mask = route_hub_mask & opportunity.get(
        "internal_transfer_evidence_tier",
        pd.Series(index=opportunity.index),
    ).eq(SPECIALIST_SCOPE_INTERNAL_TIER)
    route_hub_scope = opportunity[route_hub_mask].copy()
    specialist_scope = opportunity[specialist_mask].copy()
    specialist_scope["opportunity_deliverable_mwh"] = pd.to_numeric(
        specialist_scope.get("opportunity_deliverable_mwh", pd.Series(0.0, index=specialist_scope.index)),
        errors="coerce",
    ).fillna(0.0)
    specialist_scope["route_price_score_eur_per_mwh"] = pd.to_numeric(
        specialist_scope.get("route_price_score_eur_per_mwh", pd.Series(0.0, index=specialist_scope.index)),
        errors="coerce",
    ).fillna(0.0)
    specialist_scope["deliverable_mw_proxy"] = pd.to_numeric(
        specialist_scope.get("deliverable_mw_proxy", pd.Series(0.0, index=specialist_scope.index)),
        errors="coerce",
    ).fillna(0.0)

    predictions = fact_backtest_prediction_hourly.copy() if fact_backtest_prediction_hourly is not None else pd.DataFrame()
    predictions["forecast_horizon_hours"] = pd.to_numeric(predictions.get("forecast_horizon_hours"), errors="coerce")
    predictions["prediction_eligible_flag"] = predictions.get(
        "prediction_eligible_flag",
        pd.Series(False, index=predictions.index),
    ).fillna(False).astype(bool)
    baseline_scope = predictions[
        predictions.get("model_key", pd.Series(index=predictions.index)).eq(baseline_model_key)
        & predictions["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)
        & predictions["prediction_eligible_flag"]
        & predictions.get("route_name", pd.Series(index=predictions.index)).eq(SPECIALIST_SCOPE_ROUTE_NAME)
        & predictions.get("hub_key", pd.Series(index=predictions.index)).eq(SPECIALIST_SCOPE_HUB_KEY)
        & predictions.get("internal_transfer_evidence_tier", pd.Series(index=predictions.index)).eq(
            SPECIALIST_SCOPE_INTERNAL_TIER
        )
    ].copy()
    baseline_scope["opportunity_deliverable_abs_error_mwh"] = pd.to_numeric(
        baseline_scope.get(
            "opportunity_deliverable_abs_error_mwh",
            pd.Series(0.0, index=baseline_scope.index),
        ),
        errors="coerce",
    ).fillna(0.0)

    informative_scope = specialist_scope
    informative_baseline_scope = baseline_scope
    if informative_scope.empty and not route_hub_scope.empty:
        informative_scope = route_hub_scope.copy()
        informative_scope["opportunity_deliverable_mwh"] = pd.to_numeric(
            informative_scope.get("opportunity_deliverable_mwh", pd.Series(0.0, index=informative_scope.index)),
            errors="coerce",
        ).fillna(0.0)
        informative_baseline_scope = predictions[
            predictions.get("model_key", pd.Series(index=predictions.index)).eq(baseline_model_key)
            & predictions["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)
            & predictions["prediction_eligible_flag"]
            & predictions.get("route_name", pd.Series(index=predictions.index)).eq(SPECIALIST_SCOPE_ROUTE_NAME)
            & predictions.get("hub_key", pd.Series(index=predictions.index)).eq(SPECIALIST_SCOPE_HUB_KEY)
        ].copy()
        informative_baseline_scope["opportunity_deliverable_abs_error_mwh"] = pd.to_numeric(
            informative_baseline_scope.get(
                "opportunity_deliverable_abs_error_mwh",
                pd.Series(0.0, index=informative_baseline_scope.index),
            ),
            errors="coerce",
        ).fillna(0.0)

    actual_sum = float(specialist_scope["opportunity_deliverable_mwh"].sum()) if not specialist_scope.empty else 0.0
    baseline_abs_error_sum = (
        float(baseline_scope["opportunity_deliverable_abs_error_mwh"].sum()) if not baseline_scope.empty else 0.0
    )
    informative_actual_sum = (
        float(informative_scope["opportunity_deliverable_mwh"].sum()) if not informative_scope.empty else 0.0
    )
    informative_baseline_abs_error_sum = (
        float(informative_baseline_scope["opportunity_deliverable_abs_error_mwh"].sum())
        if not informative_baseline_scope.empty
        else 0.0
    )
    if actual_sum > INFORMATIVE_WINDOW_SIGNAL_EPSILON_MWH:
        informative_window_flag = True
        informative_signal_basis = "reviewed_actual_deliverable_mwh_sum"
    elif baseline_abs_error_sum > INFORMATIVE_WINDOW_SIGNAL_EPSILON_MWH:
        informative_window_flag = True
        informative_signal_basis = "reviewed_baseline_abs_error_mwh_sum"
    elif int(len(specialist_scope)) <= 0 and not route_hub_scope.empty:
        if informative_actual_sum > INFORMATIVE_WINDOW_SIGNAL_EPSILON_MWH:
            informative_window_flag = True
            informative_signal_basis = "proxy_route_hub_actual_deliverable_mwh_sum"
        elif informative_baseline_abs_error_sum > INFORMATIVE_WINDOW_SIGNAL_EPSILON_MWH:
            informative_window_flag = True
            informative_signal_basis = "proxy_route_hub_baseline_abs_error_mwh_sum"
        else:
            informative_window_flag = False
            informative_signal_basis = "proxy_route_hub_perfect_zero_window"
    elif int(len(specialist_scope)) <= 0:
        informative_window_flag = False
        informative_signal_basis = "no_reviewed_scope_rows"
    else:
        informative_window_flag = False
        informative_signal_basis = "reviewed_perfect_zero_window"

    row = {
        "benchmark_window_key": benchmark_window_key,
        "benchmark_window_label": benchmark_window_label,
        "benchmark_window_start_date": benchmark_window_start_date,
        "benchmark_window_end_date": benchmark_window_end_date,
        "opportunity_input_path": opportunity_input_path,
        "baseline_model_key": baseline_model_key,
        "scope_route_name": SPECIALIST_SCOPE_ROUTE_NAME,
        "scope_hub_key": SPECIALIST_SCOPE_HUB_KEY,
        "scope_internal_transfer_evidence_tier": SPECIALIST_SCOPE_INTERNAL_TIER,
        "window_day_count": int(specialist_scope["date"].nunique()) if "date" in specialist_scope.columns else 0,
        "specialist_scope_row_count": int(len(specialist_scope)),
        "specialist_scope_actual_opportunity_deliverable_mwh_sum": actual_sum,
        "specialist_scope_nonzero_actual_row_count": int(specialist_scope["opportunity_deliverable_mwh"].gt(0.0).sum()),
        "specialist_scope_positive_route_price_row_count": int(
            specialist_scope["route_price_score_eur_per_mwh"].gt(0.0).sum()
        ),
        "specialist_scope_positive_deliverable_proxy_row_count": int(
            specialist_scope["deliverable_mw_proxy"].gt(0.0).sum()
        ),
        "baseline_specialist_t_plus_1h_deliverable_abs_error_mwh_sum": baseline_abs_error_sum,
        "informative_window_flag": informative_window_flag,
        "informative_signal_basis": informative_signal_basis,
        "source_lineage": "fact_curtailment_opportunity_hourly|fact_backtest_prediction_hourly",
    }
    return pd.DataFrame([row], columns=_empty_model_benchmark_window_scout_frame().columns)


def materialize_model_benchmark_window_scout(
    output_dir: str | Path,
    *,
    opportunity_input_path: str | Path,
    readiness_start: dt.date,
    readiness_end: dt.date,
    benchmark_window_key: str,
    benchmark_window_label: str,
    baseline_model_key: str = MODEL_POTENTIAL_RATIO_V2,
) -> Dict[str, pd.DataFrame]:
    opportunity_input = load_curtailment_opportunity_input(opportunity_input_path)
    opportunity_input = _filter_opportunity_input_to_window(
        opportunity_input,
        readiness_start=readiness_start,
        readiness_end=readiness_end,
    )
    with tempfile.TemporaryDirectory() as tmp_backtest_dir:
        backtest_frames = materialize_opportunity_backtest(
            output_dir=tmp_backtest_dir,
            fact_curtailment_opportunity_hourly=opportunity_input,
            model_key=baseline_model_key,
            forecast_horizons=[READINESS_HORIZON_HOURS],
        )
    scout = build_fact_model_benchmark_window_scout(
        fact_curtailment_opportunity_hourly=opportunity_input,
        fact_backtest_prediction_hourly=backtest_frames[BACKTEST_PREDICTION_TABLE],
        benchmark_window_key=benchmark_window_key,
        benchmark_window_label=benchmark_window_label,
        benchmark_window_start_date=readiness_start.isoformat(),
        benchmark_window_end_date=readiness_end.isoformat(),
        opportunity_input_path=str(opportunity_input_path),
        baseline_model_key=baseline_model_key,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scout.to_csv(output_path / f"{MODEL_BENCHMARK_WINDOW_SCOUT_TABLE}.csv", index=False)
    return {MODEL_BENCHMARK_WINDOW_SCOUT_TABLE: scout}


def materialize_model_benchmark_suite(
    output_dir: str | Path,
    *,
    benchmark_windows: list[BenchmarkWindowSpec],
    model_key: str,
    forecast_horizons: Iterable[int],
    baseline_model_key: str = DEFAULT_READINESS_MODEL_KEY,
    candidate_model_key: str | None = DEFAULT_CANDIDATE_MODEL_KEY,
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


def materialize_reviewed_bundle_batch_evaluation(
    output_dir: str | Path,
    *,
    root_dir: str | Path = ".",
    bundle_glob: str = DEFAULT_REVIEWED_BUNDLE_BATCH_GLOB,
    model_key: str = DEFAULT_READINESS_MODEL_KEY,
    forecast_horizons: Iterable[int] = (READINESS_HORIZON_HOURS,),
    baseline_model_key: str = DEFAULT_READINESS_MODEL_KEY,
) -> Dict[str, pd.DataFrame]:
    discovered_windows = discover_reviewed_bundle_batch_windows(root_dir, bundle_glob=bundle_glob)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    windows_path = output_path / "windows"
    windows_path.mkdir(parents=True, exist_ok=True)

    runtime_windows: list[BenchmarkWindowSpec] = []
    scout_frames: list[pd.DataFrame] = []
    readiness_frames_annotated: list[pd.DataFrame] = []
    blocker_rows_annotated: list[pd.DataFrame] = []
    window_summary_frames: list[pd.DataFrame] = []

    for discovered_spec in discovered_windows:
        opportunity_input = load_curtailment_opportunity_input(discovered_spec.opportunity_input_path)
        actual_start, actual_end = _infer_readiness_window_from_opportunity_input(opportunity_input)
        runtime_spec = BenchmarkWindowSpec(
            benchmark_suite_name=discovered_spec.benchmark_suite_name,
            benchmark_window_key=discovered_spec.benchmark_window_key,
            benchmark_window_label=discovered_spec.benchmark_window_label,
            opportunity_input_path=discovered_spec.opportunity_input_path,
            readiness_start=actual_start,
            readiness_end=actual_end,
            benchmark_window_family=discovered_spec.benchmark_window_family,
            benchmark_role=discovered_spec.benchmark_role,
            promotion_window_flag=discovered_spec.promotion_window_flag,
            display_order=discovered_spec.display_order,
            window_notes=discovered_spec.window_notes,
        )
        runtime_windows.append(runtime_spec)
        window_output_dir = windows_path / runtime_spec.benchmark_window_key
        scoped_input = _filter_opportunity_input_to_window(
            opportunity_input,
            readiness_start=runtime_spec.readiness_start,
            readiness_end=runtime_spec.readiness_end,
        )
        backtest_frames = materialize_opportunity_backtest(
            output_dir=window_output_dir,
            fact_curtailment_opportunity_hourly=scoped_input,
            model_key=model_key,
            forecast_horizons=forecast_horizons,
        )
        selected_model_keys = set(backtest_frames[BACKTEST_PREDICTION_TABLE]["model_key"].dropna())
        readiness_model_key = baseline_model_key if baseline_model_key in selected_model_keys else model_key
        readiness_frames = materialize_model_readiness_review(
            output_dir=window_output_dir,
            fact_backtest_prediction_hourly=backtest_frames[BACKTEST_PREDICTION_TABLE],
            fact_backtest_summary_slice=backtest_frames[BACKTEST_SUMMARY_SLICE_TABLE],
            fact_backtest_top_error_hourly=backtest_frames[BACKTEST_TOP_ERROR_TABLE],
            fact_drift_window=backtest_frames[DRIFT_WINDOW_TABLE],
            model_key=readiness_model_key,
            baseline_model_key=baseline_model_key,
            candidate_model_key=None,
        )
        scout = build_fact_model_benchmark_window_scout(
            fact_curtailment_opportunity_hourly=scoped_input,
            fact_backtest_prediction_hourly=backtest_frames[BACKTEST_PREDICTION_TABLE],
            benchmark_window_key=runtime_spec.benchmark_window_key,
            benchmark_window_label=runtime_spec.benchmark_window_label,
            benchmark_window_start_date=runtime_spec.readiness_start.isoformat(),
            benchmark_window_end_date=runtime_spec.readiness_end.isoformat(),
            opportunity_input_path=runtime_spec.opportunity_input_path,
            baseline_model_key=baseline_model_key,
        )
        scout.to_csv(window_output_dir / f"{MODEL_BENCHMARK_WINDOW_SCOUT_TABLE}.csv", index=False)
        readiness_daily = readiness_frames.get(MODEL_READINESS_TABLE, pd.DataFrame())
        blocker_rows = readiness_frames.get(MODEL_BLOCKER_PRIORITY_TABLE, pd.DataFrame())
        scout_frames.append(_annotate_reviewed_bundle_scout(scout, spec=runtime_spec))
        readiness_frames_annotated.append(
            _annotate_reviewed_bundle_readiness_daily(readiness_daily, spec=runtime_spec)
        )
        blocker_rows_annotated.append(
            _annotate_reviewed_bundle_blocker_rows(blocker_rows, spec=runtime_spec)
        )
        window_summary_frames.append(
            _build_reviewed_bundle_window_summary(readiness_daily, scout, spec=runtime_spec)
        )

    dim_window = build_dim_reviewed_bundle_batch_window(
        runtime_windows,
        discovery_source=f"{Path(root_dir)}::{bundle_glob}",
    )
    scout_root = (
        pd.concat(scout_frames, ignore_index=True)
        if any(not frame.empty for frame in scout_frames)
        else _empty_reviewed_bundle_batch_scout_frame()
    )
    readiness_root = (
        pd.concat(readiness_frames_annotated, ignore_index=True)
        if any(not frame.empty for frame in readiness_frames_annotated)
        else _empty_reviewed_bundle_batch_readiness_daily_frame()
    )
    nonempty_blocker_frames = [frame for frame in blocker_rows_annotated if frame is not None and not frame.empty]
    blocker_detail = pd.concat(nonempty_blocker_frames, ignore_index=True) if nonempty_blocker_frames else pd.DataFrame()
    blocker_summary = _build_reviewed_bundle_blocker_summary(blocker_detail)
    window_summary = (
        pd.concat(window_summary_frames, ignore_index=True)
        if any(not frame.empty for frame in window_summary_frames)
        else _empty_reviewed_bundle_batch_window_summary_frame()
    )
    window_summary = window_summary.sort_values(["display_order", "benchmark_window_key"]).reset_index(drop=True)

    dim_window.to_csv(output_path / f"{REVIEWED_BUNDLE_BATCH_WINDOW_TABLE}.csv", index=False)
    scout_root.to_csv(output_path / f"{REVIEWED_BUNDLE_BATCH_SCOUT_TABLE}.csv", index=False)
    readiness_root.to_csv(output_path / f"{REVIEWED_BUNDLE_BATCH_READINESS_DAILY_TABLE}.csv", index=False)
    window_summary.to_csv(output_path / f"{REVIEWED_BUNDLE_BATCH_WINDOW_SUMMARY_TABLE}.csv", index=False)
    blocker_summary.to_csv(output_path / f"{REVIEWED_BUNDLE_BATCH_BLOCKER_SUMMARY_TABLE}.csv", index=False)

    return {
        REVIEWED_BUNDLE_BATCH_WINDOW_TABLE: dim_window,
        REVIEWED_BUNDLE_BATCH_SCOUT_TABLE: scout_root,
        REVIEWED_BUNDLE_BATCH_READINESS_DAILY_TABLE: readiness_root,
        REVIEWED_BUNDLE_BATCH_WINDOW_SUMMARY_TABLE: window_summary,
        REVIEWED_BUNDLE_BATCH_BLOCKER_SUMMARY_TABLE: blocker_summary,
    }
