from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from opportunity_backtest import MODEL_GB_NL_REVIEWED_SPECIALIST_V3, MODEL_POTENTIAL_RATIO_V2


MODEL_READINESS_TABLE = "fact_model_readiness_daily"
MODEL_BLOCKER_PRIORITY_TABLE = "fact_model_blocker_priority"
MODEL_CANDIDATE_COMPARE_TABLE = "fact_model_candidate_compare_daily"
MODEL_CANDIDATE_COMPARE_WINDOW_TABLE = "fact_model_candidate_compare_window"
MODEL_CANDIDATE_COMPARE_SUITE_TABLE = "fact_model_candidate_compare_suite"
DEFAULT_READINESS_MODEL_KEY = MODEL_POTENTIAL_RATIO_V2
DEFAULT_CANDIDATE_MODEL_KEY = MODEL_GB_NL_REVIEWED_SPECIALIST_V3
TARGET_ROUTE_NAME = "R2_netback_GB_NL_DE_PL"
TARGET_T1_DELIVERABLE_MAE = 0.50
TARGET_GBNL_T1_DELIVERABLE_MAE = 1.50
TARGET_PROXY_SHARE_MAX = 0.45
TARGET_CAPACITY_UNKNOWN_SHARE_MAX = 0.25
SEVERE_ROUTE_VOLUME_MWH = 25.0
SEVERE_ROUTE_MAE_MWH = 1.50
READINESS_HORIZON_HOURS = 1

BLOCKER_WEIGHTS = {
    "gb_nl_t_plus_1h_mae_above_target": 1.60,
    "route_drift_warn": 1.45,
    "cluster_drift_warn": 1.35,
    "severe_fallback_route_focus_remaining": 1.30,
    "proxy_internal_transfer_share_too_high": 1.20,
    "capacity_unknown_route_share_too_high": 1.15,
    "overall_t_plus_1h_mae_above_target": 1.00,
}

BLOCKER_SCOPE = {
    "overall_t_plus_1h_mae_above_target": "global_slice",
    "gb_nl_t_plus_1h_mae_above_target": "route_slice",
    "proxy_internal_transfer_share_too_high": "internal_transfer_slice",
    "capacity_unknown_route_share_too_high": "route_delivery_slice",
    "severe_fallback_route_focus_remaining": "fallback_slice",
    "route_drift_warn": "route_drift_slice",
    "cluster_drift_warn": "cluster_drift_slice",
}


def _empty_model_readiness_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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


def _empty_model_blocker_priority_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "window_date",
            "model_key",
            "forecast_horizon_hours",
            "blocker_type",
            "blocker_scope",
            "slice_dimension",
            "slice_value",
            "blocker_slice_key",
            "route_name",
            "cluster_key",
            "hub_key",
            "internal_transfer_evidence_tier",
            "route_delivery_tier",
            "error_focus_area",
            "drift_scope",
            "drift_state",
            "eligible_row_count",
            "top_error_row_count",
            "summary_slice_eligible_row_count",
            "actual_volume_mwh",
            "mean_deliverable_abs_error_mwh",
            "max_deliverable_abs_error_mwh",
            "summary_mae_opportunity_deliverable_mwh",
            "summary_mae_opportunity_gross_value_eur",
            "summary_error_reduction_priority_rank",
            "feature_drift_score",
            "target_drift_score",
            "residual_drift_score",
            "blocker_priority_score",
            "blocker_priority_rank",
            "blocker_summary",
            "recommended_next_step",
            "source_lineage",
        ]
    )


def _empty_model_candidate_compare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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


def _empty_model_candidate_compare_window_frame() -> pd.DataFrame:
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
            "baseline_model_key",
            "candidate_model_key",
            "window_day_count",
            "candidate_scope_row_count",
            "overall_joined_row_count",
            "baseline_overall_deliverable_abs_error_mwh_sum",
            "candidate_overall_deliverable_abs_error_mwh_sum",
            "overall_t_plus_1h_deliverable_mae_delta_mwh",
            "gb_nl_joined_row_count",
            "baseline_gb_nl_deliverable_abs_error_mwh_sum",
            "candidate_gb_nl_deliverable_abs_error_mwh_sum",
            "gb_nl_t_plus_1h_deliverable_mae_delta_mwh",
            "gb_nl_reviewed_internal_joined_row_count",
            "baseline_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum",
            "candidate_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum",
            "gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh",
            "baseline_blocker_row_count",
            "candidate_blocker_row_count",
            "blocker_row_delta",
            "baseline_severe_focus_area_count",
            "candidate_severe_focus_area_count",
            "severe_focus_area_delta",
            "promotion_state",
            "source_lineage",
        ]
    )


def _empty_model_candidate_compare_suite_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "benchmark_suite_name",
            "suite_scope",
            "baseline_model_key",
            "candidate_model_key",
            "window_count",
            "benchmark_day_count",
            "candidate_scope_row_count",
            "overall_joined_row_count",
            "baseline_overall_deliverable_abs_error_mwh_sum",
            "candidate_overall_deliverable_abs_error_mwh_sum",
            "overall_t_plus_1h_deliverable_mae_delta_mwh",
            "gb_nl_joined_row_count",
            "baseline_gb_nl_deliverable_abs_error_mwh_sum",
            "candidate_gb_nl_deliverable_abs_error_mwh_sum",
            "gb_nl_t_plus_1h_deliverable_mae_delta_mwh",
            "gb_nl_reviewed_internal_joined_row_count",
            "baseline_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum",
            "candidate_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum",
            "gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh",
            "baseline_blocker_row_count",
            "candidate_blocker_row_count",
            "blocker_row_delta",
            "baseline_severe_focus_area_count",
            "candidate_severe_focus_area_count",
            "severe_focus_area_delta",
            "candidate_beats_window_count",
            "candidate_mixed_window_count",
            "candidate_regresses_window_count",
            "candidate_insufficient_coverage_window_count",
            "promotion_state",
            "source_lineage",
        ]
    )


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _daily_prediction_slice(
    predictions: pd.DataFrame,
    *,
    model_key: str,
    horizon_hours: int,
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["window_date"] = frame["interval_start_utc"].dt.floor("d")
    eligible = frame[
        frame["model_key"].eq(model_key)
        & frame["forecast_horizon_hours"].eq(horizon_hours)
        & frame["prediction_eligible_flag"].fillna(False)
    ].copy()
    return eligible


def _daily_route_severe_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    grouped = frame.groupby("route_name", dropna=False).agg(
        actual_volume=("actual_opportunity_deliverable_mwh", lambda values: float(pd.to_numeric(values, errors="coerce").sum())),
        deliverable_mae=("opportunity_deliverable_abs_error_mwh", lambda values: float(pd.to_numeric(values, errors="coerce").mean())),
        proxy_share=("internal_transfer_evidence_tier", lambda values: float(pd.Series(values).fillna("").eq("gb_topology_transfer_gate_proxy").mean())),
        capacity_unknown_share=("route_delivery_tier", lambda values: float(pd.Series(values).fillna("").eq("capacity_unknown").mean())),
    )
    severe = grouped[
        grouped["actual_volume"].ge(SEVERE_ROUTE_VOLUME_MWH)
        & grouped["deliverable_mae"].ge(SEVERE_ROUTE_MAE_MWH)
        & (
            grouped["proxy_share"].ge(TARGET_PROXY_SHARE_MAX)
            | grouped["capacity_unknown_share"].ge(TARGET_CAPACITY_UNKNOWN_SHARE_MAX)
        )
    ]
    return int(len(severe))


def build_fact_model_readiness_daily(
    fact_backtest_prediction_hourly: pd.DataFrame,
    fact_drift_window: pd.DataFrame,
    model_key: str = DEFAULT_READINESS_MODEL_KEY,
) -> pd.DataFrame:
    if fact_backtest_prediction_hourly is None or fact_backtest_prediction_hourly.empty:
        return _empty_model_readiness_frame()

    predictions = fact_backtest_prediction_hourly.copy()
    predictions["interval_start_utc"] = pd.to_datetime(predictions["interval_start_utc"], utc=True, errors="coerce")
    predictions["window_date"] = predictions["interval_start_utc"].dt.floor("d")

    drift = fact_drift_window.copy() if fact_drift_window is not None else pd.DataFrame()
    if not drift.empty:
        drift["window_date"] = pd.to_datetime(drift["window_date"], errors="coerce", utc=True)

    t1 = _daily_prediction_slice(predictions, model_key=model_key, horizon_hours=1)
    t6 = _daily_prediction_slice(predictions, model_key=model_key, horizon_hours=6)
    if t1.empty and t6.empty:
        return _empty_model_readiness_frame()

    all_dates = sorted(set(t1["window_date"]).union(set(t6["window_date"])))
    rows = []
    for window_date in all_dates:
        t1_day = t1[t1["window_date"].eq(window_date)].copy()
        t6_day = t6[t6["window_date"].eq(window_date)].copy()
        gb_nl_t1 = t1_day[t1_day["route_name"].eq(TARGET_ROUTE_NAME)].copy()
        route_warn_count = 0
        cluster_warn_count = 0
        if not drift.empty:
            drift_day = drift[
                drift["model_key"].eq(model_key)
                & drift["forecast_horizon_hours"].eq(1)
                & drift["window_date"].eq(window_date)
            ].copy()
            route_warn_count = int(
                drift_day[drift_day["drift_scope"].eq("route_daily")]["drift_state"].fillna("warmup").eq("warn").sum()
            )
            cluster_warn_count = int(
                drift_day[drift_day["drift_scope"].eq("cluster_daily")]["drift_state"].fillna("warmup").eq("warn").sum()
            )

        overall_t1_mae = _safe_mean(t1_day, "opportunity_deliverable_abs_error_mwh")
        overall_t6_mae = _safe_mean(t6_day, "opportunity_deliverable_abs_error_mwh")
        gb_nl_t1_mae = _safe_mean(gb_nl_t1, "opportunity_deliverable_abs_error_mwh")
        proxy_share = float(t1_day["internal_transfer_evidence_tier"].fillna("").eq("gb_topology_transfer_gate_proxy").mean()) if not t1_day.empty else np.nan
        reviewed_share = float(t1_day["internal_transfer_evidence_tier"].fillna("").ne("gb_topology_transfer_gate_proxy").mean()) if not t1_day.empty else np.nan
        capacity_unknown_share = float(t1_day["route_delivery_tier"].fillna("").eq("capacity_unknown").mean()) if not t1_day.empty else np.nan
        severe_focus_count = _daily_route_severe_count(t1_day)

        blocking_reasons = []
        if pd.notna(overall_t1_mae) and overall_t1_mae > TARGET_T1_DELIVERABLE_MAE:
            blocking_reasons.append("overall_t_plus_1h_mae_above_target")
        if pd.notna(gb_nl_t1_mae) and gb_nl_t1_mae > TARGET_GBNL_T1_DELIVERABLE_MAE:
            blocking_reasons.append("gb_nl_t_plus_1h_mae_above_target")
        if pd.notna(proxy_share) and proxy_share > TARGET_PROXY_SHARE_MAX:
            blocking_reasons.append("proxy_internal_transfer_share_too_high")
        if pd.notna(capacity_unknown_share) and capacity_unknown_share > TARGET_CAPACITY_UNKNOWN_SHARE_MAX:
            blocking_reasons.append("capacity_unknown_route_share_too_high")
        if severe_focus_count > 0:
            blocking_reasons.append("severe_fallback_route_focus_remaining")
        if route_warn_count > 0:
            blocking_reasons.append("route_drift_warn")
        if cluster_warn_count > 0:
            blocking_reasons.append("cluster_drift_warn")

        rows.append(
            {
                "window_date": window_date,
                "model_key": model_key,
                "overall_t_plus_1h_deliverable_mae_mwh": overall_t1_mae,
                "overall_t_plus_6h_deliverable_mae_mwh": overall_t6_mae,
                "gb_nl_t_plus_1h_deliverable_mae_mwh": gb_nl_t1_mae,
                "proxy_internal_transfer_share_t_plus_1h": proxy_share,
                "reviewed_internal_transfer_share_t_plus_1h": reviewed_share,
                "capacity_unknown_route_share_t_plus_1h": capacity_unknown_share,
                "route_warn_count_t_plus_1h": route_warn_count,
                "cluster_warn_count_t_plus_1h": cluster_warn_count,
                "severe_unresolved_focus_area_count_t_plus_1h": severe_focus_count,
                "model_ready_flag": len(blocking_reasons) == 0,
                "model_readiness_state": "ready_for_map" if len(blocking_reasons) == 0 else "not_ready",
                "blocking_reasons": ";".join(blocking_reasons),
                "source_lineage": "fact_backtest_prediction_hourly|fact_drift_window",
            }
        )

    readiness = pd.DataFrame(rows, columns=_empty_model_readiness_frame().columns)
    return readiness.sort_values(["window_date", "model_key"]).reset_index(drop=True)


def _split_blocking_reasons(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _normalize_window_dates(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "window_date" in normalized.columns:
        normalized["window_date"] = pd.to_datetime(normalized["window_date"], utc=True, errors="coerce")
    return normalized


def _candidate_compare_join(
    predictions: pd.DataFrame,
    *,
    baseline_model_key: str,
    candidate_model_key: str,
) -> pd.DataFrame:
    empty_columns = [
        "forecast_horizon_hours",
        "interval_start_utc",
        "cluster_key",
        "route_name",
        "hub_key",
        "baseline_window_date",
        "baseline_prediction_eligible_flag",
        "baseline_deliverable_abs_error_mwh",
        "baseline_internal_transfer_evidence_tier",
        "candidate_window_date",
        "candidate_prediction_eligible_flag",
        "candidate_deliverable_abs_error_mwh",
        "candidate_internal_transfer_evidence_tier",
        "window_date",
    ]
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=empty_columns)
    frame = predictions.copy()
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["window_date"] = frame["interval_start_utc"].dt.floor("d")
    frame["forecast_horizon_hours"] = pd.to_numeric(frame["forecast_horizon_hours"], errors="coerce")
    frame["prediction_eligible_flag"] = frame["prediction_eligible_flag"].fillna(False).astype(bool)
    t1 = frame[frame["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)].copy()
    baseline = t1[t1["model_key"].eq(baseline_model_key)].copy()
    candidate = t1[t1["model_key"].eq(candidate_model_key)].copy()
    if baseline.empty or candidate.empty:
        return pd.DataFrame(columns=empty_columns)

    join_keys = [
        "forecast_horizon_hours",
        "interval_start_utc",
        "cluster_key",
        "route_name",
        "hub_key",
    ]
    baseline = baseline[
        [
            *join_keys,
            "window_date",
            "prediction_eligible_flag",
            "opportunity_deliverable_abs_error_mwh",
            "internal_transfer_evidence_tier",
        ]
    ].rename(
        columns={
            "window_date": "baseline_window_date",
            "prediction_eligible_flag": "baseline_prediction_eligible_flag",
            "opportunity_deliverable_abs_error_mwh": "baseline_deliverable_abs_error_mwh",
            "internal_transfer_evidence_tier": "baseline_internal_transfer_evidence_tier",
        }
    )
    candidate = candidate[
        [
            *join_keys,
            "window_date",
            "prediction_eligible_flag",
            "opportunity_deliverable_abs_error_mwh",
            "internal_transfer_evidence_tier",
        ]
    ].rename(
        columns={
            "window_date": "candidate_window_date",
            "prediction_eligible_flag": "candidate_prediction_eligible_flag",
            "opportunity_deliverable_abs_error_mwh": "candidate_deliverable_abs_error_mwh",
            "internal_transfer_evidence_tier": "candidate_internal_transfer_evidence_tier",
        }
    )
    joined = baseline.merge(candidate, on=join_keys, how="inner")
    joined["window_date"] = joined["candidate_window_date"].where(
        joined["candidate_window_date"].notna(),
        joined["baseline_window_date"],
    )
    joined = joined[
        joined["baseline_prediction_eligible_flag"] & joined["candidate_prediction_eligible_flag"]
    ].copy()
    for column in empty_columns:
        if column not in joined.columns:
            joined[column] = pd.NA
    return joined[empty_columns]


def _candidate_mae_delta(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    candidate_mae = pd.to_numeric(frame["candidate_deliverable_abs_error_mwh"], errors="coerce").mean()
    baseline_mae = pd.to_numeric(frame["baseline_deliverable_abs_error_mwh"], errors="coerce").mean()
    if pd.isna(candidate_mae) or pd.isna(baseline_mae):
        return np.nan
    return float(candidate_mae - baseline_mae)


def _promotion_state_from_deltas(
    *,
    candidate_scope_row_count: int,
    deltas: list[float],
) -> str:
    valid_deltas = [float(value) for value in deltas if pd.notna(value)]
    if candidate_scope_row_count <= 0 or not valid_deltas:
        return "candidate_insufficient_coverage"
    if all(value <= 0.0 for value in valid_deltas) and any(value < 0.0 for value in valid_deltas):
        return "candidate_beats_baseline"
    if all(value >= 0.0 for value in valid_deltas) and any(value > 0.0 for value in valid_deltas):
        return "candidate_regresses_baseline"
    return "candidate_mixed"


def _candidate_compare_slice_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    empty = {
        "row_count": 0,
        "baseline_abs_error_sum": 0.0,
        "candidate_abs_error_sum": 0.0,
        "mae_delta_mwh": np.nan,
    }
    if frame.empty:
        return empty
    baseline = pd.to_numeric(frame["baseline_deliverable_abs_error_mwh"], errors="coerce")
    candidate = pd.to_numeric(frame["candidate_deliverable_abs_error_mwh"], errors="coerce")
    valid = frame[baseline.notna() & candidate.notna()].copy()
    if valid.empty:
        return empty
    baseline_valid = pd.to_numeric(valid["baseline_deliverable_abs_error_mwh"], errors="coerce")
    candidate_valid = pd.to_numeric(valid["candidate_deliverable_abs_error_mwh"], errors="coerce")
    row_count = int(len(valid))
    baseline_sum = float(baseline_valid.sum())
    candidate_sum = float(candidate_valid.sum())
    return {
        "row_count": row_count,
        "baseline_abs_error_sum": baseline_sum,
        "candidate_abs_error_sum": candidate_sum,
        "mae_delta_mwh": float((candidate_sum / row_count) - (baseline_sum / row_count)),
    }


def _safe_numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce")
    if not series.notna().any():
        return 0.0
    return float(series.sum())


def _mae_delta_from_sums(*, baseline_sum: float, candidate_sum: float, row_count: int) -> float:
    if row_count <= 0:
        return np.nan
    return float((candidate_sum / row_count) - (baseline_sum / row_count))


def _summary_row_for_candidate(
    summary: pd.DataFrame,
    *,
    model_key: str,
    slice_dimension: str | None,
    slice_value: object,
) -> pd.Series | None:
    if summary.empty or not slice_dimension:
        return None
    expected_value = "<NA>" if pd.isna(slice_value) else str(slice_value)
    subset = summary[
        summary["model_key"].eq(model_key)
        & summary["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)
        & summary["slice_dimension"].eq(slice_dimension)
        & summary["slice_value"].astype(str).eq(expected_value)
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(
        ["error_reduction_priority_rank", "mae_opportunity_deliverable_mwh", "eligible_row_count"],
        ascending=[True, False, False],
        na_position="last",
    )
    return subset.iloc[0]


def _summary_dimension_for_reason(reason: str, candidate: dict[str, object]) -> tuple[str | None, object]:
    if reason in {"gb_nl_t_plus_1h_mae_above_target", "route_drift_warn"} and pd.notna(candidate.get("route_name")):
        return "route_name", candidate.get("route_name")
    if reason == "cluster_drift_warn" and pd.notna(candidate.get("cluster_key")):
        return "cluster_key", candidate.get("cluster_key")
    if reason == "proxy_internal_transfer_share_too_high":
        return "internal_transfer_evidence_tier", "gb_topology_transfer_gate_proxy"
    if reason == "capacity_unknown_route_share_too_high":
        return "route_delivery_tier", "capacity_unknown"
    if reason == "severe_fallback_route_focus_remaining" and pd.notna(candidate.get("route_name")):
        return "route_name", candidate.get("route_name")
    if pd.notna(candidate.get("route_name")):
        return "route_name", candidate.get("route_name")
    if pd.notna(candidate.get("cluster_key")):
        return "cluster_key", candidate.get("cluster_key")
    return None, pd.NA


def _matching_drift_row(
    route_warns: pd.DataFrame,
    cluster_warns: pd.DataFrame,
    *,
    route_name: object,
    cluster_key: object,
) -> pd.Series | None:
    matches = []
    if pd.notna(route_name) and not route_warns.empty:
        route_match = route_warns[route_warns["route_name"].eq(route_name)].copy()
        if not route_match.empty:
            matches.append(route_match)
    if pd.notna(cluster_key) and not cluster_warns.empty:
        cluster_match = cluster_warns[cluster_warns["cluster_key"].eq(cluster_key)].copy()
        if not cluster_match.empty:
            matches.append(cluster_match)
    if not matches:
        return None
    combined = pd.concat(matches, ignore_index=True)
    combined = combined.sort_values(
        ["residual_drift_score", "target_drift_score", "feature_drift_score"],
        ascending=[False, False, False],
        na_position="last",
    )
    return combined.iloc[0]


def _recommended_next_step(reason: str) -> str:
    mapping = {
        "overall_t_plus_1h_mae_above_target": "reduce high-volume slice error before the next readiness rerun",
        "gb_nl_t_plus_1h_mae_above_target": "inspect GB-NL route slices and worst hours before another model change",
        "proxy_internal_transfer_share_too_high": "replace proxy internal-transfer reliance with reviewed evidence on this slice",
        "capacity_unknown_route_share_too_high": "tighten route-capacity evidence or keep this slice explicitly conservative",
        "severe_fallback_route_focus_remaining": "reduce fallback-heavy route dependence before map gating",
        "route_drift_warn": "inspect warned route slices and regime-transition hours",
        "cluster_drift_warn": "inspect warned cluster slices and internal-transfer assumptions",
    }
    return mapping.get(reason, "inspect the ranked slice before changing the model")


def _blocker_slice_key(candidate: dict[str, object]) -> str:
    parts = [
        f"route={candidate.get('route_name', '<NA>')}",
        f"cluster={candidate.get('cluster_key', '<NA>')}",
        f"hub={candidate.get('hub_key', '<NA>')}",
        f"internal={candidate.get('internal_transfer_evidence_tier', '<NA>')}",
        f"tier={candidate.get('route_delivery_tier', '<NA>')}",
        f"focus={candidate.get('error_focus_area', '<NA>')}",
    ]
    return "|".join(str(part) for part in parts)


def _aggregate_top_error_candidates(top_errors: pd.DataFrame) -> list[dict[str, object]]:
    if top_errors.empty:
        return []
    grouped = top_errors.groupby(
        [
            "route_name",
            "cluster_key",
            "hub_key",
            "internal_transfer_evidence_tier",
            "route_delivery_tier",
            "error_focus_area",
        ],
        dropna=False,
    )
    rows: list[dict[str, object]] = []
    for group_keys, group_frame in grouped:
        (
            route_name,
            cluster_key,
            hub_key,
            internal_transfer_evidence_tier,
            route_delivery_tier,
            error_focus_area,
        ) = group_keys
        rows.append(
            {
                "route_name": route_name,
                "cluster_key": cluster_key,
                "hub_key": hub_key,
                "internal_transfer_evidence_tier": internal_transfer_evidence_tier,
                "route_delivery_tier": route_delivery_tier,
                "error_focus_area": error_focus_area,
                "top_error_row_count": int(len(group_frame)),
                "eligible_row_count": int(len(group_frame)),
                "actual_volume_mwh": float(
                    pd.to_numeric(group_frame["actual_opportunity_deliverable_mwh"], errors="coerce").sum()
                ),
                "mean_deliverable_abs_error_mwh": float(
                    pd.to_numeric(group_frame["opportunity_deliverable_abs_error_mwh"], errors="coerce").mean()
                ),
                "max_deliverable_abs_error_mwh": float(
                    pd.to_numeric(group_frame["opportunity_deliverable_abs_error_mwh"], errors="coerce").max()
                ),
            }
        )
    return rows


def _filtered_top_errors_for_reason(
    top_errors: pd.DataFrame,
    *,
    blocker_type: str,
    route_warns: pd.DataFrame,
    cluster_warns: pd.DataFrame,
) -> pd.DataFrame:
    filtered = top_errors.copy()
    if blocker_type == "gb_nl_t_plus_1h_mae_above_target":
        filtered = filtered[filtered["route_name"].eq(TARGET_ROUTE_NAME)].copy()
    elif blocker_type == "proxy_internal_transfer_share_too_high":
        filtered = filtered[
            filtered["internal_transfer_evidence_tier"].eq("gb_topology_transfer_gate_proxy")
        ].copy()
    elif blocker_type == "capacity_unknown_route_share_too_high":
        filtered = filtered[filtered["route_delivery_tier"].eq("capacity_unknown")].copy()
    elif blocker_type == "severe_fallback_route_focus_remaining":
        filtered = filtered[
            filtered["internal_transfer_evidence_tier"].eq("gb_topology_transfer_gate_proxy")
            | filtered["route_delivery_tier"].eq("capacity_unknown")
        ].copy()
    elif blocker_type == "route_drift_warn" and not route_warns.empty:
        filtered = filtered[filtered["route_name"].isin(route_warns["route_name"])].copy()
    elif blocker_type == "cluster_drift_warn" and not cluster_warns.empty:
        filtered = filtered[filtered["cluster_key"].isin(cluster_warns["cluster_key"])].copy()
    return filtered


def _fallback_candidates_for_reason(
    *,
    blocker_type: str,
    route_warns: pd.DataFrame,
    cluster_warns: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if blocker_type == "gb_nl_t_plus_1h_mae_above_target":
        rows.append(
            {
                "route_name": TARGET_ROUTE_NAME,
                "cluster_key": pd.NA,
                "hub_key": pd.NA,
                "internal_transfer_evidence_tier": pd.NA,
                "route_delivery_tier": pd.NA,
                "error_focus_area": "gb_nl_route",
                "top_error_row_count": 0,
                "eligible_row_count": 0,
                "actual_volume_mwh": 0.0,
                "mean_deliverable_abs_error_mwh": np.nan,
                "max_deliverable_abs_error_mwh": np.nan,
            }
        )
    elif blocker_type == "proxy_internal_transfer_share_too_high":
        rows.append(
            {
                "route_name": pd.NA,
                "cluster_key": pd.NA,
                "hub_key": pd.NA,
                "internal_transfer_evidence_tier": "gb_topology_transfer_gate_proxy",
                "route_delivery_tier": pd.NA,
                "error_focus_area": "proxy_internal_transfer",
                "top_error_row_count": 0,
                "eligible_row_count": 0,
                "actual_volume_mwh": 0.0,
                "mean_deliverable_abs_error_mwh": np.nan,
                "max_deliverable_abs_error_mwh": np.nan,
            }
        )
    elif blocker_type == "capacity_unknown_route_share_too_high":
        rows.append(
            {
                "route_name": pd.NA,
                "cluster_key": pd.NA,
                "hub_key": pd.NA,
                "internal_transfer_evidence_tier": pd.NA,
                "route_delivery_tier": "capacity_unknown",
                "error_focus_area": "capacity_unknown",
                "top_error_row_count": 0,
                "eligible_row_count": 0,
                "actual_volume_mwh": 0.0,
                "mean_deliverable_abs_error_mwh": np.nan,
                "max_deliverable_abs_error_mwh": np.nan,
            }
        )
    elif blocker_type == "route_drift_warn":
        for _, drift_row in route_warns.iterrows():
            rows.append(
                {
                    "route_name": drift_row.get("route_name", pd.NA),
                    "cluster_key": pd.NA,
                    "hub_key": pd.NA,
                    "internal_transfer_evidence_tier": pd.NA,
                    "route_delivery_tier": pd.NA,
                    "error_focus_area": "route_drift_warn",
                    "top_error_row_count": 0,
                    "eligible_row_count": int(drift_row.get("eligible_row_count", 0) or 0),
                    "actual_volume_mwh": 0.0,
                    "mean_deliverable_abs_error_mwh": np.nan,
                    "max_deliverable_abs_error_mwh": np.nan,
                }
            )
    elif blocker_type == "cluster_drift_warn":
        for _, drift_row in cluster_warns.iterrows():
            rows.append(
                {
                    "route_name": pd.NA,
                    "cluster_key": drift_row.get("cluster_key", pd.NA),
                    "hub_key": pd.NA,
                    "internal_transfer_evidence_tier": pd.NA,
                    "route_delivery_tier": pd.NA,
                    "error_focus_area": "cluster_drift_warn",
                    "top_error_row_count": 0,
                    "eligible_row_count": int(drift_row.get("eligible_row_count", 0) or 0),
                    "actual_volume_mwh": 0.0,
                    "mean_deliverable_abs_error_mwh": np.nan,
                    "max_deliverable_abs_error_mwh": np.nan,
                }
            )
    return rows


def _priority_score(
    blocker_type: str,
    *,
    actual_volume_mwh: float,
    mean_deliverable_abs_error_mwh: float,
    max_deliverable_abs_error_mwh: float,
    summary_mae_opportunity_deliverable_mwh: float,
    summary_error_reduction_priority_rank: float,
    feature_drift_score: float,
    target_drift_score: float,
    residual_drift_score: float,
) -> float:
    blocker_weight = BLOCKER_WEIGHTS.get(blocker_type, 1.0)
    summary_rank_bonus = 0.0
    if pd.notna(summary_error_reduction_priority_rank) and float(summary_error_reduction_priority_rank) > 0:
        summary_rank_bonus = 1.0 / float(summary_error_reduction_priority_rank)
    raw_score = (
        (0.0 if pd.isna(max_deliverable_abs_error_mwh) else float(max_deliverable_abs_error_mwh) * 3.0)
        + (0.0 if pd.isna(mean_deliverable_abs_error_mwh) else float(mean_deliverable_abs_error_mwh) * 2.0)
        + (0.0 if pd.isna(summary_mae_opportunity_deliverable_mwh) else float(summary_mae_opportunity_deliverable_mwh))
        + min(500.0, 0.0 if pd.isna(actual_volume_mwh) else float(actual_volume_mwh)) / 25.0
        + (0.0 if pd.isna(feature_drift_score) else float(feature_drift_score) * 5.0)
        + (0.0 if pd.isna(target_drift_score) else float(target_drift_score) * 5.0)
        + (0.0 if pd.isna(residual_drift_score) else float(residual_drift_score) * 5.0)
        + summary_rank_bonus
    )
    return blocker_weight * raw_score


def build_fact_model_blocker_priority(
    fact_model_readiness_daily: pd.DataFrame,
    fact_backtest_summary_slice: pd.DataFrame,
    fact_backtest_top_error_hourly: pd.DataFrame,
    fact_drift_window: pd.DataFrame,
) -> pd.DataFrame:
    if fact_model_readiness_daily is None or fact_model_readiness_daily.empty:
        return _empty_model_blocker_priority_frame()

    readiness = _normalize_window_dates(fact_model_readiness_daily)
    summary = (
        fact_backtest_summary_slice.copy()
        if fact_backtest_summary_slice is not None and not fact_backtest_summary_slice.empty
        else pd.DataFrame(
            columns=[
                "model_key",
                "forecast_horizon_hours",
                "slice_dimension",
                "slice_value",
                "eligible_row_count",
                "mae_opportunity_deliverable_mwh",
                "mae_opportunity_gross_value_eur",
                "error_reduction_priority_rank",
            ]
        )
    )
    top_error = (
        fact_backtest_top_error_hourly.copy()
        if fact_backtest_top_error_hourly is not None and not fact_backtest_top_error_hourly.empty
        else pd.DataFrame(
            columns=[
                "model_key",
                "forecast_horizon_hours",
                "window_date",
                "interval_start_utc",
                "route_name",
                "cluster_key",
                "hub_key",
                "internal_transfer_evidence_tier",
                "route_delivery_tier",
                "error_focus_area",
                "actual_opportunity_deliverable_mwh",
                "opportunity_deliverable_abs_error_mwh",
            ]
        )
    )
    drift = (
        _normalize_window_dates(fact_drift_window.copy())
        if fact_drift_window is not None and not fact_drift_window.empty
        else pd.DataFrame(
            columns=[
                "model_key",
                "forecast_horizon_hours",
                "window_date",
                "drift_scope",
                "drift_state",
                "route_name",
                "cluster_key",
                "eligible_row_count",
                "feature_drift_score",
                "target_drift_score",
                "residual_drift_score",
            ]
        )
    )

    if not top_error.empty:
        top_error["interval_start_utc"] = pd.to_datetime(top_error["interval_start_utc"], utc=True, errors="coerce")
        top_error["window_date"] = top_error["interval_start_utc"].dt.floor("d")
    if not summary.empty:
        summary["forecast_horizon_hours"] = pd.to_numeric(summary["forecast_horizon_hours"], errors="coerce")
    if not drift.empty:
        drift["forecast_horizon_hours"] = pd.to_numeric(drift["forecast_horizon_hours"], errors="coerce")

    rows: list[dict[str, object]] = []
    for _, readiness_row in readiness.iterrows():
        window_date = readiness_row["window_date"]
        model_key = readiness_row["model_key"]
        blocker_types = _split_blocking_reasons(readiness_row.get("blocking_reasons"))
        if not blocker_types:
            continue

        day_top_error = top_error[
            top_error["model_key"].eq(model_key)
            & top_error["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)
            & top_error["window_date"].eq(window_date)
        ].copy()
        day_route_warns = drift[
            drift["model_key"].eq(model_key)
            & drift["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)
            & drift["window_date"].eq(window_date)
            & drift["drift_scope"].eq("route_daily")
            & drift["drift_state"].eq("warn")
        ].copy()
        day_cluster_warns = drift[
            drift["model_key"].eq(model_key)
            & drift["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)
            & drift["window_date"].eq(window_date)
            & drift["drift_scope"].eq("cluster_daily")
            & drift["drift_state"].eq("warn")
        ].copy()

        for blocker_type in blocker_types:
            blocker_top_error = _filtered_top_errors_for_reason(
                day_top_error,
                blocker_type=blocker_type,
                route_warns=day_route_warns,
                cluster_warns=day_cluster_warns,
            )
            candidates = _aggregate_top_error_candidates(blocker_top_error)
            if not candidates:
                candidates = _fallback_candidates_for_reason(
                    blocker_type=blocker_type,
                    route_warns=day_route_warns,
                    cluster_warns=day_cluster_warns,
                )

            for candidate in candidates:
                slice_dimension, slice_value = _summary_dimension_for_reason(blocker_type, candidate)
                summary_row = _summary_row_for_candidate(
                    summary,
                    model_key=model_key,
                    slice_dimension=slice_dimension,
                    slice_value=slice_value,
                )
                drift_row = _matching_drift_row(
                    day_route_warns,
                    day_cluster_warns,
                    route_name=candidate.get("route_name"),
                    cluster_key=candidate.get("cluster_key"),
                )
                summary_eligible_row_count = (
                    int(summary_row["eligible_row_count"]) if summary_row is not None and pd.notna(summary_row["eligible_row_count"]) else 0
                )
                summary_mae_deliv = (
                    float(summary_row["mae_opportunity_deliverable_mwh"])
                    if summary_row is not None and pd.notna(summary_row["mae_opportunity_deliverable_mwh"])
                    else np.nan
                )
                summary_mae_gross = (
                    float(summary_row["mae_opportunity_gross_value_eur"])
                    if summary_row is not None and pd.notna(summary_row["mae_opportunity_gross_value_eur"])
                    else np.nan
                )
                summary_priority_rank = (
                    float(summary_row["error_reduction_priority_rank"])
                    if summary_row is not None and pd.notna(summary_row["error_reduction_priority_rank"])
                    else np.nan
                )
                feature_drift_score = (
                    float(drift_row["feature_drift_score"])
                    if drift_row is not None and pd.notna(drift_row.get("feature_drift_score"))
                    else np.nan
                )
                target_drift_score = (
                    float(drift_row["target_drift_score"])
                    if drift_row is not None and pd.notna(drift_row.get("target_drift_score"))
                    else np.nan
                )
                residual_drift_score = (
                    float(drift_row["residual_drift_score"])
                    if drift_row is not None and pd.notna(drift_row.get("residual_drift_score"))
                    else np.nan
                )
                priority_score = _priority_score(
                    blocker_type,
                    actual_volume_mwh=float(candidate.get("actual_volume_mwh", np.nan)),
                    mean_deliverable_abs_error_mwh=float(candidate.get("mean_deliverable_abs_error_mwh", np.nan)),
                    max_deliverable_abs_error_mwh=float(candidate.get("max_deliverable_abs_error_mwh", np.nan)),
                    summary_mae_opportunity_deliverable_mwh=summary_mae_deliv,
                    summary_error_reduction_priority_rank=summary_priority_rank,
                    feature_drift_score=feature_drift_score,
                    target_drift_score=target_drift_score,
                    residual_drift_score=residual_drift_score,
                )
                blocker_summary = (
                    f"{blocker_type}: "
                    f"route={candidate.get('route_name', '<NA>')} "
                    f"cluster={candidate.get('cluster_key', '<NA>')} "
                    f"mean_abs={candidate.get('mean_deliverable_abs_error_mwh', np.nan):.3f} "
                    f"max_abs={candidate.get('max_deliverable_abs_error_mwh', np.nan):.3f} "
                    f"actual={candidate.get('actual_volume_mwh', np.nan):.1f}"
                )
                rows.append(
                    {
                        "window_date": window_date,
                        "model_key": model_key,
                        "forecast_horizon_hours": READINESS_HORIZON_HOURS,
                        "blocker_type": blocker_type,
                        "blocker_scope": BLOCKER_SCOPE.get(blocker_type, "slice"),
                        "slice_dimension": slice_dimension if slice_dimension is not None else pd.NA,
                        "slice_value": slice_value,
                        "blocker_slice_key": _blocker_slice_key(candidate),
                        "route_name": candidate.get("route_name", pd.NA),
                        "cluster_key": candidate.get("cluster_key", pd.NA),
                        "hub_key": candidate.get("hub_key", pd.NA),
                        "internal_transfer_evidence_tier": candidate.get("internal_transfer_evidence_tier", pd.NA),
                        "route_delivery_tier": candidate.get("route_delivery_tier", pd.NA),
                        "error_focus_area": candidate.get("error_focus_area", "general"),
                        "drift_scope": drift_row.get("drift_scope", pd.NA) if drift_row is not None else pd.NA,
                        "drift_state": drift_row.get("drift_state", pd.NA) if drift_row is not None else pd.NA,
                        "eligible_row_count": int(candidate.get("eligible_row_count", 0) or 0),
                        "top_error_row_count": int(candidate.get("top_error_row_count", 0) or 0),
                        "summary_slice_eligible_row_count": summary_eligible_row_count,
                        "actual_volume_mwh": float(candidate.get("actual_volume_mwh", np.nan)),
                        "mean_deliverable_abs_error_mwh": float(candidate.get("mean_deliverable_abs_error_mwh", np.nan)),
                        "max_deliverable_abs_error_mwh": float(candidate.get("max_deliverable_abs_error_mwh", np.nan)),
                        "summary_mae_opportunity_deliverable_mwh": summary_mae_deliv,
                        "summary_mae_opportunity_gross_value_eur": summary_mae_gross,
                        "summary_error_reduction_priority_rank": summary_priority_rank,
                        "feature_drift_score": feature_drift_score,
                        "target_drift_score": target_drift_score,
                        "residual_drift_score": residual_drift_score,
                        "blocker_priority_score": priority_score,
                        "blocker_priority_rank": np.nan,
                        "blocker_summary": blocker_summary,
                        "recommended_next_step": _recommended_next_step(blocker_type),
                        "source_lineage": "fact_model_readiness_daily|fact_backtest_summary_slice|fact_backtest_top_error_hourly|fact_drift_window",
                    }
                )

    blocker = pd.DataFrame(rows, columns=_empty_model_blocker_priority_frame().columns)
    if blocker.empty:
        return blocker

    blocker["blocker_priority_rank"] = blocker.groupby(["window_date", "model_key"], dropna=False)[
        "blocker_priority_score"
    ].rank(method="first", ascending=False)
    return blocker.sort_values(
        ["window_date", "model_key", "blocker_priority_rank", "blocker_type", "blocker_slice_key"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def build_fact_model_candidate_compare_daily(
    fact_backtest_prediction_hourly: pd.DataFrame,
    fact_backtest_summary_slice: pd.DataFrame,
    fact_backtest_top_error_hourly: pd.DataFrame,
    fact_drift_window: pd.DataFrame,
    *,
    baseline_model_key: str = DEFAULT_READINESS_MODEL_KEY,
    candidate_model_key: str = DEFAULT_CANDIDATE_MODEL_KEY,
) -> pd.DataFrame:
    if fact_backtest_prediction_hourly is None or fact_backtest_prediction_hourly.empty:
        return _empty_model_candidate_compare_frame()

    predictions = fact_backtest_prediction_hourly.copy()
    available_model_keys = set(predictions.get("model_key", pd.Series(dtype=object)).dropna())
    if baseline_model_key not in available_model_keys or candidate_model_key not in available_model_keys:
        return _empty_model_candidate_compare_frame()

    predictions["interval_start_utc"] = pd.to_datetime(predictions["interval_start_utc"], utc=True, errors="coerce")
    predictions["window_date"] = predictions["interval_start_utc"].dt.floor("d")
    predictions["forecast_horizon_hours"] = pd.to_numeric(predictions["forecast_horizon_hours"], errors="coerce")
    joined = _candidate_compare_join(
        predictions,
        baseline_model_key=baseline_model_key,
        candidate_model_key=candidate_model_key,
    )
    t1 = predictions[
        predictions["forecast_horizon_hours"].eq(READINESS_HORIZON_HOURS)
        & predictions["model_key"].isin([baseline_model_key, candidate_model_key])
    ].copy()
    if t1.empty:
        return _empty_model_candidate_compare_frame()

    baseline_readiness = build_fact_model_readiness_daily(
        fact_backtest_prediction_hourly=predictions,
        fact_drift_window=fact_drift_window,
        model_key=baseline_model_key,
    )
    candidate_readiness = build_fact_model_readiness_daily(
        fact_backtest_prediction_hourly=predictions,
        fact_drift_window=fact_drift_window,
        model_key=candidate_model_key,
    )
    baseline_blocker = build_fact_model_blocker_priority(
        fact_model_readiness_daily=baseline_readiness,
        fact_backtest_summary_slice=fact_backtest_summary_slice,
        fact_backtest_top_error_hourly=fact_backtest_top_error_hourly,
        fact_drift_window=fact_drift_window,
    )
    candidate_blocker = build_fact_model_blocker_priority(
        fact_model_readiness_daily=candidate_readiness,
        fact_backtest_summary_slice=fact_backtest_summary_slice,
        fact_backtest_top_error_hourly=fact_backtest_top_error_hourly,
        fact_drift_window=fact_drift_window,
    )

    all_dates = sorted(set(t1["window_date"]))
    rows: list[dict[str, object]] = []
    for window_date in all_dates:
        date_joined = joined[joined["window_date"].eq(window_date)].copy()
        gb_nl_joined = date_joined[date_joined["route_name"].eq(TARGET_ROUTE_NAME)].copy()
        reviewed_joined = gb_nl_joined[
            gb_nl_joined["candidate_internal_transfer_evidence_tier"].ne("gb_topology_transfer_gate_proxy")
        ].copy()
        candidate_scope_row_count = int(len(date_joined))

        overall_delta = _candidate_mae_delta(date_joined)
        gb_nl_delta = _candidate_mae_delta(gb_nl_joined)
        reviewed_delta = _candidate_mae_delta(reviewed_joined)

        blocker_delta = np.nan
        severe_delta = np.nan
        if candidate_scope_row_count > 0:
            baseline_blocker_count = int(len(baseline_blocker[baseline_blocker["window_date"].eq(window_date)]))
            candidate_blocker_count = int(len(candidate_blocker[candidate_blocker["window_date"].eq(window_date)]))
            blocker_delta = float(candidate_blocker_count - baseline_blocker_count)

            baseline_row = baseline_readiness[baseline_readiness["window_date"].eq(window_date)].copy()
            candidate_row = candidate_readiness[candidate_readiness["window_date"].eq(window_date)].copy()
            if not baseline_row.empty and not candidate_row.empty:
                baseline_severe = pd.to_numeric(
                    baseline_row["severe_unresolved_focus_area_count_t_plus_1h"],
                    errors="coerce",
                ).iloc[0]
                candidate_severe = pd.to_numeric(
                    candidate_row["severe_unresolved_focus_area_count_t_plus_1h"],
                    errors="coerce",
                ).iloc[0]
                if pd.notna(baseline_severe) and pd.notna(candidate_severe):
                    severe_delta = float(candidate_severe - baseline_severe)

        promotion_state = _promotion_state_from_deltas(
            candidate_scope_row_count=candidate_scope_row_count,
            deltas=[overall_delta, gb_nl_delta, reviewed_delta, blocker_delta, severe_delta],
        )
        rows.append(
            {
                "window_date": window_date,
                "baseline_model_key": baseline_model_key,
                "candidate_model_key": candidate_model_key,
                "overall_t_plus_1h_deliverable_mae_delta_mwh": overall_delta,
                "gb_nl_t_plus_1h_deliverable_mae_delta_mwh": gb_nl_delta,
                "gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh": reviewed_delta,
                "blocker_row_delta": blocker_delta,
                "severe_focus_area_delta": severe_delta,
                "candidate_scope_row_count": candidate_scope_row_count,
                "promotion_state": promotion_state,
                "source_lineage": (
                    "fact_backtest_prediction_hourly|fact_backtest_summary_slice|"
                    "fact_backtest_top_error_hourly|fact_drift_window"
                ),
            }
        )

    compare = pd.DataFrame(rows, columns=_empty_model_candidate_compare_frame().columns)
    return compare.sort_values(["window_date", "baseline_model_key", "candidate_model_key"]).reset_index(drop=True)


def build_fact_model_candidate_compare_window(
    fact_backtest_prediction_hourly: pd.DataFrame,
    fact_backtest_summary_slice: pd.DataFrame,
    fact_backtest_top_error_hourly: pd.DataFrame,
    fact_drift_window: pd.DataFrame,
    *,
    benchmark_suite_name: str,
    benchmark_window_key: str,
    benchmark_window_label: str,
    benchmark_window_start_date: str,
    benchmark_window_end_date: str,
    benchmark_window_family: str,
    benchmark_role: str,
    promotion_window_flag: bool,
    display_order: int = 0,
    baseline_model_key: str = DEFAULT_READINESS_MODEL_KEY,
    candidate_model_key: str = DEFAULT_CANDIDATE_MODEL_KEY,
) -> pd.DataFrame:
    predictions = fact_backtest_prediction_hourly.copy() if fact_backtest_prediction_hourly is not None else pd.DataFrame()
    predictions["interval_start_utc"] = pd.to_datetime(predictions.get("interval_start_utc"), utc=True, errors="coerce")
    predictions["forecast_horizon_hours"] = pd.to_numeric(predictions.get("forecast_horizon_hours"), errors="coerce")

    joined = _candidate_compare_join(
        predictions,
        baseline_model_key=baseline_model_key,
        candidate_model_key=candidate_model_key,
    )
    window_day_count = int(joined["window_date"].nunique()) if "window_date" in joined.columns else 0

    overall_stats = _candidate_compare_slice_stats(joined)
    gb_nl_joined = joined[joined["route_name"].eq(TARGET_ROUTE_NAME)].copy() if not joined.empty else pd.DataFrame()
    gb_nl_stats = _candidate_compare_slice_stats(gb_nl_joined)
    reviewed_joined = (
        gb_nl_joined[gb_nl_joined["candidate_internal_transfer_evidence_tier"].ne("gb_topology_transfer_gate_proxy")].copy()
        if not gb_nl_joined.empty
        else pd.DataFrame()
    )
    reviewed_stats = _candidate_compare_slice_stats(reviewed_joined)

    baseline_readiness = build_fact_model_readiness_daily(
        fact_backtest_prediction_hourly=predictions,
        fact_drift_window=fact_drift_window,
        model_key=baseline_model_key,
    )
    candidate_readiness = build_fact_model_readiness_daily(
        fact_backtest_prediction_hourly=predictions,
        fact_drift_window=fact_drift_window,
        model_key=candidate_model_key,
    )
    baseline_blocker = build_fact_model_blocker_priority(
        fact_model_readiness_daily=baseline_readiness,
        fact_backtest_summary_slice=fact_backtest_summary_slice,
        fact_backtest_top_error_hourly=fact_backtest_top_error_hourly,
        fact_drift_window=fact_drift_window,
    )
    candidate_blocker = build_fact_model_blocker_priority(
        fact_model_readiness_daily=candidate_readiness,
        fact_backtest_summary_slice=fact_backtest_summary_slice,
        fact_backtest_top_error_hourly=fact_backtest_top_error_hourly,
        fact_drift_window=fact_drift_window,
    )

    baseline_blocker_row_count = int(len(baseline_blocker))
    candidate_blocker_row_count = int(len(candidate_blocker))
    baseline_severe_focus_area_count = int(
        round(_safe_numeric_sum(baseline_readiness, "severe_unresolved_focus_area_count_t_plus_1h"))
    )
    candidate_severe_focus_area_count = int(
        round(_safe_numeric_sum(candidate_readiness, "severe_unresolved_focus_area_count_t_plus_1h"))
    )
    blocker_row_delta = float(candidate_blocker_row_count - baseline_blocker_row_count)
    severe_focus_area_delta = float(candidate_severe_focus_area_count - baseline_severe_focus_area_count)
    candidate_scope_row_count = int(overall_stats["row_count"])
    promotion_state = _promotion_state_from_deltas(
        candidate_scope_row_count=candidate_scope_row_count,
        deltas=[
            float(overall_stats["mae_delta_mwh"]) if pd.notna(overall_stats["mae_delta_mwh"]) else np.nan,
            float(gb_nl_stats["mae_delta_mwh"]) if pd.notna(gb_nl_stats["mae_delta_mwh"]) else np.nan,
            float(reviewed_stats["mae_delta_mwh"]) if pd.notna(reviewed_stats["mae_delta_mwh"]) else np.nan,
            blocker_row_delta,
            severe_focus_area_delta,
        ],
    )

    row = {
        "benchmark_suite_name": benchmark_suite_name,
        "benchmark_window_key": benchmark_window_key,
        "benchmark_window_label": benchmark_window_label,
        "benchmark_window_start_date": benchmark_window_start_date,
        "benchmark_window_end_date": benchmark_window_end_date,
        "benchmark_window_family": benchmark_window_family,
        "benchmark_role": benchmark_role,
        "promotion_window_flag": bool(promotion_window_flag),
        "display_order": int(display_order),
        "baseline_model_key": baseline_model_key,
        "candidate_model_key": candidate_model_key,
        "window_day_count": window_day_count,
        "candidate_scope_row_count": candidate_scope_row_count,
        "overall_joined_row_count": int(overall_stats["row_count"]),
        "baseline_overall_deliverable_abs_error_mwh_sum": float(overall_stats["baseline_abs_error_sum"]),
        "candidate_overall_deliverable_abs_error_mwh_sum": float(overall_stats["candidate_abs_error_sum"]),
        "overall_t_plus_1h_deliverable_mae_delta_mwh": overall_stats["mae_delta_mwh"],
        "gb_nl_joined_row_count": int(gb_nl_stats["row_count"]),
        "baseline_gb_nl_deliverable_abs_error_mwh_sum": float(gb_nl_stats["baseline_abs_error_sum"]),
        "candidate_gb_nl_deliverable_abs_error_mwh_sum": float(gb_nl_stats["candidate_abs_error_sum"]),
        "gb_nl_t_plus_1h_deliverable_mae_delta_mwh": gb_nl_stats["mae_delta_mwh"],
        "gb_nl_reviewed_internal_joined_row_count": int(reviewed_stats["row_count"]),
        "baseline_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum": float(
            reviewed_stats["baseline_abs_error_sum"]
        ),
        "candidate_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum": float(
            reviewed_stats["candidate_abs_error_sum"]
        ),
        "gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh": reviewed_stats["mae_delta_mwh"],
        "baseline_blocker_row_count": baseline_blocker_row_count,
        "candidate_blocker_row_count": candidate_blocker_row_count,
        "blocker_row_delta": blocker_row_delta,
        "baseline_severe_focus_area_count": baseline_severe_focus_area_count,
        "candidate_severe_focus_area_count": candidate_severe_focus_area_count,
        "severe_focus_area_delta": severe_focus_area_delta,
        "promotion_state": promotion_state,
        "source_lineage": (
            "fact_backtest_prediction_hourly|fact_backtest_summary_slice|"
            "fact_backtest_top_error_hourly|fact_drift_window"
        ),
    }
    return pd.DataFrame([row], columns=_empty_model_candidate_compare_window_frame().columns)


def build_fact_model_candidate_compare_suite(
    fact_model_candidate_compare_window: pd.DataFrame,
) -> pd.DataFrame:
    if fact_model_candidate_compare_window is None or fact_model_candidate_compare_window.empty:
        return _empty_model_candidate_compare_suite_frame()

    window_compare = fact_model_candidate_compare_window.copy()
    window_compare["promotion_window_flag"] = window_compare["promotion_window_flag"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    group_keys = ["benchmark_suite_name", "baseline_model_key", "candidate_model_key"]

    for (suite_name, baseline_model_key, candidate_model_key), suite_frame in window_compare.groupby(group_keys, dropna=False):
        for suite_scope, subset in (
            ("all_windows", suite_frame.copy()),
            ("promotion_windows", suite_frame[suite_frame["promotion_window_flag"]].copy()),
        ):
            window_count = int(len(subset))
            benchmark_day_count = int(pd.to_numeric(subset.get("window_day_count"), errors="coerce").fillna(0).sum()) if window_count > 0 else 0
            candidate_scope_row_count = int(
                pd.to_numeric(subset.get("candidate_scope_row_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0
            overall_joined_row_count = int(
                pd.to_numeric(subset.get("overall_joined_row_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0
            baseline_overall_sum = _safe_numeric_sum(subset, "baseline_overall_deliverable_abs_error_mwh_sum")
            candidate_overall_sum = _safe_numeric_sum(subset, "candidate_overall_deliverable_abs_error_mwh_sum")
            gb_nl_joined_row_count = int(
                pd.to_numeric(subset.get("gb_nl_joined_row_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0
            baseline_gb_nl_sum = _safe_numeric_sum(subset, "baseline_gb_nl_deliverable_abs_error_mwh_sum")
            candidate_gb_nl_sum = _safe_numeric_sum(subset, "candidate_gb_nl_deliverable_abs_error_mwh_sum")
            reviewed_joined_row_count = int(
                pd.to_numeric(subset.get("gb_nl_reviewed_internal_joined_row_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0
            baseline_reviewed_sum = _safe_numeric_sum(
                subset,
                "baseline_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum",
            )
            candidate_reviewed_sum = _safe_numeric_sum(
                subset,
                "candidate_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum",
            )
            baseline_blocker_row_count = int(
                pd.to_numeric(subset.get("baseline_blocker_row_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0
            candidate_blocker_row_count = int(
                pd.to_numeric(subset.get("candidate_blocker_row_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0
            baseline_severe_focus_area_count = int(
                pd.to_numeric(subset.get("baseline_severe_focus_area_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0
            candidate_severe_focus_area_count = int(
                pd.to_numeric(subset.get("candidate_severe_focus_area_count"), errors="coerce").fillna(0).sum()
            ) if window_count > 0 else 0

            overall_delta = _mae_delta_from_sums(
                baseline_sum=baseline_overall_sum,
                candidate_sum=candidate_overall_sum,
                row_count=overall_joined_row_count,
            )
            gb_nl_delta = _mae_delta_from_sums(
                baseline_sum=baseline_gb_nl_sum,
                candidate_sum=candidate_gb_nl_sum,
                row_count=gb_nl_joined_row_count,
            )
            reviewed_delta = _mae_delta_from_sums(
                baseline_sum=baseline_reviewed_sum,
                candidate_sum=candidate_reviewed_sum,
                row_count=reviewed_joined_row_count,
            )
            blocker_row_delta = float(candidate_blocker_row_count - baseline_blocker_row_count)
            severe_focus_area_delta = float(candidate_severe_focus_area_count - baseline_severe_focus_area_count)
            aggregate_promotion_state = _promotion_state_from_deltas(
                candidate_scope_row_count=candidate_scope_row_count,
                deltas=[overall_delta, gb_nl_delta, reviewed_delta, blocker_row_delta, severe_focus_area_delta],
            )
            candidate_beats_window_count = int(subset["promotion_state"].fillna("").eq("candidate_beats_baseline").sum())
            candidate_mixed_window_count = int(subset["promotion_state"].fillna("").eq("candidate_mixed").sum())
            candidate_regresses_window_count = int(
                subset["promotion_state"].fillna("").eq("candidate_regresses_baseline").sum()
            )
            candidate_insufficient_coverage_window_count = int(
                subset["promotion_state"].fillna("").eq("candidate_insufficient_coverage").sum()
            )
            promotion_state = aggregate_promotion_state
            if aggregate_promotion_state == "candidate_beats_baseline" and (
                candidate_mixed_window_count > 0 or candidate_regresses_window_count > 0
            ):
                promotion_state = "candidate_mixed"
            if aggregate_promotion_state == "candidate_regresses_baseline" and (
                candidate_mixed_window_count > 0 or candidate_beats_window_count > 0
            ):
                promotion_state = "candidate_mixed"

            rows.append(
                {
                    "benchmark_suite_name": suite_name,
                    "suite_scope": suite_scope,
                    "baseline_model_key": baseline_model_key,
                    "candidate_model_key": candidate_model_key,
                    "window_count": window_count,
                    "benchmark_day_count": benchmark_day_count,
                    "candidate_scope_row_count": candidate_scope_row_count,
                    "overall_joined_row_count": overall_joined_row_count,
                    "baseline_overall_deliverable_abs_error_mwh_sum": baseline_overall_sum,
                    "candidate_overall_deliverable_abs_error_mwh_sum": candidate_overall_sum,
                    "overall_t_plus_1h_deliverable_mae_delta_mwh": overall_delta,
                    "gb_nl_joined_row_count": gb_nl_joined_row_count,
                    "baseline_gb_nl_deliverable_abs_error_mwh_sum": baseline_gb_nl_sum,
                    "candidate_gb_nl_deliverable_abs_error_mwh_sum": candidate_gb_nl_sum,
                    "gb_nl_t_plus_1h_deliverable_mae_delta_mwh": gb_nl_delta,
                    "gb_nl_reviewed_internal_joined_row_count": reviewed_joined_row_count,
                    "baseline_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum": baseline_reviewed_sum,
                    "candidate_gb_nl_reviewed_internal_deliverable_abs_error_mwh_sum": candidate_reviewed_sum,
                    "gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh": reviewed_delta,
                    "baseline_blocker_row_count": baseline_blocker_row_count,
                    "candidate_blocker_row_count": candidate_blocker_row_count,
                    "blocker_row_delta": blocker_row_delta,
                    "baseline_severe_focus_area_count": baseline_severe_focus_area_count,
                    "candidate_severe_focus_area_count": candidate_severe_focus_area_count,
                    "severe_focus_area_delta": severe_focus_area_delta,
                    "candidate_beats_window_count": candidate_beats_window_count,
                    "candidate_mixed_window_count": candidate_mixed_window_count,
                    "candidate_regresses_window_count": candidate_regresses_window_count,
                    "candidate_insufficient_coverage_window_count": candidate_insufficient_coverage_window_count,
                    "promotion_state": promotion_state,
                    "source_lineage": "fact_model_candidate_compare_window",
                }
            )

    suite = pd.DataFrame(rows, columns=_empty_model_candidate_compare_suite_frame().columns)
    return suite.sort_values(["benchmark_suite_name", "suite_scope", "baseline_model_key", "candidate_model_key"]).reset_index(
        drop=True
    )


def materialize_model_readiness_daily(
    output_dir: str | Path,
    fact_backtest_prediction_hourly: pd.DataFrame,
    fact_drift_window: pd.DataFrame,
    model_key: str = DEFAULT_READINESS_MODEL_KEY,
) -> Dict[str, pd.DataFrame]:
    readiness = build_fact_model_readiness_daily(
        fact_backtest_prediction_hourly=fact_backtest_prediction_hourly,
        fact_drift_window=fact_drift_window,
        model_key=model_key,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    readiness.to_csv(output_path / f"{MODEL_READINESS_TABLE}.csv", index=False)
    return {MODEL_READINESS_TABLE: readiness}


def materialize_model_readiness_review(
    output_dir: str | Path,
    fact_backtest_prediction_hourly: pd.DataFrame,
    fact_backtest_summary_slice: pd.DataFrame,
    fact_backtest_top_error_hourly: pd.DataFrame,
    fact_drift_window: pd.DataFrame,
    model_key: str = DEFAULT_READINESS_MODEL_KEY,
    baseline_model_key: str = DEFAULT_READINESS_MODEL_KEY,
    candidate_model_key: str = DEFAULT_CANDIDATE_MODEL_KEY,
) -> Dict[str, pd.DataFrame]:
    readiness = build_fact_model_readiness_daily(
        fact_backtest_prediction_hourly=fact_backtest_prediction_hourly,
        fact_drift_window=fact_drift_window,
        model_key=model_key,
    )
    blocker = build_fact_model_blocker_priority(
        fact_model_readiness_daily=readiness,
        fact_backtest_summary_slice=fact_backtest_summary_slice,
        fact_backtest_top_error_hourly=fact_backtest_top_error_hourly,
        fact_drift_window=fact_drift_window,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    readiness.to_csv(output_path / f"{MODEL_READINESS_TABLE}.csv", index=False)
    blocker.to_csv(output_path / f"{MODEL_BLOCKER_PRIORITY_TABLE}.csv", index=False)
    frames = {
        MODEL_READINESS_TABLE: readiness,
        MODEL_BLOCKER_PRIORITY_TABLE: blocker,
    }
    compare = build_fact_model_candidate_compare_daily(
        fact_backtest_prediction_hourly=fact_backtest_prediction_hourly,
        fact_backtest_summary_slice=fact_backtest_summary_slice,
        fact_backtest_top_error_hourly=fact_backtest_top_error_hourly,
        fact_drift_window=fact_drift_window,
        baseline_model_key=baseline_model_key,
        candidate_model_key=candidate_model_key,
    )
    if not compare.empty:
        compare.to_csv(output_path / f"{MODEL_CANDIDATE_COMPARE_TABLE}.csv", index=False)
        frames[MODEL_CANDIDATE_COMPARE_TABLE] = compare
    return frames
