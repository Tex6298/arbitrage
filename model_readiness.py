from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from opportunity_backtest import MODEL_POTENTIAL_RATIO_V2


MODEL_READINESS_TABLE = "fact_model_readiness_daily"
DEFAULT_READINESS_MODEL_KEY = MODEL_POTENTIAL_RATIO_V2
TARGET_ROUTE_NAME = "R2_netback_GB_NL_DE_PL"
TARGET_T1_DELIVERABLE_MAE = 0.50
TARGET_GBNL_T1_DELIVERABLE_MAE = 1.50
TARGET_PROXY_SHARE_MAX = 0.45
TARGET_CAPACITY_UNKNOWN_SHARE_MAX = 0.25
SEVERE_ROUTE_VOLUME_MWH = 25.0
SEVERE_ROUTE_MAE_MWH = 1.50


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
