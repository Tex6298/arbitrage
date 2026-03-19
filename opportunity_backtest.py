from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from curtailment_opportunity import CURTAILMENT_OPPORTUNITY_TABLE


BACKTEST_PREDICTION_TABLE = "fact_backtest_prediction_hourly"
BACKTEST_SUMMARY_SLICE_TABLE = "fact_backtest_summary_slice"
BACKTEST_TOP_ERROR_TABLE = "fact_backtest_top_error_hourly"
DRIFT_WINDOW_TABLE = "fact_drift_window"

MODEL_GROUP_MEAN_NOTICE_V1 = "opportunity_group_mean_notice_v1"
MODEL_POTENTIAL_RATIO_V2 = "opportunity_potential_ratio_v2"
MODEL_GB_NL_REVIEWED_SPECIALIST_V3 = "opportunity_gb_nl_reviewed_specialist_v3"
VALID_BACKTEST_MODEL_KEYS = {
    MODEL_GROUP_MEAN_NOTICE_V1,
    MODEL_POTENTIAL_RATIO_V2,
    MODEL_GB_NL_REVIEWED_SPECIALIST_V3,
}
VALID_BACKTEST_MODEL_SELECTIONS = {"all", *VALID_BACKTEST_MODEL_KEYS}
TARGETED_TRANSITION_ROUTE_NAMES = {"R2_netback_GB_NL_DE_PL"}
TARGETED_OPENING_GUARDRAIL_ROUTE_NAMES = {
    "R1_netback_GB_FR_DE_PL",
    "R2_netback_GB_NL_DE_PL",
}
EVENT_PHASE_CALIBRATION_ROUTE_NAME = "R1_netback_GB_FR_DE_PL"
EVENT_PHASE_CALIBRATION_SOURCE_FAMILY = "day_ahead_constraint_boundary"

DEFAULT_BACKTEST_MODEL_KEY = MODEL_GROUP_MEAN_NOTICE_V1
DEFAULT_BACKTEST_MODEL_SELECTION = "all"
DEFAULT_FORECAST_HORIZON_HOURS = (1, 6, 24, 168)

SPLIT_STRATEGY_GROUP_MEAN = "walk_forward_group_mean"
SPLIT_STRATEGY_POTENTIAL_RATIO = "walk_forward_potential_ratio"
SPLIT_STRATEGY_GB_NL_REVIEWED_SPECIALIST = "walk_forward_gb_nl_reviewed_specialist"

EXACT_MIN_HISTORY = 1
CLUSTER_ROUTE_MIN_HISTORY = 3
ROUTE_STATE_MIN_HISTORY = 6
GLOBAL_MIN_HISTORY = 24

RATIO_EXACT_MIN_HISTORY = 1
RATIO_CLUSTER_ROUTE_MARKET_STATE_MIN_HISTORY = 1
RATIO_ROUTE_MARKET_STATE_MIN_HISTORY = 1
RATIO_ROUTE_MARKET_PATH_MIN_HISTORY = 1
RATIO_CLUSTER_ROUTE_TRANSITION_REGIME_MIN_HISTORY = 1
RATIO_ROUTE_TRANSITION_REGIME_MIN_HISTORY = 1
RATIO_ROUTE_TRANSITION_HOUR_MIN_HISTORY = 1
RATIO_ROUTE_TRANSITION_PATH_MIN_HISTORY = 1
RATIO_ROUTE_NOTICE_MIN_HISTORY = 3
RATIO_ROUTE_TIER_MIN_HISTORY = 6
RATIO_GLOBAL_MIN_HISTORY = 24

FEATURE_DRIFT_WARN_THRESHOLD = 0.15
TARGET_DRIFT_WARN_THRESHOLD = 0.50
RESIDUAL_DRIFT_WARN_THRESHOLD = 0.50
ZERO_ACTIVITY_DRIFT_EPSILON = 1e-6
OPENING_GUARDRAIL_PREDICTION_EPSILON = 1e-6
OPENING_GUARDRAIL_PREOPEN_ORIGIN_HOUR_MAX = 3
OPENING_GUARDRAIL_EXTREME_UPSTREAM_STATES = {
    "day_ahead_much_weaker_than_forward",
    "day_ahead_much_stronger_than_forward",
}
R2_REVIEWED_EVENT_ROUTE_NAME = "R2_netback_GB_NL_DE_PL"
R2_REVIEWED_EVENT_SOURCE_FAMILY = "day_ahead_constraint_boundary"
R2_REVIEWED_EVENT_PREOPEN_ORIGIN_HOUR = 4.0
R2_REVIEWED_EVENT_OPEN_ORIGIN_HOUR = 5.0
R2_REVIEWED_EVENT_CLOSE_ORIGIN_HOUR = 6.0
R2_REVIEWED_EVENT_LATE_REOPEN_ORIGIN_HOUR = 14.0
R2_REVIEWED_EVENT_PREOPEN_UPSTREAM_STATES = {"day_ahead_near_forward"}
R2_REVIEWED_EVENT_LATE_REOPEN_UPSTREAM_STATES = {"day_ahead_weaker_than_forward"}
R2_REVIEWED_EVENT_LATE_REOPEN_CLUSTERS = {
    "dogger_hornsea_offshore",
    "east_anglia_offshore",
    "humber_offshore",
}
R2_REVIEWED_EVENT_LATE_REOPEN_CURTAILMENT_RATIO = 1.05
R2_REVIEWED_EVENT_LATE_REOPEN_CLUSTER_CAP_MWH = {
    "dogger_hornsea_offshore": 170.0,
}
R2_2025_REGIME_WORK_START_UTC = pd.Timestamp("2025-03-01T00:00:00Z")
EVENT_PHASE_CALIBRATION_RATIO_EPSILON = 1e-6
EVENT_PHASE_CALIBRATION_MIN_HISTORY = 3
EVENT_PHASE_CALIBRATION_RATIO_MIN = 0.0
EVENT_PHASE_CALIBRATION_RATIO_MAX = 1.10
PERSIST_CLOSE_SUPPRESSOR_POSITIVE_EPSILON = 1e-6
PERSIST_CLOSE_SUPPRESSOR_CLUSTER_HOUR_MIN_HISTORY = 2
PERSIST_CLOSE_SUPPRESSOR_ROUTE_HOUR_MIN_HISTORY = 3
PERSIST_CLOSE_SUPPRESSOR_CLUSTER_MIN_HISTORY = 3
PERSIST_CLOSE_SUPPRESSOR_MAX_POSITIVE_SHARE = 0.10
REVIEWED_EVENT_TARGET_SHIFT_MIN_REVIEWED_INTERNAL_SHARE = 0.80
REVIEWED_EVENT_TARGET_SHIFT_MAX_PROXY_SHARE = 0.20
REVIEWED_EVENT_TARGET_SHIFT_MAX_CAPACITY_UNKNOWN_SHARE = 0.15
REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_MAE_MWH = 1.0
REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_RATIO = 0.15
REVIEWED_EVENT_TARGET_SHIFT_ACTIVITY_SCALE_FLOOR_MWH = 5.0

ROUTE_PRICE_LOW_POSITIVE_MAX_EUR_PER_MWH = 25.0
ROUTE_PRICE_HIGH_POSITIVE_MIN_EUR_PER_MWH = 75.0
ROUTE_PRICE_SOFT_MOVE_EUR_PER_MWH = 10.0
ROUTE_PRICE_JUMP_MOVE_EUR_PER_MWH = 40.0

SPECIALIST_SCOPE_FORECAST_HORIZON_HOURS = 1
SPECIALIST_SCOPE_ROUTE_NAME = "R2_netback_GB_NL_DE_PL"
SPECIALIST_SCOPE_HUB_KEY = "britned"
SPECIALIST_SCOPE_INTERNAL_TIER = "reviewed_internal_constraint_boundary"
SPECIALIST_RATIO_EPSILON = 1e-6
SPECIALIST_CLASSIFIER_POSITIVE_WEIGHT_CAP = 12.0
SPECIALIST_FLIP_CLASSIFIER_POSITIVE_WEIGHT_CAP = 64.0
SPECIALIST_FLIP_OPENING_GUARDRAIL_PREDICTION_MAX_MWH = 1.0
SPECIALIST_FLIP_OPENING_GUARDRAIL_ORIGIN_HOURS = {4.0, 5.0}
SPECIALIST_FLIP_OPENING_GUARDRAIL_UPSTREAM_STATES = {
    "day_ahead_near_forward",
    "day_ahead_stronger_than_forward",
}
SPECIALIST_FLIP_OPENING_GUARDRAIL_ROUTE_TIERS = {"no_price_signal", "reviewed"}
SPECIALIST_FLIP_OPENING_GUARDRAIL_ROUTE_TRANSITIONS = {
    "price_non_positive->price_non_positive",
    "price_non_positive->price_mid_positive",
}

SPECIALIST_NUMERIC_FEATURE_COLUMNS = (
    "feature_specialist_openable_potential_mwh_asof",
    "feature_specialist_zero_proxy_flag_asof",
    "feature_curtailment_selected_mwh_asof",
    "feature_deliverable_mw_proxy_asof",
    "feature_route_price_score_eur_per_mwh_asof",
    "feature_upstream_forward_price_eur_per_mwh_asof",
    "feature_upstream_day_ahead_price_eur_per_mwh_asof",
    "feature_upstream_intraday_price_eur_per_mwh_asof",
    "feature_route_price_feasible_flag_asof",
    "feature_upstream_market_state_feed_available_flag_asof",
    "feature_system_balance_feed_available_flag_asof",
    "feature_system_balance_known_flag_asof",
    "feature_system_balance_active_flag_asof",
    "feature_connector_capacity_tight_now_flag_asof",
    "feature_market_knew_connector_restriction_flag_asof",
)

SPECIALIST_BOOLEAN_NUMERIC_FEATURE_COLUMNS = {
    "feature_specialist_zero_proxy_flag_asof",
    "feature_route_price_feasible_flag_asof",
    "feature_upstream_market_state_feed_available_flag_asof",
    "feature_system_balance_feed_available_flag_asof",
    "feature_system_balance_known_flag_asof",
    "feature_system_balance_active_flag_asof",
    "feature_connector_capacity_tight_now_flag_asof",
    "feature_market_knew_connector_restriction_flag_asof",
}

SPECIALIST_CATEGORICAL_FEATURE_COLUMNS = (
    "cluster_key",
    "feature_internal_transfer_source_family_asof",
    "feature_internal_transfer_source_key_asof",
    "feature_internal_transfer_boundary_family_asof",
    "feature_connector_itl_source_key_asof",
    "feature_internal_transfer_gate_state_asof",
    "feature_internal_transfer_gate_transition_state_asof",
    "feature_internal_transfer_gate_state_path_asof",
    "feature_route_price_state_asof",
    "feature_route_price_transition_state_asof",
    "feature_route_price_persistence_bucket_asof",
    "feature_upstream_market_state_asof",
    "feature_upstream_day_ahead_to_intraday_spread_bucket_asof",
    "feature_upstream_forward_to_day_ahead_spread_bucket_asof",
    "feature_system_balance_state_asof",
    "feature_system_balance_transition_state_asof",
    "feature_system_balance_persistence_bucket_asof",
    "feature_connector_itl_state_asof",
    "feature_connector_itl_transition_state_asof",
    "feature_connector_notice_market_state_asof",
    "feature_hour_of_day",
    "feature_origin_hour_of_day",
)

SUMMARY_SLICE_DIMENSIONS = (
    "all",
    "cluster_key",
    "hub_key",
    "route_name",
    "route_delivery_tier",
    "internal_transfer_evidence_tier",
    "internal_transfer_gate_state",
    "upstream_market_state",
    "system_balance_state",
    "connector_notice_market_state",
    "curtailment_source_tier",
    "feature_hour_of_day",
)


def _empty_backtest_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "forecast_horizon_hours",
            "forecast_horizon_label",
            "forecast_origin_utc",
            "feature_asof_utc",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "cluster_mapping_confidence",
            "cluster_connection_context",
            "cluster_preferred_hub_candidates",
            "cluster_curation_version",
            "hub_key",
            "hub_label",
            "route_name",
            "route_label",
            "route_border_key",
            "route_delivery_tier",
            "internal_transfer_evidence_tier",
            "internal_transfer_gate_state",
            "internal_transfer_source_family",
            "internal_transfer_source_key",
            "upstream_market_state",
            "system_balance_state",
            "connector_notice_market_state",
            "connector_itl_source_key",
            "curtailment_source_tier",
            "model_key",
            "split_strategy",
            "feature_hour_of_day",
            "feature_day_of_week",
            "feature_origin_hour_of_day",
            "feature_origin_day_of_week",
            "feature_upstream_market_state_feed_available_flag_asof",
            "feature_upstream_market_state_asof",
            "feature_upstream_forward_price_eur_per_mwh_asof",
            "feature_upstream_day_ahead_price_eur_per_mwh_asof",
            "feature_upstream_intraday_price_eur_per_mwh_asof",
            "feature_upstream_day_ahead_to_intraday_spread_bucket_asof",
            "feature_upstream_forward_to_day_ahead_spread_bucket_asof",
            "feature_system_balance_feed_available_flag_asof",
            "feature_system_balance_known_flag_asof",
            "feature_system_balance_active_flag_asof",
            "feature_system_balance_state_asof",
            "feature_system_balance_imbalance_direction_bucket_asof",
            "feature_system_balance_margin_direction_bucket_asof",
            "feature_system_balance_transition_state_asof",
            "feature_system_balance_persistence_bucket_asof",
            "feature_route_price_score_eur_per_mwh_asof",
            "feature_route_price_feasible_flag_asof",
            "feature_route_price_bottleneck_asof",
            "feature_route_price_state_asof",
            "feature_route_price_delta_bucket_asof",
            "feature_route_price_transition_state_asof",
            "feature_route_price_persistence_bucket_asof",
            "feature_route_delivery_tier_asof",
            "feature_connector_notice_market_state_asof",
            "feature_connector_itl_state_asof",
            "feature_connector_itl_source_key_asof",
            "feature_internal_transfer_gate_state_asof",
            "feature_internal_transfer_source_family_asof",
            "feature_internal_transfer_source_key_asof",
            "feature_internal_transfer_boundary_family_asof",
            "feature_internal_transfer_gate_bucket_asof",
            "feature_route_delivery_transition_state_asof",
            "feature_connector_itl_transition_state_asof",
            "feature_internal_transfer_gate_transition_state_asof",
            "feature_route_state_persistence_bucket_asof",
            "feature_connector_itl_state_path_asof",
            "feature_internal_transfer_gate_state_path_asof",
            "feature_curtailment_selected_mwh_asof",
            "feature_deliverable_mw_proxy_asof",
            "feature_specialist_openable_potential_mwh_asof",
            "feature_specialist_zero_proxy_flag_asof",
            "feature_route_score_eur_per_mwh_asof",
            "feature_connector_capacity_tight_now_flag_asof",
            "feature_market_knew_connector_restriction_flag_asof",
            "prediction_basis",
            "training_sample_count",
            "prediction_eligible_flag",
            "actual_opportunity_deliverable_mwh",
            "predicted_opportunity_deliverable_mwh",
            "opportunity_deliverable_residual_mwh",
            "opportunity_deliverable_abs_error_mwh",
            "actual_opportunity_gross_value_eur",
            "predicted_opportunity_gross_value_eur",
            "opportunity_gross_value_residual_eur",
            "opportunity_gross_value_abs_error_eur",
            "source_lineage",
        ]
    )


def _empty_backtest_summary_slice_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_key",
            "forecast_horizon_hours",
            "forecast_horizon_label",
            "slice_dimension",
            "slice_value",
            "error_focus_area",
            "error_reduction_priority_rank",
            "window_start_utc",
            "window_end_utc",
            "row_count",
            "eligible_row_count",
            "prediction_eligibility_rate",
            "actual_opportunity_deliverable_mean_mwh",
            "predicted_opportunity_deliverable_mean_mwh",
            "mae_opportunity_deliverable_mwh",
            "bias_opportunity_deliverable_mwh",
            "actual_opportunity_gross_value_mean_eur",
            "predicted_opportunity_gross_value_mean_eur",
            "mae_opportunity_gross_value_eur",
            "bias_opportunity_gross_value_eur",
            "source_lineage",
        ]
    )


def _empty_backtest_top_error_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_key",
            "forecast_horizon_hours",
            "forecast_horizon_label",
            "top_error_rank",
            "deliverable_abs_error_rank",
            "gross_value_abs_error_rank",
            "error_focus_area",
            "date",
            "forecast_origin_utc",
            "interval_start_utc",
            "interval_end_utc",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "cluster_mapping_confidence",
            "cluster_connection_context",
            "cluster_preferred_hub_candidates",
            "cluster_curation_version",
            "hub_key",
            "hub_label",
            "route_name",
            "route_label",
            "route_border_key",
            "route_delivery_tier",
            "internal_transfer_evidence_tier",
            "internal_transfer_gate_state",
            "upstream_market_state",
            "system_balance_state",
            "connector_notice_market_state",
            "curtailment_source_tier",
            "prediction_basis",
            "training_sample_count",
            "actual_opportunity_deliverable_mwh",
            "predicted_opportunity_deliverable_mwh",
            "opportunity_deliverable_residual_mwh",
            "opportunity_deliverable_abs_error_mwh",
            "actual_opportunity_gross_value_eur",
            "predicted_opportunity_gross_value_eur",
            "opportunity_gross_value_residual_eur",
            "opportunity_gross_value_abs_error_eur",
            "source_lineage",
        ]
    )


def _empty_drift_window_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_key",
            "forecast_horizon_hours",
            "forecast_horizon_label",
            "window_start_utc",
            "window_end_utc",
            "window_date",
            "drift_scope",
            "cluster_key",
            "route_name",
            "row_count",
            "eligible_row_count",
            "reviewed_route_share",
            "capacity_unknown_route_share",
            "reviewed_internal_transfer_share",
            "proxy_internal_transfer_share",
            "blocked_internal_reviewed_share",
            "known_connector_restriction_share",
            "system_balance_stress_share",
            "system_balance_known_share",
            "truth_backed_curtailment_share",
            "actual_opportunity_deliverable_mean_mwh",
            "predicted_opportunity_deliverable_mean_mwh",
            "residual_bias_mwh",
            "residual_mae_mwh",
            "actual_opportunity_gross_value_mean_eur",
            "predicted_opportunity_gross_value_mean_eur",
            "residual_bias_eur",
            "residual_mae_eur",
            "feature_drift_score",
            "target_drift_score",
            "residual_drift_score",
            "drift_state",
            "source_lineage",
        ]
    )


def _prior_mean_by_group(frame: pd.DataFrame, keys: list[str], target_col: str, prefix: str) -> pd.DataFrame:
    working = frame.loc[:, [*keys, "forecast_origin_utc"]].copy()
    merge_keys = [*keys, "forecast_origin_utc"]
    if not keys:
        working["__all_group"] = "__all__"
        merge_keys = ["__all_group", "forecast_origin_utc"]
    working["__target_value"] = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0)

    origin_stats = (
        working.groupby(merge_keys, dropna=False)["__target_value"]
        .agg(origin_row_count="size", origin_target_sum="sum")
        .reset_index()
        .sort_values(merge_keys, kind="mergesort")
        .reset_index(drop=True)
    )
    grouped = origin_stats.groupby(merge_keys[:-1], dropna=False) if len(merge_keys) > 1 else None
    if grouped is not None:
        origin_stats[f"{prefix}_prior_count"] = grouped["origin_row_count"].cumsum() - origin_stats["origin_row_count"]
        origin_stats["_prior_target_sum"] = grouped["origin_target_sum"].cumsum() - origin_stats["origin_target_sum"]
    else:
        origin_stats[f"{prefix}_prior_count"] = 0
        origin_stats["_prior_target_sum"] = 0.0
    origin_stats[f"{prefix}_prior_mean"] = np.where(
        origin_stats[f"{prefix}_prior_count"] > 0,
        origin_stats["_prior_target_sum"] / origin_stats[f"{prefix}_prior_count"],
        np.nan,
    )

    base = frame.loc[:, [*keys, "forecast_origin_utc"]].copy()
    if not keys:
        base["__all_group"] = "__all__"
    base["__row_index"] = frame.index
    merged = (
        base.merge(
            origin_stats[[*merge_keys, f"{prefix}_prior_count", f"{prefix}_prior_mean"]],
            on=merge_keys,
            how="left",
            sort=False,
        )
        .set_index("__row_index")
        .reindex(frame.index)
    )
    result = pd.DataFrame(index=frame.index)
    result[f"{prefix}_prior_count"] = pd.to_numeric(
        merged[f"{prefix}_prior_count"], errors="coerce"
    ).fillna(0).astype(int)
    result[f"{prefix}_prior_mean"] = pd.to_numeric(merged[f"{prefix}_prior_mean"], errors="coerce")
    return result


def _coerce_bool_series(values: pd.Series | object, default: bool = False) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)
    if series.dtype == bool:
        return series.fillna(default)
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "nan": np.nan,
            "<na>": np.nan,
            "none": np.nan,
        }
    )
    if mapped.notna().any():
        return mapped.where(mapped.notna(), default).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(int(default)).astype(int).astype(bool)
    return series.where(series.notna(), default).astype(bool)


def _append_prediction_basis_suffix(basis: pd.Series, suffix: str) -> pd.Series:
    basis_text = basis.astype("string").fillna("")
    return pd.Series(
        np.where(basis_text.eq(""), suffix, basis_text + "_" + suffix),
        index=basis.index,
        dtype="string",
    )


def _build_prior_group_key(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    if not keys:
        return pd.Series("__all__", index=frame.index, dtype="string")
    key_parts = frame.loc[:, keys].astype("string").fillna("<NA>")
    return key_parts.agg("|".join, axis=1).astype("string")


def _prior_mean_from_history_by_group(
    target_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    keys: list[str],
    target_col: str,
    prefix: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=target_frame.index)
    result[f"{prefix}_prior_count"] = 0
    result[f"{prefix}_prior_mean"] = np.nan
    if target_frame.empty:
        return result
    if history_frame is None or history_frame.empty:
        return result

    history = history_frame.loc[:, [*keys, "forecast_origin_utc", target_col]].copy()
    history["forecast_origin_utc"] = pd.to_datetime(history["forecast_origin_utc"], utc=True, errors="coerce")
    history[target_col] = pd.to_numeric(history[target_col], errors="coerce")
    history = history[history["forecast_origin_utc"].notna() & history[target_col].notna()].copy()
    if history.empty:
        return result

    history["__group_key"] = _build_prior_group_key(history, keys)
    origin_stats = (
        history.groupby(["__group_key", "forecast_origin_utc"], dropna=False)[target_col]
        .agg(origin_row_count="size", origin_target_sum="sum")
        .reset_index()
        .sort_values(["__group_key", "forecast_origin_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    origin_stats[f"{prefix}_prior_count"] = origin_stats.groupby("__group_key", dropna=False)["origin_row_count"].cumsum()
    origin_stats["_prior_target_sum"] = origin_stats.groupby("__group_key", dropna=False)["origin_target_sum"].cumsum()

    target = target_frame.loc[:, [*keys, "forecast_origin_utc"]].copy()
    target["forecast_origin_utc"] = pd.to_datetime(target["forecast_origin_utc"], utc=True, errors="coerce")
    target["__group_key"] = _build_prior_group_key(target, keys)
    target["__row_index"] = target_frame.index
    valid_target = target[target["forecast_origin_utc"].notna()].copy()
    if valid_target.empty:
        return result

    merged_groups = []
    merge_columns = [
        "__group_key",
        "forecast_origin_utc",
        f"{prefix}_prior_count",
        "_prior_target_sum",
    ]
    for group_key, target_group in valid_target.groupby("__group_key", dropna=False, sort=False):
        sorted_target_group = target_group.sort_values("forecast_origin_utc", kind="mergesort")
        history_group = origin_stats[origin_stats["__group_key"].eq(group_key)].copy()
        if history_group.empty:
            sorted_target_group[f"{prefix}_prior_count"] = 0
            sorted_target_group["_prior_target_sum"] = np.nan
            merged_groups.append(sorted_target_group)
            continue
        history_group = history_group.loc[:, merge_columns].sort_values("forecast_origin_utc", kind="mergesort")
        merged_group = pd.merge_asof(
            sorted_target_group,
            history_group,
            on="forecast_origin_utc",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_groups.append(merged_group)
    merged = pd.concat(merged_groups, ignore_index=False).set_index("__row_index").reindex(target_frame.index)
    prior_count = pd.to_numeric(merged[f"{prefix}_prior_count"], errors="coerce").fillna(0).astype(int)
    prior_target_sum = pd.to_numeric(merged["_prior_target_sum"], errors="coerce")

    result[f"{prefix}_prior_count"] = prior_count
    result[f"{prefix}_prior_mean"] = np.where(
        prior_count > 0,
        prior_target_sum / prior_count,
        np.nan,
    )
    return result


def _derive_internal_transfer_boundary_family(
    internal_transfer_source_family: pd.Series,
    internal_transfer_source_key: pd.Series,
    internal_transfer_evidence_tier: pd.Series,
) -> pd.Series:
    family = internal_transfer_source_family.astype("string")
    source_key = internal_transfer_source_key.astype("string")
    evidence_tier = internal_transfer_evidence_tier.astype("string")
    derived = family.copy()
    derived = derived.where(
        ~source_key.fillna("").str.startswith("fact_day_ahead_constraint_boundary_half_hourly:"),
        "day_ahead_constraint_boundary",
    )
    derived = derived.where(
        ~(
            evidence_tier.eq(SPECIALIST_SCOPE_INTERNAL_TIER)
            & derived.isna()
        ),
        "reviewed_internal_constraint_boundary",
    )
    return derived.astype(object)


def _specialist_scope_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["forecast_horizon_hours"], errors="coerce").eq(SPECIALIST_SCOPE_FORECAST_HORIZON_HOURS)
        & frame["route_name"].eq(SPECIALIST_SCOPE_ROUTE_NAME)
        & frame["hub_key"].eq(SPECIALIST_SCOPE_HUB_KEY)
        & frame["internal_transfer_evidence_tier"].eq(SPECIALIST_SCOPE_INTERNAL_TIER)
    )


def _compute_specialist_openable_potential(
    curtailment_selected_mwh: pd.Series,
    deliverable_mw_proxy: pd.Series,
) -> pd.Series:
    curtailment_selected = pd.to_numeric(curtailment_selected_mwh, errors="coerce").fillna(0.0).clip(lower=0.0)
    deliverable_proxy = pd.to_numeric(deliverable_mw_proxy, errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(
        np.where(
            deliverable_proxy.gt(SPECIALIST_RATIO_EPSILON),
            np.minimum(curtailment_selected, deliverable_proxy),
            curtailment_selected,
        ),
        index=curtailment_selected.index,
        dtype=float,
    )


def _specialist_flip_focus_mask(frame: pd.DataFrame) -> pd.Series:
    route_delivery_tier = frame.get(
        "feature_route_delivery_tier_asof",
        pd.Series(pd.NA, index=frame.index),
    ).astype("string")
    gate_state = frame.get(
        "feature_internal_transfer_gate_state_asof",
        pd.Series(pd.NA, index=frame.index),
    ).fillna("").astype(str)
    connector_notice = frame.get(
        "feature_connector_notice_market_state_asof",
        pd.Series(pd.NA, index=frame.index),
    ).astype("string")
    connector_itl_state = frame.get(
        "feature_connector_itl_state_asof",
        pd.Series(pd.NA, index=frame.index),
    ).astype("string")
    openable_potential = pd.to_numeric(
        frame.get("feature_specialist_openable_potential_mwh_asof", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    return (
        route_delivery_tier.isin(["no_price_signal", "reviewed"])
        & ~gate_state.str.startswith("blocked_")
        & connector_notice.eq("no_public_connector_restriction")
        & connector_itl_state.isin(["published_restriction", "blocked_zero_or_negative_itl"])
        & openable_potential.gt(0.0)
    )


def _specialist_flip_suppressed_mask(frame: pd.DataFrame) -> pd.Series:
    connector_itl_state = frame.get(
        "feature_connector_itl_state_asof",
        pd.Series(pd.NA, index=frame.index),
    ).astype("string")
    upstream_market_state = frame.get(
        "feature_upstream_market_state_asof",
        pd.Series(pd.NA, index=frame.index),
    ).astype("string")
    route_delivery_tier = frame.get(
        "feature_route_delivery_tier_asof",
        pd.Series(pd.NA, index=frame.index),
    ).astype("string")
    gate_state = frame.get(
        "feature_internal_transfer_gate_state_asof",
        pd.Series(pd.NA, index=frame.index),
    ).fillna("").astype(str)
    return (
        connector_itl_state.eq("blocked_zero_or_negative_itl")
        & upstream_market_state.eq("day_ahead_weaker_than_forward")
        & route_delivery_tier.eq("no_price_signal")
        & ~gate_state.str.startswith("blocked_")
    )


def _apply_gb_nl_specialist_flip_opening_guardrail(frame: pd.DataFrame) -> pd.DataFrame:
    adjusted = frame.copy()
    prediction_basis = adjusted.get(
        "prediction_basis",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    predicted = pd.to_numeric(
        adjusted.get("predicted_opportunity_deliverable_mwh"),
        errors="coerce",
    ).fillna(0.0)
    openable_potential = pd.to_numeric(
        adjusted.get("feature_specialist_openable_potential_mwh_asof"),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    origin_hour = pd.to_numeric(
        adjusted.get("feature_origin_hour_of_day"),
        errors="coerce",
    )
    connector_itl_state = adjusted.get(
        "feature_connector_itl_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).astype("string")
    upstream_market_state = adjusted.get(
        "feature_upstream_market_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).astype("string")
    route_delivery_tier = adjusted.get(
        "feature_route_delivery_tier_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).astype("string")
    route_price_transition = adjusted.get(
        "feature_route_price_transition_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).astype("string")
    gate_state = adjusted.get(
        "feature_internal_transfer_gate_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    mask = (
        prediction_basis.str.contains("flip_open_specialist", regex=False)
        & predicted.le(SPECIALIST_FLIP_OPENING_GUARDRAIL_PREDICTION_MAX_MWH)
        & openable_potential.gt(0.0)
        & origin_hour.isin(SPECIALIST_FLIP_OPENING_GUARDRAIL_ORIGIN_HOURS)
        & connector_itl_state.eq("published_restriction")
        & upstream_market_state.isin(SPECIALIST_FLIP_OPENING_GUARDRAIL_UPSTREAM_STATES)
        & route_delivery_tier.isin(SPECIALIST_FLIP_OPENING_GUARDRAIL_ROUTE_TIERS)
        & route_price_transition.isin(SPECIALIST_FLIP_OPENING_GUARDRAIL_ROUTE_TRANSITIONS)
        & ~gate_state.str.startswith("blocked_")
    )
    if mask.any():
        adjusted.loc[mask, "predicted_opportunity_deliverable_mwh"] = openable_potential.loc[mask]
        adjusted.loc[mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[mask, "prediction_basis"],
            "flip_opening_guardrail",
        ).values
    return adjusted


def _make_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_specialist_pipeline(kind: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                list(SPECIALIST_NUMERIC_FEATURE_COLUMNS),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", _make_one_hot_encoder()),
                    ]
                ),
                list(SPECIALIST_CATEGORICAL_FEATURE_COLUMNS),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    if kind == "classifier":
        estimator = HistGradientBoostingClassifier(random_state=42)
    elif kind == "regressor":
        estimator = HistGradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(f"unsupported specialist pipeline kind '{kind}'")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )


def _prepare_specialist_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    for column in SPECIALIST_NUMERIC_FEATURE_COLUMNS:
        values = features.get(column, pd.Series(np.nan, index=features.index))
        if column in SPECIALIST_BOOLEAN_NUMERIC_FEATURE_COLUMNS:
            features[column] = _coerce_bool_series(values, default=False).astype(float)
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            if not numeric.notna().any():
                numeric = pd.Series(0.0, index=features.index, dtype=float)
            features[column] = numeric
    for column in SPECIALIST_CATEGORICAL_FEATURE_COLUMNS:
        values = features.get(column, pd.Series(pd.NA, index=features.index))
        categorical = pd.Series("missing", index=features.index, dtype=object)
        non_missing = values.notna()
        categorical.loc[non_missing] = values.loc[non_missing].astype(str)
        features[column] = categorical
    return features[[*SPECIALIST_NUMERIC_FEATURE_COLUMNS, *SPECIALIST_CATEGORICAL_FEATURE_COLUMNS]]


def load_curtailment_opportunity_input(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.is_dir():
        input_path = input_path / f"{CURTAILMENT_OPPORTUNITY_TABLE}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"curtailment opportunity input does not exist: {input_path}")
    frame = pd.read_csv(input_path)
    for column in ("interval_start_utc", "interval_end_utc", "interval_start_local", "interval_end_local"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=column.endswith("_utc"))
    return frame


def coerce_forecast_horizons(value: str | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_FORECAST_HORIZON_HOURS
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        horizons = tuple(int(token) for token in tokens)
    else:
        horizons = tuple(int(token) for token in value)
    if not horizons:
        raise ValueError("forecast horizons must not be empty")
    if any(token <= 0 for token in horizons):
        raise ValueError("forecast horizons must be positive integers")
    return tuple(dict.fromkeys(horizons))


def _prepare_backtest_input(fact_curtailment_opportunity_hourly: pd.DataFrame) -> pd.DataFrame:
    frame = fact_curtailment_opportunity_hourly.copy()
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["interval_end_utc"] = pd.to_datetime(frame["interval_end_utc"], utc=True, errors="coerce")
    frame["interval_start_local"] = pd.to_datetime(frame["interval_start_local"], errors="coerce")
    frame["interval_end_local"] = pd.to_datetime(frame["interval_end_local"], errors="coerce")
    frame["actual_opportunity_deliverable_mwh"] = pd.to_numeric(
        frame["opportunity_deliverable_mwh"], errors="coerce"
    ).fillna(0.0)
    frame["actual_opportunity_gross_value_eur"] = pd.to_numeric(
        frame["opportunity_gross_value_eur"], errors="coerce"
    ).fillna(0.0)
    frame["upstream_market_state_feed_available_flag"] = frame.get(
        "upstream_market_state_feed_available_flag",
        pd.Series(False, index=frame.index),
    )
    frame["upstream_market_state_feed_available_flag"] = _coerce_bool_series(
        frame["upstream_market_state_feed_available_flag"]
    )
    frame["upstream_market_state"] = frame.get(
        "upstream_market_state",
        pd.Series("no_upstream_feed", index=frame.index),
    ).fillna("no_upstream_feed")
    frame["system_balance_feed_available_flag"] = frame.get(
        "system_balance_feed_available_flag",
        pd.Series(False, index=frame.index),
    )
    frame["system_balance_feed_available_flag"] = _coerce_bool_series(
        frame["system_balance_feed_available_flag"]
    )
    frame["system_balance_known_flag"] = frame.get(
        "system_balance_known_flag",
        pd.Series(False, index=frame.index),
    )
    frame["system_balance_known_flag"] = _coerce_bool_series(frame["system_balance_known_flag"])
    frame["system_balance_active_flag"] = frame.get(
        "system_balance_active_flag",
        pd.Series(False, index=frame.index),
    )
    frame["system_balance_active_flag"] = _coerce_bool_series(frame["system_balance_active_flag"])
    frame["system_balance_state"] = frame.get(
        "system_balance_state",
        pd.Series("no_public_system_balance", index=frame.index),
    ).fillna("no_public_system_balance")
    frame["system_balance_imbalance_direction_bucket"] = frame.get(
        "system_balance_imbalance_direction_bucket",
        pd.Series("imbalance_unknown", index=frame.index),
    ).fillna("imbalance_unknown")
    frame["system_balance_margin_direction_bucket"] = frame.get(
        "system_balance_margin_direction_bucket",
        pd.Series("margin_unknown", index=frame.index),
    ).fillna("margin_unknown")
    for column in (
        "upstream_forward_price_eur_per_mwh",
        "upstream_day_ahead_price_eur_per_mwh",
        "upstream_intraday_price_eur_per_mwh",
    ):
        frame[column] = pd.to_numeric(
            frame.get(column, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
    frame["upstream_day_ahead_to_intraday_spread_bucket"] = frame.get(
        "upstream_day_ahead_to_intraday_spread_bucket",
        pd.Series("spread_unknown", index=frame.index),
    ).fillna("spread_unknown")
    frame["upstream_forward_to_day_ahead_spread_bucket"] = frame.get(
        "upstream_forward_to_day_ahead_spread_bucket",
        pd.Series("spread_unknown", index=frame.index),
    ).fillna("spread_unknown")
    frame["route_price_score_eur_per_mwh"] = pd.to_numeric(
        frame.get(
            "route_price_score_eur_per_mwh",
            frame.get("deliverable_route_score_eur_per_mwh", pd.Series(0.0, index=frame.index)),
        ),
        errors="coerce",
    ).fillna(0.0)
    route_price_feasible_default = frame["route_price_score_eur_per_mwh"].gt(0.0)
    frame["route_price_feasible_flag"] = frame.get(
        "route_price_feasible_flag",
        route_price_feasible_default,
    )
    frame["route_price_feasible_flag"] = _coerce_bool_series(frame["route_price_feasible_flag"])
    frame["route_price_bottleneck"] = frame.get(
        "route_price_bottleneck",
        pd.Series("unknown_price_bottleneck", index=frame.index),
    ).fillna("unknown_price_bottleneck")
    frame["deliverable_route_score_eur_per_mwh"] = pd.to_numeric(
        frame["deliverable_route_score_eur_per_mwh"], errors="coerce"
    ).fillna(0.0)
    frame["curtailment_selected_mwh"] = pd.to_numeric(
        frame.get("curtailment_selected_mwh", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    frame["deliverable_mw_proxy"] = pd.to_numeric(
        frame.get("deliverable_mw_proxy", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    frame["connector_capacity_tight_now_flag"] = (
        pd.to_numeric(
            frame.get("connector_capacity_tight_now_flag", pd.Series(0, index=frame.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
    )
    frame["market_knew_connector_restriction_flag"] = (
        pd.to_numeric(
            frame.get("market_knew_connector_restriction_flag", pd.Series(0, index=frame.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
    )
    frame["feature_hour_of_day"] = frame["interval_start_utc"].dt.hour
    frame["feature_day_of_week"] = frame["interval_start_utc"].dt.dayofweek
    frame["cluster_mapping_confidence"] = frame.get(
        "cluster_mapping_confidence",
        pd.Series("unknown", index=frame.index),
    ).fillna("unknown")
    frame["cluster_connection_context"] = frame.get(
        "cluster_connection_context",
        pd.Series(pd.NA, index=frame.index),
    )
    frame["cluster_preferred_hub_candidates"] = frame.get(
        "cluster_preferred_hub_candidates",
        pd.Series(pd.NA, index=frame.index),
    )
    frame["cluster_curation_version"] = frame.get(
        "cluster_curation_version",
        pd.Series(pd.NA, index=frame.index),
    )
    frame["route_delivery_tier"] = frame["route_delivery_tier"].fillna("unknown")
    frame["internal_transfer_evidence_tier"] = frame.get(
        "internal_transfer_evidence_tier",
        pd.Series("gb_topology_transfer_gate_proxy", index=frame.index),
    ).fillna("gb_topology_transfer_gate_proxy")
    frame["internal_transfer_source_family"] = frame.get(
        "internal_transfer_source_family",
        pd.Series(pd.NA, index=frame.index),
    )
    frame["internal_transfer_source_key"] = frame.get(
        "internal_transfer_source_key",
        pd.Series(pd.NA, index=frame.index),
    )
    frame["internal_transfer_gate_state"] = frame.get(
        "internal_transfer_gate_state",
        pd.Series("capacity_unknown_reachable", index=frame.index),
    ).fillna("capacity_unknown_reachable")
    frame["connector_itl_state"] = frame.get(
        "connector_itl_state",
        pd.Series("no_public_itl_restriction", index=frame.index),
    ).fillna("no_public_itl_restriction")
    frame["connector_itl_source_key"] = frame.get(
        "connector_itl_source_key",
        pd.Series(pd.NA, index=frame.index),
    )
    frame["connector_notice_market_state"] = frame["connector_notice_market_state"].fillna(
        "no_public_connector_restriction"
    )
    frame["curtailment_source_tier"] = frame["curtailment_source_tier"].fillna("unknown")
    source_family = frame["internal_transfer_source_family"].astype("string")
    source_family = source_family.where(source_family.fillna("").str.strip().ne(""), pd.NA)
    frame["internal_transfer_source_family"] = source_family
    frame["internal_transfer_boundary_family"] = _derive_internal_transfer_boundary_family(
        frame["internal_transfer_source_family"],
        frame["internal_transfer_source_key"],
        frame["internal_transfer_evidence_tier"],
    )
    frame["internal_transfer_source_family"] = frame["internal_transfer_source_family"].where(
        frame["internal_transfer_source_family"].notna(),
        frame["internal_transfer_boundary_family"],
    )
    internal_gate_state = frame["internal_transfer_gate_state"].astype(str)
    frame["internal_transfer_gate_bucket"] = np.where(
        internal_gate_state.str.startswith("blocked_reviewed"),
        "blocked_reviewed",
        np.where(
            internal_gate_state.eq("blocked_upstream_dependency"),
            "blocked_upstream_dependency",
            "nonblocking_transfer",
        ),
    )
    frame["potential_opportunity_mwh"] = np.minimum(
        frame["curtailment_selected_mwh"].clip(lower=0.0),
        frame["deliverable_mw_proxy"].clip(lower=0.0),
    )
    potential_nonzero = frame["potential_opportunity_mwh"].gt(0.0)
    frame["realized_potential_ratio"] = np.where(
        potential_nonzero,
        frame["actual_opportunity_deliverable_mwh"] / frame["potential_opportunity_mwh"],
        0.0,
    )
    frame["realized_potential_ratio"] = pd.to_numeric(
        frame["realized_potential_ratio"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    frame["route_price_state"] = np.where(
        frame["route_price_score_eur_per_mwh"].le(0.0),
        "price_non_positive",
        np.where(
            frame["route_price_score_eur_per_mwh"].lt(ROUTE_PRICE_LOW_POSITIVE_MAX_EUR_PER_MWH),
            "price_low_positive",
            np.where(
                frame["route_price_score_eur_per_mwh"].lt(ROUTE_PRICE_HIGH_POSITIVE_MIN_EUR_PER_MWH),
                "price_mid_positive",
                "price_high_positive",
            ),
        ),
    )

    frame = frame.sort_values(["cluster_key", "route_name", "hub_key", "interval_start_utc"]).reset_index(drop=True)
    group_keys = ["cluster_key", "route_name", "hub_key"]
    grouped = frame.groupby(group_keys, dropna=False)
    prev_interval_start = grouped["interval_start_utc"].shift(1)
    prev2_interval_start = grouped["interval_start_utc"].shift(2)
    prev_route_tier = grouped["route_delivery_tier"].shift(1)
    prev_connector_itl = grouped["connector_itl_state"].shift(1)
    prev_internal_gate = grouped["internal_transfer_gate_state"].shift(1)
    prev2_connector_itl = grouped["connector_itl_state"].shift(2)
    prev2_internal_gate = grouped["internal_transfer_gate_state"].shift(2)
    prev_route_price_state = grouped["route_price_state"].shift(1)
    prev_route_price_score = grouped["route_price_score_eur_per_mwh"].shift(1)
    prev_system_balance_state = grouped["system_balance_state"].shift(1)

    contiguous_prev = prev_interval_start.eq(frame["interval_start_utc"] - pd.Timedelta(hours=1))
    contiguous_prev2 = prev2_interval_start.eq(frame["interval_start_utc"] - pd.Timedelta(hours=2))
    prev_route_tier = prev_route_tier.where(contiguous_prev)
    prev_connector_itl = prev_connector_itl.where(contiguous_prev)
    prev_internal_gate = prev_internal_gate.where(contiguous_prev)
    prev2_connector_itl = prev2_connector_itl.where(contiguous_prev & contiguous_prev2)
    prev2_internal_gate = prev2_internal_gate.where(contiguous_prev & contiguous_prev2)
    prev_route_price_state = prev_route_price_state.where(contiguous_prev)
    prev_route_price_score = prev_route_price_score.where(contiguous_prev)
    prev_system_balance_state = prev_system_balance_state.where(contiguous_prev)

    route_price_delta = frame["route_price_score_eur_per_mwh"] - prev_route_price_score
    frame["route_price_delta_bucket"] = np.where(
        ~contiguous_prev,
        "price_no_prior",
        np.where(
            route_price_delta.le(-ROUTE_PRICE_JUMP_MOVE_EUR_PER_MWH),
            "price_jump_down",
            np.where(
                route_price_delta.le(-ROUTE_PRICE_SOFT_MOVE_EUR_PER_MWH),
                "price_soft_down",
                np.where(
                    route_price_delta.lt(ROUTE_PRICE_SOFT_MOVE_EUR_PER_MWH),
                    "price_flat",
                    np.where(
                        route_price_delta.lt(ROUTE_PRICE_JUMP_MOVE_EUR_PER_MWH),
                        "price_soft_up",
                        "price_jump_up",
                    ),
                ),
            ),
        ),
    )

    def _transition_state(previous: pd.Series, current: pd.Series) -> pd.Series:
        prev_label = previous.fillna("START").astype(str)
        curr_label = current.fillna("unknown").astype(str)
        return prev_label + "->" + curr_label

    def _state_path(prev2: pd.Series, prev1: pd.Series, current: pd.Series) -> pd.Series:
        prev2_label = prev2.fillna("START").astype(str)
        prev1_label = prev1.fillna("START").astype(str)
        curr_label = current.fillna("unknown").astype(str)
        return prev2_label + "|" + prev1_label + "|" + curr_label

    frame["route_delivery_transition_state"] = _transition_state(prev_route_tier, frame["route_delivery_tier"])
    frame["connector_itl_transition_state"] = _transition_state(prev_connector_itl, frame["connector_itl_state"])
    frame["internal_transfer_gate_transition_state"] = _transition_state(
        prev_internal_gate, frame["internal_transfer_gate_state"]
    )
    frame["system_balance_transition_state"] = _transition_state(
        prev_system_balance_state,
        frame["system_balance_state"],
    )
    frame["route_price_transition_state"] = _transition_state(prev_route_price_state, frame["route_price_state"])
    frame["connector_itl_state_path"] = _state_path(prev2_connector_itl, prev_connector_itl, frame["connector_itl_state"])
    frame["internal_transfer_gate_state_path"] = _state_path(
        prev2_internal_gate,
        prev_internal_gate,
        frame["internal_transfer_gate_state"],
    )

    route_run_break = (~contiguous_prev) | prev_route_tier.ne(frame["route_delivery_tier"])
    route_run_id = route_run_break.groupby([frame[key] for key in group_keys], dropna=False).cumsum()
    frame["route_state_persistence_hours"] = (
        frame.groupby([*group_keys, route_run_id], dropna=False).cumcount() + 1
    ).astype(int)
    frame["route_state_persistence_bucket"] = np.where(
        frame["route_state_persistence_hours"].le(1),
        "persist_1h",
        np.where(frame["route_state_persistence_hours"].eq(2), "persist_2h", "persist_3h_plus"),
    )
    system_balance_run_break = (~contiguous_prev) | prev_system_balance_state.ne(frame["system_balance_state"])
    system_balance_run_id = system_balance_run_break.groupby([frame[key] for key in group_keys], dropna=False).cumsum()
    frame["system_balance_persistence_hours"] = (
        frame.groupby([*group_keys, system_balance_run_id], dropna=False).cumcount() + 1
    ).astype(int)
    frame["system_balance_persistence_bucket"] = np.where(
        frame["system_balance_persistence_hours"].le(1),
        "system_balance_persist_1h",
        np.where(
            frame["system_balance_persistence_hours"].eq(2),
            "system_balance_persist_2h",
            "system_balance_persist_3h_plus",
        ),
    )
    route_price_run_break = (~contiguous_prev) | prev_route_price_state.ne(frame["route_price_state"])
    route_price_run_id = route_price_run_break.groupby([frame[key] for key in group_keys], dropna=False).cumsum()
    frame["route_price_persistence_hours"] = (
        frame.groupby([*group_keys, route_price_run_id], dropna=False).cumcount() + 1
    ).astype(int)
    frame["route_price_persistence_bucket"] = np.where(
        frame["route_price_persistence_hours"].le(1),
        "price_persist_1h",
        np.where(frame["route_price_persistence_hours"].eq(2), "price_persist_2h", "price_persist_3h_plus"),
    )
    return frame.sort_values(["interval_start_utc", "cluster_key", "route_name", "hub_key"]).reset_index(drop=True)


def _build_horizon_example_frame(
    frame: pd.DataFrame,
    forecast_horizon_hours: int,
    *,
    origin_source_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target = frame.copy()
    if origin_source_frame is None:
        origin_source_frame = frame
    target["forecast_horizon_hours"] = forecast_horizon_hours
    target["forecast_horizon_label"] = f"t_plus_{forecast_horizon_hours}h"
    target["forecast_origin_utc"] = target["interval_start_utc"] - pd.Timedelta(hours=forecast_horizon_hours)
    target["feature_asof_utc"] = target["forecast_origin_utc"]

    origin = origin_source_frame[
        [
            "cluster_key",
            "route_name",
            "hub_key",
            "interval_start_utc",
            "feature_hour_of_day",
            "feature_day_of_week",
            "cluster_mapping_confidence",
            "cluster_connection_context",
            "cluster_preferred_hub_candidates",
            "cluster_curation_version",
            "upstream_market_state_feed_available_flag",
            "upstream_market_state",
            "upstream_forward_price_eur_per_mwh",
            "upstream_day_ahead_price_eur_per_mwh",
            "upstream_intraday_price_eur_per_mwh",
            "upstream_day_ahead_to_intraday_spread_bucket",
            "upstream_forward_to_day_ahead_spread_bucket",
            "system_balance_feed_available_flag",
            "system_balance_known_flag",
            "system_balance_active_flag",
            "system_balance_state",
            "system_balance_imbalance_direction_bucket",
            "system_balance_margin_direction_bucket",
            "system_balance_transition_state",
            "system_balance_persistence_bucket",
            "route_price_score_eur_per_mwh",
            "route_price_feasible_flag",
            "route_price_bottleneck",
            "route_price_state",
            "route_price_delta_bucket",
            "route_price_transition_state",
            "route_price_persistence_bucket",
            "route_delivery_tier",
            "connector_notice_market_state",
            "connector_itl_state",
            "connector_itl_source_key",
            "internal_transfer_gate_state",
            "internal_transfer_source_family",
            "internal_transfer_source_key",
            "internal_transfer_boundary_family",
            "internal_transfer_gate_bucket",
            "route_delivery_transition_state",
            "connector_itl_transition_state",
            "internal_transfer_gate_transition_state",
            "route_state_persistence_bucket",
            "connector_itl_state_path",
            "internal_transfer_gate_state_path",
            "curtailment_selected_mwh",
            "deliverable_mw_proxy",
            "deliverable_route_score_eur_per_mwh",
            "connector_capacity_tight_now_flag",
            "market_knew_connector_restriction_flag",
        ]
    ].copy()
    origin = origin.rename(
        columns={
            "interval_start_utc": "feature_asof_utc",
            "feature_hour_of_day": "feature_origin_hour_of_day",
            "feature_day_of_week": "feature_origin_day_of_week",
            "cluster_mapping_confidence": "cluster_mapping_confidence",
            "cluster_connection_context": "cluster_connection_context",
            "cluster_preferred_hub_candidates": "cluster_preferred_hub_candidates",
            "cluster_curation_version": "cluster_curation_version",
            "upstream_market_state_feed_available_flag": "feature_upstream_market_state_feed_available_flag_asof",
            "upstream_market_state": "feature_upstream_market_state_asof",
            "upstream_forward_price_eur_per_mwh": "feature_upstream_forward_price_eur_per_mwh_asof",
            "upstream_day_ahead_price_eur_per_mwh": "feature_upstream_day_ahead_price_eur_per_mwh_asof",
            "upstream_intraday_price_eur_per_mwh": "feature_upstream_intraday_price_eur_per_mwh_asof",
            "upstream_day_ahead_to_intraday_spread_bucket": "feature_upstream_day_ahead_to_intraday_spread_bucket_asof",
            "upstream_forward_to_day_ahead_spread_bucket": "feature_upstream_forward_to_day_ahead_spread_bucket_asof",
            "system_balance_feed_available_flag": "feature_system_balance_feed_available_flag_asof",
            "system_balance_known_flag": "feature_system_balance_known_flag_asof",
            "system_balance_active_flag": "feature_system_balance_active_flag_asof",
            "system_balance_state": "feature_system_balance_state_asof",
            "system_balance_imbalance_direction_bucket": "feature_system_balance_imbalance_direction_bucket_asof",
            "system_balance_margin_direction_bucket": "feature_system_balance_margin_direction_bucket_asof",
            "system_balance_transition_state": "feature_system_balance_transition_state_asof",
            "system_balance_persistence_bucket": "feature_system_balance_persistence_bucket_asof",
            "route_price_score_eur_per_mwh": "feature_route_price_score_eur_per_mwh_asof",
            "route_price_feasible_flag": "feature_route_price_feasible_flag_asof",
            "route_price_bottleneck": "feature_route_price_bottleneck_asof",
            "route_price_state": "feature_route_price_state_asof",
            "route_price_delta_bucket": "feature_route_price_delta_bucket_asof",
            "route_price_transition_state": "feature_route_price_transition_state_asof",
            "route_price_persistence_bucket": "feature_route_price_persistence_bucket_asof",
            "route_delivery_tier": "feature_route_delivery_tier_asof",
            "connector_notice_market_state": "feature_connector_notice_market_state_asof",
            "connector_itl_state": "feature_connector_itl_state_asof",
            "connector_itl_source_key": "feature_connector_itl_source_key_asof",
            "internal_transfer_gate_state": "feature_internal_transfer_gate_state_asof",
            "internal_transfer_source_family": "feature_internal_transfer_source_family_asof",
            "internal_transfer_source_key": "feature_internal_transfer_source_key_asof",
            "internal_transfer_boundary_family": "feature_internal_transfer_boundary_family_asof",
            "internal_transfer_gate_bucket": "feature_internal_transfer_gate_bucket_asof",
            "route_delivery_transition_state": "feature_route_delivery_transition_state_asof",
            "connector_itl_transition_state": "feature_connector_itl_transition_state_asof",
            "internal_transfer_gate_transition_state": "feature_internal_transfer_gate_transition_state_asof",
            "route_state_persistence_bucket": "feature_route_state_persistence_bucket_asof",
            "connector_itl_state_path": "feature_connector_itl_state_path_asof",
            "internal_transfer_gate_state_path": "feature_internal_transfer_gate_state_path_asof",
            "curtailment_selected_mwh": "feature_curtailment_selected_mwh_asof",
            "deliverable_mw_proxy": "feature_deliverable_mw_proxy_asof",
            "deliverable_route_score_eur_per_mwh": "feature_route_score_eur_per_mwh_asof",
            "connector_capacity_tight_now_flag": "feature_connector_capacity_tight_now_flag_asof",
            "market_knew_connector_restriction_flag": "feature_market_knew_connector_restriction_flag_asof",
        }
    )
    joined = target.merge(
        origin,
        on=["cluster_key", "route_name", "hub_key", "feature_asof_utc"],
        how="left",
    )
    joined["feature_specialist_openable_potential_mwh_asof"] = _compute_specialist_openable_potential(
        joined.get("feature_curtailment_selected_mwh_asof", pd.Series(np.nan, index=joined.index)),
        joined.get("feature_deliverable_mw_proxy_asof", pd.Series(np.nan, index=joined.index)),
    )
    deliverable_proxy = pd.to_numeric(
        joined.get("feature_deliverable_mw_proxy_asof", pd.Series(np.nan, index=joined.index)),
        errors="coerce",
    ).fillna(0.0)
    joined["feature_specialist_zero_proxy_flag_asof"] = deliverable_proxy.le(SPECIALIST_RATIO_EPSILON)
    return joined.sort_values(
        ["forecast_origin_utc", "interval_start_utc", "cluster_key", "route_name", "hub_key"]
    ).reset_index(drop=True)


def _finalize_prediction_frame(
    frame: pd.DataFrame,
    model_key: str,
    split_strategy: str,
    source_lineage: str,
) -> pd.DataFrame:
    frame["prediction_eligible_flag"] = frame["predicted_opportunity_deliverable_mwh"].notna()
    frame["training_sample_count"] = pd.to_numeric(frame["training_sample_count"], errors="coerce").fillna(0).astype(int)
    frame["prediction_eligible_flag"] = frame["prediction_eligible_flag"].fillna(False).astype(bool)
    frame["predicted_opportunity_deliverable_mwh"] = pd.to_numeric(
        frame["predicted_opportunity_deliverable_mwh"], errors="coerce"
    ).clip(lower=0.0)
    frame["opportunity_deliverable_residual_mwh"] = (
        frame["actual_opportunity_deliverable_mwh"] - frame["predicted_opportunity_deliverable_mwh"]
    )
    frame["opportunity_deliverable_abs_error_mwh"] = frame["opportunity_deliverable_residual_mwh"].abs()
    frame["predicted_opportunity_gross_value_eur"] = (
        frame["predicted_opportunity_deliverable_mwh"].fillna(0.0) * frame["deliverable_route_score_eur_per_mwh"]
    )
    frame["opportunity_gross_value_residual_eur"] = (
        frame["actual_opportunity_gross_value_eur"] - frame["predicted_opportunity_gross_value_eur"]
    )
    frame["opportunity_gross_value_abs_error_eur"] = frame["opportunity_gross_value_residual_eur"].abs()
    frame["model_key"] = model_key
    frame["split_strategy"] = split_strategy
    frame["source_lineage"] = source_lineage

    keep_columns = list(_empty_backtest_prediction_frame().columns)
    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[keep_columns]


def _build_group_mean_backtest(frame: pd.DataFrame) -> pd.DataFrame:
    exact = _prior_mean_by_group(
        frame,
        [
            "cluster_key",
            "route_name",
            "feature_route_delivery_tier_asof",
            "feature_connector_notice_market_state_asof",
            "feature_origin_hour_of_day",
        ],
        "actual_opportunity_deliverable_mwh",
        "exact",
    )
    cluster_route = _prior_mean_by_group(
        frame,
        ["cluster_key", "route_name", "feature_route_delivery_tier_asof"],
        "actual_opportunity_deliverable_mwh",
        "cluster_route",
    )
    route_state = _prior_mean_by_group(
        frame,
        ["route_name", "feature_route_delivery_tier_asof"],
        "actual_opportunity_deliverable_mwh",
        "route_state",
    )
    global_prior = _prior_mean_by_group(
        frame,
        [],
        "actual_opportunity_deliverable_mwh",
        "global",
    )

    result = pd.concat([frame.copy(), exact, cluster_route, route_state, global_prior], axis=1)
    result["prediction_basis"] = pd.NA
    result["training_sample_count"] = 0
    result["predicted_opportunity_deliverable_mwh"] = np.nan

    origin_available = result["feature_origin_hour_of_day"].notna()
    exact_mask = origin_available & (result["exact_prior_count"] >= EXACT_MIN_HISTORY)
    cluster_route_mask = origin_available & ~exact_mask & (result["cluster_route_prior_count"] >= CLUSTER_ROUTE_MIN_HISTORY)
    route_state_mask = (
        origin_available
        & ~exact_mask
        & ~cluster_route_mask
        & (result["route_state_prior_count"] >= ROUTE_STATE_MIN_HISTORY)
    )
    global_mask = (
        origin_available
        & ~exact_mask
        & ~cluster_route_mask
        & ~route_state_mask
        & (result["global_prior_count"] >= GLOBAL_MIN_HISTORY)
    )

    result.loc[exact_mask, "prediction_basis"] = "exact_notice_hour"
    result.loc[exact_mask, "training_sample_count"] = result.loc[exact_mask, "exact_prior_count"]
    result.loc[exact_mask, "predicted_opportunity_deliverable_mwh"] = result.loc[exact_mask, "exact_prior_mean"]

    result.loc[cluster_route_mask, "prediction_basis"] = "cluster_route_state"
    result.loc[cluster_route_mask, "training_sample_count"] = result.loc[cluster_route_mask, "cluster_route_prior_count"]
    result.loc[cluster_route_mask, "predicted_opportunity_deliverable_mwh"] = result.loc[
        cluster_route_mask, "cluster_route_prior_mean"
    ]

    result.loc[route_state_mask, "prediction_basis"] = "route_state"
    result.loc[route_state_mask, "training_sample_count"] = result.loc[route_state_mask, "route_state_prior_count"]
    result.loc[route_state_mask, "predicted_opportunity_deliverable_mwh"] = result.loc[
        route_state_mask, "route_state_prior_mean"
    ]

    result.loc[global_mask, "prediction_basis"] = "global"
    result.loc[global_mask, "training_sample_count"] = result.loc[global_mask, "global_prior_count"]
    result.loc[global_mask, "predicted_opportunity_deliverable_mwh"] = result.loc[global_mask, "global_prior_mean"]

    return _finalize_prediction_frame(
        result,
        model_key=MODEL_GROUP_MEAN_NOTICE_V1,
        split_strategy=SPLIT_STRATEGY_GROUP_MEAN,
        source_lineage="fact_curtailment_opportunity_hourly|walk_forward_group_mean",
    )


def _build_potential_ratio_backtest(
    frame: pd.DataFrame,
    history_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_frame = frame.copy()
    target_frame["__is_backtest_target"] = True
    combined_frames = [target_frame]
    if history_frame is not None and not history_frame.empty:
        historical = history_frame.copy()
        historical["__is_backtest_target"] = False
        combined_frames.insert(0, historical)
    frame = pd.concat(combined_frames, ignore_index=True, sort=False)
    frame["origin_potential_opportunity_mwh"] = np.minimum(
        pd.to_numeric(frame["feature_curtailment_selected_mwh_asof"], errors="coerce").fillna(0.0).clip(lower=0.0),
        pd.to_numeric(frame["feature_deliverable_mw_proxy_asof"], errors="coerce").fillna(0.0).clip(lower=0.0),
    )
    origin_potential_nonzero = frame["origin_potential_opportunity_mwh"].gt(0.0)
    frame["realized_origin_potential_ratio"] = np.where(
        origin_potential_nonzero,
        frame["actual_opportunity_deliverable_mwh"] / frame["origin_potential_opportunity_mwh"],
        0.0,
    )
    frame["realized_origin_potential_ratio"] = pd.to_numeric(
        frame["realized_origin_potential_ratio"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0, upper=10.0)
    cluster_upstream_market_state = _prior_mean_by_group(
        frame,
        [
            "cluster_key",
            "route_name",
            "feature_hour_of_day",
            "feature_upstream_market_state_asof",
            "feature_upstream_day_ahead_to_intraday_spread_bucket_asof",
            "feature_upstream_forward_to_day_ahead_spread_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_cluster_upstream_market_state",
    )
    route_upstream_market_state = _prior_mean_by_group(
        frame,
        [
            "route_name",
            "feature_hour_of_day",
            "feature_upstream_market_state_asof",
            "feature_upstream_day_ahead_to_intraday_spread_bucket_asof",
            "feature_upstream_forward_to_day_ahead_spread_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_route_upstream_market_state",
    )
    cluster_system_balance = _prior_mean_by_group(
        frame,
        [
            "cluster_key",
            "route_name",
            "feature_hour_of_day",
            "feature_system_balance_state_asof",
            "feature_system_balance_imbalance_direction_bucket_asof",
            "feature_system_balance_margin_direction_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_cluster_system_balance",
    )
    route_system_balance = _prior_mean_by_group(
        frame,
        [
            "route_name",
            "feature_hour_of_day",
            "feature_system_balance_state_asof",
            "feature_system_balance_persistence_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_route_system_balance",
    )
    cluster_market_state = _prior_mean_by_group(
        frame,
        [
            "cluster_key",
            "route_name",
            "feature_hour_of_day",
            "feature_route_price_state_asof",
            "feature_route_price_delta_bucket_asof",
            "feature_connector_itl_state_asof",
            "feature_internal_transfer_gate_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_cluster_market_state",
    )
    route_market_state = _prior_mean_by_group(
        frame,
        [
            "route_name",
            "feature_hour_of_day",
            "feature_route_price_state_asof",
            "feature_route_price_delta_bucket_asof",
            "feature_route_price_feasible_flag_asof",
            "feature_route_price_bottleneck_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_route_market_state",
    )
    route_market_path = _prior_mean_by_group(
        frame,
        [
            "route_name",
            "feature_hour_of_day",
            "feature_route_price_transition_state_asof",
            "feature_connector_itl_state_path_asof",
            "feature_internal_transfer_gate_state_path_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_route_market_path",
    )
    cluster_transition_regime = _prior_mean_by_group(
        frame,
        [
            "cluster_key",
            "route_name",
            "feature_hour_of_day",
            "feature_route_delivery_tier_asof",
            "feature_connector_itl_state_asof",
            "feature_internal_transfer_gate_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_cluster_transition_regime",
    )
    transition_regime = _prior_mean_by_group(
        frame,
        [
            "route_name",
            "feature_hour_of_day",
            "feature_route_delivery_tier_asof",
            "feature_connector_itl_state_asof",
            "feature_internal_transfer_gate_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_transition_regime",
    )
    transition_hour = _prior_mean_by_group(
        frame,
        [
            "route_name",
            "feature_hour_of_day",
            "feature_route_delivery_tier_asof",
            "feature_connector_itl_state_asof",
            "feature_internal_transfer_gate_state_asof",
            "feature_route_state_persistence_bucket_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_transition_hour",
    )
    transition_path = _prior_mean_by_group(
        frame,
        [
            "route_name",
            "feature_hour_of_day",
            "feature_route_delivery_transition_state_asof",
            "feature_connector_itl_transition_state_asof",
            "feature_internal_transfer_gate_transition_state_asof",
        ],
        "realized_origin_potential_ratio",
        "ratio_transition_path",
    )
    exact = _prior_mean_by_group(
        frame,
        ["route_name", "feature_route_delivery_tier_asof", "feature_connector_notice_market_state_asof", "feature_origin_hour_of_day"],
        "realized_origin_potential_ratio",
        "ratio_exact",
    )
    route_notice = _prior_mean_by_group(
        frame,
        ["route_name", "feature_route_delivery_tier_asof", "feature_connector_notice_market_state_asof"],
        "realized_origin_potential_ratio",
        "ratio_route_notice",
    )
    route_tier = _prior_mean_by_group(
        frame,
        ["route_name", "feature_route_delivery_tier_asof"],
        "realized_origin_potential_ratio",
        "ratio_route_tier",
    )
    global_prior = _prior_mean_by_group(
        frame,
        [],
        "realized_origin_potential_ratio",
        "global",
    )

    result = pd.concat(
        [
            frame.copy(),
            cluster_upstream_market_state,
            route_upstream_market_state,
            cluster_system_balance,
            route_system_balance,
            cluster_market_state,
            route_market_state,
            route_market_path,
            cluster_transition_regime,
            transition_regime,
            transition_hour,
            transition_path,
            exact,
            route_notice,
            route_tier,
            global_prior,
        ],
        axis=1,
    )
    result["prediction_basis"] = pd.NA
    result["training_sample_count"] = 0
    result["predicted_ratio"] = np.nan

    origin_available = result["feature_origin_hour_of_day"].notna()
    targeted_transition_route = result["route_name"].isin(TARGETED_TRANSITION_ROUTE_NAMES)
    upstream_feed_available = _coerce_bool_series(
        result["feature_upstream_market_state_feed_available_flag_asof"],
        default=False,
    )
    system_balance_available = _coerce_bool_series(
        result["feature_system_balance_feed_available_flag_asof"],
        default=False,
    ) & _coerce_bool_series(
        result["feature_system_balance_known_flag_asof"],
        default=False,
    )
    cluster_upstream_market_state_mask = (
        origin_available
        & upstream_feed_available
        & (result["ratio_cluster_upstream_market_state_prior_count"] >= RATIO_CLUSTER_ROUTE_MARKET_STATE_MIN_HISTORY)
    )
    route_upstream_market_state_mask = (
        origin_available
        & upstream_feed_available
        & ~cluster_upstream_market_state_mask
        & (result["ratio_route_upstream_market_state_prior_count"] >= RATIO_ROUTE_MARKET_STATE_MIN_HISTORY)
    )
    cluster_system_balance_mask = (
        origin_available
        & system_balance_available
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & (result["ratio_cluster_system_balance_prior_count"] >= RATIO_CLUSTER_ROUTE_MARKET_STATE_MIN_HISTORY)
    )
    route_system_balance_mask = (
        origin_available
        & system_balance_available
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & (result["ratio_route_system_balance_prior_count"] >= RATIO_ROUTE_MARKET_STATE_MIN_HISTORY)
    )
    cluster_market_state_mask = (
        origin_available
        & targeted_transition_route
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & ~route_system_balance_mask
        & (result["ratio_cluster_market_state_prior_count"] >= RATIO_CLUSTER_ROUTE_MARKET_STATE_MIN_HISTORY)
    )
    route_market_state_mask = (
        origin_available
        & targeted_transition_route
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & ~route_system_balance_mask
        & ~cluster_market_state_mask
        & (result["ratio_route_market_state_prior_count"] >= RATIO_ROUTE_MARKET_STATE_MIN_HISTORY)
    )
    route_market_path_mask = (
        origin_available
        & targeted_transition_route
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & ~route_system_balance_mask
        & ~cluster_market_state_mask
        & ~route_market_state_mask
        & (result["ratio_route_market_path_prior_count"] >= RATIO_ROUTE_MARKET_PATH_MIN_HISTORY)
    )
    cluster_transition_regime_mask = (
        origin_available
        & targeted_transition_route
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & ~route_system_balance_mask
        & ~cluster_market_state_mask
        & ~route_market_state_mask
        & ~route_market_path_mask
        & (result["ratio_cluster_transition_regime_prior_count"] >= RATIO_CLUSTER_ROUTE_TRANSITION_REGIME_MIN_HISTORY)
    )
    transition_regime_mask = (
        origin_available
        & targeted_transition_route
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & ~route_system_balance_mask
        & ~cluster_market_state_mask
        & ~route_market_state_mask
        & ~route_market_path_mask
        & ~cluster_transition_regime_mask
        & (result["ratio_transition_regime_prior_count"] >= RATIO_ROUTE_TRANSITION_REGIME_MIN_HISTORY)
    )
    transition_hour_mask = (
        origin_available
        & targeted_transition_route
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & ~route_system_balance_mask
        & ~cluster_market_state_mask
        & ~route_market_state_mask
        & ~route_market_path_mask
        & ~cluster_transition_regime_mask
        & ~transition_regime_mask
        & (result["ratio_transition_hour_prior_count"] >= RATIO_ROUTE_TRANSITION_HOUR_MIN_HISTORY)
    )
    transition_path_mask = (
        origin_available
        & targeted_transition_route
        & ~cluster_upstream_market_state_mask
        & ~route_upstream_market_state_mask
        & ~cluster_system_balance_mask
        & ~route_system_balance_mask
        & ~cluster_market_state_mask
        & ~route_market_state_mask
        & ~route_market_path_mask
        & ~cluster_transition_regime_mask
        & ~transition_regime_mask
        & ~transition_hour_mask
        & (result["ratio_transition_path_prior_count"] >= RATIO_ROUTE_TRANSITION_PATH_MIN_HISTORY)
    )
    exact_mask = origin_available & ~cluster_upstream_market_state_mask & ~route_upstream_market_state_mask & ~cluster_system_balance_mask & ~route_system_balance_mask & ~cluster_market_state_mask & ~route_market_state_mask & ~route_market_path_mask & ~cluster_transition_regime_mask & ~transition_regime_mask & ~transition_hour_mask & ~transition_path_mask & (
        result["ratio_exact_prior_count"] >= RATIO_EXACT_MIN_HISTORY
    )
    route_notice_mask = origin_available & ~cluster_upstream_market_state_mask & ~route_upstream_market_state_mask & ~cluster_system_balance_mask & ~route_system_balance_mask & ~cluster_market_state_mask & ~route_market_state_mask & ~route_market_path_mask & ~cluster_transition_regime_mask & ~transition_regime_mask & ~transition_hour_mask & ~transition_path_mask & ~exact_mask & (
        result["ratio_route_notice_prior_count"] >= RATIO_ROUTE_NOTICE_MIN_HISTORY
    )
    route_tier_mask = origin_available & ~cluster_upstream_market_state_mask & ~route_upstream_market_state_mask & ~cluster_system_balance_mask & ~route_system_balance_mask & ~cluster_market_state_mask & ~route_market_state_mask & ~route_market_path_mask & ~cluster_transition_regime_mask & ~transition_regime_mask & ~transition_hour_mask & ~transition_path_mask & ~exact_mask & ~route_notice_mask & (
        result["ratio_route_tier_prior_count"] >= RATIO_ROUTE_TIER_MIN_HISTORY
    )
    global_mask = origin_available & ~cluster_upstream_market_state_mask & ~route_upstream_market_state_mask & ~cluster_system_balance_mask & ~route_system_balance_mask & ~cluster_market_state_mask & ~route_market_state_mask & ~route_market_path_mask & ~cluster_transition_regime_mask & ~transition_regime_mask & ~transition_hour_mask & ~transition_path_mask & ~exact_mask & ~route_notice_mask & ~route_tier_mask & (
        result["global_prior_count"] >= RATIO_GLOBAL_MIN_HISTORY
    )

    result.loc[cluster_upstream_market_state_mask, "prediction_basis"] = "ratio_cluster_route_upstream_market_state"
    result.loc[cluster_upstream_market_state_mask, "training_sample_count"] = result.loc[
        cluster_upstream_market_state_mask, "ratio_cluster_upstream_market_state_prior_count"
    ]
    result.loc[cluster_upstream_market_state_mask, "predicted_ratio"] = result.loc[
        cluster_upstream_market_state_mask, "ratio_cluster_upstream_market_state_prior_mean"
    ]

    result.loc[route_upstream_market_state_mask, "prediction_basis"] = "ratio_route_upstream_market_state"
    result.loc[route_upstream_market_state_mask, "training_sample_count"] = result.loc[
        route_upstream_market_state_mask, "ratio_route_upstream_market_state_prior_count"
    ]
    result.loc[route_upstream_market_state_mask, "predicted_ratio"] = result.loc[
        route_upstream_market_state_mask, "ratio_route_upstream_market_state_prior_mean"
    ]

    result.loc[cluster_system_balance_mask, "prediction_basis"] = "ratio_cluster_route_system_balance"
    result.loc[cluster_system_balance_mask, "training_sample_count"] = result.loc[
        cluster_system_balance_mask, "ratio_cluster_system_balance_prior_count"
    ]
    result.loc[cluster_system_balance_mask, "predicted_ratio"] = result.loc[
        cluster_system_balance_mask, "ratio_cluster_system_balance_prior_mean"
    ]

    result.loc[route_system_balance_mask, "prediction_basis"] = "ratio_route_system_balance"
    result.loc[route_system_balance_mask, "training_sample_count"] = result.loc[
        route_system_balance_mask, "ratio_route_system_balance_prior_count"
    ]
    result.loc[route_system_balance_mask, "predicted_ratio"] = result.loc[
        route_system_balance_mask, "ratio_route_system_balance_prior_mean"
    ]

    result.loc[cluster_market_state_mask, "prediction_basis"] = "ratio_cluster_route_market_state"
    result.loc[cluster_market_state_mask, "training_sample_count"] = result.loc[
        cluster_market_state_mask, "ratio_cluster_market_state_prior_count"
    ]
    result.loc[cluster_market_state_mask, "predicted_ratio"] = result.loc[
        cluster_market_state_mask, "ratio_cluster_market_state_prior_mean"
    ]

    result.loc[route_market_state_mask, "prediction_basis"] = "ratio_route_market_state"
    result.loc[route_market_state_mask, "training_sample_count"] = result.loc[
        route_market_state_mask, "ratio_route_market_state_prior_count"
    ]
    result.loc[route_market_state_mask, "predicted_ratio"] = result.loc[
        route_market_state_mask, "ratio_route_market_state_prior_mean"
    ]

    result.loc[route_market_path_mask, "prediction_basis"] = "ratio_route_market_path"
    result.loc[route_market_path_mask, "training_sample_count"] = result.loc[
        route_market_path_mask, "ratio_route_market_path_prior_count"
    ]
    result.loc[route_market_path_mask, "predicted_ratio"] = result.loc[
        route_market_path_mask, "ratio_route_market_path_prior_mean"
    ]

    result.loc[cluster_transition_regime_mask, "prediction_basis"] = "ratio_cluster_route_transition_regime"
    result.loc[cluster_transition_regime_mask, "training_sample_count"] = result.loc[
        cluster_transition_regime_mask, "ratio_cluster_transition_regime_prior_count"
    ]
    result.loc[cluster_transition_regime_mask, "predicted_ratio"] = result.loc[
        cluster_transition_regime_mask, "ratio_cluster_transition_regime_prior_mean"
    ]

    result.loc[transition_regime_mask, "prediction_basis"] = "ratio_route_transition_regime"
    result.loc[transition_regime_mask, "training_sample_count"] = result.loc[
        transition_regime_mask, "ratio_transition_regime_prior_count"
    ]
    result.loc[transition_regime_mask, "predicted_ratio"] = result.loc[
        transition_regime_mask, "ratio_transition_regime_prior_mean"
    ]

    result.loc[transition_hour_mask, "prediction_basis"] = "ratio_route_transition_hour"
    result.loc[transition_hour_mask, "training_sample_count"] = result.loc[
        transition_hour_mask, "ratio_transition_hour_prior_count"
    ]
    result.loc[transition_hour_mask, "predicted_ratio"] = result.loc[
        transition_hour_mask, "ratio_transition_hour_prior_mean"
    ]

    result.loc[transition_path_mask, "prediction_basis"] = "ratio_route_transition_path"
    result.loc[transition_path_mask, "training_sample_count"] = result.loc[
        transition_path_mask, "ratio_transition_path_prior_count"
    ]
    result.loc[transition_path_mask, "predicted_ratio"] = result.loc[
        transition_path_mask, "ratio_transition_path_prior_mean"
    ]

    result.loc[exact_mask, "prediction_basis"] = "ratio_exact_notice_hour"
    result.loc[exact_mask, "training_sample_count"] = result.loc[exact_mask, "ratio_exact_prior_count"]
    result.loc[exact_mask, "predicted_ratio"] = result.loc[exact_mask, "ratio_exact_prior_mean"]

    result.loc[route_notice_mask, "prediction_basis"] = "ratio_route_notice_state"
    result.loc[route_notice_mask, "training_sample_count"] = result.loc[
        route_notice_mask, "ratio_route_notice_prior_count"
    ]
    result.loc[route_notice_mask, "predicted_ratio"] = result.loc[
        route_notice_mask, "ratio_route_notice_prior_mean"
    ]

    result.loc[route_tier_mask, "prediction_basis"] = "ratio_route_delivery_tier"
    result.loc[route_tier_mask, "training_sample_count"] = result.loc[
        route_tier_mask, "ratio_route_tier_prior_count"
    ]
    result.loc[route_tier_mask, "predicted_ratio"] = result.loc[
        route_tier_mask, "ratio_route_tier_prior_mean"
    ]

    result.loc[global_mask, "prediction_basis"] = "ratio_global"
    result.loc[global_mask, "training_sample_count"] = result.loc[global_mask, "global_prior_count"]
    result.loc[global_mask, "predicted_ratio"] = result.loc[global_mask, "global_prior_mean"]

    result["predicted_ratio"] = pd.to_numeric(result["predicted_ratio"], errors="coerce").clip(lower=0.0, upper=10.0)
    result["predicted_opportunity_deliverable_mwh"] = result["origin_potential_opportunity_mwh"] * result["predicted_ratio"]
    result = _apply_potential_ratio_opening_guardrail(result)
    result = _apply_potential_ratio_r2_reviewed_event_lifecycle(result)
    result = _apply_potential_ratio_r2_supported_range_suppressors(result)
    result = _apply_potential_ratio_event_phase_calibration(result)
    result = _apply_potential_ratio_persist_close_suppressor(result)
    result = result[result["__is_backtest_target"].fillna(False)].copy()

    return _finalize_prediction_frame(
        result,
        model_key=MODEL_POTENTIAL_RATIO_V2,
        split_strategy=SPLIT_STRATEGY_POTENTIAL_RATIO,
        source_lineage="fact_curtailment_opportunity_hourly|walk_forward_potential_ratio",
    )


def _apply_potential_ratio_opening_guardrail(result: pd.DataFrame) -> pd.DataFrame:
    adjusted = result.copy()
    forecast_horizon = pd.to_numeric(adjusted.get("forecast_horizon_hours"), errors="coerce")
    route_name = adjusted.get("route_name", pd.Series(pd.NA, index=adjusted.index)).astype("string")
    predicted = pd.to_numeric(adjusted.get("predicted_opportunity_deliverable_mwh"), errors="coerce")
    curtailment_selected = pd.to_numeric(
        adjusted.get("feature_curtailment_selected_mwh_asof"),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    route_score = pd.to_numeric(
        adjusted.get("feature_route_price_score_eur_per_mwh_asof"),
        errors="coerce",
    ).fillna(-np.inf)
    route_price_feasible = _coerce_bool_series(
        adjusted.get("feature_route_price_feasible_flag_asof", pd.Series(False, index=adjusted.index)),
        default=False,
    )
    transition_state = adjusted.get(
        "feature_route_price_transition_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    source_family = adjusted.get(
        "feature_internal_transfer_source_family_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    gate_state = adjusted.get(
        "feature_internal_transfer_gate_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    connector_notice_state = adjusted.get(
        "feature_connector_notice_market_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    connector_itl_state = adjusted.get(
        "feature_connector_itl_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    upstream_state = adjusted.get(
        "feature_upstream_market_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    origin_hour = pd.to_numeric(
        adjusted.get("feature_origin_hour_of_day"),
        errors="coerce",
    )
    base_mask = (
        forecast_horizon.eq(1)
        & route_name.isin(TARGETED_OPENING_GUARDRAIL_ROUTE_NAMES)
        & (predicted.isna() | predicted.le(OPENING_GUARDRAIL_PREDICTION_EPSILON))
        & curtailment_selected.gt(0.0)
        & source_family.eq("day_ahead_constraint_boundary")
        & ~gate_state.str.startswith("blocked_")
        & connector_notice_state.eq("no_public_connector_restriction")
    )
    jump_mask = (
        base_mask
        & transition_state.eq("price_non_positive->price_high_positive")
        & route_price_feasible
    )
    preopen_mask = (
        base_mask
        & ~(
            route_name.eq(R2_REVIEWED_EVENT_ROUTE_NAME)
            & connector_itl_state.eq("published_restriction")
        )
        & ~connector_itl_state.eq("blocked_zero_or_negative_itl")
        & transition_state.eq("price_non_positive->price_non_positive")
        & ~route_price_feasible
        & route_score.gt(-ROUTE_PRICE_SOFT_MOVE_EUR_PER_MWH)
        & origin_hour.le(OPENING_GUARDRAIL_PREOPEN_ORIGIN_HOUR_MAX)
        & ~upstream_state.isin(OPENING_GUARDRAIL_EXTREME_UPSTREAM_STATES)
    )
    if jump_mask.any():
        adjusted.loc[jump_mask, "predicted_opportunity_deliverable_mwh"] = curtailment_selected.loc[jump_mask]
        adjusted.loc[jump_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[jump_mask, "prediction_basis"],
            "opening_guardrail_jump",
        ).values
    if preopen_mask.any():
        adjusted.loc[preopen_mask, "predicted_opportunity_deliverable_mwh"] = curtailment_selected.loc[preopen_mask]
        adjusted.loc[preopen_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[preopen_mask, "prediction_basis"],
            "opening_guardrail_preopen",
        ).values
    return adjusted


def _apply_potential_ratio_r2_reviewed_event_lifecycle(result: pd.DataFrame) -> pd.DataFrame:
    adjusted = result.copy()
    forecast_horizon = pd.to_numeric(adjusted.get("forecast_horizon_hours"), errors="coerce")
    route_name = adjusted.get("route_name", pd.Series(pd.NA, index=adjusted.index)).astype("string")
    cluster_key = adjusted.get("cluster_key", pd.Series(pd.NA, index=adjusted.index)).fillna("").astype(str)
    prediction_basis = adjusted.get("prediction_basis", pd.Series(pd.NA, index=adjusted.index)).astype("string").fillna("")
    predicted = pd.to_numeric(
        adjusted.get("predicted_opportunity_deliverable_mwh"),
        errors="coerce",
    ).fillna(0.0)
    curtailment_selected = pd.to_numeric(
        adjusted.get("feature_curtailment_selected_mwh_asof"),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    source_family = adjusted.get(
        "feature_internal_transfer_source_family_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    gate_state = adjusted.get(
        "feature_internal_transfer_gate_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    transition_state = adjusted.get(
        "feature_route_price_transition_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    route_delivery_tier = adjusted.get(
        "feature_route_delivery_tier_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    connector_itl_state = adjusted.get(
        "feature_connector_itl_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    upstream_state = adjusted.get(
        "feature_upstream_market_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    origin_hour = pd.to_numeric(
        adjusted.get("feature_origin_hour_of_day"),
        errors="coerce",
    )
    route_score = pd.to_numeric(
        adjusted.get("feature_route_price_score_eur_per_mwh_asof"),
        errors="coerce",
    ).fillna(-np.inf)
    late_reopen_scope = (
        forecast_horizon.eq(1)
        & route_name.eq(R2_REVIEWED_EVENT_ROUTE_NAME)
        & source_family.eq(R2_REVIEWED_EVENT_SOURCE_FAMILY)
        & curtailment_selected.gt(0.0)
        & predicted.le(OPENING_GUARDRAIL_PREDICTION_EPSILON)
        & gate_state.eq("reviewed_boundary_cap")
        & route_delivery_tier.eq("no_price_signal")
        & transition_state.eq("price_non_positive->price_non_positive")
        & connector_itl_state.eq("blocked_zero_or_negative_itl")
        & prediction_basis.str.startswith("ratio_route_notice_state")
        & origin_hour.eq(R2_REVIEWED_EVENT_LATE_REOPEN_ORIGIN_HOUR)
        & route_score.le(-ROUTE_PRICE_SOFT_MOVE_EUR_PER_MWH)
        & upstream_state.isin(R2_REVIEWED_EVENT_LATE_REOPEN_UPSTREAM_STATES)
        & cluster_key.isin(R2_REVIEWED_EVENT_LATE_REOPEN_CLUSTERS)
    )
    base_scope = (
        forecast_horizon.eq(1)
        & route_name.eq(R2_REVIEWED_EVENT_ROUTE_NAME)
        & source_family.eq(R2_REVIEWED_EVENT_SOURCE_FAMILY)
        & curtailment_selected.gt(0.0)
        & ~gate_state.str.startswith("blocked_")
        & connector_itl_state.eq("published_restriction")
    )
    preopen_open_mask = (
        base_scope
        & predicted.le(OPENING_GUARDRAIL_PREDICTION_EPSILON)
        & route_delivery_tier.eq("no_price_signal")
        & transition_state.eq("price_non_positive->price_non_positive")
        & origin_hour.eq(R2_REVIEWED_EVENT_PREOPEN_ORIGIN_HOUR)
        & route_score.gt(-ROUTE_PRICE_SOFT_MOVE_EUR_PER_MWH)
        & upstream_state.isin(R2_REVIEWED_EVENT_PREOPEN_UPSTREAM_STATES)
    )
    reviewed_open_mask = (
        base_scope
        & predicted.le(OPENING_GUARDRAIL_PREDICTION_EPSILON)
        & route_delivery_tier.eq("reviewed")
        & transition_state.eq("price_non_positive->price_mid_positive")
        & origin_hour.eq(R2_REVIEWED_EVENT_OPEN_ORIGIN_HOUR)
        & route_score.gt(0.0)
    )
    jump_suppress_mask = (
        base_scope
        & predicted.gt(OPENING_GUARDRAIL_PREDICTION_EPSILON)
        & route_delivery_tier.eq("reviewed")
        & transition_state.eq("price_non_positive->price_high_positive")
    )
    close_suppress_mask = (
        base_scope
        & predicted.gt(OPENING_GUARDRAIL_PREDICTION_EPSILON)
        & route_delivery_tier.eq("reviewed")
        & transition_state.eq("price_mid_positive->price_low_positive")
        & origin_hour.eq(R2_REVIEWED_EVENT_CLOSE_ORIGIN_HOUR)
    )
    if late_reopen_scope.any():
        late_reopen_prediction = curtailment_selected * R2_REVIEWED_EVENT_LATE_REOPEN_CURTAILMENT_RATIO
        for capped_cluster_key, capped_cluster_mwh in R2_REVIEWED_EVENT_LATE_REOPEN_CLUSTER_CAP_MWH.items():
            cluster_mask = cluster_key.eq(capped_cluster_key)
            late_reopen_prediction = late_reopen_prediction.where(
                ~cluster_mask,
                np.minimum(late_reopen_prediction, capped_cluster_mwh),
            )
        adjusted.loc[late_reopen_scope, "predicted_opportunity_deliverable_mwh"] = late_reopen_prediction.loc[
            late_reopen_scope
        ]
        adjusted.loc[late_reopen_scope, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[late_reopen_scope, "prediction_basis"],
            "r2_reviewed_event_late_reopen",
        ).values
    if preopen_open_mask.any():
        adjusted.loc[preopen_open_mask, "predicted_opportunity_deliverable_mwh"] = curtailment_selected.loc[
            preopen_open_mask
        ]
        adjusted.loc[preopen_open_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[preopen_open_mask, "prediction_basis"],
            "r2_reviewed_event_preopen_open",
        ).values
    if reviewed_open_mask.any():
        adjusted.loc[reviewed_open_mask, "predicted_opportunity_deliverable_mwh"] = curtailment_selected.loc[
            reviewed_open_mask
        ]
        adjusted.loc[reviewed_open_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[reviewed_open_mask, "prediction_basis"],
            "r2_reviewed_event_open",
        ).values
    if jump_suppress_mask.any():
        adjusted.loc[jump_suppress_mask, "predicted_opportunity_deliverable_mwh"] = 0.0
        adjusted.loc[jump_suppress_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[jump_suppress_mask, "prediction_basis"],
            "r2_reviewed_event_jump_suppressor",
        ).values
    if close_suppress_mask.any():
        adjusted.loc[close_suppress_mask, "predicted_opportunity_deliverable_mwh"] = 0.0
        adjusted.loc[close_suppress_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[close_suppress_mask, "prediction_basis"],
            "r2_reviewed_event_close_suppressor",
        ).values
    return adjusted


def _apply_potential_ratio_r2_supported_range_suppressors(result: pd.DataFrame) -> pd.DataFrame:
    adjusted = result.copy()
    forecast_horizon = pd.to_numeric(adjusted.get("forecast_horizon_hours"), errors="coerce")
    route_name = adjusted.get("route_name", pd.Series(pd.NA, index=adjusted.index)).astype("string")
    source_family = adjusted.get(
        "feature_internal_transfer_source_family_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    connector_itl_state = adjusted.get(
        "feature_connector_itl_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    forecast_origin_utc = pd.to_datetime(
        adjusted.get("forecast_origin_utc", pd.Series(pd.NaT, index=adjusted.index)),
        utc=True,
        errors="coerce",
    )
    route_delivery_tier = adjusted.get(
        "feature_route_delivery_tier_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    transition_state = adjusted.get(
        "feature_route_price_transition_state_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).fillna("").astype(str)
    prediction_basis = adjusted.get(
        "prediction_basis",
        pd.Series(pd.NA, index=adjusted.index),
    ).astype("string").fillna("")
    predicted = pd.to_numeric(
        adjusted.get("predicted_opportunity_deliverable_mwh"),
        errors="coerce",
    ).fillna(0.0)
    origin_hour = pd.to_numeric(
        adjusted.get("feature_origin_hour_of_day"),
        errors="coerce",
    )

    supported_range_scope = (
        forecast_horizon.eq(1)
        & route_name.eq(R2_REVIEWED_EVENT_ROUTE_NAME)
        & source_family.eq(R2_REVIEWED_EVENT_SOURCE_FAMILY)
        & forecast_origin_utc.ge(R2_2025_REGIME_WORK_START_UTC)
    )
    if not supported_range_scope.any():
        return adjusted

    no_public_preopen_suppress_mask = (
        supported_range_scope
        & predicted.gt(OPENING_GUARDRAIL_PREDICTION_EPSILON)
        & connector_itl_state.eq("no_public_itl_restriction")
        & route_delivery_tier.eq("no_price_signal")
        & transition_state.eq("price_non_positive->price_non_positive")
        & origin_hour.le(OPENING_GUARDRAIL_PREOPEN_ORIGIN_HOUR_MAX)
        & prediction_basis.str.contains("opening_guardrail_preopen", regex=False)
    )
    published_reviewed_event_suppress_mask = (
        supported_range_scope
        & predicted.gt(OPENING_GUARDRAIL_PREDICTION_EPSILON)
        & connector_itl_state.eq("published_restriction")
        & origin_hour.isin([R2_REVIEWED_EVENT_PREOPEN_ORIGIN_HOUR, R2_REVIEWED_EVENT_OPEN_ORIGIN_HOUR])
        & (
            prediction_basis.str.contains("r2_reviewed_event_preopen_open", regex=False)
            | prediction_basis.str.contains("r2_reviewed_event_open", regex=False)
        )
    )

    if no_public_preopen_suppress_mask.any():
        adjusted.loc[no_public_preopen_suppress_mask, "predicted_opportunity_deliverable_mwh"] = 0.0
        adjusted.loc[no_public_preopen_suppress_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[no_public_preopen_suppress_mask, "prediction_basis"],
            "r2_2025_no_public_preopen_suppressor",
        ).values
    if published_reviewed_event_suppress_mask.any():
        adjusted.loc[published_reviewed_event_suppress_mask, "predicted_opportunity_deliverable_mwh"] = 0.0
        adjusted.loc[published_reviewed_event_suppress_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            adjusted.loc[published_reviewed_event_suppress_mask, "prediction_basis"],
            "r2_2025_reviewed_event_suppressor",
        ).values
    return adjusted


def _classify_potential_ratio_event_phase(result: pd.DataFrame) -> pd.Series:
    basis_text = result.get("prediction_basis", pd.Series(pd.NA, index=result.index)).astype("string").fillna("")
    route_name = result.get("route_name", pd.Series(pd.NA, index=result.index)).astype("string")
    forecast_horizon = pd.to_numeric(result.get("forecast_horizon_hours"), errors="coerce")
    source_family = result.get(
        "feature_internal_transfer_source_family_asof",
        pd.Series(pd.NA, index=result.index),
    ).fillna("").astype(str)
    transition_state = result.get(
        "feature_route_price_transition_state_asof",
        pd.Series(pd.NA, index=result.index),
    ).fillna("").astype(str)
    curtailment_selected = pd.to_numeric(
        result.get("feature_curtailment_selected_mwh_asof"),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    base_prediction = pd.to_numeric(
        result.get("predicted_opportunity_deliverable_mwh"),
        errors="coerce",
    )
    reviewed_event_family_mask = (
        basis_text.str.startswith("opening_guardrail_")
        | basis_text.str.startswith("ratio_route_notice_state")
        | basis_text.str.startswith("ratio_exact_notice_hour")
    )
    scoped_mask = (
        forecast_horizon.eq(1)
        & route_name.eq(EVENT_PHASE_CALIBRATION_ROUTE_NAME)
        & source_family.eq(EVENT_PHASE_CALIBRATION_SOURCE_FAMILY)
        & base_prediction.gt(EVENT_PHASE_CALIBRATION_RATIO_EPSILON)
        & reviewed_event_family_mask
    )

    event_phase = pd.Series(pd.NA, index=result.index, dtype="string")
    preopen_mask = (
        scoped_mask
        & basis_text.str.contains("opening_guardrail_preopen", regex=False)
        & transition_state.eq("price_non_positive->price_non_positive")
    )
    jump_mask = (
        scoped_mask
        & basis_text.str.contains("opening_guardrail_jump", regex=False)
        & transition_state.eq("price_non_positive->price_high_positive")
    )
    persist_mask = (
        scoped_mask
        & transition_state.eq("price_high_positive->price_high_positive")
        & curtailment_selected.gt(0.0)
    )
    event_phase.loc[preopen_mask] = "preopen"
    event_phase.loc[jump_mask] = "jump"
    event_phase.loc[persist_mask] = "persist"
    return event_phase


def _apply_potential_ratio_event_phase_calibration(result: pd.DataFrame) -> pd.DataFrame:
    adjusted = result.copy()
    base_prediction = pd.to_numeric(
        adjusted.get("predicted_opportunity_deliverable_mwh"),
        errors="coerce",
    )
    actual = pd.to_numeric(
        adjusted.get("actual_opportunity_deliverable_mwh"),
        errors="coerce",
    ).fillna(0.0)
    curtailment_selected = pd.to_numeric(
        adjusted.get("feature_curtailment_selected_mwh_asof"),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    event_phase = _classify_potential_ratio_event_phase(adjusted)
    calibration_scope = event_phase.notna()
    if not calibration_scope.any():
        return adjusted

    target_frame = adjusted.loc[:, ["forecast_origin_utc", "route_name", "cluster_key"]].copy()
    target_frame["event_phase"] = event_phase

    reference = pd.Series(
        np.maximum(base_prediction.fillna(0.0), EVENT_PHASE_CALIBRATION_RATIO_EPSILON),
        index=adjusted.index,
        dtype=float,
    )
    history_frame = adjusted.loc[calibration_scope, ["forecast_origin_utc", "route_name", "cluster_key"]].copy()
    history_frame["event_phase"] = event_phase.loc[calibration_scope].astype("string")
    history_frame["event_phase_realized_ratio"] = (
        actual.loc[calibration_scope] / reference.loc[calibration_scope]
    ).clip(
        lower=EVENT_PHASE_CALIBRATION_RATIO_MIN,
        upper=EVENT_PHASE_CALIBRATION_RATIO_MAX,
    )

    cluster_phase_prior = _prior_mean_from_history_by_group(
        target_frame,
        history_frame,
        ["route_name", "cluster_key", "event_phase"],
        "event_phase_realized_ratio",
        "event_phase_cluster",
    )
    route_phase_prior = _prior_mean_from_history_by_group(
        target_frame,
        history_frame,
        ["route_name", "event_phase"],
        "event_phase_realized_ratio",
        "event_phase_route",
    )
    route_prior = _prior_mean_from_history_by_group(
        target_frame,
        history_frame,
        ["route_name"],
        "event_phase_realized_ratio",
        "event_phase_route_only",
    )

    learned_ratio = pd.Series(np.nan, index=adjusted.index, dtype=float)
    learned_count = pd.Series(0, index=adjusted.index, dtype=int)
    cluster_mask = (
        calibration_scope
        & cluster_phase_prior["event_phase_cluster_prior_count"].ge(EVENT_PHASE_CALIBRATION_MIN_HISTORY)
        & cluster_phase_prior["event_phase_cluster_prior_mean"].notna()
    )
    route_phase_mask = (
        calibration_scope
        & ~cluster_mask
        & route_phase_prior["event_phase_route_prior_count"].ge(EVENT_PHASE_CALIBRATION_MIN_HISTORY)
        & route_phase_prior["event_phase_route_prior_mean"].notna()
    )
    route_mask = (
        calibration_scope
        & event_phase.eq("persist")
        & ~cluster_mask
        & ~route_phase_mask
        & route_prior["event_phase_route_only_prior_count"].ge(EVENT_PHASE_CALIBRATION_MIN_HISTORY)
        & route_prior["event_phase_route_only_prior_mean"].notna()
    )

    learned_ratio.loc[cluster_mask] = cluster_phase_prior.loc[cluster_mask, "event_phase_cluster_prior_mean"]
    learned_count.loc[cluster_mask] = cluster_phase_prior.loc[cluster_mask, "event_phase_cluster_prior_count"]
    learned_ratio.loc[route_phase_mask] = route_phase_prior.loc[route_phase_mask, "event_phase_route_prior_mean"]
    learned_count.loc[route_phase_mask] = route_phase_prior.loc[route_phase_mask, "event_phase_route_prior_count"]
    learned_ratio.loc[route_mask] = route_prior.loc[route_mask, "event_phase_route_only_prior_mean"]
    learned_count.loc[route_mask] = route_prior.loc[route_mask, "event_phase_route_only_prior_count"]

    apply_mask = learned_ratio.notna()
    if not apply_mask.any():
        return adjusted

    calibrated_prediction = (base_prediction * learned_ratio).clip(lower=0.0)
    calibrated_cap = curtailment_selected.where(curtailment_selected.gt(0.0), np.inf)
    calibrated_prediction = pd.Series(
        np.minimum(calibrated_prediction, calibrated_cap),
        index=adjusted.index,
        dtype=float,
    )
    adjusted.loc[apply_mask, "predicted_opportunity_deliverable_mwh"] = calibrated_prediction.loc[apply_mask]

    current_training_count = pd.to_numeric(
        adjusted.get("training_sample_count", pd.Series(0, index=adjusted.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    adjusted.loc[apply_mask, "training_sample_count"] = np.maximum(
        current_training_count.loc[apply_mask],
        learned_count.loc[apply_mask],
    ).astype(int)

    for phase_name in ("preopen", "jump", "persist"):
        phase_mask = apply_mask & event_phase.eq(phase_name)
        if phase_mask.any():
            adjusted.loc[phase_mask, "prediction_basis"] = _append_prediction_basis_suffix(
                adjusted.loc[phase_mask, "prediction_basis"],
                f"event_phase_calibrated_{phase_name}",
            ).values
    return adjusted


def _apply_potential_ratio_persist_close_suppressor(result: pd.DataFrame) -> pd.DataFrame:
    adjusted = result.copy()
    event_phase = _classify_potential_ratio_event_phase(adjusted)
    persist_mask = event_phase.eq("persist")
    if not persist_mask.any():
        return adjusted

    actual = pd.to_numeric(
        adjusted.get("actual_opportunity_deliverable_mwh"),
        errors="coerce",
    ).fillna(0.0)
    predicted = pd.to_numeric(
        adjusted.get("predicted_opportunity_deliverable_mwh"),
        errors="coerce",
    ).fillna(0.0)
    route_delivery_tier = adjusted.get(
        "feature_route_delivery_tier_asof",
        pd.Series(pd.NA, index=adjusted.index),
    ).astype("string")
    origin_hour = pd.to_numeric(
        adjusted.get("feature_origin_hour_of_day"),
        errors="coerce",
    )

    target_frame = adjusted.loc[:, ["forecast_origin_utc", "route_name", "cluster_key", "feature_origin_hour_of_day"]].copy()
    history_frame = adjusted.loc[persist_mask, ["forecast_origin_utc", "route_name", "cluster_key", "feature_origin_hour_of_day"]].copy()
    history_frame["persist_actual_positive_flag"] = actual.loc[persist_mask].gt(
        PERSIST_CLOSE_SUPPRESSOR_POSITIVE_EPSILON
    ).astype(float)

    cluster_hour_prior = _prior_mean_from_history_by_group(
        target_frame,
        history_frame,
        ["route_name", "cluster_key", "feature_origin_hour_of_day"],
        "persist_actual_positive_flag",
        "persist_close_cluster_hour",
    )
    route_hour_prior = _prior_mean_from_history_by_group(
        target_frame,
        history_frame,
        ["route_name", "feature_origin_hour_of_day"],
        "persist_actual_positive_flag",
        "persist_close_route_hour",
    )
    cluster_prior = _prior_mean_from_history_by_group(
        target_frame,
        history_frame,
        ["route_name", "cluster_key"],
        "persist_actual_positive_flag",
        "persist_close_cluster",
    )

    positive_share = pd.Series(np.nan, index=adjusted.index, dtype=float)
    positive_count = pd.Series(0, index=adjusted.index, dtype=int)
    cluster_hour_mask = (
        persist_mask
        & cluster_hour_prior["persist_close_cluster_hour_prior_count"].ge(
            PERSIST_CLOSE_SUPPRESSOR_CLUSTER_HOUR_MIN_HISTORY
        )
        & cluster_hour_prior["persist_close_cluster_hour_prior_mean"].notna()
    )
    route_hour_mask = (
        persist_mask
        & ~cluster_hour_mask
        & route_hour_prior["persist_close_route_hour_prior_count"].ge(
            PERSIST_CLOSE_SUPPRESSOR_ROUTE_HOUR_MIN_HISTORY
        )
        & route_hour_prior["persist_close_route_hour_prior_mean"].notna()
    )
    cluster_mask = (
        persist_mask
        & ~cluster_hour_mask
        & ~route_hour_mask
        & cluster_prior["persist_close_cluster_prior_count"].ge(
            PERSIST_CLOSE_SUPPRESSOR_CLUSTER_MIN_HISTORY
        )
        & cluster_prior["persist_close_cluster_prior_mean"].notna()
    )

    positive_share.loc[cluster_hour_mask] = cluster_hour_prior.loc[
        cluster_hour_mask, "persist_close_cluster_hour_prior_mean"
    ]
    positive_count.loc[cluster_hour_mask] = cluster_hour_prior.loc[
        cluster_hour_mask, "persist_close_cluster_hour_prior_count"
    ]
    positive_share.loc[route_hour_mask] = route_hour_prior.loc[
        route_hour_mask, "persist_close_route_hour_prior_mean"
    ]
    positive_count.loc[route_hour_mask] = route_hour_prior.loc[
        route_hour_mask, "persist_close_route_hour_prior_count"
    ]
    positive_share.loc[cluster_mask] = cluster_prior.loc[
        cluster_mask, "persist_close_cluster_prior_mean"
    ]
    positive_count.loc[cluster_mask] = cluster_prior.loc[
        cluster_mask, "persist_close_cluster_prior_count"
    ]

    history_suppress_mask = positive_share.notna() & positive_share.le(PERSIST_CLOSE_SUPPRESSOR_MAX_POSITIVE_SHARE)
    fallback_close_rule_mask = (
        persist_mask
        & positive_share.isna()
        & route_delivery_tier.eq("capacity_unknown")
        & origin_hour.ge(4.0)
    )
    suppress_mask = history_suppress_mask | fallback_close_rule_mask
    if not suppress_mask.any():
        return adjusted

    suppressed_prediction = (predicted * positive_share.fillna(1.0)).clip(lower=0.0)
    suppressed_prediction.loc[fallback_close_rule_mask] = 0.0
    adjusted.loc[suppress_mask, "predicted_opportunity_deliverable_mwh"] = suppressed_prediction.loc[suppress_mask]

    current_training_count = pd.to_numeric(
        adjusted.get("training_sample_count", pd.Series(0, index=adjusted.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    adjusted.loc[suppress_mask, "training_sample_count"] = np.maximum(
        current_training_count.loc[suppress_mask],
        positive_count.loc[suppress_mask],
    ).astype(int)
    adjusted.loc[suppress_mask, "prediction_basis"] = _append_prediction_basis_suffix(
        adjusted.loc[suppress_mask, "prediction_basis"],
        "persist_close_suppressor",
    ).values
    return adjusted


def _build_gb_nl_reviewed_specialist_v3_backtest(frame: pd.DataFrame) -> pd.DataFrame:
    scoped = frame[_specialist_scope_mask(frame)].copy()
    if scoped.empty:
        return _empty_backtest_prediction_frame()

    scoped = scoped.sort_values(
        ["forecast_origin_utc", "interval_start_utc", "cluster_key", "route_name", "hub_key"]
    ).reset_index(drop=True)
    openable_potential = pd.to_numeric(
        scoped.get("feature_specialist_openable_potential_mwh_asof"),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    scoped["specialist_open_target"] = scoped["actual_opportunity_deliverable_mwh"].gt(0.0).astype(int)
    scoped["specialist_ratio_target"] = (
        scoped["actual_opportunity_deliverable_mwh"]
        / np.maximum(openable_potential, SPECIALIST_RATIO_EPSILON)
    )
    scoped["specialist_ratio_target"] = pd.to_numeric(
        scoped["specialist_ratio_target"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    scoped["prediction_basis"] = pd.NA
    scoped["training_sample_count"] = 0
    scoped["predicted_open_probability"] = np.nan
    scoped["predicted_ratio"] = np.nan
    feature_frame = _prepare_specialist_feature_frame(scoped)
    flip_focus_mask = _specialist_flip_focus_mask(scoped)
    flip_suppressed_mask = _specialist_flip_suppressed_mask(scoped)

    for forecast_origin_utc, origin_frame in scoped.groupby("forecast_origin_utc", sort=True, dropna=False):
        train_mask = scoped["forecast_origin_utc"].lt(forecast_origin_utc)
        train_frame = scoped.loc[train_mask].copy()
        if train_frame.empty:
            continue

        train_count = int(len(train_frame))
        target_frame = scoped.loc[origin_frame.index].copy()
        x_train = feature_frame.loc[train_mask]
        x_target = feature_frame.loc[origin_frame.index]
        open_target = train_frame["specialist_open_target"].astype(int)
        open_probability = np.full(len(target_frame), float(open_target.mean()), dtype=float)
        basis_parts = ["specialist_v3"]

        if open_target.nunique(dropna=False) >= 2:
            classifier = _build_specialist_pipeline("classifier")
            positive_count = int(open_target.sum())
            negative_count = int(len(open_target) - positive_count)
            positive_weight = 1.0
            if positive_count > 0 and negative_count > 0:
                positive_weight = min(
                    max(negative_count / positive_count, 1.0),
                    SPECIALIST_CLASSIFIER_POSITIVE_WEIGHT_CAP,
                )
            sample_weight = np.where(open_target.eq(1), positive_weight, 1.0)
            classifier.fit(x_train, open_target, estimator__sample_weight=sample_weight)
            open_probability = classifier.predict_proba(x_target)[:, 1]
            basis_parts.append("hybrid_open")
        else:
            basis_parts.append("constant_open")

        origin_flip_mask = flip_focus_mask.loc[origin_frame.index].fillna(False).to_numpy(dtype=bool)
        train_flip_mask = flip_focus_mask.loc[train_frame.index].fillna(False)
        flip_train_frame = train_frame.loc[train_flip_mask].copy()
        if origin_flip_mask.any() and not flip_train_frame.empty:
            flip_x_train = feature_frame.loc[flip_train_frame.index]
            flip_open_target = flip_train_frame["specialist_open_target"].astype(int)
            if flip_open_target.nunique(dropna=False) >= 2:
                flip_classifier = _build_specialist_pipeline("classifier")
                flip_positive_count = int(flip_open_target.sum())
                flip_negative_count = int(len(flip_open_target) - flip_positive_count)
                flip_positive_weight = 1.0
                if flip_positive_count > 0 and flip_negative_count > 0:
                    flip_positive_weight = min(
                        max(flip_negative_count / flip_positive_count, 1.0),
                        SPECIALIST_FLIP_CLASSIFIER_POSITIVE_WEIGHT_CAP,
                    )
                flip_sample_weight = np.where(flip_open_target.eq(1), flip_positive_weight, 1.0)
                flip_classifier.fit(flip_x_train, flip_open_target, estimator__sample_weight=flip_sample_weight)
                flip_open_probability = flip_classifier.predict_proba(x_target.loc[origin_frame.index[origin_flip_mask]])[:, 1]
                open_probability[origin_flip_mask] = np.maximum(open_probability[origin_flip_mask], flip_open_probability)

        positive_train_mask = train_frame["specialist_open_target"].eq(1)
        positive_train = train_frame.loc[positive_train_mask].copy()
        predicted_ratio = np.zeros(len(target_frame), dtype=float)
        if not positive_train.empty:
            positive_ratio = positive_train["specialist_ratio_target"]
            x_positive = feature_frame.loc[positive_train.index]
            predicted_ratio = np.full(len(target_frame), float(positive_ratio.mean()), dtype=float)
            if len(positive_train) >= 2 and positive_ratio.nunique(dropna=False) >= 2:
                regressor = _build_specialist_pipeline("regressor")
                regressor.fit(x_positive, positive_ratio)
                predicted_ratio = regressor.predict(x_target)
                basis_parts.append("hybrid_ratio")
            else:
                basis_parts.append("constant_ratio")
        else:
            basis_parts.append("no_positive_history")

        flip_positive_train = flip_train_frame[flip_train_frame["specialist_open_target"].eq(1)].copy()
        if origin_flip_mask.any() and len(flip_positive_train) >= 2:
            flip_positive_ratio = flip_positive_train["specialist_ratio_target"]
            if flip_positive_ratio.nunique(dropna=False) >= 2:
                flip_regressor = _build_specialist_pipeline("regressor")
                flip_regressor.fit(feature_frame.loc[flip_positive_train.index], flip_positive_ratio)
                flip_ratio = flip_regressor.predict(x_target.loc[origin_frame.index[origin_flip_mask]])
                predicted_ratio[origin_flip_mask] = np.maximum(predicted_ratio[origin_flip_mask], flip_ratio)

        scoped.loc[origin_frame.index, "training_sample_count"] = train_count
        scoped.loc[origin_frame.index, "prediction_basis"] = "_".join(basis_parts)
        scoped.loc[origin_frame.index, "predicted_open_probability"] = open_probability
        scoped.loc[origin_frame.index, "predicted_ratio"] = predicted_ratio
        if origin_flip_mask.any():
            scoped.loc[origin_frame.index[origin_flip_mask], "prediction_basis"] = _append_prediction_basis_suffix(
                scoped.loc[origin_frame.index[origin_flip_mask], "prediction_basis"],
                "flip_open_specialist",
            ).values

    if flip_suppressed_mask.any():
        scoped.loc[flip_suppressed_mask, "predicted_open_probability"] = 0.0
        scoped.loc[flip_suppressed_mask, "predicted_ratio"] = 0.0
        scoped.loc[flip_suppressed_mask, "prediction_basis"] = _append_prediction_basis_suffix(
            scoped.loc[flip_suppressed_mask, "prediction_basis"],
            "suppressed_blocked_itl_weaker_forward",
        ).values

    scoped["predicted_open_probability"] = pd.to_numeric(
        scoped["predicted_open_probability"],
        errors="coerce",
    ).clip(lower=0.0, upper=1.0)
    scoped["predicted_ratio"] = pd.to_numeric(scoped["predicted_ratio"], errors="coerce").clip(lower=0.0, upper=1.0)
    scoped["predicted_opportunity_deliverable_mwh"] = (
        scoped["predicted_open_probability"]
        * scoped["predicted_ratio"]
        * openable_potential
    ).clip(lower=0.0)
    scoped["predicted_opportunity_deliverable_mwh"] = np.minimum(
        scoped["predicted_opportunity_deliverable_mwh"],
        openable_potential,
    )
    scoped = _apply_gb_nl_specialist_flip_opening_guardrail(scoped)

    return _finalize_prediction_frame(
        scoped,
        model_key=MODEL_GB_NL_REVIEWED_SPECIALIST_V3,
        split_strategy=SPLIT_STRATEGY_GB_NL_REVIEWED_SPECIALIST,
        source_lineage="fact_curtailment_opportunity_hourly|walk_forward_gb_nl_reviewed_specialist",
    )


def build_fact_backtest_prediction_hourly(
    fact_curtailment_opportunity_hourly: pd.DataFrame,
    model_key: str = DEFAULT_BACKTEST_MODEL_KEY,
    forecast_horizons: Iterable[int] | None = None,
    historical_fact_curtailment_opportunity_hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if fact_curtailment_opportunity_hourly is None or fact_curtailment_opportunity_hourly.empty:
        return _empty_backtest_prediction_frame()
    if model_key not in VALID_BACKTEST_MODEL_KEYS:
        raise ValueError(
            f"unsupported backtest model '{model_key}'. Expected one of: {', '.join(sorted(VALID_BACKTEST_MODEL_KEYS))}"
        )

    frame = _prepare_backtest_input(fact_curtailment_opportunity_hourly)
    historical_frame = (
        _prepare_backtest_input(historical_fact_curtailment_opportunity_hourly)
        if historical_fact_curtailment_opportunity_hourly is not None
        and not historical_fact_curtailment_opportunity_hourly.empty
        else pd.DataFrame()
    )
    horizon_frames = []
    for forecast_horizon_hours in coerce_forecast_horizons(forecast_horizons):
        origin_source_frame = (
            pd.concat([historical_frame, frame], ignore_index=True, sort=False)
            if not historical_frame.empty
            else frame
        )
        horizon_frame = _build_horizon_example_frame(
            frame,
            forecast_horizon_hours,
            origin_source_frame=origin_source_frame,
        )
        historical_horizon_frame = pd.DataFrame()
        if not historical_frame.empty:
            historical_horizon_frame = _build_horizon_example_frame(
                historical_frame,
                forecast_horizon_hours,
                origin_source_frame=historical_frame,
            )
            min_target_origin = pd.to_datetime(
                horizon_frame.get("forecast_origin_utc", pd.Series(pd.NaT, index=horizon_frame.index)),
                utc=True,
                errors="coerce",
            ).min()
            if pd.notna(min_target_origin):
                historical_horizon_frame = historical_horizon_frame[
                    pd.to_datetime(
                        historical_horizon_frame.get(
                            "forecast_origin_utc",
                            pd.Series(pd.NaT, index=historical_horizon_frame.index),
                        ),
                        utc=True,
                        errors="coerce",
                    ).lt(min_target_origin)
                ].copy()
        if model_key == MODEL_GROUP_MEAN_NOTICE_V1:
            horizon_frames.append(_build_group_mean_backtest(horizon_frame))
        elif model_key == MODEL_POTENTIAL_RATIO_V2:
            horizon_frames.append(
                _build_potential_ratio_backtest(
                    horizon_frame,
                    history_frame=historical_horizon_frame,
                )
            )
        elif model_key == MODEL_GB_NL_REVIEWED_SPECIALIST_V3:
            horizon_frames.append(_build_gb_nl_reviewed_specialist_v3_backtest(horizon_frame))
        else:
            raise ValueError(f"unsupported backtest model '{model_key}'")
    non_empty_horizon_frames = [horizon_frame for horizon_frame in horizon_frames if not horizon_frame.empty]
    if not non_empty_horizon_frames:
        return _empty_backtest_prediction_frame()
    return pd.concat(non_empty_horizon_frames, ignore_index=True)


def _model_selection_to_keys(model_key: str) -> list[str]:
    if model_key == "all":
        return [
            MODEL_GROUP_MEAN_NOTICE_V1,
            MODEL_POTENTIAL_RATIO_V2,
            MODEL_GB_NL_REVIEWED_SPECIALIST_V3,
        ]
    if model_key in VALID_BACKTEST_MODEL_KEYS:
        return [model_key]
    raise ValueError(
        f"unsupported backtest model selection '{model_key}'. Expected one of: {', '.join(sorted(VALID_BACKTEST_MODEL_SELECTIONS))}"
    )


def summarize_backtest_prediction_hourly(fact_backtest_prediction_hourly: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_key",
        "forecast_horizon_hours",
        "forecast_horizon_label",
        "row_count",
        "eligible_row_count",
        "mae_opportunity_deliverable_mwh",
        "bias_opportunity_deliverable_mwh",
        "mae_opportunity_gross_value_eur",
        "bias_opportunity_gross_value_eur",
    ]
    if fact_backtest_prediction_hourly is None or fact_backtest_prediction_hourly.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (model_key, forecast_horizon_hours), model_frame in fact_backtest_prediction_hourly.groupby(
        ["model_key", "forecast_horizon_hours"], dropna=False
    ):
        eligible = model_frame[model_frame["prediction_eligible_flag"].fillna(False)].copy()
        rows.append(
            {
                "model_key": model_key,
                "forecast_horizon_hours": forecast_horizon_hours,
                "forecast_horizon_label": model_frame["forecast_horizon_label"].iloc[0],
                "row_count": len(model_frame),
                "eligible_row_count": len(eligible),
                "mae_opportunity_deliverable_mwh": float(
                    pd.to_numeric(eligible["opportunity_deliverable_abs_error_mwh"], errors="coerce").mean()
                )
                if not eligible.empty
                else np.nan,
                "bias_opportunity_deliverable_mwh": float(
                    pd.to_numeric(eligible["opportunity_deliverable_residual_mwh"], errors="coerce").mean()
                )
                if not eligible.empty
                else np.nan,
                "mae_opportunity_gross_value_eur": float(
                    pd.to_numeric(eligible["opportunity_gross_value_abs_error_eur"], errors="coerce").mean()
                )
                if not eligible.empty
                else np.nan,
                "bias_opportunity_gross_value_eur": float(
                    pd.to_numeric(eligible["opportunity_gross_value_residual_eur"], errors="coerce").mean()
                )
                if not eligible.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["model_key", "forecast_horizon_hours"]).reset_index(drop=True)


def _summary_focus_area(slice_dimension: str, slice_value: object) -> str:
    if slice_dimension == "route_delivery_tier" and slice_value in {"reviewed", "capacity_unknown"}:
        return str(slice_value)
    if slice_dimension == "internal_transfer_evidence_tier":
        if slice_value == "gb_topology_transfer_gate_proxy":
            return "proxy_internal_transfer"
        if slice_value not in {"<NA>", pd.NA}:
            return "reviewed_internal_transfer"
    if slice_dimension == "internal_transfer_gate_state" and str(slice_value).startswith("blocked_reviewed"):
        return "blocked_internal_reviewed"
    if slice_dimension == "upstream_market_state" and slice_value != "no_upstream_feed":
        return "upstream_market_state"
    if slice_dimension == "system_balance_state" and slice_value != "no_public_system_balance":
        return "system_balance_state"
    if slice_dimension == "connector_notice_market_state" and slice_value != "no_public_connector_restriction":
        return "connector_restriction_state"
    if slice_dimension == "hub_key" and slice_value in {"ifa", "ifa2", "eleclink"}:
        return "gb_fr_connector_route"
    return "general"


def _build_summary_row(
    model_key: str,
    forecast_horizon_hours: int,
    forecast_horizon_label: str,
    slice_dimension: str,
    slice_value: object,
    frame: pd.DataFrame,
) -> dict:
    eligible = frame[frame["prediction_eligible_flag"].fillna(False)].copy()
    return {
        "model_key": model_key,
        "forecast_horizon_hours": forecast_horizon_hours,
        "forecast_horizon_label": forecast_horizon_label,
        "slice_dimension": slice_dimension,
        "slice_value": slice_value,
        "error_focus_area": _summary_focus_area(slice_dimension, slice_value),
        "error_reduction_priority_rank": np.nan,
        "window_start_utc": frame["interval_start_utc"].min(),
        "window_end_utc": frame["interval_end_utc"].max(),
        "row_count": len(frame),
        "eligible_row_count": len(eligible),
        "prediction_eligibility_rate": float(len(eligible) / len(frame)) if len(frame) else np.nan,
        "actual_opportunity_deliverable_mean_mwh": float(
            pd.to_numeric(eligible["actual_opportunity_deliverable_mwh"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "predicted_opportunity_deliverable_mean_mwh": float(
            pd.to_numeric(eligible["predicted_opportunity_deliverable_mwh"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "mae_opportunity_deliverable_mwh": float(
            pd.to_numeric(eligible["opportunity_deliverable_abs_error_mwh"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "bias_opportunity_deliverable_mwh": float(
            pd.to_numeric(eligible["opportunity_deliverable_residual_mwh"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "actual_opportunity_gross_value_mean_eur": float(
            pd.to_numeric(eligible["actual_opportunity_gross_value_eur"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "predicted_opportunity_gross_value_mean_eur": float(
            pd.to_numeric(eligible["predicted_opportunity_gross_value_eur"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "mae_opportunity_gross_value_eur": float(
            pd.to_numeric(eligible["opportunity_gross_value_abs_error_eur"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "bias_opportunity_gross_value_eur": float(
            pd.to_numeric(eligible["opportunity_gross_value_residual_eur"], errors="coerce").mean()
        )
        if not eligible.empty
        else np.nan,
        "source_lineage": "fact_backtest_prediction_hourly",
    }


def build_fact_backtest_summary_slice(fact_backtest_prediction_hourly: pd.DataFrame) -> pd.DataFrame:
    if fact_backtest_prediction_hourly is None or fact_backtest_prediction_hourly.empty:
        return _empty_backtest_summary_slice_frame()

    rows = []
    for (model_key, forecast_horizon_hours), model_frame in fact_backtest_prediction_hourly.groupby(
        ["model_key", "forecast_horizon_hours"], dropna=False
    ):
        forecast_horizon_label = model_frame["forecast_horizon_label"].iloc[0]
        rows.append(_build_summary_row(model_key, forecast_horizon_hours, forecast_horizon_label, "all", "all", model_frame))
        for slice_dimension in SUMMARY_SLICE_DIMENSIONS:
            if slice_dimension == "all" or slice_dimension not in model_frame.columns:
                continue
            for slice_value, slice_frame in model_frame.groupby(slice_dimension, dropna=False):
                rows.append(
                    _build_summary_row(
                        model_key,
                        forecast_horizon_hours,
                        forecast_horizon_label,
                        slice_dimension,
                        "<NA>" if pd.isna(slice_value) else slice_value,
                        slice_frame,
                    )
                )

    summary = pd.DataFrame(rows, columns=_empty_backtest_summary_slice_frame().columns)
    focus_mask = summary["error_focus_area"].ne("general")
    summary.loc[focus_mask, "error_reduction_priority_rank"] = summary[focus_mask].groupby(
        ["model_key", "forecast_horizon_hours"], dropna=False
    )["mae_opportunity_gross_value_eur"].rank(method="first", ascending=False)
    return summary.sort_values(
        ["model_key", "forecast_horizon_hours", "slice_dimension", "slice_value"]
    ).reset_index(drop=True)


def build_fact_backtest_top_error_hourly(fact_backtest_prediction_hourly: pd.DataFrame) -> pd.DataFrame:
    if fact_backtest_prediction_hourly is None or fact_backtest_prediction_hourly.empty:
        return _empty_backtest_top_error_frame()

    eligible = fact_backtest_prediction_hourly[
        fact_backtest_prediction_hourly["prediction_eligible_flag"].fillna(False)
    ].copy()
    if eligible.empty:
        return _empty_backtest_top_error_frame()

    eligible["deliverable_abs_error_rank"] = eligible.groupby(["model_key", "forecast_horizon_hours"], dropna=False)[
        "opportunity_deliverable_abs_error_mwh"
    ].rank(method="first", ascending=False)
    eligible["gross_value_abs_error_rank"] = eligible.groupby(["model_key", "forecast_horizon_hours"], dropna=False)[
        "opportunity_gross_value_abs_error_eur"
    ].rank(method="first", ascending=False)
    eligible["error_focus_area"] = "general"
    eligible.loc[eligible["route_delivery_tier"].eq("reviewed"), "error_focus_area"] = "reviewed"
    eligible.loc[eligible["route_delivery_tier"].eq("capacity_unknown"), "error_focus_area"] = "capacity_unknown"
    eligible.loc[
        eligible["internal_transfer_evidence_tier"].ne("gb_topology_transfer_gate_proxy"),
        "error_focus_area",
    ] = "reviewed_internal_transfer"
    eligible.loc[
        eligible["internal_transfer_evidence_tier"].eq("gb_topology_transfer_gate_proxy"),
        "error_focus_area",
    ] = "proxy_internal_transfer"
    eligible.loc[
        eligible["internal_transfer_gate_state"].fillna("").astype(str).str.startswith("blocked_reviewed"),
        "error_focus_area",
    ] = "blocked_internal_reviewed"
    eligible.loc[
        eligible["upstream_market_state"].fillna("no_upstream_feed").ne("no_upstream_feed"),
        "error_focus_area",
    ] = "upstream_market_state"
    eligible.loc[
        eligible["system_balance_state"].fillna("no_public_system_balance").ne("no_public_system_balance"),
        "error_focus_area",
    ] = "system_balance_state"
    eligible.loc[
        eligible["connector_notice_market_state"].ne("no_public_connector_restriction"),
        "error_focus_area",
    ] = "connector_restriction_state"
    eligible.loc[eligible["hub_key"].isin(["ifa", "ifa2", "eleclink"]), "error_focus_area"] = "gb_fr_connector_route"
    eligible.loc[
        eligible["internal_transfer_evidence_tier"].eq("gb_topology_transfer_gate_proxy"),
        "error_focus_area",
    ] = "proxy_internal_transfer"
    eligible.loc[
        eligible["internal_transfer_evidence_tier"].ne("gb_topology_transfer_gate_proxy"),
        "error_focus_area",
    ] = "reviewed_internal_transfer"
    eligible.loc[
        eligible["internal_transfer_gate_state"].fillna("").astype(str).str.startswith("blocked_reviewed"),
        "error_focus_area",
    ] = "blocked_internal_reviewed"
    eligible = eligible.sort_values(
        [
            "model_key",
            "forecast_horizon_hours",
            "opportunity_gross_value_abs_error_eur",
            "opportunity_deliverable_abs_error_mwh",
            "interval_start_utc",
            "cluster_key",
            "route_name",
            "hub_key",
        ],
        ascending=[True, True, False, False, True, True, True, True],
    ).reset_index(drop=True)
    eligible["top_error_rank"] = eligible.groupby(["model_key", "forecast_horizon_hours"], dropna=False).cumcount() + 1

    keep_columns = list(_empty_backtest_top_error_frame().columns)
    for column in keep_columns:
        if column not in eligible.columns:
            eligible[column] = pd.NA
    return eligible[keep_columns]


def build_fact_drift_window(fact_backtest_prediction_hourly: pd.DataFrame) -> pd.DataFrame:
    if fact_backtest_prediction_hourly is None or fact_backtest_prediction_hourly.empty:
        return _empty_drift_window_frame()

    frame = fact_backtest_prediction_hourly.copy()
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["interval_end_utc"] = pd.to_datetime(frame["interval_end_utc"], utc=True, errors="coerce")
    frame["window_date"] = frame["interval_start_utc"].dt.floor("d")

    scope_specs = (
        ("global_daily", []),
        ("route_daily", ["route_name"]),
        ("cluster_daily", ["cluster_key"]),
    )
    rows = []
    for drift_scope, scope_columns in scope_specs:
        group_columns = ["model_key", "forecast_horizon_hours", "forecast_horizon_label", "window_date", *scope_columns]
        for group_keys, window_frame in frame.groupby(group_columns, dropna=False):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            key_iter = iter(group_keys)
            model_key = next(key_iter)
            forecast_horizon_hours = next(key_iter)
            forecast_horizon_label = next(key_iter)
            window_date = next(key_iter)
            scope_values = {column: next(key_iter) for column in scope_columns}
            eligible = window_frame[window_frame["prediction_eligible_flag"].fillna(False)].copy()
            route_tier = window_frame["route_delivery_tier"].fillna("unknown")
            internal_tier = window_frame["internal_transfer_evidence_tier"].fillna("gb_topology_transfer_gate_proxy")
            internal_gate = window_frame["internal_transfer_gate_state"].fillna("capacity_unknown_reachable")
            notice_state = window_frame["connector_notice_market_state"].fillna("no_public_connector_restriction")
            system_balance_state = window_frame["system_balance_state"].fillna("no_public_system_balance")
            system_balance_known = _coerce_bool_series(
                window_frame.get("feature_system_balance_known_flag_asof", window_frame.get("system_balance_known_flag", pd.Series(False, index=window_frame.index))),
                default=False,
            )
            source_tier = window_frame["curtailment_source_tier"].fillna("unknown")
            rows.append(
                {
                    "model_key": model_key,
                    "forecast_horizon_hours": forecast_horizon_hours,
                    "forecast_horizon_label": forecast_horizon_label,
                    "window_start_utc": window_frame["interval_start_utc"].min(),
                    "window_end_utc": window_frame["interval_end_utc"].max(),
                    "window_date": window_date,
                    "drift_scope": drift_scope,
                    "cluster_key": scope_values.get("cluster_key", pd.NA),
                    "route_name": scope_values.get("route_name", pd.NA),
                    "row_count": len(window_frame),
                    "eligible_row_count": len(eligible),
                    "reviewed_route_share": float(route_tier.eq("reviewed").mean()),
                    "capacity_unknown_route_share": float(route_tier.eq("capacity_unknown").mean()),
                    "reviewed_internal_transfer_share": float(
                        internal_tier.ne("gb_topology_transfer_gate_proxy").mean()
                    ),
                    "proxy_internal_transfer_share": float(
                        internal_tier.eq("gb_topology_transfer_gate_proxy").mean()
                    ),
                    "blocked_internal_reviewed_share": float(
                        internal_gate.astype(str).str.startswith("blocked_reviewed").mean()
                    ),
                    "known_connector_restriction_share": float(
                        notice_state.ne("no_public_connector_restriction").mean()
                    ),
                    "system_balance_stress_share": float(
                        system_balance_state.isin(
                            ["tight_margin", "tight_margin_and_active_imbalance", "active_imbalance", "moderate_imbalance"]
                        ).mean()
                    ),
                    "system_balance_known_share": float(system_balance_known.mean()),
                    "truth_backed_curtailment_share": float(source_tier.ne("regional_proxy").mean()),
                    "actual_opportunity_deliverable_mean_mwh": float(
                        pd.to_numeric(eligible["actual_opportunity_deliverable_mwh"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "predicted_opportunity_deliverable_mean_mwh": float(
                        pd.to_numeric(eligible["predicted_opportunity_deliverable_mwh"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "residual_bias_mwh": float(
                        pd.to_numeric(eligible["opportunity_deliverable_residual_mwh"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "residual_mae_mwh": float(
                        pd.to_numeric(eligible["opportunity_deliverable_abs_error_mwh"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "actual_opportunity_gross_value_mean_eur": float(
                        pd.to_numeric(eligible["actual_opportunity_gross_value_eur"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "predicted_opportunity_gross_value_mean_eur": float(
                        pd.to_numeric(eligible["predicted_opportunity_gross_value_eur"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "residual_bias_eur": float(
                        pd.to_numeric(eligible["opportunity_gross_value_residual_eur"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "residual_mae_eur": float(
                        pd.to_numeric(eligible["opportunity_gross_value_abs_error_eur"], errors="coerce").mean()
                    )
                    if not eligible.empty
                    else np.nan,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            )

    drift = pd.DataFrame(rows)
    drift = drift.sort_values(["model_key", "forecast_horizon_hours", "drift_scope", "route_name", "cluster_key", "window_start_utc"]).reset_index(drop=True)
    drift["feature_drift_score"] = np.nan
    drift["target_drift_score"] = np.nan
    drift["residual_drift_score"] = np.nan
    drift["drift_state"] = "warmup"

    for _, model_frame in drift.groupby(
        ["model_key", "forecast_horizon_hours", "drift_scope", "route_name", "cluster_key"],
        dropna=False,
    ):
        previous = model_frame.shift(1)
        current_actual_mean = pd.to_numeric(
            model_frame["actual_opportunity_deliverable_mean_mwh"],
            errors="coerce",
        ).abs()
        previous_actual_mean = pd.to_numeric(
            previous["actual_opportunity_deliverable_mean_mwh"],
            errors="coerce",
        ).abs()
        current_predicted_mean = pd.to_numeric(
            model_frame["predicted_opportunity_deliverable_mean_mwh"],
            errors="coerce",
        ).abs()
        previous_predicted_mean = pd.to_numeric(
            previous["predicted_opportunity_deliverable_mean_mwh"],
            errors="coerce",
        ).abs()
        current_residual_mae = pd.to_numeric(model_frame["residual_mae_mwh"], errors="coerce").abs()
        previous_residual_mae = pd.to_numeric(previous["residual_mae_mwh"], errors="coerce").abs()
        current_reviewed_internal_share = pd.to_numeric(
            model_frame["reviewed_internal_transfer_share"],
            errors="coerce",
        )
        current_proxy_share = pd.to_numeric(
            model_frame["proxy_internal_transfer_share"],
            errors="coerce",
        )
        current_capacity_unknown_share = pd.to_numeric(
            model_frame["capacity_unknown_route_share"],
            errors="coerce",
        )
        previous_eligible_row_count = pd.to_numeric(previous["eligible_row_count"], errors="coerce")
        previous_reviewed_internal_share = pd.to_numeric(
            previous["reviewed_internal_transfer_share"],
            errors="coerce",
        )
        previous_proxy_share = pd.to_numeric(
            previous["proxy_internal_transfer_share"],
            errors="coerce",
        )
        previous_capacity_unknown_share = pd.to_numeric(
            previous["capacity_unknown_route_share"],
            errors="coerce",
        )
        current_activity_scale = pd.concat(
            [current_actual_mean, current_predicted_mean],
            axis=1,
        ).max(axis=1).clip(lower=REVIEWED_EVENT_TARGET_SHIFT_ACTIVITY_SCALE_FLOOR_MWH)
        previous_activity_scale = pd.concat(
            [previous_actual_mean, previous_predicted_mean],
            axis=1,
        ).max(axis=1).clip(lower=REVIEWED_EVENT_TARGET_SHIFT_ACTIVITY_SCALE_FLOOR_MWH)
        current_residual_ratio = current_residual_mae.fillna(np.inf) / current_activity_scale
        previous_residual_ratio = previous_residual_mae.fillna(np.inf) / previous_activity_scale
        feature_scores = pd.concat(
            [
                (model_frame["reviewed_route_share"] - previous["reviewed_route_share"]).abs(),
                (model_frame["capacity_unknown_route_share"] - previous["capacity_unknown_route_share"]).abs(),
                (model_frame["reviewed_internal_transfer_share"] - previous["reviewed_internal_transfer_share"]).abs(),
                (model_frame["proxy_internal_transfer_share"] - previous["proxy_internal_transfer_share"]).abs(),
                (model_frame["blocked_internal_reviewed_share"] - previous["blocked_internal_reviewed_share"]).abs(),
                (model_frame["known_connector_restriction_share"] - previous["known_connector_restriction_share"]).abs(),
                (model_frame["system_balance_stress_share"] - previous["system_balance_stress_share"]).abs(),
                (model_frame["system_balance_known_share"] - previous["system_balance_known_share"]).abs(),
                (model_frame["truth_backed_curtailment_share"] - previous["truth_backed_curtailment_share"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        target_scores = (
            (model_frame["actual_opportunity_deliverable_mean_mwh"] - previous["actual_opportunity_deliverable_mean_mwh"]).abs()
            / previous["actual_opportunity_deliverable_mean_mwh"].abs().clip(lower=1.0)
        )
        residual_scores = pd.concat(
            [
                (
                    (model_frame["residual_bias_mwh"] - previous["residual_bias_mwh"]).abs()
                    / previous["actual_opportunity_deliverable_mean_mwh"].abs().clip(lower=1.0)
                ),
                (
                    (model_frame["residual_mae_mwh"] - previous["residual_mae_mwh"]).abs()
                    / previous["residual_mae_mwh"].abs().clip(lower=1.0)
                ),
            ],
            axis=1,
        ).max(axis=1)

        drift.loc[model_frame.index, "feature_drift_score"] = feature_scores.values
        drift.loc[model_frame.index, "target_drift_score"] = target_scores.values
        drift.loc[model_frame.index, "residual_drift_score"] = residual_scores.values

        pass_mask = previous["window_start_utc"].notna()
        zero_activity_feature_only_mask = (
            pass_mask
            & feature_scores.ge(FEATURE_DRIFT_WARN_THRESHOLD)
            & target_scores.fillna(0.0).lt(TARGET_DRIFT_WARN_THRESHOLD)
            & residual_scores.fillna(0.0).lt(RESIDUAL_DRIFT_WARN_THRESHOLD)
            & current_actual_mean.fillna(np.inf)
            .le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & previous_actual_mean.fillna(np.inf)
            .le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & current_predicted_mean.fillna(np.inf)
            .le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & previous_predicted_mean.fillna(np.inf)
            .le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & current_residual_mae.fillna(np.inf)
            .le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & previous_residual_mae.fillna(np.inf)
            .le(ZERO_ACTIVITY_DRIFT_EPSILON)
        )
        zero_activity_warmup_mask = (
            pass_mask
            & previous_eligible_row_count.fillna(0.0).le(0.0)
            & current_actual_mean.fillna(np.inf).le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & current_predicted_mean.fillna(np.inf).le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & current_residual_mae.fillna(np.inf).le(ZERO_ACTIVITY_DRIFT_EPSILON)
        )
        reviewed_event_target_shift_mask = (
            pass_mask
            & pd.Series(drift_scope, index=model_frame.index).isin(["route_daily", "cluster_daily"])
            & previous_actual_mean.fillna(np.inf).le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & previous_predicted_mean.fillna(np.inf).le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & previous_residual_mae.fillna(np.inf).le(ZERO_ACTIVITY_DRIFT_EPSILON)
            & current_actual_mean.fillna(0.0).gt(ZERO_ACTIVITY_DRIFT_EPSILON)
            & current_predicted_mean.fillna(0.0).gt(ZERO_ACTIVITY_DRIFT_EPSILON)
            & current_reviewed_internal_share.fillna(0.0).ge(REVIEWED_EVENT_TARGET_SHIFT_MIN_REVIEWED_INTERNAL_SHARE)
            & current_proxy_share.fillna(1.0).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_PROXY_SHARE)
            & current_capacity_unknown_share.fillna(1.0).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_CAPACITY_UNKNOWN_SHARE)
            & current_residual_mae.fillna(np.inf).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_MAE_MWH)
            & current_residual_ratio.le(REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_RATIO)
        )
        reviewed_event_stable_shift_mask = (
            pass_mask
            & pd.Series(drift_scope, index=model_frame.index).isin(["route_daily", "cluster_daily"])
            & current_reviewed_internal_share.fillna(0.0).ge(REVIEWED_EVENT_TARGET_SHIFT_MIN_REVIEWED_INTERNAL_SHARE)
            & current_proxy_share.fillna(1.0).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_PROXY_SHARE)
            & current_capacity_unknown_share.fillna(1.0).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_CAPACITY_UNKNOWN_SHARE)
            & current_residual_mae.fillna(np.inf).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_MAE_MWH)
            & current_residual_ratio.le(REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_RATIO)
            & (
                previous_eligible_row_count.fillna(0.0).le(0.0)
                | (
                    previous_reviewed_internal_share.fillna(0.0).ge(
                        REVIEWED_EVENT_TARGET_SHIFT_MIN_REVIEWED_INTERNAL_SHARE
                    )
                    & previous_proxy_share.fillna(1.0).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_PROXY_SHARE)
                    & previous_capacity_unknown_share.fillna(1.0).le(
                        REVIEWED_EVENT_TARGET_SHIFT_MAX_CAPACITY_UNKNOWN_SHARE
                    )
                    & previous_residual_mae.fillna(np.inf).le(REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_MAE_MWH)
                    & previous_residual_ratio.le(REVIEWED_EVENT_TARGET_SHIFT_MAX_RESIDUAL_RATIO)
                )
            )
        )
        warn_mask = (
            (feature_scores >= FEATURE_DRIFT_WARN_THRESHOLD)
            | (target_scores >= TARGET_DRIFT_WARN_THRESHOLD)
            | (residual_scores >= RESIDUAL_DRIFT_WARN_THRESHOLD)
        ) & pass_mask & ~zero_activity_feature_only_mask & ~zero_activity_warmup_mask & ~reviewed_event_target_shift_mask & ~reviewed_event_stable_shift_mask
        drift.loc[model_frame.index[pass_mask], "drift_state"] = "pass"
        drift.loc[model_frame.index[warn_mask], "drift_state"] = "warn"

    keep_columns = list(_empty_drift_window_frame().columns)
    for column in keep_columns:
        if column not in drift.columns:
            drift[column] = pd.NA
    return drift[keep_columns]


def materialize_opportunity_backtest(
    output_dir: str | Path,
    fact_curtailment_opportunity_hourly: pd.DataFrame,
    model_key: str = DEFAULT_BACKTEST_MODEL_SELECTION,
    forecast_horizons: Iterable[int] | None = None,
    historical_fact_curtailment_opportunity_hourly: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    prediction_frames = [
        build_fact_backtest_prediction_hourly(
            fact_curtailment_opportunity_hourly,
            selected_model_key,
            forecast_horizons=forecast_horizons,
            historical_fact_curtailment_opportunity_hourly=historical_fact_curtailment_opportunity_hourly,
        )
        for selected_model_key in _model_selection_to_keys(model_key)
    ]
    non_empty_prediction_frames = [prediction_frame for prediction_frame in prediction_frames if not prediction_frame.empty]
    predictions = (
        pd.concat(non_empty_prediction_frames, ignore_index=True)
        if non_empty_prediction_frames
        else _empty_backtest_prediction_frame()
    )
    summary_slice = build_fact_backtest_summary_slice(predictions)
    top_error = build_fact_backtest_top_error_hourly(predictions)
    drift_window = build_fact_drift_window(predictions)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / f"{BACKTEST_PREDICTION_TABLE}.csv", index=False)
    summary_slice.to_csv(output_path / f"{BACKTEST_SUMMARY_SLICE_TABLE}.csv", index=False)
    top_error.to_csv(output_path / f"{BACKTEST_TOP_ERROR_TABLE}.csv", index=False)
    drift_window.to_csv(output_path / f"{DRIFT_WINDOW_TABLE}.csv", index=False)

    return {
        BACKTEST_PREDICTION_TABLE: predictions,
        BACKTEST_SUMMARY_SLICE_TABLE: summary_slice,
        BACKTEST_TOP_ERROR_TABLE: top_error,
        DRIFT_WINDOW_TABLE: drift_window,
    }
