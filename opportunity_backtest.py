from __future__ import annotations

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
VALID_BACKTEST_MODEL_KEYS = {MODEL_GROUP_MEAN_NOTICE_V1, MODEL_POTENTIAL_RATIO_V2}
VALID_BACKTEST_MODEL_SELECTIONS = {"all", *VALID_BACKTEST_MODEL_KEYS}

DEFAULT_BACKTEST_MODEL_KEY = MODEL_GROUP_MEAN_NOTICE_V1
DEFAULT_BACKTEST_MODEL_SELECTION = "all"
DEFAULT_FORECAST_HORIZON_HOURS = (1, 6, 24, 168)

SPLIT_STRATEGY_GROUP_MEAN = "walk_forward_group_mean"
SPLIT_STRATEGY_POTENTIAL_RATIO = "walk_forward_potential_ratio"

EXACT_MIN_HISTORY = 1
CLUSTER_ROUTE_MIN_HISTORY = 3
ROUTE_STATE_MIN_HISTORY = 6
GLOBAL_MIN_HISTORY = 24

RATIO_EXACT_MIN_HISTORY = 1
RATIO_ROUTE_NOTICE_MIN_HISTORY = 3
RATIO_ROUTE_TIER_MIN_HISTORY = 6
RATIO_GLOBAL_MIN_HISTORY = 24

FEATURE_DRIFT_WARN_THRESHOLD = 0.15
TARGET_DRIFT_WARN_THRESHOLD = 0.50
RESIDUAL_DRIFT_WARN_THRESHOLD = 0.50

SUMMARY_SLICE_DIMENSIONS = (
    "all",
    "cluster_key",
    "hub_key",
    "route_name",
    "route_delivery_tier",
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
            "hub_key",
            "hub_label",
            "route_name",
            "route_label",
            "route_border_key",
            "route_delivery_tier",
            "connector_notice_market_state",
            "curtailment_source_tier",
            "model_key",
            "split_strategy",
            "feature_hour_of_day",
            "feature_day_of_week",
            "feature_origin_hour_of_day",
            "feature_origin_day_of_week",
            "feature_route_delivery_tier_asof",
            "feature_connector_notice_market_state_asof",
            "feature_curtailment_selected_mwh_asof",
            "feature_deliverable_mw_proxy_asof",
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
            "hub_key",
            "hub_label",
            "route_name",
            "route_label",
            "route_border_key",
            "route_delivery_tier",
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
            "known_connector_restriction_share",
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
    grouped = frame.groupby(keys, dropna=False)
    prior_count = grouped.cumcount()
    prior_sum = grouped[target_col].cumsum() - frame[target_col].fillna(0.0)
    result = pd.DataFrame(index=frame.index)
    result[f"{prefix}_prior_count"] = prior_count
    result[f"{prefix}_prior_mean"] = np.where(prior_count > 0, prior_sum / prior_count, np.nan)
    return result


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
    frame["route_delivery_tier"] = frame["route_delivery_tier"].fillna("unknown")
    frame["connector_notice_market_state"] = frame["connector_notice_market_state"].fillna(
        "no_public_connector_restriction"
    )
    frame["curtailment_source_tier"] = frame["curtailment_source_tier"].fillna("unknown")
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
    return frame.sort_values(["interval_start_utc", "cluster_key", "route_name", "hub_key"]).reset_index(drop=True)


def _build_horizon_example_frame(frame: pd.DataFrame, forecast_horizon_hours: int) -> pd.DataFrame:
    target = frame.copy()
    target["forecast_horizon_hours"] = forecast_horizon_hours
    target["forecast_horizon_label"] = f"t_plus_{forecast_horizon_hours}h"
    target["forecast_origin_utc"] = target["interval_start_utc"] - pd.Timedelta(hours=forecast_horizon_hours)
    target["feature_asof_utc"] = target["forecast_origin_utc"]

    origin = frame[
        [
            "cluster_key",
            "route_name",
            "hub_key",
            "interval_start_utc",
            "feature_hour_of_day",
            "feature_day_of_week",
            "route_delivery_tier",
            "connector_notice_market_state",
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
            "route_delivery_tier": "feature_route_delivery_tier_asof",
            "connector_notice_market_state": "feature_connector_notice_market_state_asof",
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
    global_count = pd.Series(np.arange(len(frame)), index=frame.index)
    global_sum = frame["actual_opportunity_deliverable_mwh"].cumsum() - frame["actual_opportunity_deliverable_mwh"]
    global_mean = np.where(global_count > 0, global_sum / global_count, np.nan)

    result = pd.concat([frame.copy(), exact, cluster_route, route_state], axis=1)
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
    global_mask = origin_available & ~exact_mask & ~cluster_route_mask & ~route_state_mask & (global_count >= GLOBAL_MIN_HISTORY)

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
    result.loc[global_mask, "training_sample_count"] = global_count[global_mask]
    result.loc[global_mask, "predicted_opportunity_deliverable_mwh"] = global_mean[global_mask]

    return _finalize_prediction_frame(
        result,
        model_key=MODEL_GROUP_MEAN_NOTICE_V1,
        split_strategy=SPLIT_STRATEGY_GROUP_MEAN,
        source_lineage="fact_curtailment_opportunity_hourly|walk_forward_group_mean",
    )


def _build_potential_ratio_backtest(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
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
    global_count = pd.Series(np.arange(len(frame)), index=frame.index)
    global_sum = frame["realized_origin_potential_ratio"].cumsum() - frame["realized_origin_potential_ratio"]
    global_mean = np.where(global_count > 0, global_sum / global_count, np.nan)

    result = pd.concat([frame.copy(), exact, route_notice, route_tier], axis=1)
    result["prediction_basis"] = pd.NA
    result["training_sample_count"] = 0
    result["predicted_ratio"] = np.nan

    origin_available = result["feature_origin_hour_of_day"].notna()
    exact_mask = origin_available & (result["ratio_exact_prior_count"] >= RATIO_EXACT_MIN_HISTORY)
    route_notice_mask = origin_available & ~exact_mask & (
        result["ratio_route_notice_prior_count"] >= RATIO_ROUTE_NOTICE_MIN_HISTORY
    )
    route_tier_mask = origin_available & ~exact_mask & ~route_notice_mask & (
        result["ratio_route_tier_prior_count"] >= RATIO_ROUTE_TIER_MIN_HISTORY
    )
    global_mask = origin_available & ~exact_mask & ~route_notice_mask & ~route_tier_mask & (
        global_count >= RATIO_GLOBAL_MIN_HISTORY
    )

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
    result.loc[global_mask, "training_sample_count"] = global_count[global_mask]
    result.loc[global_mask, "predicted_ratio"] = global_mean[global_mask]

    result["predicted_ratio"] = pd.to_numeric(result["predicted_ratio"], errors="coerce").clip(lower=0.0, upper=10.0)
    result["predicted_opportunity_deliverable_mwh"] = result["origin_potential_opportunity_mwh"] * result["predicted_ratio"]

    return _finalize_prediction_frame(
        result,
        model_key=MODEL_POTENTIAL_RATIO_V2,
        split_strategy=SPLIT_STRATEGY_POTENTIAL_RATIO,
        source_lineage="fact_curtailment_opportunity_hourly|walk_forward_potential_ratio",
    )


def build_fact_backtest_prediction_hourly(
    fact_curtailment_opportunity_hourly: pd.DataFrame,
    model_key: str = DEFAULT_BACKTEST_MODEL_KEY,
    forecast_horizons: Iterable[int] | None = None,
) -> pd.DataFrame:
    if fact_curtailment_opportunity_hourly is None or fact_curtailment_opportunity_hourly.empty:
        return _empty_backtest_prediction_frame()
    if model_key not in VALID_BACKTEST_MODEL_KEYS:
        raise ValueError(
            f"unsupported backtest model '{model_key}'. Expected one of: {', '.join(sorted(VALID_BACKTEST_MODEL_KEYS))}"
        )

    frame = _prepare_backtest_input(fact_curtailment_opportunity_hourly)
    horizon_frames = []
    for forecast_horizon_hours in coerce_forecast_horizons(forecast_horizons):
        horizon_frame = _build_horizon_example_frame(frame, forecast_horizon_hours)
        if model_key == MODEL_GROUP_MEAN_NOTICE_V1:
            horizon_frames.append(_build_group_mean_backtest(horizon_frame))
        elif model_key == MODEL_POTENTIAL_RATIO_V2:
            horizon_frames.append(_build_potential_ratio_backtest(horizon_frame))
        else:
            raise ValueError(f"unsupported backtest model '{model_key}'")
    if not horizon_frames:
        return _empty_backtest_prediction_frame()
    return pd.concat(horizon_frames, ignore_index=True)


def _model_selection_to_keys(model_key: str) -> list[str]:
    if model_key == "all":
        return [MODEL_GROUP_MEAN_NOTICE_V1, MODEL_POTENTIAL_RATIO_V2]
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
        eligible["connector_notice_market_state"].ne("no_public_connector_restriction"),
        "error_focus_area",
    ] = "connector_restriction_state"
    eligible.loc[eligible["hub_key"].isin(["ifa", "ifa2", "eleclink"]), "error_focus_area"] = "gb_fr_connector_route"
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
            notice_state = window_frame["connector_notice_market_state"].fillna("no_public_connector_restriction")
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
                    "known_connector_restriction_share": float(
                        notice_state.ne("no_public_connector_restriction").mean()
                    ),
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
        feature_scores = pd.concat(
            [
                (model_frame["reviewed_route_share"] - previous["reviewed_route_share"]).abs(),
                (model_frame["capacity_unknown_route_share"] - previous["capacity_unknown_route_share"]).abs(),
                (model_frame["known_connector_restriction_share"] - previous["known_connector_restriction_share"]).abs(),
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
        warn_mask = (
            (feature_scores >= FEATURE_DRIFT_WARN_THRESHOLD)
            | (target_scores >= TARGET_DRIFT_WARN_THRESHOLD)
            | (residual_scores >= RESIDUAL_DRIFT_WARN_THRESHOLD)
        ) & pass_mask
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
) -> Dict[str, pd.DataFrame]:
    prediction_frames = [
        build_fact_backtest_prediction_hourly(
            fact_curtailment_opportunity_hourly,
            selected_model_key,
            forecast_horizons=forecast_horizons,
        )
        for selected_model_key in _model_selection_to_keys(model_key)
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else _empty_backtest_prediction_frame()
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
