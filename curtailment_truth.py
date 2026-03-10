from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from bmu_availability import (
    build_fact_bmu_availability_half_hourly,
    fetch_remit_event_detail,
    fetch_uou2t14d_summary,
)
from bmu_dispatch import (
    build_fact_bmu_acceptance_event,
    build_fact_bmu_dispatch_acceptance_half_hourly,
    fetch_boalf_acceptances,
)
from bmu_generation import (
    build_dim_bmu_asset,
    build_fact_bmu_generation_half_hourly,
    fetch_b1610_generation,
    fetch_bmu_reference_all,
)
from bmu_physical import build_fact_bmu_physical_position_half_hourly, fetch_balancing_physical
from bmu_truth_utils import build_bmu_interval_spine
from curtailment_signals import fetch_constraint_daily
from weather_history import build_fact_weather_hourly


VALID_TRUTH_PROFILES = {"all", "precision", "research"}
WEATHER_METHODS = {
    "bmu_weather_power_curve",
    "cluster_weather_power_curve",
    "parent_region_weather_power_curve",
}


def _scheme_year_label(value: dt.date) -> str:
    start_year = value.year if value.month >= 4 else value.year - 1
    return f"{start_year}-{start_year + 1}"


def _fetch_constraint_daily_for_range(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    labels = []
    day = start_date
    while day <= end_date:
        label = _scheme_year_label(day)
        if label not in labels:
            labels.append(label)
        day += dt.timedelta(days=1)

    frames = []
    for label in labels:
        frame = fetch_constraint_daily(label)
        frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def _build_weather_curve(
    calibration_frame: pd.DataFrame,
    group_column: str,
    min_observations: int,
    curve_scope_type: str,
) -> pd.DataFrame:
    if group_column not in calibration_frame.columns:
        return pd.DataFrame()

    curve_input = calibration_frame[
        [
            group_column,
            "weather_wind_speed_100m_ms",
            "observed_capacity_factor",
        ]
    ].dropna(subset=[group_column, "weather_wind_speed_100m_ms", "observed_capacity_factor"]).copy()
    if curve_input.empty:
        return pd.DataFrame()

    curve_input["wind_bin_ms"] = curve_input["weather_wind_speed_100m_ms"].clip(lower=0.0, upper=40.0).round().astype(int)
    observation_counts = (
        curve_input.groupby(group_column, as_index=False)
        .agg(curve_observation_count=("observed_capacity_factor", "count"))
    )
    curve_input = curve_input.merge(observation_counts, on=group_column, how="left")
    curve_input = curve_input[curve_input["curve_observation_count"] >= min_observations].copy()
    if curve_input.empty:
        return pd.DataFrame()

    curve = (
        curve_input.groupby([group_column, "wind_bin_ms", "curve_observation_count"], as_index=False)
        .agg(
            q95_capacity_factor=("observed_capacity_factor", lambda values: float(values.quantile(0.95))),
            median_capacity_factor=("observed_capacity_factor", "median"),
            bin_observation_count=("observed_capacity_factor", "count"),
        )
        .sort_values([group_column, "wind_bin_ms"])
        .reset_index(drop=True)
    )
    curve["envelope_capacity_factor"] = (
        curve.groupby(group_column)["q95_capacity_factor"].cummax().clip(upper=1.0)
    )
    curve["curve_scope_type"] = curve_scope_type
    return curve


def _apply_curve_estimates(
    frame: pd.DataFrame,
    curve_frame: pd.DataFrame,
    id_column: str,
    method_name: str,
    curve_scope_type: str,
) -> pd.DataFrame:
    if curve_frame.empty:
        return frame

    candidate_mask = (
        (~frame["counterfactual_valid_flag"])
        & frame["availability_state"].eq("available")
        & frame[id_column].notna()
        & frame["generation_mwh"].notna()
        & frame["generation_capacity_mw"].notna()
        & (frame["generation_capacity_mw"] > 0)
        & frame["weather_wind_speed_100m_ms"].notna()
    )
    if not bool(candidate_mask.any()):
        return frame

    left = frame.loc[
        candidate_mask,
        [
            id_column,
            "generation_mwh",
            "generation_capacity_mw",
            "weather_wind_speed_100m_ms",
        ],
    ].copy()
    left["_row_index"] = left.index
    left["weather_wind_bin_ms"] = left["weather_wind_speed_100m_ms"].clip(lower=0.0, upper=40.0).round().astype(float)
    left = left.sort_values([id_column, "weather_wind_bin_ms", "_row_index"])

    right = curve_frame[
        [
            id_column,
            "wind_bin_ms",
            "envelope_capacity_factor",
            "curve_observation_count",
        ]
    ].sort_values([id_column, "wind_bin_ms"])
    right["wind_bin_ms"] = right["wind_bin_ms"].astype(float)

    estimated = pd.merge_asof(
        left,
        right,
        by=id_column,
        left_on="weather_wind_bin_ms",
        right_on="wind_bin_ms",
        direction="backward",
    )
    estimated["estimated_counterfactual_mwh"] = (
        estimated["envelope_capacity_factor"] * (estimated["generation_capacity_mw"] * 0.5)
    )
    valid_mask = (
        estimated["estimated_counterfactual_mwh"].notna()
        & (estimated["estimated_counterfactual_mwh"] + 1e-6 >= estimated["generation_mwh"])
        & estimated["curve_observation_count"].notna()
    )
    if not bool(valid_mask.any()):
        return frame

    valid = estimated.loc[valid_mask].set_index("_row_index")
    frame.loc[valid.index, "counterfactual_method"] = method_name
    frame.loc[valid.index, "counterfactual_mwh"] = valid["estimated_counterfactual_mwh"]
    frame.loc[valid.index, "counterfactual_valid_flag"] = True
    frame.loc[valid.index, "weather_curve_observation_count"] = valid["curve_observation_count"].astype(int)
    frame.loc[valid.index, "weather_curve_scope_type"] = curve_scope_type
    frame.loc[valid.index, "weather_curve_scope_key"] = valid[id_column].astype(str)
    return frame


def _apply_weather_calibration(
    frame: pd.DataFrame,
    fact_weather_hourly: pd.DataFrame,
) -> pd.DataFrame:
    frame["hour_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce").dt.floor("h")
    frame["weather_scope_type"] = np.where(
        frame["cluster_key"].notna(),
        "cluster",
        np.where(frame["parent_region"].notna(), "parent_region", pd.NA),
    )
    frame["weather_scope_key"] = np.where(frame["cluster_key"].notna(), frame["cluster_key"], frame["parent_region"])
    frame["weather_curve_scope_type"] = pd.NA
    frame["weather_curve_scope_key"] = pd.NA
    frame["weather_curve_observation_count"] = 0

    if fact_weather_hourly.empty:
        for column in (
            "weather_temperature_2m_c",
            "weather_pressure_msl_hpa",
            "weather_cloud_cover_pct",
            "weather_wind_speed_10m_ms",
            "weather_wind_speed_100m_ms",
            "weather_wind_direction_100m_deg",
            "weather_wind_gusts_10m_ms",
            "weather_wind_power_index_100m",
            "weather_anchor_count",
            "weather_weight_sum_mw",
        ):
            frame[column] = np.nan
        return frame

    weather = fact_weather_hourly[fact_weather_hourly["scope_type"].isin(["cluster", "parent_region"])].copy()
    weather = weather.rename(
        columns={
            "scope_type": "weather_scope_type_join",
            "scope_key": "weather_scope_key_join",
            "temperature_2m_c": "weather_temperature_2m_c",
            "pressure_msl_hpa": "weather_pressure_msl_hpa",
            "cloud_cover_pct": "weather_cloud_cover_pct",
            "wind_speed_10m_ms": "weather_wind_speed_10m_ms",
            "wind_speed_100m_ms": "weather_wind_speed_100m_ms",
            "wind_direction_100m_deg": "weather_wind_direction_100m_deg",
            "wind_gusts_10m_ms": "weather_wind_gusts_10m_ms",
            "wind_power_index_100m": "weather_wind_power_index_100m",
            "weather_anchor_count": "weather_anchor_count",
            "weather_weight_sum_mw": "weather_weight_sum_mw",
        }
    )
    weather_columns = [
        "hour_start_utc",
        "weather_scope_type_join",
        "weather_scope_key_join",
        "weather_temperature_2m_c",
        "weather_pressure_msl_hpa",
        "weather_cloud_cover_pct",
        "weather_wind_speed_10m_ms",
        "weather_wind_speed_100m_ms",
        "weather_wind_direction_100m_deg",
        "weather_wind_gusts_10m_ms",
        "weather_wind_power_index_100m",
        "weather_anchor_count",
        "weather_weight_sum_mw",
    ]
    frame = frame.merge(
        weather[weather_columns],
        left_on=["hour_start_utc", "weather_scope_type", "weather_scope_key"],
        right_on=["hour_start_utc", "weather_scope_type_join", "weather_scope_key_join"],
        how="left",
    ).drop(columns=["weather_scope_type_join", "weather_scope_key_join"])

    calibration_mask = (
        (~frame["dispatch_truth_flag"])
        & frame["availability_state"].eq("available")
        & frame["generation_mwh"].notna()
        & frame["generation_capacity_mw"].notna()
        & (frame["generation_capacity_mw"] > 0)
        & frame["weather_wind_speed_100m_ms"].notna()
    )
    calibration = frame.loc[
        calibration_mask,
        [
            "elexon_bm_unit",
            "cluster_key",
            "parent_region",
            "generation_mwh",
            "generation_capacity_mw",
            "weather_wind_speed_100m_ms",
        ],
    ].copy()
    if calibration.empty:
        return frame

    calibration["observed_capacity_factor"] = (
        calibration["generation_mwh"] / (calibration["generation_capacity_mw"] * 0.5)
    ).clip(lower=0.0, upper=1.25)
    calibration = calibration[calibration["observed_capacity_factor"].notna()].copy()
    if calibration.empty:
        return frame

    bmu_curve = _build_weather_curve(calibration, "elexon_bm_unit", min_observations=24, curve_scope_type="bmu")
    cluster_curve = _build_weather_curve(calibration, "cluster_key", min_observations=96, curve_scope_type="cluster")
    parent_region_curve = _build_weather_curve(calibration, "parent_region", min_observations=192, curve_scope_type="parent_region")

    frame = _apply_curve_estimates(
        frame,
        curve_frame=bmu_curve,
        id_column="elexon_bm_unit",
        method_name="bmu_weather_power_curve",
        curve_scope_type="bmu",
    )
    frame = _apply_curve_estimates(
        frame,
        curve_frame=cluster_curve,
        id_column="cluster_key",
        method_name="cluster_weather_power_curve",
        curve_scope_type="cluster",
    )
    frame = _apply_curve_estimates(
        frame,
        curve_frame=parent_region_curve,
        id_column="parent_region",
        method_name="parent_region_weather_power_curve",
        curve_scope_type="parent_region",
    )
    return frame


def _apply_reconciliation(frame: pd.DataFrame, fact_constraint_daily: pd.DataFrame) -> pd.DataFrame:
    if fact_constraint_daily.empty:
        frame["gb_daily_estimated_lost_energy_mwh"] = frame.groupby("settlement_date")["lost_energy_mwh"].transform("sum")
        frame["gb_daily_truth_curtailment_mwh"] = np.nan
        frame["reconciliation_abs_error_mwh"] = np.nan
        frame["reconciliation_relative_error"] = np.nan
        frame["reconciliation_status"] = "warn"
        return frame

    estimated_daily = (
        frame.groupby("settlement_date", as_index=False)
        .agg(gb_daily_estimated_lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())))
    )
    daily = estimated_daily.merge(
        fact_constraint_daily[["date", "total_curtailment_mwh"]],
        left_on="settlement_date",
        right_on="date",
        how="left",
    ).drop(columns="date")
    daily = daily.rename(columns={"total_curtailment_mwh": "gb_daily_truth_curtailment_mwh"})
    daily["reconciliation_abs_error_mwh"] = (
        daily["gb_daily_estimated_lost_energy_mwh"] - daily["gb_daily_truth_curtailment_mwh"]
    ).abs()
    daily["reconciliation_relative_error"] = np.where(
        daily["gb_daily_truth_curtailment_mwh"] > 0,
        daily["reconciliation_abs_error_mwh"] / daily["gb_daily_truth_curtailment_mwh"],
        np.where(daily["gb_daily_estimated_lost_energy_mwh"] <= 1e-6, 0.0, np.nan),
    )
    daily["reconciliation_status"] = np.select(
        [
            daily["reconciliation_relative_error"].le(0.25),
            daily["reconciliation_relative_error"].le(0.50),
        ],
        ["pass", "warn"],
        default="fail",
    )
    daily.loc[daily["gb_daily_truth_curtailment_mwh"].isna(), "reconciliation_status"] = "warn"

    frame = frame.merge(daily, on="settlement_date", how="left")
    return frame


def build_fact_bmu_curtailment_truth_half_hourly(
    dim_bmu_asset: pd.DataFrame,
    fact_bmu_generation_half_hourly: pd.DataFrame,
    fact_bmu_dispatch_acceptance_half_hourly: pd.DataFrame,
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_availability_half_hourly: pd.DataFrame,
    fact_constraint_daily: pd.DataFrame,
    fact_weather_hourly: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    frame = build_bmu_interval_spine(dim_bmu_asset, start_date, end_date)
    generation_columns = ["settlement_date", "settlement_period", "elexon_bm_unit", "generation_mwh"]
    frame = frame.merge(
        fact_bmu_generation_half_hourly[generation_columns],
        on=["settlement_date", "settlement_period", "elexon_bm_unit"],
        how="left",
    )

    dispatch_columns = [
        "settlement_date",
        "settlement_period",
        "elexon_bm_unit",
        "accepted_down_delta_mwh_lower_bound",
        "accepted_up_delta_mwh_lower_bound",
        "dispatch_down_flag",
        "dispatch_up_flag",
        "acceptance_event_count",
        "distinct_acceptance_number_count",
    ]
    frame = frame.merge(
        fact_bmu_dispatch_acceptance_half_hourly[dispatch_columns],
        on=["settlement_date", "settlement_period", "elexon_bm_unit"],
        how="left",
    )

    physical_columns = [
        "settlement_date",
        "settlement_period",
        "elexon_bm_unit",
        "physical_baseline_source_dataset",
        "physical_baseline_mwh",
        "physical_consistency_flag",
        "counterfactual_method",
        "counterfactual_valid_flag",
    ]
    frame = frame.merge(
        fact_bmu_physical_position_half_hourly[physical_columns],
        on=["settlement_date", "settlement_period", "elexon_bm_unit"],
        how="left",
    )

    availability_columns = [
        "settlement_date",
        "settlement_period",
        "elexon_bm_unit",
        "remit_active_flag",
        "availability_state",
        "availability_confidence",
        "uou_output_usable_mw",
    ]
    frame = frame.merge(
        fact_bmu_availability_half_hourly[availability_columns],
        on=["settlement_date", "settlement_period", "elexon_bm_unit"],
        how="left",
    )

    frame["accepted_down_delta_mwh_lower_bound"] = pd.to_numeric(
        frame["accepted_down_delta_mwh_lower_bound"], errors="coerce"
    ).fillna(0.0)
    frame["accepted_up_delta_mwh_lower_bound"] = pd.to_numeric(
        frame["accepted_up_delta_mwh_lower_bound"], errors="coerce"
    ).fillna(0.0)
    frame["dispatch_truth_flag"] = frame["accepted_down_delta_mwh_lower_bound"] > 0
    frame["remit_active_flag"] = frame["remit_active_flag"].where(frame["remit_active_flag"].notna(), False).astype(bool)
    frame["availability_state"] = frame["availability_state"].fillna("unknown")
    frame["availability_confidence"] = frame["availability_confidence"].fillna("low")

    frame["counterfactual_method"] = frame["counterfactual_method"].fillna("none")
    frame["counterfactual_mwh"] = pd.to_numeric(frame["physical_baseline_mwh"], errors="coerce")
    frame["counterfactual_valid_flag"] = (
        frame["counterfactual_valid_flag"].where(frame["counterfactual_valid_flag"].notna(), False).astype(bool)
    )

    frame = _apply_weather_calibration(frame, fact_weather_hourly=fact_weather_hourly)

    frame["truth_tier"] = "dispatch_only"
    frame.loc[
        frame["counterfactual_valid_flag"] & frame["counterfactual_method"].eq("pn_qpn_physical_max"),
        "truth_tier",
    ] = "physical_baseline"
    frame.loc[
        frame["counterfactual_valid_flag"] & frame["counterfactual_method"].isin(WEATHER_METHODS),
        "truth_tier",
    ] = "weather_calibrated"

    frame["lost_energy_mwh"] = 0.0
    valid_lost_energy_mask = (
        frame["dispatch_truth_flag"]
        & frame["availability_state"].eq("available")
        & frame["counterfactual_valid_flag"]
        & frame["generation_mwh"].notna()
        & frame["counterfactual_mwh"].notna()
    )
    frame.loc[frame["dispatch_truth_flag"] & ~valid_lost_energy_mask, "lost_energy_mwh"] = np.nan
    frame.loc[valid_lost_energy_mask, "lost_energy_mwh"] = (
        frame.loc[valid_lost_energy_mask, "counterfactual_mwh"] - frame.loc[valid_lost_energy_mask, "generation_mwh"]
    ).clip(lower=0.0)
    frame["lost_energy_estimate_flag"] = valid_lost_energy_mask

    frame = _apply_reconciliation(frame, fact_constraint_daily)

    frame["precision_profile_include"] = (
        frame["truth_tier"].isin({"physical_baseline", "weather_calibrated"})
        & frame["availability_state"].eq("available")
        & frame["counterfactual_valid_flag"]
        & frame["reconciliation_status"].eq("pass")
    )
    frame["research_profile_include"] = frame["dispatch_truth_flag"] & ~frame["availability_state"].eq("outage")
    frame["target_is_proxy"] = False

    frame["source_lineage"] = "B1610|BOALF|REMIT|constraint_breakdown"
    frame.loc[frame["truth_tier"].eq("physical_baseline"), "source_lineage"] = (
        "B1610|BOALF|balancing_physical|REMIT|constraint_breakdown"
    )
    frame.loc[frame["truth_tier"].eq("weather_calibrated"), "source_lineage"] = (
        "B1610|BOALF|REMIT|open_meteo_archive|constraint_breakdown"
    )
    frame.loc[frame["uou_output_usable_mw"].notna(), "source_lineage"] = (
        frame.loc[frame["uou_output_usable_mw"].notna(), "source_lineage"] + "|UOU2T14D"
    )

    keep_columns = [
        "settlement_date",
        "settlement_period",
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "generation_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "accepted_up_delta_mwh_lower_bound",
        "availability_state",
        "remit_active_flag",
        "availability_confidence",
        "counterfactual_method",
        "counterfactual_mwh",
        "counterfactual_valid_flag",
        "lost_energy_mwh",
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "truth_tier",
        "precision_profile_include",
        "research_profile_include",
        "reconciliation_status",
        "source_lineage",
        "target_is_proxy",
        "gb_daily_estimated_lost_energy_mwh",
        "gb_daily_truth_curtailment_mwh",
        "reconciliation_abs_error_mwh",
        "reconciliation_relative_error",
        "physical_baseline_source_dataset",
        "physical_consistency_flag",
        "weather_scope_type",
        "weather_scope_key",
        "weather_curve_scope_type",
        "weather_curve_scope_key",
        "weather_curve_observation_count",
        "weather_temperature_2m_c",
        "weather_pressure_msl_hpa",
        "weather_cloud_cover_pct",
        "weather_wind_speed_10m_ms",
        "weather_wind_speed_100m_ms",
        "weather_wind_direction_100m_deg",
        "weather_wind_gusts_10m_ms",
        "weather_wind_power_index_100m",
        "weather_anchor_count",
        "weather_weight_sum_mw",
    ]
    return frame[keep_columns].sort_values(["interval_start_utc", "elexon_bm_unit"]).reset_index(drop=True)


def filter_truth_profile(frame: pd.DataFrame, truth_profile: str) -> pd.DataFrame:
    if truth_profile not in VALID_TRUTH_PROFILES:
        raise ValueError(
            f"unsupported truth profile '{truth_profile}'. Expected one of: {', '.join(sorted(VALID_TRUTH_PROFILES))}"
        )
    if truth_profile == "all":
        return frame
    if truth_profile == "precision":
        return frame[frame["precision_profile_include"]].reset_index(drop=True)
    return frame[frame["research_profile_include"]].reset_index(drop=True)


def materialize_bmu_curtailment_truth(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    truth_profile: str = "all",
) -> Dict[str, pd.DataFrame]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if truth_profile not in VALID_TRUTH_PROFILES:
        raise ValueError(f"unsupported truth profile '{truth_profile}'")

    reference = fetch_bmu_reference_all()
    dim_bmu_asset = build_dim_bmu_asset(reference)

    raw_generation = fetch_b1610_generation(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    fact_bmu_generation_half_hourly = build_fact_bmu_generation_half_hourly(dim_bmu_asset, raw_generation)

    raw_acceptance = fetch_boalf_acceptances(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    fact_bmu_acceptance_event = build_fact_bmu_acceptance_event(dim_bmu_asset, raw_acceptance)
    fact_bmu_dispatch_acceptance_half_hourly = build_fact_bmu_dispatch_acceptance_half_hourly(
        fact_bmu_acceptance_event,
        start_date=start_date,
        end_date=end_date,
    )

    raw_physical = fetch_balancing_physical(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    fact_bmu_physical_position_half_hourly = build_fact_bmu_physical_position_half_hourly(
        dim_bmu_asset=dim_bmu_asset,
        fact_bmu_generation_half_hourly=fact_bmu_generation_half_hourly,
        raw_physical_frame=raw_physical,
        start_date=start_date,
        end_date=end_date,
    )

    remit_fetch_ok = True
    try:
        raw_remit = fetch_remit_event_detail(start_date, end_date)
    except Exception:
        remit_fetch_ok = False
        raw_remit = pd.DataFrame()

    try:
        raw_uou = fetch_uou2t14d_summary(dim_bmu_asset["elexon_bm_unit"].tolist())
    except Exception:
        raw_uou = pd.DataFrame()

    fact_bmu_availability_half_hourly = build_fact_bmu_availability_half_hourly(
        dim_bmu_asset=dim_bmu_asset,
        raw_remit_frame=raw_remit,
        raw_uou_frame=raw_uou,
        start_date=start_date,
        end_date=end_date,
        remit_fetch_ok=remit_fetch_ok,
    )

    try:
        fact_weather_hourly = build_fact_weather_hourly(start_date, end_date)
    except Exception:
        fact_weather_hourly = pd.DataFrame()

    try:
        fact_constraint_daily = _fetch_constraint_daily_for_range(start_date, end_date)
    except Exception:
        fact_constraint_daily = pd.DataFrame()

    fact_bmu_curtailment_truth_half_hourly = build_fact_bmu_curtailment_truth_half_hourly(
        dim_bmu_asset=dim_bmu_asset,
        fact_bmu_generation_half_hourly=fact_bmu_generation_half_hourly,
        fact_bmu_dispatch_acceptance_half_hourly=fact_bmu_dispatch_acceptance_half_hourly,
        fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
        fact_bmu_availability_half_hourly=fact_bmu_availability_half_hourly,
        fact_constraint_daily=fact_constraint_daily,
        fact_weather_hourly=fact_weather_hourly,
        start_date=start_date,
        end_date=end_date,
    )
    fact_bmu_curtailment_truth_half_hourly = filter_truth_profile(
        fact_bmu_curtailment_truth_half_hourly,
        truth_profile=truth_profile,
    )

    frames = {
        "fact_bmu_physical_position_half_hourly": fact_bmu_physical_position_half_hourly,
        "fact_bmu_availability_half_hourly": fact_bmu_availability_half_hourly,
        "fact_bmu_curtailment_truth_half_hourly": fact_bmu_curtailment_truth_half_hourly,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)
    return frames
