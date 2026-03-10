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
from curtailment_signals import CONSTRAINT_QA_TARGET_DEFINITION, fetch_constraint_daily
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
    left["weather_wind_bin_ms"] = left["weather_wind_speed_100m_ms"].clip(lower=0.0, upper=40.0).round().astype(int)

    expanded_curve_rows = []
    for scope_key, group in curve_frame.groupby(id_column, dropna=False):
        if pd.isna(scope_key):
            continue
        base_curve = (
            group[["wind_bin_ms", "envelope_capacity_factor", "curve_observation_count"]]
            .drop_duplicates(subset=["wind_bin_ms"], keep="last")
            .sort_values("wind_bin_ms")
            .set_index("wind_bin_ms")
        )
        expanded_curve = base_curve.reindex(range(41)).ffill()
        expanded_curve["curve_observation_count"] = int(group["curve_observation_count"].max())
        expanded_curve[id_column] = scope_key
        expanded_curve["wind_bin_ms"] = expanded_curve.index.astype(int)
        expanded_curve_rows.append(
            expanded_curve.reset_index(drop=True)[
                [id_column, "wind_bin_ms", "envelope_capacity_factor", "curve_observation_count"]
            ]
        )

    if not expanded_curve_rows:
        return frame

    expanded_curve_frame = pd.concat(expanded_curve_rows, ignore_index=True)
    estimated = left.merge(
        expanded_curve_frame,
        left_on=[id_column, "weather_wind_bin_ms"],
        right_on=[id_column, "wind_bin_ms"],
        how="left",
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


def _safe_target_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            denominator > 0,
            numerator / denominator,
            np.where(numerator <= 1e-6, 0.0, np.nan),
        ),
        index=numerator.index,
    )


def _build_reconciliation_view(
    estimated_daily: pd.DataFrame,
    fact_constraint_daily: pd.DataFrame,
    source_target_column: str,
    output_target_column: str,
    output_abs_error_column: str,
    output_relative_error_column: str,
    output_status_column: str,
    definition_source_column: str | None = None,
    definition_output_column: str | None = None,
) -> pd.DataFrame:
    daily = estimated_daily.copy()
    if fact_constraint_daily.empty or source_target_column not in fact_constraint_daily.columns:
        daily[output_target_column] = np.nan
        daily[output_abs_error_column] = np.nan
        daily[output_relative_error_column] = np.nan
        daily[output_status_column] = "warn"
        if definition_output_column:
            daily[definition_output_column] = pd.NA
        return daily

    columns = ["date", source_target_column]
    if definition_source_column and definition_source_column in fact_constraint_daily.columns:
        columns.append(definition_source_column)
    truth = fact_constraint_daily[columns].copy()
    rename_map = {source_target_column: output_target_column}
    if definition_source_column and definition_source_column in truth.columns and definition_output_column:
        rename_map[definition_source_column] = definition_output_column
    daily = daily.merge(
        truth.rename(columns=rename_map),
        left_on="settlement_date",
        right_on="date",
        how="left",
    ).drop(columns="date")
    if definition_output_column and definition_output_column not in daily.columns:
        daily[definition_output_column] = pd.NA

    daily[output_abs_error_column] = (
        daily["gb_daily_estimated_lost_energy_mwh"] - daily[output_target_column]
    ).abs()
    daily[output_relative_error_column] = _safe_target_ratio(
        daily[output_abs_error_column],
        daily[output_target_column],
    )
    daily[output_status_column] = np.select(
        [
            daily[output_relative_error_column].le(0.25),
            daily[output_relative_error_column].le(0.50),
        ],
        ["pass", "warn"],
        default="fail",
    )
    daily.loc[daily[output_target_column].isna(), output_status_column] = "warn"
    return daily


def _apply_reconciliation(frame: pd.DataFrame, fact_constraint_daily: pd.DataFrame) -> pd.DataFrame:
    estimated_daily = (
        frame.groupby("settlement_date", as_index=False)
        .agg(gb_daily_estimated_lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())))
    )
    raw_daily = _build_reconciliation_view(
        estimated_daily=estimated_daily,
        fact_constraint_daily=fact_constraint_daily,
        source_target_column="total_curtailment_mwh",
        output_target_column="gb_daily_raw_constraint_total_mwh",
        output_abs_error_column="raw_reconciliation_abs_error_mwh",
        output_relative_error_column="raw_reconciliation_relative_error",
        output_status_column="raw_reconciliation_status",
    )
    qa_daily = _build_reconciliation_view(
        estimated_daily=estimated_daily,
        fact_constraint_daily=fact_constraint_daily,
        source_target_column="qa_wind_relevant_positive_mwh",
        output_target_column="gb_daily_qa_target_mwh",
        output_abs_error_column="qa_reconciliation_abs_error_mwh",
        output_relative_error_column="qa_reconciliation_relative_error",
        output_status_column="qa_reconciliation_status",
        definition_source_column="qa_target_definition",
        definition_output_column="qa_target_definition",
    )
    daily = raw_daily.merge(
        qa_daily[
            [
                "settlement_date",
                "qa_target_definition",
                "gb_daily_qa_target_mwh",
                "qa_reconciliation_abs_error_mwh",
                "qa_reconciliation_relative_error",
                "qa_reconciliation_status",
            ]
        ],
        on="settlement_date",
        how="left",
    )
    daily["gb_daily_truth_curtailment_mwh"] = daily["gb_daily_raw_constraint_total_mwh"]
    daily["reconciliation_abs_error_mwh"] = daily["raw_reconciliation_abs_error_mwh"]
    daily["reconciliation_relative_error"] = daily["raw_reconciliation_relative_error"]
    daily["reconciliation_status"] = daily["raw_reconciliation_status"]

    frame = frame.merge(daily, on="settlement_date", how="left")
    return frame


def _first_mode(values: pd.Series) -> object:
    counts = pd.Series(values).dropna().astype(str).value_counts()
    if counts.empty:
        return pd.NA
    return counts.index[0]


def _derive_counterfactual_invalid_reason(frame: pd.DataFrame) -> pd.Series:
    reason = pd.Series(pd.NA, index=frame.index, dtype="object")
    invalid_mask = ~frame["counterfactual_valid_flag"]
    if not bool(invalid_mask.any()):
        return reason

    physical_mask = frame["counterfactual_method"].eq("pn_qpn_physical_max")
    physical_consistency_mask = frame["physical_consistency_flag"].astype("boolean").fillna(False).astype(bool)
    missing_generation_mask = invalid_mask & frame["generation_mwh"].isna()
    inconsistent_physical_mask = invalid_mask & physical_mask & ~physical_consistency_mask
    below_generation_mask = (
        invalid_mask
        & physical_mask
        & frame["counterfactual_mwh"].notna()
        & frame["generation_mwh"].notna()
        & (frame["counterfactual_mwh"] + 1e-6 < frame["generation_mwh"])
    )
    missing_physical_mask = invalid_mask & physical_mask & frame["physical_baseline_source_dataset"].isna()
    no_weather_scope_mask = invalid_mask & reason.isna() & frame["weather_scope_key"].isna()
    missing_weather_observation_mask = (
        invalid_mask
        & reason.isna()
        & frame["weather_scope_key"].notna()
        & frame["weather_wind_speed_100m_ms"].isna()
    )

    reason.loc[missing_generation_mask] = "missing_generation"
    reason.loc[inconsistent_physical_mask] = "physical_inconsistent"
    reason.loc[below_generation_mask] = "physical_below_generation"
    reason.loc[missing_physical_mask & reason.isna()] = "missing_physical_baseline"
    reason.loc[no_weather_scope_mask & reason.isna()] = "no_weather_scope"
    reason.loc[missing_weather_observation_mask & reason.isna()] = "missing_weather_observation"
    reason.loc[invalid_mask & reason.isna()] = "weather_curve_unavailable"
    return reason


def _derive_lost_energy_block_reason(frame: pd.DataFrame) -> pd.Series:
    reason = pd.Series(pd.NA, index=frame.index, dtype="object")
    dispatch_mask = frame["dispatch_truth_flag"]
    if not bool(dispatch_mask.any()):
        reason.loc[~dispatch_mask] = "no_dispatch_truth"
        return reason

    reason.loc[~dispatch_mask] = "no_dispatch_truth"
    reason.loc[dispatch_mask & frame["lost_energy_estimate_flag"]] = "estimated"
    reason.loc[dispatch_mask & frame["availability_state"].eq("outage") & reason.ne("estimated")] = "outage"
    reason.loc[dispatch_mask & frame["availability_state"].eq("unknown") & reason.ne("estimated")] = "availability_unknown"
    reason.loc[dispatch_mask & frame["generation_mwh"].isna() & reason.ne("estimated")] = "missing_generation"
    reason.loc[
        dispatch_mask
        & ~frame["counterfactual_valid_flag"]
        & reason.ne("estimated")
        & reason.ne("outage")
        & reason.ne("availability_unknown")
        & reason.ne("missing_generation"),
    ] = frame.loc[
        dispatch_mask
        & ~frame["counterfactual_valid_flag"]
        & reason.ne("estimated")
        & reason.ne("outage")
        & reason.ne("availability_unknown")
        & reason.ne("missing_generation"),
        "counterfactual_invalid_reason",
    ]
    no_positive_gap_mask = (
        dispatch_mask
        & frame["counterfactual_valid_flag"]
        & frame["generation_mwh"].notna()
        & frame["counterfactual_mwh"].notna()
        & ((frame["counterfactual_mwh"] - frame["generation_mwh"]).clip(lower=0.0) <= 1e-6)
        & reason.isna()
    )
    reason.loc[no_positive_gap_mask] = "no_positive_gap"
    reason.loc[dispatch_mask & reason.isna()] = "dispatch_row_unclassified"
    return reason


def _ensure_truth_diagnostic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    defaults = {
        "mapping_status": "unknown",
        "availability_state": "unknown",
        "counterfactual_method": "none",
        "physical_consistency_flag": True,
        "lost_energy_mwh": 0.0,
        "accepted_down_delta_mwh_lower_bound": 0.0,
        "dispatch_truth_flag": False,
        "lost_energy_estimate_flag": False,
        "counterfactual_valid_flag": False,
    }
    for column, default_value in defaults.items():
        if column not in prepared.columns:
            prepared[column] = default_value

    def _backfill_alias(target_column: str, source_column: str, default_value: object = np.nan) -> None:
        if target_column not in prepared.columns:
            if source_column in prepared.columns:
                prepared[target_column] = prepared[source_column]
            else:
                prepared[target_column] = default_value
        elif source_column in prepared.columns:
            source_mask = prepared[target_column].isna()
            prepared.loc[source_mask, target_column] = prepared.loc[source_mask, source_column]

    nullable_columns = [
        "generation_mwh",
        "counterfactual_mwh",
        "physical_baseline_source_dataset",
        "weather_scope_key",
        "weather_wind_speed_100m_ms",
    ]
    for column in nullable_columns:
        if column not in prepared.columns:
            prepared[column] = np.nan

    if "gb_daily_estimated_lost_energy_mwh" not in prepared.columns:
        prepared["gb_daily_estimated_lost_energy_mwh"] = (
            prepared.groupby("settlement_date")["lost_energy_mwh"].transform(lambda values: float(pd.Series(values).fillna(0.0).sum()))
        )

    _backfill_alias("gb_daily_raw_constraint_total_mwh", "gb_daily_truth_curtailment_mwh")
    _backfill_alias("raw_reconciliation_abs_error_mwh", "reconciliation_abs_error_mwh")
    _backfill_alias("raw_reconciliation_relative_error", "reconciliation_relative_error")
    _backfill_alias("raw_reconciliation_status", "reconciliation_status", default_value="warn")
    _backfill_alias("gb_daily_truth_curtailment_mwh", "gb_daily_raw_constraint_total_mwh")
    _backfill_alias("reconciliation_abs_error_mwh", "raw_reconciliation_abs_error_mwh")
    _backfill_alias("reconciliation_relative_error", "raw_reconciliation_relative_error")
    _backfill_alias("reconciliation_status", "raw_reconciliation_status", default_value="warn")
    if "qa_target_definition" not in prepared.columns:
        prepared["qa_target_definition"] = pd.NA
    if "gb_daily_qa_target_mwh" not in prepared.columns:
        prepared["gb_daily_qa_target_mwh"] = np.nan
    if "qa_reconciliation_abs_error_mwh" not in prepared.columns:
        prepared["qa_reconciliation_abs_error_mwh"] = np.nan
    if "qa_reconciliation_relative_error" not in prepared.columns:
        prepared["qa_reconciliation_relative_error"] = np.nan
    if "qa_reconciliation_status" not in prepared.columns:
        prepared["qa_reconciliation_status"] = "warn"
    if "counterfactual_invalid_reason" not in prepared.columns:
        prepared["counterfactual_invalid_reason"] = _derive_counterfactual_invalid_reason(prepared)
    if "lost_energy_block_reason" not in prepared.columns:
        prepared["lost_energy_block_reason"] = _derive_lost_energy_block_reason(prepared)
    return prepared


def build_fact_curtailment_reconciliation_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "settlement_date",
                "gb_daily_estimated_lost_energy_mwh",
                "qa_target_definition",
                "gb_daily_raw_constraint_total_mwh",
                "raw_reconciliation_abs_error_mwh",
                "raw_reconciliation_relative_error",
                "raw_reconciliation_status",
                "gb_daily_qa_target_mwh",
                "qa_reconciliation_abs_error_mwh",
                "qa_reconciliation_relative_error",
                "qa_reconciliation_status",
                "gb_daily_truth_curtailment_mwh",
                "reconciliation_abs_error_mwh",
                "reconciliation_relative_error",
                "reconciliation_status",
                "dispatch_half_hour_count",
                "distinct_dispatch_bmu_count",
                "dispatch_down_mwh_lower_bound",
                "lost_energy_estimate_half_hour_count",
                "physical_baseline_row_count",
                "weather_calibrated_row_count",
                "dispatch_available_row_count",
                "dispatch_unknown_availability_row_count",
                "dispatch_outage_row_count",
                "mapped_dispatch_row_count",
                "unmapped_dispatch_row_count",
                "dispatch_coverage_ratio_vs_raw_total",
                "dispatch_coverage_ratio_vs_qa_target",
                "dispatch_coverage_ratio_vs_gb_truth",
                "lost_energy_capture_ratio_vs_raw_total",
                "lost_energy_capture_ratio_vs_qa_target",
                "lost_energy_capture_ratio_vs_gb_truth",
                "lost_energy_to_dispatch_ratio",
                "primary_dispatch_block_reason",
            ]
        )

    frame = _ensure_truth_diagnostic_columns(frame)
    dispatch_mask = frame["dispatch_truth_flag"]
    grouped = frame.groupby("settlement_date", as_index=False).agg(
        gb_daily_estimated_lost_energy_mwh=("gb_daily_estimated_lost_energy_mwh", "max"),
        qa_target_definition=("qa_target_definition", _first_mode),
        gb_daily_raw_constraint_total_mwh=("gb_daily_raw_constraint_total_mwh", "max"),
        raw_reconciliation_abs_error_mwh=("raw_reconciliation_abs_error_mwh", "max"),
        raw_reconciliation_relative_error=("raw_reconciliation_relative_error", "max"),
        raw_reconciliation_status=("raw_reconciliation_status", _first_mode),
        gb_daily_qa_target_mwh=("gb_daily_qa_target_mwh", "max"),
        qa_reconciliation_abs_error_mwh=("qa_reconciliation_abs_error_mwh", "max"),
        qa_reconciliation_relative_error=("qa_reconciliation_relative_error", "max"),
        qa_reconciliation_status=("qa_reconciliation_status", _first_mode),
        gb_daily_truth_curtailment_mwh=("gb_daily_truth_curtailment_mwh", "max"),
        reconciliation_abs_error_mwh=("reconciliation_abs_error_mwh", "max"),
        reconciliation_relative_error=("reconciliation_relative_error", "max"),
        reconciliation_status=("reconciliation_status", _first_mode),
        dispatch_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        distinct_dispatch_bmu_count=("elexon_bm_unit", lambda values: int(frame.loc[values.index, "elexon_bm_unit"][dispatch_mask.loc[values.index]].nunique())),
        dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[dispatch_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        lost_energy_estimate_half_hour_count=(
            "lost_energy_estimate_flag",
            lambda values: int(pd.Series(values).fillna(False).astype(bool).sum()),
        ),
        physical_baseline_row_count=("truth_tier", lambda values: int(pd.Series(values).eq("physical_baseline").sum())),
        weather_calibrated_row_count=("truth_tier", lambda values: int(pd.Series(values).eq("weather_calibrated").sum())),
        dispatch_available_row_count=(
            "availability_state",
            lambda values: int(((pd.Series(values) == "available") & dispatch_mask.loc[values.index]).sum()),
        ),
        dispatch_unknown_availability_row_count=(
            "availability_state",
            lambda values: int(((pd.Series(values) == "unknown") & dispatch_mask.loc[values.index]).sum()),
        ),
        dispatch_outage_row_count=(
            "availability_state",
            lambda values: int(((pd.Series(values) == "outage") & dispatch_mask.loc[values.index]).sum()),
        ),
        mapped_dispatch_row_count=(
            "mapping_status",
            lambda values: int(((pd.Series(values) == "mapped") & dispatch_mask.loc[values.index]).sum()),
        ),
        unmapped_dispatch_row_count=(
            "mapping_status",
            lambda values: int(((pd.Series(values) != "mapped") & dispatch_mask.loc[values.index]).sum()),
        ),
        primary_dispatch_block_reason=(
            "lost_energy_block_reason",
            lambda values: _first_mode(pd.Series(values)[dispatch_mask.loc[values.index] & ~frame.loc[values.index, "lost_energy_estimate_flag"]]),
        ),
    )
    grouped["dispatch_coverage_ratio_vs_raw_total"] = _safe_target_ratio(
        grouped["dispatch_down_mwh_lower_bound"],
        grouped["gb_daily_raw_constraint_total_mwh"],
    )
    grouped["dispatch_coverage_ratio_vs_qa_target"] = _safe_target_ratio(
        grouped["dispatch_down_mwh_lower_bound"],
        grouped["gb_daily_qa_target_mwh"],
    )
    grouped["dispatch_coverage_ratio_vs_gb_truth"] = np.where(
        grouped["gb_daily_truth_curtailment_mwh"] > 0,
        grouped["dispatch_down_mwh_lower_bound"] / grouped["gb_daily_truth_curtailment_mwh"],
        np.nan,
    )
    grouped["lost_energy_capture_ratio_vs_raw_total"] = _safe_target_ratio(
        grouped["gb_daily_estimated_lost_energy_mwh"],
        grouped["gb_daily_raw_constraint_total_mwh"],
    )
    grouped["lost_energy_capture_ratio_vs_qa_target"] = _safe_target_ratio(
        grouped["gb_daily_estimated_lost_energy_mwh"],
        grouped["gb_daily_qa_target_mwh"],
    )
    grouped["lost_energy_capture_ratio_vs_gb_truth"] = np.where(
        grouped["gb_daily_truth_curtailment_mwh"] > 0,
        grouped["gb_daily_estimated_lost_energy_mwh"] / grouped["gb_daily_truth_curtailment_mwh"],
        np.nan,
    )
    grouped["lost_energy_to_dispatch_ratio"] = np.where(
        grouped["dispatch_down_mwh_lower_bound"] > 0,
        grouped["gb_daily_estimated_lost_energy_mwh"] / grouped["dispatch_down_mwh_lower_bound"],
        np.nan,
    )
    return grouped.sort_values("settlement_date").reset_index(drop=True)


def build_fact_dispatch_alignment_daily(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "qa_target_definition",
        "gb_daily_qa_target_mwh",
        "gb_daily_estimated_lost_energy_mwh",
        "qa_target_shortfall_mwh",
        "total_dispatch_down_mwh_lower_bound",
        "estimated_dispatch_half_hour_count",
        "blocked_dispatch_half_hour_count",
        "estimated_dispatch_down_mwh_lower_bound",
        "blocked_dispatch_down_mwh_lower_bound",
        "blocked_outage_dispatch_down_mwh_lower_bound",
        "blocked_availability_unknown_dispatch_down_mwh_lower_bound",
        "blocked_physical_below_generation_dispatch_down_mwh_lower_bound",
        "blocked_physical_inconsistent_dispatch_down_mwh_lower_bound",
        "blocked_other_dispatch_down_mwh_lower_bound",
        "mapped_dispatch_down_mwh_lower_bound",
        "region_only_dispatch_down_mwh_lower_bound",
        "unmapped_dispatch_down_mwh_lower_bound",
        "mapped_estimated_lost_energy_mwh",
        "region_only_estimated_lost_energy_mwh",
        "unmapped_estimated_lost_energy_mwh",
        "mapped_dispatch_bmu_count",
        "region_only_dispatch_bmu_count",
        "unmapped_dispatch_bmu_count",
        "estimated_dispatch_share_of_total_dispatch",
        "blocked_dispatch_share_of_total_dispatch",
        "unmapped_dispatch_share_of_total_dispatch",
        "estimated_capture_ratio_vs_qa_target",
        "blocked_dispatch_ratio_vs_qa_target",
        "qa_shortfall_after_blocked_dispatch_mwh",
        "dispatch_alignment_inference",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame = _ensure_truth_diagnostic_columns(frame)
    dispatch = frame[frame["dispatch_truth_flag"]].copy()
    if dispatch.empty:
        return pd.DataFrame(columns=columns)

    mapped_mask = dispatch["mapping_status"].eq("mapped")
    region_only_mask = dispatch["mapping_status"].eq("region_only")
    unmapped_mask = ~(mapped_mask | region_only_mask)
    blocked_mask = ~dispatch["lost_energy_estimate_flag"]
    estimated_mask = dispatch["lost_energy_estimate_flag"]

    grouped = dispatch.groupby("settlement_date", as_index=False).agg(
        qa_target_definition=("qa_target_definition", _first_mode),
        gb_daily_qa_target_mwh=("gb_daily_qa_target_mwh", "max"),
        gb_daily_estimated_lost_energy_mwh=("gb_daily_estimated_lost_energy_mwh", "max"),
        total_dispatch_down_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        estimated_dispatch_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        blocked_dispatch_half_hour_count=("lost_energy_estimate_flag", lambda values: int((~pd.Series(values).fillna(False).astype(bool)).sum())),
        estimated_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[estimated_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        blocked_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[blocked_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        blocked_outage_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("outage")]
                .fillna(0.0)
                .sum()
            ),
        ),
        blocked_availability_unknown_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("availability_unknown")]
                .fillna(0.0)
                .sum()
            ),
        ),
        blocked_physical_below_generation_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("physical_below_generation")]
                .fillna(0.0)
                .sum()
            ),
        ),
        blocked_physical_inconsistent_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("physical_inconsistent")]
                .fillna(0.0)
                .sum()
            ),
        ),
        mapped_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[mapped_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        region_only_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[region_only_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        unmapped_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[unmapped_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        mapped_estimated_lost_energy_mwh=(
            "lost_energy_mwh",
            lambda values: float(pd.Series(values)[mapped_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        region_only_estimated_lost_energy_mwh=(
            "lost_energy_mwh",
            lambda values: float(pd.Series(values)[region_only_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        unmapped_estimated_lost_energy_mwh=(
            "lost_energy_mwh",
            lambda values: float(pd.Series(values)[unmapped_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        mapped_dispatch_bmu_count=(
            "elexon_bm_unit",
            lambda values: int(dispatch.loc[values.index, "elexon_bm_unit"][mapped_mask.loc[values.index]].nunique()),
        ),
        region_only_dispatch_bmu_count=(
            "elexon_bm_unit",
            lambda values: int(dispatch.loc[values.index, "elexon_bm_unit"][region_only_mask.loc[values.index]].nunique()),
        ),
        unmapped_dispatch_bmu_count=(
            "elexon_bm_unit",
            lambda values: int(dispatch.loc[values.index, "elexon_bm_unit"][unmapped_mask.loc[values.index]].nunique()),
        ),
    )

    grouped["blocked_other_dispatch_down_mwh_lower_bound"] = (
        grouped["blocked_dispatch_down_mwh_lower_bound"]
        - grouped["blocked_outage_dispatch_down_mwh_lower_bound"]
        - grouped["blocked_availability_unknown_dispatch_down_mwh_lower_bound"]
        - grouped["blocked_physical_below_generation_dispatch_down_mwh_lower_bound"]
        - grouped["blocked_physical_inconsistent_dispatch_down_mwh_lower_bound"]
    )
    grouped["qa_target_shortfall_mwh"] = grouped["gb_daily_qa_target_mwh"] - grouped["gb_daily_estimated_lost_energy_mwh"]
    grouped["estimated_dispatch_share_of_total_dispatch"] = _safe_target_ratio(
        grouped["estimated_dispatch_down_mwh_lower_bound"],
        grouped["total_dispatch_down_mwh_lower_bound"],
    )
    grouped["blocked_dispatch_share_of_total_dispatch"] = _safe_target_ratio(
        grouped["blocked_dispatch_down_mwh_lower_bound"],
        grouped["total_dispatch_down_mwh_lower_bound"],
    )
    grouped["unmapped_dispatch_share_of_total_dispatch"] = _safe_target_ratio(
        grouped["unmapped_dispatch_down_mwh_lower_bound"],
        grouped["total_dispatch_down_mwh_lower_bound"],
    )
    grouped["estimated_capture_ratio_vs_qa_target"] = _safe_target_ratio(
        grouped["gb_daily_estimated_lost_energy_mwh"],
        grouped["gb_daily_qa_target_mwh"],
    )
    grouped["blocked_dispatch_ratio_vs_qa_target"] = _safe_target_ratio(
        grouped["blocked_dispatch_down_mwh_lower_bound"],
        grouped["gb_daily_qa_target_mwh"],
    )
    grouped["qa_shortfall_after_blocked_dispatch_mwh"] = (
        grouped["qa_target_shortfall_mwh"] - grouped["blocked_dispatch_down_mwh_lower_bound"]
    )
    grouped["dispatch_alignment_inference"] = np.select(
        [
            grouped["gb_daily_qa_target_mwh"].isna(),
            grouped["qa_target_shortfall_mwh"].le(0),
            grouped["qa_shortfall_after_blocked_dispatch_mwh"].gt(0),
            grouped["blocked_dispatch_down_mwh_lower_bound"].gt(0),
        ],
        [
            "qa_target_missing",
            "qa_target_met_or_exceeded",
            "dispatch_source_shortfall_inferred",
            "dispatch_blocking_material",
        ],
        default="dispatch_alignment_unclear",
    )
    return grouped[columns].sort_values("settlement_date").reset_index(drop=True)


def build_fact_dispatch_alignment_bmu_daily(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "dispatch_half_hour_count",
        "estimated_dispatch_half_hour_count",
        "blocked_dispatch_half_hour_count",
        "accepted_down_delta_mwh_lower_bound",
        "estimated_dispatch_down_mwh_lower_bound",
        "blocked_dispatch_down_mwh_lower_bound",
        "blocked_outage_dispatch_down_mwh_lower_bound",
        "blocked_availability_unknown_dispatch_down_mwh_lower_bound",
        "blocked_physical_below_generation_dispatch_down_mwh_lower_bound",
        "blocked_physical_inconsistent_dispatch_down_mwh_lower_bound",
        "lost_energy_mwh",
        "dispatch_alignment_state",
        "primary_dispatch_block_reason",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame = _ensure_truth_diagnostic_columns(frame)
    dispatch = frame[frame["dispatch_truth_flag"]].copy()
    if dispatch.empty:
        return pd.DataFrame(columns=columns)

    blocked_mask = ~dispatch["lost_energy_estimate_flag"]
    estimated_mask = dispatch["lost_energy_estimate_flag"]

    grouped = dispatch.groupby(
        [
            "settlement_date",
            "elexon_bm_unit",
            "national_grid_bm_unit",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "mapping_status",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        dispatch_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        estimated_dispatch_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        blocked_dispatch_half_hour_count=("lost_energy_estimate_flag", lambda values: int((~pd.Series(values).fillna(False).astype(bool)).sum())),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        estimated_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[estimated_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        blocked_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(pd.Series(values)[blocked_mask.loc[values.index]].fillna(0.0).sum()),
        ),
        blocked_outage_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("outage")]
                .fillna(0.0)
                .sum()
            ),
        ),
        blocked_availability_unknown_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("availability_unknown")]
                .fillna(0.0)
                .sum()
            ),
        ),
        blocked_physical_below_generation_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("physical_below_generation")]
                .fillna(0.0)
                .sum()
            ),
        ),
        blocked_physical_inconsistent_dispatch_down_mwh_lower_bound=(
            "accepted_down_delta_mwh_lower_bound",
            lambda values: float(
                pd.Series(values)[blocked_mask.loc[values.index] & dispatch.loc[values.index, "lost_energy_block_reason"].eq("physical_inconsistent")]
                .fillna(0.0)
                .sum()
            ),
        ),
        lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        primary_dispatch_block_reason=(
            "lost_energy_block_reason",
            lambda values: _first_mode(pd.Series(values)[pd.Series(values).ne("estimated")]),
        ),
    )
    grouped["dispatch_alignment_state"] = np.select(
        [
            grouped["blocked_dispatch_half_hour_count"].eq(0),
            grouped["estimated_dispatch_half_hour_count"].eq(0),
        ],
        [
            "fully_estimated",
            "fully_blocked",
        ],
        default="partially_blocked",
    )
    return grouped[columns].sort_values(
        ["settlement_date", "blocked_dispatch_down_mwh_lower_bound", "accepted_down_delta_mwh_lower_bound"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_fact_curtailment_gap_reason_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "settlement_date",
                "lost_energy_block_reason",
                "dispatch_half_hour_count",
                "distinct_bmu_count",
                "accepted_down_delta_mwh_lower_bound",
                "lost_energy_mwh",
                "share_of_dispatch_rows",
                "share_of_dispatch_down_mwh_lower_bound",
            ]
        )

    frame = _ensure_truth_diagnostic_columns(frame)
    dispatch_frame = frame[frame["dispatch_truth_flag"]].copy()
    if dispatch_frame.empty:
        return pd.DataFrame(
            columns=[
                "settlement_date",
                "lost_energy_block_reason",
                "dispatch_half_hour_count",
                "distinct_bmu_count",
                "accepted_down_delta_mwh_lower_bound",
                "lost_energy_mwh",
                "share_of_dispatch_rows",
                "share_of_dispatch_down_mwh_lower_bound",
            ]
        )

    grouped = dispatch_frame.groupby(["settlement_date", "lost_energy_block_reason"], as_index=False).agg(
        dispatch_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        distinct_bmu_count=("elexon_bm_unit", "nunique"),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())),
    )
    daily_totals = grouped.groupby("settlement_date", as_index=False).agg(
        total_dispatch_half_hour_count=("dispatch_half_hour_count", "sum"),
        total_dispatch_down_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
    )
    grouped = grouped.merge(daily_totals, on="settlement_date", how="left")
    grouped["share_of_dispatch_rows"] = np.where(
        grouped["total_dispatch_half_hour_count"] > 0,
        grouped["dispatch_half_hour_count"] / grouped["total_dispatch_half_hour_count"],
        np.nan,
    )
    grouped["share_of_dispatch_down_mwh_lower_bound"] = np.where(
        grouped["total_dispatch_down_mwh_lower_bound"] > 0,
        grouped["accepted_down_delta_mwh_lower_bound"] / grouped["total_dispatch_down_mwh_lower_bound"],
        np.nan,
    )
    return grouped.drop(columns=["total_dispatch_half_hour_count", "total_dispatch_down_mwh_lower_bound"]).sort_values(
        ["settlement_date", "accepted_down_delta_mwh_lower_bound"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_fact_bmu_curtailment_gap_bmu_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "settlement_date",
                "elexon_bm_unit",
                "national_grid_bm_unit",
                "cluster_key",
                "cluster_label",
                "parent_region",
                "mapping_status",
                "dispatch_half_hour_count",
                "lost_energy_estimate_half_hour_count",
                "accepted_down_delta_mwh_lower_bound",
                "lost_energy_mwh",
                "dispatch_minus_lost_energy_gap_mwh",
                "primary_dispatch_block_reason",
            ]
        )

    frame = _ensure_truth_diagnostic_columns(frame)
    dispatch_frame = frame[frame["dispatch_truth_flag"]].copy()
    if dispatch_frame.empty:
        return pd.DataFrame(
            columns=[
                "settlement_date",
                "elexon_bm_unit",
                "national_grid_bm_unit",
                "cluster_key",
                "cluster_label",
                "parent_region",
                "mapping_status",
                "dispatch_half_hour_count",
                "lost_energy_estimate_half_hour_count",
                "accepted_down_delta_mwh_lower_bound",
                "lost_energy_mwh",
                "dispatch_minus_lost_energy_gap_mwh",
                "primary_dispatch_block_reason",
            ]
        )

    grouped = dispatch_frame.groupby(
        [
            "settlement_date",
            "elexon_bm_unit",
            "national_grid_bm_unit",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "mapping_status",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        dispatch_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        lost_energy_estimate_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        primary_dispatch_block_reason=(
            "lost_energy_block_reason",
            lambda values: _first_mode(pd.Series(values)[pd.Series(values).ne("estimated")]),
        ),
    )
    grouped["dispatch_minus_lost_energy_gap_mwh"] = (
        grouped["accepted_down_delta_mwh_lower_bound"] - grouped["lost_energy_mwh"]
    )
    return grouped.sort_values(
        ["settlement_date", "dispatch_minus_lost_energy_gap_mwh", "accepted_down_delta_mwh_lower_bound"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


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
    frame["counterfactual_invalid_reason"] = _derive_counterfactual_invalid_reason(frame)
    frame["lost_energy_block_reason"] = _derive_lost_energy_block_reason(frame)

    frame["precision_profile_include"] = (
        frame["truth_tier"].isin({"physical_baseline", "weather_calibrated"})
        & frame["availability_state"].eq("available")
        & frame["counterfactual_valid_flag"]
        & frame["qa_reconciliation_status"].eq("pass")
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
        "mapping_status",
        "generation_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "accepted_up_delta_mwh_lower_bound",
        "availability_state",
        "remit_active_flag",
        "availability_confidence",
        "counterfactual_method",
        "counterfactual_mwh",
        "counterfactual_valid_flag",
        "counterfactual_invalid_reason",
        "lost_energy_mwh",
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "lost_energy_block_reason",
        "truth_tier",
        "precision_profile_include",
        "research_profile_include",
        "qa_target_definition",
        "qa_reconciliation_status",
        "gb_daily_qa_target_mwh",
        "qa_reconciliation_abs_error_mwh",
        "qa_reconciliation_relative_error",
        "gb_daily_raw_constraint_total_mwh",
        "raw_reconciliation_abs_error_mwh",
        "raw_reconciliation_relative_error",
        "raw_reconciliation_status",
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

    fact_bmu_curtailment_truth_half_hourly_all = build_fact_bmu_curtailment_truth_half_hourly(
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
    fact_curtailment_reconciliation_daily = build_fact_curtailment_reconciliation_daily(
        fact_bmu_curtailment_truth_half_hourly_all
    )
    fact_dispatch_alignment_daily = build_fact_dispatch_alignment_daily(
        fact_bmu_curtailment_truth_half_hourly_all
    )
    fact_dispatch_alignment_bmu_daily = build_fact_dispatch_alignment_bmu_daily(
        fact_bmu_curtailment_truth_half_hourly_all
    )
    fact_curtailment_gap_reason_daily = build_fact_curtailment_gap_reason_daily(
        fact_bmu_curtailment_truth_half_hourly_all
    )
    fact_bmu_curtailment_gap_bmu_daily = build_fact_bmu_curtailment_gap_bmu_daily(
        fact_bmu_curtailment_truth_half_hourly_all
    )
    fact_bmu_curtailment_truth_half_hourly = filter_truth_profile(
        fact_bmu_curtailment_truth_half_hourly_all,
        truth_profile=truth_profile,
    )

    frames = {
        "fact_bmu_physical_position_half_hourly": fact_bmu_physical_position_half_hourly,
        "fact_bmu_availability_half_hourly": fact_bmu_availability_half_hourly,
        "fact_bmu_curtailment_truth_half_hourly": fact_bmu_curtailment_truth_half_hourly,
        "fact_curtailment_reconciliation_daily": fact_curtailment_reconciliation_daily,
        "fact_dispatch_alignment_daily": fact_dispatch_alignment_daily,
        "fact_dispatch_alignment_bmu_daily": fact_dispatch_alignment_bmu_daily,
        "fact_curtailment_gap_reason_daily": fact_curtailment_gap_reason_daily,
        "fact_bmu_curtailment_gap_bmu_daily": fact_bmu_curtailment_gap_bmu_daily,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)
    return frames
