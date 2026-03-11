from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from history_store import upsert_frame_to_sqlite
from truth_store_focus import (
    _coerce_bool_series,
    _first_mode,
    _load_table,
    _prepare_dispatch_source_gap_candidates,
    materialize_truth_store_source_focus,
)


FAMILY_DISPATCH_FORENSIC_DAILY_TABLE = "fact_family_dispatch_forensic_daily"
FAMILY_DISPATCH_FORENSIC_BMU_TABLE = "fact_family_dispatch_forensic_bmu_daily"
FAMILY_DISPATCH_FORENSIC_HALF_HOURLY_TABLE = "fact_family_dispatch_forensic_half_hourly"
FAMILY_PHYSICAL_FORENSIC_DAILY_TABLE = "fact_family_physical_forensic_daily"
FAMILY_PHYSICAL_FORENSIC_BMU_TABLE = "fact_family_physical_forensic_bmu_daily"
FAMILY_PHYSICAL_FORENSIC_HALF_HOURLY_TABLE = "fact_family_physical_forensic_half_hourly"
FAMILY_PUBLICATION_AUDIT_DAILY_TABLE = "fact_family_publication_audit_daily"
FAMILY_PUBLICATION_AUDIT_BMU_TABLE = "fact_family_publication_audit_bmu_daily"
FAMILY_SUPPORT_EVIDENCE_HALF_HOURLY_TABLE = "fact_family_support_evidence_half_hourly"


def normalize_forensic_family_keys(family_keys: Sequence[str] | str | None) -> list[str]:
    if family_keys is None:
        values: Iterable[str] = ["HOWAO", "HOWBO"]
    elif isinstance(family_keys, str):
        values = family_keys.split(",")
    else:
        values = family_keys
    normalized = sorted({str(value).strip().upper() for value in values if str(value).strip()})
    if not normalized:
        raise ValueError("at least one forensic family key is required")
    return normalized


def forensic_scope_key_for_family_keys(family_keys: Sequence[str] | str | None) -> str:
    return "+".join(normalize_forensic_family_keys(family_keys))


def _apply_date_filter(
    frame: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if frame.empty or "settlement_date" not in frame.columns:
        return frame.copy()
    prepared = frame.copy()
    dates = prepared["settlement_date"].astype(str)
    if start_date:
        prepared = prepared[dates >= start_date]
        dates = prepared["settlement_date"].astype(str)
    if end_date:
        prepared = prepared[dates <= end_date]
    return prepared.reset_index(drop=True)


def _build_forensic_state(
    mapping_status: pd.Series,
    no_window_candidate: pd.Series,
    family_window_candidate: pd.Series,
    acceptance_window_candidate: pd.Series,
    source_gap_candidate: pd.Series,
) -> pd.Series:
    return np.select(
        [
            mapping_status.fillna("unmapped").eq("unmapped"),
            mapping_status.fillna("unmapped").eq("region_only"),
            no_window_candidate.gt(0.0),
            family_window_candidate.gt(0.0) | acceptance_window_candidate.gt(0.0),
            source_gap_candidate.le(0.0),
        ],
        [
            "mapping_gap",
            "region_only_source_gap",
            "mapped_no_window_source_gap",
            "window_rule_gap",
            "no_source_gap_candidates",
        ],
        default="inspect",
    )


def _build_publication_audit_state(
    physical_without_boalf_negative_bid_count: pd.Series,
    availability_like_dynamic_limit_count: pd.Series,
    physical_without_boalf_count: pd.Series,
    dispatch_truth_count: pd.Series,
) -> pd.Series:
    return np.select(
        [
            physical_without_boalf_negative_bid_count.gt(0),
            availability_like_dynamic_limit_count.gt(0),
            physical_without_boalf_count.gt(0),
            dispatch_truth_count.gt(0),
        ],
        [
            "physical_without_boalf_negative_bid",
            "availability_like_dynamic_limit",
            "physical_without_boalf",
            "captured_dispatch_present",
        ],
        default="no_publication_audit_signal",
    )


def _build_support_question_code(
    sentinel_count: pd.Series,
    physical_without_boalf_negative_bid_count: pd.Series,
    availability_like_dynamic_limit_count: pd.Series,
    physical_without_boalf_count: pd.Series,
) -> pd.Series:
    return np.select(
        [
            sentinel_count.gt(0),
            physical_without_boalf_negative_bid_count.gt(0),
            availability_like_dynamic_limit_count.gt(0),
            physical_without_boalf_count.gt(0),
        ],
        [
            "query_bod_sentinel_and_missing_boalf",
            "query_missing_boalf_with_negative_bid_and_physical_gap",
            "query_dynamic_limit_change_without_boalf",
            "query_physical_gap_without_boalf",
        ],
        default="no_support_case",
    )


def _build_support_recommended_action(
    support_question_code: pd.Series,
) -> pd.Series:
    return np.select(
        [
            support_question_code.eq("query_bod_sentinel_and_missing_boalf"),
            support_question_code.eq("query_missing_boalf_with_negative_bid_and_physical_gap"),
            support_question_code.eq("query_dynamic_limit_change_without_boalf"),
            support_question_code.eq("query_physical_gap_without_boalf"),
        ],
        [
            "ask_elexon_about_suspect_bod_sentinel_and_missing_published_boalf",
            "ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf",
            "ask_elexon_whether_dynamic_limit_changes_can_occur_without_published_boalf",
            "ask_elexon_why_physical_gap_exists_without_published_boalf",
        ],
        default="no_support_escalation",
    )


def _prepare_family_forensics_context(
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    normalized_family_keys = normalize_forensic_family_keys(family_keys)
    scope_key = forensic_scope_key_for_family_keys(normalized_family_keys)

    truth = _prepare_dispatch_source_gap_candidates(fact_bmu_curtailment_truth_half_hourly)
    truth = truth[truth["bmu_family_key"].isin(normalized_family_keys)].copy()
    truth = _apply_date_filter(truth, start_date=start_date, end_date=end_date)
    optional_truth_defaults = {
        "interval_start_utc": pd.NA,
        "interval_end_utc": pd.NA,
        "national_grid_bm_unit": pd.NA,
        "availability_state": pd.NA,
        "counterfactual_invalid_reason": pd.NA,
        "dispatch_truth_source_tier": "none",
        "dispatch_inference_scope": "none",
        "negative_bid_pair_count": 0,
        "dispatch_acceptance_window_flag": False,
        "family_day_dispatch_window_flag": False,
        "valid_negative_bid_pair_count": 0,
        "sentinel_bid_pair_count": 0,
        "sentinel_offer_pair_count": 0,
        "sentinel_pair_count": 0,
        "sentinel_bid_available_flag": False,
        "sentinel_offer_available_flag": False,
        "sentinel_pair_available_flag": False,
    }
    for column, default in optional_truth_defaults.items():
        if column not in truth.columns:
            truth[column] = default
    truth_numeric_defaults = {
        "negative_bid_pair_count": 0,
        "valid_negative_bid_pair_count": 0,
        "sentinel_bid_pair_count": 0,
        "sentinel_offer_pair_count": 0,
        "sentinel_pair_count": 0,
    }
    for column, default in truth_numeric_defaults.items():
        truth[column] = pd.to_numeric(truth[column], errors="coerce").fillna(default)
    truth_bool_columns = [
        "dispatch_acceptance_window_flag",
        "family_day_dispatch_window_flag",
        "sentinel_bid_available_flag",
        "sentinel_offer_available_flag",
        "sentinel_pair_available_flag",
    ]
    for column in truth_bool_columns:
        truth[column] = _coerce_bool_series(truth[column], index=truth.index)

    family_gap = fact_dispatch_source_gap_family_daily.copy()
    if "bmu_family_key" not in family_gap.columns:
        family_gap["bmu_family_key"] = pd.NA
    family_gap = family_gap[family_gap["bmu_family_key"].isin(normalized_family_keys)].copy()
    family_gap = _apply_date_filter(family_gap, start_date=start_date, end_date=end_date)
    optional_gap_defaults = {
        "qa_reconciliation_status": pd.NA,
        "recoverability_audit_state": pd.NA,
        "source_gap_next_action": pd.NA,
        "family_source_gap_next_action": pd.NA,
        "source_gap_share_of_day_total": np.nan,
        "source_gap_share_of_remaining_qa_shortfall": np.nan,
    }
    for column, default in optional_gap_defaults.items():
        if column not in family_gap.columns:
            family_gap[column] = default
    return truth.reset_index(drop=True), family_gap.reset_index(drop=True), scope_key


def _prepare_family_physical_forensics_context(
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    truth, family_gap, scope_key = _prepare_family_forensics_context(
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    physical = fact_bmu_physical_position_half_hourly.copy()
    if physical.empty:
        return pd.DataFrame(), family_gap, scope_key

    physical = _apply_date_filter(physical, start_date=start_date, end_date=end_date)
    physical_columns = [
        "settlement_date",
        "settlement_period",
        "elexon_bm_unit",
        "pn_mwh",
        "qpn_mwh",
        "mils_mwh",
        "mels_mwh",
        "generation_mwh",
        "physical_baseline_source_dataset",
        "physical_baseline_mwh",
        "physical_consistency_flag",
        "counterfactual_valid_flag",
    ]
    for column in physical_columns:
        if column not in physical.columns:
            physical[column] = np.nan if column.endswith("_mwh") else pd.NA

    merged = truth.merge(
        physical[physical_columns],
        on=["settlement_date", "settlement_period", "elexon_bm_unit"],
        how="left",
        suffixes=("", "_physical"),
    )
    backfill_columns = [
        "pn_mwh",
        "qpn_mwh",
        "mils_mwh",
        "mels_mwh",
        "generation_mwh",
        "physical_baseline_source_dataset",
        "physical_baseline_mwh",
        "physical_consistency_flag",
        "counterfactual_valid_flag",
    ]
    for column in backfill_columns:
        physical_column = f"{column}_physical"
        if physical_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[physical_column]
        else:
            merged[column] = merged[column].where(merged[column].notna(), merged[physical_column])
        merged = merged.drop(columns=[physical_column])

    numeric_defaults = {
        "pn_mwh": 0.0,
        "qpn_mwh": 0.0,
        "mils_mwh": 0.0,
        "mels_mwh": 0.0,
        "generation_mwh": 0.0,
        "physical_baseline_mwh": 0.0,
        "accepted_down_delta_mwh_lower_bound": 0.0,
        "dispatch_down_evidence_mwh_lower_bound": 0.0,
        "lost_energy_mwh": 0.0,
        "physical_dispatch_down_gap_mwh": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column not in merged.columns:
            merged[column] = default
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(default)

    bool_defaults = {
        "negative_bid_available_flag": False,
        "sentinel_pair_available_flag": False,
        "dispatch_truth_flag": False,
        "lost_energy_estimate_flag": False,
        "counterfactual_valid_flag": False,
        "physical_consistency_flag": False,
    }
    for column, default in bool_defaults.items():
        if column not in merged.columns:
            merged[column] = default
        merged[column] = _coerce_bool_series(merged[column], index=merged.index)

    merged["positive_pn_qpn_gap_flag"] = merged["pn_mwh"].gt(merged["qpn_mwh"] + 1e-6)
    merged["positive_zero_boalf_gap_flag"] = (
        merged["positive_pn_qpn_gap_flag"] & merged["accepted_down_delta_mwh_lower_bound"].le(0.0)
    )
    merged["published_boalf_absent_flag"] = merged["accepted_down_delta_mwh_lower_bound"].le(0.0)
    merged["positive_zero_boalf_negative_bid_gap_flag"] = (
        merged["positive_zero_boalf_gap_flag"] & merged["negative_bid_available_flag"]
    )
    merged["positive_zero_boalf_sentinel_gap_flag"] = (
        merged["positive_zero_boalf_gap_flag"] & merged["sentinel_pair_available_flag"]
    )
    merged["mels_reduction_flag"] = merged["mels_mwh"].notna() & merged["pn_mwh"].gt(merged["mels_mwh"] + 1e-6)
    merged["mils_floor_flag"] = merged["mils_mwh"].notna() & merged["qpn_mwh"].lt(merged["mils_mwh"] - 1e-6)
    merged["availability_like_dynamic_limit_flag"] = (
        merged["positive_zero_boalf_gap_flag"]
        & ~merged["negative_bid_available_flag"]
        & (merged["mels_reduction_flag"] | merged["mils_floor_flag"])
    )
    merged["physical_without_boalf_gap_mwh"] = merged["physical_dispatch_down_gap_mwh"].where(
        merged["positive_zero_boalf_gap_flag"], 0.0
    )
    merged["physical_without_boalf_negative_bid_gap_mwh"] = merged["physical_dispatch_down_gap_mwh"].where(
        merged["positive_zero_boalf_negative_bid_gap_flag"], 0.0
    )
    merged["physical_without_boalf_sentinel_gap_mwh"] = merged["physical_dispatch_down_gap_mwh"].where(
        merged["positive_zero_boalf_sentinel_gap_flag"], 0.0
    )
    merged["availability_like_dynamic_limit_gap_mwh"] = merged["physical_dispatch_down_gap_mwh"].where(
        merged["availability_like_dynamic_limit_flag"], 0.0
    )
    merged["support_case_ready_flag"] = (
        merged["positive_zero_boalf_gap_flag"]
        | merged["availability_like_dynamic_limit_flag"]
        | merged["sentinel_pair_available_flag"]
    )
    return merged.reset_index(drop=True), family_gap.reset_index(drop=True), scope_key


def build_fact_family_dispatch_forensic_daily(
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "source_gap_next_action",
        "family_source_gap_next_action",
        "distinct_bmu_count",
        "half_hour_count",
        "negative_bid_half_hour_count",
        "dispatch_truth_half_hour_count",
        "lost_energy_estimate_half_hour_count",
        "source_gap_candidate_row_count",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_mwh_lower_bound",
        "same_bmu_dispatch_increment_mwh_lower_bound",
        "family_day_dispatch_increment_mwh_lower_bound",
        "source_gap_candidate_mwh_lower_bound",
        "acceptance_window_candidate_mwh_lower_bound",
        "family_window_candidate_mwh_lower_bound",
        "no_window_candidate_mwh_lower_bound",
        "physical_dispatch_down_gap_mwh",
        "lost_energy_mwh",
        "available_half_hour_count",
        "unknown_availability_half_hour_count",
        "outage_half_hour_count",
        "most_negative_bid_gbp_per_mwh",
        "primary_lost_energy_block_reason",
        "source_gap_share_of_day_total",
        "source_gap_share_of_remaining_qa_shortfall",
        "forensic_priority_rank",
        "forensic_state",
    ]
    truth, family_gap, scope_key = _prepare_family_forensics_context(
        fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily,
        family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if truth.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "negative_bid_available_flag",
        "source_gap_candidate_flag",
    ]:
        truth[column] = _coerce_bool_series(truth.get(column, False), index=truth.index)

    grouped = truth.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label"],
        as_index=False,
        dropna=False,
    ).agg(
        cluster_key=("cluster_key", _first_mode),
        cluster_label=("cluster_label", _first_mode),
        parent_region=("parent_region", _first_mode),
        mapping_status=("mapping_status", _first_mode),
        distinct_bmu_count=("elexon_bm_unit", "nunique"),
        half_hour_count=("settlement_period", "count"),
        negative_bid_half_hour_count=("negative_bid_available_flag", lambda values: int(pd.Series(values).sum())),
        dispatch_truth_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).sum())),
        lost_energy_estimate_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).sum())),
        source_gap_candidate_row_count=("source_gap_candidate_flag", lambda values: int(pd.Series(values).sum())),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
        dispatch_down_mwh_lower_bound=("dispatch_down_evidence_mwh_lower_bound", "sum"),
        same_bmu_dispatch_increment_mwh_lower_bound=("same_bmu_dispatch_increment_mwh_lower_bound", "sum"),
        family_day_dispatch_increment_mwh_lower_bound=("family_day_dispatch_increment_mwh_lower_bound", "sum"),
        source_gap_candidate_mwh_lower_bound=("source_gap_candidate_mwh_lower_bound", "sum"),
        acceptance_window_candidate_mwh_lower_bound=("acceptance_window_candidate_mwh_lower_bound", "sum"),
        family_window_candidate_mwh_lower_bound=("family_window_candidate_mwh_lower_bound", "sum"),
        no_window_candidate_mwh_lower_bound=("no_window_candidate_mwh_lower_bound", "sum"),
        physical_dispatch_down_gap_mwh=("physical_dispatch_down_gap_mwh", "sum"),
        lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        available_half_hour_count=("availability_state", lambda values: int(pd.Series(values).fillna("").eq("available").sum())),
        unknown_availability_half_hour_count=("availability_state", lambda values: int(pd.Series(values).fillna("").eq("unknown").sum())),
        outage_half_hour_count=("availability_state", lambda values: int(pd.Series(values).fillna("").eq("outage").sum())),
        most_negative_bid_gbp_per_mwh=("most_negative_bid_gbp_per_mwh", lambda values: float(pd.Series(values).dropna().min()) if pd.Series(values).dropna().size else np.nan),
        primary_lost_energy_block_reason=("lost_energy_block_reason", lambda values: _first_mode(pd.Series(values)[pd.Series(values).fillna("").ne("estimated")])),
    )
    grouped["forensic_scope_key"] = scope_key
    grouped = grouped.merge(
        family_gap[
            [
                "settlement_date",
                "bmu_family_key",
                "qa_reconciliation_status",
                "recoverability_audit_state",
                "source_gap_next_action",
                "family_source_gap_next_action",
                "source_gap_share_of_day_total",
                "source_gap_share_of_remaining_qa_shortfall",
            ]
        ],
        on=["settlement_date", "bmu_family_key"],
        how="left",
    )
    grouped["forensic_priority_rank"] = (
        grouped.groupby("settlement_date")["source_gap_candidate_mwh_lower_bound"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    grouped["forensic_state"] = _build_forensic_state(
        grouped["mapping_status"],
        grouped["no_window_candidate_mwh_lower_bound"],
        grouped["family_window_candidate_mwh_lower_bound"],
        grouped["acceptance_window_candidate_mwh_lower_bound"],
        grouped["source_gap_candidate_mwh_lower_bound"],
    )
    return grouped[columns].sort_values(
        ["settlement_date", "forensic_priority_rank", "source_gap_candidate_mwh_lower_bound"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_fact_family_dispatch_forensic_bmu_daily(
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "half_hour_count",
        "negative_bid_half_hour_count",
        "dispatch_truth_half_hour_count",
        "lost_energy_estimate_half_hour_count",
        "source_gap_candidate_row_count",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_mwh_lower_bound",
        "source_gap_candidate_mwh_lower_bound",
        "acceptance_window_candidate_mwh_lower_bound",
        "family_window_candidate_mwh_lower_bound",
        "no_window_candidate_mwh_lower_bound",
        "physical_dispatch_down_gap_mwh",
        "lost_energy_mwh",
        "most_negative_bid_gbp_per_mwh",
        "primary_lost_energy_block_reason",
        "bmu_forensic_rank_within_family_day",
        "bmu_forensic_state",
    ]
    truth, family_gap, scope_key = _prepare_family_forensics_context(
        fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily,
        family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if truth.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "negative_bid_available_flag",
        "source_gap_candidate_flag",
    ]:
        truth[column] = _coerce_bool_series(truth.get(column, False), index=truth.index)

    grouped = truth.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label", "elexon_bm_unit"],
        as_index=False,
        dropna=False,
    ).agg(
        national_grid_bm_unit=("national_grid_bm_unit", _first_mode),
        cluster_key=("cluster_key", _first_mode),
        cluster_label=("cluster_label", _first_mode),
        parent_region=("parent_region", _first_mode),
        mapping_status=("mapping_status", _first_mode),
        half_hour_count=("settlement_period", "count"),
        negative_bid_half_hour_count=("negative_bid_available_flag", lambda values: int(pd.Series(values).sum())),
        dispatch_truth_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).sum())),
        lost_energy_estimate_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).sum())),
        source_gap_candidate_row_count=("source_gap_candidate_flag", lambda values: int(pd.Series(values).sum())),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
        dispatch_down_mwh_lower_bound=("dispatch_down_evidence_mwh_lower_bound", "sum"),
        source_gap_candidate_mwh_lower_bound=("source_gap_candidate_mwh_lower_bound", "sum"),
        acceptance_window_candidate_mwh_lower_bound=("acceptance_window_candidate_mwh_lower_bound", "sum"),
        family_window_candidate_mwh_lower_bound=("family_window_candidate_mwh_lower_bound", "sum"),
        no_window_candidate_mwh_lower_bound=("no_window_candidate_mwh_lower_bound", "sum"),
        physical_dispatch_down_gap_mwh=("physical_dispatch_down_gap_mwh", "sum"),
        lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        most_negative_bid_gbp_per_mwh=("most_negative_bid_gbp_per_mwh", lambda values: float(pd.Series(values).dropna().min()) if pd.Series(values).dropna().size else np.nan),
        primary_lost_energy_block_reason=("lost_energy_block_reason", lambda values: _first_mode(pd.Series(values)[pd.Series(values).fillna("").ne("estimated")])),
    )
    grouped["forensic_scope_key"] = scope_key
    grouped = grouped.merge(
        family_gap[
            [
                "settlement_date",
                "bmu_family_key",
                "qa_reconciliation_status",
                "source_gap_next_action",
            ]
        ],
        on=["settlement_date", "bmu_family_key"],
        how="left",
    )
    grouped["bmu_forensic_rank_within_family_day"] = (
        grouped.groupby(["settlement_date", "bmu_family_key"])["source_gap_candidate_mwh_lower_bound"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    grouped["bmu_forensic_state"] = _build_forensic_state(
        grouped["mapping_status"],
        grouped["no_window_candidate_mwh_lower_bound"],
        grouped["family_window_candidate_mwh_lower_bound"],
        grouped["acceptance_window_candidate_mwh_lower_bound"],
        grouped["source_gap_candidate_mwh_lower_bound"],
    )
    return grouped[columns].sort_values(
        ["settlement_date", "bmu_family_key", "bmu_forensic_rank_within_family_day", "source_gap_candidate_mwh_lower_bound"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def build_fact_family_dispatch_forensic_half_hourly(
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "settlement_period",
        "interval_start_utc",
        "interval_end_utc",
        "bmu_family_key",
        "bmu_family_label",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "negative_bid_available_flag",
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "source_gap_candidate_flag",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_evidence_mwh_lower_bound",
        "same_bmu_dispatch_increment_mwh_lower_bound",
        "family_day_dispatch_increment_mwh_lower_bound",
        "source_gap_candidate_mwh_lower_bound",
        "acceptance_window_candidate_mwh_lower_bound",
        "family_window_candidate_mwh_lower_bound",
        "no_window_candidate_mwh_lower_bound",
        "physical_dispatch_down_gap_mwh",
        "lost_energy_mwh",
        "availability_state",
        "counterfactual_invalid_reason",
        "lost_energy_block_reason",
        "dispatch_truth_source_tier",
        "dispatch_inference_scope",
        "most_negative_bid_gbp_per_mwh",
        "negative_bid_pair_count",
        "half_hour_forensic_rank_within_family_day",
        "half_hour_forensic_state",
    ]
    truth, family_gap, scope_key = _prepare_family_forensics_context(
        fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily,
        family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if truth.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "negative_bid_available_flag",
        "source_gap_candidate_flag",
    ]:
        truth[column] = _coerce_bool_series(truth.get(column, False), index=truth.index)

    interesting_mask = (
        truth["negative_bid_available_flag"]
        | truth["dispatch_truth_flag"]
        | truth["lost_energy_estimate_flag"]
        | truth["source_gap_candidate_flag"]
    )
    interesting = truth[interesting_mask].copy()
    if interesting.empty:
        return pd.DataFrame(columns=columns)

    interesting["forensic_scope_key"] = scope_key
    interesting = interesting.merge(
        family_gap[
            [
                "settlement_date",
                "bmu_family_key",
                "qa_reconciliation_status",
                "source_gap_next_action",
            ]
        ],
        on=["settlement_date", "bmu_family_key"],
        how="left",
        suffixes=("", "_family_gap"),
    )
    if "qa_reconciliation_status_family_gap" in interesting.columns:
        interesting["qa_reconciliation_status"] = interesting["qa_reconciliation_status_family_gap"].where(
            interesting["qa_reconciliation_status_family_gap"].notna(),
            interesting["qa_reconciliation_status"],
        )
        interesting = interesting.drop(columns=["qa_reconciliation_status_family_gap"])
    interesting["half_hour_forensic_rank_within_family_day"] = (
        interesting.groupby(["settlement_date", "bmu_family_key"])["source_gap_candidate_mwh_lower_bound"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    interesting["half_hour_forensic_state"] = np.select(
        [
            interesting["source_gap_candidate_flag"] & interesting["no_window_candidate_mwh_lower_bound"].gt(0.0),
            interesting["source_gap_candidate_flag"] & interesting["family_window_candidate_mwh_lower_bound"].gt(0.0),
            interesting["source_gap_candidate_flag"] & interesting["acceptance_window_candidate_mwh_lower_bound"].gt(0.0),
            interesting["dispatch_truth_flag"],
            interesting["negative_bid_available_flag"],
        ],
        [
            "no_window_source_gap",
            "family_window_source_gap",
            "same_bmu_window_source_gap",
            "captured_dispatch",
            "negative_bid_context",
        ],
        default="inspect",
    )
    return interesting[columns].sort_values(
        ["settlement_date", "bmu_family_key", "half_hour_forensic_rank_within_family_day", "settlement_period", "elexon_bm_unit"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def build_fact_family_physical_forensic_daily(
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "distinct_bmu_count",
        "half_hour_count",
        "positive_pn_qpn_gap_row_count",
        "positive_zero_boalf_gap_row_count",
        "positive_zero_boalf_negative_bid_gap_row_count",
        "positive_zero_boalf_sentinel_gap_row_count",
        "dispatch_truth_half_hour_count",
        "lost_energy_estimate_half_hour_count",
        "pn_mwh",
        "qpn_mwh",
        "mils_mwh",
        "mels_mwh",
        "physical_baseline_mwh",
        "physical_dispatch_down_gap_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_evidence_mwh_lower_bound",
        "lost_energy_mwh",
        "most_negative_bid_gbp_per_mwh",
        "physical_forensic_priority_rank",
        "physical_forensic_state",
    ]
    merged, family_gap, scope_key = _prepare_family_physical_forensics_context(
        fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    grouped = merged.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label"],
        as_index=False,
        dropna=False,
    ).agg(
        cluster_key=("cluster_key", _first_mode),
        cluster_label=("cluster_label", _first_mode),
        parent_region=("parent_region", _first_mode),
        mapping_status=("mapping_status", _first_mode),
        distinct_bmu_count=("elexon_bm_unit", "nunique"),
        half_hour_count=("settlement_period", "count"),
        positive_pn_qpn_gap_row_count=("positive_pn_qpn_gap_flag", lambda values: int(pd.Series(values).sum())),
        positive_zero_boalf_gap_row_count=("positive_zero_boalf_gap_flag", lambda values: int(pd.Series(values).sum())),
        positive_zero_boalf_negative_bid_gap_row_count=(
            "positive_zero_boalf_negative_bid_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        positive_zero_boalf_sentinel_gap_row_count=(
            "positive_zero_boalf_sentinel_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        dispatch_truth_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).sum())),
        lost_energy_estimate_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).sum())),
        pn_mwh=("pn_mwh", "sum"),
        qpn_mwh=("qpn_mwh", "sum"),
        mils_mwh=("mils_mwh", "sum"),
        mels_mwh=("mels_mwh", "sum"),
        physical_baseline_mwh=("physical_baseline_mwh", "sum"),
        physical_dispatch_down_gap_mwh=("physical_dispatch_down_gap_mwh", "sum"),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
        dispatch_down_evidence_mwh_lower_bound=("dispatch_down_evidence_mwh_lower_bound", "sum"),
        lost_energy_mwh=("lost_energy_mwh", "sum"),
        most_negative_bid_gbp_per_mwh=(
            "most_negative_bid_gbp_per_mwh",
            lambda values: float(pd.Series(values).dropna().min()) if pd.Series(values).dropna().size else np.nan,
        ),
    )
    grouped["forensic_scope_key"] = scope_key
    grouped = grouped.merge(
        family_gap[["settlement_date", "bmu_family_key", "qa_reconciliation_status", "source_gap_next_action"]],
        on=["settlement_date", "bmu_family_key"],
        how="left",
    )
    grouped["physical_forensic_priority_rank"] = (
        grouped.groupby("settlement_date")["positive_zero_boalf_gap_row_count"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    grouped["physical_forensic_state"] = np.select(
        [
            grouped["positive_zero_boalf_negative_bid_gap_row_count"].gt(0),
            grouped["positive_zero_boalf_sentinel_gap_row_count"].gt(0),
            grouped["positive_zero_boalf_gap_row_count"].gt(0),
            grouped["dispatch_truth_half_hour_count"].gt(0),
        ],
        [
            "positive_zero_boalf_negative_bid_gap",
            "positive_zero_boalf_sentinel_gap",
            "positive_zero_boalf_gap",
            "captured_dispatch_present",
        ],
        default="no_positive_pn_qpn_gap",
    )
    return grouped[columns].sort_values(
        ["settlement_date", "physical_forensic_priority_rank", "positive_zero_boalf_gap_row_count"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_fact_family_physical_forensic_bmu_daily(
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "half_hour_count",
        "positive_pn_qpn_gap_row_count",
        "positive_zero_boalf_gap_row_count",
        "positive_zero_boalf_negative_bid_gap_row_count",
        "positive_zero_boalf_sentinel_gap_row_count",
        "dispatch_truth_half_hour_count",
        "lost_energy_estimate_half_hour_count",
        "pn_mwh",
        "qpn_mwh",
        "mils_mwh",
        "mels_mwh",
        "physical_dispatch_down_gap_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_evidence_mwh_lower_bound",
        "lost_energy_mwh",
        "most_negative_bid_gbp_per_mwh",
        "bmu_physical_forensic_rank",
        "bmu_physical_forensic_state",
    ]
    merged, family_gap, scope_key = _prepare_family_physical_forensics_context(
        fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    grouped = merged.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label", "elexon_bm_unit"],
        as_index=False,
        dropna=False,
    ).agg(
        national_grid_bm_unit=("national_grid_bm_unit", _first_mode),
        cluster_key=("cluster_key", _first_mode),
        cluster_label=("cluster_label", _first_mode),
        parent_region=("parent_region", _first_mode),
        mapping_status=("mapping_status", _first_mode),
        half_hour_count=("settlement_period", "count"),
        positive_pn_qpn_gap_row_count=("positive_pn_qpn_gap_flag", lambda values: int(pd.Series(values).sum())),
        positive_zero_boalf_gap_row_count=("positive_zero_boalf_gap_flag", lambda values: int(pd.Series(values).sum())),
        positive_zero_boalf_negative_bid_gap_row_count=(
            "positive_zero_boalf_negative_bid_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        positive_zero_boalf_sentinel_gap_row_count=(
            "positive_zero_boalf_sentinel_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        dispatch_truth_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).sum())),
        lost_energy_estimate_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).sum())),
        pn_mwh=("pn_mwh", "sum"),
        qpn_mwh=("qpn_mwh", "sum"),
        mils_mwh=("mils_mwh", "sum"),
        mels_mwh=("mels_mwh", "sum"),
        physical_dispatch_down_gap_mwh=("physical_dispatch_down_gap_mwh", "sum"),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
        dispatch_down_evidence_mwh_lower_bound=("dispatch_down_evidence_mwh_lower_bound", "sum"),
        lost_energy_mwh=("lost_energy_mwh", "sum"),
        most_negative_bid_gbp_per_mwh=(
            "most_negative_bid_gbp_per_mwh",
            lambda values: float(pd.Series(values).dropna().min()) if pd.Series(values).dropna().size else np.nan,
        ),
    )
    grouped["forensic_scope_key"] = scope_key
    grouped = grouped.merge(
        family_gap[["settlement_date", "bmu_family_key", "qa_reconciliation_status", "source_gap_next_action"]],
        on=["settlement_date", "bmu_family_key"],
        how="left",
    )
    grouped["bmu_physical_forensic_rank"] = (
        grouped.groupby(["settlement_date", "bmu_family_key"])["positive_zero_boalf_gap_row_count"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    grouped["bmu_physical_forensic_state"] = np.select(
        [
            grouped["positive_zero_boalf_negative_bid_gap_row_count"].gt(0),
            grouped["positive_zero_boalf_sentinel_gap_row_count"].gt(0),
            grouped["positive_zero_boalf_gap_row_count"].gt(0),
            grouped["dispatch_truth_half_hour_count"].gt(0),
        ],
        [
            "positive_zero_boalf_negative_bid_gap",
            "positive_zero_boalf_sentinel_gap",
            "positive_zero_boalf_gap",
            "captured_dispatch_present",
        ],
        default="no_positive_pn_qpn_gap",
    )
    return grouped[columns].sort_values(
        ["settlement_date", "bmu_family_key", "bmu_physical_forensic_rank", "positive_zero_boalf_gap_row_count"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def build_fact_family_physical_forensic_half_hourly(
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "settlement_period",
        "interval_start_utc",
        "interval_end_utc",
        "bmu_family_key",
        "bmu_family_label",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "pn_mwh",
        "qpn_mwh",
        "mils_mwh",
        "mels_mwh",
        "physical_baseline_mwh",
        "physical_baseline_source_dataset",
        "physical_consistency_flag",
        "counterfactual_valid_flag",
        "generation_mwh",
        "physical_dispatch_down_gap_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_evidence_mwh_lower_bound",
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "negative_bid_available_flag",
        "sentinel_pair_available_flag",
        "most_negative_bid_gbp_per_mwh",
        "lost_energy_block_reason",
        "dispatch_acceptance_window_flag",
        "family_day_dispatch_window_flag",
        "physical_half_hour_forensic_rank",
        "physical_half_hour_forensic_state",
    ]
    merged, family_gap, scope_key = _prepare_family_physical_forensics_context(
        fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    interesting = merged[
        merged["positive_pn_qpn_gap_flag"]
        | merged["dispatch_truth_flag"]
        | merged["sentinel_pair_available_flag"]
        | merged["negative_bid_available_flag"]
    ].copy()
    if interesting.empty:
        return pd.DataFrame(columns=columns)

    interesting["forensic_scope_key"] = scope_key
    interesting = interesting.merge(
        family_gap[["settlement_date", "bmu_family_key", "qa_reconciliation_status", "source_gap_next_action"]],
        on=["settlement_date", "bmu_family_key"],
        how="left",
        suffixes=("", "_family_gap"),
    )
    if "qa_reconciliation_status_family_gap" in interesting.columns:
        interesting["qa_reconciliation_status"] = interesting["qa_reconciliation_status_family_gap"].where(
            interesting["qa_reconciliation_status_family_gap"].notna(),
            interesting["qa_reconciliation_status"],
        )
        interesting = interesting.drop(columns=["qa_reconciliation_status_family_gap"])
    interesting["physical_half_hour_forensic_rank"] = (
        interesting.groupby(["settlement_date", "bmu_family_key"])["physical_dispatch_down_gap_mwh"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    interesting["physical_half_hour_forensic_state"] = np.select(
        [
            interesting["positive_zero_boalf_negative_bid_gap_flag"],
            interesting["positive_zero_boalf_sentinel_gap_flag"],
            interesting["positive_zero_boalf_gap_flag"],
            interesting["dispatch_truth_flag"],
        ],
        [
            "positive_zero_boalf_negative_bid_gap",
            "positive_zero_boalf_sentinel_gap",
            "positive_zero_boalf_gap",
            "captured_dispatch_present",
        ],
        default="physical_context_only",
    )
    return interesting[columns].sort_values(
        ["settlement_date", "bmu_family_key", "physical_half_hour_forensic_rank", "settlement_period", "elexon_bm_unit"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def build_fact_family_publication_audit_daily(
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "distinct_bmu_count",
        "half_hour_count",
        "published_boalf_absent_half_hour_count",
        "physical_without_boalf_half_hour_count",
        "physical_without_boalf_negative_bid_half_hour_count",
        "physical_without_boalf_sentinel_half_hour_count",
        "availability_like_dynamic_limit_half_hour_count",
        "dispatch_truth_half_hour_count",
        "lost_energy_estimate_half_hour_count",
        "physical_without_boalf_gap_mwh",
        "physical_without_boalf_negative_bid_gap_mwh",
        "physical_without_boalf_sentinel_gap_mwh",
        "availability_like_dynamic_limit_gap_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_evidence_mwh_lower_bound",
        "physical_dispatch_down_gap_mwh",
        "lost_energy_mwh",
        "most_negative_bid_gbp_per_mwh",
        "support_question_code",
        "support_recommended_action",
        "publication_audit_priority_rank",
        "publication_audit_state",
    ]
    merged, family_gap, scope_key = _prepare_family_physical_forensics_context(
        fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    grouped = merged.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label"],
        as_index=False,
        dropna=False,
    ).agg(
        cluster_key=("cluster_key", _first_mode),
        cluster_label=("cluster_label", _first_mode),
        parent_region=("parent_region", _first_mode),
        mapping_status=("mapping_status", _first_mode),
        distinct_bmu_count=("elexon_bm_unit", "nunique"),
        half_hour_count=("settlement_period", "count"),
        published_boalf_absent_half_hour_count=("published_boalf_absent_flag", lambda values: int(pd.Series(values).sum())),
        physical_without_boalf_half_hour_count=("positive_zero_boalf_gap_flag", lambda values: int(pd.Series(values).sum())),
        physical_without_boalf_negative_bid_half_hour_count=(
            "positive_zero_boalf_negative_bid_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        physical_without_boalf_sentinel_half_hour_count=(
            "positive_zero_boalf_sentinel_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        availability_like_dynamic_limit_half_hour_count=(
            "availability_like_dynamic_limit_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        dispatch_truth_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).sum())),
        lost_energy_estimate_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).sum())),
        physical_without_boalf_gap_mwh=("physical_without_boalf_gap_mwh", "sum"),
        physical_without_boalf_negative_bid_gap_mwh=("physical_without_boalf_negative_bid_gap_mwh", "sum"),
        physical_without_boalf_sentinel_gap_mwh=("physical_without_boalf_sentinel_gap_mwh", "sum"),
        availability_like_dynamic_limit_gap_mwh=("availability_like_dynamic_limit_gap_mwh", "sum"),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
        dispatch_down_evidence_mwh_lower_bound=("dispatch_down_evidence_mwh_lower_bound", "sum"),
        physical_dispatch_down_gap_mwh=("physical_dispatch_down_gap_mwh", "sum"),
        lost_energy_mwh=("lost_energy_mwh", "sum"),
        most_negative_bid_gbp_per_mwh=(
            "most_negative_bid_gbp_per_mwh",
            lambda values: float(pd.Series(values).dropna().min()) if pd.Series(values).dropna().size else np.nan,
        ),
    )
    grouped["forensic_scope_key"] = scope_key
    grouped = grouped.merge(
        family_gap[["settlement_date", "bmu_family_key", "qa_reconciliation_status", "source_gap_next_action"]],
        on=["settlement_date", "bmu_family_key"],
        how="left",
    )
    grouped["publication_audit_state"] = _build_publication_audit_state(
        grouped["physical_without_boalf_negative_bid_half_hour_count"],
        grouped["availability_like_dynamic_limit_half_hour_count"],
        grouped["physical_without_boalf_half_hour_count"],
        grouped["dispatch_truth_half_hour_count"],
    )
    grouped["support_question_code"] = _build_support_question_code(
        grouped["physical_without_boalf_sentinel_half_hour_count"],
        grouped["physical_without_boalf_negative_bid_half_hour_count"],
        grouped["availability_like_dynamic_limit_half_hour_count"],
        grouped["physical_without_boalf_half_hour_count"],
    )
    grouped["support_recommended_action"] = _build_support_recommended_action(grouped["support_question_code"])
    grouped["publication_audit_priority_rank"] = (
        grouped.groupby("settlement_date")["physical_without_boalf_gap_mwh"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    return grouped[columns].sort_values(
        ["settlement_date", "publication_audit_priority_rank", "physical_without_boalf_gap_mwh"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_fact_family_publication_audit_bmu_daily(
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "half_hour_count",
        "published_boalf_absent_half_hour_count",
        "physical_without_boalf_half_hour_count",
        "physical_without_boalf_negative_bid_half_hour_count",
        "physical_without_boalf_sentinel_half_hour_count",
        "availability_like_dynamic_limit_half_hour_count",
        "dispatch_truth_half_hour_count",
        "lost_energy_estimate_half_hour_count",
        "physical_without_boalf_gap_mwh",
        "physical_without_boalf_negative_bid_gap_mwh",
        "physical_without_boalf_sentinel_gap_mwh",
        "availability_like_dynamic_limit_gap_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_evidence_mwh_lower_bound",
        "physical_dispatch_down_gap_mwh",
        "lost_energy_mwh",
        "most_negative_bid_gbp_per_mwh",
        "support_question_code",
        "support_recommended_action",
        "bmu_publication_audit_rank",
        "bmu_publication_audit_state",
    ]
    merged, family_gap, scope_key = _prepare_family_physical_forensics_context(
        fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    grouped = merged.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label", "elexon_bm_unit"],
        as_index=False,
        dropna=False,
    ).agg(
        national_grid_bm_unit=("national_grid_bm_unit", _first_mode),
        cluster_key=("cluster_key", _first_mode),
        cluster_label=("cluster_label", _first_mode),
        parent_region=("parent_region", _first_mode),
        mapping_status=("mapping_status", _first_mode),
        half_hour_count=("settlement_period", "count"),
        published_boalf_absent_half_hour_count=("published_boalf_absent_flag", lambda values: int(pd.Series(values).sum())),
        physical_without_boalf_half_hour_count=("positive_zero_boalf_gap_flag", lambda values: int(pd.Series(values).sum())),
        physical_without_boalf_negative_bid_half_hour_count=(
            "positive_zero_boalf_negative_bid_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        physical_without_boalf_sentinel_half_hour_count=(
            "positive_zero_boalf_sentinel_gap_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        availability_like_dynamic_limit_half_hour_count=(
            "availability_like_dynamic_limit_flag",
            lambda values: int(pd.Series(values).sum()),
        ),
        dispatch_truth_half_hour_count=("dispatch_truth_flag", lambda values: int(pd.Series(values).sum())),
        lost_energy_estimate_half_hour_count=("lost_energy_estimate_flag", lambda values: int(pd.Series(values).sum())),
        physical_without_boalf_gap_mwh=("physical_without_boalf_gap_mwh", "sum"),
        physical_without_boalf_negative_bid_gap_mwh=("physical_without_boalf_negative_bid_gap_mwh", "sum"),
        physical_without_boalf_sentinel_gap_mwh=("physical_without_boalf_sentinel_gap_mwh", "sum"),
        availability_like_dynamic_limit_gap_mwh=("availability_like_dynamic_limit_gap_mwh", "sum"),
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
        dispatch_down_evidence_mwh_lower_bound=("dispatch_down_evidence_mwh_lower_bound", "sum"),
        physical_dispatch_down_gap_mwh=("physical_dispatch_down_gap_mwh", "sum"),
        lost_energy_mwh=("lost_energy_mwh", "sum"),
        most_negative_bid_gbp_per_mwh=(
            "most_negative_bid_gbp_per_mwh",
            lambda values: float(pd.Series(values).dropna().min()) if pd.Series(values).dropna().size else np.nan,
        ),
    )
    grouped["forensic_scope_key"] = scope_key
    grouped = grouped.merge(
        family_gap[["settlement_date", "bmu_family_key", "qa_reconciliation_status", "source_gap_next_action"]],
        on=["settlement_date", "bmu_family_key"],
        how="left",
    )
    grouped["bmu_publication_audit_state"] = _build_publication_audit_state(
        grouped["physical_without_boalf_negative_bid_half_hour_count"],
        grouped["availability_like_dynamic_limit_half_hour_count"],
        grouped["physical_without_boalf_half_hour_count"],
        grouped["dispatch_truth_half_hour_count"],
    )
    grouped["support_question_code"] = _build_support_question_code(
        grouped["physical_without_boalf_sentinel_half_hour_count"],
        grouped["physical_without_boalf_negative_bid_half_hour_count"],
        grouped["availability_like_dynamic_limit_half_hour_count"],
        grouped["physical_without_boalf_half_hour_count"],
    )
    grouped["support_recommended_action"] = _build_support_recommended_action(grouped["support_question_code"])
    grouped["bmu_publication_audit_rank"] = (
        grouped.groupby(["settlement_date", "bmu_family_key"])["physical_without_boalf_gap_mwh"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    return grouped[columns].sort_values(
        ["settlement_date", "bmu_family_key", "bmu_publication_audit_rank", "physical_without_boalf_gap_mwh"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def build_fact_family_support_evidence_half_hourly(
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    family_keys: Sequence[str] | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "forensic_scope_key",
        "support_case_key",
        "support_question_code",
        "support_recommended_action",
        "settlement_date",
        "settlement_period",
        "interval_start_utc",
        "interval_end_utc",
        "bmu_family_key",
        "bmu_family_label",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "qa_reconciliation_status",
        "source_gap_next_action",
        "published_boalf_absent_flag",
        "positive_pn_qpn_gap_flag",
        "negative_bid_available_flag",
        "valid_negative_bid_pair_count",
        "negative_bid_pair_count",
        "sentinel_bid_pair_count",
        "sentinel_offer_pair_count",
        "sentinel_pair_count",
        "sentinel_pair_available_flag",
        "most_negative_bid_gbp_per_mwh",
        "pn_mwh",
        "qpn_mwh",
        "mils_mwh",
        "mels_mwh",
        "generation_mwh",
        "physical_baseline_source_dataset",
        "physical_baseline_mwh",
        "physical_consistency_flag",
        "counterfactual_valid_flag",
        "physical_dispatch_down_gap_mwh",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_evidence_mwh_lower_bound",
        "dispatch_truth_flag",
        "lost_energy_estimate_flag",
        "lost_energy_mwh",
        "lost_energy_block_reason",
        "publication_audit_state",
        "support_priority_rank",
    ]
    merged, family_gap, scope_key = _prepare_family_physical_forensics_context(
        fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    interesting = merged[merged["support_case_ready_flag"]].copy()
    if interesting.empty:
        return pd.DataFrame(columns=columns)

    interesting["forensic_scope_key"] = scope_key
    interesting = interesting.merge(
        family_gap[["settlement_date", "bmu_family_key", "qa_reconciliation_status", "source_gap_next_action"]],
        on=["settlement_date", "bmu_family_key"],
        how="left",
        suffixes=("", "_family_gap"),
    )
    if "qa_reconciliation_status_family_gap" in interesting.columns:
        interesting["qa_reconciliation_status"] = interesting["qa_reconciliation_status_family_gap"].where(
            interesting["qa_reconciliation_status_family_gap"].notna(),
            interesting["qa_reconciliation_status"],
        )
        interesting = interesting.drop(columns=["qa_reconciliation_status_family_gap"])
    interesting["publication_audit_state"] = np.select(
        [
            interesting["positive_zero_boalf_sentinel_gap_flag"],
            interesting["positive_zero_boalf_negative_bid_gap_flag"],
            interesting["availability_like_dynamic_limit_flag"],
            interesting["positive_zero_boalf_gap_flag"],
            interesting["sentinel_pair_available_flag"],
            interesting["dispatch_truth_flag"],
        ],
        [
            "physical_without_boalf_sentinel_bod_present",
            "physical_without_boalf_negative_bid",
            "availability_like_dynamic_limit",
            "physical_without_boalf",
            "sentinel_bod_without_physical_gap",
            "captured_dispatch_present",
        ],
        default="support_context_only",
    )
    interesting["support_question_code"] = _build_support_question_code(
        interesting["positive_zero_boalf_sentinel_gap_flag"].astype(int),
        interesting["positive_zero_boalf_negative_bid_gap_flag"].astype(int),
        interesting["availability_like_dynamic_limit_flag"].astype(int),
        interesting["positive_zero_boalf_gap_flag"].astype(int),
    )
    interesting["support_recommended_action"] = _build_support_recommended_action(interesting["support_question_code"])
    settlement_period_text = pd.to_numeric(interesting["settlement_period"], errors="coerce").astype("Int64").astype(str)
    interesting["support_case_key"] = (
        scope_key
        + ":"
        + interesting["settlement_date"].astype(str)
        + ":"
        + interesting["elexon_bm_unit"].astype(str)
        + ":SP"
        + settlement_period_text
    )
    interesting["support_priority_rank"] = (
        interesting.groupby(["settlement_date", "bmu_family_key"])["physical_dispatch_down_gap_mwh"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    return interesting[columns].sort_values(
        ["settlement_date", "bmu_family_key", "support_priority_rank", "settlement_period", "elexon_bm_unit"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def materialize_truth_store_family_forensics(
    db_path: str | Path,
    family_keys: Sequence[str] | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, pd.DataFrame]:
    target_path = Path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"truth store does not exist: {target_path}")

    materialize_truth_store_source_focus(target_path)
    truth_half_hourly = _load_table(target_path, "fact_bmu_curtailment_truth_half_hourly")
    physical_half_hourly = _load_table(target_path, "fact_bmu_physical_position_half_hourly")
    source_gap_family_daily = _load_table(target_path, "fact_dispatch_source_gap_family_daily")

    fact_family_dispatch_forensic_daily = build_fact_family_dispatch_forensic_daily(
        truth_half_hourly,
        source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_dispatch_forensic_bmu_daily = build_fact_family_dispatch_forensic_bmu_daily(
        truth_half_hourly,
        source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_dispatch_forensic_half_hourly = build_fact_family_dispatch_forensic_half_hourly(
        truth_half_hourly,
        source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_physical_forensic_daily = build_fact_family_physical_forensic_daily(
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_physical_forensic_bmu_daily = build_fact_family_physical_forensic_bmu_daily(
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_physical_forensic_half_hourly = build_fact_family_physical_forensic_half_hourly(
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_publication_audit_daily = build_fact_family_publication_audit_daily(
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_publication_audit_bmu_daily = build_fact_family_publication_audit_bmu_daily(
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )
    fact_family_support_evidence_half_hourly = build_fact_family_support_evidence_half_hourly(
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        family_keys=family_keys,
        start_date=start_date,
        end_date=end_date,
    )

    scope_key = forensic_scope_key_for_family_keys(family_keys)
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_DISPATCH_FORENSIC_DAILY_TABLE,
        frame=fact_family_dispatch_forensic_daily,
        primary_keys=["forensic_scope_key", "settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_DISPATCH_FORENSIC_BMU_TABLE,
        frame=fact_family_dispatch_forensic_bmu_daily,
        primary_keys=["forensic_scope_key", "settlement_date", "bmu_family_key", "elexon_bm_unit"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_DISPATCH_FORENSIC_HALF_HOURLY_TABLE,
        frame=fact_family_dispatch_forensic_half_hourly,
        primary_keys=["forensic_scope_key", "settlement_date", "settlement_period", "elexon_bm_unit"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_PHYSICAL_FORENSIC_DAILY_TABLE,
        frame=fact_family_physical_forensic_daily,
        primary_keys=["forensic_scope_key", "settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_PHYSICAL_FORENSIC_BMU_TABLE,
        frame=fact_family_physical_forensic_bmu_daily,
        primary_keys=["forensic_scope_key", "settlement_date", "bmu_family_key", "elexon_bm_unit"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_PHYSICAL_FORENSIC_HALF_HOURLY_TABLE,
        frame=fact_family_physical_forensic_half_hourly,
        primary_keys=["forensic_scope_key", "settlement_date", "settlement_period", "elexon_bm_unit"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_PUBLICATION_AUDIT_DAILY_TABLE,
        frame=fact_family_publication_audit_daily,
        primary_keys=["forensic_scope_key", "settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_PUBLICATION_AUDIT_BMU_TABLE,
        frame=fact_family_publication_audit_bmu_daily,
        primary_keys=["forensic_scope_key", "settlement_date", "bmu_family_key", "elexon_bm_unit"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=FAMILY_SUPPORT_EVIDENCE_HALF_HOURLY_TABLE,
        frame=fact_family_support_evidence_half_hourly,
        primary_keys=["forensic_scope_key", "settlement_date", "settlement_period", "elexon_bm_unit"],
    )

    return {
        FAMILY_DISPATCH_FORENSIC_DAILY_TABLE: fact_family_dispatch_forensic_daily,
        FAMILY_DISPATCH_FORENSIC_BMU_TABLE: fact_family_dispatch_forensic_bmu_daily,
        FAMILY_DISPATCH_FORENSIC_HALF_HOURLY_TABLE: fact_family_dispatch_forensic_half_hourly,
        FAMILY_PHYSICAL_FORENSIC_DAILY_TABLE: fact_family_physical_forensic_daily,
        FAMILY_PHYSICAL_FORENSIC_BMU_TABLE: fact_family_physical_forensic_bmu_daily,
        FAMILY_PHYSICAL_FORENSIC_HALF_HOURLY_TABLE: fact_family_physical_forensic_half_hourly,
        FAMILY_PUBLICATION_AUDIT_DAILY_TABLE: fact_family_publication_audit_daily,
        FAMILY_PUBLICATION_AUDIT_BMU_TABLE: fact_family_publication_audit_bmu_daily,
        FAMILY_SUPPORT_EVIDENCE_HALF_HOURLY_TABLE: fact_family_support_evidence_half_hourly,
    }


def read_truth_store_family_forensics(
    db_path: str | Path,
    family_keys: Sequence[str] | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, pd.DataFrame]:
    scope_key = forensic_scope_key_for_family_keys(family_keys)
    filters = [f"forensic_scope_key = '{scope_key}'"]
    if start_date:
        filters.append(f"settlement_date >= '{start_date}'")
    if end_date:
        filters.append(f"settlement_date <= '{end_date}'")
    where_clause = " AND ".join(filters)

    def _load_filtered(table_name: str) -> pd.DataFrame:
        frame = _load_table(db_path, table_name)
        if frame.empty or "forensic_scope_key" not in frame.columns:
            return frame
        prepared = frame[frame["forensic_scope_key"].astype(str) == scope_key].copy()
        return _apply_date_filter(prepared, start_date=start_date, end_date=end_date)

    return {
        FAMILY_DISPATCH_FORENSIC_DAILY_TABLE: _load_filtered(FAMILY_DISPATCH_FORENSIC_DAILY_TABLE),
        FAMILY_DISPATCH_FORENSIC_BMU_TABLE: _load_filtered(FAMILY_DISPATCH_FORENSIC_BMU_TABLE),
        FAMILY_DISPATCH_FORENSIC_HALF_HOURLY_TABLE: _load_filtered(FAMILY_DISPATCH_FORENSIC_HALF_HOURLY_TABLE),
        FAMILY_PHYSICAL_FORENSIC_DAILY_TABLE: _load_filtered(FAMILY_PHYSICAL_FORENSIC_DAILY_TABLE),
        FAMILY_PHYSICAL_FORENSIC_BMU_TABLE: _load_filtered(FAMILY_PHYSICAL_FORENSIC_BMU_TABLE),
        FAMILY_PHYSICAL_FORENSIC_HALF_HOURLY_TABLE: _load_filtered(FAMILY_PHYSICAL_FORENSIC_HALF_HOURLY_TABLE),
        FAMILY_PUBLICATION_AUDIT_DAILY_TABLE: _load_filtered(FAMILY_PUBLICATION_AUDIT_DAILY_TABLE),
        FAMILY_PUBLICATION_AUDIT_BMU_TABLE: _load_filtered(FAMILY_PUBLICATION_AUDIT_BMU_TABLE),
        FAMILY_SUPPORT_EVIDENCE_HALF_HOURLY_TABLE: _load_filtered(FAMILY_SUPPORT_EVIDENCE_HALF_HOURLY_TABLE),
    }


def write_family_support_extract_csvs(
    frames: Dict[str, pd.DataFrame],
    output_dir: str | Path,
    family_keys: Sequence[str] | str | None = None,
) -> Dict[str, Path]:
    scope_key = forensic_scope_key_for_family_keys(family_keys)
    target_dir = Path(output_dir) / scope_key
    target_dir.mkdir(parents=True, exist_ok=True)
    export_tables = {
        FAMILY_PUBLICATION_AUDIT_DAILY_TABLE: frames.get(FAMILY_PUBLICATION_AUDIT_DAILY_TABLE, pd.DataFrame()),
        FAMILY_PUBLICATION_AUDIT_BMU_TABLE: frames.get(FAMILY_PUBLICATION_AUDIT_BMU_TABLE, pd.DataFrame()),
        FAMILY_SUPPORT_EVIDENCE_HALF_HOURLY_TABLE: frames.get(FAMILY_SUPPORT_EVIDENCE_HALF_HOURLY_TABLE, pd.DataFrame()),
    }
    written: Dict[str, Path] = {}
    for table_name, frame in export_tables.items():
        path = target_dir / f"{table_name}.csv"
        frame.to_csv(path, index=False)
        written[table_name] = path
    return written
