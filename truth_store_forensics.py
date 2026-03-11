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
    }
    for column, default in optional_truth_defaults.items():
        if column not in truth.columns:
            truth[column] = default

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

    return {
        FAMILY_DISPATCH_FORENSIC_DAILY_TABLE: fact_family_dispatch_forensic_daily,
        FAMILY_DISPATCH_FORENSIC_BMU_TABLE: fact_family_dispatch_forensic_bmu_daily,
        FAMILY_DISPATCH_FORENSIC_HALF_HOURLY_TABLE: fact_family_dispatch_forensic_half_hourly,
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
    }
