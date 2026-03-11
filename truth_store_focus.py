from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from history_store import upsert_frame_to_sqlite


SOURCE_COMPLETENESS_DAILY_TABLE = "fact_source_completeness_focus_daily"
SOURCE_COMPLETENESS_FAMILY_TABLE = "fact_source_completeness_focus_family_daily"
SOURCE_GAP_DAILY_TABLE = "fact_dispatch_source_gap_daily"
SOURCE_GAP_FAMILY_TABLE = "fact_dispatch_source_gap_family_daily"


def _load_table(db_path: str | Path, table_name: str) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as connection:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)


def _status_filter_mask(frame: pd.DataFrame, status_mode: str) -> pd.Series:
    status = frame["qa_reconciliation_status"].fillna("warn").astype(str)
    if status_mode == "all":
        return pd.Series(True, index=frame.index)
    if status_mode == "fail":
        return status.eq("fail")
    if status_mode == "fail_warn":
        return status.isin(["fail", "warn"])
    raise ValueError(f"unsupported status mode '{status_mode}'")


def _coerce_bool_series(values: pd.Series | object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, index=index)
    if series.empty:
        return series.astype(bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).ne(0)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y", "t"])


def _derive_bmu_family_key(values: pd.Series) -> pd.Series:
    normalized = pd.Series(values, copy=False).fillna("").astype(str).str.upper().str.strip()
    normalized = normalized.str.removeprefix("T_").str.removeprefix("E_")
    family = normalized.str.split("-", n=1).str[0].str.strip()
    return family.where(family.ne(""), pd.NA)


def _ensure_family_columns(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "bmu_family_key" not in prepared.columns:
        prepared["bmu_family_key"] = _derive_bmu_family_key(prepared.get("elexon_bm_unit", pd.Series(index=prepared.index)))
    if "bmu_family_label" not in prepared.columns:
        prepared["bmu_family_label"] = prepared["bmu_family_key"]
    return prepared


def _first_mode(values: pd.Series) -> object:
    series = pd.Series(values).dropna()
    if series.empty:
        return pd.NA
    modes = series.mode(dropna=True)
    if modes.empty:
        return series.iloc[0]
    return modes.iloc[0]


def _prepare_dispatch_source_gap_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _ensure_family_columns(frame)

    numeric_defaults = {
        "physical_dispatch_down_gap_mwh": 0.0,
        "accepted_down_delta_mwh_lower_bound": 0.0,
        "dispatch_down_evidence_mwh_lower_bound": 0.0,
        "same_bmu_dispatch_increment_mwh_lower_bound": 0.0,
        "family_day_dispatch_increment_mwh_lower_bound": 0.0,
        "lost_energy_mwh": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column not in prepared.columns:
            prepared[column] = default
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(default)

    bool_defaults = {
        "dispatch_truth_flag": False,
        "negative_bid_available_flag": False,
        "dispatch_acceptance_window_flag": False,
        "family_day_dispatch_window_flag": False,
        "family_day_dispatch_expansion_eligible_flag": False,
    }
    for column, default in bool_defaults.items():
        if column not in prepared.columns:
            prepared[column] = default
        prepared[column] = _coerce_bool_series(prepared[column], index=prepared.index)

    if "qa_reconciliation_status" not in prepared.columns:
        prepared["qa_reconciliation_status"] = "warn"
    if "mapping_status" not in prepared.columns:
        prepared["mapping_status"] = "unmapped"
    if "lost_energy_block_reason" not in prepared.columns:
        prepared["lost_energy_block_reason"] = pd.NA

    no_dispatch_truth_mask = prepared["lost_energy_block_reason"].fillna("no_dispatch_truth").eq("no_dispatch_truth")
    prepared["source_gap_candidate_flag"] = (
        ~prepared["dispatch_truth_flag"]
        & prepared["negative_bid_available_flag"]
        & prepared["physical_dispatch_down_gap_mwh"].gt(0.0)
        & no_dispatch_truth_mask
    )
    prepared["source_gap_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"],
        prepared["physical_dispatch_down_gap_mwh"],
        0.0,
    )
    prepared["acceptance_window_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"] & prepared["dispatch_acceptance_window_flag"],
        prepared["source_gap_candidate_mwh_lower_bound"],
        0.0,
    )
    prepared["family_window_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"] & prepared["family_day_dispatch_window_flag"],
        prepared["source_gap_candidate_mwh_lower_bound"],
        0.0,
    )
    prepared["no_window_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"]
        & ~prepared["dispatch_acceptance_window_flag"]
        & ~prepared["family_day_dispatch_window_flag"],
        prepared["source_gap_candidate_mwh_lower_bound"],
        0.0,
    )
    prepared["family_eligible_no_window_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"]
        & prepared["family_day_dispatch_expansion_eligible_flag"]
        & ~prepared["family_day_dispatch_window_flag"],
        prepared["source_gap_candidate_mwh_lower_bound"],
        0.0,
    )
    prepared["mapped_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"] & prepared["mapping_status"].eq("mapped"),
        prepared["source_gap_candidate_mwh_lower_bound"],
        0.0,
    )
    prepared["region_only_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"] & prepared["mapping_status"].eq("region_only"),
        prepared["source_gap_candidate_mwh_lower_bound"],
        0.0,
    )
    prepared["unmapped_candidate_mwh_lower_bound"] = np.where(
        prepared["source_gap_candidate_flag"] & ~prepared["mapping_status"].isin(["mapped", "region_only"]),
        prepared["source_gap_candidate_mwh_lower_bound"],
        0.0,
    )
    return prepared


def _dominant_scope_column(frame: pd.DataFrame, scope_columns: list[str]) -> pd.Series:
    numeric = frame[scope_columns].fillna(0.0)
    winner = numeric.idxmax(axis=1)
    nonzero = numeric.max(axis=1).gt(0.0)
    labels = {
        "acceptance_window_candidate_mwh_lower_bound": "same_bmu_window",
        "family_window_candidate_mwh_lower_bound": "family_window",
        "no_window_candidate_mwh_lower_bound": "no_window",
    }
    return winner.map(labels).where(nonzero, "none")


def build_fact_source_completeness_focus_daily(
    fact_curtailment_reconciliation_daily: pd.DataFrame,
    fact_constraint_target_audit_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "qa_target_definition",
        "gb_daily_qa_target_mwh",
        "gb_daily_estimated_lost_energy_mwh",
        "remaining_qa_shortfall_mwh",
        "lost_energy_capture_ratio_vs_qa_target",
        "dispatch_half_hour_count",
        "dispatch_family_day_inference_row_count",
        "family_day_dispatch_increment_mwh_lower_bound",
        "source_completeness_focus_flag",
        "focus_priority_rank",
        "next_action",
    ]
    if fact_curtailment_reconciliation_daily.empty:
        return pd.DataFrame(columns=columns)

    daily = fact_curtailment_reconciliation_daily.copy()
    audit = fact_constraint_target_audit_daily.copy()
    if not audit.empty:
        keep = [
            "settlement_date",
            "qa_target_definition",
            "recoverability_audit_state",
        ]
        daily = daily.merge(audit[keep], on="settlement_date", how="left", suffixes=("", "_audit"))
        if "qa_target_definition_audit" in daily.columns:
            daily["qa_target_definition"] = daily["qa_target_definition"].where(
                daily["qa_target_definition"].notna(),
                daily["qa_target_definition_audit"],
            )
            daily = daily.drop(columns=["qa_target_definition_audit"])
    else:
        daily["recoverability_audit_state"] = pd.NA

    for column in [
        "gb_daily_qa_target_mwh",
        "gb_daily_estimated_lost_energy_mwh",
        "lost_energy_capture_ratio_vs_qa_target",
        "family_day_dispatch_increment_mwh_lower_bound",
    ]:
        if column not in daily.columns:
            daily[column] = np.nan
    if "dispatch_family_day_inference_row_count" not in daily.columns:
        daily["dispatch_family_day_inference_row_count"] = 0

    daily["remaining_qa_shortfall_mwh"] = (
        pd.to_numeric(daily["gb_daily_qa_target_mwh"], errors="coerce")
        - pd.to_numeric(daily["gb_daily_estimated_lost_energy_mwh"], errors="coerce")
    ).clip(lower=0.0)
    daily["source_completeness_focus_flag"] = ~daily["qa_reconciliation_status"].fillna("warn").eq("pass")

    focus_scores = daily["remaining_qa_shortfall_mwh"].fillna(0.0)
    focus_mask = daily["source_completeness_focus_flag"]
    daily["focus_priority_rank"] = pd.NA
    if bool(focus_mask.any()):
        daily.loc[focus_mask, "focus_priority_rank"] = (
            focus_scores[focus_mask].rank(method="dense", ascending=False).astype("Int64")
        )

    daily["next_action"] = np.select(
        [
            ~daily["source_completeness_focus_flag"],
            daily["recoverability_audit_state"].eq("counterfactual_or_definition_limited"),
            daily["recoverability_audit_state"].eq("partially_recovered"),
            daily["family_day_dispatch_increment_mwh_lower_bound"].fillna(0.0).gt(0.0),
            daily["recoverability_audit_state"].eq("source_limited"),
        ],
        [
            "no_action",
            "counterfactual_or_target_audit",
            "expand_source_after_family_window",
            "expand_source_after_family_window",
            "add_dispatch_source",
        ],
        default="inspect",
    )
    return daily[columns].sort_values(["source_completeness_focus_flag", "focus_priority_rank", "settlement_date"], ascending=[False, True, True]).reset_index(drop=True)


def build_fact_source_completeness_focus_family_daily(
    fact_bmu_family_shortfall_daily: pd.DataFrame,
    fact_source_completeness_focus_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "next_action",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "dispatch_half_hour_count",
        "lost_energy_estimate_half_hour_count",
        "dispatch_minus_lost_energy_gap_mwh",
        "share_of_day_remaining_qa_shortfall",
        "family_day_dispatch_increment_mwh_lower_bound",
        "day_family_rank_by_gap",
        "family_next_action",
    ]
    if fact_bmu_family_shortfall_daily.empty or fact_source_completeness_focus_daily.empty:
        return pd.DataFrame(columns=columns)

    daily = fact_source_completeness_focus_daily.copy()
    focus_days = daily[daily["source_completeness_focus_flag"]].copy()
    if focus_days.empty:
        return pd.DataFrame(columns=columns)

    family = fact_bmu_family_shortfall_daily.copy()
    family = family.merge(
        focus_days[
            [
                "settlement_date",
                "qa_reconciliation_status",
                "recoverability_audit_state",
                "next_action",
            ]
        ],
        on="settlement_date",
        how="inner",
    )
    family["dispatch_minus_lost_energy_gap_mwh"] = pd.to_numeric(
        family["dispatch_minus_lost_energy_gap_mwh"], errors="coerce"
    ).fillna(0.0)
    family["family_day_dispatch_increment_mwh_lower_bound"] = pd.to_numeric(
        family.get("family_day_dispatch_increment_mwh_lower_bound", 0.0),
        errors="coerce",
    ).fillna(0.0)
    family["share_of_day_remaining_qa_shortfall"] = pd.to_numeric(
        family["share_of_day_remaining_qa_shortfall"], errors="coerce"
    )
    family = family[family["dispatch_minus_lost_energy_gap_mwh"] > 0].copy()
    if family.empty:
        return pd.DataFrame(columns=columns)

    family["day_family_rank_by_gap"] = (
        family.groupby("settlement_date")["dispatch_minus_lost_energy_gap_mwh"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    family["family_next_action"] = np.select(
        [
            family["recoverability_audit_state"].eq("counterfactual_or_definition_limited"),
            family["family_day_dispatch_increment_mwh_lower_bound"].gt(0.0),
            family["recoverability_audit_state"].eq("source_limited"),
        ],
        [
            "counterfactual_or_target_audit",
            "expand_dispatch_source_beyond_family_window",
            "missing_dispatch_evidence",
        ],
        default="inspect",
    )
    return family[columns].sort_values(
        ["settlement_date", "day_family_rank_by_gap", "dispatch_minus_lost_energy_gap_mwh"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_fact_dispatch_source_gap_daily(
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_source_completeness_focus_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "next_action",
        "remaining_qa_shortfall_mwh",
        "source_gap_candidate_row_count",
        "source_gap_candidate_distinct_bmu_count",
        "source_gap_candidate_mwh_lower_bound",
        "acceptance_window_candidate_mwh_lower_bound",
        "family_window_candidate_mwh_lower_bound",
        "no_window_candidate_mwh_lower_bound",
        "family_eligible_no_window_candidate_mwh_lower_bound",
        "mapped_candidate_mwh_lower_bound",
        "region_only_candidate_mwh_lower_bound",
        "unmapped_candidate_mwh_lower_bound",
        "source_gap_share_of_remaining_qa_shortfall",
        "source_gap_dominant_scope",
        "source_gap_next_action",
    ]
    if fact_source_completeness_focus_daily.empty:
        return pd.DataFrame(columns=columns)

    focus_daily = fact_source_completeness_focus_daily.copy()
    prepared = _prepare_dispatch_source_gap_candidates(fact_bmu_curtailment_truth_half_hourly)
    candidates = prepared[prepared["source_gap_candidate_flag"]].copy()

    if candidates.empty:
        result = focus_daily[
            [
                "settlement_date",
                "qa_reconciliation_status",
                "recoverability_audit_state",
                "next_action",
                "remaining_qa_shortfall_mwh",
            ]
        ].copy()
        numeric_columns = {
            "source_gap_candidate_row_count",
            "source_gap_candidate_distinct_bmu_count",
            "source_gap_candidate_mwh_lower_bound",
            "acceptance_window_candidate_mwh_lower_bound",
            "family_window_candidate_mwh_lower_bound",
            "no_window_candidate_mwh_lower_bound",
            "family_eligible_no_window_candidate_mwh_lower_bound",
            "mapped_candidate_mwh_lower_bound",
            "region_only_candidate_mwh_lower_bound",
            "unmapped_candidate_mwh_lower_bound",
            "source_gap_share_of_remaining_qa_shortfall",
        }
        for column in columns:
            if column in result.columns:
                continue
            result[column] = 0.0 if column in numeric_columns else "none"
        result["source_gap_candidate_row_count"] = 0
        result["source_gap_candidate_distinct_bmu_count"] = 0
        result["source_gap_dominant_scope"] = "none"
        result["source_gap_next_action"] = "no_gap_candidates"
        return result[columns]

    aggregated = candidates.groupby("settlement_date", as_index=False).agg(
        source_gap_candidate_row_count=("source_gap_candidate_flag", lambda values: int(pd.Series(values).sum())),
        source_gap_candidate_distinct_bmu_count=("elexon_bm_unit", "nunique"),
        source_gap_candidate_mwh_lower_bound=("source_gap_candidate_mwh_lower_bound", "sum"),
        acceptance_window_candidate_mwh_lower_bound=("acceptance_window_candidate_mwh_lower_bound", "sum"),
        family_window_candidate_mwh_lower_bound=("family_window_candidate_mwh_lower_bound", "sum"),
        no_window_candidate_mwh_lower_bound=("no_window_candidate_mwh_lower_bound", "sum"),
        family_eligible_no_window_candidate_mwh_lower_bound=(
            "family_eligible_no_window_candidate_mwh_lower_bound",
            "sum",
        ),
        mapped_candidate_mwh_lower_bound=("mapped_candidate_mwh_lower_bound", "sum"),
        region_only_candidate_mwh_lower_bound=("region_only_candidate_mwh_lower_bound", "sum"),
        unmapped_candidate_mwh_lower_bound=("unmapped_candidate_mwh_lower_bound", "sum"),
    )

    result = focus_daily[
        [
            "settlement_date",
            "qa_reconciliation_status",
            "recoverability_audit_state",
            "next_action",
            "remaining_qa_shortfall_mwh",
        ]
    ].merge(aggregated, on="settlement_date", how="left")
    for column in [
        "source_gap_candidate_row_count",
        "source_gap_candidate_distinct_bmu_count",
        "source_gap_candidate_mwh_lower_bound",
        "acceptance_window_candidate_mwh_lower_bound",
        "family_window_candidate_mwh_lower_bound",
        "no_window_candidate_mwh_lower_bound",
        "family_eligible_no_window_candidate_mwh_lower_bound",
        "mapped_candidate_mwh_lower_bound",
        "region_only_candidate_mwh_lower_bound",
        "unmapped_candidate_mwh_lower_bound",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result["source_gap_candidate_row_count"] = result["source_gap_candidate_row_count"].astype("Int64")
    result["source_gap_candidate_distinct_bmu_count"] = result["source_gap_candidate_distinct_bmu_count"].astype("Int64")
    denominator = pd.to_numeric(result["remaining_qa_shortfall_mwh"], errors="coerce").replace(0.0, np.nan)
    result["source_gap_share_of_remaining_qa_shortfall"] = (
        result["source_gap_candidate_mwh_lower_bound"] / denominator
    ).fillna(0.0)
    result["source_gap_dominant_scope"] = _dominant_scope_column(
        result,
        [
            "acceptance_window_candidate_mwh_lower_bound",
            "family_window_candidate_mwh_lower_bound",
            "no_window_candidate_mwh_lower_bound",
        ],
    )
    result["source_gap_next_action"] = np.select(
        [
            result["source_gap_candidate_mwh_lower_bound"].le(0.0),
            result["unmapped_candidate_mwh_lower_bound"].ge(result["source_gap_candidate_mwh_lower_bound"] * 0.5)
            & result["source_gap_candidate_mwh_lower_bound"].gt(0.0),
            result["family_eligible_no_window_candidate_mwh_lower_bound"].gt(0.0),
            result["no_window_candidate_mwh_lower_bound"].gt(0.0),
            result["acceptance_window_candidate_mwh_lower_bound"].gt(0.0)
            | result["family_window_candidate_mwh_lower_bound"].gt(0.0),
        ],
        [
            "no_gap_candidates",
            "mapping_and_source_audit",
            "expand_dispatch_source_beyond_family_window",
            "add_dispatch_source",
            "inspect_window_rule_thresholds",
        ],
        default="inspect",
    )
    return result[columns].sort_values(
        ["source_gap_candidate_mwh_lower_bound", "settlement_date"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_fact_dispatch_source_gap_family_daily(
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "next_action",
        "source_gap_next_action",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "source_gap_candidate_row_count",
        "source_gap_candidate_distinct_bmu_count",
        "source_gap_candidate_mwh_lower_bound",
        "acceptance_window_candidate_mwh_lower_bound",
        "family_window_candidate_mwh_lower_bound",
        "no_window_candidate_mwh_lower_bound",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_mwh_lower_bound",
        "lost_energy_mwh",
        "family_day_dispatch_increment_mwh_lower_bound",
        "source_gap_share_of_day_total",
        "source_gap_share_of_remaining_qa_shortfall",
        "source_gap_dominant_scope",
        "day_family_rank_by_source_gap",
        "family_source_gap_next_action",
    ]
    if fact_dispatch_source_gap_daily.empty:
        return pd.DataFrame(columns=columns)

    prepared = _prepare_dispatch_source_gap_candidates(fact_bmu_curtailment_truth_half_hourly)
    candidates = prepared[prepared["source_gap_candidate_flag"]].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    focus_days = fact_dispatch_source_gap_daily[
        [
            "settlement_date",
            "qa_reconciliation_status",
            "recoverability_audit_state",
            "next_action",
            "source_gap_next_action",
            "remaining_qa_shortfall_mwh",
        ]
    ].copy()
    family_candidates = candidates.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label"],
        as_index=False,
        dropna=False,
    ).agg(
        cluster_key=("cluster_key", _first_mode),
        cluster_label=("cluster_label", _first_mode),
        parent_region=("parent_region", _first_mode),
        mapping_status=("mapping_status", _first_mode),
        source_gap_candidate_row_count=("source_gap_candidate_flag", lambda values: int(pd.Series(values).sum())),
        source_gap_candidate_distinct_bmu_count=("elexon_bm_unit", "nunique"),
        source_gap_candidate_mwh_lower_bound=("source_gap_candidate_mwh_lower_bound", "sum"),
        acceptance_window_candidate_mwh_lower_bound=("acceptance_window_candidate_mwh_lower_bound", "sum"),
        family_window_candidate_mwh_lower_bound=("family_window_candidate_mwh_lower_bound", "sum"),
        no_window_candidate_mwh_lower_bound=("no_window_candidate_mwh_lower_bound", "sum"),
    )
    family_context = prepared.groupby(
        ["settlement_date", "bmu_family_key", "bmu_family_label"],
        as_index=False,
        dropna=False,
    ).agg(
        accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
        dispatch_down_mwh_lower_bound=("dispatch_down_evidence_mwh_lower_bound", "sum"),
        lost_energy_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        family_day_dispatch_increment_mwh_lower_bound=(
            "family_day_dispatch_increment_mwh_lower_bound",
            "sum",
        ),
    )

    family = family_candidates.merge(
        focus_days,
        on="settlement_date",
        how="inner",
    ).merge(
        family_context,
        on=["settlement_date", "bmu_family_key", "bmu_family_label"],
        how="left",
    )
    day_totals = family.groupby("settlement_date", as_index=False).agg(
        day_source_gap_candidate_mwh_lower_bound=("source_gap_candidate_mwh_lower_bound", "sum")
    )
    family = family.merge(day_totals, on="settlement_date", how="left")
    family["source_gap_share_of_day_total"] = (
        family["source_gap_candidate_mwh_lower_bound"]
        / family["day_source_gap_candidate_mwh_lower_bound"].replace(0.0, np.nan)
    ).fillna(0.0)
    family["source_gap_share_of_remaining_qa_shortfall"] = (
        family["source_gap_candidate_mwh_lower_bound"]
        / pd.to_numeric(family["remaining_qa_shortfall_mwh"], errors="coerce").replace(0.0, np.nan)
    ).fillna(0.0)
    family["source_gap_dominant_scope"] = _dominant_scope_column(
        family,
        [
            "acceptance_window_candidate_mwh_lower_bound",
            "family_window_candidate_mwh_lower_bound",
            "no_window_candidate_mwh_lower_bound",
        ],
    )
    family["day_family_rank_by_source_gap"] = (
        family.groupby("settlement_date")["source_gap_candidate_mwh_lower_bound"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    family["family_source_gap_next_action"] = np.select(
        [
            family["mapping_status"].fillna("unmapped").eq("unmapped"),
            family["no_window_candidate_mwh_lower_bound"].gt(0.0)
            & family["family_day_dispatch_increment_mwh_lower_bound"].gt(0.0),
            family["no_window_candidate_mwh_lower_bound"].gt(0.0),
            family["acceptance_window_candidate_mwh_lower_bound"].gt(0.0)
            | family["family_window_candidate_mwh_lower_bound"].gt(0.0),
        ],
        [
            "mapping_and_source_audit",
            "expand_dispatch_source_beyond_family_window",
            "add_dispatch_source",
            "inspect_window_rule_thresholds",
        ],
        default="inspect",
    )
    return family[columns].sort_values(
        ["settlement_date", "day_family_rank_by_source_gap", "source_gap_candidate_mwh_lower_bound"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def materialize_truth_store_source_focus(
    db_path: str | Path,
) -> Dict[str, pd.DataFrame]:
    target_path = Path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"truth store does not exist: {target_path}")

    reconciliation_daily = _load_table(target_path, "fact_curtailment_reconciliation_daily")
    constraint_target_audit_daily = _load_table(target_path, "fact_constraint_target_audit_daily")
    family_shortfall_daily = _load_table(target_path, "fact_bmu_family_shortfall_daily")
    truth_half_hourly = _load_table(target_path, "fact_bmu_curtailment_truth_half_hourly")

    fact_source_completeness_focus_daily = build_fact_source_completeness_focus_daily(
        reconciliation_daily,
        constraint_target_audit_daily,
    )
    fact_source_completeness_focus_family_daily = build_fact_source_completeness_focus_family_daily(
        family_shortfall_daily,
        fact_source_completeness_focus_daily,
    )
    fact_dispatch_source_gap_daily = build_fact_dispatch_source_gap_daily(
        truth_half_hourly,
        fact_source_completeness_focus_daily,
    )
    fact_dispatch_source_gap_family_daily = build_fact_dispatch_source_gap_family_daily(
        truth_half_hourly,
        fact_dispatch_source_gap_daily,
    )

    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SOURCE_COMPLETENESS_DAILY_TABLE,
        frame=fact_source_completeness_focus_daily,
        primary_keys=["settlement_date"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SOURCE_COMPLETENESS_FAMILY_TABLE,
        frame=fact_source_completeness_focus_family_daily,
        primary_keys=["settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SOURCE_GAP_DAILY_TABLE,
        frame=fact_dispatch_source_gap_daily,
        primary_keys=["settlement_date"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SOURCE_GAP_FAMILY_TABLE,
        frame=fact_dispatch_source_gap_family_daily,
        primary_keys=["settlement_date", "bmu_family_key"],
    )

    return {
        SOURCE_COMPLETENESS_DAILY_TABLE: fact_source_completeness_focus_daily,
        SOURCE_COMPLETENESS_FAMILY_TABLE: fact_source_completeness_focus_family_daily,
        SOURCE_GAP_DAILY_TABLE: fact_dispatch_source_gap_daily,
        SOURCE_GAP_FAMILY_TABLE: fact_dispatch_source_gap_family_daily,
    }


def read_truth_store_source_focus(
    db_path: str | Path,
    status_mode: str = "fail_warn",
) -> Dict[str, pd.DataFrame]:
    daily = _load_table(db_path, SOURCE_COMPLETENESS_DAILY_TABLE)
    family = _load_table(db_path, SOURCE_COMPLETENESS_FAMILY_TABLE)
    source_gap_daily = _load_table(db_path, SOURCE_GAP_DAILY_TABLE)
    source_gap_family = _load_table(db_path, SOURCE_GAP_FAMILY_TABLE)
    mask = _status_filter_mask(daily, status_mode)
    filtered_daily = daily[mask].reset_index(drop=True)
    filtered_family = family[family["settlement_date"].isin(filtered_daily["settlement_date"])].reset_index(drop=True)
    filtered_source_gap_daily = source_gap_daily[
        source_gap_daily["settlement_date"].isin(filtered_daily["settlement_date"])
    ].reset_index(drop=True)
    filtered_source_gap_family = source_gap_family[
        source_gap_family["settlement_date"].isin(filtered_daily["settlement_date"])
    ].reset_index(drop=True)
    return {
        SOURCE_COMPLETENESS_DAILY_TABLE: filtered_daily,
        SOURCE_COMPLETENESS_FAMILY_TABLE: filtered_family,
        SOURCE_GAP_DAILY_TABLE: filtered_source_gap_daily,
        SOURCE_GAP_FAMILY_TABLE: filtered_source_gap_family,
    }
