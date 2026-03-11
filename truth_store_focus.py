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


def materialize_truth_store_source_focus(
    db_path: str | Path,
) -> Dict[str, pd.DataFrame]:
    target_path = Path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"truth store does not exist: {target_path}")

    reconciliation_daily = _load_table(target_path, "fact_curtailment_reconciliation_daily")
    constraint_target_audit_daily = _load_table(target_path, "fact_constraint_target_audit_daily")
    family_shortfall_daily = _load_table(target_path, "fact_bmu_family_shortfall_daily")

    fact_source_completeness_focus_daily = build_fact_source_completeness_focus_daily(
        reconciliation_daily,
        constraint_target_audit_daily,
    )
    fact_source_completeness_focus_family_daily = build_fact_source_completeness_focus_family_daily(
        family_shortfall_daily,
        fact_source_completeness_focus_daily,
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

    return {
        SOURCE_COMPLETENESS_DAILY_TABLE: fact_source_completeness_focus_daily,
        SOURCE_COMPLETENESS_FAMILY_TABLE: fact_source_completeness_focus_family_daily,
    }


def read_truth_store_source_focus(
    db_path: str | Path,
    status_mode: str = "fail_warn",
) -> Dict[str, pd.DataFrame]:
    daily = _load_table(db_path, SOURCE_COMPLETENESS_DAILY_TABLE)
    family = _load_table(db_path, SOURCE_COMPLETENESS_FAMILY_TABLE)
    mask = _status_filter_mask(daily, status_mode)
    filtered_daily = daily[mask].reset_index(drop=True)
    filtered_family = family[family["settlement_date"].isin(filtered_daily["settlement_date"])].reset_index(drop=True)
    return {
        SOURCE_COMPLETENESS_DAILY_TABLE: filtered_daily,
        SOURCE_COMPLETENESS_FAMILY_TABLE: filtered_family,
    }
