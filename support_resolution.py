from __future__ import annotations

import datetime as dt
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict

import pandas as pd

from history_store import upsert_frame_to_sqlite
from truth_store_focus import _load_table


SUPPORT_CASE_FAMILY_TABLE = "fact_support_case_family_daily"
SUPPORT_CASE_RESOLUTION_TABLE = "fact_support_case_resolution"
SUPPORT_RESOLUTION_DAILY_TABLE = "fact_support_resolution_daily"
SUPPORT_RESOLUTION_BATCH_TABLE = "fact_support_resolution_batch"
SUPPORT_RERUN_GATE_DAILY_TABLE = "fact_support_rerun_gate_daily"
SUPPORT_RERUN_GATE_BATCH_TABLE = "fact_support_rerun_gate_batch"
SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE = "fact_support_open_case_priority_family_daily"
SUPPORT_RERUN_CANDIDATE_DAILY_TABLE = "fact_support_rerun_candidate_daily"
SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE = "fact_support_rerun_candidate_family_daily"
SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE = "fact_support_resolution_pattern_summary"
SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE = "fact_support_resolution_pattern_member_family_daily"
VALID_RESOLUTION_STATES = (
    "open",
    "confirmed_publication_gap",
    "confirmed_non_boalf_pattern",
    "confirmed_source_artifact",
    "not_reproducible",
)
VALID_TRUTH_POLICY_ACTIONS = (
    "keep_out_of_precision",
    "eligible_for_new_evidence_tier",
    "fix_source_and_rerun",
    "close_no_change",
)
VALID_RESOLUTION_FILTERS = tuple(dict.fromkeys(("all", "open", "resolved", *VALID_RESOLUTION_STATES)))
VALID_SUPPORT_GATE_FILTERS = ("all", "blocked", "ready_for_rerun", "no_rerun_required")
VALID_SUPPORT_RERUN_CANDIDATE_FILTERS = ("all", "fix_source_and_rerun", "eligible_for_new_evidence_tier")
VALID_SUPPORT_PATTERN_FILTERS = ("all", "open", "single_day", "multi_day")


def _resolve_generated_at_utc(value: str | None = None) -> str:
    if value:
        return value
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _series_or_default(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def _support_resolution_pattern_key(frame: pd.DataFrame) -> pd.Series:
    family_key = _series_or_default(frame, "bmu_family_key", "unknown_family")
    anomaly_state = _series_or_default(frame, "publication_anomaly_dominant_state", "unknown_state")
    question_code = _series_or_default(frame, "support_question_code", "unknown_question")
    mapping_status = _series_or_default(frame, "mapping_status", "unknown_mapping")
    return family_key + "::" + anomaly_state + "::" + question_code + "::" + mapping_status


def _resolution_columns() -> list[str]:
    return [
        "support_batch_id",
        "support_case_family_key",
        "support_generated_at_utc",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "support_case_day_rank",
        "support_case_family_rank",
        "support_question_code",
        "support_recommended_action",
        "resolution_state",
        "resolution_note",
        "source_reference",
        "truth_policy_action",
        "resolution_updated_at_utc",
    ]


def _table_exists(db_path: str | Path, table_name: str) -> bool:
    with closing(sqlite3.connect(db_path)) as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


def _delete_rows_for_support_batches(
    db_path: str | Path,
    table_name: str,
    support_batch_ids: list[str],
) -> None:
    batch_ids = [str(value) for value in support_batch_ids if str(value)]
    if not batch_ids or not _table_exists(db_path, table_name):
        return
    placeholders = ", ".join("?" for _ in batch_ids)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            f'DELETE FROM "{table_name}" WHERE "support_batch_id" IN ({placeholders})',
            batch_ids,
        )
        connection.commit()


def _empty_resolution_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_resolution_columns())


def _coerce_int64_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype("Int64")
    return result


def _resolution_review_state_and_action(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    open_count = pd.to_numeric(frame["open_family_count"], errors="coerce").fillna(0)
    fix_count = pd.to_numeric(frame["fix_source_and_rerun_count"], errors="coerce").fillna(0)
    eligible_count = pd.to_numeric(frame["eligible_for_new_evidence_tier_count"], errors="coerce").fillna(0)
    keep_out_count = pd.to_numeric(frame["keep_out_of_precision_count"], errors="coerce").fillna(0)
    close_count = pd.to_numeric(frame["close_no_change_count"], errors="coerce").fillna(0)
    selected_count = pd.to_numeric(frame["selected_family_count"], errors="coerce").fillna(0)

    state = pd.Series("resolved_mixed", index=frame.index, dtype="object")
    action = pd.Series("review_resolution_mix", index=frame.index, dtype="object")

    open_mask = open_count.gt(0)
    fix_mask = ~open_mask & fix_count.gt(0)
    eligible_mask = ~open_mask & ~fix_mask & eligible_count.gt(0)
    keep_mask = ~open_mask & ~fix_mask & ~eligible_mask & keep_out_count.ge(selected_count) & selected_count.gt(0)
    close_mask = ~open_mask & ~fix_mask & ~eligible_mask & ~keep_mask & close_count.ge(selected_count) & selected_count.gt(0)

    state.loc[open_mask] = "blocked_by_open_cases"
    action.loc[open_mask] = "await_support_resolution"

    state.loc[fix_mask] = "resolved_fix_and_rerun"
    action.loc[fix_mask] = "fix_source_and_rerun"

    state.loc[eligible_mask] = "resolved_review_new_evidence_tier"
    action.loc[eligible_mask] = "review_new_evidence_tier"

    state.loc[keep_mask] = "resolved_keep_out_of_precision"
    action.loc[keep_mask] = "keep_out_of_precision"

    state.loc[close_mask] = "resolved_close_no_change"
    action.loc[close_mask] = "close_no_change"
    return state, action


def _support_rerun_gate_state_and_action(
    frame: pd.DataFrame,
    ready_state: str,
    no_rerun_state: str,
) -> tuple[pd.Series, pd.Series]:
    open_count = pd.to_numeric(frame["open_family_count"], errors="coerce").fillna(0)
    fix_count = pd.to_numeric(frame["fix_source_and_rerun_count"], errors="coerce").fillna(0)
    eligible_count = pd.to_numeric(frame["eligible_for_new_evidence_tier_count"], errors="coerce").fillna(0)

    state = pd.Series(no_rerun_state, index=frame.index, dtype="object")
    action = pd.Series("lock_truth_policy_no_rerun", index=frame.index, dtype="object")

    open_mask = open_count.gt(0)
    rerun_mask = ~open_mask & (fix_count + eligible_count).gt(0)

    state.loc[open_mask] = "blocked_by_open_cases"
    action.loc[open_mask] = "await_support_resolution"

    state.loc[rerun_mask] = ready_state
    action.loc[rerun_mask] = "prepare_targeted_truth_rerun"
    return state, action


def _derive_support_case_daily_from_family(fact_support_case_family_daily: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "support_case_day_rank",
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "publication_anomaly_dominant_state",
        "selected_family_count",
    ]
    if fact_support_case_family_daily.empty:
        return pd.DataFrame(columns=columns)

    daily = fact_support_case_family_daily.groupby(
        ["support_batch_id", "settlement_date"],
        as_index=False,
        dropna=False,
    ).agg(
        support_generated_at_utc=("support_generated_at_utc", "first"),
        support_status_mode=("support_status_mode", "first"),
        support_top_days=("support_top_days", "first"),
        support_top_families_per_day=("support_top_families_per_day", "first"),
        support_case_day_rank=("support_case_day_rank", "first"),
        qa_reconciliation_status=("qa_reconciliation_status", "first"),
        recoverability_audit_state=("recoverability_audit_state", "first"),
        publication_anomaly_dominant_state=("publication_anomaly_dominant_state", "first"),
        selected_family_count=("bmu_family_key", "nunique"),
    )
    return daily[columns].sort_values(
        ["support_batch_id", "support_case_day_rank", "settlement_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_fact_support_case_resolution(
    fact_support_case_family_daily: pd.DataFrame,
    existing_resolution: pd.DataFrame | None = None,
    generated_at_utc: str | None = None,
) -> pd.DataFrame:
    columns = _resolution_columns()
    if fact_support_case_family_daily.empty:
        return pd.DataFrame(columns=columns)

    generated_value = _resolve_generated_at_utc(generated_at_utc)
    base = fact_support_case_family_daily[
        [
            "support_batch_id",
            "support_case_family_key",
            "support_generated_at_utc",
            "settlement_date",
            "bmu_family_key",
            "bmu_family_label",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "mapping_status",
            "support_case_day_rank",
            "support_case_family_rank",
            "support_question_code",
            "support_recommended_action",
        ]
    ].copy()
    base["resolution_state"] = "open"
    base["resolution_note"] = pd.NA
    base["source_reference"] = pd.NA
    base["truth_policy_action"] = "keep_out_of_precision"
    base["resolution_updated_at_utc"] = generated_value

    if existing_resolution is None or existing_resolution.empty:
        return base[columns].sort_values(
            ["support_batch_id", "settlement_date", "support_case_family_rank", "bmu_family_key"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    existing = existing_resolution.copy()
    keep_columns = [
        "support_batch_id",
        "settlement_date",
        "bmu_family_key",
        "resolution_state",
        "resolution_note",
        "source_reference",
        "truth_policy_action",
        "resolution_updated_at_utc",
    ]
    existing = existing[[column for column in keep_columns if column in existing.columns]].copy()
    merged = base.merge(
        existing,
        on=["support_batch_id", "settlement_date", "bmu_family_key"],
        how="left",
        suffixes=("", "_existing"),
    )
    for column in [
        "resolution_state",
        "resolution_note",
        "source_reference",
        "truth_policy_action",
        "resolution_updated_at_utc",
    ]:
        existing_column = f"{column}_existing"
        if existing_column not in merged.columns:
            continue
        merged[column] = merged[existing_column].where(merged[existing_column].notna(), merged[column])
        merged = merged.drop(columns=[existing_column])

    merged["resolution_state"] = merged["resolution_state"].fillna("open")
    merged["truth_policy_action"] = merged["truth_policy_action"].fillna("keep_out_of_precision")
    merged["resolution_updated_at_utc"] = merged["resolution_updated_at_utc"].fillna(generated_value)
    return merged[columns].sort_values(
        ["support_batch_id", "settlement_date", "support_case_family_rank", "bmu_family_key"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_fact_support_resolution_daily(
    fact_support_case_daily: pd.DataFrame,
    fact_support_case_family_daily: pd.DataFrame,
    fact_support_case_resolution: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "support_case_day_rank",
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "publication_anomaly_dominant_state",
        "selected_family_count",
        "open_family_count",
        "resolved_family_count",
        "confirmed_publication_gap_count",
        "confirmed_non_boalf_pattern_count",
        "confirmed_source_artifact_count",
        "not_reproducible_count",
        "keep_out_of_precision_count",
        "eligible_for_new_evidence_tier_count",
        "fix_source_and_rerun_count",
        "close_no_change_count",
        "support_resolution_state",
        "support_resolution_next_action",
    ]
    if fact_support_case_daily.empty:
        return pd.DataFrame(columns=columns)

    daily = fact_support_case_daily.copy()
    family = fact_support_case_family_daily.copy()
    resolution = fact_support_case_resolution.copy()
    merged = family.merge(
        resolution[
            [
                "support_batch_id",
                "settlement_date",
                "bmu_family_key",
                "resolution_state",
                "truth_policy_action",
            ]
        ],
        on=["support_batch_id", "settlement_date", "bmu_family_key"],
        how="left",
    )
    merged["resolution_state"] = merged["resolution_state"].fillna("open")
    merged["truth_policy_action"] = merged["truth_policy_action"].fillna("keep_out_of_precision")

    summary = merged.groupby(["support_batch_id", "settlement_date"], as_index=False).agg(
        open_family_count=("resolution_state", lambda values: int(pd.Series(values).eq("open").sum())),
        resolved_family_count=("resolution_state", lambda values: int(pd.Series(values).ne("open").sum())),
        confirmed_publication_gap_count=("resolution_state", lambda values: int(pd.Series(values).eq("confirmed_publication_gap").sum())),
        confirmed_non_boalf_pattern_count=("resolution_state", lambda values: int(pd.Series(values).eq("confirmed_non_boalf_pattern").sum())),
        confirmed_source_artifact_count=("resolution_state", lambda values: int(pd.Series(values).eq("confirmed_source_artifact").sum())),
        not_reproducible_count=("resolution_state", lambda values: int(pd.Series(values).eq("not_reproducible").sum())),
        keep_out_of_precision_count=("truth_policy_action", lambda values: int(pd.Series(values).eq("keep_out_of_precision").sum())),
        eligible_for_new_evidence_tier_count=("truth_policy_action", lambda values: int(pd.Series(values).eq("eligible_for_new_evidence_tier").sum())),
        fix_source_and_rerun_count=("truth_policy_action", lambda values: int(pd.Series(values).eq("fix_source_and_rerun").sum())),
        close_no_change_count=("truth_policy_action", lambda values: int(pd.Series(values).eq("close_no_change").sum())),
    )
    daily = daily.merge(summary, on=["support_batch_id", "settlement_date"], how="left")
    for column in [
        "selected_family_count",
        "open_family_count",
        "resolved_family_count",
        "confirmed_publication_gap_count",
        "confirmed_non_boalf_pattern_count",
        "confirmed_source_artifact_count",
        "not_reproducible_count",
        "keep_out_of_precision_count",
        "eligible_for_new_evidence_tier_count",
        "fix_source_and_rerun_count",
        "close_no_change_count",
    ]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0).astype("Int64")
    state, action = _resolution_review_state_and_action(daily)
    daily["support_resolution_state"] = state
    daily["support_resolution_next_action"] = action
    return daily[columns].sort_values(
        ["support_batch_id", "support_case_day_rank", "settlement_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_fact_support_resolution_batch(
    fact_support_resolution_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "selected_day_count",
        "blocked_day_count",
        "resolved_day_count",
        "selected_family_count",
        "open_family_count",
        "resolved_family_count",
        "confirmed_publication_gap_count",
        "confirmed_non_boalf_pattern_count",
        "confirmed_source_artifact_count",
        "not_reproducible_count",
        "keep_out_of_precision_count",
        "eligible_for_new_evidence_tier_count",
        "fix_source_and_rerun_count",
        "close_no_change_count",
        "support_resolution_state",
        "support_resolution_next_action",
    ]
    if fact_support_resolution_daily.empty:
        return pd.DataFrame(columns=columns)

    batch = fact_support_resolution_daily.groupby(
        ["support_batch_id", "support_generated_at_utc", "support_status_mode", "support_top_days", "support_top_families_per_day"],
        as_index=False,
    ).agg(
        selected_day_count=("settlement_date", "nunique"),
        blocked_day_count=("support_resolution_state", lambda values: int(pd.Series(values).eq("blocked_by_open_cases").sum())),
        resolved_day_count=("support_resolution_state", lambda values: int(pd.Series(values).ne("blocked_by_open_cases").sum())),
        selected_family_count=("selected_family_count", "sum"),
        open_family_count=("open_family_count", "sum"),
        resolved_family_count=("resolved_family_count", "sum"),
        confirmed_publication_gap_count=("confirmed_publication_gap_count", "sum"),
        confirmed_non_boalf_pattern_count=("confirmed_non_boalf_pattern_count", "sum"),
        confirmed_source_artifact_count=("confirmed_source_artifact_count", "sum"),
        not_reproducible_count=("not_reproducible_count", "sum"),
        keep_out_of_precision_count=("keep_out_of_precision_count", "sum"),
        eligible_for_new_evidence_tier_count=("eligible_for_new_evidence_tier_count", "sum"),
        fix_source_and_rerun_count=("fix_source_and_rerun_count", "sum"),
        close_no_change_count=("close_no_change_count", "sum"),
    )
    state, action = _resolution_review_state_and_action(batch)
    batch["support_resolution_state"] = state
    batch["support_resolution_next_action"] = action
    for column in [
        "selected_day_count",
        "blocked_day_count",
        "resolved_day_count",
        "selected_family_count",
        "open_family_count",
        "resolved_family_count",
        "confirmed_publication_gap_count",
        "confirmed_non_boalf_pattern_count",
        "confirmed_source_artifact_count",
        "not_reproducible_count",
        "keep_out_of_precision_count",
        "eligible_for_new_evidence_tier_count",
        "fix_source_and_rerun_count",
        "close_no_change_count",
    ]:
        batch[column] = pd.to_numeric(batch[column], errors="coerce").fillna(0).astype("Int64")
    return batch[columns].sort_values(["support_generated_at_utc", "support_batch_id"], ascending=[True, True]).reset_index(drop=True)


def build_fact_support_rerun_gate_daily(
    fact_support_resolution_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "support_case_day_rank",
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "publication_anomaly_dominant_state",
        "selected_family_count",
        "open_family_count",
        "resolved_family_count",
        "keep_out_of_precision_count",
        "eligible_for_new_evidence_tier_count",
        "fix_source_and_rerun_count",
        "close_no_change_count",
        "support_resolution_state",
        "support_resolution_next_action",
        "support_rerun_gate_state",
        "support_rerun_next_action",
    ]
    if fact_support_resolution_daily.empty:
        return pd.DataFrame(columns=columns)

    daily = fact_support_resolution_daily.copy()
    daily = _coerce_int64_columns(
        daily,
        [
            "selected_family_count",
            "open_family_count",
            "resolved_family_count",
            "keep_out_of_precision_count",
            "eligible_for_new_evidence_tier_count",
            "fix_source_and_rerun_count",
            "close_no_change_count",
        ],
    )
    state, action = _support_rerun_gate_state_and_action(
        daily,
        ready_state="candidate_targeted_rerun",
        no_rerun_state="candidate_policy_lock",
    )
    daily["support_rerun_gate_state"] = state
    daily["support_rerun_next_action"] = action
    return daily[columns].sort_values(
        ["support_batch_id", "support_case_day_rank", "settlement_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_fact_support_rerun_gate_batch(
    fact_support_resolution_batch: pd.DataFrame,
    fact_support_rerun_gate_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "selected_day_count",
        "blocked_day_count",
        "resolved_day_count",
        "selected_family_count",
        "open_family_count",
        "resolved_family_count",
        "keep_out_of_precision_count",
        "eligible_for_new_evidence_tier_count",
        "fix_source_and_rerun_count",
        "close_no_change_count",
        "support_resolution_state",
        "support_resolution_next_action",
        "support_rerun_gate_state",
        "support_rerun_next_action",
        "candidate_rerun_day_count",
        "candidate_rerun_first_date",
        "candidate_rerun_last_date",
    ]
    if fact_support_resolution_batch.empty:
        return pd.DataFrame(columns=columns)

    batch = fact_support_resolution_batch.copy()
    batch = _coerce_int64_columns(
        batch,
        [
            "selected_day_count",
            "blocked_day_count",
            "resolved_day_count",
            "selected_family_count",
            "open_family_count",
            "resolved_family_count",
            "keep_out_of_precision_count",
            "eligible_for_new_evidence_tier_count",
            "fix_source_and_rerun_count",
            "close_no_change_count",
        ],
    )
    state, action = _support_rerun_gate_state_and_action(
        batch,
        ready_state="ready_for_targeted_rerun",
        no_rerun_state="no_rerun_required",
    )
    batch["support_rerun_gate_state"] = state
    batch["support_rerun_next_action"] = action

    candidate_day_summary = pd.DataFrame(
        columns=[
            "support_batch_id",
            "candidate_rerun_day_count",
            "candidate_rerun_first_date",
            "candidate_rerun_last_date",
        ]
    )
    if not fact_support_rerun_gate_daily.empty:
        candidate_daily = fact_support_rerun_gate_daily[
            fact_support_rerun_gate_daily["support_rerun_gate_state"]
            .fillna("")
            .astype(str)
            .eq("candidate_targeted_rerun")
        ].copy()
        if not candidate_daily.empty:
            candidate_day_summary = candidate_daily.groupby("support_batch_id", as_index=False).agg(
                candidate_rerun_day_count=("settlement_date", "nunique"),
                candidate_rerun_first_date=("settlement_date", "min"),
                candidate_rerun_last_date=("settlement_date", "max"),
            )
    batch = batch.merge(candidate_day_summary, on="support_batch_id", how="left")
    batch = _coerce_int64_columns(batch, ["candidate_rerun_day_count"])
    return batch[columns].sort_values(
        ["support_generated_at_utc", "support_batch_id"],
        ascending=[True, True],
    ).reset_index(drop=True)


def build_fact_support_open_case_priority_family_daily(
    fact_support_case_family_daily: pd.DataFrame,
    fact_support_case_resolution: pd.DataFrame,
    fact_support_rerun_gate_daily: pd.DataFrame,
    fact_support_rerun_gate_batch: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "support_case_day_rank",
        "support_case_family_rank",
        "support_case_family_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "publication_anomaly_candidate_mwh_lower_bound",
        "publication_anomaly_dominant_state",
        "family_publication_anomaly_priority_rank",
        "support_question_code",
        "family_publication_anomaly_next_action",
        "support_recommended_action",
        "resolution_state",
        "truth_policy_action",
        "support_rerun_gate_state_daily",
        "support_rerun_next_action_daily",
        "support_rerun_gate_state_batch",
        "support_rerun_next_action_batch",
        "open_case_priority_rank",
        "priority_reason",
    ]
    if fact_support_case_family_daily.empty:
        return pd.DataFrame(columns=columns)

    family = fact_support_case_family_daily.copy()
    resolution = fact_support_case_resolution.copy()
    if resolution.empty:
        family["resolution_state"] = "open"
        family["truth_policy_action"] = "keep_out_of_precision"
    else:
        keep_columns = [
            "support_batch_id",
            "settlement_date",
            "bmu_family_key",
            "resolution_state",
            "truth_policy_action",
        ]
        family = family.merge(
            resolution[[column for column in keep_columns if column in resolution.columns]],
            on=["support_batch_id", "settlement_date", "bmu_family_key"],
            how="left",
        )
        family["resolution_state"] = family["resolution_state"].fillna("open")
        family["truth_policy_action"] = family["truth_policy_action"].fillna("keep_out_of_precision")

    family = family[family["resolution_state"].fillna("open").astype(str).eq("open")].copy()
    if family.empty:
        return pd.DataFrame(columns=columns)

    family["publication_anomaly_candidate_mwh_lower_bound"] = pd.to_numeric(
        family["publication_anomaly_candidate_mwh_lower_bound"], errors="coerce"
    ).fillna(0.0)
    rank_source = (
        pd.to_numeric(family["day_family_rank_by_publication_anomaly"], errors="coerce")
        if "day_family_rank_by_publication_anomaly" in family.columns
        else pd.Series(pd.NA, index=family.index, dtype="object")
    )
    fallback_rank = (
        pd.to_numeric(family["support_case_family_rank"], errors="coerce")
        if "support_case_family_rank" in family.columns
        else pd.Series(pd.NA, index=family.index, dtype="object")
    )
    family["family_publication_anomaly_priority_rank"] = rank_source.where(rank_source.notna(), fallback_rank)
    family["family_publication_anomaly_priority_rank"] = (
        pd.to_numeric(family["family_publication_anomaly_priority_rank"], errors="coerce").fillna(999999).astype("Int64")
    )
    family["family_publication_anomaly_next_action"] = (
        family["publication_anomaly_next_action"]
        if "publication_anomaly_next_action" in family.columns
        else ""
    )

    if not fact_support_rerun_gate_daily.empty:
        daily_gate = fact_support_rerun_gate_daily[
            [
                "support_batch_id",
                "settlement_date",
                "support_rerun_gate_state",
                "support_rerun_next_action",
            ]
        ].rename(
            columns={
                "support_rerun_gate_state": "support_rerun_gate_state_daily",
                "support_rerun_next_action": "support_rerun_next_action_daily",
            }
        )
        family = family.merge(daily_gate, on=["support_batch_id", "settlement_date"], how="left")
    else:
        family["support_rerun_gate_state_daily"] = pd.NA
        family["support_rerun_next_action_daily"] = pd.NA

    if not fact_support_rerun_gate_batch.empty:
        batch_gate = fact_support_rerun_gate_batch[
            ["support_batch_id", "support_rerun_gate_state", "support_rerun_next_action"]
        ].rename(
            columns={
                "support_rerun_gate_state": "support_rerun_gate_state_batch",
                "support_rerun_next_action": "support_rerun_next_action_batch",
            }
        )
        family = family.merge(batch_gate, on="support_batch_id", how="left")
    else:
        family["support_rerun_gate_state_batch"] = pd.NA
        family["support_rerun_next_action_batch"] = pd.NA

    family = family.sort_values(
        [
            "support_batch_id",
            "publication_anomaly_candidate_mwh_lower_bound",
            "family_publication_anomaly_priority_rank",
            "settlement_date",
            "bmu_family_key",
        ],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    family["open_case_priority_rank"] = family.groupby("support_batch_id").cumcount().add(1).astype("Int64")
    family["priority_reason"] = (
        "open_case::"
        + family["publication_anomaly_dominant_state"].fillna("unknown").astype(str)
        + "::"
        + family["support_question_code"].fillna("no_question_code").astype(str)
    )
    return family[columns].reset_index(drop=True)


def build_fact_support_resolution_pattern_summary(
    fact_support_open_case_priority_family_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_resolution_pattern_key",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "bmu_family_key",
        "bmu_family_label",
        "publication_anomaly_dominant_state",
        "support_question_code",
        "support_recommended_action",
        "mapping_status",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "open_case_count",
        "open_day_count",
        "first_settlement_date",
        "last_settlement_date",
        "pattern_publication_anomaly_mwh",
        "pattern_max_case_mwh",
        "first_open_case_priority_rank",
        "support_rerun_gate_state_batch",
        "support_rerun_next_action_batch",
        "pattern_review_state",
    ]
    if fact_support_open_case_priority_family_daily.empty:
        return pd.DataFrame(columns=columns)

    open_cases = fact_support_open_case_priority_family_daily.copy()
    open_cases["support_resolution_pattern_key"] = _support_resolution_pattern_key(open_cases)
    open_cases["publication_anomaly_candidate_mwh_lower_bound"] = pd.to_numeric(
        open_cases["publication_anomaly_candidate_mwh_lower_bound"], errors="coerce"
    ).fillna(0.0)
    open_cases["open_case_priority_rank"] = pd.to_numeric(
        open_cases["open_case_priority_rank"], errors="coerce"
    ).fillna(999999)

    summary = open_cases.groupby(
        ["support_batch_id", "support_resolution_pattern_key"],
        as_index=False,
        dropna=False,
    ).agg(
        support_generated_at_utc=("support_generated_at_utc", "first"),
        support_status_mode=("support_status_mode", "first"),
        support_top_days=("support_top_days", "first"),
        support_top_families_per_day=("support_top_families_per_day", "first"),
        bmu_family_key=("bmu_family_key", "first"),
        bmu_family_label=("bmu_family_label", "first"),
        publication_anomaly_dominant_state=("publication_anomaly_dominant_state", "first"),
        support_question_code=("support_question_code", "first"),
        support_recommended_action=("support_recommended_action", "first"),
        mapping_status=("mapping_status", "first"),
        cluster_key=("cluster_key", "first"),
        cluster_label=("cluster_label", "first"),
        parent_region=("parent_region", "first"),
        open_case_count=("support_case_family_key", "nunique"),
        open_day_count=("settlement_date", "nunique"),
        first_settlement_date=("settlement_date", "min"),
        last_settlement_date=("settlement_date", "max"),
        pattern_publication_anomaly_mwh=("publication_anomaly_candidate_mwh_lower_bound", "sum"),
        pattern_max_case_mwh=("publication_anomaly_candidate_mwh_lower_bound", "max"),
        first_open_case_priority_rank=("open_case_priority_rank", "min"),
        support_rerun_gate_state_batch=("support_rerun_gate_state_batch", "first"),
        support_rerun_next_action_batch=("support_rerun_next_action_batch", "first"),
    )
    summary = _coerce_int64_columns(summary, ["open_case_count", "open_day_count"])
    summary["first_open_case_priority_rank"] = pd.to_numeric(
        summary["first_open_case_priority_rank"], errors="coerce"
    ).fillna(999999).astype("Int64")
    summary["pattern_review_state"] = pd.Series("single_day", index=summary.index, dtype="object")
    summary.loc[summary["open_day_count"].gt(1), "pattern_review_state"] = "multi_day"
    return summary[columns].sort_values(
        [
            "support_batch_id",
            "first_open_case_priority_rank",
            "pattern_publication_anomaly_mwh",
            "bmu_family_key",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def build_fact_support_resolution_pattern_member_family_daily(
    fact_support_open_case_priority_family_daily: pd.DataFrame,
    fact_support_resolution_pattern_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_resolution_pattern_key",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "support_case_day_rank",
        "support_case_family_rank",
        "support_case_family_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "publication_anomaly_dominant_state",
        "support_question_code",
        "support_recommended_action",
        "mapping_status",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "publication_anomaly_candidate_mwh_lower_bound",
        "open_case_priority_rank",
        "support_rerun_gate_state_daily",
        "support_rerun_next_action_daily",
        "support_rerun_gate_state_batch",
        "support_rerun_next_action_batch",
        "pattern_review_state",
    ]
    if fact_support_open_case_priority_family_daily.empty:
        return pd.DataFrame(columns=columns)

    members = fact_support_open_case_priority_family_daily.copy()
    members["support_resolution_pattern_key"] = _support_resolution_pattern_key(members)
    if fact_support_resolution_pattern_summary.empty:
        members["pattern_review_state"] = pd.NA
    else:
        members = members.merge(
            fact_support_resolution_pattern_summary[
                ["support_batch_id", "support_resolution_pattern_key", "pattern_review_state"]
            ],
            on=["support_batch_id", "support_resolution_pattern_key"],
            how="left",
        )
    return members[columns].sort_values(
        [
            "support_batch_id",
            "open_case_priority_rank",
            "settlement_date",
            "bmu_family_key",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_fact_support_rerun_candidate_daily(
    fact_support_rerun_gate_daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "support_case_day_rank",
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "publication_anomaly_dominant_state",
        "selected_family_count",
        "resolved_family_count",
        "eligible_for_new_evidence_tier_count",
        "fix_source_and_rerun_count",
        "close_no_change_count",
        "keep_out_of_precision_count",
        "support_rerun_gate_state",
        "support_rerun_next_action",
        "rerun_candidate_action_count",
        "rerun_candidate_action_mix",
    ]
    if fact_support_rerun_gate_daily.empty:
        return pd.DataFrame(columns=columns)

    daily = fact_support_rerun_gate_daily.copy()
    daily = daily[
        daily["support_rerun_gate_state"].fillna("").astype(str).eq("candidate_targeted_rerun")
    ].copy()
    if daily.empty:
        return pd.DataFrame(columns=columns)

    daily = _coerce_int64_columns(
        daily,
        [
            "selected_family_count",
            "resolved_family_count",
            "eligible_for_new_evidence_tier_count",
            "fix_source_and_rerun_count",
            "close_no_change_count",
            "keep_out_of_precision_count",
        ],
    )
    daily["rerun_candidate_action_count"] = (
        daily["eligible_for_new_evidence_tier_count"] + daily["fix_source_and_rerun_count"]
    ).astype("Int64")
    mix = pd.Series("none", index=daily.index, dtype="object")
    fix_mask = daily["fix_source_and_rerun_count"].gt(0)
    eligible_mask = daily["eligible_for_new_evidence_tier_count"].gt(0)
    mix.loc[fix_mask & ~eligible_mask] = "fix_source_and_rerun_only"
    mix.loc[~fix_mask & eligible_mask] = "eligible_for_new_evidence_tier_only"
    mix.loc[fix_mask & eligible_mask] = "mixed_fix_and_evidence_tier"
    daily["rerun_candidate_action_mix"] = mix
    return daily[columns].sort_values(
        ["support_batch_id", "support_case_day_rank", "settlement_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_fact_support_rerun_candidate_family_daily(
    fact_support_case_family_daily: pd.DataFrame,
    fact_support_case_resolution: pd.DataFrame,
    fact_support_rerun_gate_daily: pd.DataFrame,
    fact_support_rerun_gate_batch: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "support_batch_id",
        "support_generated_at_utc",
        "support_status_mode",
        "support_top_days",
        "support_top_families_per_day",
        "support_case_day_rank",
        "support_case_family_rank",
        "support_case_family_key",
        "settlement_date",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "publication_anomaly_candidate_mwh_lower_bound",
        "publication_anomaly_dominant_state",
        "support_question_code",
        "support_recommended_action",
        "resolution_state",
        "truth_policy_action",
        "rerun_candidate_action",
        "support_rerun_gate_state_daily",
        "support_rerun_next_action_daily",
        "support_rerun_gate_state_batch",
        "support_rerun_next_action_batch",
        "rerun_candidate_family_rank",
    ]
    if (
        fact_support_case_family_daily.empty
        or fact_support_case_resolution.empty
        or fact_support_rerun_gate_daily.empty
    ):
        return pd.DataFrame(columns=columns)

    family = fact_support_case_family_daily.copy()
    resolution = fact_support_case_resolution.copy()
    family = family.merge(
        resolution[
            [
                "support_batch_id",
                "settlement_date",
                "bmu_family_key",
                "resolution_state",
                "truth_policy_action",
            ]
        ],
        on=["support_batch_id", "settlement_date", "bmu_family_key"],
        how="inner",
    )
    family = family[
        family["truth_policy_action"]
        .fillna("")
        .astype(str)
        .isin({"fix_source_and_rerun", "eligible_for_new_evidence_tier"})
    ].copy()
    if family.empty:
        return pd.DataFrame(columns=columns)

    daily_gate = fact_support_rerun_gate_daily[
        fact_support_rerun_gate_daily["support_rerun_gate_state"]
        .fillna("")
        .astype(str)
        .eq("candidate_targeted_rerun")
    ][
        [
            "support_batch_id",
            "settlement_date",
            "support_rerun_gate_state",
            "support_rerun_next_action",
        ]
    ].rename(
        columns={
            "support_rerun_gate_state": "support_rerun_gate_state_daily",
            "support_rerun_next_action": "support_rerun_next_action_daily",
        }
    )
    family = family.merge(daily_gate, on=["support_batch_id", "settlement_date"], how="inner")
    if family.empty:
        return pd.DataFrame(columns=columns)

    if not fact_support_rerun_gate_batch.empty:
        batch_gate = fact_support_rerun_gate_batch[
            ["support_batch_id", "support_rerun_gate_state", "support_rerun_next_action"]
        ].rename(
            columns={
                "support_rerun_gate_state": "support_rerun_gate_state_batch",
                "support_rerun_next_action": "support_rerun_next_action_batch",
            }
        )
        family = family.merge(batch_gate, on="support_batch_id", how="left")
    else:
        family["support_rerun_gate_state_batch"] = pd.NA
        family["support_rerun_next_action_batch"] = pd.NA

    family["publication_anomaly_candidate_mwh_lower_bound"] = pd.to_numeric(
        family["publication_anomaly_candidate_mwh_lower_bound"], errors="coerce"
    ).fillna(0.0)
    family["rerun_candidate_action"] = family["truth_policy_action"].map(
        {
            "fix_source_and_rerun": "rerun_after_source_fix",
            "eligible_for_new_evidence_tier": "review_new_evidence_tier_then_rerun",
        }
    ).fillna("rerun_candidate_review")
    family = family.sort_values(
        [
            "support_batch_id",
            "support_case_day_rank",
            "support_case_family_rank",
            "publication_anomaly_candidate_mwh_lower_bound",
            "bmu_family_key",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    family["rerun_candidate_family_rank"] = family.groupby(
        ["support_batch_id", "settlement_date"]
    ).cumcount().add(1).astype("Int64")
    return family[columns].reset_index(drop=True)


def materialize_truth_store_support_resolution(
    db_path: str | Path,
    support_batch_id: str | None = None,
    fact_support_case_family_daily: pd.DataFrame | None = None,
    generated_at_utc: str | None = None,
) -> Dict[str, pd.DataFrame]:
    target_path = Path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"truth store does not exist: {target_path}")

    if fact_support_case_family_daily is None:
        support_family = _load_table(target_path, SUPPORT_CASE_FAMILY_TABLE)
    else:
        support_family = fact_support_case_family_daily.copy()
    if support_batch_id:
        support_family = support_family[
            support_family["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))
        ].copy()

    existing = _load_table(target_path, SUPPORT_CASE_RESOLUTION_TABLE) if _table_exists(target_path, SUPPORT_CASE_RESOLUTION_TABLE) else _empty_resolution_frame()
    if support_batch_id and not existing.empty:
        existing = existing[existing["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()

    if support_batch_id and fact_support_case_family_daily is None:
        support_family = _load_table(target_path, SUPPORT_CASE_FAMILY_TABLE)
        support_family = support_family[
            support_family["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))
        ].copy()

    resolution = build_fact_support_case_resolution(
        fact_support_case_family_daily=support_family,
        existing_resolution=existing,
        generated_at_utc=generated_at_utc,
    )
    if _table_exists(target_path, "fact_support_case_daily"):
        support_daily = _load_table(target_path, "fact_support_case_daily")
        if support_batch_id:
            support_daily = support_daily[
                support_daily["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))
            ].copy()
    else:
        support_daily = _derive_support_case_daily_from_family(support_family)
    if support_daily.empty and fact_support_case_family_daily is not None:
        support_daily = _derive_support_case_daily_from_family(support_family)
    support_resolution_daily = build_fact_support_resolution_daily(
        fact_support_case_daily=support_daily,
        fact_support_case_family_daily=support_family,
        fact_support_case_resolution=resolution,
    )
    support_resolution_batch = build_fact_support_resolution_batch(support_resolution_daily)
    support_rerun_gate_daily = build_fact_support_rerun_gate_daily(support_resolution_daily)
    support_rerun_gate_batch = build_fact_support_rerun_gate_batch(
        fact_support_resolution_batch=support_resolution_batch,
        fact_support_rerun_gate_daily=support_rerun_gate_daily,
    )
    support_open_case_priority = build_fact_support_open_case_priority_family_daily(
        fact_support_case_family_daily=support_family,
        fact_support_case_resolution=resolution,
        fact_support_rerun_gate_daily=support_rerun_gate_daily,
        fact_support_rerun_gate_batch=support_rerun_gate_batch,
    )
    support_pattern_summary = build_fact_support_resolution_pattern_summary(support_open_case_priority)
    support_pattern_members = build_fact_support_resolution_pattern_member_family_daily(
        fact_support_open_case_priority_family_daily=support_open_case_priority,
        fact_support_resolution_pattern_summary=support_pattern_summary,
    )
    support_rerun_candidate_daily = build_fact_support_rerun_candidate_daily(support_rerun_gate_daily)
    support_rerun_candidate_family = build_fact_support_rerun_candidate_family_daily(
        fact_support_case_family_daily=support_family,
        fact_support_case_resolution=resolution,
        fact_support_rerun_gate_daily=support_rerun_gate_daily,
        fact_support_rerun_gate_batch=support_rerun_gate_batch,
    )
    batch_ids_to_refresh = (
        [str(support_batch_id)]
        if support_batch_id
        else sorted(
            {
                str(value)
                for value in support_family.get("support_batch_id", pd.Series(dtype="object")).dropna().astype(str)
                if str(value)
            }
        )
    )
    for table_name in [
        SUPPORT_CASE_RESOLUTION_TABLE,
        SUPPORT_RESOLUTION_DAILY_TABLE,
        SUPPORT_RESOLUTION_BATCH_TABLE,
        SUPPORT_RERUN_GATE_DAILY_TABLE,
        SUPPORT_RERUN_GATE_BATCH_TABLE,
        SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE,
        SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE,
        SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE,
        SUPPORT_RERUN_CANDIDATE_DAILY_TABLE,
        SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE,
    ]:
        _delete_rows_for_support_batches(target_path, table_name, batch_ids_to_refresh)
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_CASE_RESOLUTION_TABLE,
        frame=resolution,
        primary_keys=["support_batch_id", "settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RESOLUTION_DAILY_TABLE,
        frame=support_resolution_daily,
        primary_keys=["support_batch_id", "settlement_date"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RESOLUTION_BATCH_TABLE,
        frame=support_resolution_batch,
        primary_keys=["support_batch_id"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RERUN_GATE_DAILY_TABLE,
        frame=support_rerun_gate_daily,
        primary_keys=["support_batch_id", "settlement_date"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RERUN_GATE_BATCH_TABLE,
        frame=support_rerun_gate_batch,
        primary_keys=["support_batch_id"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE,
        frame=support_open_case_priority,
        primary_keys=["support_batch_id", "settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE,
        frame=support_pattern_summary,
        primary_keys=["support_batch_id", "support_resolution_pattern_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE,
        frame=support_pattern_members,
        primary_keys=["support_batch_id", "settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RERUN_CANDIDATE_DAILY_TABLE,
        frame=support_rerun_candidate_daily,
        primary_keys=["support_batch_id", "settlement_date"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE,
        frame=support_rerun_candidate_family,
        primary_keys=["support_batch_id", "settlement_date", "bmu_family_key"],
    )
    return {
        SUPPORT_CASE_RESOLUTION_TABLE: resolution,
        SUPPORT_RESOLUTION_DAILY_TABLE: support_resolution_daily,
        SUPPORT_RESOLUTION_BATCH_TABLE: support_resolution_batch,
        SUPPORT_RERUN_GATE_DAILY_TABLE: support_rerun_gate_daily,
        SUPPORT_RERUN_GATE_BATCH_TABLE: support_rerun_gate_batch,
        SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE: support_open_case_priority,
        SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE: support_pattern_summary,
        SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE: support_pattern_members,
        SUPPORT_RERUN_CANDIDATE_DAILY_TABLE: support_rerun_candidate_daily,
        SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE: support_rerun_candidate_family,
    }


def annotate_support_case_resolution(
    db_path: str | Path,
    support_batch_id: str,
    settlement_date: str,
    bmu_family_key: str,
    resolution_state: str,
    truth_policy_action: str,
    resolution_note: str | None = None,
    source_reference: str | None = None,
    generated_at_utc: str | None = None,
) -> pd.DataFrame:
    if resolution_state not in VALID_RESOLUTION_STATES:
        raise ValueError(f"unsupported resolution state '{resolution_state}'")
    if truth_policy_action not in VALID_TRUTH_POLICY_ACTIONS:
        raise ValueError(f"unsupported truth policy action '{truth_policy_action}'")

    target_path = Path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"truth store does not exist: {target_path}")

    resolution_frame = (
        _load_table(target_path, SUPPORT_CASE_RESOLUTION_TABLE)
        if _table_exists(target_path, SUPPORT_CASE_RESOLUTION_TABLE)
        else _empty_resolution_frame()
    )
    row_mask = (
        resolution_frame["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))
        & resolution_frame["settlement_date"].fillna("").astype(str).eq(str(settlement_date))
        & resolution_frame["bmu_family_key"].fillna("").astype(str).eq(str(bmu_family_key))
    )
    if not resolution_frame.empty and bool(row_mask.any()):
        row = resolution_frame[row_mask].copy().iloc[[0]]
    else:
        support_family = _load_table(target_path, SUPPORT_CASE_FAMILY_TABLE)
        support_family = support_family[
            support_family["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))
            & support_family["settlement_date"].fillna("").astype(str).eq(str(settlement_date))
            & support_family["bmu_family_key"].fillna("").astype(str).eq(str(bmu_family_key))
        ].copy()
        if support_family.empty:
            raise ValueError(
                "support-case family row not found; materialize the support loop or resolution table first"
            )
        row = build_fact_support_case_resolution(
            fact_support_case_family_daily=support_family,
            existing_resolution=None,
            generated_at_utc=generated_at_utc,
        ).iloc[[0]]

    updated_at = _resolve_generated_at_utc(generated_at_utc)
    row["resolution_state"] = resolution_state
    row["truth_policy_action"] = truth_policy_action
    if resolution_note is not None:
        row["resolution_note"] = resolution_note
    if source_reference is not None:
        row["source_reference"] = source_reference
    row["resolution_updated_at_utc"] = updated_at

    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_CASE_RESOLUTION_TABLE,
        frame=row[_resolution_columns()],
        primary_keys=["support_batch_id", "settlement_date", "bmu_family_key"],
    )
    materialize_truth_store_support_resolution(
        db_path=target_path,
        support_batch_id=support_batch_id,
        generated_at_utc=updated_at,
    )
    return row[_resolution_columns()].reset_index(drop=True)


def annotate_support_resolution_pattern(
    db_path: str | Path,
    support_batch_id: str,
    resolution_pattern_key: str,
    resolution_state: str,
    truth_policy_action: str,
    resolution_note: str | None = None,
    source_reference: str | None = None,
    generated_at_utc: str | None = None,
) -> pd.DataFrame:
    if resolution_state not in VALID_RESOLUTION_STATES:
        raise ValueError(f"unsupported resolution state '{resolution_state}'")
    if truth_policy_action not in VALID_TRUTH_POLICY_ACTIONS:
        raise ValueError(f"unsupported truth policy action '{truth_policy_action}'")

    target_path = Path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"truth store does not exist: {target_path}")

    updated_at = _resolve_generated_at_utc(generated_at_utc)
    materialized = materialize_truth_store_support_resolution(
        db_path=target_path,
        support_batch_id=support_batch_id,
        generated_at_utc=updated_at,
    )
    pattern_members = materialized[SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE]
    selected_members = pattern_members[
        pattern_members["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))
        & pattern_members["support_resolution_pattern_key"].fillna("").astype(str).eq(str(resolution_pattern_key))
    ].copy()
    if selected_members.empty:
        raise ValueError(
            "support-resolution pattern row not found; materialize the support-resolution patterns first"
        )

    support_family = _load_table(target_path, SUPPORT_CASE_FAMILY_TABLE)
    support_family = support_family[
        support_family["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))
    ].copy()
    support_family = support_family.merge(
        selected_members[["support_batch_id", "settlement_date", "bmu_family_key"]].drop_duplicates(),
        on=["support_batch_id", "settlement_date", "bmu_family_key"],
        how="inner",
    )
    if support_family.empty:
        raise ValueError("support-case family rows for the selected pattern are missing")

    existing = _load_table(target_path, SUPPORT_CASE_RESOLUTION_TABLE)
    existing = existing[existing["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
    rows = build_fact_support_case_resolution(
        fact_support_case_family_daily=support_family,
        existing_resolution=existing,
        generated_at_utc=updated_at,
    )
    rows["resolution_state"] = resolution_state
    rows["truth_policy_action"] = truth_policy_action
    if resolution_note is not None:
        rows["resolution_note"] = resolution_note
    if source_reference is not None:
        rows["source_reference"] = source_reference
    rows["resolution_updated_at_utc"] = updated_at

    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_CASE_RESOLUTION_TABLE,
        frame=rows[_resolution_columns()],
        primary_keys=["support_batch_id", "settlement_date", "bmu_family_key"],
    )
    materialize_truth_store_support_resolution(
        db_path=target_path,
        support_batch_id=support_batch_id,
        generated_at_utc=updated_at,
    )
    return rows[_resolution_columns()].sort_values(
        ["settlement_date", "support_case_family_rank", "bmu_family_key"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def read_support_case_resolution(
    db_path: str | Path,
    support_batch_id: str | None = None,
    resolution_filter: str = "all",
) -> pd.DataFrame:
    if resolution_filter not in VALID_RESOLUTION_FILTERS:
        raise ValueError(f"unsupported resolution filter '{resolution_filter}'")
    if not _table_exists(db_path, SUPPORT_CASE_RESOLUTION_TABLE):
        return _empty_resolution_frame()

    frame = _load_table(db_path, SUPPORT_CASE_RESOLUTION_TABLE)
    if support_batch_id:
        frame = frame[frame["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
    if resolution_filter == "all":
        pass
    elif resolution_filter == "resolved":
        frame = frame[~frame["resolution_state"].fillna("open").astype(str).eq("open")].copy()
    else:
        frame = frame[frame["resolution_state"].fillna("open").astype(str).eq(str(resolution_filter))].copy()
    return frame.sort_values(
        ["support_batch_id", "settlement_date", "support_case_family_rank", "bmu_family_key"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def read_support_resolution_review(
    db_path: str | Path,
    support_batch_id: str | None = None,
) -> Dict[str, pd.DataFrame]:
    daily = _load_table(db_path, SUPPORT_RESOLUTION_DAILY_TABLE) if _table_exists(db_path, SUPPORT_RESOLUTION_DAILY_TABLE) else pd.DataFrame()
    batch = _load_table(db_path, SUPPORT_RESOLUTION_BATCH_TABLE) if _table_exists(db_path, SUPPORT_RESOLUTION_BATCH_TABLE) else pd.DataFrame()
    if support_batch_id:
        if not daily.empty:
            daily = daily[daily["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
        if not batch.empty:
            batch = batch[batch["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
    return {
        SUPPORT_RESOLUTION_DAILY_TABLE: daily.reset_index(drop=True),
        SUPPORT_RESOLUTION_BATCH_TABLE: batch.reset_index(drop=True),
    }


def read_support_resolution_pattern_review(
    db_path: str | Path,
    support_batch_id: str | None = None,
    pattern_filter: str = "all",
    resolution_pattern_key: str | None = None,
) -> Dict[str, pd.DataFrame]:
    if pattern_filter not in VALID_SUPPORT_PATTERN_FILTERS:
        raise ValueError(f"unsupported support pattern filter '{pattern_filter}'")

    summary = (
        _load_table(db_path, SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE)
        if _table_exists(db_path, SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE)
        else pd.DataFrame()
    )
    members = (
        _load_table(db_path, SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE)
        if _table_exists(db_path, SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE)
        else pd.DataFrame()
    )

    if support_batch_id:
        if not summary.empty:
            summary = summary[summary["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
        if not members.empty:
            members = members[members["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
    if resolution_pattern_key:
        if not summary.empty:
            summary = summary[
                summary["support_resolution_pattern_key"].fillna("").astype(str).eq(str(resolution_pattern_key))
            ].copy()
        if not members.empty:
            members = members[
                members["support_resolution_pattern_key"].fillna("").astype(str).eq(str(resolution_pattern_key))
            ].copy()
    if pattern_filter == "single_day" and not summary.empty:
        summary = summary[summary["pattern_review_state"].fillna("").astype(str).eq("single_day")].copy()
    elif pattern_filter == "multi_day" and not summary.empty:
        summary = summary[summary["pattern_review_state"].fillna("").astype(str).eq("multi_day")].copy()

    if not members.empty:
        if summary.empty:
            members = members.iloc[0:0].copy()
        else:
            members = members.merge(
                summary[["support_batch_id", "support_resolution_pattern_key"]].drop_duplicates(),
                on=["support_batch_id", "support_resolution_pattern_key"],
                how="inner",
            )

    if not summary.empty:
        summary = summary.sort_values(
            [
                "support_batch_id",
                "first_open_case_priority_rank",
                "pattern_publication_anomaly_mwh",
                "bmu_family_key",
            ],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
    if not members.empty:
        members = members.sort_values(
            ["support_batch_id", "open_case_priority_rank", "settlement_date", "bmu_family_key"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    return {
        SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE: summary,
        SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE: members,
    }


def read_support_rerun_gate_review(
    db_path: str | Path,
    support_batch_id: str | None = None,
    gate_filter: str = "all",
) -> Dict[str, pd.DataFrame]:
    if gate_filter not in VALID_SUPPORT_GATE_FILTERS:
        raise ValueError(f"unsupported support gate filter '{gate_filter}'")

    daily = (
        _load_table(db_path, SUPPORT_RERUN_GATE_DAILY_TABLE)
        if _table_exists(db_path, SUPPORT_RERUN_GATE_DAILY_TABLE)
        else pd.DataFrame()
    )
    batch = (
        _load_table(db_path, SUPPORT_RERUN_GATE_BATCH_TABLE)
        if _table_exists(db_path, SUPPORT_RERUN_GATE_BATCH_TABLE)
        else pd.DataFrame()
    )
    priority = (
        _load_table(db_path, SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE)
        if _table_exists(db_path, SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE)
        else pd.DataFrame()
    )

    if support_batch_id:
        if not daily.empty:
            daily = daily[daily["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
        if not batch.empty:
            batch = batch[batch["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
        if not priority.empty:
            priority = priority[priority["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()

    if gate_filter != "all":
        batch_state_map = {
            "blocked": "blocked_by_open_cases",
            "ready_for_rerun": "ready_for_targeted_rerun",
            "no_rerun_required": "no_rerun_required",
        }
        daily_state_map = {
            "blocked": "blocked_by_open_cases",
            "ready_for_rerun": "candidate_targeted_rerun",
            "no_rerun_required": "candidate_policy_lock",
        }
        if not batch.empty:
            batch = batch[
                batch["support_rerun_gate_state"].fillna("").astype(str).eq(batch_state_map[gate_filter])
            ].copy()
        if not daily.empty:
            daily = daily[
                daily["support_rerun_gate_state"].fillna("").astype(str).eq(daily_state_map[gate_filter])
            ].copy()

    if not priority.empty:
        if daily.empty:
            priority = priority.iloc[0:0].copy()
        else:
            priority = priority.merge(
                daily[["support_batch_id", "settlement_date"]].drop_duplicates(),
                on=["support_batch_id", "settlement_date"],
                how="inner",
            )
        priority = priority.sort_values(
            ["support_batch_id", "open_case_priority_rank", "settlement_date", "bmu_family_key"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    return {
        SUPPORT_RERUN_GATE_DAILY_TABLE: daily.reset_index(drop=True),
        SUPPORT_RERUN_GATE_BATCH_TABLE: batch.reset_index(drop=True),
        SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE: priority.reset_index(drop=True),
    }


def read_support_rerun_candidate_review(
    db_path: str | Path,
    support_batch_id: str | None = None,
    candidate_filter: str = "all",
) -> Dict[str, pd.DataFrame]:
    if candidate_filter not in VALID_SUPPORT_RERUN_CANDIDATE_FILTERS:
        raise ValueError(f"unsupported support rerun candidate filter '{candidate_filter}'")

    daily = (
        _load_table(db_path, SUPPORT_RERUN_CANDIDATE_DAILY_TABLE)
        if _table_exists(db_path, SUPPORT_RERUN_CANDIDATE_DAILY_TABLE)
        else pd.DataFrame()
    )
    family = (
        _load_table(db_path, SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE)
        if _table_exists(db_path, SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE)
        else pd.DataFrame()
    )

    if support_batch_id:
        if not daily.empty:
            daily = daily[daily["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()
        if not family.empty:
            family = family[family["support_batch_id"].fillna("").astype(str).eq(str(support_batch_id))].copy()

    if candidate_filter != "all":
        if not family.empty:
            family = family[
                family["truth_policy_action"].fillna("").astype(str).eq(candidate_filter)
            ].copy()
        if not daily.empty:
            if family.empty:
                daily = daily.iloc[0:0].copy()
            else:
                daily = daily.merge(
                    family[["support_batch_id", "settlement_date"]].drop_duplicates(),
                    on=["support_batch_id", "settlement_date"],
                    how="inner",
                )

    if not daily.empty:
        daily = daily.sort_values(
            ["support_batch_id", "support_case_day_rank", "settlement_date"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    if not family.empty:
        family = family.sort_values(
            ["support_batch_id", "support_case_day_rank", "rerun_candidate_family_rank", "bmu_family_key"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    return {
        SUPPORT_RERUN_CANDIDATE_DAILY_TABLE: daily,
        SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE: family,
    }
