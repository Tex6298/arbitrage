from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import pandas as pd

from history_store import upsert_frame_to_sqlite
from truth_store_focus import (
    PUBLICATION_ANOMALY_DAILY_TABLE,
    PUBLICATION_ANOMALY_FAMILY_TABLE,
    SOURCE_GAP_FAMILY_TABLE,
    _load_table,
    _status_filter_mask,
    materialize_truth_store_source_focus,
)
from truth_store_forensics import (
    build_fact_family_publication_audit_daily,
    build_fact_family_support_evidence_half_hourly,
)


SUPPORT_CASE_DAILY_TABLE = "fact_support_case_daily"
SUPPORT_CASE_FAMILY_TABLE = "fact_support_case_family_daily"
SUPPORT_CASE_HALF_HOURLY_TABLE = "fact_support_case_half_hourly"
SUPPORT_SUMMARY_FILENAME = "support_case_summary.md"
VALID_SUPPORT_STATUS_MODES = ("all", "fail", "fail_warn")


def _support_batch_id(
    status_mode: str,
    top_days: int,
    top_families_per_day: int,
    selected_dates: list[str],
) -> str:
    if selected_dates:
        start_date = min(selected_dates)
        end_date = max(selected_dates)
    else:
        start_date = "empty"
        end_date = "empty"
    return (
        f"support_{status_mode}_days{top_days}_families{top_families_per_day}_"
        f"{start_date}_{end_date}"
    )


def _resolve_generated_at_utc(value: str | None = None) -> str:
    if value:
        return value
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _ensure_string_column(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def _select_support_days(
    publication_anomaly_daily: pd.DataFrame,
    status_mode: str,
    top_days: int,
) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "next_action",
        "remaining_qa_shortfall_mwh",
        "publication_anomaly_family_count",
        "publication_anomaly_row_count",
        "publication_anomaly_distinct_bmu_count",
        "publication_anomaly_candidate_mwh_lower_bound",
        "publication_anomaly_negative_bid_mwh_lower_bound",
        "publication_anomaly_sentinel_mwh_lower_bound",
        "publication_anomaly_dynamic_limit_mwh_lower_bound",
        "publication_anomaly_other_mwh_lower_bound",
        "publication_anomaly_share_of_remaining_qa_shortfall",
        "publication_anomaly_dominant_state",
        "publication_anomaly_priority_rank",
        "publication_anomaly_next_action",
    ]
    if publication_anomaly_daily.empty:
        return pd.DataFrame(columns=columns + ["support_case_day_rank"])

    prepared = publication_anomaly_daily.copy()
    prepared = prepared[_status_filter_mask(prepared, status_mode)].copy()
    next_action = _ensure_string_column(prepared, "publication_anomaly_next_action")
    prepared = prepared[next_action.str.startswith("support_query_")].copy()
    if prepared.empty:
        return pd.DataFrame(columns=columns + ["support_case_day_rank"])

    prepared["publication_anomaly_priority_rank"] = pd.to_numeric(
        prepared["publication_anomaly_priority_rank"], errors="coerce"
    )
    prepared["publication_anomaly_candidate_mwh_lower_bound"] = pd.to_numeric(
        prepared["publication_anomaly_candidate_mwh_lower_bound"], errors="coerce"
    ).fillna(0.0)
    prepared = prepared.sort_values(
        ["publication_anomaly_priority_rank", "publication_anomaly_candidate_mwh_lower_bound", "settlement_date"],
        ascending=[True, False, True],
    ).head(top_days)
    prepared["support_case_day_rank"] = range(1, len(prepared) + 1)
    return prepared[columns + ["support_case_day_rank"]].reset_index(drop=True)


def _select_support_families(
    publication_anomaly_family_daily: pd.DataFrame,
    selected_days: pd.DataFrame,
    top_families_per_day: int,
) -> pd.DataFrame:
    columns = [
        "settlement_date",
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "next_action",
        "publication_anomaly_next_action",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "publication_anomaly_row_count",
        "publication_anomaly_distinct_bmu_count",
        "publication_anomaly_candidate_mwh_lower_bound",
        "publication_anomaly_negative_bid_mwh_lower_bound",
        "publication_anomaly_sentinel_mwh_lower_bound",
        "publication_anomaly_dynamic_limit_mwh_lower_bound",
        "publication_anomaly_other_mwh_lower_bound",
        "accepted_down_delta_mwh_lower_bound",
        "dispatch_down_mwh_lower_bound",
        "lost_energy_mwh",
        "publication_anomaly_share_of_day_total",
        "publication_anomaly_share_of_remaining_qa_shortfall",
        "publication_anomaly_dominant_state",
        "day_family_rank_by_publication_anomaly",
        "family_publication_anomaly_next_action",
        "support_case_day_rank",
    ]
    if publication_anomaly_family_daily.empty or selected_days.empty:
        return pd.DataFrame(columns=columns + ["support_case_family_rank"])

    selected_dates = selected_days["settlement_date"].astype(str).tolist()
    prepared = publication_anomaly_family_daily.copy()
    prepared = prepared[prepared["settlement_date"].astype(str).isin(selected_dates)].copy()
    next_action = _ensure_string_column(prepared, "family_publication_anomaly_next_action")
    prepared = prepared[next_action.str.startswith("support_query_")].copy()
    if prepared.empty:
        return pd.DataFrame(columns=columns + ["support_case_family_rank"])

    prepared = prepared.merge(
        selected_days[["settlement_date", "support_case_day_rank"]],
        on="settlement_date",
        how="inner",
    )
    prepared["day_family_rank_by_publication_anomaly"] = pd.to_numeric(
        prepared["day_family_rank_by_publication_anomaly"], errors="coerce"
    )
    prepared["publication_anomaly_candidate_mwh_lower_bound"] = pd.to_numeric(
        prepared["publication_anomaly_candidate_mwh_lower_bound"], errors="coerce"
    ).fillna(0.0)
    prepared = prepared.sort_values(
        [
            "support_case_day_rank",
            "day_family_rank_by_publication_anomaly",
            "publication_anomaly_candidate_mwh_lower_bound",
            "bmu_family_key",
        ],
        ascending=[True, True, False, True],
    )
    prepared = (
        prepared.groupby("settlement_date", as_index=False, group_keys=False)
        .head(top_families_per_day)
        .reset_index(drop=True)
    )
    prepared["support_case_family_rank"] = prepared["day_family_rank_by_publication_anomaly"].astype("Int64")
    return prepared[columns + ["support_case_family_rank"]].reset_index(drop=True)


def build_fact_support_case_daily(
    selected_support_days: pd.DataFrame,
    selected_support_families: pd.DataFrame,
    support_batch_id: str,
    status_mode: str,
    top_days: int,
    top_families_per_day: int,
    generated_at_utc: str,
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
        "next_action",
        "publication_anomaly_next_action",
        "publication_anomaly_priority_rank",
        "publication_anomaly_candidate_mwh_lower_bound",
        "publication_anomaly_dominant_state",
        "publication_anomaly_family_count",
        "publication_anomaly_row_count",
        "publication_anomaly_distinct_bmu_count",
        "publication_anomaly_share_of_remaining_qa_shortfall",
        "remaining_qa_shortfall_mwh",
        "support_question_code",
        "support_recommended_action",
        "selected_family_count",
    ]
    if selected_support_days.empty:
        return pd.DataFrame(columns=columns)

    daily = selected_support_days.copy()
    if selected_support_families.empty:
        family_counts = pd.DataFrame(columns=["settlement_date", "selected_family_count"])
    else:
        family_counts = (
            selected_support_families.groupby("settlement_date", as_index=False)
            .agg(selected_family_count=("bmu_family_key", "nunique"))
        )
    daily = daily.merge(family_counts, on="settlement_date", how="left")
    daily["selected_family_count"] = pd.to_numeric(daily["selected_family_count"], errors="coerce").fillna(0).astype("Int64")
    daily["support_batch_id"] = support_batch_id
    daily["support_generated_at_utc"] = generated_at_utc
    daily["support_status_mode"] = status_mode
    daily["support_top_days"] = top_days
    daily["support_top_families_per_day"] = top_families_per_day
    daily["support_question_code"] = _ensure_string_column(daily, "publication_anomaly_dominant_state").map(
        {
            "sentinel_bod_present": "query_bod_sentinel_and_missing_boalf",
            "negative_bid_without_boalf": "query_missing_boalf_with_negative_bid_and_physical_gap",
            "dynamic_limit_like_without_boalf": "query_dynamic_limit_change_without_boalf",
            "physical_without_boalf": "query_physical_gap_without_boalf",
        }
    ).fillna("no_support_case")
    daily["support_recommended_action"] = daily["support_question_code"].map(
        {
            "query_bod_sentinel_and_missing_boalf": "ask_elexon_about_suspect_bod_sentinel_and_missing_published_boalf",
            "query_missing_boalf_with_negative_bid_and_physical_gap": "ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf",
            "query_dynamic_limit_change_without_boalf": "ask_elexon_whether_dynamic_limit_changes_can_occur_without_published_boalf",
            "query_physical_gap_without_boalf": "ask_elexon_why_physical_gap_exists_without_published_boalf",
        }
    ).fillna("no_support_escalation")
    return daily[columns].sort_values(["support_case_day_rank", "settlement_date"]).reset_index(drop=True)


def build_fact_support_case_family_daily(
    selected_support_families: pd.DataFrame,
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    support_batch_id: str,
    status_mode: str,
    top_days: int,
    top_families_per_day: int,
    generated_at_utc: str,
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
        "qa_reconciliation_status",
        "recoverability_audit_state",
        "next_action",
        "publication_anomaly_next_action",
        "bmu_family_key",
        "bmu_family_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "mapping_status",
        "publication_anomaly_candidate_mwh_lower_bound",
        "publication_anomaly_share_of_day_total",
        "publication_anomaly_share_of_remaining_qa_shortfall",
        "publication_anomaly_dominant_state",
        "day_family_rank_by_publication_anomaly",
        "support_question_code",
        "support_recommended_action",
        "distinct_bmu_count",
        "half_hour_count",
        "published_boalf_absent_half_hour_count",
        "physical_without_boalf_half_hour_count",
        "physical_without_boalf_negative_bid_half_hour_count",
        "physical_without_boalf_sentinel_half_hour_count",
        "availability_like_dynamic_limit_half_hour_count",
        "physical_without_boalf_gap_mwh",
        "most_negative_bid_gbp_per_mwh",
    ]
    if selected_support_families.empty:
        return pd.DataFrame(columns=columns)

    audit_rows = []
    for row in selected_support_families.itertuples(index=False):
        audit = build_fact_family_publication_audit_daily(
            fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
            fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
            fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
            family_keys=[row.bmu_family_key],
            start_date=str(row.settlement_date),
            end_date=str(row.settlement_date),
        )
        if audit.empty:
            continue
        audit_rows.append(audit.iloc[[0]].copy())

    audit_frame = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    family = selected_support_families.copy()
    if not audit_frame.empty:
        audit_keep = [
            "settlement_date",
            "bmu_family_key",
            "support_question_code",
            "support_recommended_action",
            "distinct_bmu_count",
            "half_hour_count",
            "published_boalf_absent_half_hour_count",
            "physical_without_boalf_half_hour_count",
            "physical_without_boalf_negative_bid_half_hour_count",
            "physical_without_boalf_sentinel_half_hour_count",
            "availability_like_dynamic_limit_half_hour_count",
            "physical_without_boalf_gap_mwh",
            "most_negative_bid_gbp_per_mwh",
        ]
        family = family.merge(
            audit_frame[audit_keep],
            on=["settlement_date", "bmu_family_key"],
            how="left",
        )
    else:
        for column in [
            "support_question_code",
            "support_recommended_action",
            "distinct_bmu_count",
            "half_hour_count",
            "published_boalf_absent_half_hour_count",
            "physical_without_boalf_half_hour_count",
            "physical_without_boalf_negative_bid_half_hour_count",
            "physical_without_boalf_sentinel_half_hour_count",
            "availability_like_dynamic_limit_half_hour_count",
            "physical_without_boalf_gap_mwh",
            "most_negative_bid_gbp_per_mwh",
        ]:
            family[column] = pd.NA

    family["support_batch_id"] = support_batch_id
    family["support_generated_at_utc"] = generated_at_utc
    family["support_status_mode"] = status_mode
    family["support_top_days"] = top_days
    family["support_top_families_per_day"] = top_families_per_day
    family["support_case_family_key"] = (
        support_batch_id
        + ":"
        + family["settlement_date"].astype(str)
        + ":"
        + family["bmu_family_key"].astype(str)
    )
    return family[columns].sort_values(
        ["support_case_day_rank", "support_case_family_rank", "bmu_family_key"]
    ).reset_index(drop=True)


def build_fact_support_case_half_hourly(
    selected_support_families: pd.DataFrame,
    fact_bmu_physical_position_half_hourly: pd.DataFrame,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame,
    fact_dispatch_source_gap_family_daily: pd.DataFrame,
    support_batch_id: str,
    status_mode: str,
    top_days: int,
    top_families_per_day: int,
    generated_at_utc: str,
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
        "support_case_key",
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
        "support_question_code",
        "support_recommended_action",
        "publication_anomaly_dominant_state",
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
    if selected_support_families.empty:
        return pd.DataFrame(columns=columns)

    frames = []
    selected = selected_support_families.copy()
    for row in selected.itertuples(index=False):
        support_rows = build_fact_family_support_evidence_half_hourly(
            fact_bmu_physical_position_half_hourly=fact_bmu_physical_position_half_hourly,
            fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
            fact_dispatch_source_gap_family_daily=fact_dispatch_source_gap_family_daily,
            family_keys=[row.bmu_family_key],
            start_date=str(row.settlement_date),
            end_date=str(row.settlement_date),
        )
        if support_rows.empty:
            continue
        support_rows = support_rows.merge(
            selected[
                [
                    "settlement_date",
                    "bmu_family_key",
                    "publication_anomaly_dominant_state",
                    "support_case_day_rank",
                    "support_case_family_rank",
                ]
            ].drop_duplicates(),
            on=["settlement_date", "bmu_family_key"],
            how="left",
        )
        support_rows["support_case_family_key"] = (
            support_batch_id
            + ":"
            + support_rows["settlement_date"].astype(str)
            + ":"
            + support_rows["bmu_family_key"].astype(str)
        )
        support_rows["support_batch_id"] = support_batch_id
        support_rows["support_generated_at_utc"] = generated_at_utc
        support_rows["support_status_mode"] = status_mode
        support_rows["support_top_days"] = top_days
        support_rows["support_top_families_per_day"] = top_families_per_day
        frames.append(support_rows)

    if not frames:
        return pd.DataFrame(columns=columns)

    half_hourly = pd.concat(frames, ignore_index=True)
    return half_hourly[columns].sort_values(
        ["support_case_day_rank", "support_case_family_rank", "support_priority_rank", "settlement_period", "elexon_bm_unit"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in subset.itertuples(index=False, name=None):
        values = []
        for value in record:
            if value is None or pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_support_case_summary_markdown(
    fact_support_case_daily: pd.DataFrame,
    fact_support_case_family_daily: pd.DataFrame,
    fact_support_case_half_hourly: pd.DataFrame,
    source_db_path: str | Path,
    support_batch_id: str,
    support_status_mode: str,
    support_top_days: int,
    support_top_families_per_day: int,
    support_generated_at_utc: str,
    example_half_hour_limit: int = 10,
) -> str:
    lines: list[str] = [
        "# Support Case Summary",
        "",
        f"- Generated at UTC: `{support_generated_at_utc}`",
        f"- Source DB: `{Path(source_db_path).resolve()}`",
        f"- Batch ID: `{support_batch_id}`",
        f"- Status filter: `{support_status_mode}`",
        f"- Top days: `{support_top_days}`",
        f"- Top families per day: `{support_top_families_per_day}`",
        f"- Selected days: `{len(fact_support_case_daily)}`",
        f"- Selected family-days: `{len(fact_support_case_family_daily)}`",
        "",
    ]
    if fact_support_case_daily.empty:
        lines.append("No support-query publication anomaly cases were selected for this batch.")
        return "\n".join(lines) + "\n"

    family_frame = fact_support_case_family_daily.copy()
    half_hourly = fact_support_case_half_hourly.copy()
    for day in fact_support_case_daily.sort_values(["support_case_day_rank", "settlement_date"]).itertuples(index=False):
        lines.extend(
            [
                f"## Day {int(day.support_case_day_rank)}: {day.settlement_date}",
                "",
                f"- QA status: `{day.qa_reconciliation_status}`",
                f"- Recoverability state: `{day.recoverability_audit_state}`",
                f"- Dominant anomaly: `{day.publication_anomaly_dominant_state}`",
                f"- Anomaly MWh: `{float(day.publication_anomaly_candidate_mwh_lower_bound):.3f}`",
                f"- Remaining QA shortfall MWh: `{float(day.remaining_qa_shortfall_mwh):.3f}`",
                f"- Recommended support action: `{day.publication_anomaly_next_action}`",
                f"- Selected families: `{int(day.selected_family_count)}`",
                "",
            ]
        )
        day_families = family_frame[family_frame["settlement_date"].astype(str) == str(day.settlement_date)].copy()
        for family in day_families.sort_values(["support_case_family_rank", "bmu_family_key"]).itertuples(index=False):
            family_rows = half_hourly[
                (half_hourly["settlement_date"].astype(str) == str(family.settlement_date))
                & (half_hourly["bmu_family_key"].astype(str) == str(family.bmu_family_key))
            ].copy()
            bmu_list = ", ".join(sorted({str(value) for value in family_rows["elexon_bm_unit"].dropna().astype(str)}))
            zero_boalf_rows = int(pd.to_numeric(family_rows["published_boalf_absent_flag"], errors="coerce").fillna(0).sum())
            negative_bid_rows = int(pd.to_numeric(family_rows["negative_bid_available_flag"], errors="coerce").fillna(0).sum())
            sentinel_rows = int(pd.to_numeric(family_rows["sentinel_pair_available_flag"], errors="coerce").fillna(0).sum())
            dynamic_limit_rows = int((family_rows["publication_audit_state"].fillna("").astype(str) == "availability_like_dynamic_limit").sum())
            example_rows = family_rows.sort_values(
                ["physical_dispatch_down_gap_mwh", "settlement_period", "elexon_bm_unit"],
                ascending=[False, True, True],
            ).head(example_half_hour_limit)
            lines.extend(
                [
                    f"### Family {int(family.support_case_family_rank)}: {family.bmu_family_key} ({family.bmu_family_label})",
                    "",
                    f"- Anomaly type: `{family.publication_anomaly_dominant_state}`",
                    f"- Anomaly MWh: `{float(family.publication_anomaly_candidate_mwh_lower_bound):.3f}`",
                    f"- Region / Cluster: `{family.parent_region}` / `{family.cluster_label}`",
                    f"- Mapping status: `{family.mapping_status}`",
                    f"- BMUs: `{bmu_list}`",
                    f"- Zero-BOALF rows: `{zero_boalf_rows}`",
                    f"- Negative-bid rows: `{negative_bid_rows}`",
                    f"- Sentinel rows: `{sentinel_rows}`",
                    f"- Dynamic-limit-like rows: `{dynamic_limit_rows}`",
                    f"- Support question: `{family.support_question_code}`",
                    f"- Recommended action: `{family.support_recommended_action}`",
                    "",
                    "Example half-hours:",
                    "",
                    _markdown_table(
                        example_rows,
                        [
                            "settlement_period",
                            "interval_start_utc",
                            "elexon_bm_unit",
                            "publication_audit_state",
                            "physical_dispatch_down_gap_mwh",
                            "most_negative_bid_gbp_per_mwh",
                            "sentinel_pair_count",
                            "accepted_down_delta_mwh_lower_bound",
                        ],
                    ),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_support_loop_outputs(
    frames: Dict[str, pd.DataFrame],
    summary_markdown: str,
    output_dir: str | Path,
    support_batch_id: str,
) -> Dict[str, Path]:
    target_dir = Path(output_dir) / support_batch_id
    target_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for table_name in [
        SUPPORT_CASE_DAILY_TABLE,
        SUPPORT_CASE_FAMILY_TABLE,
        SUPPORT_CASE_HALF_HOURLY_TABLE,
    ]:
        path = target_dir / f"{table_name}.csv"
        frames[table_name].to_csv(path, index=False)
        written[table_name] = path
    summary_path = target_dir / SUPPORT_SUMMARY_FILENAME
    summary_path.write_text(summary_markdown, encoding="utf-8")
    written[SUPPORT_SUMMARY_FILENAME] = summary_path
    return written


def materialize_truth_store_support_loop(
    db_path: str | Path,
    status_mode: str = "fail_warn",
    top_days: int = 7,
    top_families_per_day: int = 5,
    output_dir: str | Path | None = None,
    generated_at_utc: str | None = None,
) -> tuple[str, Dict[str, pd.DataFrame], str, Dict[str, Path]]:
    if status_mode not in VALID_SUPPORT_STATUS_MODES:
        raise ValueError(f"unsupported support status mode '{status_mode}'")
    if top_days <= 0:
        raise ValueError("top_days must be positive")
    if top_families_per_day <= 0:
        raise ValueError("top_families_per_day must be positive")

    target_path = Path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"truth store does not exist: {target_path}")

    materialize_truth_store_source_focus(target_path)
    publication_anomaly_daily = _load_table(target_path, PUBLICATION_ANOMALY_DAILY_TABLE)
    publication_anomaly_family_daily = _load_table(target_path, PUBLICATION_ANOMALY_FAMILY_TABLE)
    truth_half_hourly = _load_table(target_path, "fact_bmu_curtailment_truth_half_hourly")
    physical_half_hourly = _load_table(target_path, "fact_bmu_physical_position_half_hourly")
    source_gap_family_daily = _load_table(target_path, SOURCE_GAP_FAMILY_TABLE)

    selected_days = _select_support_days(
        publication_anomaly_daily=publication_anomaly_daily,
        status_mode=status_mode,
        top_days=top_days,
    )
    selected_families = _select_support_families(
        publication_anomaly_family_daily=publication_anomaly_family_daily,
        selected_days=selected_days,
        top_families_per_day=top_families_per_day,
    )
    support_batch_id = _support_batch_id(
        status_mode=status_mode,
        top_days=top_days,
        top_families_per_day=top_families_per_day,
        selected_dates=selected_days["settlement_date"].astype(str).tolist() if not selected_days.empty else [],
    )
    generated_value = _resolve_generated_at_utc(generated_at_utc)

    fact_support_case_daily = build_fact_support_case_daily(
        selected_support_days=selected_days,
        selected_support_families=selected_families,
        support_batch_id=support_batch_id,
        status_mode=status_mode,
        top_days=top_days,
        top_families_per_day=top_families_per_day,
        generated_at_utc=generated_value,
    )
    fact_support_case_family_daily = build_fact_support_case_family_daily(
        selected_support_families=selected_families,
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        support_batch_id=support_batch_id,
        status_mode=status_mode,
        top_days=top_days,
        top_families_per_day=top_families_per_day,
        generated_at_utc=generated_value,
    )
    fact_support_case_half_hourly = build_fact_support_case_half_hourly(
        selected_support_families=selected_families,
        fact_bmu_physical_position_half_hourly=physical_half_hourly,
        fact_bmu_curtailment_truth_half_hourly=truth_half_hourly,
        fact_dispatch_source_gap_family_daily=source_gap_family_daily,
        support_batch_id=support_batch_id,
        status_mode=status_mode,
        top_days=top_days,
        top_families_per_day=top_families_per_day,
        generated_at_utc=generated_value,
    )
    frames = {
        SUPPORT_CASE_DAILY_TABLE: fact_support_case_daily,
        SUPPORT_CASE_FAMILY_TABLE: fact_support_case_family_daily,
        SUPPORT_CASE_HALF_HOURLY_TABLE: fact_support_case_half_hourly,
    }
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_CASE_DAILY_TABLE,
        frame=fact_support_case_daily,
        primary_keys=["support_batch_id", "settlement_date"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_CASE_FAMILY_TABLE,
        frame=fact_support_case_family_daily,
        primary_keys=["support_batch_id", "settlement_date", "bmu_family_key"],
    )
    upsert_frame_to_sqlite(
        db_path=target_path,
        table_name=SUPPORT_CASE_HALF_HOURLY_TABLE,
        frame=fact_support_case_half_hourly,
        primary_keys=["support_batch_id", "settlement_date", "settlement_period", "elexon_bm_unit"],
    )

    summary_markdown = build_support_case_summary_markdown(
        fact_support_case_daily=fact_support_case_daily,
        fact_support_case_family_daily=fact_support_case_family_daily,
        fact_support_case_half_hourly=fact_support_case_half_hourly,
        source_db_path=target_path,
        support_batch_id=support_batch_id,
        support_status_mode=status_mode,
        support_top_days=top_days,
        support_top_families_per_day=top_families_per_day,
        support_generated_at_utc=generated_value,
    )
    written_paths: Dict[str, Path] = {}
    if output_dir is not None:
        written_paths = write_support_loop_outputs(
            frames=frames,
            summary_markdown=summary_markdown,
            output_dir=output_dir,
            support_batch_id=support_batch_id,
        )
    return support_batch_id, frames, summary_markdown, written_paths
