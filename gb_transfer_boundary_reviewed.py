from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from asset_mapping import cluster_frame
from day_ahead_constraint_boundary import (
    DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE,
    build_fact_day_ahead_constraint_boundary_half_hourly,
)
from gb_topology import interconnector_hub_frame, reachability_frame
from gb_transfer_gate import CONFIDENCE_GATE_FACTOR, STATUS_GATE_FACTOR


GB_TRANSFER_BOUNDARY_REVIEWED_TABLE = "fact_gb_transfer_boundary_reviewed_hourly"
GB_TRANSFER_BOUNDARY_SOURCE_PROVIDER = "neso"
GB_TRANSFER_BOUNDARY_SOURCE_FAMILY = "day_ahead_constraint_boundary"
GB_TRANSFER_BOUNDARY_EVIDENCE_TIER = "reviewed_internal_constraint_boundary"
GB_TRANSFER_BOUNDARY_REVIEW_STATE = "accepted_reviewed_tier"
GB_TRANSFER_BOUNDARY_POLICY_ACTION = "allow_boundary_day_ahead_gate"


@dataclass(frozen=True)
class BoundaryScopeRule:
    boundary_key: str
    rule_key: str
    parent_regions: tuple[str, ...] = ()
    cluster_keys: tuple[str, ...] = ()
    hub_keys: tuple[str, ...] = ()
    transfer_requirements: tuple[str, ...] = ()
    note: str = ""


BOUNDARY_SCOPE_RULES: tuple[BoundaryScopeRule, ...] = (
    BoundaryScopeRule(
        boundary_key="FLOWSTH",
        rule_key="england_east_to_south_export_corridor",
        cluster_keys=("east_anglia_offshore", "humber_offshore", "dogger_hornsea_offshore"),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=(
            "east_coast_transfer",
            "east_to_south_transfer",
            "south_east_bias",
            "south_to_south_coast_transfer",
        ),
        note="NESO FLOWSTH day-ahead boundary used as a first-pass east and south export corridor cap for east-facing England clusters.",
    ),
    BoundaryScopeRule(
        boundary_key="SCOTEX",
        rule_key="scotland_to_south_export_corridor",
        parent_regions=("Scotland",),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=("north_to_south_transfer",),
        note="NESO SCOTEX day-ahead boundary used as a first-pass Scotland-to-south export corridor cap.",
    ),
    BoundaryScopeRule(
        boundary_key="NKILGRMO",
        rule_key="scotland_to_south_export_corridor",
        parent_regions=("Scotland",),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=("north_to_south_transfer",),
        note="NESO NKILGRMO (B5 family) day-ahead boundary used as a first-pass Scotland-to-south export corridor cap.",
    ),
    BoundaryScopeRule(
        boundary_key="HARSPNBLY",
        rule_key="scotland_to_south_export_corridor",
        parent_regions=("Scotland",),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=("north_to_south_transfer",),
        note="NESO HARSPNBLY (B6a family) day-ahead boundary used as a first-pass Scotland-to-south export corridor cap.",
    ),
    BoundaryScopeRule(
        boundary_key="SSE-SP2",
        rule_key="scotland_to_south_export_corridor",
        parent_regions=("Scotland",),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=("north_to_south_transfer",),
        note="NESO SSE-SP2 day-ahead boundary used as a first-pass Scotland-to-south export corridor cap.",
    ),
    BoundaryScopeRule(
        boundary_key="SSEN-S",
        rule_key="scotland_to_south_export_corridor",
        parent_regions=("Scotland",),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=("north_to_south_transfer",),
        note="NESO SSEN-S day-ahead boundary used as a first-pass Scotland-to-south export corridor cap.",
    ),
    BoundaryScopeRule(
        boundary_key="SSHARN3",
        rule_key="scotland_to_south_export_corridor",
        parent_regions=("Scotland",),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=("north_to_south_transfer",),
        note="NESO SSHARN3 day-ahead boundary used as a first-pass Scotland-to-south export corridor cap.",
    ),
    BoundaryScopeRule(
        boundary_key="SEIMPPR23",
        rule_key="south_east_england_export_corridor",
        cluster_keys=("east_anglia_offshore", "humber_offshore", "dogger_hornsea_offshore"),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=(
            "east_coast_bias",
            "east_coast_transfer",
            "south_east_bias",
            "east_to_south_transfer",
            "south_to_south_coast_transfer",
        ),
        note="NESO SEIMPPR23 (LE1 South East England family) day-ahead boundary used as a first-pass south-east export corridor cap for east-facing England routes into BritNed and the France-facing hubs.",
    ),
    BoundaryScopeRule(
        boundary_key="GM+SNOW5A",
        rule_key="northern_transfer_complementary_corridor",
        parent_regions=("Scotland",),
        hub_keys=("britned", "ifa", "ifa2", "eleclink"),
        transfer_requirements=("north_to_south_transfer",),
        note="NESO GM+SNOW5A acts as a complementary northern transfer boundary to B7 and is influenced by Western Link loading, so it is used as a first-pass additional Scotland-to-south export corridor cap.",
    ),
)


def _empty_boundary_reviewed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start_utc",
            "interval_end_utc",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "hub_key",
            "hub_label",
            "scope_key",
            "scope_granularity",
            "reviewed_scope",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_gate_state",
            "reviewed_capacity_limit_mw",
            "reviewed_gate_fraction",
            "source_provider",
            "source_family",
            "source_key",
            "source_label",
            "source_document_title",
            "source_document_url",
            "source_reference",
            "source_published_utc",
            "source_published_date",
            "review_group_key",
            "source_revision_rank",
            "review_note",
            "reviewed_source_count",
            "boundary_key",
            "boundary_label",
            "boundary_state",
            "boundary_limit_mw",
            "boundary_flow_mw",
            "boundary_remaining_headroom_mw",
            "boundary_utilization_ratio",
            "structural_gate_mw_proxy",
        ]
    )


def _scope_candidates() -> pd.DataFrame:
    reachability = reachability_frame().copy()
    if reachability.empty:
        return pd.DataFrame()
    reachability = reachability[~reachability["status"].eq("upstream_dependency")].copy()
    cluster_lookup = cluster_frame()[["cluster_key", "cluster_label", "parent_region", "approx_capacity_mw"]].drop_duplicates(
        "cluster_key"
    )
    reachability = reachability.merge(cluster_lookup, on=["cluster_key", "cluster_label", "parent_region"], how="left")
    if "hub_label" not in reachability.columns:
        hub_lookup = interconnector_hub_frame()[["key", "label"]].rename(columns={"key": "hub_key", "label": "hub_label"})
        reachability = reachability.merge(hub_lookup, on="hub_key", how="left")
    reachability["status_gate_factor"] = reachability["status"].map(STATUS_GATE_FACTOR).fillna(0.0)
    reachability["confidence_gate_factor"] = reachability["confidence"].map(CONFIDENCE_GATE_FACTOR).fillna(0.0)
    reachability["structural_gate_mw_proxy"] = (
        pd.to_numeric(reachability["approx_capacity_mw"], errors="coerce")
        * reachability["status_gate_factor"].astype(float)
        * reachability["confidence_gate_factor"].astype(float)
    )
    return reachability[
        [
            "cluster_key",
            "cluster_label",
            "parent_region",
            "hub_key",
            "hub_label",
            "transfer_requirement",
            "status",
            "confidence",
            "approx_capacity_mw",
            "structural_gate_mw_proxy",
        ]
    ].copy()


def _hourly_boundary_frame(boundary_frame: pd.DataFrame) -> pd.DataFrame:
    if boundary_frame is None or boundary_frame.empty:
        return pd.DataFrame()
    frame = boundary_frame.copy()
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["interval_end_utc"] = pd.to_datetime(frame["interval_end_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["interval_start_utc", "boundary_key"]).copy()
    frame["hour_start_utc"] = frame["interval_start_utc"].dt.floor("h")
    frame["hour_end_utc"] = frame["hour_start_utc"] + pd.Timedelta(hours=1)

    state_rank = {
        "constraint_boundary_zero_limit": 0,
        "constraint_boundary_at_or_above_limit": 0,
        "constraint_boundary_tight": 1,
        "constraint_boundary_available": 2,
    }
    frame["_state_rank"] = frame["boundary_state"].map(state_rank).fillna(9)
    frame["_cap_mw"] = pd.to_numeric(frame["remaining_headroom_mw"], errors="coerce").clip(lower=0)
    frame["_limit_mw"] = pd.to_numeric(frame["limit_mw"], errors="coerce")
    frame["_flow_mw"] = pd.to_numeric(frame["flow_mw"], errors="coerce")
    frame["_utilization_ratio"] = pd.to_numeric(frame["utilization_ratio"], errors="coerce")
    frame = frame.sort_values(
        ["hour_start_utc", "boundary_key", "_state_rank", "_cap_mw", "_utilization_ratio"],
        ascending=[True, True, True, True, False],
    )

    def _first_value(values: pd.Series):
        return values.iloc[0] if not values.empty else pd.NA

    aggregated = frame.groupby(["hour_start_utc", "hour_end_utc", "boundary_key"], as_index=False).agg(
        boundary_label=("boundary_label", _first_value),
        boundary_state=("boundary_state", _first_value),
        boundary_limit_mw=("_limit_mw", "min"),
        boundary_flow_mw=("_flow_mw", "max"),
        boundary_remaining_headroom_mw=("_cap_mw", "min"),
        boundary_utilization_ratio=("_utilization_ratio", "max"),
        source_document_url=("source_document_url", _first_value),
        source_key=("source_key", _first_value),
        source_label=("source_label", _first_value),
    )
    return aggregated.rename(columns={"hour_start_utc": "interval_start_utc", "hour_end_utc": "interval_end_utc"})


def _match_scope_rows(rule: BoundaryScopeRule, scope_rows: pd.DataFrame) -> pd.DataFrame:
    matched = scope_rows.copy()
    if rule.parent_regions:
        matched = matched[matched["parent_region"].isin(rule.parent_regions)].copy()
    if rule.cluster_keys:
        matched = matched[matched["cluster_key"].isin(rule.cluster_keys)].copy()
    if rule.hub_keys:
        matched = matched[matched["hub_key"].isin(rule.hub_keys)].copy()
    if rule.transfer_requirements:
        matched = matched[matched["transfer_requirement"].isin(rule.transfer_requirements)].copy()
    return matched


def _boundary_gate_state(boundary_state: object) -> str:
    normalized = "" if pd.isna(boundary_state) else str(boundary_state)
    if normalized in {"constraint_boundary_zero_limit", "constraint_boundary_at_or_above_limit"}:
        return "blocked_reviewed_boundary"
    if normalized == "constraint_boundary_tight":
        return "reviewed_boundary_tight"
    return "reviewed_boundary_cap"


def _choose_binding_boundary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_boundary_reviewed_frame()
    sort_frame = frame.copy()
    sort_frame["_blocked_rank"] = sort_frame["reviewed_gate_state"].fillna("").astype(str).str.startswith("blocked_").map(
        {True: 0, False: 1}
    )
    sort_frame["_limit_sort"] = pd.to_numeric(sort_frame["reviewed_capacity_limit_mw"], errors="coerce").fillna(np.inf)
    sort_frame["_tier_sort"] = 0
    sort_frame = sort_frame.sort_values(
        ["interval_start_utc", "cluster_key", "hub_key", "_blocked_rank", "_limit_sort", "boundary_utilization_ratio"],
        ascending=[True, True, True, True, True, False],
    )
    binding = sort_frame.drop_duplicates(["interval_start_utc", "cluster_key", "hub_key"], keep="first").copy()
    binding["reviewed_source_count"] = (
        sort_frame.groupby(["interval_start_utc", "cluster_key", "hub_key"])["boundary_key"].transform("size")
        .loc[binding.index]
        .astype(int)
    )
    return binding[_empty_boundary_reviewed_frame().columns].sort_values(
        ["interval_start_utc", "cluster_key", "hub_key"]
    ).reset_index(drop=True)


def build_fact_gb_transfer_boundary_reviewed_hourly(
    start_date: dt.date,
    end_date: dt.date,
    day_ahead_constraint_boundary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    boundary_frame = day_ahead_constraint_boundary
    if boundary_frame is None:
        boundary_frame = build_fact_day_ahead_constraint_boundary_half_hourly(start_date=start_date, end_date=end_date)
    hourly_boundary = _hourly_boundary_frame(boundary_frame)
    if hourly_boundary.empty:
        return _empty_boundary_reviewed_frame()

    scope_rows = _scope_candidates()
    if scope_rows.empty:
        return _empty_boundary_reviewed_frame()

    candidate_rows = []
    for rule in BOUNDARY_SCOPE_RULES:
        boundary_rows = hourly_boundary[hourly_boundary["boundary_key"].eq(rule.boundary_key)].copy()
        if boundary_rows.empty:
            continue
        matched_scope = _match_scope_rows(rule, scope_rows)
        if matched_scope.empty:
            continue
        expanded = boundary_rows.assign(_join_key=1).merge(matched_scope.assign(_join_key=1), on="_join_key", how="inner").drop(
            columns="_join_key"
        )
        expanded["scope_key"] = expanded["cluster_key"]
        expanded["scope_granularity"] = "cluster_hub"
        expanded["reviewed_scope"] = "gb_internal_constraint_boundary"
        expanded["review_state"] = GB_TRANSFER_BOUNDARY_REVIEW_STATE
        expanded["reviewed_evidence_tier"] = GB_TRANSFER_BOUNDARY_EVIDENCE_TIER
        expanded["reviewed_tier_accepted_flag"] = True
        expanded["capacity_policy_action"] = GB_TRANSFER_BOUNDARY_POLICY_ACTION
        expanded["reviewed_gate_state"] = expanded["boundary_state"].map(_boundary_gate_state)
        expanded["reviewed_capacity_limit_mw"] = pd.concat(
            [
                pd.to_numeric(expanded["boundary_remaining_headroom_mw"], errors="coerce").clip(lower=0),
                pd.to_numeric(expanded["structural_gate_mw_proxy"], errors="coerce"),
            ],
            axis=1,
        ).min(axis=1)
        expanded["reviewed_gate_fraction"] = np.where(
            pd.to_numeric(expanded["approx_capacity_mw"], errors="coerce").gt(0),
            pd.to_numeric(expanded["reviewed_capacity_limit_mw"], errors="coerce")
            / pd.to_numeric(expanded["approx_capacity_mw"], errors="coerce"),
            np.nan,
        )
        tightening_mask = (
            expanded["reviewed_gate_state"].eq("blocked_reviewed_boundary")
            | pd.to_numeric(expanded["reviewed_capacity_limit_mw"], errors="coerce").lt(
                pd.to_numeric(expanded["structural_gate_mw_proxy"], errors="coerce")
            )
        )
        expanded = expanded[tightening_mask].copy()
        if expanded.empty:
            continue
        expanded["source_provider"] = GB_TRANSFER_BOUNDARY_SOURCE_PROVIDER
        expanded["source_family"] = GB_TRANSFER_BOUNDARY_SOURCE_FAMILY
        expanded["source_key"] = expanded["boundary_key"].map(lambda value: f"{DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE}:{value}")
        expanded["source_label"] = expanded["boundary_label"].map(lambda value: f"NESO day-ahead boundary {value}")
        expanded["source_document_title"] = expanded["boundary_label"].map(
            lambda value: f"NESO Day Ahead Constraint Flows and Limits - {value}"
        )
        expanded["source_reference"] = expanded["boundary_key"]
        expanded["source_published_utc"] = pd.NaT
        expanded["source_published_date"] = pd.NaT
        expanded["review_group_key"] = expanded.apply(
            lambda row: f"{row['boundary_key']}|{row['cluster_key']}|{row['hub_key']}|{row['interval_start_utc']}",
            axis=1,
        )
        expanded["source_revision_rank"] = 1
        expanded["review_note"] = rule.note
        candidate_rows.append(expanded)

    if not candidate_rows:
        return _empty_boundary_reviewed_frame()

    combined = pd.concat(candidate_rows, ignore_index=True)
    return _choose_binding_boundary(combined)


def materialize_gb_transfer_boundary_reviewed_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    day_ahead_constraint_boundary: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_gb_transfer_boundary_reviewed_hourly(
        start_date=start_date,
        end_date=end_date,
        day_ahead_constraint_boundary=day_ahead_constraint_boundary,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{GB_TRANSFER_BOUNDARY_REVIEWED_TABLE}.csv", index=False)
    return {GB_TRANSFER_BOUNDARY_REVIEWED_TABLE: fact}
