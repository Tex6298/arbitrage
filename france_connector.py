from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from gb_topology import INTERCONNECTOR_HUBS
from france_connector_reviewed import build_fact_france_connector_reviewed_hourly
from interconnector_capacity import (
    INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE,
    INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE,
    build_fact_interconnector_capacity_hourly,
    build_interconnector_capacity_review_policy,
    build_interconnector_capacity_reviewed_hourly,
    build_interconnector_capacity_source_audit,
)
from interconnector_flow import build_fact_interconnector_flow_hourly
from network_overlay import build_border_network_overlay


DIM_INTERCONNECTOR_CABLE_TABLE = "dim_interconnector_cable"
FRANCE_CONNECTOR_TABLE = "fact_france_connector_hourly"
FRANCE_BORDER_KEY = "GB-FR"
FRANCE_NEIGHBOR_DOMAIN_KEY = "FR"
FRANCE_CONNECTOR_SOURCE_KEY = "gb_france_connector_proxy"
FRANCE_CONNECTOR_SOURCE_LABEL = "GB-France connector-layer proxy"


@dataclass(frozen=True)
class FranceConnectorSpec:
    connector_key: str
    connector_label: str
    hub_key: str
    operator_name: str
    commercial_model: str
    service_state: str
    commissioning_year: int
    nominal_capacity_mw: float
    evidence_source_label: str
    evidence_source_url: str


FRANCE_CONNECTOR_SPECS: Tuple[FranceConnectorSpec, ...] = (
    FranceConnectorSpec(
        connector_key="ifa",
        connector_label="IFA",
        hub_key="ifa",
        operator_name="National Grid Interconnectors / RTE",
        commercial_model="joint_venture_regulated",
        service_state="current",
        commissioning_year=1986,
        nominal_capacity_mw=2000.0,
        evidence_source_label="IFA official interconnector site",
        evidence_source_url="https://www.ifa1interconnector.com/about-us/",
    ),
    FranceConnectorSpec(
        connector_key="ifa2",
        connector_label="IFA2",
        hub_key="ifa2",
        operator_name="National Grid IFA2 Limited / RTE",
        commercial_model="joint_venture_regulated",
        service_state="current",
        commissioning_year=2021,
        nominal_capacity_mw=1000.0,
        evidence_source_label="National Grid IFA2 project page",
        evidence_source_url="https://www.nationalgrid.com/national-grid-ventures/interconnectors-uk/ifa2-interconnector-uk-france",
    ),
    FranceConnectorSpec(
        connector_key="eleclink",
        connector_label="ElecLink",
        hub_key="eleclink",
        operator_name="ElecLink Limited / Getlink",
        commercial_model="merchant",
        service_state="current",
        commissioning_year=2022,
        nominal_capacity_mw=1000.0,
        evidence_source_label="ElecLink official website",
        evidence_source_url="https://www.eleclink.co.uk/about-us/",
    ),
)


def interconnector_cable_frame() -> pd.DataFrame:
    total_france_capacity = float(sum(spec.nominal_capacity_mw for spec in FRANCE_CONNECTOR_SPECS))
    rows = []
    for spec in FRANCE_CONNECTOR_SPECS:
        hub = INTERCONNECTOR_HUBS[spec.hub_key]
        rows.append(
            {
                "connector_key": spec.connector_key,
                "connector_label": spec.connector_label,
                "connector_scope": "france_first_pass",
                "border_key": FRANCE_BORDER_KEY,
                "border_label": "Great Britain to France cable set",
                "target_zone": "FR",
                "neighbor_domain_key": FRANCE_NEIGHBOR_DOMAIN_KEY,
                "hub_key": spec.hub_key,
                "hub_label": hub.label,
                "landing_bias": hub.landing_bias,
                "current_route_fit": hub.current_route_fit,
                "operator_name": spec.operator_name,
                "commercial_model": spec.commercial_model,
                "service_state": spec.service_state,
                "commissioning_year": spec.commissioning_year,
                "nominal_capacity_mw": spec.nominal_capacity_mw,
                "nominal_capacity_share_of_border": spec.nominal_capacity_mw / total_france_capacity,
                "evidence_source_label": spec.evidence_source_label,
                "evidence_source_url": spec.evidence_source_url,
                "target_is_proxy": False,
            }
        )
    return pd.DataFrame(rows).sort_values("connector_key").reset_index(drop=True)


def _empty_france_connector_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "source_key",
            "source_label",
            "target_is_proxy",
            "connector_scope",
            "connector_assignment_method",
            "connector_flow_assignment_method",
            "connector_capacity_assignment_method",
            "border_key",
            "border_label",
            "target_zone",
            "neighbor_domain_key",
            "connector_key",
            "connector_label",
            "hub_key",
            "hub_label",
            "landing_bias",
            "current_route_fit",
            "operator_name",
            "commercial_model",
            "service_state",
            "commissioning_year",
            "nominal_capacity_mw",
            "nominal_capacity_share_of_border",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "border_flow_mw",
            "border_flow_state",
            "border_offered_capacity_mw",
            "border_capacity_state",
            "border_headroom_proxy_mw",
            "reviewed_border_offered_capacity_mw",
            "reviewed_border_capacity_state",
            "reviewed_border_headroom_proxy_mw",
            "reviewed_publication_state",
            "reviewed_publication_evidence_tier",
            "reviewed_publication_tier_accepted_flag",
            "reviewed_publication_capacity_policy_action",
            "reviewed_publication_capacity_limit_mw",
            "reviewed_publication_source_provider",
            "reviewed_publication_source_family",
            "reviewed_publication_source_key",
            "reviewed_publication_source_label",
            "reviewed_publication_source_document_title",
            "reviewed_publication_source_document_url",
            "reviewed_publication_source_reference",
            "reviewed_publication_source_published_date",
            "reviewed_publication_source_count",
            "operator_source_provider",
            "operator_availability_state",
            "operator_capacity_evidence_tier",
            "operator_capacity_limit_mw",
            "operator_source_gap_reason",
            "connector_signed_flow_from_gb_mw_proxy",
            "connector_export_flow_mw_proxy",
            "connector_offered_capacity_mw_proxy",
            "connector_headroom_proxy_mw",
            "connector_capacity_evidence_tier",
            "connector_gate_state",
            "connector_gate_reason",
        ]
    )


def _hourly_window_frame(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    start_local = pd.Timestamp(start_date, tz="Europe/London")
    end_local = pd.Timestamp(end_date + dt.timedelta(days=1), tz="Europe/London")
    interval_start_local = pd.date_range(start=start_local, end=end_local, freq="h", inclusive="left")
    frame = pd.DataFrame({"interval_start_local": interval_start_local})
    frame["interval_end_local"] = frame["interval_start_local"] + pd.Timedelta(hours=1)
    frame["interval_start_utc"] = frame["interval_start_local"].dt.tz_convert("UTC")
    frame["interval_end_utc"] = frame["interval_end_local"].dt.tz_convert("UTC")
    frame["date"] = frame["interval_start_local"].dt.date
    return frame


def _review_policy_defaults() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "border_key": FRANCE_BORDER_KEY,
                "direction_key": "gb_to_neighbor",
                "review_state": "capacity_unknown_default",
                "reviewed_evidence_tier": "none",
                "reviewed_tier_accepted_flag": False,
                "capacity_policy_action": "keep_capacity_unknown",
            }
        ]
    )


def build_fact_france_connector_hourly(
    start_date: dt.date,
    end_date: dt.date,
    interconnector_flow: pd.DataFrame | None = None,
    interconnector_capacity: pd.DataFrame | None = None,
    interconnector_capacity_review_policy: pd.DataFrame | None = None,
    interconnector_capacity_reviewed: pd.DataFrame | None = None,
    france_connector_reviewed_period: pd.DataFrame | None = None,
    france_connector_availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    connectors = interconnector_cable_frame()
    if connectors.empty:
        return _empty_france_connector_frame()

    hours = _hourly_window_frame(start_date, end_date)
    base = hours.assign(_join_key=1).merge(connectors.assign(_join_key=1), on="_join_key", how="inner").drop(columns="_join_key")

    first_pass_overlay = build_border_network_overlay(interconnector_flow, interconnector_capacity)
    first_pass_overlay = first_pass_overlay[first_pass_overlay["border_key"] == FRANCE_BORDER_KEY].copy()

    reviewed_overlay = build_border_network_overlay(interconnector_flow, interconnector_capacity_reviewed)
    reviewed_overlay = reviewed_overlay[reviewed_overlay["border_key"] == FRANCE_BORDER_KEY].copy()
    if not reviewed_overlay.empty:
        reviewed_overlay = reviewed_overlay.rename(
            columns={
                "border_offered_capacity_mw": "reviewed_border_offered_capacity_mw",
                "border_capacity_published_flag": "reviewed_border_capacity_published_flag",
                "positive_export_flow_mw": "reviewed_positive_export_flow_mw",
                "border_headroom_proxy_mw": "reviewed_border_headroom_proxy_mw",
                "border_flow_state": "reviewed_border_flow_state",
                "border_capacity_state": "reviewed_border_capacity_state",
                "border_gate_state": "reviewed_border_gate_state",
            }
        )
        reviewed_overlay = reviewed_overlay[
            [
                "interval_start_utc",
                "border_key",
                "reviewed_border_offered_capacity_mw",
                "reviewed_border_capacity_published_flag",
                "reviewed_positive_export_flow_mw",
                "reviewed_border_headroom_proxy_mw",
                "reviewed_border_flow_state",
                "reviewed_border_capacity_state",
                "reviewed_border_gate_state",
            ]
        ]

    connector_reviewed = build_fact_france_connector_reviewed_hourly(
        start_date=start_date,
        end_date=end_date,
        reviewed_period=france_connector_reviewed_period,
    )
    if not connector_reviewed.empty:
        connector_reviewed = connector_reviewed.rename(
            columns={
                "reviewed_evidence_tier": "reviewed_publication_evidence_tier",
                "reviewed_tier_accepted_flag": "reviewed_publication_tier_accepted_flag",
                "capacity_policy_action": "reviewed_publication_capacity_policy_action",
                "reviewed_capacity_limit_mw": "reviewed_publication_capacity_limit_mw",
                "source_provider": "reviewed_publication_source_provider",
                "source_family": "reviewed_publication_source_family",
                "source_key": "reviewed_publication_source_key",
                "source_label": "reviewed_publication_source_label",
                "source_document_title": "reviewed_publication_source_document_title",
                "source_document_url": "reviewed_publication_source_document_url",
                "source_reference": "reviewed_publication_source_reference",
                "source_published_date": "reviewed_publication_source_published_date",
                "reviewed_source_count": "reviewed_publication_source_count",
            }
        )
        connector_reviewed = connector_reviewed[
            [
                "interval_start_utc",
                "connector_key",
                "direction_key",
                "reviewed_publication_state",
                "reviewed_publication_evidence_tier",
                "reviewed_publication_tier_accepted_flag",
                "reviewed_publication_capacity_policy_action",
                "reviewed_publication_capacity_limit_mw",
                "reviewed_available_capacity_mw",
                "reviewed_unavailable_capacity_mw",
                "reviewed_publication_source_provider",
                "reviewed_publication_source_family",
                "reviewed_publication_source_key",
                "reviewed_publication_source_label",
                "reviewed_publication_source_document_title",
                "reviewed_publication_source_document_url",
                "reviewed_publication_source_reference",
                "reviewed_publication_source_published_date",
                "reviewed_publication_source_count",
                "review_note",
            ]
        ]
        connector_reviewed = connector_reviewed[connector_reviewed["direction_key"].isin(["gb_to_neighbor", "both"])].copy()

    review_policy = (
        interconnector_capacity_review_policy.copy()
        if interconnector_capacity_review_policy is not None and not interconnector_capacity_review_policy.empty
        else _review_policy_defaults()
    )
    review_policy = review_policy[
        (review_policy["border_key"] == FRANCE_BORDER_KEY) & (review_policy["direction_key"] == "gb_to_neighbor")
    ].copy()
    if review_policy.empty:
        review_policy = _review_policy_defaults()
    review_policy = review_policy[
        [
            "border_key",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
        ]
    ].drop_duplicates(subset=["border_key"])

    if not first_pass_overlay.empty:
        base = base.merge(first_pass_overlay, on=["interval_start_utc", "border_key"], how="left")
    else:
        base["border_flow_mw"] = np.nan
        base["border_flow_published_flag"] = False
        base["border_offered_capacity_mw"] = np.nan
        base["border_capacity_published_flag"] = False
        base["positive_export_flow_mw"] = np.nan
        base["border_headroom_proxy_mw"] = np.nan
        base["border_flow_state"] = "flow_unknown"
        base["border_capacity_state"] = "capacity_unknown"
        base["border_gate_state"] = "capacity_unknown"

    if not reviewed_overlay.empty:
        base = base.merge(reviewed_overlay, on=["interval_start_utc", "border_key"], how="left")
    else:
        base["reviewed_border_offered_capacity_mw"] = np.nan
        base["reviewed_border_capacity_published_flag"] = False
        base["reviewed_positive_export_flow_mw"] = np.nan
        base["reviewed_border_headroom_proxy_mw"] = np.nan
        base["reviewed_border_flow_state"] = "flow_unknown"
        base["reviewed_border_capacity_state"] = "capacity_unknown"
        base["reviewed_border_gate_state"] = "capacity_unknown"

    if not connector_reviewed.empty:
        base = base.merge(
            connector_reviewed.drop(columns=["direction_key"], errors="ignore"),
            on=["interval_start_utc", "connector_key"],
            how="left",
        )
    else:
        base["reviewed_publication_state"] = pd.NA
        base["reviewed_publication_evidence_tier"] = pd.NA
        base["reviewed_publication_tier_accepted_flag"] = False
        base["reviewed_publication_capacity_policy_action"] = pd.NA
        base["reviewed_publication_capacity_limit_mw"] = np.nan
        base["reviewed_available_capacity_mw"] = np.nan
        base["reviewed_unavailable_capacity_mw"] = np.nan
        base["reviewed_publication_source_provider"] = pd.NA
        base["reviewed_publication_source_family"] = pd.NA
        base["reviewed_publication_source_key"] = pd.NA
        base["reviewed_publication_source_label"] = pd.NA
        base["reviewed_publication_source_document_title"] = pd.NA
        base["reviewed_publication_source_document_url"] = pd.NA
        base["reviewed_publication_source_reference"] = pd.NA
        base["reviewed_publication_source_published_date"] = pd.NaT
        base["reviewed_publication_source_count"] = np.nan
        base["review_note"] = pd.NA

    base["border_flow_published_flag"] = base["border_flow_published_flag"].where(
        base["border_flow_published_flag"].notna(),
        False,
    ).astype(bool)
    base["border_capacity_published_flag"] = base["border_capacity_published_flag"].where(
        base["border_capacity_published_flag"].notna(),
        False,
    ).astype(bool)
    base["border_flow_state"] = base["border_flow_state"].fillna("flow_unknown")
    base["border_capacity_state"] = base["border_capacity_state"].fillna("capacity_unknown")
    base["border_gate_state"] = base["border_gate_state"].fillna("capacity_unknown")
    base["reviewed_border_capacity_published_flag"] = base["reviewed_border_capacity_published_flag"].where(
        base["reviewed_border_capacity_published_flag"].notna(),
        False,
    ).astype(bool)
    base["reviewed_border_flow_state"] = base["reviewed_border_flow_state"].fillna("flow_unknown")
    base["reviewed_border_capacity_state"] = base["reviewed_border_capacity_state"].fillna("capacity_unknown")
    base["reviewed_border_gate_state"] = base["reviewed_border_gate_state"].fillna("capacity_unknown")
    base["reviewed_publication_tier_accepted_flag"] = base["reviewed_publication_tier_accepted_flag"].where(
        base["reviewed_publication_tier_accepted_flag"].notna(),
        False,
    ).astype(bool)
    base["reviewed_publication_state"] = base["reviewed_publication_state"].fillna(pd.NA)
    base["reviewed_publication_evidence_tier"] = base["reviewed_publication_evidence_tier"].fillna("none")
    base["reviewed_publication_capacity_policy_action"] = base["reviewed_publication_capacity_policy_action"].fillna(
        "keep_capacity_unknown"
    )

    base = base.merge(review_policy, on=["border_key"], how="left")
    base["review_state"] = base["review_state"].fillna("capacity_unknown_default")
    base["reviewed_evidence_tier"] = base["reviewed_evidence_tier"].fillna("none")
    base["reviewed_tier_accepted_flag"] = base["reviewed_tier_accepted_flag"].where(
        base["reviewed_tier_accepted_flag"].notna(),
        False,
    ).astype(bool)
    base["capacity_policy_action"] = base["capacity_policy_action"].fillna("keep_capacity_unknown")

    availability = france_connector_availability.copy() if france_connector_availability is not None else pd.DataFrame()
    if not availability.empty:
        availability["interval_start_utc"] = pd.to_datetime(availability["interval_start_utc"], utc=True, errors="coerce")
        availability = availability[
            [
                "interval_start_utc",
                "connector_key",
                "source_provider",
                "operator_availability_state",
                "operator_capacity_evidence_tier",
                "operator_capacity_limit_mw",
                "operator_source_gap_reason",
            ]
        ].copy()
        availability = availability.rename(columns={"source_provider": "operator_source_provider"})
        base = base.merge(availability, on=["interval_start_utc", "connector_key"], how="left")
    else:
        base["operator_source_provider"] = pd.NA
        base["operator_availability_state"] = pd.NA
        base["operator_capacity_evidence_tier"] = pd.NA
        base["operator_capacity_limit_mw"] = np.nan
        base["operator_source_gap_reason"] = pd.NA

    base["source_key"] = FRANCE_CONNECTOR_SOURCE_KEY
    base["source_label"] = FRANCE_CONNECTOR_SOURCE_LABEL
    base["target_is_proxy"] = True
    base["connector_scope"] = "gb_france_cable_proxy"
    base["connector_assignment_method"] = "hub_key_equals_connector_key"
    base["connector_flow_assignment_method"] = "capacity_weighted_border_flow_split_proxy"
    base["connector_capacity_assignment_method"] = "first_pass_then_reviewed_then_nominal_proxy"

    share = pd.to_numeric(base["nominal_capacity_share_of_border"], errors="coerce")
    signed_border_flow = pd.to_numeric(base["border_flow_mw"], errors="coerce")
    base["connector_signed_flow_from_gb_mw_proxy"] = signed_border_flow * share
    base["connector_export_flow_mw_proxy"] = pd.to_numeric(
        base["connector_signed_flow_from_gb_mw_proxy"], errors="coerce"
    ).clip(lower=0)

    first_pass_capacity_proxy = pd.to_numeric(base["border_offered_capacity_mw"], errors="coerce") * share
    first_pass_headroom_proxy = pd.to_numeric(base["border_headroom_proxy_mw"], errors="coerce") * share
    reviewed_capacity_proxy = pd.to_numeric(base["reviewed_border_offered_capacity_mw"], errors="coerce") * share
    reviewed_headroom_proxy = pd.to_numeric(base["reviewed_border_headroom_proxy_mw"], errors="coerce") * share

    nominal_capacity = pd.to_numeric(base["nominal_capacity_mw"], errors="coerce")
    nominal_headroom_proxy = nominal_capacity - pd.to_numeric(base["connector_export_flow_mw_proxy"], errors="coerce").fillna(0.0)
    nominal_headroom_proxy = nominal_headroom_proxy.clip(lower=0)

    reviewed_gate_state = base["reviewed_border_gate_state"].fillna("capacity_unknown")
    reviewed_limit_proxy = reviewed_headroom_proxy.where(
        ~reviewed_gate_state.eq("flow_unknown_capacity_published"),
        reviewed_capacity_proxy,
    )

    base["connector_offered_capacity_mw_proxy"] = nominal_capacity
    base["connector_headroom_proxy_mw"] = nominal_headroom_proxy
    base["connector_capacity_evidence_tier"] = "nominal_static"
    base["connector_gate_state"] = "nominal_capacity_only"
    base["connector_gate_reason"] = (
        "GB-FR published capacity is unavailable, so the cable layer falls back to nominal capacity with border-flow-weighted utilization."
    )

    first_pass_published = base["border_capacity_state"].eq("published_positive")
    first_pass_blocked_zero = base["border_gate_state"].eq("blocked_zero_offered_capacity")
    first_pass_blocked_headroom = base["border_gate_state"].eq("blocked_headroom_proxy")
    reviewed_available = base["reviewed_tier_accepted_flag"] & reviewed_gate_state.isin(
        ["pass", "flow_unknown_capacity_published"]
    )

    first_pass_limit_proxy = first_pass_headroom_proxy.where(
        ~base["border_gate_state"].eq("flow_unknown_capacity_published"),
        first_pass_capacity_proxy,
    )
    base.loc[first_pass_published, "connector_offered_capacity_mw_proxy"] = first_pass_capacity_proxy[first_pass_published]
    base.loc[first_pass_published, "connector_headroom_proxy_mw"] = first_pass_limit_proxy[first_pass_published].clip(lower=0)
    base.loc[first_pass_published, "connector_capacity_evidence_tier"] = "first_pass_border_split_proxy"
    base.loc[first_pass_published, "connector_gate_state"] = "first_pass_split_pass"
    base.loc[first_pass_published, "connector_gate_reason"] = (
        "Published GB-FR border capacity exists, so the France cable layer allocates border headroom by nominal cable share."
    )

    base.loc[first_pass_blocked_zero, "connector_offered_capacity_mw_proxy"] = 0.0
    base.loc[first_pass_blocked_zero, "connector_headroom_proxy_mw"] = 0.0
    base.loc[first_pass_blocked_zero, "connector_capacity_evidence_tier"] = "first_pass_border_split_proxy"
    base.loc[first_pass_blocked_zero, "connector_gate_state"] = "first_pass_split_blocked_zero_capacity"
    base.loc[first_pass_blocked_zero, "connector_gate_reason"] = (
        "Published GB-FR border capacity is zero or negative, so every France cable proxy is blocked for the hour."
    )

    base.loc[first_pass_blocked_headroom, "connector_offered_capacity_mw_proxy"] = first_pass_capacity_proxy[first_pass_blocked_headroom]
    base.loc[first_pass_blocked_headroom, "connector_headroom_proxy_mw"] = 0.0
    base.loc[first_pass_blocked_headroom, "connector_capacity_evidence_tier"] = "first_pass_border_split_proxy"
    base.loc[first_pass_blocked_headroom, "connector_gate_state"] = "first_pass_split_blocked_headroom"
    base.loc[first_pass_blocked_headroom, "connector_gate_reason"] = (
        "Observed GB-FR export flow already consumes the first-pass border headroom, so the France cable split is blocked."
    )

    reviewed_mask = ~first_pass_published & ~first_pass_blocked_zero & ~first_pass_blocked_headroom & reviewed_available
    base.loc[reviewed_mask, "connector_offered_capacity_mw_proxy"] = reviewed_capacity_proxy[reviewed_mask]
    base.loc[reviewed_mask, "connector_headroom_proxy_mw"] = reviewed_limit_proxy[reviewed_mask].clip(lower=0)
    base.loc[reviewed_mask, "connector_capacity_evidence_tier"] = "reviewed_border_split_proxy"
    base.loc[reviewed_mask, "connector_gate_state"] = "reviewed_split_pass"
    base.loc[reviewed_mask, "connector_gate_reason"] = (
        "A reviewed GB-FR capacity tier exists, so the France cable layer allocates reviewed border headroom by nominal cable share."
    )

    nominal_blocked = ~first_pass_published & ~reviewed_mask & nominal_headroom_proxy.le(0)
    base.loc[nominal_blocked, "connector_headroom_proxy_mw"] = 0.0
    base.loc[nominal_blocked, "connector_gate_state"] = "nominal_proxy_blocked"
    base.loc[nominal_blocked, "connector_gate_reason"] = (
        "The nominal-capacity proxy is fully consumed by the split border export flow on this France cable."
    )

    nominal_headroom = ~first_pass_published & ~reviewed_mask & nominal_headroom_proxy.gt(0) & base["border_flow_state"].ne("flow_unknown")
    base.loc[nominal_headroom, "connector_gate_state"] = "nominal_headroom_proxy"
    base.loc[nominal_headroom, "connector_gate_reason"] = (
        "Published GB-FR capacity is unavailable, so cable headroom is only a nominal-share proxy anchored to observed border flow."
    )

    reviewed_publication_limit = pd.to_numeric(base["reviewed_publication_capacity_limit_mw"], errors="coerce")
    reviewed_publication_state = base["reviewed_publication_state"].fillna("")
    reviewed_publication_available = (
        base["reviewed_publication_tier_accepted_flag"]
        & reviewed_publication_state.isin(["available", "partial_capacity"])
        & reviewed_publication_limit.gt(0)
    )
    reviewed_publication_blocked = (
        base["reviewed_publication_tier_accepted_flag"]
        & (
            reviewed_publication_state.eq("outage")
            | reviewed_publication_limit.fillna(0).le(0)
        )
    )

    current_offered_before_publication = pd.to_numeric(base["connector_offered_capacity_mw_proxy"], errors="coerce")
    current_headroom_before_publication = pd.to_numeric(base["connector_headroom_proxy_mw"], errors="coerce")
    publication_limit_effective = reviewed_publication_limit.where(
        reviewed_publication_limit.notna(),
        current_offered_before_publication,
    )

    base.loc[reviewed_publication_blocked, "connector_offered_capacity_mw_proxy"] = 0.0
    base.loc[reviewed_publication_blocked, "connector_headroom_proxy_mw"] = 0.0
    base.loc[reviewed_publication_blocked, "connector_capacity_evidence_tier"] = "reviewed_public_doc_period"
    base.loc[reviewed_publication_blocked, "connector_gate_state"] = "reviewed_publication_blocked"
    base.loc[reviewed_publication_blocked, "connector_gate_reason"] = (
        "A reviewed public France-connector publication reports the connector unavailable for this period."
    )

    reviewed_publication_promote = reviewed_publication_available & ~reviewed_publication_blocked & (
        ~first_pass_published
        | publication_limit_effective.lt(current_headroom_before_publication.fillna(np.inf))
        | base["connector_capacity_evidence_tier"].isin(
            ["nominal_static", "operator_no_active_outage", "reviewed_border_split_proxy"]
        )
    )
    base.loc[reviewed_publication_promote, "connector_offered_capacity_mw_proxy"] = pd.concat(
        [
            current_offered_before_publication.loc[reviewed_publication_promote],
            publication_limit_effective.loc[reviewed_publication_promote],
        ],
        axis=1,
    ).min(axis=1)
    base.loc[reviewed_publication_promote, "connector_headroom_proxy_mw"] = pd.concat(
        [
            current_headroom_before_publication.loc[reviewed_publication_promote],
            publication_limit_effective.loc[reviewed_publication_promote],
        ],
        axis=1,
    ).min(axis=1)
    base.loc[reviewed_publication_promote, "connector_capacity_evidence_tier"] = "reviewed_public_doc_period"

    publication_binding = reviewed_publication_promote & publication_limit_effective.lt(
        current_headroom_before_publication.fillna(np.inf)
    )
    publication_non_binding = reviewed_publication_promote & ~publication_binding
    base.loc[publication_binding, "connector_gate_state"] = "reviewed_publication_cap"
    base.loc[publication_binding, "connector_gate_reason"] = (
        "A reviewed public France-connector publication caps this cable below the current proxy headroom for the period."
    )
    base.loc[publication_non_binding, "connector_gate_state"] = "reviewed_publication_pass"
    base.loc[publication_non_binding, "connector_gate_reason"] = (
        "A reviewed public France-connector publication confirms connector availability for the period, so the route can use a reviewed connector tier instead of staying capacity-unknown."
    )

    base["operator_capacity_limit_mw"] = pd.to_numeric(base["operator_capacity_limit_mw"], errors="coerce")
    operator_outage = base["operator_availability_state"].eq("outage")
    operator_partial = base["operator_availability_state"].eq("partial_outage")
    operator_available = base["operator_availability_state"].eq("available")

    base.loc[operator_outage, "connector_offered_capacity_mw_proxy"] = pd.concat(
        [
            pd.to_numeric(base.loc[operator_outage, "connector_offered_capacity_mw_proxy"], errors="coerce"),
            base.loc[operator_outage, "operator_capacity_limit_mw"].fillna(0.0),
        ],
        axis=1,
    ).min(axis=1)
    base.loc[operator_outage, "connector_headroom_proxy_mw"] = base.loc[operator_outage, "operator_capacity_limit_mw"].fillna(0.0)
    base.loc[operator_outage, "connector_capacity_evidence_tier"] = "operator_outage_truth"
    base.loc[operator_outage, "connector_gate_state"] = "operator_outage_blocked"
    base.loc[operator_outage, "connector_gate_reason"] = (
        "Operator outage messages report the France connector unavailable for the hour."
    )

    base.loc[operator_partial, "connector_offered_capacity_mw_proxy"] = pd.concat(
        [
            pd.to_numeric(base.loc[operator_partial, "connector_offered_capacity_mw_proxy"], errors="coerce"),
            base.loc[operator_partial, "operator_capacity_limit_mw"],
        ],
        axis=1,
    ).min(axis=1)
    base.loc[operator_partial, "connector_headroom_proxy_mw"] = pd.concat(
        [
            pd.to_numeric(base.loc[operator_partial, "connector_headroom_proxy_mw"], errors="coerce"),
            base.loc[operator_partial, "operator_capacity_limit_mw"],
        ],
        axis=1,
    ).min(axis=1)
    base.loc[operator_partial, "connector_capacity_evidence_tier"] = "operator_partial_capacity_truth"
    base.loc[operator_partial, "connector_gate_state"] = "operator_partial_capacity_cap"
    base.loc[operator_partial, "connector_gate_reason"] = (
        "Operator outage messages cap the France connector below nominal capacity for the hour."
    )

    operator_available_nominal = operator_available & base["connector_capacity_evidence_tier"].eq("nominal_static")
    base.loc[operator_available_nominal, "connector_capacity_evidence_tier"] = "operator_no_active_outage"

    column_order = list(_empty_france_connector_frame().columns)
    return base[column_order].sort_values(["interval_start_utc", "connector_key"]).reset_index(drop=True)


def materialize_france_connector_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    token: str | None = None,
    interconnector_flow: pd.DataFrame | None = None,
    interconnector_capacity: pd.DataFrame | None = None,
    interconnector_capacity_review_policy: pd.DataFrame | None = None,
    interconnector_capacity_reviewed: pd.DataFrame | None = None,
    france_connector_reviewed_period: pd.DataFrame | None = None,
    france_connector_availability: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    resolved_flow = interconnector_flow
    resolved_capacity = interconnector_capacity
    resolved_review_policy = interconnector_capacity_review_policy
    resolved_reviewed_capacity = interconnector_capacity_reviewed

    if resolved_flow is None and token:
        resolved_flow = build_fact_interconnector_flow_hourly(start_date=start_date, end_date=end_date, token=token)
    if resolved_capacity is None and token:
        resolved_capacity = build_fact_interconnector_capacity_hourly(start_date=start_date, end_date=end_date, token=token)
    if resolved_review_policy is None and token:
        review_audit = build_interconnector_capacity_source_audit(start_date=start_date, end_date=end_date, token=token)
        resolved_review_policy = build_interconnector_capacity_review_policy(review_audit[INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE])
    if resolved_reviewed_capacity is None and token and resolved_review_policy is not None:
        resolved_reviewed_capacity = build_interconnector_capacity_reviewed_hourly(
            start_date=start_date,
            end_date=end_date,
            token=token,
            review_policy=resolved_review_policy,
        )

    dim_cable = interconnector_cable_frame()
    fact = build_fact_france_connector_hourly(
        start_date=start_date,
        end_date=end_date,
        interconnector_flow=resolved_flow,
        interconnector_capacity=resolved_capacity,
        interconnector_capacity_review_policy=resolved_review_policy,
        interconnector_capacity_reviewed=resolved_reviewed_capacity,
        france_connector_reviewed_period=france_connector_reviewed_period,
        france_connector_availability=france_connector_availability,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dim_cable.to_csv(output_path / f"{DIM_INTERCONNECTOR_CABLE_TABLE}.csv", index=False)
    fact.to_csv(output_path / f"{FRANCE_CONNECTOR_TABLE}.csv", index=False)
    return {
        DIM_INTERCONNECTOR_CABLE_TABLE: dim_cable,
        FRANCE_CONNECTOR_TABLE: fact,
    }
