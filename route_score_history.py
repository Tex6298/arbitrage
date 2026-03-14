from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from asset_mapping import cluster_frame
from gb_topology import INTERCONNECTOR_HUBS, ROUTE_HUB_OPTIONS
from network_overlay import build_border_network_overlay
from physical_constraints import ROUTE_BORDER_KEYS, ROUTES, compute_netbacks


ROUTE_SCORE_TABLE = "fact_route_score_hourly"


def _empty_route_score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "hub_key",
            "hub_label",
            "hub_target_zone",
            "hub_neighbor_domain_key",
            "hub_current_route_fit",
            "route_name",
            "route_label",
            "route_target_zone",
            "route_hub_preference_rank",
            "route_border_key",
            "route_price_score_eur_per_mwh",
            "route_price_feasible_flag",
            "route_price_bottleneck",
            "transfer_gate_mw_proxy",
            "transfer_gate_utilization_proxy",
            "transfer_gate_state",
            "transfer_gate_reason",
            "internal_transfer_evidence_tier",
            "internal_transfer_review_state",
            "internal_transfer_tier_accepted_flag",
            "internal_transfer_capacity_policy_action",
            "internal_transfer_gate_state",
            "internal_transfer_capacity_limit_mw",
            "internal_transfer_source_provider",
            "internal_transfer_source_family",
            "internal_transfer_source_key",
            "border_observed_flow_mw",
            "first_pass_border_offered_capacity_mw",
            "first_pass_border_headroom_proxy_mw",
            "first_pass_border_gate_state",
            "first_pass_border_flow_state",
            "first_pass_border_capacity_state",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_border_offered_capacity_mw",
            "reviewed_border_headroom_proxy_mw",
            "reviewed_border_gate_state",
            "reviewed_border_flow_state",
            "reviewed_border_capacity_state",
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
            "connector_notice_state",
            "connector_notice_known_flag",
            "connector_notice_active_flag",
            "connector_notice_upcoming_flag",
            "connector_notice_group_key",
            "connector_notice_planning_state",
            "connector_notice_planned_outage_flag",
            "connector_notice_expected_capacity_limit_mw",
            "connector_notice_hours_until_start",
            "connector_notice_days_until_start",
            "connector_notice_hours_since_publication",
            "connector_notice_lead_time_hours",
            "connector_notice_revision_count",
            "connector_notice_source_revision_rank",
            "connector_notice_source_provider",
            "connector_notice_source_family",
            "connector_notice_source_key",
            "connector_notice_source_label",
            "connector_notice_source_document_title",
            "connector_notice_source_document_url",
            "connector_notice_source_reference",
            "connector_notice_source_published_utc",
            "connector_itl_state",
            "connector_itl_evidence_tier",
            "connector_itl_tier_accepted_flag",
            "connector_itl_capacity_limit_mw",
            "connector_itl_auction_type",
            "connector_itl_restriction_reason",
            "connector_itl_source_provider",
            "connector_itl_source_key",
            "connector_itl_source_published_utc",
            "connector_key",
            "connector_label",
            "connector_operator",
            "connector_operator_source_provider",
            "connector_operator_availability_state",
            "connector_operator_capacity_evidence_tier",
            "connector_operator_capacity_limit_mw",
            "connector_nominal_capacity_mw",
            "connector_nominal_capacity_share_of_border",
            "connector_capacity_evidence_tier",
            "connector_headroom_proxy_mw",
            "connector_gate_state",
            "connector_gate_reason",
            "route_delivery_tier",
            "route_delivery_signal",
            "deliverable_mw_proxy",
            "deliverable_route_score_eur_per_mwh",
            "route_delivery_reason",
        ]
    )


def _route_hub_preferences() -> Dict[str, Tuple[str, ...]]:
    return {option.route_name: option.preferred_hubs for option in ROUTE_HUB_OPTIONS}


def _cluster_preferred_hub_lookup() -> Dict[str, Tuple[str, ...]]:
    clusters = cluster_frame()[["cluster_key", "preferred_hub_candidates"]].copy()
    lookup: Dict[str, Tuple[str, ...]] = {}
    for row in clusters.itertuples(index=False):
        preferred_hubs = tuple(
            value.strip()
            for value in str(row.preferred_hub_candidates).split(",")
            if value and value.strip()
        )
        lookup[str(row.cluster_key)] = preferred_hubs
    return lookup


def _hub_allowed_for_cluster(hub_key: str, allowed_hubs: object) -> bool:
    if not isinstance(allowed_hubs, tuple) or not allowed_hubs:
        return True
    return hub_key in allowed_hubs


def _overlay_lookup(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "interval_start_utc",
                "border_key",
                f"{prefix}_border_offered_capacity_mw",
                f"{prefix}_border_headroom_proxy_mw",
                f"{prefix}_border_gate_state",
                f"{prefix}_border_flow_state",
                f"{prefix}_border_capacity_state",
                f"{prefix}_border_observed_flow_mw",
            ]
        )
    overlay = frame[
        [
            "interval_start_utc",
            "border_key",
            "border_offered_capacity_mw",
            "border_headroom_proxy_mw",
            "border_gate_state",
            "border_flow_state",
            "border_capacity_state",
            "border_flow_mw",
        ]
    ].copy()
    return overlay.rename(
        columns={
            "border_offered_capacity_mw": f"{prefix}_border_offered_capacity_mw",
            "border_headroom_proxy_mw": f"{prefix}_border_headroom_proxy_mw",
            "border_gate_state": f"{prefix}_border_gate_state",
            "border_flow_state": f"{prefix}_border_flow_state",
            "border_capacity_state": f"{prefix}_border_capacity_state",
            "border_flow_mw": f"{prefix}_border_observed_flow_mw",
        }
    )


def _aggregate_internal_review_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    work["internal_transfer_tier_accepted_flag"] = work["internal_transfer_tier_accepted_flag"].where(
        work["internal_transfer_tier_accepted_flag"].notna(),
        False,
    ).astype(bool)
    work = work[work["internal_transfer_tier_accepted_flag"]].copy()
    if work.empty:
        return work
    work["internal_transfer_reviewed_capacity_limit_mw"] = pd.to_numeric(
        work["internal_transfer_reviewed_capacity_limit_mw"],
        errors="coerce",
    )
    work["_blocked_rank"] = work["internal_transfer_reviewed_gate_state"].fillna("").astype(str).str.startswith("blocked_").map(
        {True: 0, False: 1}
    )
    work["_limit_sort"] = work["internal_transfer_reviewed_capacity_limit_mw"].fillna(np.inf)
    work["_tier_sort"] = work["internal_transfer_reviewed_evidence_tier"].map(
        {
            "reviewed_internal_constraint_boundary": 0,
            "reviewed_internal_transfer_period": 1,
        }
    ).fillna(9)
    work = work.sort_values(
        [
            "interval_start_utc",
            "cluster_key",
            "hub_key",
            "_blocked_rank",
            "_limit_sort",
            "_tier_sort",
        ],
        ascending=[True, True, True, True, True, True],
    )
    return work.drop_duplicates(["interval_start_utc", "cluster_key", "hub_key"], keep="first").copy()


def build_fact_route_score_hourly(
    prices: pd.DataFrame,
    gb_transfer_gate: pd.DataFrame,
    gb_transfer_reviewed_hourly: pd.DataFrame | None = None,
    interconnector_itl: pd.DataFrame | None = None,
    interconnector_flow: pd.DataFrame | None = None,
    interconnector_capacity: pd.DataFrame | None = None,
    interconnector_capacity_reviewed: pd.DataFrame | None = None,
    interconnector_capacity_review_policy: pd.DataFrame | None = None,
    france_connector: pd.DataFrame | None = None,
    france_connector_notice: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if gb_transfer_gate is None or gb_transfer_gate.empty:
        return _empty_route_score_frame()

    netbacks = compute_netbacks(
        prices,
        interconnector_flow=interconnector_flow,
        interconnector_capacity=interconnector_capacity,
    ).copy()
    netbacks.index = pd.to_datetime(netbacks.index, utc=True)
    netbacks = netbacks.reset_index(names="interval_start_utc")
    netbacks["interval_start_local"] = netbacks["interval_start_utc"].dt.tz_convert("Europe/London")
    netbacks["interval_end_utc"] = netbacks["interval_start_utc"] + pd.Timedelta(hours=1)
    netbacks["interval_end_local"] = netbacks["interval_end_utc"].dt.tz_convert("Europe/London")
    netbacks["date"] = netbacks["interval_start_local"].dt.date

    review_policy = interconnector_capacity_review_policy.copy() if interconnector_capacity_review_policy is not None else pd.DataFrame()
    if review_policy.empty:
        review_policy = pd.DataFrame(
            columns=[
                "border_key",
                "direction_key",
                "review_state",
                "reviewed_evidence_tier",
                "reviewed_tier_accepted_flag",
                "capacity_policy_action",
            ]
        )
    review_policy = review_policy[review_policy["direction_key"] == "gb_to_neighbor"].copy()
    review_policy = review_policy[
        [
            "border_key",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
        ]
    ].drop_duplicates(subset=["border_key"])

    reviewed_overlay = _overlay_lookup(
        build_border_network_overlay(interconnector_flow, interconnector_capacity_reviewed),
        prefix="reviewed",
    )
    reviewed_overlay = reviewed_overlay.sort_values(["interval_start_utc", "border_key"]).reset_index(drop=True)
    connector_notice_lookup = france_connector_notice.copy() if france_connector_notice is not None else pd.DataFrame()
    if connector_notice_lookup.empty:
        connector_notice_lookup = pd.DataFrame(
            columns=[
                "interval_start_utc",
                "connector_key",
                "direction_key",
                "notice_state",
                "notice_known_flag",
                "notice_active_flag",
                "notice_upcoming_flag",
                "notice_group_key",
                "notice_planning_state",
                "planned_outage_flag",
                "expected_capacity_limit_mw",
                "hours_until_notice_start",
                "days_until_notice_start",
                "hours_since_notice_publication",
                "notice_lead_time_hours",
                "notice_revision_count",
                "source_revision_rank",
                "source_provider",
                "source_family",
                "source_key",
                "source_label",
                "source_document_title",
                "source_document_url",
                "source_reference",
                "source_published_utc",
            ]
        )
    connector_notice_lookup["interval_start_utc"] = pd.to_datetime(
        connector_notice_lookup["interval_start_utc"],
        utc=True,
        errors="coerce",
    )
    connector_notice_lookup = connector_notice_lookup[
        connector_notice_lookup["direction_key"].eq("gb_to_neighbor")
    ].copy()
    connector_lookup = france_connector.copy() if france_connector is not None else pd.DataFrame()
    if connector_lookup.empty:
        connector_lookup = pd.DataFrame(
            columns=[
                "interval_start_utc",
                "connector_key",
                "connector_label",
                "operator_name",
                "operator_source_provider",
                "operator_availability_state",
                "operator_capacity_evidence_tier",
                "operator_capacity_limit_mw",
                "nominal_capacity_mw",
                "nominal_capacity_share_of_border",
                "connector_capacity_evidence_tier",
                "connector_headroom_proxy_mw",
                "connector_gate_state",
                "connector_gate_reason",
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
            ]
        )
    for column, default in (
        ("connector_key", pd.NA),
        ("connector_label", pd.NA),
        ("operator_name", pd.NA),
        ("operator_source_provider", pd.NA),
        ("operator_availability_state", pd.NA),
        ("operator_capacity_evidence_tier", pd.NA),
        ("operator_capacity_limit_mw", np.nan),
        ("nominal_capacity_mw", np.nan),
        ("nominal_capacity_share_of_border", np.nan),
            ("connector_capacity_evidence_tier", pd.NA),
            ("connector_headroom_proxy_mw", np.nan),
            ("connector_gate_state", pd.NA),
            ("connector_gate_reason", pd.NA),
            ("reviewed_publication_state", pd.NA),
            ("reviewed_publication_evidence_tier", pd.NA),
            ("reviewed_publication_tier_accepted_flag", False),
            ("reviewed_publication_capacity_policy_action", pd.NA),
            ("reviewed_publication_capacity_limit_mw", np.nan),
            ("reviewed_publication_source_provider", pd.NA),
            ("reviewed_publication_source_family", pd.NA),
            ("reviewed_publication_source_key", pd.NA),
            ("reviewed_publication_source_label", pd.NA),
            ("reviewed_publication_source_document_title", pd.NA),
            ("reviewed_publication_source_document_url", pd.NA),
            ("reviewed_publication_source_reference", pd.NA),
            ("reviewed_publication_source_published_date", pd.NaT),
            ("reviewed_publication_source_count", np.nan),
        ):
            if column not in connector_lookup.columns:
                connector_lookup[column] = default
    connector_lookup["interval_start_utc"] = pd.to_datetime(
        connector_lookup["interval_start_utc"],
        utc=True,
        errors="coerce",
    )
    connector_itl_lookup = interconnector_itl.copy() if interconnector_itl is not None else pd.DataFrame()
    if connector_itl_lookup.empty:
        connector_itl_lookup = pd.DataFrame(
            columns=[
                "interval_start_utc",
                "connector_key",
                "connector_label",
                "direction_key",
                "itl_state",
                "itl_mw",
                "auction_type",
                "restriction_reason",
                "source_provider",
                "source_key",
                "source_published_utc",
            ]
        )
    connector_itl_lookup["interval_start_utc"] = pd.to_datetime(
        connector_itl_lookup["interval_start_utc"],
        utc=True,
        errors="coerce",
    )
    connector_itl_lookup["source_published_utc"] = pd.to_datetime(
        connector_itl_lookup.get("source_published_utc"),
        utc=True,
        errors="coerce",
    )
    connector_itl_lookup = connector_itl_lookup[
        connector_itl_lookup["direction_key"].eq("gb_to_neighbor")
    ].copy()
    connector_itl_lookup = connector_itl_lookup.rename(
        columns={
            "connector_key": "itl_connector_key",
            "connector_label": "itl_connector_label",
            "itl_state": "connector_itl_state",
            "itl_mw": "connector_itl_capacity_limit_mw",
            "auction_type": "connector_itl_auction_type",
            "restriction_reason": "connector_itl_restriction_reason",
            "source_provider": "connector_itl_source_provider",
            "source_key": "connector_itl_source_key",
            "source_published_utc": "connector_itl_source_published_utc",
        }
    )
    connector_itl_lookup["connector_itl_evidence_tier"] = "neso_interconnector_itl"
    connector_itl_lookup["connector_itl_tier_accepted_flag"] = True
    connector_itl_lookup = connector_itl_lookup[
        [
            "interval_start_utc",
            "itl_connector_key",
            "itl_connector_label",
            "connector_itl_state",
            "connector_itl_evidence_tier",
            "connector_itl_tier_accepted_flag",
            "connector_itl_capacity_limit_mw",
            "connector_itl_auction_type",
            "connector_itl_restriction_reason",
            "connector_itl_source_provider",
            "connector_itl_source_key",
            "connector_itl_source_published_utc",
        ]
    ].drop_duplicates(subset=["interval_start_utc", "itl_connector_key"], keep="last")

    route_preferences = _route_hub_preferences()
    cluster_preferred_hubs = _cluster_preferred_hub_lookup()
    transfer_gate = gb_transfer_gate.copy()
    transfer_gate["interval_start_utc"] = pd.to_datetime(transfer_gate["interval_start_utc"], utc=True, errors="coerce")
    internal_review_lookup = gb_transfer_reviewed_hourly.copy() if gb_transfer_reviewed_hourly is not None else pd.DataFrame()
    if internal_review_lookup.empty:
        internal_review_lookup = pd.DataFrame(
            columns=[
                "interval_start_utc",
                "cluster_key",
                "hub_key",
                "review_state",
                "reviewed_evidence_tier",
                "reviewed_tier_accepted_flag",
                "capacity_policy_action",
                "reviewed_gate_state",
                "reviewed_capacity_limit_mw",
                "source_provider",
                "source_family",
                "source_key",
            ]
        )
    internal_review_lookup["interval_start_utc"] = pd.to_datetime(
        internal_review_lookup["interval_start_utc"],
        utc=True,
        errors="coerce",
    )
    internal_review_lookup = internal_review_lookup[
        [
            "interval_start_utc",
            "cluster_key",
            "hub_key",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_gate_state",
            "reviewed_capacity_limit_mw",
            "source_provider",
            "source_family",
            "source_key",
        ]
    ].copy()
    internal_review_lookup = internal_review_lookup.rename(
        columns={
            "review_state": "internal_transfer_review_state",
            "reviewed_evidence_tier": "internal_transfer_reviewed_evidence_tier",
            "reviewed_tier_accepted_flag": "internal_transfer_tier_accepted_flag",
            "capacity_policy_action": "internal_transfer_capacity_policy_action",
            "reviewed_gate_state": "internal_transfer_reviewed_gate_state",
            "reviewed_capacity_limit_mw": "internal_transfer_reviewed_capacity_limit_mw",
            "source_provider": "internal_transfer_source_provider",
            "source_family": "internal_transfer_source_family",
            "source_key": "internal_transfer_source_key",
        }
    )
    internal_review_lookup = _aggregate_internal_review_lookup(internal_review_lookup)

    rows = []
    for route_name, preferred_hubs in route_preferences.items():
        route_transfer = transfer_gate[transfer_gate["hub_key"].isin(preferred_hubs)].copy()
        if not route_transfer.empty:
            cluster_allowed_hubs = route_transfer["cluster_key"].map(cluster_preferred_hubs)
            route_transfer = route_transfer[
                pd.Series(
                    [
                        _hub_allowed_for_cluster(row.hub_key, allowed_hubs)
                        for row, allowed_hubs in zip(route_transfer.itertuples(index=False), cluster_allowed_hubs)
                    ],
                    index=route_transfer.index,
                )
            ].copy()
        if route_transfer.empty:
            continue

        border_key = ROUTE_BORDER_KEYS[route_name]
        route_label = ROUTES[route_name].label
        target_zone = next(
            (INTERCONNECTOR_HUBS[hub_key].target_zone for hub_key in preferred_hubs if hub_key in INTERCONNECTOR_HUBS),
            None,
        )

        route_transfer["route_name"] = route_name
        route_transfer["route_label"] = route_label
        route_transfer["route_target_zone"] = target_zone
        route_transfer["route_hub_preference_rank"] = route_transfer["hub_key"].map(
            {hub_key: rank + 1 for rank, hub_key in enumerate(preferred_hubs)}
        )
        route_transfer["route_border_key"] = border_key
        route_transfer = route_transfer.merge(
            internal_review_lookup,
            on=["interval_start_utc", "cluster_key", "hub_key"],
            how="left",
        )
        for source_column, target_column in (
            ("border_offered_capacity_mw", "first_pass_border_offered_capacity_mw"),
            ("border_headroom_proxy_mw", "first_pass_border_headroom_proxy_mw"),
            ("border_gate_state", "first_pass_border_gate_state"),
            ("border_flow_state", "first_pass_border_flow_state"),
            ("border_capacity_state", "first_pass_border_capacity_state"),
        ):
            if source_column in route_transfer.columns and target_column not in route_transfer.columns:
                route_transfer[target_column] = route_transfer[source_column]

        required_netback_columns = [
            route_name,
            f"{route_name}_feasible",
            f"{route_name}_bottleneck",
            f"{route_name}_gb_border_observed_flow_mw",
            f"{route_name}_gb_border_offered_capacity_mw",
            f"{route_name}_gb_border_headroom_proxy_mw",
            f"{route_name}_gb_border_network_gate_state",
            f"{route_name}_gb_border_flow_state",
            f"{route_name}_gb_border_capacity_state",
        ]
        for column in required_netback_columns:
            if column in netbacks.columns:
                continue
            if column.endswith("_gate_state") or column.endswith("_flow_state"):
                netbacks[column] = "capacity_unknown" if column.endswith("_gate_state") else "flow_unknown"
            elif column.endswith("_capacity_state"):
                netbacks[column] = "capacity_unknown"
            else:
                netbacks[column] = np.nan

        merge_columns = ["interval_start_utc", route_name, f"{route_name}_feasible", f"{route_name}_bottleneck"]
        rename_map = {
            route_name: "route_price_score_eur_per_mwh",
            f"{route_name}_feasible": "route_price_feasible_flag",
            f"{route_name}_bottleneck": "route_price_bottleneck",
        }
        optional_first_pass_sources = {
            f"{route_name}_gb_border_observed_flow_mw": "border_observed_flow_mw",
            f"{route_name}_gb_border_offered_capacity_mw": "first_pass_border_offered_capacity_mw",
            f"{route_name}_gb_border_headroom_proxy_mw": "first_pass_border_headroom_proxy_mw",
            f"{route_name}_gb_border_network_gate_state": "first_pass_border_gate_state",
            f"{route_name}_gb_border_flow_state": "first_pass_border_flow_state",
            f"{route_name}_gb_border_capacity_state": "first_pass_border_capacity_state",
        }
        for source_column, target_column in optional_first_pass_sources.items():
            if target_column in route_transfer.columns:
                continue
            merge_columns.append(source_column)
            rename_map[source_column] = target_column

        route_transfer = route_transfer.merge(
            netbacks[merge_columns].rename(columns=rename_map),
            on=["interval_start_utc"],
            how="left",
        )
        route_transfer = route_transfer.merge(review_policy, left_on="route_border_key", right_on="border_key", how="left")
        route_transfer = route_transfer.drop(columns=["border_key_y"], errors="ignore").rename(
            columns={"border_key_x": "border_key"}
        )
        route_transfer = route_transfer.merge(
            reviewed_overlay,
            left_on=["interval_start_utc", "route_border_key"],
            right_on=["interval_start_utc", "border_key"],
            how="left",
        )
        route_transfer = route_transfer.drop(columns=["border_key"], errors="ignore")
        route_transfer = route_transfer.merge(
            connector_itl_lookup,
            left_on=["interval_start_utc", "hub_key"],
            right_on=["interval_start_utc", "itl_connector_key"],
            how="left",
        )
        if border_key == "GB-FR" and not connector_lookup.empty:
            route_transfer = route_transfer.merge(
                connector_lookup[
                    [
                        "interval_start_utc",
                        "connector_key",
                        "connector_label",
                        "operator_name",
                        "operator_source_provider",
                        "operator_availability_state",
                        "operator_capacity_evidence_tier",
                        "operator_capacity_limit_mw",
                        "nominal_capacity_mw",
                        "nominal_capacity_share_of_border",
                        "connector_capacity_evidence_tier",
                        "connector_headroom_proxy_mw",
                        "connector_gate_state",
                        "connector_gate_reason",
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
                    ]
                ],
                left_on=["interval_start_utc", "hub_key"],
                right_on=["interval_start_utc", "connector_key"],
                how="left",
            )
        if border_key == "GB-FR" and not connector_notice_lookup.empty:
            route_transfer = route_transfer.merge(
                connector_notice_lookup[
                    [
                        "interval_start_utc",
                        "connector_key",
                        "direction_key",
                        "notice_state",
                        "notice_known_flag",
                        "notice_active_flag",
                        "notice_upcoming_flag",
                        "notice_group_key",
                        "notice_planning_state",
                        "planned_outage_flag",
                        "expected_capacity_limit_mw",
                        "hours_until_notice_start",
                        "days_until_notice_start",
                        "hours_since_notice_publication",
                        "notice_lead_time_hours",
                        "notice_revision_count",
                        "source_revision_rank",
                        "source_provider",
                        "source_family",
                        "source_key",
                        "source_label",
                        "source_document_title",
                        "source_document_url",
                        "source_reference",
                        "source_published_utc",
                    ]
                ],
                left_on=["interval_start_utc", "hub_key"],
                right_on=["interval_start_utc", "connector_key"],
                how="left",
                suffixes=("", "_notice"),
            )
        for column, default in (
            ("itl_connector_key", pd.NA),
            ("itl_connector_label", pd.NA),
            ("connector_itl_state", pd.NA),
            ("connector_itl_evidence_tier", pd.NA),
            ("connector_itl_tier_accepted_flag", False),
            ("connector_itl_capacity_limit_mw", np.nan),
            ("connector_itl_auction_type", pd.NA),
            ("connector_itl_restriction_reason", pd.NA),
            ("connector_itl_source_provider", pd.NA),
            ("connector_itl_source_key", pd.NA),
            ("connector_itl_source_published_utc", pd.NaT),
            ("connector_key", pd.NA),
            ("connector_label", pd.NA),
            ("operator_name", pd.NA),
            ("operator_source_provider", pd.NA),
            ("operator_availability_state", pd.NA),
            ("operator_capacity_evidence_tier", pd.NA),
            ("operator_capacity_limit_mw", np.nan),
            ("nominal_capacity_mw", np.nan),
            ("nominal_capacity_share_of_border", np.nan),
            ("connector_capacity_evidence_tier", pd.NA),
            ("connector_headroom_proxy_mw", np.nan),
            ("connector_gate_state", pd.NA),
            ("connector_gate_reason", pd.NA),
            ("reviewed_publication_state", pd.NA),
            ("reviewed_publication_evidence_tier", pd.NA),
            ("reviewed_publication_tier_accepted_flag", False),
            ("reviewed_publication_capacity_policy_action", pd.NA),
            ("reviewed_publication_capacity_limit_mw", np.nan),
            ("reviewed_publication_source_provider", pd.NA),
            ("reviewed_publication_source_family", pd.NA),
            ("reviewed_publication_source_key", pd.NA),
            ("reviewed_publication_source_label", pd.NA),
            ("reviewed_publication_source_document_title", pd.NA),
            ("reviewed_publication_source_document_url", pd.NA),
            ("reviewed_publication_source_reference", pd.NA),
            ("reviewed_publication_source_published_date", pd.NaT),
            ("reviewed_publication_source_count", np.nan),
            ("notice_state", pd.NA),
            ("notice_known_flag", False),
            ("notice_active_flag", False),
            ("notice_upcoming_flag", False),
            ("notice_group_key", pd.NA),
            ("notice_planning_state", pd.NA),
            ("planned_outage_flag", False),
            ("expected_capacity_limit_mw", np.nan),
            ("hours_until_notice_start", np.nan),
            ("days_until_notice_start", np.nan),
            ("hours_since_notice_publication", np.nan),
            ("notice_lead_time_hours", np.nan),
            ("notice_revision_count", np.nan),
            ("source_revision_rank", np.nan),
            ("source_provider", pd.NA),
            ("source_family", pd.NA),
            ("source_key", pd.NA),
            ("source_label", pd.NA),
            ("source_document_title", pd.NA),
            ("source_document_url", pd.NA),
            ("source_reference", pd.NA),
            ("source_published_utc", pd.NaT),
            ("internal_transfer_review_state", pd.NA),
            ("internal_transfer_reviewed_evidence_tier", pd.NA),
            ("internal_transfer_tier_accepted_flag", False),
            ("internal_transfer_capacity_policy_action", pd.NA),
            ("internal_transfer_reviewed_gate_state", pd.NA),
            ("internal_transfer_reviewed_capacity_limit_mw", np.nan),
            ("internal_transfer_source_provider", pd.NA),
            ("internal_transfer_source_family", pd.NA),
            ("internal_transfer_source_key", pd.NA),
        ):
            if column not in route_transfer.columns:
                route_transfer[column] = default
        route_transfer["connector_itl_tier_accepted_flag"] = route_transfer[
            "connector_itl_tier_accepted_flag"
        ].where(route_transfer["connector_itl_tier_accepted_flag"].notna(), False).astype(bool)
        route_transfer["connector_key"] = route_transfer["connector_key"].where(
            route_transfer["connector_key"].notna(),
            route_transfer["itl_connector_key"],
        )
        route_transfer["connector_label"] = route_transfer["connector_label"].where(
            route_transfer["connector_label"].notna(),
            route_transfer["itl_connector_label"],
        )
        existing_connector_limit = pd.to_numeric(route_transfer["connector_headroom_proxy_mw"], errors="coerce")
        connector_itl_limit = pd.to_numeric(route_transfer["connector_itl_capacity_limit_mw"], errors="coerce")
        connector_itl_blocked = route_transfer["connector_itl_tier_accepted_flag"] & connector_itl_limit.fillna(np.inf).le(0)
        connector_itl_binding = (
            route_transfer["connector_itl_tier_accepted_flag"]
            & connector_itl_limit.notna()
            & (~existing_connector_limit.notna() | connector_itl_limit.lt(existing_connector_limit))
        )
        route_transfer.loc[connector_itl_binding, "connector_headroom_proxy_mw"] = connector_itl_limit.loc[
            connector_itl_binding
        ]
        route_transfer.loc[connector_itl_binding, "connector_capacity_evidence_tier"] = "neso_interconnector_itl"
        route_transfer.loc[connector_itl_binding & ~connector_itl_blocked, "connector_gate_state"] = "itl_capacity_cap"
        route_transfer.loc[connector_itl_binding & ~connector_itl_blocked, "connector_gate_reason"] = (
            "NESO interconnector ITL caps exportable connector headroom for this hour."
        )
        route_transfer.loc[connector_itl_blocked, "connector_gate_state"] = "itl_blocked"
        route_transfer.loc[connector_itl_blocked, "connector_gate_reason"] = (
            "NESO interconnector ITL reports zero or negative export headroom for this hour."
        )

        internal_review_accepted = route_transfer["internal_transfer_tier_accepted_flag"].where(
            route_transfer["internal_transfer_tier_accepted_flag"].notna(),
            False,
        ).astype(bool)
        hard_upstream_dependency_block = route_transfer["gate_state"].fillna("").eq("blocked_upstream_dependency")
        internal_review_effective = internal_review_accepted & ~hard_upstream_dependency_block
        route_transfer["internal_transfer_evidence_tier"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_reviewed_evidence_tier"],
            "gb_topology_transfer_gate_proxy",
        )
        route_transfer["internal_transfer_review_state"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_review_state"],
            "proxy_fallback",
        )
        route_transfer["internal_transfer_capacity_policy_action"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_capacity_policy_action"],
            "proxy_fallback",
        )
        route_transfer["internal_transfer_gate_state"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_reviewed_gate_state"],
            route_transfer["gate_state"],
        )
        route_transfer["internal_transfer_capacity_limit_mw"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_reviewed_capacity_limit_mw"],
            route_transfer["transfer_gate_mw_proxy"],
        )
        route_transfer["internal_transfer_source_provider"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_source_provider"],
            "proxy",
        )
        route_transfer["internal_transfer_source_family"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_source_family"],
            "gb_topology_transfer_gate_proxy",
        )
        route_transfer["internal_transfer_source_key"] = np.where(
            internal_review_effective,
            route_transfer["internal_transfer_source_key"],
            "gb_topology_transfer_gate_proxy",
        )

        price_positive = (
            pd.to_numeric(route_transfer["route_price_score_eur_per_mwh"], errors="coerce").gt(0)
            & route_transfer["route_price_feasible_flag"].where(
                route_transfer["route_price_feasible_flag"].notna(),
                False,
            ).astype(bool)
        )
        transfer_blocked = (
            route_transfer["internal_transfer_gate_state"].fillna("").astype(str).str.startswith("blocked_")
            | pd.to_numeric(route_transfer["internal_transfer_capacity_limit_mw"], errors="coerce").fillna(0).le(0)
        )
        connector_limit = pd.to_numeric(route_transfer.get("connector_headroom_proxy_mw"), errors="coerce")
        connector_blocked = price_positive & ~transfer_blocked & connector_limit.fillna(np.inf).le(0)
        confirmed_available = route_transfer["first_pass_border_gate_state"].eq("pass")
        reviewed_available = (
            route_transfer["capacity_policy_action"].eq("allow_reviewed_explicit_daily")
            & route_transfer["reviewed_border_gate_state"].isin(["pass", "flow_unknown_capacity_published"])
        )
        connector_itl_reviewed_available = (
            route_transfer["connector_itl_tier_accepted_flag"]
            & pd.to_numeric(route_transfer["connector_itl_capacity_limit_mw"], errors="coerce").fillna(0).gt(0)
        )
        connector_reviewed_available = (
            route_transfer["route_border_key"].eq("GB-FR")
            & route_transfer["connector_capacity_evidence_tier"].eq("reviewed_public_doc_period")
            & connector_limit.fillna(0).gt(0)
            & route_transfer["reviewed_publication_tier_accepted_flag"].where(
                route_transfer["reviewed_publication_tier_accepted_flag"].notna(),
                False,
            ).astype(bool)
        )

        route_transfer["route_delivery_tier"] = "blocked"
        route_transfer["route_delivery_signal"] = "HOLD"
        route_transfer["deliverable_mw_proxy"] = np.nan
        route_transfer["deliverable_route_score_eur_per_mwh"] = np.nan
        route_transfer["route_delivery_reason"] = "The route is not yet deliverable under the current gate stack."

        route_transfer.loc[~price_positive, "route_delivery_tier"] = "no_price_signal"
        route_transfer.loc[~price_positive, "route_delivery_reason"] = (
            "The route netback is not positive after route-leg losses and fees."
        )
        route_transfer.loc[price_positive & transfer_blocked, "route_delivery_tier"] = "blocked_internal_transfer"
        route_transfer.loc[price_positive & transfer_blocked, "route_delivery_reason"] = (
            "The cluster-to-hub transfer gate blocks or zeroes this route before border capacity matters."
        )
        reviewed_transfer_blocked = (
            price_positive
            & route_transfer["internal_transfer_gate_state"].fillna("").astype(str).str.startswith("blocked_reviewed")
        )
        route_transfer.loc[reviewed_transfer_blocked, "route_delivery_reason"] = (
            "An accepted reviewed internal-transfer tier blocks or zeroes this route before border capacity matters."
        )
        route_transfer.loc[connector_blocked, "route_delivery_tier"] = "blocked_connector_capacity"
        route_transfer.loc[connector_blocked, "route_delivery_reason"] = (
            "The selected connector has no deliverable headroom after cable limits and operator availability are applied."
        )
        route_transfer.loc[
            connector_blocked & route_transfer["connector_gate_state"].eq("itl_blocked"),
            "route_delivery_reason",
        ] = "NESO interconnector ITL blocks exportable connector headroom for this hour."

        confirmed_mask = price_positive & ~transfer_blocked & ~connector_blocked & confirmed_available
        route_transfer.loc[confirmed_mask, "route_delivery_tier"] = "confirmed"
        route_transfer.loc[confirmed_mask, "route_delivery_signal"] = "EXPORT_CONFIRMED"
        route_transfer.loc[confirmed_mask, "deliverable_mw_proxy"] = pd.concat(
            [
                pd.to_numeric(route_transfer.loc[confirmed_mask, "internal_transfer_capacity_limit_mw"], errors="coerce"),
                pd.to_numeric(route_transfer.loc[confirmed_mask, "first_pass_border_headroom_proxy_mw"], errors="coerce"),
                connector_limit.loc[confirmed_mask],
            ],
            axis=1,
        ).min(axis=1)
        route_transfer.loc[confirmed_mask, "deliverable_route_score_eur_per_mwh"] = route_transfer.loc[
            confirmed_mask, "route_price_score_eur_per_mwh"
        ]
        route_transfer.loc[confirmed_mask, "route_delivery_reason"] = (
            "The route passes both the internal transfer gate and the first-pass direct border-capacity gate."
        )

        reviewed_mask = (
            price_positive
            & ~transfer_blocked
            & ~connector_blocked
            & ~confirmed_available
            & (reviewed_available | connector_reviewed_available | connector_itl_reviewed_available)
        )
        route_transfer.loc[reviewed_mask, "route_delivery_tier"] = "reviewed"
        route_transfer.loc[reviewed_mask, "route_delivery_signal"] = "EXPORT_REVIEWED"
        reviewed_headroom = pd.to_numeric(route_transfer.loc[reviewed_mask, "reviewed_border_headroom_proxy_mw"], errors="coerce")
        reviewed_capacity = pd.to_numeric(route_transfer.loc[reviewed_mask, "reviewed_border_offered_capacity_mw"], errors="coerce")
        reviewed_gate_state = route_transfer.loc[reviewed_mask, "reviewed_border_gate_state"]
        reviewed_limit = reviewed_headroom.where(~reviewed_gate_state.eq("flow_unknown_capacity_published"), reviewed_capacity)
        route_transfer.loc[reviewed_mask, "deliverable_mw_proxy"] = pd.concat(
            [
                pd.to_numeric(route_transfer.loc[reviewed_mask, "internal_transfer_capacity_limit_mw"], errors="coerce"),
                reviewed_limit,
                connector_limit.loc[reviewed_mask],
            ],
            axis=1,
        ).min(axis=1)
        route_transfer.loc[reviewed_mask, "deliverable_route_score_eur_per_mwh"] = route_transfer.loc[
            reviewed_mask, "route_price_score_eur_per_mwh"
        ]
        route_transfer.loc[reviewed_mask, "route_delivery_reason"] = (
            "The first-pass direct border-capacity gate is unavailable, but an accepted reviewed-capacity tier exists for this border."
        )
        reviewed_publication_only = reviewed_mask & ~reviewed_available & connector_reviewed_available
        route_transfer.loc[reviewed_publication_only, "route_delivery_reason"] = (
            "The route is price-positive and internally reachable, and a reviewed France connector publication period provides an auditable cable-level reviewed tier even though border capacity is still unpublished."
        )
        reviewed_itl_only = reviewed_mask & ~reviewed_available & ~connector_reviewed_available & connector_itl_reviewed_available
        route_transfer.loc[reviewed_itl_only, "route_delivery_reason"] = (
            "The route is price-positive and internally reachable, and NESO interconnector ITL provides an auditable connector-level reviewed tier even though the border-capacity gate is still unpublished or unaccepted."
        )

        unknown_mask = (
            price_positive
            & ~transfer_blocked
            & ~connector_blocked
            & ~confirmed_available
            & ~reviewed_available
            & ~connector_reviewed_available
            & ~connector_itl_reviewed_available
        )
        route_transfer.loc[unknown_mask, "route_delivery_tier"] = "capacity_unknown"
        route_transfer.loc[unknown_mask, "route_delivery_signal"] = "EXPORT_CAPACITY_UNKNOWN"
        route_transfer.loc[unknown_mask, "deliverable_mw_proxy"] = pd.concat(
            [
                pd.to_numeric(route_transfer.loc[unknown_mask, "internal_transfer_capacity_limit_mw"], errors="coerce"),
                connector_limit.loc[unknown_mask],
            ],
            axis=1,
        ).min(axis=1)
        route_transfer.loc[unknown_mask, "deliverable_route_score_eur_per_mwh"] = route_transfer.loc[
            unknown_mask, "route_price_score_eur_per_mwh"
        ]
        route_transfer.loc[unknown_mask, "route_delivery_reason"] = (
            "The route is price-positive and internally reachable, but border capacity remains unpublished or unaccepted for gating."
        )
        france_unknown_mask = unknown_mask & route_transfer["route_border_key"].eq("GB-FR") & connector_limit.notna()
        route_transfer.loc[france_unknown_mask, "route_delivery_reason"] = (
            "The route is price-positive and internally reachable, but GB-FR cable deliverability is still only a nominal-share proxy because published France border capacity is unavailable."
        )

        rows.append(route_transfer)

    if not rows:
        return _empty_route_score_frame()

    fact = pd.concat(rows, ignore_index=True)
    fact = fact.rename(
        columns={
            "gate_state": "transfer_gate_state",
            "gate_reason": "transfer_gate_reason",
            "operator_name": "connector_operator",
            "operator_source_provider": "connector_operator_source_provider",
            "operator_availability_state": "connector_operator_availability_state",
            "operator_capacity_evidence_tier": "connector_operator_capacity_evidence_tier",
            "operator_capacity_limit_mw": "connector_operator_capacity_limit_mw",
            "nominal_capacity_mw": "connector_nominal_capacity_mw",
            "nominal_capacity_share_of_border": "connector_nominal_capacity_share_of_border",
            "notice_state": "connector_notice_state",
            "notice_known_flag": "connector_notice_known_flag",
            "notice_active_flag": "connector_notice_active_flag",
            "notice_upcoming_flag": "connector_notice_upcoming_flag",
            "notice_group_key": "connector_notice_group_key",
            "notice_planning_state": "connector_notice_planning_state",
            "planned_outage_flag": "connector_notice_planned_outage_flag",
            "expected_capacity_limit_mw": "connector_notice_expected_capacity_limit_mw",
            "hours_until_notice_start": "connector_notice_hours_until_start",
            "days_until_notice_start": "connector_notice_days_until_start",
            "hours_since_notice_publication": "connector_notice_hours_since_publication",
            "notice_lead_time_hours": "connector_notice_lead_time_hours",
            "notice_revision_count": "connector_notice_revision_count",
            "source_revision_rank": "connector_notice_source_revision_rank",
            "source_provider": "connector_notice_source_provider",
            "source_family": "connector_notice_source_family",
            "source_key": "connector_notice_source_key",
            "source_label": "connector_notice_source_label",
            "source_document_title": "connector_notice_source_document_title",
            "source_document_url": "connector_notice_source_document_url",
            "source_reference": "connector_notice_source_reference",
            "source_published_utc": "connector_notice_source_published_utc",
        }
    )
    fact["reviewed_tier_accepted_flag"] = fact["reviewed_tier_accepted_flag"].where(
        fact["reviewed_tier_accepted_flag"].notna(),
        False,
    ).astype(bool)
    for column, default in (
        ("connector_key", pd.NA),
        ("connector_label", pd.NA),
        ("connector_operator", pd.NA),
        ("connector_operator_source_provider", pd.NA),
        ("connector_operator_availability_state", pd.NA),
        ("connector_operator_capacity_evidence_tier", pd.NA),
        ("connector_operator_capacity_limit_mw", np.nan),
        ("connector_nominal_capacity_mw", np.nan),
        ("connector_nominal_capacity_share_of_border", np.nan),
        ("connector_capacity_evidence_tier", pd.NA),
        ("connector_headroom_proxy_mw", np.nan),
        ("connector_gate_state", pd.NA),
        ("connector_gate_reason", pd.NA),
        ("reviewed_publication_state", pd.NA),
        ("reviewed_publication_evidence_tier", pd.NA),
        ("reviewed_publication_tier_accepted_flag", False),
        ("reviewed_publication_capacity_policy_action", pd.NA),
        ("reviewed_publication_capacity_limit_mw", np.nan),
        ("reviewed_publication_source_provider", pd.NA),
        ("reviewed_publication_source_family", pd.NA),
        ("reviewed_publication_source_key", pd.NA),
        ("reviewed_publication_source_label", pd.NA),
        ("reviewed_publication_source_document_title", pd.NA),
        ("reviewed_publication_source_document_url", pd.NA),
        ("reviewed_publication_source_reference", pd.NA),
        ("reviewed_publication_source_published_date", pd.NaT),
        ("reviewed_publication_source_count", np.nan),
        ("connector_notice_state", pd.NA),
        ("connector_notice_known_flag", False),
        ("connector_notice_active_flag", False),
        ("connector_notice_upcoming_flag", False),
        ("connector_notice_group_key", pd.NA),
        ("connector_notice_planning_state", pd.NA),
        ("connector_notice_planned_outage_flag", False),
        ("connector_notice_expected_capacity_limit_mw", np.nan),
        ("connector_notice_hours_until_start", np.nan),
        ("connector_notice_days_until_start", np.nan),
        ("connector_notice_hours_since_publication", np.nan),
        ("connector_notice_lead_time_hours", np.nan),
        ("connector_notice_revision_count", np.nan),
        ("connector_notice_source_revision_rank", np.nan),
        ("connector_notice_source_provider", pd.NA),
        ("connector_notice_source_family", pd.NA),
        ("connector_notice_source_key", pd.NA),
        ("connector_notice_source_label", pd.NA),
        ("connector_notice_source_document_title", pd.NA),
        ("connector_notice_source_document_url", pd.NA),
        ("connector_notice_source_reference", pd.NA),
        ("connector_notice_source_published_utc", pd.NaT),
        ("connector_itl_state", pd.NA),
        ("connector_itl_evidence_tier", pd.NA),
        ("connector_itl_tier_accepted_flag", False),
        ("connector_itl_capacity_limit_mw", np.nan),
        ("connector_itl_auction_type", pd.NA),
        ("connector_itl_restriction_reason", pd.NA),
        ("connector_itl_source_provider", pd.NA),
        ("connector_itl_source_key", pd.NA),
        ("connector_itl_source_published_utc", pd.NaT),
    ):
        if column not in fact.columns:
            fact[column] = default
    column_order = list(_empty_route_score_frame().columns)
    return fact[column_order].sort_values(
        ["interval_start_utc", "cluster_key", "route_name", "route_hub_preference_rank", "hub_key"]
    ).reset_index(drop=True)


def materialize_route_score_history(
    output_dir: str | Path,
    prices: pd.DataFrame,
    gb_transfer_gate: pd.DataFrame,
    gb_transfer_reviewed_hourly: pd.DataFrame | None = None,
    interconnector_itl: pd.DataFrame | None = None,
    interconnector_flow: pd.DataFrame | None = None,
    interconnector_capacity: pd.DataFrame | None = None,
    interconnector_capacity_reviewed: pd.DataFrame | None = None,
    interconnector_capacity_review_policy: pd.DataFrame | None = None,
    france_connector: pd.DataFrame | None = None,
    france_connector_notice: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_route_score_hourly(
        prices=prices,
        gb_transfer_gate=gb_transfer_gate,
        gb_transfer_reviewed_hourly=gb_transfer_reviewed_hourly,
        interconnector_itl=interconnector_itl,
        interconnector_flow=interconnector_flow,
        interconnector_capacity=interconnector_capacity,
        interconnector_capacity_reviewed=interconnector_capacity_reviewed,
        interconnector_capacity_review_policy=interconnector_capacity_review_policy,
        france_connector=france_connector,
        france_connector_notice=france_connector_notice,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{ROUTE_SCORE_TABLE}.csv", index=False)
    return {ROUTE_SCORE_TABLE: fact}
