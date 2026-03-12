from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import pandas as pd

from asset_mapping import cluster_frame
from gb_topology import INTERCONNECTOR_HUBS, reachability_frame
from interconnector_capacity import build_fact_interconnector_capacity_hourly
from interconnector_flow import BORDER_FLOW_SPECS, build_fact_interconnector_flow_hourly, parse_iso_date
from network_overlay import build_border_network_overlay


GB_TRANSFER_GATE_TABLE = "fact_gb_transfer_gate_hourly"
GB_TRANSFER_GATE_SOURCE_KEY = "gb_topology_transfer_gate_proxy"
GB_TRANSFER_GATE_SOURCE_LABEL = "GB topology transfer-gate proxy"

STATUS_GATE_FACTOR: Dict[str, float] = {
    "near": 0.90,
    "conditional": 0.65,
    "stretched": 0.30,
    "upstream_dependency": 0.10,
}

CONFIDENCE_GATE_FACTOR: Dict[str, float] = {
    "high": 1.00,
    "medium": 0.85,
    "low": 0.70,
}

INTERNAL_GATE_STATE: Dict[str, str] = {
    "near": "pass",
    "conditional": "pass_conditional",
    "stretched": "pass_stretched",
    "upstream_dependency": "blocked_upstream_dependency",
}

HUB_BORDER_KEYS: Dict[str, str] = {}
HUB_NEIGHBOR_KEYS: Dict[str, str] = {}
for hub_key, hub in INTERCONNECTOR_HUBS.items():
    matching_spec = next((spec for spec in BORDER_FLOW_SPECS if spec.target_zone == hub.target_zone), None)
    if matching_spec is None:
        continue
    HUB_BORDER_KEYS[hub_key] = matching_spec.border_key
    HUB_NEIGHBOR_KEYS[hub_key] = matching_spec.neighbor_domain_key


def _empty_transfer_gate_frame() -> pd.DataFrame:
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
            "gate_scope",
            "gate_method",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "hub_key",
            "hub_label",
            "hub_target_zone",
            "hub_neighbor_domain_key",
            "hub_landing_bias",
            "hub_current_route_fit",
            "hub_note",
            "border_key",
            "status",
            "transfer_requirement",
            "confidence",
            "approx_cluster_capacity_mw",
            "status_gate_factor",
            "confidence_gate_factor",
            "structural_gate_fraction",
            "structural_gate_mw_proxy",
            "internal_gate_state",
            "border_flow_published_flag",
            "border_capacity_published_flag",
            "border_observed_flow_mw",
            "border_offered_capacity_mw",
            "border_headroom_proxy_mw",
            "border_flow_state",
            "border_capacity_state",
            "border_gate_state",
            "transfer_gate_mw_proxy",
            "transfer_gate_utilization_proxy",
            "gate_state",
            "gate_reason",
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


def _cluster_capacity_lookup() -> pd.DataFrame:
    frame = cluster_frame()[["cluster_key", "approx_capacity_mw"]].copy()
    frame = frame.rename(columns={"approx_capacity_mw": "approx_cluster_capacity_mw"})
    return frame


def build_fact_gb_transfer_gate_hourly(
    start_date: dt.date,
    end_date: dt.date,
    interconnector_flow: pd.DataFrame | None = None,
    interconnector_capacity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    reachability = reachability_frame().copy()
    if reachability.empty:
        return _empty_transfer_gate_frame()

    capacity_lookup = _cluster_capacity_lookup()
    reachability = reachability.merge(capacity_lookup, on="cluster_key", how="left")
    reachability["hub_target_zone"] = reachability["hub_key"].map(
        {hub_key: hub.target_zone for hub_key, hub in INTERCONNECTOR_HUBS.items()}
    )
    reachability["hub_neighbor_domain_key"] = reachability["hub_key"].map(HUB_NEIGHBOR_KEYS)
    reachability["hub_landing_bias"] = reachability["hub_key"].map(
        {hub_key: hub.landing_bias for hub_key, hub in INTERCONNECTOR_HUBS.items()}
    )
    reachability["hub_current_route_fit"] = reachability["hub_key"].map(
        {hub_key: hub.current_route_fit for hub_key, hub in INTERCONNECTOR_HUBS.items()}
    )
    reachability["hub_note"] = reachability["hub_key"].map({hub_key: hub.note for hub_key, hub in INTERCONNECTOR_HUBS.items()})
    reachability["border_key"] = reachability["hub_key"].map(HUB_BORDER_KEYS)
    reachability["status_gate_factor"] = reachability["status"].map(STATUS_GATE_FACTOR).fillna(0.0)
    reachability["confidence_gate_factor"] = reachability["confidence"].map(CONFIDENCE_GATE_FACTOR).fillna(0.0)
    reachability["structural_gate_fraction"] = (
        reachability["status_gate_factor"].astype(float) * reachability["confidence_gate_factor"].astype(float)
    )
    reachability["structural_gate_mw_proxy"] = (
        pd.to_numeric(reachability["approx_cluster_capacity_mw"], errors="coerce")
        * reachability["structural_gate_fraction"].astype(float)
    )
    reachability["internal_gate_state"] = reachability["status"].map(INTERNAL_GATE_STATE).fillna("unknown")

    hours = _hourly_window_frame(start_date, end_date)
    base = (
        hours.assign(_join_key=1)
        .merge(reachability.assign(_join_key=1), on="_join_key", how="inner")
        .drop(columns="_join_key")
    )

    border_overlay = build_border_network_overlay(
        interconnector_flow=interconnector_flow,
        interconnector_capacity=interconnector_capacity,
    )
    if not border_overlay.empty:
        overlay = border_overlay[
            [
                "interval_start_utc",
                "border_key",
                "border_flow_mw",
                "border_flow_published_flag",
                "border_offered_capacity_mw",
                "border_capacity_published_flag",
                "border_headroom_proxy_mw",
                "border_flow_state",
                "border_capacity_state",
                "border_gate_state",
            ]
        ].copy()
        overlay = overlay.rename(columns={"border_flow_mw": "border_observed_flow_mw"})
        base = base.merge(overlay, on=["interval_start_utc", "border_key"], how="left")
    else:
        base["border_observed_flow_mw"] = pd.NA
        base["border_flow_published_flag"] = False
        base["border_offered_capacity_mw"] = pd.NA
        base["border_capacity_published_flag"] = False
        base["border_headroom_proxy_mw"] = pd.NA
        base["border_flow_state"] = "flow_unknown"
        base["border_capacity_state"] = "capacity_unknown"
        base["border_gate_state"] = "capacity_unknown"

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

    base["source_key"] = GB_TRANSFER_GATE_SOURCE_KEY
    base["source_label"] = GB_TRANSFER_GATE_SOURCE_LABEL
    base["target_is_proxy"] = True
    base["gate_scope"] = "cluster_to_hub_internal_transfer_proxy"
    base["gate_method"] = "reachability_factor_plus_border_overlay_v1"

    base["transfer_gate_mw_proxy"] = pd.to_numeric(base["structural_gate_mw_proxy"], errors="coerce")
    upstream_mask = base["status"].eq("upstream_dependency")
    base.loc[upstream_mask, "transfer_gate_mw_proxy"] = 0.0

    border_block_mask = (
        ~upstream_mask & base["border_gate_state"].isin(["blocked_zero_offered_capacity", "blocked_headroom_proxy"])
    )
    base.loc[border_block_mask, "transfer_gate_mw_proxy"] = 0.0

    pass_mask = ~upstream_mask & base["border_gate_state"].eq("pass") & base["border_headroom_proxy_mw"].notna()
    base.loc[pass_mask, "transfer_gate_mw_proxy"] = pd.concat(
        [
            pd.to_numeric(base.loc[pass_mask, "structural_gate_mw_proxy"], errors="coerce"),
            pd.to_numeric(base.loc[pass_mask, "border_headroom_proxy_mw"], errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)

    published_capacity_mask = (
        ~upstream_mask
        &
        base["border_gate_state"].eq("flow_unknown_capacity_published") & base["border_offered_capacity_mw"].notna()
    )
    base.loc[published_capacity_mask, "transfer_gate_mw_proxy"] = pd.concat(
        [
            pd.to_numeric(base.loc[published_capacity_mask, "structural_gate_mw_proxy"], errors="coerce"),
            pd.to_numeric(base.loc[published_capacity_mask, "border_offered_capacity_mw"], errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)

    denominator = pd.to_numeric(base["approx_cluster_capacity_mw"], errors="coerce")
    base["transfer_gate_utilization_proxy"] = pd.NA
    valid_denominator = denominator.gt(0)
    base.loc[valid_denominator, "transfer_gate_utilization_proxy"] = (
        pd.to_numeric(base.loc[valid_denominator, "transfer_gate_mw_proxy"], errors="coerce") / denominator[valid_denominator]
    )

    base["gate_state"] = "capacity_unknown_reachable"
    base["gate_reason"] = "No published border capacity is available, so the gate falls back to structural reachability."
    base.loc[base["status"].eq("conditional"), "gate_state"] = "capacity_unknown_conditional"
    base.loc[base["status"].eq("conditional"), "gate_reason"] = (
        "The cluster-to-hub path is plausible but still depends on internal GB transfer assumptions."
    )
    base.loc[base["status"].eq("stretched"), "gate_state"] = "capacity_unknown_stretched"
    base.loc[base["status"].eq("stretched"), "gate_reason"] = (
        "The cluster-to-hub path is only a stretched corridor assumption until internal transfer truth exists."
    )
    base.loc[upstream_mask, "gate_state"] = "blocked_upstream_dependency"
    base.loc[upstream_mask, "gate_reason"] = (
        "The cluster depends on upstream mainland transfer before any interconnector-hub reachability can apply."
    )
    base.loc[border_block_mask, "gate_state"] = base.loc[border_block_mask, "border_gate_state"]
    base.loc[border_block_mask, "gate_reason"] = (
        "The border overlay is already blocked, so the internal transfer gate is zeroed for this hub-hour proxy."
    )
    base.loc[pass_mask & base["status"].eq("near"), "gate_state"] = "pass"
    base.loc[pass_mask & base["status"].eq("near"), "gate_reason"] = (
        "Both structural reachability and observed border headroom support the cluster-to-hub proxy path."
    )
    base.loc[pass_mask & base["status"].eq("conditional"), "gate_state"] = "pass_conditional"
    base.loc[pass_mask & base["status"].eq("conditional"), "gate_reason"] = (
        "Border headroom exists, but the internal transfer path is still only conditionally supported."
    )
    base.loc[pass_mask & base["status"].eq("stretched"), "gate_state"] = "pass_stretched"
    base.loc[pass_mask & base["status"].eq("stretched"), "gate_reason"] = (
        "Border headroom exists, but the internal transfer path remains a stretched proxy assumption."
    )
    base.loc[published_capacity_mask & base["status"].eq("near"), "gate_state"] = "published_capacity_pass"
    base.loc[published_capacity_mask & base["status"].eq("near"), "gate_reason"] = (
        "Published border capacity exists, but flow is unknown for the hour so the gate remains a conservative proxy."
    )
    base.loc[published_capacity_mask & base["status"].eq("conditional"), "gate_state"] = "published_capacity_conditional"
    base.loc[published_capacity_mask & base["status"].eq("conditional"), "gate_reason"] = (
        "Published border capacity exists, but both border flow and internal transfer still remain conditional."
    )
    base.loc[published_capacity_mask & base["status"].eq("stretched"), "gate_state"] = "published_capacity_stretched"
    base.loc[published_capacity_mask & base["status"].eq("stretched"), "gate_reason"] = (
        "Published border capacity exists, but the internal transfer path is still only a stretched proxy."
    )

    column_order = list(_empty_transfer_gate_frame().columns)
    return base[column_order].sort_values(["interval_start_utc", "cluster_key", "hub_key"]).reset_index(drop=True)


def materialize_gb_transfer_gate_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    token: str | None = None,
    interconnector_flow: pd.DataFrame | None = None,
    interconnector_capacity: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    resolved_flow = interconnector_flow
    resolved_capacity = interconnector_capacity
    if resolved_flow is None and token:
        resolved_flow = build_fact_interconnector_flow_hourly(start_date=start_date, end_date=end_date, token=token)
    if resolved_capacity is None and token:
        resolved_capacity = build_fact_interconnector_capacity_hourly(start_date=start_date, end_date=end_date, token=token)

    fact = build_fact_gb_transfer_gate_hourly(
        start_date=start_date,
        end_date=end_date,
        interconnector_flow=resolved_flow,
        interconnector_capacity=resolved_capacity,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{GB_TRANSFER_GATE_TABLE}.csv", index=False)
    return {GB_TRANSFER_GATE_TABLE: fact}
