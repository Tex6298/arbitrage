from __future__ import annotations

import pandas as pd


def prepare_border_flow_overlay(interconnector_flow: pd.DataFrame | None) -> pd.DataFrame:
    if interconnector_flow is None or interconnector_flow.empty:
        return pd.DataFrame(columns=["interval_start_utc", "border_key", "border_flow_mw", "border_flow_published_flag"])

    frame = interconnector_flow.copy()
    if "direction_key" in frame.columns:
        frame = frame[frame["direction_key"] == "gb_to_neighbor"].copy()
    if frame.empty:
        return pd.DataFrame(columns=["interval_start_utc", "border_key", "border_flow_mw", "border_flow_published_flag"])

    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["border_flow_mw"] = pd.to_numeric(frame["signed_flow_from_gb_mw"], errors="coerce")
    grouped = (
        frame.groupby(["interval_start_utc", "border_key"], dropna=False, sort=True)["border_flow_mw"]
        .sum(min_count=1)
        .reset_index()
    )
    grouped["border_flow_published_flag"] = grouped["border_flow_mw"].notna()
    return grouped


def prepare_border_capacity_overlay(interconnector_capacity: pd.DataFrame | None) -> pd.DataFrame:
    if interconnector_capacity is None or interconnector_capacity.empty:
        return pd.DataFrame(
            columns=["interval_start_utc", "border_key", "border_offered_capacity_mw", "border_capacity_published_flag"]
        )

    frame = interconnector_capacity.copy()
    if "direction_key" in frame.columns:
        frame = frame[frame["direction_key"] == "gb_to_neighbor"].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["interval_start_utc", "border_key", "border_offered_capacity_mw", "border_capacity_published_flag"]
        )

    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["border_offered_capacity_mw"] = pd.to_numeric(frame["offered_capacity_mw"], errors="coerce")
    grouped = (
        frame.groupby(["interval_start_utc", "border_key"], dropna=False, sort=True)["border_offered_capacity_mw"]
        .max()
        .reset_index()
    )
    grouped["border_capacity_published_flag"] = grouped["border_offered_capacity_mw"].notna()
    return grouped


def build_border_network_overlay(
    interconnector_flow: pd.DataFrame | None,
    interconnector_capacity: pd.DataFrame | None,
) -> pd.DataFrame:
    flow_overlay = prepare_border_flow_overlay(interconnector_flow)
    capacity_overlay = prepare_border_capacity_overlay(interconnector_capacity)

    if flow_overlay.empty and capacity_overlay.empty:
        return pd.DataFrame(
            columns=[
                "interval_start_utc",
                "border_key",
                "border_flow_mw",
                "border_flow_published_flag",
                "border_offered_capacity_mw",
                "border_capacity_published_flag",
                "positive_export_flow_mw",
                "border_headroom_proxy_mw",
                "border_flow_state",
                "border_capacity_state",
                "border_gate_state",
            ]
        )

    overlay = flow_overlay.merge(
        capacity_overlay,
        on=["interval_start_utc", "border_key"],
        how="outer",
    ).sort_values(["interval_start_utc", "border_key"]).reset_index(drop=True)

    overlay["border_flow_published_flag"] = overlay["border_flow_published_flag"].where(
        overlay["border_flow_published_flag"].notna(),
        False,
    ).astype(bool)
    overlay["border_capacity_published_flag"] = overlay["border_capacity_published_flag"].where(
        overlay["border_capacity_published_flag"].notna(),
        False,
    ).astype(bool)
    overlay["positive_export_flow_mw"] = pd.to_numeric(overlay["border_flow_mw"], errors="coerce").clip(lower=0)
    overlay["border_headroom_proxy_mw"] = (
        pd.to_numeric(overlay["border_offered_capacity_mw"], errors="coerce") - overlay["positive_export_flow_mw"]
    )

    overlay["border_flow_state"] = "flow_unknown"
    overlay.loc[
        overlay["border_flow_published_flag"] & pd.to_numeric(overlay["border_flow_mw"], errors="coerce").lt(0),
        "border_flow_state",
    ] = "gb_import_observed"
    overlay.loc[
        overlay["border_flow_published_flag"] & pd.to_numeric(overlay["border_flow_mw"], errors="coerce").eq(0),
        "border_flow_state",
    ] = "no_observed_flow"
    overlay.loc[
        overlay["border_flow_published_flag"] & pd.to_numeric(overlay["border_flow_mw"], errors="coerce").gt(0),
        "border_flow_state",
    ] = "gb_export_observed"

    overlay["border_capacity_state"] = "capacity_unknown"
    overlay.loc[
        overlay["border_capacity_published_flag"]
        & pd.to_numeric(overlay["border_offered_capacity_mw"], errors="coerce").le(0),
        "border_capacity_state",
    ] = "published_zero_or_negative"
    overlay.loc[
        overlay["border_capacity_published_flag"]
        & pd.to_numeric(overlay["border_offered_capacity_mw"], errors="coerce").gt(0),
        "border_capacity_state",
    ] = "published_positive"

    overlay["border_gate_state"] = "capacity_unknown"
    overlay.loc[
        overlay["border_capacity_published_flag"] & ~overlay["border_flow_published_flag"],
        "border_gate_state",
    ] = "flow_unknown_capacity_published"
    overlay.loc[
        overlay["border_capacity_published_flag"]
        & pd.to_numeric(overlay["border_offered_capacity_mw"], errors="coerce").le(0),
        "border_gate_state",
    ] = "blocked_zero_offered_capacity"
    overlay.loc[
        overlay["border_capacity_published_flag"]
        & overlay["border_flow_published_flag"]
        & pd.to_numeric(overlay["border_offered_capacity_mw"], errors="coerce").gt(0)
        & overlay["border_headroom_proxy_mw"].le(0),
        "border_gate_state",
    ] = "blocked_headroom_proxy"
    overlay.loc[
        overlay["border_capacity_published_flag"]
        & overlay["border_flow_published_flag"]
        & pd.to_numeric(overlay["border_offered_capacity_mw"], errors="coerce").gt(0)
        & overlay["border_headroom_proxy_mw"].gt(0),
        "border_gate_state",
    ] = "pass"
    return overlay
