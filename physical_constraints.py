from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from network_overlay import build_border_network_overlay


@dataclass(frozen=True)
class ConstraintAssumption:
    key: str
    question: str
    current_assumption: str
    risk: str
    next_solution: str


@dataclass(frozen=True)
class RouteSpec:
    label: str
    legs: Tuple[Dict[str, float | str], ...]


# Simple route heuristics. The model scores each border leg independently and treats
# a route as blocked if any leg is underwater for that hour.
ROUTES: Dict[str, RouteSpec] = {
    "R1_netback_GB_FR_DE_PL": RouteSpec(
        label="GB->FR->DE->PL",
        legs=(
            {"from": "GB", "to": "FR", "loss": 0.020, "fee": 0.60},
            {"from": "FR", "to": "DE", "loss": 0.010, "fee": 0.20},
            {"from": "DE", "to": "PL", "loss": 0.010, "fee": 0.30},
        ),
    ),
    "R2_netback_GB_NL_DE_PL": RouteSpec(
        label="GB->NL->DE->PL",
        legs=(
            {"from": "GB", "to": "NL", "loss": 0.020, "fee": 0.65},
            {"from": "NL", "to": "DE", "loss": 0.010, "fee": 0.15},
            {"from": "DE", "to": "PL", "loss": 0.010, "fee": 0.30},
        ),
    ),
}

ROUTE_BORDER_KEYS: Dict[str, str] = {
    "R1_netback_GB_FR_DE_PL": "GB-FR",
    "R2_netback_GB_NL_DE_PL": "GB-NL",
}


ASSUMPTIONS: Tuple[ConstraintAssumption, ...] = (
    ConstraintAssumption(
        key="gb_price_proxy",
        question="Does the national GB price represent curtailed wind that can actually reach an interconnector?",
        current_assumption="Yes. The model treats GB MID as a tradable export proxy for all GB generation.",
        risk="High. Curtailment is locational and national prices can hide north-south transmission stress.",
        next_solution="Introduce GB generation clusters and map each cluster to reachable interconnector landing points.",
    ),
    ConstraintAssumption(
        key="windfarm_mapping",
        question="Which wind farms or offshore clusters are behind the opportunity?",
        current_assumption="First-pass cluster scaffolding exists in asset_mapping.py, but it is still approximate and user-seeded.",
        risk="High. Until the registry is curated with real connection points and ownership data, route relevance is only directional.",
        next_solution="Replace the seed registry with confirmed asset, node, and owner metadata, then link each cluster to actual curtailment signals.",
    ),
    ConstraintAssumption(
        key="curtailment_signal",
        question="Are we using actual curtailment or constraint-management data yet?",
        current_assumption="No. The model only uses price spreads.",
        risk="High. Price spreads can miss curtailment-driven forced selling or redispatch conditions.",
        next_solution="Add curtailment and dispatch-down signals, then score route opportunities only when curtailment pressure is present.",
    ),
    ConstraintAssumption(
        key="internal_transfer",
        question="Can power move from the national connection point to the international connector in the same hour?",
        current_assumption="A first-pass fact_gb_transfer_gate_hourly proxy now exists, and route/opportunity layers can override it with a separate reviewed internal-transfer tier built from public boundary and constraint evidence. The legacy live route CSV path is still simpler than the historical cluster-aware stack.",
        risk="High. The fallback gate is still a proxy, and the reviewed tier is only as strong as the public evidence provided, so deliverability can still be overstated when internal GB transfer is tighter than the heuristic or the reviewed inputs are incomplete.",
        next_solution="Keep the reviewed tier explicit, expand the public internal evidence base, and replace the reviewed-input path with stronger operator or API feeds when available before attempting PTDF-style limits.",
    ),
    ConstraintAssumption(
        key="cross_border_capacity",
        question="Is interconnector capacity actually available on the route?",
        current_assumption="Route scoring now uses first-pass border flow plus first-pass offered capacity, a separate review-policy surface exists for alternate explicit-daily capacity on GB-NL, GB-BE, and GB-DK1, and GB-FR now has a France-specific cable layer with separate operator truth, reviewed public-document tiers, and an as-of notice/publication-time feature layer.",
        risk="Medium to high. Auctioned capacity, outages, and counterflows can invalidate the route.",
        next_solution="Decide whether reviewed explicit-daily tiers should be promoted into route scoring, keep improving the switchable ElecLink operator and public-document source stack, then replace the remaining border proxies with ATC/NTC, auction allocations, physical flow saturation by border and hour, and publication-time-aware capacity expectations.",
    ),
    ConstraintAssumption(
        key="route_costs",
        question="Are route costs dynamic and interconnector-specific?",
        current_assumption="No. Losses and fees are static heuristics.",
        risk="Medium. Relative route ranking can flip if one cable has materially different costs or availability.",
        next_solution="Replace fixed fees with connector-specific commercial and technical cost curves.",
    ),
)


def assumption_frame() -> pd.DataFrame:
    return pd.DataFrame([assumption.__dict__ for assumption in ASSUMPTIONS])


def remaining_workstreams() -> List[str]:
    return [
        "Replace the seed cluster registry in asset_mapping.py with confirmed wind farm, node, and ownership metadata.",
        "Add curtailment or redispatch signals so the model distinguishes generic spreads from forced-export conditions.",
        "Expand the reviewed internal-transfer evidence tier so more cluster-to-hub corridors are covered by auditable public boundary or constraint evidence instead of proxy fallback.",
        "Decide whether reviewed explicit-daily ENTSO-E capacity for GB-NL, GB-BE, and GB-DK1 is safe to promote beyond a reviewed evidence tier.",
        "Upgrade the switchable France connector source stack so better operator or API feeds can replace manual reviewed-period inputs without changing route-scoring contracts.",
        "Replace the reviewed internal-transfer input path with a stronger operator or API feed if one appears, without changing route or opportunity contracts.",
        "Upgrade fact_interconnector_capacity_hourly from offered-capacity first pass toward ATC/NTC, outages, and post-auction headroom by hour.",
        "Replace static leg fees and losses with connector-specific parameters and time-varying availability.",
    ]


def compute_route_metrics(df: pd.DataFrame, route_name: str, route_spec: RouteSpec) -> None:
    leg_margin_cols = []
    leg_labels = []

    for leg in route_spec.legs:
        source = str(leg["from"])
        sink = str(leg["to"])
        loss = float(leg["loss"])
        fee = float(leg["fee"])
        col = f"{route_name}_leg_{source}_{sink}"
        df[col] = (df[sink] * (1 - loss)) - df[source] - fee
        leg_margin_cols.append(col)
        leg_labels.append(f"{source}->{sink}")

    gross_col = f"{route_name}_gross"
    feasible_col = f"{route_name}_feasible"
    bottleneck_col = f"{route_name}_bottleneck"

    leg_margins = df[leg_margin_cols]
    df[gross_col] = leg_margins.sum(axis=1)
    df[feasible_col] = leg_margins.gt(0).all(axis=1)
    df[bottleneck_col] = leg_margins.idxmin(axis=1).map(dict(zip(leg_margin_cols, leg_labels)))
    df[route_name] = np.where(df[feasible_col], df[gross_col], leg_margins.min(axis=1))


def _argmax_label(frame: pd.DataFrame, columns: list[str], label_map: Dict[str, str]) -> pd.Series:
    if not columns:
        return pd.Series(index=frame.index, dtype="object")
    values = frame[columns]
    result = pd.Series(index=frame.index, dtype="object")
    valid_mask = values.notna().any(axis=1)
    if bool(valid_mask.any()):
        winning_columns = values[valid_mask].idxmax(axis=1)
        result.loc[valid_mask] = winning_columns.map(label_map)
    return result


def apply_interconnector_border_overlay(
    df: pd.DataFrame,
    interconnector_flow: pd.DataFrame | None,
    interconnector_capacity: pd.DataFrame | None,
) -> pd.DataFrame:
    if interconnector_flow is None and interconnector_capacity is None:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    border_overlay = build_border_network_overlay(
        interconnector_flow=interconnector_flow,
        interconnector_capacity=interconnector_capacity,
    )

    confirmed_score_cols = []
    relaxed_score_cols = []
    relaxed_route_gate_cols = {}
    confirmed_label_map = {}
    relaxed_label_map = {}

    for route_name, border_key in ROUTE_BORDER_KEYS.items():
        feasible_col = f"{route_name}_feasible"
        overlay_frame = border_overlay[border_overlay["border_key"] == border_key].set_index("interval_start_utc")

        flow_series = (
            overlay_frame["border_flow_mw"].reindex(out.index)
            if not overlay_frame.empty
            else pd.Series(index=out.index, dtype=float)
        )
        capacity_series = (
            overlay_frame["border_offered_capacity_mw"].reindex(out.index)
            if not overlay_frame.empty
            else pd.Series(index=out.index, dtype=float)
        )
        headroom_proxy = (
            overlay_frame["border_headroom_proxy_mw"].reindex(out.index)
            if not overlay_frame.empty
            else pd.Series(index=out.index, dtype=float)
        )
        flow_state = (
            overlay_frame["border_flow_state"].reindex(out.index, fill_value="flow_unknown")
            if not overlay_frame.empty
            else pd.Series("flow_unknown", index=out.index, dtype="object")
        )
        capacity_state = (
            overlay_frame["border_capacity_state"].reindex(out.index, fill_value="capacity_unknown")
            if not overlay_frame.empty
            else pd.Series("capacity_unknown", index=out.index, dtype="object")
        )
        gate_state = (
            overlay_frame["border_gate_state"].reindex(out.index, fill_value="capacity_unknown")
            if not overlay_frame.empty
            else pd.Series("capacity_unknown", index=out.index, dtype="object")
        )

        border_prefix = f"{route_name}_gb_border"
        out[f"{border_prefix}_key"] = border_key
        out[f"{border_prefix}_observed_flow_mw"] = flow_series
        out[f"{border_prefix}_offered_capacity_mw"] = capacity_series
        out[f"{border_prefix}_headroom_proxy_mw"] = headroom_proxy
        out[f"{border_prefix}_flow_state"] = flow_state
        out[f"{border_prefix}_capacity_state"] = capacity_state
        out[f"{border_prefix}_network_gate_state"] = gate_state

        confirmed_col = f"{route_name}_network_score_confirmed"
        relaxed_col = f"{route_name}_network_score_relaxed"
        out[confirmed_col] = np.where(out[feasible_col] & gate_state.eq("pass"), out[route_name], np.nan)
        out[relaxed_col] = np.where(
            out[feasible_col] & gate_state.isin(["pass", "capacity_unknown"]),
            out[route_name],
            np.nan,
        )
        confirmed_score_cols.append(confirmed_col)
        relaxed_score_cols.append(relaxed_col)
        confirmed_label_map[confirmed_col] = ROUTES[route_name].label
        relaxed_label_map[relaxed_col] = ROUTES[route_name].label
        relaxed_route_gate_cols[route_name] = f"{border_prefix}_network_gate_state"

    out["best_netback_network_confirmed"] = out[confirmed_score_cols].max(axis=1, skipna=True)
    out.loc[out[confirmed_score_cols].notna().sum(axis=1).eq(0), "best_netback_network_confirmed"] = np.nan
    out["best_route_network_confirmed"] = _argmax_label(out, confirmed_score_cols, confirmed_label_map)

    out["best_netback_network_relaxed"] = out[relaxed_score_cols].max(axis=1, skipna=True)
    out.loc[out[relaxed_score_cols].notna().sum(axis=1).eq(0), "best_netback_network_relaxed"] = np.nan
    out["best_route_network_relaxed"] = _argmax_label(out, relaxed_score_cols, relaxed_label_map)

    confirmed_positive = out["best_netback_network_confirmed"].fillna(-np.inf) > 0
    relaxed_positive = out["best_netback_network_relaxed"].fillna(-np.inf) > 0
    out["export_signal_network"] = np.where(
        confirmed_positive,
        "EXPORT_CONFIRMED",
        np.where(relaxed_positive, "EXPORT_CAPACITY_UNKNOWN", "HOLD"),
    )

    route_to_label = {route_name: route_spec.label for route_name, route_spec in ROUTES.items()}
    label_to_route = {label: route_name for route_name, label in route_to_label.items()}
    out["best_route_network_gate_state"] = None
    for route_label, route_name in label_to_route.items():
        mask = out["best_route_network_relaxed"] == route_label
        if not bool(mask.any()):
            continue
        gate_column = relaxed_route_gate_cols[route_name]
        out.loc[mask, "best_route_network_gate_state"] = out.loc[mask, gate_column]
    return out


def compute_netbacks(
    prices: pd.DataFrame,
    interconnector_flow: pd.DataFrame | None = None,
    interconnector_capacity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    prices: index=UTC hour, columns ['GB', 'FR', 'NL', 'DE', 'PL', 'CZ'] in EUR/MWh.
    Returns prices plus route metrics and a simple route signal.
    """
    required = {"GB", "FR", "NL", "DE", "PL", "CZ"}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise RuntimeError(f"missing price columns: {', '.join(missing)}")

    df = prices.copy().sort_index().interpolate(limit_direction="both")
    for route_name, route_spec in ROUTES.items():
        compute_route_metrics(df, route_name, route_spec)

    route_cols = list(ROUTES)
    route_label_map = {route_name: route_spec.label for route_name, route_spec in ROUTES.items()}

    df["best_netback"] = df[route_cols].max(axis=1)
    df["best_route"] = np.where(
        df["R1_netback_GB_FR_DE_PL"] >= df["R2_netback_GB_NL_DE_PL"],
        route_label_map["R1_netback_GB_FR_DE_PL"],
        route_label_map["R2_netback_GB_NL_DE_PL"],
    )
    df["export_signal"] = np.where(df["best_netback"] > 0, "EXPORT", "HOLD")
    return apply_interconnector_border_overlay(
        df,
        interconnector_flow=interconnector_flow,
        interconnector_capacity=interconnector_capacity,
    )
