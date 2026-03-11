from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


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
        current_assumption="A coarse reachability scaffold exists in gb_topology.py, but live route scoring still assumes internal transfer is available.",
        risk="High. The scaffold is only a status matrix; it does not yet gate or derate routes hour by hour.",
        next_solution="Turn the scaffold into explicit transfer gates or zonal/PTDF-style limits between clusters and interconnector hubs.",
    ),
    ConstraintAssumption(
        key="cross_border_capacity",
        question="Is interconnector capacity actually available on the route?",
        current_assumption="Implicitly yes. The model uses fixed losses and placeholder fees only.",
        risk="Medium to high. Auctioned capacity, outages, and counterflows can invalidate the route.",
        next_solution="Layer in ATC/NTC, auction allocations, outage flags, and physical flow saturation by border and hour.",
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
        "Materialize fact_gb_transfer_gate_hourly so the coarse hub reachability matrix becomes an hourly cluster-to-hub deliverability gate.",
        "Materialize fact_interconnector_flow_hourly so route scoring can see observed cable loading and direction by hour.",
        "Materialize fact_interconnector_capacity_hourly so route scoring can see tradable headroom, outages, and ATC or NTC constraints by hour.",
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


def compute_netbacks(prices: pd.DataFrame) -> pd.DataFrame:
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
    return df
