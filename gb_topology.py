from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from asset_mapping import ASSET_CLUSTERS


@dataclass(frozen=True)
class InterconnectorHub:
    key: str
    label: str
    landing_bias: str
    target_zone: str
    current_route_fit: str
    note: str


@dataclass(frozen=True)
class ReachabilityRule:
    cluster_key: str
    hub_key: str
    status: str
    transfer_requirement: str
    confidence: str
    note: str


@dataclass(frozen=True)
class RouteHubOption:
    route_name: str
    target_zone: str
    preferred_hubs: Tuple[str, ...]
    note: str


INTERCONNECTOR_HUBS: Dict[str, InterconnectorHub] = {
    "ifa": InterconnectorHub("ifa", "IFA", "south-east", "FR", "current", "French landing option from the south-east corridor."),
    "ifa2": InterconnectorHub("ifa2", "IFA2", "south coast", "FR", "current", "French landing option with a south-coast bias."),
    "eleclink": InterconnectorHub("eleclink", "ElecLink", "south-east", "FR", "future", "Additional French route candidate if the model expands beyond the current two routes."),
    "britned": InterconnectorHub("britned", "BritNed", "south-east east-coast", "NL", "current", "Dutch landing option and the most direct fit to the current NL route."),
    "nemo": InterconnectorHub("nemo", "Nemo", "south-east east-coast", "BE", "future", "Belgian landing option and a proxy for east-coast exportability."),
    "nsl": InterconnectorHub("nsl", "North Sea Link", "north-east", "NO", "future", "North-east landing useful for testing northern export reachability."),
    "ewic": InterconnectorHub("ewic", "East-West", "north-west", "IE", "future", "Irish Sea landing relevant to North Wales and western flows."),
    "viking_link": InterconnectorHub("viking_link", "Viking Link", "east-coast", "DK", "future", "East-coast landing that may matter if Nordic or Danish routes are added."),
}


ROUTE_HUB_OPTIONS: Tuple[RouteHubOption, ...] = (
    RouteHubOption(
        route_name="R1_netback_GB_FR_DE_PL",
        target_zone="FR",
        preferred_hubs=("ifa", "ifa2", "eleclink"),
        note="Current FR route should eventually choose from real French-facing interconnector hubs instead of treating GB as one node.",
    ),
    RouteHubOption(
        route_name="R2_netback_GB_NL_DE_PL",
        target_zone="NL",
        preferred_hubs=("britned",),
        note="Current NL route maps most directly to BritNed in the first-pass topology.",
    ),
)


REACHABILITY_RULES: Tuple[ReachabilityRule, ...] = (
    ReachabilityRule(
        "moray_firth_offshore",
        "nsl",
        "conditional",
        "same_coast_transfer",
        "medium",
        "Northern east-coast bias is more plausible here than a south-east export hop.",
    ),
    ReachabilityRule(
        "moray_firth_offshore",
        "britned",
        "stretched",
        "north_to_south_transfer",
        "low",
        "Dutch export only works after substantial internal transfer through GB bottlenecks.",
    ),
    ReachabilityRule(
        "moray_firth_offshore",
        "ifa",
        "stretched",
        "north_to_south_transfer",
        "low",
        "French export is geographically possible only after a long southbound transfer chain.",
    ),
    ReachabilityRule(
        "moray_firth_offshore",
        "ifa2",
        "stretched",
        "north_to_south_transfer",
        "low",
        "Treat as an even weaker version of the French route until detailed topology is added.",
    ),
    ReachabilityRule(
        "east_coast_scotland_offshore",
        "nsl",
        "conditional",
        "same_coast_transfer",
        "medium",
        "Still north-heavy, but closer to east-coast export corridors than Moray Firth.",
    ),
    ReachabilityRule(
        "east_coast_scotland_offshore",
        "britned",
        "conditional",
        "north_to_south_transfer",
        "medium",
        "Dutch export is plausible only if east-coast transfer is available.",
    ),
    ReachabilityRule(
        "east_coast_scotland_offshore",
        "nemo",
        "conditional",
        "north_to_south_transfer",
        "medium",
        "Belgian landing is a reasonable east-coast proxy once internal transfer is modeled.",
    ),
    ReachabilityRule(
        "east_coast_scotland_offshore",
        "ifa",
        "stretched",
        "north_to_south_transfer",
        "low",
        "French export remains a second-order candidate in this first-pass scaffold.",
    ),
    ReachabilityRule(
        "shetland_wind",
        "nsl",
        "upstream_dependency",
        "island_to_mainland_then_transfer",
        "low",
        "Any export path depends on mainland access before interconnector reachability even starts.",
    ),
    ReachabilityRule(
        "shetland_wind",
        "britned",
        "upstream_dependency",
        "island_to_mainland_then_transfer",
        "low",
        "Do not treat Dutch export as available until island-to-mainland transfer is explicit.",
    ),
    ReachabilityRule(
        "shetland_wind",
        "ifa",
        "upstream_dependency",
        "island_to_mainland_then_transfer",
        "low",
        "French export is structurally weaker than the already-unproven mainland path.",
    ),
    ReachabilityRule(
        "dogger_hornsea_offshore",
        "britned",
        "near",
        "east_coast_bias",
        "medium",
        "Best first-pass fit for Dutch export from a North Sea offshore cluster.",
    ),
    ReachabilityRule(
        "dogger_hornsea_offshore",
        "nemo",
        "conditional",
        "east_coast_transfer",
        "medium",
        "Belgian export is plausible through the same general east-coast corridor family.",
    ),
    ReachabilityRule(
        "dogger_hornsea_offshore",
        "viking_link",
        "conditional",
        "east_coast_transfer",
        "medium",
        "Useful future candidate if Nordic or Danish routing becomes relevant.",
    ),
    ReachabilityRule(
        "dogger_hornsea_offshore",
        "ifa",
        "stretched",
        "east_to_south_transfer",
        "low",
        "French export requires moving beyond the more natural east-coast hub set.",
    ),
    ReachabilityRule(
        "east_anglia_offshore",
        "britned",
        "near",
        "south_east_bias",
        "high",
        "Closest first-pass fit between a generation cluster and the Dutch-facing hub.",
    ),
    ReachabilityRule(
        "east_anglia_offshore",
        "nemo",
        "conditional",
        "south_east_bias",
        "medium",
        "Belgian export is plausible from the same broad coastal corridor.",
    ),
    ReachabilityRule(
        "east_anglia_offshore",
        "ifa",
        "conditional",
        "south_east_bias",
        "medium",
        "French-facing landings are more plausible here than for northern clusters.",
    ),
    ReachabilityRule(
        "east_anglia_offshore",
        "ifa2",
        "conditional",
        "south_to_south_coast_transfer",
        "medium",
        "Still a transfer problem, but materially more plausible than a Scottish route.",
    ),
    ReachabilityRule(
        "humber_offshore",
        "britned",
        "conditional",
        "east_coast_transfer",
        "medium",
        "Reasonable east-coast candidate, weaker than Dogger/Hornsea but still relevant.",
    ),
    ReachabilityRule(
        "humber_offshore",
        "nemo",
        "conditional",
        "east_coast_transfer",
        "medium",
        "Belgian landing is plausible with the same east-coast corridor assumption.",
    ),
    ReachabilityRule(
        "humber_offshore",
        "ifa",
        "stretched",
        "east_to_south_transfer",
        "low",
        "French export requires leaving the more natural east-coast route family.",
    ),
    ReachabilityRule(
        "north_wales_offshore",
        "ewic",
        "near",
        "irish_sea_bias",
        "medium",
        "Natural western export bias but outside the current FR/NL arbitrage model.",
    ),
    ReachabilityRule(
        "north_wales_offshore",
        "ifa",
        "stretched",
        "west_to_south_transfer",
        "low",
        "French export is not a first-pass natural fit for North Wales offshore generation.",
    ),
    ReachabilityRule(
        "north_wales_offshore",
        "britned",
        "stretched",
        "west_to_east_transfer",
        "low",
        "Dutch export is even weaker unless internal GB transfer is explicitly modeled.",
    ),
)


def interconnector_hub_frame() -> pd.DataFrame:
    return pd.DataFrame([hub.__dict__ for hub in INTERCONNECTOR_HUBS.values()]).sort_values("key").reset_index(drop=True)


def reachability_frame() -> pd.DataFrame:
    rows = []
    for rule in REACHABILITY_RULES:
        cluster = ASSET_CLUSTERS[rule.cluster_key]
        hub = INTERCONNECTOR_HUBS[rule.hub_key]
        rows.append(
            {
                "cluster_key": rule.cluster_key,
                "cluster_label": cluster.label,
                "parent_region": cluster.parent_region,
                "hub_key": rule.hub_key,
                "hub_label": hub.label,
                "target_zone": hub.target_zone,
                "status": rule.status,
                "transfer_requirement": rule.transfer_requirement,
                "confidence": rule.confidence,
                "note": rule.note,
            }
        )
    return pd.DataFrame(rows).sort_values(["parent_region", "cluster_key", "hub_key"]).reset_index(drop=True)


def route_hub_frame() -> pd.DataFrame:
    rows = []
    for route in ROUTE_HUB_OPTIONS:
        rows.append(
            {
                "route_name": route.route_name,
                "target_zone": route.target_zone,
                "preferred_hubs": ", ".join(route.preferred_hubs),
                "note": route.note,
            }
        )
    return pd.DataFrame(rows).sort_values("route_name").reset_index(drop=True)


def cluster_hub_matrix() -> pd.DataFrame:
    matrix = reachability_frame().pivot(index="cluster_key", columns="hub_key", values="status")
    return matrix.reset_index().sort_values("cluster_key").reset_index(drop=True)
