from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class AssetAnchor:
    key: str
    label: str
    latitude: float
    longitude: float
    capacity_mw: float
    note: str


@dataclass(frozen=True)
class AssetCluster:
    key: str
    label: str
    parent_region: str
    anchor_keys: Tuple[str, ...]
    grid_note: str
    curtailment_note: str
    weather_note: str
    confidence: str


@dataclass(frozen=True)
class SignalSource:
    key: str
    label: str
    granularity: str
    coverage: str
    role: str
    status: str
    note: str


PARENT_REGIONS: Dict[str, str] = {
    "Scotland": "Wind-heavy clusters with the highest north-to-south transfer risk.",
    "England/Wales": "Wind-heavy clusters with better proximity to south-east and east-coast export landings.",
}


# Seed registry from user-provided notes. Treat coordinates and capacities as approximate
# placeholders until replaced with a curated asset and connection-point register.
ASSET_ANCHORS: Dict[str, AssetAnchor] = {
    "beatrice": AssetAnchor("beatrice", "Beatrice", 58.1, -2.6, 588.0, "Approximate offshore anchor."),
    "moray_east": AssetAnchor("moray_east", "Moray East", 57.7, -2.3, 950.0, "Approximate offshore anchor."),
    "seagreen": AssetAnchor("seagreen", "Seagreen", 56.6, -1.9, 1075.0, "Approximate offshore anchor."),
    "viking": AssetAnchor("viking", "Viking", 60.3, -1.2, 443.0, "Approximate onshore Shetland anchor."),
    "hornsea_one": AssetAnchor("hornsea_one", "Hornsea One", 53.8, 0.3, 1218.0, "Approximate offshore anchor."),
    "hornsea_two": AssetAnchor("hornsea_two", "Hornsea Two", 54.0, 0.2, 1386.0, "Approximate offshore anchor."),
    "dogger_bank": AssetAnchor("dogger_bank", "Dogger Bank Cluster", 54.6, 0.9, 2400.0, "Aggregate placeholder anchor."),
    "east_anglia_one": AssetAnchor("east_anglia_one", "East Anglia One", 52.6, 1.7, 1020.0, "Approximate offshore anchor."),
    "gwynt_y_mor": AssetAnchor("gwynt_y_mor", "Gwynt y Mor", 53.4, -3.7, 576.0, "Approximate offshore anchor."),
    "triton_knoll": AssetAnchor("triton_knoll", "Triton Knoll", 53.3, 0.6, 857.0, "Approximate offshore anchor."),
}


ASSET_CLUSTERS: Dict[str, AssetCluster] = {
    "moray_firth_offshore": AssetCluster(
        key="moray_firth_offshore",
        label="Moray Firth Offshore",
        parent_region="Scotland",
        anchor_keys=("beatrice", "moray_east"),
        grid_note="North-east Scotland offshore complex with strong curtailment relevance.",
        curtailment_note="High-value cluster for daily curtailment and weather overlay work.",
        weather_note="Use capacity-weighted offshore weather anchors.",
        confidence="medium",
    ),
    "east_coast_scotland_offshore": AssetCluster(
        key="east_coast_scotland_offshore",
        label="East Coast Scotland Offshore",
        parent_region="Scotland",
        anchor_keys=("seagreen",),
        grid_note="East coast offshore generation south of Moray but still exposed to internal GB transfer constraints.",
        curtailment_note="Important when Scottish wind is high but southern transfer is tight.",
        weather_note="Single-anchor seed; expand to a fuller east-coast registry later.",
        confidence="medium",
    ),
    "shetland_wind": AssetCluster(
        key="shetland_wind",
        label="Shetland Wind",
        parent_region="Scotland",
        anchor_keys=("viking",),
        grid_note="Island-to-mainland dependency makes export reachability highly conditional.",
        curtailment_note="Useful as a stress-test cluster because upstream transfer matters before interconnector access.",
        weather_note="Single-anchor seed; should later include measured wind and cable status.",
        confidence="low",
    ),
    "dogger_hornsea_offshore": AssetCluster(
        key="dogger_hornsea_offshore",
        label="Dogger and Hornsea Offshore",
        parent_region="England/Wales",
        anchor_keys=("hornsea_one", "hornsea_two", "dogger_bank"),
        grid_note="North Sea cluster with better alignment to east-coast export assets.",
        curtailment_note="Candidate cluster for east-coast battery siting and Dutch/Belgian route screening.",
        weather_note="Use capacity-weighted offshore weather anchors.",
        confidence="medium",
    ),
    "east_anglia_offshore": AssetCluster(
        key="east_anglia_offshore",
        label="East Anglia Offshore",
        parent_region="England/Wales",
        anchor_keys=("east_anglia_one",),
        grid_note="South-east facing offshore cluster close to Dutch and Belgian landing bias.",
        curtailment_note="Good proxy cluster for south-east export opportunities.",
        weather_note="Single-anchor seed; expand with additional East Anglia projects later.",
        confidence="medium",
    ),
    "humber_offshore": AssetCluster(
        key="humber_offshore",
        label="Humber Offshore",
        parent_region="England/Wales",
        anchor_keys=("triton_knoll",),
        grid_note="East-coast offshore cluster between Humber and wider North Sea export corridors.",
        curtailment_note="Useful bridge cluster between Dogger/Hornsea and south-east landings.",
        weather_note="Single-anchor seed; suitable for a first-pass regional weather feature.",
        confidence="low",
    ),
    "north_wales_offshore": AssetCluster(
        key="north_wales_offshore",
        label="North Wales Offshore",
        parent_region="England/Wales",
        anchor_keys=("gwynt_y_mor",),
        grid_note="Irish Sea cluster with weak fit to the current FR/NL route model.",
        curtailment_note="Useful counterexample cluster when checking whether a route is even geographically plausible.",
        weather_note="Single-anchor seed; should later include Irish Sea-specific assets.",
        confidence="low",
    ),
}


SIGNAL_SOURCES: Tuple[SignalSource, ...] = (
    SignalSource(
        key="gb_mid_price",
        label="Elexon MID price",
        granularity="half-hourly -> hourly",
        coverage="GB-wide",
        role="Tradable GB price proxy used by the live arbitrage script.",
        status="implemented",
        note="Useful for spreads, not sufficient as curtailment ground truth.",
    ),
    SignalSource(
        key="constraint_breakdown",
        label="NESO constraint breakdown",
        granularity="daily",
        coverage="GB-wide",
        role="Curtailment cost and volume labels for training or backtesting.",
        status="implemented",
        note="Good daily truth set, but not locational and not intraday.",
    ),
    SignalSource(
        key="wind_split",
        label="NESO metered wind split",
        granularity="half-hourly",
        coverage="Scotland vs England/Wales",
        role="Regional allocation weights for decomposing GB-wide signals.",
        status="implemented",
        note="Useful bridge signal before asset-level curtailment is available.",
    ),
    SignalSource(
        key="anchor_weather",
        label="Capacity-weighted weather anchors",
        granularity="hourly",
        coverage="Cluster and region level",
        role="Forecast and actual weather features for the physical side of the model.",
        status="implemented_first_pass",
        note="Observed history now materializes from anchor points; forecast surfaces and richer offshore refinement still come later.",
    ),
    SignalSource(
        key="connector_capacity",
        label="Interconnector and internal transfer capacity",
        granularity="hourly",
        coverage="Hub and corridor level",
        role="Physical feasibility gate for route scoring.",
        status="missing",
        note="Still required before route scores can be treated as operational decisions.",
    ),
)


def _cluster_capacity(anchor_keys: Sequence[str]) -> float:
    return sum(ASSET_ANCHORS[key].capacity_mw for key in anchor_keys)


def _cluster_centroid(anchor_keys: Sequence[str]) -> Tuple[float, float]:
    total_capacity = _cluster_capacity(anchor_keys)
    if total_capacity <= 0:
        raise ValueError("cluster capacity must be positive")

    latitude = sum(ASSET_ANCHORS[key].latitude * ASSET_ANCHORS[key].capacity_mw for key in anchor_keys) / total_capacity
    longitude = sum(ASSET_ANCHORS[key].longitude * ASSET_ANCHORS[key].capacity_mw for key in anchor_keys) / total_capacity
    return latitude, longitude


def weather_anchor_frame() -> pd.DataFrame:
    return pd.DataFrame([anchor.__dict__ for anchor in ASSET_ANCHORS.values()]).sort_values("key").reset_index(drop=True)


def cluster_frame() -> pd.DataFrame:
    rows = []
    for cluster in ASSET_CLUSTERS.values():
        latitude, longitude = _cluster_centroid(cluster.anchor_keys)
        rows.append(
            {
                "cluster_key": cluster.key,
                "cluster_label": cluster.label,
                "parent_region": cluster.parent_region,
                "anchor_count": len(cluster.anchor_keys),
                "approx_capacity_mw": _cluster_capacity(cluster.anchor_keys),
                "centroid_latitude": round(latitude, 4),
                "centroid_longitude": round(longitude, 4),
                "anchors": ", ".join(cluster.anchor_keys),
                "grid_note": cluster.grid_note,
                "curtailment_note": cluster.curtailment_note,
                "weather_note": cluster.weather_note,
                "confidence": cluster.confidence,
            }
        )
    return pd.DataFrame(rows).sort_values(["parent_region", "cluster_key"]).reset_index(drop=True)


def parent_region_frame() -> pd.DataFrame:
    cluster_rows = cluster_frame()
    rows = []
    for region, description in PARENT_REGIONS.items():
        subset = cluster_rows[cluster_rows["parent_region"] == region]
        rows.append(
            {
                "parent_region": region,
                "cluster_count": int(len(subset)),
                "approx_capacity_mw": float(subset["approx_capacity_mw"].sum()),
                "description": description,
            }
        )
    return pd.DataFrame(rows).sort_values("parent_region").reset_index(drop=True)


def signal_source_frame() -> pd.DataFrame:
    return pd.DataFrame([signal.__dict__ for signal in SIGNAL_SOURCES]).sort_values("key").reset_index(drop=True)
