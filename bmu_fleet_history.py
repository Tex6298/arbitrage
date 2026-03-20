from __future__ import annotations

import argparse
import datetime as dt
import re
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from asset_mapping import ASSET_CLUSTERS
from bmu_availability import (
    build_fact_bmu_availability_half_hourly,
    fetch_remit_event_detail_with_status,
    fetch_json,
)
from bmu_dispatch import (
    build_fact_bmu_acceptance_event,
    build_fact_bmu_bid_offer_half_hourly,
    build_fact_bmu_dispatch_acceptance_half_hourly,
    fetch_bod_bid_offers,
    fetch_boalf_acceptances,
)
from bmu_generation import (
    BMU_CLUSTER_RULES,
    BMU_PARENT_REGION_RULES,
    ELEXON_BASE,
    ElexonError,
    _chunked,
    build_fact_bmu_generation_half_hourly,
    fetch_b1610_generation,
    fetch_bmu_reference_all,
    parse_iso_date,
)


BMU_FLEET_DIM_TABLE = "dim_bmu_fleet_asset"
BMU_FLEET_GENERATION_TABLE = "fact_bmu_generation_half_hourly"
BMU_FLEET_ACCEPTANCE_EVENT_TABLE = "fact_bmu_acceptance_event"
BMU_FLEET_DISPATCH_TABLE = "fact_bmu_dispatch_acceptance_half_hourly"
BMU_FLEET_BID_OFFER_TABLE = "fact_bmu_bid_offer_half_hourly"
BMU_FLEET_AVAILABILITY_TABLE = "fact_bmu_availability_half_hourly"

SCOTLAND_GSP_IDS = {"_N", "_P"}


def _text_for_matching(row: pd.Series) -> str:
    return " | ".join(
        str(row.get(column, "") or "")
        for column in ("national_grid_bm_unit", "elexon_bm_unit", "bm_unit_name", "lead_party_name", "gsp_group_name")
    )


def _infer_parent_region_from_gsp(frame: pd.DataFrame) -> pd.Series:
    gsp_id = frame.get("gsp_group_id", pd.Series(pd.NA, index=frame.index)).fillna("").astype(str)
    gsp_name = frame.get("gsp_group_name", pd.Series(pd.NA, index=frame.index)).fillna("").astype(str).str.upper()
    scotland = gsp_id.isin(SCOTLAND_GSP_IDS) | gsp_name.str.contains("SCOTLAND", na=False)
    known = gsp_id.ne("") | gsp_name.ne("")
    return pd.Series(
        pd.NA,
        index=frame.index,
        dtype="object",
    ).mask(scotland, "Scotland").mask(~scotland & known, "England/Wales")


def build_dim_bmu_fleet_asset(reference_frame: pd.DataFrame) -> pd.DataFrame:
    fleet = reference_frame.copy()
    fleet = fleet[fleet["production_or_consumption_flag"].eq("P")].copy()
    if fleet.empty:
        raise ElexonError("no production BMUs found in reference data")

    fleet["source_key"] = "bmu_reference_all"
    fleet["source_label"] = "Elexon BMU standing data"
    fleet["target_is_proxy"] = False
    fleet["mapping_status"] = "unmapped"
    fleet["mapping_confidence"] = "none"
    fleet["mapping_rule"] = ""
    fleet["cluster_key"] = pd.NA
    fleet["cluster_label"] = pd.NA
    fleet["parent_region"] = pd.NA

    for cluster_key, patterns, rule_note in BMU_CLUSTER_RULES:
        cluster = ASSET_CLUSTERS[cluster_key]
        regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
        match_mask = fleet.apply(lambda row: bool(regex.search(_text_for_matching(row))), axis=1)
        fleet.loc[match_mask, "mapping_status"] = "mapped"
        fleet.loc[match_mask, "mapping_confidence"] = "high"
        fleet.loc[match_mask, "mapping_rule"] = rule_note
        fleet.loc[match_mask, "cluster_key"] = cluster.key
        fleet.loc[match_mask, "cluster_label"] = cluster.label
        fleet.loc[match_mask, "parent_region"] = cluster.parent_region

    unmapped_mask = fleet["mapping_status"].eq("unmapped")
    for parent_region, patterns, rule_note in BMU_PARENT_REGION_RULES:
        regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
        match_mask = unmapped_mask & fleet.apply(lambda row: bool(regex.search(_text_for_matching(row))), axis=1)
        fleet.loc[match_mask, "mapping_status"] = "region_only"
        fleet.loc[match_mask, "mapping_confidence"] = "medium"
        fleet.loc[match_mask, "mapping_rule"] = rule_note
        fleet.loc[match_mask, "parent_region"] = parent_region
        unmapped_mask = fleet["mapping_status"].eq("unmapped")

    fallback_parent_region = _infer_parent_region_from_gsp(fleet)
    fallback_mask = fleet["mapping_status"].eq("unmapped") & fallback_parent_region.notna()
    fleet.loc[fallback_mask, "mapping_status"] = "region_only"
    fleet.loc[fallback_mask, "mapping_confidence"] = "low"
    fleet.loc[fallback_mask, "mapping_rule"] = "Fallback parent-region classification from GSP group metadata."
    fleet.loc[fallback_mask, "parent_region"] = fallback_parent_region[fallback_mask]

    fleet = fleet.drop_duplicates(subset=["elexon_bm_unit"], keep="first")
    fleet = fleet.sort_values(
        ["mapping_status", "cluster_key", "parent_region", "fuel_type", "national_grid_bm_unit", "elexon_bm_unit"],
        na_position="last",
    )
    return fleet.reset_index(drop=True)


def fetch_uou2t14d_summary_all_fuels(
    elexon_bm_units: Sequence[str],
    batch_size: int = 50,
) -> pd.DataFrame:
    if not elexon_bm_units:
        raise ElexonError("no BMUs provided for UOU2T14D fetch")

    frames: list[pd.DataFrame] = []
    for batch in _chunked(list(elexon_bm_units), batch_size):
        params = [("bmUnit", bm_unit) for bm_unit in batch]
        url = f"{ELEXON_BASE}/datasets/UOU2T14D/stream?{urllib.parse.urlencode(params, doseq=True)}"
        rows = fetch_json(url)
        if not rows:
            continue
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def materialize_bmu_fleet_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    reference = fetch_bmu_reference_all()
    dim_bmu_asset = build_dim_bmu_fleet_asset(reference)
    elexon_bm_units = dim_bmu_asset["elexon_bm_unit"].dropna().astype(str).tolist()

    raw_generation = fetch_b1610_generation(elexon_bm_units, start_date, end_date)
    fact_bmu_generation_half_hourly = build_fact_bmu_generation_half_hourly(dim_bmu_asset, raw_generation)

    try:
        raw_acceptance = fetch_boalf_acceptances(elexon_bm_units, start_date, end_date)
    except Exception:
        raw_acceptance = pd.DataFrame()
    try:
        raw_bid_offer = fetch_bod_bid_offers(elexon_bm_units, start_date, end_date)
    except Exception:
        raw_bid_offer = pd.DataFrame()

    fact_bmu_acceptance_event = build_fact_bmu_acceptance_event(dim_bmu_asset, raw_acceptance)
    fact_bmu_dispatch_acceptance_half_hourly = build_fact_bmu_dispatch_acceptance_half_hourly(
        fact_bmu_acceptance_event,
        start_date=start_date,
        end_date=end_date,
    )
    fact_bmu_bid_offer_half_hourly = build_fact_bmu_bid_offer_half_hourly(dim_bmu_asset, raw_bid_offer)

    remit_fetch_ok = True
    remit_fetch_status = pd.DataFrame()
    try:
        raw_remit, remit_fetch_status = fetch_remit_event_detail_with_status(start_date, end_date)
        if not remit_fetch_status.empty:
            remit_fetch_ok = bool(remit_fetch_status["remit_fetch_ok"].fillna(False).all())
    except Exception:
        remit_fetch_ok = False
        raw_remit = pd.DataFrame()
        remit_fetch_status = pd.DataFrame()

    try:
        raw_uou = fetch_uou2t14d_summary_all_fuels(elexon_bm_units)
    except Exception:
        raw_uou = pd.DataFrame()

    fact_bmu_availability_half_hourly = build_fact_bmu_availability_half_hourly(
        dim_bmu_asset=dim_bmu_asset,
        raw_remit_frame=raw_remit,
        raw_uou_frame=raw_uou,
        start_date=start_date,
        end_date=end_date,
        remit_fetch_ok=remit_fetch_ok,
        remit_fetch_status_by_date=remit_fetch_status,
    )

    frames = {
        BMU_FLEET_DIM_TABLE: dim_bmu_asset,
        BMU_FLEET_GENERATION_TABLE: fact_bmu_generation_half_hourly,
        BMU_FLEET_ACCEPTANCE_EVENT_TABLE: fact_bmu_acceptance_event,
        BMU_FLEET_DISPATCH_TABLE: fact_bmu_dispatch_acceptance_half_hourly,
        BMU_FLEET_BID_OFFER_TABLE: fact_bmu_bid_offer_half_hourly,
        BMU_FLEET_AVAILABILITY_TABLE: fact_bmu_availability_half_hourly,
    }

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--output-dir", required=True, help="Output directory for all-fuel BMU fleet history")
    args = parser.parse_args()

    start_date = parse_iso_date(args.start)
    end_date = parse_iso_date(args.end)
    if end_date < start_date:
        raise SystemExit("--end must be on or after --start")

    frames = materialize_bmu_fleet_history(start_date=start_date, end_date=end_date, output_dir=args.output_dir)
    print(
        f"[source=bmu_fleet_history] Materialized {len(frames)} tables "
        f"start={start_date.isoformat()} end={end_date.isoformat()} output={args.output_dir}"
    )
    for table_name, frame in frames.items():
        print(f"{table_name}: rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
