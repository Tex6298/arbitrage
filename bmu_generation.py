from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from asset_mapping import ASSET_CLUSTERS


ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc


class ElexonError(RuntimeError):
    pass


def _fetch_json(url: str) -> list[dict]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore").strip()
        detail = f": {body}" if body else ""
        raise ElexonError(f"Elexon request failed with HTTP {exc.code}{detail}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ElexonError(f"Elexon request failed: {reason}") from exc
    except TimeoutError as exc:
        raise ElexonError("Elexon request timed out") from exc
    except json.JSONDecodeError as exc:
        raise ElexonError("Elexon returned invalid JSON") from exc


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def _local_day_to_utc_window(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    start_local = dt.datetime.combine(day, dt.time.min, tzinfo=LONDON_TZ)
    end_local = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, tzinfo=LONDON_TZ)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def _rfc3339_utc(value: dt.datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _text_for_matching(row: pd.Series) -> str:
    return " | ".join(
        str(row.get(column, "") or "")
        for column in ("national_grid_bm_unit", "elexon_bm_unit", "bm_unit_name", "lead_party_name", "gsp_group_name")
    )


BMU_CLUSTER_RULES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    (
        "moray_firth_offshore",
        (r"\bBEATO-\d+\b", r"\bMOWEO-\d+\b", r"\bMOWWO-\d+\b", r"\bBEATRICE\b", r"\bMORAY\b"),
        "Direct BMU and name matches for Beatrice and Moray projects.",
    ),
    (
        "east_coast_scotland_offshore",
        (r"\bSGRWO-\d+\b", r"\bSEAGREEN\b", r"\bABRBO-\d+\b", r"\bABERDEEN OFFSHORE\b"),
        "Direct BMU and name matches for Seagreen and Aberdeen offshore projects.",
    ),
    (
        "shetland_wind",
        (r"\bVKNGW-\d+\b", r"\bVIKING ENERGY\b"),
        "Direct BMU and name matches for Viking Energy.",
    ),
    (
        "dogger_hornsea_offshore",
        (r"\bDBAWO-\d+\b", r"\bDBBWO-\d+\b", r"\bDBCWO-\d+\b", r"\bDOGGER BANK\b", r"\bHOWAO-\d+\b", r"\bHOWBO-\d+\b", r"\bHORNSEA\b"),
        "Direct BMU and name matches for Dogger Bank and Hornsea families.",
    ),
    (
        "east_anglia_offshore",
        (r"\bEAAO-\d+\b", r"\bEAST ANGLIA\b", r"\bGNFSW-\d+\b", r"\bGUNFLEET SANDS\b"),
        "Direct BMU and name matches for East Anglia and Gunfleet Sands offshore projects.",
    ),
    (
        "humber_offshore",
        (r"\bTKNEW-\d+\b", r"\bTKNWW-\d+\b", r"\bTRITON KNOLL\b", r"\bRCBKO-\d+\b", r"\bRACE BANK\b"),
        "Direct BMU and name matches for Triton Knoll and Race Bank.",
    ),
    (
        "north_wales_offshore",
        (r"\bGYMRO-\d+\b", r"\bGWYNT Y MOR\b", r"\bGYM OSP\b"),
        "Direct BMU and name matches for Gwynt y Mor.",
    ),
)

BMU_PARENT_REGION_RULES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    (
        "Scotland",
        (r"\bCLDCW-\d+\b", r"\bCLDNW-\d+\b", r"\bCLYDE (CENTRAL|NORTH)\b"),
        "Direct BMU and name matches for Clyde onshore wind; assign parent region without forcing a cluster.",
    ),
    (
        "England/Wales",
        (r"\bPNYCW-\d+\b", r"\bPEN Y CYMOEDD\b"),
        "Direct BMU and name matches for Pen y Cymoedd; assign parent region without forcing a cluster.",
    ),
)


def fetch_bmu_reference_all() -> pd.DataFrame:
    rows = _fetch_json(f"{ELEXON_BASE}/reference/bmunits/all")
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ElexonError("reference/bmunits/all returned no rows")

    frame = frame.rename(
        columns={
            "nationalGridBmUnit": "national_grid_bm_unit",
            "elexonBmUnit": "elexon_bm_unit",
            "bmUnitName": "bm_unit_name",
            "leadPartyName": "lead_party_name",
            "leadPartyId": "lead_party_id",
            "fuelType": "fuel_type",
            "bmUnitType": "bm_unit_type",
            "productionOrConsumptionFlag": "production_or_consumption_flag",
            "gspGroupId": "gsp_group_id",
            "gspGroupName": "gsp_group_name",
            "generationCapacity": "generation_capacity_mw",
            "demandCapacity": "demand_capacity_mw",
            "transmissionLossFactor": "transmission_loss_factor",
            "interconnectorId": "interconnector_id",
        }
    )
    for column in ("generation_capacity_mw", "demand_capacity_mw", "transmission_loss_factor"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_dim_bmu_asset(reference_frame: pd.DataFrame) -> pd.DataFrame:
    wind = reference_frame.copy()
    wind = wind[
        (wind["fuel_type"] == "WIND")
        & (wind["production_or_consumption_flag"] == "P")
    ].copy()
    if wind.empty:
        raise ElexonError("no wind production BMUs found in reference data")

    wind["source_key"] = "bmu_reference_all"
    wind["source_label"] = "Elexon BMU standing data"
    wind["target_is_proxy"] = False
    wind["mapping_status"] = "unmapped"
    wind["mapping_confidence"] = "none"
    wind["mapping_rule"] = ""
    wind["cluster_key"] = pd.NA
    wind["cluster_label"] = pd.NA
    wind["parent_region"] = pd.NA

    for cluster_key, patterns, rule_note in BMU_CLUSTER_RULES:
        cluster = ASSET_CLUSTERS[cluster_key]
        regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
        match_mask = wind.apply(lambda row: bool(regex.search(_text_for_matching(row))), axis=1)
        wind.loc[match_mask, "mapping_status"] = "mapped"
        wind.loc[match_mask, "mapping_confidence"] = "high"
        wind.loc[match_mask, "mapping_rule"] = rule_note
        wind.loc[match_mask, "cluster_key"] = cluster.key
        wind.loc[match_mask, "cluster_label"] = cluster.label
        wind.loc[match_mask, "parent_region"] = cluster.parent_region

    unmapped_mask = wind["mapping_status"].eq("unmapped")
    for parent_region, patterns, rule_note in BMU_PARENT_REGION_RULES:
        regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
        match_mask = unmapped_mask & wind.apply(lambda row: bool(regex.search(_text_for_matching(row))), axis=1)
        wind.loc[match_mask, "mapping_status"] = "region_only"
        wind.loc[match_mask, "mapping_confidence"] = "medium"
        wind.loc[match_mask, "mapping_rule"] = rule_note
        wind.loc[match_mask, "parent_region"] = parent_region
        unmapped_mask = wind["mapping_status"].eq("unmapped")

    wind = wind.drop_duplicates(subset=["elexon_bm_unit"], keep="first")
    wind = wind.sort_values(["mapping_status", "cluster_key", "national_grid_bm_unit", "elexon_bm_unit"], na_position="last")
    return wind.reset_index(drop=True)


def fetch_b1610_generation(
    elexon_bm_units: Sequence[str],
    start_date: dt.date,
    end_date: dt.date,
    batch_size: int = 25,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not elexon_bm_units:
        raise ElexonError("no BMUs provided for B1610 fetch")

    frames = []
    day = start_date
    while day <= end_date:
        start_utc, end_utc = _local_day_to_utc_window(day)
        for batch in _chunked(list(elexon_bm_units), batch_size):
            params = [
                ("from", _rfc3339_utc(start_utc)),
                ("to", _rfc3339_utc(end_utc)),
            ]
            params.extend(("bmUnit", bmu) for bmu in batch)
            url = f"{ELEXON_BASE}/datasets/B1610/stream?{urllib.parse.urlencode(params, doseq=True)}"
            rows = _fetch_json(url)
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["source_local_day"] = day.isoformat()
            frames.append(frame)
        day += dt.timedelta(days=1)

    if not frames:
        raise ElexonError(f"B1610 returned no rows for {start_date} to {end_date}")
    return pd.concat(frames, ignore_index=True)


def build_fact_bmu_generation_half_hourly(
    dim_bmu_asset: pd.DataFrame,
    raw_generation_frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = raw_generation_frame.rename(
        columns={
            "dataset": "source_dataset",
            "psrType": "psr_type",
            "bmUnit": "elexon_bm_unit",
            "nationalGridBmUnitId": "national_grid_bm_unit_from_fact",
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "halfHourEndTime": "half_hour_end_time_utc",
            "quantity": "generation_mwh",
        }
    ).copy()

    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"], errors="coerce").dt.date
    frame["settlement_period"] = pd.to_numeric(frame["settlement_period"], errors="coerce").astype("Int64")
    frame["half_hour_end_time_utc"] = pd.to_datetime(frame["half_hour_end_time_utc"], utc=True, errors="coerce")
    frame["half_hour_start_time_utc"] = frame["half_hour_end_time_utc"] - pd.Timedelta(minutes=30)
    frame["half_hour_start_time_local"] = frame["half_hour_start_time_utc"].dt.tz_convert("Europe/London")
    frame["half_hour_end_time_local"] = frame["half_hour_end_time_utc"].dt.tz_convert("Europe/London")
    frame["generation_mwh"] = pd.to_numeric(frame["generation_mwh"], errors="coerce")
    frame["source_key"] = "B1610"
    frame["source_label"] = "Elexon actual generation by BMU"
    frame["target_is_proxy"] = False

    dim_columns = [
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "bm_unit_name",
        "lead_party_name",
        "fuel_type",
        "bm_unit_type",
        "gsp_group_id",
        "gsp_group_name",
        "generation_capacity_mw",
        "mapping_status",
        "mapping_confidence",
        "mapping_rule",
        "cluster_key",
        "cluster_label",
        "parent_region",
    ]
    frame = frame.merge(dim_bmu_asset[dim_columns], on="elexon_bm_unit", how="left")
    frame["generation_mw_equivalent"] = frame["generation_mwh"] * 2.0
    frame["capacity_factor"] = frame["generation_mw_equivalent"] / frame["generation_capacity_mw"]
    frame.loc[frame["generation_capacity_mw"] <= 0, "capacity_factor"] = pd.NA
    frame = frame.sort_values(["half_hour_start_time_utc", "elexon_bm_unit"])
    frame = frame.drop_duplicates(subset=["settlement_date", "settlement_period", "elexon_bm_unit"], keep="last")

    keep_columns = [
        "settlement_date",
        "settlement_period",
        "half_hour_start_time_local",
        "half_hour_end_time_local",
        "half_hour_start_time_utc",
        "half_hour_end_time_utc",
        "source_key",
        "source_label",
        "source_dataset",
        "target_is_proxy",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "national_grid_bm_unit_from_fact",
        "bm_unit_name",
        "lead_party_name",
        "fuel_type",
        "bm_unit_type",
        "gsp_group_id",
        "gsp_group_name",
        "generation_capacity_mw",
        "generation_mwh",
        "generation_mw_equivalent",
        "capacity_factor",
        "mapping_status",
        "mapping_confidence",
        "mapping_rule",
        "cluster_key",
        "cluster_label",
        "parent_region",
    ]
    return frame[keep_columns].sort_values(["half_hour_start_time_utc", "elexon_bm_unit"]).reset_index(drop=True)


def materialize_bmu_generation_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    reference = fetch_bmu_reference_all()
    dim_bmu_asset = build_dim_bmu_asset(reference)
    raw_generation = fetch_b1610_generation(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    fact_bmu_generation_half_hourly = build_fact_bmu_generation_half_hourly(dim_bmu_asset, raw_generation)

    frames = {
        "dim_bmu_asset": dim_bmu_asset,
        "fact_bmu_generation_half_hourly": fact_bmu_generation_half_hourly,
    }

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)

    return frames
