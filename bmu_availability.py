from __future__ import annotations

import datetime as dt
import json
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from bmu_generation import ElexonError, build_dim_bmu_asset, fetch_bmu_reference_all
from bmu_truth_utils import (
    ELEXON_BASE,
    build_bmu_interval_spine,
    bmu_identifier_candidates,
    chunked,
    fetch_json,
    local_date_range_to_utc_window,
    overlap_hours,
    rfc3339_utc,
    unwrap_data_rows,
)


def _remit_detail_urls(start_date: dt.date, end_date: dt.date) -> Iterable[str]:
    day = start_date
    while day <= end_date:
        start_utc, end_utc = local_date_range_to_utc_window(day, day)
        url = (
            f"{ELEXON_BASE}/remit/list/by-event/stream?"
            f"{urllib.parse.urlencode({'from': rfc3339_utc(start_utc), 'to': rfc3339_utc(end_utc)})}"
        )
        rows = unwrap_data_rows(fetch_json(url))
        for row in rows:
            detail_url = row.get("url")
            if isinstance(detail_url, str) and detail_url.strip():
                yield detail_url.strip()
        day += dt.timedelta(days=1)


def fetch_remit_event_detail(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    frames = []
    for detail_url in dict.fromkeys(_remit_detail_urls(start_date, end_date)):
        rows = unwrap_data_rows(fetch_json(detail_url))
        if not rows:
            continue
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if "mrid" in frame.columns:
        frame["revisionNumber"] = pd.to_numeric(frame.get("revisionNumber"), errors="coerce")
        frame["publishTime"] = pd.to_datetime(frame.get("publishTime"), utc=True, errors="coerce")
        frame["createdTime"] = pd.to_datetime(frame.get("createdTime"), utc=True, errors="coerce")
        frame = frame.sort_values(["mrid", "revisionNumber", "publishTime", "createdTime"], na_position="last")
        frame = frame.drop_duplicates(subset=["mrid"], keep="last")
    return frame.reset_index(drop=True)


def fetch_uou2t14d_summary(
    elexon_bm_units: Sequence[str],
    batch_size: int = 50,
) -> pd.DataFrame:
    if not elexon_bm_units:
        raise ElexonError("no BMUs provided for UOU2T14D fetch")

    frames = []
    for batch in chunked(list(elexon_bm_units), batch_size):
        params = [("fuelType", "WIND")]
        params.extend(("bmUnit", bm_unit) for bm_unit in batch)
        url = f"{ELEXON_BASE}/datasets/UOU2T14D/stream?{urllib.parse.urlencode(params, doseq=True)}"
        rows = unwrap_data_rows(fetch_json(url))
        if not rows:
            continue
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_bmu_lookup(dim_bmu_asset: pd.DataFrame) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    for row in dim_bmu_asset.itertuples(index=False):
        for raw_value in (row.elexon_bm_unit, row.national_grid_bm_unit):
            for candidate in bmu_identifier_candidates(raw_value):
                lookup.setdefault(candidate, set()).add(row.elexon_bm_unit)
    return lookup


def _match_remit_bmus(row: pd.Series, lookup: dict[str, set[str]]) -> list[str]:
    matches: set[str] = set()
    for raw_value in (
        row.get("affected_unit"),
        row.get("asset_id"),
        row.get("registration_code"),
        row.get("affected_unit_eic"),
    ):
        for candidate in bmu_identifier_candidates(raw_value):
            matches.update(lookup.get(candidate, set()))
    return sorted(matches)


def _iter_remit_windows(row: pd.Series) -> Iterable[dict]:
    event_status = str(row.get("event_status") or "").strip().lower()
    if event_status == "dismissed":
        return []

    normal_capacity_mw = pd.to_numeric(pd.Series([row.get("normal_capacity_mw")]), errors="coerce").iloc[0]
    available_capacity_mw = pd.to_numeric(pd.Series([row.get("available_capacity_mw")]), errors="coerce").iloc[0]
    unavailable_capacity_mw = pd.to_numeric(pd.Series([row.get("unavailable_capacity_mw")]), errors="coerce").iloc[0]

    profile = row.get("outage_profile")
    windows = []
    if isinstance(profile, list) and profile:
        for segment in profile:
            if not isinstance(segment, dict):
                continue
            start_utc = pd.to_datetime(segment.get("startTime"), utc=True, errors="coerce")
            end_utc = pd.to_datetime(segment.get("endTime"), utc=True, errors="coerce")
            segment_capacity = pd.to_numeric(pd.Series([segment.get("capacity")]), errors="coerce").iloc[0]
            if pd.isna(start_utc) or pd.isna(end_utc) or end_utc <= start_utc:
                continue
            if pd.notna(normal_capacity_mw) and pd.notna(segment_capacity) and segment_capacity + 1e-6 >= normal_capacity_mw:
                continue
            windows.append(
                {
                    "window_start_utc": start_utc,
                    "window_end_utc": end_utc,
                    "window_capacity_mw": segment_capacity,
                    "normal_capacity_mw": normal_capacity_mw,
                    "available_capacity_mw": available_capacity_mw,
                    "unavailable_capacity_mw": unavailable_capacity_mw,
                }
            )

    if windows:
        return windows

    start_utc = pd.to_datetime(row.get("event_start_utc"), utc=True, errors="coerce")
    end_utc = pd.to_datetime(row.get("event_end_utc"), utc=True, errors="coerce")
    if pd.isna(start_utc) or pd.isna(end_utc) or end_utc <= start_utc:
        return []
    if pd.notna(unavailable_capacity_mw) and unavailable_capacity_mw <= 0 and pd.notna(available_capacity_mw):
        return []
    return [
        {
            "window_start_utc": start_utc,
            "window_end_utc": end_utc,
            "window_capacity_mw": available_capacity_mw,
            "normal_capacity_mw": normal_capacity_mw,
            "available_capacity_mw": available_capacity_mw,
            "unavailable_capacity_mw": unavailable_capacity_mw,
        }
    ]


def build_fact_bmu_availability_half_hourly(
    dim_bmu_asset: pd.DataFrame,
    raw_remit_frame: pd.DataFrame,
    raw_uou_frame: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    remit_fetch_ok: bool,
) -> pd.DataFrame:
    spine = build_bmu_interval_spine(dim_bmu_asset, start_date, end_date)
    frame = spine.copy()
    frame["source_key"] = "REMIT|UOU2T14D"
    frame["source_label"] = "REMIT outage windows with UOU availability QA"
    frame["target_is_proxy"] = False
    frame["remit_fetch_ok"] = remit_fetch_ok

    remit_interval_rows = []
    if remit_fetch_ok and not raw_remit_frame.empty:
        lookup = _build_bmu_lookup(dim_bmu_asset)
        remit = raw_remit_frame.rename(
            columns={
                "mrid": "mrid",
                "revisionNumber": "revision_number",
                "publishTime": "publish_time_utc",
                "eventStatus": "event_status",
                "eventStartTime": "event_start_utc",
                "eventEndTime": "event_end_utc",
                "affectedUnit": "affected_unit",
                "affectedUnitEIC": "affected_unit_eic",
                "assetId": "asset_id",
                "registrationCode": "registration_code",
                "outageProfile": "outage_profile",
                "normalCapacity": "normal_capacity_mw",
                "availableCapacity": "available_capacity_mw",
                "unavailableCapacity": "unavailable_capacity_mw",
                "messageType": "message_type",
                "eventType": "event_type",
                "unavailabilityType": "unavailability_type",
                "fuelType": "fuel_type_from_remit",
            }
        ).copy()
        remit["publish_time_utc"] = pd.to_datetime(remit["publish_time_utc"], utc=True, errors="coerce")
        remit["event_start_utc"] = pd.to_datetime(remit["event_start_utc"], utc=True, errors="coerce")
        remit["event_end_utc"] = pd.to_datetime(remit["event_end_utc"], utc=True, errors="coerce")
        remit["normal_capacity_mw"] = pd.to_numeric(remit["normal_capacity_mw"], errors="coerce")
        remit["available_capacity_mw"] = pd.to_numeric(remit["available_capacity_mw"], errors="coerce")
        remit["unavailable_capacity_mw"] = pd.to_numeric(remit["unavailable_capacity_mw"], errors="coerce")

        interval_index = spine[["settlement_date", "settlement_period", "interval_start_utc", "interval_end_utc"]].drop_duplicates()
        for row in remit.itertuples(index=False):
            series = pd.Series(row._asdict())
            matched_bmus = _match_remit_bmus(series, lookup)
            if not matched_bmus:
                continue
            windows = list(_iter_remit_windows(series))
            if not windows:
                continue
            for bm_unit in matched_bmus:
                matching_intervals = interval_index[
                    (interval_index["interval_end_utc"] > min(window["window_start_utc"] for window in windows))
                    & (interval_index["interval_start_utc"] < max(window["window_end_utc"] for window in windows))
                ]
                for interval in matching_intervals.itertuples(index=False):
                    total_overlap_hours = 0.0
                    max_available_capacity_mw = np.nan
                    max_unavailable_capacity_mw = np.nan
                    max_normal_capacity_mw = np.nan
                    for window in windows:
                        window_overlap = overlap_hours(
                            interval.interval_start_utc,
                            interval.interval_end_utc,
                            window["window_start_utc"],
                            window["window_end_utc"],
                        )
                        if window_overlap <= 0:
                            continue
                        total_overlap_hours += window_overlap
                        max_available_capacity_mw = np.nanmax([max_available_capacity_mw, window["available_capacity_mw"]])
                        max_unavailable_capacity_mw = np.nanmax([max_unavailable_capacity_mw, window["unavailable_capacity_mw"]])
                        max_normal_capacity_mw = np.nanmax([max_normal_capacity_mw, window["normal_capacity_mw"]])
                    if total_overlap_hours <= 0:
                        continue
                    remit_interval_rows.append(
                        {
                            "settlement_date": interval.settlement_date,
                            "settlement_period": interval.settlement_period,
                            "elexon_bm_unit": bm_unit,
                            "remit_event_count": 1,
                            "remit_overlap_hours": total_overlap_hours,
                            "remit_active_flag": True,
                            "remit_max_available_capacity_mw": max_available_capacity_mw,
                            "remit_max_unavailable_capacity_mw": max_unavailable_capacity_mw,
                            "remit_normal_capacity_mw": max_normal_capacity_mw,
                            "latest_remit_publish_time_utc": series.get("publish_time_utc"),
                        }
                    )

    remit_interval_frame = pd.DataFrame(remit_interval_rows)
    if remit_interval_frame.empty:
        remit_interval_frame = pd.DataFrame(
            columns=[
                "settlement_date",
                "settlement_period",
                "elexon_bm_unit",
                "remit_event_count",
                "remit_overlap_hours",
                "remit_active_flag",
                "remit_max_available_capacity_mw",
                "remit_max_unavailable_capacity_mw",
                "remit_normal_capacity_mw",
                "latest_remit_publish_time_utc",
            ]
        )
    else:
        remit_interval_frame = (
            remit_interval_frame.groupby(["settlement_date", "settlement_period", "elexon_bm_unit"], as_index=False, dropna=False)
            .agg(
                remit_event_count=("remit_event_count", "sum"),
                remit_overlap_hours=("remit_overlap_hours", "sum"),
                remit_active_flag=("remit_active_flag", "max"),
                remit_max_available_capacity_mw=("remit_max_available_capacity_mw", "max"),
                remit_max_unavailable_capacity_mw=("remit_max_unavailable_capacity_mw", "max"),
                remit_normal_capacity_mw=("remit_normal_capacity_mw", "max"),
                latest_remit_publish_time_utc=("latest_remit_publish_time_utc", "max"),
            )
        )

    frame = frame.merge(
        remit_interval_frame,
        on=["settlement_date", "settlement_period", "elexon_bm_unit"],
        how="left",
    )
    frame["remit_active_flag"] = frame["remit_active_flag"].where(frame["remit_active_flag"].notna(), False).astype(bool)
    frame["remit_event_count"] = pd.to_numeric(frame["remit_event_count"], errors="coerce").fillna(0).astype(int)
    frame["remit_overlap_hours"] = pd.to_numeric(frame["remit_overlap_hours"], errors="coerce")
    frame["remit_max_available_capacity_mw"] = pd.to_numeric(frame["remit_max_available_capacity_mw"], errors="coerce")
    frame["remit_max_unavailable_capacity_mw"] = pd.to_numeric(frame["remit_max_unavailable_capacity_mw"], errors="coerce")
    frame["remit_normal_capacity_mw"] = pd.to_numeric(frame["remit_normal_capacity_mw"], errors="coerce")
    frame["remit_partial_availability_flag"] = (
        frame["remit_active_flag"]
        & frame["remit_max_available_capacity_mw"].notna()
        & (frame["remit_max_available_capacity_mw"] > 0)
    )

    uou_frame = pd.DataFrame()
    if not raw_uou_frame.empty:
        uou_frame = raw_uou_frame.rename(
            columns={
                "dataset": "uou_source_dataset",
                "bmUnit": "elexon_bm_unit",
                "nationalGridBmUnit": "national_grid_bm_unit_from_fact",
                "publishTime": "uou_publish_time_utc",
                "forecastDate": "forecast_date",
                "outputUsable": "uou_output_usable_mw",
            }
        ).copy()
        uou_frame["forecast_date"] = pd.to_datetime(uou_frame["forecast_date"], errors="coerce").dt.date
        uou_frame["uou_publish_time_utc"] = pd.to_datetime(uou_frame["uou_publish_time_utc"], utc=True, errors="coerce")
        uou_frame["uou_output_usable_mw"] = pd.to_numeric(uou_frame["uou_output_usable_mw"], errors="coerce")
        uou_frame = (
            uou_frame.sort_values(["elexon_bm_unit", "forecast_date", "uou_publish_time_utc"])
            .drop_duplicates(subset=["elexon_bm_unit", "forecast_date"], keep="last")
        )
        uou_frame = uou_frame[["elexon_bm_unit", "forecast_date", "uou_publish_time_utc", "uou_output_usable_mw"]]

    if uou_frame.empty:
        frame["uou_publish_time_utc"] = pd.NaT
        frame["uou_output_usable_mw"] = np.nan
    else:
        frame = frame.merge(
            uou_frame,
            left_on=["elexon_bm_unit", "settlement_date"],
            right_on=["elexon_bm_unit", "forecast_date"],
            how="left",
        ).drop(columns="forecast_date")

    frame["availability_state"] = "unknown"
    if remit_fetch_ok:
        frame["availability_state"] = "available"
    frame.loc[frame["remit_active_flag"], "availability_state"] = "outage"
    frame.loc[frame["remit_partial_availability_flag"], "availability_state"] = "unknown"

    frame["availability_confidence"] = "low"
    if remit_fetch_ok:
        frame["availability_confidence"] = "medium"
    frame.loc[frame["remit_active_flag"], "availability_confidence"] = "high"
    frame.loc[frame["remit_partial_availability_flag"], "availability_confidence"] = "medium"
    frame.loc[
        (frame["availability_state"] == "available") & frame["uou_output_usable_mw"].notna() & (frame["uou_output_usable_mw"] > 0),
        "availability_confidence",
    ] = "high"

    keep_columns = [
        "settlement_date",
        "settlement_period",
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "source_key",
        "source_label",
        "target_is_proxy",
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
        "remit_fetch_ok",
        "remit_event_count",
        "remit_overlap_hours",
        "remit_active_flag",
        "remit_partial_availability_flag",
        "remit_max_available_capacity_mw",
        "remit_max_unavailable_capacity_mw",
        "remit_normal_capacity_mw",
        "latest_remit_publish_time_utc",
        "uou_publish_time_utc",
        "uou_output_usable_mw",
        "availability_state",
        "availability_confidence",
    ]
    return frame[keep_columns].sort_values(["interval_start_utc", "elexon_bm_unit"]).reset_index(drop=True)


def materialize_bmu_availability_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    reference = fetch_bmu_reference_all()
    dim_bmu_asset = build_dim_bmu_asset(reference)

    remit_fetch_ok = True
    try:
        raw_remit = fetch_remit_event_detail(start_date, end_date)
    except Exception:
        remit_fetch_ok = False
        raw_remit = pd.DataFrame()

    try:
        raw_uou = fetch_uou2t14d_summary(dim_bmu_asset["elexon_bm_unit"].tolist())
    except Exception:
        raw_uou = pd.DataFrame()

    fact_bmu_availability_half_hourly = build_fact_bmu_availability_half_hourly(
        dim_bmu_asset=dim_bmu_asset,
        raw_remit_frame=raw_remit,
        raw_uou_frame=raw_uou,
        start_date=start_date,
        end_date=end_date,
        remit_fetch_ok=remit_fetch_ok,
    )

    frames = {
        "fact_bmu_availability_half_hourly": fact_bmu_availability_half_hourly,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)
    return frames
