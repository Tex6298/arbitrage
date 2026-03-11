from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from bmu_generation import ElexonError, build_dim_bmu_asset, fetch_bmu_reference_all, parse_iso_date


ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc
SENTINEL_BID_FLOOR_GBP_PER_MWH = -9999.0
SENTINEL_OFFER_CEILING_GBP_PER_MWH = 9999.0


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


def _local_day_to_utc_window(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    start_local = dt.datetime.combine(day, dt.time.min, tzinfo=LONDON_TZ)
    end_local = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, tzinfo=LONDON_TZ)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def _rfc3339_utc(value: dt.datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def clip_raw_dispatch_rows_to_requested_window(
    frame: pd.DataFrame,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    prepared = frame.copy()
    if {"timeFrom", "timeTo"}.issubset(prepared.columns):
        time_from = pd.to_datetime(prepared["timeFrom"], utc=True, errors="coerce")
        time_to = pd.to_datetime(prepared["timeTo"], utc=True, errors="coerce")
        overlap_mask = time_to.gt(pd.Timestamp(start_utc)) & time_from.lt(pd.Timestamp(end_utc))
        prepared = prepared[overlap_mask].copy()

    if "settlementDate" in prepared.columns:
        settlement_date = pd.to_datetime(prepared["settlementDate"], errors="coerce").dt.date
        prepared = prepared[settlement_date.notna()].copy()

    return prepared.reset_index(drop=True)


def _build_half_hour_interval_frame(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    rows = []
    day = start_date
    while day <= end_date:
        start_local = dt.datetime.combine(day, dt.time.min, tzinfo=LONDON_TZ)
        end_local = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, tzinfo=LONDON_TZ)
        current_utc = start_local.astimezone(UTC)
        day_end_utc = end_local.astimezone(UTC)
        settlement_period = 1
        while current_utc < day_end_utc:
            next_utc = current_utc + dt.timedelta(minutes=30)
            rows.append(
                {
                    "settlement_date": day,
                    "settlement_period": settlement_period,
                    "interval_start_utc": current_utc,
                    "interval_end_utc": next_utc,
                    "interval_start_local": current_utc.astimezone(LONDON_TZ),
                    "interval_end_local": next_utc.astimezone(LONDON_TZ),
                }
            )
            current_utc = next_utc
            settlement_period += 1
        day += dt.timedelta(days=1)

    return pd.DataFrame(rows)


def _overlap_hours(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> float:
    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    seconds = (overlap_end - overlap_start).total_seconds()
    return max(seconds, 0.0) / 3600.0


def _coerce_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def fetch_boalf_acceptances(
    elexon_bm_units: Sequence[str],
    start_date: dt.date,
    end_date: dt.date,
    batch_size: int = 25,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not elexon_bm_units:
        raise ElexonError("no BMUs provided for BOALF fetch")

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
            url = f"{ELEXON_BASE}/datasets/BOALF/stream?{urllib.parse.urlencode(params, doseq=True)}"
            rows = _fetch_json(url)
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame = clip_raw_dispatch_rows_to_requested_window(frame, start_utc=start_utc, end_utc=end_utc)
            if frame.empty:
                continue
            frame["source_local_day"] = day.isoformat()
            frames.append(frame)
        day += dt.timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_bod_bid_offers(
    elexon_bm_units: Sequence[str],
    start_date: dt.date,
    end_date: dt.date,
    batch_size: int = 25,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not elexon_bm_units:
        raise ElexonError("no BMUs provided for BOD fetch")

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
            url = f"{ELEXON_BASE}/datasets/BOD/stream?{urllib.parse.urlencode(params, doseq=True)}"
            rows = _fetch_json(url)
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame = clip_raw_dispatch_rows_to_requested_window(frame, start_utc=start_utc, end_utc=end_utc)
            if frame.empty:
                continue
            frame["source_local_day"] = day.isoformat()
            frames.append(frame)
        day += dt.timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_fact_bmu_acceptance_event(
    dim_bmu_asset: pd.DataFrame,
    raw_acceptance_frame: pd.DataFrame,
) -> pd.DataFrame:
    keep_columns = [
        "settlement_date",
        "settlement_period_from",
        "settlement_period_to",
        "time_from_utc",
        "time_to_utc",
        "time_from_local",
        "time_to_local",
        "acceptance_time_utc",
        "acceptance_time_local",
        "duration_hours",
        "duration_minutes",
        "source_key",
        "source_label",
        "source_dataset",
        "target_is_proxy",
        "dispatch_truth_tier",
        "is_lower_bound_metric",
        "acceptance_number",
        "amendment_flag",
        "deemed_bo_flag",
        "so_flag",
        "stor_flag",
        "rr_flag",
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
        "level_from_mw",
        "level_to_mw",
        "level_delta_mw",
        "accepted_level_mean_mw",
        "accepted_down_delta_mw",
        "accepted_up_delta_mw",
        "accepted_down_delta_mwh_lower_bound",
        "accepted_up_delta_mwh_lower_bound",
        "dispatch_direction",
        "mapping_status",
        "mapping_confidence",
        "mapping_rule",
        "cluster_key",
        "cluster_label",
        "parent_region",
    ]
    if raw_acceptance_frame.empty:
        return pd.DataFrame(columns=keep_columns)

    frame = raw_acceptance_frame.rename(
        columns={
            "dataset": "source_dataset",
            "settlementDate": "settlement_date",
            "settlementPeriodFrom": "settlement_period_from",
            "settlementPeriodTo": "settlement_period_to",
            "timeFrom": "time_from_utc",
            "timeTo": "time_to_utc",
            "levelFrom": "level_from_mw",
            "levelTo": "level_to_mw",
            "acceptanceTime": "acceptance_time_utc",
            "acceptanceNumber": "acceptance_number",
            "amendmentFlag": "amendment_flag",
            "deemedBoFlag": "deemed_bo_flag",
            "soFlag": "so_flag",
            "storFlag": "stor_flag",
            "rrFlag": "rr_flag",
            "bmUnit": "elexon_bm_unit",
            "nationalGridBmUnit": "national_grid_bm_unit_from_fact",
        }
    ).copy()

    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"], errors="coerce").dt.date
    frame["settlement_period_from"] = pd.to_numeric(frame["settlement_period_from"], errors="coerce").astype("Int64")
    frame["settlement_period_to"] = pd.to_numeric(frame["settlement_period_to"], errors="coerce").astype("Int64")
    frame["time_from_utc"] = pd.to_datetime(frame["time_from_utc"], utc=True, errors="coerce")
    frame["time_to_utc"] = pd.to_datetime(frame["time_to_utc"], utc=True, errors="coerce")
    frame["time_from_local"] = frame["time_from_utc"].dt.tz_convert("Europe/London")
    frame["time_to_local"] = frame["time_to_utc"].dt.tz_convert("Europe/London")
    frame["acceptance_time_utc"] = pd.to_datetime(frame["acceptance_time_utc"], utc=True, errors="coerce")
    frame["acceptance_time_local"] = frame["acceptance_time_utc"].dt.tz_convert("Europe/London")
    frame["duration_hours"] = (frame["time_to_utc"] - frame["time_from_utc"]).dt.total_seconds() / 3600.0
    frame["duration_minutes"] = frame["duration_hours"] * 60.0
    frame["level_from_mw"] = pd.to_numeric(frame["level_from_mw"], errors="coerce")
    frame["level_to_mw"] = pd.to_numeric(frame["level_to_mw"], errors="coerce")
    frame["level_delta_mw"] = frame["level_to_mw"] - frame["level_from_mw"]
    frame["accepted_level_mean_mw"] = (frame["level_from_mw"] + frame["level_to_mw"]) / 2.0
    frame["accepted_down_delta_mw"] = (frame["level_from_mw"] - frame["level_to_mw"]).clip(lower=0.0)
    frame["accepted_up_delta_mw"] = (frame["level_to_mw"] - frame["level_from_mw"]).clip(lower=0.0)
    frame["accepted_down_delta_mwh_lower_bound"] = frame["accepted_down_delta_mw"] * frame["duration_hours"]
    frame["accepted_up_delta_mwh_lower_bound"] = frame["accepted_up_delta_mw"] * frame["duration_hours"]
    frame["dispatch_direction"] = np.select(
        [frame["accepted_down_delta_mw"] > 0, frame["accepted_up_delta_mw"] > 0],
        ["down", "up"],
        default="flat",
    )
    frame["source_key"] = "BOALF"
    frame["source_label"] = "Elexon bid-offer acceptance levels"
    frame["target_is_proxy"] = False
    frame["dispatch_truth_tier"] = "dispatch_acceptance_lower_bound"
    frame["is_lower_bound_metric"] = True

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
    return frame[keep_columns].sort_values(["time_from_utc", "elexon_bm_unit", "acceptance_number"]).reset_index(drop=True)


def build_fact_bmu_dispatch_acceptance_half_hourly(
    fact_bmu_acceptance_event: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
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
        "dispatch_truth_tier",
        "is_lower_bound_metric",
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
        "acceptance_event_count",
        "distinct_acceptance_number_count",
        "accepted_down_event_count",
        "accepted_up_event_count",
        "dispatch_down_flag",
        "dispatch_up_flag",
        "overlap_hours",
        "overlap_minutes",
        "accepted_level_time_weighted_mwh",
        "accepted_level_time_weighted_mean_mw",
        "accepted_down_delta_mwh_lower_bound",
        "accepted_up_delta_mwh_lower_bound",
        "max_accepted_down_delta_mw",
        "max_accepted_up_delta_mw",
        "any_deemed_bo_flag",
        "any_so_flag",
        "any_stor_flag",
        "any_rr_flag",
        "earliest_acceptance_time_utc",
        "latest_acceptance_time_utc",
    ]
    if fact_bmu_acceptance_event.empty:
        return pd.DataFrame(columns=keep_columns)

    intervals = _build_half_hour_interval_frame(start_date, end_date)
    interval_rows = []
    for event in fact_bmu_acceptance_event.itertuples(index=False):
        matching = intervals[
            (intervals["interval_end_utc"] > event.time_from_utc)
            & (intervals["interval_start_utc"] < event.time_to_utc)
        ]
        for interval in matching.itertuples(index=False):
            overlap_hours = _overlap_hours(
                event.time_from_utc,
                event.time_to_utc,
                interval.interval_start_utc,
                interval.interval_end_utc,
            )
            if overlap_hours <= 0:
                continue
            interval_rows.append(
                {
                    "settlement_date": interval.settlement_date,
                    "settlement_period": interval.settlement_period,
                    "interval_start_local": interval.interval_start_local,
                    "interval_end_local": interval.interval_end_local,
                    "interval_start_utc": interval.interval_start_utc,
                    "interval_end_utc": interval.interval_end_utc,
                    "source_key": event.source_key,
                    "source_label": event.source_label,
                    "target_is_proxy": event.target_is_proxy,
                    "dispatch_truth_tier": event.dispatch_truth_tier,
                    "is_lower_bound_metric": event.is_lower_bound_metric,
                    "elexon_bm_unit": event.elexon_bm_unit,
                    "national_grid_bm_unit": event.national_grid_bm_unit,
                    "bm_unit_name": event.bm_unit_name,
                    "lead_party_name": event.lead_party_name,
                    "fuel_type": event.fuel_type,
                    "bm_unit_type": event.bm_unit_type,
                    "gsp_group_id": event.gsp_group_id,
                    "gsp_group_name": event.gsp_group_name,
                    "generation_capacity_mw": event.generation_capacity_mw,
                    "mapping_status": event.mapping_status,
                    "mapping_confidence": event.mapping_confidence,
                    "mapping_rule": event.mapping_rule,
                    "cluster_key": event.cluster_key,
                    "cluster_label": event.cluster_label,
                    "parent_region": event.parent_region,
                    "acceptance_number": event.acceptance_number,
                    "acceptance_time_utc": event.acceptance_time_utc,
                    "deemed_bo_flag": _coerce_bool(event.deemed_bo_flag),
                    "so_flag": _coerce_bool(event.so_flag),
                    "stor_flag": _coerce_bool(event.stor_flag),
                    "rr_flag": _coerce_bool(event.rr_flag),
                    "overlap_hours": overlap_hours,
                    "accepted_level_time_weighted_mwh": event.accepted_level_mean_mw * overlap_hours,
                    "accepted_down_delta_mwh_lower_bound": event.accepted_down_delta_mw * overlap_hours,
                    "accepted_up_delta_mwh_lower_bound": event.accepted_up_delta_mw * overlap_hours,
                    "accepted_down_delta_mw": event.accepted_down_delta_mw,
                    "accepted_up_delta_mw": event.accepted_up_delta_mw,
                    "is_down_event": int(event.accepted_down_delta_mw > 0),
                    "is_up_event": int(event.accepted_up_delta_mw > 0),
                }
            )

    if not interval_rows:
        return pd.DataFrame(columns=keep_columns)

    frame = pd.DataFrame(interval_rows)
    group_columns = [
        "settlement_date",
        "settlement_period",
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "source_key",
        "source_label",
        "target_is_proxy",
        "dispatch_truth_tier",
        "is_lower_bound_metric",
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
    aggregated = (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            acceptance_event_count=("acceptance_number", "count"),
            distinct_acceptance_number_count=("acceptance_number", "nunique"),
            accepted_down_event_count=("is_down_event", "sum"),
            accepted_up_event_count=("is_up_event", "sum"),
            overlap_hours=("overlap_hours", "sum"),
            accepted_level_time_weighted_mwh=("accepted_level_time_weighted_mwh", "sum"),
            accepted_down_delta_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
            accepted_up_delta_mwh_lower_bound=("accepted_up_delta_mwh_lower_bound", "sum"),
            max_accepted_down_delta_mw=("accepted_down_delta_mw", "max"),
            max_accepted_up_delta_mw=("accepted_up_delta_mw", "max"),
            any_deemed_bo_flag=("deemed_bo_flag", "max"),
            any_so_flag=("so_flag", "max"),
            any_stor_flag=("stor_flag", "max"),
            any_rr_flag=("rr_flag", "max"),
            earliest_acceptance_time_utc=("acceptance_time_utc", "min"),
            latest_acceptance_time_utc=("acceptance_time_utc", "max"),
        )
    )
    aggregated["overlap_minutes"] = aggregated["overlap_hours"] * 60.0
    aggregated["accepted_level_time_weighted_mean_mw"] = np.where(
        aggregated["overlap_hours"] > 0,
        aggregated["accepted_level_time_weighted_mwh"] / aggregated["overlap_hours"],
        np.nan,
    )
    aggregated["dispatch_down_flag"] = aggregated["accepted_down_event_count"] > 0
    aggregated["dispatch_up_flag"] = aggregated["accepted_up_event_count"] > 0
    return aggregated[keep_columns].sort_values(["interval_start_utc", "elexon_bm_unit"]).reset_index(drop=True)


def build_fact_bmu_bid_offer_half_hourly(
    dim_bmu_asset: pd.DataFrame,
    raw_bid_offer_frame: pd.DataFrame,
) -> pd.DataFrame:
    keep_columns = [
        "settlement_date",
        "settlement_period",
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "source_key",
        "source_label",
        "source_dataset",
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
        "bid_offer_pair_count",
        "negative_bid_pair_count",
        "valid_negative_bid_pair_count",
        "negative_bid_available_flag",
        "sentinel_bid_pair_count",
        "sentinel_offer_pair_count",
        "sentinel_pair_count",
        "sentinel_bid_available_flag",
        "sentinel_offer_available_flag",
        "sentinel_pair_available_flag",
        "minimum_bid_gbp_per_mwh",
        "maximum_bid_gbp_per_mwh",
        "most_negative_bid_gbp_per_mwh",
        "least_negative_bid_gbp_per_mwh",
        "minimum_offer_gbp_per_mwh",
        "maximum_offer_gbp_per_mwh",
    ]
    if raw_bid_offer_frame.empty:
        return pd.DataFrame(columns=keep_columns)

    frame = raw_bid_offer_frame.rename(
        columns={
            "dataset": "source_dataset",
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "timeFrom": "interval_start_utc",
            "timeTo": "interval_end_utc",
            "pairId": "pair_id",
            "offer": "offer_gbp_per_mwh",
            "bid": "bid_gbp_per_mwh",
            "bmUnit": "elexon_bm_unit",
            "nationalGridBmUnit": "national_grid_bm_unit_from_fact",
        }
    ).copy()
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"], errors="coerce").dt.date
    frame["settlement_period"] = pd.to_numeric(frame["settlement_period"], errors="coerce").astype("Int64")
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame["interval_end_utc"] = pd.to_datetime(frame["interval_end_utc"], utc=True, errors="coerce")
    frame["interval_start_local"] = frame["interval_start_utc"].dt.tz_convert("Europe/London")
    frame["interval_end_local"] = frame["interval_end_utc"].dt.tz_convert("Europe/London")
    frame["pair_id"] = pd.to_numeric(frame["pair_id"], errors="coerce")
    frame["offer_gbp_per_mwh"] = pd.to_numeric(frame["offer_gbp_per_mwh"], errors="coerce")
    frame["bid_gbp_per_mwh"] = pd.to_numeric(frame["bid_gbp_per_mwh"], errors="coerce")
    frame["sentinel_bid_flag"] = frame["bid_gbp_per_mwh"] <= SENTINEL_BID_FLOOR_GBP_PER_MWH
    frame["sentinel_offer_flag"] = frame["offer_gbp_per_mwh"] >= SENTINEL_OFFER_CEILING_GBP_PER_MWH
    frame["sentinel_pair_flag"] = frame["sentinel_bid_flag"] | frame["sentinel_offer_flag"]
    frame["negative_bid_flag"] = frame["bid_gbp_per_mwh"] < 0
    frame["valid_negative_bid_flag"] = frame["negative_bid_flag"] & ~frame["sentinel_pair_flag"]
    frame["source_key"] = "BOD"
    frame["source_label"] = "Elexon bid-offer data"
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

    group_columns = [
        "settlement_date",
        "settlement_period",
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "source_key",
        "source_label",
        "source_dataset",
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
    ]
    aggregated = (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            bid_offer_pair_count=("pair_id", "count"),
            negative_bid_pair_count=("negative_bid_flag", "sum"),
            valid_negative_bid_pair_count=("valid_negative_bid_flag", "sum"),
            sentinel_bid_pair_count=("sentinel_bid_flag", "sum"),
            sentinel_offer_pair_count=("sentinel_offer_flag", "sum"),
            sentinel_pair_count=("sentinel_pair_flag", "sum"),
            minimum_bid_gbp_per_mwh=("bid_gbp_per_mwh", "min"),
            maximum_bid_gbp_per_mwh=("bid_gbp_per_mwh", "max"),
            minimum_offer_gbp_per_mwh=("offer_gbp_per_mwh", "min"),
            maximum_offer_gbp_per_mwh=("offer_gbp_per_mwh", "max"),
            most_negative_bid_gbp_per_mwh=(
                "bid_gbp_per_mwh",
                lambda values: float(pd.Series(values)[pd.Series(values) < 0].min())
                if bool((pd.Series(values) < 0).any())
                else np.nan,
            ),
            least_negative_bid_gbp_per_mwh=(
                "bid_gbp_per_mwh",
                lambda values: float(pd.Series(values)[pd.Series(values) < 0].max())
                if bool((pd.Series(values) < 0).any())
                else np.nan,
            ),
        )
    )
    aggregated["negative_bid_available_flag"] = aggregated["valid_negative_bid_pair_count"] > 0
    aggregated["sentinel_bid_available_flag"] = aggregated["sentinel_bid_pair_count"] > 0
    aggregated["sentinel_offer_available_flag"] = aggregated["sentinel_offer_pair_count"] > 0
    aggregated["sentinel_pair_available_flag"] = aggregated["sentinel_pair_count"] > 0
    return aggregated[keep_columns].sort_values(["interval_start_utc", "elexon_bm_unit"]).reset_index(drop=True)


def materialize_bmu_dispatch_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    reference = fetch_bmu_reference_all()
    dim_bmu_asset = build_dim_bmu_asset(reference)
    raw_acceptance = fetch_boalf_acceptances(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    raw_bid_offer = fetch_bod_bid_offers(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    fact_bmu_acceptance_event = build_fact_bmu_acceptance_event(dim_bmu_asset, raw_acceptance)
    fact_bmu_dispatch_acceptance_half_hourly = build_fact_bmu_dispatch_acceptance_half_hourly(
        fact_bmu_acceptance_event,
        start_date=start_date,
        end_date=end_date,
    )
    fact_bmu_bid_offer_half_hourly = build_fact_bmu_bid_offer_half_hourly(dim_bmu_asset, raw_bid_offer)

    frames = {
        "dim_bmu_asset": dim_bmu_asset,
        "fact_bmu_acceptance_event": fact_bmu_acceptance_event,
        "fact_bmu_dispatch_acceptance_half_hourly": fact_bmu_dispatch_acceptance_half_hourly,
        "fact_bmu_bid_offer_half_hourly": fact_bmu_bid_offer_half_hourly,
    }

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)

    return frames
