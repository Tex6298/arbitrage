from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from typing import Any, Iterable, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from bmu_generation import ElexonError


ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc


def fetch_json(url: str) -> Any:
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


def local_day_to_utc_window(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    start_local = dt.datetime.combine(day, dt.time.min, tzinfo=LONDON_TZ)
    end_local = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, tzinfo=LONDON_TZ)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def local_date_range_to_utc_window(start_date: dt.date, end_date: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    start_utc, _ = local_day_to_utc_window(start_date)
    _, end_utc = local_day_to_utc_window(end_date)
    return start_utc, end_utc


def rfc3339_utc(value: dt.datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def build_half_hour_interval_frame(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
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


def build_bmu_interval_spine(
    dim_bmu_asset: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    intervals = build_half_hour_interval_frame(start_date, end_date)
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
    dim = dim_bmu_asset[dim_columns].copy()
    intervals["__join_key"] = 1
    dim["__join_key"] = 1
    spine = intervals.merge(dim, on="__join_key", how="inner").drop(columns="__join_key")
    return spine.sort_values(["interval_start_utc", "elexon_bm_unit"]).reset_index(drop=True)


def overlap_hours(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> float:
    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    seconds = (overlap_end - overlap_start).total_seconds()
    return max(seconds, 0.0) / 3600.0


def normalize_bmu_identifier(value: object) -> str:
    text = str(value or "").strip().upper()
    return text


def bmu_identifier_candidates(value: object) -> list[str]:
    normalized = normalize_bmu_identifier(value)
    if not normalized:
        return []
    candidates = {normalized}
    if normalized.startswith("T_"):
        candidates.add(normalized[2:])
    else:
        candidates.add(f"T_{normalized}")
    return sorted(candidates)


def coerce_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def unwrap_data_rows(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        data = payload.get("data", [])
        if isinstance(data, list):
            return data
        raise ElexonError("Elexon returned a dict payload without list data rows")
    if isinstance(payload, list):
        return payload
    raise ElexonError(f"Elexon returned unsupported payload type {type(payload).__name__}")
