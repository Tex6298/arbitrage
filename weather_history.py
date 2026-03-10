from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from asset_mapping import ASSET_ANCHORS, ASSET_CLUSTERS, AssetAnchor


OPEN_METEO_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc
WEATHER_VARIABLES: Tuple[str, ...] = (
    "temperature_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
)


class WeatherHistoryError(RuntimeError):
    pass


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore").strip()
        detail = f": {body}" if body else ""
        raise WeatherHistoryError(f"Open-Meteo request failed with HTTP {exc.code}{detail}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise WeatherHistoryError(f"Open-Meteo request failed: {reason}") from exc
    except TimeoutError as exc:
        raise WeatherHistoryError("Open-Meteo request timed out") from exc
    except json.JSONDecodeError as exc:
        raise WeatherHistoryError("Open-Meteo returned invalid JSON") from exc


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def _anchor_cell_selection(anchor: AssetAnchor) -> str:
    note = (anchor.note or "").lower()
    if "offshore" in note:
        return "sea"
    return "land"


def _anchor_membership_frame() -> pd.DataFrame:
    rows = []
    for cluster in ASSET_CLUSTERS.values():
        for anchor_key in cluster.anchor_keys:
            anchor = ASSET_ANCHORS[anchor_key]
            rows.append(
                {
                    "anchor_key": anchor.key,
                    "anchor_label": anchor.label,
                    "anchor_latitude": anchor.latitude,
                    "anchor_longitude": anchor.longitude,
                    "anchor_capacity_mw": anchor.capacity_mw,
                    "cluster_key": cluster.key,
                    "cluster_label": cluster.label,
                    "parent_region": cluster.parent_region,
                }
            )
    return pd.DataFrame(rows).sort_values(["parent_region", "cluster_key", "anchor_key"]).reset_index(drop=True)


def _build_anchor_weather_frame(anchor: AssetAnchor, payload: dict, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise WeatherHistoryError(f"Open-Meteo returned no hourly payload for anchor '{anchor.key}'")

    time_values = hourly.get("time")
    if not isinstance(time_values, list) or not time_values:
        raise WeatherHistoryError(f"Open-Meteo returned no hourly times for anchor '{anchor.key}'")

    frame = pd.DataFrame({"hour_start_local_text": time_values})
    frame["hour_start_local"] = pd.to_datetime(frame["hour_start_local_text"], errors="coerce")
    frame["hour_start_local"] = frame["hour_start_local"].dt.tz_localize(
        "Europe/London",
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    frame["hour_end_local"] = frame["hour_start_local"] + pd.Timedelta(hours=1)
    frame["hour_start_utc"] = frame["hour_start_local"].dt.tz_convert("UTC")
    frame["hour_end_utc"] = frame["hour_end_local"].dt.tz_convert("UTC")
    frame["date"] = frame["hour_start_local"].dt.date
    frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()

    rename_map = {
        "temperature_2m": "temperature_2m_c",
        "pressure_msl": "pressure_msl_hpa",
        "cloud_cover": "cloud_cover_pct",
        "wind_speed_10m": "wind_speed_10m_ms",
        "wind_speed_100m": "wind_speed_100m_ms",
        "wind_direction_100m": "wind_direction_100m_deg",
        "wind_gusts_10m": "wind_gusts_10m_ms",
    }
    for source_name, target_name in rename_map.items():
        values = hourly.get(source_name, [])
        frame[target_name] = pd.to_numeric(pd.Series(values), errors="coerce").reindex(frame.index)

    radians = np.deg2rad(frame["wind_direction_100m_deg"])
    frame["wind_u_100m_ms"] = -frame["wind_speed_100m_ms"] * np.sin(radians)
    frame["wind_v_100m_ms"] = -frame["wind_speed_100m_ms"] * np.cos(radians)
    frame["wind_power_index_100m"] = frame["wind_speed_100m_ms"] ** 3
    frame["wind_speed_ratio_100m_to_10m"] = np.where(
        frame["wind_speed_10m_ms"] > 0,
        frame["wind_speed_100m_ms"] / frame["wind_speed_10m_ms"],
        np.nan,
    )
    frame["source_key"] = "open_meteo_archive"
    frame["source_label"] = "Open-Meteo historical weather archive"
    frame["source_dataset"] = "historical_weather_best_match"
    frame["target_is_proxy"] = False
    frame["anchor_key"] = anchor.key
    frame["anchor_label"] = anchor.label
    frame["requested_latitude"] = anchor.latitude
    frame["requested_longitude"] = anchor.longitude
    frame["resolved_latitude"] = pd.to_numeric(payload.get("latitude"), errors="coerce")
    frame["resolved_longitude"] = pd.to_numeric(payload.get("longitude"), errors="coerce")
    frame["resolved_elevation_m"] = pd.to_numeric(payload.get("elevation"), errors="coerce")
    frame["cell_selection"] = _anchor_cell_selection(anchor)

    keep_columns = [
        "date",
        "hour_start_local",
        "hour_end_local",
        "hour_start_utc",
        "hour_end_utc",
        "source_key",
        "source_label",
        "source_dataset",
        "target_is_proxy",
        "anchor_key",
        "anchor_label",
        "requested_latitude",
        "requested_longitude",
        "resolved_latitude",
        "resolved_longitude",
        "resolved_elevation_m",
        "cell_selection",
        "temperature_2m_c",
        "pressure_msl_hpa",
        "cloud_cover_pct",
        "wind_speed_10m_ms",
        "wind_speed_100m_ms",
        "wind_direction_100m_deg",
        "wind_gusts_10m_ms",
        "wind_u_100m_ms",
        "wind_v_100m_ms",
        "wind_power_index_100m",
        "wind_speed_ratio_100m_to_10m",
    ]
    return frame[keep_columns].sort_values("hour_start_utc").reset_index(drop=True)


def fetch_anchor_weather_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    frames = []
    for anchor in ASSET_ANCHORS.values():
        params = {
            "latitude": anchor.latitude,
            "longitude": anchor.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(WEATHER_VARIABLES),
            "wind_speed_unit": "ms",
            "timezone": "Europe/London",
            "cell_selection": _anchor_cell_selection(anchor),
        }
        url = f"{OPEN_METEO_ARCHIVE_BASE}?{urllib.parse.urlencode(params)}"
        payload = _fetch_json(url)
        frames.append(_build_anchor_weather_frame(anchor, payload, start_date=start_date, end_date=end_date))

    if not frames:
        raise WeatherHistoryError(f"no anchor weather rows returned for {start_date} to {end_date}")
    return pd.concat(frames, ignore_index=True)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not bool(valid.any()):
        return np.nan
    value_array = values[valid].astype(float).to_numpy()
    weight_array = weights[valid].astype(float).to_numpy()
    total_weight = float(weight_array.sum())
    if total_weight <= 0:
        return np.nan
    return float(np.average(value_array, weights=weight_array))


def _aggregate_scope_weather(
    frame: pd.DataFrame,
    group_columns: list[str],
    scope_type: str,
    scope_key_column: str,
    scope_label_column: str,
) -> pd.DataFrame:
    rows = []
    scalar_columns = [
        "temperature_2m_c",
        "pressure_msl_hpa",
        "cloud_cover_pct",
        "wind_speed_10m_ms",
        "wind_speed_100m_ms",
        "wind_gusts_10m_ms",
        "wind_power_index_100m",
        "wind_speed_ratio_100m_to_10m",
        "resolved_latitude",
        "resolved_longitude",
        "resolved_elevation_m",
    ]
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        weights = pd.to_numeric(group["anchor_capacity_mw"], errors="coerce")
        mean_u = _weighted_average(group["wind_u_100m_ms"], weights)
        mean_v = _weighted_average(group["wind_v_100m_ms"], weights)
        wind_direction = np.nan
        if pd.notna(mean_u) and pd.notna(mean_v) and (abs(mean_u) > 1e-9 or abs(mean_v) > 1e-9):
            wind_direction = float((np.degrees(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)
        rows.append(
            {
                **record,
                "scope_type": scope_type,
                "scope_key": record[scope_key_column],
                "scope_label": record[scope_label_column],
                "source_key": "open_meteo_archive",
                "source_label": "Open-Meteo historical weather archive",
                "source_dataset": "historical_weather_best_match",
                "target_is_proxy": False,
                "weather_anchor_count": int(group["anchor_key"].nunique()),
                "weather_weight_sum_mw": float(pd.to_numeric(group["anchor_capacity_mw"], errors="coerce").fillna(0.0).sum()),
                "wind_direction_100m_deg": wind_direction,
                "wind_u_100m_ms": mean_u,
                "wind_v_100m_ms": mean_v,
                **{
                    column: _weighted_average(group[column], weights)
                    for column in scalar_columns
                },
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_fact_weather_hourly_from_anchor_weather(anchor_weather_hourly: pd.DataFrame) -> pd.DataFrame:
    if anchor_weather_hourly.empty:
        raise WeatherHistoryError("anchor_weather_hourly is empty")

    membership = _anchor_membership_frame()
    anchor_frame = anchor_weather_hourly.merge(
        membership,
        on=["anchor_key", "anchor_label"],
        how="left",
    )
    if anchor_frame["cluster_key"].isna().any():
        missing_anchor_keys = sorted(anchor_frame.loc[anchor_frame["cluster_key"].isna(), "anchor_key"].dropna().unique())
        raise WeatherHistoryError(f"anchor weather rows missing cluster mapping: {', '.join(missing_anchor_keys)}")

    anchor_rows = anchor_frame.copy()
    anchor_rows["scope_type"] = "anchor"
    anchor_rows["scope_key"] = anchor_rows["anchor_key"]
    anchor_rows["scope_label"] = anchor_rows["anchor_label"]
    anchor_rows["weather_anchor_count"] = 1
    anchor_rows["weather_weight_sum_mw"] = pd.to_numeric(anchor_rows["anchor_capacity_mw"], errors="coerce")

    cluster_rows = _aggregate_scope_weather(
        anchor_frame,
        group_columns=[
            "date",
            "hour_start_local",
            "hour_end_local",
            "hour_start_utc",
            "hour_end_utc",
            "cluster_key",
            "cluster_label",
            "parent_region",
        ],
        scope_type="cluster",
        scope_key_column="cluster_key",
        scope_label_column="cluster_label",
    )

    parent_region_rows = _aggregate_scope_weather(
        anchor_frame,
        group_columns=[
            "date",
            "hour_start_local",
            "hour_end_local",
            "hour_start_utc",
            "hour_end_utc",
            "parent_region",
        ],
        scope_type="parent_region",
        scope_key_column="parent_region",
        scope_label_column="parent_region",
    )
    if not parent_region_rows.empty:
        parent_region_rows["cluster_key"] = pd.NA
        parent_region_rows["cluster_label"] = pd.NA

    common_columns = [
        "date",
        "hour_start_local",
        "hour_end_local",
        "hour_start_utc",
        "hour_end_utc",
        "scope_type",
        "scope_key",
        "scope_label",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "source_key",
        "source_label",
        "source_dataset",
        "target_is_proxy",
        "weather_anchor_count",
        "weather_weight_sum_mw",
        "temperature_2m_c",
        "pressure_msl_hpa",
        "cloud_cover_pct",
        "wind_speed_10m_ms",
        "wind_speed_100m_ms",
        "wind_direction_100m_deg",
        "wind_gusts_10m_ms",
        "wind_u_100m_ms",
        "wind_v_100m_ms",
        "wind_power_index_100m",
        "wind_speed_ratio_100m_to_10m",
        "resolved_latitude",
        "resolved_longitude",
        "resolved_elevation_m",
    ]
    for column in ("cluster_key", "cluster_label", "parent_region"):
        if column not in anchor_rows.columns:
            anchor_rows[column] = pd.NA

    out = pd.concat(
        [
            anchor_rows[common_columns],
            cluster_rows[common_columns] if not cluster_rows.empty else pd.DataFrame(columns=common_columns),
            parent_region_rows[common_columns] if not parent_region_rows.empty else pd.DataFrame(columns=common_columns),
        ],
        ignore_index=True,
        sort=False,
    )
    return out.sort_values(["scope_type", "scope_key", "hour_start_utc"]).reset_index(drop=True)


def build_fact_weather_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    anchor_weather_hourly = fetch_anchor_weather_hourly(start_date, end_date)
    return build_fact_weather_hourly_from_anchor_weather(anchor_weather_hourly)


def materialize_weather_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    fact_weather_hourly = build_fact_weather_hourly(start_date, end_date)
    frames = {
        "fact_weather_hourly": fact_weather_hourly,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)
    return frames
