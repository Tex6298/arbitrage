from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from curtailment_signals import _datapackage_show, _fetch_csv


DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE = "fact_day_ahead_constraint_boundary_half_hourly"
DAY_AHEAD_CONSTRAINT_BOUNDARY_SOURCE_KEY = "neso_day_ahead_constraint_boundary"
DAY_AHEAD_CONSTRAINT_BOUNDARY_SOURCE_LABEL = "NESO day-ahead constraint flows and limits"
DAY_AHEAD_CONSTRAINT_BOUNDARY_DATASET_ID = "cf3cbc92-2d5d-4c2b-bd29-e11a21070b26"
DAY_AHEAD_CONSTRAINT_BOUNDARY_RESOURCE_NAME = "Day Ahead Constraint Flows and Limits"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def _empty_boundary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "boundary_key",
            "boundary_label",
            "source_key",
            "source_label",
            "source_provider",
            "source_dataset_id",
            "source_resource_id",
            "source_resource_name",
            "source_document_url",
            "target_is_proxy",
            "limit_mw",
            "flow_mw",
            "remaining_headroom_mw",
            "utilization_ratio",
            "boundary_state",
        ]
    )


def _resource_metadata() -> dict:
    metadata = _datapackage_show(DAY_AHEAD_CONSTRAINT_BOUNDARY_DATASET_ID)
    resources = metadata.get("result", {}).get("resources", [])
    for resource in resources:
        if str(resource.get("name") or "").strip().lower() == DAY_AHEAD_CONSTRAINT_BOUNDARY_RESOURCE_NAME.lower():
            return resource
    raise RuntimeError("NESO day-ahead constraint boundary CSV resource not found")


def build_fact_day_ahead_constraint_boundary_half_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    resource = _resource_metadata()
    raw = _fetch_csv(str(resource.get("url") or ""))
    if raw.empty:
        return _empty_boundary_frame()

    # The NESO file currently includes two trailing blank columns; keep only the first four signal columns.
    raw = raw.iloc[:, :4].copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    boundary_column = raw.columns[0]
    timestamp_column = raw.columns[1]
    limit_column = raw.columns[2]
    flow_column = raw.columns[3]

    interval_start_local = pd.to_datetime(raw[timestamp_column], errors="coerce").dt.tz_localize(
        LONDON_TZ,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    interval_end_local = interval_start_local + pd.Timedelta(minutes=30)
    interval_start_utc = interval_start_local.dt.tz_convert(UTC)
    interval_end_utc = interval_end_local.dt.tz_convert(UTC)

    frame = pd.DataFrame(
        {
            "date": interval_start_local.dt.date,
            "interval_start_local": interval_start_local,
            "interval_end_local": interval_end_local,
            "interval_start_utc": interval_start_utc,
            "interval_end_utc": interval_end_utc,
            "boundary_key": raw[boundary_column].astype("object"),
            "boundary_label": raw[boundary_column].astype("object"),
            "source_key": DAY_AHEAD_CONSTRAINT_BOUNDARY_SOURCE_KEY,
            "source_label": DAY_AHEAD_CONSTRAINT_BOUNDARY_SOURCE_LABEL,
            "source_provider": "neso",
            "source_dataset_id": DAY_AHEAD_CONSTRAINT_BOUNDARY_DATASET_ID,
            "source_resource_id": resource.get("id"),
            "source_resource_name": resource.get("name"),
            "source_document_url": resource.get("url"),
            "target_is_proxy": False,
            "limit_mw": pd.to_numeric(raw[limit_column], errors="coerce"),
            "flow_mw": pd.to_numeric(raw[flow_column], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["interval_start_utc", "boundary_key"]).copy()
    window_start = pd.Timestamp(start_date, tz=LONDON_TZ).tz_convert(UTC)
    window_end = pd.Timestamp(end_date + dt.timedelta(days=1), tz=LONDON_TZ).tz_convert(UTC)
    frame = frame[
        frame["interval_start_utc"].lt(window_end) & frame["interval_end_utc"].gt(window_start)
    ].copy()
    if frame.empty:
        return _empty_boundary_frame()

    frame["remaining_headroom_mw"] = pd.to_numeric(frame["limit_mw"], errors="coerce") - pd.to_numeric(
        frame["flow_mw"], errors="coerce"
    )
    limit_abs = pd.to_numeric(frame["limit_mw"], errors="coerce").abs()
    frame["utilization_ratio"] = np.where(
        limit_abs.gt(0),
        pd.to_numeric(frame["flow_mw"], errors="coerce").abs() / limit_abs,
        np.nan,
    )
    frame["boundary_state"] = "constraint_boundary_available"
    frame.loc[pd.to_numeric(frame["limit_mw"], errors="coerce").le(0), "boundary_state"] = "constraint_boundary_zero_limit"
    frame.loc[
        pd.to_numeric(frame["remaining_headroom_mw"], errors="coerce").le(0)
        & pd.to_numeric(frame["limit_mw"], errors="coerce").gt(0),
        "boundary_state",
    ] = "constraint_boundary_at_or_above_limit"
    frame.loc[
        frame["boundary_state"].eq("constraint_boundary_available")
        & pd.to_numeric(frame["utilization_ratio"], errors="coerce").ge(0.9),
        "boundary_state",
    ] = "constraint_boundary_tight"

    return frame[_empty_boundary_frame().columns].sort_values(
        ["interval_start_utc", "boundary_key"]
    ).reset_index(drop=True)


def materialize_day_ahead_constraint_boundary_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_day_ahead_constraint_boundary_half_hourly(start_date=start_date, end_date=end_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE}.csv", index=False)
    return {DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE: fact}
