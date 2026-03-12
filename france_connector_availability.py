from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from bmu_availability import _iter_remit_windows, fetch_remit_event_detail_with_status
from france_connector import FRANCE_CONNECTOR_SPECS, interconnector_cable_frame


FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE = "fact_france_connector_operator_event"
FRANCE_CONNECTOR_AVAILABILITY_TABLE = "fact_france_connector_availability_hourly"


@dataclass(frozen=True)
class FranceConnectorOperatorSource:
    connector_key: str
    source_provider: str
    source_key: str
    source_label: str
    match_patterns: Tuple[str, ...]
    requires_external_export: bool
    note: str


FRANCE_CONNECTOR_OPERATOR_SOURCES: Tuple[FranceConnectorOperatorSource, ...] = (
    FranceConnectorOperatorSource(
        connector_key="ifa2",
        source_provider="elexon_remit",
        source_key="elexon_remit_connector",
        source_label="Elexon REMIT transmission-unavailability messages",
        match_patterns=(r"\bifa2\b", r"i_ied-ifa2", r"i_ieg-ifa2"),
        requires_external_export=False,
        note="IFA2 publishes transmission-unavailability messages through Elexon REMIT.",
    ),
    FranceConnectorOperatorSource(
        connector_key="ifa",
        source_provider="elexon_remit",
        source_key="elexon_remit_connector",
        source_label="Elexon REMIT transmission-unavailability messages",
        match_patterns=(r"i_ied-ifa(?!2)", r"i_ieg-ifa(?!2)", r"\bifa\b"),
        requires_external_export=False,
        note="IFA publishes transmission-unavailability messages through Elexon REMIT.",
    ),
    FranceConnectorOperatorSource(
        connector_key="eleclink",
        source_provider="nordpool_umm_export",
        source_key="nordpool_umm_export",
        source_label="Nord Pool UMM export",
        match_patterns=(r"\beleclink\b",),
        requires_external_export=True,
        note="ElecLink moved outage publication to Nord Pool UMM on June 3, 2024; the repo supports a manual export path until credentials are wired.",
    ),
)


def _hourly_window_frame(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    start_local = pd.Timestamp(start_date, tz="Europe/London")
    end_local = pd.Timestamp(end_date + dt.timedelta(days=1), tz="Europe/London")
    interval_start_local = pd.date_range(start=start_local, end=end_local, freq="h", inclusive="left")
    frame = pd.DataFrame({"interval_start_local": interval_start_local})
    frame["interval_end_local"] = frame["interval_start_local"] + pd.Timedelta(hours=1)
    frame["interval_start_utc"] = frame["interval_start_local"].dt.tz_convert("UTC")
    frame["interval_end_utc"] = frame["interval_end_local"].dt.tz_convert("UTC")
    frame["date"] = frame["interval_start_local"].dt.date
    return frame


def _connector_source_lookup() -> dict[str, FranceConnectorOperatorSource]:
    return {spec.connector_key: spec for spec in FRANCE_CONNECTOR_OPERATOR_SOURCES}


def _connector_text_blob(frame: pd.DataFrame) -> pd.Series:
    text_columns = [
        column
        for column in (
            "affectedUnit",
            "assetId",
            "registrationCode",
            "affectedUnitEIC",
            "messageType",
            "eventType",
            "unavailabilityType",
            "fuelType",
            "marketParticipantName",
        )
        if column in frame.columns
    ]
    if not text_columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[text_columns].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()


def _match_connector_from_text(blob: str) -> str | None:
    for spec in FRANCE_CONNECTOR_OPERATOR_SOURCES:
        for pattern in spec.match_patterns:
            if re.search(pattern, blob):
                return spec.connector_key
    return None


def _empty_operator_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "connector_key",
            "connector_label",
            "source_provider",
            "source_key",
            "source_label",
            "source_truth_tier",
            "connector_match_rule",
            "target_is_proxy",
            "publish_time_utc",
            "event_start_utc",
            "event_end_utc",
            "event_status",
            "message_type",
            "event_type",
            "unavailability_type",
            "affected_unit",
            "asset_id",
            "registration_code",
            "normal_capacity_mw",
            "available_capacity_mw",
            "unavailable_capacity_mw",
        ]
    )


def load_eleclink_umm_export(export_path: str | Path | None) -> pd.DataFrame:
    if not export_path:
        return pd.DataFrame()

    path = Path(export_path)
    if not path.exists():
        raise FileNotFoundError(f"ElecLink UMM export does not exist: {path}")

    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(records)
    else:
        frame = pd.read_csv(path)

    if frame.empty:
        return pd.DataFrame()

    frame = frame.rename(
        columns={
            "publishTime": "publish_time_utc",
            "published_at": "publish_time_utc",
            "eventStartTime": "event_start_utc",
            "startTime": "event_start_utc",
            "eventEndTime": "event_end_utc",
            "endTime": "event_end_utc",
            "normalCapacity": "normal_capacity_mw",
            "availableCapacity": "available_capacity_mw",
            "unavailableCapacity": "unavailable_capacity_mw",
            "messageType": "message_type",
            "eventType": "event_type",
            "eventStatus": "event_status",
            "unavailabilityType": "unavailability_type",
            "assetId": "asset_id",
            "affectedUnit": "affected_unit",
            "registrationCode": "registration_code",
        }
    ).copy()

    frame["connector_key"] = frame.get("connector_key", "eleclink").fillna("eleclink")
    frame = frame[frame["connector_key"].astype(str).str.lower().eq("eleclink")].copy()
    frame["publish_time_utc"] = pd.to_datetime(frame.get("publish_time_utc"), utc=True, errors="coerce")
    frame["event_start_utc"] = pd.to_datetime(frame.get("event_start_utc"), utc=True, errors="coerce")
    frame["event_end_utc"] = pd.to_datetime(frame.get("event_end_utc"), utc=True, errors="coerce")
    frame["normal_capacity_mw"] = pd.to_numeric(frame.get("normal_capacity_mw"), errors="coerce")
    frame["available_capacity_mw"] = pd.to_numeric(frame.get("available_capacity_mw"), errors="coerce")
    frame["unavailable_capacity_mw"] = pd.to_numeric(frame.get("unavailable_capacity_mw"), errors="coerce")
    frame["connector_label"] = "ElecLink"
    frame["source_provider"] = "nordpool_umm_export"
    frame["source_key"] = "nordpool_umm_export"
    frame["source_label"] = "Nord Pool UMM export"
    frame["source_truth_tier"] = "operator_outage_truth"
    frame["connector_match_rule"] = "manual_eleclink_export"
    frame["target_is_proxy"] = False
    return frame[
        list(_empty_operator_event_frame().columns)
    ].dropna(subset=["event_start_utc", "event_end_utc"], how="any").reset_index(drop=True)


def build_france_connector_operator_event_frame(
    raw_remit_frame: pd.DataFrame,
    eleclink_umm_export: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = []
    source_lookup = _connector_source_lookup()
    if raw_remit_frame is not None and not raw_remit_frame.empty:
        remit = raw_remit_frame.copy()
        blobs = _connector_text_blob(remit)
        matched_connector = blobs.map(_match_connector_from_text)
        remit = remit[matched_connector.notna()].copy()
        remit["connector_key"] = matched_connector[matched_connector.notna()].values
        if not remit.empty:
            remit = remit.rename(
                columns={
                    "publishTime": "publish_time_utc",
                    "eventStatus": "event_status",
                    "eventStartTime": "event_start_utc",
                    "eventEndTime": "event_end_utc",
                    "affectedUnit": "affected_unit",
                    "assetId": "asset_id",
                    "registrationCode": "registration_code",
                    "messageType": "message_type",
                    "eventType": "event_type",
                    "unavailabilityType": "unavailability_type",
                    "normalCapacity": "normal_capacity_mw",
                    "availableCapacity": "available_capacity_mw",
                    "unavailableCapacity": "unavailable_capacity_mw",
                    "outageProfile": "outage_profile",
                }
            ).copy()
            remit["publish_time_utc"] = pd.to_datetime(remit["publish_time_utc"], utc=True, errors="coerce")
            remit["event_start_utc"] = pd.to_datetime(remit["event_start_utc"], utc=True, errors="coerce")
            remit["event_end_utc"] = pd.to_datetime(remit["event_end_utc"], utc=True, errors="coerce")
            remit["normal_capacity_mw"] = pd.to_numeric(remit["normal_capacity_mw"], errors="coerce")
            remit["available_capacity_mw"] = pd.to_numeric(remit["available_capacity_mw"], errors="coerce")
            remit["unavailable_capacity_mw"] = pd.to_numeric(remit["unavailable_capacity_mw"], errors="coerce")

            event_rows = []
            for row in remit.itertuples(index=False):
                series = pd.Series(row._asdict())
                connector_key = str(series["connector_key"])
                connector_source = source_lookup[connector_key]
                connector_label = next(spec.connector_label for spec in FRANCE_CONNECTOR_SPECS if spec.connector_key == connector_key)
                windows = list(_iter_remit_windows(series))
                if not windows:
                    continue
                for window in windows:
                    event_rows.append(
                        {
                            "connector_key": connector_key,
                            "connector_label": connector_label,
                            "source_provider": connector_source.source_provider,
                            "source_key": connector_source.source_key,
                            "source_label": connector_source.source_label,
                            "source_truth_tier": "operator_outage_truth",
                            "connector_match_rule": "remit_text_match",
                            "target_is_proxy": False,
                            "publish_time_utc": series.get("publish_time_utc"),
                            "event_start_utc": window["window_start_utc"],
                            "event_end_utc": window["window_end_utc"],
                            "event_status": series.get("event_status"),
                            "message_type": series.get("message_type"),
                            "event_type": series.get("event_type"),
                            "unavailability_type": series.get("unavailability_type"),
                            "affected_unit": series.get("affected_unit"),
                            "asset_id": series.get("asset_id"),
                            "registration_code": series.get("registration_code"),
                            "normal_capacity_mw": window["normal_capacity_mw"],
                            "available_capacity_mw": window["available_capacity_mw"],
                            "unavailable_capacity_mw": window["unavailable_capacity_mw"],
                        }
                    )
            if event_rows:
                frames.append(pd.DataFrame(event_rows))

    if eleclink_umm_export is not None and not eleclink_umm_export.empty:
        frames.append(eleclink_umm_export[list(_empty_operator_event_frame().columns)].copy())

    if not frames:
        return _empty_operator_event_frame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["connector_key", "event_start_utc", "publish_time_utc"], na_position="last")
        .reset_index(drop=True)
    )


def _empty_availability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "connector_key",
            "connector_label",
            "operator_name",
            "nominal_capacity_mw",
            "source_provider",
            "source_key",
            "source_label",
            "target_is_proxy",
            "operator_fetch_ok",
            "operator_source_gap_reason",
            "operator_event_count",
            "operator_active_flag",
            "operator_partial_availability_flag",
            "operator_availability_state",
            "operator_capacity_evidence_tier",
            "operator_capacity_limit_mw",
            "operator_min_available_capacity_mw",
            "operator_max_unavailable_capacity_mw",
            "operator_normal_capacity_mw",
            "latest_operator_publish_time_utc",
        ]
    )


def build_fact_france_connector_availability_hourly(
    start_date: dt.date,
    end_date: dt.date,
    operator_event_frame: pd.DataFrame,
    remit_fetch_status_by_date: pd.DataFrame | None = None,
    eleclink_export_loaded: bool = False,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    cables = interconnector_cable_frame()
    if cables.empty:
        return _empty_availability_frame()

    hours = _hourly_window_frame(start_date, end_date)
    base = hours.assign(_join_key=1).merge(
        cables[["connector_key", "connector_label", "operator_name", "nominal_capacity_mw"]].assign(_join_key=1),
        on="_join_key",
        how="inner",
    ).drop(columns="_join_key")

    source_lookup = _connector_source_lookup()
    base["source_provider"] = base["connector_key"].map(lambda key: source_lookup[key].source_provider)
    base["source_key"] = base["connector_key"].map(lambda key: source_lookup[key].source_key)
    base["source_label"] = base["connector_key"].map(lambda key: source_lookup[key].source_label)
    base["target_is_proxy"] = False

    remit_status = remit_fetch_status_by_date.copy() if remit_fetch_status_by_date is not None else pd.DataFrame()
    if remit_status.empty:
        remit_status = pd.DataFrame({"settlement_date": pd.date_range(start=start_date, end=end_date, freq="D").date})
        remit_status["remit_fetch_ok"] = False
    remit_status["settlement_date"] = pd.to_datetime(remit_status["settlement_date"], errors="coerce").dt.date
    remit_status["remit_fetch_ok"] = remit_status["remit_fetch_ok"].where(
        remit_status["remit_fetch_ok"].notna(),
        False,
    ).astype(bool)
    base = base.merge(remit_status[["settlement_date", "remit_fetch_ok"]], left_on="date", right_on="settlement_date", how="left")
    base = base.drop(columns=["settlement_date"], errors="ignore")
    base["operator_fetch_ok"] = False
    remit_connector_mask = base["source_provider"].eq("elexon_remit")
    base.loc[remit_connector_mask, "operator_fetch_ok"] = base.loc[remit_connector_mask, "remit_fetch_ok"].fillna(False)
    base.loc[base["connector_key"].eq("eleclink"), "operator_fetch_ok"] = bool(eleclink_export_loaded)
    base = base.drop(columns=["remit_fetch_ok"], errors="ignore")

    base["operator_source_gap_reason"] = pd.NA
    base.loc[remit_connector_mask & ~base["operator_fetch_ok"], "operator_source_gap_reason"] = "elexon_remit_fetch_failed"
    base.loc[base["connector_key"].eq("eleclink") & ~base["operator_fetch_ok"], "operator_source_gap_reason"] = (
        "nordpool_umm_export_not_provided"
    )

    interval_rows = []
    if operator_event_frame is not None and not operator_event_frame.empty:
        interval_index = hours[["date", "interval_start_utc", "interval_end_utc"]].drop_duplicates()
        events = operator_event_frame.copy()
        events["event_start_utc"] = pd.to_datetime(events["event_start_utc"], utc=True, errors="coerce")
        events["event_end_utc"] = pd.to_datetime(events["event_end_utc"], utc=True, errors="coerce")
        events["publish_time_utc"] = pd.to_datetime(events["publish_time_utc"], utc=True, errors="coerce")
        events["normal_capacity_mw"] = pd.to_numeric(events["normal_capacity_mw"], errors="coerce")
        events["available_capacity_mw"] = pd.to_numeric(events["available_capacity_mw"], errors="coerce")
        events["unavailable_capacity_mw"] = pd.to_numeric(events["unavailable_capacity_mw"], errors="coerce")
        for row in events.itertuples(index=False):
            if pd.isna(row.event_start_utc) or pd.isna(row.event_end_utc) or row.event_end_utc <= row.event_start_utc:
                continue
            matching_intervals = interval_index[
                (interval_index["interval_end_utc"] > row.event_start_utc)
                & (interval_index["interval_start_utc"] < row.event_end_utc)
            ]
            for interval in matching_intervals.itertuples(index=False):
                interval_rows.append(
                    {
                        "date": interval.date,
                        "interval_start_utc": interval.interval_start_utc,
                        "connector_key": row.connector_key,
                        "operator_event_count": 1,
                        "operator_active_flag": True,
                        "operator_min_available_capacity_mw": row.available_capacity_mw,
                        "operator_max_unavailable_capacity_mw": row.unavailable_capacity_mw,
                        "operator_normal_capacity_mw": row.normal_capacity_mw,
                        "latest_operator_publish_time_utc": row.publish_time_utc,
                    }
                )

    interval_frame = pd.DataFrame(interval_rows)
    if interval_frame.empty:
        interval_frame = pd.DataFrame(
            columns=[
                "date",
                "interval_start_utc",
                "connector_key",
                "operator_event_count",
                "operator_active_flag",
                "operator_min_available_capacity_mw",
                "operator_max_unavailable_capacity_mw",
                "operator_normal_capacity_mw",
                "latest_operator_publish_time_utc",
            ]
        )
    else:
        interval_frame = (
            interval_frame.groupby(["date", "interval_start_utc", "connector_key"], as_index=False, dropna=False)
            .agg(
                operator_event_count=("operator_event_count", "sum"),
                operator_active_flag=("operator_active_flag", "max"),
                operator_min_available_capacity_mw=("operator_min_available_capacity_mw", "min"),
                operator_max_unavailable_capacity_mw=("operator_max_unavailable_capacity_mw", "max"),
                operator_normal_capacity_mw=("operator_normal_capacity_mw", "max"),
                latest_operator_publish_time_utc=("latest_operator_publish_time_utc", "max"),
            )
        )

    base = base.merge(interval_frame, on=["date", "interval_start_utc", "connector_key"], how="left")
    base["operator_event_count"] = pd.to_numeric(base["operator_event_count"], errors="coerce").fillna(0).astype(int)
    base["operator_active_flag"] = base["operator_active_flag"].where(base["operator_active_flag"].notna(), False).astype(bool)
    base["operator_min_available_capacity_mw"] = pd.to_numeric(base["operator_min_available_capacity_mw"], errors="coerce")
    base["operator_max_unavailable_capacity_mw"] = pd.to_numeric(base["operator_max_unavailable_capacity_mw"], errors="coerce")
    base["operator_normal_capacity_mw"] = pd.to_numeric(base["operator_normal_capacity_mw"], errors="coerce")

    base["operator_partial_availability_flag"] = (
        base["operator_active_flag"]
        & base["operator_min_available_capacity_mw"].notna()
        & (base["operator_min_available_capacity_mw"] > 0)
        & (
            base["operator_normal_capacity_mw"].isna()
            | (base["operator_min_available_capacity_mw"] + 1e-6 < base["operator_normal_capacity_mw"])
        )
    )

    base["operator_availability_state"] = "unknown_source"
    base.loc[base["operator_fetch_ok"], "operator_availability_state"] = "available"
    base.loc[base["operator_active_flag"], "operator_availability_state"] = "outage"
    base.loc[base["operator_partial_availability_flag"], "operator_availability_state"] = "partial_outage"

    base["operator_capacity_limit_mw"] = np.nan
    base.loc[base["operator_availability_state"].eq("available"), "operator_capacity_limit_mw"] = pd.to_numeric(
        base["nominal_capacity_mw"], errors="coerce"
    )
    base.loc[
        base["operator_availability_state"].eq("partial_outage")
        & base["operator_min_available_capacity_mw"].notna(),
        "operator_capacity_limit_mw",
    ] = base["operator_min_available_capacity_mw"]
    base.loc[base["operator_availability_state"].eq("outage"), "operator_capacity_limit_mw"] = base[
        "operator_min_available_capacity_mw"
    ].fillna(0.0)

    base["operator_capacity_evidence_tier"] = "source_unavailable"
    base.loc[base["operator_availability_state"].eq("available"), "operator_capacity_evidence_tier"] = "operator_no_active_outage"
    base.loc[
        base["operator_availability_state"].isin(["partial_outage", "outage"]),
        "operator_capacity_evidence_tier",
    ] = "operator_outage_truth"

    column_order = list(_empty_availability_frame().columns)
    return base[column_order].sort_values(["interval_start_utc", "connector_key"]).reset_index(drop=True)


def materialize_france_connector_availability_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    eleclink_umm_export_path: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    raw_remit, remit_status = fetch_remit_event_detail_with_status(start_date, end_date)
    eleclink_export = load_eleclink_umm_export(eleclink_umm_export_path)
    operator_event = build_france_connector_operator_event_frame(raw_remit, eleclink_umm_export=eleclink_export)
    availability = build_fact_france_connector_availability_hourly(
        start_date=start_date,
        end_date=end_date,
        operator_event_frame=operator_event,
        remit_fetch_status_by_date=remit_status,
        eleclink_export_loaded=not eleclink_export.empty,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    operator_event.to_csv(output_path / f"{FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE}.csv", index=False)
    availability.to_csv(output_path / f"{FRANCE_CONNECTOR_AVAILABILITY_TABLE}.csv", index=False)
    return {
        FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE: operator_event,
        FRANCE_CONNECTOR_AVAILABILITY_TABLE: availability,
    }
