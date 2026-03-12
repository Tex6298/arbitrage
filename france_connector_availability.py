from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from bmu_availability import _iter_remit_windows, fetch_remit_event_detail_with_status
from france_connector import FRANCE_CONNECTOR_SPECS, interconnector_cable_frame


FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE = "fact_france_connector_operator_event"
FRANCE_CONNECTOR_AVAILABILITY_TABLE = "fact_france_connector_availability_hourly"
FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE = "fact_france_connector_operator_source_compare"

NORDPOOL_UMM_TOKEN_URL = "https://sts.nordpoolgroup.com/connect/token"
NORDPOOL_UMM_MESSAGES_URL = "https://ummapi.nordpoolgroup.com/messages"


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
        source_provider="nordpool_umm",
        source_key="nordpool_umm",
        source_label="Nord Pool UMM",
        match_patterns=(r"\beleclink\b",),
        requires_external_export=True,
        note="ElecLink moved outage publication to Nord Pool UMM on June 3, 2024; the repo supports both an authenticated API path and a manual export fallback.",
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


def _normalize_eleclink_umm_frame(
    frame: pd.DataFrame,
    *,
    source_key: str,
    source_label: str,
    connector_match_rule: str,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_operator_event_frame()

    frame = frame.rename(
        columns={
            "publishTime": "publish_time_utc",
            "publishedAt": "publish_time_utc",
            "published_at": "publish_time_utc",
            "eventStartTime": "event_start_utc",
            "startTime": "event_start_utc",
            "eventEndTime": "event_end_utc",
            "endTime": "event_end_utc",
            "normalCapacity": "normal_capacity_mw",
            "nominalCapacity": "normal_capacity_mw",
            "availableCapacity": "available_capacity_mw",
            "unavailableCapacity": "unavailable_capacity_mw",
            "messageType": "message_type",
            "eventType": "event_type",
            "eventStatus": "event_status",
            "status": "event_status",
            "unavailabilityType": "unavailability_type",
            "assetId": "asset_id",
            "assetName": "asset_id",
            "affectedUnit": "affected_unit",
            "registrationCode": "registration_code",
        }
    ).copy()

    if "connector_key" not in frame.columns:
        blob = _connector_text_blob(frame)
        frame["connector_key"] = blob.map(_match_connector_from_text)
    frame["connector_key"] = frame["connector_key"].fillna("eleclink")
    frame = frame[frame["connector_key"].astype(str).str.lower().eq("eleclink")].copy()
    if frame.empty:
        return _empty_operator_event_frame()

    frame["publish_time_utc"] = pd.to_datetime(frame.get("publish_time_utc"), utc=True, errors="coerce")
    frame["event_start_utc"] = pd.to_datetime(frame.get("event_start_utc"), utc=True, errors="coerce")
    frame["event_end_utc"] = pd.to_datetime(frame.get("event_end_utc"), utc=True, errors="coerce")
    frame["normal_capacity_mw"] = pd.to_numeric(frame.get("normal_capacity_mw"), errors="coerce")
    frame["available_capacity_mw"] = pd.to_numeric(frame.get("available_capacity_mw"), errors="coerce")
    frame["unavailable_capacity_mw"] = pd.to_numeric(frame.get("unavailable_capacity_mw"), errors="coerce")
    frame["connector_label"] = "ElecLink"
    frame["source_provider"] = "nordpool_umm"
    frame["source_key"] = source_key
    frame["source_label"] = source_label
    frame["source_truth_tier"] = "operator_outage_truth"
    frame["connector_match_rule"] = connector_match_rule
    frame["target_is_proxy"] = False
    return frame[
        list(_empty_operator_event_frame().columns)
    ].dropna(subset=["event_start_utc", "event_end_utc"], how="any").reset_index(drop=True)


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

    return _normalize_eleclink_umm_frame(
        frame,
        source_key="nordpool_umm_export",
        source_label="Nord Pool UMM export",
        connector_match_rule="manual_eleclink_export",
    )


def _extract_umm_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("items", "messages", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    return []


def fetch_eleclink_umm_authenticated(
    *,
    username: str | None = None,
    password: str | None = None,
    client_authorization: str | None = None,
    scope: str | None = None,
    access_token: str | None = None,
    token_url: str = NORDPOOL_UMM_TOKEN_URL,
    messages_url: str = NORDPOOL_UMM_MESSAGES_URL,
    timeout_seconds: int = 30,
) -> tuple[pd.DataFrame, dict]:
    status = {
        "connector_key": "eleclink",
        "source_variant_key": "nordpool_umm_authenticated_api",
        "source_provider": "nordpool_umm",
        "source_key": "nordpool_umm_authenticated_api",
        "source_label": "Nord Pool UMM authenticated API",
        "source_attempted_flag": False,
        "source_fetch_ok": False,
        "source_gap_reason": pd.NA,
    }

    token = (access_token or "").strip()
    username = (username or "").strip()
    password = (password or "").strip()
    client_authorization = (client_authorization or "").strip()
    scope = (scope or "").strip()

    if not token and not (username and password and client_authorization):
        status["source_gap_reason"] = "nordpool_umm_credentials_not_provided"
        return _empty_operator_event_frame(), status

    status["source_attempted_flag"] = True
    try:
        if not token:
            token_body = {
                "grant_type": "password",
                "username": username,
                "password": password,
            }
            if scope:
                token_body["scope"] = scope
            auth_header = client_authorization
            if auth_header and not auth_header.lower().startswith("basic "):
                auth_header = f"Basic {auth_header}"
            token_request = urllib.request.Request(
                token_url,
                data=urllib.parse.urlencode(token_body).encode("utf-8"),
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(token_request, timeout=timeout_seconds) as response:
                token_payload = json.loads(response.read().decode("utf-8"))
            token = str(token_payload.get("access_token") or "").strip()
            if not token:
                raise RuntimeError("Nord Pool token response did not include access_token")

        message_request = urllib.request.Request(
            messages_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            method="GET",
        )
        with urllib.request.urlopen(message_request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        frame = pd.DataFrame(_extract_umm_records(payload))
        status["source_fetch_ok"] = True
        status["source_gap_reason"] = pd.NA
        normalized = _normalize_eleclink_umm_frame(
            frame,
            source_key="nordpool_umm_authenticated_api",
            source_label="Nord Pool UMM authenticated API",
            connector_match_rule="authenticated_api_match",
        )
        return normalized, status
    except urllib.error.HTTPError as exc:
        status["source_gap_reason"] = f"nordpool_umm_http_{exc.code}"
        return _empty_operator_event_frame(), status
    except Exception as exc:
        status["source_gap_reason"] = f"nordpool_umm_fetch_error:{type(exc).__name__}"
        return _empty_operator_event_frame(), status


def _requested_window_bounds(start_date: dt.date, end_date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_utc = pd.Timestamp(start_date, tz="Europe/London").tz_convert("UTC")
    end_utc = pd.Timestamp(end_date + dt.timedelta(days=1), tz="Europe/London").tz_convert("UTC")
    return start_utc, end_utc


def _event_overlap_summary(
    frame: pd.DataFrame,
    *,
    start_date: dt.date,
    end_date: dt.date,
) -> dict:
    event_count = len(frame)
    latest_publish_time = pd.NaT
    overlap_event_count = 0
    overlap_hour_count = 0
    if frame is not None and not frame.empty:
        working = frame.copy()
        working["event_start_utc"] = pd.to_datetime(working["event_start_utc"], utc=True, errors="coerce")
        working["event_end_utc"] = pd.to_datetime(working["event_end_utc"], utc=True, errors="coerce")
        working["publish_time_utc"] = pd.to_datetime(working["publish_time_utc"], utc=True, errors="coerce")
        latest_publish_time = working["publish_time_utc"].max()
        window_start, window_end = _requested_window_bounds(start_date, end_date)
        overlap = working[
            working["event_start_utc"].notna()
            & working["event_end_utc"].notna()
            & (working["event_end_utc"] > window_start)
            & (working["event_start_utc"] < window_end)
        ].copy()
        overlap_event_count = len(overlap)
        if not overlap.empty:
            interval_index = _hourly_window_frame(start_date, end_date)
            overlap_hours = 0
            for row in overlap.itertuples(index=False):
                matching = interval_index[
                    (interval_index["interval_end_utc"] > row.event_start_utc)
                    & (interval_index["interval_start_utc"] < row.event_end_utc)
                ]
                overlap_hours += len(matching)
            overlap_hour_count = overlap_hours
    return {
        "source_event_count": int(event_count),
        "source_overlap_event_count": int(overlap_event_count),
        "source_overlap_hour_count": int(overlap_hour_count),
        "source_latest_publish_time_utc": latest_publish_time,
    }


def _empty_source_compare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "requested_start_date",
            "requested_end_date",
            "selection_context",
            "connector_key",
            "source_variant_key",
            "source_provider",
            "source_key",
            "source_label",
            "source_attempted_flag",
            "source_fetch_ok",
            "source_event_count",
            "source_overlap_event_count",
            "source_overlap_hour_count",
            "source_latest_publish_time_utc",
            "source_selected_flag",
            "source_selection_rank",
            "source_selection_reason",
            "source_gap_reason",
        ]
    )


def build_eleclink_operator_source_compare(
    *,
    start_date: dt.date,
    end_date: dt.date,
    authenticated_frame: pd.DataFrame,
    authenticated_status: dict,
    export_frame: pd.DataFrame,
    export_attempted_flag: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    today = dt.datetime.now(dt.timezone.utc).date()
    historical_replay = end_date < (today - dt.timedelta(days=7))
    selection_context = "historical_replay" if historical_replay else "current_operational"

    auth_summary = {
        **authenticated_status,
        **_event_overlap_summary(authenticated_frame, start_date=start_date, end_date=end_date),
    }
    export_summary = {
        "connector_key": "eleclink",
        "source_variant_key": "nordpool_umm_export",
        "source_provider": "nordpool_umm",
        "source_key": "nordpool_umm_export",
        "source_label": "Nord Pool UMM export",
        "source_attempted_flag": bool(export_attempted_flag),
        "source_fetch_ok": bool(export_attempted_flag),
        "source_gap_reason": (pd.NA if export_attempted_flag else "nordpool_umm_export_not_provided"),
        **_event_overlap_summary(export_frame, start_date=start_date, end_date=end_date),
    }

    selected_variant = None
    selection_reason = "no_eleclink_source_available"
    if historical_replay:
        if export_summary["source_fetch_ok"]:
            selected_variant = "nordpool_umm_export"
            selection_reason = "manual_export_preferred_for_historical_replay"
        elif auth_summary["source_fetch_ok"] and auth_summary["source_overlap_hour_count"] > 0:
            selected_variant = "nordpool_umm_authenticated_api"
            selection_reason = "authenticated_api_has_historical_overlap"
    else:
        if auth_summary["source_fetch_ok"]:
            selected_variant = "nordpool_umm_authenticated_api"
            selection_reason = "authenticated_api_preferred_for_current_window"
        elif export_summary["source_fetch_ok"]:
            selected_variant = "nordpool_umm_export"
            selection_reason = "manual_export_fallback_for_current_window"

    compare_rows = []
    for rank, summary in enumerate((auth_summary, export_summary), start=1):
        row = {
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "selection_context": selection_context,
            **summary,
            "source_selected_flag": summary["source_variant_key"] == selected_variant,
            "source_selection_rank": rank,
            "source_selection_reason": selection_reason if summary["source_variant_key"] == selected_variant else pd.NA,
        }
        compare_rows.append(row)

    compare = pd.DataFrame(compare_rows, columns=list(_empty_source_compare_frame().columns))
    selected_frame = (
        authenticated_frame.copy()
        if selected_variant == "nordpool_umm_authenticated_api"
        else export_frame.copy()
        if selected_variant == "nordpool_umm_export"
        else _empty_operator_event_frame()
    )
    selected_row = compare[compare["source_selected_flag"]].head(1)
    resolution = {
        "selected_variant_key": selected_variant,
        "selection_context": selection_context,
        "selection_reason": selection_reason,
        "source_provider": "nordpool_umm",
        "source_key": (selected_row.iloc[0]["source_key"] if not selected_row.empty else "nordpool_umm"),
        "source_label": (selected_row.iloc[0]["source_label"] if not selected_row.empty else "Nord Pool UMM"),
        "source_fetch_ok": (bool(selected_row.iloc[0]["source_fetch_ok"]) if not selected_row.empty else False),
        "source_gap_reason": (selected_row.iloc[0]["source_gap_reason"] if not selected_row.empty else "no_eleclink_source_available"),
    }
    return selected_frame, compare, resolution


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
    eleclink_source_resolution: dict | None = None,
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
    eleclink_mask = base["connector_key"].eq("eleclink")
    eleclink_resolution = eleclink_source_resolution or {}
    base.loc[eleclink_mask, "operator_fetch_ok"] = bool(eleclink_resolution.get("source_fetch_ok", False))
    if eleclink_resolution:
        base.loc[eleclink_mask, "source_provider"] = eleclink_resolution.get("source_provider", "nordpool_umm")
        base.loc[eleclink_mask, "source_key"] = eleclink_resolution.get("source_key", "nordpool_umm")
        base.loc[eleclink_mask, "source_label"] = eleclink_resolution.get("source_label", "Nord Pool UMM")
    base = base.drop(columns=["remit_fetch_ok"], errors="ignore")

    base["operator_source_gap_reason"] = pd.NA
    base.loc[remit_connector_mask & ~base["operator_fetch_ok"], "operator_source_gap_reason"] = "elexon_remit_fetch_failed"
    base.loc[eleclink_mask & ~base["operator_fetch_ok"], "operator_source_gap_reason"] = eleclink_resolution.get(
        "source_gap_reason",
        "nordpool_umm_source_not_available",
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
    eleclink_umm_authenticated: pd.DataFrame | None = None,
    eleclink_authenticated_status: dict | None = None,
) -> Dict[str, pd.DataFrame]:
    raw_remit, remit_status = fetch_remit_event_detail_with_status(start_date, end_date)
    eleclink_export = load_eleclink_umm_export(eleclink_umm_export_path)
    export_attempted_flag = bool(eleclink_umm_export_path)
    authenticated_frame = eleclink_umm_authenticated if eleclink_umm_authenticated is not None else _empty_operator_event_frame()
    authenticated_status = eleclink_authenticated_status or {
        "connector_key": "eleclink",
        "source_variant_key": "nordpool_umm_authenticated_api",
        "source_provider": "nordpool_umm",
        "source_key": "nordpool_umm_authenticated_api",
        "source_label": "Nord Pool UMM authenticated API",
        "source_attempted_flag": False,
        "source_fetch_ok": False,
        "source_gap_reason": "nordpool_umm_credentials_not_provided",
    }
    selected_eleclink, source_compare, eleclink_resolution = build_eleclink_operator_source_compare(
        start_date=start_date,
        end_date=end_date,
        authenticated_frame=authenticated_frame,
        authenticated_status=authenticated_status,
        export_frame=eleclink_export,
        export_attempted_flag=export_attempted_flag,
    )
    operator_event = build_france_connector_operator_event_frame(raw_remit, eleclink_umm_export=selected_eleclink)
    availability = build_fact_france_connector_availability_hourly(
        start_date=start_date,
        end_date=end_date,
        operator_event_frame=operator_event,
        remit_fetch_status_by_date=remit_status,
        eleclink_source_resolution=eleclink_resolution,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    operator_event.to_csv(output_path / f"{FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE}.csv", index=False)
    availability.to_csv(output_path / f"{FRANCE_CONNECTOR_AVAILABILITY_TABLE}.csv", index=False)
    source_compare.to_csv(output_path / f"{FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE}.csv", index=False)
    return {
        FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE: operator_event,
        FRANCE_CONNECTOR_AVAILABILITY_TABLE: availability,
        FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE: source_compare,
    }
