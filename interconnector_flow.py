from __future__ import annotations

import datetime as dt
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from gb_topology import INTERCONNECTOR_HUBS


ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
GB_DOMAIN_EIC = "10YGB----------A"
INTERCONNECTOR_FLOW_TABLE = "fact_interconnector_flow_hourly"
INTERCONNECTOR_FLOW_SOURCE_KEY = "entsoe_actual_physical_flow"
INTERCONNECTOR_FLOW_SOURCE_LABEL = "ENTSO-E actual physical flow"
INTERCONNECTOR_FLOW_DOCUMENT_TYPE = "A11"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc


@dataclass(frozen=True)
class BorderFlowSpec:
    border_key: str
    border_label: str
    target_zone: str
    neighbor_domain_key: str
    neighbor_domain_eic: str


BORDER_FLOW_SPECS: Tuple[BorderFlowSpec, ...] = (
    BorderFlowSpec(
        border_key="GB-FR",
        border_label="Great Britain to France aggregate border",
        target_zone="FR",
        neighbor_domain_key="FR",
        neighbor_domain_eic="10YFR-RTE------C",
    ),
    BorderFlowSpec(
        border_key="GB-NL",
        border_label="Great Britain to Netherlands aggregate border",
        target_zone="NL",
        neighbor_domain_key="NL",
        neighbor_domain_eic="10YNL----------L",
    ),
    BorderFlowSpec(
        border_key="GB-BE",
        border_label="Great Britain to Belgium aggregate border",
        target_zone="BE",
        neighbor_domain_key="BE",
        neighbor_domain_eic="10YBE----------2",
    ),
    BorderFlowSpec(
        border_key="GB-NO2",
        border_label="Great Britain to Norway bidding zone NO2 aggregate border",
        target_zone="NO",
        neighbor_domain_key="NO2",
        neighbor_domain_eic="10YNO-2--------T",
    ),
    BorderFlowSpec(
        border_key="GB-IE",
        border_label="Great Britain to Ireland aggregate border",
        target_zone="IE",
        neighbor_domain_key="IE",
        neighbor_domain_eic="10Y1001A1001A59C",
    ),
    BorderFlowSpec(
        border_key="GB-DK1",
        border_label="Great Britain to Denmark bidding zone DK1 aggregate border",
        target_zone="DK",
        neighbor_domain_key="DK1",
        neighbor_domain_eic="10YDK-1--------W",
    ),
)


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def _entsoe_token_from_env() -> str:
    return os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""


def _utc_window_for_local_date_range(start_date: dt.date, end_date: dt.date) -> tuple[dt.datetime, dt.datetime]:
    start_local = dt.datetime.combine(start_date, dt.time.min, tzinfo=LONDON_TZ)
    end_local = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time.min, tzinfo=LONDON_TZ)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def _iso_interval(start_utc: dt.datetime, end_utc: dt.datetime) -> tuple[str, str]:
    return start_utc.strftime("%Y%m%d%H%M"), end_utc.strftime("%Y%m%d%H%M")


def _parse_resolution_to_timedelta(resolution: str) -> pd.Timedelta:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", resolution or "")
    if not match:
        raise RuntimeError(f"unsupported ENTSO-E flow resolution: {resolution}")

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    step = pd.Timedelta(hours=hours, minutes=minutes)
    if step <= pd.Timedelta(0) or step > pd.Timedelta(hours=1):
        raise RuntimeError(f"unsupported ENTSO-E flow resolution: {resolution}")
    return step


def _parse_entsoe_error(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    namespace = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"ns": namespace} if namespace else {}
    text_xpath = ".//ns:text" if ns else ".//text"
    code_xpath = ".//ns:code" if ns else ".//code"

    codes = [value.text.strip() for value in root.findall(code_xpath, ns) if value.text]
    texts = [value.text.strip() for value in root.findall(text_xpath, ns) if value.text]

    if codes and texts:
        return "; ".join(f"{code}: {text}" for code, text in zip(codes, texts))
    if texts:
        return "; ".join(texts)
    return ""


def _fetch_entsoe_payload(url: str, source_name: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read()
        error_text = _parse_entsoe_error(error_body)
        if error_text:
            raise RuntimeError(f"{source_name} request failed ({exc.code}): {error_text}") from exc
        raise RuntimeError(f"{source_name} request failed with HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"{source_name} request failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{source_name} request timed out") from exc


def _hub_keys_for_target_zone(target_zone: str) -> tuple[str, ...]:
    return tuple(hub.key for hub in INTERCONNECTOR_HUBS.values() if hub.target_zone == target_zone)


def _hub_labels_for_target_zone(target_zone: str) -> tuple[str, ...]:
    return tuple(hub.label for hub in INTERCONNECTOR_HUBS.values() if hub.target_zone == target_zone)


def _direction_label(spec: BorderFlowSpec, direction_key: str) -> str:
    if direction_key == "gb_to_neighbor":
        return f"Great Britain to {spec.neighbor_domain_key}"
    return f"{spec.neighbor_domain_key} to Great Britain"


def _findtext_first(element: ET.Element, ns: dict[str, str], paths: Iterable[str]) -> str:
    for path in paths:
        value = element.findtext(path, namespaces=ns)
        if value:
            return value.strip()
    return ""


def _empty_flow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "source_key",
            "source_label",
            "source_document_type",
            "source_resolution",
            "normalized_resolution",
            "target_is_proxy",
            "flow_scope",
            "hub_assignment_mode",
            "hub_assignment_is_proxy",
            "border_key",
            "border_label",
            "target_zone",
            "neighbor_domain_key",
            "gb_domain_eic",
            "neighbor_domain_eic",
            "out_domain_eic",
            "in_domain_eic",
            "direction_key",
            "direction_label",
            "candidate_hub_keys",
            "candidate_hub_labels",
            "candidate_hub_count",
            "flow_mw",
            "flow_mwh",
            "signed_flow_from_gb_mw",
            "signed_flow_from_gb_mwh",
        ]
    )


def _duration_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not bool(valid.any()):
        return float("nan")
    filtered_values = values[valid].astype(float)
    filtered_weights = weights[valid].astype(float)
    total_weight = float(filtered_weights.sum())
    if total_weight <= 0:
        return float("nan")
    return float((filtered_values * filtered_weights).sum() / total_weight)


def _normalize_flow_rows_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_flow_frame()

    normalized = frame.copy()
    normalized["interval_hours"] = (
        normalized["interval_end_utc"] - normalized["interval_start_utc"]
    ) / pd.Timedelta(hours=1)
    normalized["hour_start_utc"] = normalized["interval_start_utc"].dt.floor("h")
    identity_columns = [
        "source_key",
        "source_label",
        "source_document_type",
        "target_is_proxy",
        "flow_scope",
        "hub_assignment_mode",
        "hub_assignment_is_proxy",
        "border_key",
        "border_label",
        "target_zone",
        "neighbor_domain_key",
        "gb_domain_eic",
        "neighbor_domain_eic",
        "out_domain_eic",
        "in_domain_eic",
        "direction_key",
        "direction_label",
        "candidate_hub_keys",
        "candidate_hub_labels",
        "candidate_hub_count",
    ]
    rows = []
    grouped = normalized.groupby(identity_columns + ["hour_start_utc"], dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(identity_columns + ["hour_start_utc"], keys))
        flow_mw = _duration_weighted_average(group["flow_mw"], group["interval_hours"])
        signed_flow_from_gb_mw = _duration_weighted_average(group["signed_flow_from_gb_mw"], group["interval_hours"])
        source_resolutions = sorted({str(value) for value in group["source_resolution"].dropna().unique()})
        hour_start_utc = pd.Timestamp(record.pop("hour_start_utc"))
        hour_end_utc = hour_start_utc + pd.Timedelta(hours=1)
        hour_start_local = hour_start_utc.tz_convert("Europe/London")
        hour_end_local = hour_end_utc.tz_convert("Europe/London")
        rows.append(
            {
                **record,
                "date": hour_start_local.date(),
                "interval_start_local": hour_start_local,
                "interval_end_local": hour_end_local,
                "interval_start_utc": hour_start_utc,
                "interval_end_utc": hour_end_utc,
                "source_resolution": ",".join(source_resolutions),
                "normalized_resolution": "PT60M",
                "flow_mw": flow_mw,
                "flow_mwh": float(group["flow_mw"].astype(float).mul(group["interval_hours"].astype(float)).sum()),
                "signed_flow_from_gb_mw": signed_flow_from_gb_mw,
                "signed_flow_from_gb_mwh": float(
                    group["signed_flow_from_gb_mw"].astype(float).mul(group["interval_hours"].astype(float)).sum()
                ),
            }
        )

    column_order = list(_empty_flow_frame().columns)
    return pd.DataFrame(rows)[column_order].sort_values(
        ["interval_start_utc", "border_key", "direction_key"]
    ).reset_index(drop=True)


def _expand_period_points(
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    step: pd.Timedelta,
    curve_type: str,
    points: list[ET.Element],
    ns: dict[str, str],
) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    expanded: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    parsed_points = []
    for point in points:
        position_text = _findtext_first(point, ns, ("ns:position", "position"))
        quantity_text = _findtext_first(point, ns, ("ns:quantity", "quantity"))
        if not position_text or not quantity_text:
            continue
        parsed_points.append((int(position_text), float(quantity_text)))

    if not parsed_points:
        return expanded

    parsed_points.sort(key=lambda item: item[0])
    total_step_count = int((period_end - period_start) / step)
    step_count = max(total_step_count, parsed_points[-1][0])

    if curve_type == "A03":
        for idx, (position, quantity) in enumerate(parsed_points):
            next_position = parsed_points[idx + 1][0] if idx + 1 < len(parsed_points) else step_count + 1
            for step_position in range(position, next_position):
                interval_start = period_start + (step_position - 1) * step
                interval_end = interval_start + step
                if interval_start >= period_end:
                    break
                expanded.append((interval_start, min(interval_end, period_end), quantity))
        return expanded

    for position, quantity in parsed_points:
        interval_start = period_start + (position - 1) * step
        interval_end = interval_start + step
        if interval_start >= period_end:
            continue
        expanded.append((interval_start, min(interval_end, period_end), quantity))
    return expanded


def parse_entsoe_interconnector_flow_xml(
    xml_bytes: bytes,
    spec: BorderFlowSpec,
    direction_key: str,
    requested_start_utc: dt.datetime,
    requested_end_utc: dt.datetime,
) -> pd.DataFrame:
    error_text = _parse_entsoe_error(xml_bytes)
    if error_text and "no matching data" in error_text.lower():
        return _empty_flow_frame()

    root = ET.fromstring(xml_bytes)
    namespace = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"ns": namespace} if namespace else {}
    time_series = root.findall(".//ns:TimeSeries", ns) if ns else root.findall(".//TimeSeries")
    if not time_series:
        if error_text:
            raise RuntimeError(f"{spec.border_key} {direction_key} returned no usable flow series: {error_text}")
        raise RuntimeError(f"{spec.border_key} {direction_key} returned no usable flow series")

    hub_keys = _hub_keys_for_target_zone(spec.target_zone)
    hub_labels = _hub_labels_for_target_zone(spec.target_zone)
    hub_assignment_mode = "aggregate_border_candidate_hubs" if len(hub_keys) > 1 else "single_hub_border_proxy"
    sign_from_gb = 1.0 if direction_key == "gb_to_neighbor" else -1.0

    rows = []
    for ts in time_series:
        curve_type = _findtext_first(ts, ns, ("ns:curveType", "curveType"))
        out_domain_eic = _findtext_first(
            ts,
            ns,
            (
                "ns:outBiddingZone_Domain.mRID",
                "ns:out_Domain.mRID",
                "ns:outArea_Domain.mRID",
            ),
        )
        in_domain_eic = _findtext_first(
            ts,
            ns,
            (
                "ns:inBiddingZone_Domain.mRID",
                "ns:in_Domain.mRID",
                "ns:inArea_Domain.mRID",
            ),
        )
        for period in ts.findall(".//ns:Period", ns) if ns else ts.findall(".//Period"):
            start_text = _findtext_first(period, ns, ("ns:timeInterval/ns:start", "timeInterval/start"))
            end_text = _findtext_first(period, ns, ("ns:timeInterval/ns:end", "timeInterval/end"))
            resolution = _findtext_first(period, ns, ("ns:resolution", "resolution")) or "PT60M"
            if not start_text or not end_text:
                continue
            step = _parse_resolution_to_timedelta(resolution)
            period_start = pd.to_datetime(start_text, utc=True)
            period_end = pd.to_datetime(end_text, utc=True)

            points = period.findall("ns:Point", ns) if ns else period.findall("Point")
            for interval_start, interval_end, flow_mw in _expand_period_points(
                period_start=period_start,
                period_end=period_end,
                step=step,
                curve_type=curve_type,
                points=points,
                ns=ns,
            ):
                if interval_start < requested_start_utc or interval_start >= requested_end_utc:
                    continue
                rows.append(
                    {
                        "interval_start_utc": interval_start,
                        "interval_end_utc": interval_end,
                        "source_resolution": resolution,
                        "out_domain_eic": out_domain_eic or (GB_DOMAIN_EIC if direction_key == "gb_to_neighbor" else spec.neighbor_domain_eic),
                        "in_domain_eic": in_domain_eic or (spec.neighbor_domain_eic if direction_key == "gb_to_neighbor" else GB_DOMAIN_EIC),
                        "flow_mw": flow_mw,
                        "signed_flow_from_gb_mw": sign_from_gb * flow_mw,
                    }
                )

    if not rows:
        if error_text and "no matching data" in error_text.lower():
            return _empty_flow_frame()
        return _empty_flow_frame()

    frame = pd.DataFrame(rows).sort_values("interval_start_utc").drop_duplicates(
        subset=["interval_start_utc", "out_domain_eic", "in_domain_eic"],
        keep="last",
    )
    frame["source_key"] = INTERCONNECTOR_FLOW_SOURCE_KEY
    frame["source_label"] = INTERCONNECTOR_FLOW_SOURCE_LABEL
    frame["source_document_type"] = INTERCONNECTOR_FLOW_DOCUMENT_TYPE
    frame["target_is_proxy"] = False
    frame["flow_scope"] = "aggregate_border_bidding_zone"
    frame["hub_assignment_mode"] = hub_assignment_mode
    frame["hub_assignment_is_proxy"] = True
    frame["border_key"] = spec.border_key
    frame["border_label"] = spec.border_label
    frame["target_zone"] = spec.target_zone
    frame["neighbor_domain_key"] = spec.neighbor_domain_key
    frame["gb_domain_eic"] = GB_DOMAIN_EIC
    frame["neighbor_domain_eic"] = spec.neighbor_domain_eic
    frame["direction_key"] = direction_key
    frame["direction_label"] = _direction_label(spec, direction_key)
    frame["candidate_hub_keys"] = ",".join(hub_keys)
    frame["candidate_hub_labels"] = ",".join(hub_labels)
    frame["candidate_hub_count"] = len(hub_keys)
    return _normalize_flow_rows_to_hourly(frame)


def fetch_interconnector_flow_direction(
    spec: BorderFlowSpec,
    direction_key: str,
    start_date: dt.date,
    end_date: dt.date,
    token: str,
) -> pd.DataFrame:
    if direction_key not in {"gb_to_neighbor", "neighbor_to_gb"}:
        raise ValueError(f"unsupported direction_key '{direction_key}'")

    requested_start_utc, requested_end_utc = _utc_window_for_local_date_range(start_date, end_date)
    period_start, period_end = _iso_interval(requested_start_utc, requested_end_utc)
    out_domain = GB_DOMAIN_EIC if direction_key == "gb_to_neighbor" else spec.neighbor_domain_eic
    in_domain = spec.neighbor_domain_eic if direction_key == "gb_to_neighbor" else GB_DOMAIN_EIC
    params = {
        "securityToken": token,
        "documentType": INTERCONNECTOR_FLOW_DOCUMENT_TYPE,
        "out_Domain": out_domain,
        "in_Domain": in_domain,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    url = ENTSOE_ENDPOINT + "?" + urllib.parse.urlencode(params)
    payload = _fetch_entsoe_payload(url, f"{spec.border_key} {direction_key}")
    return parse_entsoe_interconnector_flow_xml(
        payload,
        spec=spec,
        direction_key=direction_key,
        requested_start_utc=requested_start_utc,
        requested_end_utc=requested_end_utc,
    )


def build_fact_interconnector_flow_hourly(
    start_date: dt.date,
    end_date: dt.date,
    token: str,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not token:
        raise RuntimeError("missing ENTSO-E token; set ENTOS_E_TOKEN or ENTSOE_TOKEN")

    frames = []
    for spec in BORDER_FLOW_SPECS:
        for direction_key in ("gb_to_neighbor", "neighbor_to_gb"):
            frame = fetch_interconnector_flow_direction(
                spec=spec,
                direction_key=direction_key,
                start_date=start_date,
                end_date=end_date,
                token=token,
            )
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return _empty_flow_frame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["interval_start_utc", "border_key", "direction_key"])
        .reset_index(drop=True)
    )


def materialize_interconnector_flow_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    token: str | None = None,
) -> Dict[str, pd.DataFrame]:
    resolved_token = token or _entsoe_token_from_env()
    fact = build_fact_interconnector_flow_hourly(start_date=start_date, end_date=end_date, token=resolved_token)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{INTERCONNECTOR_FLOW_TABLE}.csv", index=False)
    return {INTERCONNECTOR_FLOW_TABLE: fact}
