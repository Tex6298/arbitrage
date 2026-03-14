#!/usr/bin/env python3
"""
inline_arbitrage_live.py
------------------------
Fetches GB market-index prices from Elexon MID and continental day-ahead prices from
ENTSO-E for FR, NL, DE-LU, PL, CZ, then computes simple GB->(FR|NL)->DE->PL netbacks
per hour, including basic losses and placeholder fees.

Important behavior:
- Uses the current Elexon market-index feed for GB and documented ENTSO-E area EIC
  codes for continental zones.
- Queries per local market day for each zone, then converts timestamps to UTC.
- Synthetic data is available only with --dry. Missing tokens and API failures are
  surfaced as errors instead of silently falling back.
- GB prices are published in GBP, so you must provide a GBP->EUR conversion rate via
  --gbp-eur or GBP_EUR / GBP_EUR_RATE.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from asset_mapping import cluster_frame, parent_region_frame, signal_source_frame, weather_anchor_frame
from bmu_availability import fetch_remit_event_detail_with_status
from bmu_dispatch import materialize_bmu_dispatch_history, parse_iso_date as parse_dispatch_iso_date
from bmu_generation import materialize_bmu_generation_history, parse_iso_date as parse_bmu_iso_date
from curtailment_truth import materialize_bmu_curtailment_truth
from curtailment_signals import materialize_curtailed_history, parse_iso_date
from curtailment_opportunity import (
    CURTAILMENT_OPPORTUNITY_TABLE,
    VALID_OPPORTUNITY_TRUTH_PROFILES,
    fetch_cluster_curtailment_proxy_hourly,
    materialize_curtailment_opportunity_history,
)
from day_ahead_constraint_boundary import (
    DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE,
    build_fact_day_ahead_constraint_boundary_half_hourly,
    materialize_day_ahead_constraint_boundary_history,
    parse_iso_date as parse_boundary_iso_date,
)
from opportunity_backtest import (
    BACKTEST_PREDICTION_TABLE,
    BACKTEST_SUMMARY_SLICE_TABLE,
    BACKTEST_TOP_ERROR_TABLE,
    DEFAULT_BACKTEST_MODEL_SELECTION,
    DEFAULT_FORECAST_HORIZON_HOURS,
    DRIFT_WINDOW_TABLE,
    VALID_BACKTEST_MODEL_SELECTIONS,
    coerce_forecast_horizons,
    load_curtailment_opportunity_input,
    materialize_opportunity_backtest,
    summarize_backtest_prediction_hourly,
)
from exploration_plan import backtest_plan_frame, dataset_plan_frame, drift_monitor_plan_frame, map_layer_plan_frame
from france_connector import (
    DIM_INTERCONNECTOR_CABLE_TABLE,
    FRANCE_CONNECTOR_TABLE,
    build_fact_france_connector_hourly,
    interconnector_cable_frame,
    materialize_france_connector_history,
)
from france_connector_reviewed import (
    FRANCE_CONNECTOR_NOTICE_TABLE,
    FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE,
    build_fact_france_connector_notice_hourly,
    build_fact_france_connector_reviewed_period,
    write_normalized_france_connector_reviewed_input,
)
from france_connector_availability import (
    FRANCE_CONNECTOR_AVAILABILITY_TABLE,
    FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE,
    FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE,
    NORDPOOL_UMM_MESSAGES_URL,
    NORDPOOL_UMM_TOKEN_URL,
    build_fact_france_connector_availability_hourly,
    build_eleclink_operator_source_compare,
    build_france_connector_operator_event_frame,
    fetch_eleclink_umm_authenticated,
    load_eleclink_umm_export,
)
from gb_transfer_gate import (
    GB_TRANSFER_GATE_TABLE,
    build_fact_gb_transfer_gate_hourly,
    materialize_gb_transfer_gate_history,
    parse_iso_date as parse_transfer_iso_date,
)
from gb_transfer_boundary_reviewed import (
    GB_TRANSFER_BOUNDARY_REVIEWED_TABLE,
    build_fact_gb_transfer_boundary_reviewed_hourly,
)
from gb_transfer_reviewed import (
    GB_TRANSFER_REVIEW_POLICY_TABLE,
    GB_TRANSFER_REVIEWED_HOURLY_TABLE,
    GB_TRANSFER_REVIEWED_PERIOD_TABLE,
    build_fact_gb_transfer_review_policy,
    build_fact_gb_transfer_reviewed_hourly,
    build_fact_gb_transfer_reviewed_period,
    materialize_gb_transfer_reviewed_history,
    write_normalized_gb_transfer_reviewed_input,
)
from gb_topology import cluster_hub_matrix, interconnector_hub_frame, reachability_frame, route_hub_frame
from history_store import ingest_truth_csv_tree_to_sqlite, upsert_truth_frames_to_sqlite
from interconnector_capacity import (
    INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE,
    INTERCONNECTOR_CAPACITY_AUDIT_VARIANT_TABLE,
    INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE,
    INTERCONNECTOR_CAPACITY_REVIEWED_TABLE,
    build_fact_interconnector_capacity_hourly,
    build_interconnector_capacity_reviewed_hourly,
    build_interconnector_capacity_source_audit,
    build_interconnector_capacity_review_policy,
    materialize_interconnector_capacity_review_policy,
    materialize_interconnector_capacity_source_audit,
    materialize_interconnector_capacity_history,
    parse_iso_date as parse_capacity_iso_date,
)
from interconnector_flow import (
    build_fact_interconnector_flow_hourly,
    materialize_interconnector_flow_history,
    parse_iso_date as parse_flow_iso_date,
)
from interconnector_itl import (
    INTERCONNECTOR_ITL_TABLE,
    build_fact_interconnector_itl_hourly,
    materialize_interconnector_itl_history,
    parse_iso_date as parse_itl_iso_date,
)
from market_state_feed import (
    UPSTREAM_MARKET_STATE_TABLE,
    build_fact_upstream_market_state_hourly,
    build_fact_upstream_market_state_hourly_from_price_frame,
    materialize_upstream_market_state_history,
    write_normalized_upstream_market_state_input,
)
from physical_constraints import assumption_frame, compute_netbacks
from route_score_history import ROUTE_SCORE_TABLE, materialize_route_score_history
from support_resolution import (
    SUPPORT_CASE_RESOLUTION_TABLE,
    SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE,
    SUPPORT_RERUN_CANDIDATE_DAILY_TABLE,
    SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE,
    SUPPORT_RERUN_GATE_BATCH_TABLE,
    SUPPORT_RERUN_GATE_DAILY_TABLE,
    SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE,
    SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE,
    SUPPORT_RESOLUTION_BATCH_TABLE,
    SUPPORT_RESOLUTION_DAILY_TABLE,
    VALID_RESOLUTION_FILTERS,
    VALID_RESOLUTION_STATES,
    VALID_SUPPORT_GATE_FILTERS,
    VALID_SUPPORT_PATTERN_FILTERS,
    VALID_SUPPORT_RERUN_CANDIDATE_FILTERS,
    VALID_TRUTH_POLICY_ACTIONS,
    annotate_support_resolution_pattern,
    annotate_support_case_resolution,
    materialize_truth_store_support_resolution,
    read_support_case_resolution,
    read_support_resolution_pattern_review,
    read_support_rerun_candidate_review,
    read_support_rerun_gate_review,
    read_support_resolution_review,
)
from support_loop import (
    SUPPORT_CASE_DAILY_TABLE,
    SUPPORT_CASE_FAMILY_TABLE,
    SUPPORT_CASE_HALF_HOURLY_TABLE,
    SUPPORT_SUMMARY_FILENAME,
    materialize_truth_store_support_loop,
)
from truth_store_forensics import (
    forensic_scope_key_for_family_keys,
    materialize_truth_store_family_forensics,
    read_truth_store_family_forensics,
    write_family_support_extract_csvs,
)
from truth_store_focus import materialize_truth_store_source_focus, read_truth_store_source_focus
from weather_history import materialize_weather_history

# Optional: .env support
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
ELEXON_ENDPOINT = "https://data.elexon.co.uk/bmrs/api/v1"
BMRS_KEY_ENV_NAMES = ("BMRS_API_KEY",)
GB_PROVIDER_ENV_NAMES = ("GB_DATA_PROVIDER", "BMRS_DATA_PROVIDER")
FX_ENV_NAMES = ("GBP_EUR", "GBP_EUR_RATE")
DEFAULT_GB_PROVIDER = "APXMIDP"
NORDPOOL_UMM_USERNAME_ENV_NAMES = ("NORDPOOL_UMM_USERNAME",)
NORDPOOL_UMM_PASSWORD_ENV_NAMES = ("NORDPOOL_UMM_PASSWORD",)
NORDPOOL_UMM_CLIENT_AUTH_ENV_NAMES = ("NORDPOOL_UMM_CLIENT_AUTHORIZATION",)
NORDPOOL_UMM_SCOPE_ENV_NAMES = ("NORDPOOL_UMM_SCOPE",)
NORDPOOL_UMM_ACCESS_TOKEN_ENV_NAMES = ("NORDPOOL_UMM_ACCESS_TOKEN",)


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    eic: str
    timezone: str


# ENTSO-E continental bidding zone EIC codes and their local market time zones.
CONTINENTAL_ZONES: Dict[str, ZoneSpec] = {
    "FR": ZoneSpec("FR", "10YFR-RTE------C", "Europe/Paris"),
    "NL": ZoneSpec("NL", "10YNL----------L", "Europe/Amsterdam"),
    "DE": ZoneSpec("DE", "10Y1001A1001A82H", "Europe/Berlin"),
    "PL": ZoneSpec("PL", "10YPL-AREA-----S", "Europe/Warsaw"),
    "CZ": ZoneSpec("CZ", "10YCZ-CEPS-----N", "Europe/Prague"),
}


def parse_market_day(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def iter_market_days(start_day: dt.date, end_day: dt.date) -> Iterable[dt.date]:
    day = start_day
    while day < end_day:
        yield day
        day += dt.timedelta(days=1)


def resolve_market_days(args: argparse.Namespace) -> List[dt.date]:
    if args.date and (args.start or args.end):
        raise ValueError("use either --date or --start/--end, not both")

    today_gb = dt.datetime.now(ZoneInfo("Europe/London")).date()

    if args.date:
        start_day = parse_market_day(args.date)
        end_day = start_day + dt.timedelta(days=1)
    else:
        start_day = parse_market_day(args.start) if args.start else today_gb
        end_day = parse_market_day(args.end) if args.end else start_day + dt.timedelta(days=1)

    if end_day <= start_day:
        raise ValueError("--end must be after --start")

    return list(iter_market_days(start_day, end_day))


def entsoe_utc_window_for_local_day(day: dt.date, tz_name: str) -> Tuple[dt.datetime, dt.datetime]:
    tz = ZoneInfo(tz_name)
    start_local = dt.datetime.combine(day, dt.time.min, tzinfo=tz)
    end_local = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, tzinfo=tz)
    return start_local.astimezone(dt.timezone.utc), end_local.astimezone(dt.timezone.utc)


def iso_interval(start_utc: dt.datetime, end_utc: dt.datetime) -> Tuple[str, str]:
    return start_utc.strftime("%Y%m%d%H%M"), end_utc.strftime("%Y%m%d%H%M")


def rfc3339_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def parse_resolution_to_timedelta(resolution: str) -> pd.Timedelta:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", resolution or "")
    if not match:
        raise RuntimeError(f"unsupported ENTSO-E resolution: {resolution}")

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    step = pd.Timedelta(hours=hours, minutes=minutes)
    if step <= pd.Timedelta(0):
        raise RuntimeError(f"unsupported ENTSO-E resolution: {resolution}")
    return step


def parse_entsoe_error(xml_bytes: bytes) -> str:
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


def parse_json_error(payload: bytes) -> str:
    try:
        body = json.loads(payload.decode("utf-8"))
    except Exception:
        return payload.decode("utf-8", "ignore").strip()

    if isinstance(body, dict):
        for key in ("detail", "description", "message", "title", "error"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            for key in ("description", "message"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return ""


def _first_env_value(names: Tuple[str, ...]) -> str | None:
    return next((os.environ.get(name) for name in names if os.environ.get(name)), None)


def resolve_eleclink_umm_auth_args(args: argparse.Namespace) -> dict:
    return {
        "username": args.eleclink_umm_username or _first_env_value(NORDPOOL_UMM_USERNAME_ENV_NAMES),
        "password": args.eleclink_umm_password or _first_env_value(NORDPOOL_UMM_PASSWORD_ENV_NAMES),
        "client_authorization": args.eleclink_umm_client_authorization or _first_env_value(NORDPOOL_UMM_CLIENT_AUTH_ENV_NAMES),
        "scope": args.eleclink_umm_scope or _first_env_value(NORDPOOL_UMM_SCOPE_ENV_NAMES),
        "access_token": args.eleclink_umm_access_token or _first_env_value(NORDPOOL_UMM_ACCESS_TOKEN_ENV_NAMES),
        "token_url": args.eleclink_umm_token_url,
        "messages_url": args.eleclink_umm_messages_url,
    }


def fetch_entsoe_payload(url: str, zone_name: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read()
        error_text = parse_entsoe_error(error_body)
        if error_text:
            raise RuntimeError(f"{zone_name} request failed ({exc.code}): {error_text}") from exc
        raise RuntimeError(f"{zone_name} request failed with HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"{zone_name} request failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{zone_name} request timed out") from exc


def fetch_elexon_payload(url: str, source_name: str, api_key: str | None) -> bytes:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Ocp-Apim-Subscription-Key"] = api_key

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read()
        error_text = parse_json_error(error_body)
        if error_text:
            raise RuntimeError(f"{source_name} request failed ({exc.code}): {error_text}") from exc
        raise RuntimeError(f"{source_name} request failed with HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"{source_name} request failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{source_name} request timed out") from exc


def parse_entsoe_price_xml(xml_bytes: bytes) -> Tuple[pd.DataFrame, str, str]:
    root = ET.fromstring(xml_bytes)
    namespace = root.tag.split("}")[0].strip("{")
    ns = {"ns": namespace}

    frames = []
    currencies = set()
    resolutions = set()

    for ts in root.findall(".//ns:TimeSeries", ns):
        currency = (ts.findtext(".//ns:currency_Unit.name", default="", namespaces=ns) or "").strip().upper()
        if currency:
            currencies.add(currency)

        for period in ts.findall(".//ns:Period", ns):
            start_text = period.findtext("ns:timeInterval/ns:start", namespaces=ns)
            if not start_text:
                continue

            resolution = (period.findtext("ns:resolution", namespaces=ns) or "PT60M").strip()
            resolutions.add(resolution)
            step = parse_resolution_to_timedelta(resolution)
            start = pd.to_datetime(start_text, utc=True)

            rows = []
            for point in period.findall("ns:Point", ns):
                pos_text = point.findtext("ns:position", namespaces=ns)
                value_text = point.findtext("ns:price.amount", namespaces=ns)
                if not pos_text or not value_text:
                    continue
                timestamp = start + (int(pos_text) - 1) * step
                rows.append((timestamp, float(value_text)))

            if rows:
                frames.append(pd.DataFrame(rows, columns=["time_utc", "price_eur_mwh"]))

    if not frames:
        raise RuntimeError("failed to parse day-ahead prices from ENTSO-E payload")

    if len(currencies) > 1:
        raise RuntimeError(f"mixed currencies in ENTSO-E payload: {sorted(currencies)}")

    resolution = sorted(resolutions)[0] if len(resolutions) == 1 else "mixed"

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
    out.set_index("time_utc", inplace=True)
    return out, (next(iter(currencies)) if currencies else ""), resolution


def normalize_currency(df: pd.DataFrame, currency: str, zone_name: str, gbp_eur: float | None) -> pd.DataFrame:
    if not currency or currency == "EUR":
        return df

    if currency == "GBP":
        if gbp_eur is None:
            raise RuntimeError(
                f"{zone_name} prices are reported in GBP. Pass --gbp-eur or set GBP_EUR / GBP_EUR_RATE."
            )
        converted = df.copy()
        converted["price_eur_mwh"] = converted["price_eur_mwh"] * gbp_eur
        return converted

    raise RuntimeError(f"unsupported currency for {zone_name}: {currency}")


def normalize_resolution(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    if resolution in {"PT60M", "PT1H"}:
        return df

    if resolution == "mixed":
        return df.resample("1h").mean()

    step = parse_resolution_to_timedelta(resolution)
    if step > pd.Timedelta(hours=1):
        raise RuntimeError(f"cannot normalize resolution larger than 1 hour: {resolution}")

    return df.resample("1h").mean()


def fetch_entsoe_da_price(zone: ZoneSpec, market_day: dt.date, token: str) -> pd.DataFrame:
    start_utc, end_utc = entsoe_utc_window_for_local_day(market_day, zone.timezone)
    period_start, period_end = iso_interval(start_utc, end_utc)

    params = {
        "securityToken": token,
        "documentType": "A44",
        "in_Domain": zone.eic,
        "out_Domain": zone.eic,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    url = ENTSOE_ENDPOINT + "?" + urllib.parse.urlencode(params)
    xml_data = fetch_entsoe_payload(url, zone.name)

    if b"Publication_MarketDocument" not in xml_data:
        error_text = parse_entsoe_error(xml_data)
        if error_text:
            raise RuntimeError(f"{zone.name} request failed: {error_text}")
        raise RuntimeError(f"{zone.name} returned a non-price payload")

    prices, currency, resolution = parse_entsoe_price_xml(xml_data)
    prices = normalize_currency(prices, currency, zone.name, None)
    prices = normalize_resolution(prices, resolution)
    return prices


def fetch_prices_entsoe(
    zones: Dict[str, ZoneSpec],
    market_days: List[dt.date],
    token: str,
) -> pd.DataFrame:
    frames = {}

    for name, zone in zones.items():
        zone_frames = []
        for market_day in market_days:
            zone_frames.append(fetch_entsoe_da_price(zone, market_day, token))

        zone_df = pd.concat(zone_frames).sort_index()
        zone_df = zone_df[~zone_df.index.duplicated(keep="last")]
        frames[name] = zone_df.rename(columns={"price_eur_mwh": name})

    joined = None
    for frame in frames.values():
        joined = frame if joined is None else joined.join(frame, how="outer")

    return joined


def fetch_gb_market_index(
    market_days: List[dt.date],
    gbp_eur: float | None,
    provider: str,
    bmrs_api_key: str | None,
) -> Tuple[pd.DataFrame, str]:
    if gbp_eur is None:
        raise RuntimeError("GB prices are reported in GBP. Pass --gbp-eur or set GBP_EUR / GBP_EUR_RATE.")

    start_utc, _ = entsoe_utc_window_for_local_day(market_days[0], "Europe/London")
    _, end_utc = entsoe_utc_window_for_local_day(market_days[-1], "Europe/London")

    params = {
        "from": rfc3339_utc(start_utc),
        "to": rfc3339_utc(end_utc),
        "dataProviders": provider,
        "format": "json",
    }
    url = ELEXON_ENDPOINT + "/balancing/pricing/market-index?" + urllib.parse.urlencode(params)
    payload = fetch_elexon_payload(url, "GB Elexon MID", bmrs_api_key)

    try:
        body = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("GB Elexon MID returned invalid JSON") from exc

    rows = body.get("data")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"GB Elexon MID returned no data for provider {provider}")

    df = pd.DataFrame(rows)
    required = {"startTime", "dataProvider", "price", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"GB Elexon MID payload missing columns: {', '.join(missing)}")

    df = df[df["dataProvider"] == provider].copy()
    if df.empty:
        raise RuntimeError(f"GB Elexon MID returned no rows for provider {provider}")

    if (pd.to_numeric(df["volume"], errors="coerce").fillna(0) <= 0).all():
        raise RuntimeError(f"GB Elexon MID provider {provider} returned no positive-volume rows")

    df["time_utc"] = pd.to_datetime(df["startTime"], utc=True)
    df["price_gbp_mwh"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume_mwh"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["time_utc", "price_gbp_mwh"])
    df = df[(df["time_utc"] >= pd.Timestamp(start_utc)) & (df["time_utc"] < pd.Timestamp(end_utc))]
    if df.empty:
        raise RuntimeError(f"GB Elexon MID returned no rows inside the requested UTC window for provider {provider}")

    df = df[["time_utc", "price_gbp_mwh"]].drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
    df["price_eur_mwh"] = df["price_gbp_mwh"] * gbp_eur
    df = df.set_index("time_utc")[["price_eur_mwh"]]
    df = df.resample("1h").mean()
    return df, provider


def fetch_prices(
    market_days: List[dt.date],
    entsoe_token: str,
    gbp_eur: float | None,
    bmrs_api_key: str | None,
    gb_provider: str,
) -> Tuple[pd.DataFrame, str]:
    gb_prices, provider_used = fetch_gb_market_index(market_days, gbp_eur, gb_provider, bmrs_api_key)
    gb_prices = gb_prices.rename(columns={"price_eur_mwh": "GB"})
    continental = fetch_prices_entsoe(CONTINENTAL_ZONES, market_days, entsoe_token)
    return gb_prices.join(continental, how="outer"), provider_used


def resolve_live_price_fetch_config(args: argparse.Namespace) -> tuple[str, float, str, str | None]:
    entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
    if not entsoe_token:
        raise RuntimeError("ENTOS_E_TOKEN or ENTSOE_TOKEN is required for the free upstream market-state feed")

    gbp_eur = args.gbp_eur
    if gbp_eur is None:
        for env_name in FX_ENV_NAMES:
            if os.environ.get(env_name):
                gbp_eur = parse_gbp_eur(os.environ[env_name])
                break
    if gbp_eur is None:
        raise RuntimeError("GBP_EUR or GBP_EUR_RATE is required for the free upstream market-state feed")

    bmrs_api_key = next((os.environ.get(name) for name in BMRS_KEY_ENV_NAMES if os.environ.get(name)), None)
    gb_provider = args.gb_provider or next(
        (os.environ.get(name) for name in GB_PROVIDER_ENV_NAMES if os.environ.get(name)),
        DEFAULT_GB_PROVIDER,
    )
    return entsoe_token, gbp_eur, gb_provider, bmrs_api_key


def synthetic_prices(market_days: List[dt.date]) -> pd.DataFrame:
    start_day = market_days[0]
    periods = 24 * len(market_days)
    idx = pd.date_range(pd.Timestamp(start_day, tz="UTC"), periods=periods, freq="1h")

    rng = np.random.default_rng(42)
    gb = -5 + 10 * np.sin(np.linspace(0, 3 * np.pi, periods)) + rng.normal(0, 4, periods)
    fr = 20 + 12 * np.sin(np.linspace(0.2, 3.2 * np.pi, periods)) + rng.normal(0, 4, periods)
    nl = 22 + 12 * np.sin(np.linspace(0.3, 3.1 * np.pi, periods)) + rng.normal(0, 4, periods)
    de = 24 + 11 * np.sin(np.linspace(0.4, 3.0 * np.pi, periods)) + rng.normal(0, 4, periods)
    pl = 55 + 18 * np.sin(np.linspace(0.5, 2.6 * np.pi, periods)) + rng.normal(0, 6, periods)
    cz = 52 + 16 * np.sin(np.linspace(0.6, 2.7 * np.pi, periods)) + rng.normal(0, 5, periods)

    return pd.DataFrame({"GB": gb, "FR": fr, "NL": nl, "DE": de, "PL": pl, "CZ": cz}, index=idx).round(2)


def parse_gbp_eur(value: str | None) -> float | None:
    if value is None:
        return None
    rate = float(value)
    if rate <= 0:
        raise ValueError("GBP/EUR rate must be positive")
    return rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Single local market day (YYYY-MM-DD)")
    parser.add_argument("--start", help="First local market day (YYYY-MM-DD)")
    parser.add_argument("--end", help="Last local market day, exclusive (YYYY-MM-DD)")
    parser.add_argument("--save", help="Path to save CSV", default="inline_arbitrage_live_output.csv")
    parser.add_argument("--dry", action="store_true", help="Use synthetic data explicitly")
    parser.add_argument("--gbp-eur", type=float, help="GBP->EUR conversion rate for GB prices")
    parser.add_argument("--gb-provider", default=None, help="GB Elexon MID data provider (default: APXMIDP)")
    parser.add_argument(
        "--materialize-curtailment-history",
        action="store_true",
        help="Fetch and save the first three historical curtailment tables, then exit",
    )
    parser.add_argument("--history-year", help="Constraint breakdown scheme year label, for example 2024-2025")
    parser.add_argument("--history-start", help="Historical materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--history-end", help="Historical materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--history-output-dir",
        default="curtailment_history",
        help="Output directory for materialized historical tables",
    )
    parser.add_argument(
        "--materialize-bmu-generation",
        action="store_true",
        help="Fetch and save BMU standing data plus first-pass B1610 generation history, then exit",
    )
    parser.add_argument("--bmu-start", help="BMU generation materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--bmu-end", help="BMU generation materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--bmu-output-dir",
        default="bmu_history",
        help="Output directory for BMU standing data and generation history",
    )
    parser.add_argument(
        "--materialize-bmu-dispatch",
        action="store_true",
        help="Fetch and save BOALF acceptance history, BOD bid-offer evidence, and half-hour BMU dispatch facts, then exit",
    )
    parser.add_argument("--dispatch-start", help="BMU dispatch materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--dispatch-end", help="BMU dispatch materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--dispatch-output-dir",
        default="bmu_dispatch_history",
        help="Output directory for BMU dispatch acceptance history",
    )
    parser.add_argument(
        "--materialize-bmu-curtailment-truth",
        action="store_true",
        help="Fetch and save BMU physical, availability, and curtailment-truth history, then exit",
    )
    parser.add_argument("--truth-start", help="BMU curtailment-truth materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--truth-end", help="BMU curtailment-truth materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--truth-output-dir",
        default="bmu_truth_history",
        help="Output directory for BMU physical, availability, and curtailment-truth history",
    )
    parser.add_argument(
        "--truth-profile",
        default="all",
        choices=("all", "precision", "research"),
        help="Export profile for the truth table: all, precision, or research",
    )
    parser.add_argument(
        "--truth-store-db-path",
        help="SQLite path for deduped BMU truth tables and QA outputs",
    )
    parser.add_argument(
        "--fill-truth-store-from-dir",
        help="Recursively ingest BMU truth CSV outputs from a directory tree into the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--materialize-truth-store-source-focus",
        action="store_true",
        help="Build store-backed source-completeness focus tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--show-truth-store-source-focus",
        action="store_true",
        help="Print the store-backed source-completeness focus summary from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--source-focus-status",
        default="fail_warn",
        choices=("all", "fail", "fail_warn"),
        help="Status filter for truth-store source focus views: all, fail, or fail_warn",
    )
    parser.add_argument(
        "--source-focus-limit",
        type=int,
        default=15,
        help="Maximum number of source-focus family rows to print",
    )
    parser.add_argument(
        "--materialize-truth-store-family-forensics",
        action="store_true",
        help="Build store-backed family dispatch forensics tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--show-truth-store-family-forensics",
        action="store_true",
        help="Print the store-backed family dispatch forensics summary from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--forensic-family-keys",
        default="HOWAO,HOWBO",
        help="Comma-separated BMU family keys for store-backed forensics (default: HOWAO,HOWBO for Hornsea)",
    )
    parser.add_argument(
        "--forensic-start",
        help="Optional forensic start date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--forensic-end",
        help="Optional forensic end date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--forensic-limit",
        type=int,
        default=20,
        help="Maximum number of forensic BMU or half-hour rows to print",
    )
    parser.add_argument(
        "--forensic-output-dir",
        help="Optional directory for scoped family publication-audit and support-extract CSVs",
    )
    parser.add_argument(
        "--materialize-truth-store-support-loop",
        action="store_true",
        help="Build store-backed publication-anomaly support-case tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--show-truth-store-support-loop",
        action="store_true",
        help="Print the store-backed support-case batch summary from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--support-output-dir",
        help="Optional directory for ranked support-case CSVs and a Markdown support dossier",
    )
    parser.add_argument(
        "--support-status",
        default="fail_warn",
        choices=("all", "fail", "fail_warn"),
        help="Status filter for support-loop selection: all, fail, or fail_warn",
    )
    parser.add_argument(
        "--support-top-days",
        type=int,
        default=7,
        help="Maximum number of support-ready days to include in the support batch",
    )
    parser.add_argument(
        "--support-top-families-per-day",
        type=int,
        default=5,
        help="Maximum number of support-ready family-days to include per selected day",
    )
    parser.add_argument(
        "--support-half-hour-limit",
        type=int,
        default=20,
        help="Maximum number of support half-hour evidence rows to print",
    )
    parser.add_argument(
        "--materialize-truth-store-support-resolution",
        action="store_true",
        help="Build or refresh the support-case resolution ledger from store-backed support batches, then exit",
    )
    parser.add_argument(
        "--show-truth-store-support-resolution",
        action="store_true",
        help="Print the support-case resolution ledger from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--annotate-truth-store-support-resolution",
        action="store_true",
        help="Upsert one support-case resolution annotation into the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--resolution-batch-id",
        help="Support batch id for support-resolution materialization, annotation, or review",
    )
    parser.add_argument(
        "--resolution-date",
        help="Support-resolution settlement date for annotation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--resolution-family-key",
        help="BMU family key for support-resolution annotation",
    )
    parser.add_argument(
        "--resolution-state",
        choices=VALID_RESOLUTION_STATES,
        help="Resolution state for support-case annotation",
    )
    parser.add_argument(
        "--resolution-truth-policy-action",
        choices=VALID_TRUTH_POLICY_ACTIONS,
        help="Truth-policy action for support-case annotation",
    )
    parser.add_argument(
        "--resolution-note",
        help="Optional analyst note for support-case annotation",
    )
    parser.add_argument(
        "--resolution-source-reference",
        help="Optional source or ticket reference for support-case annotation",
    )
    parser.add_argument(
        "--resolution-filter",
        default="all",
        choices=VALID_RESOLUTION_FILTERS,
        help="Filter for support-case resolution review: all, open, resolved, or a specific resolution state",
    )
    parser.add_argument(
        "--resolution-limit",
        type=int,
        default=20,
        help="Maximum number of support-case resolution rows to print",
    )
    parser.add_argument(
        "--materialize-truth-store-support-gate",
        action="store_true",
        help="Build or refresh the support rerun-gate and open-case priority tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--show-truth-store-support-gate",
        action="store_true",
        help="Print the support rerun-gate and open-case priority tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--support-gate-filter",
        default="all",
        choices=VALID_SUPPORT_GATE_FILTERS,
        help="Filter for support rerun-gate review: all, blocked, ready_for_rerun, or no_rerun_required",
    )
    parser.add_argument(
        "--support-open-case-limit",
        type=int,
        default=20,
        help="Maximum number of open-case priority rows to print",
    )
    parser.add_argument(
        "--materialize-truth-store-rerun-candidates",
        action="store_true",
        help="Build or refresh the support rerun-candidate tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--show-truth-store-rerun-candidates",
        action="store_true",
        help="Print the support rerun-candidate tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--support-rerun-candidate-filter",
        default="all",
        choices=VALID_SUPPORT_RERUN_CANDIDATE_FILTERS,
        help="Filter for support rerun candidates: all, fix_source_and_rerun, or eligible_for_new_evidence_tier",
    )
    parser.add_argument(
        "--support-rerun-candidate-limit",
        type=int,
        default=20,
        help="Maximum number of rerun-candidate family rows to print",
    )
    parser.add_argument(
        "--materialize-truth-store-support-resolution-patterns",
        action="store_true",
        help="Build or refresh repeated open-case resolution-pattern tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--show-truth-store-support-resolution-patterns",
        action="store_true",
        help="Print repeated open-case resolution-pattern tables from the SQLite truth store, then exit",
    )
    parser.add_argument(
        "--apply-truth-store-support-resolution-pattern",
        action="store_true",
        help="Apply one resolution annotation to all currently open family-days in a selected support-resolution pattern, then exit",
    )
    parser.add_argument(
        "--resolution-pattern-key",
        help="Support-resolution pattern key for review or bulk annotation",
    )
    parser.add_argument(
        "--support-pattern-filter",
        default="all",
        choices=VALID_SUPPORT_PATTERN_FILTERS,
        help="Filter for support-resolution patterns: all, open, single_day, or multi_day",
    )
    parser.add_argument(
        "--support-pattern-limit",
        type=int,
        default=20,
        help="Maximum number of support-resolution pattern rows to print",
    )
    parser.add_argument(
        "--materialize-weather-history",
        action="store_true",
        help="Fetch and save anchor, cluster, and parent-region weather history, then exit",
    )
    parser.add_argument(
        "--materialize-interconnector-flow",
        action="store_true",
        help="Fetch and save border-level ENTSO-E physical interconnector flow history, then exit",
    )
    parser.add_argument(
        "--materialize-interconnector-itl",
        action="store_true",
        help="Fetch and save NESO interconnector ITL history for connector-specific export and import limits, then exit",
    )
    parser.add_argument(
        "--materialize-interconnector-capacity",
        action="store_true",
        help="Fetch and save border-level ENTSO-E offered interconnector capacity history, then exit",
    )
    parser.add_argument(
        "--materialize-day-ahead-constraint-boundary",
        action="store_true",
        help="Fetch and save NESO day-ahead constraint boundary flow and limit history, then exit",
    )
    parser.add_argument(
        "--materialize-interconnector-capacity-audit",
        action="store_true",
        help="Fetch and save a broader ENTSO-E interconnector-capacity source audit across border and query variants, then exit",
    )
    parser.add_argument(
        "--materialize-interconnector-capacity-review-policy",
        action="store_true",
        help="Fetch and save the interconnector-capacity review policy that keeps alternate explicit-daily evidence separate from the first-pass gate, then exit",
    )
    parser.add_argument(
        "--materialize-gb-transfer-gate",
        action="store_true",
        help="Fetch network overlays and save the hourly GB cluster-to-hub transfer-gate proxy, then exit",
    )
    parser.add_argument(
        "--materialize-france-connector-layer",
        action="store_true",
        help="Fetch network overlays and save the France-specific cable layer for IFA, IFA2, and ElecLink, then exit",
    )
    parser.add_argument(
        "--materialize-route-score-history",
        action="store_true",
        help="Fetch prices and network overlays, then save the first-pass cluster-to-hub-to-route score history, then exit",
    )
    parser.add_argument(
        "--materialize-curtailment-opportunity-history",
        action="store_true",
        help="Fetch route inputs plus cluster curtailment and save the hourly curtailment-opportunity surface, then exit",
    )
    parser.add_argument(
        "--materialize-upstream-market-state-feed",
        action="store_true",
        help="Normalize and save an upstream market-state feed for route-level forward, day-ahead, and intraday features, then exit",
    )
    parser.add_argument(
        "--materialize-opportunity-backtest",
        action="store_true",
        help="Build and save the first-pass walk-forward opportunity backtest from an existing opportunity history, then exit",
    )
    parser.add_argument("--flow-start", help="Interconnector flow materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--flow-end", help="Interconnector flow materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--flow-output-dir",
        default="interconnector_flow_history",
        help="Output directory for interconnector flow history",
    )
    parser.add_argument("--itl-start", help="Interconnector ITL materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--itl-end", help="Interconnector ITL materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--itl-output-dir",
        default="interconnector_itl_history",
        help="Output directory for interconnector ITL history",
    )
    parser.add_argument(
        "--capacity-start",
        help="Interconnector capacity materialization start date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capacity-end",
        help="Interconnector capacity materialization end date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capacity-output-dir",
        default="interconnector_capacity_history",
        help="Output directory for interconnector capacity history",
    )
    parser.add_argument(
        "--boundary-start",
        help="Day-ahead constraint boundary materialization start date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--boundary-end",
        help="Day-ahead constraint boundary materialization end date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--boundary-output-dir",
        default="day_ahead_constraint_boundary_history",
        help="Output directory for day-ahead constraint boundary history",
    )
    parser.add_argument(
        "--market-state-start",
        help="Upstream market-state materialization start date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--market-state-end",
        help="Upstream market-state materialization end date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--market-state-output-dir",
        default="upstream_market_state_history",
        help="Output directory for upstream market-state history",
    )
    parser.add_argument(
        "--normalize-upstream-market-state-input",
        action="store_true",
        help="Normalize a messy upstream market-state CSV, TSV, TXT, or JSON into the canonical hourly feed schema, then exit",
    )
    parser.add_argument(
        "--upstream-market-state-raw-path",
        help="Raw upstream market-state file to normalize",
    )
    parser.add_argument(
        "--upstream-market-state-normalized-output",
        default="upstream_market_state_input.normalized.csv",
        help="Output CSV path for the normalized upstream market-state input",
    )
    parser.add_argument(
        "--market-state-input-path",
        help="Optional local CSV or JSON input for an upstream market-state feed with forward, day-ahead, or intraday price-state fields",
    )
    parser.add_argument(
        "--opportunity-start",
        help="Curtailment-opportunity materialization start date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--opportunity-end",
        help="Curtailment-opportunity materialization end date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--opportunity-output-dir",
        default="curtailment_opportunity_history",
        help="Output directory for hourly curtailment-opportunity history",
    )
    parser.add_argument(
        "--opportunity-input-path",
        default="curtailment_opportunity_history",
        help="Input directory or CSV path for fact_curtailment_opportunity_hourly",
    )
    parser.add_argument(
        "--opportunity-truth-profile",
        default="proxy",
        choices=sorted(VALID_OPPORTUNITY_TRUTH_PROFILES),
        help="Curtailment source tier for the opportunity layer: proxy, research, precision, or all",
    )
    parser.add_argument(
        "--backtest-output-dir",
        default="opportunity_backtest_history",
        help="Output directory for fact_backtest_prediction_hourly",
    )
    parser.add_argument(
        "--backtest-model-key",
        default=DEFAULT_BACKTEST_MODEL_SELECTION,
        choices=sorted(VALID_BACKTEST_MODEL_SELECTIONS),
        help="Backtest model selection: all, opportunity_group_mean_notice_v1, or opportunity_potential_ratio_v2",
    )
    parser.add_argument(
        "--backtest-horizons",
        default=",".join(str(value) for value in DEFAULT_FORECAST_HORIZON_HOURS),
        help="Comma-separated forecast horizons in hours for the opportunity backtest, e.g. 1,6,24,168",
    )
    parser.add_argument(
        "--capacity-audit-start",
        help="Interconnector capacity source-audit start date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capacity-audit-end",
        help="Interconnector capacity source-audit end date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capacity-audit-output-dir",
        default="interconnector_capacity_audit",
        help="Output directory for interconnector capacity source audit",
    )
    parser.add_argument(
        "--capacity-review-start",
        help="Interconnector capacity review-policy start date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capacity-review-end",
        help="Interconnector capacity review-policy end date, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capacity-review-output-dir",
        default="interconnector_capacity_review",
        help="Output directory for interconnector capacity review policy",
    )
    parser.add_argument("--transfer-start", help="GB transfer-gate materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--transfer-end", help="GB transfer-gate materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--transfer-output-dir",
        default="gb_transfer_gate_history",
        help="Output directory for GB transfer-gate history",
    )
    parser.add_argument(
        "--normalize-gb-transfer-reviewed-input",
        action="store_true",
        help="Normalize a messy GB internal-transfer reviewed-input CSV, TSV, TXT, or JSON into the canonical reviewed-period CSV schema, then exit",
    )
    parser.add_argument(
        "--gb-transfer-reviewed-raw-path",
        help="Raw GB internal-transfer reviewed-input file to normalize",
    )
    parser.add_argument(
        "--gb-transfer-reviewed-normalized-output",
        default="gb_transfer_reviewed_input.normalized.csv",
        help="Output CSV path for the normalized GB internal-transfer reviewed input",
    )
    parser.add_argument(
        "--gb-transfer-reviewed-input-path",
        help="Optional local CSV or JSON input normalized from public internal boundary/constraint evidence to materialize the reviewed GB transfer tier",
    )
    parser.add_argument("--france-start", help="France connector-layer materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--france-end", help="France connector-layer materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--france-output-dir",
        default="france_connector_history",
        help="Output directory for the France connector layer",
    )
    parser.add_argument(
        "--normalize-france-reviewed-input",
        action="store_true",
        help="Normalize a messy France reviewed-input CSV, TSV, TXT, or JSON into the canonical reviewed-period CSV schema, then exit",
    )
    parser.add_argument(
        "--france-reviewed-raw-path",
        help="Raw France reviewed-input file to normalize",
    )
    parser.add_argument(
        "--france-reviewed-normalized-output",
        default="france_connector_reviewed_input.normalized.csv",
        help="Output CSV path for the normalized France reviewed input",
    )
    parser.add_argument(
        "--eleclink-umm-export-path",
        help="Optional local CSV or JSON export of ElecLink Nord Pool UMM outages to tighten France connector availability",
    )
    parser.add_argument(
        "--france-reviewed-input-path",
        help="Optional local CSV or JSON input normalized from ElecLink public documents and JAO notices to materialize fact_france_connector_reviewed_period",
    )
    parser.add_argument("--eleclink-umm-username", help="Optional Nord Pool UMM username for authenticated ElecLink access")
    parser.add_argument("--eleclink-umm-password", help="Optional Nord Pool UMM password for authenticated ElecLink access")
    parser.add_argument(
        "--eleclink-umm-client-authorization",
        help="Optional Nord Pool client authorization string for token retrieval; if no Basic prefix is present it will be added",
    )
    parser.add_argument("--eleclink-umm-scope", help="Optional Nord Pool UMM token scope")
    parser.add_argument("--eleclink-umm-access-token", help="Optional pre-fetched Nord Pool UMM bearer token")
    parser.add_argument(
        "--eleclink-umm-token-url",
        default=NORDPOOL_UMM_TOKEN_URL,
        help="Nord Pool STS token endpoint for authenticated ElecLink UMM access",
    )
    parser.add_argument(
        "--eleclink-umm-messages-url",
        default=NORDPOOL_UMM_MESSAGES_URL,
        help="Nord Pool UMM messages endpoint for authenticated ElecLink UMM access",
    )
    parser.add_argument("--route-score-start", help="Route-score materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--route-score-end", help="Route-score materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--route-score-output-dir",
        default="route_score_history",
        help="Output directory for route-score history",
    )
    parser.add_argument("--weather-start", help="Weather materialization start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--weather-end", help="Weather materialization end date, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--weather-output-dir",
        default="weather_history",
        help="Output directory for weather history",
    )
    parser.add_argument(
        "--show-constraint-assumptions",
        action="store_true",
        help="Print the physical-network assumptions register and exit",
    )
    parser.add_argument(
        "--show-asset-mapping",
        action="store_true",
        help="Print the first-pass GB asset and signal registry and exit",
    )
    parser.add_argument(
        "--show-gb-topology",
        action="store_true",
        help="Print the first-pass GB cluster-to-hub reachability scaffold and exit",
    )
    parser.add_argument(
        "--show-exploration-plan",
        action="store_true",
        help="Print the historical-data, map, backtest, and drift plan and exit",
    )
    args = parser.parse_args()

    if (
        args.show_constraint_assumptions
        or args.show_asset_mapping
        or args.show_gb_topology
        or args.show_exploration_plan
    ):
        if args.show_constraint_assumptions:
            print("Constraint assumptions")
            print(assumption_frame().to_string(index=False))

        if args.show_asset_mapping:
            if args.show_constraint_assumptions:
                print()
            print("Parent regions")
            print(parent_region_frame().to_string(index=False))
            print()
            print("Asset clusters")
            print(cluster_frame().to_string(index=False))
            print()
            print("Weather anchors")
            print(weather_anchor_frame().to_string(index=False))
            print()
            print("Signal sources")
            print(signal_source_frame().to_string(index=False))

        if args.show_gb_topology:
            if args.show_constraint_assumptions or args.show_asset_mapping:
                print()
            print("Interconnector hubs")
            print(interconnector_hub_frame().to_string(index=False))
            print()
            print("Route hub options")
            print(route_hub_frame().to_string(index=False))
            print()
            print("Cluster reachability")
            print(reachability_frame().to_string(index=False))
            print()
            print("Cluster-hub matrix")
            print(cluster_hub_matrix().to_string(index=False))

        if args.show_exploration_plan:
            if args.show_constraint_assumptions or args.show_asset_mapping or args.show_gb_topology:
                print()
            print("Dataset plan")
            print(dataset_plan_frame().to_string(index=False))
            print()
            print("Map layer plan")
            print(map_layer_plan_frame().to_string(index=False))
            print()
            print("Backtest plan")
            print(backtest_plan_frame().to_string(index=False))
            print()
            print("Drift monitor plan")
            print(drift_monitor_plan_frame().to_string(index=False))
        return 0

    if args.fill_truth_store_from_dir:
        if not args.truth_store_db_path:
            raise SystemExit("--fill-truth-store-from-dir requires --truth-store-db-path")
        try:
            summary = ingest_truth_csv_tree_to_sqlite(
                root_dir=args.fill_truth_store_from_dir,
                db_path=args.truth_store_db_path,
            )
            print(
                f"[store=sqlite] Ingested truth CSV tree from {args.fill_truth_store_from_dir} into {args.truth_store_db_path}"
            )
            if summary.empty:
                print("No recognized truth CSV tables were found.")
            else:
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: files={int(row['files_loaded'])} "
                        f"rows_loaded={int(row['rows_loaded'])} table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_truth_store_source_focus or args.show_truth_store_source_focus:
        if not args.truth_store_db_path:
            raise SystemExit(
                "--materialize-truth-store-source-focus and --show-truth-store-source-focus require --truth-store-db-path"
            )
        try:
            frames = materialize_truth_store_source_focus(args.truth_store_db_path)
            if args.materialize_truth_store_source_focus:
                print(f"[store=sqlite] Materialized source-focus tables in {args.truth_store_db_path}")
                for table_name, frame in frames.items():
                    print(f"{table_name}: rows={len(frame)}")
            if args.show_truth_store_source_focus:
                filtered = read_truth_store_source_focus(
                    db_path=args.truth_store_db_path,
                    status_mode=args.source_focus_status,
                )
                print("Source Focus Daily")
                daily = filtered["fact_source_completeness_focus_daily"]
                if daily.empty:
                    print("No matching source-focus days.")
                else:
                    print(daily.to_string(index=False))
                print()
                print("Source Focus Families")
                family = filtered["fact_source_completeness_focus_family_daily"].head(args.source_focus_limit)
                if family.empty:
                    print("No matching source-focus families.")
                else:
                    print(family.to_string(index=False))
                print()
                print("Dispatch Source Gaps Daily")
                gap_daily = filtered["fact_dispatch_source_gap_daily"]
                if gap_daily.empty:
                    print("No matching dispatch source-gap days.")
                else:
                    print(gap_daily.to_string(index=False))
                print()
                print("Dispatch Source Gaps Families")
                gap_family = filtered["fact_dispatch_source_gap_family_daily"].head(args.source_focus_limit)
                if gap_family.empty:
                    print("No matching dispatch source-gap families.")
                else:
                    print(gap_family.to_string(index=False))
                print()
                print("Publication Anomalies Daily")
                anomaly_daily = filtered["fact_publication_anomaly_daily"]
                if anomaly_daily.empty:
                    print("No matching publication-anomaly days.")
                else:
                    print(anomaly_daily.to_string(index=False))
                print()
                print("Publication Anomalies Families")
                anomaly_family = filtered["fact_publication_anomaly_family_daily"].head(args.source_focus_limit)
                if anomaly_family.empty:
                    print("No matching publication-anomaly families.")
                else:
                    print(anomaly_family.to_string(index=False))
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_truth_store_family_forensics or args.show_truth_store_family_forensics or args.forensic_output_dir:
        if not args.truth_store_db_path:
            raise SystemExit(
                "--materialize-truth-store-family-forensics, --show-truth-store-family-forensics, and --forensic-output-dir require --truth-store-db-path"
            )
        forensic_start = parse_bmu_iso_date(args.forensic_start).isoformat() if args.forensic_start else None
        forensic_end = parse_bmu_iso_date(args.forensic_end).isoformat() if args.forensic_end else None
        try:
            frames = materialize_truth_store_family_forensics(
                db_path=args.truth_store_db_path,
                family_keys=args.forensic_family_keys,
                start_date=forensic_start,
                end_date=forensic_end,
            )
            scope_key = forensic_scope_key_for_family_keys(args.forensic_family_keys)
            if args.materialize_truth_store_family_forensics:
                print(
                    f"[store=sqlite] Materialized family forensics tables in {args.truth_store_db_path} "
                    f"for scope={scope_key}"
                )
                for table_name, frame in frames.items():
                    print(f"{table_name}: rows={len(frame)}")
            filtered = None
            if args.show_truth_store_family_forensics or args.forensic_output_dir:
                filtered = read_truth_store_family_forensics(
                    db_path=args.truth_store_db_path,
                    family_keys=args.forensic_family_keys,
                    start_date=forensic_start,
                    end_date=forensic_end,
                )
            if args.show_truth_store_family_forensics:
                assert filtered is not None
                print(f"Family Dispatch Forensics Daily ({scope_key})")
                daily = filtered["fact_family_dispatch_forensic_daily"]
                if daily.empty:
                    print("No matching forensic daily rows.")
                else:
                    print(daily.to_string(index=False))
                print()
                print(f"Family Dispatch Forensics BMU Daily ({scope_key})")
                bmu_daily = filtered["fact_family_dispatch_forensic_bmu_daily"].head(args.forensic_limit)
                if bmu_daily.empty:
                    print("No matching forensic BMU rows.")
                else:
                    print(bmu_daily.to_string(index=False))
                print()
                print(f"Family Dispatch Forensics Half-Hourly ({scope_key})")
                half_hourly = filtered["fact_family_dispatch_forensic_half_hourly"].head(args.forensic_limit)
                if half_hourly.empty:
                    print("No matching forensic half-hour rows.")
                else:
                    print(half_hourly.to_string(index=False))
                print()
                print(f"Family Physical Forensics Daily ({scope_key})")
                physical_daily = filtered["fact_family_physical_forensic_daily"]
                if physical_daily.empty:
                    print("No matching physical forensic daily rows.")
                else:
                    print(physical_daily.to_string(index=False))
                print()
                print(f"Family Physical Forensics BMU Daily ({scope_key})")
                physical_bmu_daily = filtered["fact_family_physical_forensic_bmu_daily"].head(args.forensic_limit)
                if physical_bmu_daily.empty:
                    print("No matching physical forensic BMU rows.")
                else:
                    print(physical_bmu_daily.to_string(index=False))
                print()
                print(f"Family Physical Forensics Half-Hourly ({scope_key})")
                physical_half_hourly = filtered["fact_family_physical_forensic_half_hourly"].head(args.forensic_limit)
                if physical_half_hourly.empty:
                    print("No matching physical forensic half-hour rows.")
                else:
                    print(physical_half_hourly.to_string(index=False))
                print()
                print(f"Family Publication Audit Daily ({scope_key})")
                publication_daily = filtered["fact_family_publication_audit_daily"]
                if publication_daily.empty:
                    print("No matching publication-audit daily rows.")
                else:
                    print(publication_daily.to_string(index=False))
                print()
                print(f"Family Publication Audit BMU Daily ({scope_key})")
                publication_bmu_daily = filtered["fact_family_publication_audit_bmu_daily"].head(args.forensic_limit)
                if publication_bmu_daily.empty:
                    print("No matching publication-audit BMU rows.")
                else:
                    print(publication_bmu_daily.to_string(index=False))
                print()
                print(f"Family Support Evidence Half-Hourly ({scope_key})")
                support_half_hourly = filtered["fact_family_support_evidence_half_hourly"].head(args.forensic_limit)
                if support_half_hourly.empty:
                    print("No matching support-evidence half-hour rows.")
                else:
                    print(support_half_hourly.to_string(index=False))
            if args.forensic_output_dir:
                assert filtered is not None
                written = write_family_support_extract_csvs(
                    frames=filtered,
                    output_dir=args.forensic_output_dir,
                    family_keys=args.forensic_family_keys,
                )
                print()
                print(f"Support extract CSVs ({scope_key})")
                for table_name, path in written.items():
                    print(f"{table_name}: path={path}")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_curtailment_history:
        if not args.history_year or not args.history_start or not args.history_end:
            raise SystemExit(
                "--materialize-curtailment-history requires --history-year, --history-start, and --history-end"
            )

        history_start = parse_iso_date(args.history_start)
        history_end = parse_iso_date(args.history_end)
        if history_end < history_start:
            raise SystemExit("--history-end must be on or after --history-start")

        try:
            frames = materialize_curtailed_history(
                year_label=args.history_year,
                start_date=history_start,
                end_date=history_end,
                output_dir=args.history_output_dir,
            )
            print(
                f"[source=neso] Materialized {len(frames)} tables for {history_start} to {history_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.history_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_bmu_generation:
        if not args.bmu_start or not args.bmu_end:
            raise SystemExit("--materialize-bmu-generation requires --bmu-start and --bmu-end")

        bmu_start = parse_bmu_iso_date(args.bmu_start)
        bmu_end = parse_bmu_iso_date(args.bmu_end)
        if bmu_end < bmu_start:
            raise SystemExit("--bmu-end must be on or after --bmu-start")

        try:
            frames = materialize_bmu_generation_history(
                start_date=bmu_start,
                end_date=bmu_end,
                output_dir=args.bmu_output_dir,
            )
            print(f"[source=elexon] Materialized {len(frames)} tables for {bmu_start} to {bmu_end} (inclusive)")
            for table_name, frame in frames.items():
                output_path = os.path.join(args.bmu_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_bmu_dispatch:
        if not args.dispatch_start or not args.dispatch_end:
            raise SystemExit("--materialize-bmu-dispatch requires --dispatch-start and --dispatch-end")

        dispatch_start = parse_dispatch_iso_date(args.dispatch_start)
        dispatch_end = parse_dispatch_iso_date(args.dispatch_end)
        if dispatch_end < dispatch_start:
            raise SystemExit("--dispatch-end must be on or after --dispatch-start")

        try:
            frames = materialize_bmu_dispatch_history(
                start_date=dispatch_start,
                end_date=dispatch_end,
                output_dir=args.dispatch_output_dir,
            )
            print(
                f"[source=elexon] Materialized {len(frames)} tables for {dispatch_start} to {dispatch_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.dispatch_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_bmu_curtailment_truth:
        if not args.truth_start or not args.truth_end:
            raise SystemExit("--materialize-bmu-curtailment-truth requires --truth-start and --truth-end")

        truth_start = parse_bmu_iso_date(args.truth_start)
        truth_end = parse_bmu_iso_date(args.truth_end)
        if truth_end < truth_start:
            raise SystemExit("--truth-end must be on or after --truth-start")

        try:
            frames = materialize_bmu_curtailment_truth(
                start_date=truth_start,
                end_date=truth_end,
                output_dir=args.truth_output_dir,
                truth_profile=args.truth_profile,
            )
            print(
                f"[source=elexon+neso] Materialized {len(frames)} tables for {truth_start} to {truth_end} "
                f"(inclusive), profile={args.truth_profile}"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.truth_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted truth tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_truth_store_support_loop or args.show_truth_store_support_loop or args.support_output_dir:
        if not args.truth_store_db_path:
            raise SystemExit(
                "--materialize-truth-store-support-loop, --show-truth-store-support-loop, and --support-output-dir require --truth-store-db-path"
            )
        try:
            support_batch_id, frames, summary_markdown, written_paths = materialize_truth_store_support_loop(
                db_path=args.truth_store_db_path,
                status_mode=args.support_status,
                top_days=args.support_top_days,
                top_families_per_day=args.support_top_families_per_day,
                output_dir=args.support_output_dir,
            )
            if args.materialize_truth_store_support_loop:
                print(
                    f"[store=sqlite] Materialized support-loop tables in {args.truth_store_db_path} "
                    f"for batch={support_batch_id}"
                )
                for table_name in [
                    SUPPORT_CASE_DAILY_TABLE,
                    SUPPORT_CASE_FAMILY_TABLE,
                    SUPPORT_CASE_HALF_HOURLY_TABLE,
                    SUPPORT_CASE_RESOLUTION_TABLE,
                ]:
                    print(f"{table_name}: rows={len(frames[table_name])}")
            if args.show_truth_store_support_loop:
                print(f"Support Case Daily ({support_batch_id})")
                daily = frames[SUPPORT_CASE_DAILY_TABLE]
                if daily.empty:
                    print("No support-selected days.")
                else:
                    print(daily.to_string(index=False))
                print()
                print(f"Support Case Family Daily ({support_batch_id})")
                family = frames[SUPPORT_CASE_FAMILY_TABLE]
                if family.empty:
                    print("No support-selected family-days.")
                else:
                    print(family.to_string(index=False))
                print()
                print(f"Support Case Half-Hourly ({support_batch_id})")
                half_hourly = frames[SUPPORT_CASE_HALF_HOURLY_TABLE].head(args.support_half_hour_limit)
                if half_hourly.empty:
                    print("No support-selected half-hour rows.")
                else:
                    print(half_hourly.to_string(index=False))
                print()
                print("Support Case Summary Preview")
                preview_lines = summary_markdown.strip().splitlines()
                print("\n".join(preview_lines[: min(len(preview_lines), 60)]))
            if args.support_output_dir:
                for name in [
                    SUPPORT_CASE_DAILY_TABLE,
                    SUPPORT_CASE_FAMILY_TABLE,
                    SUPPORT_CASE_HALF_HOURLY_TABLE,
                    SUPPORT_SUMMARY_FILENAME,
                ]:
                    if name in written_paths:
                        print(f"{name}: {written_paths[name]}")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if (
        args.materialize_truth_store_support_resolution
        or args.show_truth_store_support_resolution
        or args.annotate_truth_store_support_resolution
        or args.materialize_truth_store_support_resolution_patterns
        or args.show_truth_store_support_resolution_patterns
        or args.apply_truth_store_support_resolution_pattern
        or args.materialize_truth_store_support_gate
        or args.show_truth_store_support_gate
        or args.materialize_truth_store_rerun_candidates
        or args.show_truth_store_rerun_candidates
    ):
        if not args.truth_store_db_path:
            raise SystemExit(
                "--materialize-truth-store-support-resolution, --show-truth-store-support-resolution, "
                "--annotate-truth-store-support-resolution, --materialize-truth-store-support-resolution-patterns, "
                "--show-truth-store-support-resolution-patterns, --apply-truth-store-support-resolution-pattern, "
                "--materialize-truth-store-support-gate, and --show-truth-store-support-gate, "
                "--materialize-truth-store-rerun-candidates, and "
                "--show-truth-store-rerun-candidates require --truth-store-db-path"
            )
        try:
            if args.annotate_truth_store_support_resolution:
                missing = [
                    flag
                    for flag, value in [
                        ("--resolution-batch-id", args.resolution_batch_id),
                        ("--resolution-date", args.resolution_date),
                        ("--resolution-family-key", args.resolution_family_key),
                        ("--resolution-state", args.resolution_state),
                        ("--resolution-truth-policy-action", args.resolution_truth_policy_action),
                    ]
                    if not value
                ]
                if missing:
                    raise SystemExit(
                        "--annotate-truth-store-support-resolution requires "
                        + ", ".join(missing)
                    )
                annotated = annotate_support_case_resolution(
                    db_path=args.truth_store_db_path,
                    support_batch_id=args.resolution_batch_id,
                    settlement_date=args.resolution_date,
                    bmu_family_key=args.resolution_family_key,
                    resolution_state=args.resolution_state,
                    truth_policy_action=args.resolution_truth_policy_action,
                    resolution_note=args.resolution_note,
                    source_reference=args.resolution_source_reference,
                )
                print("Annotated Support Case Resolution")
                print(annotated.to_string(index=False))
                return 0
            if args.apply_truth_store_support_resolution_pattern:
                missing = [
                    flag
                    for flag, value in [
                        ("--resolution-batch-id", args.resolution_batch_id),
                        ("--resolution-pattern-key", args.resolution_pattern_key),
                        ("--resolution-state", args.resolution_state),
                        ("--resolution-truth-policy-action", args.resolution_truth_policy_action),
                    ]
                    if not value
                ]
                if missing:
                    raise SystemExit(
                        "--apply-truth-store-support-resolution-pattern requires "
                        + ", ".join(missing)
                    )
                annotated = annotate_support_resolution_pattern(
                    db_path=args.truth_store_db_path,
                    support_batch_id=args.resolution_batch_id,
                    resolution_pattern_key=args.resolution_pattern_key,
                    resolution_state=args.resolution_state,
                    truth_policy_action=args.resolution_truth_policy_action,
                    resolution_note=args.resolution_note,
                    source_reference=args.resolution_source_reference,
                )
                print("Annotated Support Resolution Pattern")
                print(annotated.to_string(index=False))
                return 0

            resolution_frame = pd.DataFrame()
            review_frames = {}
            pattern_frames = {}
            gate_frames = {}
            candidate_frames = {}
            if (
                args.materialize_truth_store_support_resolution
                or args.materialize_truth_store_support_resolution_patterns
                or args.materialize_truth_store_support_gate
                or args.materialize_truth_store_rerun_candidates
            ):
                materialized = materialize_truth_store_support_resolution(
                    db_path=args.truth_store_db_path,
                    support_batch_id=args.resolution_batch_id,
                )
                resolution_frame = materialized[SUPPORT_CASE_RESOLUTION_TABLE]
                review_frames = {
                    SUPPORT_RESOLUTION_DAILY_TABLE: materialized[SUPPORT_RESOLUTION_DAILY_TABLE],
                    SUPPORT_RESOLUTION_BATCH_TABLE: materialized[SUPPORT_RESOLUTION_BATCH_TABLE],
                }
                pattern_frames = {
                    SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE: materialized[SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE],
                    SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE: materialized[SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE],
                }
                gate_frames = {
                    SUPPORT_RERUN_GATE_DAILY_TABLE: materialized[SUPPORT_RERUN_GATE_DAILY_TABLE],
                    SUPPORT_RERUN_GATE_BATCH_TABLE: materialized[SUPPORT_RERUN_GATE_BATCH_TABLE],
                    SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE: materialized[SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE],
                }
                candidate_frames = {
                    SUPPORT_RERUN_CANDIDATE_DAILY_TABLE: materialized[SUPPORT_RERUN_CANDIDATE_DAILY_TABLE],
                    SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE: materialized[SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE],
                }
                batch_text = f" batch={args.resolution_batch_id}" if args.resolution_batch_id else ""
                if args.materialize_truth_store_support_resolution:
                    print(
                        f"[store=sqlite] Materialized support-case resolution table in {args.truth_store_db_path}{batch_text}"
                    )
                    print(f"{SUPPORT_CASE_RESOLUTION_TABLE}: rows={len(resolution_frame)}")
                    print(f"{SUPPORT_RESOLUTION_DAILY_TABLE}: rows={len(review_frames[SUPPORT_RESOLUTION_DAILY_TABLE])}")
                    print(f"{SUPPORT_RESOLUTION_BATCH_TABLE}: rows={len(review_frames[SUPPORT_RESOLUTION_BATCH_TABLE])}")
                if args.materialize_truth_store_support_resolution_patterns:
                    print(
                        f"[store=sqlite] Materialized support-resolution pattern tables in {args.truth_store_db_path}{batch_text}"
                    )
                    print(
                        f"{SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE}: "
                        f"rows={len(pattern_frames[SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE])}"
                    )
                    print(
                        f"{SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE}: "
                        f"rows={len(pattern_frames[SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE])}"
                    )
                if args.materialize_truth_store_support_gate:
                    print(
                        f"[store=sqlite] Materialized support rerun-gate tables in {args.truth_store_db_path}{batch_text}"
                    )
                    print(f"{SUPPORT_RERUN_GATE_DAILY_TABLE}: rows={len(gate_frames[SUPPORT_RERUN_GATE_DAILY_TABLE])}")
                    print(f"{SUPPORT_RERUN_GATE_BATCH_TABLE}: rows={len(gate_frames[SUPPORT_RERUN_GATE_BATCH_TABLE])}")
                    print(
                        f"{SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE}: "
                        f"rows={len(gate_frames[SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE])}"
                    )
                if args.materialize_truth_store_rerun_candidates:
                    print(
                        f"[store=sqlite] Materialized support rerun-candidate tables in {args.truth_store_db_path}{batch_text}"
                    )
                    print(
                        f"{SUPPORT_RERUN_CANDIDATE_DAILY_TABLE}: "
                        f"rows={len(candidate_frames[SUPPORT_RERUN_CANDIDATE_DAILY_TABLE])}"
                    )
                    print(
                        f"{SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE}: "
                        f"rows={len(candidate_frames[SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE])}"
                    )
            if args.show_truth_store_support_resolution:
                resolution_frame = read_support_case_resolution(
                    db_path=args.truth_store_db_path,
                    support_batch_id=args.resolution_batch_id,
                    resolution_filter=args.resolution_filter,
                )
                if not review_frames:
                    review_frames = read_support_resolution_review(
                        db_path=args.truth_store_db_path,
                        support_batch_id=args.resolution_batch_id,
                    )
                print("Support Resolution Batch Review")
                batch_review = review_frames[SUPPORT_RESOLUTION_BATCH_TABLE]
                if batch_review.empty:
                    print("No matching support-resolution batch rows.")
                else:
                    print(batch_review.to_string(index=False))
                print()
                print("Support Resolution Daily Review")
                daily_review = review_frames[SUPPORT_RESOLUTION_DAILY_TABLE]
                if daily_review.empty:
                    print("No matching support-resolution daily rows.")
                else:
                    print(daily_review.head(args.resolution_limit).to_string(index=False))
                print()
                print("Support Case Resolution")
                if resolution_frame.empty:
                    print("No matching support-case resolution rows.")
                else:
                    print(resolution_frame.head(args.resolution_limit).to_string(index=False))
                if (
                    args.show_truth_store_support_resolution_patterns
                    or args.show_truth_store_support_gate
                    or args.show_truth_store_rerun_candidates
                ):
                    print()
            if args.show_truth_store_support_resolution_patterns:
                pattern_frames = read_support_resolution_pattern_review(
                    db_path=args.truth_store_db_path,
                    support_batch_id=args.resolution_batch_id,
                    pattern_filter=args.support_pattern_filter,
                    resolution_pattern_key=args.resolution_pattern_key,
                )
                print("Support Resolution Pattern Summary")
                pattern_summary = pattern_frames[SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE]
                if pattern_summary.empty:
                    print("No matching support-resolution pattern rows.")
                else:
                    print(pattern_summary.head(args.support_pattern_limit).to_string(index=False))
                print()
                print("Support Resolution Pattern Members")
                pattern_members = pattern_frames[SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE]
                if pattern_members.empty:
                    print("No matching support-resolution pattern member rows.")
                else:
                    print(pattern_members.head(args.support_pattern_limit).to_string(index=False))
                if args.show_truth_store_support_gate or args.show_truth_store_rerun_candidates:
                    print()
            if args.show_truth_store_support_gate:
                gate_frames = read_support_rerun_gate_review(
                    db_path=args.truth_store_db_path,
                    support_batch_id=args.resolution_batch_id,
                    gate_filter=args.support_gate_filter,
                )
                print("Support Rerun Gate Batch")
                batch_gate = gate_frames[SUPPORT_RERUN_GATE_BATCH_TABLE]
                if batch_gate.empty:
                    print("No matching support rerun-gate batch rows.")
                else:
                    print(batch_gate.to_string(index=False))
                print()
                print("Support Rerun Gate Daily")
                daily_gate = gate_frames[SUPPORT_RERUN_GATE_DAILY_TABLE]
                if daily_gate.empty:
                    print("No matching support rerun-gate daily rows.")
                else:
                    print(daily_gate.head(args.resolution_limit).to_string(index=False))
                print()
                print("Support Open-Case Priority")
                open_case_priority = gate_frames[SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE]
                if open_case_priority.empty:
                    print("No matching open-case priority rows.")
                else:
                    print(open_case_priority.head(args.support_open_case_limit).to_string(index=False))
                if args.show_truth_store_rerun_candidates:
                    print()
            if args.show_truth_store_rerun_candidates:
                candidate_frames = read_support_rerun_candidate_review(
                    db_path=args.truth_store_db_path,
                    support_batch_id=args.resolution_batch_id,
                    candidate_filter=args.support_rerun_candidate_filter,
                )
                print("Support Rerun Candidate Daily")
                candidate_daily = candidate_frames[SUPPORT_RERUN_CANDIDATE_DAILY_TABLE]
                if candidate_daily.empty:
                    print("No matching support rerun-candidate daily rows.")
                else:
                    print(candidate_daily.to_string(index=False))
                print()
                print("Support Rerun Candidate Family Daily")
                candidate_family = candidate_frames[SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE]
                if candidate_family.empty:
                    print("No matching support rerun-candidate family rows.")
                else:
                    print(candidate_family.head(args.support_rerun_candidate_limit).to_string(index=False))
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_weather_history:
        if not args.weather_start or not args.weather_end:
            raise SystemExit("--materialize-weather-history requires --weather-start and --weather-end")

        weather_start = parse_bmu_iso_date(args.weather_start)
        weather_end = parse_bmu_iso_date(args.weather_end)
        if weather_end < weather_start:
            raise SystemExit("--weather-end must be on or after --weather-start")

        try:
            frames = materialize_weather_history(
                start_date=weather_start,
                end_date=weather_end,
                output_dir=args.weather_output_dir,
            )
            print(
                f"[source=open_meteo] Materialized {len(frames)} tables for {weather_start} to {weather_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.weather_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.normalize_gb_transfer_reviewed_input:
        if not args.gb_transfer_reviewed_raw_path:
            raise SystemExit("--normalize-gb-transfer-reviewed-input requires --gb-transfer-reviewed-raw-path")
        try:
            normalized = write_normalized_gb_transfer_reviewed_input(
                input_path=args.gb_transfer_reviewed_raw_path,
                output_path=args.gb_transfer_reviewed_normalized_output,
            )
            print(
                f"[source=manual_reviewed_input] Normalized GB transfer reviewed input rows={len(normalized)} "
                f"path={args.gb_transfer_reviewed_normalized_output}"
            )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.normalize_upstream_market_state_input:
        if not args.upstream_market_state_raw_path:
            raise SystemExit("--normalize-upstream-market-state-input requires --upstream-market-state-raw-path")
        try:
            normalized = write_normalized_upstream_market_state_input(
                raw_path=args.upstream_market_state_raw_path,
                output_path=args.upstream_market_state_normalized_output,
            )
            print(
                f"[source=manual_market_state_input] Normalized upstream market-state input rows={len(normalized)} "
                f"path={args.upstream_market_state_normalized_output}"
            )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.normalize_france_reviewed_input:
        if not args.france_reviewed_raw_path:
            raise SystemExit("--normalize-france-reviewed-input requires --france-reviewed-raw-path")
        try:
            normalized = write_normalized_france_connector_reviewed_input(
                input_path=args.france_reviewed_raw_path,
                output_path=args.france_reviewed_normalized_output,
            )
            print(
                f"[source=manual_reviewed_input] Normalized France reviewed input rows={len(normalized)} "
                f"path={args.france_reviewed_normalized_output}"
            )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_upstream_market_state_feed:
        if not args.market_state_start or not args.market_state_end:
            raise SystemExit("--materialize-upstream-market-state-feed requires --market-state-start and --market-state-end")
        market_state_start = parse_market_day(args.market_state_start)
        market_state_end = parse_market_day(args.market_state_end)
        if market_state_end < market_state_start:
            raise SystemExit("--market-state-end must be on or after --market-state-start")
        try:
            if args.market_state_input_path:
                frames = materialize_upstream_market_state_history(
                    output_dir=args.market_state_output_dir,
                    start_date=market_state_start,
                    end_date=market_state_end,
                    input_path=args.market_state_input_path,
                )
                source_label = "manual_or_api_market_state"
            else:
                market_days = list(iter_market_days(market_state_start, market_state_end + dt.timedelta(days=1)))
                entsoe_token, gbp_eur, gb_provider, bmrs_api_key = resolve_live_price_fetch_config(args)
                prices, provider_used = fetch_prices(
                    market_days=market_days,
                    entsoe_token=entsoe_token,
                    gbp_eur=gbp_eur,
                    bmrs_api_key=bmrs_api_key,
                    gb_provider=gb_provider,
                )
                fact = build_fact_upstream_market_state_hourly_from_price_frame(
                    prices=prices,
                    gb_source_provider=provider_used,
                )
                frames = materialize_upstream_market_state_history(
                    output_dir=args.market_state_output_dir,
                    start_date=market_state_start,
                    end_date=market_state_end,
                    input_path=None,
                )
                frames[UPSTREAM_MARKET_STATE_TABLE] = fact
                fact.to_csv(
                    os.path.join(args.market_state_output_dir, f"{UPSTREAM_MARKET_STATE_TABLE}.csv"),
                    index=False,
                )
                source_label = f"free_entsoe_day_ahead_plus_elexon_mid:{provider_used}"
            print(
                f"[source={source_label}] Materialized {len(frames)} tables for "
                f"{market_state_start} to {market_state_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.market_state_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted upstream market-state tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_interconnector_flow:
        if not args.flow_start or not args.flow_end:
            raise SystemExit("--materialize-interconnector-flow requires --flow-start and --flow-end")

        flow_start = parse_flow_iso_date(args.flow_start)
        flow_end = parse_flow_iso_date(args.flow_end)
        if flow_end < flow_start:
            raise SystemExit("--flow-end must be on or after --flow-start")

        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit("--materialize-interconnector-flow requires ENTOS_E_TOKEN or ENTSOE_TOKEN")

        try:
            frames = materialize_interconnector_flow_history(
                start_date=flow_start,
                end_date=flow_end,
                output_dir=args.flow_output_dir,
                token=entsoe_token,
            )
            print(
                f"[source=entsoe] Materialized {len(frames)} tables for {flow_start} to {flow_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.flow_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted interconnector flow tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_interconnector_itl:
        if not args.itl_start or not args.itl_end:
            raise SystemExit("--materialize-interconnector-itl requires --itl-start and --itl-end")

        itl_start = parse_itl_iso_date(args.itl_start)
        itl_end = parse_itl_iso_date(args.itl_end)
        if itl_end < itl_start:
            raise SystemExit("--itl-end must be on or after --itl-start")

        try:
            frames = materialize_interconnector_itl_history(
                start_date=itl_start,
                end_date=itl_end,
                output_dir=args.itl_output_dir,
            )
            print(
                f"[source=neso] Materialized {len(frames)} tables for {itl_start} to {itl_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.itl_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted interconnector ITL tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_interconnector_capacity:
        if not args.capacity_start or not args.capacity_end:
            raise SystemExit("--materialize-interconnector-capacity requires --capacity-start and --capacity-end")

        capacity_start = parse_capacity_iso_date(args.capacity_start)
        capacity_end = parse_capacity_iso_date(args.capacity_end)
        if capacity_end < capacity_start:
            raise SystemExit("--capacity-end must be on or after --capacity-start")

        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit("--materialize-interconnector-capacity requires ENTOS_E_TOKEN or ENTSOE_TOKEN")

        try:
            frames = materialize_interconnector_capacity_history(
                start_date=capacity_start,
                end_date=capacity_end,
                output_dir=args.capacity_output_dir,
                token=entsoe_token,
            )
            print(
                f"[source=entsoe] Materialized {len(frames)} tables for {capacity_start} to {capacity_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.capacity_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted interconnector capacity tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_day_ahead_constraint_boundary:
        if not args.boundary_start or not args.boundary_end:
            raise SystemExit(
                "--materialize-day-ahead-constraint-boundary requires --boundary-start and --boundary-end"
            )

        boundary_start = parse_boundary_iso_date(args.boundary_start)
        boundary_end = parse_boundary_iso_date(args.boundary_end)
        if boundary_end < boundary_start:
            raise SystemExit("--boundary-end must be on or after --boundary-start")

        try:
            frames = materialize_day_ahead_constraint_boundary_history(
                start_date=boundary_start,
                end_date=boundary_end,
                output_dir=args.boundary_output_dir,
            )
            print(
                f"[source=neso] Materialized {len(frames)} tables for {boundary_start} to {boundary_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.boundary_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted day-ahead constraint boundary tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_interconnector_capacity_audit:
        if not args.capacity_audit_start or not args.capacity_audit_end:
            raise SystemExit(
                "--materialize-interconnector-capacity-audit requires --capacity-audit-start and --capacity-audit-end"
            )

        capacity_audit_start = parse_capacity_iso_date(args.capacity_audit_start)
        capacity_audit_end = parse_capacity_iso_date(args.capacity_audit_end)
        if capacity_audit_end < capacity_audit_start:
            raise SystemExit("--capacity-audit-end must be on or after --capacity-audit-start")

        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit("--materialize-interconnector-capacity-audit requires ENTOS_E_TOKEN or ENTSOE_TOKEN")

        try:
            frames = materialize_interconnector_capacity_source_audit(
                start_date=capacity_audit_start,
                end_date=capacity_audit_end,
                output_dir=args.capacity_audit_output_dir,
                token=entsoe_token,
            )
            print(
                f"[source=entsoe] Materialized {len(frames)} tables for {capacity_audit_start} to {capacity_audit_end} "
                f"(inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.capacity_audit_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted interconnector capacity audit tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_interconnector_capacity_review_policy:
        if not args.capacity_review_start or not args.capacity_review_end:
            raise SystemExit(
                "--materialize-interconnector-capacity-review-policy requires --capacity-review-start and --capacity-review-end"
            )

        capacity_review_start = parse_capacity_iso_date(args.capacity_review_start)
        capacity_review_end = parse_capacity_iso_date(args.capacity_review_end)
        if capacity_review_end < capacity_review_start:
            raise SystemExit("--capacity-review-end must be on or after --capacity-review-start")

        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit(
                "--materialize-interconnector-capacity-review-policy requires ENTOS_E_TOKEN or ENTSOE_TOKEN"
            )

        try:
            frames = materialize_interconnector_capacity_review_policy(
                start_date=capacity_review_start,
                end_date=capacity_review_end,
                output_dir=args.capacity_review_output_dir,
                token=entsoe_token,
            )
            print(
                f"[source=entsoe] Materialized {len(frames)} tables for {capacity_review_start} to {capacity_review_end} "
                f"(inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.capacity_review_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            policy = frames[INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE]
            if not policy.empty:
                print()
                print("Interconnector Capacity Review Policy")
                print(policy.to_string(index=False))
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted interconnector capacity review tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_gb_transfer_gate:
        if not args.transfer_start or not args.transfer_end:
            raise SystemExit("--materialize-gb-transfer-gate requires --transfer-start and --transfer-end")

        transfer_start = parse_transfer_iso_date(args.transfer_start)
        transfer_end = parse_transfer_iso_date(args.transfer_end)
        if transfer_end < transfer_start:
            raise SystemExit("--transfer-end must be on or after --transfer-start")

        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit("--materialize-gb-transfer-gate requires ENTOS_E_TOKEN or ENTSOE_TOKEN")

        try:
            frames = materialize_gb_transfer_gate_history(
                start_date=transfer_start,
                end_date=transfer_end,
                output_dir=args.transfer_output_dir,
                token=entsoe_token,
            )
            reviewed_frames = materialize_gb_transfer_reviewed_history(
                start_date=transfer_start,
                end_date=transfer_end,
                output_dir=args.transfer_output_dir,
                reviewed_input_path=args.gb_transfer_reviewed_input_path,
            )
            frames = {**frames, **reviewed_frames}
            print(
                f"[source=entsoe+topology+internal_reviewed] Materialized {len(frames)} tables for "
                f"{transfer_start} to {transfer_end} "
                f"(inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.transfer_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted GB transfer-gate tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_france_connector_layer:
        if not args.france_start or not args.france_end:
            raise SystemExit("--materialize-france-connector-layer requires --france-start and --france-end")

        france_start = parse_transfer_iso_date(args.france_start)
        france_end = parse_transfer_iso_date(args.france_end)
        if france_end < france_start:
            raise SystemExit("--france-end must be on or after --france-start")

        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit("--materialize-france-connector-layer requires ENTOS_E_TOKEN or ENTSOE_TOKEN")

        try:
            raw_remit, remit_status = fetch_remit_event_detail_with_status(france_start, france_end)
            france_reviewed_period = build_fact_france_connector_reviewed_period(
                start_date=france_start,
                end_date=france_end,
                reviewed_input_path=args.france_reviewed_input_path,
            )
            france_reviewed_notice = build_fact_france_connector_notice_hourly(
                start_date=france_start,
                end_date=france_end,
                reviewed_period=france_reviewed_period,
            )
            eleclink_auth_kwargs = resolve_eleclink_umm_auth_args(args)
            eleclink_auth_frame, eleclink_auth_status = fetch_eleclink_umm_authenticated(**eleclink_auth_kwargs)
            eleclink_export = load_eleclink_umm_export(args.eleclink_umm_export_path)
            selected_eleclink, eleclink_source_compare, eleclink_resolution = build_eleclink_operator_source_compare(
                start_date=france_start,
                end_date=france_end,
                authenticated_frame=eleclink_auth_frame,
                authenticated_status=eleclink_auth_status,
                export_frame=eleclink_export,
                export_attempted_flag=bool(args.eleclink_umm_export_path),
            )
            france_operator_event = build_france_connector_operator_event_frame(
                raw_remit,
                eleclink_umm_export=selected_eleclink,
            )
            france_availability = build_fact_france_connector_availability_hourly(
                start_date=france_start,
                end_date=france_end,
                operator_event_frame=france_operator_event,
                remit_fetch_status_by_date=remit_status,
                eleclink_source_resolution=eleclink_resolution,
            )
            frames = materialize_france_connector_history(
                start_date=france_start,
                end_date=france_end,
                output_dir=args.france_output_dir,
                token=entsoe_token,
                france_connector_reviewed_period=france_reviewed_period,
                france_connector_availability=france_availability,
            )
            frames[FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE] = france_reviewed_period
            frames[FRANCE_CONNECTOR_NOTICE_TABLE] = france_reviewed_notice
            frames[FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE] = france_operator_event
            frames[FRANCE_CONNECTOR_AVAILABILITY_TABLE] = france_availability
            frames[FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE] = eleclink_source_compare
            france_reviewed_period.to_csv(
                os.path.join(args.france_output_dir, f"{FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE}.csv"),
                index=False,
            )
            france_reviewed_notice.to_csv(
                os.path.join(args.france_output_dir, f"{FRANCE_CONNECTOR_NOTICE_TABLE}.csv"),
                index=False,
            )
            france_operator_event.to_csv(
                os.path.join(args.france_output_dir, f"{FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE}.csv"),
                index=False,
            )
            france_availability.to_csv(
                os.path.join(args.france_output_dir, f"{FRANCE_CONNECTOR_AVAILABILITY_TABLE}.csv"),
                index=False,
            )
            eleclink_source_compare.to_csv(
                os.path.join(args.france_output_dir, f"{FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE}.csv"),
                index=False,
            )
            print(
                f"[source=entsoe+elexon_remit+france_connector_policy] Materialized {len(frames)} tables for "
                f"{france_start} to {france_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.france_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted France connector-layer tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_curtailment_opportunity_history:
        if not args.opportunity_start or not args.opportunity_end:
            raise SystemExit(
                "--materialize-curtailment-opportunity-history requires --opportunity-start and --opportunity-end"
            )

        opportunity_start = parse_market_day(args.opportunity_start)
        opportunity_end = parse_market_day(args.opportunity_end)
        if opportunity_end < opportunity_start:
            raise SystemExit("--opportunity-end must be on or after --opportunity-start")

        market_days = list(iter_market_days(opportunity_start, opportunity_end + dt.timedelta(days=1)))
        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit(
                "--materialize-curtailment-opportunity-history requires ENTOS_E_TOKEN or ENTSOE_TOKEN"
            )

        bmrs_api_key = next((os.environ.get(name) for name in BMRS_KEY_ENV_NAMES if os.environ.get(name)), None)
        gbp_eur = args.gbp_eur
        if gbp_eur is None:
            for env_name in FX_ENV_NAMES:
                if os.environ.get(env_name):
                    gbp_eur = parse_gbp_eur(os.environ[env_name])
                    break
        gb_provider = args.gb_provider or next(
            (os.environ.get(name) for name in GB_PROVIDER_ENV_NAMES if os.environ.get(name)),
            DEFAULT_GB_PROVIDER,
        )

        try:
            prices, provider_used = fetch_prices(market_days, entsoe_token, gbp_eur, bmrs_api_key, gb_provider)
            network_start = market_days[0]
            network_end = market_days[-1]
            interconnector_flow = build_fact_interconnector_flow_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            interconnector_itl = build_fact_interconnector_itl_hourly(
                start_date=network_start,
                end_date=network_end,
            )
            interconnector_capacity = build_fact_interconnector_capacity_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            review_audit = build_interconnector_capacity_source_audit(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            review_policy = build_interconnector_capacity_review_policy(
                review_audit[INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE]
            )
            reviewed_capacity = build_interconnector_capacity_reviewed_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
                review_policy=review_policy,
            )
            raw_remit, remit_status = fetch_remit_event_detail_with_status(network_start, network_end)
            france_reviewed_period = build_fact_france_connector_reviewed_period(
                start_date=network_start,
                end_date=network_end,
                reviewed_input_path=args.france_reviewed_input_path,
            )
            france_reviewed_notice = build_fact_france_connector_notice_hourly(
                start_date=network_start,
                end_date=network_end,
                reviewed_period=france_reviewed_period,
            )
            eleclink_auth_kwargs = resolve_eleclink_umm_auth_args(args)
            eleclink_auth_frame, eleclink_auth_status = fetch_eleclink_umm_authenticated(**eleclink_auth_kwargs)
            eleclink_export = load_eleclink_umm_export(args.eleclink_umm_export_path)
            selected_eleclink, eleclink_source_compare, eleclink_resolution = build_eleclink_operator_source_compare(
                start_date=network_start,
                end_date=network_end,
                authenticated_frame=eleclink_auth_frame,
                authenticated_status=eleclink_auth_status,
                export_frame=eleclink_export,
                export_attempted_flag=bool(args.eleclink_umm_export_path),
            )
            france_operator_event = build_france_connector_operator_event_frame(
                raw_remit,
                eleclink_umm_export=selected_eleclink,
            )
            france_availability = build_fact_france_connector_availability_hourly(
                start_date=network_start,
                end_date=network_end,
                operator_event_frame=france_operator_event,
                remit_fetch_status_by_date=remit_status,
                eleclink_source_resolution=eleclink_resolution,
            )
            france_connector = build_fact_france_connector_hourly(
                start_date=network_start,
                end_date=network_end,
                interconnector_flow=interconnector_flow,
                interconnector_capacity=interconnector_capacity,
                interconnector_capacity_review_policy=review_policy,
                interconnector_capacity_reviewed=reviewed_capacity,
                france_connector_reviewed_period=france_reviewed_period,
                france_connector_availability=france_availability,
            )
            gb_transfer_gate = build_fact_gb_transfer_gate_hourly(
                start_date=network_start,
                end_date=network_end,
                interconnector_flow=interconnector_flow,
                interconnector_capacity=interconnector_capacity,
            )
            day_ahead_constraint_boundary = build_fact_day_ahead_constraint_boundary_half_hourly(
                start_date=network_start,
                end_date=network_end,
            )
            gb_transfer_reviewed_period = build_fact_gb_transfer_reviewed_period(
                start_date=network_start,
                end_date=network_end,
                reviewed_input_path=args.gb_transfer_reviewed_input_path,
            )
            gb_transfer_review_policy = build_fact_gb_transfer_review_policy(gb_transfer_reviewed_period)
            gb_transfer_reviewed_hourly = build_fact_gb_transfer_reviewed_hourly(
                start_date=network_start,
                end_date=network_end,
                reviewed_period=gb_transfer_reviewed_period,
                review_policy=gb_transfer_review_policy,
            )
            gb_transfer_boundary_reviewed_hourly = build_fact_gb_transfer_boundary_reviewed_hourly(
                start_date=network_start,
                end_date=network_end,
                day_ahead_constraint_boundary=day_ahead_constraint_boundary,
            )
            combined_gb_transfer_reviewed_hourly = pd.concat(
                [gb_transfer_reviewed_hourly, gb_transfer_boundary_reviewed_hourly],
                ignore_index=True,
                sort=False,
            )
            route_frames = materialize_route_score_history(
                output_dir=args.opportunity_output_dir,
                prices=prices,
                gb_transfer_gate=gb_transfer_gate,
                interconnector_itl=interconnector_itl,
                interconnector_flow=interconnector_flow,
                interconnector_capacity=interconnector_capacity,
                interconnector_capacity_reviewed=reviewed_capacity,
                interconnector_capacity_review_policy=review_policy,
                france_connector=france_connector,
                france_connector_notice=france_reviewed_notice,
                gb_transfer_reviewed_hourly=combined_gb_transfer_reviewed_hourly,
            )
            route_score = route_frames[ROUTE_SCORE_TABLE]
            if args.market_state_input_path:
                upstream_market_state = build_fact_upstream_market_state_hourly(
                    start_date=network_start,
                    end_date=network_end,
                    input_path=args.market_state_input_path,
                )
            else:
                upstream_market_state = build_fact_upstream_market_state_hourly_from_price_frame(
                    prices=prices,
                    gb_source_provider=provider_used,
                )
            cluster_curtailment_proxy = fetch_cluster_curtailment_proxy_hourly(
                start_date=network_start,
                end_date=network_end,
            )
            truth_frame = None
            if args.opportunity_truth_profile != "proxy":
                with tempfile.TemporaryDirectory() as temp_truth_dir:
                    truth_frames = materialize_bmu_curtailment_truth(
                        start_date=network_start,
                        end_date=network_end,
                        output_dir=temp_truth_dir,
                        truth_profile=args.opportunity_truth_profile,
                    )
                truth_frame = truth_frames["fact_bmu_curtailment_truth_half_hourly"]
            opportunity_frames = materialize_curtailment_opportunity_history(
                output_dir=args.opportunity_output_dir,
                fact_route_score_hourly=route_score,
                fact_regional_curtailment_hourly_proxy=cluster_curtailment_proxy,
                fact_bmu_curtailment_truth_half_hourly=truth_frame,
                fact_upstream_market_state_hourly=upstream_market_state,
                truth_profile=args.opportunity_truth_profile,
            )
            cluster_curtailment_proxy.to_csv(
                os.path.join(args.opportunity_output_dir, "fact_regional_curtailment_hourly_proxy.csv"),
                index=False,
            )
            gb_transfer_reviewed_period.to_csv(
                os.path.join(args.opportunity_output_dir, f"{GB_TRANSFER_REVIEWED_PERIOD_TABLE}.csv"),
                index=False,
            )
            gb_transfer_review_policy.to_csv(
                os.path.join(args.opportunity_output_dir, f"{GB_TRANSFER_REVIEW_POLICY_TABLE}.csv"),
                index=False,
            )
            gb_transfer_reviewed_hourly.to_csv(
                os.path.join(args.opportunity_output_dir, f"{GB_TRANSFER_REVIEWED_HOURLY_TABLE}.csv"),
                index=False,
            )
            gb_transfer_boundary_reviewed_hourly.to_csv(
                os.path.join(args.opportunity_output_dir, f"{GB_TRANSFER_BOUNDARY_REVIEWED_TABLE}.csv"),
                index=False,
            )
            day_ahead_constraint_boundary.to_csv(
                os.path.join(args.opportunity_output_dir, f"{DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE}.csv"),
                index=False,
            )
            interconnector_itl.to_csv(
                os.path.join(args.opportunity_output_dir, f"{INTERCONNECTOR_ITL_TABLE}.csv"),
                index=False,
            )
            upstream_market_state.to_csv(
                os.path.join(args.opportunity_output_dir, f"{UPSTREAM_MARKET_STATE_TABLE}.csv"),
                index=False,
            )
            frames = {
                ROUTE_SCORE_TABLE: route_score,
                "fact_regional_curtailment_hourly_proxy": cluster_curtailment_proxy,
                DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE: day_ahead_constraint_boundary,
                GB_TRANSFER_BOUNDARY_REVIEWED_TABLE: gb_transfer_boundary_reviewed_hourly,
                GB_TRANSFER_REVIEWED_PERIOD_TABLE: gb_transfer_reviewed_period,
                GB_TRANSFER_REVIEW_POLICY_TABLE: gb_transfer_review_policy,
                GB_TRANSFER_REVIEWED_HOURLY_TABLE: gb_transfer_reviewed_hourly,
                INTERCONNECTOR_ITL_TABLE: interconnector_itl,
                **opportunity_frames,
            }
            upstream_source_label = (
                "manual_or_api_market_state"
                if args.market_state_input_path
                else f"free_entsoe_day_ahead_plus_elexon_mid:{provider_used}"
            )
            print(
                f"[source=elexon_mid:{provider_used}+entsoe+neso+route_opportunity+{upstream_source_label}] "
                f"Materialized {len(frames)} tables for {opportunity_start} to {opportunity_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.opportunity_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                store_frames = dict(frames)
                store_frames[GB_TRANSFER_GATE_TABLE] = gb_transfer_gate
                store_frames[DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE] = day_ahead_constraint_boundary
                store_frames[GB_TRANSFER_BOUNDARY_REVIEWED_TABLE] = gb_transfer_boundary_reviewed_hourly
                store_frames[INTERCONNECTOR_ITL_TABLE] = interconnector_itl
                store_frames[GB_TRANSFER_REVIEWED_PERIOD_TABLE] = gb_transfer_reviewed_period
                store_frames[GB_TRANSFER_REVIEW_POLICY_TABLE] = gb_transfer_review_policy
                store_frames[GB_TRANSFER_REVIEWED_HOURLY_TABLE] = gb_transfer_reviewed_hourly
                store_frames[INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE] = review_policy
                store_frames[INTERCONNECTOR_CAPACITY_REVIEWED_TABLE] = reviewed_capacity
                store_frames[DIM_INTERCONNECTOR_CABLE_TABLE] = interconnector_cable_frame()
                store_frames[FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE] = france_reviewed_period
                store_frames[FRANCE_CONNECTOR_NOTICE_TABLE] = france_reviewed_notice
                store_frames[FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE] = france_operator_event
                store_frames[FRANCE_CONNECTOR_AVAILABILITY_TABLE] = france_availability
                store_frames[FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE] = eleclink_source_compare
                store_frames[FRANCE_CONNECTOR_TABLE] = france_connector
                if truth_frame is not None:
                    store_frames["fact_bmu_curtailment_truth_half_hourly"] = truth_frame
                summary = upsert_truth_frames_to_sqlite(store_frames, args.truth_store_db_path)
                print(
                    f"[store=sqlite] Upserted curtailment opportunity and supporting tables into {args.truth_store_db_path}"
                )
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_opportunity_backtest:
        try:
            opportunity_input = load_curtailment_opportunity_input(args.opportunity_input_path)
            forecast_horizons = coerce_forecast_horizons(args.backtest_horizons)
            frames = materialize_opportunity_backtest(
                output_dir=args.backtest_output_dir,
                fact_curtailment_opportunity_hourly=opportunity_input,
                model_key=args.backtest_model_key,
                forecast_horizons=forecast_horizons,
            )
            summary = summarize_backtest_prediction_hourly(frames[BACKTEST_PREDICTION_TABLE])
            print(
                f"[source={CURTAILMENT_OPPORTUNITY_TABLE}+horizon_backtest] "
                f"Materialized {len(frames)} tables from {args.opportunity_input_path} "
                f"for horizons={','.join(str(value) for value in forecast_horizons)}"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.backtest_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if not summary.empty:
                print(summary.round(4).to_string(index=False))
            if args.truth_store_db_path:
                store_summary = upsert_truth_frames_to_sqlite(frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted opportunity backtest tables into {args.truth_store_db_path}")
                for _, row in store_summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.materialize_route_score_history:
        if not args.route_score_start or not args.route_score_end:
            raise SystemExit("--materialize-route-score-history requires --route-score-start and --route-score-end")

        route_score_start = parse_market_day(args.route_score_start)
        route_score_end = parse_market_day(args.route_score_end)
        if route_score_end < route_score_start:
            raise SystemExit("--route-score-end must be on or after --route-score-start")

        market_days = list(iter_market_days(route_score_start, route_score_end + dt.timedelta(days=1)))
        entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
        if not entsoe_token:
            raise SystemExit("--materialize-route-score-history requires ENTOS_E_TOKEN or ENTSOE_TOKEN")

        bmrs_api_key = next((os.environ.get(name) for name in BMRS_KEY_ENV_NAMES if os.environ.get(name)), None)
        gbp_eur = args.gbp_eur
        if gbp_eur is None:
            for env_name in FX_ENV_NAMES:
                if os.environ.get(env_name):
                    gbp_eur = parse_gbp_eur(os.environ[env_name])
                    break
        gb_provider = args.gb_provider or next(
            (os.environ.get(name) for name in GB_PROVIDER_ENV_NAMES if os.environ.get(name)),
            DEFAULT_GB_PROVIDER,
        )

        try:
            prices, provider_used = fetch_prices(market_days, entsoe_token, gbp_eur, bmrs_api_key, gb_provider)
            network_start = market_days[0]
            network_end = market_days[-1]
            interconnector_flow = build_fact_interconnector_flow_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            interconnector_itl = build_fact_interconnector_itl_hourly(
                start_date=network_start,
                end_date=network_end,
            )
            interconnector_capacity = build_fact_interconnector_capacity_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            review_audit = build_interconnector_capacity_source_audit(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            review_policy = build_interconnector_capacity_review_policy(
                review_audit[INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE]
            )
            reviewed_capacity = build_interconnector_capacity_reviewed_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
                review_policy=review_policy,
            )
            raw_remit, remit_status = fetch_remit_event_detail_with_status(network_start, network_end)
            france_reviewed_period = build_fact_france_connector_reviewed_period(
                start_date=network_start,
                end_date=network_end,
                reviewed_input_path=args.france_reviewed_input_path,
            )
            france_reviewed_notice = build_fact_france_connector_notice_hourly(
                start_date=network_start,
                end_date=network_end,
                reviewed_period=france_reviewed_period,
            )
            eleclink_auth_kwargs = resolve_eleclink_umm_auth_args(args)
            eleclink_auth_frame, eleclink_auth_status = fetch_eleclink_umm_authenticated(**eleclink_auth_kwargs)
            eleclink_export = load_eleclink_umm_export(args.eleclink_umm_export_path)
            selected_eleclink, eleclink_source_compare, eleclink_resolution = build_eleclink_operator_source_compare(
                start_date=network_start,
                end_date=network_end,
                authenticated_frame=eleclink_auth_frame,
                authenticated_status=eleclink_auth_status,
                export_frame=eleclink_export,
                export_attempted_flag=bool(args.eleclink_umm_export_path),
            )
            france_operator_event = build_france_connector_operator_event_frame(
                raw_remit,
                eleclink_umm_export=selected_eleclink,
            )
            france_availability = build_fact_france_connector_availability_hourly(
                start_date=network_start,
                end_date=network_end,
                operator_event_frame=france_operator_event,
                remit_fetch_status_by_date=remit_status,
                eleclink_source_resolution=eleclink_resolution,
            )
            france_connector = build_fact_france_connector_hourly(
                start_date=network_start,
                end_date=network_end,
                interconnector_flow=interconnector_flow,
                interconnector_capacity=interconnector_capacity,
                interconnector_capacity_review_policy=review_policy,
                interconnector_capacity_reviewed=reviewed_capacity,
                france_connector_reviewed_period=france_reviewed_period,
                france_connector_availability=france_availability,
            )
            gb_transfer_gate = build_fact_gb_transfer_gate_hourly(
                start_date=network_start,
                end_date=network_end,
                interconnector_flow=interconnector_flow,
                interconnector_capacity=interconnector_capacity,
            )
            day_ahead_constraint_boundary = build_fact_day_ahead_constraint_boundary_half_hourly(
                start_date=network_start,
                end_date=network_end,
            )
            gb_transfer_reviewed_period = build_fact_gb_transfer_reviewed_period(
                start_date=network_start,
                end_date=network_end,
                reviewed_input_path=args.gb_transfer_reviewed_input_path,
            )
            gb_transfer_review_policy = build_fact_gb_transfer_review_policy(gb_transfer_reviewed_period)
            gb_transfer_reviewed_hourly = build_fact_gb_transfer_reviewed_hourly(
                start_date=network_start,
                end_date=network_end,
                reviewed_period=gb_transfer_reviewed_period,
                review_policy=gb_transfer_review_policy,
            )
            gb_transfer_boundary_reviewed_hourly = build_fact_gb_transfer_boundary_reviewed_hourly(
                start_date=network_start,
                end_date=network_end,
                day_ahead_constraint_boundary=day_ahead_constraint_boundary,
            )
            combined_gb_transfer_reviewed_hourly = pd.concat(
                [gb_transfer_reviewed_hourly, gb_transfer_boundary_reviewed_hourly],
                ignore_index=True,
                sort=False,
            )
            frames = materialize_route_score_history(
                output_dir=args.route_score_output_dir,
                prices=prices,
                gb_transfer_gate=gb_transfer_gate,
                interconnector_itl=interconnector_itl,
                interconnector_flow=interconnector_flow,
                interconnector_capacity=interconnector_capacity,
                interconnector_capacity_reviewed=reviewed_capacity,
                interconnector_capacity_review_policy=review_policy,
                france_connector=france_connector,
                france_connector_notice=france_reviewed_notice,
                gb_transfer_reviewed_hourly=combined_gb_transfer_reviewed_hourly,
            )
            frames[DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE] = day_ahead_constraint_boundary
            frames[GB_TRANSFER_BOUNDARY_REVIEWED_TABLE] = gb_transfer_boundary_reviewed_hourly
            frames[GB_TRANSFER_REVIEWED_PERIOD_TABLE] = gb_transfer_reviewed_period
            frames[GB_TRANSFER_REVIEW_POLICY_TABLE] = gb_transfer_review_policy
            frames[GB_TRANSFER_REVIEWED_HOURLY_TABLE] = gb_transfer_reviewed_hourly
            frames[INTERCONNECTOR_ITL_TABLE] = interconnector_itl
            day_ahead_constraint_boundary.to_csv(
                os.path.join(args.route_score_output_dir, f"{DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE}.csv"),
                index=False,
            )
            gb_transfer_boundary_reviewed_hourly.to_csv(
                os.path.join(args.route_score_output_dir, f"{GB_TRANSFER_BOUNDARY_REVIEWED_TABLE}.csv"),
                index=False,
            )
            gb_transfer_reviewed_period.to_csv(
                os.path.join(args.route_score_output_dir, f"{GB_TRANSFER_REVIEWED_PERIOD_TABLE}.csv"),
                index=False,
            )
            gb_transfer_review_policy.to_csv(
                os.path.join(args.route_score_output_dir, f"{GB_TRANSFER_REVIEW_POLICY_TABLE}.csv"),
                index=False,
            )
            gb_transfer_reviewed_hourly.to_csv(
                os.path.join(args.route_score_output_dir, f"{GB_TRANSFER_REVIEWED_HOURLY_TABLE}.csv"),
                index=False,
            )
            interconnector_itl.to_csv(
                os.path.join(args.route_score_output_dir, f"{INTERCONNECTOR_ITL_TABLE}.csv"),
                index=False,
            )
            print(
                f"[source=elexon_mid:{provider_used}+entsoe+entsoe_network] Materialized {len(frames)} tables for "
                f"{route_score_start} to {route_score_end} (inclusive)"
            )
            for table_name, frame in frames.items():
                output_path = os.path.join(args.route_score_output_dir, f"{table_name}.csv")
                print(f"{table_name}: rows={len(frame)} path={output_path}")
            if args.truth_store_db_path:
                store_frames = dict(frames)
                store_frames[GB_TRANSFER_GATE_TABLE] = gb_transfer_gate
                store_frames[DAY_AHEAD_CONSTRAINT_BOUNDARY_TABLE] = day_ahead_constraint_boundary
                store_frames[GB_TRANSFER_BOUNDARY_REVIEWED_TABLE] = gb_transfer_boundary_reviewed_hourly
                store_frames[INTERCONNECTOR_ITL_TABLE] = interconnector_itl
                store_frames[DIM_INTERCONNECTOR_CABLE_TABLE] = interconnector_cable_frame()
                store_frames[INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE] = review_policy
                store_frames[INTERCONNECTOR_CAPACITY_REVIEWED_TABLE] = reviewed_capacity
                store_frames[FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE] = france_reviewed_period
                store_frames[FRANCE_CONNECTOR_NOTICE_TABLE] = france_reviewed_notice
                store_frames[FRANCE_CONNECTOR_OPERATOR_EVENT_TABLE] = france_operator_event
                store_frames[FRANCE_CONNECTOR_AVAILABILITY_TABLE] = france_availability
                store_frames[FRANCE_CONNECTOR_OPERATOR_SOURCE_COMPARE_TABLE] = eleclink_source_compare
                store_frames[FRANCE_CONNECTOR_TABLE] = france_connector
                summary = upsert_truth_frames_to_sqlite(store_frames, args.truth_store_db_path)
                print(f"[store=sqlite] Upserted route-score and reviewed-capacity tables into {args.truth_store_db_path}")
                for _, row in summary.iterrows():
                    print(
                        f"{row['table_name']}: rows_loaded={int(row['rows_loaded'])} "
                        f"table_rows={int(row['table_row_count'])}"
                    )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    market_days = resolve_market_days(args)
    entsoe_token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""
    bmrs_api_key = next((os.environ.get(name) for name in BMRS_KEY_ENV_NAMES if os.environ.get(name)), None)
    gbp_eur = args.gbp_eur
    if gbp_eur is None:
        for env_name in FX_ENV_NAMES:
            if os.environ.get(env_name):
                gbp_eur = parse_gbp_eur(os.environ[env_name])
                break
    gb_provider = args.gb_provider or next(
        (os.environ.get(name) for name in GB_PROVIDER_ENV_NAMES if os.environ.get(name)),
        DEFAULT_GB_PROVIDER,
    )

    try:
        if args.dry:
            prices = synthetic_prices(market_days)
            used_source = "synthetic (--dry)"
            interconnector_flow = None
            interconnector_capacity = None
        else:
            if not entsoe_token:
                raise RuntimeError("missing ENTSO-E token; set ENTOS_E_TOKEN or ENTSOE_TOKEN, or use --dry")
            prices, provider_used = fetch_prices(market_days, entsoe_token, gbp_eur, bmrs_api_key, gb_provider)
            network_start = market_days[0]
            network_end = market_days[-1]
            interconnector_flow = build_fact_interconnector_flow_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            interconnector_capacity = build_fact_interconnector_capacity_hourly(
                start_date=network_start,
                end_date=network_end,
                token=entsoe_token,
            )
            used_source = f"elexon_mid:{provider_used}+entsoe+entsoe_network"

        out = compute_netbacks(
            prices,
            interconnector_flow=interconnector_flow,
            interconnector_capacity=interconnector_capacity,
        )
        out.to_csv(args.save)

        market_day_end = market_days[-1] + dt.timedelta(days=1)
        print(f"[source={used_source}] Market days: {market_days[0]} to {market_day_end} (exclusive), rows={len(out)}")
        print(out.head(6).round(2).to_string())
        print(f"\nSaved: {args.save}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
