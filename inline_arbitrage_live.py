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
from bmu_dispatch import materialize_bmu_dispatch_history, parse_iso_date as parse_dispatch_iso_date
from bmu_generation import materialize_bmu_generation_history, parse_iso_date as parse_bmu_iso_date
from curtailment_truth import materialize_bmu_curtailment_truth
from curtailment_signals import materialize_curtailed_history, parse_iso_date
from exploration_plan import backtest_plan_frame, dataset_plan_frame, drift_monitor_plan_frame, map_layer_plan_frame
from gb_topology import cluster_hub_matrix, interconnector_hub_frame, reachability_frame, route_hub_frame
from history_store import ingest_truth_csv_tree_to_sqlite, upsert_truth_frames_to_sqlite
from physical_constraints import assumption_frame, compute_netbacks
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
    if resolution == "mixed":
        raise RuntimeError(f"mixed resolutions in ENTSO-E payload: {sorted(resolutions)}")

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
        "--materialize-weather-history",
        action="store_true",
        help="Fetch and save anchor, cluster, and parent-region weather history, then exit",
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
        else:
            if not entsoe_token:
                raise RuntimeError("missing ENTSO-E token; set ENTOS_E_TOKEN or ENTSOE_TOKEN, or use --dry")
            prices, provider_used = fetch_prices(market_days, entsoe_token, gbp_eur, bmrs_api_key, gb_provider)
            used_source = f"elexon_mid:{provider_used}+entsoe"

        out = compute_netbacks(prices)
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
