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

# Simple route heuristics. The model scores each border leg independently and treats
# a route as blocked if any leg is underwater for that hour.
ROUTES = {
    "R1_netback_GB_FR_DE_PL": {
        "label": "GB->FR->DE->PL",
        "legs": (
            {"from": "GB", "to": "FR", "loss": 0.020, "fee": 0.60},
            {"from": "FR", "to": "DE", "loss": 0.010, "fee": 0.20},
            {"from": "DE", "to": "PL", "loss": 0.010, "fee": 0.30},
        ),
    },
    "R2_netback_GB_NL_DE_PL": {
        "label": "GB->NL->DE->PL",
        "legs": (
            {"from": "GB", "to": "NL", "loss": 0.020, "fee": 0.65},
            {"from": "NL", "to": "DE", "loss": 0.010, "fee": 0.15},
            {"from": "DE", "to": "PL", "loss": 0.010, "fee": 0.30},
        ),
    },
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


def compute_route_metrics(df: pd.DataFrame, route_name: str, route_spec: Dict[str, object]) -> None:
    leg_margin_cols = []
    leg_labels = []

    for leg in route_spec["legs"]:
        source = leg["from"]
        sink = leg["to"]
        loss = float(leg["loss"])
        fee = float(leg["fee"])
        col = f"{route_name}_leg_{source}_{sink}"
        df[col] = (df[sink] * (1 - loss)) - df[source] - fee
        leg_margin_cols.append(col)
        leg_labels.append(f"{source}->{sink}")

    gross_col = f"{route_name}_gross"
    feasible_col = f"{route_name}_feasible"
    bottleneck_col = f"{route_name}_bottleneck"

    leg_margins = df[leg_margin_cols]
    df[gross_col] = leg_margins.sum(axis=1)
    df[feasible_col] = leg_margins.gt(0).all(axis=1)
    df[bottleneck_col] = leg_margins.idxmin(axis=1).map(dict(zip(leg_margin_cols, leg_labels)))
    df[route_name] = np.where(df[feasible_col], df[gross_col], leg_margins.min(axis=1))


def compute_netbacks(prices: pd.DataFrame) -> pd.DataFrame:
    """
    prices: index=UTC hour, columns ['GB', 'FR', 'NL', 'DE', 'PL', 'CZ'] in EUR/MWh.
    Returns prices plus route metrics and a simple route signal.
    """
    required = {"GB", "FR", "NL", "DE", "PL", "CZ"}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise RuntimeError(f"missing price columns: {', '.join(missing)}")

    df = prices.copy().sort_index().interpolate(limit_direction="both")
    for route_name, route_spec in ROUTES.items():
        compute_route_metrics(df, route_name, route_spec)

    route_cols = list(ROUTES)
    route_label_map = {route_name: route_spec["label"] for route_name, route_spec in ROUTES.items()}

    df["best_netback"] = df[route_cols].max(axis=1)
    df["best_route"] = np.where(
        df["R1_netback_GB_FR_DE_PL"] >= df["R2_netback_GB_NL_DE_PL"],
        route_label_map["R1_netback_GB_FR_DE_PL"],
        route_label_map["R2_netback_GB_NL_DE_PL"],
    )
    df["export_signal"] = np.where(df["best_netback"] > 0, "EXPORT", "HOLD")
    return df


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
    args = parser.parse_args()

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
