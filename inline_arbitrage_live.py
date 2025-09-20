
#!/usr/bin/env python3
"""
inline_arbitrage_live.py
------------------------
Fetches day-ahead electricity prices from ENTSO-E for GB, FR, NL, DE-LU, PL, CZ,
then computes simple GB->(FR|NL)->DE->PL netbacks per hour, including basic losses & fees.
If ENTOS_E_TOKEN is missing or network fails, falls back to synthetic demo data.

Usage:
  python inline_arbitrage_live.py --date 2025-09-20
  python inline_arbitrage_live.py --start 2025-09-20 --end 2025-09-21
  python inline_arbitrage_live.py --save out.csv

Set your ENTSO-E token via ENV var ENTOS_E_TOKEN or a .env file in the same directory.

Notes:
- This script focuses on DA price spreads; capacity/congestion is approximated by simple heuristics.
- For production, add physical flow/ATC checks and imbalance/capacity cost models.
"""

import os
import sys
import argparse
import datetime as dt
from typing import List, Dict
import pandas as pd
import numpy as np

# Optional: .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request


ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
# ENTSO-E bidding zone EIC codes
ZONES = {
    "GB":   "BZN|GB",
    "FR":   "BZN|FR",
    "NL":   "BZN|NL",
    "DE":   "BZN|DE-LU",
    "PL":   "BZN|PL",
    "CZ":   "BZN|CZ",
}

def iso_interval(start: dt.datetime, end: dt.datetime) -> str:
    # ENTSO-E expects UTC in YYYYMMDDHHMM format
    return start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")

def fetch_entsoe_da_price(zone_code: str, start: dt.datetime, end: dt.datetime, token: str) -> pd.DataFrame:
    """
    Fetches day-ahead prices (documentType A44) for a given bidding zone between [start, end).
    Returns a DataFrame with UTC timestamps and price in EUR/MWh.
    """
    params = {
        "securityToken": token,
        "documentType": "A44",
        "in_Domain": zone_code,
        "out_Domain": zone_code,
    }
    periodStart, periodEnd = iso_interval(start, end)
    params["periodStart"] = periodStart
    params["periodEnd"] = periodEnd
    url = ENTSOE_ENDPOINT + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        ns = {'ns': 'urn:entsoe.eu:wgedi:acknowledgementdocument:7:0'}
        # Prices are typically in "Publication_MarketDocument" with TimeSeries/Period/Point
        # Handle common namespaces
        if b"Publication_MarketDocument" not in xml_data:
            # Could be an error message from API
            raise RuntimeError("ENTSO-E returned an unexpected payload (maybe invalid token or window).")
        # Try generic parse
        return parse_entsoe_price_xml(xml_data)
    except Exception as e:
        raise

def parse_entsoe_price_xml(xml_bytes: bytes) -> pd.DataFrame:
    # A light-weight parser that extracts (time, price) from the common ENTSO-E DA price schema.
    # The schema can vary; we handle the typical case here.
    root = ET.fromstring(xml_bytes)
    ns = {'ns': root.tag.split('}')[0].strip('{')}
    series = []
    for ts in root.findall('.//ns:TimeSeries', ns):
        currency = ts.findtext('.//ns:currency_Unit.name', default="", namespaces=ns)
        price_unit = ts.findtext('.//ns:price_Measure_Unit.name', default="", namespaces=ns)
        for period in ts.findall('.//ns:Period', ns):
            start_text = period.findtext('ns:timeInterval/ns:start', namespaces=ns)
            resolution = period.findtext('ns:resolution', namespaces=ns) or 'PT60M'
            start = pd.to_datetime(start_text, utc=True)
            points = []
            for p in period.findall('ns:Point', ns):
                pos = int(p.findtext('ns:position', namespaces=ns))
                val = float(p.findtext('ns:price.amount', namespaces=ns))
                points.append((pos, val))
            points.sort(key=lambda x: x[0])
            # Build hourly index
            if resolution != 'PT60M':
                # For simplicity, we assume hourly; extend here if you need quarter-hours
                pass
            times = [start + pd.Timedelta(hours=i) for i,_ in points]
            vals = [v for _,v in points]
            df = pd.DataFrame({"time_utc": times, "price_eur_mwh": vals})
            series.append(df)
    if not series:
        raise RuntimeError("Failed to parse DA prices from ENTSO-E payload.")
    out = pd.concat(series, ignore_index=True).drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
    out.set_index("time_utc", inplace=True)
    return out

def fetch_prices_entsoe(zones: Dict[str,str], start: dt.datetime, end: dt.datetime, token: str) -> pd.DataFrame:
    frames = {}
    for name, code in zones.items():
        df = fetch_entsoe_da_price(code, start, end, token)
        frames[name] = df.rename(columns={"price_eur_mwh": name})
    # outer-join on UTC timestamps
    all_df = None
    for name, df in frames.items():
        all_df = df if all_df is None else all_df.join(df, how="outer")
    return all_df

def compute_netbacks(prices: pd.DataFrame) -> pd.DataFrame:
    """
    prices: index=UTC hour, columns ['GB','FR','NL','DE','PL','CZ']
    Returns prices + netback columns + route & signal.
    """
    df = prices.copy()
    # Fill small gaps with forward fill to avoid dropping whole rows
    df = df.sort_index().interpolate(limit_direction='both')
    # Route losses (very simplified): HVDC 2% each, AC 1% each
    loss_R = 1 - (1-0.02)*(1-0.01)*(1-0.01)  # ~0.0394
    fees_total = 0.8 + 0.3  # capacity+ops placeholders
    # Capacity heuristics (dummy: allow all hours by default)
    cap_R1 = 1.0
    cap_R2 = 1.0

    df["R1_netback_GB_FR_DE_PL"] = ((df["PL"] * (1 - loss_R)) - df["GB"] - fees_total) * cap_R1
    df["R2_netback_GB_NL_DE_PL"] = ((df["PL"] * (1 - loss_R)) - df["GB"] - fees_total) * cap_R2
    df["best_netback"] = df[["R1_netback_GB_FR_DE_PL","R2_netback_GB_NL_DE_PL"]].max(axis=1)
    df["best_route"] = np.where(
        df["R1_netback_GB_FR_DE_PL"] >= df["R2_netback_GB_NL_DE_PL"],
        "GB→FR→DE→PL",
        "GB→NL→DE→PL"
    )
    df["export_signal"] = np.where(df["best_netback"] > 0, "EXPORT", "HOLD")
    return df

def synthetic_prices(day: dt.date) -> pd.DataFrame:
    # Mirror the demo shape but time-zone aware UTC hours for the chosen day
    idx = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=24, freq="H")
    rng = np.random.default_rng(42)
    gb = -5 + 10*np.sin(np.linspace(0, 3*np.pi, 24)) + rng.normal(0, 4, 24)
    fr = 20 + 12*np.sin(np.linspace(0.2, 3.2*np.pi, 24)) + rng.normal(0, 4, 24)
    nl = 22 + 12*np.sin(np.linspace(0.3, 3.1*np.pi, 24)) + rng.normal(0, 4, 24)
    de = 24 + 11*np.sin(np.linspace(0.4, 3.0*np.pi, 24)) + rng.normal(0, 4, 24)
    pl = 55 + 18*np.sin(np.linspace(0.5, 2.6*np.pi, 24)) + rng.normal(0, 6, 24)
    cz = 52 + 16*np.sin(np.linspace(0.6, 2.7*np.pi, 24)) + rng.normal(0, 5, 24)
    df = pd.DataFrame({"GB":gb,"FR":fr,"NL":nl,"DE":de,"PL":pl,"CZ":cz}, index=idx).round(2)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Single UTC date (YYYY-MM-DD)")
    parser.add_argument("--start", help="UTC start (YYYY-MM-DD)")
    parser.add_argument("--end", help="UTC end (YYYY-MM-DD, exclusive)")
    parser.add_argument("--save", help="Path to save CSV", default="inline_arbitrage_live_output.csv")
    parser.add_argument("--dry", action="store_true", help="Use synthetic data (no API calls)")
    args = parser.parse_args()

    token = os.environ.get("ENTOS_E_TOKEN") or os.environ.get("ENTSOE_TOKEN") or ""

    # Determine window
    if args.date:
        start = dt.datetime.fromisoformat(args.date).replace(tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=1)
    else:
        if not args.start:
            # default: today UTC
            start = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc)
        else:
            start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
        if not args.end:
            end = start + dt.timedelta(days=1)
        else:
            end = dt.datetime.fromisoformat(args.end).replace(tzinfo=dt.timezone.utc)

    # Get prices
    if args.dry or not token:
        prices = synthetic_prices(start.date())
        used_source = "synthetic"
    else:
        try:
            prices = fetch_prices_entsoe(ZONES, start, end, token)
            used_source = "entsoe"
        except Exception as e:
            # fallback
            prices = synthetic_prices(start.date())
            used_source = f"fallback_synthetic_due_to_error: {e}"

    out = compute_netbacks(prices)
    out.to_csv(args.save)

    print(f"[source={used_source}] Window: {start} to {end}, rows={len(out)}")
    print(out.head(6).round(2).to_string())
    print(f"\nSaved: {args.save}")

if __name__ == "__main__":
    main()
