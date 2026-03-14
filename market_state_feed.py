from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from physical_constraints import ROUTES


UPSTREAM_MARKET_STATE_TABLE = "fact_upstream_market_state_hourly"

_KNOWN_ROUTE_NAMES = set(ROUTES)
_SPREAD_FLAT_ABS_MAX = 5.0
_SPREAD_WIDE_ABS_MAX = 20.0
_ROUTE_DAY_AHEAD_ZONE = {
    "R1_netback_GB_FR_DE_PL": "FR",
    "R2_netback_GB_NL_DE_PL": "NL",
}

_COLUMN_ALIASES = {
    "interval_start_utc": "interval_start_utc",
    "interval_start": "interval_start_utc",
    "timestamp_utc": "interval_start_utc",
    "delivery_start_utc": "interval_start_utc",
    "route_name": "route_name",
    "route": "route_name",
    "source_provider": "source_provider",
    "provider": "source_provider",
    "source_family": "source_family",
    "family": "source_family",
    "source_key": "source_key",
    "feed_key": "source_key",
    "source_published_utc": "source_published_utc",
    "published_utc": "source_published_utc",
    "published_at_utc": "source_published_utc",
    "forward_price_eur_per_mwh": "forward_price_eur_per_mwh",
    "forward_price": "forward_price_eur_per_mwh",
    "day_ahead_price_eur_per_mwh": "day_ahead_price_eur_per_mwh",
    "day_ahead_price": "day_ahead_price_eur_per_mwh",
    "da_price_eur_per_mwh": "day_ahead_price_eur_per_mwh",
    "intraday_price_eur_per_mwh": "intraday_price_eur_per_mwh",
    "intraday_price": "intraday_price_eur_per_mwh",
    "id_price_eur_per_mwh": "intraday_price_eur_per_mwh",
    "imbalance_price_eur_per_mwh": "imbalance_price_eur_per_mwh",
    "imbalance_price": "imbalance_price_eur_per_mwh",
}


def _empty_upstream_market_state_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start_utc",
            "interval_end_utc",
            "route_name",
            "source_provider",
            "source_family",
            "source_key",
            "source_published_utc",
            "forward_price_eur_per_mwh",
            "day_ahead_price_eur_per_mwh",
            "intraday_price_eur_per_mwh",
            "imbalance_price_eur_per_mwh",
            "forward_to_day_ahead_spread_eur_per_mwh",
            "day_ahead_to_intraday_spread_eur_per_mwh",
            "forward_to_day_ahead_spread_bucket",
            "day_ahead_to_intraday_spread_bucket",
            "upstream_market_state",
            "upstream_market_state_feed_available_flag",
        ]
    )


def _canonicalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("(eur/mwh)", "").replace("(utc)", "")
    text = text.replace("/", " ").replace("-", " ").replace(".", " ")
    text = "_".join(token for token in text.split() if token)
    return _COLUMN_ALIASES.get(text, text)


def _read_tabular_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("rows", "data", "items", "records"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError(f"unsupported upstream market-state JSON payload: {path}")

    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")

    raise ValueError(f"unsupported upstream market-state input format: {path.suffix}")


def _spread_bucket(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.where(
            numeric.isna(),
            "spread_unknown",
            np.where(
                numeric.le(-_SPREAD_WIDE_ABS_MAX),
                "spread_strong_negative",
                np.where(
                    numeric.le(-_SPREAD_FLAT_ABS_MAX),
                    "spread_negative",
                    np.where(
                        numeric.lt(_SPREAD_FLAT_ABS_MAX),
                        "spread_flat",
                        np.where(
                            numeric.lt(_SPREAD_WIDE_ABS_MAX),
                            "spread_positive",
                            "spread_strong_positive",
                        ),
                    ),
                ),
            ),
        ),
        index=values.index,
    )


def _market_state_label(frame: pd.DataFrame) -> pd.Series:
    available = frame["upstream_market_state_feed_available_flag"]
    da_id_bucket = frame["day_ahead_to_intraday_spread_bucket"]
    fwd_da_bucket = frame["forward_to_day_ahead_spread_bucket"]
    state = pd.Series("no_upstream_feed", index=frame.index, dtype="object")
    state.loc[available & da_id_bucket.eq("spread_strong_positive")] = "intraday_much_stronger_than_day_ahead"
    state.loc[available & da_id_bucket.eq("spread_positive")] = "intraday_stronger_than_day_ahead"
    state.loc[available & da_id_bucket.eq("spread_flat")] = "intraday_near_day_ahead"
    state.loc[available & da_id_bucket.eq("spread_negative")] = "intraday_weaker_than_day_ahead"
    state.loc[available & da_id_bucket.eq("spread_strong_negative")] = "intraday_much_weaker_than_day_ahead"
    state.loc[
        available
        & da_id_bucket.eq("spread_unknown")
        & fwd_da_bucket.eq("spread_strong_positive")
    ] = "day_ahead_much_stronger_than_forward"
    state.loc[
        available
        & da_id_bucket.eq("spread_unknown")
        & fwd_da_bucket.eq("spread_positive")
    ] = "day_ahead_stronger_than_forward"
    state.loc[
        available
        & da_id_bucket.eq("spread_unknown")
        & fwd_da_bucket.eq("spread_flat")
    ] = "day_ahead_near_forward"
    state.loc[
        available
        & da_id_bucket.eq("spread_unknown")
        & fwd_da_bucket.eq("spread_negative")
    ] = "day_ahead_weaker_than_forward"
    state.loc[
        available
        & da_id_bucket.eq("spread_unknown")
        & fwd_da_bucket.eq("spread_strong_negative")
    ] = "day_ahead_much_weaker_than_forward"
    state.loc[available & state.eq("no_upstream_feed")] = "partial_upstream_feed"
    return state


def normalize_upstream_market_state_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_upstream_market_state_frame()

    normalized = frame.copy()
    normalized.columns = [_canonicalize_column_name(column) for column in normalized.columns]

    if "interval_start_utc" not in normalized.columns:
        raise ValueError("upstream market-state input requires interval_start_utc")
    if "route_name" not in normalized.columns:
        raise ValueError("upstream market-state input requires route_name")

    normalized["interval_start_utc"] = pd.to_datetime(
        normalized["interval_start_utc"],
        utc=True,
        errors="coerce",
    )
    normalized = normalized[normalized["interval_start_utc"].notna()].copy()
    normalized["interval_end_utc"] = normalized.get("interval_end_utc")
    normalized["interval_end_utc"] = pd.to_datetime(
        normalized["interval_end_utc"],
        utc=True,
        errors="coerce",
    )
    normalized["interval_end_utc"] = normalized["interval_end_utc"].where(
        normalized["interval_end_utc"].notna(),
        normalized["interval_start_utc"] + pd.Timedelta(hours=1),
    )

    normalized["route_name"] = normalized["route_name"].astype(str).str.strip()
    normalized = normalized[normalized["route_name"].isin(_KNOWN_ROUTE_NAMES)].copy()
    if normalized.empty:
        return _empty_upstream_market_state_frame()

    for column, default in (
        ("source_provider", "manual_upstream_market_state"),
        ("source_family", "manual_market_state_feed"),
        ("source_key", "manual_upstream_market_state"),
    ):
        normalized[column] = normalized.get(column, default)
        normalized[column] = normalized[column].where(normalized[column].notna(), default)

    normalized["source_published_utc"] = pd.to_datetime(
        normalized.get("source_published_utc"),
        utc=True,
        errors="coerce",
    )

    for column in (
        "forward_price_eur_per_mwh",
        "day_ahead_price_eur_per_mwh",
        "intraday_price_eur_per_mwh",
        "imbalance_price_eur_per_mwh",
    ):
        normalized[column] = pd.to_numeric(normalized.get(column), errors="coerce")

    normalized["forward_to_day_ahead_spread_eur_per_mwh"] = (
        normalized["day_ahead_price_eur_per_mwh"] - normalized["forward_price_eur_per_mwh"]
    )
    normalized["day_ahead_to_intraday_spread_eur_per_mwh"] = (
        normalized["intraday_price_eur_per_mwh"] - normalized["day_ahead_price_eur_per_mwh"]
    )
    normalized["forward_to_day_ahead_spread_bucket"] = _spread_bucket(
        normalized["forward_to_day_ahead_spread_eur_per_mwh"]
    )
    normalized["day_ahead_to_intraday_spread_bucket"] = _spread_bucket(
        normalized["day_ahead_to_intraday_spread_eur_per_mwh"]
    )
    normalized["upstream_market_state_feed_available_flag"] = normalized[
        [
            "forward_price_eur_per_mwh",
            "day_ahead_price_eur_per_mwh",
            "intraday_price_eur_per_mwh",
            "imbalance_price_eur_per_mwh",
        ]
    ].notna().any(axis=1)
    normalized["upstream_market_state"] = _market_state_label(normalized)

    normalized = normalized.sort_values(
        ["interval_start_utc", "route_name", "source_published_utc", "source_key"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    normalized = normalized.drop_duplicates(["interval_start_utc", "route_name"], keep="last")

    keep_columns = list(_empty_upstream_market_state_frame().columns)
    for column in keep_columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized[keep_columns].sort_values(["interval_start_utc", "route_name"]).reset_index(drop=True)


def build_fact_upstream_market_state_hourly_from_price_frame(
    prices: pd.DataFrame | None,
    *,
    gb_source_provider: str = "APXMIDP",
) -> pd.DataFrame:
    if prices is None or prices.empty:
        return _empty_upstream_market_state_frame()

    price_frame = prices.copy()
    if "GB" not in price_frame.columns:
        return _empty_upstream_market_state_frame()

    price_frame.index = pd.to_datetime(price_frame.index, utc=True, errors="coerce")
    price_frame = price_frame[price_frame.index.notna()].sort_index()
    if price_frame.empty:
        return _empty_upstream_market_state_frame()

    route_frames = []
    for route_name, day_ahead_zone in _ROUTE_DAY_AHEAD_ZONE.items():
        if day_ahead_zone not in price_frame.columns:
            continue

        route_frame = pd.DataFrame(
            {
                "interval_start_utc": price_frame.index,
                "interval_end_utc": price_frame.index + pd.Timedelta(hours=1),
                "route_name": route_name,
                "source_provider": f"elexon_mid:{gb_source_provider}+entsoe_day_ahead",
                "source_family": "free_entsoe_day_ahead_plus_elexon_mid",
                "source_key": f"free_entsoe_day_ahead_plus_elexon_mid:{route_name}",
                # This first free feed is source-lineage accurate, but it does not yet
                # recover the original publication timestamps for each upstream source.
                "source_published_utc": pd.NaT,
                # Use GB MID as the source-side near-term market reference and the
                # first foreign zone day-ahead price as the external pull signal.
                "forward_price_eur_per_mwh": pd.to_numeric(price_frame["GB"], errors="coerce").to_numpy(),
                "day_ahead_price_eur_per_mwh": pd.to_numeric(price_frame[day_ahead_zone], errors="coerce").to_numpy(),
                "intraday_price_eur_per_mwh": np.nan,
                "imbalance_price_eur_per_mwh": np.nan,
            }
        )
        route_frames.append(route_frame)

    if not route_frames:
        return _empty_upstream_market_state_frame()

    fact = pd.concat(route_frames, ignore_index=True, sort=False)
    fact["forward_to_day_ahead_spread_eur_per_mwh"] = (
        fact["day_ahead_price_eur_per_mwh"] - fact["forward_price_eur_per_mwh"]
    )
    fact["day_ahead_to_intraday_spread_eur_per_mwh"] = np.nan
    fact["forward_to_day_ahead_spread_bucket"] = _spread_bucket(
        fact["forward_to_day_ahead_spread_eur_per_mwh"]
    )
    fact["day_ahead_to_intraday_spread_bucket"] = "spread_unknown"
    fact["upstream_market_state_feed_available_flag"] = (
        fact[["forward_price_eur_per_mwh", "day_ahead_price_eur_per_mwh"]].notna().all(axis=1)
    )
    fact["upstream_market_state"] = _market_state_label(fact)
    keep_columns = list(_empty_upstream_market_state_frame().columns)
    return fact[keep_columns].sort_values(["interval_start_utc", "route_name"]).reset_index(drop=True)


def build_fact_upstream_market_state_hourly(
    start_date: pd.Timestamp | str | object,
    end_date: pd.Timestamp | str | object,
    input_path: str | Path | None,
) -> pd.DataFrame:
    if not input_path:
        return _empty_upstream_market_state_frame()

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"upstream market-state input does not exist: {input_file}")

    start_ts = pd.Timestamp(start_date).tz_localize("UTC") if pd.Timestamp(start_date).tzinfo is None else pd.Timestamp(start_date).tz_convert("UTC")
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") if pd.Timestamp(end_date).tzinfo is None else pd.Timestamp(end_date).tz_convert("UTC")
    end_exclusive = end_ts + pd.Timedelta(days=1)

    normalized = normalize_upstream_market_state_input_frame(_read_tabular_input(input_file))
    if normalized.empty:
        return normalized
    normalized = normalized[
        normalized["interval_start_utc"].between(start_ts, end_exclusive, inclusive="left")
    ].copy()
    return normalized.sort_values(["interval_start_utc", "route_name"]).reset_index(drop=True)


def materialize_upstream_market_state_history(
    output_dir: str | Path,
    start_date: pd.Timestamp | str | object,
    end_date: pd.Timestamp | str | object,
    input_path: str | Path | None,
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_upstream_market_state_hourly(
        start_date=start_date,
        end_date=end_date,
        input_path=input_path,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{UPSTREAM_MARKET_STATE_TABLE}.csv", index=False)
    return {UPSTREAM_MARKET_STATE_TABLE: fact}


def write_normalized_upstream_market_state_input(
    raw_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    input_file = Path(raw_path)
    if not input_file.exists():
        raise FileNotFoundError(f"raw upstream market-state input does not exist: {input_file}")
    normalized = normalize_upstream_market_state_input_frame(_read_tabular_input(input_file))
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_file, index=False)
    return normalized
