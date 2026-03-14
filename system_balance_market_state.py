from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SYSTEM_BALANCE_MARKET_STATE_TABLE = "fact_system_balance_market_state_hourly"
ELEXON_DATASET_BASE = "https://data.elexon.co.uk/bmrs/api/v1/datasets"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc

_IMBALANCE_STRONG_THRESHOLD_MW = 1000.0
_IMBALANCE_MODERATE_THRESHOLD_MW = 250.0
_MARGIN_TIGHT_RATIO = 0.05
_MARGIN_ELEVATED_RATIO = 0.10
_MARGIN_NEUTRAL_RATIO = 0.20
_MARGIN_TIGHT_ABS_MW = 1000.0
_MARGIN_ELEVATED_ABS_MW = 2000.0
_MARGIN_NEUTRAL_ABS_MW = 4000.0


@dataclass(frozen=True)
class SystemBalanceDatasetSpec:
    dataset_key: str
    metric_column: str
    value_aliases: tuple[str, ...]


DATASET_SPECS: tuple[SystemBalanceDatasetSpec, ...] = (
    SystemBalanceDatasetSpec(
        dataset_key="IMBALNGC",
        metric_column="system_balance_imbalance_mw",
        value_aliases=(
            "imbalance",
            "imbalance_mw",
            "indicative_imbalance",
            "system_imbalance",
            "quantity",
            "value",
        ),
    ),
    SystemBalanceDatasetSpec(
        dataset_key="INDDEM",
        metric_column="system_balance_indicated_demand_mw",
        value_aliases=(
            "demand",
            "indicated_demand",
            "indicated_demand_mw",
            "national_demand",
            "demand_mw",
            "quantity",
            "value",
        ),
    ),
    SystemBalanceDatasetSpec(
        dataset_key="INDGEN",
        metric_column="system_balance_indicated_generation_mw",
        value_aliases=(
            "generation",
            "indicated_generation",
            "indicated_generation_mw",
            "national_generation",
            "generation_mw",
            "quantity",
            "value",
        ),
    ),
    SystemBalanceDatasetSpec(
        dataset_key="MELNGC",
        metric_column="system_balance_indicated_margin_mw",
        value_aliases=(
            "margin",
            "indicated_margin",
            "indicated_margin_mw",
            "system_margin",
            "margin_mw",
            "quantity",
            "value",
        ),
    ),
)


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def _empty_system_balance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "system_balance_source_provider",
            "system_balance_source_family",
            "system_balance_source_key",
            "system_balance_source_dataset_keys",
            "system_balance_source_published_utc",
            "system_balance_feed_available_flag",
            "system_balance_known_flag",
            "system_balance_active_flag",
            "system_balance_state",
            "system_balance_imbalance_mw",
            "system_balance_indicated_demand_mw",
            "system_balance_indicated_generation_mw",
            "system_balance_indicated_margin_mw",
            "system_balance_demand_minus_generation_mw",
            "system_balance_margin_ratio",
            "system_balance_imbalance_direction_bucket",
            "system_balance_margin_direction_bucket",
            "source_lineage",
        ]
    )


def _empty_dataset_metric_frame(metric_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start_utc",
            metric_column,
            f"{metric_column}_source_published_utc",
            f"{metric_column}_source_row_count",
        ]
    )


def _canonicalize_column_name(value: object) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value).strip())
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def _fetch_elexon_payload(url: str, source_name: str, api_key: str | None) -> bytes:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Ocp-Apim-Subscription-Key"] = api_key
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore").strip()
        detail = f": {body}" if body else ""
        raise RuntimeError(f"{source_name} request failed with HTTP {exc.code}{detail}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"{source_name} request failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{source_name} request timed out") from exc


def _requested_window_utc(start_date: dt.date, end_date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_utc = pd.Timestamp(start_date, tz=LONDON_TZ).tz_convert("UTC")
    end_utc = pd.Timestamp(end_date + dt.timedelta(days=1), tz=LONDON_TZ).tz_convert("UTC")
    return start_utc, end_utc


def _first_series(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return None


def _settlement_period_start_utc(settlement_date: object, settlement_period: object) -> pd.Timestamp:
    try:
        date_value = dt.date.fromisoformat(str(settlement_date))
        period_value = int(float(settlement_period))
    except Exception:
        return pd.NaT
    if period_value <= 0:
        return pd.NaT
    start_local = dt.datetime.combine(date_value, dt.time.min, tzinfo=LONDON_TZ) + dt.timedelta(
        minutes=30 * (period_value - 1)
    )
    return pd.Timestamp(start_local.astimezone(UTC))


def _coerce_interval_start_utc(frame: pd.DataFrame) -> pd.Series:
    for column_name in (
        "start_time",
        "start_datetime",
        "delivery_start_time",
        "start",
        "time_from",
    ):
        if column_name in frame.columns:
            timestamps = pd.to_datetime(frame[column_name], errors="coerce", utc=True)
            if timestamps.notna().any():
                return timestamps

    settlement_date = _first_series(frame, ("settlement_date", "date"))
    settlement_period = _first_series(frame, ("settlement_period", "period"))
    if settlement_date is not None and settlement_period is not None:
        return pd.Series(
            [
                _settlement_period_start_utc(date_value, period_value)
                for date_value, period_value in zip(settlement_date, settlement_period)
            ],
            index=frame.index,
            dtype="datetime64[ns, UTC]",
        )

    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")


def _coerce_source_published_utc(frame: pd.DataFrame) -> pd.Series:
    publish_series = _first_series(
        frame,
        (
            "publish_time",
            "publish_datetime",
            "published_at",
            "created_time",
            "inserted_time",
            "created_datetime",
        ),
    )
    if publish_series is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(publish_series, errors="coerce", utc=True)


def _pick_metric_series(frame: pd.DataFrame, spec: SystemBalanceDatasetSpec) -> pd.Series:
    for alias in spec.value_aliases:
        if alias in frame.columns:
            return pd.to_numeric(frame[alias], errors="coerce")

    excluded = {
        "dataset",
        "boundary",
        "area",
        "settlement_date",
        "settlement_period",
        "publish_time",
        "publish_datetime",
        "start_time",
        "start_datetime",
        "date",
        "from",
        "to",
    }
    numeric_candidates = []
    for column in frame.columns:
        if column in excluded or column.endswith("_utc"):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            numeric_candidates.append((column, int(numeric.notna().sum()), numeric))
    if not numeric_candidates:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    numeric_candidates.sort(key=lambda item: (-item[1], item[0]))
    return numeric_candidates[0][2]


def _filter_dataset_scope(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    for scope_column in ("boundary", "area", "national_boundary"):
        if scope_column not in filtered.columns:
            continue
        values = filtered[scope_column].astype(str).str.lower()
        preferred = values.str.contains("national|gb|great_britain|system", regex=True, na=False)
        if preferred.any():
            filtered = filtered[preferred].copy()
    return filtered


def normalize_system_balance_dataset_frame(
    raw_frame: pd.DataFrame,
    spec: SystemBalanceDatasetSpec,
    *,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    if raw_frame is None or raw_frame.empty:
        return _empty_dataset_metric_frame(spec.metric_column)

    frame = raw_frame.copy()
    frame.columns = [_canonicalize_column_name(column) for column in frame.columns]
    frame = _filter_dataset_scope(frame)
    frame["interval_start_utc"] = _coerce_interval_start_utc(frame)
    frame["source_published_utc"] = _coerce_source_published_utc(frame)
    frame[spec.metric_column] = _pick_metric_series(frame, spec)
    frame = frame.dropna(subset=["interval_start_utc"]).copy()
    if frame.empty:
        return _empty_dataset_metric_frame(spec.metric_column)

    window_start, window_end = _requested_window_utc(start_date, end_date)
    frame = frame[
        frame["interval_start_utc"].ge(window_start) & frame["interval_start_utc"].lt(window_end)
    ].copy()
    if frame.empty:
        return _empty_dataset_metric_frame(spec.metric_column)

    frame["interval_start_utc"] = frame["interval_start_utc"].dt.floor("30min")
    frame = frame.sort_values(
        ["interval_start_utc", "source_published_utc"],
        ascending=[True, True],
        na_position="last",
    )
    frame = frame.drop_duplicates(["interval_start_utc"], keep="last")
    frame["hour_start_utc"] = frame["interval_start_utc"].dt.floor("h")

    hourly = frame.groupby("hour_start_utc", as_index=False).agg(
        **{
            spec.metric_column: (spec.metric_column, "mean"),
            f"{spec.metric_column}_source_published_utc": ("source_published_utc", "max"),
            f"{spec.metric_column}_source_row_count": (spec.metric_column, "count"),
        }
    )
    return hourly.rename(columns={"hour_start_utc": "interval_start_utc"}).sort_values("interval_start_utc").reset_index(
        drop=True
    )


def _load_dataset_frame(
    spec: SystemBalanceDatasetSpec,
    *,
    start_date: dt.date,
    end_date: dt.date,
    api_key: str | None,
) -> pd.DataFrame:
    window_start, window_end = _requested_window_utc(start_date, end_date)
    frames = []
    chunk_start = window_start
    while chunk_start < window_end:
        chunk_end = min(chunk_start + pd.Timedelta(days=1), window_end)
        params = {
            "publishDateTimeFrom": chunk_start.strftime("%Y-%m-%dT%H:%MZ"),
            "publishDateTimeTo": chunk_end.strftime("%Y-%m-%dT%H:%MZ"),
            "format": "json",
        }
        url = f"{ELEXON_DATASET_BASE}/{spec.dataset_key}?{urllib.parse.urlencode(params)}"
        payload = _fetch_elexon_payload(url, f"Elexon {spec.dataset_key}", api_key)
        try:
            body = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Elexon {spec.dataset_key} returned invalid JSON") from exc

        rows = body.get("data") if isinstance(body, dict) else body
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            rows = [rows]
        frames.append(pd.DataFrame(rows))
        chunk_start = chunk_end
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _imbalance_bucket(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.where(
            numeric.isna(),
            "imbalance_unknown",
            np.where(
                numeric.le(-_IMBALANCE_STRONG_THRESHOLD_MW),
                "imbalance_strong_negative",
                np.where(
                    numeric.le(-_IMBALANCE_MODERATE_THRESHOLD_MW),
                    "imbalance_negative",
                    np.where(
                        numeric.lt(_IMBALANCE_MODERATE_THRESHOLD_MW),
                        "imbalance_neutral",
                        np.where(
                            numeric.lt(_IMBALANCE_STRONG_THRESHOLD_MW),
                            "imbalance_positive",
                            "imbalance_strong_positive",
                        ),
                    ),
                ),
            ),
        ),
        index=values.index,
    )


def _margin_bucket(margin_ratio: pd.Series, margin_mw: pd.Series) -> pd.Series:
    ratio = pd.to_numeric(margin_ratio, errors="coerce")
    margin = pd.to_numeric(margin_mw, errors="coerce")
    return pd.Series(
        np.where(
            ratio.notna(),
            np.where(
                ratio.le(_MARGIN_TIGHT_RATIO),
                "margin_very_tight",
                np.where(
                    ratio.le(_MARGIN_ELEVATED_RATIO),
                    "margin_tight",
                    np.where(ratio.le(_MARGIN_NEUTRAL_RATIO), "margin_neutral", "margin_loose"),
                ),
            ),
            np.where(
                margin.isna(),
                "margin_unknown",
                np.where(
                    margin.le(_MARGIN_TIGHT_ABS_MW),
                    "margin_very_tight",
                    np.where(
                        margin.le(_MARGIN_ELEVATED_ABS_MW),
                        "margin_tight",
                        np.where(margin.le(_MARGIN_NEUTRAL_ABS_MW), "margin_neutral", "margin_loose"),
                    ),
                ),
            ),
        ),
        index=margin.index,
    )


def _system_balance_state(frame: pd.DataFrame) -> pd.Series:
    available = frame["system_balance_feed_available_flag"]
    imbalance_bucket = frame["system_balance_imbalance_direction_bucket"]
    margin_bucket = frame["system_balance_margin_direction_bucket"]
    stress = pd.Series("no_public_system_balance", index=frame.index, dtype="object")
    stress.loc[available] = "balanced_or_loose"
    tight_margin = margin_bucket.isin({"margin_very_tight", "margin_tight"})
    strong_imbalance = imbalance_bucket.isin({"imbalance_strong_negative", "imbalance_strong_positive"})
    moderate_imbalance = imbalance_bucket.isin({"imbalance_negative", "imbalance_positive"})
    stress.loc[available & tight_margin & strong_imbalance] = "tight_margin_and_active_imbalance"
    stress.loc[available & tight_margin & ~strong_imbalance] = "tight_margin"
    stress.loc[available & ~tight_margin & strong_imbalance] = "active_imbalance"
    stress.loc[
        available
        & ~tight_margin
        & ~strong_imbalance
        & moderate_imbalance
    ] = "moderate_imbalance"
    return stress


def build_fact_system_balance_market_state_hourly(
    start_date: dt.date,
    end_date: dt.date,
    api_key: str | None = None,
    dataset_frames: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    metric_frames = []
    for spec in DATASET_SPECS:
        raw_frame = dataset_frames.get(spec.dataset_key) if dataset_frames is not None else None
        if raw_frame is None:
            raw_frame = _load_dataset_frame(spec, start_date=start_date, end_date=end_date, api_key=api_key)
        metric_frames.append(
            normalize_system_balance_dataset_frame(
                raw_frame,
                spec,
                start_date=start_date,
                end_date=end_date,
            )
        )

    if not metric_frames:
        return _empty_system_balance_frame()

    combined = None
    for metric_frame in metric_frames:
        if combined is None:
            combined = metric_frame
        else:
            combined = combined.merge(metric_frame, on="interval_start_utc", how="outer")
    if combined is None or combined.empty:
        return _empty_system_balance_frame()

    combined["interval_start_utc"] = pd.to_datetime(combined["interval_start_utc"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["interval_start_utc"]).copy()
    combined["interval_end_utc"] = combined["interval_start_utc"] + pd.Timedelta(hours=1)
    combined["interval_start_local"] = combined["interval_start_utc"].dt.tz_convert(LONDON_TZ)
    combined["interval_end_local"] = combined["interval_end_utc"].dt.tz_convert(LONDON_TZ)
    combined["date"] = combined["interval_start_local"].dt.date

    combined["system_balance_feed_available_flag"] = combined[
        [
            "system_balance_imbalance_mw",
            "system_balance_indicated_demand_mw",
            "system_balance_indicated_generation_mw",
            "system_balance_indicated_margin_mw",
        ]
    ].notna().any(axis=1)
    combined["system_balance_demand_minus_generation_mw"] = (
        pd.to_numeric(combined["system_balance_indicated_demand_mw"], errors="coerce")
        - pd.to_numeric(combined["system_balance_indicated_generation_mw"], errors="coerce")
    )
    demand = pd.to_numeric(combined["system_balance_indicated_demand_mw"], errors="coerce")
    margin = pd.to_numeric(combined["system_balance_indicated_margin_mw"], errors="coerce")
    combined["system_balance_margin_ratio"] = np.where(demand.abs().gt(0), margin / demand.abs(), np.nan)
    combined["system_balance_imbalance_direction_bucket"] = _imbalance_bucket(combined["system_balance_imbalance_mw"])
    combined["system_balance_margin_direction_bucket"] = _margin_bucket(
        combined["system_balance_margin_ratio"],
        combined["system_balance_indicated_margin_mw"],
    )

    published_columns = [
        f"{spec.metric_column}_source_published_utc"
        for spec in DATASET_SPECS
        if f"{spec.metric_column}_source_published_utc" in combined.columns
    ]
    if published_columns:
        for column in published_columns:
            combined[column] = pd.to_datetime(combined[column], utc=True, errors="coerce")
        combined["system_balance_source_published_utc"] = combined[published_columns].apply(
            lambda row: row.dropna().max() if row.notna().any() else pd.NaT,
            axis=1,
        )
    else:
        combined["system_balance_source_published_utc"] = pd.NaT
    combined["system_balance_source_dataset_keys"] = combined.apply(
        lambda row: "|".join(
            spec.dataset_key
            for spec in DATASET_SPECS
            if pd.notna(row.get(spec.metric_column))
        ),
        axis=1,
    )
    combined["system_balance_source_provider"] = "elexon"
    combined["system_balance_source_family"] = "public_system_balance"
    combined["system_balance_source_key"] = combined["system_balance_source_dataset_keys"].where(
        combined["system_balance_source_dataset_keys"].astype(str).str.len().gt(0),
        "public_system_balance",
    )
    combined["system_balance_known_flag"] = (
        pd.to_datetime(combined["system_balance_source_published_utc"], utc=True, errors="coerce")
        .le(combined["interval_start_utc"])
        .fillna(False)
    )
    combined["system_balance_active_flag"] = combined["system_balance_feed_available_flag"].astype(bool)
    combined["system_balance_state"] = _system_balance_state(combined)
    combined["source_lineage"] = "elexon:IMBALNGC|INDDEM|INDGEN|MELNGC"

    keep_columns = list(_empty_system_balance_frame().columns)
    for column in keep_columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined[keep_columns].sort_values("interval_start_utc").reset_index(drop=True)


def materialize_system_balance_market_state_history(
    output_dir: str | Path,
    start_date: dt.date,
    end_date: dt.date,
    api_key: str | None = None,
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_system_balance_market_state_hourly(
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{SYSTEM_BALANCE_MARKET_STATE_TABLE}.csv", index=False)
    return {SYSTEM_BALANCE_MARKET_STATE_TABLE: fact}
