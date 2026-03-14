from __future__ import annotations

import datetime as dt
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from asset_mapping import cluster_frame


NESO_BASE = "https://api.neso.energy"
NESO_ACTION_BASE = f"{NESO_BASE}/api/3/action"
NESO_CONSTRAINT_BREAKDOWN_DATASET_ID = "fb56b46e-cef3-4eb8-9294-0ca19769b7eb"
NESO_CONSTRAINT_BREAKDOWN_RESOURCES: Dict[str, str] = {
    "2025-2026": "6afe1c2b-6d70-4e76-8e74-0952b0a2beab",
    "2024-2025": "748557ef-2bb3-41c0-8181-5f1a148c1ff4",
}
NESO_METERED_WIND_RESOURCES: Dict[str, str] = {
    "2018-2019": "bf03c648-98d8-40f3-b5d9-e174cb2c1f81",
    "2019-2020": "c36868d9-9d43-47ce-ac3e-de69f0bb6222",
    "2020-2021": "043d5423-f877-4d28-864e-210dc68711d4",
    "2021-2022": "e5df03f3-a25b-4df4-a95c-8a709815924b",
    "2022-2023": "f732e9bb-b573-46a7-8767-3affbbb29b45",
    "2023-2024": "c47155bc-71b8-4b0a-aba3-0e2d1295daea",
    "2024-2025": "3ac8c742-d601-4b6c-9779-e69cea94e134",
    "2025-2026": "7622b040-977a-45a6-924e-f158df6c29f0",
}
CONSTRAINT_QA_TARGET_DEFINITION = "wind_constraints_positive_only_v1"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json,text/csv;q=0.9,*/*;q=0.8"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore").strip()
        detail = f": {body}" if body else ""
        raise RuntimeError(f"NESO request failed with HTTP {exc.code}{detail}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"NESO request failed: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("NESO request timed out") from exc


def _fetch_json(url: str) -> dict:
    payload = _fetch_bytes(url)
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("NESO returned invalid JSON") from exc


def _fetch_csv(url: str) -> pd.DataFrame:
    payload = _fetch_bytes(url)
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(payload.decode(encoding)))
        except UnicodeDecodeError:
            continue
    raise RuntimeError("NESO returned CSV data in an unsupported encoding")


def _datapackage_show(dataset_id: str) -> dict:
    url = f"{NESO_ACTION_BASE}/datapackage_show?id={urllib.parse.quote(dataset_id)}"
    return _fetch_json(url)


def _resolve_resource_download_url(dataset_id: str, resource_id: str) -> str:
    metadata = _datapackage_show(dataset_id)
    resources = metadata.get("result", {}).get("resources", [])
    for resource in resources:
        if resource.get("id") != resource_id:
            continue
        path = resource.get("path") or ""
        if isinstance(path, str) and path.startswith("http"):
            return path
        return f"{NESO_BASE}/dataset/{dataset_id}/resource/{resource_id}/download/{path or 'data.csv'}"
    raise RuntimeError(f"resource {resource_id} not found in dataset {dataset_id}")


def _datastore_search_sql(resource_id: str, sql: str) -> pd.DataFrame:
    params = urllib.parse.urlencode({"sql": sql})
    url = f"{NESO_ACTION_BASE}/datastore_search_sql?{params}"
    payload = _fetch_json(url)
    rows = payload.get("result", {}).get("records", [])
    return pd.DataFrame.from_records(rows)


def _settlement_period_start(settlement_date: dt.date, settlement_period: int) -> dt.datetime:
    midnight_local = dt.datetime.combine(settlement_date, dt.time.min, tzinfo=LONDON_TZ)
    return midnight_local + dt.timedelta(minutes=30 * (settlement_period - 1))


def _scheme_year_label(value: dt.date) -> str:
    start_year = value.year if value.month >= 4 else value.year - 1
    return f"{start_year}-{start_year + 1}"


def _scheme_year_labels_for_range(start_date: dt.date, end_date: dt.date) -> Tuple[str, ...]:
    labels = []
    cursor = start_date
    while cursor <= end_date:
        label = _scheme_year_label(cursor)
        if not labels or labels[-1] != label:
            labels.append(label)
        cursor += dt.timedelta(days=1)
    return tuple(labels)


def _match_column(columns: Tuple[str, ...], *tokens: str) -> str | None:
    normalized = {column: _normalize_column_name(column) for column in columns}
    for column, normalized_name in normalized.items():
        if all(token in normalized_name for token in tokens):
            return column
    return None


def _constraint_column_map(columns: Tuple[str, ...]) -> Dict[str, str]:
    categories = {
        "reducing_largest_loss": ("largest", "loss"),
        "increasing_system_inertia": ("system", "inertia"),
        "voltage_constraints": ("voltage",),
        "thermal_constraints": ("thermal",),
    }
    measures = {
        "volume_mwh": ("volume",),
        "cost_gbp": ("cost",),
    }

    column_map: Dict[str, str] = {}
    for category_key, category_tokens in categories.items():
        for measure_key, measure_tokens in measures.items():
            matched = _match_column(columns, *(category_tokens + measure_tokens))
            if matched:
                column_map[f"{category_key}_{measure_key}"] = matched
    return column_map


def add_constraint_qa_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["qa_target_definition"] = CONSTRAINT_QA_TARGET_DEFINITION

    source_to_target = {
        "voltage_constraints_volume_mwh": "qa_wind_voltage_positive_mwh",
        "thermal_constraints_volume_mwh": "qa_wind_thermal_positive_mwh",
        "increasing_system_inertia_volume_mwh": "qa_inertia_positive_mwh",
        "reducing_largest_loss_volume_mwh": "qa_largest_loss_positive_mwh",
    }
    for source_column, target_column in source_to_target.items():
        if source_column not in out.columns:
            out[target_column] = np.nan
            continue
        out[target_column] = pd.to_numeric(out[source_column], errors="coerce").clip(lower=0.0)

    out["qa_wind_relevant_positive_mwh"] = out[
        ["qa_wind_voltage_positive_mwh", "qa_wind_thermal_positive_mwh"]
    ].sum(axis=1, min_count=2)
    return out


def fetch_constraint_daily(year_label: str) -> pd.DataFrame:
    resource_id = NESO_CONSTRAINT_BREAKDOWN_RESOURCES.get(year_label)
    if not resource_id:
        raise RuntimeError(
            f"unsupported constraint year '{year_label}'. Supported years: {', '.join(sorted(NESO_CONSTRAINT_BREAKDOWN_RESOURCES))}"
        )

    url = (
        f"{NESO_BASE}/dataset/{NESO_CONSTRAINT_BREAKDOWN_DATASET_ID}/resource/{resource_id}/download/"
        f"constraint-breakdown-{year_label}.csv"
    )
    raw = _fetch_csv(url)
    raw.rename(columns={column: column.strip() for column in raw.columns}, inplace=True)

    date_column = next((column for column in raw.columns if _normalize_column_name(column) == "date"), None)
    if not date_column:
        raise RuntimeError("constraint breakdown payload has no Date column")

    column_map = _constraint_column_map(tuple(raw.columns))
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_column], errors="coerce").dt.date,
            "source_key": "constraint_breakdown",
            "source_label": "NESO constraint breakdown",
            "source_year_label": year_label,
            "source_dataset_id": NESO_CONSTRAINT_BREAKDOWN_DATASET_ID,
            "source_resource_id": resource_id,
            "target_is_proxy": False,
        }
    )

    for canonical_name in (
        "reducing_largest_loss_volume_mwh",
        "reducing_largest_loss_cost_gbp",
        "increasing_system_inertia_volume_mwh",
        "increasing_system_inertia_cost_gbp",
        "voltage_constraints_volume_mwh",
        "voltage_constraints_cost_gbp",
        "thermal_constraints_volume_mwh",
        "thermal_constraints_cost_gbp",
    ):
        raw_column = column_map.get(canonical_name)
        out[canonical_name] = pd.to_numeric(raw[raw_column], errors="coerce") if raw_column else np.nan

    volume_columns = [column for column in out.columns if column.endswith("_volume_mwh")]
    cost_columns = [column for column in out.columns if column.endswith("_cost_gbp")]
    out["total_curtailment_mwh"] = out[volume_columns].sum(axis=1, min_count=1)
    out["total_curtailment_cost_gbp"] = out[cost_columns].sum(axis=1, min_count=1)
    out = add_constraint_qa_columns(out)

    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_wind_split_half_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    raw_frames = []
    for scheme_year in _scheme_year_labels_for_range(start_date, end_date):
        resource_id = NESO_METERED_WIND_RESOURCES.get(scheme_year)
        if not resource_id:
            raise RuntimeError(
                f"unsupported wind split year '{scheme_year}'. Supported years: {', '.join(sorted(NESO_METERED_WIND_RESOURCES))}"
            )

        sql = f"""
        SELECT "Sett_Date","Sett_Period",
               "Scottish Wind Output","England/Wales Wind Output","Total"
          FROM "{resource_id}"
         WHERE "Sett_Date" BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
         ORDER BY "Sett_Date","Sett_Period"
        """
        frame = _datastore_search_sql(resource_id, sql)
        if frame.empty:
            continue
        frame["source_scheme_year"] = scheme_year
        frame["source_resource_id"] = resource_id
        raw_frames.append(frame)

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if raw.empty:
        raise RuntimeError(f"no NESO wind split rows returned for {start_date} to {end_date}")

    settlement_dates = pd.to_datetime(raw["Sett_Date"], errors="coerce").dt.date
    settlement_periods = pd.to_numeric(raw["Sett_Period"], errors="coerce").astype("Int64")
    local_starts = [
        _settlement_period_start(settlement_date, int(settlement_period))
        if pd.notna(settlement_date) and pd.notna(settlement_period)
        else pd.NaT
        for settlement_date, settlement_period in zip(settlement_dates, settlement_periods)
    ]
    interval_start_local = pd.Series(pd.to_datetime(local_starts), index=raw.index)
    interval_end_local = interval_start_local + pd.Timedelta(minutes=30)

    out = pd.DataFrame(
        {
            "date": settlement_dates,
            "settlement_period": settlement_periods.astype("Int64"),
            "interval_start_local": interval_start_local,
            "interval_end_local": interval_end_local,
            "interval_start_utc": interval_start_local.dt.tz_convert("UTC"),
            "interval_end_utc": interval_end_local.dt.tz_convert("UTC"),
            "scotland_wind_mw": pd.to_numeric(raw["Scottish Wind Output"], errors="coerce"),
            "england_wales_wind_mw": pd.to_numeric(raw["England/Wales Wind Output"], errors="coerce"),
            "gb_total_wind_mw": pd.to_numeric(raw["Total"], errors="coerce"),
            "source_key": "wind_split",
            "source_label": "NESO metered wind split",
            "source_scheme_year": raw["source_scheme_year"],
            "source_resource_id": raw["source_resource_id"],
            "target_is_proxy": False,
        }
    )
    out = out.dropna(subset=["date", "settlement_period", "interval_start_utc"]).sort_values("interval_start_utc").reset_index(drop=True)
    return out


def build_regional_curtailment_hourly_proxy(
    fact_constraint_daily: pd.DataFrame,
    fact_wind_split_half_hourly: pd.DataFrame,
) -> pd.DataFrame:
    if fact_constraint_daily.empty:
        raise RuntimeError("fact_constraint_daily is empty")
    if fact_wind_split_half_hourly.empty:
        raise RuntimeError("fact_wind_split_half_hourly is empty")

    constraints = fact_constraint_daily.copy()
    constraints["date"] = pd.to_datetime(constraints["date"], errors="coerce").dt.date
    constraints = constraints.dropna(subset=["date", "total_curtailment_mwh"])

    wind = fact_wind_split_half_hourly.copy()
    wind["date"] = pd.to_datetime(wind["date"], errors="coerce").dt.date

    long = pd.concat(
        [
            wind[
                [
                    "date",
                    "interval_start_local",
                    "interval_start_utc",
                    "scotland_wind_mw",
                ]
            ].rename(columns={"scotland_wind_mw": "wind_mw"}).assign(parent_region="Scotland"),
            wind[
                [
                    "date",
                    "interval_start_local",
                    "interval_start_utc",
                    "england_wales_wind_mw",
                ]
            ].rename(columns={"england_wales_wind_mw": "wind_mw"}).assign(parent_region="England/Wales"),
        ],
        ignore_index=True,
    )
    long["wind_mw"] = pd.to_numeric(long["wind_mw"], errors="coerce").fillna(0.0)
    long["hour_start_utc"] = long["interval_start_utc"].dt.floor("h")
    long["hour_start_local"] = long["interval_start_local"].dt.floor("h")

    daily_region = (
        long.groupby(["date", "parent_region"], as_index=False)
        .agg(region_daily_wind_weight_mw=("wind_mw", "sum"))
    )
    daily_total = (
        daily_region.groupby("date", as_index=False)
        .agg(gb_daily_wind_weight_mw=("region_daily_wind_weight_mw", "sum"))
    )
    daily_region = daily_region.merge(daily_total, on="date", how="left")
    daily_region["regional_daily_wind_share"] = np.where(
        daily_region["gb_daily_wind_weight_mw"] > 0,
        daily_region["region_daily_wind_weight_mw"] / daily_region["gb_daily_wind_weight_mw"],
        0.5,
    )

    daily_region = daily_region.merge(
        constraints[
            [
                "date",
                "source_year_label",
                "total_curtailment_mwh",
                "total_curtailment_cost_gbp",
            ]
        ],
        on="date",
        how="left",
    )
    daily_region["regional_daily_alloc_mwh"] = daily_region["total_curtailment_mwh"] * daily_region["regional_daily_wind_share"]
    daily_region["regional_daily_alloc_cost_gbp"] = (
        daily_region["total_curtailment_cost_gbp"] * daily_region["regional_daily_wind_share"]
    )

    hourly_region = (
        long.groupby(["date", "parent_region", "hour_start_local", "hour_start_utc"], as_index=False)
        .agg(region_hourly_wind_weight_mw=("wind_mw", "sum"))
    )
    hourly_region = hourly_region.merge(
        daily_region[
            [
                "date",
                "parent_region",
                "source_year_label",
                "total_curtailment_mwh",
                "regional_daily_alloc_mwh",
                "regional_daily_alloc_cost_gbp",
                "regional_daily_wind_share",
                "region_daily_wind_weight_mw",
            ]
        ],
        on=["date", "parent_region"],
        how="left",
    )
    hourly_region["intraday_wind_share"] = np.where(
        hourly_region["region_daily_wind_weight_mw"] > 0,
        hourly_region["region_hourly_wind_weight_mw"] / hourly_region["region_daily_wind_weight_mw"],
        0.0,
    )
    hourly_region["hourly_curtailment_proxy_mwh"] = (
        hourly_region["regional_daily_alloc_mwh"] * hourly_region["intraday_wind_share"]
    )
    hourly_region["hourly_curtailment_proxy_cost_gbp"] = (
        hourly_region["regional_daily_alloc_cost_gbp"] * hourly_region["intraday_wind_share"]
    )

    parent_region_rows = hourly_region.rename(
        columns={
            "hour_start_local": "interval_start_local",
            "hour_start_utc": "interval_start_utc",
        }
    ).assign(
        scope_type="parent_region",
        scope_key=lambda frame: frame["parent_region"],
        scope_label=lambda frame: frame["parent_region"],
        cluster_capacity_share=np.nan,
        allocation_method="gb_daily_curtailment * regional_daily_wind_share * intraday_wind_share",
        source_key="regional_curtailment_hourly_proxy",
        source_label="Derived regional hourly curtailment proxy",
        target_is_proxy=True,
    )

    cluster_weights = cluster_frame()[
        [
            "cluster_key",
            "cluster_label",
            "parent_region",
            "approx_capacity_mw",
        ]
    ].copy()
    cluster_weights["parent_region_capacity_mw"] = cluster_weights.groupby("parent_region")["approx_capacity_mw"].transform("sum")
    cluster_weights["cluster_capacity_share"] = np.where(
        cluster_weights["parent_region_capacity_mw"] > 0,
        cluster_weights["approx_capacity_mw"] / cluster_weights["parent_region_capacity_mw"],
        np.nan,
    )

    cluster_rows = hourly_region.merge(
        cluster_weights[
            [
                "cluster_key",
                "cluster_label",
                "parent_region",
                "cluster_capacity_share",
            ]
        ],
        on="parent_region",
        how="inner",
    )
    cluster_rows = cluster_rows.rename(
        columns={
            "hour_start_local": "interval_start_local",
            "hour_start_utc": "interval_start_utc",
            "cluster_key": "scope_key",
            "cluster_label": "scope_label",
        }
    )
    cluster_rows["hourly_curtailment_proxy_mwh"] = (
        cluster_rows["hourly_curtailment_proxy_mwh"] * cluster_rows["cluster_capacity_share"]
    )
    cluster_rows["hourly_curtailment_proxy_cost_gbp"] = (
        cluster_rows["hourly_curtailment_proxy_cost_gbp"] * cluster_rows["cluster_capacity_share"]
    )
    cluster_rows = cluster_rows.assign(
        scope_type="cluster",
        allocation_method="gb_daily_curtailment * regional_daily_wind_share * intraday_wind_share * cluster_capacity_share",
        source_key="regional_curtailment_hourly_proxy",
        source_label="Derived regional hourly curtailment proxy",
        target_is_proxy=True,
    )

    out = pd.concat([parent_region_rows, cluster_rows], ignore_index=True, sort=False)
    out["interval_end_local"] = out["interval_start_local"] + pd.Timedelta(hours=1)
    out["interval_end_utc"] = out["interval_start_utc"] + pd.Timedelta(hours=1)

    keep_columns = [
        "date",
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "scope_type",
        "scope_key",
        "scope_label",
        "parent_region",
        "source_key",
        "source_label",
        "source_year_label",
        "target_is_proxy",
        "total_curtailment_mwh",
        "regional_daily_alloc_mwh",
        "regional_daily_alloc_cost_gbp",
        "regional_daily_wind_share",
        "region_daily_wind_weight_mw",
        "region_hourly_wind_weight_mw",
        "intraday_wind_share",
        "cluster_capacity_share",
        "hourly_curtailment_proxy_mwh",
        "hourly_curtailment_proxy_cost_gbp",
        "allocation_method",
    ]
    return out[keep_columns].sort_values(["scope_type", "scope_key", "interval_start_utc"]).reset_index(drop=True)


def materialize_curtailed_history(
    year_label: str,
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    fact_constraint_daily = fetch_constraint_daily(year_label)
    fact_constraint_daily = fact_constraint_daily[
        (fact_constraint_daily["date"] >= start_date) & (fact_constraint_daily["date"] <= end_date)
    ].reset_index(drop=True)
    if fact_constraint_daily.empty:
        raise RuntimeError(f"no constraint rows returned for {start_date} to {end_date} in {year_label}")

    fact_wind_split_half_hourly = fetch_wind_split_half_hourly(start_date, end_date)
    fact_regional_curtailment_hourly_proxy = build_regional_curtailment_hourly_proxy(
        fact_constraint_daily,
        fact_wind_split_half_hourly,
    )

    frames = {
        "fact_constraint_daily": fact_constraint_daily,
        "fact_wind_split_half_hourly": fact_wind_split_half_hourly,
        "fact_regional_curtailment_hourly_proxy": fact_regional_curtailment_hourly_proxy,
    }

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)

    return frames
