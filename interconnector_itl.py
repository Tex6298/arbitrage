from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from curtailment_signals import _datapackage_show, _fetch_csv


INTERCONNECTOR_ITL_TABLE = "fact_interconnector_itl_hourly"
NESO_ITL_SOURCE_KEY = "neso_interconnector_itl"
NESO_ITL_SOURCE_LABEL = "NESO interconnector ITL"
NESO_SOURCE_PROVIDER = "neso"
LONDON_TZ = ZoneInfo("Europe/London")
UTC = dt.timezone.utc


@dataclass(frozen=True)
class ITLDatasetSpec:
    connector_key: str
    connector_label: str
    border_key: str
    target_zone: str
    neighbor_domain_key: str
    dataset_key: str
    dataset_id: str
    current_resource_name: str | None
    archived_resource_name: str
    parse_mode: str


ITL_DATASET_SPECS: tuple[ITLDatasetSpec, ...] = (
    ITLDatasetSpec(
        connector_key="ifa",
        connector_label="IFA",
        border_key="GB-FR",
        target_zone="FR",
        neighbor_domain_key="FR",
        dataset_key="ifa",
        dataset_id="a95d222b-e390-4ad0-af17-09de957616ed",
        current_resource_name="IFA ITL Data",
        archived_resource_name="Archived IFA DA & ID Weekly ITLs",
        parse_mode="neso_current_itl",
    ),
    ITLDatasetSpec(
        connector_key="ifa2",
        connector_label="IFA2",
        border_key="GB-FR",
        target_zone="FR",
        neighbor_domain_key="FR",
        dataset_key="ifa2",
        dataset_id="860c8221-c656-4af3-800a-281b9e500489",
        current_resource_name="IFA2 ITL Data",
        archived_resource_name="Archived IFA2 DA & ID Weekly ITLs",
        parse_mode="neso_current_itl",
    ),
    ITLDatasetSpec(
        connector_key="britned",
        connector_label="BritNed",
        border_key="GB-NL",
        target_zone="NL",
        neighbor_domain_key="NL",
        dataset_key="britned",
        dataset_id="1a9fa49a-dea1-4468-9a8c-7800db9d3ff4",
        current_resource_name=None,
        archived_resource_name="BritNed DA & ID Weekly ITLs",
        parse_mode="britned_weekly_itl",
    ),
    ITLDatasetSpec(
        connector_key="eleclink",
        connector_label="ElecLink",
        border_key="GB-FR",
        target_zone="FR",
        neighbor_domain_key="FR",
        dataset_key="eleclink",
        dataset_id="46daeb44-95a7-4032-8a22-914c896ed261",
        current_resource_name="ElecLink NTC Data",
        archived_resource_name="Archived ElecLink NTC Data",
        parse_mode="neso_current_itl",
    ),
)


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD") from exc


def _empty_itl_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "connector_key",
            "connector_label",
            "border_key",
            "target_zone",
            "neighbor_domain_key",
            "direction_key",
            "direction_label",
            "source_key",
            "source_label",
            "source_provider",
            "source_dataset_key",
            "source_dataset_id",
            "source_resource_id",
            "source_resource_name",
            "source_document_url",
            "source_table_variant",
            "source_published_utc",
            "source_published_date",
            "auction_type",
            "itl_scope",
            "target_is_proxy",
            "itl_mw",
            "restriction_reason",
            "restriction_active_flag",
            "itl_state",
        ]
    )


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _match_column(columns: Iterable[str], *tokens: str) -> str | None:
    normalized = {column: _normalize_column_name(column) for column in columns}
    for column, normalized_name in normalized.items():
        if all(token in normalized_name for token in tokens):
            return column
    return None


def _direction_label(spec: ITLDatasetSpec, direction_key: str) -> str:
    if direction_key == "gb_to_neighbor":
        return f"Great Britain to {spec.neighbor_domain_key}"
    return f"{spec.neighbor_domain_key} to Great Britain"


def _resource_publication_timestamp(resource: dict, resource_date: dt.date | None) -> pd.Timestamp:
    for key in ("metadata_modified", "created"):
        value = resource.get(key)
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(timestamp):
            return timestamp
    if resource_date is not None:
        return pd.Timestamp(resource_date, tz=UTC)
    return pd.NaT


def _extract_resource_date(resource: dict) -> dt.date | None:
    candidates = [str(resource.get("name") or ""), str(resource.get("url") or "")]
    for candidate in candidates:
        match = re.search(r"(20\d{6})", candidate)
        if not match:
            continue
        try:
            return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
    return None


def _resource_rows(spec: ITLDatasetSpec) -> list[dict]:
    metadata = _datapackage_show(spec.dataset_id)
    resources = metadata.get("result", {}).get("resources", [])
    if not isinstance(resources, list):
        raise RuntimeError(f"{spec.dataset_key} dataset returned invalid resource metadata")
    return resources


def _resource_url(resource: dict) -> str:
    url = resource.get("url")
    if isinstance(url, str) and url:
        return url
    raise RuntimeError(f"resource {resource.get('id')} has no download URL")


def _pick_current_resource(resources: list[dict], resource_name: str) -> dict | None:
    for resource in resources:
        if str(resource.get("name") or "").strip().lower() == resource_name.strip().lower():
            return resource
    return None


def _pick_archived_resources(resources: list[dict], spec: ITLDatasetSpec, start_date: dt.date, end_date: dt.date) -> list[dict]:
    lower_bound = start_date - dt.timedelta(days=7)
    upper_bound = end_date + dt.timedelta(days=7)
    selected = []
    for resource in resources:
        name = str(resource.get("name") or "")
        if spec.archived_resource_name.lower() not in name.lower():
            continue
        resource_date = _extract_resource_date(resource)
        if resource_date is None:
            selected.append(resource)
            continue
        if lower_bound <= resource_date <= upper_bound:
            selected.append(resource)
    return selected


def _parse_current_operational_period(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True)


def _parse_britned_operational_period(value: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    if pd.isna(value):
        return pd.NaT, pd.NaT
    raw_value = str(value).strip()
    match = re.fullmatch(r"(\d{8})\s+(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})", raw_value)
    if not match:
        return pd.NaT, pd.NaT
    start_day = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    start_local = pd.Timestamp(f"{start_day.isoformat()} {match.group(2)}", tz=LONDON_TZ)
    end_local = pd.Timestamp(f"{start_day.isoformat()} {match.group(3)}", tz=LONDON_TZ)
    if end_local <= start_local:
        end_local = end_local + pd.Timedelta(days=1)
    return start_local.tz_convert(UTC), end_local.tz_convert(UTC)


def _auction_rank(value: object) -> int:
    if pd.isna(value):
        return -1
    text = str(value).strip().lower()
    if text == "day ahead":
        return 0
    match = re.search(r"intraday\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return 99


def _itl_state(limit_mw: float, reason: str) -> str:
    normalized_reason = str(reason or "").strip().lower()
    if pd.isna(limit_mw) or float(limit_mw) <= 0:
        return "blocked_zero_or_negative_itl"
    if normalized_reason in {"", "no restriction"}:
        return "published_no_restriction"
    return "published_restriction"


def _build_direction_frame(
    base: pd.DataFrame,
    spec: ITLDatasetSpec,
    resource: dict,
    source_table_variant: str,
    direction_key: str,
    itl_column: str,
    reason_column: str,
    source_published_utc: pd.Series,
    auction_type: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": base["interval_start_local"].dt.date,
            "interval_start_local": base["interval_start_local"],
            "interval_end_local": base["interval_end_local"],
            "interval_start_utc": base["interval_start_utc"],
            "interval_end_utc": base["interval_end_utc"],
            "connector_key": spec.connector_key,
            "connector_label": spec.connector_label,
            "border_key": spec.border_key,
            "target_zone": spec.target_zone,
            "neighbor_domain_key": spec.neighbor_domain_key,
            "direction_key": direction_key,
            "direction_label": _direction_label(spec, direction_key),
            "source_key": NESO_ITL_SOURCE_KEY,
            "source_label": NESO_ITL_SOURCE_LABEL,
            "source_provider": NESO_SOURCE_PROVIDER,
            "source_dataset_key": spec.dataset_key,
            "source_dataset_id": spec.dataset_id,
            "source_resource_id": resource.get("id"),
            "source_resource_name": resource.get("name"),
            "source_document_url": _resource_url(resource),
            "source_table_variant": source_table_variant,
            "source_published_utc": source_published_utc,
            "source_published_date": source_published_utc.dt.date,
            "auction_type": auction_type,
            "itl_scope": "connector_da_id_submission",
            "target_is_proxy": False,
            "itl_mw": pd.to_numeric(base[itl_column], errors="coerce"),
            "restriction_reason": base[reason_column].astype("object"),
        }
    )
    out["restriction_active_flag"] = ~out["restriction_reason"].fillna("").astype(str).str.strip().str.lower().isin(
        ["", "no restriction"]
    )
    out["itl_state"] = [
        _itl_state(limit_mw, reason)
        for limit_mw, reason in zip(out["itl_mw"], out["restriction_reason"])
    ]
    return out


def _parse_neso_current_itl(resource_frame: pd.DataFrame, spec: ITLDatasetSpec, resource: dict, source_table_variant: str) -> pd.DataFrame:
    columns = tuple(resource_frame.columns)
    start_column = _match_column(columns, "operational", "period", "start")
    upload_column = _match_column(columns, "data", "upload", "time")
    auction_column = _match_column(columns, "auction", "type")
    to_gb_column = _match_column(columns, "flow", "to", "gb")
    from_gb_column = _match_column(columns, "flow", "from", "gb")
    reason_to_gb_column = _match_column(columns, "reason", "to", "gb")
    reason_from_gb_column = _match_column(columns, "reason", "from", "gb")

    if not all([start_column, to_gb_column, from_gb_column, reason_to_gb_column, reason_from_gb_column]):
        raise RuntimeError(f"{spec.dataset_key} ITL current resource has an unsupported schema")

    start_utc = _parse_current_operational_period(resource_frame[start_column])
    if upload_column is None:
        published_utc = pd.Series([_resource_publication_timestamp(resource, None)] * len(resource_frame), index=resource_frame.index)
    else:
        published_utc = pd.to_datetime(resource_frame[upload_column], errors="coerce", utc=True)
    auction_type = resource_frame[auction_column] if auction_column else pd.Series(["unknown"] * len(resource_frame), index=resource_frame.index)
    base = pd.DataFrame(
        {
            "interval_start_utc": start_utc,
            "interval_end_utc": start_utc + pd.Timedelta(hours=1),
        }
    )
    base["interval_start_local"] = base["interval_start_utc"].dt.tz_convert(LONDON_TZ)
    base["interval_end_local"] = base["interval_end_utc"].dt.tz_convert(LONDON_TZ)
    base[to_gb_column] = resource_frame[to_gb_column]
    base[from_gb_column] = resource_frame[from_gb_column]
    base[reason_to_gb_column] = resource_frame[reason_to_gb_column]
    base[reason_from_gb_column] = resource_frame[reason_from_gb_column]

    to_gb = _build_direction_frame(
        base=base,
        spec=spec,
        resource=resource,
        source_table_variant=source_table_variant,
        direction_key="neighbor_to_gb",
        itl_column=to_gb_column,
        reason_column=reason_to_gb_column,
        source_published_utc=published_utc,
        auction_type=auction_type,
    )
    from_gb = _build_direction_frame(
        base=base,
        spec=spec,
        resource=resource,
        source_table_variant=source_table_variant,
        direction_key="gb_to_neighbor",
        itl_column=from_gb_column,
        reason_column=reason_from_gb_column,
        source_published_utc=published_utc,
        auction_type=auction_type,
    )
    return pd.concat([to_gb, from_gb], ignore_index=True)


def _parse_britned_weekly_itl(resource_frame: pd.DataFrame, spec: ITLDatasetSpec, resource: dict, source_table_variant: str) -> pd.DataFrame:
    columns = tuple(resource_frame.columns)
    period_column = _match_column(columns, "operational", "date", "time")
    from_gb_column = _match_column(columns, "flow", "from", "gb")
    to_gb_column = _match_column(columns, "flow", "to", "gb")
    reason_column = _match_column(columns, "reason")
    if not all([period_column, from_gb_column, to_gb_column, reason_column]):
        raise RuntimeError("britned ITL archived resource has an unsupported schema")

    parsed_periods = resource_frame[period_column].map(_parse_britned_operational_period)
    start_utc = parsed_periods.map(lambda item: item[0])
    end_utc = parsed_periods.map(lambda item: item[1])
    published_utc = pd.Series(
        [_resource_publication_timestamp(resource, _extract_resource_date(resource))] * len(resource_frame),
        index=resource_frame.index,
        dtype="datetime64[ns, UTC]",
    )
    base = pd.DataFrame(
        {
            "interval_start_utc": start_utc,
            "interval_end_utc": end_utc,
        }
    )
    base["interval_start_local"] = pd.to_datetime(base["interval_start_utc"], utc=True, errors="coerce").dt.tz_convert(LONDON_TZ)
    base["interval_end_local"] = pd.to_datetime(base["interval_end_utc"], utc=True, errors="coerce").dt.tz_convert(LONDON_TZ)
    base[from_gb_column] = resource_frame[from_gb_column]
    base[to_gb_column] = resource_frame[to_gb_column]
    base[reason_column] = resource_frame[reason_column]
    auction_type = pd.Series(["weekly_archive"] * len(resource_frame), index=resource_frame.index)

    to_gb = _build_direction_frame(
        base=base,
        spec=spec,
        resource=resource,
        source_table_variant=source_table_variant,
        direction_key="neighbor_to_gb",
        itl_column=to_gb_column,
        reason_column=reason_column,
        source_published_utc=published_utc,
        auction_type=auction_type,
    )
    from_gb = _build_direction_frame(
        base=base,
        spec=spec,
        resource=resource,
        source_table_variant=source_table_variant,
        direction_key="gb_to_neighbor",
        itl_column=from_gb_column,
        reason_column=reason_column,
        source_published_utc=published_utc,
        auction_type=auction_type,
    )
    return pd.concat([to_gb, from_gb], ignore_index=True)


def _parse_resource_frame(resource_frame: pd.DataFrame, spec: ITLDatasetSpec, resource: dict, source_table_variant: str) -> pd.DataFrame:
    if spec.parse_mode == "neso_current_itl":
        return _parse_neso_current_itl(resource_frame, spec, resource, source_table_variant)
    if spec.parse_mode == "britned_weekly_itl":
        return _parse_britned_weekly_itl(resource_frame, spec, resource, source_table_variant)
    raise RuntimeError(f"unsupported ITL parse mode for {spec.dataset_key}: {spec.parse_mode}")


def _filter_requested_window(frame: pd.DataFrame, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if frame.empty:
        return _empty_itl_frame()
    start_local = pd.Timestamp(start_date, tz=LONDON_TZ)
    end_local = pd.Timestamp(end_date + dt.timedelta(days=1), tz=LONDON_TZ)
    start_utc = start_local.tz_convert(UTC)
    end_utc = end_local.tz_convert(UTC)
    filtered = frame[
        frame["interval_start_utc"].lt(end_utc) & frame["interval_end_utc"].gt(start_utc)
    ].copy()
    if filtered.empty:
        return _empty_itl_frame()
    return filtered


def _dedupe_itl(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_itl_frame()
    deduped = frame.copy()
    deduped["source_published_utc"] = pd.to_datetime(deduped["source_published_utc"], utc=True, errors="coerce")
    deduped["_published_sort"] = deduped["source_published_utc"].fillna(pd.Timestamp("1900-01-01", tz=UTC))
    deduped["_auction_sort"] = deduped["auction_type"].map(_auction_rank)
    deduped["_variant_sort"] = deduped["source_table_variant"].map({"archived_upload": 0, "current_datastore": 1}).fillna(0)
    deduped = deduped.sort_values(
        ["interval_start_utc", "connector_key", "direction_key", "_published_sort", "_variant_sort", "_auction_sort"],
        ascending=[True, True, True, True, True, True],
    )
    deduped = deduped.drop_duplicates(
        subset=["interval_start_utc", "connector_key", "direction_key"],
        keep="last",
    )
    return deduped[_empty_itl_frame().columns].sort_values(
        ["interval_start_utc", "connector_key", "direction_key"]
    ).reset_index(drop=True)


def _fetch_spec_itl(spec: ITLDatasetSpec, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    resources = _resource_rows(spec)
    frames = []

    if spec.current_resource_name:
        current_resource = _pick_current_resource(resources, spec.current_resource_name)
        if current_resource is not None:
            current_frame = _fetch_csv(_resource_url(current_resource))
            parsed_current = _parse_resource_frame(current_frame, spec, current_resource, "current_datastore")
            filtered_current = _filter_requested_window(parsed_current, start_date, end_date)
            if not filtered_current.empty:
                frames.append(filtered_current)

    archived_resources = _pick_archived_resources(resources, spec, start_date, end_date)
    for resource in archived_resources:
        archived_frame = _fetch_csv(_resource_url(resource))
        parsed_archived = _parse_resource_frame(archived_frame, spec, resource, "archived_upload")
        filtered_archived = _filter_requested_window(parsed_archived, start_date, end_date)
        if not filtered_archived.empty:
            frames.append(filtered_archived)

    if not frames:
        return _empty_itl_frame()
    return _dedupe_itl(pd.concat(frames, ignore_index=True))


def build_fact_interconnector_itl_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    frames = []
    for spec in ITL_DATASET_SPECS:
        frame = _fetch_spec_itl(spec, start_date=start_date, end_date=end_date)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return _empty_itl_frame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["interval_start_utc", "connector_key", "direction_key"]
    ).reset_index(drop=True)


def materialize_interconnector_itl_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_interconnector_itl_hourly(start_date=start_date, end_date=end_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{INTERCONNECTOR_ITL_TABLE}.csv", index=False)
    return {INTERCONNECTOR_ITL_TABLE: fact}
