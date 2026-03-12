from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE = "fact_france_connector_reviewed_period"

DEFAULT_REVIEWED_SOURCE_PROVIDER = "public_reviewed_doc"
DEFAULT_REVIEW_STATE = "accepted_reviewed_tier"
DEFAULT_REVIEWED_EVIDENCE_TIER = "reviewed_public_doc_period"
DEFAULT_CAPACITY_POLICY_ACTION = "allow_reviewed_public_period"
DEFAULT_PERIOD_TIMEZONE = "UTC"

_CONNECTOR_METADATA: Dict[str, dict[str, object]] = {
    "ifa": {"connector_label": "IFA", "nominal_capacity_mw": 2000.0},
    "ifa2": {"connector_label": "IFA2", "nominal_capacity_mw": 1000.0},
    "eleclink": {"connector_label": "ElecLink", "nominal_capacity_mw": 1000.0},
}

_SOURCE_CATALOG: Dict[str, dict[str, str]] = {
    "eleclink_planned_outage_programme": {
        "source_family": "eleclink_public_doc",
        "source_label": "ElecLink planned outage programme",
        "source_url": "https://www.eleclink.co.uk/publications/planned-outage-programme",
    },
    "eleclink_capacity_split": {
        "source_family": "eleclink_public_doc",
        "source_label": "ElecLink capacity split document",
        "source_url": "https://www.eleclink.co.uk/customers/document-library",
    },
    "eleclink_ntc_restriction": {
        "source_family": "eleclink_public_doc",
        "source_label": "ElecLink NTC restriction statement",
        "source_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
    },
    "jao_ifa_notice": {
        "source_family": "jao_public_notice",
        "source_label": "JAO IFA notice",
        "source_url": "https://www.jao.eu/news",
    },
    "jao_ifa2_notice": {
        "source_family": "jao_public_notice",
        "source_label": "JAO IFA2 notice",
        "source_url": "https://www.jao.eu/news",
    },
    "jao_frgb_notice_generic": {
        "source_family": "jao_public_notice",
        "source_label": "JAO FR-GB notice",
        "source_url": "https://www.jao.eu/news",
    },
}


def _empty_reviewed_period_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "connector_key",
            "connector_label",
            "direction_key",
            "reviewed_scope",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_publication_state",
            "period_start_utc",
            "period_end_utc",
            "period_timezone",
            "connector_nominal_capacity_mw",
            "reviewed_capacity_limit_mw",
            "reviewed_available_capacity_mw",
            "reviewed_unavailable_capacity_mw",
            "source_provider",
            "source_family",
            "source_key",
            "source_label",
            "source_document_title",
            "source_document_url",
            "source_reference",
            "source_published_date",
            "review_note",
            "target_is_proxy",
        ]
    )


def _empty_reviewed_hourly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start_utc",
            "connector_key",
            "direction_key",
            "reviewed_publication_state",
            "reviewed_capacity_limit_mw",
            "reviewed_available_capacity_mw",
            "reviewed_unavailable_capacity_mw",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "source_provider",
            "source_family",
            "source_key",
            "source_label",
            "source_document_title",
            "source_document_url",
            "source_reference",
            "source_published_date",
            "review_note",
            "reviewed_source_count",
        ]
    )


def _normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in renamed.columns
    ]
    return renamed


def _read_input_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"France reviewed-period input does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)
    raise ValueError("France reviewed-period input must be .csv or .json")


def _first_series(frame: pd.DataFrame, names: list[str], default: object = pd.NA) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def _parse_timestamp_series(
    start_values: pd.Series,
    time_values: pd.Series | None = None,
    timezone_values: pd.Series | None = None,
) -> pd.Series:
    timezone_series = timezone_values if timezone_values is not None else pd.Series(
        [DEFAULT_PERIOD_TIMEZONE] * len(start_values),
        index=start_values.index,
        dtype="object",
    )
    if time_values is None:
        time_series = pd.Series(["00:00"] * len(start_values), index=start_values.index, dtype="object")
    else:
        time_series = time_values.fillna("00:00").astype(str)

    parsed_values: list[pd.Timestamp] = []
    for index, value in start_values.items():
        if pd.isna(value):
            parsed_values.append(pd.NaT)
            continue
        timezone_name = timezone_series.loc[index]
        timezone_name = DEFAULT_PERIOD_TIMEZONE if pd.isna(timezone_name) or str(timezone_name).strip() == "" else str(timezone_name)
        raw_value = str(value).strip()
        try:
            parsed = pd.Timestamp(raw_value)
        except Exception:
            parsed_values.append(pd.NaT)
            continue
        if parsed.tzinfo is not None:
            parsed_values.append(parsed.tz_convert("UTC"))
            continue
        if len(raw_value) <= 10:
            combined = f"{raw_value} {str(time_series.loc[index]).strip()}"
            try:
                parsed = pd.Timestamp(combined)
            except Exception:
                parsed_values.append(pd.NaT)
                continue
        try:
            localized = parsed.tz_localize(timezone_name)
        except Exception:
            parsed_values.append(pd.NaT)
            continue
        parsed_values.append(localized.tz_convert("UTC"))
    return pd.Series(parsed_values, index=start_values.index, dtype="datetime64[ns, UTC]")


def _normalize_direction(direction: object) -> str:
    value = str(direction or "").strip().lower()
    if value in {"", "gb_to_neighbor", "gb->fr", "gb_to_fr", "export", "fr_import"}:
        return "gb_to_neighbor"
    if value in {"neighbor_to_gb", "fr_to_gb", "fr->gb", "import", "fr_export"}:
        return "neighbor_to_gb"
    if value in {"both", "bi", "bidirectional"}:
        return "both"
    raise ValueError(f"unsupported France reviewed-period direction: {direction}")


def _derive_publication_state(
    explicit_state: object,
    capacity_limit_mw: object,
    nominal_capacity_mw: object,
) -> str:
    value = str(explicit_state or "").strip().lower()
    if value in {"outage", "blocked", "unavailable", "closed"}:
        return "outage"
    if value in {"partial_capacity", "partial", "capped", "restricted"}:
        return "partial_capacity"
    if value in {"available", "open", "normal"}:
        return "available"
    if value in {"informational", "info"}:
        return "informational"

    limit = pd.to_numeric(pd.Series([capacity_limit_mw]), errors="coerce").iloc[0]
    nominal = pd.to_numeric(pd.Series([nominal_capacity_mw]), errors="coerce").iloc[0]
    if pd.isna(limit):
        return "informational"
    if float(limit) <= 0:
        return "outage"
    if pd.notna(nominal) and float(limit) < float(nominal):
        return "partial_capacity"
    return "available"


def load_france_connector_reviewed_input(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return _empty_reviewed_period_frame()

    raw = _normalized_columns(_read_input_frame(path))
    if raw.empty:
        return _empty_reviewed_period_frame()

    connector_key = _first_series(raw, ["connector_key", "connector", "cable_key", "cable"]).astype(str).str.strip().str.lower()
    unknown_connectors = sorted(set(connector_key) - set(_CONNECTOR_METADATA))
    if unknown_connectors:
        raise ValueError(f"unsupported France connector key(s): {', '.join(unknown_connectors)}")

    source_key = _first_series(raw, ["source_key", "reviewed_source_key", "source_type"]).astype(str).str.strip().str.lower()
    unknown_sources = sorted({key for key in source_key if key and key not in _SOURCE_CATALOG})
    if unknown_sources:
        raise ValueError(f"unsupported France reviewed source key(s): {', '.join(unknown_sources)}")

    period_timezone = _first_series(raw, ["period_timezone", "timezone", "tz"], DEFAULT_PERIOD_TIMEZONE).fillna(DEFAULT_PERIOD_TIMEZONE)
    period_start = _parse_timestamp_series(
        _first_series(raw, ["period_start_utc", "period_start", "start_utc", "start_date", "start"]),
        _first_series(raw, ["period_start_time", "start_time"], "00:00"),
        period_timezone,
    )
    end_base = _first_series(raw, ["period_end_utc", "period_end", "end_utc", "end_date", "end"])
    period_end = _parse_timestamp_series(
        end_base,
        _first_series(raw, ["period_end_time", "end_time"], "00:00"),
        period_timezone,
    )
    date_only_end_mask = end_base.astype(str).str.len().le(10) & period_end.notna()
    period_end = period_end.where(~date_only_end_mask, period_end + pd.Timedelta(days=1))

    if period_start.isna().any() or period_end.isna().any():
        raise ValueError("France reviewed-period input must provide parseable start and end timestamps or dates")
    if bool((period_end <= period_start).any()):
        raise ValueError("France reviewed-period input contains end times that are not after the start time")

    metadata = connector_key.map(_CONNECTOR_METADATA)
    nominal_capacity = metadata.map(lambda row: row["nominal_capacity_mw"])
    connector_label = metadata.map(lambda row: row["connector_label"])

    capacity_limit = pd.to_numeric(
        _first_series(raw, ["capacity_limit_mw", "available_capacity_mw", "capacity_mw"]),
        errors="coerce",
    )
    unavailable_capacity = pd.to_numeric(
        _first_series(raw, ["unavailable_capacity_mw", "unavailable_mw"]),
        errors="coerce",
    )
    missing_limit = capacity_limit.isna() & unavailable_capacity.notna() & nominal_capacity.notna()
    capacity_limit = capacity_limit.where(~missing_limit, nominal_capacity - unavailable_capacity)

    explicit_state = _first_series(raw, ["reviewed_publication_state", "publication_state", "state", "availability_state"])
    derived_state = pd.Series(
        [
            _derive_publication_state(explicit_state.iloc[index], capacity_limit.iloc[index], nominal_capacity.iloc[index])
            for index in range(len(raw))
        ],
        index=raw.index,
        dtype="object",
    )

    capacity_limit = capacity_limit.where(~derived_state.eq("outage"), 0.0)
    capacity_limit = capacity_limit.where(~(derived_state.eq("available") & capacity_limit.isna()), nominal_capacity)
    available_capacity = capacity_limit.copy()
    unavailable_capacity = unavailable_capacity.where(unavailable_capacity.notna(), nominal_capacity - available_capacity)

    catalog = source_key.map(_SOURCE_CATALOG)
    source_family = pd.Series(
        [
            row.get("source_family") if isinstance(row, dict) else pd.NA
            for row in catalog
        ],
        index=raw.index,
        dtype="object",
    )
    source_label = _first_series(raw, ["source_label", "reviewed_source_label"]).where(
        _first_series(raw, ["source_label", "reviewed_source_label"]).notna(),
        pd.Series(
            [row.get("source_label") if isinstance(row, dict) else pd.NA for row in catalog],
            index=raw.index,
            dtype="object",
        ),
    )
    source_url = _first_series(raw, ["source_document_url", "source_url", "reviewed_source_url"]).where(
        _first_series(raw, ["source_document_url", "source_url", "reviewed_source_url"]).notna(),
        pd.Series(
            [row.get("source_url") if isinstance(row, dict) else pd.NA for row in catalog],
            index=raw.index,
            dtype="object",
        ),
    )

    frame = pd.DataFrame(
        {
            "connector_key": connector_key,
            "connector_label": connector_label,
            "direction_key": _first_series(raw, ["direction_key", "direction"], "gb_to_neighbor").map(_normalize_direction),
            "reviewed_scope": "france_connector_public_doc_period",
            "review_state": _first_series(raw, ["review_state"], DEFAULT_REVIEW_STATE).fillna(DEFAULT_REVIEW_STATE),
            "reviewed_evidence_tier": _first_series(raw, ["reviewed_evidence_tier"], DEFAULT_REVIEWED_EVIDENCE_TIER).fillna(
                DEFAULT_REVIEWED_EVIDENCE_TIER
            ),
            "reviewed_tier_accepted_flag": _first_series(raw, ["reviewed_tier_accepted_flag"], True)
            .where(_first_series(raw, ["reviewed_tier_accepted_flag"], True).notna(), True)
            .astype(bool),
            "capacity_policy_action": _first_series(
                raw,
                ["capacity_policy_action"],
                DEFAULT_CAPACITY_POLICY_ACTION,
            ).fillna(DEFAULT_CAPACITY_POLICY_ACTION),
            "reviewed_publication_state": derived_state,
            "period_start_utc": period_start,
            "period_end_utc": period_end,
            "period_timezone": period_timezone,
            "connector_nominal_capacity_mw": nominal_capacity,
            "reviewed_capacity_limit_mw": capacity_limit,
            "reviewed_available_capacity_mw": available_capacity,
            "reviewed_unavailable_capacity_mw": unavailable_capacity,
            "source_provider": _first_series(raw, ["source_provider"], DEFAULT_REVIEWED_SOURCE_PROVIDER).fillna(
                DEFAULT_REVIEWED_SOURCE_PROVIDER
            ),
            "source_family": source_family,
            "source_key": source_key,
            "source_label": source_label,
            "source_document_title": _first_series(raw, ["source_document_title", "document_title", "title"]),
            "source_document_url": source_url,
            "source_reference": _first_series(raw, ["source_reference", "document_reference", "reference"]),
            "source_published_date": pd.to_datetime(
                _first_series(raw, ["source_published_date", "published_date", "document_date"]),
                errors="coerce",
            ).dt.date,
            "review_note": _first_series(raw, ["review_note", "note", "notes"]),
            "target_is_proxy": False,
        }
    )
    return frame.sort_values(["period_start_utc", "connector_key", "source_key"]).reset_index(drop=True)


def build_fact_france_connector_reviewed_period(
    start_date: dt.date,
    end_date: dt.date,
    reviewed_input: pd.DataFrame | None = None,
    reviewed_input_path: str | Path | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    frame = reviewed_input.copy() if reviewed_input is not None else load_france_connector_reviewed_input(reviewed_input_path)
    if frame.empty:
        return _empty_reviewed_period_frame()
    frame["period_start_utc"] = pd.to_datetime(frame["period_start_utc"], utc=True, errors="coerce")
    frame["period_end_utc"] = pd.to_datetime(frame["period_end_utc"], utc=True, errors="coerce")
    window_start = pd.Timestamp(start_date, tz="UTC")
    window_end = pd.Timestamp(end_date + dt.timedelta(days=1), tz="UTC")
    overlapping = frame[
        frame["period_start_utc"].lt(window_end) & frame["period_end_utc"].gt(window_start)
    ].copy()
    if overlapping.empty:
        return _empty_reviewed_period_frame()
    column_order = list(_empty_reviewed_period_frame().columns)
    return overlapping[column_order].sort_values(["period_start_utc", "connector_key", "source_key"]).reset_index(drop=True)


def build_fact_france_connector_reviewed_hourly(
    start_date: dt.date,
    end_date: dt.date,
    reviewed_period: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if reviewed_period is None or reviewed_period.empty:
        return _empty_reviewed_hourly_frame()

    frame = reviewed_period.copy()
    frame["period_start_utc"] = pd.to_datetime(frame["period_start_utc"], utc=True, errors="coerce")
    frame["period_end_utc"] = pd.to_datetime(frame["period_end_utc"], utc=True, errors="coerce")
    frame = frame[frame["period_end_utc"].gt(frame["period_start_utc"])].copy()
    if frame.empty:
        return _empty_reviewed_hourly_frame()

    interval_start_utc = pd.date_range(
        start=pd.Timestamp(start_date, tz="UTC"),
        end=pd.Timestamp(end_date + dt.timedelta(days=1), tz="UTC"),
        freq="h",
        inclusive="left",
    )
    hours = pd.DataFrame({"interval_start_utc": interval_start_utc})
    hours["interval_end_utc"] = hours["interval_start_utc"] + pd.Timedelta(hours=1)

    joined = hours.assign(_join_key=1).merge(frame.assign(_join_key=1), on="_join_key", how="inner").drop(columns="_join_key")
    joined = joined[
        joined["interval_start_utc"].lt(joined["period_end_utc"])
        & joined["interval_end_utc"].gt(joined["period_start_utc"])
    ].copy()
    if joined.empty:
        return _empty_reviewed_hourly_frame()

    severity_rank = joined["reviewed_publication_state"].map(
        {"outage": 0, "partial_capacity": 1, "available": 2, "informational": 3}
    ).fillna(9)
    joined["reviewed_capacity_limit_mw"] = pd.to_numeric(joined["reviewed_capacity_limit_mw"], errors="coerce")
    joined["reviewed_available_capacity_mw"] = pd.to_numeric(joined["reviewed_available_capacity_mw"], errors="coerce")
    joined["reviewed_unavailable_capacity_mw"] = pd.to_numeric(joined["reviewed_unavailable_capacity_mw"], errors="coerce")
    joined = joined.assign(_severity_rank=severity_rank)
    joined = joined.sort_values(
        [
            "interval_start_utc",
            "connector_key",
            "direction_key",
            "_severity_rank",
            "reviewed_capacity_limit_mw",
            "source_published_date",
        ],
        ascending=[True, True, True, True, True, False],
        na_position="last",
    )

    group_columns = ["interval_start_utc", "connector_key", "direction_key"]
    representative = joined.groupby(group_columns, as_index=False).first()
    aggregated_limit = joined.groupby(group_columns)["reviewed_capacity_limit_mw"].min().reset_index()
    aggregated_available = joined.groupby(group_columns)["reviewed_available_capacity_mw"].min().reset_index()
    aggregated_unavailable = joined.groupby(group_columns)["reviewed_unavailable_capacity_mw"].max().reset_index()
    source_count = joined.groupby(group_columns)["source_key"].nunique().reset_index(name="reviewed_source_count")

    hourly = representative.merge(aggregated_limit, on=group_columns, how="left", suffixes=("", "_min"))
    hourly = hourly.merge(aggregated_available, on=group_columns, how="left", suffixes=("", "_min"))
    hourly = hourly.merge(aggregated_unavailable, on=group_columns, how="left", suffixes=("", "_max"))
    hourly = hourly.merge(source_count, on=group_columns, how="left")

    hourly["reviewed_capacity_limit_mw"] = hourly["reviewed_capacity_limit_mw_min"].where(
        hourly["reviewed_capacity_limit_mw_min"].notna(),
        hourly["reviewed_capacity_limit_mw"],
    )
    hourly["reviewed_available_capacity_mw"] = hourly["reviewed_available_capacity_mw_min"].where(
        hourly["reviewed_available_capacity_mw_min"].notna(),
        hourly["reviewed_available_capacity_mw"],
    )
    hourly["reviewed_unavailable_capacity_mw"] = hourly["reviewed_unavailable_capacity_mw_max"].where(
        hourly["reviewed_unavailable_capacity_mw_max"].notna(),
        hourly["reviewed_unavailable_capacity_mw"],
    )
    hourly = hourly.drop(
        columns=[
            "_severity_rank",
            "interval_end_utc",
            "period_start_utc",
            "period_end_utc",
            "reviewed_capacity_limit_mw_min",
            "reviewed_available_capacity_mw_min",
            "reviewed_unavailable_capacity_mw_max",
            "reviewed_scope",
            "period_timezone",
            "connector_nominal_capacity_mw",
            "target_is_proxy",
        ],
        errors="ignore",
    )

    column_order = list(_empty_reviewed_hourly_frame().columns)
    return hourly[column_order].sort_values(group_columns).reset_index(drop=True)


def materialize_france_connector_reviewed_period(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    reviewed_input: pd.DataFrame | None = None,
    reviewed_input_path: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_france_connector_reviewed_period(
        start_date=start_date,
        end_date=end_date,
        reviewed_input=reviewed_input,
        reviewed_input_path=reviewed_input_path,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE}.csv", index=False)
    return {FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE: fact}
