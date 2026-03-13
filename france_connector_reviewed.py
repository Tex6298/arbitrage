from __future__ import annotations

import datetime as dt
import json
import re
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE = "fact_france_connector_reviewed_period"
FRANCE_CONNECTOR_NOTICE_TABLE = "fact_france_connector_notice_hourly"

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


def _requested_window_utc(start_date: dt.date, end_date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_local = pd.Timestamp(start_date, tz="Europe/London")
    end_local = pd.Timestamp(end_date + dt.timedelta(days=1), tz="Europe/London")
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


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
            "source_published_utc",
            "source_published_date",
            "notice_group_key",
            "notice_planning_state",
            "planned_outage_flag",
            "source_revision_rank",
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


def _empty_notice_hourly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start_utc",
            "interval_end_utc",
            "connector_key",
            "connector_label",
            "direction_key",
            "notice_state",
            "notice_known_flag",
            "notice_active_flag",
            "notice_upcoming_flag",
            "notice_group_key",
            "notice_planning_state",
            "planned_outage_flag",
            "expected_capacity_limit_mw",
            "hours_until_notice_start",
            "days_until_notice_start",
            "hours_since_notice_publication",
            "notice_lead_time_hours",
            "notice_revision_count",
            "source_revision_rank",
            "source_provider",
            "source_family",
            "source_key",
            "source_label",
            "source_document_title",
            "source_document_url",
            "source_reference",
            "source_published_utc",
            "source_published_date",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_publication_state",
            "review_note",
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
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        for column in renamed.columns
    ]
    return renamed


def _read_text_table(path: Path) -> pd.DataFrame:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()
    if "\t" in lines[0]:
        splitter = lambda line: [cell.strip() for cell in line.split("\t")]
    else:
        splitter = lambda line: [cell.strip() for cell in re.split(r"\s{2,}", line.strip())]
    rows = [splitter(line) for line in lines]
    header = rows[0]
    body = [row for row in rows[1:] if len(row) == len(header)]
    return pd.DataFrame(body, columns=header)


def _read_input_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"France reviewed-period input does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(file_path, sep=None, engine="python")
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)
    if suffix == ".txt":
        return _read_text_table(file_path)
    raise ValueError("France reviewed-period input must be .csv, .tsv, .txt, or .json")


def _first_series(frame: pd.DataFrame, names: list[str], default: object = pd.NA) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def _coalesce_series(frame: pd.DataFrame, names: list[str], default: object = pd.NA) -> pd.Series:
    available = [frame[name] for name in names if name in frame.columns]
    if not available:
        return pd.Series([default] * len(frame), index=frame.index, dtype="object")
    combined = pd.concat(available, axis=1)
    resolved = combined.bfill(axis=1).iloc[:, 0]
    if default is not pd.NA:
        resolved = resolved.where(resolved.notna(), default)
    return resolved


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


def _parse_delivery_period_range(
    date_values: pd.Series,
    period_values: pd.Series,
    timezone_values: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    start_values: list[pd.Timestamp] = []
    end_values: list[pd.Timestamp] = []
    for index in date_values.index:
        raw_date = date_values.loc[index]
        raw_period = period_values.loc[index]
        if pd.isna(raw_date) or pd.isna(raw_period):
            start_values.append(pd.NaT)
            end_values.append(pd.NaT)
            continue
        match = re.match(r"^\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*$", str(raw_period))
        if not match:
            start_values.append(pd.NaT)
            end_values.append(pd.NaT)
            continue
        timezone_name = timezone_values.loc[index]
        timezone_name = DEFAULT_PERIOD_TIMEZONE if pd.isna(timezone_name) or str(timezone_name).strip() == "" else str(timezone_name)
        day = pd.Timestamp(str(raw_date).strip())
        start_local = pd.Timestamp(f"{day.date()} {match.group(1)}").tz_localize(timezone_name)
        end_local = pd.Timestamp(f"{day.date()} {match.group(2)}").tz_localize(timezone_name)
        if end_local <= start_local:
            end_local = end_local + pd.Timedelta(days=1)
        start_values.append(start_local.tz_convert("UTC"))
        end_values.append(end_local.tz_convert("UTC"))
    return (
        pd.Series(start_values, index=date_values.index, dtype="datetime64[ns, UTC]"),
        pd.Series(end_values, index=date_values.index, dtype="datetime64[ns, UTC]"),
    )


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
    value = "" if pd.isna(explicit_state) else str(explicit_state).strip().lower()
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


def _derive_notice_planning_state(source_key: object, explicit_value: object) -> str:
    if not pd.isna(explicit_value):
        value = str(explicit_value).strip().lower()
        if value in {"planned", "market_notice", "operational_restriction", "unplanned", "unknown"}:
            return value
    source = str(source_key or "").strip().lower()
    if source == "eleclink_planned_outage_programme":
        return "planned"
    if source == "eleclink_capacity_split":
        return "planned"
    if source == "eleclink_ntc_restriction":
        return "operational_restriction"
    if source.startswith("jao_"):
        return "market_notice"
    return "unknown"


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

    period_timezone = _coalesce_series(raw, ["period_timezone", "timezone", "tz"], DEFAULT_PERIOD_TIMEZONE).fillna(DEFAULT_PERIOD_TIMEZONE)
    period_start_input = _coalesce_series(raw, ["period_start_utc", "period_start", "start_utc", "start_date", "start"])
    period_end_input = _coalesce_series(raw, ["period_end_utc", "period_end", "end_utc", "end_date", "end"])
    delivery_date = _coalesce_series(raw, ["delivery_date", "date", "period_date", "settlement_date"])
    delivery_period = _coalesce_series(raw, ["delivery_period_gmt", "delivery_period", "time_interval", "period_range"])
    period_start = _parse_timestamp_series(
        period_start_input,
        _coalesce_series(raw, ["period_start_time", "start_time"], "00:00"),
        period_timezone,
    )
    end_base = period_end_input
    period_end = _parse_timestamp_series(
        end_base,
        _coalesce_series(raw, ["period_end_time", "end_time"], "00:00"),
        period_timezone,
    )
    delivery_period_start, delivery_period_end = _parse_delivery_period_range(
        delivery_date,
        delivery_period,
        period_timezone,
    )
    period_start = period_start.where(period_start.notna(), delivery_period_start)
    period_end = period_end.where(period_end.notna(), delivery_period_end)
    date_only_end_mask = end_base.notna() & end_base.astype(str).str.len().le(10) & period_end.notna()
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

    explicit_state = _coalesce_series(raw, ["reviewed_publication_state", "publication_state", "state", "availability_state"])
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
    source_label = _coalesce_series(raw, ["source_label", "reviewed_source_label"]).where(
        _coalesce_series(raw, ["source_label", "reviewed_source_label"]).notna(),
        pd.Series(
            [row.get("source_label") if isinstance(row, dict) else pd.NA for row in catalog],
            index=raw.index,
            dtype="object",
        ),
    )
    source_url = _coalesce_series(raw, ["source_document_url", "source_url", "reviewed_source_url"]).where(
        _coalesce_series(raw, ["source_document_url", "source_url", "reviewed_source_url"]).notna(),
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
            "direction_key": _coalesce_series(raw, ["direction_key", "direction"], "gb_to_neighbor").map(_normalize_direction),
            "reviewed_scope": "france_connector_public_doc_period",
            "review_state": _coalesce_series(raw, ["review_state"], DEFAULT_REVIEW_STATE).fillna(DEFAULT_REVIEW_STATE),
            "reviewed_evidence_tier": _coalesce_series(raw, ["reviewed_evidence_tier"], DEFAULT_REVIEWED_EVIDENCE_TIER).fillna(
                DEFAULT_REVIEWED_EVIDENCE_TIER
            ),
            "reviewed_tier_accepted_flag": _coalesce_series(raw, ["reviewed_tier_accepted_flag"], True)
            .where(_coalesce_series(raw, ["reviewed_tier_accepted_flag"], True).notna(), True)
            .astype(bool),
            "capacity_policy_action": _coalesce_series(
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
            "source_provider": _coalesce_series(raw, ["source_provider"], DEFAULT_REVIEWED_SOURCE_PROVIDER).fillna(
                DEFAULT_REVIEWED_SOURCE_PROVIDER
            ),
            "source_family": source_family,
            "source_key": source_key,
            "source_label": source_label,
            "source_document_title": _coalesce_series(raw, ["source_document_title", "document_title", "title"]),
            "source_document_url": source_url,
            "source_reference": _coalesce_series(raw, ["source_reference", "document_reference", "reference"]),
            "source_published_utc": _parse_timestamp_series(
                _coalesce_series(raw, ["source_published_utc", "published_utc", "publish_time", "publication_time", "published_at"]),
                _coalesce_series(raw, ["source_published_time", "published_time"], "00:00"),
                _coalesce_series(raw, ["source_published_timezone", "published_timezone", "publication_timezone"], DEFAULT_PERIOD_TIMEZONE),
            ),
            "source_published_date": pd.to_datetime(
                _coalesce_series(raw, ["source_published_date", "published_date", "document_date"]),
                errors="coerce",
            ).dt.date,
            "review_note": _coalesce_series(raw, ["review_note", "note", "notes"]),
            "target_is_proxy": False,
        }
    )
    frame["source_published_utc"] = frame["source_published_utc"].where(
        frame["source_published_utc"].notna(),
        pd.to_datetime(frame["source_published_date"], errors="coerce", utc=True),
    )
    frame["notice_group_key"] = (
        frame["connector_key"].astype(str)
        + "|"
        + frame["direction_key"].astype(str)
        + "|"
        + frame["period_start_utc"].astype(str)
        + "|"
        + frame["period_end_utc"].astype(str)
    )
    frame["notice_planning_state"] = [
        _derive_notice_planning_state(source_key.iloc[index], _coalesce_series(raw, ["notice_planning_state", "planned_vs_unplanned"]).iloc[index])
        for index in range(len(frame))
    ]
    frame["planned_outage_flag"] = frame["notice_planning_state"].eq("planned")
    frame = frame.sort_values(
        ["notice_group_key", "source_published_utc", "source_key"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    frame["source_revision_rank"] = frame.groupby("notice_group_key").cumcount() + 1
    return frame.sort_values(["period_start_utc", "connector_key", "source_key"]).reset_index(drop=True)


def normalize_france_connector_reviewed_input(reviewed_input: pd.DataFrame) -> pd.DataFrame:
    if reviewed_input.empty:
        return _empty_reviewed_period_frame()
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / "france_reviewed.csv"
        reviewed_input.to_csv(temp_path, index=False)
        return load_france_connector_reviewed_input(temp_path)


def write_normalized_france_connector_reviewed_input(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    normalized = load_france_connector_reviewed_input(input_path)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(target_path, index=False)
    return normalized


def build_fact_france_connector_reviewed_period(
    start_date: dt.date,
    end_date: dt.date,
    reviewed_input: pd.DataFrame | None = None,
    reviewed_input_path: str | Path | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    frame = (
        normalize_france_connector_reviewed_input(reviewed_input)
        if reviewed_input is not None
        else load_france_connector_reviewed_input(reviewed_input_path)
    )
    if frame.empty:
        return _empty_reviewed_period_frame()
    frame["period_start_utc"] = pd.to_datetime(frame["period_start_utc"], utc=True, errors="coerce")
    frame["period_end_utc"] = pd.to_datetime(frame["period_end_utc"], utc=True, errors="coerce")
    window_start, window_end = _requested_window_utc(start_date, end_date)
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

    window_start, window_end = _requested_window_utc(start_date, end_date)
    interval_start_utc = pd.date_range(
        start=window_start,
        end=window_end,
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


def build_fact_france_connector_notice_hourly(
    start_date: dt.date,
    end_date: dt.date,
    reviewed_period: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if reviewed_period is None or reviewed_period.empty:
        return _empty_notice_hourly_frame()

    frame = reviewed_period.copy()
    frame["period_start_utc"] = pd.to_datetime(frame["period_start_utc"], utc=True, errors="coerce")
    frame["period_end_utc"] = pd.to_datetime(frame["period_end_utc"], utc=True, errors="coerce")
    frame["source_published_utc"] = pd.to_datetime(frame["source_published_utc"], utc=True, errors="coerce")
    frame = frame[
        frame["period_end_utc"].gt(frame["period_start_utc"])
        & frame["source_published_utc"].notna()
    ].copy()
    if frame.empty:
        return _empty_notice_hourly_frame()

    window_start, window_end = _requested_window_utc(start_date, end_date)
    hours = pd.DataFrame(
        {"interval_start_utc": pd.date_range(start=window_start, end=window_end, freq="h", inclusive="left")}
    )
    hours["interval_end_utc"] = hours["interval_start_utc"] + pd.Timedelta(hours=1)
    connectors = pd.DataFrame(
        [
            {
                "connector_key": connector_key,
                "connector_label": metadata["connector_label"],
                "direction_key": direction_key,
            }
            for connector_key, metadata in _CONNECTOR_METADATA.items()
            for direction_key in ("gb_to_neighbor", "neighbor_to_gb")
        ]
    )
    base = hours.assign(_join_key=1).merge(connectors.assign(_join_key=1), on="_join_key", how="inner").drop(columns="_join_key")

    candidates = base.assign(_join_key=1).merge(frame.assign(_join_key=1), on="_join_key", how="left").drop(columns="_join_key")
    candidates = candidates[
        candidates["connector_key_x"].eq(candidates["connector_key_y"])
        & (
            candidates["direction_key_y"].eq("both")
            | candidates["direction_key_x"].eq(candidates["direction_key_y"])
        )
    ].copy()
    if candidates.empty:
        empty = _empty_notice_hourly_frame()
        dense = base.copy()
        dense["notice_state"] = "no_notice"
        dense["notice_known_flag"] = False
        dense["notice_active_flag"] = False
        dense["notice_upcoming_flag"] = False
        for column in empty.columns:
            if column in dense.columns:
                continue
            dense[column] = pd.NA
        dense["notice_state"] = dense["notice_state"].fillna("no_notice")
        dense["notice_known_flag"] = dense["notice_known_flag"].fillna(False)
        dense["notice_active_flag"] = dense["notice_active_flag"].fillna(False)
        dense["notice_upcoming_flag"] = dense["notice_upcoming_flag"].fillna(False)
        return dense[empty.columns].sort_values(["interval_start_utc", "connector_key", "direction_key"]).reset_index(drop=True)

    candidates = candidates[
        candidates["source_published_utc"].le(candidates["interval_start_utc"])
        & candidates["period_end_utc"].gt(candidates["interval_start_utc"])
    ].copy()
    if candidates.empty:
        result = base.copy()
        result["notice_state"] = "no_notice"
        result["notice_known_flag"] = False
        result["notice_active_flag"] = False
        result["notice_upcoming_flag"] = False
        for column in _empty_notice_hourly_frame().columns:
            if column in result.columns:
                continue
            result[column] = pd.NA
        result["notice_state"] = result["notice_state"].fillna("no_notice")
        result["notice_known_flag"] = result["notice_known_flag"].fillna(False)
        result["notice_active_flag"] = result["notice_active_flag"].fillna(False)
        result["notice_upcoming_flag"] = result["notice_upcoming_flag"].fillna(False)
        return result[_empty_notice_hourly_frame().columns].sort_values(
            ["interval_start_utc", "connector_key", "direction_key"]
        ).reset_index(drop=True)

    candidates["connector_key"] = candidates["connector_key_x"]
    candidates["connector_label"] = candidates["connector_label_x"]
    candidates["direction_key"] = candidates["direction_key_x"]
    candidates["notice_state"] = np.where(
        candidates["period_start_utc"].le(candidates["interval_start_utc"]),
        "active",
        "upcoming",
    )
    candidates["notice_state_rank"] = candidates["notice_state"].map({"active": 0, "upcoming": 1}).fillna(9)
    candidates = candidates.sort_values(
        [
            "interval_start_utc",
            "connector_key",
            "direction_key",
            "notice_state_rank",
            "period_start_utc",
            "source_published_utc",
        ],
        ascending=[True, True, True, True, True, False],
        na_position="last",
    )
    group_columns = ["interval_start_utc", "connector_key", "direction_key"]
    selected = candidates.groupby(group_columns, as_index=False).first()
    revision_counts = (
        candidates.groupby(group_columns + ["notice_group_key"])["source_revision_rank"]
        .max()
        .reset_index(name="notice_revision_count")
    )
    selected = selected.merge(revision_counts, on=group_columns + ["notice_group_key"], how="left")
    selected["notice_known_flag"] = True
    selected["notice_active_flag"] = selected["notice_state"].eq("active")
    selected["notice_upcoming_flag"] = selected["notice_state"].eq("upcoming")
    selected["expected_capacity_limit_mw"] = pd.to_numeric(selected["reviewed_capacity_limit_mw"], errors="coerce")
    selected["hours_until_notice_start"] = (
        (selected["period_start_utc"] - selected["interval_start_utc"]) / pd.Timedelta(hours=1)
    ).clip(lower=0)
    selected["days_until_notice_start"] = selected["hours_until_notice_start"] / 24.0
    selected["hours_since_notice_publication"] = (
        (selected["interval_start_utc"] - selected["source_published_utc"]) / pd.Timedelta(hours=1)
    ).clip(lower=0)
    selected["notice_lead_time_hours"] = (
        (selected["period_start_utc"] - selected["source_published_utc"]) / pd.Timedelta(hours=1)
    )

    selected = selected[
        [
            "interval_start_utc",
            "interval_end_utc",
            "connector_key",
            "connector_label",
            "direction_key",
            "notice_state",
            "notice_known_flag",
            "notice_active_flag",
            "notice_upcoming_flag",
            "notice_group_key",
            "notice_planning_state",
            "planned_outage_flag",
            "expected_capacity_limit_mw",
            "hours_until_notice_start",
            "days_until_notice_start",
            "hours_since_notice_publication",
            "notice_lead_time_hours",
            "notice_revision_count",
            "source_revision_rank",
            "source_provider",
            "source_family",
            "source_key",
            "source_label",
            "source_document_title",
            "source_document_url",
            "source_reference",
            "source_published_utc",
            "source_published_date",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_publication_state",
            "review_note",
        ]
    ].copy()

    result = base.merge(selected, on=["interval_start_utc", "interval_end_utc", "connector_key", "connector_label", "direction_key"], how="left")
    result["notice_state"] = result["notice_state"].fillna("no_notice")
    result["notice_known_flag"] = result["notice_known_flag"].where(result["notice_known_flag"].notna(), False).astype(bool)
    result["notice_active_flag"] = result["notice_active_flag"].where(result["notice_active_flag"].notna(), False).astype(bool)
    result["notice_upcoming_flag"] = result["notice_upcoming_flag"].where(result["notice_upcoming_flag"].notna(), False).astype(bool)
    result["planned_outage_flag"] = result["planned_outage_flag"].where(result["planned_outage_flag"].notna(), False).astype(bool)
    result["reviewed_tier_accepted_flag"] = result["reviewed_tier_accepted_flag"].where(
        result["reviewed_tier_accepted_flag"].notna(),
        False,
    ).astype(bool)
    column_order = list(_empty_notice_hourly_frame().columns)
    return result[column_order].sort_values(["interval_start_utc", "connector_key", "direction_key"]).reset_index(drop=True)


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
    notice = build_fact_france_connector_notice_hourly(
        start_date=start_date,
        end_date=end_date,
        reviewed_period=fact,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE}.csv", index=False)
    notice.to_csv(output_path / f"{FRANCE_CONNECTOR_NOTICE_TABLE}.csv", index=False)
    return {
        FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE: fact,
        FRANCE_CONNECTOR_NOTICE_TABLE: notice,
    }
