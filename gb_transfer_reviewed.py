from __future__ import annotations

import datetime as dt
import json
import re
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from asset_mapping import cluster_frame
from gb_topology import interconnector_hub_frame


GB_TRANSFER_REVIEWED_PERIOD_TABLE = "fact_gb_transfer_reviewed_period"
GB_TRANSFER_REVIEW_POLICY_TABLE = "fact_gb_transfer_review_policy"
GB_TRANSFER_REVIEWED_HOURLY_TABLE = "fact_gb_transfer_reviewed_hourly"

DEFAULT_REVIEWED_SOURCE_PROVIDER = "public_reviewed_doc"
DEFAULT_REVIEW_STATE = "accepted_reviewed_tier"
DEFAULT_REVIEWED_EVIDENCE_TIER = "reviewed_internal_transfer_period"
DEFAULT_CAPACITY_POLICY_ACTION = "allow_reviewed_internal_period"
DEFAULT_PERIOD_TIMEZONE = "UTC"
SCOTLAND_NORTH_TO_SOUTH_SOURCE_FAMILY = "scotland_north_to_south_review"
SHETLAND_DEPENDENCY_SOURCE_FAMILY = "shetland_dependency_review"

_SOURCE_CATALOG: Dict[str, dict[str, str]] = {
    "internal_boundary_restriction": {
        "source_family": "public_boundary_doc",
        "source_label": "Internal boundary restriction",
        "source_url": "",
    },
    "reviewed_transfer_cap_window": {
        "source_family": "public_transfer_review",
        "source_label": "Reviewed transfer-cap window",
        "source_url": "",
    },
    "public_constraint_period": {
        "source_family": "public_constraint_doc",
        "source_label": "Public constraint period",
        "source_url": "",
    },
    "etys_2023_b6_capability": {
        "source_family": SCOTLAND_NORTH_TO_SOUTH_SOURCE_FAMILY,
        "source_label": "ETYS 2023 B6 north-to-south capability",
        "source_url": "https://www.neso.energy/document/286591/download",
    },
    "scotland_north_to_south_gap_review": {
        "source_family": SCOTLAND_NORTH_TO_SOUTH_SOURCE_FAMILY,
        "source_label": "Scotland north-to-south gap-hours review",
        "source_url": "",
    },
    "shetland_island_link_dependency_review": {
        "source_family": SHETLAND_DEPENDENCY_SOURCE_FAMILY,
        "source_label": "Shetland island-link dependency review",
        "source_url": "",
    },
}

_CLUSTER_LOOKUP = cluster_frame()[
    ["cluster_key", "cluster_label", "parent_region", "approx_capacity_mw"]
].drop_duplicates("cluster_key")
_REGION_CAPACITY_LOOKUP = (
    _CLUSTER_LOOKUP.groupby("parent_region", as_index=False)["approx_capacity_mw"]
    .sum()
    .rename(columns={"approx_capacity_mw": "region_approx_capacity_mw"})
)
_HUB_LOOKUP = interconnector_hub_frame()[["key", "label"]].rename(
    columns={"key": "hub_key", "label": "hub_label"}
).drop_duplicates("hub_key")


def _requested_window_utc(start_date: dt.date, end_date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_local = pd.Timestamp(start_date, tz="Europe/London")
    end_local = pd.Timestamp(end_date + dt.timedelta(days=1), tz="Europe/London")
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def _empty_reviewed_period_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scope_key",
            "scope_granularity",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "hub_key",
            "hub_label",
            "reviewed_scope",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_gate_state",
            "period_start_utc",
            "period_end_utc",
            "period_timezone",
            "approx_cluster_capacity_mw",
            "region_approx_capacity_mw",
            "reviewed_capacity_limit_mw",
            "reviewed_gate_fraction",
            "source_provider",
            "source_family",
            "source_key",
            "source_label",
            "source_document_title",
            "source_document_url",
            "source_reference",
            "source_published_utc",
            "source_published_date",
            "review_group_key",
            "source_revision_rank",
            "review_note",
            "target_is_proxy",
        ]
    )


def _empty_review_policy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_key",
            "source_family",
            "source_provider",
            "source_label",
            "source_document_url",
            "policy_scope",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "policy_note",
            "source_lineage",
        ]
    )


def _empty_reviewed_hourly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start_utc",
            "interval_end_utc",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "hub_key",
            "hub_label",
            "scope_key",
            "scope_granularity",
            "reviewed_scope",
            "review_state",
            "reviewed_evidence_tier",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "reviewed_gate_state",
            "reviewed_capacity_limit_mw",
            "reviewed_gate_fraction",
            "source_provider",
            "source_family",
            "source_key",
            "source_label",
            "source_document_title",
            "source_document_url",
            "source_reference",
            "source_published_utc",
            "source_published_date",
            "review_group_key",
            "source_revision_rank",
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
        raise FileNotFoundError(f"GB transfer reviewed input does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(file_path, sep=None, engine="python")
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)
    if suffix == ".txt":
        return _read_text_table(file_path)
    raise ValueError("GB transfer reviewed input must be .csv, .tsv, .txt, or .json")


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


def _normalize_review_state(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in {"accepted_reviewed_tier", "accepted", "allow"}:
        return "accepted_reviewed_tier"
    if text in {"keep_proxy_fallback", "proxy_fallback"}:
        return "keep_proxy_fallback"
    if text in {"audit_only", "not_accepted"}:
        return "audit_only"
    return DEFAULT_REVIEW_STATE


def _normalize_capacity_policy_action(value: object, review_state: str) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text:
        return text
    if review_state == "accepted_reviewed_tier":
        return DEFAULT_CAPACITY_POLICY_ACTION
    if review_state == "keep_proxy_fallback":
        return "keep_proxy_fallback"
    return "audit_only"


def _normalize_explicit_gate_state(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"blocked_reviewed_capacity", "blocked", "outage", "closed", "zero"}:
        return "blocked_reviewed_capacity"
    if text in {"reviewed_pass", "pass", "open", "available"}:
        return "reviewed_pass"
    if text in {"reviewed_pass_restricted", "restricted", "partial", "conditional", "capped"}:
        return "reviewed_pass_restricted"
    if text in {"audit_only"}:
        return "audit_only"
    raise ValueError(f"unsupported internal reviewed gate state: {value}")


def normalize_gb_transfer_reviewed_input(reviewed_input: pd.DataFrame) -> pd.DataFrame:
    raw = _normalized_columns(reviewed_input.copy())
    if raw.empty:
        return _empty_reviewed_period_frame()

    cluster_key = _coalesce_series(raw, ["cluster_key", "cluster"], pd.NA).astype("object")
    cluster_key = cluster_key.where(cluster_key.isna(), cluster_key.astype(str).str.strip().str.lower())
    parent_region = _coalesce_series(raw, ["parent_region", "region"], pd.NA).astype("object")
    parent_region = parent_region.where(parent_region.isna(), parent_region.astype(str).str.strip())
    hub_key = _first_series(raw, ["hub_key", "hub"], pd.NA).astype("object")
    hub_key = hub_key.where(hub_key.isna(), hub_key.astype(str).str.strip().str.lower())

    if (~cluster_key.isna() | ~parent_region.isna()).sum() != len(raw):
        raise ValueError("GB transfer reviewed input must provide cluster_key or parent_region for every row")
    if hub_key.isna().any():
        raise ValueError("GB transfer reviewed input must provide hub_key for every row")

    unknown_clusters = sorted(set(cluster_key.dropna()) - set(_CLUSTER_LOOKUP["cluster_key"]))
    unknown_regions = sorted(set(parent_region.dropna()) - set(_REGION_CAPACITY_LOOKUP["parent_region"]))
    unknown_hubs = sorted(set(hub_key.dropna()) - set(_HUB_LOOKUP["hub_key"]))
    if unknown_clusters:
        raise ValueError(f"unsupported cluster_key(s): {', '.join(str(value) for value in unknown_clusters)}")
    if unknown_regions:
        raise ValueError(f"unsupported parent_region(s): {', '.join(str(value) for value in unknown_regions)}")
    if unknown_hubs:
        raise ValueError(f"unsupported hub_key(s): {', '.join(str(value) for value in unknown_hubs)}")

    period_timezone = _coalesce_series(raw, ["period_timezone", "timezone", "tz"], DEFAULT_PERIOD_TIMEZONE).fillna(
        DEFAULT_PERIOD_TIMEZONE
    )
    period_start_input = _coalesce_series(raw, ["period_start_utc", "period_start", "start_utc", "start_date", "start"])
    period_end_input = _coalesce_series(raw, ["period_end_utc", "period_end", "end_utc", "end_date", "end"])
    delivery_date = _coalesce_series(raw, ["delivery_date", "date", "period_date", "settlement_date"])
    delivery_period = _coalesce_series(raw, ["delivery_period_gmt", "delivery_period", "time_interval", "period_range"])
    period_start = _parse_timestamp_series(
        period_start_input,
        _coalesce_series(raw, ["period_start_time", "start_time"], "00:00"),
        period_timezone,
    )
    period_end = _parse_timestamp_series(
        period_end_input,
        _coalesce_series(raw, ["period_end_time", "end_time"], "00:00"),
        period_timezone,
    )
    delivery_period_start, delivery_period_end = _parse_delivery_period_range(delivery_date, delivery_period, period_timezone)
    period_start = period_start.where(period_start.notna(), delivery_period_start)
    period_end = period_end.where(period_end.notna(), delivery_period_end)
    date_only_end_mask = period_end_input.notna() & period_end_input.astype(str).str.len().le(10) & period_end.notna()
    period_end = period_end.where(~date_only_end_mask, period_end + pd.Timedelta(days=1))

    if period_start.isna().any() or period_end.isna().any():
        raise ValueError("GB transfer reviewed input must provide parseable start and end timestamps or dates")
    if period_end.le(period_start).any():
        raise ValueError("GB transfer reviewed input must have period_end_utc after period_start_utc")

    source_key = _first_series(raw, ["source_key", "reviewed_source_key", "source_type"], "public_constraint_period")
    source_key = source_key.astype(str).str.strip().str.lower()
    source_provider = _coalesce_series(raw, ["source_provider"], DEFAULT_REVIEWED_SOURCE_PROVIDER).astype("object")
    source_family = pd.Series(
        [_SOURCE_CATALOG.get(key, {}).get("source_family", "custom_reviewed_input") for key in source_key],
        index=raw.index,
        dtype="object",
    )
    source_label = _coalesce_series(raw, ["source_label"], pd.NA).astype("object")
    source_label = source_label.where(
        source_label.notna(),
        pd.Series([_SOURCE_CATALOG.get(key, {}).get("source_label", key) for key in source_key], index=raw.index),
    )
    source_document_url = _coalesce_series(raw, ["source_document_url", "source_url"], pd.NA).astype("object")
    source_document_url = source_document_url.where(
        source_document_url.notna(),
        pd.Series([_SOURCE_CATALOG.get(key, {}).get("source_url", "") for key in source_key], index=raw.index),
    )
    source_document_title = _coalesce_series(raw, ["source_document_title", "document_title"], pd.NA).astype("object")
    source_reference = _coalesce_series(raw, ["source_reference", "reference"], pd.NA).astype("object")
    source_published_utc = _parse_timestamp_series(
        _coalesce_series(raw, ["source_published_utc", "published_utc", "published_at"]),
        timezone_values=_coalesce_series(raw, ["source_published_timezone"], DEFAULT_PERIOD_TIMEZONE),
    )
    source_published_date = pd.to_datetime(
        _coalesce_series(raw, ["source_published_date", "published_date"]),
        errors="coerce",
    ).dt.date
    source_published_utc = source_published_utc.where(
        source_published_utc.notna(),
        pd.to_datetime(source_published_date, errors="coerce", utc=True),
    )

    review_state = _coalesce_series(raw, ["review_state"], DEFAULT_REVIEW_STATE).map(_normalize_review_state)
    reviewed_evidence_tier = _coalesce_series(
        raw,
        ["reviewed_evidence_tier", "evidence_tier"],
        DEFAULT_REVIEWED_EVIDENCE_TIER,
    ).astype("object")
    capacity_policy_action = pd.Series(
        [
            _normalize_capacity_policy_action(value, state)
            for value, state in zip(_coalesce_series(raw, ["capacity_policy_action"], pd.NA), review_state)
        ],
        index=raw.index,
        dtype="object",
    )
    reviewed_tier_accepted_flag = review_state.eq("accepted_reviewed_tier")
    explicit_gate_state = _coalesce_series(raw, ["reviewed_gate_state", "gate_state"], pd.NA).map(
        _normalize_explicit_gate_state
    )
    reviewed_capacity_limit_mw = pd.to_numeric(
        _coalesce_series(raw, ["capacity_limit_mw", "available_capacity_mw"]),
        errors="coerce",
    )
    reviewed_gate_fraction = pd.to_numeric(_coalesce_series(raw, ["gate_fraction"], pd.NA), errors="coerce")
    source_revision_rank = pd.to_numeric(
        _coalesce_series(raw, ["source_revision_rank", "revision_rank"], 1),
        errors="coerce",
    ).fillna(1).astype(int)
    review_note = _coalesce_series(raw, ["review_note", "note"], pd.NA).astype("object")

    frame = pd.DataFrame(
        {
            "cluster_key": cluster_key,
            "parent_region": parent_region,
            "hub_key": hub_key,
            "review_state": review_state,
            "reviewed_evidence_tier": reviewed_evidence_tier,
            "reviewed_tier_accepted_flag": reviewed_tier_accepted_flag,
            "capacity_policy_action": capacity_policy_action,
            "reviewed_gate_state": explicit_gate_state,
            "period_start_utc": period_start,
            "period_end_utc": period_end,
            "period_timezone": period_timezone,
            "reviewed_capacity_limit_mw": reviewed_capacity_limit_mw,
            "reviewed_gate_fraction": reviewed_gate_fraction,
            "source_provider": source_provider,
            "source_family": source_family,
            "source_key": source_key,
            "source_label": source_label,
            "source_document_title": source_document_title,
            "source_document_url": source_document_url,
            "source_reference": source_reference,
            "source_published_utc": source_published_utc,
            "source_published_date": source_published_date,
            "source_revision_rank": source_revision_rank,
            "review_note": review_note,
        }
    )
    cluster_meta = _CLUSTER_LOOKUP.rename(columns={"approx_capacity_mw": "approx_cluster_capacity_mw"}).copy()
    frame = frame.merge(
        cluster_meta[["cluster_key", "cluster_label", "parent_region", "approx_cluster_capacity_mw"]],
        on="cluster_key",
        how="left",
        suffixes=("", "_lookup"),
    )
    if "parent_region_lookup" in frame.columns:
        frame["parent_region"] = frame["parent_region"].where(frame["parent_region"].notna(), frame["parent_region_lookup"])
    frame["cluster_label"] = frame.get("cluster_label")
    if "cluster_label_lookup" in frame.columns:
        frame["cluster_label"] = frame["cluster_label"].where(frame["cluster_label"].notna(), frame["cluster_label_lookup"])
    frame = frame.drop(columns=["parent_region_lookup", "cluster_label_lookup"], errors="ignore")
    frame = frame.merge(_REGION_CAPACITY_LOOKUP, on="parent_region", how="left")
    frame = frame.merge(_HUB_LOOKUP, on="hub_key", how="left")
    frame["scope_granularity"] = np.where(frame["cluster_key"].notna(), "cluster_hub", "parent_region_hub")
    frame["scope_key"] = np.where(frame["cluster_key"].notna(), frame["cluster_key"], frame["parent_region"])
    frame["reviewed_scope"] = "gb_internal_transfer_reviewed_period"
    frame["review_group_key"] = (
        frame["source_key"].astype(str)
        + "|"
        + frame["scope_key"].astype(str)
        + "|"
        + frame["hub_key"].astype(str)
        + "|"
        + frame["period_start_utc"].astype(str)
        + "|"
        + frame["period_end_utc"].astype(str)
    )
    frame["target_is_proxy"] = False

    column_order = list(_empty_reviewed_period_frame().columns)
    for column in column_order:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[column_order].sort_values(["period_start_utc", "scope_key", "hub_key", "source_key"]).reset_index(drop=True)


def load_gb_transfer_reviewed_input(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return _empty_reviewed_period_frame()
    return normalize_gb_transfer_reviewed_input(_read_input_frame(path))


def write_normalized_gb_transfer_reviewed_input(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    normalized = load_gb_transfer_reviewed_input(input_path)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(target_path, index=False)
    return normalized


def build_fact_gb_transfer_reviewed_period(
    start_date: dt.date,
    end_date: dt.date,
    reviewed_input: pd.DataFrame | None = None,
    reviewed_input_path: str | Path | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    frame = (
        normalize_gb_transfer_reviewed_input(reviewed_input)
        if reviewed_input is not None
        else load_gb_transfer_reviewed_input(reviewed_input_path)
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
    return overlapping[column_order].sort_values(["period_start_utc", "scope_key", "hub_key", "source_key"]).reset_index(drop=True)


def build_fact_gb_transfer_review_policy(reviewed_period: pd.DataFrame | None) -> pd.DataFrame:
    if reviewed_period is None or reviewed_period.empty:
        return _empty_review_policy_frame()
    frame = reviewed_period.copy()
    frame["source_published_utc"] = pd.to_datetime(frame["source_published_utc"], utc=True, errors="coerce")
    frame = frame.sort_values(
        ["source_key", "source_published_utc", "source_revision_rank"],
        ascending=[True, False, False],
        na_position="last",
    )
    rows = []
    for source_key, subset in frame.groupby("source_key", dropna=False):
        representative = subset.iloc[0]
        accepted = bool(subset["reviewed_tier_accepted_flag"].fillna(False).any())
        review_state = "accepted_reviewed_tier" if accepted else representative["review_state"]
        capacity_policy_action = representative["capacity_policy_action"]
        if accepted and capacity_policy_action in {"keep_proxy_fallback", "audit_only"}:
            capacity_policy_action = DEFAULT_CAPACITY_POLICY_ACTION
        rows.append(
            {
                "source_key": source_key,
                "source_family": representative["source_family"],
                "source_provider": representative["source_provider"],
                "source_label": representative["source_label"],
                "source_document_url": representative["source_document_url"],
                "policy_scope": "gb_internal_transfer",
                "review_state": review_state,
                "reviewed_evidence_tier": representative["reviewed_evidence_tier"],
                "reviewed_tier_accepted_flag": accepted,
                "capacity_policy_action": capacity_policy_action,
                "policy_note": (
                    "Accepted reviewed internal-transfer evidence may override the proxy gate."
                    if accepted
                    else "This reviewed source stays audit-only and falls back to the proxy gate."
                ),
                "source_lineage": GB_TRANSFER_REVIEWED_PERIOD_TABLE,
            }
        )
    return pd.DataFrame(rows, columns=_empty_review_policy_frame().columns).sort_values("source_key").reset_index(drop=True)


def _expand_reviewed_scope(frame: pd.DataFrame) -> pd.DataFrame:
    cluster_meta = _CLUSTER_LOOKUP.rename(columns={"approx_capacity_mw": "approx_cluster_capacity_mw"}).copy()
    region_meta = _REGION_CAPACITY_LOOKUP.copy()

    cluster_rows = frame[frame["cluster_key"].notna()].copy()
    if not cluster_rows.empty:
        cluster_rows = cluster_rows.drop(columns=["cluster_label", "parent_region", "approx_cluster_capacity_mw"], errors="ignore")
        cluster_rows = cluster_rows.merge(
            cluster_meta[["cluster_key", "cluster_label", "parent_region", "approx_cluster_capacity_mw"]],
            on="cluster_key",
            how="left",
        )

    region_rows = frame[frame["cluster_key"].isna() & frame["parent_region"].notna()].copy()
    if not region_rows.empty:
        region_rows = region_rows.drop(columns=["cluster_key", "cluster_label", "approx_cluster_capacity_mw"], errors="ignore")
        region_rows = region_rows.merge(
            cluster_meta[["cluster_key", "cluster_label", "parent_region", "approx_cluster_capacity_mw"]],
            on="parent_region",
            how="left",
        )
        region_rows = region_rows.merge(region_meta, on="parent_region", how="left")

    expanded = (
        pd.concat([cluster_rows, region_rows], ignore_index=True)
        if not cluster_rows.empty or not region_rows.empty
        else pd.DataFrame()
    )
    if expanded.empty:
        return expanded

    expanded["hub_label"] = expanded["hub_label"].where(
        expanded["hub_label"].notna(),
        expanded["hub_key"].map(_HUB_LOOKUP.set_index("hub_key")["hub_label"]),
    )
    return expanded


def _effective_capacity_limit(row: pd.Series) -> float:
    cluster_capacity = pd.to_numeric(pd.Series([row.get("approx_cluster_capacity_mw")]), errors="coerce").iloc[0]
    region_capacity = pd.to_numeric(pd.Series([row.get("region_approx_capacity_mw")]), errors="coerce").iloc[0]
    capacity_limit = pd.to_numeric(pd.Series([row.get("reviewed_capacity_limit_mw")]), errors="coerce").iloc[0]
    gate_fraction = pd.to_numeric(pd.Series([row.get("reviewed_gate_fraction")]), errors="coerce").iloc[0]

    candidates: list[float] = []
    if pd.notna(cluster_capacity) and pd.notna(gate_fraction):
        candidates.append(float(cluster_capacity) * float(gate_fraction))
    if pd.notna(capacity_limit):
        if row.get("scope_granularity") == "parent_region_hub" and pd.notna(region_capacity) and float(region_capacity) > 0 and pd.notna(cluster_capacity):
            candidates.append(float(capacity_limit) * float(cluster_capacity) / float(region_capacity))
        else:
            candidates.append(float(capacity_limit))
    if not candidates:
        return np.nan
    return float(min(candidates))


def _derive_hourly_gate_state(row: pd.Series) -> str:
    explicit = row.get("reviewed_gate_state")
    if pd.notna(explicit):
        return str(explicit)
    if not bool(row.get("reviewed_tier_accepted_flag", False)):
        return "audit_only"
    limit = pd.to_numeric(pd.Series([row.get("reviewed_capacity_limit_mw")]), errors="coerce").iloc[0]
    cluster_capacity = pd.to_numeric(pd.Series([row.get("approx_cluster_capacity_mw")]), errors="coerce").iloc[0]
    if pd.notna(limit) and float(limit) <= 0:
        return "blocked_reviewed_capacity"
    if pd.notna(limit) and pd.notna(cluster_capacity) and float(cluster_capacity) > 0 and float(limit) >= 0.95 * float(cluster_capacity):
        return "reviewed_pass"
    if pd.notna(limit):
        return "reviewed_pass_restricted"
    return "reviewed_pass"


def build_fact_gb_transfer_reviewed_hourly(
    start_date: dt.date,
    end_date: dt.date,
    reviewed_period: pd.DataFrame | None = None,
    review_policy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if reviewed_period is None or reviewed_period.empty:
        return _empty_reviewed_hourly_frame()

    frame = reviewed_period.copy()
    policy = review_policy.copy() if review_policy is not None else _empty_review_policy_frame()
    if not policy.empty:
        policy = policy[
            [
                "source_key",
                "review_state",
                "reviewed_evidence_tier",
                "reviewed_tier_accepted_flag",
                "capacity_policy_action",
            ]
        ].drop_duplicates(subset=["source_key"])
        frame = frame.drop(
            columns=["review_state", "reviewed_evidence_tier", "reviewed_tier_accepted_flag", "capacity_policy_action"],
            errors="ignore",
        ).merge(policy, on="source_key", how="left")

    frame["period_start_utc"] = pd.to_datetime(frame["period_start_utc"], utc=True, errors="coerce")
    frame["period_end_utc"] = pd.to_datetime(frame["period_end_utc"], utc=True, errors="coerce")
    frame["source_published_utc"] = pd.to_datetime(frame["source_published_utc"], utc=True, errors="coerce")
    frame = frame[frame["period_end_utc"].gt(frame["period_start_utc"])].copy()
    if frame.empty:
        return _empty_reviewed_hourly_frame()

    expanded = _expand_reviewed_scope(frame)
    if expanded.empty:
        return _empty_reviewed_hourly_frame()

    window_start, window_end = _requested_window_utc(start_date, end_date)
    hours = pd.DataFrame({"interval_start_utc": pd.date_range(start=window_start, end=window_end, freq="h", inclusive="left")})
    hours["interval_end_utc"] = hours["interval_start_utc"] + pd.Timedelta(hours=1)

    joined = hours.assign(_join_key=1).merge(expanded.assign(_join_key=1), on="_join_key", how="inner").drop(columns="_join_key")
    joined = joined[
        joined["interval_start_utc"].lt(joined["period_end_utc"])
        & joined["interval_end_utc"].gt(joined["period_start_utc"])
    ].copy()
    if joined.empty:
        return _empty_reviewed_hourly_frame()

    joined["reviewed_capacity_limit_mw"] = joined.apply(_effective_capacity_limit, axis=1)
    cluster_capacity = pd.to_numeric(joined["approx_cluster_capacity_mw"], errors="coerce")
    joined["reviewed_gate_fraction"] = np.where(
        cluster_capacity.gt(0),
        pd.to_numeric(joined["reviewed_capacity_limit_mw"], errors="coerce") / cluster_capacity,
        np.nan,
    )
    joined["reviewed_gate_state"] = joined.apply(_derive_hourly_gate_state, axis=1)
    joined["scope_rank"] = joined["scope_granularity"].map({"cluster_hub": 0, "parent_region_hub": 1}).fillna(9)

    within_family = joined.sort_values(
        [
            "interval_start_utc",
            "cluster_key",
            "hub_key",
            "source_family",
            "review_group_key",
            "source_published_utc",
            "source_revision_rank",
        ],
        ascending=[True, True, True, True, True, False, False],
        na_position="last",
    )
    within_family = within_family.groupby(
        ["interval_start_utc", "cluster_key", "hub_key", "source_family", "review_group_key"],
        as_index=False,
    ).first()

    within_family["_limit_sort"] = pd.to_numeric(within_family["reviewed_capacity_limit_mw"], errors="coerce").fillna(np.inf)
    resolved = within_family.sort_values(
        [
            "interval_start_utc",
            "cluster_key",
            "hub_key",
            "scope_rank",
            "_limit_sort",
            "source_published_utc",
            "source_revision_rank",
        ],
        ascending=[True, True, True, True, True, False, False],
        na_position="last",
    )
    group_columns = ["interval_start_utc", "cluster_key", "hub_key"]
    representative = resolved.groupby(group_columns, as_index=False).first()
    source_count = resolved.groupby(group_columns)["source_key"].nunique().reset_index(name="reviewed_source_count")
    representative = representative.merge(source_count, on=group_columns, how="left")
    representative = representative.drop(
        columns=[
            "_limit_sort",
            "scope_rank",
            "period_start_utc",
            "period_end_utc",
            "period_timezone",
            "target_is_proxy",
            "approx_cluster_capacity_mw",
            "region_approx_capacity_mw",
        ],
        errors="ignore",
    )

    column_order = list(_empty_reviewed_hourly_frame().columns)
    for column in column_order:
        if column not in representative.columns:
            representative[column] = pd.NA
    return representative[column_order].sort_values(group_columns).reset_index(drop=True)


def materialize_gb_transfer_reviewed_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    reviewed_input: pd.DataFrame | None = None,
    reviewed_input_path: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    period = build_fact_gb_transfer_reviewed_period(
        start_date=start_date,
        end_date=end_date,
        reviewed_input=reviewed_input,
        reviewed_input_path=reviewed_input_path,
    )
    policy = build_fact_gb_transfer_review_policy(period)
    hourly = build_fact_gb_transfer_reviewed_hourly(
        start_date=start_date,
        end_date=end_date,
        reviewed_period=period,
        review_policy=policy,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    period.to_csv(output_path / f"{GB_TRANSFER_REVIEWED_PERIOD_TABLE}.csv", index=False)
    policy.to_csv(output_path / f"{GB_TRANSFER_REVIEW_POLICY_TABLE}.csv", index=False)
    hourly.to_csv(output_path / f"{GB_TRANSFER_REVIEWED_HOURLY_TABLE}.csv", index=False)
    return {
        GB_TRANSFER_REVIEWED_PERIOD_TABLE: period,
        GB_TRANSFER_REVIEW_POLICY_TABLE: policy,
        GB_TRANSFER_REVIEWED_HOURLY_TABLE: hourly,
    }


def write_temp_normalized_gb_transfer_reviewed_input(raw_text: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / "gb_transfer_reviewed_input.txt"
        temp_path.write_text(raw_text, encoding="utf-8")
        return load_gb_transfer_reviewed_input(temp_path)
