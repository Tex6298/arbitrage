from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from interconnector_flow import (
    BORDER_FLOW_SPECS,
    BorderFlowSpec,
    ENTSOE_ENDPOINT,
    GB_DOMAIN_EIC,
    _direction_label,
    _empty_flow_frame,
    _entsoe_token_from_env,
    _expand_period_points,
    _fetch_entsoe_payload,
    _findtext_first,
    _hub_keys_for_target_zone,
    _hub_labels_for_target_zone,
    _iso_interval,
    _parse_entsoe_error,
    _parse_resolution_to_timedelta,
    _utc_window_for_local_date_range,
    parse_iso_date,
)
import urllib.parse
import xml.etree.ElementTree as ET


INTERCONNECTOR_CAPACITY_TABLE = "fact_interconnector_capacity_hourly"
INTERCONNECTOR_CAPACITY_REVIEWED_TABLE = "fact_interconnector_capacity_reviewed_hourly"
INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE = "fact_interconnector_capacity_source_audit_daily"
INTERCONNECTOR_CAPACITY_AUDIT_VARIANT_TABLE = "fact_interconnector_capacity_source_audit_variant"
INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE = "fact_interconnector_capacity_review_policy"
INTERCONNECTOR_CAPACITY_SOURCE_KEY = "entsoe_offered_capacity"
INTERCONNECTOR_CAPACITY_SOURCE_LABEL = "ENTSO-E offered capacity"
INTERCONNECTOR_CAPACITY_DOCUMENT_TYPE = "A31"
INTERCONNECTOR_CAPACITY_ARTICLE = "11.1.A"
INTERCONNECTOR_CAPACITY_AUCTION_TYPE = "A01"
INTERCONNECTOR_CAPACITY_CONTRACT_TYPE = "A01"
REVIEWED_EXPLICIT_DAILY_BORDERS = frozenset({"GB-NL", "GB-BE", "GB-DK1"})


@dataclass(frozen=True)
class CapacityAuditVariant:
    variant_key: str
    variant_label: str
    document_type: str
    auction_type: str | None
    contract_market_agreement_type: str | None


CAPACITY_AUDIT_VARIANTS: Tuple[CapacityAuditVariant, ...] = (
    CapacityAuditVariant(
        variant_key="a31_implicit_daily",
        variant_label="A31 implicit daily offered capacity",
        document_type="A31",
        auction_type="A01",
        contract_market_agreement_type="A01",
    ),
    CapacityAuditVariant(
        variant_key="a31_implicit_weekly",
        variant_label="A31 implicit weekly agreed capacity",
        document_type="A31",
        auction_type="A01",
        contract_market_agreement_type="A02",
    ),
    CapacityAuditVariant(
        variant_key="a31_implicit_monthly",
        variant_label="A31 implicit monthly agreed capacity",
        document_type="A31",
        auction_type="A01",
        contract_market_agreement_type="A03",
    ),
    CapacityAuditVariant(
        variant_key="a31_implicit_yearly",
        variant_label="A31 implicit yearly agreed capacity",
        document_type="A31",
        auction_type="A01",
        contract_market_agreement_type="A04",
    ),
    CapacityAuditVariant(
        variant_key="a31_implicit_intraday",
        variant_label="A31 implicit intraday agreed capacity",
        document_type="A31",
        auction_type="A01",
        contract_market_agreement_type="A07",
    ),
    CapacityAuditVariant(
        variant_key="a31_explicit_daily",
        variant_label="A31 explicit daily agreed capacity",
        document_type="A31",
        auction_type="A02",
        contract_market_agreement_type="A01",
    ),
)


def _empty_capacity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "source_key",
            "source_label",
            "source_document_type",
            "source_article",
            "source_resolution",
            "normalized_resolution",
            "target_is_proxy",
            "capacity_scope",
            "hub_assignment_mode",
            "hub_assignment_is_proxy",
            "border_key",
            "border_label",
            "target_zone",
            "neighbor_domain_key",
            "gb_domain_eic",
            "neighbor_domain_eic",
            "out_domain_eic",
            "in_domain_eic",
            "direction_key",
            "direction_label",
            "candidate_hub_keys",
            "candidate_hub_labels",
            "candidate_hub_count",
            "auction_type",
            "contract_market_agreement_type",
            "business_type",
            "offered_capacity_mw",
            "offered_capacity_mwh",
        ]
    )


def _normalize_capacity_rows_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_capacity_frame()

    normalized = frame.copy()
    normalized["interval_hours"] = (
        normalized["interval_end_utc"] - normalized["interval_start_utc"]
    ) / pd.Timedelta(hours=1)
    normalized["hour_start_utc"] = normalized["interval_start_utc"].dt.floor("h")
    identity_columns = [
        "source_key",
        "source_label",
        "source_document_type",
        "source_article",
        "target_is_proxy",
        "capacity_scope",
        "hub_assignment_mode",
        "hub_assignment_is_proxy",
        "border_key",
        "border_label",
        "target_zone",
        "neighbor_domain_key",
        "gb_domain_eic",
        "neighbor_domain_eic",
        "out_domain_eic",
        "in_domain_eic",
        "direction_key",
        "direction_label",
        "candidate_hub_keys",
        "candidate_hub_labels",
        "candidate_hub_count",
        "auction_type",
        "contract_market_agreement_type",
        "business_type",
    ]
    rows = []
    grouped = normalized.groupby(identity_columns + ["hour_start_utc"], dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(identity_columns + ["hour_start_utc"], keys))
        hour_start_utc = pd.Timestamp(record.pop("hour_start_utc"))
        hour_end_utc = hour_start_utc + pd.Timedelta(hours=1)
        hour_start_local = hour_start_utc.tz_convert("Europe/London")
        hour_end_local = hour_end_utc.tz_convert("Europe/London")
        source_resolutions = sorted({str(value) for value in group["source_resolution"].dropna().unique()})
        offered_capacity_mw = float(
            (group["offered_capacity_mw"].astype(float) * group["interval_hours"].astype(float)).sum()
            / float(group["interval_hours"].astype(float).sum())
        )
        rows.append(
            {
                **record,
                "date": hour_start_local.date(),
                "interval_start_local": hour_start_local,
                "interval_end_local": hour_end_local,
                "interval_start_utc": hour_start_utc,
                "interval_end_utc": hour_end_utc,
                "source_resolution": ",".join(source_resolutions),
                "normalized_resolution": "PT60M",
                "offered_capacity_mw": offered_capacity_mw,
                "offered_capacity_mwh": float(
                    group["offered_capacity_mw"].astype(float).mul(group["interval_hours"].astype(float)).sum()
                ),
            }
        )

    column_order = list(_empty_capacity_frame().columns)
    return pd.DataFrame(rows)[column_order].sort_values(
        ["interval_start_utc", "border_key", "direction_key"]
    ).reset_index(drop=True)


def parse_entsoe_interconnector_capacity_xml(
    xml_bytes: bytes,
    spec: BorderFlowSpec,
    direction_key: str,
    requested_start_utc: dt.datetime,
    requested_end_utc: dt.datetime,
) -> pd.DataFrame:
    error_text = _parse_entsoe_error(xml_bytes)
    if error_text and "no matching data" in error_text.lower():
        return _empty_capacity_frame()

    root = ET.fromstring(xml_bytes)
    namespace = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"ns": namespace} if namespace else {}
    time_series = root.findall(".//ns:TimeSeries", ns) if ns else root.findall(".//TimeSeries")
    if not time_series:
        if error_text:
            raise RuntimeError(f"{spec.border_key} {direction_key} returned no usable capacity series: {error_text}")
        raise RuntimeError(f"{spec.border_key} {direction_key} returned no usable capacity series")

    hub_keys = _hub_keys_for_target_zone(spec.target_zone)
    hub_labels = _hub_labels_for_target_zone(spec.target_zone)
    hub_assignment_mode = "aggregate_border_candidate_hubs" if len(hub_keys) > 1 else "single_hub_border_proxy"

    rows = []
    for ts in time_series:
        curve_type = _findtext_first(ts, ns, ("ns:curveType", "curveType"))
        out_domain_eic = _findtext_first(
            ts,
            ns,
            (
                "ns:outBiddingZone_Domain.mRID",
                "ns:out_Domain.mRID",
                "ns:outArea_Domain.mRID",
            ),
        )
        in_domain_eic = _findtext_first(
            ts,
            ns,
            (
                "ns:inBiddingZone_Domain.mRID",
                "ns:in_Domain.mRID",
                "ns:inArea_Domain.mRID",
            ),
        )
        auction_type = _findtext_first(ts, ns, ("ns:auction.type", "auction.type")) or INTERCONNECTOR_CAPACITY_AUCTION_TYPE
        contract_type = _findtext_first(
            ts,
            ns,
            ("ns:contract_MarketAgreement.type", "contract_MarketAgreement.type"),
        ) or INTERCONNECTOR_CAPACITY_CONTRACT_TYPE
        business_type = _findtext_first(ts, ns, ("ns:businessType", "businessType"))

        for period in ts.findall(".//ns:Period", ns) if ns else ts.findall(".//Period"):
            start_text = _findtext_first(period, ns, ("ns:timeInterval/ns:start", "timeInterval/start"))
            end_text = _findtext_first(period, ns, ("ns:timeInterval/ns:end", "timeInterval/end"))
            resolution = _findtext_first(period, ns, ("ns:resolution", "resolution")) or "PT60M"
            if not start_text or not end_text:
                continue
            step = _parse_resolution_to_timedelta(resolution)
            period_start = pd.to_datetime(start_text, utc=True)
            period_end = pd.to_datetime(end_text, utc=True)

            points = period.findall("ns:Point", ns) if ns else period.findall("Point")
            for interval_start, interval_end, offered_capacity_mw in _expand_period_points(
                period_start=period_start,
                period_end=period_end,
                step=step,
                curve_type=curve_type,
                points=points,
                ns=ns,
            ):
                if interval_start < requested_start_utc or interval_start >= requested_end_utc:
                    continue
                rows.append(
                    {
                        "interval_start_utc": interval_start,
                        "interval_end_utc": interval_end,
                        "source_resolution": resolution,
                        "out_domain_eic": out_domain_eic
                        or (GB_DOMAIN_EIC if direction_key == "gb_to_neighbor" else spec.neighbor_domain_eic),
                        "in_domain_eic": in_domain_eic
                        or (spec.neighbor_domain_eic if direction_key == "gb_to_neighbor" else GB_DOMAIN_EIC),
                        "auction_type": auction_type,
                        "contract_market_agreement_type": contract_type,
                        "business_type": business_type,
                        "offered_capacity_mw": offered_capacity_mw,
                    }
                )

    if not rows:
        return _empty_capacity_frame()

    frame = pd.DataFrame(rows).sort_values("interval_start_utc").drop_duplicates(
        subset=[
            "interval_start_utc",
            "out_domain_eic",
            "in_domain_eic",
            "auction_type",
            "contract_market_agreement_type",
            "business_type",
        ],
        keep="last",
    )
    frame["source_key"] = INTERCONNECTOR_CAPACITY_SOURCE_KEY
    frame["source_label"] = INTERCONNECTOR_CAPACITY_SOURCE_LABEL
    frame["source_document_type"] = INTERCONNECTOR_CAPACITY_DOCUMENT_TYPE
    frame["source_article"] = INTERCONNECTOR_CAPACITY_ARTICLE
    frame["target_is_proxy"] = False
    frame["capacity_scope"] = "aggregate_border_bidding_zone"
    frame["hub_assignment_mode"] = hub_assignment_mode
    frame["hub_assignment_is_proxy"] = True
    frame["border_key"] = spec.border_key
    frame["border_label"] = spec.border_label
    frame["target_zone"] = spec.target_zone
    frame["neighbor_domain_key"] = spec.neighbor_domain_key
    frame["gb_domain_eic"] = GB_DOMAIN_EIC
    frame["neighbor_domain_eic"] = spec.neighbor_domain_eic
    frame["direction_key"] = direction_key
    frame["direction_label"] = _direction_label(spec, direction_key)
    frame["candidate_hub_keys"] = ",".join(hub_keys)
    frame["candidate_hub_labels"] = ",".join(hub_labels)
    frame["candidate_hub_count"] = len(hub_keys)
    return _normalize_capacity_rows_to_hourly(frame)


def fetch_interconnector_capacity_direction(
    spec: BorderFlowSpec,
    direction_key: str,
    start_date: dt.date,
    end_date: dt.date,
    token: str,
    document_type: str = INTERCONNECTOR_CAPACITY_DOCUMENT_TYPE,
    auction_type: str | None = INTERCONNECTOR_CAPACITY_AUCTION_TYPE,
    contract_market_agreement_type: str | None = INTERCONNECTOR_CAPACITY_CONTRACT_TYPE,
) -> pd.DataFrame:
    if direction_key not in {"gb_to_neighbor", "neighbor_to_gb"}:
        raise ValueError(f"unsupported direction_key '{direction_key}'")

    requested_start_utc, requested_end_utc = _utc_window_for_local_date_range(start_date, end_date)
    period_start, period_end = _iso_interval(requested_start_utc, requested_end_utc)
    out_domain = GB_DOMAIN_EIC if direction_key == "gb_to_neighbor" else spec.neighbor_domain_eic
    in_domain = spec.neighbor_domain_eic if direction_key == "gb_to_neighbor" else GB_DOMAIN_EIC
    params = {
        "securityToken": token,
        "documentType": document_type,
        "out_Domain": out_domain,
        "in_Domain": in_domain,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    if auction_type:
        params["auction.Type"] = auction_type
    if contract_market_agreement_type:
        params["contract_MarketAgreement.Type"] = contract_market_agreement_type
    url = ENTSOE_ENDPOINT + "?" + urllib.parse.urlencode(params)
    payload = _fetch_entsoe_payload(
        url,
        f"{spec.border_key} {direction_key} offered_capacity {document_type}/{auction_type or 'na'}/{contract_market_agreement_type or 'na'}",
    )
    return parse_entsoe_interconnector_capacity_xml(
        payload,
        spec=spec,
        direction_key=direction_key,
        requested_start_utc=requested_start_utc,
        requested_end_utc=requested_end_utc,
    )


def build_fact_interconnector_capacity_hourly(
    start_date: dt.date,
    end_date: dt.date,
    token: str,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not token:
        raise RuntimeError("missing ENTSO-E token; set ENTOS_E_TOKEN or ENTSOE_TOKEN")

    frames = []
    for spec in BORDER_FLOW_SPECS:
        for direction_key in ("gb_to_neighbor", "neighbor_to_gb"):
            frame = fetch_interconnector_capacity_direction(
                spec=spec,
                direction_key=direction_key,
                start_date=start_date,
                end_date=end_date,
                token=token,
            )
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return _empty_capacity_frame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            [
                "interval_start_utc",
                "border_key",
                "direction_key",
                "auction_type",
                "contract_market_agreement_type",
            ]
        )
        .reset_index(drop=True)
    )


def materialize_interconnector_capacity_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    token: str | None = None,
) -> Dict[str, pd.DataFrame]:
    resolved_token = token or _entsoe_token_from_env()
    fact = build_fact_interconnector_capacity_hourly(start_date=start_date, end_date=end_date, token=resolved_token)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{INTERCONNECTOR_CAPACITY_TABLE}.csv", index=False)
    return {INTERCONNECTOR_CAPACITY_TABLE: fact}


def _empty_capacity_audit_variant_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "border_key",
            "border_label",
            "target_zone",
            "neighbor_domain_key",
            "direction_key",
            "direction_label",
            "variant_key",
            "variant_label",
            "document_type",
            "auction_type",
            "contract_market_agreement_type",
            "rows_returned",
            "first_interval_start_utc",
            "last_interval_start_utc",
            "published_any_rows",
            "audit_status",
            "audit_note",
            "query_error",
        ]
    )


def _empty_capacity_audit_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "border_key",
            "border_label",
            "target_zone",
            "neighbor_domain_key",
            "direction_key",
            "direction_label",
            "first_pass_rows_returned",
            "alternate_variant_rows_returned",
            "first_published_variant_key",
            "published_variant_keys",
            "audit_status",
            "recommended_gate_policy",
            "audit_note",
        ]
    )


def _empty_capacity_reviewed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "source_key",
            "source_label",
            "source_document_type",
            "source_article",
            "source_resolution",
            "normalized_resolution",
            "target_is_proxy",
            "capacity_scope",
            "hub_assignment_mode",
            "hub_assignment_is_proxy",
            "border_key",
            "border_label",
            "target_zone",
            "neighbor_domain_key",
            "gb_domain_eic",
            "neighbor_domain_eic",
            "out_domain_eic",
            "in_domain_eic",
            "direction_key",
            "direction_label",
            "candidate_hub_keys",
            "candidate_hub_labels",
            "candidate_hub_count",
            "auction_type",
            "contract_market_agreement_type",
            "business_type",
            "offered_capacity_mw",
            "offered_capacity_mwh",
            "review_state",
            "reviewed_evidence_tier",
            "accepted_variant_key",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "review_note",
        ]
    )


def _empty_capacity_review_policy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "border_key",
            "border_label",
            "target_zone",
            "neighbor_domain_key",
            "direction_key",
            "direction_label",
            "audit_status",
            "recommended_gate_policy",
            "first_pass_rows_returned",
            "alternate_variant_rows_returned",
            "first_published_variant_key",
            "published_variant_keys",
            "review_state",
            "reviewed_evidence_tier",
            "accepted_variant_key",
            "reviewed_tier_accepted_flag",
            "capacity_policy_action",
            "review_note",
        ]
    )


def build_interconnector_capacity_source_audit(
    start_date: dt.date,
    end_date: dt.date,
    token: str,
    variants: Iterable[CapacityAuditVariant] = CAPACITY_AUDIT_VARIANTS,
) -> Dict[str, pd.DataFrame]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not token:
        raise RuntimeError("missing ENTSO-E token; set ENTOS_E_TOKEN or ENTSOE_TOKEN")

    variant_rows = []
    for spec in BORDER_FLOW_SPECS:
        for direction_key in ("gb_to_neighbor", "neighbor_to_gb"):
            for variant in variants:
                try:
                    frame = fetch_interconnector_capacity_direction(
                        spec=spec,
                        direction_key=direction_key,
                        start_date=start_date,
                        end_date=end_date,
                        token=token,
                        document_type=variant.document_type,
                        auction_type=variant.auction_type,
                        contract_market_agreement_type=variant.contract_market_agreement_type,
                    )
                    query_error = None
                    rows_returned = int(len(frame))
                    published_any_rows = rows_returned > 0
                    audit_status = "published" if published_any_rows else "no_matching_data"
                    audit_note = (
                        f"Published {rows_returned} hourly rows."
                        if published_any_rows
                        else "No matching rows returned for this official ENTSO-E query variant."
                    )
                    first_interval_start_utc = frame["interval_start_utc"].min() if published_any_rows else pd.NaT
                    last_interval_start_utc = frame["interval_start_utc"].max() if published_any_rows else pd.NaT
                except Exception as exc:
                    rows_returned = 0
                    published_any_rows = False
                    audit_status = "query_error"
                    audit_note = "The official ENTSO-E query variant was rejected or failed."
                    query_error = str(exc)
                    first_interval_start_utc = pd.NaT
                    last_interval_start_utc = pd.NaT
                variant_rows.append(
                    {
                        "border_key": spec.border_key,
                        "border_label": spec.border_label,
                        "target_zone": spec.target_zone,
                        "neighbor_domain_key": spec.neighbor_domain_key,
                        "direction_key": direction_key,
                        "direction_label": _direction_label(spec, direction_key),
                        "variant_key": variant.variant_key,
                        "variant_label": variant.variant_label,
                        "document_type": variant.document_type,
                        "auction_type": variant.auction_type,
                        "contract_market_agreement_type": variant.contract_market_agreement_type,
                        "rows_returned": rows_returned,
                        "first_interval_start_utc": first_interval_start_utc,
                        "last_interval_start_utc": last_interval_start_utc,
                        "published_any_rows": published_any_rows,
                        "audit_status": audit_status,
                        "audit_note": audit_note,
                        "query_error": query_error,
                    }
                )

    variant_frame = (
        pd.DataFrame(variant_rows)
        if variant_rows
        else _empty_capacity_audit_variant_frame()
    )
    if variant_frame.empty:
        return {
            INTERCONNECTOR_CAPACITY_AUDIT_VARIANT_TABLE: _empty_capacity_audit_variant_frame(),
            INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE: _empty_capacity_audit_daily_frame(),
        }

    daily_rows = []
    grouped = variant_frame.groupby(
        ["border_key", "border_label", "target_zone", "neighbor_domain_key", "direction_key", "direction_label"],
        dropna=False,
        sort=True,
    )
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(
            zip(
                ["border_key", "border_label", "target_zone", "neighbor_domain_key", "direction_key", "direction_label"],
                keys,
            )
        )
        first_pass = group[group["variant_key"] == "a31_implicit_daily"]
        first_pass_rows = int(first_pass["rows_returned"].sum()) if not first_pass.empty else 0
        alternate = group[group["variant_key"] != "a31_implicit_daily"]
        alternate_rows = int(alternate["rows_returned"].sum()) if not alternate.empty else 0
        published = group[group["published_any_rows"]]
        query_error_count = int(group["audit_status"].eq("query_error").sum())
        published_variant_keys = ",".join(published["variant_key"].tolist())
        if first_pass_rows > 0:
            audit_status = "first_pass_published"
            recommended_gate_policy = "eligible_first_pass_gate"
            audit_note = "The first-pass offered-capacity query returns data for this border-direction."
        elif not published.empty:
            audit_status = "alternate_variant_published"
            recommended_gate_policy = "audit_before_gate"
            audit_note = "Only alternate official query variants returned rows. Do not use as a full-border gate without review."
        elif query_error_count > 0:
            audit_status = "query_error_or_unpublished"
            recommended_gate_policy = "audit_before_gate"
            audit_note = "One or more official query variants were rejected, and no published rows were found. Do not use as a full-border gate yet."
        else:
            audit_status = "no_variant_published"
            recommended_gate_policy = "capacity_unknown_default"
            audit_note = "No tested official query variant returned rows. Treat capacity as unknown, not blocked."
        first_published_variant_key = published.iloc[0]["variant_key"] if not published.empty else None
        daily_rows.append(
            {
                **record,
                "first_pass_rows_returned": first_pass_rows,
                "alternate_variant_rows_returned": alternate_rows,
                "first_published_variant_key": first_published_variant_key,
                "published_variant_keys": published_variant_keys,
                "audit_status": audit_status,
                "recommended_gate_policy": recommended_gate_policy,
                "audit_note": audit_note,
            }
        )

    daily_frame = pd.DataFrame(daily_rows).sort_values(["border_key", "direction_key"]).reset_index(drop=True)
    variant_frame = variant_frame.sort_values(["border_key", "direction_key", "variant_key"]).reset_index(drop=True)
    return {
        INTERCONNECTOR_CAPACITY_AUDIT_VARIANT_TABLE: variant_frame,
        INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE: daily_frame,
    }


def build_interconnector_capacity_review_policy(
    audit_daily: pd.DataFrame,
) -> pd.DataFrame:
    if audit_daily.empty:
        return _empty_capacity_review_policy_frame()

    rows = []
    for row in audit_daily.to_dict(orient="records"):
        published_variant_keys = str(row.get("published_variant_keys") or "")
        published_variants = {value.strip() for value in published_variant_keys.split(",") if value.strip()}
        border_key = str(row.get("border_key") or "")
        audit_status = str(row.get("audit_status") or "")

        if audit_status == "first_pass_published":
            review_state = "first_pass_direct"
            reviewed_evidence_tier = "first_pass_implicit_daily"
            accepted_variant_key = "a31_implicit_daily"
            reviewed_tier_accepted_flag = True
            capacity_policy_action = "eligible_first_pass_gate"
            review_note = "The first-pass implicit-daily query is already published, so no alternate reviewed tier is needed."
        elif "a31_explicit_daily" in published_variants and border_key in REVIEWED_EXPLICIT_DAILY_BORDERS:
            review_state = "accepted_reviewed_tier"
            reviewed_evidence_tier = "reviewed_explicit_daily"
            accepted_variant_key = "a31_explicit_daily"
            reviewed_tier_accepted_flag = True
            capacity_policy_action = "allow_reviewed_explicit_daily"
            review_note = (
                "The official ENTSO-E explicit-daily variant is acceptable as a reviewed evidence tier for this border,"
                " but it remains distinct from the first-pass direct gate."
            )
        elif "a31_explicit_daily" in published_variants:
            review_state = "open_review_required"
            reviewed_evidence_tier = "none"
            accepted_variant_key = None
            reviewed_tier_accepted_flag = False
            capacity_policy_action = "keep_capacity_unknown"
            review_note = "An explicit-daily variant published, but it is not on the accepted reviewed-tier border list."
        else:
            review_state = "capacity_unknown_default"
            reviewed_evidence_tier = "none"
            accepted_variant_key = None
            reviewed_tier_accepted_flag = False
            capacity_policy_action = "keep_capacity_unknown"
            review_note = "No accepted reviewed-capacity tier exists for this border-direction, so capacity stays unknown."

        rows.append(
            {
                "border_key": row.get("border_key"),
                "border_label": row.get("border_label"),
                "target_zone": row.get("target_zone"),
                "neighbor_domain_key": row.get("neighbor_domain_key"),
                "direction_key": row.get("direction_key"),
                "direction_label": row.get("direction_label"),
                "audit_status": audit_status,
                "recommended_gate_policy": row.get("recommended_gate_policy"),
                "first_pass_rows_returned": row.get("first_pass_rows_returned"),
                "alternate_variant_rows_returned": row.get("alternate_variant_rows_returned"),
                "first_published_variant_key": row.get("first_published_variant_key"),
                "published_variant_keys": row.get("published_variant_keys"),
                "review_state": review_state,
                "reviewed_evidence_tier": reviewed_evidence_tier,
                "accepted_variant_key": accepted_variant_key,
                "reviewed_tier_accepted_flag": reviewed_tier_accepted_flag,
                "capacity_policy_action": capacity_policy_action,
                "review_note": review_note,
            }
        )

    return pd.DataFrame(rows).sort_values(["border_key", "direction_key"]).reset_index(drop=True)


def _capacity_variant_lookup() -> Dict[str, CapacityAuditVariant]:
    return {variant.variant_key: variant for variant in CAPACITY_AUDIT_VARIANTS}


def build_interconnector_capacity_reviewed_hourly(
    start_date: dt.date,
    end_date: dt.date,
    token: str,
    review_policy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not token:
        raise RuntimeError("missing ENTSO-E token; set ENTOS_E_TOKEN or ENTSOE_TOKEN")

    resolved_policy = review_policy
    if resolved_policy is None:
        audit_frames = build_interconnector_capacity_source_audit(start_date=start_date, end_date=end_date, token=token)
        resolved_policy = build_interconnector_capacity_review_policy(
            audit_frames[INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE]
        )

    if resolved_policy.empty:
        return _empty_capacity_reviewed_frame()

    variant_lookup = _capacity_variant_lookup()
    rows = []
    for policy_row in resolved_policy.to_dict(orient="records"):
        if not bool(policy_row.get("reviewed_tier_accepted_flag")):
            continue
        if str(policy_row.get("capacity_policy_action") or "") != "allow_reviewed_explicit_daily":
            continue
        variant_key = str(policy_row.get("accepted_variant_key") or "")
        variant = variant_lookup.get(variant_key)
        if variant is None:
            continue
        border_key = str(policy_row.get("border_key") or "")
        direction_key = str(policy_row.get("direction_key") or "")
        spec = next((candidate for candidate in BORDER_FLOW_SPECS if candidate.border_key == border_key), None)
        if spec is None:
            continue
        frame = fetch_interconnector_capacity_direction(
            spec=spec,
            direction_key=direction_key,
            start_date=start_date,
            end_date=end_date,
            token=token,
            document_type=variant.document_type,
            auction_type=variant.auction_type,
            contract_market_agreement_type=variant.contract_market_agreement_type,
        )
        if frame.empty:
            continue
        reviewed = frame.copy()
        reviewed["source_key"] = "entsoe_offered_capacity_reviewed"
        reviewed["source_label"] = "ENTSO-E offered capacity reviewed tier"
        reviewed["review_state"] = policy_row.get("review_state")
        reviewed["reviewed_evidence_tier"] = policy_row.get("reviewed_evidence_tier")
        reviewed["accepted_variant_key"] = policy_row.get("accepted_variant_key")
        reviewed["reviewed_tier_accepted_flag"] = policy_row.get("reviewed_tier_accepted_flag")
        reviewed["capacity_policy_action"] = policy_row.get("capacity_policy_action")
        reviewed["review_note"] = policy_row.get("review_note")
        rows.append(reviewed)

    if not rows:
        return _empty_capacity_reviewed_frame()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["interval_start_utc", "border_key", "direction_key", "accepted_variant_key"])
        .reset_index(drop=True)
    )


def materialize_interconnector_capacity_source_audit(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    token: str | None = None,
) -> Dict[str, pd.DataFrame]:
    resolved_token = token or _entsoe_token_from_env()
    frames = build_interconnector_capacity_source_audit(start_date=start_date, end_date=end_date, token=resolved_token)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(output_path / f"{table_name}.csv", index=False)
    return frames


def materialize_interconnector_capacity_review_policy(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
    token: str | None = None,
) -> Dict[str, pd.DataFrame]:
    resolved_token = token or _entsoe_token_from_env()
    frames = build_interconnector_capacity_source_audit(start_date=start_date, end_date=end_date, token=resolved_token)
    review_policy = build_interconnector_capacity_review_policy(frames[INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE])
    reviewed_hourly = build_interconnector_capacity_reviewed_hourly(
        start_date=start_date,
        end_date=end_date,
        token=resolved_token,
        review_policy=review_policy,
    )
    materialized = dict(frames)
    materialized[INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE] = review_policy
    materialized[INTERCONNECTOR_CAPACITY_REVIEWED_TABLE] = reviewed_hourly
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for table_name, frame in materialized.items():
        frame.to_csv(output_path / f"{table_name}.csv", index=False)
    return materialized
