from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from asset_mapping import cluster_frame
from curtailment_signals import fetch_constraint_daily, fetch_wind_split_half_hourly, build_regional_curtailment_hourly_proxy
from system_balance_market_state import SYSTEM_BALANCE_MARKET_STATE_TABLE


CURTAILMENT_OPPORTUNITY_TABLE = "fact_curtailment_opportunity_hourly"
VALID_OPPORTUNITY_TRUTH_PROFILES = {"proxy", "research", "precision", "all"}


def _empty_curtailment_opportunity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "cluster_mapping_confidence",
            "cluster_connection_context",
            "cluster_preferred_hub_candidates",
            "cluster_curation_version",
            "hub_key",
            "hub_label",
            "route_name",
            "route_label",
            "route_border_key",
            "route_target_zone",
            "route_price_score_eur_per_mwh",
            "route_price_feasible_flag",
            "route_price_bottleneck",
            "upstream_market_state_feed_available_flag",
            "upstream_market_state",
            "upstream_forward_price_eur_per_mwh",
            "upstream_day_ahead_price_eur_per_mwh",
            "upstream_intraday_price_eur_per_mwh",
            "upstream_imbalance_price_eur_per_mwh",
            "upstream_forward_to_day_ahead_spread_eur_per_mwh",
            "upstream_day_ahead_to_intraday_spread_eur_per_mwh",
            "upstream_forward_to_day_ahead_spread_bucket",
            "upstream_day_ahead_to_intraday_spread_bucket",
            "upstream_market_state_source_provider",
            "upstream_market_state_source_family",
            "upstream_market_state_source_key",
            "upstream_market_state_source_published_utc",
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
            "system_balance_source_provider",
            "system_balance_source_family",
            "system_balance_source_key",
            "system_balance_source_dataset_keys",
            "system_balance_source_published_utc",
            "route_delivery_tier",
            "route_delivery_signal",
            "route_delivery_reason",
            "deliverable_mw_proxy",
            "deliverable_route_score_eur_per_mwh",
            "internal_transfer_evidence_tier",
            "internal_transfer_gate_state",
            "internal_transfer_capacity_limit_mw",
            "internal_transfer_source_provider",
            "internal_transfer_source_family",
            "internal_transfer_source_key",
            "curtailment_source_tier",
            "curtailment_truth_profile",
            "curtailment_source_target_is_proxy",
            "curtailment_proxy_mwh",
            "curtailment_proxy_cost_gbp",
            "curtailment_truth_mwh",
            "curtailment_truth_dispatch_mwh_lower_bound",
            "curtailment_truth_half_hour_count",
            "curtailment_truth_bmu_count",
            "curtailment_selected_mwh",
            "curtailment_present_flag",
            "export_candidate_flag",
            "opportunity_deliverable_mwh",
            "opportunity_spill_mwh",
            "opportunity_gross_value_eur",
            "opportunity_state",
            "connector_capacity_tight_now_flag",
            "market_knew_connector_restriction_flag",
            "connector_notice_market_state",
            "connector_notice_state",
            "connector_notice_known_flag",
            "connector_notice_active_flag",
            "connector_notice_upcoming_flag",
            "connector_notice_hours_until_start",
            "connector_notice_hours_since_publication",
            "connector_notice_lead_time_hours",
            "connector_notice_revision_count",
            "connector_notice_source_key",
            "connector_itl_state",
            "connector_itl_capacity_limit_mw",
            "connector_itl_source_key",
            "connector_gate_state",
            "connector_capacity_evidence_tier",
            "reviewed_publication_state",
            "source_lineage",
        ]
    )


def _coerce_bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(default)
    return values.where(values.notna(), default).astype(bool)


def _scheme_year_label(value: dt.date) -> str:
    start_year = value.year if value.month >= 4 else value.year - 1
    return f"{start_year}-{start_year + 1}"


def _fetch_constraint_daily_for_range(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    labels: list[str] = []
    cursor = start_date
    while cursor <= end_date:
        label = _scheme_year_label(cursor)
        if label not in labels:
            labels.append(label)
        cursor += dt.timedelta(days=1)

    frames = []
    for label in labels:
        frame = fetch_constraint_daily(label)
        frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fetch_cluster_curtailment_proxy_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    constraint_daily = _fetch_constraint_daily_for_range(start_date, end_date)
    if constraint_daily.empty:
        return pd.DataFrame()
    wind_split = fetch_wind_split_half_hourly(start_date, end_date)
    if wind_split.empty:
        return pd.DataFrame()
    return build_regional_curtailment_hourly_proxy(constraint_daily, wind_split)


def _build_cluster_truth_hourly(
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame | None,
    truth_profile: str,
) -> pd.DataFrame:
    columns = [
        "interval_start_utc",
        "cluster_key",
        "curtailment_truth_mwh",
        "curtailment_truth_dispatch_mwh_lower_bound",
        "curtailment_truth_half_hour_count",
        "curtailment_truth_bmu_count",
    ]
    if (
        fact_bmu_curtailment_truth_half_hourly is None
        or fact_bmu_curtailment_truth_half_hourly.empty
        or truth_profile == "proxy"
    ):
        return pd.DataFrame(columns=columns)

    frame = fact_bmu_curtailment_truth_half_hourly.copy()
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
    frame = frame[frame["cluster_key"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    if truth_profile == "research":
        frame = frame[_coerce_bool_series(frame["research_profile_include"])]
    elif truth_profile == "precision":
        frame = frame[_coerce_bool_series(frame["precision_profile_include"])]
    elif truth_profile == "all":
        frame = frame[_coerce_bool_series(frame["lost_energy_estimate_flag"])]
    else:
        raise ValueError(f"unsupported opportunity truth profile: {truth_profile}")

    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["interval_start_utc"] = frame["interval_start_utc"].dt.floor("h")
    grouped = frame.groupby(["interval_start_utc", "cluster_key"], as_index=False).agg(
        curtailment_truth_mwh=("lost_energy_mwh", lambda values: float(pd.Series(values).fillna(0.0).sum())),
        curtailment_truth_dispatch_mwh_lower_bound=(
            "dispatch_down_evidence_mwh_lower_bound",
            lambda values: float(pd.Series(values).fillna(0.0).sum()),
        ),
        curtailment_truth_half_hour_count=("elexon_bm_unit", "count"),
        curtailment_truth_bmu_count=("elexon_bm_unit", "nunique"),
    )
    return grouped.sort_values(["interval_start_utc", "cluster_key"]).reset_index(drop=True)


def build_cluster_curtailment_signal_hourly(
    fact_regional_curtailment_hourly_proxy: pd.DataFrame | None,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame | None = None,
    truth_profile: str = "proxy",
) -> pd.DataFrame:
    if truth_profile not in VALID_OPPORTUNITY_TRUTH_PROFILES:
        raise ValueError(
            f"unsupported opportunity truth profile '{truth_profile}'. "
            f"Expected one of: {', '.join(sorted(VALID_OPPORTUNITY_TRUTH_PROFILES))}"
        )

    proxy_columns = [
        "interval_start_utc",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "curtailment_proxy_mwh",
        "curtailment_proxy_cost_gbp",
    ]
    proxy = pd.DataFrame(columns=proxy_columns)
    if fact_regional_curtailment_hourly_proxy is not None and not fact_regional_curtailment_hourly_proxy.empty:
        proxy = fact_regional_curtailment_hourly_proxy.copy()
        proxy = proxy[proxy["scope_type"].eq("cluster")].copy()
        proxy["interval_start_utc"] = pd.to_datetime(proxy["interval_start_utc"], utc=True, errors="coerce")
        proxy = proxy.rename(
            columns={
                "scope_key": "cluster_key",
                "scope_label": "cluster_label",
                "hourly_curtailment_proxy_mwh": "curtailment_proxy_mwh",
                "hourly_curtailment_proxy_cost_gbp": "curtailment_proxy_cost_gbp",
            }
        )
        proxy = proxy[
            [
                "interval_start_utc",
                "cluster_key",
                "cluster_label",
                "parent_region",
                "curtailment_proxy_mwh",
                "curtailment_proxy_cost_gbp",
            ]
        ].copy()

    truth_hourly = _build_cluster_truth_hourly(
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        truth_profile=truth_profile,
    )

    if proxy.empty and truth_hourly.empty:
        return pd.DataFrame(
            columns=proxy_columns
            + [
                "curtailment_truth_mwh",
                "curtailment_truth_dispatch_mwh_lower_bound",
                "curtailment_truth_half_hour_count",
                "curtailment_truth_bmu_count",
                "curtailment_source_tier",
                "curtailment_truth_profile",
                "curtailment_source_target_is_proxy",
                "curtailment_selected_mwh",
                "curtailment_present_flag",
            ]
        )

    cluster_lookup = cluster_frame()[["cluster_key", "cluster_label", "parent_region"]].drop_duplicates("cluster_key")
    combined = proxy.merge(truth_hourly, on=["interval_start_utc", "cluster_key"], how="outer")
    combined = combined.merge(cluster_lookup, on="cluster_key", how="left", suffixes=("", "_lookup"))
    combined["cluster_label"] = combined["cluster_label"].where(
        combined["cluster_label"].notna(),
        combined["cluster_label_lookup"],
    )
    combined["parent_region"] = combined["parent_region"].where(
        combined["parent_region"].notna(),
        combined["parent_region_lookup"],
    )
    combined = combined.drop(columns=["cluster_label_lookup", "parent_region_lookup"], errors="ignore")

    for column in (
        "curtailment_proxy_mwh",
        "curtailment_proxy_cost_gbp",
        "curtailment_truth_mwh",
        "curtailment_truth_dispatch_mwh_lower_bound",
        "curtailment_truth_half_hour_count",
        "curtailment_truth_bmu_count",
    ):
        if column not in combined.columns:
            combined[column] = np.nan

    truth_available = pd.to_numeric(combined["curtailment_truth_half_hour_count"], errors="coerce").fillna(0).gt(0)
    selected_truth_tier = f"cluster_truth_{truth_profile}" if truth_profile != "proxy" else "regional_proxy"
    combined["curtailment_source_tier"] = np.where(truth_available, selected_truth_tier, "regional_proxy")
    combined["curtailment_truth_profile"] = truth_profile
    combined["curtailment_source_target_is_proxy"] = ~truth_available
    combined["curtailment_selected_mwh"] = np.where(
        truth_available,
        pd.to_numeric(combined["curtailment_truth_mwh"], errors="coerce").fillna(0.0),
        pd.to_numeric(combined["curtailment_proxy_mwh"], errors="coerce").fillna(0.0),
    )
    combined["curtailment_present_flag"] = pd.to_numeric(
        combined["curtailment_selected_mwh"], errors="coerce"
    ).fillna(0.0).gt(0)
    return combined.sort_values(["interval_start_utc", "cluster_key"]).reset_index(drop=True)


def build_fact_curtailment_opportunity_hourly(
    fact_route_score_hourly: pd.DataFrame,
    fact_regional_curtailment_hourly_proxy: pd.DataFrame | None,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame | None = None,
    fact_upstream_market_state_hourly: pd.DataFrame | None = None,
    fact_system_balance_market_state_hourly: pd.DataFrame | None = None,
    truth_profile: str = "proxy",
) -> pd.DataFrame:
    if fact_route_score_hourly is None or fact_route_score_hourly.empty:
        return _empty_curtailment_opportunity_frame()
    if truth_profile not in VALID_OPPORTUNITY_TRUTH_PROFILES:
        raise ValueError(
            f"unsupported opportunity truth profile '{truth_profile}'. "
            f"Expected one of: {', '.join(sorted(VALID_OPPORTUNITY_TRUTH_PROFILES))}"
        )

    route = fact_route_score_hourly.copy()
    route["interval_start_utc"] = pd.to_datetime(route["interval_start_utc"], utc=True, errors="coerce")
    route["interval_end_utc"] = pd.to_datetime(route["interval_end_utc"], utc=True, errors="coerce")
    signal = build_cluster_curtailment_signal_hourly(
        fact_regional_curtailment_hourly_proxy=fact_regional_curtailment_hourly_proxy,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        truth_profile=truth_profile,
    )
    cluster_lookup = cluster_frame()[
        [
            "cluster_key",
            "cluster_label",
            "parent_region",
            "mapping_confidence",
            "connection_context",
            "preferred_hub_candidates",
            "curation_version",
        ]
    ].drop_duplicates("cluster_key")

    fact = route.merge(
        signal,
        on=["interval_start_utc", "cluster_key"],
        how="left",
        suffixes=("", "_signal"),
    )
    fact = fact.merge(cluster_lookup, on=["cluster_key"], how="left", suffixes=("", "_lookup"))
    market_state = (
        fact_upstream_market_state_hourly.copy()
        if fact_upstream_market_state_hourly is not None and not fact_upstream_market_state_hourly.empty
        else pd.DataFrame()
    )
    if not market_state.empty:
        market_state["interval_start_utc"] = pd.to_datetime(
            market_state["interval_start_utc"], utc=True, errors="coerce"
        )
        market_state = market_state.rename(
            columns={
                "forward_price_eur_per_mwh": "upstream_forward_price_eur_per_mwh",
                "day_ahead_price_eur_per_mwh": "upstream_day_ahead_price_eur_per_mwh",
                "intraday_price_eur_per_mwh": "upstream_intraday_price_eur_per_mwh",
                "imbalance_price_eur_per_mwh": "upstream_imbalance_price_eur_per_mwh",
                "forward_to_day_ahead_spread_eur_per_mwh": "upstream_forward_to_day_ahead_spread_eur_per_mwh",
                "day_ahead_to_intraday_spread_eur_per_mwh": "upstream_day_ahead_to_intraday_spread_eur_per_mwh",
                "forward_to_day_ahead_spread_bucket": "upstream_forward_to_day_ahead_spread_bucket",
                "day_ahead_to_intraday_spread_bucket": "upstream_day_ahead_to_intraday_spread_bucket",
                "source_provider": "upstream_market_state_source_provider",
                "source_family": "upstream_market_state_source_family",
                "source_key": "upstream_market_state_source_key",
                "source_published_utc": "upstream_market_state_source_published_utc",
            }
        )
        market_state = market_state.drop_duplicates(["interval_start_utc", "route_name"], keep="last")
        fact = fact.merge(market_state, on=["interval_start_utc", "route_name"], how="left")
    system_balance = (
        fact_system_balance_market_state_hourly.copy()
        if fact_system_balance_market_state_hourly is not None and not fact_system_balance_market_state_hourly.empty
        else pd.DataFrame()
    )
    if not system_balance.empty:
        system_balance["interval_start_utc"] = pd.to_datetime(
            system_balance["interval_start_utc"], utc=True, errors="coerce"
        )
        system_balance = system_balance.drop_duplicates(["interval_start_utc"], keep="last")
        fact = fact.merge(system_balance, on=["interval_start_utc"], how="left", suffixes=("", "_system"))
    fact["cluster_label"] = fact["cluster_label"].where(fact["cluster_label"].notna(), fact.get("cluster_label_signal"))
    fact["parent_region"] = fact["parent_region"].where(fact["parent_region"].notna(), fact.get("parent_region_signal"))
    fact["cluster_label"] = fact["cluster_label"].where(fact["cluster_label"].notna(), fact.get("cluster_label_lookup"))
    fact["parent_region"] = fact["parent_region"].where(fact["parent_region"].notna(), fact.get("parent_region_lookup"))
    fact["cluster_mapping_confidence"] = fact.get("mapping_confidence")
    fact["cluster_connection_context"] = fact.get("connection_context")
    fact["cluster_preferred_hub_candidates"] = fact.get("preferred_hub_candidates")
    fact["cluster_curation_version"] = fact.get("curation_version")
    fact = fact.drop(
        columns=[
            "cluster_label_signal",
            "parent_region_signal",
            "cluster_label_lookup",
            "parent_region_lookup",
            "mapping_confidence",
            "connection_context",
            "preferred_hub_candidates",
            "curation_version",
        ],
        errors="ignore",
    )

    default_values: dict[str, object] = {
        "curtailment_source_tier": "regional_proxy",
        "curtailment_truth_profile": truth_profile,
        "curtailment_source_target_is_proxy": True,
        "curtailment_proxy_mwh": 0.0,
        "curtailment_proxy_cost_gbp": 0.0,
        "curtailment_truth_mwh": 0.0,
        "curtailment_truth_dispatch_mwh_lower_bound": 0.0,
        "curtailment_truth_half_hour_count": 0,
        "curtailment_truth_bmu_count": 0,
        "curtailment_selected_mwh": 0.0,
        "curtailment_present_flag": False,
        "internal_transfer_evidence_tier": "gb_topology_transfer_gate_proxy",
        "internal_transfer_gate_state": "capacity_unknown_reachable",
        "internal_transfer_capacity_limit_mw": 0.0,
        "internal_transfer_source_provider": "proxy",
        "internal_transfer_source_family": "gb_topology_transfer_gate_proxy",
        "internal_transfer_source_key": "gb_topology_transfer_gate_proxy",
        "route_price_score_eur_per_mwh": 0.0,
        "route_price_feasible_flag": False,
        "route_price_bottleneck": pd.NA,
        "upstream_market_state_feed_available_flag": False,
        "upstream_market_state": "no_upstream_feed",
        "upstream_forward_price_eur_per_mwh": np.nan,
        "upstream_day_ahead_price_eur_per_mwh": np.nan,
        "upstream_intraday_price_eur_per_mwh": np.nan,
        "upstream_imbalance_price_eur_per_mwh": np.nan,
        "upstream_forward_to_day_ahead_spread_eur_per_mwh": np.nan,
        "upstream_day_ahead_to_intraday_spread_eur_per_mwh": np.nan,
        "upstream_forward_to_day_ahead_spread_bucket": "spread_unknown",
        "upstream_day_ahead_to_intraday_spread_bucket": "spread_unknown",
        "upstream_market_state_source_provider": pd.NA,
        "upstream_market_state_source_family": pd.NA,
        "upstream_market_state_source_key": pd.NA,
        "upstream_market_state_source_published_utc": pd.NaT,
        "system_balance_feed_available_flag": False,
        "system_balance_known_flag": False,
        "system_balance_active_flag": False,
        "system_balance_state": "no_public_system_balance",
        "system_balance_imbalance_mw": np.nan,
        "system_balance_indicated_demand_mw": np.nan,
        "system_balance_indicated_generation_mw": np.nan,
        "system_balance_indicated_margin_mw": np.nan,
        "system_balance_demand_minus_generation_mw": np.nan,
        "system_balance_margin_ratio": np.nan,
        "system_balance_imbalance_direction_bucket": "imbalance_unknown",
        "system_balance_margin_direction_bucket": "margin_unknown",
        "system_balance_source_provider": pd.NA,
        "system_balance_source_family": pd.NA,
        "system_balance_source_key": pd.NA,
        "system_balance_source_dataset_keys": pd.NA,
        "system_balance_source_published_utc": pd.NaT,
        "connector_notice_state": pd.NA,
        "connector_notice_known_flag": False,
        "connector_notice_active_flag": False,
        "connector_notice_upcoming_flag": False,
        "connector_notice_hours_until_start": np.nan,
        "connector_notice_hours_since_publication": np.nan,
        "connector_notice_lead_time_hours": np.nan,
        "connector_notice_revision_count": np.nan,
        "connector_notice_source_key": pd.NA,
        "connector_itl_state": pd.NA,
        "connector_itl_capacity_limit_mw": np.nan,
        "connector_itl_source_key": pd.NA,
        "connector_gate_state": pd.NA,
        "connector_capacity_evidence_tier": pd.NA,
        "reviewed_publication_state": pd.NA,
    }
    for column, default in default_values.items():
        if column not in fact.columns:
            fact[column] = default

    fact["curtailment_source_target_is_proxy"] = _coerce_bool_series(
        fact["curtailment_source_target_is_proxy"]
    )
    fact["curtailment_present_flag"] = _coerce_bool_series(fact["curtailment_present_flag"])
    fact["connector_notice_known_flag"] = _coerce_bool_series(fact["connector_notice_known_flag"])
    fact["connector_notice_active_flag"] = _coerce_bool_series(fact["connector_notice_active_flag"])
    fact["connector_notice_upcoming_flag"] = _coerce_bool_series(fact["connector_notice_upcoming_flag"])
    fact["route_price_feasible_flag"] = _coerce_bool_series(fact["route_price_feasible_flag"])
    fact["upstream_market_state_feed_available_flag"] = _coerce_bool_series(
        fact["upstream_market_state_feed_available_flag"]
    )
    fact["system_balance_feed_available_flag"] = _coerce_bool_series(fact["system_balance_feed_available_flag"])
    fact["system_balance_known_flag"] = _coerce_bool_series(fact["system_balance_known_flag"])
    fact["system_balance_active_flag"] = _coerce_bool_series(fact["system_balance_active_flag"])

    fact["route_price_score_eur_per_mwh"] = pd.to_numeric(
        fact["route_price_score_eur_per_mwh"], errors="coerce"
    )
    for column in (
        "upstream_forward_price_eur_per_mwh",
        "upstream_day_ahead_price_eur_per_mwh",
        "upstream_intraday_price_eur_per_mwh",
        "upstream_imbalance_price_eur_per_mwh",
        "upstream_forward_to_day_ahead_spread_eur_per_mwh",
        "upstream_day_ahead_to_intraday_spread_eur_per_mwh",
        "system_balance_imbalance_mw",
        "system_balance_indicated_demand_mw",
        "system_balance_indicated_generation_mw",
        "system_balance_indicated_margin_mw",
        "system_balance_demand_minus_generation_mw",
        "system_balance_margin_ratio",
    ):
        fact[column] = pd.to_numeric(fact[column], errors="coerce")
    fact["deliverable_mw_proxy"] = pd.to_numeric(fact["deliverable_mw_proxy"], errors="coerce")
    fact["deliverable_route_score_eur_per_mwh"] = pd.to_numeric(
        fact["deliverable_route_score_eur_per_mwh"], errors="coerce"
    )
    fact["curtailment_selected_mwh"] = pd.to_numeric(fact["curtailment_selected_mwh"], errors="coerce").fillna(0.0)
    fact["curtailment_proxy_mwh"] = pd.to_numeric(fact["curtailment_proxy_mwh"], errors="coerce").fillna(0.0)
    fact["curtailment_proxy_cost_gbp"] = pd.to_numeric(fact["curtailment_proxy_cost_gbp"], errors="coerce").fillna(0.0)
    fact["curtailment_truth_mwh"] = pd.to_numeric(fact["curtailment_truth_mwh"], errors="coerce").fillna(0.0)
    fact["curtailment_truth_dispatch_mwh_lower_bound"] = pd.to_numeric(
        fact["curtailment_truth_dispatch_mwh_lower_bound"],
        errors="coerce",
    ).fillna(0.0)
    fact["curtailment_truth_half_hour_count"] = pd.to_numeric(
        fact["curtailment_truth_half_hour_count"], errors="coerce"
    ).fillna(0).astype(int)
    fact["curtailment_truth_bmu_count"] = pd.to_numeric(
        fact["curtailment_truth_bmu_count"], errors="coerce"
    ).fillna(0).astype(int)

    export_candidate = fact["route_delivery_signal"].isin(
        ["EXPORT_CONFIRMED", "EXPORT_REVIEWED", "EXPORT_CAPACITY_UNKNOWN"]
    )
    fact["export_candidate_flag"] = fact["curtailment_present_flag"] & export_candidate
    fact["opportunity_deliverable_mwh"] = np.where(
        fact["export_candidate_flag"],
        np.minimum(
            fact["curtailment_selected_mwh"],
            fact["deliverable_mw_proxy"].fillna(0.0).clip(lower=0.0),
        ),
        0.0,
    )
    fact["opportunity_spill_mwh"] = (
        fact["curtailment_selected_mwh"] - fact["opportunity_deliverable_mwh"]
    ).clip(lower=0.0)
    fact["opportunity_gross_value_eur"] = (
        fact["opportunity_deliverable_mwh"] * fact["deliverable_route_score_eur_per_mwh"].fillna(0.0)
    )

    reviewed_publication_tight_now = fact["reviewed_publication_state"].isin(["outage", "partial_capacity"]) & (
        ~fact["connector_notice_upcoming_flag"] | fact["connector_notice_active_flag"]
    )
    tight_now_flag = (
        fact["connector_notice_active_flag"]
        | reviewed_publication_tight_now
        | fact["connector_gate_state"].isin(
            ["operator_outage_blocked", "operator_partial_capacity_cap"]
        )
        | fact["route_delivery_tier"].eq("blocked_connector_capacity")
    )
    market_knew_flag = fact["connector_notice_known_flag"] & (
        fact["connector_notice_active_flag"] | fact["connector_notice_upcoming_flag"]
    )
    fact["connector_capacity_tight_now_flag"] = tight_now_flag.astype(bool)
    fact["market_knew_connector_restriction_flag"] = market_knew_flag.astype(bool)
    fact["connector_notice_market_state"] = "no_public_connector_restriction"
    fact.loc[tight_now_flag & market_knew_flag, "connector_notice_market_state"] = (
        "tight_now_and_publicly_known"
    )
    fact.loc[tight_now_flag & ~market_knew_flag, "connector_notice_market_state"] = (
        "tight_now_without_public_notice"
    )
    fact.loc[~tight_now_flag & fact["connector_notice_upcoming_flag"], "connector_notice_market_state"] = (
        "known_upcoming_restriction"
    )

    fact["opportunity_state"] = "no_curtailment"
    curtailment_mask = fact["curtailment_present_flag"]
    fact.loc[curtailment_mask & fact["route_delivery_tier"].eq("no_price_signal"), "opportunity_state"] = (
        "curtailment_no_price_signal"
    )
    fact.loc[curtailment_mask & fact["route_delivery_tier"].eq("blocked_internal_transfer"), "opportunity_state"] = (
        "curtailment_blocked_internal_transfer"
    )
    fact.loc[curtailment_mask & fact["route_delivery_tier"].eq("blocked_connector_capacity"), "opportunity_state"] = (
        "curtailment_blocked_connector_capacity"
    )
    fact.loc[curtailment_mask & fact["route_delivery_tier"].eq("confirmed"), "opportunity_state"] = (
        "curtailment_export_confirmed"
    )
    fact.loc[curtailment_mask & fact["route_delivery_tier"].eq("reviewed"), "opportunity_state"] = (
        "curtailment_export_reviewed"
    )
    fact.loc[curtailment_mask & fact["route_delivery_tier"].eq("capacity_unknown"), "opportunity_state"] = (
        "curtailment_export_capacity_unknown"
    )
    fact.loc[
        curtailment_mask
        & ~fact["route_delivery_tier"].isin(
            [
                "no_price_signal",
                "blocked_internal_transfer",
                "blocked_connector_capacity",
                "confirmed",
                "reviewed",
                "capacity_unknown",
            ]
        ),
        "opportunity_state",
    ] = "curtailment_blocked_other"

    fact["source_lineage"] = "fact_route_score_hourly|fact_regional_curtailment_hourly_proxy"
    truth_lineage_mask = fact["curtailment_source_tier"].ne("regional_proxy")
    fact.loc[truth_lineage_mask, "source_lineage"] = (
        "fact_route_score_hourly|fact_bmu_curtailment_truth_half_hourly|fact_regional_curtailment_hourly_proxy"
    )
    fact.loc[fact["upstream_market_state_feed_available_flag"], "source_lineage"] = (
        fact.loc[fact["upstream_market_state_feed_available_flag"], "source_lineage"]
        + "|fact_upstream_market_state_hourly"
    )
    fact.loc[fact["system_balance_feed_available_flag"], "source_lineage"] = (
        fact.loc[fact["system_balance_feed_available_flag"], "source_lineage"]
        + f"|{SYSTEM_BALANCE_MARKET_STATE_TABLE}"
    )

    keep_columns = list(_empty_curtailment_opportunity_frame().columns)
    for column in keep_columns:
        if column not in fact.columns:
            fact[column] = pd.NA
    return fact[keep_columns].sort_values(
        ["interval_start_utc", "cluster_key", "route_name", "hub_key"]
    ).reset_index(drop=True)


def materialize_curtailment_opportunity_history(
    output_dir: str | Path,
    fact_route_score_hourly: pd.DataFrame,
    fact_regional_curtailment_hourly_proxy: pd.DataFrame | None,
    fact_bmu_curtailment_truth_half_hourly: pd.DataFrame | None = None,
    fact_upstream_market_state_hourly: pd.DataFrame | None = None,
    fact_system_balance_market_state_hourly: pd.DataFrame | None = None,
    truth_profile: str = "proxy",
) -> Dict[str, pd.DataFrame]:
    fact = build_fact_curtailment_opportunity_hourly(
        fact_route_score_hourly=fact_route_score_hourly,
        fact_regional_curtailment_hourly_proxy=fact_regional_curtailment_hourly_proxy,
        fact_bmu_curtailment_truth_half_hourly=fact_bmu_curtailment_truth_half_hourly,
        fact_upstream_market_state_hourly=fact_upstream_market_state_hourly,
        fact_system_balance_market_state_hourly=fact_system_balance_market_state_hourly,
        truth_profile=truth_profile,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fact.to_csv(output_path / f"{CURTAILMENT_OPPORTUNITY_TABLE}.csv", index=False)
    frames = {CURTAILMENT_OPPORTUNITY_TABLE: fact}
    if fact_upstream_market_state_hourly is not None and not fact_upstream_market_state_hourly.empty:
        frames["fact_upstream_market_state_hourly"] = fact_upstream_market_state_hourly.copy()
        fact_upstream_market_state_hourly.to_csv(output_path / "fact_upstream_market_state_hourly.csv", index=False)
    if fact_system_balance_market_state_hourly is not None and not fact_system_balance_market_state_hourly.empty:
        frames[SYSTEM_BALANCE_MARKET_STATE_TABLE] = fact_system_balance_market_state_hourly.copy()
        fact_system_balance_market_state_hourly.to_csv(
            output_path / f"{SYSTEM_BALANCE_MARKET_STATE_TABLE}.csv",
            index=False,
        )
    return frames
