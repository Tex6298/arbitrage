import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opportunity_backtest import (
    BACKTEST_PREDICTION_TABLE,
    BACKTEST_SUMMARY_SLICE_TABLE,
    BACKTEST_TOP_ERROR_TABLE,
    DRIFT_WINDOW_TABLE,
    MODEL_GB_NL_REVIEWED_SPECIALIST_V3,
    MODEL_GROUP_MEAN_NOTICE_V1,
    MODEL_POTENTIAL_RATIO_V2,
    build_fact_backtest_prediction_hourly,
    build_fact_backtest_summary_slice,
    build_fact_backtest_top_error_hourly,
    build_fact_drift_window,
    materialize_opportunity_backtest,
    summarize_backtest_prediction_hourly,
)


def _opportunity_row(
    interval_start_utc: str,
    cluster_key: str,
    route_name: str,
    hub_key: str,
    route_delivery_tier: str,
    connector_notice_market_state: str,
    opportunity_deliverable_mwh: float,
    deliverable_route_score_eur_per_mwh: float,
    *,
    curtailment_selected_mwh: float | None = None,
    deliverable_mw_proxy: float | None = None,
    curtailment_source_tier: str = "regional_proxy",
    internal_transfer_evidence_tier: str = "gb_topology_transfer_gate_proxy",
    internal_transfer_gate_state: str = "capacity_unknown_reachable",
    internal_transfer_source_family: str | None = None,
    internal_transfer_source_key: str | None = None,
    connector_itl_state: str = "no_public_itl_restriction",
    connector_itl_source_key: str | None = None,
    route_price_score_eur_per_mwh: float | None = None,
    route_price_feasible_flag: bool | None = None,
    route_price_bottleneck: str | None = None,
    upstream_market_state_feed_available_flag: bool = False,
    upstream_market_state: str = "no_upstream_feed",
    upstream_day_ahead_price_eur_per_mwh: float | None = None,
    upstream_intraday_price_eur_per_mwh: float | None = None,
    upstream_forward_price_eur_per_mwh: float | None = None,
    upstream_day_ahead_to_intraday_spread_bucket: str = "spread_unknown",
    upstream_forward_to_day_ahead_spread_bucket: str = "spread_unknown",
    system_balance_feed_available_flag: bool = False,
    system_balance_known_flag: bool = False,
    system_balance_active_flag: bool = False,
    system_balance_state: str = "no_public_system_balance",
    system_balance_imbalance_direction_bucket: str = "imbalance_unknown",
    system_balance_margin_direction_bucket: str = "margin_unknown",
) -> dict:
    interval_start = pd.Timestamp(interval_start_utc)
    interval_end = interval_start + pd.Timedelta(hours=1)
    interval_start_local = interval_start.tz_convert("Europe/London")
    interval_end_local = interval_end.tz_convert("Europe/London")
    if curtailment_selected_mwh is None:
        curtailment_selected_mwh = opportunity_deliverable_mwh
    if deliverable_mw_proxy is None:
        deliverable_mw_proxy = opportunity_deliverable_mwh
    if route_price_score_eur_per_mwh is None:
        route_price_score_eur_per_mwh = deliverable_route_score_eur_per_mwh
    if route_price_feasible_flag is None:
        route_price_feasible_flag = route_price_score_eur_per_mwh > 0.0
    if route_price_bottleneck is None:
        route_price_bottleneck = "GB->NL" if "GB_NL" in route_name else "GB->FR"
    if internal_transfer_source_family is None:
        internal_transfer_source_family = (
            "day_ahead_constraint_boundary"
            if internal_transfer_evidence_tier == "reviewed_internal_constraint_boundary"
            else "gb_topology_transfer_gate_proxy"
        )
    if internal_transfer_source_key is None:
        internal_transfer_source_key = (
            "fact_day_ahead_constraint_boundary_half_hourly:SSE-SP2"
            if internal_transfer_evidence_tier == "reviewed_internal_constraint_boundary"
            else "gb_topology_transfer_gate_proxy"
        )
    if connector_itl_source_key is None:
        connector_itl_source_key = "neso_interconnector_itl"
    return {
        "date": interval_start.date(),
        "interval_start_local": interval_start_local,
        "interval_end_local": interval_end_local,
        "interval_start_utc": interval_start,
        "interval_end_utc": interval_end,
        "cluster_key": cluster_key,
        "cluster_label": cluster_key.replace("_", " ").title(),
        "parent_region": "England/Wales",
        "hub_key": hub_key,
        "hub_label": hub_key.title(),
        "route_name": route_name,
        "route_label": route_name,
        "route_border_key": "GB-FR",
        "route_price_score_eur_per_mwh": route_price_score_eur_per_mwh,
        "route_price_feasible_flag": route_price_feasible_flag,
        "route_price_bottleneck": route_price_bottleneck,
        "route_delivery_tier": route_delivery_tier,
        "connector_notice_market_state": connector_notice_market_state,
        "curtailment_source_tier": curtailment_source_tier,
        "upstream_market_state_feed_available_flag": upstream_market_state_feed_available_flag,
        "upstream_market_state": upstream_market_state,
        "upstream_day_ahead_price_eur_per_mwh": upstream_day_ahead_price_eur_per_mwh,
        "upstream_intraday_price_eur_per_mwh": upstream_intraday_price_eur_per_mwh,
        "upstream_forward_price_eur_per_mwh": upstream_forward_price_eur_per_mwh,
        "upstream_day_ahead_to_intraday_spread_bucket": upstream_day_ahead_to_intraday_spread_bucket,
        "upstream_forward_to_day_ahead_spread_bucket": upstream_forward_to_day_ahead_spread_bucket,
        "system_balance_feed_available_flag": system_balance_feed_available_flag,
        "system_balance_known_flag": system_balance_known_flag,
        "system_balance_active_flag": system_balance_active_flag,
        "system_balance_state": system_balance_state,
        "system_balance_imbalance_direction_bucket": system_balance_imbalance_direction_bucket,
        "system_balance_margin_direction_bucket": system_balance_margin_direction_bucket,
        "curtailment_selected_mwh": curtailment_selected_mwh,
        "deliverable_mw_proxy": deliverable_mw_proxy,
        "opportunity_deliverable_mwh": opportunity_deliverable_mwh,
        "opportunity_gross_value_eur": opportunity_deliverable_mwh * deliverable_route_score_eur_per_mwh,
        "deliverable_route_score_eur_per_mwh": deliverable_route_score_eur_per_mwh,
        "internal_transfer_evidence_tier": internal_transfer_evidence_tier,
        "internal_transfer_gate_state": internal_transfer_gate_state,
        "internal_transfer_source_family": internal_transfer_source_family,
        "internal_transfer_source_key": internal_transfer_source_key,
        "connector_itl_state": connector_itl_state,
        "connector_itl_source_key": connector_itl_source_key,
    }


class OpportunityBacktestTests(unittest.TestCase):
    def test_build_fact_backtest_prediction_hourly_uses_exact_prior_mean_for_v1(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    10.0,
                    50.0,
                ),
                _opportunity_row(
                    "2024-10-02T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    30.0,
                    60.0,
                ),
                _opportunity_row(
                    "2024-10-03T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    50.0,
                    60.0,
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_GROUP_MEAN_NOTICE_V1, forecast_horizons=(24,)
        )
        third = backtest.iloc[2]
        self.assertTrue(bool(third["prediction_eligible_flag"]))
        self.assertEqual(third["prediction_basis"], "exact_notice_hour")
        self.assertEqual(int(third["training_sample_count"]), 1)
        self.assertEqual(int(third["forecast_horizon_hours"]), 24)
        self.assertEqual(third["forecast_origin_utc"], pd.Timestamp("2024-10-02T09:00:00+00:00"))
        self.assertEqual(third["feature_route_delivery_tier_asof"], "reviewed")
        self.assertAlmostEqual(float(third["predicted_opportunity_deliverable_mwh"]), 30.0)
        self.assertAlmostEqual(float(third["predicted_opportunity_gross_value_eur"]), 1800.0)

    def test_build_fact_backtest_prediction_hourly_uses_potential_ratio_v2(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    10.0,
                    50.0,
                    curtailment_selected_mwh=12.0,
                    deliverable_mw_proxy=10.0,
                ),
                _opportunity_row(
                    "2024-10-02T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    9.0,
                    60.0,
                    curtailment_selected_mwh=18.0,
                    deliverable_mw_proxy=10.0,
                ),
                _opportunity_row(
                    "2024-10-03T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    12.0,
                    60.0,
                    curtailment_selected_mwh=20.0,
                    deliverable_mw_proxy=10.0,
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(24,)
        )
        third = backtest.iloc[2]
        self.assertTrue(bool(third["prediction_eligible_flag"]))
        self.assertEqual(third["prediction_basis"], "ratio_exact_notice_hour")
        self.assertEqual(int(third["training_sample_count"]), 1)
        self.assertAlmostEqual(float(third["predicted_opportunity_deliverable_mwh"]), 9.0)
        self.assertAlmostEqual(float(third["predicted_opportunity_gross_value_eur"]), 540.0)

    def test_build_fact_backtest_prediction_hourly_uses_market_state_features_for_britned_flip(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=150.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    connector_itl_state="published_restriction",
                    route_price_score_eur_per_mwh=-5.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                ),
                _opportunity_row(
                    "2024-10-01T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    100.0,
                    60.0,
                    curtailment_selected_mwh=150.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    connector_itl_state="published_restriction",
                    route_price_score_eur_per_mwh=60.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->NL",
                ),
                _opportunity_row(
                    "2024-10-02T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=180.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    connector_itl_state="published_restriction",
                    route_price_score_eur_per_mwh=-2.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                ),
                _opportunity_row(
                    "2024-10-02T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    120.0,
                    65.0,
                    curtailment_selected_mwh=180.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    connector_itl_state="published_restriction",
                    route_price_score_eur_per_mwh=65.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->NL",
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(1,)
        )
        target = backtest[backtest["interval_start_utc"] == pd.Timestamp("2024-10-02T05:00:00+00:00")].iloc[0]
        self.assertTrue(bool(target["prediction_eligible_flag"]))
        self.assertEqual(target["prediction_basis"], "ratio_cluster_route_market_state")
        self.assertAlmostEqual(float(target["feature_route_price_score_eur_per_mwh_asof"]), -2.0)
        self.assertFalse(bool(target["feature_route_price_feasible_flag_asof"]))
        self.assertEqual(target["feature_route_price_bottleneck_asof"], "GB->NL")
        self.assertEqual(target["feature_route_price_state_asof"], "price_non_positive")
        self.assertEqual(target["feature_route_price_delta_bucket_asof"], "price_no_prior")
        self.assertEqual(target["feature_route_price_transition_state_asof"], "START->price_non_positive")
        self.assertEqual(target["feature_route_price_persistence_bucket_asof"], "price_persist_1h")
        self.assertEqual(target["feature_connector_itl_state_asof"], "published_restriction")
        self.assertEqual(target["feature_internal_transfer_gate_state_asof"], "reviewed_boundary_cap")
        self.assertEqual(target["feature_internal_transfer_gate_bucket_asof"], "nonblocking_transfer")
        self.assertEqual(
            target["feature_connector_itl_state_path_asof"],
            "START|START|published_restriction",
        )
        self.assertEqual(
            target["feature_internal_transfer_gate_state_path_asof"],
            "START|START|reviewed_boundary_cap",
        )
        self.assertAlmostEqual(float(target["predicted_opportunity_deliverable_mwh"]), 113.33333333333333)

    def test_build_fact_backtest_prediction_hourly_prefers_upstream_market_state_when_available(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    110.0,
                    52.0,
                    curtailment_selected_mwh=150.0,
                    deliverable_mw_proxy=170.0,
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="intraday_stronger_than_day_ahead",
                    upstream_day_ahead_price_eur_per_mwh=38.0,
                    upstream_intraday_price_eur_per_mwh=52.0,
                    upstream_forward_price_eur_per_mwh=36.0,
                    upstream_day_ahead_to_intraday_spread_bucket="spread_positive",
                    upstream_forward_to_day_ahead_spread_bucket="spread_flat",
                ),
                _opportunity_row(
                    "2024-10-01T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    100.0,
                    55.0,
                    curtailment_selected_mwh=150.0,
                    deliverable_mw_proxy=170.0,
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="intraday_stronger_than_day_ahead",
                    upstream_day_ahead_price_eur_per_mwh=40.0,
                    upstream_intraday_price_eur_per_mwh=55.0,
                    upstream_forward_price_eur_per_mwh=38.0,
                    upstream_day_ahead_to_intraday_spread_bucket="spread_positive",
                    upstream_forward_to_day_ahead_spread_bucket="spread_flat",
                ),
                _opportunity_row(
                    "2024-10-02T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    130.0,
                    57.0,
                    curtailment_selected_mwh=180.0,
                    deliverable_mw_proxy=170.0,
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="intraday_stronger_than_day_ahead",
                    upstream_day_ahead_price_eur_per_mwh=42.0,
                    upstream_intraday_price_eur_per_mwh=57.0,
                    upstream_forward_price_eur_per_mwh=40.0,
                    upstream_day_ahead_to_intraday_spread_bucket="spread_positive",
                    upstream_forward_to_day_ahead_spread_bucket="spread_flat",
                ),
                _opportunity_row(
                    "2024-10-02T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    120.0,
                    60.0,
                    curtailment_selected_mwh=180.0,
                    deliverable_mw_proxy=170.0,
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="intraday_stronger_than_day_ahead",
                    upstream_day_ahead_price_eur_per_mwh=45.0,
                    upstream_intraday_price_eur_per_mwh=60.0,
                    upstream_forward_price_eur_per_mwh=43.0,
                    upstream_day_ahead_to_intraday_spread_bucket="spread_positive",
                    upstream_forward_to_day_ahead_spread_bucket="spread_flat",
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(1,)
        )
        target = backtest.iloc[3]
        self.assertTrue(bool(target["prediction_eligible_flag"]))
        self.assertEqual(target["prediction_basis"], "ratio_cluster_route_upstream_market_state")
        self.assertTrue(bool(target["feature_upstream_market_state_feed_available_flag_asof"]))
        self.assertEqual(target["feature_upstream_market_state_asof"], "intraday_stronger_than_day_ahead")
        self.assertAlmostEqual(float(target["feature_upstream_day_ahead_price_eur_per_mwh_asof"]), 42.0)
        self.assertAlmostEqual(float(target["feature_upstream_intraday_price_eur_per_mwh_asof"]), 57.0)
        self.assertEqual(target["feature_upstream_day_ahead_to_intraday_spread_bucket_asof"], "spread_positive")
        self.assertEqual(target["feature_upstream_forward_to_day_ahead_spread_bucket_asof"], "spread_flat")
        self.assertAlmostEqual(float(target["predicted_opportunity_deliverable_mwh"]), 113.33333333333333)

    def test_build_fact_backtest_prediction_hourly_uses_system_balance_when_known(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    90.0,
                    50.0,
                    curtailment_selected_mwh=140.0,
                    deliverable_mw_proxy=150.0,
                    system_balance_feed_available_flag=True,
                    system_balance_known_flag=True,
                    system_balance_active_flag=True,
                    system_balance_state="tight_margin",
                    system_balance_imbalance_direction_bucket="imbalance_neutral",
                    system_balance_margin_direction_bucket="margin_tight",
                ),
                _opportunity_row(
                    "2024-10-01T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    95.0,
                    55.0,
                    curtailment_selected_mwh=140.0,
                    deliverable_mw_proxy=150.0,
                    system_balance_feed_available_flag=True,
                    system_balance_known_flag=True,
                    system_balance_active_flag=True,
                    system_balance_state="tight_margin",
                    system_balance_imbalance_direction_bucket="imbalance_neutral",
                    system_balance_margin_direction_bucket="margin_tight",
                ),
                _opportunity_row(
                    "2024-10-02T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    100.0,
                    52.0,
                    curtailment_selected_mwh=160.0,
                    deliverable_mw_proxy=150.0,
                    system_balance_feed_available_flag=True,
                    system_balance_known_flag=True,
                    system_balance_active_flag=True,
                    system_balance_state="tight_margin",
                    system_balance_imbalance_direction_bucket="imbalance_neutral",
                    system_balance_margin_direction_bucket="margin_tight",
                ),
                _opportunity_row(
                    "2024-10-02T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    105.0,
                    56.0,
                    curtailment_selected_mwh=160.0,
                    deliverable_mw_proxy=150.0,
                    system_balance_feed_available_flag=True,
                    system_balance_known_flag=True,
                    system_balance_active_flag=True,
                    system_balance_state="tight_margin",
                    system_balance_imbalance_direction_bucket="imbalance_neutral",
                    system_balance_margin_direction_bucket="margin_tight",
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(1,)
        )
        target = backtest.iloc[3]
        self.assertTrue(bool(target["prediction_eligible_flag"]))
        self.assertEqual(target["prediction_basis"], "ratio_cluster_route_system_balance")
        self.assertTrue(bool(target["feature_system_balance_feed_available_flag_asof"]))
        self.assertTrue(bool(target["feature_system_balance_known_flag_asof"]))
        self.assertEqual(target["feature_system_balance_state_asof"], "tight_margin")
        self.assertEqual(target["feature_system_balance_margin_direction_bucket_asof"], "margin_tight")
        self.assertEqual(target["feature_system_balance_transition_state_asof"], "START->tight_margin")
        self.assertEqual(target["feature_system_balance_persistence_bucket_asof"], "system_balance_persist_1h")

    def test_build_fact_backtest_prediction_hourly_scopes_specialist_v3_and_preserves_lineage(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=150.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SSE-SP2",
                    connector_itl_state="published_restriction",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=-5.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                ),
                _opportunity_row(
                    "2024-10-01T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    90.0,
                    60.0,
                    curtailment_selected_mwh=150.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SSE-SP2",
                    connector_itl_state="published_restriction",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=60.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->NL",
                ),
                _opportunity_row(
                    "2024-10-02T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    80.0,
                    62.0,
                    curtailment_selected_mwh=180.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:FLOWSTH",
                    connector_itl_state="published_restriction",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=58.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->NL",
                ),
                _opportunity_row(
                    "2024-10-02T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    120.0,
                    65.0,
                    curtailment_selected_mwh=180.0,
                    deliverable_mw_proxy=170.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:FLOWSTH",
                    connector_itl_state="published_restriction",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=65.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->NL",
                ),
                _opportunity_row(
                    "2024-10-02T05:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "no_public_connector_restriction",
                    70.0,
                    50.0,
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact,
            model_key=MODEL_GB_NL_REVIEWED_SPECIALIST_V3,
            forecast_horizons=(1, 24),
        )

        self.assertTrue(backtest["forecast_horizon_hours"].eq(1).all())
        self.assertTrue(backtest["route_name"].eq("R2_netback_GB_NL_DE_PL").all())
        self.assertTrue(backtest["hub_key"].eq("britned").all())
        self.assertTrue(
            backtest["internal_transfer_evidence_tier"].eq("reviewed_internal_constraint_boundary").all()
        )
        target = backtest[backtest["interval_start_utc"] == pd.Timestamp("2024-10-02T05:00:00+00:00")].iloc[0]
        self.assertTrue(bool(target["prediction_eligible_flag"]))
        self.assertGreater(int(target["training_sample_count"]), 0)
        self.assertEqual(target["internal_transfer_source_family"], "day_ahead_constraint_boundary")
        self.assertEqual(
            target["feature_internal_transfer_source_family_asof"],
            "day_ahead_constraint_boundary",
        )
        self.assertEqual(
            target["feature_internal_transfer_source_key_asof"],
            "fact_day_ahead_constraint_boundary_half_hourly:FLOWSTH",
        )
        self.assertEqual(
            target["feature_internal_transfer_boundary_family_asof"],
            "day_ahead_constraint_boundary",
        )
        self.assertEqual(target["feature_connector_itl_source_key_asof"], "neso_interconnector_itl")
        self.assertGreaterEqual(float(target["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertLessEqual(
            float(target["predicted_opportunity_deliverable_mwh"]),
            float(target["feature_deliverable_mw_proxy_asof"]),
        )

    def test_build_fact_backtest_summary_slice_includes_multiple_dimensions(self) -> None:
        backtest = pd.concat(
            [
                build_fact_backtest_prediction_hourly(
                    pd.DataFrame(
                        [
                            _opportunity_row(
                                "2024-10-01T09:00:00Z",
                                "east_anglia_offshore",
                                "R1_netback_GB_FR_DE_PL",
                                "ifa",
                                "reviewed",
                                "known_upcoming_restriction",
                                10.0,
                                50.0,
                            ),
                            _opportunity_row(
                                "2024-10-02T09:00:00Z",
                                "east_anglia_offshore",
                                "R1_netback_GB_FR_DE_PL",
                                "ifa",
                                "reviewed",
                                "known_upcoming_restriction",
                                30.0,
                                60.0,
                            ),
                            _opportunity_row(
                                "2024-10-03T09:00:00Z",
                                "east_anglia_offshore",
                                "R1_netback_GB_FR_DE_PL",
                                "ifa",
                                "reviewed",
                                "known_upcoming_restriction",
                                50.0,
                                60.0,
                            ),
                        ]
                    ),
                    model_key=MODEL_GROUP_MEAN_NOTICE_V1,
                    forecast_horizons=(24,),
                )
            ],
            ignore_index=True,
        )

        summary = build_fact_backtest_summary_slice(backtest)
        self.assertIn("all", set(summary["slice_dimension"]))
        self.assertIn("route_name", set(summary["slice_dimension"]))
        self.assertIn("hub_key", set(summary["slice_dimension"]))
        self.assertIn("internal_transfer_evidence_tier", set(summary["slice_dimension"]))
        self.assertIn("internal_transfer_gate_state", set(summary["slice_dimension"]))
        self.assertIn("system_balance_state", set(summary["slice_dimension"]))
        route_row = summary[
            (summary["slice_dimension"] == "route_name")
            & (summary["slice_value"] == "R1_netback_GB_FR_DE_PL")
        ].iloc[0]
        self.assertEqual(int(route_row["eligible_row_count"]), 1)
        self.assertAlmostEqual(float(route_row["mae_opportunity_deliverable_mwh"]), 20.0)
        self.assertEqual(int(route_row["forecast_horizon_hours"]), 24)

    def test_build_fact_backtest_top_error_hourly_ranks_largest_error_first(self) -> None:
        base = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    10.0,
                    50.0,
                ),
                _opportunity_row(
                    "2024-10-02T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    30.0,
                    60.0,
                ),
                _opportunity_row(
                    "2024-10-03T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    80.0,
                    60.0,
                ),
            ]
        )
        backtest = build_fact_backtest_prediction_hourly(
            base, model_key=MODEL_GROUP_MEAN_NOTICE_V1, forecast_horizons=(24,)
        )
        ranked = build_fact_backtest_top_error_hourly(backtest)
        self.assertEqual(int(ranked.iloc[0]["top_error_rank"]), 1)
        self.assertEqual(ranked.iloc[0]["interval_start_utc"], pd.Timestamp("2024-10-03T09:00:00+00:00"))
        self.assertEqual(int(ranked.iloc[0]["forecast_horizon_hours"]), 24)
        self.assertEqual(ranked.iloc[0]["error_focus_area"], "proxy_internal_transfer")

    def test_build_fact_backtest_top_error_hourly_flags_reviewed_internal_transfer(self) -> None:
        base = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    10.0,
                    50.0,
                    internal_transfer_evidence_tier="reviewed_internal_transfer_period",
                    internal_transfer_gate_state="reviewed_pass_restricted",
                ),
                _opportunity_row(
                    "2024-10-02T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    30.0,
                    60.0,
                    internal_transfer_evidence_tier="reviewed_internal_transfer_period",
                    internal_transfer_gate_state="reviewed_pass_restricted",
                ),
                _opportunity_row(
                    "2024-10-03T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    80.0,
                    60.0,
                    internal_transfer_evidence_tier="reviewed_internal_transfer_period",
                    internal_transfer_gate_state="reviewed_pass_restricted",
                ),
            ]
        )
        backtest = build_fact_backtest_prediction_hourly(
            base, model_key=MODEL_GROUP_MEAN_NOTICE_V1, forecast_horizons=(24,)
        )
        ranked = build_fact_backtest_top_error_hourly(backtest)
        self.assertEqual(ranked.iloc[0]["error_focus_area"], "reviewed_internal_transfer")

    def test_build_fact_drift_window_flags_change_after_warmup(self) -> None:
        rows = []
        for date_text, tier, notice_state, actual in (
            ("2024-10-01T09:00:00Z", "no_price_signal", "no_public_connector_restriction", 0.0),
            ("2024-10-01T10:00:00Z", "no_price_signal", "no_public_connector_restriction", 0.0),
            ("2024-10-02T09:00:00Z", "reviewed", "known_upcoming_restriction", 40.0),
            ("2024-10-02T10:00:00Z", "reviewed", "known_upcoming_restriction", 40.0),
        ):
            rows.append(
                _opportunity_row(
                    date_text,
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    tier,
                    notice_state,
                    actual,
                    50.0,
                    curtailment_selected_mwh=max(actual, 40.0),
                    deliverable_mw_proxy=max(actual, 40.0),
                    curtailment_source_tier="cluster_truth_research" if actual else "regional_proxy",
                )
            )
        backtest = build_fact_backtest_prediction_hourly(
            pd.DataFrame(rows), model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(24,)
        )
        drift = build_fact_drift_window(backtest)
        global_rows = drift[drift["drift_scope"] == "global_daily"].reset_index(drop=True)
        route_rows = drift[drift["drift_scope"] == "route_daily"].reset_index(drop=True)
        cluster_rows = drift[drift["drift_scope"] == "cluster_daily"].reset_index(drop=True)
        self.assertEqual(global_rows.iloc[0]["drift_state"], "warmup")
        self.assertEqual(global_rows.iloc[1]["drift_state"], "warn")
        self.assertTrue(route_rows["route_name"].eq("R1_netback_GB_FR_DE_PL").all())
        self.assertTrue(cluster_rows["cluster_key"].eq("east_anglia_offshore").all())
        self.assertIn("reviewed_internal_transfer_share", drift.columns)
        self.assertIn("proxy_internal_transfer_share", drift.columns)
        self.assertIn("blocked_internal_reviewed_share", drift.columns)

    def test_summarize_backtest_prediction_hourly_reports_both_models(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    10.0,
                    50.0,
                ),
                _opportunity_row(
                    "2024-10-02T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    30.0,
                    60.0,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_opportunity_backtest(tmp_dir, fact, model_key="all", forecast_horizons=(24,))
            summary = summarize_backtest_prediction_hourly(frames[BACKTEST_PREDICTION_TABLE])
            self.assertEqual(set(summary["model_key"]), {MODEL_GROUP_MEAN_NOTICE_V1, MODEL_POTENTIAL_RATIO_V2})
            self.assertEqual(set(summary["forecast_horizon_hours"]), {24})

    def test_materialize_opportunity_backtest_writes_all_csvs(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    10.0,
                    50.0,
                ),
                _opportunity_row(
                    "2024-10-02T09:00:00Z",
                    "east_anglia_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "known_upcoming_restriction",
                    30.0,
                    60.0,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_opportunity_backtest(tmp_dir, fact, model_key="all", forecast_horizons=(1, 24))
            self.assertEqual(
                set(frames),
                {
                    BACKTEST_PREDICTION_TABLE,
                    BACKTEST_SUMMARY_SLICE_TABLE,
                    BACKTEST_TOP_ERROR_TABLE,
                    DRIFT_WINDOW_TABLE,
                },
            )
            for table_name in frames:
                self.assertTrue((Path(tmp_dir) / f"{table_name}.csv").exists())
            self.assertEqual(
                set(frames[BACKTEST_PREDICTION_TABLE]["forecast_horizon_hours"]),
                {1, 24},
            )


if __name__ == "__main__":
    unittest.main()
