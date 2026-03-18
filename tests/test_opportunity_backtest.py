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
    _apply_gb_nl_specialist_flip_opening_guardrail,
    _apply_potential_ratio_opening_guardrail,
    _apply_potential_ratio_event_phase_calibration,
    _apply_potential_ratio_persist_close_suppressor,
    _apply_potential_ratio_r2_reviewed_event_lifecycle,
    _prepare_specialist_feature_frame,
    _prior_mean_by_group,
    _prior_mean_from_history_by_group,
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
    def test_prior_mean_by_group_excludes_same_origin_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "forecast_origin_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "cluster_key": "east_anglia_offshore",
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "feature_hour_of_day": 4,
                    "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                    "feature_upstream_day_ahead_to_intraday_spread_bucket_asof": "spread_unknown",
                    "feature_upstream_forward_to_day_ahead_spread_bucket_asof": "spread_positive",
                    "realized_ratio": 0.0,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "cluster_key": "east_anglia_offshore",
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "feature_hour_of_day": 4,
                    "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                    "feature_upstream_day_ahead_to_intraday_spread_bucket_asof": "spread_unknown",
                    "feature_upstream_forward_to_day_ahead_spread_bucket_asof": "spread_positive",
                    "realized_ratio": 1.0,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-10-02T00:00:00Z"),
                    "cluster_key": "east_anglia_offshore",
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "feature_hour_of_day": 4,
                    "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                    "feature_upstream_day_ahead_to_intraday_spread_bucket_asof": "spread_unknown",
                    "feature_upstream_forward_to_day_ahead_spread_bucket_asof": "spread_positive",
                    "realized_ratio": 0.5,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-10-02T00:00:00Z"),
                    "cluster_key": "east_anglia_offshore",
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "feature_hour_of_day": 4,
                    "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                    "feature_upstream_day_ahead_to_intraday_spread_bucket_asof": "spread_unknown",
                    "feature_upstream_forward_to_day_ahead_spread_bucket_asof": "spread_positive",
                    "realized_ratio": 0.5,
                },
            ]
        )
        grouped = _prior_mean_by_group(
            frame,
            [
                "cluster_key",
                "route_name",
                "feature_hour_of_day",
                "feature_upstream_market_state_asof",
                "feature_upstream_day_ahead_to_intraday_spread_bucket_asof",
                "feature_upstream_forward_to_day_ahead_spread_bucket_asof",
            ],
            "realized_ratio",
            "demo",
        )
        self.assertEqual(grouped["demo_prior_count"].tolist(), [0, 0, 2, 2])
        self.assertTrue(pd.isna(grouped.iloc[0]["demo_prior_mean"]))
        self.assertTrue(pd.isna(grouped.iloc[1]["demo_prior_mean"]))
        self.assertEqual(grouped.iloc[2]["demo_prior_mean"], 0.5)
        self.assertEqual(grouped.iloc[3]["demo_prior_mean"], 0.5)

        global_grouped = _prior_mean_by_group(frame, [], "realized_ratio", "global")
        self.assertEqual(global_grouped["global_prior_count"].tolist(), [0, 0, 2, 2])
        self.assertEqual(global_grouped.iloc[2]["global_prior_mean"], 0.5)
        self.assertEqual(global_grouped.iloc[3]["global_prior_mean"], 0.5)

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

    def test_build_fact_backtest_prediction_hourly_applies_opening_guardrail_on_price_jump(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T03:00:00Z",
                    "dogger_hornsea_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=210.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    route_price_score_eur_per_mwh=-5.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->FR",
                ),
                _opportunity_row(
                    "2024-10-01T04:00:00Z",
                    "dogger_hornsea_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "no_public_connector_restriction",
                    0.0,
                    80.0,
                    curtailment_selected_mwh=205.0,
                    deliverable_mw_proxy=1000.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    route_price_score_eur_per_mwh=80.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->FR",
                ),
                _opportunity_row(
                    "2024-10-01T05:00:00Z",
                    "dogger_hornsea_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "no_public_connector_restriction",
                    190.0,
                    78.0,
                    curtailment_selected_mwh=190.0,
                    deliverable_mw_proxy=1000.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    route_price_score_eur_per_mwh=78.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->FR",
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(1,)
        )
        target = backtest[backtest["interval_start_utc"] == pd.Timestamp("2024-10-01T05:00:00+00:00")].iloc[0]
        self.assertTrue(bool(target["prediction_eligible_flag"]))
        self.assertEqual(target["feature_route_price_transition_state_asof"], "price_non_positive->price_high_positive")
        self.assertEqual(target["prediction_basis"], "opening_guardrail_jump")
        self.assertAlmostEqual(float(target["feature_curtailment_selected_mwh_asof"]), 205.0)
        self.assertAlmostEqual(float(target["predicted_opportunity_deliverable_mwh"]), 205.0)

    def test_build_fact_backtest_prediction_hourly_applies_opening_guardrail_on_preopen_state(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T00:00:00Z",
                    "dogger_hornsea_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=200.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    route_price_score_eur_per_mwh=-2.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->FR",
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="day_ahead_stronger_than_forward",
                    upstream_day_ahead_to_intraday_spread_bucket="spread_unknown",
                    upstream_forward_to_day_ahead_spread_bucket="spread_positive",
                ),
                _opportunity_row(
                    "2024-10-01T01:00:00Z",
                    "dogger_hornsea_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=190.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    route_price_score_eur_per_mwh=-1.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->FR",
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="day_ahead_stronger_than_forward",
                    upstream_day_ahead_to_intraday_spread_bucket="spread_unknown",
                    upstream_forward_to_day_ahead_spread_bucket="spread_positive",
                ),
                _opportunity_row(
                    "2024-10-01T02:00:00Z",
                    "dogger_hornsea_offshore",
                    "R1_netback_GB_FR_DE_PL",
                    "ifa",
                    "reviewed",
                    "no_public_connector_restriction",
                    180.0,
                    70.0,
                    curtailment_selected_mwh=180.0,
                    deliverable_mw_proxy=500.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    route_price_score_eur_per_mwh=70.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->FR",
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="day_ahead_stronger_than_forward",
                    upstream_day_ahead_to_intraday_spread_bucket="spread_unknown",
                    upstream_forward_to_day_ahead_spread_bucket="spread_positive",
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(1,)
        )
        target = backtest[backtest["interval_start_utc"] == pd.Timestamp("2024-10-01T02:00:00+00:00")].iloc[0]
        self.assertTrue(bool(target["prediction_eligible_flag"]))
        self.assertEqual(target["feature_route_price_transition_state_asof"], "price_non_positive->price_non_positive")
        self.assertEqual(target["feature_origin_hour_of_day"], 1.0)
        self.assertEqual(target["prediction_basis"], "opening_guardrail_preopen")
        self.assertAlmostEqual(float(target["feature_curtailment_selected_mwh_asof"]), 190.0)
        self.assertAlmostEqual(float(target["predicted_opportunity_deliverable_mwh"]), 190.0)

    def test_apply_potential_ratio_opening_guardrail_skips_preopen_when_itl_is_blocked_zero(self) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 309.2517906012644,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_connector_notice_market_state_asof": "no_public_connector_restriction",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "blocked_zero_or_negative_itl",
                    "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                    "feature_origin_hour_of_day": 0.0,
                    "feature_route_price_score_eur_per_mwh_asof": -8.7316,
                    "feature_route_price_feasible_flag_asof": False,
                    "prediction_basis": "ratio_exact_notice_hour",
                }
            ]
        )

        adjusted = _apply_potential_ratio_opening_guardrail(result)

        self.assertAlmostEqual(float(adjusted.iloc[0]["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(adjusted.iloc[0]["prediction_basis"], "ratio_exact_notice_hour")

    def test_build_fact_backtest_prediction_hourly_keeps_r2_generic_preopen_when_itl_is_not_published(
        self,
    ) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-12-09T00:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=214.500000,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    connector_itl_state="no_public_itl_restriction",
                    route_price_score_eur_per_mwh=-2.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="day_ahead_near_forward",
                    upstream_day_ahead_to_intraday_spread_bucket="spread_unknown",
                    upstream_forward_to_day_ahead_spread_bucket="spread_positive",
                ),
                _opportunity_row(
                    "2024-12-09T01:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=211.021402,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    connector_itl_state="no_public_itl_restriction",
                    route_price_score_eur_per_mwh=-0.6748,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="day_ahead_near_forward",
                    upstream_day_ahead_to_intraday_spread_bucket="spread_unknown",
                    upstream_forward_to_day_ahead_spread_bucket="spread_positive",
                ),
                _opportunity_row(
                    "2024-12-09T02:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    209.267162,
                    35.0,
                    curtailment_selected_mwh=209.267162,
                    deliverable_mw_proxy=500.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    connector_itl_state="no_public_itl_restriction",
                    route_price_score_eur_per_mwh=35.0,
                    route_price_feasible_flag=True,
                    route_price_bottleneck="GB->NL",
                    upstream_market_state_feed_available_flag=True,
                    upstream_market_state="day_ahead_near_forward",
                    upstream_day_ahead_to_intraday_spread_bucket="spread_unknown",
                    upstream_forward_to_day_ahead_spread_bucket="spread_positive",
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact, model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(1,)
        )
        target = backtest[backtest["interval_start_utc"] == pd.Timestamp("2024-12-09T02:00:00+00:00")].iloc[0]
        self.assertEqual(target["feature_connector_itl_state_asof"], "no_public_itl_restriction")
        self.assertEqual(target["prediction_basis"], "opening_guardrail_preopen")
        self.assertAlmostEqual(float(target["predicted_opportunity_deliverable_mwh"]), 211.021402)

    def test_event_phase_calibration_prior_mean_excludes_same_origin_rows(self) -> None:
        target_frame = pd.DataFrame(
            [
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-01T00:00:00Z"),
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "event_phase": "persist",
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-01T00:00:00Z"),
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "event_phase": "persist",
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-02T00:00:00Z"),
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "event_phase": "persist",
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-02T00:00:00Z"),
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "event_phase": "persist",
                },
            ]
        )
        history_frame = target_frame.copy()
        history_frame["realized_ratio"] = [0.0, 1.0, 0.5, 0.5]

        grouped = _prior_mean_from_history_by_group(
            target_frame,
            history_frame,
            ["route_name", "cluster_key", "event_phase"],
            "realized_ratio",
            "event_phase_demo",
        )

        self.assertEqual(grouped["event_phase_demo_prior_count"].tolist(), [0, 0, 2, 2])
        self.assertTrue(pd.isna(grouped.iloc[0]["event_phase_demo_prior_mean"]))
        self.assertTrue(pd.isna(grouped.iloc[1]["event_phase_demo_prior_mean"]))
        self.assertEqual(grouped.iloc[2]["event_phase_demo_prior_mean"], 0.5)
        self.assertEqual(grouped.iloc[3]["event_phase_demo_prior_mean"], 0.5)

    def test_apply_potential_ratio_event_phase_calibration_decays_r1_persist_with_history(self) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-06T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 100.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 200.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-07T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 90.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 190.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-08T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 80.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 180.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 55.45372,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 207.672486,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 33,
                },
            ]
        )

        adjusted = _apply_potential_ratio_event_phase_calibration(result)
        target = adjusted.iloc[-1]

        self.assertAlmostEqual(float(target["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(
            target["prediction_basis"],
            "ratio_route_notice_state_event_phase_calibrated_persist",
        )
        self.assertEqual(int(target["training_sample_count"]), 33)

    def test_apply_potential_ratio_event_phase_calibration_scopes_to_r1_reviewed_t_plus_1_only(self) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-06T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 100.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 200.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-07T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 90.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 190.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-08T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 80.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 180.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 70.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 170.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-06T06:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 100.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 200.0,
                    "feature_internal_transfer_source_family_asof": "gb_topology_transfer_gate_proxy",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-07T06:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 90.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 190.0,
                    "feature_internal_transfer_source_family_asof": "gb_topology_transfer_gate_proxy",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-08T06:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 80.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 180.0,
                    "feature_internal_transfer_source_family_asof": "gb_topology_transfer_gate_proxy",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T06:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 70.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 170.0,
                    "feature_internal_transfer_source_family_asof": "gb_topology_transfer_gate_proxy",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-06T07:00:00Z"),
                    "forecast_horizon_hours": 6,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 100.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 200.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-07T07:00:00Z"),
                    "forecast_horizon_hours": 6,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 90.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 190.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-08T07:00:00Z"),
                    "forecast_horizon_hours": 6,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 80.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 180.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T07:00:00Z"),
                    "forecast_horizon_hours": 6,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "prediction_basis": "ratio_route_notice_state",
                    "predicted_opportunity_deliverable_mwh": 70.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 170.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 3,
                },
            ]
        )

        adjusted = _apply_potential_ratio_event_phase_calibration(result)

        self.assertEqual(float(adjusted.iloc[3]["predicted_opportunity_deliverable_mwh"]), 70.0)
        self.assertEqual(float(adjusted.iloc[7]["predicted_opportunity_deliverable_mwh"]), 70.0)
        self.assertEqual(float(adjusted.iloc[11]["predicted_opportunity_deliverable_mwh"]), 70.0)
        self.assertEqual(adjusted.iloc[3]["prediction_basis"], "ratio_route_notice_state")
        self.assertEqual(adjusted.iloc[7]["prediction_basis"], "ratio_route_notice_state")
        self.assertEqual(adjusted.iloc[11]["prediction_basis"], "ratio_route_notice_state")

    def test_apply_potential_ratio_persist_close_suppressor_zeroes_close_rows_with_zero_history(self) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-07T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "feature_origin_hour_of_day": 5.0,
                    "prediction_basis": "ratio_route_notice_state_event_phase_calibrated_persist",
                    "predicted_opportunity_deliverable_mwh": 60.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 200.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_delivery_tier_asof": "capacity_unknown",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 8,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-08T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "feature_origin_hour_of_day": 5.0,
                    "prediction_basis": "ratio_route_notice_state_event_phase_calibrated_persist",
                    "predicted_opportunity_deliverable_mwh": 58.0,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 195.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_delivery_tier_asof": "capacity_unknown",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 8,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "feature_origin_hour_of_day": 5.0,
                    "prediction_basis": "ratio_route_notice_state_event_phase_calibrated_persist",
                    "predicted_opportunity_deliverable_mwh": 53.176128,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 207.672486,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_delivery_tier_asof": "capacity_unknown",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 45,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "feature_origin_hour_of_day": 5.0,
                    "prediction_basis": "ratio_route_notice_state_opening_guardrail_jump",
                    "predicted_opportunity_deliverable_mwh": 194.648349,
                    "actual_opportunity_deliverable_mwh": 207.672486,
                    "feature_curtailment_selected_mwh_asof": 207.672486,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_high_positive",
                    "training_sample_count": 33,
                },
            ]
        )

        adjusted = _apply_potential_ratio_persist_close_suppressor(result)
        target = adjusted.iloc[2]

        self.assertAlmostEqual(float(target["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(
            target["prediction_basis"],
            "ratio_route_notice_state_event_phase_calibrated_persist_persist_close_suppressor",
        )
        self.assertEqual(float(adjusted.iloc[3]["predicted_opportunity_deliverable_mwh"]), 194.648349)
        self.assertEqual(adjusted.iloc[3]["prediction_basis"], "ratio_route_notice_state_opening_guardrail_jump")

    def test_apply_potential_ratio_persist_close_suppressor_zeroes_capacity_unknown_persist_without_history(self) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "feature_origin_hour_of_day": 5.0,
                    "prediction_basis": "ratio_route_notice_state_event_phase_calibrated_persist",
                    "predicted_opportunity_deliverable_mwh": 53.176128,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 207.672486,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_delivery_tier_asof": "capacity_unknown",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 45,
                },
                {
                    "forecast_origin_utc": pd.Timestamp("2024-12-09T05:00:00Z"),
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "feature_origin_hour_of_day": 5.0,
                    "prediction_basis": "ratio_route_notice_state_event_phase_calibrated_persist",
                    "predicted_opportunity_deliverable_mwh": 53.176128,
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 207.672486,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                    "training_sample_count": 45,
                },
            ]
        )

        adjusted = _apply_potential_ratio_persist_close_suppressor(result)

        self.assertAlmostEqual(float(adjusted.iloc[0]["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(
            adjusted.iloc[0]["prediction_basis"],
            "ratio_route_notice_state_event_phase_calibrated_persist_persist_close_suppressor",
        )
        self.assertAlmostEqual(float(adjusted.iloc[1]["predicted_opportunity_deliverable_mwh"]), 53.176128)
        self.assertEqual(
            adjusted.iloc[1]["prediction_basis"],
            "ratio_route_notice_state_event_phase_calibrated_persist",
        )

    def test_apply_potential_ratio_r2_reviewed_event_lifecycle_opens_clean_preopen_and_reviewed_open_rows(
        self,
    ) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 164.864498,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_near_forward",
                    "feature_origin_hour_of_day": 4.0,
                    "feature_route_price_score_eur_per_mwh_asof": -4.88675,
                    "prediction_basis": "ratio_cluster_route_upstream_market_state",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 290.835304,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_mid_positive",
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_near_forward",
                    "feature_origin_hour_of_day": 5.0,
                    "feature_route_price_score_eur_per_mwh_asof": 55.689525,
                    "prediction_basis": "ratio_route_notice_state",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 301.730730,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_much_weaker_than_forward",
                    "feature_origin_hour_of_day": 4.0,
                    "feature_route_price_score_eur_per_mwh_asof": -33.22665,
                    "prediction_basis": "ratio_route_notice_state",
                },
            ]
        )

        adjusted = _apply_potential_ratio_r2_reviewed_event_lifecycle(result)

        self.assertAlmostEqual(float(adjusted.iloc[0]["predicted_opportunity_deliverable_mwh"]), 164.864498)
        self.assertEqual(
            adjusted.iloc[0]["prediction_basis"],
            "ratio_cluster_route_upstream_market_state_r2_reviewed_event_preopen_open",
        )
        self.assertAlmostEqual(float(adjusted.iloc[1]["predicted_opportunity_deliverable_mwh"]), 290.835304)
        self.assertEqual(
            adjusted.iloc[1]["prediction_basis"],
            "ratio_route_notice_state_r2_reviewed_event_open",
        )
        self.assertEqual(float(adjusted.iloc[2]["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(adjusted.iloc[2]["prediction_basis"], "ratio_route_notice_state")

    def test_apply_potential_ratio_r2_reviewed_event_lifecycle_suppresses_reviewed_jump_and_close_rows(
        self,
    ) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "predicted_opportunity_deliverable_mwh": 366.489426,
                    "feature_curtailment_selected_mwh_asof": 366.489426,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_high_positive",
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_near_forward",
                    "feature_origin_hour_of_day": 15.0,
                    "feature_route_price_score_eur_per_mwh_asof": 76.9315,
                    "prediction_basis": "ratio_global_opening_guardrail_jump",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "predicted_opportunity_deliverable_mwh": 208.898806,
                    "feature_curtailment_selected_mwh_asof": 311.065798,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_mid_positive->price_low_positive",
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_near_forward",
                    "feature_origin_hour_of_day": 6.0,
                    "feature_route_price_score_eur_per_mwh_asof": 6.99345,
                    "prediction_basis": "ratio_route_notice_state",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "predicted_opportunity_deliverable_mwh": 80.0,
                    "feature_curtailment_selected_mwh_asof": 90.0,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_high_positive",
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_near_forward",
                    "feature_origin_hour_of_day": 5.0,
                    "feature_route_price_score_eur_per_mwh_asof": 100.0,
                    "prediction_basis": "ratio_route_notice_state",
                },
            ]
        )

        adjusted = _apply_potential_ratio_r2_reviewed_event_lifecycle(result)

        self.assertAlmostEqual(float(adjusted.iloc[0]["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(
            adjusted.iloc[0]["prediction_basis"],
            "ratio_global_opening_guardrail_jump_r2_reviewed_event_jump_suppressor",
        )
        self.assertAlmostEqual(float(adjusted.iloc[1]["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(
            adjusted.iloc[1]["prediction_basis"],
            "ratio_route_notice_state_r2_reviewed_event_close_suppressor",
        )
        self.assertAlmostEqual(float(adjusted.iloc[2]["predicted_opportunity_deliverable_mwh"]), 80.0)
        self.assertEqual(adjusted.iloc[2]["prediction_basis"], "ratio_route_notice_state")

    def test_apply_potential_ratio_r2_reviewed_event_lifecycle_opens_late_reopen_trio_only(self) -> None:
        result = pd.DataFrame(
            [
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "dogger_hornsea_offshore",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 345.0642357407364,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "blocked_zero_or_negative_itl",
                    "feature_upstream_market_state_asof": "day_ahead_weaker_than_forward",
                    "feature_origin_hour_of_day": 14.0,
                    "feature_route_price_score_eur_per_mwh_asof": -12.24605,
                    "prediction_basis": "ratio_route_notice_state",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "east_anglia_offshore",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 70.33683462341149,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "blocked_zero_or_negative_itl",
                    "feature_upstream_market_state_asof": "day_ahead_weaker_than_forward",
                    "feature_origin_hour_of_day": 14.0,
                    "feature_route_price_score_eur_per_mwh_asof": -12.24605,
                    "prediction_basis": "ratio_route_notice_state",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "humber_offshore",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 59.09673261986632,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "blocked_zero_or_negative_itl",
                    "feature_upstream_market_state_asof": "day_ahead_weaker_than_forward",
                    "feature_origin_hour_of_day": 14.0,
                    "feature_route_price_score_eur_per_mwh_asof": -12.24605,
                    "prediction_basis": "ratio_route_notice_state",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "east_coast_scotland_offshore",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 81.00424612655388,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "blocked_reviewed_boundary",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "blocked_zero_or_negative_itl",
                    "feature_upstream_market_state_asof": "day_ahead_weaker_than_forward",
                    "feature_origin_hour_of_day": 14.0,
                    "feature_route_price_score_eur_per_mwh_asof": -12.24605,
                    "prediction_basis": "ratio_route_notice_state",
                },
                {
                    "forecast_horizon_hours": 1,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": "east_anglia_offshore",
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "feature_curtailment_selected_mwh_asof": 68.76863102583121,
                    "feature_internal_transfer_source_family_asof": "day_ahead_constraint_boundary",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_connector_itl_state_asof": "blocked_zero_or_negative_itl",
                    "feature_upstream_market_state_asof": "day_ahead_weaker_than_forward",
                    "feature_origin_hour_of_day": 14.0,
                    "feature_route_price_score_eur_per_mwh_asof": -21.7779,
                    "prediction_basis": "ratio_cluster_route_upstream_market_state",
                },
            ]
        )

        adjusted = _apply_potential_ratio_r2_reviewed_event_lifecycle(result)

        self.assertAlmostEqual(float(adjusted.iloc[0]["predicted_opportunity_deliverable_mwh"]), 170.0)
        self.assertEqual(
            adjusted.iloc[0]["prediction_basis"],
            "ratio_route_notice_state_r2_reviewed_event_late_reopen",
        )
        self.assertAlmostEqual(float(adjusted.iloc[1]["predicted_opportunity_deliverable_mwh"]), 73.85367635458206)
        self.assertEqual(
            adjusted.iloc[1]["prediction_basis"],
            "ratio_route_notice_state_r2_reviewed_event_late_reopen",
        )
        self.assertAlmostEqual(float(adjusted.iloc[2]["predicted_opportunity_deliverable_mwh"]), 62.05156925085964)
        self.assertEqual(
            adjusted.iloc[2]["prediction_basis"],
            "ratio_route_notice_state_r2_reviewed_event_late_reopen",
        )
        self.assertEqual(float(adjusted.iloc[3]["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(adjusted.iloc[3]["prediction_basis"], "ratio_route_notice_state")
        self.assertEqual(float(adjusted.iloc[4]["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertEqual(adjusted.iloc[4]["prediction_basis"], "ratio_cluster_route_upstream_market_state")

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

    def test_build_fact_backtest_prediction_hourly_specialist_exposes_openable_potential_when_proxy_zero(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T03:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "reviewed",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=150.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family=None,
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SEIMPPR23",
                    connector_itl_state="published_restriction",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=-2.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                ),
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
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family=None,
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SEIMPPR23",
                    connector_itl_state="published_restriction",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=-2.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                )
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact,
            model_key=MODEL_GB_NL_REVIEWED_SPECIALIST_V3,
            forecast_horizons=(1,),
        )

        target = backtest[backtest["interval_start_utc"] == pd.Timestamp("2024-10-01T04:00:00+00:00")].iloc[0]
        self.assertEqual(float(target["feature_deliverable_mw_proxy_asof"]), 0.0)
        self.assertEqual(float(target["feature_specialist_openable_potential_mwh_asof"]), 150.0)
        self.assertTrue(bool(target["feature_specialist_zero_proxy_flag_asof"]))
        self.assertEqual(target["feature_internal_transfer_source_family_asof"], "day_ahead_constraint_boundary")
        self.assertEqual(target["feature_internal_transfer_boundary_family_asof"], "day_ahead_constraint_boundary")

    def test_build_fact_backtest_prediction_hourly_specialist_suppresses_blocked_itl_weaker_forward_regime(self) -> None:
        fact = pd.DataFrame(
            [
                _opportunity_row(
                    "2024-10-01T14:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=170.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SEIMPPR23",
                    connector_itl_state="blocked_zero_or_negative_itl",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=-1.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                    upstream_market_state="day_ahead_weaker_than_forward",
                ),
                _opportunity_row(
                    "2024-10-01T15:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    170.0,
                    0.0,
                    curtailment_selected_mwh=170.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SEIMPPR23",
                    connector_itl_state="blocked_zero_or_negative_itl",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=-1.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                    upstream_market_state="day_ahead_weaker_than_forward",
                ),
                _opportunity_row(
                    "2024-10-02T14:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=170.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SEIMPPR23",
                    connector_itl_state="blocked_zero_or_negative_itl",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=-1.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                    upstream_market_state="day_ahead_weaker_than_forward",
                ),
                _opportunity_row(
                    "2024-10-02T15:00:00Z",
                    "dogger_hornsea_offshore",
                    "R2_netback_GB_NL_DE_PL",
                    "britned",
                    "no_price_signal",
                    "no_public_connector_restriction",
                    0.0,
                    0.0,
                    curtailment_selected_mwh=170.0,
                    deliverable_mw_proxy=0.0,
                    internal_transfer_evidence_tier="reviewed_internal_constraint_boundary",
                    internal_transfer_gate_state="reviewed_boundary_cap",
                    internal_transfer_source_family="day_ahead_constraint_boundary",
                    internal_transfer_source_key="fact_day_ahead_constraint_boundary_half_hourly:SEIMPPR23",
                    connector_itl_state="blocked_zero_or_negative_itl",
                    connector_itl_source_key="neso_interconnector_itl",
                    route_price_score_eur_per_mwh=-1.0,
                    route_price_feasible_flag=False,
                    route_price_bottleneck="GB->NL",
                    upstream_market_state="day_ahead_weaker_than_forward",
                ),
            ]
        )

        backtest = build_fact_backtest_prediction_hourly(
            fact,
            model_key=MODEL_GB_NL_REVIEWED_SPECIALIST_V3,
            forecast_horizons=(1,),
        )

        target = backtest[backtest["interval_start_utc"] == pd.Timestamp("2024-10-02T15:00:00+00:00")].iloc[0]
        self.assertTrue(bool(target["prediction_eligible_flag"]))
        self.assertEqual(float(target["predicted_opportunity_deliverable_mwh"]), 0.0)
        self.assertIn("suppressed_blocked_itl_weaker_forward", str(target["prediction_basis"]))

    def test_apply_gb_nl_specialist_flip_opening_guardrail_only_lifts_tiny_predicted_flip_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prediction_basis": "specialist_v3_hybrid_open_hybrid_ratio_flip_open_specialist",
                    "predicted_opportunity_deliverable_mwh": 0.6,
                    "feature_specialist_openable_potential_mwh_asof": 280.0,
                    "feature_origin_hour_of_day": 4.0,
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_near_forward",
                    "feature_route_delivery_tier_asof": "no_price_signal",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_non_positive",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                },
                {
                    "prediction_basis": "specialist_v3_hybrid_open_hybrid_ratio_flip_open_specialist",
                    "predicted_opportunity_deliverable_mwh": 0.7,
                    "feature_specialist_openable_potential_mwh_asof": 291.0,
                    "feature_origin_hour_of_day": 5.0,
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_mid_positive",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                },
                {
                    "prediction_basis": "specialist_v3_hybrid_open_hybrid_ratio_flip_open_specialist",
                    "predicted_opportunity_deliverable_mwh": 3.0,
                    "feature_specialist_openable_potential_mwh_asof": 171.0,
                    "feature_origin_hour_of_day": 5.0,
                    "feature_connector_itl_state_asof": "published_restriction",
                    "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                    "feature_route_delivery_tier_asof": "reviewed",
                    "feature_route_price_transition_state_asof": "price_non_positive->price_mid_positive",
                    "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                },
            ]
        )

        adjusted = _apply_gb_nl_specialist_flip_opening_guardrail(frame)

        self.assertEqual(float(adjusted.iloc[0]["predicted_opportunity_deliverable_mwh"]), 280.0)
        self.assertEqual(float(adjusted.iloc[1]["predicted_opportunity_deliverable_mwh"]), 291.0)
        self.assertEqual(float(adjusted.iloc[2]["predicted_opportunity_deliverable_mwh"]), 3.0)
        self.assertIn("flip_opening_guardrail", str(adjusted.iloc[0]["prediction_basis"]))
        self.assertIn("flip_opening_guardrail", str(adjusted.iloc[1]["prediction_basis"]))
        self.assertNotIn("flip_opening_guardrail", str(adjusted.iloc[2]["prediction_basis"]))

    def test_prepare_specialist_feature_frame_fills_all_missing_categoricals_with_missing_label(self) -> None:
        feature_frame = _prepare_specialist_feature_frame(
            pd.DataFrame(
                [
                    {
                        "feature_deliverable_mw_proxy_asof": 170.0,
                        "cluster_key": "dogger_hornsea_offshore",
                        "feature_internal_transfer_source_family_asof": pd.NA,
                        "feature_internal_transfer_source_key_asof": pd.NA,
                        "feature_internal_transfer_boundary_family_asof": pd.NA,
                        "feature_connector_itl_source_key_asof": pd.NA,
                        "feature_internal_transfer_gate_state_asof": "reviewed_boundary_cap",
                        "feature_internal_transfer_gate_transition_state_asof": "reviewed_boundary_cap->reviewed_boundary_cap",
                        "feature_internal_transfer_gate_state_path_asof": "reviewed_boundary_cap|reviewed_boundary_cap",
                        "feature_route_price_state_asof": "price_high_positive",
                        "feature_route_price_transition_state_asof": "price_high_positive->price_high_positive",
                        "feature_route_price_persistence_bucket_asof": "steady",
                        "feature_upstream_market_state_asof": "day_ahead_stronger_than_forward",
                        "feature_upstream_day_ahead_to_intraday_spread_bucket_asof": "intraday_missing",
                        "feature_upstream_forward_to_day_ahead_spread_bucket_asof": "day_ahead_above_forward",
                        "feature_system_balance_state_asof": "margin_long",
                        "feature_system_balance_transition_state_asof": "margin_long->margin_long",
                        "feature_system_balance_persistence_bucket_asof": "steady",
                        "feature_connector_itl_state_asof": "published_restriction",
                        "feature_connector_itl_transition_state_asof": "published_restriction->published_restriction",
                        "feature_connector_notice_market_state_asof": "no_public_connector_restriction",
                        "feature_hour_of_day": 5.0,
                        "feature_origin_hour_of_day": 4.0,
                    }
                ]
            )
        )

        self.assertEqual(feature_frame.loc[0, "feature_internal_transfer_source_family_asof"], "missing")
        self.assertEqual(feature_frame.loc[0, "feature_internal_transfer_source_key_asof"], "missing")
        self.assertEqual(feature_frame.loc[0, "feature_internal_transfer_boundary_family_asof"], "missing")
        self.assertEqual(feature_frame.loc[0, "feature_connector_itl_source_key_asof"], "missing")

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

    def test_build_fact_drift_window_keeps_zero_activity_feature_only_shift_as_pass(self) -> None:
        rows = []
        for day_text, tier, notice_state, internal_tier, gate_state, system_balance_state in (
            (
                "2024-10-01",
                "no_price_signal",
                "no_public_connector_restriction",
                "gb_topology_transfer_gate_proxy",
                "capacity_unknown_reachable",
                "tight_margin_and_active_imbalance",
            ),
            (
                "2024-10-02",
                "reviewed",
                "known_upcoming_restriction",
                "reviewed_internal_constraint_boundary",
                "reviewed_boundary_cap",
                "active_imbalance",
            ),
        ):
            for hour in range(6):
                rows.append(
                    _opportunity_row(
                        f"{day_text}T0{hour}:00:00Z",
                        "east_anglia_offshore",
                        "R1_netback_GB_FR_DE_PL",
                        "ifa",
                        tier,
                        notice_state,
                        0.0,
                        50.0,
                        curtailment_selected_mwh=0.0,
                        deliverable_mw_proxy=0.0,
                        curtailment_source_tier="regional_proxy",
                        internal_transfer_evidence_tier=internal_tier,
                        internal_transfer_gate_state=gate_state,
                        system_balance_feed_available_flag=True,
                        system_balance_known_flag=True,
                        system_balance_active_flag=True,
                        system_balance_state=system_balance_state,
                    )
                )
        backtest = build_fact_backtest_prediction_hourly(
            pd.DataFrame(rows), model_key=MODEL_POTENTIAL_RATIO_V2, forecast_horizons=(1,)
        )
        drift = build_fact_drift_window(backtest)
        second_window = drift[drift["window_date"].eq(pd.Timestamp("2024-10-02T00:00:00Z"))].reset_index(drop=True)
        self.assertFalse(second_window.empty)
        self.assertTrue(second_window["feature_drift_score"].fillna(0.0).gt(0.0).all())
        self.assertTrue(second_window["target_drift_score"].fillna(0.0).eq(0.0).all())
        self.assertTrue(second_window["residual_drift_score"].fillna(0.0).eq(0.0).all())
        self.assertTrue(second_window["drift_state"].eq("pass").all())

    def test_build_fact_drift_window_keeps_well_predicted_reviewed_event_shift_as_pass(self) -> None:
        rows = []
        for day_text, actual_mwh, predicted_mwh in (
            ("2024-10-01", 0.0, 0.0),
            ("2024-10-02", 40.0, 39.6),
        ):
            for hour in range(10):
                interval_start = pd.Timestamp(f"{day_text}T{hour:02d}:00:00Z")
                route_tier = "capacity_unknown" if (day_text == "2024-10-02" and hour == 0) else "reviewed"
                rows.append(
                    {
                        "interval_start_utc": interval_start,
                        "interval_end_utc": interval_start + pd.Timedelta(hours=1),
                        "model_key": MODEL_POTENTIAL_RATIO_V2,
                        "forecast_horizon_hours": 1,
                        "forecast_horizon_label": "t+1h",
                        "cluster_key": "east_anglia_offshore",
                        "route_name": "R1_netback_GB_FR_DE_PL",
                        "prediction_eligible_flag": True,
                        "route_delivery_tier": route_tier,
                        "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                        "internal_transfer_gate_state": "reviewed_boundary_cap",
                        "connector_notice_market_state": "no_public_connector_restriction",
                        "system_balance_state": "no_public_system_balance",
                        "feature_system_balance_known_flag_asof": False,
                        "curtailment_source_tier": "cluster_truth_research" if actual_mwh > 0.0 else "regional_proxy",
                        "actual_opportunity_deliverable_mwh": actual_mwh,
                        "predicted_opportunity_deliverable_mwh": predicted_mwh,
                        "opportunity_deliverable_residual_mwh": actual_mwh - predicted_mwh,
                        "opportunity_deliverable_abs_error_mwh": abs(actual_mwh - predicted_mwh),
                        "actual_opportunity_gross_value_eur": actual_mwh * 50.0,
                        "predicted_opportunity_gross_value_eur": predicted_mwh * 50.0,
                        "opportunity_gross_value_residual_eur": (actual_mwh - predicted_mwh) * 50.0,
                        "opportunity_gross_value_abs_error_eur": abs(actual_mwh - predicted_mwh) * 50.0,
                        "source_lineage": "fact_backtest_prediction_hourly",
                    }
                )

        drift = build_fact_drift_window(pd.DataFrame(rows))
        second_window = drift[drift["window_date"].eq(pd.Timestamp("2024-10-02T00:00:00Z"))].reset_index(drop=True)
        route_rows = second_window[second_window["drift_scope"].eq("route_daily")]
        cluster_rows = second_window[second_window["drift_scope"].eq("cluster_daily")]

        self.assertFalse(route_rows.empty)
        self.assertFalse(cluster_rows.empty)
        self.assertTrue(route_rows["target_drift_score"].fillna(0.0).gt(0.0).all())
        self.assertTrue(cluster_rows["target_drift_score"].fillna(0.0).gt(0.0).all())
        self.assertTrue(route_rows["drift_state"].eq("pass").all())
        self.assertTrue(cluster_rows["drift_state"].eq("pass").all())

    def test_build_fact_drift_window_keeps_well_predicted_reviewed_close_shift_as_pass(self) -> None:
        rows = []
        for day_text, actual_mwh, predicted_mwh in (
            ("2024-10-01", 18.0, 17.5),
            ("2024-10-02", 0.0, 0.0),
        ):
            for hour in range(8):
                interval_start = pd.Timestamp(f"{day_text}T{hour:02d}:00:00Z")
                rows.append(
                    {
                        "interval_start_utc": interval_start,
                        "interval_end_utc": interval_start + pd.Timedelta(hours=1),
                        "model_key": MODEL_POTENTIAL_RATIO_V2,
                        "forecast_horizon_hours": 1,
                        "forecast_horizon_label": "t+1h",
                        "cluster_key": "dogger_hornsea_offshore",
                        "route_name": "R2_netback_GB_NL_DE_PL",
                        "prediction_eligible_flag": True,
                        "route_delivery_tier": "reviewed",
                        "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                        "internal_transfer_gate_state": "reviewed_boundary_cap",
                        "connector_notice_market_state": "known_upcoming_restriction",
                        "system_balance_state": "no_public_system_balance",
                        "feature_system_balance_known_flag_asof": False,
                        "curtailment_source_tier": "cluster_truth_research" if actual_mwh > 0.0 else "regional_proxy",
                        "actual_opportunity_deliverable_mwh": actual_mwh,
                        "predicted_opportunity_deliverable_mwh": predicted_mwh,
                        "opportunity_deliverable_residual_mwh": actual_mwh - predicted_mwh,
                        "opportunity_deliverable_abs_error_mwh": abs(actual_mwh - predicted_mwh),
                        "actual_opportunity_gross_value_eur": actual_mwh * 50.0,
                        "predicted_opportunity_gross_value_eur": predicted_mwh * 50.0,
                        "opportunity_gross_value_residual_eur": (actual_mwh - predicted_mwh) * 50.0,
                        "opportunity_gross_value_abs_error_eur": abs(actual_mwh - predicted_mwh) * 50.0,
                        "source_lineage": "fact_backtest_prediction_hourly",
                    }
                )

        drift = build_fact_drift_window(pd.DataFrame(rows))
        second_window = drift[drift["window_date"].eq(pd.Timestamp("2024-10-02T00:00:00Z"))].reset_index(drop=True)
        route_rows = second_window[second_window["drift_scope"].eq("route_daily")]
        cluster_rows = second_window[second_window["drift_scope"].eq("cluster_daily")]

        self.assertFalse(route_rows.empty)
        self.assertFalse(cluster_rows.empty)
        self.assertTrue(route_rows["target_drift_score"].fillna(0.0).gt(0.0).all())
        self.assertTrue(cluster_rows["target_drift_score"].fillna(0.0).gt(0.0).all())
        self.assertTrue(route_rows["drift_state"].eq("pass").all())
        self.assertTrue(cluster_rows["drift_state"].eq("pass").all())

    def test_build_fact_drift_window_keeps_first_reviewed_day_after_empty_warmup_as_pass(self) -> None:
        rows = []
        for hour in range(6):
            interval_start = pd.Timestamp(f"2024-09-30T{hour:02d}:00:00Z")
            rows.append(
                {
                    "interval_start_utc": interval_start,
                    "interval_end_utc": interval_start + pd.Timedelta(hours=1),
                    "model_key": MODEL_POTENTIAL_RATIO_V2,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "cluster_key": "east_anglia_offshore",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "prediction_eligible_flag": False,
                    "route_delivery_tier": "reviewed",
                    "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                    "internal_transfer_gate_state": "reviewed_boundary_cap",
                    "connector_notice_market_state": "known_upcoming_restriction",
                    "system_balance_state": "no_public_system_balance",
                    "feature_system_balance_known_flag_asof": False,
                    "curtailment_source_tier": "regional_proxy",
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "opportunity_deliverable_residual_mwh": 0.0,
                    "opportunity_deliverable_abs_error_mwh": 0.0,
                    "actual_opportunity_gross_value_eur": 0.0,
                    "predicted_opportunity_gross_value_eur": 0.0,
                    "opportunity_gross_value_residual_eur": 0.0,
                    "opportunity_gross_value_abs_error_eur": 0.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            )
        for hour in range(6):
            interval_start = pd.Timestamp(f"2024-10-01T{hour:02d}:00:00Z")
            rows.append(
                {
                    "interval_start_utc": interval_start,
                    "interval_end_utc": interval_start + pd.Timedelta(hours=1),
                    "model_key": MODEL_POTENTIAL_RATIO_V2,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "cluster_key": "east_anglia_offshore",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "prediction_eligible_flag": True,
                    "route_delivery_tier": "reviewed",
                    "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                    "internal_transfer_gate_state": "reviewed_boundary_cap",
                    "connector_notice_market_state": "known_upcoming_restriction",
                    "system_balance_state": "no_public_system_balance",
                    "feature_system_balance_known_flag_asof": False,
                    "curtailment_source_tier": "cluster_truth_research",
                    "actual_opportunity_deliverable_mwh": 3.0,
                    "predicted_opportunity_deliverable_mwh": 2.9,
                    "opportunity_deliverable_residual_mwh": 0.1,
                    "opportunity_deliverable_abs_error_mwh": 0.1,
                    "actual_opportunity_gross_value_eur": 150.0,
                    "predicted_opportunity_gross_value_eur": 145.0,
                    "opportunity_gross_value_residual_eur": 5.0,
                    "opportunity_gross_value_abs_error_eur": 5.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            )

        drift = build_fact_drift_window(pd.DataFrame(rows))
        target_window = drift[drift["window_date"].eq(pd.Timestamp("2024-10-01T00:00:00Z"))].reset_index(drop=True)
        route_rows = target_window[target_window["drift_scope"].eq("route_daily")]
        cluster_rows = target_window[target_window["drift_scope"].eq("cluster_daily")]

        self.assertFalse(route_rows.empty)
        self.assertFalse(cluster_rows.empty)
        self.assertTrue(route_rows["drift_state"].eq("pass").all())
        self.assertTrue(cluster_rows["drift_state"].eq("pass").all())

    def test_build_fact_drift_window_keeps_first_zero_activity_day_after_empty_warmup_as_pass(self) -> None:
        rows = []
        for hour in range(6):
            interval_start = pd.Timestamp(f"2024-09-30T{hour:02d}:00:00Z")
            rows.append(
                {
                    "interval_start_utc": interval_start,
                    "interval_end_utc": interval_start + pd.Timedelta(hours=1),
                    "model_key": MODEL_POTENTIAL_RATIO_V2,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "cluster_key": "shetland_wind",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "prediction_eligible_flag": False,
                    "route_delivery_tier": "no_price_signal",
                    "internal_transfer_evidence_tier": "gb_topology_transfer_gate_proxy",
                    "internal_transfer_gate_state": "capacity_unknown_reachable",
                    "connector_notice_market_state": "no_public_connector_restriction",
                    "system_balance_state": "no_public_system_balance",
                    "feature_system_balance_known_flag_asof": False,
                    "curtailment_source_tier": "regional_proxy",
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "opportunity_deliverable_residual_mwh": 0.0,
                    "opportunity_deliverable_abs_error_mwh": 0.0,
                    "actual_opportunity_gross_value_eur": 0.0,
                    "predicted_opportunity_gross_value_eur": 0.0,
                    "opportunity_gross_value_residual_eur": 0.0,
                    "opportunity_gross_value_abs_error_eur": 0.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            )
        for hour in range(6):
            interval_start = pd.Timestamp(f"2024-10-01T{hour:02d}:00:00Z")
            rows.append(
                {
                    "interval_start_utc": interval_start,
                    "interval_end_utc": interval_start + pd.Timedelta(hours=1),
                    "model_key": MODEL_POTENTIAL_RATIO_V2,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "cluster_key": "shetland_wind",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "prediction_eligible_flag": True,
                    "route_delivery_tier": "no_price_signal",
                    "internal_transfer_evidence_tier": "gb_topology_transfer_gate_proxy",
                    "internal_transfer_gate_state": "capacity_unknown_reachable",
                    "connector_notice_market_state": "known_upcoming_restriction",
                    "system_balance_state": "active_imbalance",
                    "feature_system_balance_known_flag_asof": True,
                    "curtailment_source_tier": "regional_proxy",
                    "actual_opportunity_deliverable_mwh": 0.0,
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "opportunity_deliverable_residual_mwh": 0.0,
                    "opportunity_deliverable_abs_error_mwh": 0.0,
                    "actual_opportunity_gross_value_eur": 0.0,
                    "predicted_opportunity_gross_value_eur": 0.0,
                    "opportunity_gross_value_residual_eur": 0.0,
                    "opportunity_gross_value_abs_error_eur": 0.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            )

        drift = build_fact_drift_window(pd.DataFrame(rows))
        target_window = drift[drift["window_date"].eq(pd.Timestamp("2024-10-01T00:00:00Z"))].reset_index(drop=True)
        route_rows = target_window[target_window["drift_scope"].eq("route_daily")]
        cluster_rows = target_window[target_window["drift_scope"].eq("cluster_daily")]

        self.assertFalse(route_rows.empty)
        self.assertFalse(cluster_rows.empty)
        self.assertTrue(route_rows["drift_state"].eq("pass").all())
        self.assertTrue(cluster_rows["drift_state"].eq("pass").all())

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
