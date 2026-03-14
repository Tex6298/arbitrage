import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_readiness import (
    DEFAULT_CANDIDATE_MODEL_KEY,
    DEFAULT_READINESS_MODEL_KEY,
    MODEL_BLOCKER_PRIORITY_TABLE,
    MODEL_CANDIDATE_COMPARE_TABLE,
    MODEL_CANDIDATE_COMPARE_SUITE_TABLE,
    MODEL_CANDIDATE_COMPARE_WINDOW_TABLE,
    MODEL_READINESS_TABLE,
    build_fact_model_candidate_compare_daily,
    build_fact_model_candidate_compare_suite,
    build_fact_model_candidate_compare_window,
    build_fact_model_blocker_priority,
    build_fact_model_readiness_daily,
    materialize_model_readiness_daily,
    materialize_model_readiness_review,
)


def _prediction_row(
    interval_start_utc: str,
    *,
    horizon_hours: int,
    deliverable_abs_error: float,
    model_key: str = DEFAULT_READINESS_MODEL_KEY,
    route_name: str = "R1_netback_GB_FR_DE_PL",
    actual_deliverable_mwh: float = 10.0,
    internal_transfer_evidence_tier: str = "reviewed_internal_constraint_boundary",
    cluster_key: str = "dogger_hornsea_offshore",
    hub_key: str = "britned",
    route_delivery_tier: str = "reviewed",
) -> dict:
    return {
        "interval_start_utc": pd.Timestamp(interval_start_utc),
        "model_key": model_key,
        "forecast_horizon_hours": horizon_hours,
        "prediction_eligible_flag": True,
        "cluster_key": cluster_key,
        "hub_key": hub_key,
        "route_name": route_name,
        "actual_opportunity_deliverable_mwh": actual_deliverable_mwh,
        "opportunity_deliverable_abs_error_mwh": deliverable_abs_error,
        "internal_transfer_evidence_tier": internal_transfer_evidence_tier,
        "route_delivery_tier": route_delivery_tier,
    }


class ModelReadinessTests(unittest.TestCase):
    def test_build_fact_model_readiness_daily_flags_ready_when_thresholds_pass(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction_row("2024-10-01T00:00:00Z", horizon_hours=1, deliverable_abs_error=0.35),
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    horizon_hours=1,
                    deliverable_abs_error=0.45,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=20.0,
                ),
                _prediction_row("2024-10-01T00:00:00Z", horizon_hours=6, deliverable_abs_error=0.45),
            ]
        )
        drift = pd.DataFrame(
            [
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "route_daily",
                    "drift_state": "pass",
                },
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "cluster_daily",
                    "drift_state": "pass",
                },
            ]
        )

        readiness = build_fact_model_readiness_daily(predictions, drift)

        row = readiness.iloc[0]
        self.assertTrue(bool(row["model_ready_flag"]))
        self.assertEqual(row["model_readiness_state"], "ready_for_map")
        self.assertEqual(row["blocking_reasons"], "")
        self.assertAlmostEqual(float(row["overall_t_plus_1h_deliverable_mae_mwh"]), 0.4)
        self.assertAlmostEqual(float(row["overall_t_plus_6h_deliverable_mae_mwh"]), 0.45)
        self.assertAlmostEqual(float(row["gb_nl_t_plus_1h_deliverable_mae_mwh"]), 0.45)

    def test_build_fact_model_readiness_daily_flags_blockers_when_thresholds_fail(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction_row(
                    "2024-10-01T00:00:00Z",
                    horizon_hours=1,
                    deliverable_abs_error=1.8,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                    internal_transfer_evidence_tier="gb_topology_transfer_gate_proxy",
                    route_delivery_tier="capacity_unknown",
                ),
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    horizon_hours=1,
                    deliverable_abs_error=1.7,
                    actual_deliverable_mwh=40.0,
                    internal_transfer_evidence_tier="gb_topology_transfer_gate_proxy",
                    route_delivery_tier="capacity_unknown",
                ),
                _prediction_row(
                    "2024-10-01T00:00:00Z",
                    horizon_hours=6,
                    deliverable_abs_error=0.8,
                    actual_deliverable_mwh=40.0,
                ),
            ]
        )
        drift = pd.DataFrame(
            [
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "route_daily",
                    "drift_state": "warn",
                },
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "cluster_daily",
                    "drift_state": "warn",
                },
            ]
        )

        readiness = build_fact_model_readiness_daily(predictions, drift)

        row = readiness.iloc[0]
        self.assertFalse(bool(row["model_ready_flag"]))
        self.assertEqual(row["model_readiness_state"], "not_ready")
        self.assertIn("overall_t_plus_1h_mae_above_target", row["blocking_reasons"])
        self.assertIn("gb_nl_t_plus_1h_mae_above_target", row["blocking_reasons"])
        self.assertIn("proxy_internal_transfer_share_too_high", row["blocking_reasons"])
        self.assertIn("capacity_unknown_route_share_too_high", row["blocking_reasons"])
        self.assertIn("severe_fallback_route_focus_remaining", row["blocking_reasons"])
        self.assertIn("route_drift_warn", row["blocking_reasons"])
        self.assertIn("cluster_drift_warn", row["blocking_reasons"])

    def test_materialize_model_readiness_daily_writes_csv(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction_row("2024-10-01T00:00:00Z", horizon_hours=1, deliverable_abs_error=0.40),
                _prediction_row("2024-10-01T00:00:00Z", horizon_hours=6, deliverable_abs_error=0.45),
            ]
        )
        drift = pd.DataFrame(
            [
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "route_daily",
                    "drift_state": "pass",
                },
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "cluster_daily",
                    "drift_state": "pass",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_model_readiness_daily(tmp_dir, predictions, drift)
            self.assertEqual(set(frames), {MODEL_READINESS_TABLE})
            self.assertTrue((Path(tmp_dir) / f"{MODEL_READINESS_TABLE}.csv").exists())

    def test_build_fact_model_blocker_priority_ranks_readiness_blockers_from_slices_and_errors(self) -> None:
        readiness = pd.DataFrame(
            [
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "overall_t_plus_1h_deliverable_mae_mwh": 0.8,
                    "overall_t_plus_6h_deliverable_mae_mwh": 0.7,
                    "gb_nl_t_plus_1h_deliverable_mae_mwh": 2.1,
                    "proxy_internal_transfer_share_t_plus_1h": 0.35,
                    "reviewed_internal_transfer_share_t_plus_1h": 0.65,
                    "capacity_unknown_route_share_t_plus_1h": 0.0,
                    "route_warn_count_t_plus_1h": 1,
                    "cluster_warn_count_t_plus_1h": 0,
                    "severe_unresolved_focus_area_count_t_plus_1h": 0,
                    "model_ready_flag": False,
                    "model_readiness_state": "not_ready",
                    "blocking_reasons": "gb_nl_t_plus_1h_mae_above_target;route_drift_warn",
                    "source_lineage": "fact_backtest_prediction_hourly|fact_drift_window",
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "slice_dimension": "route_name",
                    "slice_value": "R2_netback_GB_NL_DE_PL",
                    "error_focus_area": "reviewed",
                    "error_reduction_priority_rank": 1.0,
                    "window_start_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "window_end_utc": pd.Timestamp("2024-10-08T00:00:00Z"),
                    "row_count": 20,
                    "eligible_row_count": 18,
                    "prediction_eligibility_rate": 0.9,
                    "actual_opportunity_deliverable_mean_mwh": 60.0,
                    "predicted_opportunity_deliverable_mean_mwh": 55.0,
                    "mae_opportunity_deliverable_mwh": 1.9,
                    "bias_opportunity_deliverable_mwh": -0.5,
                    "actual_opportunity_gross_value_mean_eur": 3000.0,
                    "predicted_opportunity_gross_value_mean_eur": 2750.0,
                    "mae_opportunity_gross_value_eur": 120.0,
                    "bias_opportunity_gross_value_eur": -70.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            ]
        )
        top_error = pd.DataFrame(
            [
                {
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "top_error_rank": 1,
                    "deliverable_abs_error_rank": 1,
                    "gross_value_abs_error_rank": 1,
                    "error_focus_area": "reviewed",
                    "date": pd.Timestamp("2024-10-01").date(),
                    "forecast_origin_utc": pd.Timestamp("2024-10-01T04:00:00Z"),
                    "interval_start_utc": pd.Timestamp("2024-10-01T05:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T06:00:00Z"),
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "cluster_mapping_confidence": "medium",
                    "cluster_connection_context": "context",
                    "cluster_preferred_hub_candidates": "britned, ifa",
                    "cluster_curation_version": "phase2",
                    "hub_key": "britned",
                    "hub_label": "BritNed",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "route_label": "GB->NL->DE->PL",
                    "route_border_key": "GB-NL",
                    "route_delivery_tier": "reviewed",
                    "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                    "internal_transfer_gate_state": "reviewed_boundary_cap",
                    "upstream_market_state": "intraday_stronger_than_day_ahead",
                    "system_balance_state": "tight_margin",
                    "connector_notice_market_state": "no_public_connector_restriction",
                    "curtailment_source_tier": "regional_proxy",
                    "prediction_basis": "ratio_cluster_route_upstream_market_state",
                    "training_sample_count": 4,
                    "actual_opportunity_deliverable_mwh": 170.0,
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "opportunity_deliverable_residual_mwh": 170.0,
                    "opportunity_deliverable_abs_error_mwh": 170.0,
                    "actual_opportunity_gross_value_eur": 9000.0,
                    "predicted_opportunity_gross_value_eur": 0.0,
                    "opportunity_gross_value_residual_eur": 9000.0,
                    "opportunity_gross_value_abs_error_eur": 9000.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            ]
        )
        drift = pd.DataFrame(
            [
                {
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "window_start_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "window_end_utc": pd.Timestamp("2024-10-02T00:00:00Z"),
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "drift_scope": "route_daily",
                    "cluster_key": pd.NA,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "row_count": 24,
                    "eligible_row_count": 24,
                    "reviewed_route_share": 1.0,
                    "capacity_unknown_route_share": 0.0,
                    "reviewed_internal_transfer_share": 1.0,
                    "proxy_internal_transfer_share": 0.0,
                    "blocked_internal_reviewed_share": 0.0,
                    "known_connector_restriction_share": 0.0,
                    "system_balance_stress_share": 0.5,
                    "system_balance_known_share": 1.0,
                    "truth_backed_curtailment_share": 0.0,
                    "actual_opportunity_deliverable_mean_mwh": 60.0,
                    "predicted_opportunity_deliverable_mean_mwh": 50.0,
                    "residual_bias_mwh": 10.0,
                    "residual_mae_mwh": 15.0,
                    "actual_opportunity_gross_value_mean_eur": 3000.0,
                    "predicted_opportunity_gross_value_mean_eur": 2500.0,
                    "residual_bias_eur": 500.0,
                    "residual_mae_eur": 700.0,
                    "feature_drift_score": 0.3,
                    "target_drift_score": 0.6,
                    "residual_drift_score": 0.7,
                    "drift_state": "warn",
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            ]
        )

        blocker = build_fact_model_blocker_priority(readiness, summary, top_error, drift)

        self.assertEqual(set(blocker["blocker_type"]), {"gb_nl_t_plus_1h_mae_above_target", "route_drift_warn"})
        gb_nl_row = blocker[blocker["blocker_type"] == "gb_nl_t_plus_1h_mae_above_target"].iloc[0]
        self.assertEqual(gb_nl_row["slice_dimension"], "route_name")
        self.assertEqual(gb_nl_row["slice_value"], "R2_netback_GB_NL_DE_PL")
        self.assertEqual(gb_nl_row["route_name"], "R2_netback_GB_NL_DE_PL")
        self.assertEqual(gb_nl_row["hub_key"], "britned")
        self.assertGreater(float(gb_nl_row["blocker_priority_score"]), 0.0)
        self.assertIn("inspect GB-NL route slices", gb_nl_row["recommended_next_step"])

    def test_build_fact_model_candidate_compare_daily_reports_shadow_deltas(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_READINESS_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=8.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_CANDIDATE_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=2.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
            ]
        )

        compare = build_fact_model_candidate_compare_daily(
            fact_backtest_prediction_hourly=predictions,
            fact_backtest_summary_slice=pd.DataFrame(),
            fact_backtest_top_error_hourly=pd.DataFrame(),
            fact_drift_window=pd.DataFrame(),
        )

        row = compare.iloc[0]
        self.assertEqual(row["baseline_model_key"], DEFAULT_READINESS_MODEL_KEY)
        self.assertEqual(row["candidate_model_key"], DEFAULT_CANDIDATE_MODEL_KEY)
        self.assertEqual(int(row["candidate_scope_row_count"]), 1)
        self.assertLess(float(row["overall_t_plus_1h_deliverable_mae_delta_mwh"]), 0.0)
        self.assertLess(float(row["gb_nl_t_plus_1h_deliverable_mae_delta_mwh"]), 0.0)
        self.assertLess(float(row["gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh"]), 0.0)
        self.assertEqual(row["promotion_state"], "candidate_beats_baseline")

    def test_build_fact_model_candidate_compare_window_rolls_up_window_metrics(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_READINESS_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=8.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_CANDIDATE_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=2.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
                _prediction_row(
                    "2024-10-02T01:00:00Z",
                    model_key=DEFAULT_READINESS_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=6.0,
                ),
                _prediction_row(
                    "2024-10-02T01:00:00Z",
                    model_key=DEFAULT_CANDIDATE_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=4.0,
                ),
            ]
        )

        window_compare = build_fact_model_candidate_compare_window(
            fact_backtest_prediction_hourly=predictions,
            fact_backtest_summary_slice=pd.DataFrame(),
            fact_backtest_top_error_hourly=pd.DataFrame(),
            fact_drift_window=pd.DataFrame(),
            benchmark_suite_name="suite_a",
            benchmark_window_key="window_a",
            benchmark_window_label="Window A",
            benchmark_window_start_date="2024-10-01",
            benchmark_window_end_date="2024-10-02",
            benchmark_window_family="guardrail",
            benchmark_role="guardrail",
            promotion_window_flag=False,
            display_order=1,
        )

        row = window_compare.iloc[0]
        self.assertEqual(window_compare.columns.tolist(), build_fact_model_candidate_compare_window(
            fact_backtest_prediction_hourly=pd.DataFrame(),
            fact_backtest_summary_slice=pd.DataFrame(),
            fact_backtest_top_error_hourly=pd.DataFrame(),
            fact_drift_window=pd.DataFrame(),
            benchmark_suite_name="suite_a",
            benchmark_window_key="window_a",
            benchmark_window_label="Window A",
            benchmark_window_start_date="2024-10-01",
            benchmark_window_end_date="2024-10-02",
            benchmark_window_family="guardrail",
            benchmark_role="guardrail",
            promotion_window_flag=False,
            display_order=1,
        ).columns.tolist())
        self.assertEqual(row["benchmark_window_key"], "window_a")
        self.assertEqual(int(row["window_day_count"]), 2)
        self.assertEqual(int(row["candidate_scope_row_count"]), 2)
        self.assertAlmostEqual(float(row["overall_t_plus_1h_deliverable_mae_delta_mwh"]), -4.0)
        self.assertAlmostEqual(float(row["gb_nl_t_plus_1h_deliverable_mae_delta_mwh"]), -6.0)
        self.assertAlmostEqual(float(row["gb_nl_reviewed_internal_t_plus_1h_deliverable_mae_delta_mwh"]), -6.0)
        self.assertEqual(row["promotion_state"], "candidate_beats_baseline")

    def test_build_fact_model_candidate_compare_suite_separates_promotion_windows(self) -> None:
        diagnostic_predictions = pd.DataFrame(
            [
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_READINESS_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=6.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_CANDIDATE_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=1.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
            ]
        )
        promotion_predictions = pd.DataFrame(
            [
                _prediction_row(
                    "2024-10-08T01:00:00Z",
                    model_key=DEFAULT_READINESS_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=2.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
                _prediction_row(
                    "2024-10-08T01:00:00Z",
                    model_key=DEFAULT_CANDIDATE_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=7.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
            ]
        )
        window_compare = pd.concat(
            [
                build_fact_model_candidate_compare_window(
                    fact_backtest_prediction_hourly=diagnostic_predictions,
                    fact_backtest_summary_slice=pd.DataFrame(),
                    fact_backtest_top_error_hourly=pd.DataFrame(),
                    fact_drift_window=pd.DataFrame(),
                    benchmark_suite_name="suite_a",
                    benchmark_window_key="diag",
                    benchmark_window_label="Diagnostic",
                    benchmark_window_start_date="2024-10-01",
                    benchmark_window_end_date="2024-10-07",
                    benchmark_window_family="diagnostic",
                    benchmark_role="diagnostic",
                    promotion_window_flag=False,
                    display_order=1,
                ),
                build_fact_model_candidate_compare_window(
                    fact_backtest_prediction_hourly=promotion_predictions,
                    fact_backtest_summary_slice=pd.DataFrame(),
                    fact_backtest_top_error_hourly=pd.DataFrame(),
                    fact_drift_window=pd.DataFrame(),
                    benchmark_suite_name="suite_a",
                    benchmark_window_key="holdout",
                    benchmark_window_label="Holdout",
                    benchmark_window_start_date="2024-10-08",
                    benchmark_window_end_date="2024-10-14",
                    benchmark_window_family="acceptance",
                    benchmark_role="acceptance",
                    promotion_window_flag=True,
                    display_order=2,
                ),
            ],
            ignore_index=True,
        )

        suite_compare = build_fact_model_candidate_compare_suite(window_compare)

        self.assertEqual(set(suite_compare["suite_scope"]), {"all_windows", "promotion_windows"})
        all_windows = suite_compare[suite_compare["suite_scope"] == "all_windows"].iloc[0]
        promotion_windows = suite_compare[suite_compare["suite_scope"] == "promotion_windows"].iloc[0]
        self.assertEqual(int(all_windows["window_count"]), 2)
        self.assertEqual(int(promotion_windows["window_count"]), 1)
        self.assertEqual(int(all_windows["candidate_beats_window_count"]), 1)
        self.assertEqual(int(all_windows["candidate_regresses_window_count"]), 1)
        self.assertEqual(all_windows["promotion_state"], "candidate_mixed")
        self.assertAlmostEqual(float(promotion_windows["overall_t_plus_1h_deliverable_mae_delta_mwh"]), 5.0)
        self.assertEqual(promotion_windows["promotion_state"], "candidate_regresses_baseline")

    def test_materialize_model_readiness_review_writes_both_outputs(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction_row("2024-10-01T00:00:00Z", horizon_hours=1, deliverable_abs_error=0.9),
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    horizon_hours=1,
                    deliverable_abs_error=1.8,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
                _prediction_row("2024-10-01T00:00:00Z", horizon_hours=6, deliverable_abs_error=0.6),
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "slice_dimension": "route_name",
                    "slice_value": "R2_netback_GB_NL_DE_PL",
                    "error_focus_area": "reviewed",
                    "error_reduction_priority_rank": 1.0,
                    "window_start_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "window_end_utc": pd.Timestamp("2024-10-02T00:00:00Z"),
                    "row_count": 2,
                    "eligible_row_count": 2,
                    "prediction_eligibility_rate": 1.0,
                    "actual_opportunity_deliverable_mean_mwh": 25.0,
                    "predicted_opportunity_deliverable_mean_mwh": 23.0,
                    "mae_opportunity_deliverable_mwh": 1.8,
                    "bias_opportunity_deliverable_mwh": 0.0,
                    "actual_opportunity_gross_value_mean_eur": 1000.0,
                    "predicted_opportunity_gross_value_mean_eur": 900.0,
                    "mae_opportunity_gross_value_eur": 80.0,
                    "bias_opportunity_gross_value_eur": 0.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            ]
        )
        top_error = pd.DataFrame(
            [
                {
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "forecast_horizon_label": "t+1h",
                    "top_error_rank": 1,
                    "deliverable_abs_error_rank": 1,
                    "gross_value_abs_error_rank": 1,
                    "error_focus_area": "reviewed",
                    "date": pd.Timestamp("2024-10-01").date(),
                    "forecast_origin_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "interval_start_utc": pd.Timestamp("2024-10-01T01:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T02:00:00Z"),
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "cluster_mapping_confidence": "medium",
                    "cluster_connection_context": "context",
                    "cluster_preferred_hub_candidates": "britned",
                    "cluster_curation_version": "phase2",
                    "hub_key": "britned",
                    "hub_label": "BritNed",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "route_label": "GB->NL->DE->PL",
                    "route_border_key": "GB-NL",
                    "route_delivery_tier": "reviewed",
                    "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                    "internal_transfer_gate_state": "reviewed_boundary_cap",
                    "upstream_market_state": "no_upstream_feed",
                    "system_balance_state": "no_public_system_balance",
                    "connector_notice_market_state": "no_public_connector_restriction",
                    "curtailment_source_tier": "regional_proxy",
                    "prediction_basis": "ratio_global",
                    "training_sample_count": 2,
                    "actual_opportunity_deliverable_mwh": 40.0,
                    "predicted_opportunity_deliverable_mwh": 0.0,
                    "opportunity_deliverable_residual_mwh": 40.0,
                    "opportunity_deliverable_abs_error_mwh": 40.0,
                    "actual_opportunity_gross_value_eur": 1000.0,
                    "predicted_opportunity_gross_value_eur": 0.0,
                    "opportunity_gross_value_residual_eur": 1000.0,
                    "opportunity_gross_value_abs_error_eur": 1000.0,
                    "source_lineage": "fact_backtest_prediction_hourly",
                }
            ]
        )
        drift = pd.DataFrame(
            [
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "route_daily",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "cluster_key": pd.NA,
                    "feature_drift_score": 0.2,
                    "target_drift_score": 0.5,
                    "residual_drift_score": 0.6,
                    "drift_state": "warn",
                    "eligible_row_count": 2,
                },
                {
                    "window_date": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "model_key": DEFAULT_READINESS_MODEL_KEY,
                    "forecast_horizon_hours": 1,
                    "drift_scope": "cluster_daily",
                    "route_name": pd.NA,
                    "cluster_key": "dogger_hornsea_offshore",
                    "feature_drift_score": 0.1,
                    "target_drift_score": 0.2,
                    "residual_drift_score": 0.3,
                    "drift_state": "warn",
                    "eligible_row_count": 2,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_model_readiness_review(tmp_dir, predictions, summary, top_error, drift)
            self.assertEqual(set(frames), {MODEL_READINESS_TABLE, MODEL_BLOCKER_PRIORITY_TABLE})
            self.assertTrue((Path(tmp_dir) / f"{MODEL_READINESS_TABLE}.csv").exists())
            self.assertTrue((Path(tmp_dir) / f"{MODEL_BLOCKER_PRIORITY_TABLE}.csv").exists())

    def test_materialize_model_readiness_review_writes_candidate_compare_when_v2_and_v3_exist(self) -> None:
        predictions = pd.DataFrame(
            [
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_READINESS_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=6.0,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
                _prediction_row(
                    "2024-10-01T01:00:00Z",
                    model_key=DEFAULT_CANDIDATE_MODEL_KEY,
                    horizon_hours=1,
                    deliverable_abs_error=1.5,
                    route_name="R2_netback_GB_NL_DE_PL",
                    actual_deliverable_mwh=40.0,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_model_readiness_review(
                tmp_dir,
                predictions,
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )
            self.assertEqual(
                set(frames),
                {
                    MODEL_READINESS_TABLE,
                    MODEL_BLOCKER_PRIORITY_TABLE,
                    MODEL_CANDIDATE_COMPARE_TABLE,
                },
            )
            self.assertTrue((Path(tmp_dir) / f"{MODEL_CANDIDATE_COMPARE_TABLE}.csv").exists())


if __name__ == "__main__":
    unittest.main()
