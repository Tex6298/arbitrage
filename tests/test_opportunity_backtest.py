import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opportunity_backtest import (
    BACKTEST_PREDICTION_TABLE,
    BACKTEST_SUMMARY_SLICE_TABLE,
    BACKTEST_TOP_ERROR_TABLE,
    DRIFT_WINDOW_TABLE,
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
) -> dict:
    interval_start = pd.Timestamp(interval_start_utc)
    interval_end = interval_start + pd.Timedelta(hours=1)
    interval_start_local = interval_start.tz_convert("Europe/London")
    interval_end_local = interval_end.tz_convert("Europe/London")
    if curtailment_selected_mwh is None:
        curtailment_selected_mwh = opportunity_deliverable_mwh
    if deliverable_mw_proxy is None:
        deliverable_mw_proxy = opportunity_deliverable_mwh
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
        "route_delivery_tier": route_delivery_tier,
        "connector_notice_market_state": connector_notice_market_state,
        "curtailment_source_tier": curtailment_source_tier,
        "curtailment_selected_mwh": curtailment_selected_mwh,
        "deliverable_mw_proxy": deliverable_mw_proxy,
        "opportunity_deliverable_mwh": opportunity_deliverable_mwh,
        "opportunity_gross_value_eur": opportunity_deliverable_mwh * deliverable_route_score_eur_per_mwh,
        "deliverable_route_score_eur_per_mwh": deliverable_route_score_eur_per_mwh,
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
        self.assertEqual(ranked.iloc[0]["error_focus_area"], "gb_fr_connector_route")

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
