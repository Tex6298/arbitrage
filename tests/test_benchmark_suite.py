import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from benchmark_suite import (
    BENCHMARK_WINDOW_TABLE,
    MODEL_BENCHMARK_WINDOW_SCOUT_TABLE,
    MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE,
    REVIEWED_BUNDLE_BATCH_BLOCKER_SUMMARY_TABLE,
    REVIEWED_BUNDLE_BATCH_READINESS_DAILY_TABLE,
    REVIEWED_BUNDLE_BATCH_SCOUT_TABLE,
    REVIEWED_BUNDLE_BATCH_WINDOW_SUMMARY_TABLE,
    REVIEWED_BUNDLE_BATCH_WINDOW_TABLE,
    BenchmarkWindowSpec,
    build_fact_model_benchmark_window_scout,
    discover_reviewed_bundle_batch_windows,
    load_benchmark_suite_manifest,
    materialize_model_benchmark_suite,
    materialize_reviewed_bundle_batch_evaluation,
)
from model_readiness import (
    MODEL_BLOCKER_PRIORITY_TABLE,
    MODEL_CANDIDATE_COMPARE_SUITE_TABLE,
    MODEL_CANDIDATE_COMPARE_WINDOW_TABLE,
    MODEL_READINESS_TABLE,
)
from opportunity_backtest import (
    BACKTEST_PREDICTION_TABLE,
    BACKTEST_SUMMARY_SLICE_TABLE,
    BACKTEST_TOP_ERROR_TABLE,
    DRIFT_WINDOW_TABLE,
)

TEST_CANDIDATE_MODEL_KEY = "opportunity_gb_nl_reviewed_specialist_v3"


def _prediction_row(
    interval_start_utc: str,
    *,
    model_key: str,
    deliverable_abs_error: float,
    route_name: str = "R2_netback_GB_NL_DE_PL",
) -> dict:
    return {
        "interval_start_utc": pd.Timestamp(interval_start_utc),
        "model_key": model_key,
        "forecast_horizon_hours": 1,
        "prediction_eligible_flag": True,
        "cluster_key": "dogger_hornsea_offshore",
        "hub_key": "britned",
        "route_name": route_name,
        "actual_opportunity_deliverable_mwh": 40.0,
        "opportunity_deliverable_abs_error_mwh": deliverable_abs_error,
        "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
        "route_delivery_tier": "reviewed",
    }


class BenchmarkSuiteTests(unittest.TestCase):
    def test_discover_and_materialize_reviewed_bundle_batch_eval(self) -> None:
        def _bundle_input(start_text: str, end_text: str, *, deliverable_mwh: float, route_price: float) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp(start_text),
                        "interval_start_utc": pd.Timestamp(f"{start_text}T01:00:00Z"),
                        "route_name": "R2_netback_GB_NL_DE_PL",
                        "hub_key": "britned",
                        "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                        "opportunity_deliverable_mwh": deliverable_mwh,
                        "route_price_score_eur_per_mwh": route_price,
                        "deliverable_mw_proxy": deliverable_mwh,
                    },
                    {
                        "date": pd.Timestamp(end_text),
                        "interval_start_utc": pd.Timestamp(f"{end_text}T23:00:00Z"),
                        "route_name": "R2_netback_GB_NL_DE_PL",
                        "hub_key": "britned",
                        "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                        "opportunity_deliverable_mwh": 0.0,
                        "route_price_score_eur_per_mwh": route_price,
                        "deliverable_mw_proxy": 0.0,
                    },
                ]
            )

        def _readiness_frame(window_date: str, *, ready: bool, overall_t1: float, gb_nl_t1: float) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "window_date": pd.Timestamp(window_date, tz="UTC"),
                        "model_key": "opportunity_potential_ratio_v2",
                        "overall_t_plus_1h_deliverable_mae_mwh": overall_t1,
                        "overall_t_plus_6h_deliverable_mae_mwh": 0.25,
                        "gb_nl_t_plus_1h_deliverable_mae_mwh": gb_nl_t1,
                        "proxy_internal_transfer_share_t_plus_1h": 0.15,
                        "reviewed_internal_transfer_share_t_plus_1h": 0.85,
                        "capacity_unknown_route_share_t_plus_1h": 0.0,
                        "route_warn_count_t_plus_1h": 0 if ready else 1,
                        "cluster_warn_count_t_plus_1h": 0,
                        "severe_unresolved_focus_area_count_t_plus_1h": 0,
                        "model_ready_flag": ready,
                        "model_readiness_state": "ready_for_map" if ready else "not_ready",
                        "blocking_reasons": "" if ready else "route_drift_warn",
                        "source_lineage": "tests",
                    }
                ]
            )

        def _blocker_frame(window_date: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "window_date": pd.Timestamp(window_date, tz="UTC"),
                        "model_key": "opportunity_potential_ratio_v2",
                        "forecast_horizon_hours": 1,
                        "blocker_type": "route_drift_warn",
                        "blocker_scope": "route_drift_slice",
                        "slice_dimension": "route_name",
                        "slice_value": "R2_netback_GB_NL_DE_PL",
                        "blocker_slice_key": "route_name:R2_netback_GB_NL_DE_PL",
                        "route_name": "R2_netback_GB_NL_DE_PL",
                        "cluster_key": "dogger_hornsea_offshore",
                        "hub_key": "britned",
                        "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                        "route_delivery_tier": "reviewed",
                        "error_focus_area": "general",
                        "drift_scope": "route_daily",
                        "drift_state": "warn",
                        "eligible_row_count": 4,
                        "top_error_row_count": 2,
                        "summary_slice_eligible_row_count": 4,
                        "actual_volume_mwh": 42.0,
                        "mean_deliverable_abs_error_mwh": 0.9,
                        "max_deliverable_abs_error_mwh": 1.3,
                        "summary_mae_opportunity_deliverable_mwh": 0.9,
                        "summary_mae_opportunity_gross_value_eur": 12.0,
                        "summary_error_reduction_priority_rank": 1.0,
                        "feature_drift_score": 0.6,
                        "target_drift_score": 0.1,
                        "residual_drift_score": 0.2,
                        "blocker_priority_score": 2.5,
                        "blocker_priority_rank": 1.0,
                        "blocker_summary": "route drift",
                        "recommended_next_step": "tighten route-level regime features",
                        "source_lineage": "tests",
                    }
                ]
            )

        def fake_load(path: str | Path) -> pd.DataFrame:
            path_text = str(path)
            if "2024-12-07_2024-12-09" in path_text:
                return _bundle_input("2024-12-07", "2024-12-09", deliverable_mwh=42.0, route_price=80.0)
            return _bundle_input("2024-10-14", "2024-10-16", deliverable_mwh=0.0, route_price=0.0)

        def fake_backtest(*, output_dir: str | Path, **_: object) -> dict[str, pd.DataFrame]:
            output_dir_text = str(output_dir)
            if "2024-12-07_2024-12-09" in output_dir_text:
                predictions = pd.DataFrame(
                    [
                        {
                            "interval_start_utc": pd.Timestamp("2024-12-07T01:00:00Z"),
                            "model_key": "opportunity_potential_ratio_v2",
                            "forecast_horizon_hours": 1,
                            "prediction_eligible_flag": True,
                            "cluster_key": "dogger_hornsea_offshore",
                            "hub_key": "britned",
                            "route_name": "R2_netback_GB_NL_DE_PL",
                            "actual_opportunity_deliverable_mwh": 42.0,
                            "opportunity_deliverable_abs_error_mwh": 3.0,
                            "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                            "route_delivery_tier": "reviewed",
                        }
                    ]
                )
            else:
                predictions = pd.DataFrame(
                    [
                        {
                            "interval_start_utc": pd.Timestamp("2024-10-14T01:00:00Z"),
                            "model_key": "opportunity_potential_ratio_v2",
                            "forecast_horizon_hours": 1,
                            "prediction_eligible_flag": True,
                            "cluster_key": "dogger_hornsea_offshore",
                            "hub_key": "britned",
                            "route_name": "R2_netback_GB_NL_DE_PL",
                            "actual_opportunity_deliverable_mwh": 0.0,
                            "opportunity_deliverable_abs_error_mwh": 0.0,
                            "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                            "route_delivery_tier": "reviewed",
                        }
                    ]
                )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            predictions.to_csv(Path(output_dir) / f"{BACKTEST_PREDICTION_TABLE}.csv", index=False)
            return {
                BACKTEST_PREDICTION_TABLE: predictions,
                BACKTEST_SUMMARY_SLICE_TABLE: pd.DataFrame(),
                BACKTEST_TOP_ERROR_TABLE: pd.DataFrame(),
                DRIFT_WINDOW_TABLE: pd.DataFrame(),
            }

        def fake_readiness(*, output_dir: str | Path, **_: object) -> dict[str, pd.DataFrame]:
            output_dir_text = str(output_dir)
            if "2024-12-07_2024-12-09" in output_dir_text:
                readiness = pd.concat(
                    [
                        _readiness_frame("2024-12-07", ready=True, overall_t1=0.2, gb_nl_t1=0.1),
                        _readiness_frame("2024-12-08", ready=False, overall_t1=0.8, gb_nl_t1=0.4),
                    ],
                    ignore_index=True,
                )
                blocker = _blocker_frame("2024-12-08")
            else:
                readiness = _readiness_frame("2024-10-14", ready=True, overall_t1=0.0, gb_nl_t1=0.0)
                blocker = pd.DataFrame(columns=_blocker_frame("2024-12-08").columns)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            readiness.to_csv(Path(output_dir) / f"{MODEL_READINESS_TABLE}.csv", index=False)
            blocker.to_csv(Path(output_dir) / f"{MODEL_BLOCKER_PRIORITY_TABLE}.csv", index=False)
            return {
                MODEL_READINESS_TABLE: readiness,
                MODEL_BLOCKER_PRIORITY_TABLE: blocker,
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "curtailment_opportunity_live_britned_reviewed_2024-10-14_2024-10-16").mkdir()
            (root / "curtailment_opportunity_live_britned_reviewed_2024-12-07_2024-12-09").mkdir()
            (root / "curtailment_opportunity_live_britned_reviewed_refresh_2024-10-14_2024-10-16").mkdir()

            specs = discover_reviewed_bundle_batch_windows(root)
            self.assertEqual(
                [spec.benchmark_window_key for spec in specs],
                [
                    "curtailment_opportunity_live_britned_reviewed_2024-10-14_2024-10-16",
                    "curtailment_opportunity_live_britned_reviewed_2024-12-07_2024-12-09",
                ],
            )
            self.assertEqual(specs[0].readiness_start, dt.date(2024, 10, 14))
            self.assertEqual(specs[1].readiness_end, dt.date(2024, 12, 9))

            with patch("benchmark_suite.load_curtailment_opportunity_input", side_effect=fake_load), patch(
                "benchmark_suite.materialize_opportunity_backtest",
                side_effect=fake_backtest,
            ), patch(
                "benchmark_suite.materialize_model_readiness_review",
                side_effect=fake_readiness,
            ):
                frames = materialize_reviewed_bundle_batch_evaluation(
                    root / "batch_output",
                    root_dir=root,
                    model_key="opportunity_potential_ratio_v2",
                    forecast_horizons=[1, 6],
                    baseline_model_key="opportunity_potential_ratio_v2",
                )

            self.assertEqual(
                set(frames),
                {
                    REVIEWED_BUNDLE_BATCH_WINDOW_TABLE,
                    REVIEWED_BUNDLE_BATCH_SCOUT_TABLE,
                    REVIEWED_BUNDLE_BATCH_READINESS_DAILY_TABLE,
                    REVIEWED_BUNDLE_BATCH_WINDOW_SUMMARY_TABLE,
                    REVIEWED_BUNDLE_BATCH_BLOCKER_SUMMARY_TABLE,
                },
            )
            self.assertTrue((root / "batch_output" / f"{REVIEWED_BUNDLE_BATCH_WINDOW_TABLE}.csv").exists())
            self.assertTrue((root / "batch_output" / f"{REVIEWED_BUNDLE_BATCH_SCOUT_TABLE}.csv").exists())
            self.assertTrue((root / "batch_output" / f"{REVIEWED_BUNDLE_BATCH_READINESS_DAILY_TABLE}.csv").exists())
            self.assertTrue((root / "batch_output" / f"{REVIEWED_BUNDLE_BATCH_WINDOW_SUMMARY_TABLE}.csv").exists())
            self.assertTrue((root / "batch_output" / f"{REVIEWED_BUNDLE_BATCH_BLOCKER_SUMMARY_TABLE}.csv").exists())
            self.assertTrue(
                (
                    root
                    / "batch_output"
                    / "windows"
                    / "curtailment_opportunity_live_britned_reviewed_2024-12-07_2024-12-09"
                    / f"{MODEL_BENCHMARK_WINDOW_SCOUT_TABLE}.csv"
                ).exists()
            )

            scout = frames[REVIEWED_BUNDLE_BATCH_SCOUT_TABLE]
            dec_scout = scout[scout["benchmark_window_key"].str.contains("2024-12-07_2024-12-09")].iloc[0]
            oct_scout = scout[scout["benchmark_window_key"].str.contains("2024-10-14_2024-10-16")].iloc[0]
            self.assertTrue(bool(dec_scout["informative_window_flag"]))
            self.assertFalse(bool(oct_scout["informative_window_flag"]))

            summary = frames[REVIEWED_BUNDLE_BATCH_WINDOW_SUMMARY_TABLE]
            dec_summary = summary[summary["benchmark_window_key"].str.contains("2024-12-07_2024-12-09")].iloc[0]
            self.assertEqual(int(dec_summary["window_day_count"]), 2)
            self.assertEqual(int(dec_summary["ready_day_count"]), 1)
            self.assertEqual(int(dec_summary["not_ready_day_count"]), 1)
            self.assertEqual(dec_summary["informative_signal_basis"], "reviewed_actual_deliverable_mwh_sum")

            blocker_summary = frames[REVIEWED_BUNDLE_BATCH_BLOCKER_SUMMARY_TABLE]
            self.assertEqual(len(blocker_summary), 1)
            blocker_row = blocker_summary.iloc[0]
            self.assertEqual(blocker_row["blocker_type"], "route_drift_warn")
            self.assertEqual(int(blocker_row["blocker_day_count"]), 1)

    def test_load_benchmark_suite_manifest_sorts_rows_and_defaults_promotion_flag(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "benchmark_suite_name": "suite_a",
                    "benchmark_window_key": "holdout",
                    "benchmark_window_label": "Holdout",
                    "opportunity_input_path": "input_b",
                    "readiness_start": "2024-10-08",
                    "readiness_end": "2024-10-14",
                    "benchmark_role": "acceptance",
                    "display_order": 2,
                },
                {
                    "benchmark_suite_name": "suite_a",
                    "benchmark_window_key": "diag",
                    "benchmark_window_label": "Diagnostic",
                    "opportunity_input_path": "input_a",
                    "readiness_start": "2024-10-01",
                    "readiness_end": "2024-10-07",
                    "benchmark_role": "diagnostic",
                    "promotion_window_flag": False,
                    "display_order": 1,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "suite.csv"
            manifest.to_csv(manifest_path, index=False)
            specs = load_benchmark_suite_manifest(manifest_path)

        self.assertEqual([spec.benchmark_window_key for spec in specs], ["diag", "holdout"])
        self.assertFalse(specs[0].promotion_window_flag)
        self.assertTrue(specs[1].promotion_window_flag)

    def test_materialize_model_benchmark_suite_writes_root_rollups_and_window_outputs(self) -> None:
        benchmark_windows = [
            BenchmarkWindowSpec(
                benchmark_suite_name="suite_a",
                benchmark_window_key="diag",
                benchmark_window_label="Diagnostic",
                opportunity_input_path="input_diag",
                readiness_start=dt.date(2024, 10, 1),
                readiness_end=dt.date(2024, 10, 7),
                benchmark_window_family="diagnostic",
                benchmark_role="diagnostic",
                promotion_window_flag=False,
                display_order=1,
            ),
            BenchmarkWindowSpec(
                benchmark_suite_name="suite_a",
                benchmark_window_key="holdout",
                benchmark_window_label="Holdout",
                opportunity_input_path="input_holdout",
                readiness_start=dt.date(2024, 10, 8),
                readiness_end=dt.date(2024, 10, 14),
                benchmark_window_family="acceptance",
                benchmark_role="acceptance",
                promotion_window_flag=True,
                display_order=2,
            ),
        ]

        def fake_load(_: str) -> pd.DataFrame:
            return pd.DataFrame({"interval_start_utc": [pd.Timestamp("2024-10-01T00:00:00Z")]})

        def fake_backtest(*, output_dir: str | Path, **_: object) -> dict[str, pd.DataFrame]:
            output_dir_text = str(output_dir)
            if "diag" in output_dir_text:
                predictions = pd.DataFrame(
                    [
                        _prediction_row(
                            "2024-10-01T01:00:00Z",
                            model_key="opportunity_potential_ratio_v2",
                            deliverable_abs_error=6.0,
                        ),
                        _prediction_row(
                            "2024-10-01T01:00:00Z",
                            model_key="opportunity_gb_nl_reviewed_specialist_v3",
                            deliverable_abs_error=1.0,
                        ),
                    ]
                )
            else:
                predictions = pd.DataFrame(
                    [
                        _prediction_row(
                            "2024-10-08T01:00:00Z",
                            model_key="opportunity_potential_ratio_v2",
                            deliverable_abs_error=2.0,
                        ),
                        _prediction_row(
                            "2024-10-08T01:00:00Z",
                            model_key="opportunity_gb_nl_reviewed_specialist_v3",
                            deliverable_abs_error=7.0,
                        ),
                    ]
                )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            predictions.to_csv(Path(output_dir) / f"{BACKTEST_PREDICTION_TABLE}.csv", index=False)
            return {
                BACKTEST_PREDICTION_TABLE: predictions,
                BACKTEST_SUMMARY_SLICE_TABLE: pd.DataFrame(),
                BACKTEST_TOP_ERROR_TABLE: pd.DataFrame(),
                DRIFT_WINDOW_TABLE: pd.DataFrame(),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("benchmark_suite.load_curtailment_opportunity_input", side_effect=fake_load), patch(
                "benchmark_suite.materialize_opportunity_backtest",
                side_effect=fake_backtest,
            ):
                frames = materialize_model_benchmark_suite(
                    tmp_dir,
                    benchmark_windows=benchmark_windows,
                    model_key="all",
                    forecast_horizons=[1, 6, 24, 168],
                    candidate_model_key=TEST_CANDIDATE_MODEL_KEY,
                    manifest_source="testsuite",
                )

            self.assertEqual(
                set(frames),
                {
                    BENCHMARK_WINDOW_TABLE,
                    MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE,
                    MODEL_CANDIDATE_COMPARE_WINDOW_TABLE,
                    MODEL_CANDIDATE_COMPARE_SUITE_TABLE,
                },
            )
            self.assertTrue((Path(tmp_dir) / f"{BENCHMARK_WINDOW_TABLE}.csv").exists())
            self.assertTrue((Path(tmp_dir) / f"{MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE}.csv").exists())
            self.assertTrue((Path(tmp_dir) / f"{MODEL_CANDIDATE_COMPARE_WINDOW_TABLE}.csv").exists())
            self.assertTrue((Path(tmp_dir) / f"{MODEL_CANDIDATE_COMPARE_SUITE_TABLE}.csv").exists())
            self.assertTrue(
                (Path(tmp_dir) / "windows" / "diag" / "fact_model_candidate_compare_daily.csv").exists()
            )
            self.assertTrue(
                (Path(tmp_dir) / "windows" / "holdout" / "fact_model_candidate_compare_daily.csv").exists()
            )

            window_compare = frames[MODEL_CANDIDATE_COMPARE_WINDOW_TABLE]
            suite_compare = frames[MODEL_CANDIDATE_COMPARE_SUITE_TABLE]
            self.assertEqual(int(window_compare["candidate_scope_row_count"].sum()), 2)
            self.assertEqual(set(window_compare["promotion_state"]), {"candidate_beats_baseline", "candidate_regresses_baseline"})
            promotion_row = suite_compare[suite_compare["suite_scope"] == "promotion_windows"].iloc[0]
            self.assertEqual(int(promotion_row["window_count"]), 1)
            self.assertEqual(promotion_row["promotion_state"], "candidate_regresses_baseline")

    def test_materialize_model_benchmark_suite_defaults_to_no_candidate_compare(self) -> None:
        benchmark_windows = [
            BenchmarkWindowSpec(
                benchmark_suite_name="suite_a",
                benchmark_window_key="diag",
                benchmark_window_label="Diagnostic",
                opportunity_input_path="input_diag",
                readiness_start=dt.date(2024, 10, 1),
                readiness_end=dt.date(2024, 10, 7),
                benchmark_window_family="diagnostic",
                benchmark_role="diagnostic",
                promotion_window_flag=False,
                display_order=1,
            ),
        ]

        def fake_load(_: str) -> pd.DataFrame:
            return pd.DataFrame({"interval_start_utc": [pd.Timestamp("2024-10-01T00:00:00Z")]})

        def fake_backtest(*, output_dir: str | Path, **_: object) -> dict[str, pd.DataFrame]:
            predictions = pd.DataFrame(
                [
                    _prediction_row(
                        "2024-10-01T01:00:00Z",
                        model_key="opportunity_potential_ratio_v2",
                        deliverable_abs_error=6.0,
                    ),
                    _prediction_row(
                        "2024-10-01T01:00:00Z",
                        model_key=TEST_CANDIDATE_MODEL_KEY,
                        deliverable_abs_error=1.0,
                    ),
                ]
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return {
                BACKTEST_PREDICTION_TABLE: predictions,
                BACKTEST_SUMMARY_SLICE_TABLE: pd.DataFrame(),
                BACKTEST_TOP_ERROR_TABLE: pd.DataFrame(),
                DRIFT_WINDOW_TABLE: pd.DataFrame(),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("benchmark_suite.load_curtailment_opportunity_input", side_effect=fake_load), patch(
                "benchmark_suite.materialize_opportunity_backtest",
                side_effect=fake_backtest,
            ):
                frames = materialize_model_benchmark_suite(
                    tmp_dir,
                    benchmark_windows=benchmark_windows,
                    model_key="all",
                    forecast_horizons=[1, 6, 24, 168],
                    manifest_source="testsuite",
                )

            self.assertTrue(frames[MODEL_CANDIDATE_COMPARE_WINDOW_TABLE].empty)
            self.assertTrue(frames[MODEL_CANDIDATE_COMPARE_SUITE_TABLE].empty)
            self.assertTrue(frames[MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE].empty)

    def test_build_fact_model_benchmark_window_scout_marks_actual_signal_windows_informative(self) -> None:
        opportunity = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-12-09"),
                    "interval_start_utc": pd.Timestamp("2024-12-09T06:00:00Z"),
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "hub_key": "britned",
                    "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                    "opportunity_deliverable_mwh": 42.0,
                    "route_price_score_eur_per_mwh": 80.0,
                    "deliverable_mw_proxy": 120.0,
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "model_key": "opportunity_potential_ratio_v2",
                    "forecast_horizon_hours": 1,
                    "prediction_eligible_flag": True,
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "hub_key": "britned",
                    "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                    "opportunity_deliverable_abs_error_mwh": 3.5,
                }
            ]
        )

        scout = build_fact_model_benchmark_window_scout(
            fact_curtailment_opportunity_hourly=opportunity,
            fact_backtest_prediction_hourly=predictions,
            benchmark_window_key="dec_window",
            benchmark_window_label="December Window",
            benchmark_window_start_date="2024-12-07",
            benchmark_window_end_date="2024-12-09",
            opportunity_input_path="input_dir",
        )

        row = scout.iloc[0]
        self.assertEqual(scout.columns.tolist(), build_fact_model_benchmark_window_scout(
            fact_curtailment_opportunity_hourly=pd.DataFrame(),
            fact_backtest_prediction_hourly=pd.DataFrame(),
            benchmark_window_key="empty",
            benchmark_window_label="Empty",
            benchmark_window_start_date="2024-12-01",
            benchmark_window_end_date="2024-12-01",
            opportunity_input_path="input_dir",
        ).columns.tolist())
        self.assertEqual(int(row["specialist_scope_row_count"]), 1)
        self.assertAlmostEqual(float(row["specialist_scope_actual_opportunity_deliverable_mwh_sum"]), 42.0)
        self.assertTrue(bool(row["informative_window_flag"]))
        self.assertEqual(row["informative_signal_basis"], "reviewed_actual_deliverable_mwh_sum")


if __name__ == "__main__":
    unittest.main()
