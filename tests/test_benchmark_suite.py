import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from benchmark_suite import (
    BENCHMARK_WINDOW_TABLE,
    MODEL_CANDIDATE_COMPARE_WINDOW_DAILY_TABLE,
    BenchmarkWindowSpec,
    load_benchmark_suite_manifest,
    materialize_model_benchmark_suite,
)
from model_readiness import MODEL_CANDIDATE_COMPARE_SUITE_TABLE, MODEL_CANDIDATE_COMPARE_WINDOW_TABLE
from opportunity_backtest import (
    BACKTEST_PREDICTION_TABLE,
    BACKTEST_SUMMARY_SLICE_TABLE,
    BACKTEST_TOP_ERROR_TABLE,
    DRIFT_WINDOW_TABLE,
)


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


if __name__ == "__main__":
    unittest.main()
