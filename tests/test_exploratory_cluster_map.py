import tempfile
import unittest
from pathlib import Path

import pandas as pd

from exploratory_cluster_map import (
    EXPLORATORY_CLUSTER_MAP_HOURLY_TABLE,
    EXPLORATORY_CLUSTER_MAP_HTML,
    EXPLORATORY_CLUSTER_MAP_POINT_TABLE,
    build_fact_exploratory_cluster_map_hourly,
    materialize_exploratory_cluster_map,
)


def _opportunity_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "interval_start_utc": "2024-12-09T06:00:00Z",
                "interval_end_utc": "2024-12-09T07:00:00Z",
                "cluster_key": "east_anglia_offshore",
                "cluster_label": "East Anglia Offshore",
                "parent_region": "England/Wales",
                "hub_key": "ifa2",
                "route_name": "R1_netback_GB_FR_DE_PL",
                "route_price_feasible_flag": True,
                "export_candidate_flag": True,
                "internal_transfer_evidence_tier": "reviewed_internal_constraint_boundary",
                "route_price_score_eur_per_mwh": 65.0,
                "deliverable_mw_proxy": 350.0,
                "opportunity_deliverable_mwh": 120.0,
                "opportunity_gross_value_eur": 7800.0,
            },
            {
                "interval_start_utc": "2024-12-09T06:00:00Z",
                "interval_end_utc": "2024-12-09T07:00:00Z",
                "cluster_key": "east_anglia_offshore",
                "cluster_label": "East Anglia Offshore",
                "parent_region": "England/Wales",
                "hub_key": "britned",
                "route_name": "R2_netback_GB_NL_DE_PL",
                "route_price_feasible_flag": False,
                "export_candidate_flag": False,
                "internal_transfer_evidence_tier": "gb_topology_transfer_gate_proxy",
                "route_price_score_eur_per_mwh": -4.0,
                "deliverable_mw_proxy": 0.0,
                "opportunity_deliverable_mwh": 0.0,
                "opportunity_gross_value_eur": 0.0,
            },
        ]
    )


def _readiness_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_date": "2024-12-09T00:00:00Z",
                "model_key": "opportunity_potential_ratio_v2",
                "model_ready_flag": False,
                "model_readiness_state": "not_ready",
                "blocking_reasons": "proxy_internal_transfer_share_too_high",
            }
        ]
    )


class ExploratoryClusterMapTests(unittest.TestCase):
    def test_build_fact_exploratory_cluster_map_hourly_joins_readiness_and_counts_routes(self) -> None:
        fact = build_fact_exploratory_cluster_map_hourly(
            fact_curtailment_opportunity_hourly=_opportunity_rows(),
            fact_model_readiness_daily=_readiness_rows(),
        )

        self.assertEqual(len(fact), 1)
        row = fact.iloc[0]
        self.assertEqual(row["cluster_key"], "east_anglia_offshore")
        self.assertEqual(int(row["route_count"]), 2)
        self.assertEqual(int(row["feasible_route_count"]), 1)
        self.assertEqual(int(row["export_candidate_route_count"]), 1)
        self.assertEqual(int(row["proxy_internal_route_count"]), 1)
        self.assertEqual(int(row["reviewed_internal_route_count"]), 1)
        self.assertAlmostEqual(float(row["opportunity_deliverable_mwh_sum"]), 120.0)
        self.assertEqual(row["model_readiness_state"], "not_ready")
        self.assertEqual(row["blocking_reasons"], "proxy_internal_transfer_share_too_high")
        self.assertEqual(row["mapping_confidence"], "medium")

    def test_materialize_exploratory_cluster_map_writes_html_with_badges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            opportunity_dir = root / "opportunity"
            readiness_dir = root / "readiness"
            output_dir = root / "map"
            opportunity_dir.mkdir()
            readiness_dir.mkdir()
            _opportunity_rows().to_csv(
                opportunity_dir / "fact_curtailment_opportunity_hourly.csv",
                index=False,
            )
            _readiness_rows().to_csv(
                readiness_dir / "fact_model_readiness_daily.csv",
                index=False,
            )

            frames = materialize_exploratory_cluster_map(
                opportunity_input_path=opportunity_dir,
                readiness_input_path=readiness_dir,
                output_dir=output_dir,
            )

            self.assertIn(EXPLORATORY_CLUSTER_MAP_POINT_TABLE, frames)
            self.assertIn(EXPLORATORY_CLUSTER_MAP_HOURLY_TABLE, frames)
            self.assertTrue((output_dir / f"{EXPLORATORY_CLUSTER_MAP_POINT_TABLE}.csv").exists())
            self.assertTrue((output_dir / f"{EXPLORATORY_CLUSTER_MAP_HOURLY_TABLE}.csv").exists())
            html_path = output_dir / EXPLORATORY_CLUSTER_MAP_HTML
            self.assertTrue(html_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Exploratory Only", html)
            self.assertIn('badge("Confidence"', html)
            self.assertIn('badge("Readiness"', html)


if __name__ == "__main__":
    unittest.main()
