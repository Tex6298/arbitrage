import datetime as dt
import unittest

import pandas as pd

from gb_transfer_boundary_reviewed import (
    GB_TRANSFER_BOUNDARY_REVIEWED_TABLE,
    build_fact_gb_transfer_boundary_reviewed_hourly,
)


def _boundary_row(
    boundary_key: str,
    interval_start_utc: str,
    *,
    limit_mw: float,
    flow_mw: float,
    remaining_headroom_mw: float,
    utilization_ratio: float,
    boundary_state: str,
) -> dict:
    interval_start = pd.Timestamp(interval_start_utc)
    return {
        "date": interval_start.date(),
        "interval_start_local": interval_start.tz_convert("Europe/London"),
        "interval_end_local": (interval_start + pd.Timedelta(minutes=30)).tz_convert("Europe/London"),
        "interval_start_utc": interval_start,
        "interval_end_utc": interval_start + pd.Timedelta(minutes=30),
        "boundary_key": boundary_key,
        "boundary_label": boundary_key,
        "source_key": "neso_day_ahead_constraint_boundary",
        "source_label": "NESO day-ahead constraint flows and limits",
        "source_provider": "neso",
        "source_dataset_id": "dataset-id",
        "source_resource_id": "resource-id",
        "source_resource_name": "day_ahead_boundary",
        "source_document_url": "https://example.com/day-ahead-boundary",
        "target_is_proxy": False,
        "limit_mw": limit_mw,
        "flow_mw": flow_mw,
        "remaining_headroom_mw": remaining_headroom_mw,
        "utilization_ratio": utilization_ratio,
        "boundary_state": boundary_state,
    }


class GbTransferBoundaryReviewedTests(unittest.TestCase):
    def test_build_fact_gb_transfer_boundary_reviewed_hourly_blocks_flowsth_corridor(self) -> None:
        boundary = pd.DataFrame(
            [
                _boundary_row(
                    "FLOWSTH",
                    "2024-10-01T00:00:00Z",
                    limit_mw=1000.0,
                    flow_mw=1000.0,
                    remaining_headroom_mw=0.0,
                    utilization_ratio=1.0,
                    boundary_state="constraint_boundary_at_or_above_limit",
                )
            ]
        )

        fact = build_fact_gb_transfer_boundary_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            day_ahead_constraint_boundary=boundary,
        )

        row = fact[
            (fact["cluster_key"] == "east_anglia_offshore")
            & (fact["hub_key"] == "britned")
            & (fact["interval_start_utc"] == pd.Timestamp("2024-10-01T00:00:00Z"))
        ].iloc[0]
        self.assertEqual(row["reviewed_gate_state"], "blocked_reviewed_boundary")
        self.assertEqual(row["reviewed_evidence_tier"], "reviewed_internal_constraint_boundary")
        self.assertEqual(row["source_key"], "fact_day_ahead_constraint_boundary_half_hourly:FLOWSTH")
        self.assertEqual(row["source_provider"], "neso")

    def test_build_fact_gb_transfer_boundary_reviewed_hourly_skips_non_tightening_rows(self) -> None:
        boundary = pd.DataFrame(
            [
                _boundary_row(
                    "FLOWSTH",
                    "2024-10-01T00:00:00Z",
                    limit_mw=100000.0,
                    flow_mw=1000.0,
                    remaining_headroom_mw=99000.0,
                    utilization_ratio=0.01,
                    boundary_state="constraint_boundary_available",
                )
            ]
        )

        fact = build_fact_gb_transfer_boundary_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            day_ahead_constraint_boundary=boundary,
        )

        self.assertTrue(fact.empty)
        self.assertEqual(list(fact.columns), list(build_fact_gb_transfer_boundary_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            day_ahead_constraint_boundary=pd.DataFrame(),
        ).columns))

    def test_build_fact_gb_transfer_boundary_reviewed_hourly_scotland_rules_stay_scotland_only(self) -> None:
        boundary = pd.DataFrame(
            [
                _boundary_row(
                    "NKILGRMO",
                    "2024-10-01T00:00:00Z",
                    limit_mw=2000.0,
                    flow_mw=1950.0,
                    remaining_headroom_mw=50.0,
                    utilization_ratio=0.975,
                    boundary_state="constraint_boundary_tight",
                )
            ]
        )

        fact = build_fact_gb_transfer_boundary_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            day_ahead_constraint_boundary=boundary,
        )

        self.assertFalse(fact.empty)
        self.assertEqual(set(fact["parent_region"]), {"Scotland"})
        self.assertTrue((fact["boundary_key"] == "NKILGRMO").all())

    def test_build_fact_gb_transfer_boundary_reviewed_hourly_maps_seimp_to_east_anglia_corridors(self) -> None:
        boundary = pd.DataFrame(
            [
                _boundary_row(
                    "SEIMPPR23",
                    "2024-10-01T00:00:00Z",
                    limit_mw=800.0,
                    flow_mw=760.0,
                    remaining_headroom_mw=40.0,
                    utilization_ratio=0.95,
                    boundary_state="constraint_boundary_tight",
                )
            ]
        )

        fact = build_fact_gb_transfer_boundary_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            day_ahead_constraint_boundary=boundary,
        )

        self.assertFalse(fact.empty)
        self.assertEqual(
            set(fact["cluster_key"]),
            {"east_anglia_offshore", "humber_offshore", "dogger_hornsea_offshore"},
        )
        self.assertEqual(set(fact["hub_key"]), {"britned", "ifa", "ifa2", "eleclink"})
        self.assertTrue((fact["reviewed_gate_state"].isin(["reviewed_boundary_tight", "blocked_reviewed_boundary"])).all())

    def test_build_fact_gb_transfer_boundary_reviewed_hourly_maps_gm_snow_to_scotland_only(self) -> None:
        boundary = pd.DataFrame(
            [
                _boundary_row(
                    "GM+SNOW5A",
                    "2024-10-01T00:00:00Z",
                    limit_mw=900.0,
                    flow_mw=880.0,
                    remaining_headroom_mw=20.0,
                    utilization_ratio=0.977,
                    boundary_state="constraint_boundary_tight",
                )
            ]
        )

        fact = build_fact_gb_transfer_boundary_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            day_ahead_constraint_boundary=boundary,
        )

        self.assertFalse(fact.empty)
        self.assertEqual(set(fact["parent_region"]), {"Scotland"})
        self.assertTrue((fact["boundary_key"] == "GM+SNOW5A").all())


if __name__ == "__main__":
    unittest.main()
