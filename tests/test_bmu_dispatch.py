import datetime as dt
import unittest

import pandas as pd

from bmu_dispatch import build_fact_bmu_bid_offer_half_hourly, clip_raw_dispatch_rows_to_requested_window


def sample_dim_bmu_asset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "elexon_bm_unit": "T_TEST-1",
                "national_grid_bm_unit": "TEST-1",
                "bm_unit_name": "Test Wind 1",
                "lead_party_name": "Test Lead",
                "fuel_type": "WIND",
                "bm_unit_type": "GEN",
                "gsp_group_id": "_A",
                "gsp_group_name": "Test GSP",
                "generation_capacity_mw": 20.0,
                "mapping_status": "mapped",
                "mapping_confidence": "high",
                "mapping_rule": "test",
                "cluster_key": "moray_firth_offshore",
                "cluster_label": "Moray Firth Offshore",
                "parent_region": "Scotland",
            }
        ]
    )


class BMUDispatchTests(unittest.TestCase):
    def test_clip_raw_dispatch_rows_to_requested_window_excludes_next_day_sp1(self) -> None:
        start_utc = dt.datetime(2024, 10, 1, 23, 0, tzinfo=dt.timezone.utc)
        end_utc = dt.datetime(2024, 10, 2, 23, 0, tzinfo=dt.timezone.utc)
        raw = pd.DataFrame(
            [
                {
                    "settlementDate": "2024-10-02",
                    "settlementPeriod": 48,
                    "timeFrom": "2024-10-02T22:30:00Z",
                    "timeTo": "2024-10-02T23:00:00Z",
                    "bmUnit": "T_TEST-1",
                },
                {
                    "settlementDate": "2024-10-03",
                    "settlementPeriod": 1,
                    "timeFrom": "2024-10-02T23:00:00Z",
                    "timeTo": "2024-10-02T23:30:00Z",
                    "bmUnit": "T_TEST-1",
                },
            ]
        )

        clipped = clip_raw_dispatch_rows_to_requested_window(raw, start_utc=start_utc, end_utc=end_utc)
        self.assertEqual(len(clipped), 1)
        self.assertEqual(str(clipped.iloc[0]["settlementDate"]), "2024-10-02")

    def test_build_fact_bmu_bid_offer_half_hourly_excludes_sentinel_pairs_from_negative_bid_evidence(self) -> None:
        dim = sample_dim_bmu_asset()
        raw_bid_offer = pd.DataFrame(
            [
                {
                    "dataset": "BOD",
                    "settlementDate": "2024-10-01",
                    "settlementPeriod": 1,
                    "timeFrom": "2024-09-30T23:00:00Z",
                    "timeTo": "2024-09-30T23:30:00Z",
                    "pairId": -1,
                    "offer": -9998.0,
                    "bid": -9999.0,
                    "nationalGridBmUnit": "TEST-1",
                    "bmUnit": "T_TEST-1",
                },
                {
                    "dataset": "BOD",
                    "settlementDate": "2024-10-01",
                    "settlementPeriod": 1,
                    "timeFrom": "2024-09-30T23:00:00Z",
                    "timeTo": "2024-09-30T23:30:00Z",
                    "pairId": 1,
                    "offer": 9999.0,
                    "bid": 100.0,
                    "nationalGridBmUnit": "TEST-1",
                    "bmUnit": "T_TEST-1",
                },
            ]
        )

        fact = build_fact_bmu_bid_offer_half_hourly(dim, raw_bid_offer)
        first_row = fact.iloc[0]
        self.assertFalse(bool(first_row["negative_bid_available_flag"]))
        self.assertEqual(int(first_row["negative_bid_pair_count"]), 1)
        self.assertEqual(int(first_row["valid_negative_bid_pair_count"]), 0)
        self.assertEqual(int(first_row["sentinel_bid_pair_count"]), 1)
        self.assertEqual(int(first_row["sentinel_offer_pair_count"]), 1)
        self.assertEqual(int(first_row["sentinel_pair_count"]), 2)
        self.assertTrue(bool(first_row["sentinel_pair_available_flag"]))


if __name__ == "__main__":
    unittest.main()
