import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_state_feed import (
    UPSTREAM_MARKET_STATE_TABLE,
    build_fact_upstream_market_state_hourly,
    build_fact_upstream_market_state_hourly_from_price_frame,
    materialize_upstream_market_state_history,
    normalize_upstream_market_state_input_frame,
    write_normalized_upstream_market_state_input,
)


class MarketStateFeedTests(unittest.TestCase):
    def test_normalize_upstream_market_state_input_frame_computes_spreads_and_state(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T09:00:00Z",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "source_provider": "manual_curve",
                    "source_family": "route_curve",
                    "source_key": "r2_curve",
                    "source_published_utc": "2024-10-01T06:00:00Z",
                    "forward_price_eur_per_mwh": 40.0,
                    "day_ahead_price_eur_per_mwh": 55.0,
                    "intraday_price_eur_per_mwh": 70.0,
                }
            ]
        )

        normalized = normalize_upstream_market_state_input_frame(raw)
        row = normalized.iloc[0]
        self.assertEqual(row["route_name"], "R2_netback_GB_NL_DE_PL")
        self.assertTrue(bool(row["upstream_market_state_feed_available_flag"]))
        self.assertAlmostEqual(float(row["forward_to_day_ahead_spread_eur_per_mwh"]), 15.0)
        self.assertAlmostEqual(float(row["day_ahead_to_intraday_spread_eur_per_mwh"]), 15.0)
        self.assertEqual(row["forward_to_day_ahead_spread_bucket"], "spread_positive")
        self.assertEqual(row["day_ahead_to_intraday_spread_bucket"], "spread_positive")
        self.assertEqual(row["upstream_market_state"], "intraday_stronger_than_day_ahead")

    def test_build_fact_upstream_market_state_hourly_filters_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "market_state.csv"
            input_path.write_text(
                "\n".join(
                    [
                        "interval_start_utc,route_name,day_ahead_price_eur_per_mwh,intraday_price_eur_per_mwh",
                        "2024-09-30T23:00:00Z,R2_netback_GB_NL_DE_PL,40,45",
                        "2024-10-01T09:00:00Z,R2_netback_GB_NL_DE_PL,50,55",
                    ]
                ),
                encoding="utf-8",
            )
            fact = build_fact_upstream_market_state_hourly(
                start_date="2024-10-01",
                end_date="2024-10-01",
                input_path=input_path,
            )
            self.assertEqual(len(fact), 1)
            self.assertEqual(fact.iloc[0]["route_name"], "R2_netback_GB_NL_DE_PL")

    def test_build_fact_upstream_market_state_hourly_from_price_frame_uses_free_live_mapping(self) -> None:
        prices = pd.DataFrame(
            {
                "GB": [40.0, 60.0],
                "FR": [70.0, 58.0],
                "NL": [55.0, 85.0],
                "DE": [80.0, 82.0],
                "PL": [95.0, 97.0],
            },
            index=pd.to_datetime(["2024-10-01T09:00:00Z", "2024-10-01T10:00:00Z"], utc=True),
        )

        fact = build_fact_upstream_market_state_hourly_from_price_frame(
            prices,
            gb_source_provider="APXMIDP",
        )

        self.assertEqual(len(fact), 4)
        r1_first = fact[
            (fact["interval_start_utc"] == pd.Timestamp("2024-10-01T09:00:00Z"))
            & (fact["route_name"] == "R1_netback_GB_FR_DE_PL")
        ].iloc[0]
        self.assertEqual(r1_first["source_family"], "free_entsoe_day_ahead_plus_elexon_mid")
        self.assertEqual(r1_first["source_provider"], "elexon_mid:APXMIDP+entsoe_day_ahead")
        self.assertAlmostEqual(float(r1_first["forward_price_eur_per_mwh"]), 40.0)
        self.assertAlmostEqual(float(r1_first["day_ahead_price_eur_per_mwh"]), 70.0)
        self.assertEqual(r1_first["upstream_market_state"], "day_ahead_much_stronger_than_forward")

        r2_second = fact[
            (fact["interval_start_utc"] == pd.Timestamp("2024-10-01T10:00:00Z"))
            & (fact["route_name"] == "R2_netback_GB_NL_DE_PL")
        ].iloc[0]
        self.assertAlmostEqual(float(r2_second["forward_price_eur_per_mwh"]), 60.0)
        self.assertAlmostEqual(float(r2_second["day_ahead_price_eur_per_mwh"]), 85.0)
        self.assertEqual(r2_second["forward_to_day_ahead_spread_bucket"], "spread_strong_positive")

    def test_materialize_and_normalize_upstream_market_state_history_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "market_state_raw.csv"
            raw_path.write_text(
                "\n".join(
                    [
                        "timestamp_utc,route,day_ahead_price,intraday_price",
                        "2024-10-01T09:00:00Z,R1_netback_GB_FR_DE_PL,40,42",
                    ]
                ),
                encoding="utf-8",
            )
            normalized_path = Path(tmp_dir) / "market_state_normalized.csv"
            normalized = write_normalized_upstream_market_state_input(raw_path, normalized_path)
            self.assertEqual(len(normalized), 1)
            frames = materialize_upstream_market_state_history(
                output_dir=Path(tmp_dir) / "out",
                start_date="2024-10-01",
                end_date="2024-10-01",
                input_path=normalized_path,
            )
            self.assertEqual(set(frames), {UPSTREAM_MARKET_STATE_TABLE})
            self.assertTrue((Path(tmp_dir) / "out" / f"{UPSTREAM_MARKET_STATE_TABLE}.csv").exists())


if __name__ == "__main__":
    unittest.main()
