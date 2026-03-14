import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd

from system_balance_market_state import (
    SYSTEM_BALANCE_MARKET_STATE_TABLE,
    _load_dataset_frame,
    build_fact_system_balance_market_state_hourly,
    materialize_system_balance_market_state_history,
    normalize_system_balance_dataset_frame,
    DATASET_SPECS,
)


def _spec(dataset_key: str):
    for spec in DATASET_SPECS:
        if spec.dataset_key == dataset_key:
            return spec
    raise AssertionError(f"missing dataset spec {dataset_key}")


class SystemBalanceMarketStateTests(unittest.TestCase):
    def test_normalize_system_balance_dataset_frame_uses_settlement_period_and_hourly_average(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "SettlementDate": "2024-10-01",
                    "SettlementPeriod": 1,
                    "PublishTime": "2024-09-30T23:20:00Z",
                    "Quantity": 100.0,
                },
                {
                    "SettlementDate": "2024-10-01",
                    "SettlementPeriod": 2,
                    "PublishTime": "2024-09-30T23:50:00Z",
                    "Quantity": 300.0,
                },
            ]
        )

        normalized = normalize_system_balance_dataset_frame(
            raw,
            _spec("IMBALNGC"),
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.iloc[0]["interval_start_utc"], pd.Timestamp("2024-09-30T23:00:00Z"))
        self.assertAlmostEqual(float(normalized.iloc[0]["system_balance_imbalance_mw"]), 200.0)
        self.assertEqual(int(normalized.iloc[0]["system_balance_imbalance_mw_source_row_count"]), 2)
        self.assertEqual(
            normalized.iloc[0]["system_balance_imbalance_mw_source_published_utc"],
            pd.Timestamp("2024-09-30T23:50:00Z"),
        )

    def test_build_fact_system_balance_market_state_hourly_handles_dst_transition_from_settlement_periods(self) -> None:
        dataset_frames = {
            "IMBALNGC": pd.DataFrame(
                [
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 3,
                        "PublishTime": "2024-03-31T00:45:00Z",
                        "Quantity": 1200.0,
                    },
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 4,
                        "PublishTime": "2024-03-31T01:10:00Z",
                        "Quantity": 800.0,
                    },
                ]
            ),
            "INDDEM": pd.DataFrame(
                [
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 3,
                        "PublishTime": "2024-03-31T00:45:00Z",
                        "Quantity": 40000.0,
                    },
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 4,
                        "PublishTime": "2024-03-31T01:10:00Z",
                        "Quantity": 40400.0,
                    },
                ]
            ),
            "INDGEN": pd.DataFrame(
                [
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 3,
                        "PublishTime": "2024-03-31T00:45:00Z",
                        "Quantity": 39200.0,
                    },
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 4,
                        "PublishTime": "2024-03-31T01:10:00Z",
                        "Quantity": 39500.0,
                    },
                ]
            ),
            "MELNGC": pd.DataFrame(
                [
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 3,
                        "PublishTime": "2024-03-31T00:45:00Z",
                        "Quantity": 900.0,
                    },
                    {
                        "SettlementDate": "2024-03-31",
                        "SettlementPeriod": 4,
                        "PublishTime": "2024-03-31T01:10:00Z",
                        "Quantity": 800.0,
                    },
                ]
            ),
        }

        fact = build_fact_system_balance_market_state_hourly(
            start_date=dt.date(2024, 3, 31),
            end_date=dt.date(2024, 3, 31),
            dataset_frames=dataset_frames,
        )

        self.assertEqual(len(fact), 1)
        row = fact.iloc[0]
        self.assertEqual(row["interval_start_utc"], pd.Timestamp("2024-03-31T01:00:00Z"))
        self.assertEqual(row["interval_start_local"], pd.Timestamp("2024-03-31T02:00:00+01:00"))
        self.assertEqual(row["interval_end_local"], pd.Timestamp("2024-03-31T03:00:00+01:00"))

    def test_build_fact_system_balance_market_state_hourly_marks_known_only_when_published_before_interval(self) -> None:
        dataset_frames = {
            "IMBALNGC": pd.DataFrame(
                [
                    {
                        "StartTime": "2024-10-01T09:00:00Z",
                        "PublishTime": "2024-10-01T08:45:00Z",
                        "Quantity": 1200.0,
                    },
                    {
                        "StartTime": "2024-10-01T10:00:00Z",
                        "PublishTime": "2024-10-01T10:15:00Z",
                        "Quantity": 1200.0,
                    },
                ]
            ),
            "INDDEM": pd.DataFrame(
                [
                    {
                        "StartTime": "2024-10-01T09:00:00Z",
                        "PublishTime": "2024-10-01T08:45:00Z",
                        "Quantity": 42000.0,
                    },
                    {
                        "StartTime": "2024-10-01T10:00:00Z",
                        "PublishTime": "2024-10-01T10:15:00Z",
                        "Quantity": 42000.0,
                    },
                ]
            ),
            "INDGEN": pd.DataFrame(
                [
                    {
                        "StartTime": "2024-10-01T09:00:00Z",
                        "PublishTime": "2024-10-01T08:45:00Z",
                        "Quantity": 40600.0,
                    },
                    {
                        "StartTime": "2024-10-01T10:00:00Z",
                        "PublishTime": "2024-10-01T10:15:00Z",
                        "Quantity": 40600.0,
                    },
                ]
            ),
            "MELNGC": pd.DataFrame(
                [
                    {
                        "StartTime": "2024-10-01T09:00:00Z",
                        "PublishTime": "2024-10-01T08:45:00Z",
                        "Quantity": 700.0,
                    },
                    {
                        "StartTime": "2024-10-01T10:00:00Z",
                        "PublishTime": "2024-10-01T10:15:00Z",
                        "Quantity": 700.0,
                    },
                ]
            ),
        }

        fact = build_fact_system_balance_market_state_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            dataset_frames=dataset_frames,
        )

        self.assertTrue(bool(fact.iloc[0]["system_balance_known_flag"]))
        self.assertFalse(bool(fact.iloc[1]["system_balance_known_flag"]))

    def test_build_fact_system_balance_market_state_hourly_does_not_forward_fill_missing_metrics(self) -> None:
        dataset_frames = {
            "IMBALNGC": pd.DataFrame(
                [
                    {
                        "StartTime": "2024-10-01T09:00:00Z",
                        "PublishTime": "2024-10-01T08:30:00Z",
                        "Quantity": 500.0,
                    }
                ]
            ),
            "INDDEM": pd.DataFrame(
                [
                    {
                        "StartTime": "2024-10-01T10:00:00Z",
                        "PublishTime": "2024-10-01T09:30:00Z",
                        "Quantity": 43000.0,
                    }
                ]
            ),
            "INDGEN": pd.DataFrame(columns=["StartTime", "PublishTime", "Quantity"]),
            "MELNGC": pd.DataFrame(columns=["StartTime", "PublishTime", "Quantity"]),
        }

        fact = build_fact_system_balance_market_state_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            dataset_frames=dataset_frames,
        )

        self.assertEqual(len(fact), 2)
        self.assertTrue(pd.isna(fact.iloc[0]["system_balance_indicated_demand_mw"]))
        self.assertTrue(pd.isna(fact.iloc[1]["system_balance_imbalance_mw"]))

    def test_load_dataset_frame_chunks_publish_window_to_one_day_requests(self) -> None:
        requests: list[tuple[str, str]] = []

        def _fake_fetch(url: str, source_name: str, api_key: str | None) -> bytes:
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            requests.append(
                (
                    query["publishDateTimeFrom"][0],
                    query["publishDateTimeTo"][0],
                )
            )
            return json.dumps({"data": [{"value": len(requests)}]}).encode("utf-8")

        with patch("system_balance_market_state._fetch_elexon_payload", side_effect=_fake_fetch):
            frame = _load_dataset_frame(
                _spec("IMBALNGC"),
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 7),
                api_key=None,
            )

        self.assertEqual(
            requests,
            [
                ("2024-09-30T23:00Z", "2024-10-01T23:00Z"),
                ("2024-10-01T23:00Z", "2024-10-02T23:00Z"),
                ("2024-10-02T23:00Z", "2024-10-03T23:00Z"),
                ("2024-10-03T23:00Z", "2024-10-04T23:00Z"),
                ("2024-10-04T23:00Z", "2024-10-05T23:00Z"),
                ("2024-10-05T23:00Z", "2024-10-06T23:00Z"),
                ("2024-10-06T23:00Z", "2024-10-07T23:00Z"),
            ],
        )
        self.assertEqual(len(frame), len(requests))

    def test_materialize_system_balance_market_state_history_writes_csv(self) -> None:
        fact = pd.DataFrame(
            [
                {
                    "date": dt.date(2024, 10, 1),
                    "interval_start_local": pd.Timestamp("2024-10-01T10:00:00+01:00"),
                    "interval_end_local": pd.Timestamp("2024-10-01T11:00:00+01:00"),
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T10:00:00Z"),
                    "system_balance_source_provider": "elexon",
                    "system_balance_source_family": "public_system_balance",
                    "system_balance_source_key": "IMBALNGC",
                    "system_balance_source_dataset_keys": "IMBALNGC",
                    "system_balance_source_published_utc": pd.Timestamp("2024-10-01T08:30:00Z"),
                    "system_balance_feed_available_flag": True,
                    "system_balance_known_flag": True,
                    "system_balance_active_flag": True,
                    "system_balance_state": "active_imbalance",
                    "system_balance_imbalance_mw": 500.0,
                    "system_balance_indicated_demand_mw": pd.NA,
                    "system_balance_indicated_generation_mw": pd.NA,
                    "system_balance_indicated_margin_mw": pd.NA,
                    "system_balance_demand_minus_generation_mw": pd.NA,
                    "system_balance_margin_ratio": pd.NA,
                    "system_balance_imbalance_direction_bucket": "imbalance_positive",
                    "system_balance_margin_direction_bucket": "margin_unknown",
                    "source_lineage": "elexon:IMBALNGC",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "system_balance_market_state.build_fact_system_balance_market_state_hourly",
                return_value=fact,
            ):
                frames = materialize_system_balance_market_state_history(
                    tmp_dir,
                    start_date=dt.date(2024, 10, 1),
                    end_date=dt.date(2024, 10, 1),
                )

            self.assertEqual(set(frames), {SYSTEM_BALANCE_MARKET_STATE_TABLE})
            self.assertTrue((Path(tmp_dir) / f"{SYSTEM_BALANCE_MARKET_STATE_TABLE}.csv").exists())


if __name__ == "__main__":
    unittest.main()
