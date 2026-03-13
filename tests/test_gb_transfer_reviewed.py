import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gb_transfer_reviewed import (
    GB_TRANSFER_REVIEWED_HOURLY_TABLE,
    GB_TRANSFER_REVIEWED_PERIOD_TABLE,
    build_fact_gb_transfer_review_policy,
    build_fact_gb_transfer_reviewed_hourly,
    build_fact_gb_transfer_reviewed_period,
    materialize_gb_transfer_reviewed_history,
    write_normalized_gb_transfer_reviewed_input,
)


class GbTransferReviewedTests(unittest.TestCase):
    def test_write_normalized_gb_transfer_reviewed_input_parses_txt_delivery_period_rows(self) -> None:
        raw_text = "\n".join(
            [
                "Cluster Key\tHub Key\tSource Key\tDelivery Date\tDelivery period (GMT)\tCapacity limit MW\tSource Document Title",
                "east_anglia_offshore\teleclink\tinternal_boundary_restriction\t2024-10-01\t00:00 - 01:00\t200\tBoundary notice",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "gb_transfer_reviewed_raw.txt"
            out_path = Path(tmp_dir) / "gb_transfer_reviewed_normalized.csv"
            raw_path.write_text(raw_text, encoding="utf-8")

            normalized = write_normalized_gb_transfer_reviewed_input(raw_path, out_path)

            self.assertTrue(out_path.exists())
            self.assertEqual(len(normalized), 1)
            row = normalized.iloc[0]
            self.assertEqual(row["cluster_key"], "east_anglia_offshore")
            self.assertEqual(row["hub_key"], "eleclink")
            self.assertEqual(row["period_start_utc"], pd.Timestamp("2024-10-01T00:00:00Z"))
            self.assertEqual(row["period_end_utc"], pd.Timestamp("2024-10-01T01:00:00Z"))

    def test_build_fact_gb_transfer_reviewed_hourly_prefers_cluster_specific_over_region_scope(self) -> None:
        reviewed_period = pd.DataFrame(
            [
                {
                    "scope_key": "England/Wales",
                    "scope_granularity": "parent_region_hub",
                    "cluster_key": pd.NA,
                    "cluster_label": pd.NA,
                    "parent_region": "England/Wales",
                    "hub_key": "eleclink",
                    "hub_label": "ElecLink",
                    "reviewed_scope": "gb_internal_transfer_reviewed_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_internal_transfer_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_internal_period",
                    "reviewed_gate_state": pd.NA,
                    "period_start_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T02:00:00Z"),
                    "period_timezone": "UTC",
                    "approx_cluster_capacity_mw": pd.NA,
                    "region_approx_capacity_mw": 1000.0,
                    "reviewed_capacity_limit_mw": 100.0,
                    "reviewed_gate_fraction": pd.NA,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "public_boundary_doc",
                    "source_key": "internal_boundary_restriction",
                    "source_label": "Boundary restriction",
                    "source_document_title": "Boundary restriction",
                    "source_document_url": "https://example.com/boundary",
                    "source_reference": "BOUNDARY-1",
                    "source_published_utc": pd.Timestamp("2024-09-30T09:00:00Z"),
                    "source_published_date": dt.date(2024, 9, 30),
                    "review_group_key": "region",
                    "source_revision_rank": 1,
                    "review_note": pd.NA,
                    "target_is_proxy": False,
                },
                {
                    "scope_key": "east_anglia_offshore",
                    "scope_granularity": "cluster_hub",
                    "cluster_key": "east_anglia_offshore",
                    "cluster_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "hub_key": "eleclink",
                    "hub_label": "ElecLink",
                    "reviewed_scope": "gb_internal_transfer_reviewed_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_internal_transfer_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_internal_period",
                    "reviewed_gate_state": pd.NA,
                    "period_start_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T02:00:00Z"),
                    "period_timezone": "UTC",
                    "approx_cluster_capacity_mw": 300.0,
                    "region_approx_capacity_mw": 1000.0,
                    "reviewed_capacity_limit_mw": 150.0,
                    "reviewed_gate_fraction": pd.NA,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "public_constraint_doc",
                    "source_key": "public_constraint_period",
                    "source_label": "Cluster review",
                    "source_document_title": "Cluster review",
                    "source_document_url": "https://example.com/cluster",
                    "source_reference": "CLUSTER-1",
                    "source_published_utc": pd.Timestamp("2024-09-30T10:00:00Z"),
                    "source_published_date": dt.date(2024, 9, 30),
                    "review_group_key": "cluster",
                    "source_revision_rank": 1,
                    "review_note": pd.NA,
                    "target_is_proxy": False,
                },
            ]
        )
        policy = build_fact_gb_transfer_review_policy(reviewed_period)

        hourly = build_fact_gb_transfer_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            reviewed_period=reviewed_period,
            review_policy=policy,
        )

        row = hourly[
            (hourly["cluster_key"] == "east_anglia_offshore")
            & (hourly["hub_key"] == "eleclink")
            & (hourly["interval_start_utc"] == pd.Timestamp("2024-10-01T00:00:00Z"))
        ].iloc[0]
        self.assertEqual(row["scope_granularity"], "cluster_hub")
        self.assertEqual(row["source_key"], "public_constraint_period")
        self.assertAlmostEqual(float(row["reviewed_capacity_limit_mw"]), 150.0)

    def test_build_fact_gb_transfer_reviewed_hourly_prefers_lower_capacity_within_same_scope(self) -> None:
        reviewed_period = pd.DataFrame(
            [
                {
                    "scope_key": "east_anglia_offshore",
                    "scope_granularity": "cluster_hub",
                    "cluster_key": "east_anglia_offshore",
                    "cluster_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "hub_key": "eleclink",
                    "hub_label": "ElecLink",
                    "reviewed_scope": "gb_internal_transfer_reviewed_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_internal_transfer_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_internal_period",
                    "reviewed_gate_state": pd.NA,
                    "period_start_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T02:00:00Z"),
                    "period_timezone": "UTC",
                    "approx_cluster_capacity_mw": 300.0,
                    "region_approx_capacity_mw": 1000.0,
                    "reviewed_capacity_limit_mw": 180.0,
                    "reviewed_gate_fraction": pd.NA,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "public_boundary_doc",
                    "source_key": "internal_boundary_restriction",
                    "source_label": "Boundary restriction",
                    "source_document_title": "Boundary restriction",
                    "source_document_url": "https://example.com/boundary",
                    "source_reference": "BOUNDARY-1",
                    "source_published_utc": pd.Timestamp("2024-09-30T09:00:00Z"),
                    "source_published_date": dt.date(2024, 9, 30),
                    "review_group_key": "boundary",
                    "source_revision_rank": 1,
                    "review_note": pd.NA,
                    "target_is_proxy": False,
                },
                {
                    "scope_key": "east_anglia_offshore",
                    "scope_granularity": "cluster_hub",
                    "cluster_key": "east_anglia_offshore",
                    "cluster_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "hub_key": "eleclink",
                    "hub_label": "ElecLink",
                    "reviewed_scope": "gb_internal_transfer_reviewed_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_internal_transfer_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_internal_period",
                    "reviewed_gate_state": pd.NA,
                    "period_start_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T02:00:00Z"),
                    "period_timezone": "UTC",
                    "approx_cluster_capacity_mw": 300.0,
                    "region_approx_capacity_mw": 1000.0,
                    "reviewed_capacity_limit_mw": 120.0,
                    "reviewed_gate_fraction": pd.NA,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "public_transfer_review",
                    "source_key": "reviewed_transfer_cap_window",
                    "source_label": "Transfer cap",
                    "source_document_title": "Transfer cap",
                    "source_document_url": "https://example.com/cap",
                    "source_reference": "CAP-1",
                    "source_published_utc": pd.Timestamp("2024-09-30T08:00:00Z"),
                    "source_published_date": dt.date(2024, 9, 30),
                    "review_group_key": "cap",
                    "source_revision_rank": 1,
                    "review_note": pd.NA,
                    "target_is_proxy": False,
                },
            ]
        )
        policy = build_fact_gb_transfer_review_policy(reviewed_period)

        hourly = build_fact_gb_transfer_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            reviewed_period=reviewed_period,
            review_policy=policy,
        )

        row = hourly.iloc[0]
        self.assertEqual(row["source_key"], "reviewed_transfer_cap_window")
        self.assertAlmostEqual(float(row["reviewed_capacity_limit_mw"]), 120.0)
        self.assertEqual(row["reviewed_gate_state"], "reviewed_pass_restricted")

    def test_materialize_gb_transfer_reviewed_history_writes_csvs(self) -> None:
        reviewed_input = pd.DataFrame(
            [
                {
                    "cluster_key": "east_anglia_offshore",
                    "hub_key": "eleclink",
                    "source_key": "public_constraint_period",
                    "period_start_utc": "2024-10-01T00:00:00Z",
                    "period_end_utc": "2024-10-01T06:00:00Z",
                    "capacity_limit_mw": 200.0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_gb_transfer_reviewed_history(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                output_dir=tmp_dir,
                reviewed_input=reviewed_input,
            )
            self.assertIn(GB_TRANSFER_REVIEWED_PERIOD_TABLE, frames)
            self.assertIn(GB_TRANSFER_REVIEWED_HOURLY_TABLE, frames)
            self.assertTrue((Path(tmp_dir) / f"{GB_TRANSFER_REVIEWED_PERIOD_TABLE}.csv").exists())
            self.assertTrue((Path(tmp_dir) / f"{GB_TRANSFER_REVIEWED_HOURLY_TABLE}.csv").exists())


if __name__ == "__main__":
    unittest.main()
