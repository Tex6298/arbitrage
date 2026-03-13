import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from france_connector_reviewed import (
    FRANCE_CONNECTOR_NOTICE_TABLE,
    FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE,
    build_fact_france_connector_notice_hourly,
    build_fact_france_connector_reviewed_hourly,
    build_fact_france_connector_reviewed_period,
    materialize_france_connector_reviewed_period,
    write_normalized_france_connector_reviewed_input,
)


class FranceConnectorReviewedTests(unittest.TestCase):
    def test_build_fact_france_connector_reviewed_period_normalizes_public_doc_rows(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "connector_key": "eleclink",
                    "source_key": "eleclink_ntc_restriction",
                    "start_date": "2024-10-01",
                    "end_date": "2024-10-01",
                    "period_timezone": "UTC",
                    "capacity_limit_mw": 200.0,
                    "source_document_title": "ElecLink restriction",
                },
                {
                    "connector_key": "ifa2",
                    "source_key": "jao_ifa2_notice",
                    "period_start_utc": "2024-10-01T00:00:00Z",
                    "period_end_utc": "2024-10-01T12:00:00Z",
                    "reviewed_publication_state": "outage",
                    "source_document_title": "IFA2 JAO notice",
                },
            ]
        )

        fact = build_fact_france_connector_reviewed_period(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 2),
            reviewed_input=source,
        )

        self.assertEqual(set(fact["connector_key"]), {"eleclink", "ifa2"})
        eleclink = fact[fact["connector_key"] == "eleclink"].iloc[0]
        self.assertEqual(eleclink["reviewed_publication_state"], "partial_capacity")
        self.assertEqual(eleclink["capacity_policy_action"], "allow_reviewed_public_period")
        self.assertEqual(eleclink["source_document_url"], "https://www.eleclink.co.uk/publications/ntc-restrictions")
        self.assertEqual(eleclink["period_end_utc"], pd.Timestamp("2024-10-02T00:00:00Z"))

        ifa2 = fact[fact["connector_key"] == "ifa2"].iloc[0]
        self.assertEqual(ifa2["reviewed_publication_state"], "outage")
        self.assertAlmostEqual(float(ifa2["reviewed_capacity_limit_mw"]), 0.0)

    def test_build_fact_france_connector_reviewed_hourly_expands_overlapping_periods(self) -> None:
        reviewed_period = pd.DataFrame(
            [
                {
                    "connector_key": "ifa",
                    "connector_label": "IFA",
                    "direction_key": "gb_to_neighbor",
                    "reviewed_scope": "france_connector_public_doc_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_public_doc_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_public_period",
                    "reviewed_publication_state": "partial_capacity",
                    "period_start_utc": pd.Timestamp("2024-09-30T23:30:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T01:15:00Z"),
                    "period_timezone": "UTC",
                    "connector_nominal_capacity_mw": 2000.0,
                    "reviewed_capacity_limit_mw": 1200.0,
                    "reviewed_available_capacity_mw": 1200.0,
                    "reviewed_unavailable_capacity_mw": 800.0,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "jao_public_notice",
                    "source_key": "jao_ifa_notice",
                    "source_label": "JAO IFA notice",
                    "source_document_title": "IFA notice",
                    "source_document_url": "https://www.jao.eu/news",
                    "source_reference": "JAO-IFA-1",
                    "source_published_date": dt.date(2024, 9, 30),
                    "review_note": "Notice.",
                    "target_is_proxy": False,
                }
            ]
        )

        hourly = build_fact_france_connector_reviewed_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            reviewed_period=reviewed_period,
        )

        self.assertEqual(len(hourly), 3)
        self.assertTrue((hourly["connector_key"] == "ifa").all())
        self.assertTrue((hourly["reviewed_publication_state"] == "partial_capacity").all())

    def test_materialize_france_connector_reviewed_period_writes_csv(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "connector_key": "ifa2",
                    "source_key": "jao_ifa2_notice",
                    "period_start_utc": "2024-10-01T00:00:00Z",
                    "period_end_utc": "2024-10-01T06:00:00Z",
                    "capacity_limit_mw": 500.0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_france_connector_reviewed_period(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                output_dir=tmp_dir,
                reviewed_input=source,
            )

            self.assertEqual(set(frames), {FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE, FRANCE_CONNECTOR_NOTICE_TABLE})
            self.assertTrue((Path(tmp_dir) / f"{FRANCE_CONNECTOR_REVIEWED_PERIOD_TABLE}.csv").exists())
            self.assertTrue((Path(tmp_dir) / f"{FRANCE_CONNECTOR_NOTICE_TABLE}.csv").exists())

    def test_write_normalized_france_connector_reviewed_input_parses_txt_delivery_period_rows(self) -> None:
        raw_text = "\n".join(
            [
                "Connector Key\tSource Key\tDelivery Date\tDelivery period (GMT)\tCapacity limit MW\tSource Document Title",
                "eleclink\teleclink_ntc_restriction\t2024-10-01\t00:00 - 00:15\t250\tElecLink restriction",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "france_reviewed_raw.txt"
            out_path = Path(tmp_dir) / "france_reviewed_normalized.csv"
            raw_path.write_text(raw_text, encoding="utf-8")

            normalized = write_normalized_france_connector_reviewed_input(raw_path, out_path)

            self.assertTrue(out_path.exists())
            self.assertEqual(len(normalized), 1)
            row = normalized.iloc[0]
            self.assertEqual(row["connector_key"], "eleclink")
            self.assertEqual(row["period_start_utc"], pd.Timestamp("2024-10-01T00:00:00Z"))
            self.assertEqual(row["period_end_utc"], pd.Timestamp("2024-10-01T00:15:00Z"))

    def test_build_fact_france_connector_notice_hourly_tracks_upcoming_notice_and_revisions(self) -> None:
        reviewed_period = pd.DataFrame(
            [
                {
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "direction_key": "gb_to_neighbor",
                    "reviewed_scope": "france_connector_public_doc_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_public_doc_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_public_period",
                    "reviewed_publication_state": "partial_capacity",
                    "period_start_utc": pd.Timestamp("2024-10-01T12:00:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T18:00:00Z"),
                    "period_timezone": "UTC",
                    "connector_nominal_capacity_mw": 1000.0,
                    "reviewed_capacity_limit_mw": 400.0,
                    "reviewed_available_capacity_mw": 400.0,
                    "reviewed_unavailable_capacity_mw": 600.0,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "eleclink_public_doc",
                    "source_key": "eleclink_ntc_restriction",
                    "source_label": "ElecLink NTC restriction statement",
                    "source_document_title": "ElecLink restriction v1",
                    "source_document_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
                    "source_reference": "EL-NTC-1",
                    "source_published_utc": pd.Timestamp("2024-09-30T10:00:00Z"),
                    "source_published_date": dt.date(2024, 9, 30),
                    "notice_group_key": "eleclink|gb_to_neighbor|2024-10-01T12:00:00Z|2024-10-01T18:00:00Z",
                    "notice_planning_state": "operational_restriction",
                    "planned_outage_flag": False,
                    "source_revision_rank": 1,
                    "review_note": "Initial restriction.",
                    "target_is_proxy": False,
                },
                {
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "direction_key": "gb_to_neighbor",
                    "reviewed_scope": "france_connector_public_doc_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_public_doc_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_public_period",
                    "reviewed_publication_state": "partial_capacity",
                    "period_start_utc": pd.Timestamp("2024-10-01T12:00:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T18:00:00Z"),
                    "period_timezone": "UTC",
                    "connector_nominal_capacity_mw": 1000.0,
                    "reviewed_capacity_limit_mw": 250.0,
                    "reviewed_available_capacity_mw": 250.0,
                    "reviewed_unavailable_capacity_mw": 750.0,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "eleclink_public_doc",
                    "source_key": "eleclink_ntc_restriction",
                    "source_label": "ElecLink NTC restriction statement",
                    "source_document_title": "ElecLink restriction v2",
                    "source_document_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
                    "source_reference": "EL-NTC-2",
                    "source_published_utc": pd.Timestamp("2024-10-01T08:00:00Z"),
                    "source_published_date": dt.date(2024, 10, 1),
                    "notice_group_key": "eleclink|gb_to_neighbor|2024-10-01T12:00:00Z|2024-10-01T18:00:00Z",
                    "notice_planning_state": "operational_restriction",
                    "planned_outage_flag": False,
                    "source_revision_rank": 2,
                    "review_note": "Updated restriction.",
                    "target_is_proxy": False,
                },
            ]
        )

        notice = build_fact_france_connector_notice_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            reviewed_period=reviewed_period,
        )

        row = notice[
            (notice["connector_key"] == "eleclink")
            & (notice["direction_key"] == "gb_to_neighbor")
            & (notice["interval_start_utc"] == pd.Timestamp("2024-10-01T09:00:00Z"))
        ].iloc[0]
        self.assertEqual(row["notice_state"], "upcoming")
        self.assertTrue(bool(row["notice_known_flag"]))
        self.assertAlmostEqual(float(row["hours_until_notice_start"]), 3.0)
        self.assertAlmostEqual(float(row["expected_capacity_limit_mw"]), 250.0)
        self.assertEqual(int(row["notice_revision_count"]), 2)
        self.assertEqual(int(row["source_revision_rank"]), 2)


if __name__ == "__main__":
    unittest.main()
