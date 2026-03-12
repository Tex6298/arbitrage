import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from interconnector_capacity import (
    INTERCONNECTOR_CAPACITY_TABLE,
    INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE,
    INTERCONNECTOR_CAPACITY_REVIEW_POLICY_TABLE,
    INTERCONNECTOR_CAPACITY_REVIEWED_TABLE,
    INTERCONNECTOR_CAPACITY_AUDIT_VARIANT_TABLE,
    BORDER_FLOW_SPECS,
    build_fact_interconnector_capacity_hourly,
    build_interconnector_capacity_review_policy,
    build_interconnector_capacity_reviewed_hourly,
    build_interconnector_capacity_source_audit,
    materialize_interconnector_capacity_history,
    parse_entsoe_interconnector_capacity_xml,
)


def _sample_capacity_xml(
    out_domain_eic: str,
    in_domain_eic: str,
    first_quantity: float,
    second_quantity: float,
) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <businessType>A31</businessType>
    <auction.type>A01</auction.type>
    <contract_MarketAgreement.type>A01</contract_MarketAgreement.type>
    <outBiddingZone_Domain.mRID codingScheme="A01">{out_domain_eic}</outBiddingZone_Domain.mRID>
    <inBiddingZone_Domain.mRID codingScheme="A01">{in_domain_eic}</inBiddingZone_Domain.mRID>
    <Period>
      <timeInterval>
        <start>2024-09-30T23:00Z</start>
        <end>2024-10-01T01:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point>
        <position>1</position>
        <quantity>{first_quantity}</quantity>
      </Point>
      <Point>
        <position>2</position>
        <quantity>{second_quantity}</quantity>
      </Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
""".encode("utf-8")


def _sample_stepwise_capacity_xml(out_domain_eic: str, in_domain_eic: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <curveType>A03</curveType>
    <auction.type>A01</auction.type>
    <contract_MarketAgreement.type>A01</contract_MarketAgreement.type>
    <outBiddingZone_Domain.mRID codingScheme="A01">{out_domain_eic}</outBiddingZone_Domain.mRID>
    <inBiddingZone_Domain.mRID codingScheme="A01">{in_domain_eic}</inBiddingZone_Domain.mRID>
    <Period>
      <timeInterval>
        <start>2024-09-30T23:00Z</start>
        <end>2024-10-01T02:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>950</quantity></Point>
      <Point><position>3</position><quantity>0</quantity></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
""".encode("utf-8")


class InterconnectorCapacityTests(unittest.TestCase):
    def test_parse_entsoe_interconnector_capacity_xml_sets_capacity_and_metadata(self) -> None:
        spec = next(flow_spec for flow_spec in BORDER_FLOW_SPECS if flow_spec.border_key == "GB-FR")
        start_utc = dt.datetime(2024, 9, 30, 23, 0, tzinfo=dt.timezone.utc)
        end_utc = dt.datetime(2024, 10, 1, 23, 0, tzinfo=dt.timezone.utc)
        xml_bytes = _sample_capacity_xml(
            out_domain_eic="10YGB----------A",
            in_domain_eic="10YFR-RTE------C",
            first_quantity=1000.0,
            second_quantity=950.0,
        )

        fact = parse_entsoe_interconnector_capacity_xml(
            xml_bytes,
            spec=spec,
            direction_key="gb_to_neighbor",
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
        )

        self.assertEqual(len(fact), 2)
        self.assertEqual(fact.iloc[0]["border_key"], "GB-FR")
        self.assertEqual(fact.iloc[0]["direction_key"], "gb_to_neighbor")
        self.assertEqual(fact.iloc[0]["source_article"], "11.1.A")
        self.assertEqual(fact.iloc[0]["auction_type"], "A01")
        self.assertEqual(fact.iloc[0]["contract_market_agreement_type"], "A01")
        self.assertAlmostEqual(float(fact.iloc[0]["offered_capacity_mw"]), 1000.0)
        self.assertEqual(fact.iloc[0]["candidate_hub_keys"], "ifa,ifa2,eleclink")

    def test_parse_entsoe_interconnector_capacity_xml_expands_stepwise_constant_curve(self) -> None:
        spec = next(flow_spec for flow_spec in BORDER_FLOW_SPECS if flow_spec.border_key == "GB-NL")
        start_utc = dt.datetime(2024, 9, 30, 23, 0, tzinfo=dt.timezone.utc)
        end_utc = dt.datetime(2024, 10, 1, 23, 0, tzinfo=dt.timezone.utc)
        xml_bytes = _sample_stepwise_capacity_xml(
            out_domain_eic="10YNL----------L",
            in_domain_eic="10YGB----------A",
        )

        fact = parse_entsoe_interconnector_capacity_xml(
            xml_bytes,
            spec=spec,
            direction_key="neighbor_to_gb",
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
        )

        self.assertEqual(len(fact), 3)
        self.assertEqual(fact["offered_capacity_mw"].tolist(), [950.0, 950.0, 0.0])

    def test_materialize_interconnector_capacity_history_writes_csv_and_concatenates_directions(self) -> None:
        start_date = dt.date(2024, 10, 1)
        end_date = dt.date(2024, 10, 1)

        def fake_fetch(spec, direction_key, start_date, end_date, token):  # type: ignore[no-redef]
            if spec.border_key != "GB-BE":
                return pd.DataFrame(columns=["interval_start_utc"])
            return pd.DataFrame(
                [
                    {
                        "date": start_date,
                        "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                        "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                        "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                        "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                        "source_key": "entsoe_offered_capacity",
                        "source_label": "ENTSO-E offered capacity",
                        "source_document_type": "A31",
                        "source_article": "11.1.A",
                        "source_resolution": "PT60M",
                        "normalized_resolution": "PT60M",
                        "target_is_proxy": False,
                        "capacity_scope": "aggregate_border_bidding_zone",
                        "hub_assignment_mode": "single_hub_border_proxy",
                        "hub_assignment_is_proxy": True,
                        "border_key": spec.border_key,
                        "border_label": spec.border_label,
                        "target_zone": spec.target_zone,
                        "neighbor_domain_key": spec.neighbor_domain_key,
                        "gb_domain_eic": "10YGB----------A",
                        "neighbor_domain_eic": spec.neighbor_domain_eic,
                        "out_domain_eic": "10YGB----------A" if direction_key == "gb_to_neighbor" else spec.neighbor_domain_eic,
                        "in_domain_eic": spec.neighbor_domain_eic if direction_key == "gb_to_neighbor" else "10YGB----------A",
                        "direction_key": direction_key,
                        "direction_label": "test",
                        "candidate_hub_keys": "nemo",
                        "candidate_hub_labels": "Nemo",
                        "candidate_hub_count": 1,
                        "auction_type": "A01",
                        "contract_market_agreement_type": "A01",
                        "business_type": "A31",
                        "offered_capacity_mw": 1000.0,
                        "offered_capacity_mwh": 1000.0,
                    }
                ]
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("interconnector_capacity.fetch_interconnector_capacity_direction", side_effect=fake_fetch):
                frames = materialize_interconnector_capacity_history(
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=tmp_dir,
                    token="test-token",
                )

            self.assertIn(INTERCONNECTOR_CAPACITY_TABLE, frames)
            fact = frames[INTERCONNECTOR_CAPACITY_TABLE]
            self.assertEqual(len(fact), 2)
            self.assertEqual(set(fact["direction_key"]), {"gb_to_neighbor", "neighbor_to_gb"})
            output_path = Path(tmp_dir) / f"{INTERCONNECTOR_CAPACITY_TABLE}.csv"
            self.assertTrue(output_path.exists())

    def test_build_fact_interconnector_capacity_hourly_requires_token(self) -> None:
        with self.assertRaises(RuntimeError):
            build_fact_interconnector_capacity_hourly(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                token="",
            )

    def test_build_interconnector_capacity_source_audit_flags_unknown_borders(self) -> None:
        def fake_fetch(spec, direction_key, start_date, end_date, token, document_type="A31", auction_type="A01", contract_market_agreement_type="A01"):  # type: ignore[no-redef]
            if spec.border_key == "GB-NO2" and contract_market_agreement_type == "A01":
                return pd.DataFrame(
                    [
                        {
                            "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                            "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                            "source_resolution": "PT60M",
                            "out_domain_eic": "10YGB----------A",
                            "in_domain_eic": "10YNO-2--------T",
                            "auction_type": auction_type,
                            "contract_market_agreement_type": contract_market_agreement_type,
                            "business_type": document_type,
                            "offered_capacity_mw": 1449.0,
                            "source_key": "entsoe_offered_capacity",
                            "source_label": "ENTSO-E offered capacity",
                            "source_document_type": document_type,
                            "source_article": "11.1.A",
                            "target_is_proxy": False,
                            "capacity_scope": "aggregate_border_bidding_zone",
                            "hub_assignment_mode": "single_hub_border_proxy",
                            "hub_assignment_is_proxy": True,
                            "border_key": spec.border_key,
                            "border_label": spec.border_label,
                            "target_zone": spec.target_zone,
                            "neighbor_domain_key": spec.neighbor_domain_key,
                            "gb_domain_eic": "10YGB----------A",
                            "neighbor_domain_eic": spec.neighbor_domain_eic,
                            "direction_key": direction_key,
                            "direction_label": "test",
                            "candidate_hub_keys": "nsl",
                            "candidate_hub_labels": "North Sea Link",
                            "candidate_hub_count": 1,
                            "date": dt.date(2024, 10, 1),
                            "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                            "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                            "normalized_resolution": "PT60M",
                            "offered_capacity_mwh": 1449.0,
                        }
                    ]
                )
            return pd.DataFrame(columns=["interval_start_utc"])

        with patch("interconnector_capacity.fetch_interconnector_capacity_direction", side_effect=fake_fetch):
            frames = build_interconnector_capacity_source_audit(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                token="test-token",
            )

        daily = frames[INTERCONNECTOR_CAPACITY_AUDIT_DAILY_TABLE]
        variant = frames[INTERCONNECTOR_CAPACITY_AUDIT_VARIANT_TABLE]
        no2_daily = daily[(daily["border_key"] == "GB-NO2") & (daily["direction_key"] == "gb_to_neighbor")].iloc[0]
        fr_daily = daily[(daily["border_key"] == "GB-FR") & (daily["direction_key"] == "gb_to_neighbor")].iloc[0]
        self.assertEqual(no2_daily["audit_status"], "first_pass_published")
        self.assertEqual(no2_daily["recommended_gate_policy"], "eligible_first_pass_gate")
        self.assertEqual(fr_daily["audit_status"], "no_variant_published")
        self.assertEqual(fr_daily["recommended_gate_policy"], "capacity_unknown_default")
        self.assertEqual(
            int(
                variant[
                    (variant["border_key"] == "GB-NO2")
                    & (variant["direction_key"] == "gb_to_neighbor")
                    & (variant["variant_key"] == "a31_implicit_daily")
                ]["rows_returned"].iloc[0]
            ),
            1,
        )

    def test_build_interconnector_capacity_review_policy_accepts_explicit_daily_reviewed_tier(self) -> None:
        audit_daily = pd.DataFrame(
            [
                {
                    "border_key": "GB-NL",
                    "border_label": "Great Britain to Netherlands aggregate border",
                    "target_zone": "NL",
                    "neighbor_domain_key": "NL",
                    "direction_key": "gb_to_neighbor",
                    "direction_label": "Great Britain to NL",
                    "first_pass_rows_returned": 0,
                    "alternate_variant_rows_returned": 24,
                    "first_published_variant_key": "a31_explicit_daily",
                    "published_variant_keys": "a31_explicit_daily",
                    "audit_status": "alternate_variant_published",
                    "recommended_gate_policy": "audit_before_gate",
                    "audit_note": "Only alternate variants published.",
                },
                {
                    "border_key": "GB-NO2",
                    "border_label": "Great Britain to Norway bidding zone NO2 aggregate border",
                    "target_zone": "NO",
                    "neighbor_domain_key": "NO2",
                    "direction_key": "gb_to_neighbor",
                    "direction_label": "Great Britain to NO2",
                    "first_pass_rows_returned": 24,
                    "alternate_variant_rows_returned": 0,
                    "first_published_variant_key": "a31_implicit_daily",
                    "published_variant_keys": "a31_implicit_daily",
                    "audit_status": "first_pass_published",
                    "recommended_gate_policy": "eligible_first_pass_gate",
                    "audit_note": "First pass published.",
                },
                {
                    "border_key": "GB-FR",
                    "border_label": "Great Britain to France aggregate border",
                    "target_zone": "FR",
                    "neighbor_domain_key": "FR",
                    "direction_key": "gb_to_neighbor",
                    "direction_label": "Great Britain to FR",
                    "first_pass_rows_returned": 0,
                    "alternate_variant_rows_returned": 0,
                    "first_published_variant_key": None,
                    "published_variant_keys": "",
                    "audit_status": "query_error_or_unpublished",
                    "recommended_gate_policy": "audit_before_gate",
                    "audit_note": "No rows found.",
                },
            ]
        )

        review = build_interconnector_capacity_review_policy(audit_daily)

        self.assertEqual(set(review["border_key"]), {"GB-NL", "GB-NO2", "GB-FR"})
        nl_row = review[review["border_key"] == "GB-NL"].iloc[0]
        no2_row = review[review["border_key"] == "GB-NO2"].iloc[0]
        fr_row = review[review["border_key"] == "GB-FR"].iloc[0]

        self.assertEqual(nl_row["review_state"], "accepted_reviewed_tier")
        self.assertEqual(nl_row["accepted_variant_key"], "a31_explicit_daily")
        self.assertEqual(nl_row["capacity_policy_action"], "allow_reviewed_explicit_daily")
        self.assertTrue(bool(nl_row["reviewed_tier_accepted_flag"]))

        self.assertEqual(no2_row["review_state"], "first_pass_direct")
        self.assertEqual(no2_row["capacity_policy_action"], "eligible_first_pass_gate")

        self.assertEqual(fr_row["review_state"], "capacity_unknown_default")
        self.assertEqual(fr_row["capacity_policy_action"], "keep_capacity_unknown")

    def test_build_interconnector_capacity_reviewed_hourly_fetches_only_accepted_reviewed_borders(self) -> None:
        review_policy = pd.DataFrame(
            [
                {
                    "border_key": "GB-NL",
                    "border_label": "Great Britain to Netherlands aggregate border",
                    "target_zone": "NL",
                    "neighbor_domain_key": "NL",
                    "direction_key": "gb_to_neighbor",
                    "direction_label": "Great Britain to NL",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_explicit_daily",
                    "accepted_variant_key": "a31_explicit_daily",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_explicit_daily",
                    "review_note": "Accepted reviewed tier.",
                },
                {
                    "border_key": "GB-FR",
                    "border_label": "Great Britain to France aggregate border",
                    "target_zone": "FR",
                    "neighbor_domain_key": "FR",
                    "direction_key": "gb_to_neighbor",
                    "direction_label": "Great Britain to FR",
                    "review_state": "capacity_unknown_default",
                    "reviewed_evidence_tier": "none",
                    "accepted_variant_key": None,
                    "reviewed_tier_accepted_flag": False,
                    "capacity_policy_action": "keep_capacity_unknown",
                    "review_note": "No reviewed tier.",
                },
            ]
        )

        def fake_fetch(spec, direction_key, start_date, end_date, token, document_type="A31", auction_type="A01", contract_market_agreement_type="A01"):  # type: ignore[no-redef]
            self.assertEqual(spec.border_key, "GB-NL")
            self.assertEqual(direction_key, "gb_to_neighbor")
            self.assertEqual(auction_type, "A02")
            self.assertEqual(contract_market_agreement_type, "A01")
            return pd.DataFrame(
                [
                    {
                        "date": start_date,
                        "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                        "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                        "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                        "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                        "source_key": "entsoe_offered_capacity",
                        "source_label": "ENTSO-E offered capacity",
                        "source_document_type": "A31",
                        "source_article": "11.1.A",
                        "source_resolution": "PT60M",
                        "normalized_resolution": "PT60M",
                        "target_is_proxy": False,
                        "capacity_scope": "aggregate_border_bidding_zone",
                        "hub_assignment_mode": "single_hub_border_proxy",
                        "hub_assignment_is_proxy": True,
                        "border_key": spec.border_key,
                        "border_label": spec.border_label,
                        "target_zone": spec.target_zone,
                        "neighbor_domain_key": spec.neighbor_domain_key,
                        "gb_domain_eic": "10YGB----------A",
                        "neighbor_domain_eic": spec.neighbor_domain_eic,
                        "out_domain_eic": "10YGB----------A",
                        "in_domain_eic": spec.neighbor_domain_eic,
                        "direction_key": direction_key,
                        "direction_label": "test",
                        "candidate_hub_keys": "britned",
                        "candidate_hub_labels": "BritNed",
                        "candidate_hub_count": 1,
                        "auction_type": "A02",
                        "contract_market_agreement_type": "A01",
                        "business_type": "A31",
                        "offered_capacity_mw": 700.0,
                        "offered_capacity_mwh": 700.0,
                    }
                ]
            )

        with patch("interconnector_capacity.fetch_interconnector_capacity_direction", side_effect=fake_fetch):
            reviewed = build_interconnector_capacity_reviewed_hourly(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                token="test-token",
                review_policy=review_policy,
            )

        self.assertEqual(len(reviewed), 1)
        self.assertEqual(reviewed.iloc[0]["border_key"], "GB-NL")
        self.assertEqual(reviewed.iloc[0]["source_key"], "entsoe_offered_capacity_reviewed")
        self.assertEqual(reviewed.iloc[0]["accepted_variant_key"], "a31_explicit_daily")
        self.assertEqual(reviewed.iloc[0]["capacity_policy_action"], "allow_reviewed_explicit_daily")


if __name__ == "__main__":
    unittest.main()
