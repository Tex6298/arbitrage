import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from interconnector_flow import (
    BORDER_FLOW_SPECS,
    INTERCONNECTOR_FLOW_TABLE,
    build_fact_interconnector_flow_hourly,
    materialize_interconnector_flow_history,
    parse_entsoe_interconnector_flow_xml,
)


def _sample_flow_xml(out_domain_eic: str, in_domain_eic: str, first_quantity: float, second_quantity: float) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
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
</GL_MarketDocument>
""".encode("utf-8")


def _sample_quarter_hour_flow_xml(out_domain_eic: str, in_domain_eic: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <outBiddingZone_Domain.mRID codingScheme="A01">{out_domain_eic}</outBiddingZone_Domain.mRID>
    <inBiddingZone_Domain.mRID codingScheme="A01">{in_domain_eic}</inBiddingZone_Domain.mRID>
    <Period>
      <timeInterval>
        <start>2024-09-30T23:00Z</start>
        <end>2024-10-01T00:00Z</end>
      </timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>100</quantity></Point>
      <Point><position>2</position><quantity>200</quantity></Point>
      <Point><position>3</position><quantity>300</quantity></Point>
      <Point><position>4</position><quantity>400</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>
""".encode("utf-8")


def _sample_stepwise_constant_flow_xml(out_domain_eic: str, in_domain_eic: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <curveType>A03</curveType>
    <outBiddingZone_Domain.mRID codingScheme="A01">{out_domain_eic}</outBiddingZone_Domain.mRID>
    <inBiddingZone_Domain.mRID codingScheme="A01">{in_domain_eic}</inBiddingZone_Domain.mRID>
    <Period>
      <timeInterval>
        <start>2024-09-30T23:00Z</start>
        <end>2024-10-01T02:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>0</quantity></Point>
      <Point><position>3</position><quantity>500</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>
""".encode("utf-8")


class InterconnectorFlowTests(unittest.TestCase):
    def test_parse_entsoe_interconnector_flow_xml_sets_border_and_signed_direction(self) -> None:
        spec = next(flow_spec for flow_spec in BORDER_FLOW_SPECS if flow_spec.border_key == "GB-FR")
        start_utc = dt.datetime(2024, 9, 30, 23, 0, tzinfo=dt.timezone.utc)
        end_utc = dt.datetime(2024, 10, 1, 23, 0, tzinfo=dt.timezone.utc)
        xml_bytes = _sample_flow_xml(
            out_domain_eic="10YGB----------A",
            in_domain_eic="10YFR-RTE------C",
            first_quantity=750.0,
            second_quantity=700.0,
        )

        fact = parse_entsoe_interconnector_flow_xml(
            xml_bytes,
            spec=spec,
            direction_key="gb_to_neighbor",
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
        )

        self.assertEqual(len(fact), 2)
        self.assertEqual(fact.iloc[0]["border_key"], "GB-FR")
        self.assertEqual(fact.iloc[0]["target_zone"], "FR")
        self.assertEqual(fact.iloc[0]["direction_key"], "gb_to_neighbor")
        self.assertAlmostEqual(float(fact.iloc[0]["flow_mw"]), 750.0)
        self.assertAlmostEqual(float(fact.iloc[0]["signed_flow_from_gb_mw"]), 750.0)
        self.assertEqual(fact.iloc[0]["candidate_hub_keys"], "ifa,ifa2,eleclink")
        self.assertTrue(bool(fact.iloc[0]["hub_assignment_is_proxy"]))
        self.assertEqual(str(fact.iloc[0]["date"]), "2024-10-01")

    def test_parse_entsoe_interconnector_flow_xml_treats_no_matching_data_as_empty(self) -> None:
        spec = next(flow_spec for flow_spec in BORDER_FLOW_SPECS if flow_spec.border_key == "GB-DK1")
        start_utc = dt.datetime(2024, 9, 30, 23, 0, tzinfo=dt.timezone.utc)
        end_utc = dt.datetime(2024, 10, 1, 23, 0, tzinfo=dt.timezone.utc)
        xml_bytes = b"""<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <Reason>
    <code>999</code>
    <text>No matching data found</text>
  </Reason>
</Acknowledgement_MarketDocument>
"""

        fact = parse_entsoe_interconnector_flow_xml(
            xml_bytes,
            spec=spec,
            direction_key="gb_to_neighbor",
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
        )

        self.assertTrue(fact.empty)
        self.assertIn("flow_mw", fact.columns)

    def test_parse_entsoe_interconnector_flow_xml_normalizes_quarter_hour_to_hourly_average(self) -> None:
        spec = next(flow_spec for flow_spec in BORDER_FLOW_SPECS if flow_spec.border_key == "GB-NL")
        start_utc = dt.datetime(2024, 9, 30, 23, 0, tzinfo=dt.timezone.utc)
        end_utc = dt.datetime(2024, 10, 1, 23, 0, tzinfo=dt.timezone.utc)
        xml_bytes = _sample_quarter_hour_flow_xml(
            out_domain_eic="10YGB----------A",
            in_domain_eic="10YNL----------L",
        )

        fact = parse_entsoe_interconnector_flow_xml(
            xml_bytes,
            spec=spec,
            direction_key="gb_to_neighbor",
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
        )

        self.assertEqual(len(fact), 1)
        self.assertEqual(fact.iloc[0]["normalized_resolution"], "PT60M")
        self.assertEqual(fact.iloc[0]["source_resolution"], "PT15M")
        self.assertAlmostEqual(float(fact.iloc[0]["flow_mw"]), 250.0)
        self.assertAlmostEqual(float(fact.iloc[0]["flow_mwh"]), 250.0)

    def test_parse_entsoe_interconnector_flow_xml_expands_stepwise_constant_curve(self) -> None:
        spec = next(flow_spec for flow_spec in BORDER_FLOW_SPECS if flow_spec.border_key == "GB-IE")
        start_utc = dt.datetime(2024, 9, 30, 23, 0, tzinfo=dt.timezone.utc)
        end_utc = dt.datetime(2024, 10, 1, 23, 0, tzinfo=dt.timezone.utc)
        xml_bytes = _sample_stepwise_constant_flow_xml(
            out_domain_eic="10Y1001A1001A59C",
            in_domain_eic="10YGB----------A",
        )

        fact = parse_entsoe_interconnector_flow_xml(
            xml_bytes,
            spec=spec,
            direction_key="neighbor_to_gb",
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
        )

        self.assertEqual(len(fact), 3)
        self.assertEqual(fact["flow_mw"].tolist(), [0.0, 0.0, 500.0])
        self.assertEqual(fact["signed_flow_from_gb_mw"].tolist(), [0.0, 0.0, -500.0])

    def test_materialize_interconnector_flow_history_writes_csv_and_concatenates_directions(self) -> None:
        start_date = dt.date(2024, 10, 1)
        end_date = dt.date(2024, 10, 1)

        def fake_fetch(spec, direction_key, start_date, end_date, token):  # type: ignore[no-redef]
            if spec.border_key != "GB-FR":
                return pd.DataFrame(columns=["interval_start_utc"])
            sign = 1.0 if direction_key == "gb_to_neighbor" else -1.0
            return pd.DataFrame(
                [
                    {
                        "date": start_date,
                        "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                        "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                        "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                        "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                        "source_key": "entsoe_actual_physical_flow",
                        "source_label": "ENTSO-E actual physical flow",
                        "source_document_type": "A11",
                        "source_resolution": "PT60M",
                        "normalized_resolution": "PT60M",
                        "target_is_proxy": False,
                        "flow_scope": "aggregate_border_bidding_zone",
                        "hub_assignment_mode": "aggregate_border_candidate_hubs",
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
                        "candidate_hub_keys": "ifa,ifa2,eleclink",
                        "candidate_hub_labels": "IFA,IFA2,ElecLink",
                        "candidate_hub_count": 3,
                        "flow_mw": 500.0,
                        "flow_mwh": 500.0,
                        "signed_flow_from_gb_mw": sign * 500.0,
                        "signed_flow_from_gb_mwh": sign * 500.0,
                    }
                ]
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("interconnector_flow.fetch_interconnector_flow_direction", side_effect=fake_fetch):
                frames = materialize_interconnector_flow_history(
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=tmp_dir,
                    token="test-token",
                )

            self.assertIn(INTERCONNECTOR_FLOW_TABLE, frames)
            fact = frames[INTERCONNECTOR_FLOW_TABLE]
            self.assertEqual(len(fact), 2)
            self.assertEqual(set(fact["direction_key"]), {"gb_to_neighbor", "neighbor_to_gb"})
            output_path = Path(tmp_dir) / f"{INTERCONNECTOR_FLOW_TABLE}.csv"
            self.assertTrue(output_path.exists())

    def test_build_fact_interconnector_flow_hourly_requires_token(self) -> None:
        with self.assertRaises(RuntimeError):
            build_fact_interconnector_flow_hourly(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                token="",
            )


if __name__ == "__main__":
    unittest.main()
