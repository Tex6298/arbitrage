import datetime as dt
import unittest

import pandas as pd

from france_connector_availability import (
    build_eleclink_operator_source_compare,
    build_fact_france_connector_availability_hourly,
    build_france_connector_operator_event_frame,
    fetch_eleclink_umm_authenticated,
    load_eleclink_umm_export,
)


class FranceConnectorAvailabilityTests(unittest.TestCase):
    def test_build_france_connector_operator_event_frame_matches_ifa2_remit_rows(self) -> None:
        raw_remit = pd.DataFrame(
            [
                {
                    "affectedUnit": "I_IED-IFA2",
                    "assetId": "I_IED-IFA2",
                    "registrationCode": "48X0000000002400",
                    "publishTime": "2024-10-01T00:10:00Z",
                    "eventStatus": "Active",
                    "eventStartTime": "2024-10-01T00:00:00Z",
                    "eventEndTime": "2024-10-01T02:00:00Z",
                    "messageType": "UnavailabilitiesOfElectricityFacilities",
                    "eventType": "Transmission unavailability",
                    "unavailabilityType": "Unplanned",
                    "normalCapacity": 1014.0,
                    "availableCapacity": 500.0,
                    "unavailableCapacity": 514.0,
                    "outageProfile": [],
                }
            ]
        )

        events = build_france_connector_operator_event_frame(raw_remit)

        self.assertEqual(len(events), 1)
        row = events.iloc[0]
        self.assertEqual(row["connector_key"], "ifa2")
        self.assertEqual(row["source_provider"], "elexon_remit")
        self.assertAlmostEqual(float(row["available_capacity_mw"]), 500.0)

    def test_build_fact_france_connector_availability_hourly_marks_partial_outage_and_eleclink_unknown(self) -> None:
        operator_events = pd.DataFrame(
            [
                {
                    "connector_key": "ifa2",
                    "connector_label": "IFA2",
                    "source_provider": "elexon_remit",
                    "source_key": "elexon_remit_connector",
                    "source_label": "Elexon REMIT transmission-unavailability messages",
                    "source_truth_tier": "operator_outage_truth",
                    "connector_match_rule": "remit_text_match",
                    "target_is_proxy": False,
                    "publish_time_utc": "2024-10-01T00:10:00Z",
                    "event_start_utc": "2024-09-30T23:00:00Z",
                    "event_end_utc": "2024-10-01T01:00:00Z",
                    "event_status": "Active",
                    "message_type": "UnavailabilitiesOfElectricityFacilities",
                    "event_type": "Transmission unavailability",
                    "unavailability_type": "Unplanned",
                    "affected_unit": "I_IED-IFA2",
                    "asset_id": "I_IED-IFA2",
                    "registration_code": "48X0000000002400",
                    "normal_capacity_mw": 1014.0,
                    "available_capacity_mw": 500.0,
                    "unavailable_capacity_mw": 514.0,
                }
            ]
        )
        remit_status = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "remit_fetch_ok": True,
                }
            ]
        )

        fact = build_fact_france_connector_availability_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            operator_event_frame=operator_events,
            remit_fetch_status_by_date=remit_status,
            eleclink_source_resolution={},
        )

        ifa2 = fact[
            (fact["connector_key"] == "ifa2")
            & (fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z"))
        ].iloc[0]
        eleclink = fact[
            (fact["connector_key"] == "eleclink")
            & (fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z"))
        ].iloc[0]

        self.assertEqual(ifa2["operator_availability_state"], "partial_outage")
        self.assertEqual(ifa2["operator_capacity_evidence_tier"], "operator_outage_truth")
        self.assertAlmostEqual(float(ifa2["operator_capacity_limit_mw"]), 500.0)

        self.assertEqual(eleclink["operator_availability_state"], "unknown_source")
        self.assertEqual(eleclink["operator_source_gap_reason"], "nordpool_umm_source_not_available")

    def test_build_eleclink_operator_source_compare_prefers_manual_export_for_historical_window(self) -> None:
        export_frame = load_eleclink_umm_export(None)
        export_frame = pd.DataFrame(
            [
                {
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "source_provider": "nordpool_umm",
                    "source_key": "nordpool_umm_export",
                    "source_label": "Nord Pool UMM export",
                    "source_truth_tier": "operator_outage_truth",
                    "connector_match_rule": "manual_eleclink_export",
                    "target_is_proxy": False,
                    "publish_time_utc": "2024-10-01T00:10:00Z",
                    "event_start_utc": "2024-09-30T23:00:00Z",
                    "event_end_utc": "2024-10-01T02:00:00Z",
                    "event_status": "Active",
                    "message_type": "Transmission unavailability",
                    "event_type": "Transmission unavailability",
                    "unavailability_type": "Planned",
                    "affected_unit": "ElecLink",
                    "asset_id": "ElecLink",
                    "registration_code": pd.NA,
                    "normal_capacity_mw": 1000.0,
                    "available_capacity_mw": 400.0,
                    "unavailable_capacity_mw": 600.0,
                }
            ]
        )
        auth_status = {
            "connector_key": "eleclink",
            "source_variant_key": "nordpool_umm_authenticated_api",
            "source_provider": "nordpool_umm",
            "source_key": "nordpool_umm_authenticated_api",
            "source_label": "Nord Pool UMM authenticated API",
            "source_attempted_flag": True,
            "source_fetch_ok": True,
            "source_gap_reason": pd.NA,
        }
        selected, compare, resolution = build_eleclink_operator_source_compare(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            authenticated_frame=pd.DataFrame(),
            authenticated_status=auth_status,
            export_frame=export_frame,
            export_attempted_flag=True,
        )

        self.assertEqual(resolution["source_key"], "nordpool_umm_export")
        self.assertEqual(resolution["selection_context"], "historical_replay")
        self.assertEqual(len(selected), 1)
        self.assertEqual(compare.loc[compare["source_selected_flag"], "source_variant_key"].iloc[0], "nordpool_umm_export")

    def test_build_eleclink_operator_source_compare_prefers_authenticated_for_current_window(self) -> None:
        auth_frame = pd.DataFrame(
            [
                {
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "source_provider": "nordpool_umm",
                    "source_key": "nordpool_umm_authenticated_api",
                    "source_label": "Nord Pool UMM authenticated API",
                    "source_truth_tier": "operator_outage_truth",
                    "connector_match_rule": "authenticated_api_match",
                    "target_is_proxy": False,
                    "publish_time_utc": "2026-03-12T00:10:00Z",
                    "event_start_utc": "2026-03-12T00:00:00Z",
                    "event_end_utc": "2026-03-12T02:00:00Z",
                    "event_status": "Active",
                    "message_type": "Transmission unavailability",
                    "event_type": "Transmission unavailability",
                    "unavailability_type": "Unplanned",
                    "affected_unit": "ElecLink",
                    "asset_id": "ElecLink",
                    "registration_code": pd.NA,
                    "normal_capacity_mw": 1000.0,
                    "available_capacity_mw": 300.0,
                    "unavailable_capacity_mw": 700.0,
                }
            ]
        )
        auth_status = {
            "connector_key": "eleclink",
            "source_variant_key": "nordpool_umm_authenticated_api",
            "source_provider": "nordpool_umm",
            "source_key": "nordpool_umm_authenticated_api",
            "source_label": "Nord Pool UMM authenticated API",
            "source_attempted_flag": True,
            "source_fetch_ok": True,
            "source_gap_reason": pd.NA,
        }

        today = dt.datetime.now(dt.timezone.utc).date()
        selected, compare, resolution = build_eleclink_operator_source_compare(
            start_date=today,
            end_date=today,
            authenticated_frame=auth_frame,
            authenticated_status=auth_status,
            export_frame=pd.DataFrame(),
            export_attempted_flag=False,
        )

        self.assertEqual(resolution["source_key"], "nordpool_umm_authenticated_api")
        self.assertEqual(resolution["selection_context"], "current_operational")
        self.assertEqual(len(selected), 1)
        self.assertEqual(compare.loc[compare["source_selected_flag"], "source_variant_key"].iloc[0], "nordpool_umm_authenticated_api")

    def test_fetch_eleclink_umm_authenticated_requires_credentials_or_token(self) -> None:
        frame, status = fetch_eleclink_umm_authenticated()

        self.assertTrue(frame.empty)
        self.assertFalse(status["source_fetch_ok"])
        self.assertEqual(status["source_gap_reason"], "nordpool_umm_credentials_not_provided")


if __name__ == "__main__":
    unittest.main()
