import datetime as dt
import unittest

import numpy as np
import pandas as pd

from bmu_availability import build_fact_bmu_availability_half_hourly
from bmu_dispatch import build_fact_bmu_bid_offer_half_hourly, build_fact_bmu_dispatch_acceptance_half_hourly
from bmu_physical import build_fact_bmu_physical_position_half_hourly
from bmu_truth_utils import build_half_hour_interval_frame
from curtailment_signals import CONSTRAINT_QA_TARGET_DEFINITION, add_constraint_qa_columns
from curtailment_truth import (
    build_fact_bmu_family_shortfall_daily,
    build_fact_dispatch_alignment_bmu_daily,
    build_fact_dispatch_alignment_daily,
    build_fact_bmu_curtailment_gap_bmu_daily,
    build_fact_bmu_curtailment_truth_half_hourly,
    build_fact_constraint_target_audit_daily,
    build_fact_curtailment_gap_reason_daily,
    build_fact_curtailment_reconciliation_daily,
    filter_truth_profile,
)
from weather_history import build_fact_weather_hourly_from_anchor_weather


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


class CurtailmentTruthTests(unittest.TestCase):
    def test_half_hour_spine_handles_dst_days(self) -> None:
        spring = build_half_hour_interval_frame(dt.date(2026, 3, 29), dt.date(2026, 3, 29))
        autumn = build_half_hour_interval_frame(dt.date(2026, 10, 25), dt.date(2026, 10, 25))
        normal = build_half_hour_interval_frame(dt.date(2026, 3, 10), dt.date(2026, 3, 10))
        self.assertEqual(len(normal), 48)
        self.assertEqual(len(spring), 46)
        self.assertEqual(len(autumn), 50)

    def test_overlapping_acceptances_aggregate_into_one_half_hour(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period_from": 1,
                    "settlement_period_to": 1,
                    "time_from_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "time_to_utc": pd.Timestamp("2024-09-30T23:20:00Z"),
                    "time_from_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "time_to_local": pd.Timestamp("2024-10-01T00:20:00+01:00"),
                    "acceptance_time_utc": pd.Timestamp("2024-09-30T22:55:00Z"),
                    "acceptance_time_local": pd.Timestamp("2024-09-30T23:55:00+01:00"),
                    "duration_hours": 20.0 / 60.0,
                    "duration_minutes": 20.0,
                    "source_key": "BOALF",
                    "source_label": "test",
                    "source_dataset": "BOALF",
                    "target_is_proxy": False,
                    "dispatch_truth_tier": "dispatch_acceptance_lower_bound",
                    "is_lower_bound_metric": True,
                    "acceptance_number": 1,
                    "amendment_flag": False,
                    "deemed_bo_flag": False,
                    "so_flag": False,
                    "stor_flag": False,
                    "rr_flag": False,
                    "elexon_bm_unit": "T_TEST-1",
                    "national_grid_bm_unit": "TEST-1",
                    "national_grid_bm_unit_from_fact": "TEST-1",
                    "bm_unit_name": "Test Wind 1",
                    "lead_party_name": "Test Lead",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test GSP",
                    "generation_capacity_mw": 20.0,
                    "level_from_mw": 10.0,
                    "level_to_mw": 6.0,
                    "level_delta_mw": -4.0,
                    "accepted_level_mean_mw": 8.0,
                    "accepted_down_delta_mw": 4.0,
                    "accepted_up_delta_mw": 0.0,
                    "accepted_down_delta_mwh_lower_bound": 4.0 * (20.0 / 60.0),
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_direction": "down",
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period_from": 1,
                    "settlement_period_to": 1,
                    "time_from_utc": pd.Timestamp("2024-09-30T23:10:00Z"),
                    "time_to_utc": pd.Timestamp("2024-09-30T23:30:00Z"),
                    "time_from_local": pd.Timestamp("2024-10-01T00:10:00+01:00"),
                    "time_to_local": pd.Timestamp("2024-10-01T00:30:00+01:00"),
                    "acceptance_time_utc": pd.Timestamp("2024-09-30T23:05:00Z"),
                    "acceptance_time_local": pd.Timestamp("2024-10-01T00:05:00+01:00"),
                    "duration_hours": 20.0 / 60.0,
                    "duration_minutes": 20.0,
                    "source_key": "BOALF",
                    "source_label": "test",
                    "source_dataset": "BOALF",
                    "target_is_proxy": False,
                    "dispatch_truth_tier": "dispatch_acceptance_lower_bound",
                    "is_lower_bound_metric": True,
                    "acceptance_number": 2,
                    "amendment_flag": False,
                    "deemed_bo_flag": False,
                    "so_flag": False,
                    "stor_flag": False,
                    "rr_flag": False,
                    "elexon_bm_unit": "T_TEST-1",
                    "national_grid_bm_unit": "TEST-1",
                    "national_grid_bm_unit_from_fact": "TEST-1",
                    "bm_unit_name": "Test Wind 1",
                    "lead_party_name": "Test Lead",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test GSP",
                    "generation_capacity_mw": 20.0,
                    "level_from_mw": 6.0,
                    "level_to_mw": 4.0,
                    "level_delta_mw": -2.0,
                    "accepted_level_mean_mw": 5.0,
                    "accepted_down_delta_mw": 2.0,
                    "accepted_up_delta_mw": 0.0,
                    "accepted_down_delta_mwh_lower_bound": 2.0 * (20.0 / 60.0),
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_direction": "down",
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                },
            ]
        )
        fact = build_fact_bmu_dispatch_acceptance_half_hourly(events, dt.date(2024, 10, 1), dt.date(2024, 10, 1))
        first_row = fact.iloc[0]
        self.assertEqual(first_row["acceptance_event_count"], 2)
        self.assertEqual(first_row["distinct_acceptance_number_count"], 2)
        self.assertAlmostEqual(first_row["accepted_down_delta_mwh_lower_bound"], 2.0)

    def test_bid_offer_half_hourly_marks_negative_bid_availability(self) -> None:
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
                    "offer": 0.0,
                    "bid": -65.0,
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
                    "offer": 1500.0,
                    "bid": 100.0,
                    "nationalGridBmUnit": "TEST-1",
                    "bmUnit": "T_TEST-1",
                },
            ]
        )
        fact = build_fact_bmu_bid_offer_half_hourly(dim, raw_bid_offer)
        first_row = fact.iloc[0]
        self.assertTrue(bool(first_row["negative_bid_available_flag"]))
        self.assertEqual(int(first_row["negative_bid_pair_count"]), 1)
        self.assertAlmostEqual(float(first_row["most_negative_bid_gbp_per_mwh"]), -65.0)
        self.assertAlmostEqual(float(first_row["least_negative_bid_gbp_per_mwh"]), -65.0)

    def test_physical_baseline_below_generation_is_invalid(self) -> None:
        dim = sample_dim_bmu_asset()
        generation = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "half_hour_start_time_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "half_hour_end_time_local": pd.Timestamp("2024-10-01T00:30:00+01:00"),
                    "half_hour_start_time_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "half_hour_end_time_utc": pd.Timestamp("2024-09-30T23:30:00Z"),
                    "source_key": "B1610",
                    "source_label": "test",
                    "source_dataset": "B1610",
                    "target_is_proxy": False,
                    "elexon_bm_unit": "T_TEST-1",
                    "national_grid_bm_unit": "TEST-1",
                    "national_grid_bm_unit_from_fact": "TEST-1",
                    "bm_unit_name": "Test Wind 1",
                    "lead_party_name": "Test Lead",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test GSP",
                    "generation_capacity_mw": 20.0,
                    "generation_mwh": 10.0,
                    "generation_mw_equivalent": 20.0,
                    "capacity_factor": 1.0,
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                }
            ]
        )
        raw_physical = pd.DataFrame(
            [
                {
                    "dataset": "PN",
                    "settlementDate": "2024-10-01",
                    "settlementPeriod": 1,
                    "timeFrom": "2024-09-30T23:00:00Z",
                    "timeTo": "2024-09-30T23:30:00Z",
                    "levelFrom": 16.0,
                    "levelTo": 16.0,
                    "nationalGridBmUnit": "TEST-1",
                    "bmUnit": "T_TEST-1",
                }
            ]
        )
        fact = build_fact_bmu_physical_position_half_hourly(dim, generation, raw_physical, dt.date(2024, 10, 1), dt.date(2024, 10, 1))
        first_row = fact[fact["settlement_period"] == 1].iloc[0]
        self.assertFalse(bool(first_row["counterfactual_valid_flag"]))

    def test_missing_remit_keeps_availability_unknown(self) -> None:
        dim = sample_dim_bmu_asset()
        raw_uou = pd.DataFrame(
            [
                {
                    "dataset": "UOU2T14D",
                    "bmUnit": "T_TEST-1",
                    "nationalGridBmUnit": "TEST-1",
                    "publishTime": "2026-03-10T00:00:00Z",
                    "forecastDate": "2024-10-01",
                    "outputUsable": 20.0,
                }
            ]
        )
        fact = build_fact_bmu_availability_half_hourly(
            dim_bmu_asset=dim,
            raw_remit_frame=pd.DataFrame(),
            raw_uou_frame=raw_uou,
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            remit_fetch_ok=False,
        )
        first_row = fact[fact["settlement_period"] == 1].iloc[0]
        self.assertEqual(first_row["availability_state"], "unknown")

    def test_partial_day_remit_fetch_status_only_degrades_failed_days(self) -> None:
        dim = sample_dim_bmu_asset()
        remit_status = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "remit_fetch_ok": False,
                    "remit_detail_url_count": 10,
                    "remit_detail_error_count": 1,
                    "remit_first_fetch_error": "synthetic remit failure",
                },
                {
                    "settlement_date": dt.date(2024, 10, 2),
                    "remit_fetch_ok": True,
                    "remit_detail_url_count": 8,
                    "remit_detail_error_count": 0,
                    "remit_first_fetch_error": pd.NA,
                },
            ]
        )
        fact = build_fact_bmu_availability_half_hourly(
            dim_bmu_asset=dim,
            raw_remit_frame=pd.DataFrame(),
            raw_uou_frame=pd.DataFrame(),
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 2),
            remit_fetch_ok=False,
            remit_fetch_status_by_date=remit_status,
        )
        first_day = fact[
            (fact["settlement_date"] == dt.date(2024, 10, 1)) & (fact["settlement_period"] == 1)
        ].iloc[0]
        second_day = fact[
            (fact["settlement_date"] == dt.date(2024, 10, 2)) & (fact["settlement_period"] == 1)
        ].iloc[0]
        self.assertEqual(first_day["availability_state"], "unknown")
        self.assertFalse(bool(first_day["remit_fetch_ok"]))
        self.assertEqual(int(first_day["remit_detail_error_count"]), 1)
        self.assertEqual(first_day["remit_first_fetch_error"], "synthetic remit failure")
        self.assertEqual(second_day["availability_state"], "available")
        self.assertTrue(bool(second_day["remit_fetch_ok"]))
        self.assertEqual(int(second_day["remit_detail_error_count"]), 0)

    def test_partial_remit_downgrades_to_unknown_before_truth_override(self) -> None:
        dim = sample_dim_bmu_asset()
        raw_remit = pd.DataFrame(
            [
                {
                    "mrid": "1",
                    "revisionNumber": 1,
                    "publishTime": "2024-09-30T22:00:00Z",
                    "eventStatus": "active",
                    "eventStartTime": "2024-09-30T23:00:00Z",
                    "eventEndTime": "2024-09-30T23:30:00Z",
                    "affectedUnit": "T_TEST-1",
                    "availableCapacity": 10.0,
                    "normalCapacity": 20.0,
                    "unavailableCapacity": 10.0,
                }
            ]
        )
        fact = build_fact_bmu_availability_half_hourly(
            dim_bmu_asset=dim,
            raw_remit_frame=raw_remit,
            raw_uou_frame=pd.DataFrame(),
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            remit_fetch_ok=True,
        )
        first_row = fact[fact["settlement_period"] == 1].iloc[0]
        self.assertTrue(bool(first_row["remit_partial_availability_flag"]))
        self.assertEqual(first_row["availability_state"], "unknown")
        self.assertEqual(first_row["availability_confidence"], "medium")

    def test_weather_history_aggregates_anchor_weights(self) -> None:
        anchor_weather = pd.DataFrame(
            [
                {
                    "date": dt.date(2024, 10, 1),
                    "hour_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "hour_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "hour_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "hour_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "source_key": "open_meteo_archive",
                    "source_label": "test",
                    "source_dataset": "historical_weather_best_match",
                    "target_is_proxy": False,
                    "anchor_key": "beatrice",
                    "anchor_label": "Beatrice",
                    "requested_latitude": 58.1,
                    "requested_longitude": -2.6,
                    "resolved_latitude": 58.1,
                    "resolved_longitude": -2.6,
                    "resolved_elevation_m": 0.0,
                    "cell_selection": "sea",
                    "temperature_2m_c": 8.0,
                    "pressure_msl_hpa": 1005.0,
                    "cloud_cover_pct": 50.0,
                    "wind_speed_10m_ms": 9.0,
                    "wind_speed_100m_ms": 12.0,
                    "wind_direction_100m_deg": 180.0,
                    "wind_gusts_10m_ms": 12.0,
                    "wind_u_100m_ms": 0.0,
                    "wind_v_100m_ms": 12.0,
                    "wind_power_index_100m": 1728.0,
                    "wind_speed_ratio_100m_to_10m": 12.0 / 9.0,
                },
                {
                    "date": dt.date(2024, 10, 1),
                    "hour_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "hour_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "hour_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "hour_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "source_key": "open_meteo_archive",
                    "source_label": "test",
                    "source_dataset": "historical_weather_best_match",
                    "target_is_proxy": False,
                    "anchor_key": "moray_east",
                    "anchor_label": "Moray East",
                    "requested_latitude": 57.7,
                    "requested_longitude": -2.3,
                    "resolved_latitude": 57.7,
                    "resolved_longitude": -2.3,
                    "resolved_elevation_m": 0.0,
                    "cell_selection": "sea",
                    "temperature_2m_c": 10.0,
                    "pressure_msl_hpa": 1007.0,
                    "cloud_cover_pct": 70.0,
                    "wind_speed_10m_ms": 11.0,
                    "wind_speed_100m_ms": 14.0,
                    "wind_direction_100m_deg": 180.0,
                    "wind_gusts_10m_ms": 14.0,
                    "wind_u_100m_ms": 0.0,
                    "wind_v_100m_ms": 14.0,
                    "wind_power_index_100m": 2744.0,
                    "wind_speed_ratio_100m_to_10m": 14.0 / 11.0,
                },
            ]
        )
        fact_weather = build_fact_weather_hourly_from_anchor_weather(anchor_weather)
        cluster_row = fact_weather[
            (fact_weather["scope_type"] == "cluster")
            & (fact_weather["scope_key"] == "moray_firth_offshore")
        ].iloc[0]
        expected_wind_speed = (12.0 * 588.0 + 14.0 * 950.0) / (588.0 + 950.0)
        self.assertAlmostEqual(cluster_row["wind_speed_100m_ms"], expected_wind_speed)
        self.assertEqual(int(cluster_row["weather_anchor_count"]), 2)

    def test_weather_curve_can_upgrade_invalid_dispatch_row(self) -> None:
        dim = sample_dim_bmu_asset()
        base_day = dt.date(2024, 10, 1)
        generation_rows = []
        availability_rows = []
        dispatch_rows = []
        weather_rows = []
        for hour in range(24):
            hour_start_utc = pd.Timestamp("2024-09-30T23:00:00Z") + pd.Timedelta(hours=hour)
            hour_start_local = hour_start_utc.tz_convert("Europe/London")
            weather_speed = 4.0 + hour
            weather_rows.append(
                {
                    "date": hour_start_local.date(),
                    "hour_start_local": hour_start_local,
                    "hour_end_local": hour_start_local + pd.Timedelta(hours=1),
                    "hour_start_utc": hour_start_utc,
                    "hour_end_utc": hour_start_utc + pd.Timedelta(hours=1),
                    "scope_type": "cluster",
                    "scope_key": "moray_firth_offshore",
                    "scope_label": "Moray Firth Offshore",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                    "source_key": "open_meteo_archive",
                    "source_label": "test",
                    "source_dataset": "historical_weather_best_match",
                    "target_is_proxy": False,
                    "weather_anchor_count": 1,
                    "weather_weight_sum_mw": 1000.0,
                    "temperature_2m_c": 8.0,
                    "pressure_msl_hpa": 1000.0,
                    "cloud_cover_pct": 50.0,
                    "wind_speed_10m_ms": weather_speed - 2.0,
                    "wind_speed_100m_ms": weather_speed,
                    "wind_direction_100m_deg": 180.0,
                    "wind_gusts_10m_ms": weather_speed + 1.0,
                    "wind_u_100m_ms": 0.0,
                    "wind_v_100m_ms": weather_speed,
                    "wind_power_index_100m": weather_speed ** 3,
                    "wind_speed_ratio_100m_to_10m": weather_speed / (weather_speed - 2.0),
                    "resolved_latitude": 58.0,
                    "resolved_longitude": -2.0,
                    "resolved_elevation_m": 0.0,
                }
            )
            for half_hour in range(2):
                settlement_period = hour * 2 + half_hour + 1
                interval_start_utc = hour_start_utc + pd.Timedelta(minutes=30 * half_hour)
                generation_mwh = min((weather_speed / 16.0) ** 3 * 10.0, 10.0)
                if settlement_period == 25:
                    generation_mwh = 2.0
                    dispatch_rows.append(
                        {
                            "settlement_date": base_day,
                            "settlement_period": settlement_period,
                            "elexon_bm_unit": "T_TEST-1",
                            "accepted_down_delta_mwh_lower_bound": 3.0,
                            "accepted_up_delta_mwh_lower_bound": 0.0,
                            "dispatch_down_flag": True,
                            "dispatch_up_flag": False,
                            "acceptance_event_count": 1,
                            "distinct_acceptance_number_count": 1,
                        }
                    )
                generation_rows.append(
                    {
                        "settlement_date": base_day,
                        "settlement_period": settlement_period,
                        "elexon_bm_unit": "T_TEST-1",
                        "generation_mwh": generation_mwh,
                    }
                )
                availability_rows.append(
                    {
                        "settlement_date": base_day,
                        "settlement_period": settlement_period,
                        "elexon_bm_unit": "T_TEST-1",
                        "remit_active_flag": False,
                        "availability_state": "available",
                        "availability_confidence": "high",
                        "uou_output_usable_mw": 20.0,
                    }
                )

        physical = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 25,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": pd.NA,
                    "physical_baseline_mwh": np.nan,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "none",
                    "counterfactual_valid_flag": False,
                }
            ]
        )
        constraints = add_constraint_qa_columns(
            pd.DataFrame(
                [
                    {
                        "date": base_day,
                        "total_curtailment_mwh": 999.0,
                        "voltage_constraints_volume_mwh": 999.0,
                        "thermal_constraints_volume_mwh": 0.0,
                        "increasing_system_inertia_volume_mwh": 0.0,
                        "reducing_largest_loss_volume_mwh": 0.0,
                    }
                ]
            )
        )

        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=pd.DataFrame(generation_rows),
            fact_bmu_dispatch_acceptance_half_hourly=pd.DataFrame(dispatch_rows),
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=pd.DataFrame(availability_rows),
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(weather_rows),
            start_date=base_day,
            end_date=base_day,
        )
        upgraded_row = fact[fact["settlement_period"] == 25].iloc[0]
        self.assertEqual(upgraded_row["truth_tier"], "weather_calibrated")
        self.assertTrue(bool(upgraded_row["lost_energy_estimate_flag"]))
        self.assertEqual(upgraded_row["counterfactual_method"], "bmu_weather_power_curve")

    def test_dispatch_truth_expands_from_negative_bid_and_pn_qpn_gap(self) -> None:
        dim = sample_dim_bmu_asset()
        base_day = dt.date(2024, 10, 1)
        generation = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 4.0,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 7.0,
                }
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 0.5,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                }
            ]
        )
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "pn_mwh": 7.0,
                    "qpn_mwh": 2.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 7.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                }
            ]
        )
        bid_offer = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "negative_bid_pair_count": 1,
                    "negative_bid_available_flag": True,
                    "most_negative_bid_gbp_per_mwh": -80.0,
                    "least_negative_bid_gbp_per_mwh": -80.0,
                }
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                }
            ]
        )
        constraints = add_constraint_qa_columns(
            pd.DataFrame(
                [
                    {
                        "date": base_day,
                        "total_curtailment_mwh": 3.0,
                        "voltage_constraints_volume_mwh": 3.0,
                        "thermal_constraints_volume_mwh": 0.0,
                        "increasing_system_inertia_volume_mwh": 0.0,
                        "reducing_largest_loss_volume_mwh": 0.0,
                    }
                ]
            )
        )

        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=generation,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=availability,
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(),
            start_date=base_day,
            end_date=base_day,
            fact_bmu_bid_offer_half_hourly=bid_offer,
        )
        first_row = fact.iloc[0]
        self.assertTrue(bool(first_row["dispatch_truth_flag"]))
        self.assertTrue(bool(first_row["dispatch_acceptance_window_flag"]))
        self.assertEqual(first_row["dispatch_truth_source_tier"], "physical_inference")
        self.assertAlmostEqual(float(first_row["physical_dispatch_down_gap_mwh"]), 5.0)
        self.assertAlmostEqual(float(first_row["physical_dispatch_down_increment_mwh_lower_bound"]), 5.0)
        self.assertAlmostEqual(float(first_row["dispatch_down_evidence_mwh_lower_bound"]), 5.0)
        self.assertTrue(bool(first_row["research_profile_include"]))
        self.assertTrue(bool(first_row["precision_profile_include"]))
        self.assertIn("BOD", first_row["source_lineage"])
        self.assertIn("balancing_physical", first_row["source_lineage"])

    def test_family_day_dispatch_expansion_can_promote_sibling_bmu(self) -> None:
        base_day = dt.date(2024, 10, 1)
        dim = pd.DataFrame(
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
                    "generation_capacity_mw": 40.0,
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                },
                {
                    "elexon_bm_unit": "T_TEST-2",
                    "national_grid_bm_unit": "TEST-2",
                    "bm_unit_name": "Test Wind 2",
                    "lead_party_name": "Test Lead",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test GSP",
                    "generation_capacity_mw": 40.0,
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                },
            ]
        )
        generation = pd.DataFrame(
            [
                *[
                    {
                        "settlement_date": base_day,
                        "settlement_period": period,
                        "elexon_bm_unit": "T_TEST-1",
                        "generation_mwh": 10.0,
                    }
                    for period in [3, 4, 5, 6, 7]
                ],
                {
                    "settlement_date": base_day,
                    "settlement_period": 8,
                    "elexon_bm_unit": "T_TEST-2",
                    "generation_mwh": 10.0,
                },
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 5,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 1.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                }
            ]
        )
        physical = pd.DataFrame(
            [
                *[
                    {
                        "settlement_date": base_day,
                        "settlement_period": period,
                        "elexon_bm_unit": "T_TEST-1",
                        "pn_mwh": 40.0,
                        "qpn_mwh": 0.0,
                        "physical_baseline_source_dataset": "PN",
                        "physical_baseline_mwh": 20.0,
                        "physical_consistency_flag": True,
                        "counterfactual_method": "pn_qpn_physical_max",
                        "counterfactual_valid_flag": True,
                    }
                    for period in [3, 4, 5, 6, 7]
                ],
                {
                    "settlement_date": base_day,
                    "settlement_period": 8,
                    "elexon_bm_unit": "T_TEST-2",
                    "pn_mwh": 30.0,
                    "qpn_mwh": 0.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 20.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
            ]
        )
        availability = pd.DataFrame(
            [
                *[
                    {
                        "settlement_date": base_day,
                        "settlement_period": period,
                        "elexon_bm_unit": "T_TEST-1",
                        "remit_active_flag": False,
                        "availability_state": "available",
                        "availability_confidence": "high",
                        "uou_output_usable_mw": 40.0,
                    }
                    for period in [3, 4, 5, 6, 7]
                ],
                {
                    "settlement_date": base_day,
                    "settlement_period": 8,
                    "elexon_bm_unit": "T_TEST-2",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 40.0,
                },
            ]
        )
        bid_offer = pd.DataFrame(
            [
                *[
                    {
                        "settlement_date": base_day,
                        "settlement_period": period,
                        "elexon_bm_unit": "T_TEST-1",
                        "negative_bid_pair_count": 1,
                        "negative_bid_available_flag": True,
                        "most_negative_bid_gbp_per_mwh": -80.0,
                        "least_negative_bid_gbp_per_mwh": -80.0,
                    }
                    for period in [3, 4, 5, 6, 7]
                ],
                {
                    "settlement_date": base_day,
                    "settlement_period": 8,
                    "elexon_bm_unit": "T_TEST-2",
                    "negative_bid_pair_count": 1,
                    "negative_bid_available_flag": True,
                    "most_negative_bid_gbp_per_mwh": -90.0,
                    "least_negative_bid_gbp_per_mwh": -90.0,
                },
            ]
        )
        constraints = add_constraint_qa_columns(
            pd.DataFrame(
                [
                    {
                        "date": base_day,
                        "total_curtailment_mwh": 200.0,
                        "voltage_constraints_volume_mwh": 200.0,
                        "thermal_constraints_volume_mwh": 0.0,
                        "increasing_system_inertia_volume_mwh": 0.0,
                        "reducing_largest_loss_volume_mwh": 0.0,
                    }
                ]
            )
        )

        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=generation,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=availability,
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(),
            start_date=base_day,
            end_date=base_day,
            fact_bmu_bid_offer_half_hourly=bid_offer,
        )
        sibling_row = fact[
            (fact["elexon_bm_unit"] == "T_TEST-2")
            & (fact["settlement_period"] == 8)
        ].iloc[0]
        self.assertFalse(bool(sibling_row["dispatch_acceptance_window_flag"]))
        self.assertTrue(bool(sibling_row["family_day_dispatch_expansion_eligible_flag"]))
        self.assertTrue(bool(sibling_row["family_day_dispatch_window_flag"]))
        self.assertTrue(bool(sibling_row["family_day_dispatch_expansion_applied_flag"]))
        self.assertEqual(sibling_row["dispatch_inference_scope"], "family_day_window")
        self.assertEqual(sibling_row["dispatch_truth_source_tier"], "physical_inference")
        self.assertAlmostEqual(float(sibling_row["family_day_dispatch_increment_mwh_lower_bound"]), 30.0)
        self.assertAlmostEqual(float(sibling_row["dispatch_down_evidence_mwh_lower_bound"]), 30.0)
        self.assertTrue(bool(sibling_row["dispatch_truth_flag"]))

        daily = build_fact_curtailment_reconciliation_daily(fact)
        self.assertAlmostEqual(float(daily.iloc[0]["family_day_dispatch_increment_mwh_lower_bound"]), 30.0)
        self.assertEqual(int(daily.iloc[0]["dispatch_family_day_inference_row_count"]), 1)

    def test_truth_profiles_follow_row_flags(self) -> None:
        dim = sample_dim_bmu_asset()
        generation = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 4.0,
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 5.0,
                },
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 2.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                }
            ]
        )
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 6.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 5.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                },
            ]
        )
        constraints = add_constraint_qa_columns(
            pd.DataFrame(
                [
                    {
                        "date": dt.date(2024, 10, 1),
                        "total_curtailment_mwh": 2.0,
                        "voltage_constraints_volume_mwh": 2.0,
                        "thermal_constraints_volume_mwh": 0.0,
                        "increasing_system_inertia_volume_mwh": 0.0,
                        "reducing_largest_loss_volume_mwh": 0.0,
                    }
                ]
            )
        )
        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=generation,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=availability,
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(),
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
        )
        period_1 = fact[fact["settlement_period"] == 1].iloc[0]
        period_2 = fact[fact["settlement_period"] == 2].iloc[0]
        self.assertEqual(period_1["truth_tier"], "physical_baseline")
        self.assertTrue(bool(period_1["precision_profile_include"]))
        self.assertTrue(bool(period_1["research_profile_include"]))
        self.assertEqual(period_1["qa_reconciliation_status"], "pass")
        self.assertEqual(period_1["raw_reconciliation_status"], "pass")
        self.assertEqual(period_2["truth_tier"], "physical_baseline")
        self.assertTrue(bool(period_2["precision_profile_include"]))
        self.assertFalse(bool(period_2["research_profile_include"]))

        precision = filter_truth_profile(fact, "precision")
        research = filter_truth_profile(fact, "research")
        self.assertEqual(set(precision["settlement_period"]), {1, 2})
        self.assertEqual(set(research["settlement_period"]), {1})

    def test_partial_remit_can_be_overridden_when_available_capacity_supports_counterfactual(self) -> None:
        dim = sample_dim_bmu_asset()
        base_day = dt.date(2024, 10, 1)
        generation = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 4.0,
                }
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 2.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                }
            ]
        )
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 6.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                }
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": True,
                    "remit_partial_availability_flag": True,
                    "remit_max_available_capacity_mw": 12.5,
                    "remit_max_unavailable_capacity_mw": 7.5,
                    "remit_normal_capacity_mw": 20.0,
                    "availability_state": "unknown",
                    "availability_confidence": "medium",
                    "uou_output_usable_mw": np.nan,
                }
            ]
        )
        constraints = add_constraint_qa_columns(
            pd.DataFrame(
                [
                    {
                        "date": base_day,
                        "total_curtailment_mwh": 2.0,
                        "voltage_constraints_volume_mwh": 2.0,
                        "thermal_constraints_volume_mwh": 0.0,
                        "increasing_system_inertia_volume_mwh": 0.0,
                        "reducing_largest_loss_volume_mwh": 0.0,
                    }
                ]
            )
        )

        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=generation,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=availability,
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(),
            start_date=base_day,
            end_date=base_day,
        )
        first_row = fact.iloc[0]
        self.assertEqual(first_row["availability_state_raw"], "unknown")
        self.assertEqual(first_row["availability_state"], "available")
        self.assertTrue(bool(first_row["availability_override_flag"]))
        self.assertEqual(
            first_row["availability_override_reason"],
            "remit_partial_available_capacity_supports_counterfactual",
        )
        self.assertTrue(bool(first_row["lost_energy_estimate_flag"]))

    def test_reconciliation_diagnostics_explain_dispatch_gap(self) -> None:
        dim = sample_dim_bmu_asset()
        base_day = dt.date(2024, 10, 1)
        generation = pd.DataFrame(
            [
                {"settlement_date": base_day, "settlement_period": 1, "elexon_bm_unit": "T_TEST-1", "generation_mwh": 4.0},
                {"settlement_date": base_day, "settlement_period": 2, "elexon_bm_unit": "T_TEST-1", "generation_mwh": 5.0},
                {"settlement_date": base_day, "settlement_period": 3, "elexon_bm_unit": "T_TEST-1", "generation_mwh": 5.5},
                {"settlement_date": base_day, "settlement_period": 4, "elexon_bm_unit": "T_TEST-1", "generation_mwh": 6.0},
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 2.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 3.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 3,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 1.5,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 4,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 1.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                },
            ]
        )
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 6.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 7.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 3,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 7.5,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 4,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 5.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": False,
                },
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "unknown",
                    "availability_confidence": "low",
                    "uou_output_usable_mw": np.nan,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 3,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "unknown",
                    "availability_confidence": "low",
                    "uou_output_usable_mw": np.nan,
                },
                {
                    "settlement_date": base_day,
                    "settlement_period": 4,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                },
            ]
        )
        constraints = add_constraint_qa_columns(
            pd.DataFrame(
                [
                    {
                        "date": base_day,
                        "total_curtailment_mwh": 10.0,
                        "voltage_constraints_volume_mwh": 2.0,
                        "thermal_constraints_volume_mwh": 0.0,
                        "increasing_system_inertia_volume_mwh": 8.0,
                        "reducing_largest_loss_volume_mwh": 0.0,
                    }
                ]
            )
        )

        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=generation,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=availability,
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(),
            start_date=base_day,
            end_date=base_day,
        )

        first_four = fact[fact["settlement_period"].isin([1, 2, 3, 4])].set_index("settlement_period")
        self.assertEqual(first_four.loc[1, "lost_energy_block_reason"], "estimated")
        self.assertEqual(first_four.loc[2, "lost_energy_block_reason"], "availability_unknown")
        self.assertEqual(first_four.loc[3, "lost_energy_block_reason"], "availability_unknown")
        self.assertEqual(first_four.loc[4, "counterfactual_invalid_reason"], "physical_below_generation")
        self.assertEqual(first_four.loc[4, "lost_energy_block_reason"], "physical_below_generation")
        self.assertEqual(first_four.loc[1, "raw_reconciliation_status"], "fail")
        self.assertEqual(first_four.loc[1, "qa_reconciliation_status"], "pass")

        daily = build_fact_curtailment_reconciliation_daily(fact)
        self.assertEqual(len(daily), 1)
        self.assertEqual(int(daily.iloc[0]["dispatch_half_hour_count"]), 4)
        self.assertAlmostEqual(float(daily.iloc[0]["dispatch_down_mwh_lower_bound"]), 7.5)
        self.assertEqual(int(daily.iloc[0]["lost_energy_estimate_half_hour_count"]), 1)
        self.assertEqual(daily.iloc[0]["primary_dispatch_block_reason"], "availability_unknown")
        self.assertEqual(daily.iloc[0]["raw_reconciliation_status"], "fail")
        self.assertEqual(daily.iloc[0]["qa_reconciliation_status"], "pass")
        self.assertAlmostEqual(float(daily.iloc[0]["dispatch_coverage_ratio_vs_raw_total"]), 0.75)
        self.assertAlmostEqual(float(daily.iloc[0]["dispatch_coverage_ratio_vs_qa_target"]), 3.75)

        dispatch_alignment_daily = build_fact_dispatch_alignment_daily(fact)
        self.assertEqual(len(dispatch_alignment_daily), 1)
        self.assertAlmostEqual(float(dispatch_alignment_daily.iloc[0]["estimated_dispatch_down_mwh_lower_bound"]), 2.0)
        self.assertAlmostEqual(float(dispatch_alignment_daily.iloc[0]["blocked_dispatch_down_mwh_lower_bound"]), 5.5)
        self.assertAlmostEqual(
            float(dispatch_alignment_daily.iloc[0]["blocked_availability_unknown_dispatch_down_mwh_lower_bound"]),
            4.5,
        )
        self.assertAlmostEqual(
            float(dispatch_alignment_daily.iloc[0]["blocked_physical_below_generation_dispatch_down_mwh_lower_bound"]),
            1.0,
        )
        self.assertEqual(
            dispatch_alignment_daily.iloc[0]["dispatch_alignment_inference"],
            "qa_target_met_or_exceeded",
        )

        reason_daily = build_fact_curtailment_gap_reason_daily(fact)
        reason_rows = {
            row["lost_energy_block_reason"]: row
            for _, row in reason_daily.iterrows()
        }
        self.assertEqual(int(reason_rows["availability_unknown"]["dispatch_half_hour_count"]), 2)
        self.assertAlmostEqual(float(reason_rows["availability_unknown"]["accepted_down_delta_mwh_lower_bound"]), 4.5)
        self.assertEqual(int(reason_rows["physical_below_generation"]["dispatch_half_hour_count"]), 1)

        bmu_daily = build_fact_bmu_curtailment_gap_bmu_daily(fact)
        self.assertEqual(len(bmu_daily), 1)
        self.assertEqual(bmu_daily.iloc[0]["primary_dispatch_block_reason"], "availability_unknown")
        self.assertAlmostEqual(float(bmu_daily.iloc[0]["dispatch_minus_lost_energy_gap_mwh"]), 5.5)

        dispatch_alignment_bmu_daily = build_fact_dispatch_alignment_bmu_daily(fact)
        self.assertEqual(len(dispatch_alignment_bmu_daily), 1)
        self.assertEqual(dispatch_alignment_bmu_daily.iloc[0]["dispatch_alignment_state"], "partially_blocked")
        self.assertAlmostEqual(float(dispatch_alignment_bmu_daily.iloc[0]["blocked_dispatch_down_mwh_lower_bound"]), 5.5)

        constraint_audit_daily = build_fact_constraint_target_audit_daily(fact, constraints)
        self.assertEqual(len(constraint_audit_daily), 1)
        self.assertEqual(constraint_audit_daily.iloc[0]["qa_primary_driver"], "voltage_dominant")
        self.assertEqual(constraint_audit_daily.iloc[0]["recoverability_audit_state"], "overcaptured_or_definition_mismatch")

        family_shortfall_daily = build_fact_bmu_family_shortfall_daily(fact)
        self.assertEqual(len(family_shortfall_daily), 1)
        self.assertEqual(family_shortfall_daily.iloc[0]["bmu_family_key"], "TEST")
        self.assertEqual(family_shortfall_daily.iloc[0]["bmu_family_label"], "TEST")
        self.assertAlmostEqual(float(family_shortfall_daily.iloc[0]["dispatch_minus_lost_energy_gap_mwh"]), 5.5)

    def test_missing_qa_target_keeps_precision_profile_conservative(self) -> None:
        dim = sample_dim_bmu_asset()
        generation = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 4.0,
                }
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 2.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                }
            ]
        )
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 6.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                }
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                }
            ]
        )
        constraints = pd.DataFrame(
            [
                {
                    "date": dt.date(2024, 10, 1),
                    "total_curtailment_mwh": 2.0,
                }
            ]
        )

        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=generation,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=availability,
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(),
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
        )

        first_row = fact.iloc[0]
        self.assertEqual(first_row["reconciliation_status"], "pass")
        self.assertEqual(first_row["qa_reconciliation_status"], "warn")
        self.assertFalse(bool(first_row["precision_profile_include"]))

        legacy_view = fact.drop(
            columns=[
                "qa_target_definition",
                "gb_daily_qa_target_mwh",
                "qa_reconciliation_abs_error_mwh",
                "qa_reconciliation_relative_error",
                "qa_reconciliation_status",
                "gb_daily_raw_constraint_total_mwh",
                "raw_reconciliation_abs_error_mwh",
                "raw_reconciliation_relative_error",
                "raw_reconciliation_status",
            ]
        )
        daily = build_fact_curtailment_reconciliation_daily(legacy_view)
        self.assertEqual(daily.iloc[0]["qa_reconciliation_status"], "warn")

        dispatch_alignment_daily = build_fact_dispatch_alignment_daily(legacy_view)
        self.assertEqual(dispatch_alignment_daily.iloc[0]["dispatch_alignment_inference"], "qa_target_missing")

    def test_constraint_target_audit_backfills_qa_columns_from_raw_constraints(self) -> None:
        fact = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "dispatch_truth_flag": True,
                    "lost_energy_estimate_flag": True,
                    "accepted_down_delta_mwh_lower_bound": 2.0,
                    "dispatch_down_evidence_mwh_lower_bound": 2.0,
                    "lost_energy_mwh": 1.0,
                    "elexon_bm_unit": "T_TEST-1",
                    "qa_target_definition": CONSTRAINT_QA_TARGET_DEFINITION,
                    "gb_daily_qa_target_mwh": 2.0,
                    "qa_reconciliation_status": "pass",
                    "gb_daily_estimated_lost_energy_mwh": 1.0,
                    "gb_daily_raw_constraint_total_mwh": 2.0,
                    "raw_reconciliation_status": "warn",
                    "gb_daily_truth_curtailment_mwh": 2.0,
                    "reconciliation_status": "warn",
                    "mapping_status": "mapped",
                    "truth_tier": "physical_baseline",
                    "availability_state": "available",
                    "counterfactual_valid_flag": True,
                    "lost_energy_block_reason": "estimated",
                }
            ]
        )
        raw_constraints = pd.DataFrame(
            [
                {
                    "date": dt.date(2024, 10, 1),
                    "total_curtailment_mwh": 10.0,
                    "voltage_constraints_volume_mwh": 8.0,
                    "thermal_constraints_volume_mwh": 2.0,
                    "increasing_system_inertia_volume_mwh": 0.0,
                    "reducing_largest_loss_volume_mwh": 0.0,
                }
            ]
        )
        audit = build_fact_constraint_target_audit_daily(fact, raw_constraints)
        self.assertEqual(audit.iloc[0]["qa_primary_driver"], "voltage_dominant")
        self.assertAlmostEqual(float(audit.iloc[0]["qa_voltage_share_of_target"]), 0.8)


if __name__ == "__main__":
    unittest.main()
