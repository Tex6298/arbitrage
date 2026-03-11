# Support Case Summary

- Generated at UTC: `2026-03-11T14:31:22+00:00`
- Source DB: `C:\Users\marty\Documents\LocalPython\arbitrage\support_loop_smoke.sqlite`
- Batch ID: `support_fail_warn_days7_families5_2024-10-01_2024-10-07`
- Status filter: `fail_warn`
- Top days: `7`
- Top families per day: `5`
- Selected days: `7`
- Selected family-days: `35`

## Day 1: 2024-10-06

- QA status: `fail`
- Recoverability state: `source_limited`
- Dominant anomaly: `negative_bid_without_boalf`
- Anomaly MWh: `230750.492`
- Remaining QA shortfall MWh: `10225.763`
- Recommended support action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`
- Publication next action: `support_query_missing_published_boalf`
- Selected families: `5`

### Family 1: HOWAO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `25186.500`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWAO-1, T_HOWAO-2, T_HOWAO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 23 | 2024-10-06 10:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 192.75 | -151.89 | 0 | 0 |
| 23 | 2024-10-06 10:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 192.25 | -151.89 | 0 | 0 |
| 24 | 2024-10-06 10:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 191.75 | -151.89 | 0 | 0 |
| 21 | 2024-10-06 09:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 191.5 | -156.12 | 0 | 0 |
| 33 | 2024-10-06 15:00:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 191.5 | -153.66 | 0 | 0 |
| 33 | 2024-10-06 15:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191.5 | -153.66 | 0 | 0 |
| 22 | 2024-10-06 09:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 191.25 | -156.12 | 0 | 0 |
| 21 | 2024-10-06 09:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -156.12 | 0 | 0 |
| 25 | 2024-10-06 11:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 191 | -154.28 | 0 | 0 |
| 26 | 2024-10-06 11:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 191 | -154.28 | 0 | 0 |

### Family 2: HOWBO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `25147.000`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWBO-1, T_HOWBO-2, T_HOWBO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 2024-10-06 10:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 210.5 | -131.03 | 0 | 0 |
| 24 | 2024-10-06 10:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.5 | -131.03 | 0 | 0 |
| 22 | 2024-10-06 09:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 210.25 | -131.03 | 0 | 0 |
| 22 | 2024-10-06 09:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.25 | -131.03 | 0 | 0 |
| 23 | 2024-10-06 10:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 209.25 | -131.03 | 0 | 0 |
| 23 | 2024-10-06 10:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 209.25 | -131.03 | 0 | 0 |
| 25 | 2024-10-06 11:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 208 | -131.03 | 0 | 0 |
| 25 | 2024-10-06 11:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 208 | -131.03 | 0 | 0 |
| 26 | 2024-10-06 11:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 207.5 | -131.03 | 0 | 0 |
| 26 | 2024-10-06 11:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 207.5 | -131.03 | 0 | 0 |

### Family 4: MOWEO (Moray East)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `12116.500`
- Region / Cluster: `Scotland` / `Moray Firth Offshore`
- Mapping status: `mapped`
- BMUs: `T_MOWEO-1, T_MOWEO-2, T_MOWEO-3`
- Zero-BOALF rows: `142`
- Negative-bid rows: `142`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48 | 2024-10-06 22:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 122.5 | -48.34 | 0 | 0 |
| 43 | 2024-10-06 20:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 122 | -39.02 | 0 | 0 |
| 47 | 2024-10-06 22:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 122 | -48.34 | 0 | 0 |
| 34 | 2024-10-06 15:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 120 | -35.5 | 0 | 0 |
| 44 | 2024-10-06 20:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 120 | -39.02 | 0 | 0 |
| 42 | 2024-10-06 19:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 118.5 | -37.04 | 0 | 0 |
| 41 | 2024-10-06 19:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 118 | -37.04 | 0 | 0 |
| 35 | 2024-10-06 16:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 117 | -23.35 | 0 | 0 |
| 46 | 2024-10-06 21:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 116 | -48.34 | 0 | 0 |
| 45 | 2024-10-06 21:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 115.5 | -48.34 | 0 | 0 |

### Family 5: SGRWO (Seagreen)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `11264.350`
- Region / Cluster: `Scotland` / `East Coast Scotland Offshore`
- Mapping status: `mapped`
- BMUs: `T_SGRWO-1, T_SGRWO-2, T_SGRWO-3, T_SGRWO-4, T_SGRWO-5, T_SGRWO-6`
- Zero-BOALF rows: `272`
- Negative-bid rows: `272`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2024-10-05 23:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 3 | 2024-10-06 00:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 4 | 2024-10-06 00:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 5 | 2024-10-06 01:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 6 | 2024-10-06 01:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 7 | 2024-10-06 02:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 8 | 2024-10-06 02:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 9 | 2024-10-06 03:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 10 | 2024-10-06 03:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 11 | 2024-10-06 04:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |

### Family 7: EAAO (East Anglia)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `8160.000`
- Region / Cluster: `England/Wales` / `East Anglia Offshore`
- Mapping status: `mapped`
- BMUs: `T_EAAO-2`
- Zero-BOALF rows: `48`
- Negative-bid rows: `48`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2024-10-05 23:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -185.75 | 0 | 0 |
| 2 | 2024-10-05 23:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -185.75 | 0 | 0 |
| 3 | 2024-10-06 00:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -188.14 | 0 | 0 |
| 4 | 2024-10-06 00:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -188.14 | 0 | 0 |
| 5 | 2024-10-06 01:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -189.82 | 0 | 0 |
| 6 | 2024-10-06 01:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -189.82 | 0 | 0 |
| 7 | 2024-10-06 02:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -190.61 | 0 | 0 |
| 8 | 2024-10-06 02:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -190.61 | 0 | 0 |
| 9 | 2024-10-06 03:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -192.54 | 0 | 0 |
| 10 | 2024-10-06 03:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -192.54 | 0 | 0 |

## Day 2: 2024-10-01

- QA status: `fail`
- Recoverability state: `counterfactual_or_definition_limited`
- Dominant anomaly: `negative_bid_without_boalf`
- Anomaly MWh: `227076.242`
- Remaining QA shortfall MWh: `7973.682`
- Recommended support action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`
- Publication next action: `support_query_missing_published_boalf`
- Selected families: `5`

### Family 1: HOWBO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `23565.750`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWBO-1, T_HOWBO-2, T_HOWBO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 35 | 2024-10-01 16:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 211.25 | -89.33 | 0 | 0 |
| 36 | 2024-10-01 16:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 211.25 | -89.33 | 0 | 0 |
| 37 | 2024-10-01 17:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.25 | -52.82 | 0 | 0 |
| 48 | 2024-10-01 22:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 209 | -122.9 | 0 | 0 |
| 35 | 2024-10-01 16:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 208.75 | -89.33 | 0 | 0 |
| 36 | 2024-10-01 16:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 208.75 | -89.33 | 0 | 0 |
| 38 | 2024-10-01 17:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 208.75 | -52.82 | 0 | 0 |
| 39 | 2024-10-01 18:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 208.5 | -36.77 | 0 | 0 |
| 47 | 2024-10-01 22:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 208.5 | -122.9 | 0 | 0 |
| 46 | 2024-10-01 21:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 208 | -56.86 | 0 | 0 |

### Family 2: HOWAO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `21275.000`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWAO-1, T_HOWAO-2, T_HOWAO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 39 | 2024-10-01 18:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191.75 | -135.61 | 0 | 0 |
| 40 | 2024-10-01 18:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -135.61 | 0 | 0 |
| 41 | 2024-10-01 19:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -161.36 | 0 | 0 |
| 42 | 2024-10-01 19:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -161.36 | 0 | 0 |
| 43 | 2024-10-01 20:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -170 | 0 | 0 |
| 44 | 2024-10-01 20:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 190.5 | -170 | 0 | 0 |
| 45 | 2024-10-01 21:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 190.5 | -172.81 | 0 | 0 |
| 46 | 2024-10-01 21:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 190.5 | -172.81 | 0 | 0 |
| 47 | 2024-10-01 22:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 190.5 | -174.13 | 0 | 0 |
| 48 | 2024-10-01 22:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 190.5 | -174.13 | 0 | 0 |

### Family 5: RCBKO (Race Bank)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `10853.500`
- Region / Cluster: `England/Wales` / `Humber Offshore`
- Mapping status: `mapped`
- BMUs: `T_RCBKO-1, T_RCBKO-2`
- Zero-BOALF rows: `94`
- Negative-bid rows: `94`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2024-10-01 01:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.75 | -167.26 | 0 | 0 |
| 7 | 2024-10-01 02:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.75 | -167.26 | 0 | 0 |
| 9 | 2024-10-01 03:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.75 | -167.26 | 0 | 0 |
| 2 | 2024-09-30 23:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.5 | -167.26 | 0 | 0 |
| 3 | 2024-10-01 00:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.5 | -167.26 | 0 | 0 |
| 4 | 2024-10-01 00:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.5 | -167.26 | 0 | 0 |
| 5 | 2024-10-01 01:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.5 | -167.26 | 0 | 0 |
| 10 | 2024-10-01 03:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 134.5 | -167.26 | 0 | 0 |
| 8 | 2024-10-01 02:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 133.75 | -167.26 | 0 | 0 |
| 12 | 2024-10-01 04:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 133.25 | -167.26 | 0 | 0 |

### Family 6: TKNWW (Triton Knoll)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `8298.500`
- Region / Cluster: `England/Wales` / `Humber Offshore`
- Mapping status: `mapped`
- BMUs: `T_TKNWW-1`
- Zero-BOALF rows: `48`
- Negative-bid rows: `48`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 38 | 2024-10-01 17:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 206 | -57.13 | 0 | 0 |
| 39 | 2024-10-01 18:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 205.5 | -63.99 | 0 | 0 |
| 40 | 2024-10-01 18:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 205.5 | -63.99 | 0 | 0 |
| 41 | 2024-10-01 19:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 205 | -86.52 | 0 | 0 |
| 42 | 2024-10-01 19:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 205 | -86.52 | 0 | 0 |
| 44 | 2024-10-01 20:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 205 | -94.19 | 0 | 0 |
| 43 | 2024-10-01 20:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 204.5 | -94.19 | 0 | 0 |
| 45 | 2024-10-01 21:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 204.5 | -96.46 | 0 | 0 |
| 46 | 2024-10-01 21:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 204 | -96.46 | 0 | 0 |
| 1 | 2024-09-30 23:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203 | -117.43 | 0 | 0 |

### Family 7: TKNEW (Triton Knoll)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `7753.500`
- Region / Cluster: `England/Wales` / `Humber Offshore`
- Mapping status: `mapped`
- BMUs: `T_TKNEW-1`
- Zero-BOALF rows: `48`
- Negative-bid rows: `48`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2024-09-30 23:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 200 | -117.43 | 0 | 0 |
| 2 | 2024-09-30 23:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 200 | -117.43 | 0 | 0 |
| 3 | 2024-10-01 00:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 199.5 | -131.78 | 0 | 0 |
| 4 | 2024-10-01 00:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 199 | -131.78 | 0 | 0 |
| 42 | 2024-10-01 19:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 199 | -86.52 | 0 | 0 |
| 43 | 2024-10-01 20:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 199 | -94.19 | 0 | 0 |
| 5 | 2024-10-01 01:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 198.5 | -149.37 | 0 | 0 |
| 38 | 2024-10-01 17:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 198.5 | -57.13 | 0 | 0 |
| 39 | 2024-10-01 18:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 198.5 | -63.99 | 0 | 0 |
| 40 | 2024-10-01 18:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 198.5 | -63.99 | 0 | 0 |

## Day 3: 2024-10-02

- QA status: `fail`
- Recoverability state: `source_limited`
- Dominant anomaly: `negative_bid_without_boalf`
- Anomaly MWh: `190813.800`
- Remaining QA shortfall MWh: `10878.878`
- Recommended support action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`
- Publication next action: `support_query_missing_published_boalf`
- Selected families: `5`

### Family 1: HOWBO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `26624.500`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWBO-1, T_HOWBO-2, T_HOWBO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2024-10-02 03:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.5 | -131.03 | 0 | 0 |
| 11 | 2024-10-02 04:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.5 | -131.03 | 0 | 0 |
| 12 | 2024-10-02 04:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.5 | -131.03 | 0 | 0 |
| 7 | 2024-10-02 02:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.25 | -131.03 | 0 | 0 |
| 8 | 2024-10-02 02:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210.25 | -131.03 | 0 | 0 |
| 9 | 2024-10-02 03:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210 | -131.03 | 0 | 0 |
| 13 | 2024-10-02 05:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210 | -131.03 | 0 | 0 |
| 14 | 2024-10-02 05:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210 | -131.03 | 0 | 0 |
| 15 | 2024-10-02 06:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210 | -129.54 | 0 | 0 |
| 19 | 2024-10-02 08:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 210 | -131.03 | 0 | 0 |

### Family 2: HOWAO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `25967.250`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWAO-1, T_HOWAO-2, T_HOWAO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 2024-10-02 03:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191.25 | -177.96 | 0 | 0 |
| 3 | 2024-10-02 00:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -180.55 | 0 | 0 |
| 4 | 2024-10-02 00:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -180.55 | 0 | 0 |
| 6 | 2024-10-02 01:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -181.54 | 0 | 0 |
| 7 | 2024-10-02 02:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -181.53 | 0 | 0 |
| 8 | 2024-10-02 02:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -181.53 | 0 | 0 |
| 10 | 2024-10-02 03:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -177.96 | 0 | 0 |
| 11 | 2024-10-02 04:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -210.89 | 0 | 0 |
| 12 | 2024-10-02 04:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -210.89 | 0 | 0 |
| 13 | 2024-10-02 05:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 191 | -168.92 | 0 | 0 |

### Family 3: RCBKO (Race Bank)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `11630.000`
- Region / Cluster: `England/Wales` / `Humber Offshore`
- Mapping status: `mapped`
- BMUs: `T_RCBKO-1, T_RCBKO-2`
- Zero-BOALF rows: `96`
- Negative-bid rows: `96`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 2024-10-02 06:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 129.5 | -167.26 | 0 | 0 |
| 17 | 2024-10-02 07:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 129.5 | -167.26 | 0 | 0 |
| 18 | 2024-10-02 07:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 129.5 | -167.26 | 0 | 0 |
| 19 | 2024-10-02 08:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 129.25 | -167.26 | 0 | 0 |
| 15 | 2024-10-02 06:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 128.75 | -167.26 | 0 | 0 |
| 20 | 2024-10-02 08:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 128.75 | -167.26 | 0 | 0 |
| 17 | 2024-10-02 07:00:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 128.5 | -167.26 | 0 | 0 |
| 1 | 2024-10-01 23:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 128.25 | -167.26 | 0 | 0 |
| 16 | 2024-10-02 06:30:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 128.25 | -167.26 | 0 | 0 |
| 18 | 2024-10-02 07:30:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 128.25 | -167.26 | 0 | 0 |

### Family 5: TKNWW (Triton Knoll)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `8580.500`
- Region / Cluster: `England/Wales` / `Humber Offshore`
- Mapping status: `mapped`
- BMUs: `T_TKNWW-1`
- Zero-BOALF rows: `48`
- Negative-bid rows: `48`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 2024-10-02 05:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203.5 | -77.8 | 0 | 0 |
| 22 | 2024-10-02 09:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203.5 | -94.49 | 0 | 0 |
| 6 | 2024-10-02 01:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203 | -105.43 | 0 | 0 |
| 15 | 2024-10-02 06:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203 | -65.39 | 0 | 0 |
| 16 | 2024-10-02 06:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203 | -65.39 | 0 | 0 |
| 18 | 2024-10-02 07:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203 | -76.3 | 0 | 0 |
| 20 | 2024-10-02 08:30:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 203 | -82.64 | 0 | 0 |
| 19 | 2024-10-02 08:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 202.5 | -82.64 | 0 | 0 |
| 21 | 2024-10-02 09:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 202.5 | -94.49 | 0 | 0 |
| 23 | 2024-10-02 10:00:00+00:00 | T_TKNWW-1 | physical_without_boalf_negative_bid | 202.5 | -100.93 | 0 | 0 |

### Family 7: TKNEW (Triton Knoll)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `8353.000`
- Region / Cluster: `England/Wales` / `Humber Offshore`
- Mapping status: `mapped`
- BMUs: `T_TKNEW-1`
- Zero-BOALF rows: `48`
- Negative-bid rows: `48`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 2024-10-02 06:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 201.5 | -65.39 | 0 | 0 |
| 18 | 2024-10-02 07:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 201.5 | -76.3 | 0 | 0 |
| 19 | 2024-10-02 08:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 201.5 | -82.64 | 0 | 0 |
| 17 | 2024-10-02 07:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 201 | -76.3 | 0 | 0 |
| 15 | 2024-10-02 06:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 197.5 | -65.39 | 0 | 0 |
| 14 | 2024-10-02 05:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 197 | -77.8 | 0 | 0 |
| 20 | 2024-10-02 08:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 196.5 | -82.64 | 0 | 0 |
| 22 | 2024-10-02 09:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 196.5 | -94.49 | 0 | 0 |
| 6 | 2024-10-02 01:30:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 196 | -105.43 | 0 | 0 |
| 7 | 2024-10-02 02:00:00+00:00 | T_TKNEW-1 | physical_without_boalf_negative_bid | 196 | -106.88 | 0 | 0 |

## Day 4: 2024-10-05

- QA status: `fail`
- Recoverability state: `source_limited`
- Dominant anomaly: `negative_bid_without_boalf`
- Anomaly MWh: `178434.075`
- Remaining QA shortfall MWh: `8157.897`
- Recommended support action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`
- Publication next action: `support_query_missing_published_boalf`
- Selected families: `5`

### Family 2: HOWAO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `13366.750`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWAO-1, T_HOWAO-2, T_HOWAO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 2024-10-05 19:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 177.75 | -169.81 | 0 | 0 |
| 42 | 2024-10-05 19:30:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 177.5 | -169.81 | 0 | 0 |
| 42 | 2024-10-05 19:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 177.5 | -169.81 | 0 | 0 |
| 40 | 2024-10-05 18:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 176.75 | -149.01 | 0 | 0 |
| 41 | 2024-10-05 19:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 176.75 | -169.81 | 0 | 0 |
| 38 | 2024-10-05 17:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 176.25 | -127.79 | 0 | 0 |
| 40 | 2024-10-05 18:30:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 176.25 | -149.01 | 0 | 0 |
| 40 | 2024-10-05 18:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 176.25 | -149.01 | 0 | 0 |
| 41 | 2024-10-05 19:00:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 176.25 | -169.81 | 0 | 0 |
| 41 | 2024-10-05 19:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 176.25 | -169.81 | 0 | 0 |

### Family 3: HOWBO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `13291.500`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWBO-1, T_HOWBO-2, T_HOWBO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 2024-10-05 18:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 188.25 | -36.77 | 0 | 0 |
| 41 | 2024-10-05 19:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 188 | -35.2 | 0 | 0 |
| 38 | 2024-10-05 17:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 187.75 | -52.82 | 0 | 0 |
| 39 | 2024-10-05 18:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 187.75 | -36.77 | 0 | 0 |
| 40 | 2024-10-05 18:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 187 | -36.77 | 0 | 0 |
| 41 | 2024-10-05 19:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 186.75 | -35.2 | 0 | 0 |
| 42 | 2024-10-05 19:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 186.75 | -35.2 | 0 | 0 |
| 38 | 2024-10-05 17:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 186.25 | -52.82 | 0 | 0 |
| 39 | 2024-10-05 18:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 186.25 | -36.77 | 0 | 0 |
| 42 | 2024-10-05 19:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 185.5 | -35.2 | 0 | 0 |

### Family 4: MOWEO (Moray East)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `8324.000`
- Region / Cluster: `Scotland` / `Moray Firth Offshore`
- Mapping status: `mapped`
- BMUs: `T_MOWEO-1, T_MOWEO-2, T_MOWEO-3`
- Zero-BOALF rows: `122`
- Negative-bid rows: `122`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 2024-10-05 19:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 126 | -49.5 | 0 | 0 |
| 43 | 2024-10-05 20:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 123.5 | -52.9 | 0 | 0 |
| 44 | 2024-10-05 20:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 119 | -52.9 | 0 | 0 |
| 41 | 2024-10-05 19:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 116 | -49.5 | 0 | 0 |
| 42 | 2024-10-05 19:30:00+00:00 | T_MOWEO-1 | physical_without_boalf_negative_bid | 114.5 | -49.5 | 0 | 0 |
| 45 | 2024-10-05 21:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 113.5 | -59.89 | 0 | 0 |
| 46 | 2024-10-05 21:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 113 | -59.89 | 0 | 0 |
| 39 | 2024-10-05 18:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 112 | -31.33 | 0 | 0 |
| 47 | 2024-10-05 22:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 111.5 | -77.71 | 0 | 0 |
| 38 | 2024-10-05 17:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 111 | -12.82 | 0 | 0 |

### Family 6: SGRWO (Seagreen)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `6389.350`
- Region / Cluster: `Scotland` / `East Coast Scotland Offshore`
- Mapping status: `mapped`
- BMUs: `T_SGRWO-1, T_SGRWO-2, T_SGRWO-3, T_SGRWO-4, T_SGRWO-5, T_SGRWO-6`
- Zero-BOALF rows: `267`
- Negative-bid rows: `267`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 39 | 2024-10-05 18:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 40 | 2024-10-05 18:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 41 | 2024-10-05 19:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 42 | 2024-10-05 19:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 43 | 2024-10-05 20:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 44 | 2024-10-05 20:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 46 | 2024-10-05 21:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 47 | 2024-10-05 22:00:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 48 | 2024-10-05 22:30:00+00:00 | T_SGRWO-6 | physical_without_boalf_negative_bid | 179 | -19.67 | 0 | 0 |
| 38 | 2024-10-05 17:30:00+00:00 | T_SGRWO-1 | physical_without_boalf_negative_bid | 141 | -19.67 | 0 | 0 |

### Family 7: GYMR (GYMR)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `5767.500`
- Region / Cluster: `England/Wales` / `North Wales Offshore`
- Mapping status: `mapped`
- BMUs: `T_GYMR-15, T_GYMR-17, T_GYMR-26, T_GYMR-28`
- Zero-BOALF rows: `192`
- Negative-bid rows: `192`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 41 | 2024-10-05 19:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 64.5 | -181.49 | 0 | 0 |
| 44 | 2024-10-05 20:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 64 | -181.49 | 0 | 0 |
| 43 | 2024-10-05 20:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 63 | -181.49 | 0 | 0 |
| 40 | 2024-10-05 18:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 62 | -181.49 | 0 | 0 |
| 42 | 2024-10-05 19:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 61.5 | -181.49 | 0 | 0 |
| 45 | 2024-10-05 21:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 60 | -181.49 | 0 | 0 |
| 46 | 2024-10-05 21:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 59 | -181.49 | 0 | 0 |
| 47 | 2024-10-05 22:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 58 | -181.49 | 0 | 0 |
| 39 | 2024-10-05 18:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 57.5 | -181.49 | 0 | 0 |
| 48 | 2024-10-05 22:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 56.5 | -181.49 | 0 | 0 |

## Day 5: 2024-10-07

- QA status: `fail`
- Recoverability state: `source_limited`
- Dominant anomaly: `negative_bid_without_boalf`
- Anomaly MWh: `134799.658`
- Remaining QA shortfall MWh: `9804.440`
- Recommended support action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`
- Publication next action: `support_query_missing_published_boalf`
- Selected families: `5`

### Family 1: HOWAO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `18129.250`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWAO-1, T_HOWAO-2, T_HOWAO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 2024-10-07 04:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 187.75 | -175.32 | 0 | 0 |
| 12 | 2024-10-07 04:30:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 187.25 | -175.32 | 0 | 0 |
| 13 | 2024-10-07 05:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 186.25 | -150.56 | 0 | 0 |
| 11 | 2024-10-07 04:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 185.75 | -175.32 | 0 | 0 |
| 13 | 2024-10-07 05:00:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 185.75 | -150.56 | 0 | 0 |
| 11 | 2024-10-07 04:00:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 185.5 | -175.32 | 0 | 0 |
| 12 | 2024-10-07 04:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 182.25 | -175.32 | 0 | 0 |
| 10 | 2024-10-07 03:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 181.5 | -185.09 | 0 | 0 |
| 10 | 2024-10-07 03:30:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 181.25 | -185.09 | 0 | 0 |
| 14 | 2024-10-07 05:30:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 181.25 | -150.56 | 0 | 0 |

### Family 2: HOWBO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `16980.750`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWBO-1, T_HOWBO-2, T_HOWBO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 2024-10-07 04:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 200.75 | -131.03 | 0 | 0 |
| 12 | 2024-10-07 04:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 200.75 | -131.03 | 0 | 0 |
| 11 | 2024-10-07 04:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 197.5 | -131.03 | 0 | 0 |
| 11 | 2024-10-07 04:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 197.5 | -131.03 | 0 | 0 |
| 13 | 2024-10-07 05:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 194.5 | -131.03 | 0 | 0 |
| 13 | 2024-10-07 05:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 194.5 | -131.03 | 0 | 0 |
| 10 | 2024-10-07 03:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 193.5 | -131.03 | 0 | 0 |
| 10 | 2024-10-07 03:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 193.5 | -131.03 | 0 | 0 |
| 9 | 2024-10-07 03:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 188.75 | -131.03 | 0 | 0 |
| 9 | 2024-10-07 03:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 188.75 | -131.03 | 0 | 0 |

### Family 3: MOWEO (Moray East)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `12243.000`
- Region / Cluster: `Scotland` / `Moray Firth Offshore`
- Mapping status: `mapped`
- BMUs: `T_MOWEO-1, T_MOWEO-2, T_MOWEO-3`
- Zero-BOALF rows: `141`
- Negative-bid rows: `125`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2024-10-07 01:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 131 | -58.55 | 0 | 0 |
| 3 | 2024-10-07 00:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 128 | -54.63 | 0 | 0 |
| 4 | 2024-10-07 00:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 127 | -54.63 | 0 | 0 |
| 7 | 2024-10-07 02:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 127 | -61.24 | 0 | 0 |
| 2 | 2024-10-06 23:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 126.5 | -57.46 | 0 | 0 |
| 6 | 2024-10-07 01:30:00+00:00 | T_MOWEO-2 | physical_without_boalf_negative_bid | 125 | -58.55 | 0 | 0 |
| 8 | 2024-10-07 02:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 125 | -61.24 | 0 | 0 |
| 5 | 2024-10-07 01:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 123.5 | -58.55 | 0 | 0 |
| 9 | 2024-10-07 03:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 121.5 | -62.69 | 0 | 0 |
| 10 | 2024-10-07 03:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 121.5 | -62.69 | 0 | 0 |

### Family 4: MOWWO (Moray West)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `8991.500`
- Region / Cluster: `Scotland` / `Moray Firth Offshore`
- Mapping status: `mapped`
- BMUs: `T_MOWWO-1, T_MOWWO-2, T_MOWWO-3, T_MOWWO-4`
- Zero-BOALF rows: `192`
- Negative-bid rows: `192`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 2024-10-07 04:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 97 | -999 | 0 | 0 |
| 9 | 2024-10-07 03:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 96.5 | -999 | 0 | 0 |
| 10 | 2024-10-07 03:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 96 | -999 | 0 | 0 |
| 8 | 2024-10-07 02:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 95.5 | -999 | 0 | 0 |
| 3 | 2024-10-07 00:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 92.5 | -999 | 0 | 0 |
| 5 | 2024-10-07 01:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 92.5 | -999 | 0 | 0 |
| 17 | 2024-10-07 07:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 92.5 | -999 | 0 | 0 |
| 16 | 2024-10-07 06:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 92 | -999 | 0 | 0 |
| 4 | 2024-10-07 00:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 91.5 | -999 | 0 | 0 |
| 13 | 2024-10-07 05:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 91 | -999 | 0 | 0 |

### Family 5: EAAO (East Anglia)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `7404.500`
- Region / Cluster: `England/Wales` / `East Anglia Offshore`
- Mapping status: `mapped`
- BMUs: `T_EAAO-2`
- Zero-BOALF rows: `48`
- Negative-bid rows: `48`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2024-10-06 23:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -152.55 | 0 | 0 |
| 2 | 2024-10-06 23:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -152.55 | 0 | 0 |
| 3 | 2024-10-07 00:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -151.17 | 0 | 0 |
| 4 | 2024-10-07 00:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -151.17 | 0 | 0 |
| 5 | 2024-10-07 01:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -156.43 | 0 | 0 |
| 6 | 2024-10-07 01:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -156.43 | 0 | 0 |
| 7 | 2024-10-07 02:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -158.71 | 0 | 0 |
| 8 | 2024-10-07 02:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -158.71 | 0 | 0 |
| 9 | 2024-10-07 03:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -160.33 | 0 | 0 |
| 10 | 2024-10-07 03:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -160.33 | 0 | 0 |

## Day 6: 2024-10-04

- QA status: `warn`
- Recoverability state: `partially_recovered`
- Dominant anomaly: `negative_bid_without_boalf`
- Anomaly MWh: `131441.833`
- Remaining QA shortfall MWh: `1953.873`
- Recommended support action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`
- Publication next action: `support_query_missing_published_boalf`
- Selected families: `5`

### Family 2: MOWEO (Moray East)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `12000.000`
- Region / Cluster: `Scotland` / `Moray Firth Offshore`
- Mapping status: `mapped`
- BMUs: `T_MOWEO-1, T_MOWEO-2, T_MOWEO-3`
- Zero-BOALF rows: `128`
- Negative-bid rows: `128`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2024-10-04 03:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 131.5 | -56.18 | 0 | 0 |
| 4 | 2024-10-04 00:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 126.5 | -56.28 | 0 | 0 |
| 6 | 2024-10-04 01:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 126.5 | -54.84 | 0 | 0 |
| 5 | 2024-10-04 01:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 125.5 | -54.84 | 0 | 0 |
| 7 | 2024-10-04 02:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 125.5 | -62.34 | 0 | 0 |
| 8 | 2024-10-04 02:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 125.5 | -62.34 | 0 | 0 |
| 9 | 2024-10-04 03:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 125.5 | -56.18 | 0 | 0 |
| 18 | 2024-10-04 07:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 123 | -24.17 | 0 | 0 |
| 19 | 2024-10-04 08:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 123 | -39.62 | 0 | 0 |
| 17 | 2024-10-04 07:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 122.5 | -24.17 | 0 | 0 |

### Family 3: MOWWO (Moray West)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `8381.500`
- Region / Cluster: `Scotland` / `Moray Firth Offshore`
- Mapping status: `mapped`
- BMUs: `T_MOWWO-1, T_MOWWO-2, T_MOWWO-3, T_MOWWO-4`
- Zero-BOALF rows: `146`
- Negative-bid rows: `135`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `5`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 2024-10-04 00:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 101.5 | -999 | 0 | 0 |
| 2 | 2024-10-03 23:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 101 | -999 | 0 | 0 |
| 5 | 2024-10-04 01:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 99.5 | -999 | 0 | 0 |
| 7 | 2024-10-04 02:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 99.5 | -999 | 0 | 0 |
| 9 | 2024-10-04 03:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 99.5 | -999 | 0 | 0 |
| 6 | 2024-10-04 01:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 99 | -999 | 0 | 0 |
| 4 | 2024-10-04 00:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 98.5 | -999 | 0 | 0 |
| 8 | 2024-10-04 02:30:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 98.5 | -999 | 0 | 0 |
| 19 | 2024-10-04 08:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 98 | -999 | 0 | 0 |
| 11 | 2024-10-04 04:00:00+00:00 | T_MOWWO-1 | physical_without_boalf_negative_bid | 97.5 | -999 | 0 | 0 |

### Family 7: HOWBO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `3717.250`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWBO-1, T_HOWBO-2, T_HOWBO-3`
- Zero-BOALF rows: `108`
- Negative-bid rows: `108`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | 2024-10-04 20:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 91 | -40.14 | 0 | 0 |
| 43 | 2024-10-04 20:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 91 | -40.14 | 0 | 0 |
| 44 | 2024-10-04 20:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 89.25 | -40.14 | 0 | 0 |
| 44 | 2024-10-04 20:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 89.25 | -40.14 | 0 | 0 |
| 45 | 2024-10-04 21:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 84.5 | -56.86 | 0 | 0 |
| 45 | 2024-10-04 21:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 84.5 | -56.86 | 0 | 0 |
| 42 | 2024-10-04 19:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 83.25 | -35.2 | 0 | 0 |
| 42 | 2024-10-04 19:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 83.25 | -35.2 | 0 | 0 |
| 43 | 2024-10-04 20:00:00+00:00 | T_HOWBO-3 | physical_without_boalf_negative_bid | 80 | -9999 | 0 | 0 |
| 44 | 2024-10-04 20:30:00+00:00 | T_HOWBO-3 | physical_without_boalf_negative_bid | 78.5 | -9999 | 0 | 0 |

### Family 8: HOWAO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `3494.750`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWAO-1, T_HOWAO-2, T_HOWAO-3`
- Zero-BOALF rows: `105`
- Negative-bid rows: `105`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | 2024-10-04 20:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 82.75 | -163.09 | 0 | 0 |
| 43 | 2024-10-04 20:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 82.5 | -163.09 | 0 | 0 |
| 44 | 2024-10-04 20:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 81.25 | -163.09 | 0 | 0 |
| 44 | 2024-10-04 20:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 81.25 | -163.09 | 0 | 0 |
| 43 | 2024-10-04 20:00:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 81 | -163.09 | 0 | 0 |
| 44 | 2024-10-04 20:30:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 79.75 | -163.09 | 0 | 0 |
| 42 | 2024-10-04 19:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 77 | -161.05 | 0 | 0 |
| 45 | 2024-10-04 21:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 77 | -169.31 | 0 | 0 |
| 45 | 2024-10-04 21:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 77 | -169.31 | 0 | 0 |
| 42 | 2024-10-04 19:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 76.5 | -161.05 | 0 | 0 |

### Family 9: GYMR (GYMR)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `3349.000`
- Region / Cluster: `England/Wales` / `North Wales Offshore`
- Mapping status: `mapped`
- BMUs: `T_GYMR-15, T_GYMR-17, T_GYMR-26, T_GYMR-28`
- Zero-BOALF rows: `192`
- Negative-bid rows: `192`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 23 | 2024-10-04 10:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 31.5 | -181.49 | 0 | 0 |
| 24 | 2024-10-04 10:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 31 | -181.49 | 0 | 0 |
| 22 | 2024-10-04 09:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 30.5 | -181.49 | 0 | 0 |
| 25 | 2024-10-04 11:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 30 | -181.49 | 0 | 0 |
| 26 | 2024-10-04 11:30:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 28.5 | -181.49 | 0 | 0 |
| 46 | 2024-10-04 21:30:00+00:00 | T_GYMR-17 | physical_without_boalf_negative_bid | 28 | -181.49 | 0 | 0 |
| 24 | 2024-10-04 10:30:00+00:00 | T_GYMR-17 | physical_without_boalf_negative_bid | 27.5 | -181.49 | 0 | 0 |
| 25 | 2024-10-04 11:00:00+00:00 | T_GYMR-17 | physical_without_boalf_negative_bid | 27.5 | -181.49 | 0 | 0 |
| 27 | 2024-10-04 12:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 27.5 | -181.49 | 0 | 0 |
| 41 | 2024-10-04 19:00:00+00:00 | T_GYMR-15 | physical_without_boalf_negative_bid | 27.5 | -181.49 | 0 | 0 |

## Day 7: 2024-10-03

- QA status: `fail`
- Recoverability state: `source_limited`
- Dominant anomaly: `negative_bid_without_boalf`
- Anomaly MWh: `81943.683`
- Remaining QA shortfall MWh: `5672.541`
- Recommended support action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`
- Publication next action: `support_query_missing_published_boalf`
- Selected families: `5`

### Family 1: HOWAO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `6193.250`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWAO-1, T_HOWAO-2, T_HOWAO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2024-10-02 23:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 103.75 | -174.25 | 0 | 0 |
| 1 | 2024-10-02 23:00:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 102.25 | -174.25 | 0 | 0 |
| 1 | 2024-10-02 23:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 102.25 | -174.25 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 96.25 | -174.25 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 94.5 | -174.25 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 94.5 | -174.25 | 0 | 0 |
| 3 | 2024-10-03 00:00:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 85 | -172.97 | 0 | 0 |
| 3 | 2024-10-03 00:00:00+00:00 | T_HOWAO-1 | physical_without_boalf_negative_bid | 83.5 | -172.97 | 0 | 0 |
| 3 | 2024-10-03 00:00:00+00:00 | T_HOWAO-3 | physical_without_boalf_negative_bid | 83.5 | -172.97 | 0 | 0 |
| 16 | 2024-10-03 06:30:00+00:00 | T_HOWAO-2 | physical_without_boalf_negative_bid | 76.5 | -136.11 | 0 | 0 |

### Family 2: HOWBO (Hornsea)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `6128.750`
- Region / Cluster: `England/Wales` / `Dogger and Hornsea Offshore`
- Mapping status: `mapped`
- BMUs: `T_HOWBO-1, T_HOWBO-2, T_HOWBO-3`
- Zero-BOALF rows: `144`
- Negative-bid rows: `144`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2024-10-02 23:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 104.5 | -129.5 | 0 | 0 |
| 1 | 2024-10-02 23:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 103.5 | -129.5 | 0 | 0 |
| 11 | 2024-10-03 04:00:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 97.5 | -131.03 | 0 | 0 |
| 11 | 2024-10-03 04:00:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 96.25 | -131.03 | 0 | 0 |
| 12 | 2024-10-03 04:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 90.5 | -131.03 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 89.75 | -129.5 | 0 | 0 |
| 10 | 2024-10-03 03:30:00+00:00 | T_HOWBO-2 | physical_without_boalf_negative_bid | 89.25 | -131.03 | 0 | 0 |
| 12 | 2024-10-03 04:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 89.25 | -131.03 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 88.5 | -129.5 | 0 | 0 |
| 10 | 2024-10-03 03:30:00+00:00 | T_HOWBO-1 | physical_without_boalf_negative_bid | 88 | -131.03 | 0 | 0 |

### Family 3: EAAO (East Anglia)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `5742.500`
- Region / Cluster: `England/Wales` / `East Anglia Offshore`
- Mapping status: `mapped`
- BMUs: `T_EAAO-2`
- Zero-BOALF rows: `48`
- Negative-bid rows: `48`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2024-10-02 23:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -150.83 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -150.83 | 0 | 0 |
| 3 | 2024-10-03 00:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -149.69 | 0 | 0 |
| 4 | 2024-10-03 00:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -149.69 | 0 | 0 |
| 7 | 2024-10-03 02:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -149.98 | 0 | 0 |
| 8 | 2024-10-03 02:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -149.98 | 0 | 0 |
| 9 | 2024-10-03 03:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -147.85 | 0 | 0 |
| 10 | 2024-10-03 03:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 170 | -147.85 | 0 | 0 |
| 12 | 2024-10-03 04:30:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 169.5 | -143.44 | 0 | 0 |
| 11 | 2024-10-03 04:00:00+00:00 | T_EAAO-2 | physical_without_boalf_negative_bid | 167.5 | -143.44 | 0 | 0 |

### Family 8: RCBKO (Race Bank)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `3327.750`
- Region / Cluster: `England/Wales` / `Humber Offshore`
- Mapping status: `mapped`
- BMUs: `T_RCBKO-1, T_RCBKO-2`
- Zero-BOALF rows: `96`
- Negative-bid rows: `96`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2024-10-02 23:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 91.5 | -167.26 | 0 | 0 |
| 1 | 2024-10-02 23:00:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 90.5 | -167.26 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 81 | -167.26 | 0 | 0 |
| 2 | 2024-10-02 23:30:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 80.25 | -167.26 | 0 | 0 |
| 3 | 2024-10-03 00:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 74.5 | -167.26 | 0 | 0 |
| 3 | 2024-10-03 00:00:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 73.75 | -167.26 | 0 | 0 |
| 5 | 2024-10-03 01:00:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 72 | -167.26 | 0 | 0 |
| 5 | 2024-10-03 01:00:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 71.5 | -167.26 | 0 | 0 |
| 4 | 2024-10-03 00:30:00+00:00 | T_RCBKO-1 | physical_without_boalf_negative_bid | 71 | -167.26 | 0 | 0 |
| 4 | 2024-10-03 00:30:00+00:00 | T_RCBKO-2 | physical_without_boalf_negative_bid | 70.25 | -167.26 | 0 | 0 |

### Family 9: MOWEO (Moray East)

- Anomaly type: `negative_bid_without_boalf`
- Anomaly MWh: `2909.000`
- Region / Cluster: `Scotland` / `Moray Firth Offshore`
- Mapping status: `mapped`
- BMUs: `T_MOWEO-1, T_MOWEO-2, T_MOWEO-3`
- Zero-BOALF rows: `131`
- Negative-bid rows: `131`
- Sentinel rows: `0`
- Dynamic-limit-like rows: `0`
- Support question: `query_missing_boalf_with_negative_bid_and_physical_gap`
- Recommended action: `ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf`

Example half-hours:

| settlement_period | interval_start_utc | elexon_bm_unit | publication_audit_state | physical_dispatch_down_gap_mwh | most_negative_bid_gbp_per_mwh | sentinel_pair_count | accepted_down_delta_mwh_lower_bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 46 | 2024-10-03 21:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 126 | -47.64 | 0 | 0 |
| 47 | 2024-10-03 22:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 121.5 | -44.29 | 0 | 0 |
| 48 | 2024-10-03 22:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 119 | -44.29 | 0 | 0 |
| 45 | 2024-10-03 21:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 113 | -47.64 | 0 | 0 |
| 42 | 2024-10-03 19:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 108.5 | -33.77 | 0 | 0 |
| 41 | 2024-10-03 19:00:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 108 | -33.77 | 0 | 0 |
| 40 | 2024-10-03 18:30:00+00:00 | T_MOWEO-3 | physical_without_boalf_negative_bid | 102.5 | -10.82 | 0 | 0 |
| 41 | 2024-10-03 19:00:00+00:00 | T_MOWEO-1 | physical_without_boalf_negative_bid | 87 | -33.77 | 0 | 0 |
| 47 | 2024-10-03 22:00:00+00:00 | T_MOWEO-1 | physical_without_boalf_negative_bid | 87 | -44.29 | 0 | 0 |
| 43 | 2024-10-03 20:00:00+00:00 | T_MOWEO-1 | physical_without_boalf_negative_bid | 86.5 | -40.71 | 0 | 0 |
