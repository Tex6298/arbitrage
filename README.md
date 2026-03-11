# Inline Arbitrage (Live) - GB via Elexon, Continent via ENTSO-E

This script fetches GB market-index prices from Elexon MID and continental
day-ahead prices from the ENTSO-E Transparency Platform for FR, NL, DE-LU, PL, CZ,
then computes simple route netbacks for:

- `GB->FR->DE->PL`
- `GB->NL->DE->PL`

Key behavior:

- Uses Elexon MID for GB and real ENTSO-E area EIC codes for the continental zones
- Queries by local market day for each zone, then converts timestamps to UTC
- Converts GB prices from GBP to EUR using a user-supplied FX rate
- Keeps physical-network assumptions in a separate module: `physical_constraints.py`
- Adds first-pass physical scaffolding in `asset_mapping.py` and `gb_topology.py`
- Adds a formal exploration plan in `exploration_plan.py` for historical data, map layers, backtests, and drift checks
- Adds `curtailment_signals.py` to materialize the first three historical tables for backtesting
- Adds `bmu_generation.py` to materialize BMU standing data and first-pass B1610 generation history
- Adds `bmu_dispatch.py` to materialize BOALF dispatch acceptances plus BOD bid-offer evidence for BMU-level dispatch expansion
- Adds `weather_history.py` to materialize observed anchor, cluster, and parent-region weather history
- Adds `bmu_physical.py`, `bmu_availability.py`, and `curtailment_truth.py` to materialize a first-pass BMU lost-energy truth layer
- Scores routes leg-by-leg and blocks a route when any border leg is underwater
- Uses synthetic data only when you ask for it with `--dry`

## Quick start

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then put your API credentials and FX rate in it:

   ```
   ENTOS_E_TOKEN=YOUR_TOKEN_HERE
   BMRS_API_KEY=YOUR_BMRS_API_KEY_HERE
   GBP_EUR=1.17
   ```

   `BMRS_API_KEY` is read exactly by that name if you have an Iris/Elexon key.
   Either `ENTOS_E_TOKEN` or `ENTSOE_TOKEN` works for the ENTSO-E token.
   Either `GBP_EUR` or `GBP_EUR_RATE` works for the FX rate.

3. Run a real pull for one market day:

   ```bash
   python inline_arbitrage_live.py --date 2025-09-20 --save out.csv
   ```

4. Or run synthetic demo data explicitly:

   ```bash
   python inline_arbitrage_live.py --date 2025-09-20 --dry
   ```

5. Print the current physical-network assumption register:

   ```bash
   python inline_arbitrage_live.py --show-constraint-assumptions
   ```

6. Print the current asset and topology scaffolding:

   ```bash
   python inline_arbitrage_live.py --show-asset-mapping
   python inline_arbitrage_live.py --show-gb-topology
   ```

7. Print the historical-data, map, and drift plan:

   ```bash
   python inline_arbitrage_live.py --show-exploration-plan
   ```

8. Materialize the first three historical tables:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-curtailment-history ^
     --history-year 2024-2025 ^
     --history-start 2024-10-01 ^
     --history-end 2024-10-07 ^
     --history-output-dir curtailment_history
   ```

9. Materialize BMU standing data and first-pass B1610 generation history:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-bmu-generation ^
     --bmu-start 2024-10-01 ^
     --bmu-end 2024-10-03 ^
     --bmu-output-dir bmu_history
   ```

10. Materialize BMU dispatch acceptances, bid-offer evidence, and the first half-hour dispatch layer:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-bmu-dispatch ^
     --dispatch-start 2024-10-01 ^
     --dispatch-end 2024-10-03 ^
     --dispatch-output-dir bmu_dispatch_history
   ```

11. Materialize BMU physical positions, availability gates, and the tiered curtailment-truth table:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-bmu-curtailment-truth ^
     --truth-start 2024-10-01 ^
     --truth-end 2024-10-03 ^
     --truth-output-dir bmu_truth_history ^
     --truth-profile all
   ```

   That same run now also writes daily QA tables that explain the reconciliation gap against GB truth.

13. Upsert deduped BMU truth outputs into a SQLite store while materializing:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-bmu-curtailment-truth ^
     --truth-start 2024-10-01 ^
     --truth-end 2024-10-03 ^
     --truth-output-dir bmu_truth_history ^
     --truth-store-db-path bmu_truth_store.sqlite ^
     --truth-profile all
   ```

14. Backfill an existing tree of daily truth CSV outputs into the same SQLite store:

   ```bash
   python inline_arbitrage_live.py ^
     --fill-truth-store-from-dir bmu_truth_history_phase4_family_day_daily ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

15. Materialize and inspect the store-backed source-completeness focus surfaces:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-truth-store-source-focus ^
     --show-truth-store-source-focus ^
     --truth-store-db-path bmu_truth_store.sqlite ^
     --source-focus-status fail_warn ^
     --source-focus-limit 20
    ```

16. Materialize and inspect targeted family dispatch and physical forensics from the SQLite store. By default this scopes to Hornsea family keys `HOWAO,HOWBO`:

    ```bash
    python inline_arbitrage_live.py ^
      --materialize-truth-store-family-forensics ^
      --show-truth-store-family-forensics ^
      --truth-store-db-path bmu_truth_store.sqlite ^
      --forensic-family-keys HOWAO,HOWBO ^
      --forensic-limit 20
    ```

17. Export a support-ready Hornsea publication-audit packet from the SQLite store:

    ```bash
    python inline_arbitrage_live.py ^
      --materialize-truth-store-family-forensics ^
      --truth-store-db-path bmu_truth_store.sqlite ^
      --forensic-family-keys HOWAO,HOWBO ^
      --forensic-output-dir hornsea_support_extract
    ```

18. Materialize a ranked publication-anomaly support batch from the SQLite store and write the CSV evidence tables plus Markdown dossier:

    ```bash
    python inline_arbitrage_live.py ^
      --materialize-truth-store-support-loop ^
      --show-truth-store-support-loop ^
      --truth-store-db-path support_loop_smoke.sqlite ^
      --support-status fail_warn ^
      --support-top-days 7 ^
      --support-top-families-per-day 5 ^
      --support-half-hour-limit 20 ^
      --support-output-dir support_loop_output
    ```

19. Build or review the support-case resolution ledger, and annotate a specific family-day once support feedback or analyst review arrives:

    ```bash
    python inline_arbitrage_live.py ^
      --materialize-truth-store-support-resolution ^
      --show-truth-store-support-resolution ^
      --truth-store-db-path support_loop_smoke.sqlite ^
      --resolution-filter open ^
      --resolution-limit 20
    ```

    ```bash
    python inline_arbitrage_live.py ^
      --annotate-truth-store-support-resolution ^
      --truth-store-db-path support_loop_smoke.sqlite ^
      --resolution-batch-id support_fail_warn_days7_families5_2024-10-01_2024-10-07 ^
      --resolution-date 2024-10-02 ^
      --resolution-family-key HOWBO ^
      --resolution-state confirmed_publication_gap ^
      --resolution-truth-policy-action fix_source_and_rerun ^
      --resolution-note "Elexon support escalation opened for Hornsea publication anomaly." ^
      --resolution-source-reference ticket-2026-03-11-HOWBO
    ```

    The same review surface now also prints:
    - `fact_support_resolution_daily`
    - `fact_support_resolution_batch`

20. Materialize or review the support rerun-gate and unresolved-case priority surfaces:

    ```bash
    python inline_arbitrage_live.py ^
      --materialize-truth-store-support-gate ^
      --show-truth-store-support-gate ^
      --truth-store-db-path support_loop_smoke.sqlite ^
      --resolution-batch-id support_fail_warn_days7_families5_2024-10-01_2024-10-07 ^
      --support-gate-filter all ^
      --support-open-case-limit 20
    ```

    This review surface prints:
    - `fact_support_rerun_gate_batch`
    - `fact_support_rerun_gate_daily`
    - `fact_support_open_case_priority_family_daily`

21. Materialize or review repeated open-case resolution patterns, then bulk-annotate one pattern when the same support-ready family issue repeats across multiple days:

    ```bash
    python inline_arbitrage_live.py ^
      --materialize-truth-store-support-resolution-patterns ^
      --show-truth-store-support-resolution-patterns ^
      --truth-store-db-path support_loop_smoke.sqlite ^
      --resolution-batch-id support_fail_warn_days7_families5_2024-10-01_2024-10-07 ^
      --support-pattern-filter multi_day ^
      --support-pattern-limit 20
    ```

    ```bash
    python inline_arbitrage_live.py ^
      --apply-truth-store-support-resolution-pattern ^
      --truth-store-db-path support_loop_smoke.sqlite ^
      --resolution-batch-id support_fail_warn_days7_families5_2024-10-01_2024-10-07 ^
      --resolution-pattern-key HOWAO::negative_bid_without_boalf::query_missing_boalf_with_negative_bid_and_physical_gap::mapped ^
      --resolution-state confirmed_publication_gap ^
      --resolution-truth-policy-action fix_source_and_rerun ^
      --resolution-note "Bulk-reviewed repeated Hornsea HOWAO publication-gap pattern across open family-days." ^
      --resolution-source-reference bulk-pattern-2026-03-11-HOWAO
    ```

    This review/apply surface uses:
    - `fact_support_resolution_pattern_summary`
    - `fact_support_resolution_pattern_member_family_daily`

21. Materialize or review the rerun-candidate surfaces for days and family-days that are actually ready once their local resolution state allows rerun:

    ```bash
    python inline_arbitrage_live.py ^
      --materialize-truth-store-rerun-candidates ^
      --show-truth-store-rerun-candidates ^
      --truth-store-db-path support_loop_smoke.sqlite ^
      --resolution-batch-id support_fail_warn_days7_families5_2024-10-01_2024-10-07 ^
      --support-rerun-candidate-filter all ^
      --support-rerun-candidate-limit 20
    ```

    This review surface prints:
    - `fact_support_rerun_candidate_daily`
    - `fact_support_rerun_candidate_family_daily`

12. Materialize observed weather history for anchors, clusters, and parent regions:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-weather-history ^
     --weather-start 2024-10-01 ^
     --weather-end 2024-10-03 ^
     --weather-output-dir weather_history
   ```

## Notes

- GB prices come from the Elexon market-index feed and are published in GBP, so an FX rate
  is required to compare them against EUR-denominated continental markets.
- Continental ENTSO-E day-ahead prices are a local market-day product, not a generic rolling UTC feed.
- The GB feed is half-hourly and is normalized to hourly means before route calculation.
- The default GB provider is `APXMIDP`, which is the only provider returning non-zero live
  rows in the sample query I checked on March 10, 2026.
- If Elexon changes authentication behavior, the script will send `BMRS_API_KEY` when present.
- Some continental zones may publish sub-hourly prices. The script normalizes sub-hourly data to
  hourly means before calculating the simple route netbacks.
- Route scores are heuristics based on per-leg spreads, losses, and fees. They are
  stricter than a simple `PL - GB` spread because a negative intermediate leg blocks
  the route for that hour.
- `asset_mapping.py` is a seed registry only. It uses approximate user-provided anchors and
  capacities so the repo has a concrete place to hang cluster, weather, and curtailment work.
- `gb_topology.py` is also a coarse scaffold. It exposes cluster-to-hub reachability statuses,
  not a validated network model, PTDF model, or operational transfer limit.
- `exploration_plan.py` formalizes the next stage: historical datasets, interactive map layers,
  walk-forward backtests, and drift monitors. It is a plan surface, not an implemented UI.
- `curtailment_signals.py` now materializes:
  - `fact_constraint_daily`
  - `fact_wind_split_half_hourly`
  - `fact_regional_curtailment_hourly_proxy`
- `fact_constraint_daily` now keeps the raw NESO total for context and adds a wind-only QA target:
  - `qa_target_definition=wind_constraints_positive_only_v1`
  - `qa_wind_relevant_positive_mwh = max(voltage, 0) + max(thermal, 0)`
- The hourly regional table is explicitly a proxy. It allocates GB-wide daily curtailment by
  regional wind share, then distributes it intraday by wind shape and down to clusters by
  approximate capacity share.
- `bmu_generation.py` now materializes:
  - `dim_bmu_asset`
  - `fact_bmu_generation_half_hourly`
- The BMU dimension is a first pass. It maps known wind BMUs into the current cluster registry
  using explicit name rules, allows `mapping_status=region_only` when the parent region is clear
  but no current cluster should be forced, and leaves everything else as `mapping_status=unmapped`.
- `fact_bmu_generation_half_hourly` is actual generation truth from Elexon B1610, not curtailment truth.
- `bmu_dispatch.py` now materializes:
  - `fact_bmu_acceptance_event`
  - `fact_bmu_dispatch_acceptance_half_hourly`
- `fact_bmu_bid_offer_half_hourly`
- `fact_bmu_dispatch_acceptance_half_hourly` is direct dispatch truth, not lost-energy truth. Its
  `accepted_down_delta_mwh_lower_bound` field is intentionally a lower-bound dispatch metric
  because BOALF gives accepted levels, not the full no-constraint counterfactual.
- `fact_bmu_bid_offer_half_hourly` is not dispatch truth. It is an evidence layer from `BOD`
  that flags negative bid availability and carries price diagnostics so PN-QPN gaps can be used
  as a second dispatch-evidence tier without pretending `BOD` is an acceptance feed.
- Raw `BOD` and `BOALF` fetches are now clipped back to the requested local-day UTC window after
  retrieval, so next-day settlement-period 1 rows no longer leak into a daily pull when the Elexon
  endpoint treats the `to` bound as inclusive.
- `fact_bmu_bid_offer_half_hourly` now carries explicit sentinel diagnostics for suspect source
  values such as `bid <= -9999` or `offer >= 9999`. Those pairs are preserved for audit, but they
  are excluded from `negative_bid_available_flag` and from downstream dispatch inference by default.
- On March 10, 2026, the Elexon v1 endpoint returned data for `BOALF` while the plain `BOAL`
  path returned `404`, so the dispatch materializer is standardized on `BOALF`.
- On the same March 10, 2026 probe, `BOD` returned rows for wind BMUs while `NTO` and `NTB`
  were empty on the sample, so the first dispatch expansion is `BOALF + BOD + PN/QPN`, not
  a notice-to-deliver path.
- `curtailment_truth.py` now materializes:
  - `fact_bmu_bid_offer_half_hourly`
  - `fact_bmu_physical_position_half_hourly`
  - `fact_bmu_availability_half_hourly`
  - `fact_bmu_curtailment_truth_half_hourly`
  - `fact_curtailment_reconciliation_daily`
  - `fact_constraint_target_audit_daily`
  - `fact_dispatch_alignment_daily`
  - `fact_dispatch_alignment_bmu_daily`
  - `fact_curtailment_gap_reason_daily`
  - `fact_bmu_curtailment_gap_bmu_daily`
  - `fact_bmu_family_shortfall_daily`
- `weather_history.py` now materializes:
  - `fact_weather_hourly`
- `fact_weather_hourly` carries observed anchor weather plus capacity-weighted cluster and parent-region aggregates.
- `fact_bmu_curtailment_truth_half_hourly` is tiered, not flattened. It keeps:
  - `dispatch_only` rows when dispatch truth exists but a lost-energy estimate is not valid
  - `physical_baseline` rows when PN or QPN provides a valid half-hour counterfactual
  - `weather_calibrated` rows only when BMU, cluster, or parent-region weather power curves can upgrade an otherwise invalid row
- The truth table now also carries explicit dispatch-source tiers:
  - `acceptance_only` when BOALF alone supports the dispatch row
  - `physical_inference` when a nearby BOALF down-acceptance window plus a negative `BOD` bid and positive `PN-QPN` gap create a dispatch row with no same-half-hour BOALF acceptance
  - `acceptance_plus_physical_inference` when the PN-QPN gap materially exceeds the BOALF lower bound inside that BOALF-triggered window
- The truth table now also carries explicit dispatch-inference scope fields:
  - `dispatch_inference_scope` distinguishes same-BMU BOALF-window inference from the narrower family-day expansion window
  - `family_day_dispatch_increment_mwh_lower_bound` is only applied on mapped BMU-family days that already dominate the provisional family shortfall audit
- The active dispatch quantity for QA and profiles is now `dispatch_down_evidence_mwh_lower_bound`.
  The original BOALF-only field `accepted_down_delta_mwh_lower_bound` is still preserved as raw
  acceptance truth so the expansion remains auditable.
- The truth table now also carries explicit `counterfactual_invalid_reason` and `lost_energy_block_reason` fields so failed capture is diagnosable instead of just silent.
- The three reconciliation QA tables are the main debugging surface for target completeness:
  - `fact_curtailment_reconciliation_daily` shows both raw NESO-total and wind-only QA reconciliation, with BOALF acceptance, same-BMU physical inference, and family-day expansion split out separately
  - `fact_constraint_target_audit_daily` shows whether each day is voltage-dominant, thermal-dominant, or mixed inside the QA target, and classifies the day as source-limited, counterfactual-or-definition-limited, or partially recovered
  - `fact_dispatch_alignment_daily` shows whether blocked dispatch could materially close the QA-target gap, and how much of the dispatch surface is still coming from BOALF versus PN-QPN plus BOD inference
  - `fact_dispatch_alignment_bmu_daily` shows which BMUs are fully estimated, partially blocked, or fully blocked, with blocked lower-bound MWh split by reason
  - `fact_curtailment_gap_reason_daily` breaks each day down by loss-estimate failure reason
  - `fact_bmu_curtailment_gap_bmu_daily` shows which BMUs account for the biggest dispatch-to-lost-energy gap
  - `fact_bmu_family_shortfall_daily` rolls the same gap up to BMU-family and day so the shortfall can be attributed to families like Seagreen, Race Bank, Gunfleet Sands, or Moray rather than only individual BMUs, and now exposes the family-day dispatch increment separately
- `fact_bmu_curtailment_truth_half_hourly` now carries both reconciliation layers:
  - raw-context fields: `gb_daily_raw_constraint_total_mwh`, `raw_reconciliation_*`
  - precision-gate fields: `gb_daily_qa_target_mwh`, `qa_reconciliation_*`
- The BMU truth materializer can now upsert its 11 output tables into a SQLite store using
  `--truth-store-db-path`, and `--fill-truth-store-from-dir` can backfill a tree of daily CSV drops
  into that same deduped store without relying on one giant weekly CSV export.
- The repo now also materializes store-backed prioritization tables:
  - `fact_source_completeness_focus_daily`
  - `fact_source_completeness_focus_family_daily`
- The same store pass now also materializes store-backed dispatch-source gap tables:
  - `fact_dispatch_source_gap_daily`
  - `fact_dispatch_source_gap_family_daily`
- The same store pass now also materializes store-backed publication-anomaly ranking tables:
  - `fact_publication_anomaly_daily`
  - `fact_publication_anomaly_family_daily`
- The repo now also materializes targeted store-backed family forensics tables:
  - `fact_family_dispatch_forensic_daily`
  - `fact_family_dispatch_forensic_bmu_daily`
  - `fact_family_dispatch_forensic_half_hourly`
- The same scoped family-forensics pass now also materializes physical PN/QPN/MILS/MELS surfaces:
  - `fact_family_physical_forensic_daily`
  - `fact_family_physical_forensic_bmu_daily`
  - `fact_family_physical_forensic_half_hourly`
- The same scoped family-forensics pass now also materializes publication-audit and support-extract surfaces:
  - `fact_family_publication_audit_daily`
  - `fact_family_publication_audit_bmu_daily`
  - `fact_family_support_evidence_half_hourly`
- Those focus tables turn the stored QA surfaces into ranked next actions, so the next dispatch-source
  pass can target the remaining fail and warn days directly from SQLite rather than from stitched CSVs.
- The new source-gap tables are narrower: they rank days and BMU families where `negative_bid_available_flag`
  plus `physical_dispatch_down_gap_mwh` indicate missing dispatch evidence, and split that gap into
  `same_bmu_window`, `family_window`, and `no_window` scopes so source expansion is auditable.
- The publication-anomaly tables sit beside the source-gap tables, not inside truth. They rank days and
  BMU families where `physical_dispatch_down_gap_mwh` exists with no published `BOALF`, then split that
  anomaly into `sentinel_bod_present`, `negative_bid_without_boalf`, `dynamic_limit_like_without_boalf`,
  and residual `physical_without_boalf` buckets.
- The family-forensics tables are narrower again: they are a scoped inspection surface keyed by a forensic family
  scope such as `HOWAO+HOWBO`, and they break the chosen families down at daily, BMU-daily, and half-hourly
  grain so a Hornsea-first forensic pass can inspect evidence without auto-promoting those rows into dispatch truth.
- The physical-forensics tables are specifically for questions like “is the Hornsea miss visible in PN/QPN/MILS/MELS
  even when BOALF is empty?” They keep positive `PN-QPN` gaps, zero-BOALF rows, valid negative-bid rows, and sentinel
  bid artifacts separate so source issues do not get hidden inside one aggregate number.
- The publication-audit tables sit one level above that evidence tape: they classify scoped family-days and BMUs as
  `physical_without_boalf`, `physical_without_boalf_negative_bid`, or `availability_like_dynamic_limit`, and they
  attach a support question code instead of silently folding those rows back into truth.
- The support-evidence extract is the support-ready packet. It keeps row-level `PN/QPN/MILS/MELS`, generation,
  BOALF lower bound, BOD sentinel diagnostics, and the recommended support question together in one scoped CSV export.
- The support loop packages the broader store-backed publication-anomaly ranking into:
  - `fact_support_case_daily`
  - `fact_support_case_family_daily`
  - `fact_support_case_half_hourly`
  - `support_case_summary.md`
- The support loop now also keeps a manual resolution ledger:
  - `fact_support_case_resolution`
- The support-resolution workflow now also materializes review summaries:
  - `fact_support_resolution_daily`
  - `fact_support_resolution_batch`
- The support-resolution workflow now also materializes rerun-gate and unresolved-case priority surfaces:
  - `fact_support_rerun_gate_daily`
  - `fact_support_rerun_gate_batch`
  - `fact_support_open_case_priority_family_daily`
- The same support-resolution refresh path now also materializes repeated-pattern helper surfaces:
  - `fact_support_resolution_pattern_summary`
  - `fact_support_resolution_pattern_member_family_daily`
- The same refresh path now also materializes rerun-candidate prep surfaces:
  - `fact_support_rerun_candidate_daily`
  - `fact_support_rerun_candidate_family_daily`
- New support-resolution states are:
  - `open`
  - `confirmed_publication_gap`
  - `confirmed_non_boalf_pattern`
  - `confirmed_source_artifact`
  - `not_reproducible`
- New support rerun-gate states are:
  - batch: `blocked_by_open_cases`, `ready_for_targeted_rerun`, `no_rerun_required`
  - day: `blocked_by_open_cases`, `candidate_targeted_rerun`, `candidate_policy_lock`
- New truth-policy actions are:
  - `keep_out_of_precision`
  - `eligible_for_new_evidence_tier`
  - `fix_source_and_rerun`
  - `close_no_change`
- The rerun gate is mixed-mode by design:
  - batch rows are the hard stop/go surface
  - day rows are advisory so targeted reruns can be prepared without changing truth automatically
- The rerun-candidate tables are narrower than the rerun gate:
  - they only surface resolved family-days with `fix_source_and_rerun` or `eligible_for_new_evidence_tier`
  - they only emit days where the day-level rerun gate is `candidate_targeted_rerun`
- The repeated-pattern helper is narrower than the open-case priority table:
  - it groups only currently open family-days
  - it keys patterns by `bmu_family_key + anomaly state + support question code + mapping status`
  - bulk apply only touches currently open rows inside one selected batch and one selected pattern key
- The support loop is packaging and triage only. It does not change dispatch truth, truth tiers, reconciliation gates,
  or the precision profile.
- `bmu_truth_store.sqlite` in the repo may be a Git LFS pointer in some checkouts rather than a usable SQLite file.
  If so, rebuild a local store first, for example:
  - `python inline_arbitrage_live.py --fill-truth-store-from-dir bmu_truth_history_phase4_family_day --truth-store-db-path support_loop_smoke.sqlite`
- `precision_profile_include` now keys off the wind-only QA target, not the mixed raw NESO total.
- The weather-calibrated tier now uses observed weather history from capacity-weighted anchor points.
  It still does not replace a valid physical-baseline row, and it is still a first pass rather than
  a final turbine-level power model.
- `fact_bmu_availability_half_hourly` uses REMIT as the primary outage gate. If REMIT is missing
  for a run, rows degrade to `availability_state=unknown` instead of silently becoming available.
- Partial REMIT windows no longer behave like hard outages by default. When REMIT still reports
  positive available capacity, the availability table now downgrades those rows to `unknown`.
- The truth table keeps both raw and effective availability fields. It can promote a partial-REMIT
  row back to `available` only when REMIT available capacity supports the counterfactual, and it
  marks that with `availability_override_flag` and `availability_override_reason`.

## Next steps

- Replace the seed asset registry with confirmed wind farm, node, and owner metadata
- Turn the topology scaffold into actual transfer gates between clusters and hubs
- Add forecast weather history and feature versioning so weather forecast error can be backtested
- Improve dispatch-to-lost-energy capture against the wind-only QA target before relying on the precision profile
- Start with a cluster-point time-slider map, then add hub arcs and error/drift layers
- Add physical flow and ATC checks
- Calibrate fee and capacity costs with auction history
- Add imbalance-risk premia and intraday updates
- Build the interactive dashboard only after the historical and drift surfaces exist
