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

13. Materialize first-pass border-level interconnector physical flow history from ENTSO-E:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-interconnector-flow ^
     --flow-start 2024-10-01 ^
     --flow-end 2024-10-03 ^
     --flow-output-dir interconnector_flow_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

14. Materialize first-pass border-level interconnector offered-capacity history from ENTSO-E:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-interconnector-capacity ^
     --capacity-start 2024-10-01 ^
     --capacity-end 2024-10-03 ^
     --capacity-output-dir interconnector_capacity_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

15. Materialize a broader official ENTSO-E capacity-source audit before using capacity as a full-border gate:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-interconnector-capacity-audit ^
     --capacity-audit-start 2024-10-01 ^
     --capacity-audit-end 2024-10-03 ^
     --capacity-audit-output-dir interconnector_capacity_audit
   ```

16. Materialize the reviewed-capacity policy so explicit-daily ENTSO-E capacity for GB-NL, GB-BE, and GB-DK1 stays separate from the first-pass gate:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-interconnector-capacity-review-policy ^
     --capacity-review-start 2024-10-01 ^
     --capacity-review-end 2024-10-03 ^
     --capacity-review-output-dir interconnector_capacity_review
   ```

17. Materialize NESO interconnector ITL history for connector-specific reviewed caps on `IFA`, `IFA2`, `BritNed`, and `ElecLink`:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-interconnector-itl ^
     --itl-start 2024-10-01 ^
     --itl-end 2024-10-03 ^
     --itl-output-dir interconnector_itl_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

18. Materialize NESO day-ahead boundary flows and limits as a first-class internal-capacity evidence table:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-day-ahead-constraint-boundary ^
     --boundary-start 2024-10-01 ^
     --boundary-end 2024-10-03 ^
     --boundary-output-dir day_ahead_constraint_boundary_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

19. Materialize the first-pass hourly GB transfer-gate proxy from cluster to interconnector hub:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-gb-transfer-gate ^
     --transfer-start 2024-10-01 ^
     --transfer-end 2024-10-03 ^
     --transfer-output-dir gb_transfer_gate_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

   If you have reviewed public boundary or constraint evidence for internal GB transfer, add:

   ```bash
     --gb-transfer-reviewed-input-path gb_transfer_reviewed_input.csv
   ```

   A checked-in template is available at:

   ```text
   gb_transfer_reviewed_input.example.csv
   ```

   If your source is a messy CSV, TSV, TXT, or PDF-extracted table, normalize it first:

   ```bash
   python inline_arbitrage_live.py ^
     --normalize-gb-transfer-reviewed-input ^
     --gb-transfer-reviewed-raw-path gb_transfer_reviewed_raw.txt ^
     --gb-transfer-reviewed-normalized-output gb_transfer_reviewed_input.csv
   ```

20. Materialize the first-pass cluster-aware route-score history:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-route-score-history ^
     --route-score-start 2024-10-01 ^
     --route-score-end 2024-10-02 ^
     --route-score-output-dir route_score_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

21. Materialize the first curtailment-opportunity history surface:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-curtailment-opportunity-history ^
     --opportunity-start 2024-10-01 ^
     --opportunity-end 2024-10-02 ^
     --opportunity-output-dir curtailment_opportunity_history ^
     --opportunity-truth-profile proxy ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

   This writes `fact_curtailment_opportunity_hourly.csv` plus the supporting `fact_route_score_hourly.csv` and
   `fact_regional_curtailment_hourly_proxy.csv` inputs in the same output directory.

   If you have a reviewed or API-fed upstream market-state input with route-level forward, day-ahead, intraday,
   or imbalance price-state fields, add:

   ```bash
     --market-state-input-path upstream_market_state_input.csv
   ```

   A checked-in template is available at:

   ```text
   upstream_market_state_input.example.csv
   ```

   If your source is a messy CSV, TSV, TXT, or JSON file, normalize it first:

   ```bash
   python inline_arbitrage_live.py ^
     --normalize-upstream-market-state-input ^
     --upstream-market-state-raw-path upstream_market_state_raw.txt ^
     --upstream-market-state-normalized-output upstream_market_state_input.csv
   ```

22. Materialize the canonical upstream market-state feed by itself:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-upstream-market-state-feed ^
     --market-state-start 2024-10-01 ^
     --market-state-end 2024-10-02 ^
     --market-state-input-path upstream_market_state_input.csv ^
     --market-state-output-dir upstream_market_state_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

   This writes `fact_upstream_market_state_hourly.csv` and keeps explicit source lineage so a reviewed manual feed can
   be swapped later for a stronger API feed without changing the opportunity or backtest contracts.

23. Materialize the France-specific connector layer for `IFA`, `IFA2`, and `ElecLink`:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-france-connector-layer ^
     --france-start 2024-10-01 ^
     --france-end 2024-10-02 ^
     --france-output-dir france_connector_history ^
     --truth-store-db-path bmu_truth_store.sqlite
   ```

   If you have a reviewed manual input assembled from ElecLink public documents and JAO notices, add:

   ```bash
     --france-reviewed-input-path france_connector_reviewed_input.csv
   ```

   A checked-in template is available at:

   ```text
   france_connector_reviewed_input.example.csv
   ```

   If your source is a messy CSV, TSV, TXT, or PDF-extracted table, normalize it first:

   ```bash
   python inline_arbitrage_live.py ^
     --normalize-france-reviewed-input ^
     --france-reviewed-raw-path france_connector_reviewed_raw.txt ^
     --france-reviewed-normalized-output france_connector_reviewed_input.csv
   ```

   The normalizer accepts common alias columns such as `delivery date`, `delivery period (GMT)`,
   `capacity limit MW`, `available capacity MW`, `connector`, `cable`, `source key`, and plain-text
   tabular extracts with tab or repeated-space separators.

   If you have a local Nord Pool UMM export for `ElecLink`, add:

   ```bash
     --eleclink-umm-export-path eleclink_umm_export.csv
   ```

   If you have Nord Pool UMM credentials, you can also supply:

   ```bash
     --eleclink-umm-username YOUR_USERNAME ^
     --eleclink-umm-password YOUR_PASSWORD ^
     --eleclink-umm-client-authorization YOUR_CLIENT_AUTHORIZATION_STRING
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
- The repo now materializes the first network-deliverability surface:
  - `fact_interconnector_flow_hourly`
- The repo now also materializes the first commercial-capacity surface:
  - `fact_interconnector_capacity_hourly`
- The repo now also materializes a connector-specific reviewed-cap surface from NESO:
  - `fact_interconnector_itl_hourly`
- `fact_interconnector_itl_hourly` is a connector-level reviewed evidence tier for `IFA`, `IFA2`, `BritNed`, and
  `ElecLink`. It comes from NESO ITL submissions, not from final auction allocation or realized post-auction headroom,
  so it is useful for auditable route caps and blocks but should not be mistaken for full commercial-capacity truth.
- The repo now also materializes a public internal-boundary evidence surface from NESO:
  - `fact_day_ahead_constraint_boundary_half_hourly`
- `fact_day_ahead_constraint_boundary_half_hourly` is the public day-ahead boundary flow-and-limit table normalized to
  half-hour rows with headroom and utilization diagnostics. It now feeds a separate reviewed internal-transfer tightening
  layer for mapped corridors, while staying explicit and auditable rather than silently rewriting `fact_gb_transfer_gate_hourly`.
- The current first pass is real ENTSO-E `A11` border flow, but it is still border-level rather than cable-specific.
  Shared borders like GB-FR are therefore carried as aggregate border truth plus candidate hub sets, not attributed to `IFA`,
  `IFA2`, or `ElecLink` individually.
- `fact_interconnector_capacity_hourly` is a first-pass ENTSO-E offered-capacity layer from article `11.1.A`
  using `documentType=A31`, `auction.Type=A01`, and daily contract type `A01`. It is commercial offered capacity,
  not yet ATC/NTC, outage truth, or post-auction headroom.
- `interconnector_capacity.py` also materializes a broader source-audit surface:
  - `fact_interconnector_capacity_source_audit_daily`
  - `fact_interconnector_capacity_source_audit_variant`
- `interconnector_capacity.py` now also materializes:
  - `fact_interconnector_capacity_review_policy`
  - `fact_interconnector_capacity_reviewed_hourly`
- `fact_interconnector_capacity_review_policy` is the decision surface for alternate official capacity coverage.
  It accepts `a31_explicit_daily` as a reviewed evidence tier for `GB-NL`, `GB-BE`, and `GB-DK1`, but it keeps that
  tier separate from the first-pass direct gate so route scoring does not silently promote it.
- `fact_interconnector_capacity_reviewed_hourly` is the accepted reviewed-capacity surface itself.
  It currently fetches explicit-daily `A31` only for the borders allowed by the review policy, and it stays separate from
  `fact_interconnector_capacity_hourly` so reviewed coverage can be audited explicitly.
- `france_connector.py` now materializes:
  - `dim_interconnector_cable`
  - `fact_france_connector_hourly`
- `france_connector_availability.py` now materializes:
  - `fact_france_connector_operator_event`
  - `fact_france_connector_availability_hourly`
  - `fact_france_connector_operator_source_compare`
- `france_connector_reviewed.py` now materializes:
  - `fact_france_connector_reviewed_period`
- `dim_interconnector_cable` is the first cable-level connector dimension. Current scope is France-facing cables only:
  `IFA`, `IFA2`, and `ElecLink`.
- `fact_france_connector_hourly` decomposes the shared `GB-FR` border into cable rows using nominal-capacity shares,
  border flow, and any published or reviewed border-capacity overlays. It is explicitly a cable proxy layer, not
  cable-level operational truth.
- `fact_france_connector_operator_event` is the event-level source-truth surface for France connector outages. Today it
  uses Elexon REMIT for `IFA` and `IFA2`, and supports both an authenticated Nord Pool UMM path and an optional manual
  export path for `ElecLink`.
- `fact_france_connector_availability_hourly` is the hourly operator-availability layer. It can block or cap `IFA` and
  `IFA2` directly from REMIT, and it keeps `ElecLink` explicitly at `unknown_source` unless either an authenticated
  Nord Pool UMM session or a reviewed manual export is selected for the requested window.
- `fact_france_connector_operator_source_compare` is the ElecLink source-selection surface. It compares the authenticated
  Nord Pool path and the manual export path, then records which source was selected for the requested window and why.
- `fact_france_connector_reviewed_period` is a separate reviewed-evidence tier built from manual inputs normalized from
  ElecLink public documents and JAO notices. It is intentionally kept separate from operator truth so the repo can use
  reviewed public period caps now, and later swap in better API/operator sources without rewriting the route scorer.
- `fact_france_connector_notice_hourly` is the as-of publication-time feature layer derived from the same reviewed
  public inputs. It keeps notice state, lead time, publication timestamp, and revision count separate from live gating
  so future backtests can model what the market already knew without leaking later document revisions into earlier hours.
- The France source stack is now explicit:
  - operator/API truth first when available (`fact_france_connector_availability_hourly`)
  - reviewed public period inputs second (`fact_france_connector_reviewed_period`)
  - border reviewed-capacity tiers third when applicable
  - nominal connector proxies last
- `--france-reviewed-input-path` expects a reviewed CSV or JSON with normalized period rows. The minimum practical fields are:
  - `connector_key`
  - `source_key`
  - `period_start_utc` and `period_end_utc`, or `start_date` and `end_date`
  - one of `capacity_limit_mw`, `available_capacity_mw`, or an explicit `reviewed_publication_state`
- `source_published_utc` is strongly recommended. If it is missing, the normalizer falls back to `source_published_date`
  at midnight UTC, which is acceptable for coarse historical replay but weak for any notice-lead-time feature work.
- A ready-to-fill example is checked in as `france_connector_reviewed_input.example.csv`.
- A one-command normalizer is available for messy raw extracts:
  - `--normalize-france-reviewed-input`
  - `--france-reviewed-raw-path`
  - `--france-reviewed-normalized-output`
- Supported `source_key` values in the current first pass are:
  - `eleclink_planned_outage_programme`
  - `eleclink_capacity_split`
  - `eleclink_ntc_restriction`
  - `jao_ifa_notice`
  - `jao_ifa2_notice`
  - `jao_frgb_notice_generic`
- The live route scorer now joins `fact_interconnector_flow_hourly` and `fact_interconnector_capacity_hourly`
  into the GB border leg only.
  It keeps:
  - confirmed route scores when published capacity and flow imply positive headroom
  - relaxed route scores when price is positive but capacity is unpublished
  - `export_signal_network` with explicit `EXPORT_CONFIRMED`, `EXPORT_CAPACITY_UNKNOWN`, or `HOLD`
- `gb_transfer_gate.py` now materializes:
  - `fact_gb_transfer_gate_hourly`
- `fact_gb_transfer_gate_hourly` is a first-pass hourly proxy for internal GB deliverability from a generation cluster to an
  interconnector hub. It combines static reachability status with observed border flow and offered-capacity overlays, so it is
  useful for screening but is still not a validated internal-network transfer truth layer.
- `gb_transfer_reviewed.py` now materializes:
  - `fact_gb_transfer_reviewed_period`
  - `fact_gb_transfer_review_policy`
  - `fact_gb_transfer_reviewed_hourly`
- This is the internal-capacity reviewed tier above the proxy. It is fed from normalized public boundary or constraint evidence,
  keeps policy acceptance explicit, expands reviewed periods to hourly cluster-hub rows, and lets a stronger future API replace
  the reviewed-input path without changing route or opportunity contracts.
- `gb_transfer_boundary_reviewed.py` now materializes:
  - `fact_gb_transfer_boundary_reviewed_hourly`
- This is the first-pass NESO boundary-derived internal-transfer tightening layer. It maps selected day-ahead boundary rows
  onto explicit cluster-to-hub corridors, aggregates them conservatively to hourly rows, and only emits a reviewed override
  when the public boundary evidence is tighter than the structural proxy or explicitly blocked.
- Current first-pass mapped boundary families are:
  - `FLOWSTH` for east-facing England export corridors
  - `SEIMPPR23` for east-facing England south-east export corridors into `BritNed` and the France-facing hubs
  - `SCOTEX`, `NKILGRMO`, `HARSPNBLY`, `SSE-SP2`, `SSEN-S`, `SSHARN3`, and `GM+SNOW5A` for Scotland-to-south export corridors
- `route_score_history.py` now materializes:
  - `fact_route_score_hourly`
- `fact_route_score_hourly` is the first cluster-aware route-screening surface. It joins route netbacks to
  `fact_gb_transfer_gate_hourly`, `fact_gb_transfer_reviewed_hourly`, `fact_gb_transfer_boundary_reviewed_hourly`,
  first-pass border capacity, the reviewed-capacity tier, `fact_interconnector_itl_hourly`, and the France connector layer,
  then labels each row as
  `confirmed`, `reviewed`, `capacity_unknown`, `blocked_internal_transfer`, `blocked_connector_capacity`, or `no_price_signal`.
- Route rows now keep explicit internal-transfer lineage:
  - `internal_transfer_evidence_tier`
  - `internal_transfer_gate_state`
  - `internal_transfer_capacity_limit_mw`
  - `internal_transfer_source_provider`
  - `internal_transfer_source_key`
- Accepted reviewed internal evidence overrides the proxy gate for that cluster-hub-hour. If no accepted reviewed evidence
  exists, the scorer falls back explicitly to the proxy tier instead of silently relabeling it. When both manual reviewed
  internal evidence and the NESO boundary-derived reviewed layer exist, the scorer takes the tighter accepted reviewed row.
- France route rows now carry cable-specific connector metadata and headroom proxies, so `GB-FR` is no longer treated
  as one undifferentiated border inside `fact_route_score_hourly`.
- France route rows now also carry operator-availability fields, so `IFA` and `IFA2` can be capped or blocked by
  REMIT-backed connector outages before the route is scored.
- France route rows now also carry reviewed-publication fields, so `GB-FR` can move from `capacity_unknown` to an
  auditable reviewed tier when a connector-specific public period cap exists even though border capacity is still unpublished.
- France route rows now also carry connector notice fields from `fact_france_connector_notice_hourly`, including
  whether a reviewed restriction was already known, whether it was upcoming or active in that hour, how long remained
  until start, and how long since the document was published.
- Route rows now also carry connector-level ITL lineage, including:
  - `connector_itl_state`
  - `connector_itl_capacity_limit_mw`
  - `connector_itl_auction_type`
  - `connector_itl_restriction_reason`
  - `connector_itl_source_key`
- ITL can now act as an auditable reviewed connector cap or block when it is tighter than the border-level capacity
  evidence, which is especially useful on `GB-FR` and `GB-NL` when connector-specific public evidence is stronger than
  aggregate border coverage.
- `curtailment_opportunity.py` now materializes:
  - `fact_curtailment_opportunity_hourly`
- `fact_curtailment_opportunity_hourly` is the first actual curtailment-opportunity surface. It joins
  `fact_route_score_hourly` to hourly cluster curtailment magnitude, keeps the curtailment source tier explicit
  (`regional_proxy` first pass, with optional BMU-truth override), and carries the France connector notice timing fields
  into a model-ready opportunity table.
- It now also preserves the internal transfer evidence tier and gate state, so opportunity rows distinguish reviewed internal
  restrictions from proxy-only internal deliverability.
- `opportunity_backtest.py` now materializes:
  - `fact_backtest_prediction_hourly`
- `opportunity_backtest.py` also now materializes:
  - `fact_backtest_summary_slice`
  - `fact_backtest_top_error_hourly`
  - `fact_drift_window`
- `fact_backtest_prediction_hourly` is the first backtest audit trail over the opportunity surface. The current model is
  a non-leaky walk-forward audit surface over the opportunity layer. It now supports:
  - `opportunity_group_mean_notice_v1`
  - `opportunity_potential_ratio_v2`
- The backtest is now horizonized. Each prediction row carries:
  - `forecast_horizon_hours`
  - `forecast_origin_utc`
  - `feature_asof_utc`
  - as-of feature columns such as route tier, connector-notice state, curtailment magnitude, route score, and connector-tightness flags
- The horizon logic is forecast-safe by construction: each target hour is joined to its own origin-hour feature row, and
  each model only trains on earlier forecast origins for the same horizon.
- The backtest table keeps:
  - prediction basis (`exact_notice_hour`, `cluster_route_state`, `route_state`, `global`)
  - and for `v2`, calibrated ratio bases (`ratio_exact_notice_hour`, `ratio_route_notice_state`, `ratio_route_delivery_tier`, `ratio_global`) plus targeted market-state and transition-aware bases for `R2_netback_GB_NL_DE_PL`
  - training sample count
  - actuals, predictions, residuals, and absolute errors for both deliverable MWh and gross value
  - explicit `model_key` and `split_strategy`
- The horizonized `v2` backtest now preserves explicit as-of market and transition features for:
  - raw `route_price_score_eur_per_mwh`
  - route-price feasibility and bottleneck lineage
  - route-price state, hourly delta bucket, transition state, and persistence bucket
  - `connector_itl_state`
  - `internal_transfer_gate_state`
  - route-delivery, connector-ITL, and internal-gate transition states
  - route-state persistence buckets
  - multi-hour connector and internal gate state paths
- If `fact_upstream_market_state_hourly` is present, `v2` also preserves explicit as-of upstream market-state fields for:
  - forward, day-ahead, intraday, and optional imbalance prices
  - forward-to-day-ahead and day-ahead-to-intraday spread buckets
  - route-level upstream market-state labels plus source lineage
- Those features are there to target one-hour `BritNed / GB-NL` regime flips directly without leaking future route state. When no upstream feed is present, the current market-state layer still falls back to the as-of route score rather than pretending we already have those external curves.
- `fact_backtest_summary_slice` is the first slice-aware QA surface over the backtest. It aggregates error and bias by
  model, forecast horizon, cluster, connector hub, route, delivery tier, internal-transfer tier, internal-transfer gate state,
  connector-notice market state, upstream market state, curtailment source tier, and hour of day.
- The summary slice table now also carries:
  - `error_focus_area`
  - `error_reduction_priority_rank`
- Those fields make the “hard error-reduction loop” explicit for:
  - `reviewed`
  - `capacity_unknown`
  - reviewed versus proxy internal-transfer regimes
  - blocked internal reviewed states
  - connector-restriction states
  - specific GB-FR cable routes via `hub_key`
- `fact_backtest_top_error_hourly` is the ranked forensic surface for the worst eligible backtest hours by deliverable
  and gross-value error. It is now also horizon-aware and tags the same focus regimes directly on the worst hours.
- `fact_drift_window` is the first drift surface derived directly from backtest predictions. The initial implementation
  is now daily and horizon-aware across:
  - `global_daily`
  - `route_daily`
  - `cluster_daily`
- Each drift row keeps explicit feature-mix, target-shift, and residual-shift scores plus `warmup`, `pass`, and `warn`
  states so the warnings can be tied back to a specific route or cluster instead of only the whole system.
- Drift rows now also expose reviewed-internal-share, proxy-internal-share, and blocked-reviewed-internal share, so internal
  transfer regime shifts show up directly in the same route and cluster drift surface.
- The opportunity layer now distinguishes:
  - capacity is tight now
  - the market already knew a connector restriction was coming
  - no public connector restriction signal
- `--opportunity-truth-profile proxy` keeps the surface fast and reproducible. `research`, `precision`, and `all`
  can optionally pull BMU truth and override the proxy where valid cluster truth exists.
- To backtest an existing opportunity export:
  - `python inline_arbitrage_live.py --materialize-opportunity-backtest --opportunity-input-path curtailment_opportunity_history --backtest-output-dir opportunity_backtest_history`
- The backtest CLI now accepts:
  - `--backtest-model-key all`
  - `--backtest-model-key opportunity_group_mean_notice_v1`
  - `--backtest-model-key opportunity_potential_ratio_v2`
- It also accepts:
  - `--backtest-horizons 1,6,24,168`
- `all` is now the default so both baselines can be compared on the same CLI path and stored in the same backtest tables.
- For `ElecLink`, the current policy is explicit:
  - near-current windows prefer authenticated Nord Pool UMM if credentials are available
  - historical replay windows prefer a manual UMM export when supplied
  - if neither source is usable, `ElecLink` remains `unknown_source`
- The same switch pattern now applies to France reviewed evidence more broadly: manual reviewed public-doc inputs can be
  used immediately, and if a better public or authenticated API ever appears, it can replace the reviewed input path
  without changing the route-scoring contract.
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
- Multi-day REMIT fetch quality is now tracked at settlement-date scope inside
  `fact_bmu_availability_half_hourly` with `remit_fetch_ok`, `remit_detail_url_count`,
  `remit_detail_error_count`, and `remit_first_fetch_error`, so one bad REMIT day no longer
  poisons a whole weekly rerun.
- `interconnector_flow.py` now materializes:
  - `fact_interconnector_flow_hourly`
- `fact_interconnector_flow_hourly` is a first-pass border-level physical flow surface from ENTSO-E `A11`.
  It carries GB-signed direction, observed hourly MW, and candidate hub sets. It is not yet a cable-specific truth layer and it does
  not yet include ATC/NTC, outages, or utilization versus technical capacity.
- `interconnector_capacity.py` now materializes:
  - `fact_interconnector_capacity_hourly`
- `fact_interconnector_capacity_hourly` is a first-pass border-level offered-capacity surface from ENTSO-E article `11.1.A`.
  It carries hourly offered MW by border and direction plus auction and contract metadata. It is not yet cable-specific, outage-aware,
  or equivalent to post-auction available headroom.
- `fact_interconnector_capacity_source_audit_daily` and `fact_interconnector_capacity_source_audit_variant` rank whether the
  official first-pass query publishes any rows for each border and direction, and whether alternate official query variants
  publish anything at all. They are the guardrail for deciding whether missing capacity should remain `capacity_unknown`.
- `fact_interconnector_capacity_review_policy` is the next policy layer above that audit. It records whether alternate
  explicit-daily capacity is acceptable as a reviewed evidence tier, and keeps GB-FR and GB-IE on `keep_capacity_unknown`
  until a better source story exists.
- `fact_france_connector_hourly` is the current workaround for the France gap. It gives the model a cable-specific
  screen for `IFA`, `IFA2`, and `ElecLink` without pretending ENTSO-E border rows are already cable-level truth.
- `fact_france_connector_availability_hourly` is the first place where France connector operator truth enters the stack.
  It already improves `IFA` and `IFA2`; `ElecLink` still needs a stable Nord Pool UMM ingest path.
- `fact_route_score_hourly` is the first place where the reviewed-capacity tier is actually consumed.
  It still does not change the legacy national route CSV path; it is a separate historical screening surface so reviewed
  capacity stays explicit and auditable.
- Partial REMIT windows no longer behave like hard outages by default. When REMIT still reports
  positive available capacity, the availability table now downgrades those rows to `unknown`.
- The truth table keeps both raw and effective availability fields. It can promote a partial-REMIT
  row back to `available` only when REMIT available capacity supports the counterfactual, and it
  marks that with `availability_override_flag` and `availability_override_reason`.

## Next steps

- Replace the seed asset registry with confirmed wind farm, node, and owner metadata
- Decide whether the reviewed explicit-daily tier should be promoted from `fact_route_score_hourly` into the legacy live route path
- Upgrade `fact_curtailment_opportunity_hourly` from proxy-first source selection toward broader cluster truth coverage and stronger model-ready source-policy slices
- Wire `ElecLink` onto a stable Nord Pool UMM or equivalent operator outage feed
- Upgrade `fact_interconnector_capacity_hourly` toward ATC/NTC, outages, and post-allocation headroom
- Add forecast weather history and feature versioning so weather forecast error can be backtested
- Improve dispatch-to-lost-energy capture against the wind-only QA target before relying on the precision profile
- Strengthen the drift window from global-daily into route and cluster slices once the current audit trail is stable
- Add true forecast horizons on top of the opportunity backtest once the same-hour baselines stop moving
- Start with a cluster-point time-slider map, then add hub arcs and error/drift layers
- Add physical flow and ATC checks
- Calibrate fee and capacity costs with auction history
- Add imbalance-risk premia and intraday updates
- Build the interactive dashboard only after the historical and drift surfaces exist
