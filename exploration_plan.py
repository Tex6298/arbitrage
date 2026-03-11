from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    grain: str
    spatial_scope: str
    role: str
    source_plan: str
    status: str
    note: str


@dataclass(frozen=True)
class MapLayerSpec:
    key: str
    phase: str
    geometry: str
    time_axis: str
    value_encoding: str
    interaction: str
    purpose: str


@dataclass(frozen=True)
class BacktestSpec:
    key: str
    target: str
    forecast_horizon: str
    split_strategy: str
    primary_metrics: str
    slice_dimensions: str
    acceptance_gate: str


@dataclass(frozen=True)
class DriftMonitorSpec:
    key: str
    monitor_scope: str
    metric: str
    cadence: str
    trigger: str
    action: str


DATASET_SPECS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="dim_cluster",
        grain="static reference",
        spatial_scope="GB cluster",
        role="Canonical cluster registry for all joins and map points.",
        source_plan="asset_mapping.py seed registry, later replaced by curated asset metadata.",
        status="scaffolded",
        note="Cluster key is the spatial spine of the model.",
    ),
    DatasetSpec(
        key="dim_bmu_asset",
        grain="static reference",
        spatial_scope="GB BMU to cluster",
        role="Joins Elexon BMU standing data to cluster and parent-region scaffolding.",
        source_plan="Elexon reference/bmunits/all plus explicit cluster mapping rules.",
        status="implemented_first_pass",
        note="First pass covers mapped BMUs, region-only assignments where a cluster would be forced, and leaves the rest explicitly unmapped.",
    ),
    DatasetSpec(
        key="dim_hub",
        grain="static reference",
        spatial_scope="GB interconnector hub",
        role="Canonical interconnector landing and route-hub registry.",
        source_plan="gb_topology.py hub registry.",
        status="scaffolded",
        note="Needed for route maps, reachability checks, and scenario siting.",
    ),
    DatasetSpec(
        key="fact_interconnector_flow_hourly",
        grain="hourly",
        spatial_scope="Border and interconnector direction",
        role="Observed cross-border physical flow and utilization surface for deliverability checks.",
        source_plan="ENTSO-E or connector-operator actual flow history joined to dim_hub and route legs.",
        status="planned",
        note="This is the first real answer to 'was the cable already busy'; it should carry import or export direction, observed MW, and utilization versus technical limit.",
    ),
    DatasetSpec(
        key="fact_interconnector_capacity_hourly",
        grain="hourly",
        spatial_scope="Border and interconnector direction",
        role="Available cross-border transfer capability, outage, and commercial-capacity surface.",
        source_plan="Connector outage feeds, ATC or NTC series, and auction or nomination metadata where recoverable.",
        status="planned",
        note="This is not the same as observed flow; it should explain whether unused physical headroom was actually tradable or blocked by outages, counterflows, or allocations.",
    ),
    DatasetSpec(
        key="fact_gb_transfer_gate_hourly",
        grain="hourly",
        spatial_scope="Cluster to interconnector hub",
        role="Internal GB deliverability gate from curtailed cluster to landing hub.",
        source_plan="Start with topology status plus constraint-derived heuristics, then upgrade toward zonal or PTDF-style transfer gates.",
        status="planned",
        note="This is the internal-network counterpart to interconnector capacity; without it, a free cable does not mean a northern curtailed cluster can actually reach the border.",
    ),
    DatasetSpec(
        key="fact_market_price_hourly",
        grain="hourly",
        spatial_scope="GB and continental zones",
        role="Price spread and route-score inputs.",
        source_plan="Elexon MID for GB plus ENTSO-E day-ahead for FR/NL/DE/PL/CZ.",
        status="implemented",
        note="Current live script already materializes most of this surface.",
    ),
    DatasetSpec(
        key="fact_bmu_generation_half_hourly",
        grain="half-hourly",
        spatial_scope="Wind BMU",
        role="Actual generation history for BMU-level backtesting and future curtailment truth work.",
        source_plan="Elexon B1610 filtered by mapped and unmapped wind BMUs from dim_bmu_asset.",
        status="implemented_first_pass",
        note="Actual output truth, not curtailment truth; pair this with dispatch acceptances before estimating lost energy.",
    ),
    DatasetSpec(
        key="fact_bmu_acceptance_event",
        grain="acceptance event",
        spatial_scope="Wind BMU",
        role="Raw dispatch-acceptance event history for reconstructing curtailment instructions.",
        source_plan="Elexon BOALF filtered by wind BMUs from dim_bmu_asset.",
        status="implemented_first_pass",
        note="Event-level dispatch truth surface; lower-bound energy metrics only until PN or availability joins exist.",
    ),
    DatasetSpec(
        key="fact_bmu_dispatch_acceptance_half_hourly",
        grain="half-hourly",
        spatial_scope="Wind BMU",
        role="Half-hour dispatch-acceptance truth layer for backtests and future asset-level curtailment truth.",
        source_plan="BOALF acceptance events expanded into settlement intervals and aggregated by BMU.",
        status="implemented_first_pass",
        note="Not proxy data, but the accepted-down MWh field is still a lower-bound dispatch metric rather than lost-energy truth.",
    ),
    DatasetSpec(
        key="fact_bmu_bid_offer_half_hourly",
        grain="half-hourly",
        spatial_scope="Wind BMU",
        role="Half-hour bid-offer evidence layer for expanding dispatch capture beyond direct BOALF acceptances.",
        source_plan="Elexon BOD half-hour ladder aggregated by BMU, retaining negative-bid availability flags and price diagnostics.",
        status="implemented_first_pass",
        note="Evidence only, not direct dispatch truth; use this with PN/QPN gaps inside BOALF-triggered windows to infer additional dispatch rows without relabeling BOALF.",
    ),
    DatasetSpec(
        key="fact_bmu_physical_position_half_hourly",
        grain="half-hourly",
        spatial_scope="Wind BMU",
        role="Half-hour PN, QPN, MILS, and MELS evidence for physical curtailment baselines.",
        source_plan="Elexon balancing physical endpoint joined to BMU generation truth.",
        status="implemented_first_pass",
        note="Provides a physical-baseline tier and explicit invalidation when the baseline is missing or inconsistent.",
    ),
    DatasetSpec(
        key="fact_bmu_availability_half_hourly",
        grain="half-hourly",
        spatial_scope="Wind BMU",
        role="Availability gate for lost-energy estimation.",
        source_plan="REMIT outage windows mapped onto BMUs, with UOU2T14D as secondary QA where available.",
        status="implemented_first_pass",
        note="REMIT drives outage gating, but partial-availability REMIT windows now degrade to unknown and can be explicitly overridden in truth-building when available capacity supports the counterfactual.",
    ),
    DatasetSpec(
        key="fact_bmu_curtailment_truth_half_hourly",
        grain="half-hourly",
        spatial_scope="Wind BMU",
        role="Tiered BMU curtailment truth surface for precision-first and research-friendly backtests.",
        source_plan="Join B1610 generation, BOALF acceptance truth, BOD bid-offer evidence, balancing physical, REMIT availability, weather history, and GB daily reconciliation.",
        status="implemented_first_pass",
        note="Carries acceptance-only and physical-inference dispatch tiers, plus explicit same-BMU versus family-day inference scope, alongside dispatch-only, physical-baseline, and weather-calibrated lost-energy tiers.",
    ),
    DatasetSpec(
        key="fact_curtailment_reconciliation_daily",
        grain="daily",
        spatial_scope="GB-wide with BMU diagnostics",
        role="Target-quality QA surface that explains why BMU truth does or does not reconcile to GB daily curtailment labels.",
        source_plan="Aggregate fact_bmu_curtailment_truth_half_hourly against fact_constraint_daily, plus block-reason and BMU-day gap summaries.",
        status="implemented_first_pass",
        note="Carries both raw NESO-total reconciliation and the wind-only positive-voltage-plus-thermal QA target used for precision gating, with same-BMU and family-day dispatch increments broken out separately.",
    ),
    DatasetSpec(
        key="fact_constraint_target_audit_daily",
        grain="daily",
        spatial_scope="GB-wide with QA-target composition audit",
        role="Explains which QA-target categories dominate each day and whether the remaining miss looks source-limited or definition-limited.",
        source_plan="Join fact_constraint_daily category columns to daily BMU reconciliation and dispatch-alignment surfaces.",
        status="implemented_first_pass",
        note="Use this before changing the precision gate; it distinguishes voltage-led, thermal-led, and mixed shortfall days.",
    ),
    DatasetSpec(
        key="fact_dispatch_alignment_daily",
        grain="daily",
        spatial_scope="GB-wide with BMU dispatch decomposition",
        role="Explains whether blocked dispatch could close the QA-target gap or whether the current dispatch source undercaptures the target by construction.",
        source_plan="Derived from fact_bmu_curtailment_truth_half_hourly with acceptance truth, physical-inference increments, and dispatch rows split into estimated, blocked, and mapping-status buckets.",
        status="implemented_first_pass",
        note="Use this to decide whether BOALF-only undercapture is being reduced materially by the BOD plus PN/QPN expansion or whether a new source is still required.",
    ),
    DatasetSpec(
        key="fact_bmu_family_shortfall_daily",
        grain="daily",
        spatial_scope="BMU family",
        role="Attributes remaining dispatch-to-lost-energy gaps to BMU families and days rather than only individual BMUs.",
        source_plan="Group fact_bmu_curtailment_truth_half_hourly by derived BMU family code and day.",
        status="implemented_first_pass",
        note="Use this to see which families actually drive the benchmark shortfall before adding new sources; same-BMU and family-day dispatch increments are now separated so targeted expansion stays auditable.",
    ),
    DatasetSpec(
        key="fact_source_completeness_focus_daily",
        grain="daily",
        spatial_scope="GB-wide with store-backed prioritization",
        role="Ranks remaining fail and warn days by QA shortfall and recommends the next source-completeness action.",
        source_plan="Join fact_curtailment_reconciliation_daily and fact_constraint_target_audit_daily inside the SQLite truth store.",
        status="implemented_first_pass",
        note="This is the first store-native planning surface; use it to focus the next dispatch-source pass instead of stitching CSVs by hand.",
    ),
    DatasetSpec(
        key="fact_source_completeness_focus_family_daily",
        grain="daily",
        spatial_scope="BMU family",
        role="Ranks BMU families on fail and warn days by remaining dispatch-to-lost-energy gap and recommends the next family-level action.",
        source_plan="Join fact_bmu_family_shortfall_daily to fact_source_completeness_focus_daily inside the SQLite truth store.",
        status="implemented_first_pass",
        note="Use this to isolate families like Moray East, Seagreen, or Race Bank that still dominate the store-backed shortfall after family-day expansion.",
    ),
    DatasetSpec(
        key="fact_dispatch_source_gap_daily",
        grain="daily",
        spatial_scope="GB-wide with store-backed missing-source diagnostics",
        role="Quantifies how much missing dispatch evidence is implied by negative bids plus physical down gaps on each day.",
        source_plan="Aggregate fact_bmu_curtailment_truth_half_hourly inside the SQLite truth store for rows with no current dispatch evidence but strong missing-source signatures.",
        status="implemented_first_pass",
        note="Splits candidate gap MWh into same-BMU window, family-window, and no-window scopes so the next source pass can distinguish missing sources from threshold issues.",
    ),
    DatasetSpec(
        key="fact_dispatch_source_gap_family_daily",
        grain="daily",
        spatial_scope="BMU family",
        role="Ranks BMU families by store-backed missing dispatch-source candidate MWh rather than only by dispatch-to-lost-energy gap.",
        source_plan="Group store-backed truth rows with negative bids, physical down gaps, and no current dispatch evidence by BMU family and day.",
        status="implemented_first_pass",
        note="Use this to target families like Sheirds, Beinn Tharsuinn, Kype Muir, Seagreen, or Moray when the remaining miss still looks source-limited.",
    ),
    DatasetSpec(
        key="fact_publication_anomaly_daily",
        grain="daily",
        spatial_scope="GB-wide with family-backed publication diagnostics",
        role="Ranks days where physical down-gap evidence exists without published BOALF, and splits that anomaly into sentinel, negative-bid, dynamic-limit-like, or other buckets.",
        source_plan="Join store-backed truth and physical rows, isolate physical-without-BOALF cases, then aggregate them by day.",
        status="implemented_first_pass",
        note="Use this to decide whether a missing-curtailment day looks like a broader publication anomaly rather than only a dispatch-source shortfall.",
    ),
    DatasetSpec(
        key="fact_publication_anomaly_family_daily",
        grain="daily",
        spatial_scope="BMU family",
        role="Ranks family-days by publication-anomaly MWh and classifies the dominant anomaly type for support or audit follow-up.",
        source_plan="Aggregate store-backed physical-without-BOALF cases by BMU family and day, then join daily QA context.",
        status="implemented_first_pass",
        note="Use this to answer whether Hornsea is unique or part of a wider offshore publication pattern.",
    ),
    DatasetSpec(
        key="fact_family_dispatch_forensic_daily",
        grain="daily",
        spatial_scope="Scoped BMU family",
        role="Targeted forensic surface for one chosen family scope, showing current capture, missing-source candidates, and availability/block-state mix by day.",
        source_plan="Filter store-backed truth rows to the chosen family scope, then aggregate with fact_dispatch_source_gap_family_daily metadata.",
        status="implemented_first_pass",
        note="Use this for focused investigations like Hornsea before changing truth logic; it is diagnostic only and should not feed model training directly.",
    ),
    DatasetSpec(
        key="fact_family_dispatch_forensic_bmu_daily",
        grain="daily",
        spatial_scope="Scoped BMU within family",
        role="Ranks individual BMUs inside a forensic family scope by missing-source candidate MWh.",
        source_plan="Filter store-backed truth rows to the chosen family scope, then aggregate by BMU and day.",
        status="implemented_first_pass",
        note="This is the main surface for identifying which Hornsea BMUs actually drive the family-day forensic gap.",
    ),
    DatasetSpec(
        key="fact_family_dispatch_forensic_half_hourly",
        grain="half-hourly",
        spatial_scope="Scoped BMU within family",
        role="Half-hour evidence tape for a forensic family scope, retaining both current dispatch truth and missing-source candidate rows.",
        source_plan="Filter store-backed truth rows to the chosen family scope and keep half-hours with negative bids, dispatch truth, lost-energy estimates, or source-gap candidates.",
        status="implemented_first_pass",
        note="Use this to inspect whether the family miss is mainly no-window candidate activity, existing captured dispatch, or a counterfactual problem.",
    ),
    DatasetSpec(
        key="fact_family_physical_forensic_daily",
        grain="daily",
        spatial_scope="Scoped BMU family",
        role="Targeted PN/QPN/MILS/MELS forensic surface for one chosen family scope, showing whether positive physical down-gaps exist despite zero BOALF capture.",
        source_plan="Join fact_bmu_physical_position_half_hourly to store-backed truth rows for the chosen family scope, then aggregate by family and day.",
        status="implemented_first_pass",
        note="Use this to answer whether families like Hornsea have visible physical curtailment signatures even when current dispatch reconstruction stays empty.",
    ),
    DatasetSpec(
        key="fact_family_physical_forensic_bmu_daily",
        grain="daily",
        spatial_scope="Scoped BMU within family",
        role="Ranks individual BMUs inside a forensic family scope by positive PN-QPN gap with zero BOALF capture.",
        source_plan="Join fact_bmu_physical_position_half_hourly to store-backed truth rows for the chosen family scope, then aggregate by BMU and day.",
        status="implemented_first_pass",
        note="This is the main surface for identifying which Hornsea BMUs have the largest zero-BOALF physical gap.",
    ),
    DatasetSpec(
        key="fact_family_physical_forensic_half_hourly",
        grain="half-hourly",
        spatial_scope="Scoped BMU within family",
        role="Half-hour PN/QPN/MILS/MELS evidence tape for a forensic family scope, keeping zero-BOALF physical gaps, valid negative bids, and sentinel bid artifacts visible.",
        source_plan="Join fact_bmu_physical_position_half_hourly to store-backed truth rows for the chosen family scope and keep half-hours with positive physical gaps, dispatch truth, or bid diagnostics.",
        status="implemented_first_pass",
        note="Use this to inspect whether the family miss is visible as repeated positive PN-QPN gaps with zero BOALF, and whether sentinel BOD values are contaminating the evidence layer.",
    ),
    DatasetSpec(
        key="fact_family_publication_audit_daily",
        grain="daily",
        spatial_scope="Scoped BMU family",
        role="Support-oriented publication audit that asks whether published BOALF is absent even when the family shows strong physical curtailment signatures.",
        source_plan="Aggregate the scoped physical-forensics context by family and day, then classify the case into missing-BOALF, dynamic-limit-like, or no-publication-audit signal states.",
        status="implemented_first_pass",
        note="Use this to decide whether a family-day is support-ready and what the support question should actually be.",
    ),
    DatasetSpec(
        key="fact_family_publication_audit_bmu_daily",
        grain="daily",
        spatial_scope="Scoped BMU within family",
        role="Ranks BMUs inside a scoped family by support-ready physical-without-BOALF evidence.",
        source_plan="Aggregate the scoped physical-forensics context by BMU and day, then attach a support question code and rank by physical-without-BOALF gap MWh.",
        status="implemented_first_pass",
        note="Use this when a family-day is support-worthy but the exact BMUs need to be narrowed before sending evidence out.",
    ),
    DatasetSpec(
        key="fact_family_support_evidence_half_hourly",
        grain="half-hourly",
        spatial_scope="Scoped BMU within family",
        role="Support-ready evidence extract that keeps half-hour physical, bid, sentinel, and BOALF-absence diagnostics together for external investigation.",
        source_plan="Filter the scoped physical-forensics context to support-case rows and retain the row-level physical, bid, sentinel, and publication-audit fields.",
        status="implemented_first_pass",
        note="This is diagnostic and export-focused; it should not feed model training or truth gating directly.",
    ),
    DatasetSpec(
        key="fact_support_case_daily",
        grain="daily",
        spatial_scope="Store-backed support batch",
        role="Ranks support-ready publication-anomaly days and packages their QA context for escalation.",
        source_plan="Select support_query publication-anomaly days from the SQLite truth store, then attach batch metadata and recommended support actions.",
        status="implemented_first_pass",
        note="This is packaging only; it should not feed truth gating or model training directly.",
    ),
    DatasetSpec(
        key="fact_support_case_family_daily",
        grain="daily",
        spatial_scope="Store-backed support batch at BMU-family scope",
        role="Ranks support-ready family-days inside each selected publication-anomaly day and keeps mapped spatial context plus support question codes.",
        source_plan="Join selected publication-anomaly family-days to scoped publication-audit summaries from the SQLite truth store.",
        status="implemented_first_pass",
        note="This narrows the support packet to mapped family-days and excludes mapping-and-publication-audit cases.",
    ),
    DatasetSpec(
        key="fact_support_case_half_hourly",
        grain="half-hourly",
        spatial_scope="Store-backed support batch at BMU half-hour scope",
        role="Stores the full half-hour evidence tape for selected support family-days, including PN/QPN/MILS/MELS, BOALF absence, and bid diagnostics.",
        source_plan="Filter scoped family support-evidence rows to the selected support batch and persist them with batch metadata.",
        status="implemented_first_pass",
        note="This is the escalation evidence tape; it should not be auto-promoted into dispatch truth.",
    ),
    DatasetSpec(
        key="fact_support_case_resolution",
        grain="family-day resolution ledger",
        spatial_scope="Store-backed support batch at BMU family scope",
        role="Records analyst and support outcomes for publication-anomaly family-days and links them to explicit truth-policy actions.",
        source_plan="Join store-backed support-case family rows to a manual resolution ledger keyed by support batch, settlement date, and BMU family.",
        status="implemented_first_pass",
        note="This is the decision gate for publication anomalies; unresolved rows should remain out of the precision profile.",
    ),
    DatasetSpec(
        key="fact_support_resolution_daily",
        grain="daily review summary",
        spatial_scope="Store-backed support batch at day scope",
        role="Summarizes whether each publication-anomaly day is still blocked by open cases or is ready for rerun, exclusion, or evidence-tier review.",
        source_plan="Aggregate fact_support_case_resolution back onto fact_support_case_daily inside the SQLite truth store.",
        status="implemented_first_pass",
        note="Use this to answer whether a benchmark day is still blocked by unresolved publication anomalies.",
    ),
    DatasetSpec(
        key="fact_support_resolution_batch",
        grain="batch review summary",
        spatial_scope="Store-backed support batch",
        role="Summarizes whether an entire support batch is still blocked or is ready for the next truth-policy action.",
        source_plan="Aggregate fact_support_resolution_daily by support batch inside the SQLite truth store.",
        status="implemented_first_pass",
        note="This is the top-level decision surface before any rerun or truth-tier change.",
    ),
    DatasetSpec(
        key="fact_support_rerun_gate_daily",
        grain="daily rerun-gate advisory",
        spatial_scope="Store-backed support batch at day scope",
        role="Labels whether a publication-anomaly day is blocked, is a targeted rerun candidate, or should be policy-locked with no rerun.",
        source_plan="Apply mixed-mode rerun-gate rules to fact_support_resolution_daily inside the SQLite truth store.",
        status="implemented_first_pass",
        note="Day rows are advisory only; they do not override a blocked batch.",
    ),
    DatasetSpec(
        key="fact_support_rerun_gate_batch",
        grain="batch rerun gate",
        spatial_scope="Store-backed support batch",
        role="Provides the hard stop/go surface for whether a support batch can move toward a rerun decision.",
        source_plan="Apply mixed-mode rerun-gate rules to fact_support_resolution_batch and candidate rerun days inside the SQLite truth store.",
        status="implemented_first_pass",
        note="This is the operational rerun gate; future rerun workflows should consult it before acting.",
    ),
    DatasetSpec(
        key="fact_support_open_case_priority_family_daily",
        grain="family-day open-case priority",
        spatial_scope="Store-backed support batch at BMU family scope",
        role="Ranks unresolved support cases by anomaly magnitude and existing family-day priority so analysts can work the highest-leverage open cases first.",
        source_plan="Join fact_support_case_family_daily to open rows in fact_support_case_resolution, then attach current batch/day rerun-gate states.",
        status="implemented_first_pass",
        note="This is a triage surface only; it does not mutate truth policy or dispatch evidence.",
    ),
    DatasetSpec(
        key="fact_support_resolution_pattern_summary",
        grain="open-case pattern summary",
        spatial_scope="Store-backed support batch at repeated BMU-family pattern scope",
        role="Groups repeated open family-day cases into reusable analyst patterns so repeated publication-anomaly decisions can be reviewed once and applied consistently.",
        source_plan="Group fact_support_open_case_priority_family_daily by family, anomaly state, support question code, and mapping status inside the SQLite truth store.",
        status="implemented_first_pass",
        note="This is the scaling helper for repeated resolution work; it does not change truth by itself.",
    ),
    DatasetSpec(
        key="fact_support_resolution_pattern_member_family_daily",
        grain="family-day pattern member",
        spatial_scope="Store-backed support batch at BMU family scope",
        role="Lists the currently open family-days that belong to each repeated analyst pattern and are eligible for bulk resolution annotation.",
        source_plan="Attach pattern keys from fact_support_resolution_pattern_summary back onto fact_support_open_case_priority_family_daily.",
        status="implemented_first_pass",
        note="Bulk apply should only target rows surfaced here for the chosen batch and pattern key.",
    ),
    DatasetSpec(
        key="fact_support_rerun_candidate_daily",
        grain="daily rerun candidate",
        spatial_scope="Store-backed support batch at day scope",
        role="Identifies days that are actually ready for a targeted rerun once all cases on that day are resolved and at least one resolved action points to rerun or evidence-tier promotion.",
        source_plan="Filter fact_support_rerun_gate_daily to candidate_targeted_rerun days and summarize rerun-driving truth-policy actions.",
        status="implemented_first_pass",
        note="This remains advisory if the enclosing batch is still blocked by open cases elsewhere.",
    ),
    DatasetSpec(
        key="fact_support_rerun_candidate_family_daily",
        grain="family-day rerun candidate",
        spatial_scope="Store-backed support batch at BMU family scope",
        role="Lists resolved family-days inside rerun-candidate days that should drive the next targeted rerun or evidence-tier review.",
        source_plan="Join fact_support_case_family_daily to resolved rerun-driving actions in fact_support_case_resolution, then filter to candidate_targeted_rerun days.",
        status="implemented_first_pass",
        note="This is the prep surface for a future rerun workflow; it still does not change truth directly.",
    ),
    DatasetSpec(
        key="fact_weather_hourly",
        grain="hourly",
        spatial_scope="Cluster and parent-region weather anchors",
        role="Forecast and actual weather features for model training and map overlays.",
        source_plan="Capacity-weighted anchor weather per cluster, then rolled to parent regions when needed.",
        status="implemented_first_pass",
        note="Observed historical weather now materializes from anchor points; forecast joins and forecast-error backtests are still missing.",
    ),
    DatasetSpec(
        key="fact_constraint_daily",
        grain="daily",
        spatial_scope="GB-wide",
        role="Historical curtailment cost and volume truth set.",
        source_plan="NESO constraint breakdown daily series.",
        status="implemented",
        note="Now carries both raw category totals and a wind-only positive voltage plus thermal QA target for BMU truth alignment.",
    ),
    DatasetSpec(
        key="fact_wind_split_half_hourly",
        grain="half-hourly",
        spatial_scope="Scotland vs England/Wales",
        role="Bridge signal for allocating GB-wide curtailment into regional history.",
        source_plan="NESO metered wind split or equivalent regional wind outturn feed.",
        status="implemented",
        note="Used first for regional historical decomposition before asset-level truth exists.",
    ),
    DatasetSpec(
        key="fact_regional_curtailment_hourly_proxy",
        grain="hourly",
        spatial_scope="Cluster and parent-region",
        role="Trainable target surface before true locational curtailment is available.",
        source_plan="Allocate daily GB curtailment using regional or cluster wind weights and topology gates.",
        status="implemented_proxy",
        note="Must stay clearly labeled as proxy, not truth.",
    ),
    DatasetSpec(
        key="fact_route_score_hourly",
        grain="hourly",
        spatial_scope="Cluster-to-hub-to-route",
        role="Route feasibility, bottleneck, and netback surface for scenario analysis.",
        source_plan="Current route heuristics plus future fact_gb_transfer_gate_hourly, fact_interconnector_flow_hourly, and fact_interconnector_capacity_hourly inputs.",
        status="partial",
        note="Exists today only at GB node level and without physical gating; the next network phase should promote it from a spread heuristic into a deliverability surface.",
    ),
    DatasetSpec(
        key="fact_backtest_prediction_hourly",
        grain="hourly",
        spatial_scope="Cluster, region, and route",
        role="Stores predictions, actuals, residuals, and run metadata for each backtest.",
        source_plan="Generated by walk-forward training runs.",
        status="planned",
        note="This table is the audit trail; it should never be recomputed silently.",
    ),
    DatasetSpec(
        key="fact_drift_window",
        grain="rolling window",
        spatial_scope="Cluster, region, route, and feature family",
        role="Feature drift, target drift, and residual drift monitoring.",
        source_plan="Computed from historical truth, proxy targets, and stored backtest predictions.",
        status="planned",
        note="Needed to catch regime changes before we trust the map or deployment decisions.",
    ),
)


MAP_LAYER_SPECS: Tuple[MapLayerSpec, ...] = (
    MapLayerSpec(
        key="cluster_time_slider",
        phase="phase_1",
        geometry="Cluster centroids as points",
        time_axis="Hourly scrubber with day/week aggregation toggle",
        value_encoding="Point size = curtailed MWh, color = actual vs proxy vs predicted",
        interaction="Hover for lineage, click for history panel, brush to sync charts",
        purpose="First interactive map for backtesting and exploration.",
    ),
    MapLayerSpec(
        key="cluster_hub_arc_map",
        phase="phase_2",
        geometry="Arcs from cluster centroids to interconnector hubs",
        time_axis="Hourly scrubber and route selector",
        value_encoding="Arc width = feasible export volume, color = route score or blocked state",
        interaction="Toggle actual, predicted, and counterfactual battery scenarios",
        purpose="Makes the internal-transfer assumption visible instead of hiding it in a table.",
    ),
    MapLayerSpec(
        key="space_time_error_map",
        phase="phase_2",
        geometry="Clusters or H3 cells",
        time_axis="Backtest window slider",
        value_encoding="Height = absolute error, color = signed bias or drift percentile",
        interaction="Compare forecast run versions and drill into error slices",
        purpose="This is the practical 4D view: geography plus time plus magnitude plus state.",
    ),
    MapLayerSpec(
        key="battery_siting_scenario_map",
        phase="phase_3",
        geometry="Clusters, hubs, and candidate battery markers",
        time_axis="Historical replay or forecast horizon",
        value_encoding="Marker size = avoided curtailment, color = gross value or utilization",
        interaction="Adjust battery size, duration, and cluster placement live",
        purpose="Turns the exploration tool into a siting and what-if interface.",
    ),
)


BACKTEST_SPECS: Tuple[BacktestSpec, ...] = (
    BacktestSpec(
        key="regional_daily_curtailment",
        target="Daily regional curtailed MWh",
        forecast_horizon="1 to 7 days",
        split_strategy="Anchored walk-forward by month",
        primary_metrics="MAE, RMSE, MAPE, bias",
        slice_dimensions="Region, season, wind regime, constraint regime",
        acceptance_gate="No region should exceed a sustained positive or negative bias for two consecutive windows.",
    ),
    BacktestSpec(
        key="cluster_hourly_proxy",
        target="Hourly cluster curtailment proxy",
        forecast_horizon="24h and 168h",
        split_strategy="Rolling-origin hourly backtest",
        primary_metrics="MAE, pinball loss, hit rate on top decile hours",
        slice_dimensions="Cluster, hour of day, weekday, storm regime",
        acceptance_gate="Top-decile event recall must remain stable by cluster before using the map operationally.",
    ),
    BacktestSpec(
        key="route_feasibility",
        target="Feasible versus blocked route state",
        forecast_horizon="24h and 168h",
        split_strategy="Rolling-origin classification backtest",
        primary_metrics="Precision, recall, false-positive rate",
        slice_dimensions="Hub family, route, internal-transfer class",
        acceptance_gate="False positives must be tightly controlled because bad export signals are operationally expensive.",
    ),
)


DRIFT_MONITOR_SPECS: Tuple[DriftMonitorSpec, ...] = (
    DriftMonitorSpec(
        key="feature_distribution_drift",
        monitor_scope="Weather, price spread, and regional wind-weight features",
        metric="PSI or KS by feature family",
        cadence="Each retrain window",
        trigger="Feature family crosses threshold in two consecutive windows",
        action="Freeze promotion, inspect weather-anchor coverage, and compare against topology changes.",
    ),
    DriftMonitorSpec(
        key="target_definition_drift",
        monitor_scope="Daily truth versus hourly proxy allocation",
        metric="Share-of-day concentration and regional allocation shift",
        cadence="Weekly",
        trigger="Regional proxy weights move materially without a matching change in input wind structure",
        action="Revisit allocation logic before trusting regional backtests.",
    ),
    DriftMonitorSpec(
        key="residual_drift",
        monitor_scope="Prediction residuals by cluster and route",
        metric="Rolling bias and tail error",
        cadence="Every model run",
        trigger="Signed bias persists by cluster or route family",
        action="Block deployment and investigate whether topology or curtailment truth changed.",
    ),
    DriftMonitorSpec(
        key="topology_drift",
        monitor_scope="Hub availability and reachability assumptions",
        metric="Registry diff and scenario impact delta",
        cadence="Whenever hub or cluster metadata changes",
        trigger="A topology edit changes route ranking or feasible hours materially",
        action="Version the topology registry and rerun backtests before comparing model generations.",
    ),
)


def dataset_plan_frame() -> pd.DataFrame:
    return pd.DataFrame([spec.__dict__ for spec in DATASET_SPECS]).sort_values("key").reset_index(drop=True)


def map_layer_plan_frame() -> pd.DataFrame:
    return pd.DataFrame([spec.__dict__ for spec in MAP_LAYER_SPECS]).sort_values(["phase", "key"]).reset_index(drop=True)


def backtest_plan_frame() -> pd.DataFrame:
    return pd.DataFrame([spec.__dict__ for spec in BACKTEST_SPECS]).sort_values("key").reset_index(drop=True)


def drift_monitor_plan_frame() -> pd.DataFrame:
    return pd.DataFrame([spec.__dict__ for spec in DRIFT_MONITOR_SPECS]).sort_values("key").reset_index(drop=True)
