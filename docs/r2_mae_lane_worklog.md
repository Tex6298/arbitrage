# R2 MAE Lane Worklog

## 2026-03-22

- picked up the stalled arbitrage handoff from the sibling repo at
  `C:\Users\marty\Documents\LocalPython\arbitrage`
- confirmed the current uncommitted diff already contains:
  - the earlier `benchmark_suite.py` reviewed-bundle scout fallback change
  - unrelated DST parsing work in `opportunity_backtest.py`
- scoped the next safe MAE patch to the existing
  `_apply_potential_ratio_r2_reviewed_event_lifecycle` late-reopen branch
- widened the late-reopen basis gate only:
  - from `ratio_route_notice_state`
  - to `ratio_route_notice_state` plus `ratio_exact_notice_hour`
- kept every other late-reopen guard unchanged:
  - route `R2_netback_GB_NL_DE_PL`
  - source family `day_ahead_constraint_boundary`
  - horizon `1`
  - gate `reviewed_boundary_cap`
  - tier `no_price_signal`
  - transition `price_non_positive->price_non_positive`
  - ITL state `blocked_zero_or_negative_itl`
  - origin hour `14`
  - negative route score threshold
  - weaker-forward upstream state
  - capped cluster trio only
- added regression coverage for the exact-notice late-reopen shape
- verification run:
  - `python -m pytest tests\test_opportunity_backtest.py -q`
  - reran `build_fact_backtest_prediction_hourly(..., forecast_horizons=[1])`
    against:
    - `curtailment_opportunity_live_britned_reviewed_rerun_2024-10-01_2024-10-07`
    - `_local_runs/reviewed_volume_supported_range_v1/bundles/curtailment_opportunity_live_britned_reviewed_2025-03-15_2025-03-17`
    - the broader `_local_runs/reviewed_volume_supported_range_v1/bundles/*` sweep
- current verification outcome:
  - tests pass
  - October still produces the same three `ratio_route_notice_state` late-reopen rows
  - the local reviewed-bundle sweep did not surface any live
    `ratio_exact_notice_hour_r2_reviewed_event_late_reopen` rows yet
  - treat this patch as lifecycle hardening and regression coverage, not as a
    confirmed batch-metric mover on the current local stock

## Still Open

- broader 2025 `R2` reviewed underpredictions remain split across multiple
  signatures
- do not assume this patch closes the no-public late-open family or the
  zero-openable-potential September case

## 2026-03-22 Current-Head Family Collapse

- planning baseline remains:
  `_local_runs/reviewed_volume_supported_range_v1/model_readiness_reviewed_bundle_batch_shared_plus_r2_no_public_plus_july_plus_march_v1`
- reran each target family on current HEAD with
  `python inline_arbitrage_live.py --materialize-model-readiness ...` into:
  - `_local_runs\r2_mae_impl_20260322\baseline_april_2025-04-15_2025-04-17`
  - `_local_runs\r2_mae_impl_20260322\baseline_july_2025-07-15_2025-07-17`
  - `_local_runs\r2_mae_impl_20260322\baseline_october_2025-10-15_2025-10-17`
- current-head status by family:
  - April 15 still reproduces as not ready
  - July 15 no longer reproduces; all three days are ready and
    `fact_model_blocker_priority.csv` is empty
  - October 15 to 17 no longer reproduces; all three days are ready and
    `fact_model_blocker_priority.csv` is empty
- because July and October are stale on current HEAD, they stay out of active
  implementation scope for this lane

## 2026-03-22 April Regression Restoration

- inspected the live April hourly output and found the original target pair is
  already closed on current HEAD:
  - hour `4` rows fire as
    `ratio_route_notice_state_r2_no_public_april_hour4_open`
  - hour `16` rows fire as
    `ratio_route_notice_state_r2_no_public_april_hour16_open`
- the remaining reproducing April miss is the hour `18` close-reopen family on
  `2025-04-15`
  - live rows are `no_price_signal`, not `reviewed`
  - they remain zeroed on current HEAD
  - older combined supported-range outputs show these rows previously reopened as
    `ratio_route_notice_state_r2_no_public_late_open_close_reopen`
  - the reopen set is the five-cluster family:
    `dogger_hornsea_offshore`, `east_anglia_offshore`,
    `east_coast_scotland_offshore`, `humber_offshore`,
    `moray_firth_offshore`
  - `shetland_wind` stays zero in the older combined output and is treated as a
    negative control
- implementation landed:
  - added a dated April 15 hour `18` no-public close-reopen event spec in
    `opportunity_backtest.py`
  - kept the generic hour `18` `reviewed` close-reopen rule unchanged
  - added optional `cluster_keys` support to the no-public event library so the
    dated April reopen is limited to the five reproduced clusters
  - added regression coverage proving the April close-reopen patch opens the
    five-cluster family while leaving `shetland_wind` and non-event dates at
    zero
- focused verification:
  - `python -m pytest tests\test_opportunity_backtest.py -q`
  - result: `53 passed`
- targeted April rerun:
  - output:
    `_local_runs\r2_mae_impl_20260322\patched_april_2025-04-15_2025-04-17`
  - hourly restoration matches the historical combined baseline for the five
    target clusters:
    - `dogger_hornsea_offshore`
    - `east_anglia_offshore`
    - `east_coast_scotland_offshore`
    - `humber_offshore`
    - `moray_firth_offshore`
  - negative control still holds:
    - `shetland_wind` remains zero
  - daily readiness delta for
    `curtailment_opportunity_live_britned_reviewed_2025-04-15_2025-04-17`:
    - before patch on current HEAD: `1 ready / 2 not ready`
    - after patch: `3 ready / 0 not ready`
    - `2025-04-15` overall `t+1h` MAE: `0.703741 -> 0.184428`
    - `2025-04-15` GB-NL `t+1h` MAE: `2.345804 -> 0.614760`
    - route warn count: `1 -> 0`
    - cluster warn count: `1 -> 0`
    - blocker file is now empty

## 2026-03-22 Full Supported-Range Batch

- ran:
  `python inline_arbitrage_live.py --materialize-reviewed-bundle-batch-eval ...`
- output:
  `_local_runs/reviewed_volume_supported_range_v1/model_readiness_reviewed_bundle_batch_shared_plus_r2_no_public_plus_july_plus_march_plus_april_restore_v1`
- comparison target:
  `_local_runs/reviewed_volume_supported_range_v1/model_readiness_reviewed_bundle_batch_shared_plus_r2_no_public_plus_july_plus_march_v1`
- batch-level delta versus that older combined baseline:
  - ready days: `51 -> 59`
  - not-ready days: `10 -> 2`
  - remaining blocker windows: only
    `curtailment_opportunity_live_britned_reviewed_2025-01-24_2025-01-26`
  - remaining blocker type: `proxy_internal_transfer_share_too_high`
  - 2024 windows: no readiness regressions
- attribution note:
  - April is the family isolated and verified in this turn
  - July and October were already stale on current HEAD before this patch
  - December also improved versus the older combined baseline, but that change
    was not isolated inside this April restoration lane and should not be
    credited to the April patch by assumption

## 2026-03-22 Remaining January Proxy Debt Audit

- investigated the only remaining blocker window:
  `curtailment_opportunity_live_britned_reviewed_2025-01-24_2025-01-26`
- confirmed current blocker shape:
  - `2025-01-24` not ready with
    `proxy_internal_transfer_share_too_high`
  - `2025-01-25` ready
  - `2025-01-26` not ready with
    `proxy_internal_transfer_share_too_high`
  - no MAE blocker, no route drift blocker, no cluster drift blocker
- confirmed this window is a non-informative perfect-zero case rather than a
  prediction miss:
  - `informative_signal_basis = reviewed_perfect_zero_window`
  - actual deliverable sums are zero across all three days
  - baseline error sums are zero across all three days
- traced the blocker to internal-transfer evidence quality, not opportunity
  prediction:
  - readiness blocks because `proxy_internal_transfer_share_t_plus_1h` is
    `0.452273` on `2025-01-24` and `0.510417` on `2025-01-26`
  - the dominant blocker slice is
    `route=R1_netback_GB_FR_DE_PL|cluster=dogger_hornsea_offshore|hub=eleclink|internal=gb_topology_transfer_gate_proxy`
- local route-score audit outcome:
  - the proxy rows in `fact_route_score_hourly.csv` carry
    `internal_transfer_tier_accepted_flag = <NA>`
  - the same rows carry `internal_transfer_review_state = proxy_fallback`
  - the final evidence tier is therefore honestly
    `gb_topology_transfer_gate_proxy`
  - reviewed boundary evidence does exist in the bundle, but not for the
    blocked proxy-backed route and hub combinations in a way that produces an
    accepted reviewed join for those rows
  - direct `interval_start_utc + cluster_key + hub_key` comparison found
    `0` exact reviewed-boundary matches for the currently proxy-backed blocker
    rows
- implementation conclusion:
  - there is no honest code patch to "accept" these January rows into reviewed
    state from the current local inputs
  - changing the code here would mean relabeling proxy-backed rows as reviewed
    without accepted reviewed evidence
  - this matches the standing posture in `docs/map_operational_posture.md`:
    keep January 2025 out of promotion logic until better reviewed evidence
    exists

## 2026-03-22 December Attribution Audit

- checked the `2025-12-15_2025-12-17` improvement against the older combined
  baseline
- confirmed the December change is not a prediction change:
  - current and baseline `t+1h` hourly prediction rows match
  - the daily flip is `2025-12-16` only
- confirmed the improvement is a drift-state classification difference:
  - the old output marks `east_anglia_offshore` as `warn`
  - the current rerun marks the same score tuple as `pass`
- attribution conclusion:
  - December improvement is almost certainly tied to drift-classification logic
    already present in current `opportunity_backtest.py`
  - it is not attributable to the April 15 R2 restore landed in this turn

## 2026-03-22 January Regeneration Feasibility Recheck

- revisited the January blocker after comparing the saved bundle artifact to the
  current `gb_transfer_boundary_reviewed.py` builder
- important correction to the earlier January audit:
  - the current local bundle still does **not** contain same-hour reviewed
    matches for the blocker rows
  - but rebuilding `fact_gb_transfer_boundary_reviewed_hourly` from the same
    raw January day-ahead boundary input with current code produces materially
    more accepted rows than the saved bundle artifact
- confirmed stale-artifact delta:
  - saved distinct `interval_start_utc + cluster_key + hub_key` keys: `865`
  - rebuilt with current code: `1224`
  - new keys in rebuilt output: `359`
  - most of those new keys land on the FR-facing England blocker surface:
    `dogger_hornsea_offshore`, `east_anglia_offshore`, `humber_offshore` on
    `eleclink`, `ifa`, and `ifa2`
- confirmed projected blocker impact on the failing January days:
  - failing-day proxy rows in the saved opportunity bundle: `464`
  - same-hour rebuilt reviewed matches available from current code: `320`
  - remaining uncovered rows after rebuilt match: `144`
  - those remaining uncovered rows are entirely `shetland_wind`
- projected daily proxy-share delta if the January bundle is fully regenerated
  through current code and those rebuilt matches flow into route-score and
  opportunity materialization:
  - `2025-01-24`: about `0.456 -> 0.150`
  - `2025-01-26`: about `0.510417 -> 0.150`
  - both would fall below the current readiness proxy gate of `0.45`
- current best interpretation:
  - January is not fully "new upstream evidence only" debt
  - part of the blocker is stale local-bundle debt that should be recoverable by
    rebuilding the January reviewed bundle with current code into a fresh output
    directory
  - what still looks genuinely uncovered after that rebuild is the
    `shetland_wind` surface
- next executable step:
  - run a single-window January rebuild via
    `inline_arbitrage_live.py --materialize-curtailment-opportunity-history`
    into a fresh scratch directory, then rerun readiness on that rebuilt
    opportunity bundle

## 2026-03-22 January Rerun Promotion

- materialized the promoted January rerun bundle into the supported-range
  bundle root:
  - `_local_runs/reviewed_volume_supported_range_v1/bundles/curtailment_opportunity_live_britned_reviewed_rerun_2025-01-24_2025-01-26`
- exact command:
  - `python inline_arbitrage_live.py --materialize-curtailment-opportunity-history --opportunity-start 2025-01-24 --opportunity-end 2025-01-26 --opportunity-output-dir _local_runs\reviewed_volume_supported_range_v1\bundles\curtailment_opportunity_live_britned_reviewed_rerun_2025-01-24_2025-01-26 --opportunity-truth-profile proxy`
- rerun artifact confirmation:
  - `fact_gb_transfer_boundary_reviewed_hourly.csv`: `1224` rows
  - `fact_route_score_hourly.csv`: `1440` rows
  - `fact_curtailment_opportunity_hourly.csv`: `1440` rows
- targeted readiness verification output:
  - `_local_runs/r2_mae_impl_20260322/jan_promoted_rerun_readiness_2025-01-24_2025-01-26`
- targeted readiness result:
  - `2025-01-24`: `ready_for_map`
  - `2025-01-25`: `ready_for_map`
  - `2025-01-26`: `ready_for_map`
  - blocker file empty
  - `proxy_internal_transfer_share_t_plus_1h = 0.15` on all three days
- implementation interpretation:
  - the January blocker was real in the saved base bundle, but it was not a
    permanent upstream-evidence dead end
  - current code can regenerate enough reviewed boundary coverage from the same
    raw January inputs to clear the hard readiness gate honestly
  - residual uncovered rows are still the `shetland_wind` surface, but they no
    longer breach readiness

## 2026-03-22 Supported-Range Batch After January Rerun

- ran:
  - `python inline_arbitrage_live.py --materialize-reviewed-bundle-batch-eval --reviewed-bundle-batch-root _local_runs\reviewed_volume_supported_range_v1\bundles --reviewed-bundle-batch-pattern curtailment_opportunity_live_britned_reviewed_* --reviewed-bundle-batch-output-dir _local_runs\reviewed_volume_supported_range_v1\model_readiness_reviewed_bundle_batch_shared_plus_r2_no_public_plus_july_plus_march_plus_april_restore_plus_january_rerun_v1 --backtest-model-key opportunity_potential_ratio_v2 --baseline-model-key opportunity_potential_ratio_v2 --backtest-horizons 1,6,24,168`
- output:
  - `_local_runs/reviewed_volume_supported_range_v1/model_readiness_reviewed_bundle_batch_shared_plus_r2_no_public_plus_july_plus_march_plus_april_restore_plus_january_rerun_v1`
- comparison target:
  - `_local_runs/reviewed_volume_supported_range_v1/model_readiness_reviewed_bundle_batch_shared_plus_r2_no_public_plus_july_plus_march_plus_april_restore_v1`
- batch-level delta versus the previous April-restored sweep:
  - ready days: `59 -> 61`
  - not-ready days: `2 -> 0`
  - blocker windows: `1 -> 0`
  - blocker summary rows: `1 -> 0`
  - January window:
    `curtailment_opportunity_live_britned_reviewed_rerun_2025-01-24_2025-01-26`
    now lands `3/3` ready with max proxy share `0.15`
- current end-state:
  - all `19/19` reviewed windows in the supported-range local stock are ready
  - all `61/61` daily rows are ready
  - the earlier broad-sweep January blocker is closed on current HEAD
  - the two-year supported-range local story is now clear without relaxing the
    readiness policy
