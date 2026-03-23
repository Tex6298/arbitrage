# Map Operational Posture

This document records the current operational decision for the curtailment map
against the authoritative local evidence set.

## Decision

The supported-range local evidence set no longer blocks promotion.

The earlier exploratory-only posture is withdrawn after the January 2025 rerun
closed the last readiness blocker on current HEAD.

## Basis

The original narrow-stock decision was based on these authoritative outputs:

- `model_readiness_reviewed_bundle_batch_authoritative_rerun`
- `model_readiness_oct_2024_drift_policy_v3`
- `model_readiness_dec_2024_event_lifecycle`

Those outputs still show that the previously curated local windows are strong:

- all informative local reviewed windows are passing readiness
- October 1-7, 2024 is `7/7` ready days in the authoritative rerun
- December 7-9, 2024 is `3/3` ready days after event-lifecycle hardening
- November 22-24, 2024 is `3/3` ready days
- the retired `v3` specialist remains rejected in
  `model_readiness_gb_nl_shadow_suite_v1`

The broader supported-range sweep now lands cleanly:

- 19 reviewed windows were evaluated in `_local_runs/reviewed_volume_supported_range_v1`
- 11 of those windows are informative in the latest combined supported-range
  rerun
- 19 of 19 windows are now fully ready
- 61 of 61 daily rows are ready
- the blocker summary is empty in
  `_local_runs/reviewed_volume_supported_range_v1/model_readiness_reviewed_bundle_batch_shared_plus_r2_no_public_plus_july_plus_march_plus_april_restore_plus_january_rerun_v1`
- the repeated 2025 `R2_netback_GB_NL_DE_PL` failures are closed on the current
  local branch
- the January 2025 window now clears readiness through the rerun bundle
  `curtailment_opportunity_live_britned_reviewed_rerun_2025-01-24_2025-01-26`

The remaining January detail is no longer a blocker. `shetland_wind` still
uses proxy-backed internal-transfer evidence on some rows, but the rerun lifts
reviewed coverage enough that the daily proxy share is `0.15` on all three
days and the hard readiness gate still passes without policy relaxation.

## Operating Guardrails

Promotion or operational use should keep these guardrails in place:

- treat `fact_model_readiness_daily` as a hard gate
- only show map states as operational when `model_ready_flag = true`
- surface readiness state and blocking reasons in the UI or downstream consumer
- do not weaken the proxy-share gate just because current local stock passes
- rerun the batch after any upstream reviewed-evidence expansion or major model
  behavior change

The map generator now exposes both aliases:

- `--materialize-operational-cluster-map`
  - readiness-gated internal operational alias
- `--materialize-exploratory-cluster-map`
  - compatibility exploratory alias over the same aggregated map tables

## Build Command

Example operational build against the latest clean local supported-range stock:

```powershell
python inline_arbitrage_live.py `
  --materialize-operational-cluster-map `
  --opportunity-input-path _local_runs\reviewed_volume_supported_range_v1\bundles\curtailment_opportunity_live_britned_reviewed_rerun_2025-01-24_2025-01-26 `
  --operational-map-readiness-path _local_runs\r2_mae_impl_20260322\jan_promoted_rerun_readiness_2025-01-24_2025-01-26 `
  --operational-map-output-dir _local_runs\operational_cluster_map_january_ready_2025-01-24_2025-01-26
```

That command keeps the operational map tied to the same hard-gated readiness
output that cleared the January rerun window.

## What This Does Not Mean

This decision does not mean:

- the narrow curated stock is still useful as a design proof
- the broad supported-range story is now clean on current local stock
- January was resolved by regenerating stale local bundle artifacts with
  current code, not by weakening readiness policy
- some January Shetland rows still rely on proxy-backed evidence, but they are
  within the existing operational threshold
- there is no open `R2` MAE cleanup lane left on current HEAD

## Next Engineering Program

The next program is no longer another `R2` MAE lane on current HEAD.

Priority order:

1. if January Shetland evidence quality matters, expand reviewed coverage
   upstream and rerun the January bundle
2. keep promoting from the current hard-gated readiness outputs rather than
   from exploratory snapshots
3. rerun the supported-range batch after any future upstream evidence
   expansion or forecasting change
4. preserve the current readiness gate unless there is a deliberate policy
   decision to change it
