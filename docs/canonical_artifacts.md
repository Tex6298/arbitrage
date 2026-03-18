# Canonical Artifacts

This document tells you where to start, which outputs are authoritative, and
how to interpret the naming of the many generated directories in the repo.

## Edit these code paths

If you are changing behavior, these are the main modules that matter:

- `route_score_history.py`
  - route evidence, delivery tiers, reviewed versus proxy gating
- `curtailment_opportunity.py`
  - feature surface and opportunity rows
- `opportunity_backtest.py`
  - baseline forecasting logic, narrow event handling, drift-window generation
- `model_readiness.py`
  - readiness gate, blocker prioritization, daily pass or fail
- `benchmark_suite.py`
  - manifest suites, scout runs, reviewed-bundle batch evaluation
- `exploratory_cluster_map.py`
  - exploratory map output only

Generated output directories are not source code. Treat them as snapshots.

## Current authoritative generated outputs

These are the generated outputs that best represent the current state of the
repo.

### October authoritative rerun

- `curtailment_opportunity_live_britned_reviewed_rerun_2024-10-01_2024-10-07`
  - authoritative October reviewed-input opportunity bundle
- `model_readiness_oct_2024_drift_policy_v3`
  - authoritative October readiness result after the final forecast and
    drift-policy fixes

### December authoritative readiness

- `model_readiness_dec_2024_event_lifecycle`
  - authoritative December readiness result after opening, persist-close, and
    drift-policy hardening

### Current authoritative batch rollup

- `model_readiness_reviewed_bundle_batch_authoritative_rerun`
  - authoritative batch evaluation over local reviewed BritNed bundles
  - batch autodiscovery now prefers `rerun` bundles over base bundles, and base
    bundles over `refresh` bundles for the same date range

### BMU truth snapshots

- `bmu_truth_history_phase6_remit_fix`
  - current authoritative BMU truth snapshot in local stock
- `bmu_truth_history_phase4_family_day`
  - retained single-directory reference snapshot for store-backfill examples
- `bmu_truth_history_phase4_family_day_daily`
  - retained daily-tree reference snapshot for store-backfill examples

### Archived benchmark compare

- `model_readiness_gb_nl_shadow_suite_v1`
  - archived shadow-suite evidence for the retired `v3` specialist
  - useful for design history, not for current promotion decisions

## Important non-authoritative patterns

These names usually indicate a snapshot that should not be treated as the final
word if a better artifact exists for the same window.

- `*_refresh_*`
  - exploratory refresh or intermediate rebuild
  - usually superseded by a later `rerun` for the same date range
- `*_smoke*`
  - smoke or sanity check only
- `*_test*`
  - test artifact only
- `*_demo*`
  - demonstration output only
- `*_shadow*`
  - compare or candidate-evaluation output, not the baseline production path

## Naming rules that now matter

These naming conventions are worth preserving because the tooling uses them:

- `*_rerun_*`
  - authoritative rerun for a previously materialized date range
- base directory with no suffix
  - first materialized version for that date range
- `*_refresh_*`
  - intermediate refresh; lower priority than base or rerun

For reviewed-bundle batch autodiscovery, the priority is now:

1. `rerun`
2. base
3. `refresh`

## Generated artifact retention

Generated outputs now have an explicit retention policy:

- `cleanup/curtailment_generated_artifacts_manifest_v1.csv`
  - operational inventory for `keep`, `archive`, and `delete`
- `cleanup/bmu_generated_artifacts_manifest_v1.csv`
  - the same retention inventory pattern for BMU truth-history artifacts
- `_local_runs/`
  - future home for transient exploratory, smoke, scout, and stepping-stone
    outputs
- `_local_archive/generated_outputs/`
  - local-only archive target for superseded generated snapshots

Only explicitly authoritative outputs should stay tracked at the repo root.
If an output is transient or superseded, it should land in `_local_runs/` or the
local archive rather than becoming another top-level tracked snapshot.

## Map outputs

Map outputs are still exploratory unless explicitly promoted later.

- `exploratory_cluster_map_*`
  - exploratory cluster-point time-slider outputs
  - these are not the operational product surface
  - regenerate them from the current opportunity and readiness artifacts rather
    than treating an old map directory as canonical

## Practical starting points

If you need the current state quickly:

1. Read [GB Curtailment Lineage](gb_curtailment_lineage.md)
2. Open `model_readiness_reviewed_bundle_batch_authoritative_rerun`
3. Open `model_readiness_oct_2024_drift_policy_v3`
4. Open `model_readiness_dec_2024_event_lifecycle`
5. Only then drop into the older shadow or refresh directories if you need
   history
