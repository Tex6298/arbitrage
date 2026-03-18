# GB Curtailment Lineage

This document is the short design history for the GB curtailment and readiness
stack. It answers four practical questions:

- What is the current canonical path?
- What older approaches still exist in the repo?
- Why were they replaced?
- Which output directories should be treated as history versus authority?

## Current canonical stack

The current production-intent path is:

1. `route_score_history.py`
   - deterministic route evidence, gating, connector state, and reviewed versus
     proxy source selection
2. `curtailment_opportunity.py`
   - feature surface and origin-time opportunity state
3. `opportunity_backtest.py`
   - canonical baseline model `opportunity_potential_ratio_v2` plus narrow
     hardening rules for known reviewed-event regimes
4. `model_readiness.py`
   - map-readiness gate, blocker prioritization, drift surface
5. `benchmark_suite.py`
   - manifest-driven benchmark suite, scout gate, and batch evaluation
6. `exploratory_cluster_map.py`
   - exploratory UI only, driven from opportunity plus readiness outputs

The current baseline is still `opportunity_potential_ratio_v2`. The repo now
hardens that baseline with narrow event-lifecycle logic instead of introducing
new broad specialist models.

## Evolution phases

## 1. Netback and route scaffolding

The repo started as an inline arbitrage netback tool:

- `inline_arbitrage_live.py`
- `physical_constraints.py`
- `asset_mapping.py`
- `gb_topology.py`

This phase established route legs, border economics, and the first physical
assumption register. It did not yet have a formal historical readiness gate.

## 2. Historical opportunity surface and first baseline

The next phase added historical opportunity materialization and a first simple
baseline:

- `curtailment_signals.py`
- `route_score_history.py`
- `curtailment_opportunity.py`
- `opportunity_group_mean_notice_v1`

`opportunity_group_mean_notice_v1` was useful as a first auditable baseline, but
it was too coarse for reviewed open and close regimes, and too brittle when the
evidence mix changed. It remains in the repo as a historical baseline only.

## 3. Canonical baseline `v2`

`opportunity_potential_ratio_v2` replaced the first baseline because it was a
better fit for the data and the pipeline:

- it predicts a deliverable ratio instead of a raw deliverable amount
- it stays bounded by deterministic origin-time potential
- it can fall back through auditable feature hierarchies
- it can accept narrow regime fixes without becoming a separate model family

This is still the active baseline today.

## 4. GB-NL specialist shadow candidate `v3`

The repo then introduced a narrow specialist candidate:

- `opportunity_gb_nl_reviewed_specialist_v3`

The intent was to beat `v2` on the reviewed `GB-NL` / `britned` blocker week,
especially the October 1-7, 2024 failure modes.

It was not promoted. The later benchmark suite showed:

- it did not beat `v2` on informative out-of-time holdouts
- some later windows were perfect-zero guardrails and could not count as
  promotion evidence
- once the suite was made forecast-only, `v3` still regressed

`v3` is now retired as an active candidate and kept only for archived compare
paths.

## 5. Benchmark and readiness hardening

Once the specialist experiment stalled, the repo shifted from model invention to
evaluation hardening:

- `model_readiness.py` became the formal gate for map readiness
- `benchmark_suite.py` added manifest-driven suite runs
- `fact_model_benchmark_window_scout` was added so later windows could be
  screened for real signal before being counted
- suite promotion scoring became forecast-only rather than blocker-driven
- reviewed-bundle batch evaluation was added to stop manual day-by-day loops

This phase is what turned the repo from ad hoc backtesting into a defensible
validation workflow.

## 6. Evidence plumbing

Several later issues were not model issues at all; they were evidence and
source-selection issues. The repo now has explicit plumbing for:

- France reviewed connector evidence
- GB reviewed boundary pass-through on non-tightening corridors
- Scotland north-to-south reviewed gap coverage
- a dedicated Shetland dependency reviewed source family, when auditable source
  rows exist

The rule of thumb is now explicit: new reviewed evidence is acceptable when it
is auditable and same-interval. Carry-forward hacks are not treated as reviewed
truth.

## 7. Narrow hardening of `v2`

After `v3` was retired, the repo fixed the remaining material problems by
hardening `v2` in place:

- opening guardrail for one-hour open events
- `R1` reviewed event-phase calibration
- `R1` persist-close suppressor
- `R2` reviewed lifecycle handling for October published-restriction shapes
- `R2` late-reopen handling for the October 1 reviewed late reopen
- drift-policy exceptions for well-predicted reviewed event shifts and first-day
  warmup artifacts

This is the key architectural decision in the current repo state: improve the
canonical baseline with narrow auditable regime handling, instead of spawning a
new specialist branch for every failure mode.

## Supersession register

| Item | Status | Superseded by | Why |
| --- | --- | --- | --- |
| `opportunity_group_mean_notice_v1` | historical baseline only | `opportunity_potential_ratio_v2` | too coarse for reviewed event regimes |
| `opportunity_gb_nl_reviewed_specialist_v3` | retired shadow candidate | enhanced `opportunity_potential_ratio_v2` | lost on informative later holdouts |
| `curtailment_opportunity_live_britned_reviewed_refresh_2024-10-01_2024-10-07` | historical refresh snapshot | `curtailment_opportunity_live_britned_reviewed_rerun_2024-10-01_2024-10-07` | rerun is the authoritative October reviewed-input bundle |
| early `model_readiness_reviewed_bundle_batch*` snapshots | historical point-in-time outputs | `model_readiness_reviewed_bundle_batch_authoritative_rerun` | batch autodiscovery now prefers authoritative reruns |

## What is still intentionally unresolved

Two items remain intentionally outside the active forecasting scope:

- January 24-26, 2025 remains a guardrail and data-coverage issue, not a model
  target
- Shetland reviewed dependency logic exists, but it stays dormant until real
  `shetland_island_link_dependency_review` rows are supplied

## Current working position

Today the repo should be read as:

- active model: enhanced `opportunity_potential_ratio_v2`
- active readiness gate: `model_readiness.py`
- active batch validation snapshot:
  `model_readiness_reviewed_bundle_batch_authoritative_rerun`
- active October authoritative rerun:
  `curtailment_opportunity_live_britned_reviewed_rerun_2024-10-01_2024-10-07`
- retired specialist experiment:
  `opportunity_gb_nl_reviewed_specialist_v3`

If a future change needs another model family, it should clear the existing
suite and readiness machinery rather than bypass it.
