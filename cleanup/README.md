# Generated Artifact Cleanup

This folder is the first-pass retention system for generated curtailment
artifacts. It is intentionally non-destructive.

The operating model is:

1. classify every in-scope generated directory as `keep`, `archive`, or
   `delete`
2. validate that classification with a dry-run
3. review the report once
4. only then write a later execution script

## Files

- `curtailment_generated_artifacts_manifest_v1.csv`
  - tracked inventory for current curtailment-generated top-level directories
- `bmu_generated_artifacts_manifest_v1.csv`
  - tracked inventory for BMU truth-history and BMU example outputs
- `opportunity_backtest_generated_artifacts_manifest_v1.csv`
  - tracked inventory for standalone `opportunity_backtest_*` snapshots
- `dry_run_generated_artifact_cleanup.py`
  - non-mutating validator and report generator
- `execute_generated_artifact_cleanup.py`
  - separate execution path for `archive` or `delete` after a final dry-run review

## Manifest columns

- `path`
- `artifact_family`
- `artifact_kind`
- `action`
- `authority_state`
- `replacement_path`
- `archive_destination`
- `git_tracked_expected`
- `reason`
- `notes`

`action` is one of:

- `keep`
- `archive`
- `delete`

`archive_destination` is only used for `archive` rows and must stay under
`_local_archive/generated_outputs/`.

## Local-only policy

The eventual archive target is local only:

- `_local_archive/generated_outputs/` for retained local snapshots
- `_local_runs/` for transient new runs and exploratory outputs

Both paths are gitignored. The point is to remove superseded tracked snapshots
from the repo while still leaving a nearby local escape hatch during cleanup.

## Dry-run usage

```bash
python cleanup/dry_run_generated_artifact_cleanup.py ^
  --repo-root . ^
  --manifest-path cleanup/curtailment_generated_artifacts_manifest_v1.csv ^
  --show all ^
  --format table
```

For the BMU truth-history family, pass an explicit scope prefix:

```bash
python cleanup/dry_run_generated_artifact_cleanup.py ^
  --repo-root . ^
  --manifest-path cleanup/bmu_generated_artifacts_manifest_v1.csv ^
  --scope-prefix bmu_ ^
  --show all ^
  --format table
```

The script fails if:

- a manifest row points at a missing path
- an in-scope top-level directory is not classified
- an archive destination is outside `_local_archive/generated_outputs/`
- a replacement path is named but missing

The script does not move, delete, rename, or rewrite scoped artifacts.

## Execution usage

Execution is intentionally separate from the validator and requires a final
review token:

```bash
python cleanup/execute_generated_artifact_cleanup.py ^
  --repo-root . ^
  --manifest-path cleanup/curtailment_generated_artifacts_manifest_v1.csv ^
  --action archive ^
  --execute ^
  --confirm-final-review reviewed-dry-run
```

Use `--action archive` and `--action delete` as separate runs. Without
`--execute`, the executor prints the validated action plan only.

For non-curtailment families, add the same `--scope-prefix` values used during
the dry-run.

For the standalone backtest family, use `opportunity_backtest` as the scope
prefix:

```bash
python cleanup/dry_run_generated_artifact_cleanup.py ^
  --repo-root . ^
  --manifest-path cleanup/opportunity_backtest_generated_artifacts_manifest_v1.csv ^
  --scope-prefix opportunity_backtest ^
  --show all ^
  --format table
```
