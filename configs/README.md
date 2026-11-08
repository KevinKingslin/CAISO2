# CAISO configuration

`scripts/caiso` selects `experiments/amazon.yaml`, `experiments/hotpotqa.yaml`, or
`experiments/pubmedqa.yaml` for both evaluation and optimization.

- `method/default.yaml`: shared algorithm, DICAI, attribution, clustering,
  optimization, DPO, budget, error-handling and reporting settings.
- `experiments/*.yaml`: benchmark identity, launch/evaluation settings, and
  dataset-specific batches, persona catalogs, and method overrides.
- `providers.yaml`: validated Azure request/concurrency caps, retry limits,
  backoff and timeout, with optional per-deployment overrides. Defaults match the
  previous CAISO transport; credentials remain in `.env`.
- `method_config`: path relative to the experiment YAML.
- `method`: recursively overrides the shared method; unspecified nested fields keep
  their shared values.

Values were copied without retuning. Original `scripts/configs/` files and packaged
`caiso/config/default.yaml` remain unchanged and usable by the legacy launchers.
Reward-model/PPO settings in those files are unused by CAISO. Its existing DPO
settings remain in the shared method file.

To replace the default validation prefix with a reproducible random subset,
change these keys inside the experiment's existing `method` block:

```yaml
method:
  validation_sampling: seeded
  validation_batch_size: 10
  seed: 42
```

The subset is sampled from the complete validation split and stays fixed for the
initial baseline and every current/candidate acceptance comparison. Keep your
existing `validation_batch_size` to retain its sample count; change `seed` to
choose a different subset. Attribution and block-selection diagnostics sample
their independently configured mini-batches from the full validation pool.

For acceptance on the complete validation split, set
`method.validation_batch_size: all`. `validation_sampling: first` retains the
released OPTIMAS-compatible prefix behavior. Changing the protocol or seed
requires a new run; saved checkpoints verify their configuration and data hashes.

To skip block constitution induction, add this override to an experiment:

```yaml
method:
  block_constitution:
    enabled: false
```

The default is `true`. With `false`, each iteration skips preliminary preference
candidates, block preference collection, and block DICAI induction. It generates
and scores candidates once using the system constitution, including preference
labels used for DPO weight optimization. Full-system validation still decides
whether to accept an update. The system constitution is still induced normally.
The existing `04_block_constitution/status.json` records the skipped stage, and
`05_block_optimization/status.json` records `constitution_source: system`.
Use a new output directory when changing this setting; saved checkpoints require
the same method configuration. Restore `true` to enable block induction again.

Keep credentials, deployments, backend choices and service URLs in `.env`.
Prompt text is under [`caiso/prompts`](../caiso/prompts/README.md). `prompt_tips`
remains a method parameter with its existing values.

Each shipped experiment selects three dataset-specific expert profiles through
`caiso.persona_catalog`: `personas/amazon`, `personas/hotpotqa`, or
`personas/pubmedqa`. Edit the full profile text in
`caiso/prompts/systems/<dataset>/personas/*.txt`. The same three profiles are used
for both system and block constitution induction, in the fixed order of
chain-of-thought, self-consistency, and self-refinement reasoning strategies.

The legacy optimizer profiles also select these defaults by dataset. An explicit
`caiso.personas` list overrides the catalog and must have exactly three entries.
Resolved persona names and complete prompt text are included in saved experiment
configuration and DICAI cache provenance. Persona changes apply to new inductions;
start a new run to regenerate constitutions already stored in a checkpoint.

DICAI uses the supplied original code-based prompts documented in
[`docs/PROMPT_AUDIT.md`](../docs/PROMPT_AUDIT.md). Configured self-consistency
samples, refinement steps, reconstruction votes and debate gates are active.

CAISO's semantic one-quality check is configured under the shared `dicai` section
(or under an experiment's `method.dicai` override):

```yaml
dicai:
  atomicity:
    enabled: true
    batch_size: 20
    max_revisions: 2
```

It checks new chairperson lists and the abstracted constitution using the existing
induction model. Compound debate principles are split; compound abstractions are
revised to one shared quality per cluster. Valid prior lists and verified cluster
medoids provide recorded fallbacks when revisions fail. Checks and revisions
consume the existing model-call budget and are checkpointed. `enabled: false`
is available for ablations. `batch_size` limits principles per semantic request;
it is unrelated to the number of validation questions. Disabling the check
restores the original format/reconstruction-only behavior for ablations. The
original DICAI prompts and random task-tie policy are unchanged. Changing this
configuration requires a new run/output directory.

```sh
./scripts/caiso run amazon
./scripts/caiso evaluate amazon --unoptimized
./scripts/caiso evaluate results/amazon-run --state current
./scripts/caiso run --config configs/experiments/amazon.yaml
```

Run-directory evaluation supports `--state current`, `--state best` and
`--state final`. The first two freeze a saved CAISO checkpoint and work while
optimization is running. A checkpoint must already exist. With no flag, the CLI
uses the final export if present, otherwise the current checkpoint. Checkpoint
evaluations have separate `evaluation_current.json` / `evaluation_best.json`
reports, with the evaluated iteration and state fingerprint. Each invocation's
state/settings snapshot and logs are kept in its own internal evaluation folder.
