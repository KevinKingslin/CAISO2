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

To disable optimization history throughout the method:

```yaml
method:
  use_history: false
```

In the shared `method/default.yaml`, set the top-level `use_history` instead.
The default is `true`. With `false`:

- Block selection receives no summary of past optimization attempts.
- Earlier rejections do not filter out candidates, including preliminary
  candidates used to induce a block constitution.
- Prompt generation receives no earlier candidate instructions/scores and skips
  scoring performed solely to build that feedback. Complete joint candidates
  are still scored, including during workflow compatibility search; selected
  candidates still pass the full-system validation gate.
- Workflow proposals receive no recent-performance feedback (workflow search
  remains disabled for the benchmark experiments).

The current accepted system, constitutions, validation/acceptance rules and
workflow scheduling counters remain part of the algorithm's state. DICAI's
shared debate history is independent and stays enabled. Logs, plots, checkpoints
and optimization-attempt records are still saved for auditing and resuming;
they are not used as optimization feedback when this switch is off.

With the master `use_history: true`, you can disable history for block selection
alone:

```yaml
method:
  block_selection:
    use_history: false
```

The selection-only default is `true`. With `false`, selection receives an empty
history object; its constitution, diagnostic trajectories and available blocks
are unchanged.
History summaries are not generated for the judge. Optimization attempts remain
in checkpoints and continue to prevent repeating rejected candidates in the same
system state when the master switch is enabled. The selection-only setting
does not disable prompt-search feedback. The master `use_history: false`
overrides it, even when `block_selection.use_history` is `true`.

`iterations/<iteration>/03_selection/status.json` records `history_enabled`;
`block_selection.json` records the history actually supplied to the judge. Set `true`
to enable selection history again when the master is enabled. Block-constitution
and block-optimization phase status files also record the master setting as
`history_enabled`. Use a new run when changing either setting. Older
checkpoints that omit the keys can resume with the equivalent `true` defaults when
all other fingerprints match.

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

To reuse the current system's saved validation score during block acceptance,
set this override in an experiment:

```yaml
method:
  block_optimization:
    reevaluate_current: false
```

The default `true` evaluates current and candidate systems on each acceptance
comparison. With `false`, the initial baseline is still evaluated once and every
candidate is evaluated normally. Accepted candidate scores become the saved
current score; rejections retain the previous score, including after resume.
For 25 questions and one pass, an acceptance comparison then uses 25 fresh
system runs instead of 50. Attribution and diagnostic evaluations keep their
own batches. Reuse retains a historical measurement of stochastic task outputs.

The phase's `validation_metrics.json` marks current scores with `reused: true`
and `source: saved_current_score`; no fresh current trajectories or coverage are
invented. State/validation fingerprints are included. The CLI phase also says
when it is reusing the score. Change to `true` to restore fresh comparisons.
Start a new run when changing this protocol; checkpoint configuration/data
fingerprints still guard resume compatibility.
Existing checkpoints that omit this new key can resume with `true` when every
other setting, data split, provider and library fingerprint matches.

To exclude each optimizable component's current value from the candidate menus:

```yaml
method:
  block_optimization:
    include_current_configuration: false
```

The default is `true`. With `false`, prompt, model-selection, hyperparameter and
weight-adapter proposals must differ from the component's current execution
value. Metadata-only differences do not qualify. The filter applies to both
preliminary preference proposals and the final block search. Every eligible
joint candidate changes all optimizable members of its selected block; fixed
components retain their required values. If any optimizable member has no
changed option (for example, DPO has no distinct preference pairs), that proposal
pass returns no candidates instead of reinserting the current configuration.
With the toggle off, `max_joint_candidates` counts only changed combinations.

Current configurations remain available as proposal context and scoring
references. Full-system acceptance validation still determines whether to apply
an update. This toggle is separate from `reevaluate_current`, which controls
whether acceptance refreshes the incumbent's score. Set the new toggle to `false`
before starting a new run. Older checkpoints missing either default toggle can
resume with equivalent `true` settings; other fingerprint checks remain strict.

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
