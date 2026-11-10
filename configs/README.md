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

Original `scripts/configs/` files remain usable by the legacy launchers. Shared
and packaged method defaults use task-only candidate scoring:

```yaml
block_optimization:
  task_scoring:
    enabled: true
    constitution_weight: 0.0
    task_weight: 1.0
```

This uses the end-to-end training task metric for prompt feedback and candidate
selection, skips candidate constitution judging, and records `constitution_score`
as `null`. Constitution text still guides prompt generation and block selection;
the fixed validation set remains the separate acceptance gate. A positive
`constitution_weight` opts into mixed ranking with local constitution feedback;
`enabled: false` restores constitution-only scoring.

Reward-model/PPO settings in those files are unused by CAISO. Its existing DPO
settings remain in the shared method file.

Leiden blocks contain at most two components by default. To change the cap in an
experiment, override:

```yaml
method:
  leiden:
    max_comm_size: 2
```

In `method/default.yaml`, this is `leiden.max_comm_size`. Positive integers set
the maximum number of components in a block; `0` disables the cap. Singleton
blocks remain allowed. The cap is enforced during Leiden optimization, including
when the workflow is repartitioned. Use a new run when changing this setting.

The active HotpotQA profile uses a fixed seeded 50-question acceptance sample,
and PubMedQA/Amazon use the complete validation set. Each comparison uses two
fresh repetitions. Attribution and block-selection diagnostics use training
questions (`diagnostics_split: train`), keeping acceptance examples out of prompt
feedback and scheduling. The corresponding settings are:

```yaml
method:
  validation_sampling: seeded
  validation_batch_size: 50
  seed: 42
  block_optimization:
    validation_repeats: 2
```

The subset is sampled from the complete validation split and stays fixed for the
initial baseline and every current/candidate acceptance comparison. Keep your
existing `validation_batch_size` to retain its sample count; change `seed` to
choose a different subset. `diagnostics_split: validation` explicitly restores
the historical validation-based scheduling behavior for ablations.

For acceptance on the complete validation split, set
`method.validation_batch_size: all`. `validation_sampling: first` retains the
released OPTIMAS-compatible prefix behavior. Changing the protocol or seed
requires a new run; saved checkpoints verify their configuration and data hashes.

Attribution reuse is bounded independently of workflow-search plateau settings:

```yaml
method:
  attribution:
    reuse_unchanged: true
    refresh_after_unsuccessful_iterations: 5
```

An accepted change is measured immediately. After five unsuccessful iterations
since the last measurement, attribution is refreshed on new training questions
with fresh executions. Rejections and iterations without new candidates count;
the counter resets only after a successful measurement and survives resume.
Earlier v3 checkpoints without the counter force one fresh measurement when
resumed with the default limit. Explicitly changing the limit requires a new run.
See [refresh artifacts and budget behavior](../docs/OPTIMIZATION_IMPROVEMENTS.md).

To omit history from block selection:

```yaml
method:
  block_selection:
    use_history: false
```

In the shared `method/default.yaml`, set `block_selection.use_history` instead.
The default is `true`. With `false`, block selection receives an empty history
object. Its constitution, diagnostic trajectories and available blocks remain
available. All other stages keep their history:

- Prompt generation retains a checkpointed archive of earlier instructions and
  rescores up to three on the current screening batch before showing their scores.
  Within-call feedback and the current instruction's baseline score are retained.
- Earlier rejections still filter candidates in the same system context,
  including preliminary candidates used to induce a block constitution.
- Workflow proposals retain recent-performance feedback (workflow search
  remains disabled for the benchmark experiments).
- DICAI retains its shared debate history.

This is the only history toggle. The duplicate top-level `use_history` setting
has been removed; move old top-level overrides into `block_selection.use_history`.
With `false`, summaries are not generated for the selection judge. Logs, plots,
checkpoints and optimization-attempt records are still saved.

`iterations/<iteration>/03_selection/status.json` records `history_enabled`;
`block_selection.json` records the history actually supplied to the judge. Set
`block_selection.use_history: true` to enable selection history again. Block-constitution and
block-optimization phase status files keep `history_enabled: true` independently.
Use a new run when changing the setting. Older checkpoints that omit the selection
key or include the removed top-level `true` default can resume when all other
fingerprints match.

Block constitution induction is disabled by default:

```yaml
method:
  block_constitution:
    enabled: false
```

With the default `false`, each iteration skips preliminary preference
candidates, block preference collection, and block DICAI induction. It generates
and scores candidates once using the system constitution, including preference
labels used for DPO weight optimization. Full-system validation still decides
whether to accept an update. The system constitution is still induced normally.
The existing `04_block_constitution/status.json` records the skipped stage, and
`05_block_optimization/status.json` records `constitution_source: system`.
Use a new output directory when changing this setting; saved checkpoints require
the same method configuration. Set `true` to opt into block-specific constitutions.

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
comparison. With `false`, the initial baseline is still evaluated with the
configured repetition count and every
candidate is evaluated normally. Accepted candidate scores become the saved
current score; rejections retain the previous score, including after resume.
For 50 questions and two passes, an acceptance comparison then uses 100 fresh
system runs instead of 200. Attribution and diagnostic evaluations keep their
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
original DICAI prompts are unchanged. Bootstrap ties now abstain under the
[current search protocol](../docs/OPTIMIZATION_IMPROVEMENTS.md). Changing this
configuration requires a new run/output directory.

```sh
./scripts/caiso run amazon
./scripts/caiso evaluate amazon --unoptimized
./scripts/caiso evaluate hotpotqa --interactive
./scripts/caiso evaluate results/amazon-run --state current
./scripts/caiso run --config configs/experiments/amazon.yaml
```

For terminal entry of component prompts, hyperparameters, model selectors, and
existing generation settings, use `--interactive` or set
`evaluation.interactive: true` in an experiment profile. Enter keeps the current value; `/multiline` starts
a multiline prompt, ended by `/end` on its own line. Manual evaluations default
to the test split and a new `results/<dataset>-manual-<id>/` folder with
`component_config.json`, `manual_system.pth`, and `evaluation.json`. Use `--output`
to choose a new folder explicitly. Interactive mode uses dataset/YAML targets
without a saved state or optimization-run directory.

Run-directory evaluation supports `--state current`, `--state best` and
`--state final`. The first two freeze a saved CAISO checkpoint and work while
optimization is running. A checkpoint must already exist. With no flag, the CLI
uses the final export if present, otherwise the current checkpoint. Checkpoint
evaluations have separate `evaluation_current.json` / `evaluation_best.json`
reports, with the evaluated iteration and state fingerprint. Each invocation's
state/settings snapshot and logs are kept in its own internal evaluation folder.
