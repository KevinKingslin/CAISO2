# Initial DICAI preference collection

The default `preference_collection.mode: component` follows the requested
unchanged/changed configuration comparison. The protocol is
`configuration-pair-v2`.

## Procedure

1. Deduplicate training inputs and seed-shuffle their order.
2. Independently select an eligible component at random for each sample. Eligible
   components have an alternate declared configuration or a temperature setting
   different from `preference_collection.temperature`.
3. Change that component only: randomly choose an alternate value from its
   declared search space, or use the configured changed temperature when there is
   no alternate discrete value. Prompts are not rewritten.
4. Execute the upstream prefix once under the current configuration and freeze
   its outputs. The selected component is not run during prefix capture.
5. Run branch A (unchanged configuration) once from that component onward.
6. Run branch B (changed configuration) once from the same captured prefix.
   The selected component and all subsequent components execute afresh. The
   downstream configurations remain normal. No candidate output is frozen.
7. Label the branch with the higher objective task metric as preferred. On exact
   equality, choose A or B at random with equal probability. Keep the pair even
   when the outputs are identical. No score-margin threshold is applied.
8. Send both actual trajectories, the preference label and separate training
   task-score feedback to DICAI's reasoning and debate stages.

Each comparison therefore costs one upstream-prefix execution plus two suffix
executions. There is no extra full-system reference execution, four-candidate
pool, repeated downstream continuation scoring, or averaging. When the first
component is selected, the prefix is empty and the two branches are full runs.
Component selection and tie-breaking use seeded RNGs. The benchmark adapter
continues to leave actual model generations unseeded.

## Configuration

```yaml
initial_preference_pairs: 30
initial_preference_attempts: 200
preference_collection:
  mode: component
  temperature: 0.9
```

The temperature is the changed branch's actual setting, not a lower bound.
Components whose current temperature already equals it need an alternate declared
configuration to be eligible. `temperature: null` disables temperature changes
but still permits alternate discrete configurations. The collector stops at its
pair target, attempt cap or run budget. With `initial_preference_attempts: -1`, it
visits at most `initial_preference_pairs` distinct training inputs once, selecting
one random component per sample. A nonempty smaller collection is usable.

`mode: rerun` retains the original two-full-run ablation, which skips ties and
uses `initial_preference_min_margin`. That margin does not affect the default
configuration-pair collector. The former `candidates`, `downstream_rollouts` and
`pairs_per_context` controls are removed because the requested procedure always
uses exactly two configurations and one run per branch.

Supplied `caiso.preferences_path` data bypasses collection. Start a new output
directory for this protocol; previous checkpoints retain their saved datasets.
The optional block-constitution collector remains separate and disabled by
default.

## Evidence and accounting

Retained pairs contain two full recorded trajectories, not scores attached to
representative draws. A is the unchanged branch; B is the changed branch. Pair
metadata records the changed component/parameter, task scores, whether the scores
were tied, and `label_source: random_task_tie` or `training_task_metric`.
`setup/bootstrap_comparisons.jsonl` records each collected pair and filtered
comparison. Preferences also appear in `setup.json` and run checkpoints.
DICAI's trajectory projection hides task scores, targets, gold and evaluation
metadata. Reasoning and every debate variant separately receive a training-only
`score_feedback` object read from `metadata.global_scores` in A/B order. It
contains `metric`, `score_a`, `score_b`, `delta_a_minus_b` and `relation`
(`A_better`, `B_better` or `tie`). Scores never reorder candidates or replace
the supplied preference label. Pairs without stored scores receive null feedback;
malformed or nonfinite score records fail before induction requests.

`dicai.score_metric` names the recorded task metric; it does not change the
benchmark evaluator. Experiment loading defaults to `answer_f1` for HotpotQA
and `task_metric` otherwise. Override it in the experiment's method settings
when supplying another metric. For example:

```yaml
method:
  dicai:
    score_metric: answer_f1
```

The prompts use absolute scores and their gap as outcome evidence, without
treating a gap as confidence or proof of component causality. Principles must
remain applicable without scores or reference answers. Exact ties still enter
reasoning/debate and retain their existing randomly assigned A/B labels and
label-reconstruction gate. Feedback explicitly identifies the lack of a measured
task-metric advantage; this extension does not introduce tie filtering or a
ternary judge.

The shuffled reconstruction judge remains blind to scores, reference answers
and labels. After inference, each scored choice records `selected_score`,
`oracle_score` (the larger candidate score) and `regret` (oracle minus selected).
Either choice on a score tie has zero regret. Reports retain original label
`accuracy`, including ties, and add `non_tied_accuracy`/`non_tied_n` over scored
non-tied pairs, `tied_pairs`, `scored_pairs`, `scored_selections`,
`mean_selected_score`, `mean_oracle_score` and `mean_regret`. Score means use
only scored pairs with a completed choice; missing choices have null values,
not zero regret, and the coverage counts expose their exclusion. These are
training reconstruction diagnostics, not held-out benchmark results.

When optional `dicai.minimization.enabled` is true, fully scored comparison
sets use mean regret instead of label accuracy. Non-strict minimization permits
equal regret; strict minimization requires lower regret. Neither may remove a
principle when the baseline or trial has incomplete scored selections. Sets
with any unscored pair retain accuracy-based minimization. Existing component
coverage restoration and the nonempty constitution requirement remain active;
final reconstruction evaluates the list after coverage restoration.

The score-feedback prompt adaptation has its own protocol/provenance version.
The current component-format version is
`democratic-icai-c8d5-source-trajectory-v7-component-format`. Every induction
pair must include recorded execution component IDs; the current collector
already supplies them. Reasoning, debate, abstraction and repairs require
`Select the trajectory where <component_id> <condition>`. Primary subject IDs
are validated against executed/source components, including fallback and final outputs.
Start a fresh output directory to regenerate constitutions; existing checkpoints
retain saved constitutions. Existing pair JSON can be supplied using
`caiso.preferences_path` without recollecting trajectories. Original DICAI source
and reference prompts remain unchanged.

Only fresh component calls consume execution cost; prefix replay increments the
reuse counter. Each suffix completion counts as one completed system execution;
prefix capture does not. Only training inputs can use this sampling path.
Variables, generation settings and call observers are restored after every
branch, including failures. Explicit policy rejections skip the comparison
without replacement draws. Service and budget failures propagate through normal
initialization error handling.

`tests/test_preference_collection.py` verifies two actual configuration runs,
shared upstream execution, random component selection, single-run metric labels,
random exact ties including identical outputs, blind score redaction, parameter
restoration, error handling and optimizer integration without paid model calls.
`tests/test_dicai_score_feedback.py` covers score-aware reasoning/debate, blind
selection, regret accounting, and regret-preserving minimization.
