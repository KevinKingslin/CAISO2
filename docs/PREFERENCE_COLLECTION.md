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
8. Send both actual trajectories and the preference label to DICAI.

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
DICAI's existing projection hides task scores, targets, gold and evaluation
metadata from model requests. The 15 DICAI prompt templates are unchanged.

Only fresh component calls consume execution cost; prefix replay increments the
reuse counter. Each suffix completion counts as one completed system execution;
prefix capture does not. Only training inputs can use this sampling path.
Variables, generation settings and call observers are restored after every
branch, including failures. Explicit policy rejections skip the comparison
without replacement draws. Service and budget failures propagate through normal
initialization error handling.

`tests/test_preference_collection.py` verifies two actual configuration runs,
shared upstream execution, random component selection, single-run metric labels,
random exact ties including identical outputs, score redaction, parameter
restoration, error handling and optimizer integration without paid model calls.
