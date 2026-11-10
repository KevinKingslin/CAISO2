# Initial DICAI preference collection

The default `preference_collection.mode: component` collects objective training
preferences using the component-level sampling idea in
[OPTIMAS Section 4.1](https://arxiv.org/html/2507.03041#S4.SS1).
It generates real alternative component outputs through higher-temperature
decoding or alternate declared configurations. It does not generate new prompts,
inject a reference answer as an execution, or use a constitution to label ties.
The collection protocol is `component-continuations-v1`.

## Execution and scoring

1. Deduplicate training inputs and seed-shuffle their order and the optimizable
   components. Visit every component on a question before advancing to the next.
2. Execute the normal system once for that question, capturing the reference
   execution. Reuse it across the question's component pools.
3. For the selected component, retain its original output and sample up to
   `candidates - 1` alternatives against its captured prefix. Components with a
   declared search space try alternate configurations in seeded order. LLM
   components exposing a temperature setting use at least the exploration
   temperature; `null` requests no override. Model services can still restrict
   temperature support. Prompts and weights are not rewritten or trained here.
4. Deduplicate by the sampled component's actual output. A configuration change
   that produces the same output does not create another candidate.
5. Freeze the candidate output and all non-descendant outputs from the reference.
   Execute its descendants afresh, under the normal downstream configuration,
   for `downstream_rollouts` repetitions. Validate replayed inputs/configurations.
6. Compute each output's mean task metric. Consider every unordered candidate
   pair, independent of which array index has the higher score. Retain pairs with
   differences strictly greater than `initial_preference_min_margin`, prioritizing
   larger gaps and capping each pool with `pairs_per_context`. Randomize A/B order
   using the collection seed.

The mean estimates downstream performance conditional on a particular frozen
output and context. It is not the score of a representative single trajectory or
an estimate of the configuration's unconditional accuracy. Three continuations
are a bounded default, not a statistical confidence guarantee. Seeds control
question/configuration selection and identify repeats; the benchmark adapter
does not add model-generation seeds, so repeated calls are not deterministic.

Sampling is restricted to training examples. Exploratory variables, generation
settings and call observers are restored even after failure. Content-policy
rejections are logged without replacement draws. A candidate lacking its full
requested continuation count is excluded from comparisons. Service and budget
failures propagate through the normal initialization failure handling.

## Configuration and artifacts

The shared and packaged method YAMLs default to four candidates including the
original, exploration temperature 0.9, three downstream continuations, and one
retained pair per pool. The existing defaults target 30 retained pairs, cap 200
question/component pools, and require a mean-score difference greater than 0.01.
If fewer than 30 usable pairs are found, a nonempty set is used. No usable pairs
causes a clear initialization error rather than fabricated preferences.

`initial_preference_attempts: -1` performs one pass through all components on up to
`initial_preference_pairs` distinct training questions, still stopping at the
retained-pair target. `preference_collection.mode: rerun` is the earlier collector:
two full runs of one unchanged configuration, with its original attempt policy.
Explicit `caiso.preferences_path` data bypasses either collector. Existing run
checkpoints retain their saved preferences; start a new output directory for the
new resolved configuration.

The run's `setup/bootstrap_comparisons.jsonl` records candidate pools, sampling
settings, each real continuation, individual task rewards, means, retained pair
IDs and abstention reasons. Retained preferences are saved in `setup.json` and
checkpoints. Their candidate records contain the sampled prefix plus all actual
continuations. Their metadata holds the mean scores, per-run scores, label source
and changed configuration. DICAI's existing evidence projection strips reward,
target, gold, seed and evaluation metadata from every continuation before
reasoning or blind reconstruction. Component coverage includes observed IDs in
both the prefix and continuations. The 15 DICAI prompt files are unchanged.

Only fresh calls charge component execution cost; replayed records increment the
reuse counter, and each completed continuation counts as a completed system
execution. A sampling call stopping at its producer does not count as a completed
system execution. The reference baseline also consumes the normal budget.

This collector is for initial system-constitution induction through the benchmark
`UpstreamRuntime`. The separate optional block-constitution collector continues
to compare proposed block configurations and remains disabled by default. Its
existing constitution-based tie handling is not used by this initial collector.

## Validation

`tests/test_preference_collection.py` exercises actual synthetic upstream
components, including a branch not downstream of the sampled producer. It checks
generation-setting restoration, alternate hyperparameters, frozen candidates,
fresh continuations, averaged labels, policy/service/budget failures, score
redaction, and training-only execution. No paid model calls are needed.
