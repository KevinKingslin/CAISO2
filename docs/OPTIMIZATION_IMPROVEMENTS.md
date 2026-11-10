# Search protocol v3

These changes implement the optimization audit's requested repairs. They change
the optimization protocol, while retaining benchmark factories, initial prompts,
task inference interfaces, datasets, scorers and model choices. Pre-v3 checkpoints
fail the method fingerprint check. Earlier v3 checkpoints can resume with the
default attribution refresh limit; a checkpoint without its new counter forces
a fresh measurement before cached effects can be reused.

```sh
./scripts/caiso optimize hotpotqa --output results/hotpotqa-v3
```

## Evidence and validation

- Bootstrap comparison ties, identical trajectories and reward gaps at or below
  `initial_preference_min_margin: 0.01` abstain. They never receive random winners.
  Defaults request up to 30 usable pairs in at most 200 comparisons. Every attempt
  records its label or abstention reason in `setup/bootstrap_comparisons.jsonl`.
- An all-ambiguous bootstrap stops with a useful error instead of fabricating a
  constitution. For deterministic systems without differing task outcomes, supply
  genuinely labelled training preferences through `caiso.preferences_path` using
  the existing `PreferencePair` JSON schema. Fewer than the requested maximum
  pairs may be used if the bounded search finds some useful evidence.
- Active HotpotQA acceptance uses 50 fixed seeded questions sampled from all 250
  validation examples. PubMedQA uses all 25; Amazon uses all 60. Current and
  candidate each execute two fresh repetitions by default. The numerical minimum
  accepted mean improvement is 0.005. This is not a significance test.
- Attribution and block selection use training examples. Acceptance questions
  never appear in optimizer feedback or scheduling. The test split remains
  outside optimization. Active profiles perform three final test repetitions.
- Discrete joint-candidate selection uses argmax (`sample_discrete: false`).
  Explicit softmax exploration remains available as an ablation.

## Comparable feedback and history

The prompt generator receives bounded training evidence with reference targets,
final predictions, task scores, the selected component's inputs/outputs and
downstream outputs. Evidence is ordered by failure, with a successful contrast
when available. Reference answers are optimizer feedback only: runtime task
inputs and the benchmark's initial prompts are unchanged.

The current instruction is scored before proposals. A bounded archive persists
candidate instruction text across iterations and resume in
`checkpoint/checkpoint.json` under `optimizer_state.prompt_archive`. Up to three
archived instructions are rescored against the current accepted system on the
same current screening batch before their scores are shown. Historical scores
from different questions are never presented as comparable measurements.

`feedback_examples` (8), `feedback_max_chars` (2,000), `prompt_history_limit` (3)
and `prompt_archive_size` (30 per component) bound prompt and archive growth.
The archive guides proposals; it does not expand each component's three-new-
instruction menu or remove individually weak configurations before joint scoring.

## Search cost

Task-only search screens complete joint configurations on the same five training
questions. It promotes the three strongest changed states plus the unchanged
reference to all 20 training questions, evaluating only the missing 15 examples.
Every returned finalist has a score on the same complete batch. Screened-out
small-batch scores cannot compete against full-batch scores. Individually weak
changes remain eligible to cooperate during screening.

This staging applies to task-only scoring. Explicit mixed/constitution-only
modes retain their complete-batch comparison. DPO preference generation retains
its full training-reference batch even when task scoring is staged.

Training reference trajectories supply the unchanged configuration's score,
avoiding its duplicate evaluation. Within one proposal, component observations
can be replayed when example ID/input, repeat ID, actual component inputs,
configuration and generation settings agree. An input/configuration change
invalidates that component; changed outputs flow to its descendants. Identical
downstream inputs may share the same recorded output. This is shared stochastic
evidence, not additional independent samples.

Replay scopes are cleared on success and failure, refuse validation/test
examples, and never span acceptance. Policy rejections are retained within a
scope instead of resampled. Component mutations and callbacks still use the
original transactional apply/restore path. The runtime's serialization lock stays
in place.

Attribution's normal trajectories also supply matching diagnostic observations.
Edge effects are retained after rejected updates for at most five unsuccessful
iterations since the last measurement. Rejections and iterations with no new
candidate both count. On the fifth, attribution is refreshed immediately using
a new training sample and fresh executions; the counter then resets. New samples
prefer questions outside the previous measurement, reusing questions only when
the pool is too small for a full disjoint batch.

Accepted state changes trigger an immediate fresh measurement, including on the
last iteration. The accepted checkpoint is saved before that work, so a budget
limit or failed refresh cannot lose the accepted configuration. A failed periodic
refresh retains its expired counter and cannot permit stale reuse on the next
attempt. The counter persists across checkpoint/resume.

Reused attribution is explicitly marked in the phase status;
its original input IDs, seeds and deltas remain attached to the estimate.
Partitioning can still explore different groupings cheaply. Set
`attribution.reuse_unchanged: false` to collect new effects every iteration.
The positive integer `attribution.refresh_after_unsuccessful_iterations` controls
the reuse bound (default `5`). Post-decision measurements are saved separately
in `01_attribution/refresh.json` with their reason, input IDs, seeds, state hash
and effects. The original `edge_weights.json` still describes the measurement
used for that iteration's partition. Refresh events also appear in the event log.

Before the final block proposal, the optimizer reserves enough equivalent runs
to complete current/candidate acceptance and a normal attribution refresh. A
search cannot consume that reserved
allowance. If it cannot finish full-batch finalists inside its cap, optimization
stops with the accepted checkpoint rather than validating incomparable partial
scores. The reservation covers normal executions; response retries can require
additional work. Model-call and wall-clock caps still apply independently.

## Measurements and artifacts

The [offline probe](optimization_improvements_probes.json) exercises the actual
HotpotQA factory and runtime with deterministic replacement responses and guarded
network connections. For its hint/answer block, training-search costs are:

| Search | Equivalent full-system runs |
| --- | ---: |
| Historical implementation, including duplicated baseline | 340 |
| Baseline reuse, full-batch scoring | 320 |
| Staged scoring, no component replay | 140 |
| Staged scoring with component replay in this probe | 53 |

The 53-run measurement depends on the selected finalists and which inputs change;
it is not a bound for every block or real model output. With the stronger
50-question/two-repeat acceptance, that probe's rejected two-prompt iteration
uses 323 equivalent runs, including 200 for fresh validation and 70 for initial
attribution. Accepted updates require an additional immediate attribution sweep.
Rejected-state iterations reuse effects until the five-iteration refresh bound.
Archived-prompt rescoring and unsuccessful bootstrap attempts consume additional
budget and are charged normally. No HotpotQA accuracy improvement is claimed by
these offline measurements.

`full_system_runs` now also counts logical executions containing replayed
components. Use `component_executions` and `equivalent_full_system_runs` for
charged work, and `reused_component_executions` for shared observations. Replayed
component IDs are included in training trajectory metadata.

Each optimization phase saves `05_block_optimization/candidate_search.json` with
screened configurations, batch fingerprints and finalist states. Returned
candidate records include their training sample count and stage. Acceptance
metrics include all repetition IDs and full requested coverage. Prompt targets
remain in training feedback; acceptance trajectories contain no injected labels.

```sh
python -m scripts.audit_optimization --output search-probes.json
python -m scripts.audit_optimization --run results/hotpotqa-v3 --output run-audit.json
```

## Settings and limitations

The shared and packaged method YAMLs are kept identical. The new search settings
under `block_optimization` are:

```yaml
sample_discrete: false
validation_repeats: 2
reuse_component_executions: true
screening:
  enabled: true
  batch_size: 5
  finalists: 3
```

Use a new output directory for an ablation. Screening can eliminate a good
configuration through small-sample noise; compare against `screening.enabled:
false`. Repeated acceptance reduces generation noise but does not remove finite-
validation-set overfitting. No live benchmark, GPU training or historical Optimas
reproduction is established by the offline checks. Method request seeds and
schema-free response-format options are now forwarded; benchmark generation
remains unseeded and provider seeds are not guarantees of determinism.
