# Integration and benchmark provenance

## Sources

- OPTIMAS: https://github.com/snap-stanford/optimas, commit
  `71871a2dd98081231d733debd41178ba47e13528`.
- CAISO: the method and supporting modules from the user's archived CAISO checkout,
  whose last commit was `f50f6f9` ("Start DICAI cluster search at ten"). Old CAISO
  systems, dataset loaders, model transports, and experiment configurations were
  not imported. Benchmark inference and scoring are delegated to the original code.
- DICAI's adapted procedure/prompts retain their attribution and the supplied
  [license](../third_party/democratic_icai/LICENSE) and
  [source record](../third_party/democratic_icai/SOURCE.json).

`UPSTREAM.json` records every upstream tracked file, its source hash, destination,
and preservation mode. `scripts/verify_upstream.py` verifies all 107 protected files.
76 are verbatim and 31 differ only by `from/import optimas` → `from/import caiso`.
All bundled data artifacts, dataset loaders, metrics and original experiment YAMLs
remain protected. Prompt extraction is recorded below and has separate text and
rendering regression tests.

The [original-source prompt audit](PROMPT_AUDIT.md) verifies all 29 benchmark
prompt entries against pinned OPTIMAS. Active DICAI requests now follow the
user-supplied `Democratic-ICAI-C8D5` source preserved byte for byte under
`third_party/democratic_icai/original/`. All 15 original complete requests and
their trajectory versions have recorded hashes. Independent AST-based tests
reconstruct the original requests and allow only the documented trajectory
adaptations. The earlier paper-prompt protocol is retained as a historical
reference.

The source's structured reasoning/questions, sampled consensus, iterative
refinement and reconstruction-gated debate are wired into CAISO's existing
providers and recovery layer. Numeric configurations, benchmark systems, datasets,
metrics, model routes and the three dataset personas are retained. Original
creative-writing persona constants are preserved as references. The external
audit template is imported without enabling a new optimization stage or model.
The source's best-effort result at the round cap remains explicitly marked as
unfinalized if the reconstruction/minimum-question gates fail. Existing robust
format validation still requires at least one usable chairperson turn.

Debate uses the classic AutoGen API from the source: three persistent
`AssistantAgent` debaters and a `Judge` call `generate_reply` with one
`Moderator_Proxy` `UserProxyAgent` as sender. The proxy has no LLM, human input or
code execution. Agents are local to each preference pair, so concurrent debates
share no agent state. A registered AutoGen model client delegates completions to
CAISO's provider/recovery and request-cache layer, preserving role routing,
per-turn seeds, temperatures, validators, budgets and response error context.
The editable request files still supply the complete system/user messages and
bounded shared history; no implicit agent history is appended. AutoGen caching
is disabled. The runtime dependency is pinned to `autogen[openai]==0.14.1`, which
provides the classic `autogen` namespace, separate from the optional modern
`autogen_agentchat` benchmark component adapter. Implementation fingerprints
invalidate earlier induction caches after this execution change.

## Ten explicit upstream exceptions

1. `.gitignore`: adds local environment, cache, and build exclusions.
2. `caiso/__init__.py`: package metadata/logging and a Python 3.12-compatible
   replacement for the removed `distutils` boolean parser; enables a narrow
   joblib compatibility shim mapping pandas-1 numeric index classes to `Index`.
   This lets pandas 2 read the bundled test artifact without editing it or its loader.
3. `caiso/optim/optimizer.py`: replaces OPTIMAS's reward-model optimizer with
   `CaisoOptimizer`, which runs `caiso.method.algorithm.Algorithm`.
4. `scripts/optimize_system.py`: constructs the same system/dataset factories and
   replaces reward-model loading/training with CAISO setup. It saves the original
   system state format and uses the original evaluator for final test evaluation.
   Accepts an optional exact output-directory override for `scripts/caiso`.
   Writes the resolved `config.yaml` and an `evaluation.json` report alongside
   the existing checkpoint exports and raw evaluation metrics. The helper defaults
   to `results/<dataset>-<config-hash>/`; direct Python invocation retains YAML paths.
5. `scripts/eval_system.py`: import renaming plus `weights_only=False` when loading
   the original checkpoint format, which contains `SimpleNamespace` and `Path`
   objects and requires explicit full loading on modern PyTorch.
   Accepts programmatic argv and an exact output-directory override so the
   convenience launcher can reuse this evaluator without duplicating it.
6. `examples/systems/hotpotqa/five_components.py`: reads `COLBERT_URL` (or
   `colbert_url`) from the environment/.env, retaining the original retrieval URL
   as the fallback. Loads signature text from component folders under
   `caiso/prompts/systems/hotpotqa/`.
   Prompt text, k, and the search space are unchanged.
7. `examples/systems/amazon/local_models_for_next_item_selection.py`: loads the same
   completion templates and DSPy instruction/field metadata from
   component folders under `caiso/prompts/systems/amazon/`.
8. `examples/systems/pubmed/three_components_with_model_selection.py`: loads the
   same system message, initial instructions, answer format and user templates from
   `caiso/prompts/systems/pubmedqa/`.
9. `caiso/arch/base.py`: propagates explicit policy skips, exhausted transport
   retries and non-retryable service failures immediately instead of resampling
   them in the generic component retry loop. Ordinary response retries stay intact.
   Keeps component trajectories on the calling thread so concurrent evaluation
   cannot read another example's outputs when constructing its prediction.
10. `caiso/eval.py`: calls the same per-example `system.evaluate` and task scorer,
    preserves successful scores, keeps predictions in input order, counts policy
    skips as zero, records coverage, and propagates other failures. It replaces the
    upstream helper that swallowed errors and excluded failed examples from means.

The active `configs/experiments/*.yaml` profiles use shared settings in
`configs/method/default.yaml` with recursive experiment overrides. Their current
[search protocol](OPTIMIZATION_IMPROVEMENTS.md) intentionally upgrades validation,
preference labels, candidate screening, history and observation reuse. The
original `scripts/configs/*.yaml` profiles remain preserved. Legacy reward/PPO parameters remain in the
original files but are absent from active CAISO profiles because CAISO does not use
those OPTIMAS algorithms.

The upstream root README is preserved at `docs/OPTIMAS_README.md`; the root README
explains this integration. New files contain CAISO, integration tests, packaging,
and documentation. Remaining upstream reward-model collection/training utilities
and component search utilities are preserved under `/caiso` for reference and
legacy scripts. They are **not** the optimizer used by CAISO's launcher.

## Dataset and optimization defaults

| Dataset | Train / validation / test | Optimization train batch | Active acceptance set |
| --- | --- | --- | --- | --- |
| HotpotQA | 5000 / 250 / 100 | 20 | 50 seeded questions, 2 repetitions |
| PubMedQA | 475 / 25 / 500 | 20 | all 25 questions, 2 repetitions |
| Amazon | 335 / 60 / 99 | 20 | all 60 questions, 2 repetitions |

Split sizes are the released loader defaults; acceptance settings are CAISO's
current protocol, not a claim of matching the paper. HotpotQA uses the original DSPy loader and the
bundled distractor test artifact. No new filtering, splitting, answer extraction,
prompt formatting, token limit, model substitutions, or scoring functions are
introduced. PubMedQA keeps the released first-substring answer matcher, including
its known behavior on words containing "no". Amazon keeps the original raw
completion payload, including `max_new_tokens`. STaRK and BigCodeBench sources and
evaluation profiles are also retained; their external environments are required.

The active profiles use `method.validation_sampling: seeded`, retaining the complete
validation pool and draws a reproducible acceptance subset of
`validation_batch_size` examples using the method seed. That acceptance subset
stays fixed across iterations. Attribution and diagnostics use separate training
mini-batches. Setting
`method.validation_batch_size: all` uses the full pool for acceptance. These
overrides change the protocol; start a new run instead of resuming a checkpoint
created with another selection. The preserved direct-launch profiles explicitly
retain their historical validation prefixes. The test set is never passed to CAISO's optimizer.
Original preference-data/reward-model/PPO
settings are not mapped onto a different CAISO mechanism: CAISO induces its own
constitutions and uses constitution-labelled DPO for weight components.

## Execution boundary

`UpstreamRuntime.execute` calls the original `CompoundAISystem.__call__` and
`evaluate` on the original example fields. A temporary observer records component
calls and applies an edge intervention only when requested for causal attribution.
Candidate configuration changes use the original component update callbacks,
including actual adapter unload/load operations. Configuration changes are restored
on success and failure. The selected state is installed into that same system
before the original evaluation script runs.

Candidate training searches can share recorded component observations keyed by
example/repeat, actual inputs, configuration and generation settings. Changes to
inputs/configuration invalidate the corresponding component; downstream execution
uses the newly observed outputs. Replayed components restore trajectory entries,
are marked in trajectory metadata and are counted separately from executed
components. Scopes end before acceptance, which always runs fresh with the
default reevaluation policy. Replay refuses validation/test examples. Filtered
observations remain zero and are not silently resampled during promotion.

CAISO's internal graph is inferred from the executor's last-producer field bindings.
Fields retain their original values and types. Non-selector edge baselines use
zero-valued observed containers/numbers and a single blank space for text, so the
original components' nonempty-input checks still hold. Selector-edge baselines use
the original initial model, since an empty model identifier is invalid. Overrides
may specify a constant or pass-through baseline per edge/input field.

During an intervention, a downstream generator may also return an empty string
after receiving blank evidence. In the intervened component and its descendants,
the runtime represents empty text inputs with the same single blank space. For
HotpotQA, an empty/whitespace-only retrieval query produces no retrieved evidence
without a ColBERT request. This avoids repeatedly executing the strict retriever
with an empty keyword input or retrieving unrelated documents for a dummy query.
Trajectory metadata records `blank_intervention_inputs` and
`empty_query_retrievals`. If the keyword generator returns an empty or whitespace
query on an ordinary input during optimization, the runtime retries that
producer using the configured `response_retries` limit (two retries by default),
charges each execution, and logs its output. Persistent failure stops with a
producer-specific error instead of repeatedly executing the retriever on the
same invalid input. Valid upstream calls and the separate evaluation script
retain their original behavior; no fallback question, model or passage is
substituted.

Upstream component defaults, per-thread variables, trajectory storage, and adapter
IDs are mutable shared state. Optimization execution is protected by a lock;
raising CAISO's execution worker count does not make those calls simultaneous.
This preserves candidate isolation. Independent DICAI debates/reasoning remain
parallel. The original evaluation script retains its own worker settings.

Amazon DPO captures completion prompts by executing the original `forward` code
with an isolated mock HTTP function. The prompt strings are not rewritten or
maintained as a second copy. Candidate inference continues through the original
vLLM server/client rather than a second inference engine.

## Reproducibility and cost

Source preservation is checked by hashes. Tests exercise real benchmark factories
with mocked external responses; they do not demonstrate historical benchmark
scores, GPU training, external retrieval correctness, or API availability.
The upstream requirements do not pin their historical environment. CAISO pins
DSPy 3.0.4 and TRL 0.29.0 for its integration/training path, and records the installed
versions and data hashes. Do not infer an exact historical reproduction from a
matching source tree alone. Legacy reward-training scripts are retained as source;
compatibility with their historical training environment is not asserted.

CAISO seeds control its sampling, method requests and training. Bootstrap ties
abstain. Benchmark generation
requests retain their original unseeded behavior. CAISO's equivalent-run budget
counts component executions across initialization, attribution, proposals, and
validation. Partial block runs count by their fraction of full-system components.
LLM/token counters cover CAISO method services and do not intercept HTTP calls or
retries inside the unchanged benchmark components. Post-optimization test evaluation
is outside the optimization budget. Compare both optimization budgets and model/
training costs when drawing efficiency conclusions.
