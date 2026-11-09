# CAISO2

CAISO's constitution and causal block optimization, integrated with the released
[OPTIMAS](https://github.com/snap-stanford/optimas) benchmark code at commit
`71871a2dd98081231d733debd41178ba47e13528`.

The original dataset loaders, data artifacts, metrics, and experiment YAMLs are
preserved. Benchmark prompts retain their original text in the shared prompt
directory. Python imports use `caiso` instead of
`optimas`. Optimization runs `caiso/method/algorithm.py`; it does not train or use
OPTIMAS's learned local reward model.

## Install

Use Python 3.11 or 3.12, from this repository's root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[benchmarks,train]'
```

The original `requirements.txt` is included unchanged. `requirements-caiso.txt`
adds CAISO's dependencies. Additional benchmark dependencies and external services
are still required where they were required upstream; see
[the original setup instructions](docs/OPTIMAS_README.md).

Copy [`.env.example`](.env.example) to `.env` and fill in your Azure key,
resource endpoint, API version, and deployment names. With `LLM_BACKEND=azure`,
existing GPT task calls (including DSPy) and method/embedding calls use those
deployments. No YAML edits or OpenAI API key are required for these calls.
`CAISO_METHOD_MODEL` and `CAISO_EMBEDDING_MODEL` override method service models.
Both launchers load `.env`; already-exported environment variables take precedence.
To use the original providers, omit `LLM_BACKEND` or set it to `openai`.

Deployment variables follow `AZURE_<MODEL>_DEPLOYMENT`, replacing punctuation
with underscores, e.g. `gpt-4o-mini` uses `AZURE_GPT_4O_MINI_DEPLOYMENT`.
Explicit `azure/<deployment>` model identifiers also work. Routing applies to
sync/async LiteLLM completion and embedding calls and retains prompt/generation
settings. Missing mappings fail explicitly. PubMed's original search space also
contains Claude candidates: an Azure OpenAI resource alone cannot run that full
search space. Those require provider support or an explicitly changed search space.
Do not map a Claude candidate to a GPT deployment for a benchmark comparison.
The Azure OpenAI key/endpoint aliases `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_API_VERSION` are also accepted.
Keep `.env` private; only the placeholder template is tracked.

Azure request limits are in [`configs/providers.yaml`](configs/providers.yaml),
separate from model credentials. Defaults match the earlier transport: 60 requests
per minute and four concurrent requests per deployment, eight rate-limit retries,
and four transient-error retries. Limits are shared by DSPy task calls, method roles
and embeddings that use the same endpoint/deployment. Per-deployment overrides can
be added under `deployments` using the Azure deployment name as the key.

The launcher validates these limits before constructing models and prints the
effective defaults. These are local caps, not an Azure quota-discovery API; configure
them to match your deployments. HTTP 429 retries honor Azure's `Retry-After` and
millisecond headers, including waits longer than the local backoff cap. A shared
cooldown also holds concurrent callers. Built-in SDK retries are disabled on these
routes to avoid multiplying retry counts.

Explicit policy rejections and responses marked `content_filter` skip without
resampling. Optimization records affected examples/pairs through its existing skip
handlers. Evaluation counts those examples as zero in the full requested denominator
and reports coverage per repetition. Other evaluation failures, including credentials,
missing deployments, exhausted retries or invalid scores, stop evaluation instead of
silently dropping examples. Model, prompt, dataset and generation settings stay the same.

Diagnostics are under `.internal/logs/`: `transport_config.json` records limits,
`transport.jsonl` records retries/recovery, and `skipped_examples.jsonl` records policy
skips. Evaluation JSON includes `coverage` counts for `requested`, `completed` and
`skipped_content_filter`. Transport logs omit request bodies and API keys.
All-filtered evaluation batches score zero; initialization still requires enough
usable evidence to induce a constitution.

Amazon still requires the original Qwen model,
initialized LoRA adapters, and an external vLLM server. HotpotQA retains the
original ColBERT endpoint by default. Set `COLBERT_URL` in `.env` to your full
ColBERT search URL (including the route, e.g. `/wiki17_abstracts`). The lowercase
`colbert_url` alias is also accepted; `COLBERT_URL` takes precedence.
BigCodeBench requires its package/execution environment;
STaRK requires its original data/embeddings and model credentials.

## Run the same unoptimized systems

The convenience launcher uses the same files and defaults as the Python commands
below. Run it from the checkout; it selects this project's `.venv` and loads `.env`:

```sh
./scripts/caiso evaluate amazon --unoptimized
./scripts/caiso evaluate hotpotqa --unoptimized
./scripts/caiso evaluate pubmedqa --unoptimized
```

`evaluate <dataset>` also defaults to the unoptimized system. Evaluation uses the
`evaluation` sections in `configs/experiments/*.yaml`, including their three
repetitions. Use `--repeats 1` to override that, or `--split val` for validation.
Each repeat displays a progress bar with finished examples, elapsed time, and ETA.
Content-policy skips advance the bar and remain included in the reported coverage.
As each repeat finishes, its score and coverage are printed immediately, before
the next repeat starts (for example, `Evaluation test repeat 1/3: score=0.350000`).
Baseline results default to `results/<dataset>-unoptimized/evaluation_unoptimized.json`
(using `pubmedqa` as the directory label). `--output <directory>` sets the exact
result directory. The evaluator's raw `test_metrics.json`/`val_metrics.json`, logs,
and initial-state snapshot are kept under `.internal/evaluation/<variant>/<split>/`.
Validation reports use `validation_unoptimized.json`/`validation.json` to keep test
results separate. The launcher starts neither vLLM
nor ColBERT: the same external services must already be available.

Increase parallel examples for any of the three datasets or an optimized run with
`--workers` (the current experiment profiles default to four):

```sh
./scripts/caiso evaluate amazon --unoptimized --workers 10
./scripts/caiso evaluate hotpotqa --unoptimized --workers 10
./scripts/caiso evaluate pubmedqa --unoptimized --workers 10
./scripts/caiso evaluate results/amazon-run --workers 10
```

To persist that choice, set `evaluation.max_eval_workers` in the experiment YAML.
Workers overlap independent examples; each example still runs its original
components in order. The Azure transport independently caps each deployment at
four concurrent requests and 60 requests/minute by default, so more workers alone
cannot exceed those caps. Set `concurrency` and `requests_per_minute` in
`configs/providers.yaml` to fit the deployment's actual RPM and TPM quota; its
request pacing does not enforce a separate TPM limit. Amazon's local stages also
depend on vLLM throughput. Component trajectories are isolated per worker to keep
concurrent examples from overwriting each other's predictions.

For a shorter preliminary check, add `--repeats 1`; the default three repeats
remain useful for measuring variation. This reduces evaluation work rather than
speeding up each repeat.

### Enter a configuration interactively

Use `--interactive` to enter each component's settings in execution order and
evaluate the resulting system on the test set:

```sh
./scripts/caiso evaluate hotpotqa --interactive
./scripts/caiso evaluate pubmedqa --interactive --repeats 1 --output results/my-manual-system
```

Enter keeps the displayed value. Type a new prompt directly, or type `/multiline`,
paste multiple lines, and finish with `/end` on its own line. Hyperparameters and
model selectors are requested one field at a time. Numbers and booleans use JSON
syntax; strings such as model names are entered without quotes. Values may differ
from the optimizer's search grid. Existing generation settings (model,
temperature, token limits, top-p and top-k) are also offered. Components with
model-weight variables accept an existing adapter directory; fixed components
keep their behavior. The same external services and model files are required as
for ordinary evaluation.

Each invocation defaults to a new `results/<dataset>-manual-<id>/` folder containing:

- `component_config.json`: the entered component values and evaluation settings.
- `manual_system.pth`: the complete system state, including generation settings.
- `evaluation.json`: test scores, repetition results, and coverage.
- `.internal/evaluation/manual/test/`: raw metrics, logs, and rollout files.

`--output` selects an exact folder; an existing run is rejected to avoid overwriting
its configuration or results. Interactive mode starts from the factory system;
use a dataset or YAML profile, without `--state`, `--unoptimized`, or a saved state.
To enable it in an experiment YAML, add `interactive: true` under `evaluation`
(or at the top level of a legacy evaluation profile). The direct
`python -m scripts.eval_system` launcher also accepts `--interactive`.

To repeat evaluation later without entering the settings again, pass
`--system_state_dict_path results/my-manual-system/manual_system.pth` to the direct
evaluation launcher, together with the matching dataset and system name.

```sh
python -m scripts.eval_system scripts/configs/eval/hotpotqa_vanilla.yaml
python -m scripts.eval_system scripts/configs/eval/pubmed_vanilla.yaml
python -m scripts.eval_system scripts/configs/eval/amazon_vanilla.yaml
```

These use the original evaluator, factories, metrics, and test sets. The released
STaRK and BigCodeBench evaluation configurations are also included.

## Optimize with CAISO

```sh
./scripts/caiso optimize amazon --output results/amazon-run
./scripts/caiso optimize hotpotqa --output results/hotpotqa-run
./scripts/caiso optimize pubmedqa --output results/pubmed-run
```

These dispatch to `scripts/optimize_system.py` with `configs/experiments/*.yaml`.
`run` is an alias for `optimize`, so `./scripts/caiso run amazon` also works.
Omitting `--output` restores `results/<dataset>-<config-hash>/`, where the hash
reflects the current experiment and resolved method configuration. An existing
checkpoint requires `--resume` or a new `--output`. `--no-test` skips the automatic
test evaluation after optimization; `--resume <checkpoint-directory>` (also
`--checkpoint`) forwards to the optimizer's existing resume support.

To evaluate a completed optimization run, pass the actual run directory:

```sh
./scripts/caiso evaluate results/amazon-run
./scripts/caiso evaluate results/amazon-run --unoptimized
```

The first command reads `experiment.json` and loads `final_optimized_system.pth`
when available. Otherwise it evaluates the current saved CAISO checkpoint. The
second constructs the initial system.

You can evaluate an accepted system while optimization continues in another
terminal:

```sh
./scripts/caiso evaluate results/hotpotqa-run --state current
./scripts/caiso evaluate results/hotpotqa-run --state best --repeats 3
```

`current` loads the latest saved accepted state; `best` loads the checkpoint's
best validation state. These flags always use the checkpoint, including when a
previous final export exists during a resumed run. `--state final` requires the
final `.pth` export. Before the first checkpoint is saved, wait for initialization
or use `--unoptimized` explicitly.

The evaluator freezes one checkpoint version before constructing the system. It
prints and records the checkpoint iteration, validation score and state
fingerprint. Later optimizer updates do not alter that evaluation. Reports are
`evaluation_current.json` / `evaluation_best.json` (or `validation_current.json` /
`validation_best.json` with `--split val`); the final `evaluation.json` is separate.
Each invocation keeps its frozen state, original component settings and logs
under `.internal/evaluation/<current-or-best>/<split>/<id>/`. Amazon snapshot
evaluation reuses saved adapters under unique vLLM names, without recreating
adapter files or replacing the optimizer's active registrations.
Their reports are `evaluation.json` and `evaluation_unoptimized.json` at the run
root. Automatic post-optimization evaluation also writes `evaluation.json`.
The run contains `config.yaml`, `README.md`, `setup.json`, `run.json`, `history.csv`,
`iterations/`, `checkpoint/`, and `.internal/`, plus the current checkpoint exports.

The current-iteration terminal panel shows every component's original, current
accepted, and candidate configuration, with the candidate's evaluation or
acceptance status. Long values use compact previews; complete values are saved
in each iteration's `07_state/component_configurations.json`. Temporary variable
apply/restore messages and queued model logs go to the active phase's `model.log`.
Capable terminals update one panel in place; redirected output and terminals
without cursor control receive one final panel.

Explicit `--output` paths are still respected; existing results are not moved.
Both use the current evaluation profile for that dataset. An existing YAML can be
passed as the target or via `--config`; `--system amazon` is also accepted in place
of a dataset target. `pubmed` and `pubmedqa` refer to the same current PubMed files.
`python -m scripts.caiso_cli ...` is equivalent when using an activated environment.

```sh
python -m scripts.optimize_system scripts/configs/optimize/hotpotqa.yaml
python -m scripts.optimize_system scripts/configs/optimize/pubmed.yaml
python -m scripts.optimize_system scripts/configs/optimize/amazon.yaml
```

## Configuration and prompts

The main launcher uses the earlier CAISO configuration structure:

```text
configs/
  method/default.yaml
  experiments/amazon.yaml
  experiments/hotpotqa.yaml
  experiments/pubmedqa.yaml
caiso/prompts/
  dicai/reasoning/*.txt
  dicai/debate/*.txt
  dicai/{reconstruction,abstraction}.txt
  method/<stage>/*.txt
  systems/amazon/<component>/*.txt
  systems/hotpotqa/<component>/*.txt
  systems/pubmedqa/<component>/*.txt
```

`method_config` is relative to the
experiment file; the experiment's `method` mapping recursively overrides the
shared method. `optimization` contains launch settings and `evaluation` contains
the existing evaluation settings. For example, in a copy of an experiment:

```yaml
method:
  budget:
    max_equivalent_full_runs: 1000
  block_optimization:
    max_joint_candidates: 3
    task_scoring:
      enabled: true
      constitution_weight: 0.0
      task_weight: 1.0
```

Candidate selection and prompt-generation feedback default to the end-to-end
task metric on the training batch. With `constitution_weight: 0.0`, candidate
constitution judging is skipped and `constitution_score` is recorded as `null`.
The constitution still guides generation and block selection. Validation
acceptance continues to use the separate, fixed validation set.

Set a positive `constitution_weight` to opt into weighted task/constitution
ranking; in that mode prompt feedback uses the local constitution score.
Set `task_scoring.enabled: false` to use constitution-only scoring.
Task scoring runs the full system for candidate feedback and can consume more
evaluation budget than local block scoring. Scores for identical candidate states
are reused within a proposal call.

Select custom profiles with `./scripts/caiso run --config YOUR_EXPERIMENT.yaml` or
`./scripts/caiso evaluate --config YOUR_EXPERIMENT.yaml`.

`leiden.max_comm_size` defaults to `2`, limiting each discovered block to at most
two components. Override it under `method.leiden` in an experiment; use `0` for
unrestricted block sizes. Singleton blocks remain allowed.

`block_selection.use_history` defaults to `true` in the shared method file. Set
`method.block_selection.use_history: false` in an experiment to omit past
optimization attempts from block-selection requests only. This is the only
history toggle; the duplicate top-level `use_history` setting has been removed.

Prompt optimization always retains candidate instructions and task-score
feedback within each proposal call. Rejected-candidate filtering, workflow
performance feedback, and DICAI debate history remain enabled regardless of the
selection-history setting. Block preference generation also retains rejection
filtering when enabled. Phase status files record the
effective `history_enabled` setting for each stage. Logs and checkpoints continue
to retain all optimization attempts. Start a new run when changing the setting;
older checkpoints that omit the selection key or include the removed top-level
`true` default remain compatible when all other settings match. Move any old
top-level override into `block_selection.use_history` in your YAML.

Block constitution induction is off by default (`block_constitution.enabled: false`).
All block prompt optimization uses the system constitution directly,
skipping preliminary preference candidates, block preference collection, and
block induction. Task scores still drive candidate selection, and validation
controls acceptance. Set `method.block_constitution.enabled: true` in an
experiment to opt into block-specific constitutions. Changing this setting
requires a new run/output directory.

`block_optimization.reevaluate_current` also defaults to `true`. Set it to `false`
in the shared method file (or under an experiment's `method.block_optimization`)
to reuse the saved current validation score during block acceptance. The initial
baseline and all candidates are still evaluated. Reused scores are marked in the
phase's validation metrics; this saves the current-system validation runs and
changes how stochastic evaluations are compared. Start a new run after changing
the setting.

`block_optimization.include_current_configuration` defaults to `true`. Set it to
`false` to exclude unchanged component values from candidate menus for prompts,
model choices, hyperparameters and trained adapters. Every eligible joint
candidate must change each optimizable component in the selected block; if one
has no changed option, that proposal pass is skipped. Fixed components retain
their required values. The current system remains a scoring reference, and
acceptance validation still decides whether to install the proposed changes.
Use a new run when changing this search setting.

The original `scripts/configs/` files and `caiso/config/default.yaml` remain intact
and supported by direct Python commands. Their reward-model/PPO parameters belong
to the legacy OPTIMAS utilities and are unused by CAISO. CAISO weight optimization
uses its existing `block_optimization.dpo` settings; reward-model training is not
needed. Credentials, model deployments, backend choices and service URLs stay in
`.env`. See [the configuration guide](configs/README.md) and
[the prompt inventory](caiso/prompts/README.md).

Stage request files contain complete `[SYSTEM]` and `[USER]` sections. Active
DICAI prompts come from the user-supplied `Democratic-ICAI-C8D5` code. Its eight
original files are preserved in `third_party/democratic_icai/original/`; all 15
complete original requests are in `third_party/democratic_icai/prompts/`.
Their editable trajectory versions are in `caiso/prompts/dicai/`. The
[prompt audit](docs/PROMPT_AUDIT.md) records source hashes and the exact adaptations.

CAISO also enforces the existing one-quality rule with a semantic check after
new chairperson lists and after abstraction. Compound debate principles receive
bounded split revisions; compound abstractions are revised to one shared quality
per cluster. The 15 source-derived DICAI prompts remain unchanged. Complete
check/revision prompts are in `caiso/prompts/method/constitution_atomicity/`.
The existing induction model is used; `dicai.atomicity.enabled` defaults to `true`
and `max_revisions` to `2`. Reviews, repairs and verified fallbacks are recorded.
See [configuration](configs/README.md) for the optional original-behavior ablation.

Principles start with **"Select the trajectory"**. Original instructions are
retained apart from response/story terminology, provided preference labels,
structured execution evidence and the documented guidance for component criteria
and reusable principles. Constitutions remain lists of strings. No dataset-specific
answer-style requirements are added.

Reasoning returns the original `reasoning`, `dimensions`, and `questions` fields.
CoT uses one request; self-consistency samples independent chains and synthesizes
recurring dimensions; self-refinement alternates critique and revision. The
existing settings control these stages. The separate CAISO seed-drafting call
has been removed because the original rationales already supply questions.
Debate retains the source's one-quality rules, reconstruction feedback and
finalization gates. The blind judge uses the original JSON choice `1` / `2`
format, adapted to trajectory evidence. Singleton clusters retain their principle
without an abstraction call, as in the source.

The audit prompt is included as a complete template; no new auditor model or
automatic audit stage is enabled. Dataset personas, model routes, transport
limits and all existing numeric configuration values are retained. The paper
Appendix G transcriptions remain historical references, not active prompts.
Start a new optimization run after this migration: prompt schemas, call counts
and reconstruction behavior differ, and old checkpoints retain old constitutions.
Document this trajectory adaptation and the resolved configuration in research.

CAISO's prompt optimizer, block selector and workflow designer use the same
writing format: named agents, numbered tasks, explicit output formats and
labeled evidence sections. Each stage's complete request is in one `.txt` file
under `caiso/prompts/method/`.

Check that all prompt resources are present with
`python -m scripts.verify_prompts`. Optimization checks the prompts needed by
its configuration before loading benchmark models. Missing-file reports list
the package location and recovery steps; configured personas and tips do not
require their unused default files.

Method responses tolerate Markdown/list presentation differences and wrapped
JSON while retaining schema validation. Rejected answers and their pair/stage
context are recorded in `.internal/logs/phase_errors.jsonl`, and repair attempts
include the answer that failed. Successful constitution-induction requests are
checkpointed individually, so retrying an interrupted induction can reuse
completed model calls. Chairperson format failures retain the last validated
working list when available; an empty constitution is never substituted.
A candidate seed list also requires at least one usable chairperson turn before
it can be included as settled principles. A corrected induction has a new
provenance fingerprint; use a new run to replace older constitutions.

Each experiment's `caiso.persona_catalog` selects three dataset-specific expert
profiles from `caiso/prompts/systems/<dataset>/personas/*.txt` for both system and
block constitution induction. Their full text is saved with the resolved
experiment. Optional `caiso.personas` lists override the catalog;
`caiso.edge_baselines` configures per-edge interventions.
Workflow search is disabled to retain the original benchmark systems. Full-system
and component calls during CAISO optimization are serialized to avoid races in the
upstream shared variables and vLLM adapter registrations; DICAI requests remain
parallel.

The launcher saves `initial_system.pth`, `final_optimized_system.pth`, the CAISO
checkpoint, per-phase artifacts, and `benchmark_manifest.json` containing dataset
hashes, model settings, and installed package versions. It evaluates the final
system on the untouched test set unless `--no-test` is passed. Resume with:

```sh
python -m scripts.optimize_system YOUR_EXPERIMENT.yaml --resume YOUR_OUTPUT/checkpoint
```

Evaluate the resulting system with the same upstream evaluation script:

```sh
python -m scripts.eval_system --dataset hotpotqa --system_name hotpotqa_system \
  --system_state_dict_path YOUR_OUTPUT/final_optimized_system.pth --num_repeat 3
```

## Verify

The [optimization quality audit](docs/OPTIMIZATION_AUDIT.md) traces the active
search, explains its main limitations, compares it with Optimas, and includes
offline probes plus a read-only inspector for saved runs.

```sh
python -m scripts.verify_upstream
pip install -e '.[test]'
pytest -q
```

107 imported files are checked against the pinned source, allowing only package
import renaming where needed. The ten explicit integration exceptions are listed
in [`docs/INTEGRATION.md`](docs/INTEGRATION.md). Tests compare the actual HotpotQA,
PubMedQA, and Amazon benchmark requests and trajectories with deterministic mocked
services, and run CAISO's full induction/optimization/checkpoint cycle offline.
Live benchmark scores and GPU training are not established by those tests.

Exact source parity does not guarantee the historical paper scores: API model
backends, retrieval services, dependency versions, and downloaded data can change.
This checkout follows the **released code defaults**, including HotpotQA's 5000
training examples, rather than silently substituting paper-specific splits.

The upstream [Apache 2.0 license](LICENSE) and [attribution](ATTRIBUTION.md) are retained.
See [integration details](docs/INTEGRATION.md) for CAISO and DICAI provenance.
