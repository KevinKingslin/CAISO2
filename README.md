# CAISO2

CAISO's constitution and causal block optimization, integrated with the released
[OPTIMAS](https://github.com/snap-stanford/optimas) benchmark code at commit
`71871a2dd98081231d733debd41178ba47e13528`.

The original examples, dataset loaders, data artifacts, metrics, system factories,
and experiment YAMLs are preserved. Python imports use `caiso` instead of
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

Configure `.env` using the credentials expected by the original benchmark
(OpenAI/Anthropic/Gemini as applicable). CAISO's method services default to
`gpt-4o` and `text-embedding-3-small`, via LiteLLM. These are separate from the
unchanged benchmark models. Amazon still requires the original Qwen model,
initialized LoRA adapters, and an external vLLM server. HotpotQA retains the
original ColBERT endpoint. BigCodeBench requires its package/execution environment;
STaRK requires its original data/embeddings and model credentials.

## Run the same unoptimized systems

```sh
python -m scripts.eval_system scripts/configs/eval/hotpotqa_vanilla.yaml
python -m scripts.eval_system scripts/configs/eval/pubmed_vanilla.yaml
python -m scripts.eval_system scripts/configs/eval/amazon_vanilla.yaml
```

These use the original evaluator, factories, metrics, and test sets. The released
STaRK and BigCodeBench evaluation configurations are also included.

## Optimize with CAISO

```sh
python -m scripts.optimize_system scripts/configs/optimize/hotpotqa.yaml
python -m scripts.optimize_system scripts/configs/optimize/pubmed.yaml
python -m scripts.optimize_system scripts/configs/optimize/amazon.yaml
```

The original experiment files specify the system, dataset, output location,
training batch, validation prefix, iteration limit, and prompt candidate count.
The CAISO launcher replaces reward-model setup and the optimization algorithm.
The released reward-model/preference-dataset/PPO settings remain in the files for
reference; CAISO uses its own constitution induction and DPO settings instead.
There is no reward-model training prerequisite for CAISO.

Method defaults are in [`caiso/config/default.yaml`](caiso/config/default.yaml).
To change them, copy an experiment YAML and add a `caiso` section:

```yaml
caiso:
  services:
    model: gpt-4o
    embeddings:
      backend: litellm
      model: text-embedding-3-small
  method:
    budget:
      max_equivalent_full_runs: 1000
    block_optimization:
      max_joint_candidates: 3
      task_scoring:
        enabled: true
```

Experiment overrides take precedence over method defaults. Optional `caiso.personas`
and `caiso.edge_baselines` configure DICAI identities and per-edge interventions.
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

```sh
python -m scripts.verify_upstream
pip install -e '.[test]'
pytest -q
```

112 imported files are checked against the pinned source, allowing only package
import renaming where needed. The five explicit integration exceptions are listed
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
