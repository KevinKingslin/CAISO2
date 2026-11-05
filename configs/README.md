# CAISO configuration

`scripts/caiso` selects `experiments/amazon.yaml`, `experiments/hotpotqa.yaml`, or
`experiments/pubmedqa.yaml` for both evaluation and optimization.

- `method/default.yaml`: shared algorithm, DICAI, attribution, clustering,
  optimization, DPO, budget, error-handling and reporting settings.
- `experiments/*.yaml`: benchmark identity, launch/evaluation settings, and
  dataset-specific batches and method overrides.
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

Keep credentials, deployments, backend choices and service URLs in `.env`.
Prompt text is under [`caiso/prompts`](../caiso/prompts/README.md). `prompt_tips`
remains a method parameter with its existing values.

```sh
./scripts/caiso run amazon
./scripts/caiso evaluate amazon --unoptimized
./scripts/caiso run --config configs/experiments/amazon.yaml
```
