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

Keep credentials, deployments, backend choices and service URLs in `.env`.
Prompt text is under [`caiso/prompts`](../caiso/prompts/README.md). `prompt_tips`
remains a method parameter with its existing values.

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

```sh
./scripts/caiso run amazon
./scripts/caiso evaluate amazon --unoptimized
./scripts/caiso run --config configs/experiments/amazon.yaml
```
