"""Section 4.6, equations 16-17: block changes labeled by complete system outcomes."""

from pathlib import Path

from caiso.blocks.utils import block_context
from caiso.constitution.models import PreferencePair
from caiso.core.content_filter import ContentFilteredError, execute_example, record_skip
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import digest, save_json


def build_block_preferences(
    runtime,
    state,
    members,
    candidate_pairs,
    examples,
    seeds,
    system_constitution,
    judge,
    epsilon=1e-6,
    qualitative=False,
    reuse_trajectories=False,
    progress=None,
    statistics=None,
):
    if not candidate_pairs or not examples or not seeds:
        raise ValueError("Block preference data must be nonempty")
    pairs = []
    trajectories = {}
    executions, reused, skipped, completed = 0, 0, 0, 0
    total = len(candidate_pairs) * len(examples) * len(seeds)
    if progress:
        progress(0, total)

    def execute(candidate, example, seed):
        nonlocal executions, reused
        key = digest([candidate, example, seed])
        if reuse_trajectories and key in trajectories:
            reused += 1
            return trajectories[key]
        trajectory = execute_example(runtime, candidate, example, seed)
        executions += 1
        if reuse_trajectories:
            trajectories[key] = trajectory
        return trajectory

    for i, (a, b) in enumerate(candidate_pairs):
        if not (set(a) | set(b)) <= set(members):
            raise ValueError("Block preference candidates modify an outside component")
        sa, sb = state.with_configs(a), state.with_configs(b)
        for example in examples:
            for seed in seeds:
                completed += 1
                if progress:
                    progress(completed, total)
                ta = execute(sa, example, seed)
                tb = execute(sb, example, seed) if ta is not None else None
                if ta is None or tb is None:
                    skipped += 1
                    continue
                preference = "A" if ta.reward > tb.reward else "B"
                if qualitative or abs(ta.reward - tb.reward) <= epsilon:
                    try:
                        preference = judge.choose(
                            system_constitution, ta.model_dump(), tb.model_dump(), seed
                        )
                    except ContentFilteredError as error:
                        record_skip(
                            runtime.providers,
                            error,
                            "block_preferences",
                            example_id=example.example_id,
                            seed=seed,
                        )
                        skipped += 1
                        continue
                ca, cb = (
                    {n: sa.configurations[n].model_dump() for n in members},
                    {n: sb.configurations[n].model_dump() for n in members},
                )
                pairs.append(
                    PreferencePair(
                        pair_id=f"block-{digest([state, members, i, example.example_id, seed])[:20]}",
                        context={
                            "block": members,
                            "z_B_A": block_context(ta, members, state.workflow),
                            "z_B_B": block_context(tb, members, state.workflow),
                            "system_constitution": system_constitution.principles,
                        },
                        candidate_a={"configuration": ca, "trajectory": ta.model_dump()},
                        candidate_b={"configuration": cb, "trajectory": tb.model_dump()},
                        preferred=preference,
                        metadata={
                            "seed": seed,
                            "global_scores": [ta.reward, tb.reward],
                            "label_source": "constitution"
                            if qualitative or abs(ta.reward - tb.reward) <= epsilon
                            else "task_metric",
                        },
                    )
                )
    if statistics is not None:
        statistics.update(
            preference_records=len(pairs),
            full_system_executions=executions,
            reused_trajectories=reused,
            reuse_trajectories=reuse_trajectories,
            skipped_content_filter=skipped,
        )
    if not pairs:
        raise ModelResponseError(
            "No unfiltered preference examples remain for block constitution induction"
        )
    return pairs


def induce_block_constitution(
    dicai, pairs, members, seed, output=None, progress=None, cache_dir=None, activity=None
):
    options = {"component_ids": members}
    if progress:
        options["progress"] = progress
    if activity:
        options["activity"] = activity
    if output:
        options["output_dir"] = Path(output).with_suffix("")
    if cache_dir:
        options["output_dir"] = Path(cache_dir)
    constitution = dicai.induce(pairs, seed, scope="block:" + ",".join(sorted(members)), **options)
    if output:
        save_json(output, constitution)
    return constitution
