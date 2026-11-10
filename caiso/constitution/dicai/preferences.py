"""Project CAISO outputs into label-free DICAI prompt/response comparisons."""

from caiso.prompts import render_text

EVALUATION_KEYS = {"reward", "target", "preferred", "gold", "global_scores", "metadata"}


def response_view(value):
    """Exclude evaluation metadata, retaining executed inputs and outputs.

    Projection avoids dropping legitimate task fields inside component outputs.
    """
    if "continuations" in value:
        value = {**value, "continuations": [response_view(item) for item in value["continuations"]]}
    if "trajectory" in value:
        return {
            **{k: v for k, v in value.items() if k not in EVALUATION_KEYS | {"trajectory"}},
            "trajectory": response_view(value["trajectory"]),
        }
    if "executions" in value:
        return {
            **{
                k: v
                for k, v in value.items()
                if k not in EVALUATION_KEYS | {"executions", "seed", "state_hash", "input_id"}
            },
            "executions": [
                {k: e[k] for k in ("component_id", "component_input", "component_output") if k in e}
                for e in value["executions"]
            ],
        }
    return {k: v for k, v in value.items() if k not in EVALUATION_KEYS}


def task_view(context):
    # Attach candidate-specific z_B context to its response before shuffling.
    return {
        k: v
        for k, v in context.items()
        if k not in {"z_B_A", "z_B_B", "preferred", "gold", "target", "global_scores", "metadata"}
    }


def candidate_views(pair):
    a, b = response_view(pair.candidate_a), response_view(pair.candidate_b)
    for candidate, key in ((a, "z_B_A"), (b, "z_B_B")):
        if key in pair.context:
            candidate["block_context"] = pair.context[key]
    return a, b


def pair_block(pair, labeled=False):
    a, b = candidate_views(pair)
    return render_text(
        "method", "preferences.labeled_pair" if labeled else "preferences.unlabeled_pair",
        context=task_view(pair.context), response_a=a, response_b=b, preferred=pair.preferred,
    )
