"""One abstraction pass per cluster, using the existing induction model."""

from caiso.core.parallel import ordered_map
from caiso.core.recovery import ModelResponseError
from caiso.prompts import render_messages

from .reasoning_assembly import valid_principle


def abstract_clusters(
    principles,
    labels,
    providers,
    role,
    seed,
    max_workers=1,
    progress=None,
    temperature=0.2,
    config=None,
):
    def valid_abstract(text):
        try:
            return valid_principle(text.strip().strip('"'))
        except ValueError as exc:
            raise ModelResponseError("Invalid DICAI abstraction output: " + str(exc)) from exc

    def abstract_cluster(cluster):
        members = [p for p, label in zip(principles, labels, strict=True) if label == cluster]
        return providers.complete(
            role,
            render_messages(
                "dicai", "abstraction", principles="\n".join("- " + p for p in members),
            ),
            seed=seed + cluster,
            validate=valid_abstract,
            response_context={"stage": "abstraction", "cluster": cluster},
            temperature=temperature,
        )

    return ordered_map(abstract_cluster, sorted(set(labels)), max_workers, progress)
