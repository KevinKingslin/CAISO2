"""One abstraction pass per cluster, using the existing induction model."""

from caiso.core.parallel import ordered_map
from caiso.core.recovery import ModelResponseError
from caiso.prompts import render_messages

from .parsing import normalize_principle, principle_subjects, referenced_components


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
    *, component_ids=None,
):
    principles = [normalize_principle(p, component_ids) for p in principles]
    known_ids = set(component_ids) if component_ids is not None else principle_subjects(principles)

    def abstract_cluster(cluster):
        members = [p for p, label in zip(principles, labels, strict=True) if label == cluster]
        allowed = sorted(referenced_components(members, known_ids))

        def valid_abstract(text):
            try:
                return normalize_principle(text, allowed)
            except (ValueError, ModelResponseError) as exc:
                raise ModelResponseError("Invalid DICAI abstraction output: " + str(exc)) from exc

        return providers.complete(
            role,
            render_messages(
                "dicai", "abstraction", principles="\n".join("- " + p for p in members),
                component_ids=allowed,
            ),
            seed=seed + cluster,
            validate=valid_abstract,
            response_context={"stage": "abstraction", "cluster": cluster},
            temperature=temperature,
        )

    return ordered_map(abstract_cluster, sorted(set(labels)), max_workers, progress)
