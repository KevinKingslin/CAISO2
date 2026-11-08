import numpy as np

from .base import ConfigurationOptimizer, filter_current_configuration


def probabilities(scores, temperature=1.0):
    if temperature <= 0 or not scores or not np.isfinite(scores).all():
        raise ValueError("Softmax requires finite scores and a positive temperature")
    values = np.asarray(scores, dtype=float)
    exp = np.exp((values - values.max()) / temperature)
    return exp / exp.sum()


class ModelSelectorOptimizer(ConfigurationOptimizer):
    def __init__(self, temperature=1.0, sample=True):
        self.temperature, self.sample = temperature, sample

    def propose(self, component, context):
        current = context.state.configurations[component.component_id]
        candidates = []
        for value in component.config_search_space:
            c = current.model_copy(deep=True)
            c.value = value
            candidates.append(c)
        return filter_current_configuration(current, candidates, context.config)

    def select(self, candidates, scores, seed=0, final=False):
        p = probabilities(scores, self.temperature)
        if final or not self.sample:
            return candidates[int(np.argmax(scores))]
        return candidates[int(np.random.default_rng(seed).choice(len(candidates), p=p))]
