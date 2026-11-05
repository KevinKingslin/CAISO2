from .model_selector import ModelSelectorOptimizer


class HyperparameterOptimizer(ModelSelectorOptimizer):
    """Discrete hyperparameter configurations use the same softmax selection rule."""
