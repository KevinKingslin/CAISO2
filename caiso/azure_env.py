"""Route existing LiteLLM model calls using deployment details from .env."""

import os
from functools import wraps


def azure_kwargs(model, kwargs):
    if os.getenv("LLM_BACKEND", "").lower() != "azure":
        return model, kwargs
    name = model.removeprefix("openai/")
    if model.startswith("azure/"):
        routed = model
    else:
        variable = "AZURE_" + name.upper().replace("-", "_").replace(".", "_") + "_DEPLOYMENT"
        legacy = {"gpt-4o": "AZURE_GPT4O_DEPLOYMENT", "gpt-4o-mini": "AZURE_GPT4O_MINI_DEPLOYMENT"}
        deployment = (os.getenv(variable) or os.getenv(legacy.get(name, variable), "")).strip()
        if not deployment:
            raise ValueError(f"Azure routing requires {variable} for model {model!r}")
        routed = "azure/" + deployment
    key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("AZURE_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
    if not all((key, endpoint, version)):
        raise ValueError("Set AZURE_API_KEY, AZURE_API_BASE and AZURE_API_VERSION in .env")
    # DSPy may pass an explicit OpenAI key/base. Replace those on Azure routes.
    kwargs = {k: v for k, v in kwargs.items() if k not in {"api_key", "api_base", "base_url", "api_version"}}
    kwargs.update(api_key=key, api_base=endpoint, api_version=version)
    return routed, kwargs


def install_azure_routing():
    """Install once, before DSPy imports; settings are read at request time."""
    import litellm

    for name in ("completion", "acompletion", "embedding", "aembedding"):
        original = getattr(litellm, name)
        if getattr(original, "_caiso_azure", False):
            continue

        def wrap(function):
            @wraps(function)
            def routed(*args, **kwargs):
                if args:
                    model, kwargs = azure_kwargs(args[0], kwargs)
                    args = (model, *args[1:])
                elif "model" in kwargs:
                    model = kwargs.pop("model")
                    model, kwargs = azure_kwargs(model, kwargs)
                    kwargs["model"] = model
                return function(*args, **kwargs)

            routed._caiso_azure = True
            return routed

        setattr(litellm, name, wrap(original))
