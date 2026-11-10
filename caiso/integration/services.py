"""Method-only model services. Benchmark model calls stay in examples/ unchanged."""

from dataclasses import dataclass
import json
import os

import numpy as np

from caiso.core.serialization import digest
from caiso.prompts import render_messages
from caiso.providers.base import ProviderRegistry
from caiso.providers.embeddings import SentenceTransformerEmbeddings
from caiso.utils import api
from caiso.providers.transport import observe_transport


class MethodProvider:
    def __init__(self, model, budget, max_tokens=4096):
        self.model, self.budget, self.max_tokens = model, budget, max_tokens
        self.on_error = None

    def complete(
        self, messages, *, seed, schema=None, temperature=0.2, max_tokens=None,
        include_schema_instruction=True, **kwargs,
    ):
        instructions = list(messages)
        if schema is not None and include_schema_instruction:
            instructions.extend(
                render_messages(
                    "method", "services.schema_message", schema=json.dumps(schema.model_json_schema()),
                )
            )
        with observe_transport(self.budget, self.on_error):
            generation = {"seed": seed, **kwargs}
            if schema is not None:
                generation.setdefault("response_format", {"type": "json_object"})
            result = api.get_llm_output(
                instructions,
                model=self.model,
                temperature=temperature,
                max_new_tokens=max_tokens or self.max_tokens,
                # Keep the wire-level JSON mode but let the method registry
                # parse/validate the original text and handle format retries.
                json_object=False,
                **generation,
            )
        # Support test/custom API implementations that already return decoded JSON.
        return json.dumps(result) if schema is not None and isinstance(result, (dict, list)) else result

    def fingerprint(self):
        return digest({"method_model": self.model, "max_tokens": self.max_tokens})


class MethodEmbeddings:
    def __init__(self, model, budget):
        self.model, self.budget = model, budget
        self.on_error = None
        self.provider = self

    def embed(self, texts):
        import litellm

        with observe_transport(self.budget, self.on_error):
            result = litellm.embedding(model=self.model, input=list(texts))
        return np.asarray(
            [row["embedding"] for row in sorted(result.data, key=lambda row: row["index"])]
        )

    def fingerprint(self):
        return digest({"method_embedding_model": self.model})


@dataclass
class LocalModelReference:
    """Training metadata; inference continues through the original vLLM client."""

    model_id: str

    def fingerprint(self):
        return digest({"model_id": self.model_id})


def build_services(system, config, budget):
    model = os.getenv("CAISO_METHOD_MODEL") or config.get("model", "gpt-4o")
    providers = {"method": MethodProvider(model, budget, config.get("max_tokens", 4096))}
    for cid, component in system.components.items():
        if hasattr(component.config, "base_model_path"):
            providers[cid] = LocalModelReference(component.config.base_model_path)
    roles = dict.fromkeys(
        (
            "system_constitution",
            "block_constitution",
            "constitution_judge",
            "block_selection",
            "prompt_optimizer",
            "workflow_generator",
        ),
        "method",
    )
    registry = ProviderRegistry(providers, roles)
    embeddings = config.get("embeddings", {})
    if embeddings.get("backend", "litellm") == "sentence_transformers":
        embedder = SentenceTransformerEmbeddings(
            embeddings.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            budget,
            device=embeddings.get("device", "cpu"),
        )
    elif embeddings.get("backend", "litellm") == "litellm":
        embedder = MethodEmbeddings(
            os.getenv("CAISO_EMBEDDING_MODEL") or embeddings.get("model", "text-embedding-3-small"), budget
        )
    else:
        raise ValueError("Unknown CAISO embedding backend")
    return registry, embedder
