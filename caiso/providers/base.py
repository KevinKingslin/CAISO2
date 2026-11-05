from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from pydantic import BaseModel, ValidationError

from caiso.core.recovery import ModelResponseError, Recovery
from caiso.prompts import entry as prompt_entry
from caiso.core.serialization import digest

DICAI_ROLES = {"system_constitution", "block_constitution", "constitution_judge", "block_selection"}


def check_dicai_provider(role, alias):
    """Reject legacy GPT-5 routes before a request, including saved run configs."""
    normalized = alias.lower().replace("-", "").replace("_", "")
    if role in DICAI_ROLES and "gpt5" in normalized:
        raise ValueError(
            f"{role} selects {alias!r}, but GPT-5 is disabled for DICAI. "
            "Use azure_gpt4o with AZURE_GPT4O_DEPLOYMENT. "
            "Start a new run/output directory when switching an old run's model."
        )


class Provider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        schema: type[BaseModel] | None = None,
        **kwargs,
    ) -> str: ...

    def fingerprint(self) -> str: ...


class ProviderRegistry:
    def __init__(
        self,
        providers: dict[str, Provider],
        roles: dict[str, str] | None = None,
        generation: dict[str, dict] | None = None,
    ):
        self.providers = providers
        self.roles = roles or {}
        self.generation = generation or {}
        self.recovery = Recovery()

    def get(self, alias):
        role = alias
        alias = self.roles.get(alias, alias)
        check_dicai_provider(role, alias)
        if alias not in self.providers:
            raise ValueError(f"Provider alias {alias!r} is not configured")
        return self.providers[alias]

    def complete(self, alias, messages, *, seed, schema=None, validate=None, **kwargs):
        kwargs = {**self.generation.get(alias, {}), **kwargs}
        provider = self.get(alias)  # Fail configuration errors before entering a retry loop.

        def request(attempt, previous_error):
            instructions = messages
            format_error = isinstance(previous_error, (ModelResponseError, ValidationError))
            if format_error and kwargs.get("prompt_format") != "completion":
                instructions = [
                    *messages,
                    {
                        "role": "user",
                        "content": prompt_entry('method', 'providers.format_retry').format(value_0=str(previous_error)[:1500]),
                    },
                ]
            retry_seed = (seed + attempt * 104729) % (2**31 - 1) if format_error else seed
            text = provider.complete(instructions, seed=retry_seed, schema=schema, **kwargs)
            # Raw completion can stop at EOS immediately. OPTIMAS passes that
            # empty context/feedback downstream rather than resampling it.
            if not isinstance(text, str) or (
                not text.strip() and kwargs.get("prompt_format") != "completion"
            ):
                raise ModelResponseError("Model returned an empty or non-text response")
            result = schema.model_validate_json(text) if schema else text
            return validate(result) if validate else result

        return self.recovery.call(
            request, stage=f"model:{alias}", retries=self.recovery.config.get("response_retries", 2)
        )

    def fingerprint(self):
        return digest(
            {
                "roles": self.roles,
                **({"generation": self.generation} if self.generation else {}),
                "providers": {k: p.fingerprint() for k, p in self.providers.items()},
            }
        )

    def batch(self, alias, requests, max_workers=1):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda r: self.complete(alias, **r), requests))


def messages(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
