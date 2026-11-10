"""Classic AutoGen debate agents backed by CAISO's budgeted provider layer."""

from types import SimpleNamespace

import autogen

from caiso.core.recovery import ModelResponseError


class ProviderModelClient:
    """AutoGen ModelClient bridge; CAISO owns transport, retries and accounting."""

    def __init__(self, config, *, providers, role, request):
        self.model = config["model"]
        self.providers, self.role, self.request = providers, role, request

    def create(self, params):
        text = self.providers.complete(self.role, params["messages"], **self.request)
        return SimpleNamespace(
            model=self.model,
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            message_retrieval_function=self.message_retrieval,
        )

    @staticmethod
    def message_retrieval(response):
        return [choice.message.content for choice in response.choices]

    @staticmethod
    def cost(response):
        return 0.0  # Real usage and cost are already recorded by CAISO.

    @staticmethod
    def get_usage(response):
        return {}  # Do not invent token counts or duplicate provider accounting.


class DebateAgent:
    """One persistent AssistantAgent per participant, isolated to a single pair."""

    def __init__(self, name, providers, role):
        self.request = {}
        self.agent = autogen.AssistantAgent(
            name=name,
            system_message="",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config={
                "config_list": [{
                    "model": role,
                    "model_client_cls": ProviderModelClient.__name__,
                }],
                "cache_seed": None,
            },
        )
        # Keep providers (including their locks and request cache) out of the
        # llm_config that AutoGen deep-copies when constructing an agent.
        self.agent.register_model_client(
            ProviderModelClient, providers=providers, role=role, request=self.request,
        )

    def speak(self, moderator, messages, **request):
        self.agent.update_system_message(messages[0]["content"])
        self.request.update(request)
        try:
            # Match the original _speak: only the explicit current prompt is
            # sent; bounded shared history is already included in that prompt.
            reply = self.agent.generate_reply(
                messages=[dict(message) for message in messages[1:]], sender=moderator,
            )
        finally:
            self.request.clear()
        if isinstance(reply, dict):
            reply = reply.get("content", "")
        if not isinstance(reply, str) or not reply.strip():
            raise ModelResponseError(f"AutoGen agent {self.agent.name} returned no text")
        return reply.strip()


def moderator_proxy():
    return autogen.UserProxyAgent(
        name="Moderator_Proxy", human_input_mode="NEVER",
        code_execution_config=False, llm_config=False, default_auto_reply="",
    )
