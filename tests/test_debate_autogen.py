"""Exercise real AutoGen agents without making external model requests."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import autogen

from caiso.constitution.dicai.agents import DebateAgent
from caiso.constitution.dicai.debate import debate
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.models import PreferencePair
from tests.prompt_requests import PRINCIPLES, Recorder


def pair(identifier="autogen-test"):
    return PreferencePair(
        pair_id=identifier, context={"question": identifier}, preferred="A",
        candidate_a={"answer": "A"}, candidate_b={"answer": "B"},
    )


def catalog():
    return [
        {"persona": p.model_dump(), "reasoning": "Preserve the evidence.", "principles": PRINCIPLES}
        for p in assign_personas(["First", "Second", "Third"], 42)
    ]


def test_real_agents_speak_in_source_order_with_one_moderator(monkeypatch):
    turns, proxies = [], []
    generate = autogen.AssistantAgent.generate_reply
    initialize = autogen.UserProxyAgent.__init__

    def record_reply(self, *, messages, sender, **kwargs):
        turns.append((self, sender, deepcopy(messages)))
        return generate(self, messages=messages, sender=sender, **kwargs)

    def record_proxy(self, *args, **kwargs):
        proxies.append((self, dict(kwargs)))
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(autogen.AssistantAgent, "generate_reply", record_reply)
    monkeypatch.setattr(autogen.UserProxyAgent, "__init__", record_proxy)
    values, history = debate(
        pair(), catalog(), Recorder(), "audit", 42, rounds=2,
        config={"min_debate_rounds": 2, "atomicity": {"enabled": False}},
    )

    assert values == PRINCIPLES
    assert history[-1]["finalized"] is True
    assert [agent.name for agent, _, _ in turns] == ["Debater_1", "Debater_2", "Debater_3", "Judge"] * 2
    assert all(isinstance(agent, autogen.AssistantAgent) for agent, _, _ in turns)
    assert all(turns[i][0] is turns[i + 4][0] for i in range(4))
    assert len({id(agent) for agent, _, _ in turns}) == 4
    assert len(proxies) == 1
    moderator, options = proxies[0]
    assert moderator.name == "Moderator_Proxy"
    assert options["human_input_mode"] == "NEVER"
    assert options["code_execution_config"] is False
    assert options["llm_config"] is False
    assert options["default_auto_reply"] == ""
    assert all(sender is moderator for _, sender, _ in turns)
    assert all(len(messages) == 1 and messages[0]["role"] == "user" for _, _, messages in turns)
    assert "[debater_1]" in turns[1][2][0]["content"]
    assert "[debater_3]" in turns[3][2][0]["content"]


def test_model_client_preserves_provider_options_and_leaves_caching_to_caiso(monkeypatch, tmp_path):
    class Provider(Recorder):
        def complete(self, role, messages, **kwargs):
            self.options.append((role, dict(kwargs)))
            return super().complete(role, messages, **kwargs)

    monkeypatch.chdir(tmp_path)
    provider = Provider()
    provider.options = []
    speaker = DebateAgent("Debater_1", provider, "audit")
    moderator = autogen.UserProxyAgent(
        "Moderator_Proxy", human_input_mode="NEVER", code_execution_config=False, llm_config=False,
    )
    messages = [
        {"role": "system", "content": "You are a debater who must defend evidence."},
        {"role": "user", "content": "Defend Q1."},
    ]
    original = deepcopy(messages)
    validated = []

    def validate(text):
        validated.append(text)
        return text

    options = {"seed": 17, "temperature": 0.7, "validate": validate,
               "response_context": {"pair_id": "autogen-test", "stage": "debater", "cycle": 1}}
    for _ in range(2):
        assert speaker.speak(moderator, messages, **options)

    assert provider.options == [("audit", options)] * 2
    assert len(validated) == 2
    assert [call["messages"] for call in provider.calls] == [original, original]
    assert messages == original
    assert speaker.request == {}
    assert list(tmp_path.iterdir()) == []


def test_concurrent_pairs_keep_agent_requests_isolated():
    barrier = Barrier(2)

    class Provider(Recorder):
        def complete(self, role, messages, **kwargs):
            context = kwargs.get("response_context", {})
            if context.get("stage") == "debater":
                identifier = context["pair_id"]
                assert identifier in messages[1]["content"]
                expected_seed = (100 if identifier == "pair-one" else 200) + context["debater"] - 1
                assert kwargs["seed"] == expected_seed
                if context["debater"] == 1:
                    barrier.wait(timeout=10)
            return super().complete(role, messages, **kwargs)

    def run(item):
        identifier, seed = item
        return debate(
            pair(identifier), catalog(), Provider(), "audit", seed, rounds=1,
            config={"atomicity": {"enabled": False}},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [("pair-one", 100), ("pair-two", 200)]))
    assert all(values == PRINCIPLES and history[-1]["finalized"] for values, history in results)
