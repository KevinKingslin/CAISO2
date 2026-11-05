import pytest

from caiso.azure_env import azure_kwargs


def test_azure_routes_dspy_and_overrides_stale_credentials(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "azure")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    monkeypatch.setenv("AZURE_GPT_4O_MINI_DEPLOYMENT", "mini-deployment")
    model, kwargs = azure_kwargs("openai/gpt-4o-mini", {"api_key": "old", "base_url": "old", "temperature": 0.3})
    assert model == "azure/mini-deployment"
    assert kwargs == {"api_key": "test-key", "api_base": "https://test.openai.azure.com/", "api_version": "2024-10-21", "temperature": 0.3}


def test_azure_missing_mapping_fails(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "azure")
    monkeypatch.delenv("AZURE_GPT_4O_MINI_DEPLOYMENT", raising=False)
    with pytest.raises(ValueError, match="AZURE_GPT_4O_MINI_DEPLOYMENT"):
        azure_kwargs("gpt-4o-mini", {})


def test_original_routing_unchanged(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert azure_kwargs("openai/gpt-4o-mini", {"temperature": 0.6}) == ("openai/gpt-4o-mini", {"temperature": 0.6})


def test_installed_routing_covers_sync_and_async(monkeypatch):
    import asyncio
    import litellm
    from caiso.azure_env import install_azure_routing

    monkeypatch.setenv("LLM_BACKEND", "azure")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    monkeypatch.setenv("AZURE_GPT_4O_MINI_DEPLOYMENT", "mini")
    monkeypatch.setenv("AZURE_TEXT_EMBEDDING_3_SMALL_DEPLOYMENT", "embed")

    def sync(**kwargs):
        return kwargs

    async def async_call(**kwargs):
        return kwargs

    for name in ("completion", "embedding"):
        monkeypatch.setattr(litellm, name, sync)
    for name in ("acompletion", "aembedding"):
        monkeypatch.setattr(litellm, name, async_call)
    install_azure_routing()
    install_azure_routing()
    assert litellm.completion(model="openai/gpt-4o-mini")["model"] == "azure/mini"
    assert litellm.embedding(model="text-embedding-3-small")["model"] == "azure/embed"
    assert asyncio.run(litellm.acompletion(model="gpt-4o-mini"))["model"] == "azure/mini"
    assert asyncio.run(litellm.aembedding(model="text-embedding-3-small"))["model"] == "azure/embed"
