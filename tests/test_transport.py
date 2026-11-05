import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from caiso.core.content_filter import ContentFilteredError
from caiso.core.recovery import Recovery, TransportExhausted, TransportRequestError
from caiso.providers import transport


@pytest.fixture
def setup_transport(monkeypatch, tmp_path):
    config = tmp_path / "providers.yaml"
    config.write_text(
        "azure_defaults:\n  requests_per_minute: 60\n  concurrency: 1\n  rate_limit_retries: 1\n  retry_jitter_seconds: 0\n"
    )
    transport.configure_transport(tmp_path, config)
    clock = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(
        transport, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 0, sleep=sleep)
    )
    yield clock, sleeps, tmp_path
    transport.configure_transport()


def http_error(status, headers=None, body=None):
    response = httpx.Response(
        status, headers=headers, json=body, request=httpx.Request("POST", "https://test.invalid")
    )
    return httpx.HTTPStatusError("service error", request=response.request, response=response)


def test_429_honors_millisecond_header_and_counts_physical_requests(setup_transport):
    clock, _, output = setup_transport
    calls, charged = [], []

    def call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise http_error(429, {"retry-after-ms": "1500"})
        return {"choices": [{"finish_reason": "stop"}]}

    with transport.observe_transport(
        SimpleNamespace(check=lambda: None, call=lambda **kw: charged.append(kw))
    ):
        transport.request(
            call,
            "azure/test",
            (),
            {"model": "azure/test", "api_base": "https://test.invalid", "num_retries": 5},
        )
    assert len(calls) == len(charged) == 2
    assert all(call["num_retries"] == 0 for call in calls)
    assert clock[0] >= 1.5
    events = [
        json.loads(line)
        for line in (output / ".internal/logs/transport.jsonl").read_text().splitlines()
    ]
    assert events[0]["retry_after_seconds"] == 1.5
    assert events[0]["status_code"] == 429


def test_deployment_pacing_is_shared_across_callers(setup_transport):
    clock, _, _ = setup_transport
    first = transport.get_controller("azure/test", {"api_base": "https://test.invalid"})
    second = transport.get_controller("azure/test", {"api_base": "https://test.invalid/"})
    assert first is second
    assert first.delay() == 0
    assert second.delay() == 1
    assert first.gate.acquire(blocking=False)
    assert not second.gate.acquire(blocking=False)
    first.gate.release()
    clock[0] = 2
    assert second.delay() == 0


def test_transport_exhaustion_is_not_retried_by_response_loop(setup_transport):
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        raise http_error(429, {"Retry-After": "0"})

    with pytest.raises(TransportExhausted):
        Recovery({"operation_retries": 3}).call(
            lambda *args: transport.request(call, "azure/test", (), {"model": "azure/test"}),
            stage="test",
        )
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_other_service_errors_are_not_policy_skips_or_retries(setup_transport, status):
    calls = []

    def call(**kwargs):
        calls.append(1)
        raise http_error(status, body={"error": {"code": "invalid_request"}})

    with pytest.raises(TransportRequestError):
        transport.request(call, "azure/test", (), {})
    assert calls == [1]


@pytest.mark.parametrize("output_filter", [False, True])
def test_explicit_policy_filter_is_never_retried(setup_transport, output_filter):
    calls = []

    def call(**kwargs):
        calls.append(1)
        if output_filter:
            return {"choices": [{"finish_reason": "content_filter"}]}
        raise http_error(
            400,
            body={
                "error": {
                    "code": "content_filter",
                    "innererror": {"code": "ResponsibleAIPolicyViolation"},
                }
            },
        )

    with pytest.raises(ContentFilteredError):
        transport.request(call, "azure/test", (), {})
    assert calls == [1]


def test_async_transport_handles_policy_errors_and_releases_slot(setup_transport):
    async def reject(**kwargs):
        raise http_error(400, body={"error": {"code": "content_filter"}})

    with pytest.raises(ContentFilteredError):
        asyncio.run(transport.arequest(reject, "azure/test", (), {}))
    controller = transport.get_controller("azure/test", {})
    assert controller.gate.acquire(blocking=False)
    controller.gate.release()


def test_retry_after_http_date_and_long_server_wait_are_not_capped(setup_transport):
    controller = transport.get_controller("azure/test", {})
    error = http_error(429, {"Retry-After": "Thu, 01 Jan 1970 00:02:00 GMT"})
    assert controller.retry_delay(error.response, 0) == 120
    assert controller.retry_delay(http_error(429, {"Retry-After": "180"}).response, 0) == 180


def test_async_429_waits_and_recovers(setup_transport, monkeypatch):
    clock, sleeps, _ = setup_transport
    calls = []

    async def delay(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(transport, "await_delay", delay)

    async def call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise http_error(429, {"Retry-After": "2"})
        return {"choices": [{"finish_reason": "stop"}]}

    asyncio.run(transport.arequest(call, "azure/test", (), {}))
    assert len(calls) == 2
    assert clock[0] >= 2


def test_bad_limits_fail_before_any_requests(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text("azure_defaults:\n  requests_per_minute: 0\n")
    with pytest.raises(ValueError, match="positive"):
        transport.configure_transport(path=path)


def test_policy_scope_blocks_fallback_but_not_another_example(setup_transport):
    calls = []

    def reject(**kwargs):
        calls.append(1)
        return {"choices": [{"finish_reason": "content_filter"}]}

    with transport.request_scope():
        for _ in range(3):
            with pytest.raises(ContentFilteredError):
                transport.request(reject, "azure/test", (), {})
    assert calls == [1]
    with transport.request_scope():
        with pytest.raises(ContentFilteredError):
            transport.request(reject, "azure/test", (), {})
    assert calls == [1, 1]
