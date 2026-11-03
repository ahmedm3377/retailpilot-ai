from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents import model_client
from app.agents.model_client import ModelClient, ModelEndpointError, ModelUnavailableError, check_endpoint
from app.agents.schemas import CustomerMessage, json_schema
from app.agents.settings import AgentSettings


LOCAL = {"model": "provider/some-model", "api_base": "http://127.0.0.1:9"}


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:8000/v1", "http://[::1]:11434",
                                 "https://127.0.0.1:8443/v1"])
def test_local_endpoints_allowed_with_any_path(url):
    assert check_endpoint(url).startswith("http")


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1",           # hosted: the brief forbids it
    "http://10.0.0.5:11434",               # non-loopback without explicit allowlisting
    "http://models.example.com:8000/v1",
    "http://user:pw@127.0.0.1:11434",      # credentials belong in LLM_API_KEY
    "http://127.0.0.1:8000/v1?x=1",
    "file:///etc/passwd",
    "",
])
def test_hosted_or_odd_endpoints_rejected(url):
    with pytest.raises(ModelEndpointError):
        check_endpoint(url)


def test_own_server_only_when_listed_in_allowed_hosts():
    with pytest.raises(ModelEndpointError):
        ModelClient(AgentSettings(model="provider/m", api_base="http://gpu-box:8000/v1"))
    client = ModelClient(AgentSettings(model="provider/m", api_base="http://gpu-box:8000/v1", allowed_hosts="gpu-box"))
    assert client.base_url == "http://gpu-box:8000/v1"


@pytest.mark.parametrize("route", ["ollama_chat/any-model", "hosted_vllm/org/any-model", "openai/any-model"])
def test_any_litellm_route_is_accepted_on_a_local_endpoint(route):
    assert ModelClient(AgentSettings(model=route, api_base="http://127.0.0.1:9")).model == route


def test_the_call_uses_the_configured_route_endpoint_key_and_extra_params(monkeypatch):
    seen = {}

    def completion(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    monkeypatch.setattr(model_client.litellm, "completion", completion)
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    settings = AgentSettings(model="hosted_vllm/org/m", api_base="http://127.0.0.1:8000/v1",
                             extra_params='{"top_k": 5}', seed=11)
    reply = ModelClient(settings).chat_json("customer_resolution_agent", [{"role": "user", "content": "hi"}], {})
    assert (seen["model"], seen["api_base"], seen["api_key"]) == ("hosted_vllm/org/m", "http://127.0.0.1:8000/v1", "secret-key")
    assert seen["top_k"] == 5 and seen["seed"] == 11 and reply.call.completion_tokens == 2
    assert "secret-key" not in repr(settings)  # the key is read when calling, never kept where it could be recorded


def test_down_model_maps_to_unavailable_without_fallback():
    # Port 9 (discard) is closed on loopback: the call must fail, never fall back elsewhere.
    client = ModelClient(AgentSettings(timeout_s=3, **LOCAL))
    with pytest.raises(ModelUnavailableError):
        client.chat_json("customer_resolution_agent", [{"role": "user", "content": "hi"}],
                         json_schema(CustomerMessage))
    assert client.calls and not client.calls[-1].ok
    with pytest.raises(ModelUnavailableError):
        client.ready()


def test_ready_declares_what_env_says_and_marks_the_rest_unavailable(monkeypatch):
    class Http:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return SimpleNamespace(status_code=404)  # any answer means the server is up

    monkeypatch.setattr(model_client.httpx, "Client", lambda **kwargs: Http())
    info = ModelClient(AgentSettings(engine="vllm", quantization="Q8", **LOCAL)).ready()
    assert (info["engine"], info["quantization"], info["digest"], info["engine_version"]) == (
        "vllm", "Q8", "unavailable", "unavailable")


