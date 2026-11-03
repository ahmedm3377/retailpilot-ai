"""The only path from agents to an LLM: the LiteLLM SDK, to whichever model .env names.

Nothing here knows an engine. LLM_MODEL is a LiteLLM route (for example `ollama_chat/<name>` or
`hosted_vllm/<name>`), LLM_API_BASE is where it runs, and LLM_EXTRA_PARAMS carries anything
engine-specific. The chosen model must honour `response_format` json_schema, because agents emit one
schema-constrained JSON object per call, dispatched by app.agents.tool_registry.

The brief requires local inference, so the endpoint must be loopback or a host listed in
LLM_ALLOWED_HOSTS. There is no fallback model or provider: if the configured one fails, the call fails.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from app.agents.settings import AgentSettings, apply_offline_defaults
from app.observability import telemetry as tel

apply_offline_defaults()
import litellm  # noqa: E402  (must follow apply_offline_defaults)

litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.success_callback = []
litellm.failure_callback = []
litellm.callbacks = []

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
UNAVAILABLE = "unavailable"


class ModelEndpointError(ValueError):
    """The configured model or endpoint is outside the local allowlist."""


class ModelUnavailableError(RuntimeError):
    """The local model did not answer (down, timeout, bad response)."""


def check_endpoint(url: str, private_hosts: frozenset[str] = frozenset()) -> str:
    """Accept only a loopback or explicitly allowed model server: the brief forbids hosted LLMs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ModelEndpointError(f"LLM_API_BASE must be an http(s) URL on a local host, got {parsed.scheme!r}")
    if parsed.username or parsed.password:
        raise ModelEndpointError("LLM_API_BASE must not contain credentials; use LLM_API_KEY")
    if parsed.hostname not in LOCAL_HOSTS | set(private_hosts):
        raise ModelEndpointError(f"model host {parsed.hostname!r} is not local; add it to LLM_ALLOWED_HOSTS "
                                 "only if it is your own server")
    if parsed.query or parsed.fragment:
        raise ModelEndpointError("LLM_API_BASE must be a base URL, without a query or fragment")
    return url.rstrip("/")


def check_model(model: str) -> str:
    """A LiteLLM route: provider/name. Which providers are allowed is decided by the endpoint check."""
    if not model:
        raise ModelEndpointError("LLM_MODEL is not set; see .env.example")
    if "/" not in model:
        raise ModelEndpointError(f"LLM_MODEL {model!r} must be a LiteLLM route, such as provider/model-name")
    return model


@dataclass(frozen=True)
class ModelCall:
    """Audit record of one LLM call; no prompt or completion text."""

    role: str
    model: str
    latency_ms: float
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


@dataclass
class ModelReply:
    text: str
    call: ModelCall


class ChatModel(Protocol):
    """What agents depend on; ModelClient is the real one, tests may inject a fake."""

    model: str

    def chat_json(self, role: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> ModelReply: ...


@dataclass
class ModelClient:
    settings: AgentSettings
    private_hosts: frozenset[str] = frozenset()
    calls: list[ModelCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        configured = {h.strip() for h in self.settings.allowed_hosts.split(",") if h.strip()}
        self.base_url = check_endpoint(self.settings.api_base, self.private_hosts | configured)
        self.model = check_model(self.settings.model)

    def decoding(self) -> dict[str, Any]:
        s = self.settings
        return {"temperature": s.temperature, "top_p": s.top_p, "seed": s.seed, "max_tokens": s.max_tokens,
                "timeout_s": s.timeout_s, "extra_params": s.extra_param_values()}

    def chat_json(self, role: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> ModelReply:
        """One completion constrained to `schema`; returns raw text for guardrail parsing."""
        s = self.settings
        provider, model_name = self.model.split("/", 1)
        with tel.span(f"chat {model_name}", {"gen_ai.operation.name": "chat", "gen_ai.system": provider,
                                              "gen_ai.request.model": model_name,
                                              "retail.agent.role": tel.role(role)}) as span:
            tel.capture(span, input=messages)
            start = time.perf_counter()
            try:
                response = litellm.completion(
                    model=self.model, api_base=self.base_url, api_key=os.environ.get("LLM_API_KEY") or "local",
                    messages=messages,
                    response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}},
                    temperature=s.temperature, top_p=s.top_p, seed=s.seed,
                    max_tokens=s.max_tokens, timeout=s.timeout_s, num_retries=0, **s.extra_param_values())
            except Exception as error:  # noqa: BLE001 - mapped to one typed error
                call = ModelCall(role, self.model, (time.perf_counter() - start) * 1000, False,
                                 error=type(error).__name__)
                self.calls.append(call)
                tel.set_attributes(span, {"retail.llm.outcome": "error"})
                if role in tel.AGENT_ROLES:
                    tel.record_llm_call(role, call.latency_ms / 1000, False, None, None)
                raise ModelUnavailableError(f"{role}: local model call failed ({type(error).__name__})") from error
            usage = getattr(response, "usage", None)
            call = ModelCall(role, self.model, (time.perf_counter() - start) * 1000, True,
                             prompt_tokens=getattr(usage, "prompt_tokens", None),
                             completion_tokens=getattr(usage, "completion_tokens", None))
            self.calls.append(call)
            # Token attributes only when the engine reported them (None is dropped, never recorded as 0).
            tel.set_attributes(span, {"retail.llm.outcome": "ok",
                                      "gen_ai.usage.input_tokens": call.prompt_tokens,
                                      "gen_ai.usage.output_tokens": call.completion_tokens})
            if role in tel.AGENT_ROLES:
                tel.record_llm_call(role, call.latency_ms / 1000, True, call.prompt_tokens, call.completion_tokens)
            content = response.choices[0].message.content or ""
            tel.capture(span, output=content)
            return ModelReply(content, call)

    def ready(self) -> dict[str, Any]:
        """Whether the model server answers at all, plus what .env declares about the model.

        Any HTTP status counts: engines differ in what they serve at their base URL. A wrong model
        name shows up in the first real call, which the warm-up makes at startup.
        """
        s = self.settings
        try:
            with httpx.Client(timeout=5, trust_env=False, follow_redirects=False) as http:
                http.get(self.base_url)
        except httpx.HTTPError as error:
            raise ModelUnavailableError(f"model server not reachable at {self.base_url}: "
                                        f"{type(error).__name__}") from error
        return {"model": self.model, "engine": s.engine or UNAVAILABLE, "engine_version": s.engine_version or UNAVAILABLE,
                "digest": s.model_revision or UNAVAILABLE, "quantization": s.quantization or UNAVAILABLE,
                "decoding": self.decoding()}
