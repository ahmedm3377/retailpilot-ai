"""Agent, model and guardrail settings from the environment, plus offline defaults.

`apply_offline_defaults()` must run before litellm, transformers or guardrails
are imported: LiteLLM fetches its cost map from GitHub at import time and
Hugging Face libraries download models lazily.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

OFFLINE_ENV_DEFAULTS = {
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
}


def apply_offline_defaults() -> None:
    """Set offline/telemetry-off defaults without overriding explicit user values."""
    for key, value in OFFLINE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    # Guardrails ships an OpenTelemetry exporter to a hosted endpoint. When this app's own
    # (opt-in) telemetry is off, disable the OTel SDK entirely; when it is on, our private
    # providers need the SDK, and Guardrails' hub telemetry stays off via
    # app.guardrails.service.disable_guardrails_telemetry() (verified by the offline tests).
    if os.environ.get("RETAIL_OTEL_ENABLED", "false").strip().lower() not in ("1", "true", "yes", "on"):
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    # Loopback model calls must never be routed through a proxy.
    no_proxy = {h for h in os.environ.get("NO_PROXY", "").split(",") if h}
    no_proxy |= {"127.0.0.1", "localhost", "::1"}
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(sorted(no_proxy))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _flag(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AgentSettings:
    # The model and where it runs: any LiteLLM route, for example "ollama_chat/<name>" or
    # "hosted_vllm/<name>". Set in .env; there is no built-in model. LLM_API_KEY is read only when calling.
    model: str = ""
    api_base: str = ""
    allowed_hosts: str = ""  # extra non-loopback hosts the model may run on (comma separated)
    extra_params: str = "{}"  # JSON object: engine-specific parameters passed through to LiteLLM
    # What the contract's output declares about the model. No engine-neutral API reports these.
    engine: str = ""
    engine_version: str = ""
    model_revision: str = ""
    quantization: str = ""
    # For measurement only: names of the model server's processes, and how to evict the model for a cold start.
    server_processes: str = ""
    unload_command: str = ""
    timeout_s: float = 75.0
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 7
    max_tool_steps: int = 5
    checkpoint_schema: str = "retail_checkpoints"
    guardrails_enabled: bool = True
    # Measured locally: ordinary requests scored 0.71-0.75, jailbreaks 0.84 (9 samples).
    jailbreak_threshold: float = 0.8
    toxicity_threshold: float = 0.5
    fail_closed: bool = True
    max_input_chars: int = 2000
    # Confirmation turns word the stored result with the deterministic template
    # (fallback_message) instead of one more LLM call; proposals/denials/clarifications keep LLM wording.
    confirm_template: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.extra_param_values(), dict):
            raise ValueError("LLM_EXTRA_PARAMS must be a JSON object")

    def extra_param_values(self) -> dict[str, Any]:
        return json.loads(self.extra_params or "{}")

    def server_process_names(self) -> tuple[str, ...]:
        return tuple(n.strip().lower() for n in self.server_processes.split(",") if n.strip())

    @classmethod
    def from_env(cls) -> "AgentSettings":
        d = cls()
        return cls(
            model=_env("LLM_MODEL", d.model),
            api_base=_env("LLM_API_BASE", d.api_base),
            allowed_hosts=_env("LLM_ALLOWED_HOSTS", d.allowed_hosts),
            extra_params=_env("LLM_EXTRA_PARAMS", d.extra_params),
            engine=_env("LLM_ENGINE", d.engine),
            engine_version=_env("LLM_ENGINE_VERSION", d.engine_version),
            model_revision=_env("LLM_MODEL_REVISION", d.model_revision),
            quantization=_env("LLM_QUANTIZATION", d.quantization),
            server_processes=_env("LLM_SERVER_PROCESSES", d.server_processes),
            unload_command=_env("LLM_UNLOAD_COMMAND", d.unload_command),
            timeout_s=float(_env("LLM_TIMEOUT_SECONDS", str(d.timeout_s))),
            max_tokens=int(_env("LLM_MAX_TOKENS", str(d.max_tokens))),
            temperature=float(_env("LLM_TEMPERATURE", str(d.temperature))),
            top_p=float(_env("LLM_TOP_P", str(d.top_p))),
            seed=int(_env("LLM_SEED", str(d.seed))),
            max_tool_steps=int(_env("AGENT_MAX_TOOL_STEPS", str(d.max_tool_steps))),
            checkpoint_schema=_env("CHECKPOINT_SCHEMA", d.checkpoint_schema),
            guardrails_enabled=_flag("GUARDRAILS_ENABLED", d.guardrails_enabled),
            jailbreak_threshold=float(_env("GUARDRAILS_JAILBREAK_THRESHOLD", str(d.jailbreak_threshold))),
            toxicity_threshold=float(_env("GUARDRAILS_TOXICITY_THRESHOLD", str(d.toxicity_threshold))),
            fail_closed=_flag("GUARDRAILS_FAIL_CLOSED", d.fail_closed),
            max_input_chars=int(_env("GUARDRAILS_MAX_INPUT_CHARS", str(d.max_input_chars))),
            confirm_template=_flag("RETAIL_CONFIRM_TEMPLATE", d.confirm_template),
        )
