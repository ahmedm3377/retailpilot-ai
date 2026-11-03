"""Telemetry settings and the OTLP endpoint guard.

On this machine another project's collector listens on 127.0.0.1:4317/4318, so
those ports are refused outright: this app's telemetry goes only to its own
stack (observability/compose.yaml, OTLP/HTTP on 127.0.0.1:14318).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
FOREIGN_COLLECTOR_PORTS = {4317, 4318}
DEFAULT_ENDPOINT = "http://127.0.0.1:14318"


class TelemetryEndpointError(ValueError):
    pass


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def telemetry_enabled() -> bool:
    return _flag("RETAIL_OTEL_ENABLED", False)


def check_otlp_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        raise TelemetryEndpointError("OTLP endpoint must be plain http:// on loopback")
    if parsed.username or parsed.password:
        raise TelemetryEndpointError("OTLP endpoint must not contain credentials")
    if parsed.hostname not in LOCAL_HOSTS:
        raise TelemetryEndpointError(f"OTLP host {parsed.hostname!r} is not loopback")
    if parsed.port in FOREIGN_COLLECTOR_PORTS or parsed.port is None:
        raise TelemetryEndpointError("OTLP port must be explicit and not 4317/4318 (another project's collector "
                                     "listens there); use this project's 14318")
    if parsed.path not in ("", "/") or parsed.query:
        raise TelemetryEndpointError("OTLP endpoint must be the base URL (signal paths are added automatically)")
    return url.rstrip("/")


@dataclass(frozen=True)
class TelemetrySettings:
    enabled: bool = False
    endpoint: str = DEFAULT_ENDPOINT
    service_name: str = "hp-retail-agentic-ai"
    capture_content: bool = False
    max_content_chars: int = 8000
    export_interval_ms: int = 5000
    export_timeout_s: float = 2.0

    @classmethod
    def from_env(cls) -> "TelemetrySettings":
        d = cls()
        return cls(enabled=telemetry_enabled(),
                   endpoint=check_otlp_endpoint(os.environ.get("RETAIL_OTEL_ENDPOINT", d.endpoint)),
                   service_name=os.environ.get("OTEL_SERVICE_NAME", d.service_name),
                   capture_content=_flag("RETAIL_OTEL_CAPTURE_CONTENT", False),
                   max_content_chars=int(os.environ.get("RETAIL_OTEL_MAX_CONTENT_CHARS", d.max_content_chars)),
                   export_interval_ms=int(os.environ.get("RETAIL_OTEL_EXPORT_INTERVAL_MS", d.export_interval_ms)))
