"""Introspected `hardware` and `configuration` for the results document.

Everything is read from this machine at run time (registry/proc, psutil,
nvidia-smi, the declared model settings, PostgreSQL, installed package metadata). Unreadable values
are reported as "unavailable", never guessed. No credentials are included.
"""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

import psutil

from app.agents.settings import OFFLINE_ENV_DEFAULTS, AgentSettings

UNAVAILABLE = "unavailable"
PACKAGES = ("litellm", "langgraph", "langgraph-checkpoint-postgres", "langchain-core", "guardrails-ai",
            "guardrails-ai-detect-jailbreak", "guardrails-ai-toxic-language", "guardrails-ai-detect-pii",
            "presidio-analyzer", "detoxify", "transformers", "torch", "psycopg", "pydantic", "psutil")


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or UNAVAILABLE
    return platform.processor() or UNAVAILABLE


def gpus() -> list[dict[str, str]] | str:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if not out:
        return "none detected (nvidia-smi unavailable)"
    rows = []
    for line in out.splitlines():
        name, memory, driver = (part.strip() for part in line.split(","))
        rows.append({"name": name, "memory_total": memory, "driver": driver})
    return rows


def power() -> dict[str, Any]:
    info: dict[str, Any] = {}
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    info["on_ac_power"] = battery.power_plugged if battery else UNAVAILABLE
    if sys.platform == "win32":
        scheme = _run(["powercfg", "/getactivescheme"])
        info["power_scheme"] = scheme.split("(", 1)[-1].rstrip(")") if scheme and "(" in scheme else UNAVAILABLE
    return info


def hardware() -> dict[str, Any]:
    return {
        "cpu": cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram": f"{psutil.virtual_memory().total / 2**30:.1f} GiB",
        "gpu": gpus(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "power": power(),
    }


def _safe_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.hostname}:{p.port}{p.path}"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = UNAVAILABLE
    return versions


def configuration(model_info: dict[str, Any], settings: AgentSettings, guard_description: dict[str, Any],
                  database_url: str, postgres_version: str) -> dict[str, Any]:
    return {
        "model": model_info.get("model"),
        "model_digest": model_info.get("digest"),
        "quantization": model_info.get("quantization"),
        "engine": model_info.get("engine"),
        "engine_version": model_info.get("engine_version"),
        "engine_url": _safe_url(settings.api_base),
        "llm_sdk": f"litellm (Python SDK, {settings.model.split('/', 1)[0]} route)",
        "declared_by": "model revision, quantization and engine come from LLM_* in .env; no neutral API reports them",
        "decoding": model_info.get("decoding"),
        "agents": ["customer_resolution_agent", "retail_operations_agent"],
        "max_tool_steps": settings.max_tool_steps,
        "tool_transport": "standalone MCP server (package mcp_server) over Streamable HTTP on loopback",
        "concurrency": 1,
        "guardrails": guard_description,
        "database": {"engine": "postgresql", "server_version": postgres_version, "url": _safe_url(database_url),
                     "isolation": "fresh business + checkpoint schema per case run"},
        "packages": package_versions(),
        "offline_env": {k: os.environ.get(k, UNAVAILABLE)
                        for k in (*OFFLINE_ENV_DEFAULTS, "OTEL_SDK_DISABLED", "RETAIL_OTEL_ENABLED")},
    }
