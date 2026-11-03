"""Machine load snapshot and the quiet-machine gate.

Earlier runs showed 3-5x slower turns while another model client, Chrome
and low free RAM competed with the benchmark. Each variant records a snapshot
before and after; the gate reports problems instead of killing user processes.
Command lines are reduced to the executable name and arguments without paths.
"""
from __future__ import annotations

import shlex
import subprocess
from datetime import datetime, timezone
from typing import Any

import psutil

from app.evaluation.metrics import gpu_device_used_mb

GROUPS = ("chrome", "msedge", "python", "node", "firefox")
# Measured on this machine (2026-10-29): the adapter process peaks at ~2.27 GB RSS and the
# model runner (llama-server, gemma4:e2b) at ~2.77 GB. Both are the workload, not contention.
# Required headroom = what the workload still has to allocate, plus ~0.2-0.8 GB margin.
MIN_FREE_RAM_MB_MODEL_RESIDENT = 3072       # adapter 2.27 GB + margin
MIN_FREE_RAM_MB_MODEL_NOT_RESIDENT = 5376   # adapter 2.27 GB + model server 2.77 GB + margin
MAX_IDLE_GPU_UTIL_PCT = 10
MODEL_RESIDENT_RSS_MB = 1024  # a model server this large has the model in memory


def unload_model(command: str, timeout_s: float = 60.0) -> bool:
    """Evict the model with LLM_UNLOAD_COMMAND, so that cold start measures a real model load.

    Engines differ, so the command comes from .env, which is trusted configuration. No shell is used.
    """
    if not command:
        return False
    try:
        return subprocess.run(shlex.split(command), capture_output=True, timeout=timeout_s).returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _gpu_util() -> float | None:
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        return float(max(float(v) for v in out.split()))
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _on_ac_power() -> bool | None:
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    return None if battery is None else bool(battery.power_plugged)


def load_snapshot(server_names: tuple[str, ...] = ()) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    groups: dict[str, dict[str, float]] = {g: {"count": 0, "rss_mb": 0.0} for g in GROUPS}
    server_rss = 0.0
    for proc in psutil.process_iter(["name", "memory_info"]):
        name = (proc.info.get("name") or "").lower()
        rss = proc.info.get("memory_info")
        rss_mb = (rss.rss if rss else 0) / 2**20
        if server_names and name.startswith(server_names):
            server_rss += rss_mb
        group = next((g for g in GROUPS if name.startswith(g)), None)
        if group is not None:
            groups[group]["count"] += 1
            groups[group]["rss_mb"] += rss_mb
    return {
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ram_total_mb": round(vm.total / 2**20), "ram_free_mb": round(vm.available / 2**20),
        "cpu_load_pct": psutil.cpu_percent(interval=1.0),
        "gpu_device_used_mb": gpu_device_used_mb(), "gpu_util_pct": _gpu_util(),
        "model_server_rss_mb": round(server_rss),
        "process_groups": {g: {"count": int(v["count"]), "rss_mb": round(v["rss_mb"])} for g, v in groups.items()},
        # Laptops throttle GPU and CPU on battery (measured here: decode fell from ~64 to ~19 tokens/s).
        "on_ac_power": _on_ac_power(),
        # RSS of the measuring process itself: its memory is the workload, not outside contention.
        "own_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
    }


def quiet_problems(snapshot: dict[str, Any]) -> list[str]:
    problems = []
    resident = snapshot.get("model_server_rss_mb", 0) >= MODEL_RESIDENT_RSS_MB
    minimum = MIN_FREE_RAM_MB_MODEL_RESIDENT if resident else MIN_FREE_RAM_MB_MODEL_NOT_RESIDENT
    available = snapshot["ram_free_mb"] + snapshot.get("own_rss_mb", 0)
    if available < minimum:
        problems.append(f"free RAM {snapshot['ram_free_mb']} MB + own {snapshot.get('own_rss_mb', 0)} MB "
                        f"< {minimum} MB (model {'resident' if resident else 'not resident'})")
    if snapshot.get("on_ac_power") is False:
        problems.append("running on battery (GPU/CPU are power-capped); plug in AC power")
    util = snapshot.get("gpu_util_pct")
    if util is not None and util > MAX_IDLE_GPU_UTIL_PCT:
        problems.append(f"GPU utilization {util}% while idle (> {MAX_IDLE_GPU_UTIL_PCT}%)")
    return problems


MIN_PROBE_RATIO = 0.8


def probe_problems(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[str]:
    """Contention during the run: the model's output rate fell by more than 20% from start to end."""
    b = (before or {}).get("median_decode_tokens_per_s")
    a = (after or {}).get("median_decode_tokens_per_s")
    if not b or not a:
        return ["decode probe unavailable"]
    if a < MIN_PROBE_RATIO * b:
        return [f"decode rate fell during the run: {b} -> {a} tokens/s (< {int(MIN_PROBE_RATIO * 100)}%)"]
    return []
