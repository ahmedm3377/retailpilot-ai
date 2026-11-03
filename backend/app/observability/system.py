"""CPU, RAM and GPU readings for the dashboard, taken each time metrics are exported (telemetry on only)."""
from __future__ import annotations

import os
import time

import psutil

from app.evaluation.metrics import MIB, gpu_device_used_mb, gpu_utilization_pct, mcp_server_processes, server_processes


class SystemReadings:
    """One snapshot per export: every gauge's callback reads the same values."""

    MAX_AGE_S = 1.0

    def __init__(self) -> None:
        names = os.environ.get("LLM_SERVER_PROCESSES", "")
        self.server_names = tuple(n.strip().lower() for n in names.split(",") if n.strip())
        self.process = psutil.Process()
        self.taken_at = 0.0
        self.values: dict[str, float | None] = {}
        self.rss: dict[str, float] = {}
        psutil.cpu_percent(None)  # the first call only starts the measuring interval

    @staticmethod
    def _rss_mb(processes: list[psutil.Process]) -> float:
        total = 0.0
        for proc in processes:
            try:
                total += proc.memory_info().rss / MIB
            except psutil.Error:
                continue
        return round(total, 1)

    def read(self) -> "SystemReadings":
        if time.monotonic() - self.taken_at < self.MAX_AGE_S:
            return self
        self.taken_at = time.monotonic()
        self.values = {"cpu": psutil.cpu_percent(None), "memory": psutil.virtual_memory().percent,
                       "gpu_utilization": gpu_utilization_pct(), "gpu_memory": gpu_device_used_mb()}
        try:
            own = [self.process, *self.process.children(recursive=True)]
        except psutil.Error:
            own = [self.process]
        self.rss = {"app": self._rss_mb(own), "model_server": self._rss_mb(server_processes(self.server_names)),
                    "mcp_server": self._rss_mb(mcp_server_processes())}
        return self
