"""Scoped, nullable resource measurements around one case run.

RAM: peak of (this Python process RSS + its children + the processes named in
LLM_SERVER_PROCESSES + the standalone MCP server), summed per sample. VRAM: whole-device nvidia-smi memory.used,
the one reading every engine shares: per-process VRAM is not available from
nvidia-smi on Windows (WDDM reports "Insufficient Permissions").
Anything that cannot be read stays None; nothing is estimated.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

import psutil

MIB = 1024 * 1024
RAM_SCOPE = ("adapter Python process RSS + RSS of its child processes + RSS of the processes named in LLM_SERVER_PROCESSES + RSS of the standalone MCP server (sampled together)")
VRAM_SCOPE = "nvidia-smi memory.used for the whole GPU, including other applications; null without nvidia-smi"
GPU_DEVICE_SCOPE = "nvidia-smi memory.used for the whole GPU, including other applications"


@dataclass
class ResourcePeaks:
    peak_ram_mb: float | None = None
    peak_python_rss_mb: float | None = None
    peak_model_server_rss_mb: float | None = None
    peak_mcp_server_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    peak_gpu_device_used_mb: float | None = None
    samples: int = 0
    server_processes_seen: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def server_processes(names: tuple[str, ...]) -> list[psutil.Process]:
    """Local processes whose name starts with one of the configured model-server names."""
    if not names:
        return []
    found = []
    for proc in psutil.process_iter(["name"]):
        if (proc.info.get("name") or "").lower().startswith(names):
            found.append(proc)
    return found


def mcp_server_processes() -> list[psutil.Process]:
    """The standalone MCP server, whoever started it. It runs under Python, so only its command line tells."""
    found = []
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if any(a == "-m" and b == "mcp_server" for a, b in zip(cmdline, cmdline[1:])):
            found.append(proc)
    return found


def gpu_device_used_mb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        return float(sum(float(v) for v in out.split()))
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def gpu_utilization_pct() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        return max(float(v) for v in out.split())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


class ResourceSampler:
    """Background sampler; use as a context manager around the measured work."""

    def __init__(self, server_names: tuple[str, ...] = (), interval_s: float = 0.1, slow_every: int = 10) -> None:
        self.server_names = server_names
        self.interval_s = interval_s
        self.slow_every = slow_every  # process discovery and nvidia-smi run every N samples (they cost more)
        self.peaks = ResourcePeaks()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._self = psutil.Process()
        self._servers: list[psutil.Process] = []
        self._mcp: list[psutil.Process] = []

    def __enter__(self) -> "ResourceSampler":
        self._servers = server_processes(self.server_names)
        self.peaks.server_processes_seen = bool(self._servers)
        self._mcp = mcp_server_processes()
        self._sample(slow=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._sample(slow=True)
        notes = []
        if not self.peaks.server_processes_seen:
            notes.append("no model server process found (see LLM_SERVER_PROCESSES): peak_ram_mb covers this "
                         "process and its children only")
        if not self._mcp:
            notes.append("no MCP server process found: peak_ram_mb doesn't include it")
        if self.peaks.peak_vram_mb is None:
            notes.append("accelerator memory unavailable: nvidia-smi is missing or reported nothing")
        self.peaks.notes = "; ".join(notes)

    def _run(self) -> None:
        n = 0
        while not self._stop.wait(self.interval_s):
            n += 1
            self._sample(slow=n % self.slow_every == 0)

    @staticmethod
    def _peak(current: float | None, value: float | None) -> float | None:
        if value is None:
            return current
        return value if current is None else max(current, value)

    def _sample(self, slow: bool) -> None:
        p = self.peaks
        try:
            python_mb = self._self.memory_info().rss / MIB
        except psutil.Error:
            return
        if slow and not self._mcp:  # looking through command lines costs more, and the server doesn't respawn
            self._mcp = mcp_server_processes()
        mcp_mb = 0.0
        for proc in self._mcp:
            try:
                mcp_mb += proc.memory_info().rss / MIB
            except psutil.Error:
                continue
        mcp_pids = {proc.pid for proc in self._mcp}
        children_mb = 0.0
        try:
            for child in self._self.children(recursive=True):
                if child.pid in mcp_pids:  # the adapter's own server: counted once, as the MCP server
                    continue
                try:
                    children_mb += child.memory_info().rss / MIB
                except psutil.Error:
                    continue
        except psutil.Error:
            pass
        if slow:  # a server may spawn a runner process when it loads a model; pick it up.
            self._servers = server_processes(self.server_names)
            p.server_processes_seen = p.server_processes_seen or bool(self._servers)
        server_mb = 0.0
        alive = []
        for proc in self._servers:
            try:
                server_mb += proc.memory_info().rss / MIB
                alive.append(proc)
            except psutil.Error:
                continue
        self._servers = alive
        p.samples += 1
        p.peak_python_rss_mb = self._peak(p.peak_python_rss_mb, round(python_mb, 1))
        p.peak_mcp_server_rss_mb = self._peak(p.peak_mcp_server_rss_mb, round(mcp_mb, 1))
        if p.server_processes_seen:
            p.peak_model_server_rss_mb = self._peak(p.peak_model_server_rss_mb, round(server_mb, 1))
        p.peak_ram_mb = self._peak(p.peak_ram_mb, round(python_mb + children_mb + server_mb + mcp_mb, 1))
        if slow:
            device = gpu_device_used_mb()
            p.peak_gpu_device_used_mb = self._peak(p.peak_gpu_device_used_mb, device)
            p.peak_vram_mb = self._peak(p.peak_vram_mb, device)


class Timer:
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self._start) / 1e6
