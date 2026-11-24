"""Embedded time-breakdown, RAM, CPU, and PID-scoped GPU memory monitor.

Design follows ``medfm-flare/FLARE-Efficiency`` (background monotonic sampling,
trapezoidal AUC integration, and GPU ranking tolerance) while isolating RAM and
GPU measurements to the active inference process tree so shared hosts do not
contaminate CPU sanity checks or memory baselines.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import psutil

logger = logging.getLogger(__name__)

_MIB = 1024.0 * 1024.0
DEFAULT_GPU_TOLERANCE_MB = 4096.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def get_process_max_rss_mb() -> float:
    """Return kernel high-water-mark RSS for the current process in MiB.

    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` reports KiB on Linux and bytes
    on macOS/Darwin. Returns max across self and waited children when available.
    """
    divisor = _MIB if sys.platform == "darwin" else 1024.0
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / divisor
    return float(max(self_rss, child_rss))


def ranking_gpu_memory(
    max_gpu_mb: float,
    tolerance_mb: float = DEFAULT_GPU_TOLERANCE_MB,
) -> float:
    """FLARE-Efficiency GPU ranking score: ``max(0, peak - tolerance)`` rounded."""
    return float(round(max(0.0, float(max_gpu_mb) - float(tolerance_mb))))


def _auc(seconds: list[float], values: list[float]) -> float:
    """Trapezoidal integral over real sample timestamps."""
    if len(values) < 2 or len(seconds) < 2:
        return 0.0
    n = min(len(seconds), len(values))
    return float(_trapezoid(values[:n], seconds[:n]))


def _process_tree_pids(root_proc: psutil.Process) -> set[int]:
    pids = {root_proc.pid}
    try:
        for child in root_proc.children(recursive=True):
            pids.add(child.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return pids


class PidGpuSampler:
    """Sample GPU memory (MiB) allocated specifically by the target process tree.

    Uses ``pynvml`` (provided by ``nvidia-ml-py``) and falls back to
    ``nvidia-smi --query-compute-apps`` if NVML Python bindings are unavailable.
    Querying by process PID avoids false non-zero readings on shared GPUs.
    """

    def __init__(self, gpu_index: int | None = None) -> None:
        self.gpu_index = gpu_index
        self.backend = "none"
        self.gpu_name: str | None = None
        self._pynvml = None
        self._handles: list[Any] = []
        self._nvidia_smi = shutil.which("nvidia-smi")

        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                indices = [gpu_index] if (gpu_index is not None and 0 <= gpu_index < count) else list(range(count))
                for idx in indices:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    self._handles.append(handle)
                if self._handles:
                    raw_name = pynvml.nvmlDeviceGetName(self._handles[0])
                    self.gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                self._pynvml = pynvml
                self.backend = "pynvml"
            else:
                pynvml.nvmlShutdown()
        except Exception as exc:
            logger.debug("pynvml unavailable (%s); falling back to nvidia-smi/none", exc)
            self._pynvml = None
            self._handles = []
            if self._nvidia_smi:
                self.backend = "nvidia-smi"

    def read_mb(self, pids: set[int]) -> float:
        """Return total GPU memory in MiB used by any PID in ``pids``."""
        if not pids:
            return 0.0
        if self.backend == "pynvml" and self._pynvml is not None:
            total_bytes = 0
            seen_per_gpu_pid: set[tuple[int, int]] = set()
            for gpu_idx, handle in enumerate(self._handles):
                for fn_name in (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    fn = getattr(self._pynvml, fn_name, None)
                    if fn is None:
                        continue
                    try:
                        procs = fn(handle)
                    except Exception:
                        continue
                    for proc in procs:
                        pid = int(getattr(proc, "pid", -1))
                        if pid in pids and (gpu_idx, pid) not in seen_per_gpu_pid:
                            seen_per_gpu_pid.add((gpu_idx, pid))
                            used = getattr(proc, "usedGpuMemory", 0) or 0
                            total_bytes += int(used)
            return float(total_bytes / _MIB)

        if self.backend == "nvidia-smi" and self._nvidia_smi:
            try:
                res = subprocess.run(
                    [
                        self._nvidia_smi,
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if res.returncode != 0:
                    return 0.0
                total_mb = 0.0
                for line in res.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 2:
                        continue
                    try:
                        pid = int(parts[0])
                        mem = float(parts[1])
                    except ValueError:
                        continue
                    if pid in pids:
                        total_mb += mem
                return float(total_mb)
            except Exception:
                return 0.0

        return 0.0

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._pynvml = None
            self._handles = []


def collect_torch_cuda_stats(device: str = "cpu") -> dict[str, Any]:
    """Read PyTorch CUDA availability and peak memory stats without triggering CUDA init on CPU."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return {
            "torch_loaded": False,
            "cuda_available": False,
            "cuda_initialized": False,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }
    cuda_available = bool(torch_mod.cuda.is_available())
    cuda_initialized = bool(cuda_available and torch_mod.cuda.is_initialized())
    alloc_mb = 0.0
    reserved_mb = 0.0
    if cuda_initialized:
        try:
            dev_arg = torch_mod.device(device) if device.startswith("cuda") else None
            alloc_mb = float(torch_mod.cuda.max_memory_allocated(dev_arg) / _MIB)
            reserved_mb = float(torch_mod.cuda.max_memory_reserved(dev_arg) / _MIB)
        except Exception:
            alloc_mb = float(torch_mod.cuda.max_memory_allocated() / _MIB)
            reserved_mb = float(torch_mod.cuda.max_memory_reserved() / _MIB)
    return {
        "torch_loaded": True,
        "cuda_available": cuda_available,
        "cuda_initialized": cuda_initialized,
        "peak_allocated_mb": alloc_mb,
        "peak_reserved_mb": reserved_mb,
    }


@dataclass
class Samples:
    """Time series collected by :class:`CaseProfiler`."""

    interval: float
    seconds: list[float] = field(default_factory=list)
    cpu_percent: list[float] = field(default_factory=list)
    ram_mb: list[float] = field(default_factory=list)
    gpu_mb: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.seconds)


def summarize_samples(
    samples: Samples,
    timings: dict[str, float],
    *,
    device: str = "cpu",
    ram_baseline_mb: float = 0.0,
    ru_maxrss_mb: float = 0.0,
    inference_spans: list[tuple[float, float]] | None = None,
    torch_cuda_stats: dict[str, Any] | None = None,
    gpu_tolerance_mb: float = DEFAULT_GPU_TOLERANCE_MB,
) -> dict[str, Any]:
    """Compute scalar efficiency metrics from raw samples and timing phases."""
    model_load_s = float(timings.get("model_load_s", 0.0))
    inference_s = float(timings.get("inference_s", 0.0))
    runtime_s = model_load_s + inference_s

    sampled_ram_peak = max(samples.ram_mb) if samples.ram_mb else 0.0
    max_ram_mb = max(sampled_ram_peak, float(ru_maxrss_mb), float(ram_baseline_mb))
    ram_delta_mb = max(0.0, max_ram_mb - float(ram_baseline_mb))

    inf_ram_vals: list[float] = []
    if inference_spans and samples.seconds and samples.ram_mb:
        for t, r in zip(samples.seconds, samples.ram_mb):
            if any(lo <= t <= hi for lo, hi in inference_spans):
                inf_ram_vals.append(r)
    ram_inference_peak_mb = max(inf_ram_vals) if inf_ram_vals else sampled_ram_peak

    max_cpu_pct = max(samples.cpu_percent) if samples.cpu_percent else 0.0
    auc_cpu = _auc(samples.seconds, samples.cpu_percent)
    auc_ram = _auc(samples.seconds, samples.ram_mb)

    cuda_stats = torch_cuda_stats or {}
    torch_alloc_mb = float(cuda_stats.get("peak_allocated_mb", 0.0))
    torch_reserved_mb = float(cuda_stats.get("peak_reserved_mb", 0.0))
    cuda_available = bool(cuda_stats.get("cuda_available", False))

    nvml_peak_mb = max(samples.gpu_mb) if samples.gpu_mb else 0.0
    max_gpu_mb = max(nvml_peak_mb, torch_reserved_mb)
    auc_gpu = _auc(samples.seconds, samples.gpu_mb)
    rank_gpu_mb = ranking_gpu_memory(max_gpu_mb, gpu_tolerance_mb)

    is_cpu = device == "cpu"
    if is_cpu:
        gpu_sanity_ok = (
            max_gpu_mb == 0.0
            and torch_alloc_mb == 0.0
            and torch_reserved_mb == 0.0
            and not cuda_available
        )
    else:
        gpu_sanity_ok = cuda_available

    return {
        "runtime_s": round(runtime_s, 6),
        "model_load_s": round(model_load_s, 6),
        "inference_s": round(inference_s, 6),
        "data_load_s": round(float(timings.get("data_load_s", 0.0)), 6),
        "preprocess_s": round(float(timings.get("preprocess_s", 0.0)), 6),
        "postprocess_s": round(float(timings.get("postprocess_s", 0.0)), 6),
        "save_s": round(float(timings.get("save_s", 0.0)), 6),
        "wall_total_s": round(float(timings.get("wall_total_s", 0.0)), 6),
        "MaxRAM_MB": round(max_ram_mb, 3),
        "ram_baseline_mb": round(float(ram_baseline_mb), 3),
        "ram_inference_peak_mb": round(float(ram_inference_peak_mb), 3),
        "ram_delta_mb": round(ram_delta_mb, 3),
        "AUC_RAM_Time_MB_s": round(auc_ram, 3),
        "MaxCPU_Utilization_pct": round(max_cpu_pct, 3),
        "AUC_CPU_Time_pct_s": round(auc_cpu, 3),
        "MaxGPU_Mem_MB": round(max_gpu_mb, 3),
        "gpu_nvml_peak_mb": round(nvml_peak_mb, 3),
        "AUC_GPU_Time_MB_s": round(auc_gpu, 3),
        "RankGPU_Mem_MB": rank_gpu_mb,
        "gpu_torch_peak_allocated_mb": round(torch_alloc_mb, 3),
        "gpu_torch_peak_reserved_mb": round(torch_reserved_mb, 3),
        "gpu_sanity_ok": bool(gpu_sanity_ok),
    }


class CaseProfiler:
    """Context manager measuring phase timings, RAM, CPU, and PID-scoped GPU memory."""

    def __init__(
        self,
        device: str = "cpu",
        gpu_index: int | None = None,
        sample_interval: float = 0.1,
        gpu_tolerance_mb: float = DEFAULT_GPU_TOLERANCE_MB,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError("sample_interval must be > 0")
        if gpu_tolerance_mb < 0:
            raise ValueError("gpu_tolerance_mb must be >= 0")
        self.device = device
        self.gpu_index = gpu_index
        self.sample_interval = float(sample_interval)
        self.gpu_tolerance_mb = float(gpu_tolerance_mb)

        self.samples = Samples(interval=self.sample_interval)
        self.timings: dict[str, float] = {}
        self.phase_spans: dict[str, list[tuple[float, float]]] = {}
        self.per_lesion_inference_s: dict[str, float] = {}

        self._root_proc = psutil.Process(os.getpid())
        self._tracked_procs: dict[int, psutil.Process] = {self._root_proc.pid: self._root_proc}
        self._logical_cpus = max(1, psutil.cpu_count(logical=True) or 1)
        self._gpu_sampler = PidGpuSampler(gpu_index=gpu_index)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._start_mono: float = 0.0
        self._start_perf: float = 0.0
        self.ram_baseline_mb: float = 0.0
        self._finalized_record: dict[str, Any] | None = None

    def start(self) -> "CaseProfiler":
        if self._thread is not None:
            raise RuntimeError("CaseProfiler already started")
        self._start_mono = time.monotonic()
        self._start_perf = time.perf_counter()
        self.ram_baseline_mb = self._read_tree_rss_mb()
        self._prime_cpu_counters()
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="case-profiler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Samples:
        if self._stopped:
            return self.samples
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0 * self.sample_interval + 2.0)
            self._thread = None
        if "wall_total_s" not in self.timings:
            self.timings["wall_total_s"] = time.perf_counter() - self._start_perf
        self._sample_once()
        return self.samples

    def _sync_cuda_if_needed(self) -> None:
        if not self.device.startswith("cuda"):
            return
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and torch_mod.cuda.is_available() and torch_mod.cuda.is_initialized():
            torch_mod.cuda.synchronize(torch_mod.device(self.device))

    @contextmanager
    def phase(self, name: str):
        """Time a named execution phase (synchronizing CUDA on GPU runs)."""
        self._sync_cuda_if_needed()
        t0_rel = time.monotonic() - self._start_mono
        t0_perf = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda_if_needed()
            dt = time.perf_counter() - t0_perf
            t1_rel = time.monotonic() - self._start_mono
            self.timings[f"{name}_s"] = self.timings.get(f"{name}_s", 0.0) + dt
            self.phase_spans.setdefault(name, []).append((t0_rel, t1_rel))
            self._sample_once()

    def record_lesion_time(self, label_id: int, elapsed_s: float) -> None:
        self.per_lesion_inference_s[str(int(label_id))] = round(float(elapsed_s), 6)

    def _refresh_tree(self) -> list[psutil.Process]:
        procs: list[psutil.Process] = [self._root_proc]
        try:
            children = self._root_proc.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
        active_pids = {self._root_proc.pid}
        for child in children:
            active_pids.add(child.pid)
            cached = self._tracked_procs.get(child.pid)
            if cached is None:
                try:
                    child.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                self._tracked_procs[child.pid] = child
                procs.append(child)
            else:
                procs.append(cached)
        for stale in set(self._tracked_procs) - active_pids:
            self._tracked_procs.pop(stale, None)
        return procs

    def _prime_cpu_counters(self) -> None:
        for proc in self._refresh_tree():
            try:
                proc.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _read_tree_rss_mb(self) -> float:
        total_bytes = 0
        for proc in self._refresh_tree():
            try:
                total_bytes += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return float(total_bytes / _MIB)

    def _sample_once(self) -> None:
        with self._lock:
            try:
                procs = self._refresh_tree()
                pids = {p.pid for p in procs}
                total_rss_bytes = 0
                total_cpu_pct = 0.0
                for proc in procs:
                    try:
                        total_rss_bytes += proc.memory_info().rss
                        total_cpu_pct += proc.cpu_percent(None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                ram_mb = float(total_rss_bytes / _MIB)
                cpu_pct = min(100.0, float(total_cpu_pct / self._logical_cpus))
                gpu_mb = self._gpu_sampler.read_mb(pids)
                elapsed = max(0.0, time.monotonic() - self._start_mono)
            except Exception as exc:
                logger.warning("resource sample failed: %s", exc)
                return

            self.samples.seconds.append(round(elapsed, 6))
            self.samples.cpu_percent.append(round(cpu_pct, 3))
            self.samples.ram_mb.append(round(ram_mb, 3))
            self.samples.gpu_mb.append(round(gpu_mb, 3))

    def _loop(self) -> None:
        tick = 0
        while True:
            tick += 1
            delay = max(0.0, self._start_mono + tick * self.sample_interval - time.monotonic())
            if self._stop.wait(delay):
                break
            self._sample_once()

    def finalize(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Stop monitoring, compute summary metrics, close NVML, and return full record."""
        self.stop()
        torch_stats = collect_torch_cuda_stats(self.device)
        ru_maxrss_mb = get_process_max_rss_mb()
        metrics = summarize_samples(
            self.samples,
            self.timings,
            device=self.device,
            ram_baseline_mb=self.ram_baseline_mb,
            ru_maxrss_mb=ru_maxrss_mb,
            inference_spans=self.phase_spans.get("inference"),
            torch_cuda_stats=torch_stats,
            gpu_tolerance_mb=self.gpu_tolerance_mb,
        )
        host = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_mb": round(psutil.virtual_memory().total / _MIB, 1),
            "gpu_monitor_backend": self._gpu_sampler.backend,
            "gpu_name": self._gpu_sampler.gpu_name,
        }
        self._gpu_sampler.close()
        record: dict[str, Any] = {
            "device": self.device,
            "gpu_index": self.gpu_index,
            "gpu_memory_tolerance_mb": self.gpu_tolerance_mb,
            "host": host,
            "torch_cuda": torch_stats,
            "ru_maxrss_mb": round(ru_maxrss_mb, 3),
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
            "phase_spans": {
                k: [[round(a, 6), round(b, 6)] for a, b in spans]
                for k, spans in self.phase_spans.items()
            },
            "per_lesion_inference_s": dict(self.per_lesion_inference_s),
            "metrics": metrics,
            "samples": {
                "interval": self.samples.interval,
                "seconds": list(self.samples.seconds),
                "cpu_percent": list(self.samples.cpu_percent),
                "ram_mb": list(self.samples.ram_mb),
                "gpu_mb": list(self.samples.gpu_mb),
            },
        }
        if extra_metadata:
            record.update(extra_metadata)
        self._finalized_record = record
        return record

    def __enter__(self) -> "CaseProfiler":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
        self._gpu_sampler.close()
