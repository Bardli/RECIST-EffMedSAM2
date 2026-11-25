"""Embedded time-breakdown, RAM, CPU, PID-scoped GPU monitor, and trace plotting."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
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

_PHASE_COLORS = {
    "model_loading_time": ("#f59e0b", "model_loading_time"),
    "data_loading_time": ("#6b7280", "data_loading_time"),
    "image_preprocessing_time": ("#8b5cf6", "image_preprocessing_time"),
    "model_inference_time": ("#10b981", "model_inference_time"),
    "postprocessing_time": ("#ec4899", "postprocessing_time"),
    "output_saving_time": ("#3b82f6", "output_saving_time"),
}


def apply_thread_env(num_threads: int | None) -> None:
    """Limit OpenMP/BLAS thread pools before importing PyTorch or OpenCV."""
    if num_threads is None:
        return
    val = str(num_threads)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = val


def configure_device_env(device: str, gpu_index: int | None = None) -> None:
    """Enforce CPU isolation or explicit physical GPU visibility before importing PyTorch."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    elif device.startswith("cuda") and gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically using a temporary sibling file and ``os.replace``."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, dest)


def get_process_max_rss_mb() -> float:
    """Return kernel high-water-mark RSS for the current process in MiB."""
    divisor = _MIB if sys.platform == "darwin" else 1024.0
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor)


def ranking_gpu_memory(
    max_gpu_mb: float,
    tolerance_mb: float = DEFAULT_GPU_TOLERANCE_MB,
) -> float:
    """GPU ranking penalty score: ``max(0, peak - tolerance)`` rounded."""
    return float(round(max(0.0, float(max_gpu_mb) - float(tolerance_mb))))


def _auc(seconds: list[float], values: list[float]) -> float:
    if len(values) < 2 or len(seconds) < 2:
        return 0.0
    n = min(len(seconds), len(values))
    return float(_trapezoid(values[:n], seconds[:n]))


def resolve_physical_gpu_targets(
    device: str,
    physical_gpu_index: int | None = None,
    cuda_visible_devices: str | None = None,
) -> list[int | str] | None:
    """Resolve physical GPU index/UUID list for NVML/nvidia-smi monitoring."""
    if device == "cpu":
        return None
    if physical_gpu_index is not None:
        return [int(physical_gpu_index)]

    if device == "cuda":
        logical_idx = 0
    elif device.startswith("cuda:"):
        logical_idx = int(device.split(":", 1)[1])
    else:
        logical_idx = 0

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") if cuda_visible_devices is None else cuda_visible_devices
    if cvd is None:
        return [logical_idx]

    tokens = [tok.strip() for tok in cvd.split(",") if tok.strip()]
    if not tokens or logical_idx < 0 or logical_idx >= len(tokens):
        return []
    tok = tokens[logical_idx]
    if tok.isdigit():
        return [int(tok)]
    if tok == "-1" or tok.lower() == "void":
        return []
    return [tok]


class PidGpuSampler:
    """Sample GPU memory (MiB) used specifically by target PID(s); returns ``None`` if unmeasured."""

    def __init__(
        self,
        gpu_targets: list[int | str] | None = None,
        *,
        gpu_index: int | None = None,
    ) -> None:
        if gpu_targets is None and gpu_index is not None:
            gpu_targets = [int(gpu_index)]
        self.gpu_targets = gpu_targets
        self.backend = "none"
        self.gpu_name: str | None = None
        self.monitored_physical_gpus: list[int | str] = []
        self.queries_attempted = 0
        self.queries_succeeded = 0
        self.queries_failed = 0
        self.last_error: str | None = None

        self._pynvml = None
        self._handles: list[Any] = []
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._target_resolution_failed = gpu_targets == []
        if self._target_resolution_failed:
            self.last_error = "No visible physical GPU mapped for CUDA device"

        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0 and not self._target_resolution_failed:
                targets: list[int | str] = list(range(count)) if gpu_targets is None else list(gpu_targets)
                handles: list[Any] = []
                resolved: list[int | str] = []
                for target in targets:
                    if isinstance(target, int):
                        if not 0 <= target < count:
                            raise ValueError(f"Physical GPU index {target} out of range (count={count})")
                        handles.append(pynvml.nvmlDeviceGetHandleByIndex(target))
                        resolved.append(target)
                    else:
                        handles.append(pynvml.nvmlDeviceGetHandleByUUID(str(target)))
                        resolved.append(str(target))
                if handles:
                    raw_name = pynvml.nvmlDeviceGetName(handles[0])
                    self.gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                self._handles = handles
                self.monitored_physical_gpus = resolved
                self._pynvml = pynvml
                self.backend = "pynvml"
            else:
                pynvml.nvmlShutdown()
                if self._target_resolution_failed:
                    self.backend = "pynvml"
        except Exception as exc:
            logger.debug("pynvml init failed (%s); falling back to nvidia-smi/none", exc)
            self.last_error = repr(exc)
            self._pynvml = None
            self._handles = []
            if self._nvidia_smi:
                self.backend = "nvidia-smi"
                if gpu_targets:
                    self.monitored_physical_gpus = list(gpu_targets)
            elif self._target_resolution_failed:
                self.backend = "none"

    def read_mb(self, pids: set[int]) -> float | None:
        if not pids:
            return None
        self.queries_attempted += 1
        if self._target_resolution_failed:
            self.queries_failed += 1
            return None
        if self.backend == "none":
            return None

        if self.backend == "pynvml":
            if self._pynvml is None or not self._handles:
                self.queries_failed += 1
                return None
            total_bytes = 0
            seen_per_gpu_pid: set[tuple[int, int]] = set()
            all_handles_ok = True
            for gpu_idx, handle in enumerate(self._handles):
                handle_ok = False
                for fn_name in (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    fn = getattr(self._pynvml, fn_name, None)
                    if fn is None:
                        continue
                    try:
                        procs = fn(handle)
                        handle_ok = True
                    except Exception as exc:
                        self.last_error = repr(exc)
                        continue
                    for proc in procs:
                        pid = int(getattr(proc, "pid", -1))
                        if pid in pids and (gpu_idx, pid) not in seen_per_gpu_pid:
                            seen_per_gpu_pid.add((gpu_idx, pid))
                            used = getattr(proc, "usedGpuMemory", 0) or 0
                            total_bytes += int(used)
                if not handle_ok:
                    all_handles_ok = False
            if not all_handles_ok:
                self.queries_failed += 1
                return None
            self.queries_succeeded += 1
            return float(total_bytes / _MIB)

        if self.backend == "nvidia-smi" and self._nvidia_smi:
            cmd = [
                self._nvidia_smi,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
            if self.gpu_targets:
                cmd += ["-i", ",".join(str(t) for t in self.gpu_targets)]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=False)
                if res.returncode != 0:
                    self.queries_failed += 1
                    self.last_error = res.stderr.strip() or f"nvidia-smi rc={res.returncode}"
                    return None
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
                self.queries_succeeded += 1
                return float(total_mb)
            except Exception as exc:
                self.queries_failed += 1
                self.last_error = repr(exc)
                return None

        return None

    @property
    def status(self) -> str:
        if self.backend == "none":
            return "unavailable"
        if self.queries_succeeded > 0 and self.queries_failed == 0:
            return "measured"
        if self.queries_succeeded > 0 and self.queries_failed > 0:
            return "partial"
        if self.queries_failed > 0:
            return "query_failed"
        return "unavailable"

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
    gpu_mb: list[float | None] = field(default_factory=list)

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
    gpu_monitor_status: str | None = None,
    gpu_monitor_backend: str = "unknown",
    gpu_queries_failed: int = 0,
) -> dict[str, Any]:
    """Compute scalar efficiency metrics with self-explanatory timing phase names."""
    model_load_s = float(timings.get("model_loading_time_s", 0.0))
    inference_s = float(timings.get("model_inference_time_s", 0.0))
    postprocess_s = float(timings.get("postprocessing_time_s", 0.0))
    total_inference_time_s = model_load_s + inference_s + postprocess_s

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
    cuda_initialized = bool(cuda_stats.get("cuda_initialized", False))

    valid_gpu = [(t, float(g)) for t, g in zip(samples.seconds, samples.gpu_mb) if g is not None]
    gpu_valid_samples = len(valid_gpu)
    gpu_failed_samples = int(gpu_queries_failed)

    if gpu_monitor_status is None:
        if gpu_valid_samples > 0 and gpu_failed_samples == 0:
            gpu_monitor_status = "measured"
        elif gpu_valid_samples > 0:
            gpu_monitor_status = "partial"
        elif gpu_failed_samples > 0:
            gpu_monitor_status = "query_failed"
        else:
            gpu_monitor_status = "unavailable"

    if valid_gpu:
        nvml_peak_mb: float | None = round(max(g for _, g in valid_gpu), 3)
        auc_gpu: float | None = round(_auc([t for t, _ in valid_gpu], [g for _, g in valid_gpu]), 3)
    else:
        nvml_peak_mb = None
        auc_gpu = None

    is_cpu = device == "cpu"
    if nvml_peak_mb is not None:
        max_gpu_mb: float | None = round(max(nvml_peak_mb, torch_reserved_mb), 3)
    elif torch_reserved_mb > 0.0 or (not is_cpu and cuda_initialized):
        max_gpu_mb = round(torch_reserved_mb, 3)
    else:
        max_gpu_mb = None

    rank_gpu_mb: float | None = (
        ranking_gpu_memory(max_gpu_mb, gpu_tolerance_mb) if max_gpu_mb is not None else None
    )

    if is_cpu:
        if cuda_available or torch_alloc_mb > 0.0 or torch_reserved_mb > 0.0:
            gpu_sanity_ok: bool | None = False
        elif gpu_monitor_status == "measured":
            gpu_sanity_ok = bool(nvml_peak_mb == 0.0)
        elif gpu_monitor_status in ("partial", "query_failed"):
            gpu_sanity_ok = False
        else:
            gpu_sanity_ok = None
    else:
        gpu_sanity_ok = bool(cuda_available and gpu_monitor_status in ("measured", "partial"))

    return {
        "total_inference_time_s": round(total_inference_time_s, 6),
        "model_loading_time_s": round(model_load_s, 6),
        "model_inference_time_s": round(inference_s, 6),
        "postprocessing_time_s": round(postprocess_s, 6),
        "data_loading_time_s": round(float(timings.get("data_loading_time_s", 0.0)), 6),
        "image_preprocessing_time_s": round(float(timings.get("image_preprocessing_time_s", 0.0)), 6),
        "output_saving_time_s": round(float(timings.get("output_saving_time_s", 0.0)), 6),
        "end_to_end_wall_time_including_io_s": round(
            float(timings.get("end_to_end_wall_time_including_io_s", 0.0)), 6
        ),
        "MaxRAM_MB": round(max_ram_mb, 3),
        "ram_baseline_mb": round(float(ram_baseline_mb), 3),
        "ram_inference_peak_mb": round(float(ram_inference_peak_mb), 3),
        "ram_delta_mb": round(ram_delta_mb, 3),
        "AUC_RAM_Time_MB_s": round(auc_ram, 3),
        "MaxCPU_Utilization_pct": round(max_cpu_pct, 3),
        "AUC_CPU_Time_pct_s": round(auc_cpu, 3),
        "MaxGPU_Mem_MB": max_gpu_mb,
        "gpu_nvml_peak_mb": nvml_peak_mb,
        "AUC_GPU_Time_MB_s": auc_gpu,
        "RankGPU_Mem_MB": rank_gpu_mb,
        "gpu_torch_peak_allocated_mb": round(torch_alloc_mb, 3),
        "gpu_torch_peak_reserved_mb": round(torch_reserved_mb, 3),
        "gpu_monitor_status": gpu_monitor_status,
        "gpu_sanity_ok": gpu_sanity_ok,
    }


class CaseProfiler:
    """Per-case phase timer and process-tree RAM/CPU/GPU monitor."""

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
        self.target_pid = os.getpid()
        self.device = device
        self.gpu_index = gpu_index
        self.physical_gpu_targets = resolve_physical_gpu_targets(device, gpu_index)
        self.sample_interval = float(sample_interval)
        self.gpu_tolerance_mb = float(gpu_tolerance_mb)

        self.samples = Samples(interval=self.sample_interval)
        self.timings: dict[str, float] = {}
        self.phase_spans: dict[str, list[tuple[float, float]]] = {}
        self.per_lesion_inference_time_s: dict[str, float] = {}
        self.per_lesion_postprocessing_time_s: dict[str, float] = {}

        try:
            self._root_proc: psutil.Process | None = psutil.Process(self.target_pid)
        except psutil.NoSuchProcess:
            self._root_proc = None
        self._tracked_procs: dict[int, psutil.Process] = (
            {self._root_proc.pid: self._root_proc} if self._root_proc is not None else {}
        )
        self._logical_cpus = max(1, psutil.cpu_count(logical=True) or 1)
        self._gpu_sampler = PidGpuSampler(gpu_targets=self.physical_gpu_targets)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._start_mono: float = 0.0
        self._start_perf: float = 0.0
        self.ram_baseline_mb: float = 0.0

    def start(self) -> "CaseProfiler":
        if self._thread is not None:
            raise RuntimeError("Monitor already started")
        self._start_mono = time.monotonic()
        self._start_perf = time.perf_counter()
        self.ram_baseline_mb = self._read_tree_rss_mb()
        self._prime_cpu_counters()
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
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
        if "end_to_end_wall_time_including_io_s" not in self.timings:
            self.timings["end_to_end_wall_time_including_io_s"] = time.perf_counter() - self._start_perf
        self._sample_once()
        return self.samples

    def _refresh_tree(self) -> list[psutil.Process]:
        if self._root_proc is None or not self._root_proc.is_running():
            return []
        try:
            if self._root_proc.status() == psutil.STATUS_ZOMBIE:
                return []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

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
                if not procs:
                    return
                pids = {p.pid for p in procs}
                total_rss_bytes = 0
                total_cpu_pct = 0.0
                alive = False
                for proc in procs:
                    try:
                        total_rss_bytes += proc.memory_info().rss
                        total_cpu_pct += proc.cpu_percent(None)
                        alive = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if not alive:
                    return
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
            self.samples.gpu_mb.append(round(gpu_mb, 3) if gpu_mb is not None else None)

    def _loop(self) -> None:
        tick = 0
        while True:
            tick += 1
            delay = max(0.0, self._start_mono + tick * self.sample_interval - time.monotonic())
            if self._stop.wait(delay):
                break
            self._sample_once()

    def _sync_cuda_if_needed(self) -> None:
        if not self.device.startswith("cuda"):
            return
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and torch_mod.cuda.is_available() and torch_mod.cuda.is_initialized():
            torch_mod.cuda.synchronize(torch_mod.device(self.device))

    @contextmanager
    def phase(self, name: str):
        """Time a named phase (CUDA-synchronized when running on GPU)."""
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
        self.per_lesion_inference_time_s[str(int(label_id))] = round(float(elapsed_s), 6)

    def record_lesion_postprocess_time(self, label_id: int, elapsed_s: float) -> None:
        self.per_lesion_postprocessing_time_s[str(int(label_id))] = round(float(elapsed_s), 6)

    def finalize(
        self,
        extra_metadata: dict[str, Any] | None = None,
        *,
        use_ru_maxrss: bool = True,
    ) -> dict[str, Any]:
        """Stop monitoring, compute metrics, close NVML, and return full record."""
        self.stop()
        torch_stats = collect_torch_cuda_stats(self.device)
        ru_maxrss_mb = get_process_max_rss_mb() if use_ru_maxrss else 0.0
        metrics = summarize_samples(
            self.samples,
            self.timings,
            device=self.device,
            ram_baseline_mb=self.ram_baseline_mb,
            ru_maxrss_mb=ru_maxrss_mb,
            inference_spans=self.phase_spans.get("model_inference_time"),
            torch_cuda_stats=torch_stats,
            gpu_tolerance_mb=self.gpu_tolerance_mb,
            gpu_monitor_status=self._gpu_sampler.status,
            gpu_monitor_backend=self._gpu_sampler.backend,
            gpu_queries_failed=self._gpu_sampler.queries_failed,
        )
        host = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_mb": round(psutil.virtual_memory().total / _MIB, 1),
            "gpu_monitor_backend": self._gpu_sampler.backend,
            "gpu_monitor_status": self._gpu_sampler.status,
            "gpu_monitored_physical_gpus": list(self._gpu_sampler.monitored_physical_gpus),
            "gpu_name": self._gpu_sampler.gpu_name,
        }
        self._gpu_sampler.close()
        record: dict[str, Any] = {
            "device": self.device,
            "physical_gpu_index": (
                self.physical_gpu_targets[0]
                if self.physical_gpu_targets and len(self.physical_gpu_targets) == 1
                else self.gpu_index
            ),
            "gpu_memory_tolerance_mb": self.gpu_tolerance_mb,
            "host": host,
            "torch_cuda": torch_stats,
            "ru_maxrss_mb": round(ru_maxrss_mb, 3),
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
            "phase_spans": {
                k: [[round(a, 6), round(b, 6)] for a, b in spans]
                for k, spans in self.phase_spans.items()
            },
            "per_lesion_inference_time_s": dict(self.per_lesion_inference_time_s),
            "per_lesion_postprocessing_time_s": dict(self.per_lesion_postprocessing_time_s),
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
        return record

    def __enter__(self) -> "CaseProfiler":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
        self._gpu_sampler.close()


def plot_case(record: dict[str, Any], plots_dir: Path) -> list[Path]:
    """Save CPU / RAM / GPU-vs-time curve PNGs with shaded execution phases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = record.get("samples", {})
    seconds = samples.get("seconds", [])
    if not seconds:
        return []

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    case_id = record.get("case_id", "case")
    model = record.get("model_name") or record.get("model", "")
    phase_spans = record.get("phase_spans", {})

    series = [
        ("cpu_percent", "CPU Utilization (%)", f"{case_id}_CPU-Time.png"),
        ("ram_mb", "Process RAM RSS (MiB)", f"{case_id}_RAM-Time.png"),
        ("gpu_mb", "Process GPU Memory (MiB)", f"{case_id}_GPU-Time.png"),
    ]
    written: list[Path] = []
    for key, ylabel, filename in series:
        values = samples.get(key) or []
        pairs = [(t, v) for t, v in zip(seconds, values) if v is not None]
        if not pairs:
            continue
        xs = [t for t, _ in pairs]
        ys = [float(v) for _, v in pairs]

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        ax.plot(xs, ys, color="#1d4ed8", linewidth=1.6)
        labeled_phases: set[str] = set()
        for phase, spans in phase_spans.items():
            color, label = _PHASE_COLORS.get(phase, ("#9ca3af", phase))
            for t0, t1 in spans:
                lbl = label if phase not in labeled_phases else None
                labeled_phases.add(phase)
                ax.axvspan(t0, t1, color=color, alpha=0.12, label=lbl)
        tolerance = record.get("gpu_memory_tolerance_mb")
        max_gpu = record.get("metrics", {}).get("MaxGPU_Mem_MB")
        if key == "gpu_mb" and tolerance and max_gpu is not None and float(max_gpu) > 0:
            ax.axhline(
                float(tolerance),
                color="#dc2626",
                linestyle="--",
                linewidth=1.0,
                label=f"tolerance {float(tolerance):.0f} MiB",
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        title_prefix = f"{model} - " if model else ""
        ax.set_title(f"{title_prefix}{case_id} ({record.get('device', '')})", fontsize=9)
        ax.grid(alpha=0.3)
        if labeled_phases or (key == "gpu_mb" and tolerance and max_gpu is not None and float(max_gpu) > 0):
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        out_path = plots_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written
