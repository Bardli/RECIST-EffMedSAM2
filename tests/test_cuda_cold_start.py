"""Real CUDA test: run in a fresh interpreter, never after a fixture warms CUDA."""
import os
import subprocess
import sys

import pytest


def test_real_cuda_cold_start():
    script = """
import sys
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    sys.exit(77)
from recist_effmedsam2.backends.device_memory import reset_cuda_peaks
assert not torch.cuda.is_initialized(), "worker already warmed CUDA"
reset_cuda_peaks(torch.device("cpu"))
assert not torch.cuda.is_initialized(), "CPU path initialized CUDA"
if not torch.cuda.is_available():
    sys.exit(77)
assert not torch.cuda.is_initialized(), "device discovery warmed PyTorch CUDA"
dev = torch.device("cuda:0")
reset_cuda_peaks(dev)
assert torch.cuda.is_initialized()
baseline = torch.cuda.memory_allocated(dev)
tensor = torch.ones(1024 * 1024, device=dev, dtype=torch.float32)
torch.cuda.synchronize(dev)
assert torch.cuda.max_memory_allocated(dev) >= baseline + tensor.numel() * tensor.element_size()
print("cold-start CUDA validated:", torch.__version__, torch.cuda.get_device_name(dev))
"""
    result = subprocess.run([sys.executable, "-c", script], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode == 77 and os.getenv("RECIST_REQUIRE_CUDA") != "1":
        pytest.skip("requires a CUDA-capable PyTorch environment and an allocated NVIDIA GPU")
    assert result.returncode == 0, f"CUDA cold-start test failed (rc={result.returncode}): {result.stdout}"
    print(result.stdout.strip())
