"""Per-case accelerator accounting; keep CPU workers CUDA-free."""


def reset_cuda_peaks(device) -> None:
    """Initialize CUDA before resetting allocator counters."""
    if device.type == "cuda":
        import torch

        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
