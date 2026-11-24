"""Per-case accelerator accounting; keep CPU workers CUDA-free."""


def reset_cuda_peaks(device) -> None:
    """Initialize CUDA in a fresh worker before resetting allocator counters."""
    if device.type == "cuda":
        import torch

        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
