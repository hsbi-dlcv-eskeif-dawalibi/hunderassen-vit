"""Model cost: parameters, FLOPs and latency (lecture 02: "Komplexität reduzieren").

FLOPs are counted with ``torch.utils.flop_counter`` (PyTorch >= 2.1, no extra
dependency); one multiply-add counts as 2 FLOPs. Latency is the median of timed forward
passes after warm-up.
"""

from __future__ import annotations

import time


def count_flops(model, size: int = 224) -> float:
    """GFLOPs for a single 3 x size x size image."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    device = next(model.parameters()).device
    x = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(pixel_values=x)
    return counter.get_total_flops() / 1e9


def measure_latency(model, size: int = 224, batch_size: int = 1, warmup: int = 10, runs: int = 50) -> float:
    """Median milliseconds per forward pass of ``batch_size`` images."""
    import statistics

    import torch

    device = next(model.parameters()).device
    x = torch.randn(batch_size, 3, size, size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            sync()
            start = time.perf_counter()
            model(pixel_values=x)
            sync()
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def profile(model, size: int = 224) -> dict[str, float]:
    """Cost of a model: parameters (millions), GFLOPs per image and median latency (ms)."""
    total = sum(p.numel() for p in model.parameters())
    return {
        "parameters_m": total / 1e6,
        "gflops": count_flops(model, size),
        "latency_ms_bs1": measure_latency(model, size, batch_size=1),
        "latency_ms_bs32": measure_latency(model, size, batch_size=32, warmup=3, runs=10),
    }
