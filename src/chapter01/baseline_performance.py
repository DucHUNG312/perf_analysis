from __future__ import annotations


import torch
import chapter01.arch_config

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available:
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        print(
            f"WARNING: CUDA unavailable or unsupported ({exc}); falling back to CPU.")
        return torch.device("cpu")


class BaselinePerformanceBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()


def get_benchmark() -> BaseBenchmark:
    return BaselinePerformanceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
