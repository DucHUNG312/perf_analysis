"""
Auto-Optimizer Module

A general-purpose GPU code optimizer that works on:
- Standalone Python files with GPU code
- External repos (clone → analyze → optimize)
- Existing benchmark pairs (baseline_ / optimized_)
- Code from stdin/clipboard

Usage:
    python -m core.optimization.auto code.py --output optimized_code.py
    python -m core.optimization.auto https://github.com/user/repo --target src/model.py
    python -m core.optimization.auto --scan --threshold 1.1
"""

from .optimizer import AutoOptimizer
from .input_adapters import FileAdapter, RepoAdapter, BenchmarkAdapter
from .config import (
    OptimizerConfig,
    LLMConfig,
    OptimizationConfig,
    ProfilingConfig,
    load_config,
    generate_config_template,
)

__all__ = [
    'AutoOptimizer',
    'FileAdapter',
    'RepoAdapter',
    'BenchmarkAdapter',
    'OptimizerConfig',
    'LLMConfig',
    'OptimizationConfig',
    'ProfilingConfig',
    'load_config',
    'generate_config_template',
]

