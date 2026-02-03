#!/usr/bin/env python3
"""
Entry point for running parallelism_planner as a module.

Usage:
    python -m core.optimization.parallelism_planner <command> [options]
    
Commands:
    recommend   - Get parallelism recommendations (TP/PP/DP/CP/EP)
    sharding    - Get ZeRO/FSDP/HSDP sharding recommendations
    launch      - Generate framework launch commands
    pareto      - Cost/throughput Pareto analysis
    calibrate   - Calibrate from benchmark data
    presets     - List available model presets
    topology    - Detect hardware topology
    analyze     - Analyze model architecture
    
Examples:
    # Parallelism recommendations
    python -m core.optimization.parallelism_planner recommend llama-3.1-70b --training
    
    # Sharding recommendations
    python -m core.optimization.parallelism_planner sharding llama-3.1-70b --dp 4 --memory 192
    
    # Generate launch commands for multi-node training
    python -m core.optimization.parallelism_planner launch --nodes 2 --gpus 4 --tp 2 --pp 2 --dp 1 --sharding zero3
    
    # Cost analysis
    python -m core.optimization.parallelism_planner pareto llama-3.1-70b --gpu-cost 4.0
    
    # List model presets
    python -m core.optimization.parallelism_planner presets
"""

from .cli import main

if __name__ == "__main__":
    main()

