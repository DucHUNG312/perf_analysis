"""
CLI entry point for auto-optimizer.

Usage:
    python -m core.optimization.auto code.py --output optimized.py
    python -m core.optimization.auto https://github.com/user/repo --target model.py
    python -m core.optimization.auto --scan . --threshold 1.1
"""

from .optimizer import main

if __name__ == "__main__":
    main()



