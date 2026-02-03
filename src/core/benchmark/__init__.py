"""Benchmark framework core components.

This package provides the core infrastructure for benchmark verification
and correctness enforcement.

Key Modules:
- verification: Data models (InputSignature, ToleranceSpec, etc.)
- quarantine: QuarantineManager for non-compliant benchmarks  
- verify_runner: VerifyRunner for verification execution
- contract: BenchmarkContract defining required interface
"""

from core.benchmark.verification import (
    ComparisonDetails,
    EnforcementPhase,
    InputSignature,
    PrecisionFlags,
    QuarantineReason,
    QuarantineRecord,
    ToleranceSpec,
    VerifyResult,
    DEFAULT_TOLERANCES,
    get_enforcement_phase,
    get_tolerance_for_dtype,
    is_verification_enabled,
    set_deterministic_seeds,
)

from core.benchmark.quarantine import (
    QuarantineManager,
    detect_skip_flags,
    check_benchmark_compliance,
)

from core.benchmark.verify_runner import (
    GoldenOutput,
    GoldenOutputCache,
    VerifyConfig,
    VerifyRunner,
)

from core.benchmark.contract import (
    BenchmarkContract,
    check_benchmark_file,
    check_benchmark_file_ast,
)

__all__ = [
    # Verification data models
    "ComparisonDetails",
    "EnforcementPhase",
    "InputSignature",
    "PrecisionFlags",
    "QuarantineReason",
    "QuarantineRecord",
    "ToleranceSpec",
    "VerifyResult",
    "DEFAULT_TOLERANCES",
    "get_enforcement_phase",
    "get_tolerance_for_dtype",
    "is_verification_enabled",
    "set_deterministic_seeds",
    # Quarantine management
    "QuarantineManager",
    "detect_skip_flags",
    "check_benchmark_compliance",
    # Verification runner
    "GoldenOutput",
    "GoldenOutputCache",
    "VerifyConfig",
    "VerifyRunner",
    # Contract
    "BenchmarkContract",
    "check_benchmark_file",
    "check_benchmark_file_ast",
]

