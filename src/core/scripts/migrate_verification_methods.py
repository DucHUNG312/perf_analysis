#!/usr/bin/env python3
"""Auto-migration script to add verification methods to benchmarks.

This script adds the mandatory verification methods to benchmark files:
- get_input_signature(): Describes workload for equivalence checking
- validate_result(): Validates benchmark output
- get_workload_metadata(): Provides workload metrics for fair comparison

Usage:
    # Dry run on Phase 1 (ch01-ch06)
    python -m core.scripts.migrate_verification_methods --phase 1 --dry-run
    
    # Migrate Phase 1 with backup
    python -m core.scripts.migrate_verification_methods --phase 1 --backup
    
    # Migrate specific chapter
    python -m core.scripts.migrate_verification_methods --chapter ch07
    
    # Generate report
    python -m core.scripts.migrate_verification_methods --phase 1 --report artifacts/migration_report.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Configuration
# =============================================================================

# Phases for prioritized migration
PHASES = {
    1: ["ch01", "ch02", "ch03", "ch04", "ch05", "ch06"],
    2: ["ch07", "ch08", "ch09", "ch10", "ch11", "ch12", "ch13", "ch14"],
    3: ["ch15", "ch16", "ch17", "ch18", "ch19", "ch20"],
    4: ["labs"],  # Will match labs/*
}

# Common workload attributes to detect
WORKLOAD_ATTRIBUTES = {
    # Matrix/tensor dimensions
    "m", "n", "k", "M", "N", "K",
    # Batch/sequence dimensions  
    "batch_size", "seq_len", "seq_length", "sequence_length",
    # Model dimensions
    "hidden_dim", "hidden_size", "embed_dim", "d_model",
    "num_heads", "n_heads", "head_dim",
    "vocab_size", "num_layers", "n_layers",
    # Other common parameters
    "N", "num_chunks", "num_elements", "size",
    "num_tokens", "max_seq_len", "context_length",
}

# Output attributes to detect for validate_result
OUTPUT_ATTRIBUTES = {
    "output", "out", "C", "result", "results", 
    "logits", "hidden_states", "attention_output",
}

# Known dtypes
DTYPE_MAPPING = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
    "torch.int32": "int32",
    "torch.int64": "int64",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkAnalysis:
    """Analysis results for a benchmark file."""
    file_path: str
    benchmark_class: Optional[str] = None
    has_get_input_signature: bool = False
    has_validate_result: bool = False
    has_get_workload_metadata: bool = False
    has_register_workload_metadata: bool = False
    has_get_verify_output: bool = False
    
    # Detected attributes
    workload_attrs: Dict[str, Any] = field(default_factory=dict)
    output_attrs: Set[str] = field(default_factory=set)
    detected_dtypes: Set[str] = field(default_factory=set)
    detected_shapes: Dict[str, Tuple] = field(default_factory=dict)
    
    # Parent class info
    parent_class: Optional[str] = None
    is_cuda_binary: bool = False
    
    errors: List[str] = field(default_factory=list)


@dataclass
class MigrationResult:
    """Result of migrating a single file."""
    file_path: str
    success: bool
    methods_added: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None
    error: Optional[str] = None
    needs_manual_review: bool = False
    review_reason: Optional[str] = None


@dataclass
class MigrationReport:
    """Complete migration report."""
    timestamp: str
    phase: Optional[int]
    chapter: Optional[str]
    dry_run: bool
    total_files: int = 0
    files_modified: int = 0
    files_skipped: int = 0
    files_errored: int = 0
    manual_review_needed: int = 0
    results: List[MigrationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "phase": self.phase,
            "chapter": self.chapter,
            "dry_run": self.dry_run,
            "summary": {
                "total_files": self.total_files,
                "files_modified": self.files_modified,
                "files_skipped": self.files_skipped,
                "files_errored": self.files_errored,
                "manual_review_needed": self.manual_review_needed,
            },
            "results": [
                {
                    "file_path": r.file_path,
                    "success": r.success,
                    "methods_added": r.methods_added,
                    "backup_path": r.backup_path,
                    "error": r.error,
                    "needs_manual_review": r.needs_manual_review,
                    "review_reason": r.review_reason,
                }
                for r in self.results
            ],
        }


# =============================================================================
# AST Analysis
# =============================================================================

class BenchmarkAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze benchmark files."""
    
    def __init__(self):
        self.benchmark_class: Optional[str] = None
        self.parent_class: Optional[str] = None
        self.methods: Set[str] = set()
        self.workload_attrs: Dict[str, Any] = {}
        self.output_attrs: Set[str] = set()
        self.detected_dtypes: Set[str] = set()
        self.detected_shapes: Dict[str, Tuple] = {}
        self.has_register_workload_metadata: bool = False
        self._in_benchmark_class = False
        self._in_init = False
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to find benchmark classes."""
        # Check if this looks like a benchmark class
        has_benchmark_fn = any(
            isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
            for item in node.body
        )
        
        if has_benchmark_fn:
            self.benchmark_class = node.name
            
            # Get parent class
            if node.bases:
                base = node.bases[0]
                if isinstance(base, ast.Name):
                    self.parent_class = base.id
                elif isinstance(base, ast.Attribute):
                    self.parent_class = base.attr
            
            self._in_benchmark_class = True
            self.generic_visit(node)
            self._in_benchmark_class = False
        else:
            self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to detect methods."""
        if self._in_benchmark_class:
            self.methods.add(node.name)
            
            if node.name == "__init__":
                self._in_init = True
                self.generic_visit(node)
                self._in_init = False
            elif node.name == "setup":
                # Also check setup for attribute assignments
                self._in_init = True
                self.generic_visit(node)
                self._in_init = False
            else:
                self.generic_visit(node)
        else:
            self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to detect attributes."""
        if not self._in_benchmark_class:
            self.generic_visit(node)
            return
        
        for target in node.targets:
            # Check for self.attr = value patterns
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    attr_name = target.attr
                    
                    # Check if it's a workload attribute
                    if attr_name in WORKLOAD_ATTRIBUTES:
                        value = self._extract_value(node.value)
                        if value is not None:
                            self.workload_attrs[attr_name] = value
                    
                    # Check if it's an output attribute
                    if attr_name in OUTPUT_ATTRIBUTES:
                        self.output_attrs.add(attr_name)
                    
                    # Try to detect dtype from torch.randn/empty calls
                    self._detect_tensor_info(attr_name, node.value)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to detect register_workload_metadata."""
        if self._in_benchmark_class:
            # Check for self.register_workload_metadata(...)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "register_workload_metadata":
                    self.has_register_workload_metadata = True
        
        self.generic_visit(node)
    
    def _extract_value(self, node: ast.expr) -> Any:
        """Extract a constant value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Name):
            # Could be a constant like True/False/None
            if node.id in ("True", "False", "None"):
                return eval(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers
            val = self._extract_value(node.operand)
            if val is not None:
                return -val
        elif isinstance(node, ast.BinOp):
            # Handle simple expressions like 4096 * 4
            left = self._extract_value(node.left)
            right = self._extract_value(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Add):
                    return left + right
        return None
    
    def _detect_tensor_info(self, attr_name: str, node: ast.expr) -> None:
        """Detect dtype and shape from tensor initialization."""
        if not isinstance(node, ast.Call):
            return
        
        # Get function name
        func_name = None
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        
        if func_name not in ("randn", "empty", "zeros", "ones", "rand"):
            return
        
        # Look for dtype keyword argument
        for keyword in node.keywords:
            if keyword.arg == "dtype":
                dtype_str = ast.unparse(keyword.value) if hasattr(ast, 'unparse') else None
                if dtype_str and dtype_str in DTYPE_MAPPING:
                    self.detected_dtypes.add(DTYPE_MAPPING[dtype_str])
        
        # Try to extract shape from first argument
        if node.args:
            shape = self._extract_shape(node.args[0])
            if shape:
                self.detected_shapes[attr_name] = shape
    
    def _extract_shape(self, node: ast.expr) -> Optional[Tuple]:
        """Extract shape tuple from AST node."""
        if isinstance(node, ast.Tuple):
            shape = []
            for elt in node.elts:
                val = self._extract_value(elt)
                if val is not None:
                    shape.append(val)
                elif isinstance(elt, ast.Attribute):
                    # self.batch_size -> "batch_size"
                    if isinstance(elt.value, ast.Name) and elt.value.id == "self":
                        shape.append(f"self.{elt.attr}")
                else:
                    return None
            return tuple(shape) if shape else None
        elif isinstance(node, ast.Constant):
            return (node.value,)
        return None


def analyze_benchmark_file(file_path: Path) -> BenchmarkAnalysis:
    """Analyze a benchmark file for verification methods and attributes."""
    analysis = BenchmarkAnalysis(file_path=str(file_path))
    
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        analysis.errors.append(f"Syntax error: {e}")
        return analysis
    except Exception as e:
        analysis.errors.append(f"Parse error: {e}")
        return analysis
    
    analyzer = BenchmarkAnalyzer()
    analyzer.visit(tree)
    
    analysis.benchmark_class = analyzer.benchmark_class
    analysis.parent_class = analyzer.parent_class
    analysis.has_get_input_signature = "get_input_signature" in analyzer.methods
    analysis.has_validate_result = "validate_result" in analyzer.methods
    analysis.has_get_workload_metadata = "get_workload_metadata" in analyzer.methods
    analysis.has_register_workload_metadata = analyzer.has_register_workload_metadata
    analysis.has_get_verify_output = "get_verify_output" in analyzer.methods
    analysis.workload_attrs = analyzer.workload_attrs
    analysis.output_attrs = analyzer.output_attrs
    analysis.detected_dtypes = analyzer.detected_dtypes
    analysis.detected_shapes = analyzer.detected_shapes
    analysis.is_cuda_binary = analyzer.parent_class == "CudaBinaryBenchmark"
    
    return analysis


# =============================================================================
# Code Generation
# =============================================================================

def generate_get_input_signature(analysis: BenchmarkAnalysis) -> Optional[str]:
    """Generate get_input_signature method code.
    
    STRICT: Always generates explicit implementation with detected parameters.
    """
    if analysis.has_get_input_signature:
        return None
    
    # Build signature dict entries
    entries = []
    
    # Add workload attributes
    for attr, value in sorted(analysis.workload_attrs.items()):
        entries.append(f'            "{attr}": self.{attr},')
    
    # Add detected dtypes
    if analysis.detected_dtypes:
        dtypes = sorted(analysis.detected_dtypes)
        if len(dtypes) == 1:
            entries.append(f'            "dtype": "{dtypes[0]}",')
        else:
            entries.append(f'            "dtypes": {dtypes},')
    
    # If no attributes detected, force manual implementation
    if not entries:
        return '''
    def get_input_signature(self) -> dict:
        """Input signature for verification equivalence.
        
        MANDATORY: This method must be implemented explicitly.
        TODO: Add ALL workload parameters that define this benchmark's inputs.
        Common parameters: batch_size, seq_len, hidden_dim, m, n, k, dtype
        """
        raise NotImplementedError(
            "TODO: Implement get_input_signature() - return dict of workload parameters"
        )
'''
    
    # Generate the method
    entries_str = "\n".join(entries)
    return f'''
    def get_input_signature(self) -> dict:
        """Input signature for verification equivalence.
        
        MANDATORY: This method must be implemented explicitly.
        """
        return {{
{entries_str}
        }}
'''


def generate_validate_result(analysis: BenchmarkAnalysis) -> Optional[str]:
    """Generate validate_result method code.
    
    STRICT: Always generates explicit validation checks.
    """
    if analysis.has_validate_result:
        return None
    
    # Build validation checks
    checks = []
    
    # Add checks for output attributes
    for attr in sorted(analysis.output_attrs):
        checks.append(f'''        if self.{attr} is None:
            return "{attr} is None"
        if torch.isnan(self.{attr}).any():
            return "{attr} contains NaN values"
        if torch.isinf(self.{attr}).any():
            return "{attr} contains Inf values"''')
    
    # If no output attributes detected, force manual implementation
    if not checks:
        return '''
    def validate_result(self) -> Optional[str]:
        """Validate benchmark produced valid output.
        
        MANDATORY: This method must be implemented explicitly.
        TODO: Add validation checks for output tensors (None, NaN, Inf, shape, bounds).
        """
        raise NotImplementedError(
            "TODO: Implement validate_result() - check output for None/NaN/Inf/shape"
        )
'''
    
    # Generate the method with proper checks
    checks_str = "\n".join(checks)
    return f'''
    def validate_result(self) -> Optional[str]:
        """Validate benchmark produced valid output.
        
        MANDATORY: This method must be implemented explicitly.
        """
{checks_str}
        return None
'''


def generate_get_workload_metadata(analysis: BenchmarkAnalysis) -> Optional[str]:
    """Generate get_workload_metadata method code.
    
    STRICT: Always generates explicit workload calculation.
    """
    if analysis.has_get_workload_metadata or analysis.has_register_workload_metadata:
        return None
    
    # Try to compute tokens/ops from detected attributes
    workload_expr = None
    
    # Check for common patterns
    attrs = analysis.workload_attrs
    if "batch_size" in attrs and "seq_len" in attrs:
        workload_expr = "float(self.batch_size * self.seq_len)"
    elif "batch_size" in attrs and "seq_length" in attrs:
        workload_expr = "float(self.batch_size * self.seq_length)"
    elif "m" in attrs and "n" in attrs:
        workload_expr = "float(self.m * self.n)"
    elif "M" in attrs and "N" in attrs:
        workload_expr = "float(self.M * self.N)"
    elif "N" in attrs:
        workload_expr = "float(self.N)"
    elif "num_elements" in attrs:
        workload_expr = "float(self.num_elements)"
    elif "size" in attrs:
        workload_expr = "float(self.size)"
    
    if workload_expr:
        return f'''
    def get_workload_metadata(self) -> WorkloadMetadata:
        """Return workload metadata for fair comparison.
        
        MANDATORY: This method must be implemented explicitly.
        """
        return WorkloadMetadata(
            tokens_per_iteration={workload_expr},
        )
'''
    else:
        return '''
    def get_workload_metadata(self) -> WorkloadMetadata:
        """Return workload metadata for fair comparison.
        
        MANDATORY: This method must be implemented explicitly.
        TODO: Calculate actual tokens/bytes/ops per iteration.
        """
        raise NotImplementedError(
            "TODO: Implement get_workload_metadata() - return WorkloadMetadata with actual workload"
        )
'''


def generate_get_verify_output(analysis: BenchmarkAnalysis) -> Optional[str]:
    """Generate get_verify_output method code.
    
    STRICT: Always generates explicit implementation, never relies on fallbacks.
    """
    if analysis.has_get_verify_output:
        return None
    
    # Priority order for output attributes
    output_priority = ["output", "C", "result", "out", "y", "logits", "hidden_states"]
    
    # Find first detected output attribute
    detected_output = None
    for attr in output_priority:
        if attr in analysis.output_attrs:
            detected_output = attr
            break
    
    if detected_output:
        return f'''
    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison.
        
        MANDATORY: This method must be implemented explicitly.
        """
        return self.{detected_output}
'''
    else:
        # No output detected - MUST still implement, force manual review
        return '''
    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison.
        
        MANDATORY: This method must be implemented explicitly.
        TODO: Return the output tensor, or a checksum tensor for throughput-only benchmarks.
        """
        raise NotImplementedError(
            "TODO: Implement get_verify_output() - return output tensor or checksum"
        )
'''


# =============================================================================
# File Modification
# =============================================================================

def find_class_end(source: str, class_name: str) -> Optional[int]:
    """Find the line number where a class definition ends."""
    lines = source.split("\n")
    in_class = False
    class_indent = 0
    last_line_in_class = 0
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
        if stripped.startswith(f"class {class_name}"):
            in_class = True
            class_indent = current_indent
            last_line_in_class = i
            continue
        
        if in_class:
            # Empty lines are still part of the class
            if not stripped:
                continue
            
            # If we hit a line at same or lower indent (and it's not empty), class ends
            if current_indent <= class_indent and stripped:
                return last_line_in_class
            
            last_line_in_class = i
    
    return last_line_in_class if in_class else None


def add_import_if_needed(source: str, import_stmt: str, from_module: str) -> str:
    """Add an import statement if not already present."""
    if import_stmt in source:
        return source
    
    # Find existing imports from the same module
    pattern = rf"from {re.escape(from_module)} import"
    match = re.search(pattern, source)
    
    if match:
        # Add to existing import
        # Find the end of the import statement
        line_start = source.rfind("\n", 0, match.start()) + 1
        line_end = source.find("\n", match.end())
        if line_end == -1:
            line_end = len(source)
        
        import_line = source[line_start:line_end]
        
        # Check if it's a multi-line import with parentheses
        if "(" in import_line and ")" not in import_line:
            # Multi-line import - find closing paren
            paren_end = source.find(")", line_end)
            if paren_end != -1:
                line_end = source.find("\n", paren_end)
        
        # Add the new import
        if ")" in source[line_start:line_end]:
            # Has closing paren on same line
            new_import_line = source[line_start:line_end].replace(")", f", {import_stmt})")
        else:
            # Simple single-line import
            new_import_line = source[line_start:line_end].rstrip() + f", {import_stmt}"
        
        return source[:line_start] + new_import_line + source[line_end:]
    
    # No existing import from this module - add new import
    # Find a good place to add it (after other imports)
    lines = source.split("\n")
    last_import_line = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_line = i
    
    # Insert after last import
    lines.insert(last_import_line + 1, f"from {from_module} import {import_stmt}")
    return "\n".join(lines)


def migrate_file(file_path: Path, analysis: BenchmarkAnalysis, dry_run: bool = False, backup: bool = False) -> MigrationResult:
    """Migrate a single benchmark file."""
    result = MigrationResult(file_path=str(file_path), success=False)
    
    if not analysis.benchmark_class:
        result.error = "No benchmark class found"
        return result
    
    if analysis.is_cuda_binary:
        # CudaBinaryBenchmark has its own get_input_signature via workload_params
        result.success = True
        result.needs_manual_review = True
        result.review_reason = "CudaBinaryBenchmark - use workload_params in __init__"
        return result
    
    # Generate methods to add
    methods_to_add = []
    
    sig_method = generate_get_input_signature(analysis)
    if sig_method:
        methods_to_add.append(("get_input_signature", sig_method))
    
    validate_method = generate_validate_result(analysis)
    if validate_method:
        methods_to_add.append(("validate_result", validate_method))
    
    workload_method = generate_get_workload_metadata(analysis)
    if workload_method:
        methods_to_add.append(("get_workload_metadata", workload_method))
    
    verify_output_method = generate_get_verify_output(analysis)
    if verify_output_method:
        methods_to_add.append(("get_verify_output", verify_output_method))
    
    if not methods_to_add:
        result.success = True
        return result
    
    # Read source
    try:
        source = file_path.read_text()
    except Exception as e:
        result.error = f"Failed to read file: {e}"
        return result
    
    # Find where to insert methods (end of class)
    class_end_line = find_class_end(source, analysis.benchmark_class)
    if class_end_line is None:
        result.error = f"Could not find end of class {analysis.benchmark_class}"
        return result
    
    # Check if we need to add imports
    needs_optional_import = "Optional[str]" not in source and any(
        name in ("validate_result", "get_verify_output") for name, _ in methods_to_add
    )
    needs_workload_import = "WorkloadMetadata" not in source and any(
        name == "get_workload_metadata" for name, _ in methods_to_add
    )
    needs_torch_import = "import torch" not in source and any(
        name == "get_verify_output" for name, _ in methods_to_add
    )
    
    # Build new source
    lines = source.split("\n")
    
    # Add methods at end of class
    method_code = "\n".join(code for _, code in methods_to_add)
    lines.insert(class_end_line + 1, method_code)
    
    new_source = "\n".join(lines)
    
    # Add imports if needed
    if needs_optional_import:
        if "from typing import" in new_source:
            new_source = add_import_if_needed(new_source, "Optional", "typing")
        else:
            # Find first import and add typing import
            first_import = new_source.find("import ")
            if first_import != -1:
                line_start = new_source.rfind("\n", 0, first_import) + 1
                new_source = new_source[:line_start] + "from typing import Optional\n" + new_source[line_start:]
    
    if needs_workload_import:
        new_source = add_import_if_needed(new_source, "WorkloadMetadata", "core.harness.benchmark_harness")
    
    if needs_torch_import:
        # Find first import and add torch import
        lines = new_source.split("\n")
        first_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                first_import_idx = i
                break
        lines.insert(first_import_idx, "import torch")
        new_source = "\n".join(lines)
    
    result.methods_added = [name for name, _ in methods_to_add]
    
    # Check if TODOs were added (needs manual review)
    if "# TODO:" in method_code:
        result.needs_manual_review = True
        result.review_reason = "Generated methods contain TODOs requiring manual completion"
    
    if dry_run:
        result.success = True
        return result
    
    # Create backup if requested
    if backup:
        backup_path = file_path.with_suffix(".py.bak")
        try:
            shutil.copy(file_path, backup_path)
            result.backup_path = str(backup_path)
        except Exception as e:
            result.error = f"Failed to create backup: {e}"
            return result
    
    # Write modified source
    try:
        file_path.write_text(new_source)
        result.success = True
    except Exception as e:
        result.error = f"Failed to write file: {e}"
    
    return result


# =============================================================================
# Discovery
# =============================================================================

def find_benchmark_files(root_dir: Path, phase: Optional[int] = None, chapter: Optional[str] = None) -> List[Path]:
    """Find benchmark files to migrate."""
    files: List[Path] = []
    patterns = ["baseline_*.py", "optimized_*.py"]
    
    if chapter:
        # Single chapter
        search_dirs = [root_dir / chapter]
        if chapter.startswith("labs/"):
            search_dirs = [root_dir / chapter]
    elif phase:
        # Phase-based
        chapters = PHASES.get(phase, [])
        search_dirs = []
        for ch in chapters:
            if ch == "labs":
                # Include all labs subdirectories
                labs_dir = root_dir / "labs"
                if labs_dir.exists():
                    search_dirs.extend(d for d in labs_dir.iterdir() if d.is_dir())
            else:
                search_dirs.append(root_dir / ch)
    else:
        # All chapters
        search_dirs = [
            d for d in root_dir.iterdir()
            if d.is_dir() and (d.name.startswith("ch") or d.name == "labs")
        ]
        labs_dir = root_dir / "labs"
        if labs_dir.exists():
            search_dirs.extend(d for d in labs_dir.iterdir() if d.is_dir())
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            files.extend(search_dir.glob(pattern))
            files.extend(search_dir.glob(f"**/{pattern}"))
    
    return sorted(set(files))


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(report: MigrationReport) -> str:
    """Generate a Markdown summary of the migration."""
    lines = [
        "# Benchmark Migration Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Phase:** {report.phase or 'All'}",
        f"**Chapter:** {report.chapter or 'All'}",
        f"**Dry Run:** {'Yes' if report.dry_run else 'No'}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total files | {report.total_files} |",
        f"| Files modified | {report.files_modified} |",
        f"| Files skipped | {report.files_skipped} |",
        f"| Files errored | {report.files_errored} |",
        f"| Manual review needed | {report.manual_review_needed} |",
        "",
    ]
    
    # Group results by chapter
    by_chapter: Dict[str, List[MigrationResult]] = defaultdict(list)
    for r in report.results:
        chapter = Path(r.file_path).parent.name
        by_chapter[chapter].append(r)
    
    if report.files_modified > 0:
        lines.extend([
            "## Files Modified",
            "",
        ])
        for chapter in sorted(by_chapter.keys()):
            modified = [r for r in by_chapter[chapter] if r.methods_added]
            if modified:
                lines.append(f"### {chapter}")
                lines.append("")
                for r in modified:
                    methods = ", ".join(r.methods_added)
                    lines.append(f"- `{Path(r.file_path).name}`: added {methods}")
                lines.append("")
    
    if report.files_errored > 0:
        lines.extend([
            "## Errors",
            "",
        ])
        for r in report.results:
            if r.error:
                lines.append(f"- `{r.file_path}`: {r.error}")
        lines.append("")
    
    if report.manual_review_needed > 0:
        lines.extend([
            "## Manual Review Needed",
            "",
            "The following files have TODOs that require manual completion:",
            "",
        ])
        for r in report.results:
            if r.needs_manual_review:
                lines.append(f"- `{r.file_path}`: {r.review_reason}")
        lines.append("")
    
    return "\n".join(lines)


def generate_manual_review_list(report: MigrationReport) -> str:
    """Generate a list of files needing manual review."""
    lines = [
        "# Files Requiring Manual Review",
        "#",
        "# These files have generated TODOs that need to be completed manually.",
        "# Look for '# TODO:' comments in the generated methods.",
        "#",
    ]
    
    for r in report.results:
        if r.needs_manual_review:
            lines.append(f"{r.file_path}  # {r.review_reason}")
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def migrate_benchmarks(
    root_dir: Path,
    phase: Optional[int] = None,
    chapter: Optional[str] = None,
    dry_run: bool = False,
    backup: bool = False,
    report_path: Optional[Path] = None,
) -> MigrationReport:
    """Run migration on benchmark files."""
    report = MigrationReport(
        timestamp=datetime.now().isoformat(),
        phase=phase,
        chapter=chapter,
        dry_run=dry_run,
    )
    
    # Find files
    files = find_benchmark_files(root_dir, phase, chapter)
    report.total_files = len(files)
    
    print(f"Found {len(files)} benchmark files to process")
    
    for file_path in files:
        # Analyze
        analysis = analyze_benchmark_file(file_path)
        
        if analysis.errors:
            result = MigrationResult(
                file_path=str(file_path),
                success=False,
                error="; ".join(analysis.errors),
            )
            report.results.append(result)
            report.files_errored += 1
            print(f"  ERROR: {file_path.name} - {result.error}")
            continue
        
        # Check if already compliant
        if (analysis.has_get_input_signature and 
            analysis.has_validate_result and 
            (analysis.has_get_workload_metadata or analysis.has_register_workload_metadata)):
            result = MigrationResult(file_path=str(file_path), success=True)
            report.results.append(result)
            report.files_skipped += 1
            continue
        
        # Migrate
        result = migrate_file(file_path, analysis, dry_run, backup)
        report.results.append(result)
        
        if result.success:
            if result.methods_added:
                report.files_modified += 1
                status = "DRY-RUN" if dry_run else "MODIFIED"
                print(f"  {status}: {file_path.name} - added {', '.join(result.methods_added)}")
            else:
                report.files_skipped += 1
        else:
            report.files_errored += 1
            print(f"  ERROR: {file_path.name} - {result.error}")
        
        if result.needs_manual_review:
            report.manual_review_needed += 1
    
    # Generate reports
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nJSON report written to {report_path}")
        
        # Markdown summary
        md_path = report_path.with_suffix(".md")
        md_content = generate_markdown_report(report)
        md_path.write_text(md_content)
        print(f"Markdown summary written to {md_path}")
        
        # Manual review file (if any)
        if report.manual_review_needed > 0:
            review_path = report_path.parent / "manual_review_needed.txt"
            review_content = generate_manual_review_list(report)
            review_path.write_text(review_content)
            print(f"Manual review list written to {review_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("MIGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:         {report.total_files}")
    print(f"Files modified:      {report.files_modified}")
    print(f"Files skipped:       {report.files_skipped}")
    print(f"Files errored:       {report.files_errored}")
    print(f"Manual review needed: {report.manual_review_needed}")
    
    if dry_run:
        print("\n*** DRY RUN - no files were modified ***")
    
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-migrate benchmark files to add verification methods"
    )
    parser.add_argument(
        "--phase", "-p",
        type=int,
        choices=[1, 2, 3, 4],
        help="Migration phase (1=ch01-06, 2=ch07-14, 3=ch15-20, 4=labs)"
    )
    parser.add_argument(
        "--chapter", "-c",
        help="Specific chapter to migrate (e.g., ch07, labs/moe_cuda)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview changes without writing files"
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Create .bak files before modifying"
    )
    parser.add_argument(
        "--report", "-r",
        type=Path,
        help="Path to write JSON migration report"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=".",
        help="Root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    root_dir = args.root.resolve()
    
    report = migrate_benchmarks(
        root_dir=root_dir,
        phase=args.phase,
        chapter=args.chapter,
        dry_run=args.dry_run,
        backup=args.backup,
        report_path=args.report,
    )
    
    return 0 if report.files_errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

