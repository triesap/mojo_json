# json CPU backends
# Supports multiple backends:
#   - simdjson (FFI): High-performance C++ backend (default)
#   - mojo (native): Pure Mojo implementation (zero FFI)

# =============================================================================
# Common Types (backend-agnostic)
# =============================================================================

from .types import (
    JSON_TYPE_NULL,
    JSON_TYPE_BOOL,
    JSON_TYPE_INT64,
    JSON_TYPE_UINT64,
    JSON_TYPE_DOUBLE,
    JSON_TYPE_STRING,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT,
    JSON_OK,
    JSON_ERROR_INVALID,
    JSON_ERROR_CAPACITY,
    JSON_ERROR_UTF8,
    JSON_ERROR_OTHER,
)

# =============================================================================
# simdjson Backend (FFI)
# =============================================================================

from .simdjson_ffi import (
    SimdjsonFFI,
    SIMDJSON_OK,
    SIMDJSON_ERROR_INVALID_JSON,
    SIMDJSON_ERROR_CAPACITY,
    SIMDJSON_ERROR_UTF8,
    SIMDJSON_ERROR_OTHER,
    SIMDJSON_TYPE_NULL,
    SIMDJSON_TYPE_BOOL,
    SIMDJSON_TYPE_INT64,
    SIMDJSON_TYPE_UINT64,
    SIMDJSON_TYPE_DOUBLE,
    SIMDJSON_TYPE_STRING,
    SIMDJSON_TYPE_ARRAY,
    SIMDJSON_TYPE_OBJECT,
)

# =============================================================================
# Mojo Backend (Pure Native)
# =============================================================================

from .mojo_backend import parse_mojo, MojoJSONParser

# =============================================================================
# SIMD Backend (High-Performance Native)
# =============================================================================

from .simd_backend import parse_simd, FastParser

# =============================================================================
# Two-Pass Backend (v0.2 Stage 1 + Stage 2)
# =============================================================================
#
# `parse_cpu_native` is the v0.2 entry point: stage 1 builds the
# structural index, stage 2 walks it. Stage 1 ships in two
# implementations -- a scalar oracle (`stage1_scalar`) and a SIMD
# version (`stage1`). Both produce byte-identical output, enforced by
# `tests/test_stage1_equivalence.mojo`.
#
# Default is the scalar stage 1: empirically, on JSON with many short
# strings (twitter.json style) the per-position iteration cost of the
# SIMD path's bitmask resolver dominates the per-byte cost of the
# scalar walk. Callers who want to opt into SIMD can call
# `parse_cpu_native[force_scalar=False]` directly. See
# `pprint/.cursor/rules/plan.mdc` Phase 7 for the historical perf data
# motivating this default.

from ..value import Value
from .stage1_scalar import (
    parse_structural_scalar,
    StructuralIndex,
)
from .stage1 import parse_structural_simd
from .stage2 import parse_with_index, parse_two_pass


def parse_cpu_native[force_scalar: Bool = True](s: String) raises -> Value:
    """v0.2 two-pass CPU parser.

    Parameters:
        force_scalar: When True (default), use the scalar stage 1
            oracle. Set to False to use the SIMD stage 1 implementation
            (for benchmarking or on workloads that favor SIMD).

    Args:
        s: JSON input string.

    Returns:
        Parsed Value, equivalent to `parse_simd(s)`.
    """
    return parse_two_pass[force_scalar=force_scalar](s)
