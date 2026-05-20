# json CPU backends.
#
# v0.2 ships two CPU paths:
#   - `parse_cpu_native` (default): two-pass stage 1 + stage 2 walker.
#       Stage 1 has scalar (`stage1_scalar`) and SIMD (`stage1`)
#       implementations; both are byte-identical, enforced by
#       `tests/test_stage1_equivalence.mojo`. Default is scalar; opt
#       into SIMD with `parse_cpu_native[force_scalar=False]`.
#   - simdjson FFI: optional C++ backend, exposed via `SimdjsonFFI`
#       and selected with `loads[target='cpu-simdjson']`.
#
# The pre-v0.2 `parse_mojo` / `parse_simd` entry points (and their
# `MojoJSONParser` / `FastParser` structs) were deleted in v0.2-E:
# the codepaths they wrapped were dead since v0.2-A wired the
# canonical CPU pipeline through `parse_cpu_native`.

# Common type tags shared with the simdjson FFI bindings.
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

# simdjson FFI backend.
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
        Parsed Value.
    """
    return parse_two_pass[force_scalar=force_scalar](s)
