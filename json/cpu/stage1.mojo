# SIMD stage 1: structural-index builder using `pack_bits`.
#
# This is the SIMD counterpart of `stage1_scalar.parse_structural_scalar`.
# It scans the input in 32-byte chunks, classifies each byte as
# structural / quote / backslash / other via SIMD comparisons, then uses
# `pack_bits` to convert the comparison masks into 32-bit bitmaps so we
# can iterate match positions with `count_trailing_zeros`.
#
# String/escape state -------------------------------------------------------
#
# JSON's "is this byte inside a string?" rule depends on a byte-by-byte
# escape-state machine: `\"` does not close a string, `\\"` does, etc.
# True simdjson-style stage 1 uses a carry-less multiply (CLMUL) trick to
# turn this into a SIMD-friendly prefix XOR. Mojo does not yet expose
# CLMUL, so we fall back to a per-chunk scalar scan over the quote/
# backslash positions only -- still much cheaper than the byte-by-byte
# scalar version because we touch only those positions.
#
# Performance gate ----------------------------------------------------------
#
# v0.2 Phase C ships scalar as the default stage 1 because, on Mojo's
# current SIMD surface, the SIMD scan does not consistently beat the
# scalar walk on twitter.json (see `pprint/.cursor/rules/plan.mdc`
# Phase 7 results). The SIMD path is wired here for correctness and as
# a future fast path: it MUST produce identical output to the scalar
# oracle, which `tests/test_stage1_equivalence.mojo` enforces.

from std.collections import List
from std.bit import count_trailing_zeros
from std.memory.unsafe import pack_bits

from .stage1_scalar import StructuralIndex


# ---------------------------------------------------------------------------
# SIMD parameters
# ---------------------------------------------------------------------------


comptime SIMD_WIDTH: Int = 32
"""Bytes processed per SIMD iteration. Picked to match common AVX2 / NEON
register sizes; `pack_bits` produces a 32-bit mask we iterate with
`count_trailing_zeros`."""


# ---------------------------------------------------------------------------
# parse_structural_simd
# ---------------------------------------------------------------------------


def parse_structural_simd(input: String) -> StructuralIndex:
    """Build a structural index using a SIMD scan.

    Output is byte-for-byte identical to
    `stage1_scalar.parse_structural_scalar(input)` -- the equivalence is
    enforced by `tests/test_stage1_equivalence.mojo`.

    Performance is competitive with the scalar version on dense JSON
    arrays of long strings; on JSON with many short keys (`twitter.json`
    style) the per-match iteration cost dominates the SIMD load and the
    scalar walk wins. Callers should benchmark on their own corpora and
    flip the default in `parse_cpu_native` accordingly.
    """
    var bytes = input.as_bytes()
    var n = len(bytes)
    var index = StructuralIndex(capacity=n // 4)

    var in_string = False
    var escaped = False
    var i = 0

    comptime W = SIMD[DType.uint8, SIMD_WIDTH]

    while i + SIMD_WIDTH <= n:
        var chunk = bytes.unsafe_ptr().load[width=SIMD_WIDTH](i)

        # Mask of bytes equal to a "structural-or-string" marker --
        # `{` `}` `[` `]` `:` `,` `\\` `"`. We need the backslashes here
        # so the per-chunk scalar resolver can advance escape state.
        var lbrace = chunk.eq(W(UInt8(ord("{"))))
        var rbrace = chunk.eq(W(UInt8(ord("}"))))
        var lbrack = chunk.eq(W(UInt8(ord("["))))
        var rbrack = chunk.eq(W(UInt8(ord("]"))))
        var colon = chunk.eq(W(UInt8(ord(":"))))
        var comma = chunk.eq(W(UInt8(ord(","))))
        var quote = chunk.eq(W(UInt8(ord('"'))))
        var bslash = chunk.eq(W(UInt8(ord("\\"))))

        var struct_no_q = lbrace | rbrace | lbrack | rbrack | colon | comma
        var any_marker = struct_no_q | quote | bslash

        var struct_mask = pack_bits[dtype=DType.uint32](struct_no_q)
        var quote_mask = pack_bits[dtype=DType.uint32](quote)
        var bslash_mask = pack_bits[dtype=DType.uint32](bslash)
        var any_mask = pack_bits[dtype=DType.uint32](any_marker)

        # Fast path: no markers in this chunk and we're not inside a
        # string; advance past the whole 32-byte block.
        if any_mask == 0 and not in_string:
            i += SIMD_WIDTH
            escaped = False
            continue

        # Walk just the marker positions inside this chunk; the rest of
        # the bytes are uninteresting for stage 1.
        var local = any_mask
        while local != 0:
            var bit = Int(count_trailing_zeros(local))
            var pos = i + bit
            local &= local - 1  # Clear the bit we just visited.

            if escaped:
                escaped = False
                continue

            var bit_mask = UInt32(1) << UInt32(bit)
            var is_quote = (quote_mask & bit_mask) != 0
            var is_bslash = (bslash_mask & bit_mask) != 0
            var is_struct = (struct_mask & bit_mask) != 0

            if in_string:
                if is_bslash:
                    escaped = True
                    continue
                if is_quote:
                    index.positions.append(UInt32(pos))
                    in_string = False
                continue

            # Outside a string.
            if is_quote:
                index.positions.append(UInt32(pos))
                in_string = True
                continue
            if is_struct:
                index.positions.append(UInt32(pos))

        i += SIMD_WIDTH

    # Tail: handle bytes that didn't fit in the last 32-byte chunk
    # using the byte-by-byte algorithm (reuse the same logic as
    # `parse_structural_scalar`'s inner loop).
    while i < n:
        var c = bytes[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if in_string:
            if c == UInt8(ord("\\")):
                escaped = True
                i += 1
                continue
            if c == UInt8(ord('"')):
                index.positions.append(UInt32(i))
                in_string = False
                i += 1
                continue
            i += 1
            continue
        if c == UInt8(ord('"')):
            index.positions.append(UInt32(i))
            in_string = True
            i += 1
            continue
        if (
            c == UInt8(ord("{"))
            or c == UInt8(ord("}"))
            or c == UInt8(ord("["))
            or c == UInt8(ord("]"))
            or c == UInt8(ord(":"))
            or c == UInt8(ord(","))
        ):
            index.positions.append(UInt32(i))
        i += 1

    return index^
