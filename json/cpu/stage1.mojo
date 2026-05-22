# SIMD stage 1: structural-index builder using `pack_bits`.
#
# This is the SIMD counterpart of `stage1_scalar.parse_structural_scalar`.
# It scans the input in 32-byte chunks, classifies each byte via a
# PSHUFB-style nibble lookup (Phase 2 of v0.2's parity push), then uses
# `pack_bits` to convert the resulting per-category bool masks into
# 32-bit bitmaps so we can iterate match positions with
# `count_trailing_zeros`.
#
# Classifier ----------------------------------------------------------------
#
# The previous Phase-1 walker did eight independent `.eq()` SIMD compares
# per chunk -- one for each of `{`, `}`, `[`, `]`, `:`, `,`, `"`, `\\`.
# That's eight SIMD comparison instructions plus six ORs to assemble the
# per-category masks before `pack_bits`. simdjson uses two PSHUFBs (one
# for the low nibble, one for the high nibble) plus an AND -- a 4x
# reduction in classifier ops, which on Apple Silicon's NEON happens to
# be the dominant stage 1 cost.
#
# We replicate that here via Mojo's `SIMD._dynamic_shuffle`, which
# lowers to `pshufb` on x86 SSE4+ and `vqtbl1q` on ARM64. Crucially this
# is a single comptime call: the same source produces the right
# instruction on every supported ISA, with an automatic scalar fallback
# for ISAs that don't have a byte-permute. No `__attribute__((target))`
# fan-out, no per-arch source duplication.
#
# Each byte is classified by ANDing two 16-entry tables, indexed by the
# byte's low and high nibbles respectively. The table values are
# bit-encoded so that exactly one bit survives per marker class:
#
#     bit 0 = `"`              bit 4 = `\\`
#     bit 1 = `:`              bit 5 = `,`
#     bit 2 = `[`              bit 6 = `{`
#     bit 3 = `]`              bit 7 = `}`
#
# `low_table[low_nibble] & high_table[high_nibble]` then yields:
#   - 0x00         for non-markers (every other byte in the corpus),
#   - 0x01         for a `"`,
#   - 0x10         for a `\\`,
#   - 0x02 / 0x04 / 0x08 / 0x20 / 0x40 / 0x80
#                  for the six structural characters.
#
# We extract the three category bools (`struct`, `quote`, `bslash`) from
# the classifier output via cheap bitmask ANDs and feed them through
# `pack_bits` exactly as before. The escape-state resolver and the
# scalar tail loop are unchanged -- their cost is dominated by the
# carry-less-multiply gap that Phase 3 closes.
#
# String/escape state -------------------------------------------------------
#
# JSON's "is this byte inside a string?" rule depends on a byte-by-byte
# escape-state machine: `\"` does not close a string, `\\"` does, etc.
# A true SIMD stage 1 uses CLMUL to turn this into a SIMD-friendly
# prefix XOR. That's Phase 3 -- this file still falls back to a per-chunk
# scalar resolver over the marker positions only (still cheap because we
# touch only those positions, not every byte).
#
# Important subtlety: the marker walk only iterates positions that hit
# `{ } [ ] : , \\ "`. An escape consumes the byte at `bslash + 1`, which
# may NOT be a marker -- it could be `n`, `t`, `u`, an arbitrary text
# byte, etc. Carrying `escaped` to the next *marker* (regardless of
# distance) is therefore wrong: a non-marker byte between the backslash
# and the next marker would already have consumed the escape silently.
# We track the absolute position of the backslash that set the escape
# and only honor the escape when the next visited position is exactly
# `bslash_pos + 1`. The same rule applies across chunk boundaries.

from std.collections import List
from std.bit import count_trailing_zeros
from std.memory.unsafe import pack_bits
from std.sys import CompilationTarget

from .stage1_scalar import StructuralIndex


# ---------------------------------------------------------------------------
# SIMD parameters
# ---------------------------------------------------------------------------


comptime SIMD_WIDTH: Int = 32
"""Bytes processed per SIMD iteration. Picked to match common AVX2 / NEON
register sizes; `pack_bits` produces a 32-bit mask we iterate with
`count_trailing_zeros`."""


# Category-bit masks (matches the PSHUFB table encoding above).
comptime _CAT_QUOTE: UInt8 = 0x01
"""Bit 0 of the classifier output: byte is `\"`."""
comptime _CAT_BSLASH: UInt8 = 0x10
"""Bit 4 of the classifier output: byte is `\\`."""
comptime _CAT_STRUCT_MASK: UInt8 = 0xEE
"""Bits 1, 2, 3, 5, 6, 7 of the classifier output: byte is one of
`: [ ] , { }`."""


# ---------------------------------------------------------------------------
# PSHUFB-style nibble-lookup classifier.
#
# Given a 32-byte chunk, returns a 32-byte SIMD vector where each lane
# is the classifier byte for the corresponding input byte (0 for
# non-markers; one of the bits above for markers).
# ---------------------------------------------------------------------------


@always_inline
def _classify_chunk[
    W: Int
](chunk: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    """Two-PSHUFB nibble-lookup marker classifier.

    Indexes `_LOW_TABLE` by the low nibble of each input byte and
    `_HIGH_TABLE` by the high nibble; ANDs the two lookups. The result
    has at most one bit set per byte, and zero for any byte that is not
    one of `\" \\ { } [ ] : ,`.

    The lookup is implemented via `SIMD._dynamic_shuffle`, which lowers
    to `pshufb` (x86 SSE4+/AVX2) or `vqtbl1q` (ARM64 NEON). On ISAs
    without a byte-permute it falls back to an unrolled scalar gather
    -- still correct, just not the fast path.
    """
    # Tables encoded so that low_table[low(b)] & high_table[high(b)]
    # leaves exactly one category bit set per marker.
    comptime _LOW_TABLE = SIMD[DType.uint8, 16](
        # idx:  0     1     2     3     4     5     6     7
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        #       8     9     A     B     C     D     E     F
        0x00,
        0x00,
        0x02,
        0x44,
        0x30,
        0x88,
        0x00,
        0x00,
    )
    comptime _HIGH_TABLE = SIMD[DType.uint8, 16](
        # idx:  0     1     2     3     4     5     6     7
        0x00,
        0x00,
        0x21,
        0x02,
        0x00,
        0x1C,
        0x00,
        0xC0,
        #       8     9     A     B     C     D     E     F
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    )

    var low_nibble = chunk & 0x0F
    var high_nibble = chunk >> 4
    var lo_lookup = _LOW_TABLE._dynamic_shuffle(low_nibble)
    var hi_lookup = _HIGH_TABLE._dynamic_shuffle(high_nibble)
    return lo_lookup & hi_lookup


# ---------------------------------------------------------------------------
# parse_structural_simd
# ---------------------------------------------------------------------------


def parse_structural_simd(input: String) -> StructuralIndex:
    """Build a structural index using a SIMD scan.

    Output is byte-for-byte identical to
    `stage1_scalar.parse_structural_scalar(input)` -- the equivalence is
    enforced by `tests/test_stage1_equivalence.mojo`, including a
    full-document run against the benchmark corpora.

    Faster than the scalar walker by ~1.3x to ~1.5x on the benchmark
    corpora after Phase 2; this is the default stage 1 used by
    `parse_cpu_native_tape`.
    """
    var bytes = input.as_bytes()
    var n = len(bytes)
    var index = StructuralIndex(capacity=n // 4)

    var in_string = False
    # Absolute position of the most recent backslash that has not yet
    # had its escape consumed. -1 means "no pending escape". The byte
    # actually escaped is bslash_pos + 1, regardless of whether it is
    # a marker or a non-marker.
    var bslash_pos: Int = -1
    var i = 0

    while i + SIMD_WIDTH <= n:
        var chunk = bytes.unsafe_ptr().load[width=SIMD_WIDTH](i)

        # Two-PSHUFB classifier: one byte out per input byte, with at
        # most one category bit set. We read per-marker categories
        # straight out of `classified` via `extractelement`, so we only
        # need ONE `pack_bits` (for the iteration bitmap) instead of
        # the four the Phase-1 walker computed.
        var classified = _classify_chunk(chunk)

        comptime W = SIMD[DType.uint8, SIMD_WIDTH]
        var any_marker = classified.ne(W(0))
        var any_mask = pack_bits[dtype=DType.uint32](any_marker)

        # Fast path: no markers in this chunk and we're not inside a
        # string. Any pending escape that pointed inside this chunk is
        # silently consumed by a non-marker byte; pending escapes that
        # point past the end of this chunk (only possible if the prior
        # chunk ended with `\\` at its last byte) survive into the next
        # chunk.
        if any_mask == 0 and not in_string:
            if bslash_pos >= 0 and bslash_pos < i + SIMD_WIDTH - 1:
                bslash_pos = -1
            i += SIMD_WIDTH
            continue

        # Walk just the marker positions inside this chunk; the rest of
        # the bytes are uninteresting for stage 1.
        var local = any_mask
        while local != 0:
            var bit = Int(count_trailing_zeros(local))
            var pos = i + bit
            local &= local - 1  # Clear the bit we just visited.

            # Resolve any pending escape against this exact position.
            #   pos == bslash_pos + 1  -> this marker IS the escaped
            #                             byte; skip it and clear the
            #                             pending escape.
            #   pos >  bslash_pos + 1  -> the escape was consumed by a
            #                             non-marker byte that lives
            #                             between the backslash and
            #                             this marker; treat the
            #                             current marker normally.
            if bslash_pos >= 0:
                if pos == bslash_pos + 1:
                    bslash_pos = -1
                    continue
                bslash_pos = -1

            # One byte read out of the classifier SIMD: encoded
            # category bits for this marker.
            var cat = classified[bit]

            if in_string:
                if (cat & _CAT_BSLASH) != 0:
                    bslash_pos = pos
                    continue
                if (cat & _CAT_QUOTE) != 0:
                    index.positions.append(UInt32(pos))
                    in_string = False
                continue

            # Outside a string.
            if (cat & _CAT_QUOTE) != 0:
                index.positions.append(UInt32(pos))
                in_string = True
                continue
            if (cat & _CAT_STRUCT_MASK) != 0:
                index.positions.append(UInt32(pos))

        # End of chunk. If a backslash was set during this chunk and
        # the byte it would escape is still inside this chunk, that
        # byte was a non-marker (otherwise we'd have visited it above
        # and cleared bslash_pos). Drop the pending escape so we don't
        # mis-fire it on the next chunk's first marker.
        if bslash_pos >= 0 and bslash_pos < i + SIMD_WIDTH - 1:
            bslash_pos = -1

        i += SIMD_WIDTH

    # Tail: handle bytes that didn't fit in the last 32-byte chunk
    # using the byte-by-byte algorithm (reuse the same logic as
    # `parse_structural_scalar`'s inner loop).
    var escaped = bslash_pos >= 0 and bslash_pos == i - 1
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
