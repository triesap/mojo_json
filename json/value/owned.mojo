# json - Copy-on-write owned tree representation.
#
# `OwnedValue` is the structured tree representation of a JSON value.
# Phase B routes `Value.set` / `Value.append` / `Value.set_at` through
# this type instead of doing raw-string surgery on `Value._raw`. The
# rewrite path is:
#
#   raw_json --(_value_to_owned)--> OwnedValue --(mutate)--> OwnedValue
#                                                                |
#   raw_json <--(_owned_to_json)----------------------------------+
#
# This is O(N) per mutation (same big-O as v0.1 raw-string surgery)
# but eliminates the silent-bug class where surgery on a sibling
# subtree corrupts adjacent text. Phase D / a future v0.3 may keep
# the OwnedValue resident across calls instead of round-tripping
# through the raw string each time; for now the round-trip preserves
# the v0.1 invariant that `Value.raw_json()` reflects the latest
# mutation.

from std.collections import List
from std.collections.dict import Dict

from .value import Value, Null
from ..unicode import unescape_json_string


# ---------------------------------------------------------------------------
# OwnedValue
# ---------------------------------------------------------------------------


struct OwnedValue(Copyable, Movable):
    """Structured tree representation of a JSON value.

    Unlike `Value`, an `OwnedValue` for an array stores its children in
    `array_val: List[OwnedValue]` rather than as a raw JSON substring,
    so a mutation at any nesting level can be applied in place and a
    fresh JSON serialization can be produced after the fact.
    """

    # Type tag: 0=null, 1=bool, 2=int, 3=float, 4=string, 5=array, 6=object.
    var kind: Int
    var bool_val: Bool
    var int_val: Int64
    var float_val: Float64
    var str_val: String
    var array_val: List[OwnedValue]
    var object_keys: List[String]
    var object_values: List[OwnedValue]

    def __init__(out self):
        self.kind = 0
        self.bool_val = False
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = String()
        self.array_val = List[OwnedValue]()
        self.object_keys = List[String]()
        self.object_values = List[OwnedValue]()

    def copy(self) -> Self:
        var out = Self()
        out.kind = self.kind
        out.bool_val = self.bool_val
        out.int_val = self.int_val
        out.float_val = self.float_val
        out.str_val = self.str_val
        out.array_val = self.array_val.copy()
        out.object_keys = self.object_keys.copy()
        out.object_values = self.object_values.copy()
        return out^

    @staticmethod
    def make_null() -> Self:
        return Self()

    @staticmethod
    def make_bool(b: Bool) -> Self:
        var v = Self()
        v.kind = 1
        v.bool_val = b
        return v^

    @staticmethod
    def make_int(i: Int64) -> Self:
        var v = Self()
        v.kind = 2
        v.int_val = i
        return v^

    @staticmethod
    def make_float(f: Float64) -> Self:
        var v = Self()
        v.kind = 3
        v.float_val = f
        return v^

    @staticmethod
    def make_string(var s: String) -> Self:
        var v = Self()
        v.kind = 4
        v.str_val = s^
        return v^

    @staticmethod
    def make_array(var items: List[OwnedValue]) -> Self:
        var v = Self()
        v.kind = 5
        v.array_val = items^
        return v^

    @staticmethod
    def make_object(
        var keys: List[String], var values: List[OwnedValue]
    ) -> Self:
        var v = Self()
        v.kind = 6
        v.object_keys = keys^
        v.object_values = values^
        return v^


# ---------------------------------------------------------------------------
# Value <-> OwnedValue conversion
# ---------------------------------------------------------------------------


def _value_to_owned(v: Value) raises -> OwnedValue:
    """Convert a `Value` to an `OwnedValue` tree.

    For arrays and objects, this re-parses the raw JSON substring; for
    primitives it copies the inline value. Phase C will short-circuit
    this step when the source `Value` is already tape-backed.
    """
    if v.is_null():
        return OwnedValue.make_null()
    if v.is_bool():
        return OwnedValue.make_bool(v.bool_value())
    if v.is_int():
        return OwnedValue.make_int(v.int_value())
    if v.is_float():
        return OwnedValue.make_float(v.float_value())
    if v.is_string():
        return OwnedValue.make_string(v.string_value())
    if v.is_array():
        return _parse_owned_value(v.raw_json())
    if v.is_object():
        return _parse_owned_value(v.raw_json())
    raise Error("Unknown Value kind in _value_to_owned")


def _parse_owned_value(json_str: String) raises -> OwnedValue:
    """Parse a raw JSON value string into an `OwnedValue` tree.

    This is a small recursive-descent walker that mirrors the parser
    in `value.value._parse_json_value_to_value` but emits an
    `OwnedValue` directly instead of a `Value` whose array/object body
    is stored as a raw substring.
    """
    var bytes = json_str.as_bytes()
    var n = len(bytes)
    var pos = 0

    var result = _parse_owned_at(bytes, pos, n, json_str)
    return result^


def _skip_ws(bytes: Span[UInt8, _], pos: Int, n: Int) -> Int:
    var i = pos
    while i < n and (
        bytes[i] == UInt8(ord(" "))
        or bytes[i] == UInt8(ord("\t"))
        or bytes[i] == UInt8(ord("\n"))
        or bytes[i] == UInt8(ord("\r"))
    ):
        i += 1
    return i


def _parse_owned_at(
    bytes: Span[UInt8, _], start: Int, n: Int, raw_json: String
) raises -> OwnedValue:
    """Parse a single JSON value starting at `start` in `bytes`.

    `raw_json` is passed through only so error messages can include
    context; functionally only `bytes[start:n]` is consumed.
    """
    var i = _skip_ws(bytes, start, n)
    if i >= n:
        raise Error("Unexpected end of JSON")

    var c = bytes[i]

    if c == UInt8(ord("n")):
        return OwnedValue.make_null()
    if c == UInt8(ord("t")):
        return OwnedValue.make_bool(True)
    if c == UInt8(ord("f")):
        return OwnedValue.make_bool(False)
    if c == UInt8(ord('"')):
        var start_idx = i + 1
        var end_idx = start_idx
        var has_escapes = False
        while end_idx < n:
            var b = bytes[end_idx]
            if b == UInt8(ord("\\")):
                has_escapes = True
                end_idx += 2
                continue
            if b == UInt8(ord('"')):
                break
            end_idx += 1
        if not has_escapes:
            return OwnedValue.make_string(
                String(unsafe_from_utf8=bytes[start_idx:end_idx])
            )
        var bytes_list = List[UInt8](capacity=n)
        for j in range(n):
            bytes_list.append(bytes[j])
        var unescaped = unescape_json_string(bytes_list, start_idx, end_idx)
        return OwnedValue.make_string(String(unsafe_from_utf8=unescaped^))

    if c == UInt8(ord("-")) or (c >= UInt8(ord("0")) and c <= UInt8(ord("9"))):
        var num_start = i
        var is_float = False
        if c == UInt8(ord("-")):
            i += 1
        while i < n:
            var b = bytes[i]
            if (b >= UInt8(ord("0")) and b <= UInt8(ord("9"))) or b == UInt8(
                ord("+")
            ):
                i += 1
            elif (
                b == UInt8(ord("."))
                or b == UInt8(ord("e"))
                or b == UInt8(ord("E"))
            ):
                is_float = True
                i += 1
            elif b == UInt8(ord("-")):
                # Negative exponent.
                i += 1
            else:
                break
        var num_str = String(unsafe_from_utf8=bytes[num_start:i])
        if is_float:
            return OwnedValue.make_float(atof(num_str))
        return OwnedValue.make_int(Int64(atol(num_str)))

    if c == UInt8(ord("[")):
        return _parse_owned_array(bytes, i, n, raw_json)

    if c == UInt8(ord("{")):
        return _parse_owned_object(bytes, i, n, raw_json)

    raise Error("Invalid JSON value")


def _parse_owned_array(
    bytes: Span[UInt8, _], start: Int, n: Int, raw_json: String
) raises -> OwnedValue:
    var i = start + 1  # Skip '['
    var items = List[OwnedValue]()

    i = _skip_ws(bytes, i, n)
    if i < n and bytes[i] == UInt8(ord("]")):
        return OwnedValue.make_array(items^)

    while i < n:
        var elem = _parse_owned_at(bytes, i, n, raw_json)
        items.append(elem^)

        # Advance past the element we just parsed by walking until we hit
        # a top-level comma or the closing bracket.
        i = _skip_to_array_separator(bytes, i, n)

        if i >= n:
            raise Error("Unterminated array")

        if bytes[i] == UInt8(ord(",")):
            i += 1
            i = _skip_ws(bytes, i, n)
            continue
        elif bytes[i] == UInt8(ord("]")):
            return OwnedValue.make_array(items^)
        else:
            raise Error("Expected ',' or ']' in array")

    raise Error("Unterminated array")


def _parse_owned_object(
    bytes: Span[UInt8, _], start: Int, n: Int, raw_json: String
) raises -> OwnedValue:
    var i = start + 1  # Skip '{'
    var keys = List[String]()
    var values = List[OwnedValue]()

    i = _skip_ws(bytes, i, n)
    if i < n and bytes[i] == UInt8(ord("}")):
        return OwnedValue.make_object(keys^, values^)

    while i < n:
        i = _skip_ws(bytes, i, n)
        if i >= n or bytes[i] != UInt8(ord('"')):
            raise Error("Expected string key in object")
        # Parse key.
        var key_start = i + 1
        var key_end = key_start
        while key_end < n:
            var b = bytes[key_end]
            if b == UInt8(ord("\\")):
                key_end += 2
                continue
            if b == UInt8(ord('"')):
                break
            key_end += 1
        var key = String(unsafe_from_utf8=bytes[key_start:key_end])
        keys.append(key^)
        i = key_end + 1
        i = _skip_ws(bytes, i, n)
        if i >= n or bytes[i] != UInt8(ord(":")):
            raise Error("Expected ':' after object key")
        i += 1
        i = _skip_ws(bytes, i, n)

        # Parse value.
        var v = _parse_owned_at(bytes, i, n, raw_json)
        values.append(v^)

        i = _skip_to_object_separator(bytes, i, n)
        if i >= n:
            raise Error("Unterminated object")
        if bytes[i] == UInt8(ord(",")):
            i += 1
            continue
        elif bytes[i] == UInt8(ord("}")):
            return OwnedValue.make_object(keys^, values^)
        else:
            raise Error("Expected ',' or '}' in object")

    raise Error("Unterminated object")


def _skip_to_array_separator(bytes: Span[UInt8, _], start: Int, n: Int) -> Int:
    """Walk past one JSON value, returning the index of the next ',' or ']'.

    Used after parsing an element to advance past its bytes without
    re-parsing. Honors string/escape state so commas inside strings
    are ignored.
    """
    var i = start
    var depth = 0
    var in_string = False
    var escaped = False

    while i < n:
        var c = bytes[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == UInt8(ord("\\")) and in_string:
            escaped = True
            i += 1
            continue
        if c == UInt8(ord('"')):
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if c == UInt8(ord("[")) or c == UInt8(ord("{")):
            depth += 1
        elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
            if depth == 0:
                return i
            depth -= 1
        elif c == UInt8(ord(",")) and depth == 0:
            return i
        i += 1

    return i


def _skip_to_object_separator(bytes: Span[UInt8, _], start: Int, n: Int) -> Int:
    """Same as `_skip_to_array_separator` but stops at ',' or '}'."""
    var i = start
    var depth = 0
    var in_string = False
    var escaped = False

    while i < n:
        var c = bytes[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == UInt8(ord("\\")) and in_string:
            escaped = True
            i += 1
            continue
        if c == UInt8(ord('"')):
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if c == UInt8(ord("[")) or c == UInt8(ord("{")):
            depth += 1
        elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
            if depth == 0:
                return i
            depth -= 1
        elif c == UInt8(ord(",")) and depth == 0:
            return i
        i += 1

    return i


# ---------------------------------------------------------------------------
# OwnedValue serialization
# ---------------------------------------------------------------------------


def _owned_to_json(o: OwnedValue) -> String:
    """Serialize an `OwnedValue` back into a JSON string."""
    if o.kind == 0:
        return "null"
    if o.kind == 1:
        return "true" if o.bool_val else "false"
    if o.kind == 2:
        return String(o.int_val)
    if o.kind == 3:
        return String(o.float_val)
    if o.kind == 4:
        return _escape_json_string(o.str_val)
    if o.kind == 5:
        var out = String("[")
        for i in range(len(o.array_val)):
            if i > 0:
                out += ","
            out += _owned_to_json(o.array_val[i])
        out += "]"
        return out^
    if o.kind == 6:
        var out = String("{")
        for i in range(len(o.object_keys)):
            if i > 0:
                out += ","
            out += _escape_json_string(o.object_keys[i])
            out += ":"
            out += _owned_to_json(o.object_values[i])
        out += "}"
        return out^
    return "null"


def _escape_json_string(s: String) -> String:
    """Render a Mojo string as a JSON string literal."""
    var out = String('"')
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord('"')):
            out += '\\"'
        elif c == UInt8(ord("\\")):
            out += "\\\\"
        elif c == UInt8(ord("\n")):
            out += "\\n"
        elif c == UInt8(ord("\r")):
            out += "\\r"
        elif c == UInt8(ord("\t")):
            out += "\\t"
        else:
            out += chr(Int(c))
    out += '"'
    return out^


# ---------------------------------------------------------------------------
# Value mutation through OwnedValue
# ---------------------------------------------------------------------------


def _materialize_for_write(v: Value) raises -> OwnedValue:
    """Convert a Value into an OwnedValue tree for in-place mutation.

    This is the entry point for the COW path: callers materialize the
    tree, mutate it, then call `_serialize_into_value` to fold the
    result back into a `Value` whose `_raw` reflects the change.
    """
    return _value_to_owned(v)


def _serialize_into_value(o: OwnedValue) -> Value:
    """Serialize an `OwnedValue` and produce a fresh `Value`.

    For arrays and objects, the returned Value has `_raw` set to the
    serialized JSON, plus its `_keys` / `_count` populated so the
    v0.1 read path keeps working.
    """
    if o.kind == 0:
        return Value(Null())
    if o.kind == 1:
        return Value(o.bool_val)
    if o.kind == 2:
        return Value(o.int_val)
    if o.kind == 3:
        return Value(o.float_val)
    if o.kind == 4:
        return Value(o.str_val)
    if o.kind == 5:
        var raw = _owned_to_json(o)
        var v = Value(Null())
        v._type = 5
        v._raw = raw
        v._count = len(o.array_val)
        return v^
    if o.kind == 6:
        var raw = _owned_to_json(o)
        var v = Value(Null())
        v._type = 6
        v._raw = raw
        v._count = len(o.object_keys)
        v._keys = o.object_keys.copy()
        return v^
    return Value(Null())


# ---------------------------------------------------------------------------
# OwnedValue navigation (used for nested mutation via JSON Pointer)
# ---------------------------------------------------------------------------


def _set_at_pointer(
    mut tree: OwnedValue,
    tokens: List[String],
    idx: Int,
    var new_val: OwnedValue,
) raises:
    """Recursively set the value at the path `tokens[idx:]` in `tree`."""
    if idx == len(tokens):
        # Replace the entire tree -- caller guarantees this branch is
        # only reached at the top level.
        tree = new_val^
        return

    var token = tokens[idx]

    if tree.kind == 6:  # object
        var key_pos = -1
        for i in range(len(tree.object_keys)):
            if tree.object_keys[i] == token:
                key_pos = i
                break
        if idx == len(tokens) - 1:
            if key_pos >= 0:
                tree.object_values[key_pos] = new_val^
            else:
                tree.object_keys.append(token)
                tree.object_values.append(new_val^)
        else:
            if key_pos < 0:
                raise Error("Path does not exist in object: /" + token)
            _set_at_pointer(
                tree.object_values[key_pos], tokens, idx + 1, new_val^
            )
        return

    if tree.kind == 5:  # array
        var index: Int
        try:
            index = atol(token)
        except:
            raise Error("Array index must be a number: " + token)
        if index < 0 or index >= len(tree.array_val):
            raise Error("Array index out of bounds: " + token)
        if idx == len(tokens) - 1:
            tree.array_val[index] = new_val^
        else:
            _set_at_pointer(tree.array_val[index], tokens, idx + 1, new_val^)
        return

    raise Error(
        "Cannot navigate into primitive value with pointer token: /" + token
    )
