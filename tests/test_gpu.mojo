# GPU loading tests
# Tests: loads[target="gpu"](...)
#
# On Apple Silicon, loads[target="gpu"] raises an Error in v0.2 unless
# the user opted into the legacy CPU fallback by compiling with
# `-D JSON_GPU_ALLOW_APPLE_FALLBACK=1`. This file therefore short-circuits
# every GPU-runtime test on Apple-without-fallback and runs only the
# error-path assertion plus the tape-adapter unit test (which does not
# require GPU runtime). Non-Apple targets and Apple+fallback builds run
# the full suite.

from std.sys import has_apple_gpu_accelerator, is_defined
from std.testing import assert_equal, assert_true, TestSuite

from json import loads, dumps, Value, Null
from json import serialize_json, deserialize_json
from json.gpu.tape_adapter import parse_gpu_to_value
from json.cpu.stage1_scalar import parse_structural_scalar
from json.types import JSONResult


# Compile-time predicate: GPU runtime is reachable from this build.
comptime GPU_RUNTIME_AVAILABLE = (
    not has_apple_gpu_accelerator()
    or is_defined["JSON_GPU_ALLOW_APPLE_FALLBACK"]()
)


# =============================================================================
# GPU loads Tests
# =============================================================================


def test_loads_gpu_simple_object() raises:
    """Test GPU loads with simple object."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('{"name": "Alice"}')
    assert_true(v.is_object(), "GPU should return object")


def test_loads_gpu_simple_array() raises:
    """Test GPU loads with simple array."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("[1, 2, 3, 4, 5]")
    assert_true(v.is_array(), "GPU should return array")


def test_loads_gpu_nested() raises:
    """Test GPU loads with nested structure."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('{"data": {"nested": [1, 2, 3]}}')
    assert_true(v.is_object(), "GPU should handle nested structures")


def test_loads_gpu_string() raises:
    """Test GPU loads with string."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('"hello world"')
    assert_true(v.is_string(), "GPU should return string")


def test_loads_gpu_number() raises:
    """Test GPU loads with number."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("12345")
    assert_true(v.is_int() or v.is_float(), "GPU should return number")


def test_loads_gpu_bool() raises:
    """Test GPU loads with boolean."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("true")
    assert_true(v.is_bool(), "GPU should return bool")


def test_loads_gpu_null() raises:
    """Test GPU loads with null."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("null")
    assert_true(v.is_null(), "GPU should return null")


def test_loads_gpu_bool_false() raises:
    """Test GPU loads with false."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("false")
    assert_true(v.is_bool(), "GPU should return bool")
    assert_equal(v.bool_value(), False)


def test_loads_gpu_negative_number() raises:
    """Test GPU loads with negative number."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("-42")
    assert_true(v.is_int() or v.is_float(), "GPU should return number")


def test_loads_gpu_float() raises:
    """Test GPU loads with float."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("3.14159")
    assert_true(v.is_float(), "GPU should return float")


def test_loads_gpu_empty_object() raises:
    """Test GPU loads with empty object."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("{}")
    assert_true(v.is_object(), "GPU should return empty object")


def test_loads_gpu_empty_array() raises:
    """Test GPU loads with empty array."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]("[]")
    assert_true(v.is_array(), "GPU should return empty array")


def test_loads_gpu_array_of_objects() raises:
    """Test GPU loads with array of objects."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('[{"a": 1}, {"b": 2}]')
    assert_true(v.is_array(), "GPU should return array")


def test_loads_gpu_deeply_nested() raises:
    """Test GPU loads with deeply nested structure."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var v = loads[target="gpu"]('{"a": {"b": {"c": {"d": 1}}}}')
    assert_true(v.is_object(), "GPU should handle deep nesting")


# =============================================================================
# CPU/GPU Equivalence Tests
# =============================================================================


def test_cpu_gpu_equivalence_object() raises:
    """Test CPU and GPU produce equivalent results for objects."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = '{"a": 1, "b": 2}'
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_object(), gpu_result.is_object())


def test_cpu_gpu_equivalence_array() raises:
    """Test CPU and GPU produce equivalent results for arrays."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = "[1, 2, 3, 4, 5]"
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_array(), gpu_result.is_array())


def test_cpu_gpu_equivalence_nested() raises:
    """Test CPU and GPU produce equivalent results for nested structures."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = '{"users": [{"name": "Alice"}, {"name": "Bob"}], "count": 2}'
    var cpu_result = loads[target="cpu"](json)
    var gpu_result = loads[target="gpu"](json)
    assert_equal(cpu_result.is_object(), gpu_result.is_object())


def test_gpu_dumps_roundtrip() raises:
    """Test GPU loads then dumps roundtrip."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json = '{"test": "value"}'
    var v = loads[target="gpu"](json)
    var output = dumps(v)
    var v2 = loads[target="cpu"](output)
    assert_true(v2.is_object(), "GPU roundtrip should produce valid object")


@fieldwise_init
struct _GPUPerson(Defaultable, Movable):
    var name: String
    var age: Int
    var active: Bool

    def __init__(out self):
        self.name = ""
        self.age = 0
        self.active = False


def test_gpu_reflection_roundtrip() raises:
    """Test reflection-based deserialize_json with GPU backend."""
    comptime if not GPU_RUNTIME_AVAILABLE:
        return
    var json_str = '{"name":"GPU Test","age":42,"active":true}'
    var person = deserialize_json[_GPUPerson, target="gpu"](json_str)
    assert_equal(person.name, "GPU Test")
    assert_equal(person.age, 42)
    assert_equal(person.active, True)

    var back = serialize_json(person)
    var rt = deserialize_json[_GPUPerson, target="gpu"](back)
    assert_equal(rt.name, "GPU Test")
    assert_equal(rt.age, 42)


# =============================================================================
# v0.2 Phase D additions: Apple-error path + tape-adapter direct test
# =============================================================================


def test_apple_silicon_gpu_raises_without_fallback() raises:
    """`loads[target='gpu']` raises on Apple Silicon unless the fallback
    flag is set at compile time. On non-Apple builds (or Apple+fallback)
    this test is a no-op.
    """
    comptime if (
        has_apple_gpu_accelerator()
        and not is_defined["JSON_GPU_ALLOW_APPLE_FALLBACK"]()
    ):
        var raised = False
        try:
            var _v = loads[target="gpu"]('{"x": 1}')
        except _:
            raised = True
        assert_true(
            raised,
            (
                "loads[target='gpu'] must raise on Apple without"
                " JSON_GPU_ALLOW_APPLE_FALLBACK"
            ),
        )


def test_tape_adapter_roundtrip() raises:
    """End-to-end test of `parse_gpu_to_value` without requiring the GPU.

    Builds a synthetic `JSONResult` from `stage1_scalar` (filtering quote
    positions out, which mirrors what the GPU kernel emits), feeds it
    through the tape adapter, and asserts the resulting Value structurally
    matches the canonical CPU parse.
    """
    var json = '{"users":[{"name":"Alice","age":30},{"name":"Bob","age":25}]}'

    var scalar_index = parse_structural_scalar(json)
    var bytes = json.as_bytes()

    # GPU emits {}[]:, only -- filter quotes out of the scalar oracle.
    var fake_gpu_result = JSONResult()
    for i in range(len(scalar_index.positions)):
        var p = Int(scalar_index.positions[i])
        var c = bytes[p]
        if c != UInt8(ord('"')):
            fake_gpu_result.structural.append(Int32(p))
    fake_gpu_result.file_size = len(bytes)

    var adapter_value = parse_gpu_to_value(json, fake_gpu_result^)
    var cpu_value = loads(json)

    assert_true(adapter_value.is_object(), "adapter Value should be object")
    assert_equal(
        adapter_value.is_object(),
        cpu_value.is_object(),
        "adapter and CPU agree on object-ness",
    )
    var users = adapter_value["users"]
    assert_true(users.is_array(), "adapter Value users[] is array")
    assert_equal(users.array_count(), 2, "adapter Value users has 2 elements")


def main() raises:
    print("=" * 60)
    print("test_gpu.mojo - GPU loads() tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
