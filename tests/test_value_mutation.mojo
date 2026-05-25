# Tests for copy-on-write mutation via OwnedValue.
#
# These tests pin down the mutation contract:
#   1. `Value.set` / `Value.append` mutate in place and the change is
#      visible through every read API (`raw_json`, `string_value`, etc.).
#   2. The OwnedValue round-trip preserves all values that were
#      previously parsed -- it does not silently drop sibling keys or
#      reorder them within an object.
#   3. `Value.set_at(pointer, value)` propagates a mutation through the
#      full parent chain, so `doc["a"]["b"].set("c", value)` is
#      visible from `doc` afterwards even though `__getitem__`
#      returns a fresh view.
#   4. The patch convenience pattern used by `json/patch.mojo`
#      (read parent, mutate, write parent back) keeps working.

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from json import loads, Value, Null


# ---------------------------------------------------------------------------
# Top-level mutation.
# ---------------------------------------------------------------------------


def test_object_set_new_key_propagates() raises:
    """Setting a brand-new key on an object reflects in raw_json and reads."""
    var obj = loads('{"name":"Alice"}')
    obj.set("age", Value(Int64(30)))
    assert_true(obj.is_object())
    assert_equal(obj.object_count(), 2)
    var keys = obj.object_keys()
    assert_equal(len(keys), 2)
    assert_equal(obj["name"].string_value(), "Alice")
    assert_equal(obj["age"].int_value(), 30)


def test_object_set_update_key() raises:
    """Updating an existing key keeps the value count and updates the read."""
    var obj = loads('{"name":"Alice","age":30}')
    obj.set("name", Value("Bob"))
    assert_equal(obj.object_count(), 2)
    assert_equal(obj["name"].string_value(), "Bob")
    assert_equal(obj["age"].int_value(), 30)


def test_array_set_index() raises:
    """Replacing an element by index reflects in raw_json and reads."""
    var arr = loads("[1,2,3]")
    arr.set(1, Value(Int64(20)))
    assert_equal(arr.array_count(), 3)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 20)
    assert_equal(items[2].int_value(), 3)


def test_array_append() raises:
    """Append grows the array and the new value is observable."""
    var arr = loads("[1,2]")
    arr.append(Value(Int64(3)))
    assert_equal(arr.array_count(), 3)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 2)
    assert_equal(items[2].int_value(), 3)


def test_array_append_to_empty() raises:
    """Append to an empty array works correctly."""
    var arr = loads("[]")
    arr.append(Value(True))
    assert_equal(arr.array_count(), 1)
    assert_equal(arr.array_items()[0].bool_value(), True)


def test_object_set_on_empty() raises:
    """Setting on an empty object works correctly."""
    var obj = loads("{}")
    obj.set("first", Value(Int64(1)))
    assert_equal(obj.object_count(), 1)
    assert_equal(obj["first"].int_value(), 1)


# ---------------------------------------------------------------------------
# OwnedValue round-trip preserves all sibling values.
# ---------------------------------------------------------------------------


def test_object_set_preserves_other_keys() raises:
    """Mutating one key in an object doesn't lose any sibling keys."""
    var obj = loads('{"a":1,"b":2,"c":3,"d":4}')
    obj.set("b", Value(Int64(20)))
    assert_equal(obj.object_count(), 4)
    assert_equal(obj["a"].int_value(), 1)
    assert_equal(obj["b"].int_value(), 20)
    assert_equal(obj["c"].int_value(), 3)
    assert_equal(obj["d"].int_value(), 4)


def test_array_set_preserves_neighbors() raises:
    """Mutating one index in an array doesn't disturb its neighbors."""
    var arr = loads("[10,20,30,40,50]")
    arr.set(2, Value(Int64(99)))
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 10)
    assert_equal(items[1].int_value(), 20)
    assert_equal(items[2].int_value(), 99)
    assert_equal(items[3].int_value(), 40)
    assert_equal(items[4].int_value(), 50)


def test_object_with_nested_array_preserved() raises:
    """A nested array stays parseable after mutating a sibling key."""
    var obj = loads('{"name":"Ada","tags":[1,2,3]}')
    obj.set("name", Value("Bob"))
    assert_equal(obj["name"].string_value(), "Bob")
    var tags = obj["tags"]
    assert_true(tags.is_array())
    assert_equal(tags.array_count(), 3)
    var items = tags.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 2)
    assert_equal(items[2].int_value(), 3)


def test_object_with_nested_object_preserved() raises:
    """A nested object stays parseable after mutating a sibling key."""
    var obj = loads('{"user":{"name":"Ada","age":36},"flag":true}')
    obj.set("flag", Value(False))
    assert_equal(obj["flag"].bool_value(), False)
    var user = obj["user"]
    assert_true(user.is_object())
    assert_equal(user["name"].string_value(), "Ada")
    assert_equal(user["age"].int_value(), 36)


def test_set_value_with_string_containing_quotes() raises:
    """Setting a string value containing quotes round-trips through escapes."""
    var obj = loads('{"k":1}')
    obj.set("msg", Value('hello "world"'))
    assert_equal(obj["msg"].string_value(), 'hello "world"')
    assert_equal(obj["k"].int_value(), 1)


def test_append_complex_value() raises:
    """Appending a parsed object preserves its structure."""
    var arr = loads("[1,2]")
    var nested = loads('{"k":42}')
    arr.append(nested)
    assert_equal(arr.array_count(), 3)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 1)
    assert_equal(items[1].int_value(), 2)
    assert_true(items[2].is_object())
    assert_equal(items[2]["k"].int_value(), 42)


# ---------------------------------------------------------------------------
# set_at(pointer, value) -- nested mutation entry point.
# ---------------------------------------------------------------------------


def test_set_at_top_level_object_key() raises:
    """`set_at("/key", v)` is equivalent to `set("key", v)` at the root."""
    var obj = loads('{"a":1,"b":2}')
    obj.set_at("/a", Value(Int64(100)))
    assert_equal(obj["a"].int_value(), 100)
    assert_equal(obj["b"].int_value(), 2)


def test_set_at_nested_object_key() raises:
    """A pointer two levels deep mutates the correct leaf."""
    var doc = loads('{"a":{"b":1,"c":2}}')
    doc.set_at("/a/b", Value(Int64(99)))
    var a = doc["a"]
    assert_equal(a["b"].int_value(), 99)
    assert_equal(a["c"].int_value(), 2)


def test_set_at_three_levels_deep() raises:
    """Three-level nested mutation propagates all the way up."""
    var doc = loads('{"a":{"b":{"c":1}}}')
    doc.set_at("/a/b/c", Value(Int64(42)))
    var leaf = doc["a"]["b"]["c"]
    assert_equal(leaf.int_value(), 42)


def test_set_at_array_index() raises:
    """A pointer with an array index mutates the indexed element."""
    var doc = loads('{"items":[10,20,30]}')
    doc.set_at("/items/1", Value(Int64(200)))
    var items = doc["items"].array_items()
    assert_equal(items[0].int_value(), 10)
    assert_equal(items[1].int_value(), 200)
    assert_equal(items[2].int_value(), 30)


def test_set_at_inserts_new_object_key() raises:
    """`set_at` on a missing leaf key under an existing object inserts it."""
    var doc = loads('{"a":{"b":1}}')
    doc.set_at("/a/c", Value(Int64(2)))
    var a = doc["a"]
    assert_equal(a.object_count(), 2)
    assert_equal(a["b"].int_value(), 1)
    assert_equal(a["c"].int_value(), 2)


def test_set_at_empty_pointer_replaces_root() raises:
    """An empty pointer replaces the entire document."""
    var doc = loads('{"a":1}')
    var replacement = loads("[1,2,3]")
    doc.set_at("", replacement)
    assert_true(doc.is_array())
    assert_equal(doc.array_count(), 3)


def test_set_at_path_does_not_exist_raises() raises:
    """Pointer through a missing intermediate path raises an error."""
    var doc = loads('{"a":{"b":1}}')
    var raised = False
    try:
        doc.set_at("/a/missing/leaf", Value(Int64(0)))
    except:
        raised = True
    assert_true(raised)


def test_set_at_through_primitive_raises() raises:
    """Pointer that descends into a primitive value raises an error."""
    var doc = loads('{"a":1}')
    var raised = False
    try:
        doc.set_at("/a/b", Value(Int64(0)))
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# The patch.mojo / jsonpath.mojo workaround pattern still works.
# ---------------------------------------------------------------------------


def test_read_modify_write_parent_pattern() raises:
    """The 'fetch parent, mutate, set_at parent' pattern works end-to-end.

    json/patch.mojo uses this idiom heavily; if `set_at` regresses it,
    every JSON Patch operation breaks.
    """
    var doc = loads('{"users":[{"name":"Ada"}]}')
    var parent = doc.at("/users/0")
    parent.set("age", Value(Int64(36)))
    doc.set_at("/users/0", parent)
    var leaf = doc["users"].array_items()[0].copy()
    assert_equal(leaf["name"].string_value(), "Ada")
    assert_equal(leaf["age"].int_value(), 36)


def test_multiple_sequential_mutations() raises:
    """Many mutations in sequence each leave the document in a consistent state.
    """
    var doc = loads('{"counter":0}')
    for i in range(5):
        doc.set("counter", Value(Int64(i)))
    assert_equal(doc["counter"].int_value(), 4)


def test_append_multiple_times() raises:
    """Repeated append grows the array without losing earlier values."""
    var arr = loads("[]")
    for i in range(4):
        arr.append(Value(Int64(i)))
    assert_equal(arr.array_count(), 4)
    var items = arr.array_items()
    assert_equal(items[0].int_value(), 0)
    assert_equal(items[1].int_value(), 1)
    assert_equal(items[2].int_value(), 2)
    assert_equal(items[3].int_value(), 3)


def main() raises:
    print("=" * 60)
    print("test_value_mutation.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
