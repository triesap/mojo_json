# Changelog

## 0.2.0 — May 2026

### Architectural changes

- **Tape-backed `Document` + `Value` view.** `Value` is now a view over
  a packed tape stored in a `Document`. Every CPU and GPU pipeline
  emits the same shape, so downstream tooling (LazyValue, JSONPath,
  patch, schema, reflection) operates on a single representation.
- **Two-pass CPU parser.** `json.cpu.parse_cpu_native` is the canonical
  CPU entry point. Stage 1 (`stage1_scalar` oracle, `stage1` SIMD)
  builds a structural index; stage 2 walks the index and constructs
  the value tree. The scalar and SIMD stage 1 implementations are
  byte-identical, enforced by `tests/test_stage1_equivalence.mojo`.
  Default is the scalar variant; opt into SIMD with
  `parse_cpu_native[force_scalar=False]`.
- **Copy-on-write mutation.** `set` / `append` / `set_at` route through
  a new `OwnedValue` tree, so `doc["a"]["b"].set(...)` propagates
  through nested access. v0.1's raw-string mutation helpers
  (`_update_object_value`, `_add_object_key`, etc.) were deleted.
- **GPU tape adapter.** `json.gpu.tape_adapter.parse_gpu_to_value`
  bridges GPU `JSONResult` output into stage 2, eliminating the v0.1
  CPU re-scan in `_build_array` / `_build_object`.

### Breaking changes

- `loads[target='gpu']` on Apple Silicon now **raises an explicit
  Error** instead of silently falling back to CPU. Set
  `-D JSON_GPU_ALLOW_APPLE_FALLBACK=1` at compile time to opt back
  into the v0.1 silent fallback. This caught a class of perf
  surprises where users thought they were on the GPU path but were
  silently downgraded.
- **`load[format='ndjson'](path) -> List[Value]`** is the new typed
  NDJSON file API; it matches `loads[format='ndjson']`. The legacy
  extension-based auto-detection (`load("x.ndjson") -> Value`) still
  works for backward compatibility but wraps the records in an array
  Value -- new code should prefer the typed overload.
- **Removed**: deprecated `loads_with_config` (use `loads(s, config)`
  instead -- unchanged behavior, same overload set).
- **Removed**: `parse_mojo`, `parse_simd`, `MojoJSONParser`,
  `FastParser` (dead since v0.2-A wired everything through
  `parse_cpu_native`). `json.cpu.simdjson_ffi.SimdjsonFFI` is still
  exported for direct FFI access.

### Mutation-semantics fix (was a silent bug in v0.1)

In v0.1 `doc["users"][0].set("name", Value("Bob"))` would silently
drop the mutation: `__getitem__` returned a fresh `Value` and the
write rewrote that throwaway's raw JSON. v0.2 routes mutations
through `OwnedValue` with parent-chain materialization, matching
Python's `dict`/`list` semantics. See `tests/test_value_mutation.mojo`
for the full property suite.

### Internal hygiene

- Deleted `json/cpu/mojo_backend.mojo` (778 LOC dead code).
- Deleted `json/cpu/simd_backend.mojo` (703 LOC migrated to
  `stage1_scalar.mojo` in v0.2-C).
- Deleted `tests/test_mojo_backend.mojo` (covered by
  `test_backend_equivalence.mojo` and `test_stage1_equivalence.mojo`).
- `json/__init__.mojo` shrunk from 556 to ~85 lines; the long-form API
  guide moved to `docs/api.md`.

## 0.1.6 — Feb 2026

Final v0.1 release; baseline for the v0.2 redesign.

See `git log --until 2026-05-01` for the full v0.1.x history.
