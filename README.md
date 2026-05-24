# json

[![CI](https://github.com/ehsanmok/json/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/json/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/json/actions/workflows/docs.yaml/badge.svg)](https://ehsanmok.github.io/json/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**High-performance JSON for Mojo.** Pure-Mojo two-pass CPU parser, GPU-accelerated parsing on NVIDIA, AMD, and Apple Metal, tape-backed `Document` shared by every backend, reflection serde with zero boilerplate, RFC-compliant JSONPath, JSON Patch, and JSON Schema. The simdjson FFI shim is opt-in for the cases where you need it.

```mojo
from json import loads, dumps

var data = loads('{"name": "Alice", "scores": [95, 87, 92]}')
print(data["name"].string_value())     # Alice
print(data["scores"][0].int_value())   # 95
print(dumps(data, indent="  "))        # pretty print
```

## Why json

- **Pure Mojo, zero FFI on the hot path.** The default CPU parser is a 64-byte branchless SIMD scan (PSHUFB-style classifier with prefix-XOR escape tracking) that emits a packed `Document` tape. The simdjson FFI shim is opt-in via `target="cpu-simdjson"`.
- **One representation across CPU and GPU.** Every backend writes into the same tape-backed `Document`. `Value` is a stable index into that tape, so iteration is a tape walk rather than a re-parse, and nested mutation propagates through the parent (`doc["a"]["b"].set(...)` is observed by `doc`).
- **GPU that wins on big files.** AMD MI355X hits 13 GB/s (3.6x cuJSON) and NVIDIA B200 hits 8 GB/s (1.8x cuJSON) on the 804 MB `twitter_large_record.json`. Apple M3 Pro runs the lean Metal pipeline at 3.1 GB/s on the same file, no CPU fallback. [Numbers below.](#performance)
- **Reflection serde with no boilerplate.** `serialize_json(struct)` and `deserialize_json[T](json)` walk struct fields at compile time, no hand-written `to_json` / `from_json` needed. Custom traits (`JsonSerializable`, `JsonDeserializable`) override the default for one type without abandoning reflection for the rest.
- **Strict where it matters, lenient where it asks for it.** RFC 7159 by default; opt into comments, trailing commas, and a custom max-depth via `ParserConfig`.
- **Fuzzed.** Five mozz harnesses (parser, simdjson FFI, Value access, JSONPath, NDJSON) with a differential property that simdjson and the native parser must agree on canonical `dumps` output, plus an ASan harness over the FFI and tape boundaries.

## Install

```toml
[workspace]
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
json = { git = "https://github.com/ehsanmok/json.git", tag = "<latest-release>" }
```

```bash
pixi install
```

Requires [pixi](https://pixi.sh). Pin to a [released tag](https://github.com/ehsanmok/json/releases) for reproducible builds; track unreleased work via `branch = "main"` (breaking changes possible between tags).

`mojo` and `simdjson` install as transitive dependencies. The simdjson FFI wrapper builds on environment activation, no manual step.

For GPU acceleration: NVIDIA CUDA 7.0+, AMD ROCm 6+, or Apple Silicon. See [GPU compatibility](https://docs.modular.com/max/packages#gpu-compatibility).

## Quick start

The tour below walks the snippet at the top of this README from beginner to advanced. Each level adds one concept; everything compiles, and the runnable equivalents live under [`examples/`](./examples/) (every one is part of `pixi run tests`).

### Beginner: parse and serialize

```mojo
from json.prelude import *  # loads, dumps, load, dump, Value, Null, ParserConfig, SerializerConfig, ...

def main() raises:
    var data = loads('{"name": "Alice", "scores": [95, 87, 92]}')
    print(data["name"].string_value())     # Alice
    print(data["scores"][0].int_value())   # 95
    print(dumps(data, indent="  "))        # pretty print

    var config = load("config.json")
    var logs   = load[format="ndjson"]("events.ndjson")  # List[Value]
```

`json.prelude` re-exports the everyday surface (`loads`, `dumps`, `load`, `dump`, `Value`, `Null`, `ParserConfig`, `SerializerConfig`, the reflection serde shortcuts). Domain-specific surfaces (jsonpath, patch, schema, lazy, streaming, manual serde traits, simdjson FFI) stay in their own modules so an import block still documents which features a file actually uses. Full list: [`json/prelude.mojo`](./json/prelude.mojo).

### Intermediate: typed serde via reflection

Promote a hand-written `to_json` / `from_json` pair to a single struct definition. Compile-time reflection walks the fields once at compile time and emits the same code a hand-written serde would.

```mojo
from json import serialize_json, deserialize_json, try_deserialize_json

@fieldwise_init
struct Person(Defaultable, Movable):
    var name:   String
    var age:    Int
    var active: Bool

    def __init__(out self):
        self.name   = ""
        self.age    = 0
        self.active = False

def main() raises:
    var p = deserialize_json[Person]('{"name":"Alice","age":30,"active":true}')
    print(p.name, p.age, p.active)
    print(serialize_json[pretty=True](p))

    # Non-raising variant for input that may not parse.
    var maybe = try_deserialize_json[Person]('{"bad": json')
    if not maybe:
        print("invalid input")
```

Supported field types: `Int`, `Int64`, `Bool`, `Float64`, `Float32`, `String`, `List[T]`, `Optional[T]`, nested structs, `Value` (raw passthrough), `Dict[String, T]`, and nested `List` / `Optional` combinations. For full control on a single type, implement `JsonSerializable` / `JsonDeserializable`; the rest of the struct keeps using reflection. See [`docs/api.md`](./docs/api.md).

### Advanced: GPU parsing for big files

```mojo
from json import load

def main() raises:
    var data = load[target="gpu"]("twitter_large_record.json")
    print(data.array_count())
```

Dispatch is compile-time (`target="gpu"`); the runtime picks NVIDIA, AMD, or Apple Metal based on what's present. The Mojo native CPU parser is the right call below ~100 MB; GPU launch overhead dominates smaller files.

For NDJSON, lazy parsing, streaming, JSONPath, JSON Patch, and JSON Schema, see [`examples/`](./examples/) (one example per topic, indexed in [`examples/README.md`](./examples/README.md)).

## Performance

### GPU on `twitter_large_record.json` (804 MB)

| Platform | Throughput | vs cuJSON | Pipeline |
|---|---:|---|---|
| AMD MI355X | 13 GB/s | **3.6x** | single-shot |
| NVIDIA B200 | 8 GB/s | **1.8x** | single-shot |
| Apple M3 Pro | 3.1 GB/s | n/a | lean Metal |

*GPU is only beneficial for files >100 MB on discrete cards (NVIDIA, AMD).*

### CPU on Apple Silicon, M-series

`pixi run -e dev bench-cpu <file>` runs the simdjson C++ reference first, then the Mojo CPU path. Both use the same protocol: 3 warmup + 100 measured iterations, min-time-derived throughput. Two workloads:

- `parse_only`: `loads(...)` and peek the root tag. Measures parse cost in isolation.
- `parse_traverse`: parse and walk every leaf via the public API. The realistic workload for any code that actually consumes the document.

| File | Size | simdjson `parse_only` | mojo `parse_only` | simdjson `parse_traverse` | mojo `parse_traverse` |
|---|---|---|---|---|---|
| `twitter.json` | 616 KB | 0.235 ms / 2.68 GB/s | 0.54 ms / 1.17 GB/s | 0.236 ms / 2.67 GB/s | 1.12 ms / 0.57 GB/s |
| `citm_catalog.json` | 1.7 MB | 0.440 ms / 3.92 GB/s | 1.09 ms / 1.58 GB/s | 0.528 ms / 3.27 GB/s | 2.49 ms / 0.69 GB/s |

`parse_traverse` only adds a small constant on top of `parse_only` because every `Value` is a stable tape index, so traversal is a tape walk and not a re-parse. The remaining 2.3-2.5x to native simdjson on `parse_only` is algorithmic (no Eisel-Lemire float fast path, recursive emission rather than a flat tape walker, no AVX-512 64-byte chunks). Full breakdown in [`docs/performance.md`](./docs/performance.md).

```bash
# Optional: download cuJSON datasets for end-to-end GPU runs.
pixi run -e dev download-twitter-large

# CPU bench (simdjson C++ vs mojo)
pixi run -e dev bench-cpu                                            # twitter.json
pixi run -e dev bench-cpu benchmark/datasets/citm_catalog.json

# GPU bench (large files only)
pixi run bench-gpu benchmark/datasets/twitter_large_record.json

# Apple Metal sweep across multiple files
pixi run -e dev bench-gpu-apple
```

> **Apple Metal users on Xcode 26.x.** If `bench-gpu` errors with `Metal Compiler failed to compile metallib`, run `xcodebuild -downloadComponent MetalToolchain` once to register the toolchain. See [the Apple developer thread](https://developer.apple.com/forums/thread/802155) for context.

## Architecture

```
json.parser          loads / load: dispatch on target= and format=
json.serialize       dumps / dump
json.document        Tape-backed Document (single source of truth)
json.value           Value view over Document tape
json.cpu             Two-pass CPU parser (stage 1 SIMD scan, stage 2 tape emit) and simdjson FFI
json.gpu             GPU kernels (fused structural scan), stream compaction, tape adapter
json.reflection      Compile-time reflection serde
json.lazy            LazyValue (on-demand parsing of substrings)
json.streaming       Streaming parser for files larger than memory
json.patch           JSON Patch (RFC 6902), Merge Patch (RFC 7396)
json.jsonpath        JSONPath (RFC 9535)
json.schema          JSON Schema validation
json.ndjson          NDJSON
```

Every backend (Mojo native CPU, simdjson FFI, GPU) emits the same `Document` tape, so `Value` access, mutation, JSONPath, JSON Patch, and Schema validation see one DOM regardless of how the bytes arrived. Full request lifecycle and tape layout in [`docs/architecture.md`](./docs/architecture.md).

## Examples

```bash
pixi run examples                                  # every tier
pixi run example-parsing                           # one example by name
pixi run mojo -I . examples/basic/parsing.mojo     # raw mojo invoke
```

| Tier | File | What it covers |
|---|---|---|
| basic | [`parsing.mojo`](./examples/basic/parsing.mojo) | `loads` / `dumps`, scalars and nested types |
| basic | [`file_io.mojo`](./examples/basic/file_io.mojo) | `load` / `dump` for JSON files |
| basic | [`value_types.mojo`](./examples/basic/value_types.mojo) | `Value` API: `is_*`, `*_value`, `array_count`, `object_keys` |
| basic | [`error_handling.mojo`](./examples/basic/error_handling.mojo) | `try` / `except` patterns and recovery |
| intermediate | [`reflection_serde.mojo`](./examples/intermediate/reflection_serde.mojo) | Zero-boilerplate struct serde via reflection |
| intermediate | [`struct_serde.mojo`](./examples/intermediate/struct_serde.mojo) | Manual `Serializable` / `Deserializable` traits |
| intermediate | [`ndjson.mojo`](./examples/intermediate/ndjson.mojo) | NDJSON parsing and serialization |
| intermediate | [`jsonpath.mojo`](./examples/intermediate/jsonpath.mojo) | RFC 9535 JSONPath queries |
| intermediate | [`schema_validation.mojo`](./examples/intermediate/schema_validation.mojo) | JSON Schema validation |
| intermediate | [`json_patch.mojo`](./examples/intermediate/json_patch.mojo) | RFC 6902 JSON Patch and RFC 7396 Merge Patch |
| advanced | [`lazy_parsing.mojo`](./examples/advanced/lazy_parsing.mojo) | On-demand parsing for huge documents |
| advanced | [`gpu_parsing.mojo`](./examples/advanced/gpu_parsing.mojo) | GPU pipeline (NVIDIA, AMD, Apple Metal) |

The full guided tour is in [`examples/README.md`](./examples/README.md).

## Develop

```bash
git clone https://github.com/ehsanmok/json.git && cd json
pixi install                  # default env: tests, examples, library
pixi install -e dev           # adds mojodoc, pre-commit, dataset downloader
```

The project uses three pixi environments, layered:

| Env | Adds | What it unlocks |
|---|---|---|
| `default` | nothing | `tests-cpu`, `tests-gpu`, `tests-e2e`, `examples`, `format-check` |
| `dev` | mojodoc, pre-commit, gdown, sysroot pin, gxx | `format`, `docs`, `bench-cpu`, `bench-gpu`, `tests-asan`, `download-*` |
| `fuzz` | mozz on top of `dev` | `fuzz-loads`, `fuzz-simdjson`, `fuzz-value-access`, `fuzz-jsonpath`, `fuzz-ndjson`, `fuzz-all` |

Common tasks (run with `pixi run [-e <env>] <task>`):

| Task | Env | What it does |
|---|---|---|
| `tests` | default | Full unit + integration suite plus every example under [`examples/`](./examples/) |
| `tests-cpu` / `tests-gpu` / `tests-e2e` | default | Tier-specific suites |
| `tests-asan` | dev | LLVM AddressSanitizer over the FFI and tape boundaries (Linux) |
| `format-check` / `format` | default / dev | `mojo format` over `json`, `tests`, `benchmark/mojo` |
| `docs` / `docs-build` | dev | mojodoc-rendered package docstring |
| `bench-cpu` / `bench-gpu` / `bench-gpu-apple` | dev | Reproducible benches |
| `fuzz-all` | fuzz | Every harness back-to-back |

The full task list, including every per-example and per-fuzz target, is in [`pixi.toml`](./pixi.toml).

### Sanitizer harness

`pixi run -e dev tests-asan` AOT-builds the FFI- and lifetime-heavy slice of the suite with `--sanitize address` and runs each binary in isolation. The intent is to catch use-after-free, OOB, and lifetime regressions on the simdjson FFI shim, the tape-backed `Document` / `Value` view, and the COW mutation path before they hit production.

> **macOS note.** Mojo's bundled libasan looks for `__asan_version_mismatch_check_v8`; system clang on Apple Silicon ships `__asan_version_mismatch_check_apple_clang_1700`. The harness detects the mismatch and skips with a clear message, so a dev-box `pixi run tests-asan` exits 0. Linux CI runs the full harness end-to-end.

### Fuzz harnesses

Five mozz harnesses live under [`fuzz/`](./fuzz/):

- `fuzz-loads`: default Mojo two-pass parser, with a `parse -> dumps -> parse` idempotence property.
- `fuzz-simdjson`: simdjson FFI boundary, with a differential property that simdjson and the native Mojo parser must agree on canonical `dumps` output when both succeed.
- `fuzz-value-access`: `Value` access API and COW mutation.
- `fuzz-jsonpath`: RFC 9535 JSONPath engine.
- `fuzz-ndjson`: `loads[format='ndjson']` line splitter.

Crashes land under `.mozz_crashes/<harness>/`; copy regressions into `fuzz/corpus/<harness>/` to lock them in.

## Further reading

- [`docs/api.md`](./docs/api.md): full API reference.
- [`docs/architecture.md`](./docs/architecture.md): CPU and GPU backend design, tape layout, `Value` semantics.
- [`docs/performance.md`](./docs/performance.md): benchmark methodology, optimisation deep-dive, what's left to close the simdjson gap.
- [`benchmark/README.md`](./benchmark/README.md): reproducible benchmark setup.
- [`examples/README.md`](./examples/README.md): guided tour of basic, intermediate, advanced examples.

## License

[MIT](./LICENSE)
