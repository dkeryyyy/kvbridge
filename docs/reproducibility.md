# Reproducibility and validation status

Updated for the local v0.1 research preparation, 2026-10-14. **The current standalone package has no recorded successful CUDA-library build, GPU self-test or Compute Sanitizer run.** The integrated historical experiments are a separate evidence set. Host-only checks have passed as detailed in [VALIDATION.md](../VALIDATION.md). This document does not authorize running a model, connecting to another machine or bypassing an existing resource restriction.

## What can be reproduced from this package

| Item | Available | Missing or limited |
| --- | --- | --- |
| Inspect source and memory layouts | Headers, two CUDA translation units, example and CMake | No portability qualification beyond historical sm_89 context |
| Recalculate selected benchmark summaries | [64k v2-07 JSON](../data/att-bench-65536-v2-07.json) | Earlier kernel's record, not a current-source benchmark; original harness not included |
| Inspect historical FP64 qualification | [v10 oracle summary](../data/v10-oracle-summary.json) | Extracted metrics and source hashes; original output tensors/harness absent |
| Inspect attention-offload throughput | [matrix-02](../data/matrix-02-summary.json), [matrix-06](../data/matrix-06-summary.json) | Summaries only; model, engine, service and complete traces absent |
| Inspect protocol timing | [Operations](../data/blackbox-session-05-ops.json), [summary](../data/blackbox-session-05-analysis.json) | Original server diagnostic log and client harness not fully packaged |
| Recalculate KV storage | [CSV](../data/kv-memory-table.csv), [JSON](../data/kv-memory-table.json) | Arithmetic excludes runtime/scratch; no capacity admission proof |
| Run a new synthetic standalone test | Intended workflow below | Requires a suitable local CUDA environment and new validation |
| Reproduce full historical model experiments | Not available | No complete reproduction script or complete original evidence bundle |

The Chinese topic summaries preserve reported results and failures but are not executable reproductions. Internal-only commands were removed rather than presented as usable package commands. [Provenance](provenance.md) identifies the public evidence available for each claim.

Historical v2-07 p50/p95 use the original harness's nearest-rank rule, `sorted_samples[ceil(q*n)-1]` (p50 is element 15 for 32 samples, counting from zero). The release validator follows that rule and reproduces the approximately 3.262505× p50 ratio. The new example averages the two middle samples for an even-sized median; Python's `statistics.median` therefore need not reproduce the historical p50.

## Host-only checks

The shape/sizing helpers, CLI parser and FP64 fixture/reference helpers can be built without CUDA:

```sh
cmake -S . -B build-host -DKVB_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-host --config Release --parallel 2
ctest --test-dir build-host -C Release --output-on-failure
python3 tools/validate_release.py
```

Local preparation passed all three CTest targets: 98 shape, 15 CLI and 6,149 oracle-helper checks, with AppleClang `21.0.0.21000334`. The host build used `-Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer`; host sanitizer checks passed. The integrity validator uses Python 3.9+ standard-library modules. See [VALIDATION.md](../VALIDATION.md) for reproducible host commands and [CONTRIBUTING.md](../CONTRIBUTING.md) for intentional manifest updates.

These tests do not compile CUDA translation units, run kernels or validate GPU allocation/cleanup. Treat their result separately from the device checks below.

## Intended standalone workflow — execution unverified

Historical integrated platform: Windows, RTX 4060 Laptop (sm_89), CUDA 13.3.1. Other CUDA versions, GPUs and operating systems have not been qualified for this package. CMake, a compatible C++17 host compiler and CUDA must be supplied separately.

From the package root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release --parallel 2
```

For a single-configuration generator, use the following executable path. On a Windows multi-configuration generator substitute `build/Release/attention_bench.exe`.

```sh
./build/attention_bench --self-test
compute-sanitizer --tool memcheck ./build/attention_bench --self-test
compute-sanitizer --tool racecheck ./build/attention_bench --self-test
compute-sanitizer --tool synccheck ./build/attention_bench --self-test
./build/attention_bench --tokens 8192 --warmup 8 --iters 32
```

These are proposed commands, not executed evidence. The self-test has seven fixed cases: six at 256 tokens cover identity/reversed pages, partial range `[32,239)`, fully masked queries, empty range and a causal prefix; the seventh uses 513 tokens, three splits and a partial final page, with query positions on both sides of split boundaries so later splits can be entirely masked for an earlier query. Both kernels compare `n/l` against FP64 elementwise using `atol=3e-6`, `rtol=5e-5`, then check repeated raw `m/l/n` bitwise. The benchmark uses a new nonzero-query generator with all signed INT8 codes; it is not the old v2-07 fixture. CLI bounds are specified by [the example](../examples/attention_bench.cu); a nonzero exit is a failure. Record sanitizer version and limitations. A clean tiny fixture does not prove that every supported token count or shape is safe.

Only after small checks and fresh resource admission succeed, test a larger case:

```sh
./build/attention_bench --tokens 65536 --warmup 8 --iters 32
```

The benchmark alternates AB/BA order and emits per-iteration event samples. The events measure a launch span covering split plus merge; host context/preparation work between submissions may introduce GPU idle gaps, so this is not pure kernel-only time. It gates normalized output before timing and checks repeatability afterward.

Run sanitizers separately from performance measurements. Preserve failed runs and slow samples; a numerical failure invalidates a performance claim for that candidate. The example is a synthetic fixture, not a model-quality or full-service lifecycle test.

The public protocol export also supports `python3 tools/check_blackbox.py`: it recomputes all 14 rounded statistics from 130 retained duration rows, excluding the first five. The original request IDs and absolute timestamps were removed after joining; the raw client/service harness is still absent.

## Resource estimates and host responsibilities

For the intended 8-row, 24-query-head, 4-KV-head, dimension-256 configuration:

- INT8 K/V and their FP32 scales use **2,080 bytes/token**: 16.25 MiB at 8,192 tokens and 130 MiB at 65,536 tokens.
- With 256-token splits, one scratch allocation at 64k is `8*24*256*(256+2)*4 = 50,724,864` bytes (48.375 MiB). Two simultaneously live candidate scratch buffers double that allocation.
- Per candidate, `n` needs 196,608 bytes and `m/l` each need 768 bytes. Q, positions, page tables, host copies, comparison buffers, events, modules, runtime/context and allocator overhead are additional.

These are component arithmetic, **not peak process/device measurements or permission to allocate**. Inspect the current example's complete allocation plan before running. The example bounds its explicit device buffers to 256 MiB and asks for a further 256 MiB of free-memory headroom; this local check is not a full application/system admission controller. Query fresh capacity, reserve room for runtime and other work, and choose a small fixture first. Do not create memory pressure to test failure handling.

The package does not include the research system's admission controller, shared ledger or remote cleanup evidence. In a supervised deployment, all pre-existing budgets and cleanup restrictions still apply. Callers own stream completion, original allocation/context lifetime, cancellation and error cleanup; process exit alone is not a portable proof that every external resource has retired.

## What to record for a new result

Record source/manifest hash, device and compute capability, driver/CUDA/host compiler/CMake versions, build flags and architecture, exact command, exit status, allocation estimates and measured peaks. Save the fixture seed/distribution, shape, page mapping, causal positions, normalized-output definition, oracle and tolerance formulas, all errors, and sanitizer logs.

For timing, keep warmup count, repetition count, both candidate orders, event-time versus host-time definitions and all samples. Report conversion/preparation separately. Historical v2-07 timings and `n/m/l` errors must not be copied into a new `n/l` result table.
