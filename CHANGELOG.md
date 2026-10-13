# Changelog

## 0.1.0 — research source release, 2026-10-14

Initial public package: paged INT8-KV attention kernels, a bounded synthetic CUDA
example, historical offload measurements, negative results, and offline checks.
This is a research snapshot with no API stability or production-support promise.

Release preparation fixes:

- Correct the example's numerator allocation to `rows * q_heads * dim` floats.
- Bound and validate command-line inputs before CUDA; release buffers and events
  on normal exit and exceptions; fail before timing when the numerical gate fails.
- Validate shapes and allocation sizes; state caller-owned device-data contracts,
  empty-row behavior, and explicit MMA dispatch without automatic fallback.
- Set the CUDA architecture default before enabling CUDA and add CPU-only builds.
- Prepare MMA shared memory for the current context instead of caching global state.
- Guard the SIMT vector path for ranges whose beginning is not four-token aligned.
- Add host tests and a seven-case GPU/float64-oracle self-test entry point, including
  multiple splits, causal masking, page permutation, and a partially occupied page.
- Separate historical versions, measurement scopes, numerical metrics, estimates,
  and inference. Add sanitized evidence, provenance and an integrity manifest.

CPU-only checks were run locally. The final standalone CUDA package has **not**
been compiled with NVCC or exercised on a GPU; its self-test and sanitizer commands
remain unverified. Historical service results do not establish those outcomes.
See [validation status](VALIDATION.md) and [reproduction scope](docs/reproducibility.md).
