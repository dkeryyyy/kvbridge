# Kernel design and numerical boundaries

This describes the experimental source shipped in v0.1, not a new benchmark result. The current standalone CUDA build and GPU checks remain unverified. See [provenance](provenance.md) for its relationship to the historical service and [reproducibility](reproducibility.md) for tests.

## Shape, layout and ownership

The intended MMA geometry is `rows=8`, `q_heads/kv_heads=6`, `dim=256`, `page_tokens=32`, INT8 KV and page-aligned `begin`. The historical configuration uses 24 query heads and 4 KV heads. Do not treat the SIMT fallback code as a tested arbitrary-shape library; validate its limits separately.

- Q is `float[rows*q_heads*dim]`, with BF16-exact values in the intended domain.
- K is `[physical_page, kv_head, slot, dim]`; V is `[physical_page, kv_head, dim, slot]`.
- K and V each have one FP32 scale per `[physical_page, kv_head, slot]`.
- `page_table[logical_page]` selects a physical page. Absolute positions define causality: a key is visible when `k_position <= q_position`.
- The processed interval is `[begin,end)`, bounded by the supplied token and page capacities.

The caller owns allocations, streams and the device/context lifetime. Launchers enqueue work; successful return is not proof of completion. Check launch errors and synchronize at an appropriate boundary before reading results, reusing or freeing storage.

Each 256-token split writes `dim+2` FP32 values per `(row,head)`: maximum, denominator and numerator vector. `scratch_elements(s)` returns the number of floats required. Merge writes `m/l` of `rows*q_heads` floats each and **`n` of `rows*q_heads*dim` floats**. Normalize each numerator dimension by its row-head denominator when positive. An empty/fully masked row has `m=-inf, l=0, n=0`; the example uses a zero normalized vector instead of dividing by zero. Thus `m=-inf` is intentional in this case, not a nonfinite-data failure.

`attention_mma_supported` is a shape predicate, not a runtime certification. `launch_attention_mma` throws if the predicate fails or shared-memory opt-in fails; it does **not** automatically dispatch SIMT. A caller may explicitly select the reference after checking its requirements.

## SIMT reference

The shared-group path uses a block per query row, KV head and split. Grouped query heads reuse staged data. QK dots, softmax reductions and weighted V accumulation are FP32. The source retains a legacy path for larger query-head groups within the common validator's limits (`dim<=256`); this package has no general-shape GPU qualification results.

The reference is a useful comparator, not an exact real-arithmetic oracle. Exponent evaluation, FP32 rounding, reduction order and merge also affect its output.

## MMA candidate

A block covers one KV head and split, with 512 threads and 48 query-head rows. K is converted from signed INT8 to BF16 without losing its integer value; V is converted to FP16 likewise. QK uses BF16 MMA with FP32 accumulation, then applies per-token K scale and the score scale in FP32.

The PV operand is formed from the FP32 softmax weight multiplied by V scale. It is represented by an FP16 high part and an FP16 residual, consumed by two MMA operations with FP32 accumulation. **V scale therefore affects a reduced-precision hi/lo operand**, even though raw V staging is exact. Hi/lo improves accuracy but is not exact for all values; small residual underflow, large-value overflow and cancellation still need tests. Finite input alone does not establish representable intermediates.

The source uses `ldmatrix`, padded shared-memory pitches, vectorized INT8 staging, register prefetch and two QK accumulation chains. Those transformations alter arithmetic ordering and resource use. They are not evidence of a particular utilization level or speedup for the standalone package.

## Three different numerical records

| Record | Compared values | Meaning |
| --- | --- | --- |
| [Historical v2-07 benchmark](../data/att-bench-65536-v2-07.json) | Concatenated `n/m/l`; patterned KV fixture | RMS/max-absolute tolerance passed for that earlier version. Not a current hi/lo `n/l` result. |
| Later service-v10 report | Nonzero queries against an FP64 reference in an integrated qualification suite | The single-FP16-P candidate failed; hi/lo passed. [Metric summary](../data/v10-oracle-summary.json) is shipped, without raw output tensors or the full suite. |
| Current standalone example | Normalized `n/l`, plus a separate small FP64-oracle self-test | New validation scope; no successful build/GPU/sanitizer record yet. |

The example's MMA-versus-SIMT comparison uses:

```
finite AND normalized_rms <= 0.002
       AND max_abs <= max(1e-3, 0.005 * max_abs(reference))
```

The same numeric constants do not make `n/m/l` and `n/l` tests equivalent. The standalone oracle checks each `n/l` element with `abs(actual-reference) <= 3e-6 + 5e-5*abs(reference)` for both SIMT and MMA, plus bitwise repeatability of raw `m/l/n`. Its seven cases cover identity/reversed pages, a partial range, fully masked queries, an empty range and a causal prefix at 256 tokens, plus a 513-token case crossing three splits and a partial final page. The new nonzero-query generator uses all signed INT8 codes and is not the historical benchmark generator. Keep output definition, normalization denominator, query distribution, source identity and failure thresholds attached to every reported result. Do not tune a tolerance after seeing a failure.

## Performance and future work

The shipped JSON records a historical 64k forward median of 7.8141 ms for SIMT and 2.3951 ms for v2-07 MMA. The current hi/lo candidate must be timed again. The example times a launch span covering split plus merge; because host launcher preparation can occur between event submissions, that span may contain GPU idle gaps and is not a pure kernel-only metric. Pure stream-read probes omit conversion, exponentiation, reductions and merge, and may benefit from cache; they are not a measured upper bound for this attention operation.

Useful next work is a verified standalone build, adversarial small oracle fixtures, sanitizer checks, matched timing, then profiler-supported hypotheses. A preconverted KV sidecar changes memory requirements and page ownership; it is a separate design experiment, not a promised shortcut to stream-probe time.
