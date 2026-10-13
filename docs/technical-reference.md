# Technical reference and historical results

This page contains the detailed results and API notes moved out of the project homepage. For an overview, start with the [README](../README.md). Build commands, resource estimates and test procedures are in [reproducibility](reproducibility.md).

## Historical findings

| Scope | Finding | Evidence boundary |
| --- | --- | --- |
| Earlier attention benchmark, v2-07 | At 65,536 tokens: SIMT 7.8141 ms, MMA 2.3951 ms, about 3.26× | Forward-sweep medians, synthetic fixture, RTX 4060 Laptop. This predates the shipped hi/lo kernel. [Raw record](../data/att-bench-65536-v2-07.json). |
| Integrated attention offload, service v10 | At 64k context: one selected attention layer retained 95.7% of baseline decode speed; five retained 54.1% | DFlash2 B8, 512 output tokens; comparison within each run. Local backing remained allocated. [Study](../docs/whole-layer-1to5-64k.zh.md). |
| Protocol client measurement | Client RTT median 4.098 ms; server window 3.227 ms; paired transport-plus-client residual 0.867 ms | The residual is not an isolated wire measurement. The roughly 6 ms difference from another client is a diagnostic lead, not a proven causal attribution. [Data](../data/blackbox-session-05-analysis.json). |
| Five complete decoder layers, including weights | CUDA 40.271872 ms versus Mac 4.081718 ms on the same short-history component snapshot | Two CUDA replays, not end-to-end generation. The bandwidth argument is conditional on this hardware, B8 and weight-traffic assumptions. [Study](../docs/segment-validation-five-layer.zh.md). |
| KV storage arithmetic | 2,080 bytes/token/attention layer for four KV heads of dimension 256 | INT8 K/V plus FP32 scales; excludes scratch, page tables, runtime and reserves. [Table](../data/kv-memory-table.csv). |

The studies motivate measuring compute, protocol overhead and actual memory release separately. They do not establish that weight offload is universally impossible, or that KV offload always improves speed.

## API and memory contract

The interface is in [attention_common.h](../include/kvbridge/attention_common.h). The example owns buffers and events; the library launchers do not allocate them for the caller.

For a validated `Shape s`, allocate at least:

| Output | Float elements |
| --- | ---: |
| `m` and `l`, each | `s.rows * s.q_heads` |
| `n` | `s.rows * s.q_heads * s.dim` |
| `scratch` | `kvbridge::scratch_elements(s)` |

For `(row, head, d)`, normalized output is `n[(row*q_heads+head)*dim+d] / l[row*q_heads+head]` when the denominator is positive. For an empty/fully masked row the raw convention is `m=-inf, l=0, n=0`; do not divide by zero. The example treats its normalized vector as zero. Page-table entries, capacities, positions and all device pointers must match the documented layouts.

The MMA path is restricted to INT8 KV, 8 query rows, 6 query heads per KV head, dimension 256, Page32, and a page-aligned `begin`. Queries are supplied as FP32 values but must be BF16-exact for the intended numerical comparison. Check the source for all shape and launch limits.

**There is no automatic fallback in `launch_attention_mma`.** It throws on an unsupported shape or failed shared-memory opt-in. `attention_mma_supported` checks geometry; it does not prove device capability, successful allocation or numerical qualification. A caller can explicitly choose `launch_attention` after validating the reference path's own limits.

## Numerical scope

INT8-to-BF16 K staging and INT8-to-FP16 V staging exactly represent all signed INT8 codes. QK scaling occurs in FP32 after the dot product. For PV, the softmax weight is multiplied by the V scale in FP32 and then represented by an FP16 hi/lo pair. This reduces rounding error; it is not exact arithmetic or a universal relative-error guarantee. Finite inputs, representable intermediate values and the documented shape domain remain prerequisites.

The source uses different accumulation orders from SIMT, so byte equality is not assumed. Three evidence scopes must stay separate:

1. The shipped v2-07 JSON compares concatenated **`n/m/l`**, with normalized RMS about `1.283e-5`. It does not validate `n/l` for the current hi/lo source.
2. The later integrated v10 report describes a nonzero-query FP64 qualification: a single-FP16-P version failed, and the hi/lo version passed. A [sanitized metric summary](../data/v10-oracle-summary.json) is shipped; the original raw output tensors and complete qualification harness are not.
3. The standalone example compares normalized **`n/l`** and has its own small oracle test. Its current GPU results are unverified. Its nonzero-query synthetic generator differs from the historical benchmark. Passing it would be a bounded fixture result, not full service or model qualification.

See [kernel design](../docs/kernel-design.md) and [reproducibility](../docs/reproducibility.md) for acceptance rules and required records.

## Documentation

| Document | Topic |
| --- | --- |
| [Reproducibility](../docs/reproducibility.md) | What can be checked from this package; validation commands and missing artifacts |
| [Provenance](../docs/provenance.md) | Code lineage, public data index, licenses and evidence limits |
| [Kernel design](../docs/kernel-design.md) | Layout, launch domain, arithmetic and memory ownership |
| [Attention optimization](../docs/attention-kernel-mma.zh.md) | Historical v2-03–v2-07 campaign and plateau |
| [Early MMA work](../docs/attention-kernel-mma-early.zh.md) | Layout errors, vector loads and fragment assembly |
| [Attention offload, 1–5 layers](../docs/whole-layer-1to5-64k.zh.md) | Integrated service results and retained backing |
| [Protocol timing](../docs/blackbox-timing-split.zh.md) | Client/server timing and attribution limits |
| [Five-layer segment](../docs/segment-validation-five-layer.zh.md) | Complete-decoder-layer component measurement and failure |
| [Segment feasibility](../docs/segment-feasibility-review.zh.md) | Conditional bandwidth estimate and tradeoffs |
| [Q8 linear kernels](../docs/linear-kernels-q8.zh.md) | Separate projection experiments, not included library code |

## Limitations and license

This library has no model loader, network protocol, paging authority, admission controller, backward pass or production integration. The historical safeguards are not included. A host application must provide allocation limits, ownership, synchronization, cancellation and cleanup. Long-context feasibility, physical memory savings and performance on other hardware require new evidence.

The package carries [MIT](../LICENSE). CUDA/toolchains and any separately obtained models retain their own terms. The [provenance note](../docs/provenance.md) states the source-review boundary; the license file is not a claim of ownership over third-party tools or research.
