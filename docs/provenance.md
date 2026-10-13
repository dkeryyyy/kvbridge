# Source, evidence and license provenance

This note describes the local v0.1 research package as prepared on 2026-10-14. It separates source lineage from evidence of a successful build. [RELEASE-MANIFEST.json](../RELEASE-MANIFEST.json) identifies packaged files; a matching hash proves file identity, not correctness, authorship or permission to redistribute third-party material.

## Code lineage checked locally

Before the v0.1 safety/example revisions, local text comparison established:

| Packaged component | Local research ancestor | Extraction changes observed |
| --- | --- | --- |
| `include/kvbridge/partial_attention_mma.cuh` | service-v10 `native/partial_attention_mma.cuh`, SHA256 `b48ed324a2ebc4ccc9593ec04087079df69b9f171545fe0cbb6a74a1f8510db1` | Header comments, public include and explicit namespace wrapper; kernel body matched at this extraction boundary. It includes the later hi/lo PV change. |
| `src/attention_shared.cu` | service-v10 `native/partial_attention.cu`, SHA256 `f3e70962a19bf06510e28f239647bdcfc1a8b80ab82079c8d7d6c1163b752643` | Reference and merge extracted; integrated MMA auto-dispatch removed; public interface and launch organization changed. |
| `src/attention_mma.cu` and public header | Standalone extraction/adaptation of that internal interface and MMA launch path | Separate explicit launcher; unsupported shape/shared-memory setup failure throws. |
| Host shape/sizing helpers, example, tests, CMake and release documentation | Standalone package scaffolding and v0.1 preparation | Not the historical benchmark/service harness; requires independent validation. |

The ancestor files are not separately bundled as a runnable research checkout. The pre-revision packaged MMA header had SHA256 `76b42a874e26e48732a00c555a0b370e3394ec663a00aba27bc14f53bdd31c2e`; use the current manifest for current file identity. Any changes after the described extraction must be qualified as new source, not silently attributed to an old result.

## Public data index

[DATA-PROVENANCE.json](../data/DATA-PROVENANCE.json) records original file identities, transformations and which items are arithmetic rather than measurements. The blackbox export removes identifiers and absolute clocks, retaining relative durations joined before redaction.

| File | Origin and interpretation | What it does not establish |
| --- | --- | --- |
| [att-bench-65536-v2-07.json](../data/att-bench-65536-v2-07.json) | Historical v2-07, synthetic patterned INT8 Page32 KV, forward/reverse samples, timing and concatenated `n/m/l` checks | Current hi/lo kernel timing or normalized `n/l` accuracy |
| [v10-oracle-summary.json](../data/v10-oracle-summary.json) | Sanitized extraction of later integrated-service FP64 qualification metrics, tolerance formulas and source hashes; failed single-P and passed hi/lo records retained | A new experiment, current-package validation or the original raw output tensors |
| [matrix-02-summary.json](../data/matrix-02-summary.json) | Integrated 64k baseline and early 1–5-attention-layer matrix attempt; successful rows coexist with an overall failure field | A fully completed matrix or all original traces |
| [matrix-06-summary.json](../data/matrix-06-summary.json) | Separate baseline and 4/5-layer continuation | A paired reverse sweep or general model-quality result |
| [blackbox-session-05-ops.json](../data/blackbox-session-05-ops.json) | Per-operation client/service diagnostic fields and completion summary | A fresh deployment, complete server log or portable cleanup mechanism |
| [blackbox-session-05-analysis.json](../data/blackbox-session-05-analysis.json) | 130 joined operations, 125 steady operations, timing percentiles | Isolated wire latency or causal proof for another client's overhead |
| [kv-memory-table.csv](../data/kv-memory-table.csv), [JSON](../data/kv-memory-table.json) | Arithmetic for 4 KV heads, dim256, INT8 K/V and FP32 scales | Measured resident memory, admission, complete runtime footprint or a long-context acceptance |

Selected exports preserve numerical/timing results and failure states; identifying fields may be removed as documented by the release preparation. Historical fields are records, not instructions or reusable remote endpoints. The archive is incomplete by design: no model weights, full captures, service/engine binaries, complete protection state or complete original experiment harness are distributed.

The five-complete-layer and Q8-linear topic documents transcribe historical report values whose original raw records are **not included**. The later service-v10 FP64 qualification has a sanitized metric extraction here, but no raw output tensors or runnable original harness. Those summaries explain the research but cannot independently substantiate every historical claim without additional evidence. No current standalone result should cite them as its own validation.

## Third-party material and MIT boundary

The package contains an [MIT license declaration](../LICENSE) for KVBridge Offload Study software and documentation. Local inspection of the supplied CUDA files found no embedded third-party implementation attribution or separate third-party license header. That observation is a limited source review, **not proof that the entire historical chain is free of outside contributions**. The declaring maintainers must confirm rights to release the code and retain any notices required by actual incorporated sources; this note does not invent upstream ownership or relicense someone else's code.

CUDA APIs, CUDA headers, PTX instructions, NVIDIA compiler/runtime tools, CMake and host compilers are dependencies or interfaces, not vendored parts of this package. Obtain them under their own terms. Model names describe experimental context; model weights and inference-engine code are not distributed or relicensed here.

“FlashAttention-style” split/merge describes a numerical technique; it is not a claim that this package includes the FlashAttention implementation. The wider study discussed GemLite, TRT-LLM and other kernels as research references; no such library is a build dependency or supplied implementation in this package. If future contributions incorporate their code, record the exact upstream revision, copied files and applicable notices before distribution.

See [validation status](../VALIDATION.md) for completed host-only checks, [changelog](../CHANGELOG.md) for release edits and [reproducibility](reproducibility.md) for outstanding CUDA qualification.
