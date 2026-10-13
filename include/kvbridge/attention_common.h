#pragma once
// KVBridge attention library — public interface.
//
// Two implementations of the same split-K paged attention math over INT8 KV
// (GQA: q_heads / kv_heads grouped query heads share one KV head):
//   * launch_attention      — reference SIMT kernel (frozen baseline)
//   * launch_attention_mma  — tensor-core (bf16 QK / fp16 PV mma) kernel
// Both write the same scratch convention. Their accumulation orders differ;
// numerical agreement must be tested on the caller's data (see README).
//
// Memory formats (all device pointers):
//
//   q           float[rows * q_heads * dim]      row-major, fp32 (values should be bf16-exact)
//   q_positions int64[rows]                      absolute position of each query row
//   k_positions int64[tokens]                    absolute position of each KV token
//   page_table  uint32[pages]                    logical page index -> physical page id
//   k           int8[((page * kv_heads + kv) * page_tokens + slot) * dim + d]
//   v           int8[((page * kv_heads + kv) * dim + d) * page_tokens + slot]   (dim-major)
//   k_scales    float[(page * kv_heads + kv) * page_tokens + slot]
//   v_scales    float[same as k_scales]
//
// tokens is the logical backing length. The logical token interval [begin,end)
// is split into kSplitTokens-token splits. Each split produces one per-(row-head) partial
// [max, denominator, numerator] in scratch; merge_attention combines the
// splits flash-attention style into per-(row,head) m/l/n. The caller outputs
// out[row,head,dim] = n[row,head,dim] / l[row,head], only where l > 0.
// Empty or fully masked ranges return m=-inf, l=0 and n=0, not a normalized
// output. The caller decides how to represent those empty attention rows.
//
// Output allocation sizes (floats):
//   m: output_scalar_elements(shape)     = rows * q_heads
//   l: output_scalar_elements(shape)     = rows * q_heads
//   n: output_numerator_elements(shape)  = rows * q_heads * dim
//   scratch: scratch_elements(shape)
// All input/output/scratch buffers must be correctly sized, mutually
// non-overlapping, and valid in the current CUDA context until the caller
// synchronizes its stream. This library does not own or allocate buffers.
// Device page_table entries must be < pages; each physical KV page includes
// all page_tokens slots, including padding in a partially occupied last page.
// k_positions must be nondecreasing in logical token order. Q/K/V/scales must
// be finite; MMA also requires bf16-exact Q and abs(v_scales) <= 65504.
// The host shape checks cannot validate device buffer contents or lengths.
//
// scratch layout (floats):
//   scratch[((row * q_heads + head) * splits + split) * (dim + 2) + col]
//   with col 0 = split max, 1 = split denominator, 2..2+dim-1 = split numerator.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "kvbridge/attention_shape.h"

namespace kvbridge {

struct DeviceInput {
  const float* q;
  const int64_t* q_positions;
  const int64_t* k_positions;
  const uint32_t* page_table;
  const void* k;       // int8[] or (reference only) float[]
  const void* v;
  const float* k_scales;   // per (page, kv_head, slot); ignored for fp32 KV
  const float* v_scales;
};

// Eagerly loads SIMT + merge + MMA kernels and performs MMA shared-memory
// opt-in in the caller's current context. Returns the first CUDA failure;
// therefore this combined function requires an MMA-capable device. No global
// "already initialized" flag is shared across owners, contexts or devices.
cudaError_t preload_attention_kernels();
// Separate entry points for callers explicitly choosing one implementation.
cudaError_t preload_attention_reference_kernels();
cudaError_t preload_attention_mma_kernels();

// Reference SIMT implementation: valid Shape, dim <= 256, fp32 or int8 KV.
// It keeps the existing shared-group schedule for groups <=8 and legacy
// per-head schedule for larger groups; it never invokes the MMA path.
// Invalid host metadata/buffers throw std::invalid_argument before launch.
// CUDA launch/setup failures throw std::runtime_error. Launches are async;
// execution failures must be checked when the caller synchronizes the stream.
void launch_attention(const Shape&, const DeviceInput&, float* scratch,
                      float* m, float* l, float* n, cudaStream_t);

// Tensor-core implementation. Requires: int8 KV, rows == 8,
// q_heads/kv_heads == 6, dim == 256, page_tokens == 32, begin % page_tokens == 0,
// splits <= 65535, bf16-exact Q, finite inputs and abs(v_scales) <= 65504.
// Needs sm_80+ AND sufficient shared memory. Invalid/unsupported metadata
// throws std::invalid_argument; device/opt-in/launch failure throws
// std::runtime_error. There is no implicit fallback: explicitly dispatch with
// attention_mma_supported(shape) and choose reference when appropriate.
void launch_attention_mma(const Shape&, const DeviceInput&, float* scratch,
                          float* m, float* l, float* n, cudaStream_t);

// Merge split-K partials into the output sizes above. splits must equal
// split_count(shape); incorrect metadata/buffers throw before submission.
void merge_attention(const Shape&, const float* scratch, unsigned splits,
                     float* m, float* l, float* n, cudaStream_t);
}  // namespace kvbridge
