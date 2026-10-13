#include "kvbridge/attention_common.h"
#include "attention_launch_checks.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <stdexcept>


namespace kvbridge {
namespace {
constexpr unsigned kThreads = 256;
constexpr unsigned kWarps = kThreads / 32;
constexpr unsigned kMaxDim = 256;
constexpr unsigned kMaxGroup = 8;
__device__ float warp_sum(float x) {
  for (int delta = 16; delta; delta >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, delta);
  return x;
}
__device__ float warp_max(float x) {
  for (int delta = 16; delta; delta >>= 1)
    x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, delta));
  return x;
}
template <bool Q8>
__device__ float load_value(const void* p, size_t i, float scale) {
  if constexpr (Q8)
    return static_cast<float>(static_cast<const int8_t*>(p)[i]) * scale;
  else
    return static_cast<const float*>(p)[i];
}

// Shared-group kernel: one block covers one query row, one KV head and one
// bounded historical split. The group of (q_heads / kv_heads) query heads that
// share the KV head read each K/V element exactly once, and every thread
// accumulates complete dots (no warp reduction for QK). K and V are fetched
// with 4-wide loads where alignment and page structure allow.
template <bool Q8>
__global__ void split_kernel_shared(Shape s, DeviceInput in, unsigned splits,
                                    float* scratch) {
  extern __shared__ char smem_raw[];
  const unsigned group = s.q_heads / s.kv_heads;
  float* q_s = reinterpret_cast<float*>(smem_raw);
  float* w_s = q_s + static_cast<size_t>(group) * s.dim;
  float* wsc_s = w_s + static_cast<size_t>(group) * kSplitTokens;
  float* vsc_s = wsc_s + static_cast<size_t>(group) * kSplitTokens;
  unsigned* vbase_s = reinterpret_cast<unsigned*>(vsc_s + kSplitTokens);
  float* red_s = reinterpret_cast<float*>(vbase_s + kSplitTokens);
  float* maxima_s = red_s + kWarps;
  float* denom_s = maxima_s + group;

  const unsigned row = blockIdx.x / s.kv_heads;
  const unsigned kv = blockIdx.x % s.kv_heads;
  const unsigned split = blockIdx.y;
  const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned start = s.begin + split * kSplitTokens;
  const unsigned count = start < s.end ? min(kSplitTokens, s.end - start) : 0;
  const int64_t query_position = in.q_positions[row];

  for (unsigned idx = tid; idx < group * s.dim; idx += kThreads) {
    const unsigned head = idx / s.dim, d = idx % s.dim;
    q_s[idx] = in.q[static_cast<size_t>(row * s.q_heads + kv * group + head) * s.dim + d];
  }
  __syncthreads();


  // An unaligned begin could otherwise make a four-slot V load cross a page.
  const bool kv4 = (s.dim % 4u == 0u) && (s.page_tokens % 4u == 0u) && (s.begin % 4u == 0u);
  float dot[kMaxGroup];
#pragma unroll
  for (unsigned i = 0; i < kMaxGroup; ++i) dot[i] = 0.0f;

  const unsigned j = tid;
  bool visible = false;
  const void* kbase = nullptr;
  float kscale = 1.0f;
  if (j < count) {
    const unsigned token = start + j;
    visible = in.k_positions[token] <= query_position;
    if (visible) {
      const unsigned page = in.page_table[token / s.page_tokens];
      const unsigned slot = token % s.page_tokens;
      const size_t scale_idx = (static_cast<size_t>(page) * s.kv_heads + kv) * s.page_tokens + slot;
      kscale = Q8 ? in.k_scales[scale_idx] : 1.0f;
      vsc_s[j] = Q8 ? in.v_scales[scale_idx] : 1.0f;
      vbase_s[j] = static_cast<unsigned>((static_cast<size_t>(page) * s.kv_heads + kv) *
                                         s.dim * s.page_tokens + slot);
      kbase = Q8 ? static_cast<const void*>(static_cast<const int8_t*>(in.k) + scale_idx * s.dim)
                 : static_cast<const void*>(static_cast<const float*>(in.k) + scale_idx * s.dim);
    } else {
      vsc_s[j] = 1.0f;
      vbase_s[j] = 0;
    }
  } else {
    vsc_s[j] = 1.0f;
    vbase_s[j] = 0;
  }

  if (visible) {
    unsigned d = 0;
    if (kv4 && Q8) {
      const int8_t* kb = static_cast<const int8_t*>(kbase);
      for (; d + 4 <= s.dim; d += 4) {
        const unsigned packed = *reinterpret_cast<const unsigned*>(kb + d);
        const float f0 = static_cast<float>(static_cast<int8_t>(packed & 0xffu)) * kscale;
        const float f1 = static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xffu)) * kscale;
        const float f2 = static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xffu)) * kscale;
        const float f3 = static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xffu)) * kscale;
#pragma unroll
        for (unsigned i = 0; i < kMaxGroup; ++i) {
          if (i >= group) break;
          const float* qq = q_s + static_cast<size_t>(i) * s.dim + d;
          dot[i] = fmaf(qq[0], f0, fmaf(qq[1], f1, fmaf(qq[2], f2, fmaf(qq[3], f3, dot[i]))));
        }
      }
    }
    for (; d < s.dim; ++d) {
      const float kf = Q8 ? static_cast<float>(static_cast<const int8_t*>(kbase)[d]) * kscale
                          : static_cast<const float*>(kbase)[d];
#pragma unroll
      for (unsigned i = 0; i < kMaxGroup; ++i) {
        if (i >= group) break;
        dot[i] = fmaf(q_s[static_cast<size_t>(i) * s.dim + d], kf, dot[i]);
      }
    }
  }
#pragma unroll
  for (unsigned i = 0; i < kMaxGroup; ++i) {
    if (i >= group) break;
    w_s[static_cast<size_t>(i) * kSplitTokens + j] = visible ? dot[i] * s.scale : -CUDART_INF_F;
  }

  for (unsigned i = 0; i < group; ++i) {
    float* wi = w_s + static_cast<size_t>(i) * kSplitTokens;
    float x = warp_max(wi[tid]);
    if (!lane) red_s[warp] = x;
    __syncthreads();
    if (!warp) {
      x = warp_max(lane < kWarps ? red_s[lane] : -CUDART_INF_F);
      if (!lane) maxima_s[i] = x;
    }
    __syncthreads();
    const float m = maxima_s[i];
    const float w = (m == -CUDART_INF_F) ? 0.0f : expf(wi[tid] - m);
    wi[tid] = w;
    wsc_s[static_cast<size_t>(i) * kSplitTokens + tid] = w * vsc_s[tid];
    x = warp_sum(w);
    if (!lane) red_s[warp] = x;
    __syncthreads();
    if (!warp) {
      x = warp_sum(lane < kWarps ? red_s[lane] : 0.0f);
      if (!lane) denom_s[i] = x;
    }
    __syncthreads();
  }

  if (tid < s.dim) {
    float num[kMaxGroup];
#pragma unroll
    for (unsigned i = 0; i < kMaxGroup; ++i) num[i] = 0.0f;
    unsigned j2 = 0;
    if (kv4 && Q8) {
      const int8_t* vb = static_cast<const int8_t*>(in.v);
      for (; j2 + 4 <= count; j2 += 4) {
        const unsigned char* vp = reinterpret_cast<const unsigned char*>(vb + vbase_s[j2]) +
                                  static_cast<size_t>(tid) * s.page_tokens;
        const unsigned packed = *reinterpret_cast<const unsigned*>(vp);
        const float f0 = static_cast<float>(static_cast<int8_t>(packed & 0xffu));
        const float f1 = static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xffu));
        const float f2 = static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xffu));
        const float f3 = static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xffu));
#pragma unroll
        for (unsigned i = 0; i < kMaxGroup; ++i) {
          if (i >= group) break;
          const float* wi = wsc_s + static_cast<size_t>(i) * kSplitTokens + j2;
          num[i] = fmaf(wi[0], f0, fmaf(wi[1], f1, fmaf(wi[2], f2, fmaf(wi[3], f3, num[i]))));
        }
      }
    }
    for (; j2 < count; ++j2) {
      const size_t value_idx = static_cast<size_t>(vbase_s[j2]) + static_cast<size_t>(tid) * s.page_tokens;
      const float vf = load_value<Q8>(in.v, value_idx, 1.0f);
#pragma unroll
      for (unsigned i = 0; i < kMaxGroup; ++i) {
        if (i >= group) break;
        num[i] = fmaf(wsc_s[static_cast<size_t>(i) * kSplitTokens + j2], vf, num[i]);
      }
    }
#pragma unroll
    for (unsigned i = 0; i < kMaxGroup; ++i) {
      if (i >= group) break;
      const unsigned rh = row * s.q_heads + kv * group + i;
      const size_t output = (static_cast<size_t>(rh) * splits + split) * (s.dim + 2);
      scratch[output + 2 + tid] = num[i];
    }
  }
  for (unsigned i = 0; i < group; ++i) {
    const unsigned rh = row * s.q_heads + kv * group + i;
    const size_t output = (static_cast<size_t>(rh) * splits + split) * (s.dim + 2);
    if (!tid) {
      scratch[output] = maxima_s[i];
      scratch[output + 1] = denom_s[i];
    }
  }
}

// Legacy per-(row, head) kernel retained for metadata outside the shared-group
// envelope (group > 8). All launch paths validate dim <= 256 before submission.
template <bool Q8>
__global__ void split_kernel(Shape s, DeviceInput in, unsigned splits,
                             float* scratch) {
  __shared__ float query[256];
  __shared__ float weights[kSplitTokens];
  __shared__ float vsc[kSplitTokens];
  __shared__ float wscale[kSplitTokens];
  __shared__ unsigned vbase[kSplitTokens];
  __shared__ float reduction[kWarps];
  __shared__ float maximum, denominator;
  const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned rh = blockIdx.x, split = blockIdx.y;
  const unsigned row = rh / s.q_heads, head = rh % s.q_heads;
  const unsigned kv_head = head / (s.q_heads / s.kv_heads);
  const unsigned start = s.begin + split * kSplitTokens;
  const unsigned count = start < s.end ? min(kSplitTokens, s.end - start) : 0;
  const int64_t query_position = in.q_positions[row];
  if (tid < s.dim) query[tid] = in.q[static_cast<size_t>(rh) * s.dim + tid];
  __syncthreads();

  for (unsigned j = warp; j < kSplitTokens; j += kWarps) {
    const unsigned token = start + j;
    const bool visible = j < count && in.k_positions[token] <= query_position;
    float dot = 0.0f;
    if (visible) {
      const unsigned page = in.page_table[token / s.page_tokens];
      const unsigned slot = token % s.page_tokens;
      const size_t scale_idx = (static_cast<size_t>(page) * s.kv_heads + kv_head) * s.page_tokens + slot;
      const float scale = Q8 ? in.k_scales[scale_idx] : 1.0f;
      const size_t base = scale_idx * s.dim;
      if (!lane) {
        vsc[j] = Q8 ? in.v_scales[scale_idx] : 1.0f;
        vbase[j] = static_cast<unsigned>((static_cast<size_t>(page) * s.kv_heads + kv_head) *
                                             s.dim * s.page_tokens + slot);
      }
      for (unsigned d = lane; d < s.dim; d += 32)
        dot = fmaf(query[d], load_value<Q8>(in.k, base + d, scale), dot);
    } else if (!lane) {
      vsc[j] = 1.0f;
      vbase[j] = 0;
    }
    dot = warp_sum(dot);
    if (!lane) weights[j] = visible ? dot * s.scale : -CUDART_INF_F;
  }
  __syncthreads();
  float x = warp_max(weights[tid]);
  if (!lane) reduction[warp] = x;
  __syncthreads();
  if (!warp) {
    x = warp_max(lane < kWarps ? reduction[lane] : -CUDART_INF_F);
    if (!lane) maximum = x;
  }
  __syncthreads();
  const float weight = maximum == -CUDART_INF_F ? 0.0f : expf(weights[tid] - maximum);
  weights[tid] = weight;
  wscale[tid] = weight * vsc[tid];
  x = warp_sum(weight);
  if (!lane) reduction[warp] = x;
  __syncthreads();
  if (!warp) {
    x = warp_sum(lane < kWarps ? reduction[lane] : 0.0f);
    if (!lane) denominator = x;
  }
  __syncthreads();
  const size_t output = (static_cast<size_t>(rh) * splits + split) * (s.dim + 2);
  if (!tid) {
    scratch[output] = maximum;
    scratch[output + 1] = denominator;
  }
  for (unsigned d = warp; d < s.dim; d += kWarps) {
    float numerator = 0.0f;
    for (unsigned j = lane; j < count; j += 32) {
      const float w = wscale[j];
      if (w == 0.0f) continue;
      const size_t value_idx = static_cast<size_t>(vbase[j]) + static_cast<size_t>(d) * s.page_tokens;
      numerator = fmaf(w, load_value<Q8>(in.v, value_idx, 1.0f), numerator);
    }
    numerator = warp_sum(numerator);
    if (!lane) scratch[output + 2 + d] = numerator;
  }
}

__global__ void merge_kernel(Shape s, const float* scratch, unsigned splits,
                             float* m, float* l, float* n) {
  __shared__ float reduction[kWarps];
  __shared__ float maximum;
  const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned rh = blockIdx.x;
  const size_t base = static_cast<size_t>(rh) * splits * (s.dim + 2);
  float local_max = -CUDART_INF_F;
  for (unsigned part = tid; part < splits; part += kThreads)
    local_max = fmaxf(local_max, scratch[base + static_cast<size_t>(part) * (s.dim + 2)]);
  local_max = warp_max(local_max);
  if (!lane) reduction[warp] = local_max;
  __syncthreads();
  if (!warp) {
    local_max = warp_max(lane < kWarps ? reduction[lane] : -CUDART_INF_F);
    if (!lane) maximum = local_max;
  }
  __syncthreads();
  float local_sum = 0.0f;
  for (unsigned part = tid; part < splits; part += kThreads) {
    const size_t off = base + static_cast<size_t>(part) * (s.dim + 2);
    if (scratch[off + 1] > 0.0f)
      local_sum += expf(scratch[off] - maximum) * scratch[off + 1];
  }
  local_sum = warp_sum(local_sum);
  if (!lane) reduction[warp] = local_sum;
  __syncthreads();
  if (!warp) {
    local_sum = warp_sum(lane < kWarps ? reduction[lane] : 0.0f);
    if (!lane) { m[rh] = maximum; l[rh] = local_sum; }
  }
  if (tid < s.dim) {
    float numerator = 0.0f;
    for (unsigned part = 0; part < splits; ++part) {
      const size_t off = base + static_cast<size_t>(part) * (s.dim + 2);
      if (scratch[off + 1] > 0.0f)
        numerator = fmaf(expf(scratch[off] - maximum), scratch[off + 2 + tid], numerator);
    }
    n[static_cast<size_t>(rh) * s.dim + tid] = numerator;
  }
}

}  // namespace

void launch_attention(const Shape& s, const DeviceInput& in, float* scratch,
                      float* m, float* l, float* n, cudaStream_t stream) {
  detail::validate_launch(s, in, scratch, m, l, n);
  const unsigned splits = split_count(s);
  const unsigned group = s.q_heads / s.kv_heads;
  if (s.dim <= kMaxDim && group <= kMaxGroup && splits <= 65535) {
    const size_t smem = (static_cast<size_t>(group) * s.dim +
                         2 * static_cast<size_t>(group) * kSplitTokens +
                         2 * kSplitTokens) * sizeof(float) +
                        (kWarps + 2 * static_cast<size_t>(group)) * sizeof(float);
    const dim3 grid(s.rows * s.kv_heads, splits);
    if (s.int8)
      split_kernel_shared<true><<<grid, kThreads, smem, stream>>>(s, in, splits, scratch);
    else
      split_kernel_shared<false><<<grid, kThreads, smem, stream>>>(s, in, splits, scratch);
    detail::check_cuda(cudaGetLastError(), "reference split launch");
    merge_kernel<<<s.rows * s.q_heads, kThreads, 0, stream>>>(s, scratch, splits, m, l, n);
    detail::check_cuda(cudaGetLastError(), "reference merge launch");
    return;
  }
  if (splits > 65535) throw std::runtime_error("grouped schedule exceeds CUDA grid-y split limit");
  const dim3 grid(s.rows * s.q_heads, splits);
  if (s.int8) split_kernel<true><<<grid, kThreads, 0, stream>>>(s, in, splits, scratch);
  else split_kernel<false><<<grid, kThreads, 0, stream>>>(s, in, splits, scratch);
  detail::check_cuda(cudaGetLastError(), "reference legacy split launch");
  merge_kernel<<<s.rows * s.q_heads, kThreads, 0, stream>>>(s, scratch, splits, m, l, n);
  detail::check_cuda(cudaGetLastError(), "reference merge launch");
}

void merge_attention(const Shape& s, const float* scratch, unsigned splits,
                     float* m, float* l, float* n, cudaStream_t stream) {
  detail::validate_outputs(s, scratch, m, l, n);
  if (splits != split_count(s)) throw std::invalid_argument("merge split count does not match shape");
  merge_kernel<<<s.rows * s.q_heads, kThreads, 0, stream>>>(s, scratch, splits, m, l, n);
  detail::check_cuda(cudaGetLastError(), "attention merge launch");
}

cudaError_t preload_attention_reference_kernels() {
  cudaFuncAttributes a{};
  auto code = cudaFuncGetAttributes(&a, split_kernel_shared<true>); if (code != cudaSuccess) return code;
  code = cudaFuncGetAttributes(&a, split_kernel_shared<false>); if (code != cudaSuccess) return code;
  code = cudaFuncGetAttributes(&a, split_kernel<true>); if (code != cudaSuccess) return code;
  code = cudaFuncGetAttributes(&a, split_kernel<false>); if (code != cudaSuccess) return code;
  return cudaFuncGetAttributes(&a, merge_kernel);
}
cudaError_t preload_attention_kernels() {
  const auto code = preload_attention_reference_kernels();
  return code == cudaSuccess ? preload_attention_mma_kernels() : code;
}
}  // namespace kvbridge
