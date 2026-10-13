// Tensor-core attention launcher. Contains no kernels itself: the kernel lives
// in include/kvbridge/partial_attention_mma.cuh; this TU adds current-context
// shared-memory opt-in and the shape gate, and reuses the reference library's
// merge kernel for the split combination.

#include <stdexcept>

#include "kvbridge/attention_common.h"
#include "kvbridge/partial_attention_mma.cuh"
#include "attention_launch_checks.h"

namespace kvbridge {
cudaError_t preload_attention_mma_kernels() {
  // Attributes belong to the active context/function. A process-wide bool
  // would skip setup after a context is destroyed or another device selected.
  // Repeat this inexpensive host setup instead of caching an unowned context.
  int device = 0, major = 0, shared_limit = 0;
  auto code = cudaGetDevice(&device); if (code != cudaSuccess) return code;
  code = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (code != cudaSuccess) return code;
  if (major < 8) return cudaErrorNotSupported;
  code = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (code != cudaSuccess) return code;
  cudaFuncAttributes attributes{};
  code = cudaFuncGetAttributes(&attributes, split_kernel_mma);
  if (code != cudaSuccess) return code;
  if (size_t(shared_limit) < mma_detail::kSmemBytes + attributes.sharedSizeBytes ||
      attributes.maxThreadsPerBlock < int(mma_detail::mThreads)) return cudaErrorNotSupported;
  return cudaFuncSetAttribute(split_kernel_mma, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(mma_detail::kSmemBytes));
}

void launch_attention_mma(const Shape& s, const DeviceInput& in, float* scratch,
                          float* m, float* l, float* n, cudaStream_t stream) {
  detail::validate_launch(s, in, scratch, m, l, n);
  if (!attention_mma_supported(s))
    throw std::invalid_argument(
        "launch_attention_mma requires int8 KV, rows8/group6/dim256/page32 and a page-aligned begin");
  detail::check_cuda(preload_attention_mma_kernels(), "MMA current-context preparation");
  const unsigned splits = split_count(s);
  const dim3 grid(s.kv_heads, splits);
  split_kernel_mma<<<grid, mma_detail::mThreads, mma_detail::kSmemBytes, stream>>>(s, in, splits, scratch);
  detail::check_cuda(cudaGetLastError(), "MMA split launch");
  merge_attention(s, scratch, splits, m, l, n, stream);
}
}  // namespace kvbridge
