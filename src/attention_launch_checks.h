#pragma once
#include "kvbridge/attention_common.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kvbridge::detail {
inline void validate_outputs(const Shape& s, const void* scratch,
                             const float* m, const float* l, const float* n) {
  validate_shape(s);
  if (!scratch || !m || !l || !n)
    throw std::invalid_argument("scratch, m, l and n must be non-null device buffers");
}
inline void validate_launch(const Shape& s, const DeviceInput& in, const void* scratch,
                            const float* m, const float* l, const float* n) {
  validate_outputs(s, scratch, m, l, n);
  if (!in.q || !in.q_positions || !in.k_positions || !in.page_table || !in.k || !in.v ||
      (s.int8 && (!in.k_scales || !in.v_scales)))
    throw std::invalid_argument("attention input device buffers must be non-null");
  if (s.int8 && s.dim % 4 == 0 && s.page_tokens % 4 == 0 &&
      ((reinterpret_cast<uintptr_t>(in.k) % 4) || (reinterpret_cast<uintptr_t>(in.v) % 4)))
    throw std::invalid_argument("vectorized INT8 K/V buffers must be 4-byte aligned");
}
inline void check_cuda(cudaError_t code, const char* operation) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
}  // namespace kvbridge::detail
