#include "kvbridge/attention_shape.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvbridge {
namespace {
uint64_t splits_unchecked(const Shape& s) noexcept {
  const uint64_t count = uint64_t(s.end) - s.begin;
  return count == 0 ? 1 : (count + kSplitTokens - 1) / kSplitTokens;
}
bool multiply_with_limit(uint64_t& value, uint64_t factor, uint64_t limit) noexcept {
  if (factor != 0 && value > limit / factor) return false;
  value *= factor;
  return true;
}
}  // namespace

const char* attention_shape_error(const Shape& s) noexcept {
  if (!s.rows || !s.q_heads || !s.kv_heads) return "rows and head counts must be positive";
  if (s.q_heads % s.kv_heads) return "q_heads must be divisible by kv_heads";
  if (!s.dim || s.dim > 256) return "dim must be in [1,256]";
  if (!s.page_tokens || !s.pages || !s.tokens) return "page_tokens, pages and tokens must be positive";
  if (s.begin > s.end || s.end > s.tokens) return "require begin <= end <= tokens";
  if (uint64_t(s.tokens) > uint64_t(s.pages) * s.page_tokens) return "tokens exceed allocated page capacity";
  if (!std::isfinite(s.scale)) return "score scale must be finite";
  if (splits_unchecked(s) > 65535) return "range exceeds CUDA grid-y split limit";
  // Kernel row/head products use unsigned, and CUDA grid.x is at most 2^31-1.
  if (uint64_t(s.rows) * s.q_heads > 0x7fffffffu) return "row/head product exceeds CUDA grid-x limit";
  uint64_t kv = s.pages;
  const uint64_t u32max = std::numeric_limits<uint32_t>::max();
  if (!multiply_with_limit(kv, s.kv_heads, u32max) ||
      !multiply_with_limit(kv, s.page_tokens, u32max) ||
      !multiply_with_limit(kv, s.dim, u32max)) return "physical KV extent exceeds 32-bit kernel offsets";
  const uint64_t max_float_elements = std::numeric_limits<size_t>::max() / sizeof(float);
  uint64_t elements = uint64_t(s.rows) * s.q_heads;
  if (!multiply_with_limit(elements, splits_unchecked(s), max_float_elements) ||
      !multiply_with_limit(elements, uint64_t(s.dim) + 2, max_float_elements))
    return "scratch byte size exceeds size_t";
  if (kv > max_float_elements ||
      uint64_t(s.tokens) > std::numeric_limits<size_t>::max() / sizeof(int64_t) ||
      uint64_t(s.pages) > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
    return "input byte size exceeds size_t";
  return nullptr;
}

bool attention_shape_valid(const Shape& s) noexcept { return attention_shape_error(s) == nullptr; }
void validate_shape(const Shape& s) {
  if (const char* error = attention_shape_error(s)) throw std::invalid_argument(error);
}
unsigned split_count(const Shape& s) {
  validate_shape(s);
  return static_cast<unsigned>(splits_unchecked(s));
}
size_t scratch_elements(const Shape& s) {
  validate_shape(s);
  return size_t(s.rows) * s.q_heads * size_t(splits_unchecked(s)) * (s.dim + 2);
}
size_t output_scalar_elements(const Shape& s) {
  validate_shape(s);
  return size_t(s.rows) * s.q_heads;
}
size_t output_numerator_elements(const Shape& s) {
  return output_scalar_elements(s) * s.dim;
}
bool attention_mma_supported(const Shape& s) noexcept {
  return attention_shape_valid(s) && s.int8 && s.rows == 8 && s.dim == 256 &&
         s.page_tokens == 32 && s.q_heads / s.kv_heads == 6 && s.begin % 32 == 0;
}
}  // namespace kvbridge
