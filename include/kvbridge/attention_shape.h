#pragma once
// Host-only shape validation and allocation sizes; no CUDA toolkit required.
#include <cstddef>
#include <cstdint>

namespace kvbridge {
constexpr unsigned kSplitTokens = 256;

struct Shape {
  uint32_t rows, q_heads, kv_heads, dim, page_tokens, pages, tokens, begin, end;
  float scale;  // finite score scale after QK, e.g. 1/sqrt(dim)
  bool int8;    // true: INT8 KV; false: fp32 KV (reference only)
};

// Positive rows/heads/dim/pages/page_tokens/tokens; q_heads must be divisible
// by kv_heads, dim <= 256, 0 <= begin <= end <= tokens <= pages*page_tokens.
// pages describes both the page-table length and allocated physical KV pages.
// Grid dimensions, 32-bit kernel offsets and host allocation sizes are bounded.
// Empty [begin,end) is legal even though the backing token count is positive.
// Validation checks metadata only, never dereferences device pointers or starts
// CUDA. A nullptr error means valid; validate_shape throws std::invalid_argument.
const char* attention_shape_error(const Shape&) noexcept;
bool attention_shape_valid(const Shape&) noexcept;
void validate_shape(const Shape&);

// Each sizing function validates first and returns element counts, not bytes.
// An empty range still has one split, producing m=-inf, l=0, n=0.
unsigned split_count(const Shape&);
size_t scratch_elements(const Shape&);           // floats in split scratch
size_t output_scalar_elements(const Shape&);     // floats in EACH of m and l
size_t output_numerator_elements(const Shape&);  // floats in n (also q)

// Shape predicate only; false for invalid metadata. Does not inspect the GPU,
// context, data values or buffers. Actual MMA launch also requires sm_80+ and
// sufficient dynamic shared memory in the caller's current CUDA context.
bool attention_mma_supported(const Shape&) noexcept;
}  // namespace kvbridge
