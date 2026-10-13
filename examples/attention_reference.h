#pragma once
// Independent scalar float64 oracle for small synthetic INT8 KV fixtures.
// This is example/test code, not the CUDA kernel's split/merge algorithm.
#include "kvbridge/attention_shape.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace kvbridge_example {
struct Fixture {
  kvbridge::Shape shape;
  std::vector<float> q, ks, vs;
  std::vector<int64_t> qpos, kpos;
  std::vector<uint32_t> table;
  std::vector<int8_t> k, v;
};
inline Fixture make_fixture(unsigned tokens, bool permute = false) {
  Fixture f;
  f.shape = {8, 24, 4, 256, 32, (tokens + 31) / 32, tokens, 0, tokens, 0.0625f, true};
  kvbridge::validate_shape(f.shape);
  const auto& s = f.shape;
  f.q.resize(kvbridge::output_numerator_elements(s));
  f.qpos.resize(s.rows); f.kpos.resize(tokens); f.table.resize(s.pages);
  const size_t slots = size_t(s.pages) * s.kv_heads * s.page_tokens;
  f.k.resize(slots * s.dim); f.v.resize(slots * s.dim);
  f.ks.resize(slots, 1); f.vs.resize(slots, 1);
  for (unsigned p = 0; p < s.pages; ++p) f.table[p] = permute ? s.pages - 1 - p : p;
  for (size_t i = 0; i < f.q.size(); ++i)
    f.q[i] = float(int((i * 7) % 257) - 128) / 1024.0f;  // BF16-exact
  for (unsigned r = 0; r < s.rows; ++r) f.qpos[r] = int64_t(tokens) - s.rows + r;
  for (unsigned t = 0; t < tokens; ++t) {
    f.kpos[t] = t;
    const unsigned p = f.table[t / s.page_tokens], slot = t % s.page_tokens;
    for (unsigned h = 0; h < s.kv_heads; ++h) {
      const size_t si = (size_t(p) * s.kv_heads + h) * s.page_tokens + slot;
      f.ks[si] = float((t + h) % 7 + 1) / 512.0f;
      f.vs[si] = float((t * 3 + h) % 5 + 1) / 512.0f;
      for (unsigned d = 0; d < s.dim; ++d) {
        f.k[si * s.dim + d] = int8_t(int((t * 17 + h * 11 + d * 3) % 256) - 128);
        f.v[((size_t(p) * s.kv_heads + h) * s.dim + d) * s.page_tokens + slot] =
            int8_t(int((t * 7 + h * 13 + d * 5) % 256) - 128);
      }
    }
  }
  return f;
}
struct Oracle {
  std::vector<double> normalized;
  std::vector<bool> visible;
};
inline Oracle reference(const Fixture& f) {
  const auto& s = f.shape;
  kvbridge::validate_shape(s);
  Oracle o{std::vector<double>(kvbridge::output_numerator_elements(s), 0),
           std::vector<bool>(kvbridge::output_scalar_elements(s), false)};
  std::vector<double> logits(s.end - s.begin);
  const unsigned group = s.q_heads / s.kv_heads;
  for (unsigned r = 0; r < s.rows; ++r) for (unsigned h = 0; h < s.q_heads; ++h) {
    const size_t rh = size_t(r) * s.q_heads + h;
    const unsigned kv = h / group;
    double maximum = -std::numeric_limits<double>::infinity();
    for (unsigned t = s.begin; t < s.end; ++t) {
      double dot = -std::numeric_limits<double>::infinity();
      if (f.kpos[t] <= f.qpos[r]) {
        const size_t si = (size_t(f.table[t / s.page_tokens]) * s.kv_heads + kv) * s.page_tokens + t % s.page_tokens;
        dot = 0;
        for (unsigned d = 0; d < s.dim; ++d)
          dot += double(f.q[rh * s.dim + d]) * double(f.k[si * s.dim + d]) * double(f.ks[si]);
        dot *= double(s.scale);
        o.visible[rh] = true;
      }
      logits[t - s.begin] = dot;
      maximum = std::max(maximum, dot);
    }
    if (!o.visible[rh]) continue;
    double denominator = 0;
    for (unsigned t = s.begin; t < s.end; ++t) {
      const double weight = std::exp(logits[t - s.begin] - maximum);
      denominator += weight;
      const unsigned p = f.table[t / s.page_tokens], slot = t % s.page_tokens;
      const size_t si = (size_t(p) * s.kv_heads + kv) * s.page_tokens + slot;
      for (unsigned d = 0; d < s.dim; ++d)
        o.normalized[rh * s.dim + d] += weight * double(f.v[((size_t(p) * s.kv_heads + kv) * s.dim + d) * s.page_tokens + slot]) * double(f.vs[si]);
    }
    for (unsigned d = 0; d < s.dim; ++d) o.normalized[rh * s.dim + d] /= denominator;
  }
  return o;
}
}  // namespace kvbridge_example
