#include "kvbridge/attention_shape.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
unsigned checks = 0;
void expect(bool condition, const char* label) {
  ++checks;
  if (!condition) throw std::runtime_error(label);
}
template <class Change>
void reject(const kvbridge::Shape& base, Change change, const char* label) {
  auto shape = base;
  change(shape);
  expect(!kvbridge::attention_shape_valid(shape), label);
  expect(kvbridge::attention_shape_error(shape) != nullptr, label);
  expect(!kvbridge::attention_mma_supported(shape), label);
  bool thrown = false;
  try { (void)kvbridge::scratch_elements(shape); }
  catch (const std::invalid_argument&) { thrown = true; }
  expect(thrown, label);
}
}  // namespace

int main() {
  try {
    const kvbridge::Shape base{8, 24, 4, 256, 32, 2048, 65536, 0, 65536, .0625f, true};
    expect(kvbridge::attention_shape_valid(base), "valid reference shape");
    expect(kvbridge::attention_mma_supported(base), "valid MMA geometry");
    expect(kvbridge::split_count(base) == 256, "64Ki split count");
    expect(kvbridge::output_scalar_elements(base) == 192, "m/l each has 192 floats");
    expect(kvbridge::output_numerator_elements(base) == 49152, "n has 49152 floats");
    expect(kvbridge::scratch_elements(base) == size_t(192) * 256 * 258, "scratch shape");

    reject(base, [](auto& s) { s.rows = 0; }, "zero rows");
    reject(base, [](auto& s) { s.q_heads = 0; }, "zero q heads");
    reject(base, [](auto& s) { s.kv_heads = 0; }, "zero kv heads never divides");
    reject(base, [](auto& s) { s.q_heads = 25; }, "non-divisible heads must not truncate to group6");
    reject(base, [](auto& s) { s.kv_heads = 48; }, "more kv than q heads");
    reject(base, [](auto& s) { s.dim = 0; }, "zero dim");
    reject(base, [](auto& s) { s.dim = 257; }, "dim beyond shared query/output storage");
    reject(base, [](auto& s) { s.pages = 0; }, "zero pages");
    reject(base, [](auto& s) { s.page_tokens = 0; }, "zero page tokens");
    reject(base, [](auto& s) { s.tokens = 0; s.end = 0; }, "zero backing tokens");
    reject(base, [](auto& s) { s.begin = 33; s.end = 32; }, "reversed interval");
    reject(base, [](auto& s) { s.end = 65537; }, "end past token storage");
    reject(base, [](auto& s) { --s.pages; }, "token storage past page capacity");
    reject(base, [](auto& s) { s.scale = std::numeric_limits<float>::infinity(); }, "infinite scale");
    reject(base, [](auto& s) { s.scale = std::numeric_limits<float>::quiet_NaN(); }, "NaN scale");
    reject(base, [](auto& s) { s.rows = std::numeric_limits<uint32_t>::max(); }, "grid x product overflow");
    reject(base, [](auto& s) { s.pages = std::numeric_limits<uint32_t>::max(); }, "physical index overflow");
    reject(base, [](auto& s) { s.kv_heads = s.q_heads = 0xfffffffeu; }, "head/storage products bounded");

    auto s = base;
    s.begin = s.end = 32;
    expect(kvbridge::attention_shape_valid(s), "empty interval valid");
    expect(kvbridge::attention_mma_supported(s), "aligned empty interval MMA valid");
    expect(kvbridge::split_count(s) == 1, "empty interval has one zero-output split");
    expect(kvbridge::scratch_elements(s) == size_t(192) * 258, "empty scratch size");
    s.begin = 32; s.end = 33;
    expect(kvbridge::attention_mma_supported(s), "partial page supports aligned begin");
    expect(kvbridge::split_count(s) == 1, "one-token split");
    s.end = 289;
    expect(kvbridge::split_count(s) == 2, "257-token interval splits twice");
    s.begin = 31;
    expect(kvbridge::attention_shape_valid(s) && !kvbridge::attention_mma_supported(s), "unaligned reference only");
    s = base; s.int8 = false;
    expect(kvbridge::attention_shape_valid(s) && !kvbridge::attention_mma_supported(s), "fp32 reference only");
    s = base; s.rows = 1;
    expect(kvbridge::attention_shape_valid(s) && !kvbridge::attention_mma_supported(s), "one-row reference only");
    s = base; s.dim = 1;
    expect(kvbridge::attention_shape_valid(s) && !kvbridge::attention_mma_supported(s), "scalar dimension reference only");
    s = base; s.q_heads = 36;
    expect(kvbridge::attention_shape_valid(s) && !kvbridge::attention_mma_supported(s), "group9 legacy reference");
    s = base; s.tokens = 65535; s.end = 65535;
    expect(kvbridge::attention_mma_supported(s), "last page may be padded");
    s = {1, 1, 1, 1, 256, 65536, 16777216, 0, 16776960, 1.f, true};
    expect(kvbridge::split_count(s) == 65535, "largest grid y");
    reject(s, [](auto& x) { ++x.end; }, "grid y overflow");
    s = {1, 1, 1, 1, 1, 0xffffffffu, 0xffffffffu, 0xfffffffeu, 0xffffffffu, 1.f, true};
    expect(kvbridge::split_count(s) == 1, "near uint32 maximum no range addition overflow");
    s.begin = s.end;
    expect(kvbridge::split_count(s) == 1, "empty range at uint32 maximum");
    std::cout << "PASS " << checks << " host shape/API checks (no CUDA)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';
    return 1;
  }
}
