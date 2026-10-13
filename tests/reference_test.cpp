#include "attention_reference.h"
#include <iostream>
#include <stdexcept>
int main() {
  int checks = 0;
  auto check = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("oracle test failed"); };
  // Uniform QK has an analytic answer: arithmetic mean of causal V values.
  auto f = kvbridge_example::make_fixture(64, true);
  std::fill(f.q.begin(), f.q.end(), 0);
  f.qpos.assign(8, 2);
  auto o = kvbridge_example::reference(f);
  for (unsigned h = 0; h < 24; ++h) for (unsigned d = 0; d < 256; ++d) {
    double sum = 0;
    for (unsigned t = 0; t <= 2; ++t)
      sum += double(int((t * 7 + (h / 6) * 13 + d * 5) % 256) - 128) * double((t * 3 + h / 6) % 5 + 1) / 512.0;
    check(std::abs(o.normalized[h * 256 + d] - sum / 3) < 1e-14);
  }
  auto identity = kvbridge_example::make_fixture(64);
  auto permuted = kvbridge_example::make_fixture(64, true);
  check(kvbridge_example::reference(identity).normalized == kvbridge_example::reference(permuted).normalized);
  auto before = kvbridge_example::reference(f);
  // Future poison must not affect the causal result; mutate only future tokens.
  for (unsigned t = 3; t < 64; ++t) for (unsigned h = 0; h < 4; ++h) {
    const size_t si = (size_t(f.table[t / 32]) * 4 + h) * 32 + t % 32;
    for (unsigned d = 0; d < 256; ++d) {
      f.k[si * 256 + d] = 127;
      f.v[((size_t(f.table[t / 32]) * 4 + h) * 256 + d) * 32 + t % 32] = -128;
    }
  }
  check(before.normalized == kvbridge_example::reference(f).normalized);
  f.shape.begin = f.shape.end;
  o = kvbridge_example::reference(f);
  check(std::none_of(o.visible.begin(), o.visible.end(), [](bool b) { return b; }));
  check(std::all_of(o.normalized.begin(), o.normalized.end(), [](double x) { return x == 0; }));
  f.shape.begin = 0; f.qpos.assign(8, -1);
  o = kvbridge_example::reference(f);
  check(std::none_of(o.visible.begin(), o.visible.end(), [](bool b) { return b; }));
  std::cout << checks << " oracle checks passed\n";
}
