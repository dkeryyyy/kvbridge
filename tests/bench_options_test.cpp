#include "bench_options.h"
#include <iostream>
#include <vector>
#include <stdexcept>
using kvbridge_example::parse_options;
int main() {
  int checks = 0;
  auto check = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("option test failed"); };
  const char* defaults[] = {"bench"};
  check(parse_options(1, defaults).tokens == 65536);
  const char* valid[] = {"bench", "--tokens", "256", "--warmup", "0", "--iters", "256", "--self-test"};
  const auto a = parse_options(8, valid);
  check(a.tokens == 256 && a.warmup == 0 && a.iters == 256 && a.self_test);
  for (const auto& args : std::vector<std::vector<const char*>>{
      {"bench", "--tokens", "0"}, {"bench", "--tokens", "-256"}, {"bench", "--tokens", "257"},
      {"bench", "--tokens", "65792"}, {"bench", "--tokens", "999999999999999999"},
      {"bench", "--tokens", "256junk"}, {"bench", "--tokens", ""}, {"bench", "--tokens"},
      {"bench", "--iters", "0"}, {"bench", "--iters", "257"}, {"bench", "--warmup", "33"},
      {"bench", "--warmup", "-1"}, {"bench", "--unknown"}}) {
    bool rejected = false;
    try { (void)parse_options(int(args.size()), args.data()); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected);
  }
  std::cout << checks << " CLI checks passed\n";
}
