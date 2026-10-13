#pragma once
#include <charconv>
#include <stdexcept>
#include <string>

namespace kvbridge_example {
struct Options {
  int tokens = 65536, warmup = 8, iters = 32;
  bool help = false, self_test = false;
};
inline Options parse_options(int argc, const char* const* argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    if (flag == "--help" || flag == "-h") { o.help = true; continue; }
    if (flag == "--self-test") { o.self_test = true; continue; }
    if (flag != "--tokens" && flag != "--warmup" && flag != "--iters")
      throw std::invalid_argument("unknown option: " + flag);
    if (++i == argc) throw std::invalid_argument("missing value for " + flag);
    const std::string text(argv[i]);
    int value = 0;
    const auto p = std::from_chars(text.data(), text.data() + text.size(), value);
    if (p.ec != std::errc{} || p.ptr != text.data() + text.size())
      throw std::invalid_argument("invalid integer for " + flag);
    if (flag == "--tokens") o.tokens = value;
    if (flag == "--warmup") o.warmup = value;
    if (flag == "--iters") o.iters = value;
  }
  if (o.tokens < 256 || o.tokens > 65536 || o.tokens % 256)
    throw std::invalid_argument("tokens must be a multiple of 256 in [256,65536]");
  if (o.warmup < 0 || o.warmup > 32)
    throw std::invalid_argument("warmup must be in [0,32]");
  if (o.iters < 1 || o.iters > 256)
    throw std::invalid_argument("iters must be in [1,256]");
  return o;
}
}  // namespace kvbridge_example
