// Bounded synthetic example. No model weights or external services are loaded.
// Historical measurements in data/ were NOT produced by this extracted program.
#include <cuda_runtime.h>
#include "kvbridge/attention_common.h"
#include "attention_reference.h"
#include "bench_options.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool cleanup_failed = false;
void checked(cudaError_t e, const char* operation) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}
void cleanup(cudaError_t e, const char* operation) noexcept {
  if (e != cudaSuccess) {
    cleanup_failed = true;
    std::fprintf(stderr, "cleanup error (%s): %s\n", operation, cudaGetErrorString(e));
  }
}
template<class T> class Buffer {
 public:
  explicit Buffer(size_t count) : count_(count) {
    checked(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), "cudaMalloc");
  }
  explicit Buffer(const std::vector<T>& host) : Buffer(host.size()) { upload(host); }
  ~Buffer() { if (ptr_) cleanup(cudaFree(ptr_), "cudaFree"); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* get() const { return ptr_; }
  void upload(const std::vector<T>& host) {
    if (host.size() != count_) throw std::invalid_argument("buffer size mismatch");
    checked(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  std::vector<T> download() const {
    std::vector<T> host(count_);
    checked(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return host;
  }
 private:
  size_t count_;
  T* ptr_ = nullptr;
};
class Event {
 public:
  Event() { checked(cudaEventCreate(&e_), "cudaEventCreate"); }
  ~Event() { if (e_) cleanup(cudaEventDestroy(e_), "cudaEventDestroy"); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return e_; }
 private:
  cudaEvent_t e_ = nullptr;
};
struct Snapshot { std::vector<float> m, l, n; };
struct Output {
  Buffer<float> scratch, m, l, n;
  explicit Output(const kvbridge::Shape& s)
      : scratch(kvbridge::scratch_elements(s)), m(kvbridge::output_scalar_elements(s)),
        l(kvbridge::output_scalar_elements(s)), n(kvbridge::output_numerator_elements(s)) {}
  Snapshot read() const { return {m.download(), l.download(), n.download()}; }
};
size_t device_bytes(const kvbridge_example::Fixture& f) {
  const auto& s = f.shape;
  return (f.q.size() + f.ks.size() + f.vs.size()) * sizeof(float) +
         (f.qpos.size() + f.kpos.size()) * sizeof(int64_t) + f.table.size() * sizeof(uint32_t) +
         f.k.size() + f.v.size() + 2 * sizeof(float) *
           (kvbridge::scratch_elements(s) + 2 * kvbridge::output_scalar_elements(s) +
            kvbridge::output_numerator_elements(s));
}
void check_memory(const kvbridge_example::Fixture& f) {
  size_t available = 0, total = 0;
  checked(cudaMemGetInfo(&available, &total), "cudaMemGetInfo");
  const size_t need = device_bytes(f), reserve = size_t(256) * 1024 * 1024;
  std::printf("explicit device buffers: %.2f MiB; runtime/module overhead excluded\n", double(need) / (1024 * 1024));
  if (need > size_t(256) * 1024 * 1024 || available < need || available - need < reserve)
    throw std::runtime_error("insufficient free memory for bounded buffers plus 256 MiB headroom");
  // This is a local preflight, not a reservation, watchdog or multi-process guard.
}
struct DeviceFixture {
  Buffer<float> q, ks, vs;
  Buffer<int64_t> qpos, kpos;
  Buffer<uint32_t> table;
  Buffer<int8_t> k, v;
  explicit DeviceFixture(const kvbridge_example::Fixture& f)
      : q(f.q), ks(f.ks), vs(f.vs), qpos(f.qpos), kpos(f.kpos), table(f.table), k(f.k), v(f.v) {}
  kvbridge::DeviceInput input() const { return {q.get(), qpos.get(), kpos.get(), table.get(), k.get(), v.get(), ks.get(), vs.get()}; }
};
void launch(bool mma, const kvbridge::Shape& s, const DeviceFixture& f, Output& o) {
  if (mma) kvbridge::launch_attention_mma(s, f.input(), o.scratch.get(), o.m.get(), o.l.get(), o.n.get(), nullptr);
  else kvbridge::launch_attention(s, f.input(), o.scratch.get(), o.m.get(), o.l.get(), o.n.get(), nullptr);
  checked(cudaGetLastError(), "kernel launch");
}
bool normalized(const Snapshot& p, unsigned dim, std::vector<double>& out,
                std::vector<bool>& visible) {
  out.assign(p.n.size(), 0); visible.assign(p.l.size(), false);
  for (size_t rh = 0; rh < p.l.size(); ++rh) {
    if (p.l[rh] == 0) {
      if (!std::isinf(p.m[rh]) || p.m[rh] >= 0) return false;
      for (unsigned d = 0; d < dim; ++d) if (p.n[rh * dim + d] != 0) return false;
      continue;
    }
    if (!std::isfinite(p.l[rh]) || p.l[rh] < 0 || !std::isfinite(p.m[rh])) return false;
    visible[rh] = true;
    for (unsigned d = 0; d < dim; ++d) {
      const size_t i = rh * dim + d;
      if (!std::isfinite(p.n[i])) return false;
      out[i] = double(p.n[i]) / double(p.l[rh]);
    }
  }
  return true;
}
bool oracle_gate(const Snapshot& p, const kvbridge_example::Oracle& oracle, unsigned dim,
                 const char* variant) {
  std::vector<double> actual;
  std::vector<bool> visible;
  bool pass = normalized(p, dim, actual, visible);
  pass = pass && visible == oracle.visible;
  double worst = 0;
  if (pass) for (size_t i = 0; i < actual.size(); ++i) {
    const double err = std::abs(actual[i] - oracle.normalized[i]);
    worst = std::max(worst, err);
    pass = pass && err <= 3e-6 + 5e-5 * std::abs(oracle.normalized[i]);
  }
  std::printf("  %s float64 oracle: %s; max_abs(n/l)=%.9g; atol=3e-6 rtol=5e-5\n",
              variant, pass ? "PASS" : "FAIL", worst);
  return pass;
}
bool paired_gate(const Snapshot& a, const Snapshot& b, unsigned dim) {
  std::vector<double> x, y;
  std::vector<bool> vx, vy;
  bool pass = normalized(a, dim, x, vx) && normalized(b, dim, y, vy) && vx == vy;
  double err2 = 0, ref2 = 0, worst = 0, refmax = 0;
  if (pass) for (size_t i = 0; i < x.size(); ++i) {
    const double e = x[i] - y[i];
    err2 += e * e; ref2 += x[i] * x[i];
    worst = std::max(worst, std::abs(e)); refmax = std::max(refmax, std::abs(x[i]));
  }
  const double nrms = std::sqrt(err2 / a.n.size()) / std::max(std::sqrt(ref2 / a.n.size()), 1e-6);
  const double maxlimit = std::max(1e-3, .005 * refmax);
  pass = pass && std::isfinite(nrms) && nrms <= .002 && worst <= maxlimit;
  std::printf("paired n/l gate: %s; NRMS=%.9g (limit .002), max_abs=%.9g (limit %.9g)\n",
              pass ? "PASS" : "FAIL", nrms, worst, maxlimit);
  return pass;
}
bool same(const Snapshot& a, const Snapshot& b) {
  auto bytes_equal = [](const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
  };
  return bytes_equal(a.m, b.m) && bytes_equal(a.l, b.l) && bytes_equal(a.n, b.n);
}
int self_test() {
  for (int test = 0; test < 7; ++test) {
    auto f = kvbridge_example::make_fixture(test == 6 ? 513 : 256, test != 0);
    const char* name = "identity-pages";
    if (test == 1) name = "reversed-pages";
    if (test == 2) { name = "partial-range"; f.shape.begin = 32; f.shape.end = 239; }
    if (test == 3) { name = "fully-masked"; f.qpos.assign(8, -1); }
    if (test == 4) { name = "empty-range"; f.shape.begin = f.shape.end; }
    if (test == 5) { name = "causal-prefix"; for (unsigned r = 0; r < 8; ++r) f.qpos[r] = r * 17; }
    if (test == 6) {
      name = "three-splits-partial-last-page";
      f.qpos = {0, 33, 250, 255, 256, 510, 511, 512};  // includes fully masked later splits
    }
    std::printf("self-test: %s\n", name);
    const auto oracle = kvbridge_example::reference(f);
    check_memory(f);
    DeviceFixture device(f);
    Output a(f.shape), b(f.shape);
    launch(false, f.shape, device, a); launch(true, f.shape, device, b);
    checked(cudaDeviceSynchronize(), "self-test synchronize");
    const auto first_a = a.read(), first_b = b.read();
    const bool a_ok = oracle_gate(first_a, oracle, f.shape.dim, "SIMT");
    const bool b_ok = oracle_gate(first_b, oracle, f.shape.dim, "MMA");
    launch(true, f.shape, device, b); launch(false, f.shape, device, a);
    checked(cudaDeviceSynchronize(), "repeat synchronize");
    const bool stable = same(first_a, a.read()) && same(first_b, b.read());
    if (!a_ok || !b_ok || !stable) { std::fprintf(stderr, "self-test failed (repeat bitwise=%d)\n", int(stable)); return 1; }
  }
  std::puts("seven GPU/oracle cases passed; this is a synthetic subset, not model qualification");
  return 0;
}
double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  return x.size() % 2 ? x[x.size()/2] : (x[x.size()/2-1] + x[x.size()/2]) / 2;
}
int benchmark(const kvbridge_example::Options& options) {
  auto f = kvbridge_example::make_fixture(unsigned(options.tokens));
  check_memory(f);
  DeviceFixture device(f);
  Output a(f.shape), b(f.shape);
  launch(false, f.shape, device, a); launch(true, f.shape, device, b);
  checked(cudaDeviceSynchronize(), "qualification synchronize");
  const auto first_a = a.read(), first_b = b.read();
  if (!paired_gate(first_a, first_b, f.shape.dim)) return 1;  // no timings on failure
  for (int i = 0; i < options.warmup; ++i) {
    launch(i % 2 != 0, f.shape, device, i % 2 ? b : a);
    launch(i % 2 == 0, f.shape, device, i % 2 ? a : b);
  }
  checked(cudaDeviceSynchronize(), "warmup synchronize");
  Event start, stop;
  auto timed = [&](bool mma) {
    checked(cudaEventRecord(start.get()), "start event");
    launch(mma, f.shape, device, mma ? b : a);
    checked(cudaEventRecord(stop.get()), "stop event");
    checked(cudaEventSynchronize(stop.get()), "timing synchronize");
    float ms = 0;
    checked(cudaEventElapsedTime(&ms, start.get(), stop.get()), "elapsed time");
    return double(ms);
  };
  std::vector<double> simt, mma;
  std::puts("iteration,simt_ms,mma_ms,first");
  for (int i = 0; i < options.iters; ++i) {
    double ta = 0, tb = 0;
    if (i % 2) { tb = timed(true); ta = timed(false); }
    else { ta = timed(false); tb = timed(true); }
    simt.push_back(ta); mma.push_back(tb);
    std::printf("%d,%.9g,%.9g,%s\n", i, ta, tb, i % 2 ? "MMA" : "SIMT");
  }
  if (!same(first_a, a.read()) || !same(first_b, b.read())) {
    std::fprintf(stderr, "output changed during repeat/timing runs; results invalid\n"); return 1;
  }
  const double ta = median(simt), tb = median(mma);
  if (!(ta > 0 && tb > 0)) throw std::runtime_error("invalid nonpositive event timing");
  std::printf("tokens=%d warmup=%d iters=%d; alternating order; launch event span (split+merge; may include host preparation gaps)\n"
              "SIMT median %.6f ms; MMA median %.6f ms; speedup %.4fx\n", options.tokens,
              options.warmup, options.iters, ta, tb, ta/tb);
  return 0;
}
int run(const kvbridge_example::Options& o) {
  checked(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp prop{};
  checked(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::printf("device: %s sm_%d%d\n", prop.name, prop.major, prop.minor);
  checked(kvbridge::preload_attention_kernels(), "kernel preload and shared-memory opt-in");
  return o.self_test ? self_test() : benchmark(o);
}
}  // namespace
int main(int argc, char** argv) {
  int result = 2;
  try {
    const auto options = kvbridge_example::parse_options(argc, argv);  // before CUDA
    if (options.help) {
      std::puts("attention_bench [--tokens N] [--warmup W] [--iters I] [--self-test]\n"
                "tokens: multiples of 256 in [256,65536], default 65536\n"
                "warmup: [0,32], default 8; iters: [1,256], default 32\n"
                "self-test: seven fixed 256/513-token float64-oracle cases; no timing\n"
                "Device 0; synthetic INT8 KV; no external safety controller included.\n"
                "Exit: 0 passed, 1 numerical failure, 2 configuration/CUDA error, 3 cleanup error.");
      return 0;
    }
    result = run(options);  // buffers/events leave scope before cleanup status is checked
  } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); }
  return cleanup_failed ? 3 : result;
}
