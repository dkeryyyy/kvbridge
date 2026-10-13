#pragma once
// Tensor-core split-K attention kernel (bf16 QK / fp16 PV mma).
// Header-only; include after "kvbridge/attention_common.h". The kernel is
// defined inside namespace kvbridge and uses Shape/DeviceInput from there.
//
// Same Page32 INT8 KV layouts and same scratch/merge convention as the A
// kernel. Numerics: Q is bf16-exact (protocol sends bf16), K int8 -> bf16 and
// V int8 -> fp16 are exact; QK uses mma.m16n8k16.bf16.bf16.f32 and PV uses
// mma.m16n8k16.f16.f16.f32 with fp32 accumulation. Accumulation order differs
// from A's sequential FMA, so outputs are validated by the frozen tolerance
// rule (finite; normalized RMS and max-abs bounds), not byte equality.
//
// Numerics (exact wherever possible): K is staged as raw int8 -> bf16 (exact)
// and V as raw int8 -> fp16 (exact); the per-token kscale multiplies the
// scores at the S write and the per-token vscale multiplies the softmax
// weights (A's wscale convention), so no scale ever rides through a reduced-
// precision staging path. P is staged in fp16 as w * vscale, so
// |w * vscale| must be <= 65504: weights are in [0, 1] and the engine's
// vscales are model-derived and far below this bound (the small-pool fixture
// restores in-domain scales for the B4 comparison). QK uses bf16 with fp32
// accumulation.
//
// Structure per block (kv_head, split of up to 256 tokens, 8 rows x 6 grouped
// heads = 48 head-rows, 16 warps of 32 lanes):
//   1. Q_s[48][kQStride] bf16 staged once, pairwise packed u32 stores.
//   2. QK: one-page token chunks. K staging is software-pipelined: raw u32
//      units for the next chunk are loaded into registers while the current
//      chunk's QK runs, then converted/packed (raw, exact) into
//      K_s[32][kKStride] at the loop head. Fragments via ldmatrix.x4 (A from
//      Q_s) and ldmatrix.x2 (B from K_s, row-major N x K); QK keeps two
//      independent accumulation chains (even/odd k-step) summed before the
//      per-column kscale * s.scale multiply; scores to S_s fp32 with stride
//      kSStride.
//   3. Softmax per head-row with causal mask; exp in place; maxima/denoms.
//   4. PV: V staging is register-prefetched the same way (raw u32 units for
//      four slots of one dim) and packed (raw, exact) into
//      V_s[256 dims][kVStride2]; fragments via ldmatrix.x2 (B from V_s,
//      hoisted out of the m-loop); A fragments from S_s fp32 (holding
//      w * vscale) packed as fp16 hi/lo pairs (two mmas per k-tile, ~22-bit
//      effective weights) and reused across the warp's two n-tiles (16 warps
//      x 16 dims); PV accumulated in registers; scratch written in A's
//      [max, denom, numerator...] convention (merge unchanged).
//
// Padded row pitches: kQStride/kKStride = 264 bf16 (528 B rows keep every
// ldmatrix 16-byte row segment aligned and the eight rail rows on distinct
// banks), kSStride = 261 floats (worst-case 2-way bank pattern for the warp's
// (grp,tig) scalar accesses; unpadded 256 was 8-way), kVStride2 = 40 halves
// (80 B rows: 8-lane phases land on distinct banks for dim-major reads).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
#include <cstring>

#include "kvbridge/attention_common.h"

namespace kvbridge {

namespace mma_detail {
constexpr unsigned mThreads = 512;
constexpr unsigned mWarp = 32;
constexpr unsigned mWarps = mThreads / mWarp;
constexpr unsigned kRows = 8;
constexpr unsigned kGroup = 6;
constexpr unsigned kHeads = kRows * kGroup;  // 48
constexpr unsigned kTokens = kSplitTokens;   // 256
constexpr unsigned kChunk = 32;              // one page
constexpr unsigned kChunks = kTokens / kChunk;
constexpr unsigned kQStride = kTokens + 8;    // bf16 row pitch (528 B)
constexpr unsigned kKStride = kTokens + 8;    // bf16 row pitch (528 B)
constexpr unsigned kSStride = kTokens + 5;    // float row pitch
constexpr unsigned kVStride2 = kChunk + 8;    // half row pitch (80 B)

// Broadcast reductions with the same shuffle order as the A kernel.
__device__ __forceinline__ float wmax_b(float x) {
  for (int delta = 16; delta; delta >>= 1)
    x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, delta));
  return __shfl_sync(0xffffffffu, x, 0);
}
__device__ __forceinline__ float wsum_b(float x) {
  for (int delta = 16; delta; delta >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, delta);
  return __shfl_sync(0xffffffffu, x, 0);
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  uint32_t out;
  memcpy(&out, &v, 4);
  return out;
}
// P is staged as an fp16 hi/lo pair: the high part is the rounded fp16 value
// and the low part carries the residual, so the two PV mmas keep ~22 bits of
// effective weight precision. A single fp16 rounding of w * vscale measured
// ~2.5e-5 normalized error against the frozen float64 oracle bound (atol
// 3e-6); the split removes that deficit without touching smem.
__device__ __forceinline__ void pack_f16_hi_lo(float lo, float hi, uint32_t& packed_hi, uint32_t& packed_lo) {
  const __half h0 = __float2half_rn(lo), h1 = __float2half_rn(hi);
  const __half2 hv = __halves2half2(h0, h1);
  memcpy(&packed_hi, &hv, 4);
  const __half2 rv = __floats2half2_rn(lo - __half2float(h0), hi - __half2float(h1));
  memcpy(&packed_lo, &rv, 4);
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  uint32_t out;
  memcpy(&out, &v, 4);
  return out;
}
// ldmatrix fragment assemblies. Addresses are supplied per lane; for .x2 only
// lanes 0-15 are consumed (lanes 16-31 pass mirrored addresses for safety).
__device__ __forceinline__ void ldm_x4(uint32_t* r, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t* r, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}

struct Smem {
  __nv_bfloat16 q[kHeads * kQStride];            // 25.3 KB
  float s[kHeads * kSStride];                    // 50.1 KB
  union {
    __nv_bfloat16 k[kChunk * kKStride];          // 16.9 KB (one page of K)
    __half v[kTokens * kVStride2];               // 20.0 KB (one page of V, dim-major)
  } kv;
  float maxima[kHeads];
  float denom[kHeads];
};
constexpr size_t kSmemBytes = sizeof(Smem);
}  // namespace mma_detail

// One block per (kv_head, split); int8 KV only.
__global__ void __launch_bounds__(mma_detail::mThreads)
split_kernel_mma(Shape s, DeviceInput in, unsigned splits, float* scratch) {
  using namespace mma_detail;
  extern __shared__ char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const unsigned kv = blockIdx.x;
  const unsigned split = blockIdx.y;
  const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned grp = lane >> 2, tig = lane & 3;
  const unsigned start = s.begin + split * kSplitTokens;
  const unsigned count = start < s.end ? min(kTokens, s.end - start) : 0;
  const unsigned first_page = start / s.page_tokens;

  // ---- stage Q as bf16 (exact), pairwise packed stores ----
  for (unsigned i = tid; i < kHeads * kTokens / 2; i += mThreads) {
    const unsigned idx = 2 * i;
    const unsigned hr = idx / kTokens, d = idx % kTokens;
    const unsigned row = hr / kGroup, h = hr % kGroup;
    const float* base = in.q + static_cast<size_t>((row * s.q_heads + kv * kGroup + h)) * s.dim;
    const uint32_t packed = pack_bf16(base[d], base[d + 1]);
    memcpy(&sm.q[hr * kQStride + d], &packed, 4);
  }
  __syncthreads();

  // ---- QK: register-prefetched K staging, two mma accumulation chains ----
  unsigned pk[4];  // raw u32 K units (4 dims of one slot each) for the next chunk
  {
    const unsigned ct = min(kChunk, count);
    const int8_t* kb = nullptr;
    if (ct) {
      const unsigned page = in.page_table[first_page];
      kb = static_cast<const int8_t*>(in.k) +
           (static_cast<size_t>(page) * s.kv_heads + kv) * s.page_tokens * s.dim;
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      const unsigned u = tid + i * mThreads;
      const unsigned slot = u >> 6;  // 64 four-dim units per 256-dim row
      uint32_t v = 0u;
      if (ct && slot < ct)
        memcpy(&v, kb + static_cast<size_t>(slot) * s.dim + ((u & 63) << 2), 4);
      pk[i] = v;
    }
  }
  for (unsigned c = 0; c < kChunks; ++c) {
    const unsigned tok0 = c * kChunk;
    const unsigned chunk_tokens = tok0 < count ? min(kChunk, count - tok0) : 0;
    const unsigned page = chunk_tokens ? in.page_table[first_page + c] : 0u;
    // convert the prefetched raw units into packed bf16 (zero-filled tails)
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      const unsigned u = tid + i * mThreads;
      const unsigned slot = u >> 6;
      const unsigned col = (u & 63) << 2;
      uint32_t w0 = 0u, w1 = 0u;
      if (slot < chunk_tokens) {
        // Raw int8 -> bf16 is exact (|int8| <= 127). The per-token kscale is
        // NOT applied here: it multiplies the scores after the QK mma, so the
        // mma accumulates exact integer products and the scale stays in fp32
        // (the A kernel's fmaf(q, k*ks, dot) value, modulo fp32 order).
        const uint32_t u4 = pk[i];
        const float f0 = static_cast<float>(static_cast<int8_t>(u4 & 0xffu));
        const float f1 = static_cast<float>(static_cast<int8_t>((u4 >> 8) & 0xffu));
        const float f2 = static_cast<float>(static_cast<int8_t>((u4 >> 16) & 0xffu));
        const float f3 = static_cast<float>(static_cast<int8_t>((u4 >> 24) & 0xffu));
        w0 = pack_bf16(f0, f1);
        w1 = pack_bf16(f2, f3);
      }
      uint32_t* dst = reinterpret_cast<uint32_t*>(sm.kv.k + slot * kKStride + col);
      dst[0] = w0;
      dst[1] = w1;
    }
    __syncthreads();
    // prefetch raw K units for chunk c+1 while QK of chunk c runs
    if (c + 1 < kChunks) {
      const unsigned tokn = (c + 1) * kChunk;
      const unsigned ctn = tokn < count ? min(kChunk, count - tokn) : 0;
      const int8_t* kb = nullptr;
      if (ctn) {
        const unsigned pagen = in.page_table[first_page + c + 1];
        kb = static_cast<const int8_t*>(in.k) +
             (static_cast<size_t>(pagen) * s.kv_heads + kv) * s.page_tokens * s.dim;
      }
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned u = tid + i * mThreads;
        const unsigned slot = u >> 6;
        uint32_t v = 0u;
        if (ctn && slot < ctn)
          memcpy(&v, kb + static_cast<size_t>(slot) * s.dim + ((u & 63) << 2), 4);
        pk[i] = v;
      }
    }
    for (unsigned base = warp; base < 12; base += mWarps) {
      const unsigned mt = base % 3, nt = base / 3;
      float cfr[4] = {0.f, 0.f, 0.f, 0.f}, cfr2[4] = {0.f, 0.f, 0.f, 0.f};
      const unsigned g8 = lane >> 3, r8 = lane & 7;
      const unsigned rowA = mt * 16 + ((g8 & 1) << 3) + r8;
      const unsigned colAh = (g8 >> 1) << 3;
      const unsigned l16 = lane & 15;
      const unsigned rowB = nt * 8 + (l16 & 7);
      const unsigned colBh = (l16 >> 3) << 3;
#pragma unroll
      for (unsigned kt = 0; kt < 16; kt += 2) {
        // A (16x16 bf16) via ldmatrix.x4; matrix order [m0k0,m1k0,m0k1,m1k1]
        // matches the mma A operand registers a[0..3]. B (16x8 bf16) via
        // ldmatrix.x2 (no transpose) from K_s[slot][dim]: lanes 0-7 feed rows
        // nt*8+0..7 at dim k0, lanes 8-15 the same rows at k0+8. K_s rows are
        // the N dimension and columns are k (dim), so the plain row-major
        // fragment {M[g][2t], M[g][2t+1]} is exactly the mma B operand.
        uint32_t a[4], b[2], a2[4], b2[2];
        ldm_x4(a, sm.q + rowA * kQStride + kt * 16 + colAh);
        ldm_x2(b, sm.kv.k + rowB * kKStride + kt * 16 + colBh);
        mma_bf16(cfr, a, b);
        ldm_x4(a2, sm.q + rowA * kQStride + (kt + 1) * 16 + colAh);
        ldm_x2(b2, sm.kv.k + rowB * kKStride + (kt + 1) * 16 + colBh);
        mma_bf16(cfr2, a2, b2);
      }
      // Columns of this chunk all live in the chunk's page; the per-token
      // kscale is applied here together with the protocol score scale.
      const unsigned c0 = nt * 8 + 2 * tig;
      const size_t sidx = (static_cast<size_t>(page) * s.kv_heads + kv) * s.page_tokens + c0;
      const float ksc0 = in.k_scales[sidx] * s.scale;
      const float ksc1 = in.k_scales[sidx + 1] * s.scale;
      float* srow0 = sm.s + (mt * 16 + grp) * kSStride + tok0 + c0;
      float* srow1 = srow0 + 8 * kSStride;
      srow0[0] = (cfr[0] + cfr2[0]) * ksc0;
      srow0[1] = (cfr[1] + cfr2[1]) * ksc1;
      srow1[0] = (cfr[2] + cfr2[2]) * ksc0;
      srow1[1] = (cfr[3] + cfr2[3]) * ksc1;
    }
    __syncthreads();
  }

  // ---- softmax per head-row with causal mask; exp in place ----
  for (unsigned hr = warp; hr < kHeads; hr += mWarps) {
    const unsigned row = hr / kGroup;
    const int64_t qpos = in.q_positions[row];
    float local_max = -CUDART_INF_F;
    for (unsigned col = lane; col < kTokens; col += mWarp) {
      if (col < count && in.k_positions[start + col] <= qpos)
        local_max = fmaxf(local_max, sm.s[hr * kSStride + col]);
    }
    local_max = wmax_b(local_max);
    float local_sum = 0.0f;
    for (unsigned col = lane; col < kTokens; col += mWarp) {
      const bool valid = col < count && in.k_positions[start + col] <= qpos;
      // __expf: MUFU-pipe intrinsic, max relative error ~1e-5 for the
      // non-positive arguments here (S - row max), far inside the frozen
      // tolerance rule; the A kernel keeps expf (byte-exact reference).
      const float w = (valid && local_max != -CUDART_INF_F)
                          ? __expf(sm.s[hr * kSStride + col] - local_max)
                          : 0.0f;
      // V stays raw int8 (exact fp16); the per-token vscale rides on the
      // softmax weight exactly like A's wscale = w * vscale. The stored P is
      // fp16(w * vscale) and the denominator keeps the vscale-free sum.
      // The page lookup is only meaningful for col < count (tokens beyond the
      // committed range have no page); w is zero there, so any value works.
      const unsigned page_c = (col < count) ? in.page_table[first_page + (col >> 5)] : 0u;
      const float vsc = (col < count)
                            ? in.v_scales[(static_cast<size_t>(page_c) * s.kv_heads + kv) * s.page_tokens +
                                          (col & 31u)]
                            : 0.0f;
      sm.s[hr * kSStride + col] = w * vsc;
      local_sum += w;
    }
    local_sum = wsum_b(local_sum);
    if (lane == 0) {
      sm.maxima[hr] = local_max;
      sm.denom[hr] = local_sum;
    }
  }
  __syncthreads();

  // ---- PV: warp w owns 2 dim n-tiles = dims [w*16, w*16+16) ----
  float acc[2][3][4];
#pragma unroll
  for (unsigned nt = 0; nt < 2; ++nt)
#pragma unroll
    for (unsigned mt = 0; mt < 3; ++mt)
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) acc[nt][mt][i] = 0.0f;
  unsigned pv[4];  // raw u32 V units (four slots of one dim each) for the next chunk
  {
    const unsigned ct = min(kChunk, count);
    const int8_t* vb = nullptr;
    if (ct) {
      const unsigned page = in.page_table[first_page];
      vb = static_cast<const int8_t*>(in.v) +
           (static_cast<size_t>(page) * s.kv_heads + kv) * s.dim * s.page_tokens;
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      const unsigned u = tid + i * mThreads;
      const unsigned g = u & 7;   // 8 four-slot groups per page
      const unsigned d = u >> 3;
      uint32_t v = 0u;
      if (ct && 4 * g < ct)
        memcpy(&v, vb + static_cast<size_t>(d) * s.page_tokens + 4 * g, 4);
      pv[i] = v;
    }
  }
  for (unsigned c = 0; c < kChunks; ++c) {
    const unsigned tok0 = c * kChunk;
    const unsigned chunk_tokens = tok0 < count ? min(kChunk, count - tok0) : 0;
    // convert the prefetched raw units into packed fp16; V is dim-major in
    // smem (V_s[dim][slot]) so PV B fragments load without transpose.
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      const unsigned u = tid + i * mThreads;
      const unsigned g = u & 7;
      const unsigned d = u >> 3;
      const unsigned slot0 = 4 * g;
      if (slot0 + 3 < chunk_tokens) {
        // Raw int8 -> fp16 is exact (|int8| <= 127). The per-token vscale is
        // NOT applied here: it multiplies the softmax weights (A's
        // wscale = w * vscale convention), so PV accumulates exact int8
        // products of the staged P.
        const uint32_t u4 = pv[i];
        const float f0 = static_cast<float>(static_cast<int8_t>(u4 & 0xffu));
        const float f1 = static_cast<float>(static_cast<int8_t>((u4 >> 8) & 0xffu));
        const float f2 = static_cast<float>(static_cast<int8_t>((u4 >> 16) & 0xffu));
        const float f3 = static_cast<float>(static_cast<int8_t>((u4 >> 24) & 0xffu));
        const uint32_t w0 = pack_f16(f0, f1);
        const uint32_t w1 = pack_f16(f2, f3);
        uint32_t* dst = reinterpret_cast<uint32_t*>(sm.kv.v + d * kVStride2 + slot0);
        dst[0] = w0;
        dst[1] = w1;
      } else {
        for (unsigned k = 0; k < 4; ++k) {
          const unsigned slot = slot0 + k;
          const float value = slot < chunk_tokens
                                  ? static_cast<float>(
                                        static_cast<int8_t>((pv[i] >> (8 * k)) & 0xffu))
                                  : 0.0f;
          sm.kv.v[d * kVStride2 + slot] = __float2half_rn(value);
        }
      }
    }
    __syncthreads();
    // prefetch raw V units for chunk c+1 while PV of chunk c runs
    if (c + 1 < kChunks) {
      const unsigned tokn = (c + 1) * kChunk;
      const unsigned ctn = tokn < count ? min(kChunk, count - tokn) : 0;
      const int8_t* vb = nullptr;
      if (ctn) {
        const unsigned pagen = in.page_table[first_page + c + 1];
        vb = static_cast<const int8_t*>(in.v) +
             (static_cast<size_t>(pagen) * s.kv_heads + kv) * s.dim * s.page_tokens;
      }
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned u = tid + i * mThreads;
        const unsigned g = u & 7;
        const unsigned d = u >> 3;
        uint32_t v = 0u;
        if (ctn && 4 * g < ctn)
          memcpy(&v, vb + static_cast<size_t>(d) * s.page_tokens + 4 * g, 4);
        pv[i] = v;
      }
    }
    // A fragments depend only on (mt,kt); assemble once for both k-tiles and
    // reuse across the warp's two n-tiles. B fragments load per (nt,kt) via
    // ldmatrix.x2, hoisted out of the m-loop.
    uint32_t pa[2][3][4], pl[2][3][4];
#pragma unroll
    for (unsigned kt = 0; kt < 2; ++kt) {
      const unsigned t0 = kt * 16;
#pragma unroll
      for (unsigned mt = 0; mt < 3; ++mt) {
        const float* p0 = sm.s + (mt * 16 + grp) * kSStride + tok0 + t0;
        const float* p1 = p0 + 8 * kSStride;
        pack_f16_hi_lo(p0[2 * tig], p0[2 * tig + 1], pa[kt][mt][0], pl[kt][mt][0]);
        pack_f16_hi_lo(p1[2 * tig], p1[2 * tig + 1], pa[kt][mt][1], pl[kt][mt][1]);
        pack_f16_hi_lo(p0[2 * tig + 8], p0[2 * tig + 9], pa[kt][mt][2], pl[kt][mt][2]);
        pack_f16_hi_lo(p1[2 * tig + 8], p1[2 * tig + 9], pa[kt][mt][3], pl[kt][mt][3]);
      }
    }
#pragma unroll
    for (unsigned nt = 0; nt < 2; ++nt) {
      const unsigned ntile = warp * 2 + nt;
      const unsigned l16 = lane & 15;
      const unsigned vdim = ntile * 8 + (l16 & 7);
      const unsigned vslot = (l16 >> 3) << 3;
      uint32_t b0[2], b1[2];
      ldm_x2(b0, sm.kv.v + vdim * kVStride2 + vslot);
      ldm_x2(b1, sm.kv.v + vdim * kVStride2 + 16 + vslot);
#pragma unroll
      for (unsigned mt = 0; mt < 3; ++mt) {
        mma_f16(acc[nt][mt], pa[0][mt], b0);
        mma_f16(acc[nt][mt], pl[0][mt], b0);
        mma_f16(acc[nt][mt], pa[1][mt], b1);
        mma_f16(acc[nt][mt], pl[1][mt], b1);
      }
    }
    __syncthreads();
  }

  // ---- scratch in A's convention ----
  if (tid < kHeads) {
    const unsigned row = tid / kGroup, h = tid % kGroup;
    const unsigned rh = row * s.q_heads + kv * kGroup + h;
    const size_t output = (static_cast<size_t>(rh) * splits + split) * (s.dim + 2);
    scratch[output] = sm.maxima[tid];
    scratch[output + 1] = sm.denom[tid];
  }
  for (unsigned nt = 0; nt < 2; ++nt) {
    const unsigned ntile = warp * 2 + nt;
#pragma unroll
    for (unsigned mt = 0; mt < 3; ++mt) {
      const unsigned hr0 = mt * 16 + grp, hr1 = hr0 + 8;
      const unsigned dim = ntile * 8 + 2 * tig;
      const unsigned row0 = hr0 / kGroup, h0 = hr0 % kGroup;
      const unsigned row1 = hr1 / kGroup, h1 = hr1 % kGroup;
      const size_t o0 =
          (static_cast<size_t>((row0 * s.q_heads + kv * kGroup + h0)) * splits + split) * (s.dim + 2);
      const size_t o1 =
          (static_cast<size_t>((row1 * s.q_heads + kv * kGroup + h1)) * splits + split) * (s.dim + 2);
      scratch[o0 + 2 + dim] = acc[nt][mt][0];
      scratch[o0 + 2 + dim + 1] = acc[nt][mt][1];
      scratch[o1 + 2 + dim] = acc[nt][mt][2];
      scratch[o1 + 2 + dim + 1] = acc[nt][mt][3];
    }
  }
}

}  // namespace kvbridge
