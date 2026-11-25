// Inkling 2 Mini (inkling2_loop) fused Metal kernels.
// Buffers: 0 = W (mmapped GGUF, quantized weights), 1 = PB (small f32 params + bf16 matrices),
//          2 = A (f32 activation arena), 3 = KV (K/V cache), 4 = CS (f32 conv state), 5 = args.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define E      2304
#define NH     24
#define NKV    8
#define HD     128
#define NREL   16
#define EXT    1024
#define NEXP   128
#define TOPK   8
#define FF     1152
#define QK_CLIP 22.627416610717773f
#define REL_CLIP 8.0f

// ---- K/V cache element type.
// Default: fp16, the values the model computed. With KV8 (the opt-in "--kv8" cache, half the size, NOT exact): one signed
// byte per value, v[i] = q[i] * s per 128-value vector. The scale costs no space: s is one of 256 steps
// s(c) = (8 + (c & 7)) * 2^((c >> 3) - 20) (so every q * s is exact in fp16 and f32), and the 8 bits of c are the lowest bits of
// the vector's first 8 bytes (bit 7 - i in byte i) - those 8 values are rounded to the even or odd integers.
// Every reader, including the token's own step, uses the stored values, so batched and single-token paths agree.
#ifdef KV8
typedef int8_t kv_t;
typedef uint kvw_t;            // 4 consecutive cache elements, for copies
inline float kv_scale(uint c) { return float(8u + (c & 7u)) * as_type<float>(((c >> 3) + 107u) << 23); }
// smallest step that represents a vector whose largest magnitude is mx (126 steps: the first 8 values may only reach the even 126)
inline uint kv_code(float mx) {
    if (!(mx > 0.0f)) return 0u;
    const float t = mx / 126.0f;
    int ex = clamp((int) floor(log2(t)), -17, 14);
    int m = clamp((int) ceil(t * exp2((float) -ex) * 8.0f), 8, 16);
    if (m == 16) { m = 8; ex += 1; }
    uint c = (uint) min(((ex + 17) << 3) | (m - 8), 255);
    if (kv_scale(c) * 126.0f < mx && c < 255u) c += 1u;      // log2 / exp2 rounding must never make the largest value clip
    return c;
}
// the scale of the stored vector that starts at p
inline float kv_vscale(device const kv_t * p) {
    device const uint * w = (device const uint *) p;
    const uint hi = ((w[0] & 0x01010101u) * 0x08040201u) >> 24, lo = ((w[1] & 0x01010101u) * 0x08040201u) >> 24;
    return kv_scale(((hi & 15u) << 4) | (lo & 15u));
}
// element i of a vector with scale code c: the stored integer (as float) nearest to x / s
inline float kv_quant(float x, float s, uint c, ushort i) {
    const float y = x / s;
    if (i < 8) { const float b = float((c >> (7 - i)) & 1u); return clamp(2.0f * rint((y - b) * 0.5f) + b, -127.0f, 127.0f); }
    return clamp(rint(y), -127.0f, 127.0f);
}
#else
typedef half kv_t;
typedef half4 kvw_t;
#endif

struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct block_q8_0 { half d; char qs[32]; };
struct block_q5_0 { half d; uchar qh[4]; uchar qs[16]; };

inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }

// NV4 / MX8 element decoding without tables (chadlyb/erisinky): magnitude bits placed into a half give value * 2^-k,
// for normal and subnormal codes alike. FP4 E2M1 -> x 2^-14; FP8 E4M3 -> x 2^-8.
inline float fp4f(uint n) { return float(as_type<half>(ushort(((n & 8u) << 12) | ((n & 7u) << 9)))); }
inline float e4m3f(uint b) { return float(as_type<half>(ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7)))); }
inline float4 fp4f4(uchar4 q) { return float4(fp4f(q.x), fp4f(q.y), fp4f(q.z), fp4f(q.w)); }
inline float4 e4m3f4(uchar4 q) { return float4(e4m3f(q.x), e4m3f(q.y), e4m3f(q.z), e4m3f(q.w)); }
#define NV4_K 4194304.0f   // 2^22 = 2^14 (fp4) * 2^8 (e4m3 scale)
#define MX8_K 256.0f       // 2^8 (e4m3 value)
#define NV4_BAL 1
inline void fp4x8(uint q, thread float4 & lo, thread float4 & hi) {
    const uint4 w = uint4(as_type<uchar4>(q));
    lo = float4(as_type<half4>(ushort4(((w & 8u) << 12) | ((w & 7u) << 9))));
    hi = float4(as_type<half4>(ushort4(((w & 0x80u) << 8) | ((w & 0x70u) << 5))));
}
inline float2 e4m3f2(uint sc) {
    const uint t = (sc | (sc << 8)) & 0x00FF00FFu;
    return float2(as_type<half2>(((t & 0x00800080u) << 8) | ((t & 0x007F007Fu) << 7)));
}
inline void e4m3x8(device const ushort * p, thread float4 & a, thread float4 & b) {
    const uint4 w0 = uint4(as_type<uchar4>(uint(p[0]) | (uint(p[1]) << 16))), w1 = uint4(as_type<uchar4>(uint(p[2]) | (uint(p[3]) << 16)));
    a = float4(as_type<half4>(ushort4(((w0 & 0x80u) << 8) | ((w0 & 0x7Fu) << 7))));
    b = float4(as_type<half4>(ushort4(((w1 & 0x80u) << 8) | ((w1 & 0x7Fu) << 7))));
}
inline float gscale_at(device const uchar * W, uint lo, uint hi, int e) {
    return ((device const float *) (W + (ulong(hi) << 32 | lo)))[e];
}
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// sum over the whole threadgroup; every thread gets the result. sh must hold >= 32 floats.
inline float tg_sum(float v, threadgroup float * sh, ushort sg, ushort lane, ushort nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? sh[lane] : 0.0f;
    return simd_sum(r);
}
inline float tg_max(float v, threadgroup float * sh, ushort sg, ushort lane, ushort nsg) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? sh[lane] : -INFINITY;
    return simd_max(r);
}

// "last threadgroup done" detection: every TG bumps a device counter after its writes; the TG that observes
// count == total-1 knows all others finished (device-scope seq_cst fences make their writes visible) and resets it.
inline bool last_tg(device atomic_uint * ctr, uint total, ushort tid, threadgroup uint * flag) {
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint prev = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        *flag = (prev == total - 1) ? 1u : 0u;
        if (*flag) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *flag != 0;
}

inline void scale_min_k4(int j, device const uchar * q, thread float & sc, thread float & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// ---------------------------------------------------------------------------------------------
// quantized row dot products against x in threadgroup memory. NR rows, row pointers w[r].
// all 32 lanes participate; returns per-lane partial sums in sumf[r] (caller simd_sums).

template <int NR>
inline void dot_q4k(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
    const short ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
    float yl[16], yh[16];
    for (int ib = ix; ib < nb; ib += 4) {
        device const float * y4 = x + ib * 256 + 64 * iq + 8 * ir;
        float4 sumy = 0.0f;
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        for (short row = 0; row < NR; ++row) {
            device const block_q4_K * b = (device const block_q4_K *) w[row] + ib;
            device const ushort * sc = (device const ushort *) b->scales + iq;
            device const ushort * q1 = (device const ushort *) b->qs + 16 * iq + 4 * ir;
            device const ushort * q2 = q1 + 32;
            ushort sc16[4];
            thread const uchar * sc8 = (thread const uchar *) sc16;
            sc16[0] = sc[0] & 0x3f3f;
            sc16[1] = sc[2] & 0x3f3f;
            sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
            sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
            float4 acc1 = 0.0f, acc2 = 0.0f;
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += float(b->d) * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                        (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                        (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                        (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         float(b->dmin) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
    }
}

template <int NR>
inline void dot_q6k(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
    const short tid = lane / 2, ix = lane % 2, ip = tid / 8, il = tid % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
    float yl[16];
    for (int i = ix; i < nb; i += 2) {
        device const float * y = x + i * 256 + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0]; yl[4 * l + 1] = y[l + 32]; yl[4 * l + 2] = y[l + 64]; yl[4 * l + 3] = y[l + 96];
        }
        for (short row = 0; row < NR; ++row) {
            device const block_q6_K * b = (device const block_q6_K *) w[row] + i;
            device const uchar * q1 = b->ql + q_offset_l;
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = b->qh + q_offset_h;
            device const char  * sc = b->scales + is;
            float4 sums = 0.0f;
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((int8_t) ((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((int8_t) ((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((int8_t) ((q1[l] >> 4)  | ((qh[l] & 0x30) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((int8_t) ((q2[l] >> 4)  | ((qh[l] & 0xC0) >> 2)) - 32);
            }
            sumf[row] += float(b->d) * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }
}

// 4 lanes per block (8 quants each), 8 blocks per simdgroup pass (chadlyb/erisinky)
template <int NR>
inline void dot_q8(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
    const short ix = lane / 4, il = lane % 4;
    for (int ib = ix; ib < nb; ib += 8) {
        device const float4 * x4 = (device const float4 *) (x + ib * 32 + il * 8);
        const float4 xa = x4[0], xb = x4[1];
        for (short row = 0; row < NR; ++row) {
            device const block_q8_0 * b = (device const block_q8_0 *) w[row] + ib;
            device const ushort * q = (device const ushort *) b->qs + il * 4;
            const char2 q0 = as_type<char2>(q[0]), q1 = as_type<char2>(q[1]), q2 = as_type<char2>(q[2]), q3 = as_type<char2>(q[3]);
            const float s = dot(xa, float4(q0.x, q0.y, q1.x, q1.y)) + dot(xb, float4(q2.x, q2.y, q3.x, q3.y));
            sumf[row] += s * float(b->d);
        }
    }
}

template <int NR>
inline void dot_nv4(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
    const short ix = lane / 4, il = lane % 4;
    for (int ib = ix; ib < nb; ib += 8) {
        device const float * xb = x + ib * 32 + il * 4;
        const float4 xl = *(device const float4 *) xb, xh = *(device const float4 *) (xb + 16);
        for (short row = 0; row < NR; ++row) {
            device const ushort * p = (device const ushort *) (w[row] + ib * 18);
            const float2 sc = e4m3f2(p[0]);
            float4 lo, hi; fp4x8(uint(p[1 + 2 * il]) | (uint(p[2 + 2 * il]) << 16), lo, hi);
            sumf[row] += sc.x * dot(xl, lo) + sc.y * dot(xh, hi);
        }
    }
}

template <int NR>
inline void dot_mx8(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
    const short ix = lane / 4, il = lane % 4;
    for (int ib = ix; ib < nb; ib += 8) {
        device const float4 * x4 = (device const float4 *) (x + ib * 32 + il * 8);
        const float4 xa = x4[0], xb = x4[1];
        for (short row = 0; row < NR; ++row) {
            device const ushort * p = (device const ushort *) (w[row] + ib * 34);
            float4 a, b; e4m3x8(p + 1 + 4 * il, a, b);
            sumf[row] += (dot(xa, a) + dot(xb, b)) * float(as_type<half>(p[0])) * MX8_K;
        }
    }
}

template <int NR>
inline void dot_bf16(device const uchar * const * w, int K, device const float * x, ushort lane, thread float * sumf) {
    for (int i = lane * 8; i < K; i += 256) {
        const float4 xa = *(device const float4 *) (x + i), xb = *(device const float4 *) (x + i + 4);
        for (short row = 0; row < NR; ++row) {
            const ushort4 a = *(device const ushort4 *) (w[row] + 2 * i);
            const ushort4 b = *(device const ushort4 *) (w[row] + 2 * i + 8);
            sumf[row] += dot(xa, float4(bf16f(a.x), bf16f(a.y), bf16f(a.z), bf16f(a.w))) +
                         dot(xb, float4(bf16f(b.x), bf16f(b.y), bf16f(b.z), bf16f(b.w)));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// embedding lookup (Q4_K row) + RMSNorm(token_embd_norm)

struct EmbedArgs { uint emb_lo, emb_hi, row_bytes, norm_off, out0, out1, tok_index; float eps; };

kernel void k_embed(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                    constant EmbedArgs & a [[buffer(5)]], device const int * toks [[buffer(6)]], uint tok [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    const int t = toks[a.tok_index + tok];
    const ulong off = (ulong(a.emb_hi) << 32 | a.emb_lo) + (ulong) t * a.row_bytes;
    device const block_q4_K * row = (device const block_q4_K *) (W + off);
    float v[9];
    float ss = 0.0f;
    if (a.row_bytes == (E / 32) * 34) {     // Q8_0
        device const block_q8_0 * r8 = (device const block_q8_0 *) (W + off);
        for (int k = 0; k < 9; k++) { const int i = tid + k * 256; v[k] = float(r8[i / 32].d) * float(r8[i / 32].qs[i % 32]); ss += v[k] * v[k]; }
    } else
    for (int k = 0; k < 9; k++) {
        const int j = tid, pair = j / 64, l = j % 64;
        device const block_q4_K & b = row[k];
        const int sub = pair * 2 + (l >= 32 ? 1 : 0);
        const uchar q = b.qs[pair * 32 + (l & 31)];
        const int qv = l >= 32 ? (q >> 4) : (q & 0xF);
        float sc, m;
        scale_min_k4(sub, b.scales, sc, m);
        v[k] = float(b.d) * sc * qv - float(b.dmin) * m;
        ss += v[k] * v[k];
    }
    ss = tg_sum(ss, sh, sg, lane, 8);
    const float r = rsqrt(ss / E + a.eps);
    device const float * w = (device const float *) (PB + a.norm_off);
    for (int k = 0; k < 9; k++) {
        const int i = tid + k * 256;
        const float o = v[k] * r * w[i];
        A[a.out0 + tok * E + i] = o;
        A[a.out1 + tok * E + i] = o;
    }
}

// Per-span descriptor for 3D-batched Layer 3 decode dispatches (64 bytes, mirrors Rust `DecodeSpanDesc`).
struct DecodeSpanDesc {
    int   tok_id;
    uint  pos;
    uint  cs_base;
    uint  bt_full_base;
    uint  bt_win_base;
    uint  part_off;
    uint  swa_nkeys;
    uint  swa_first_pos;
    uint  swa_kps;
    uint  swa_nsplit;
    uint  full_nkeys;
    uint  full_kps;
    uint  full_nsplit;
    float full_tau;
    float swa_tau;
    uint  pad1;
};

struct EmbedDecArgs { uint emb_lo, emb_hi, row_bytes, norm_off, out_off, desc_off; float eps; uint pad; };

kernel void k_embed_dec(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                        constant EmbedDecArgs & a [[buffer(5)]], uint si [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    device const DecodeSpanDesc & d = ((device const DecodeSpanDesc *) (A + a.desc_off))[si];
    const int t = d.tok_id;
    const ulong off = (ulong(a.emb_hi) << 32 | a.emb_lo) + (ulong) t * a.row_bytes;
    device const block_q4_K * row = (device const block_q4_K *) (W + off);
    float v[9];
    float ss = 0.0f;
    if (a.row_bytes == (E / 32) * 34) {     // Q8_0
        device const block_q8_0 * r8 = (device const block_q8_0 *) (W + off);
        for (int k = 0; k < 9; k++) { const int i = tid + k * 256; v[k] = float(r8[i / 32].d) * float(r8[i / 32].qs[i % 32]); ss += v[k] * v[k]; }
    } else
    for (int k = 0; k < 9; k++) {
        const int j = tid, pair = j / 64, l = j % 64;
        device const block_q4_K & b = row[k];
        const int sub = pair * 2 + (l >= 32 ? 1 : 0);
        const uchar q = b.qs[pair * 32 + (l & 31)];
        const int qv = l >= 32 ? (q >> 4) : (q & 0xF);
        float sc, m;
        scale_min_k4(sub, b.scales, sc, m);
        v[k] = float(b.d) * sc * qv - float(b.dmin) * m;
        ss += v[k] * v[k];
    }
    ss = tg_sum(ss, sh, sg, lane, 8);
    const float r = rsqrt(ss / E + a.eps);
    device const float * w = (device const float *) (PB + a.norm_off);
    for (int k = 0; k < 9; k++) {
        const int i = tid + k * 256;
        A[a.out_off + si * E + i] = v[k] * r * w[i];
    }
}

// ---------------------------------------------------------------------------------------------
// depth attention ("block attention residuals"): out_q = sum_i softmax_i(q . n_i / sqrt(E)) n_i, n_i = rmsnorm(s_i)
// optionally followed by RMSNorm with a weight. q_off < 0: identity (out = raw s_0). One threadgroup per job, 256 threads.

// hcopy / hout: fp16 destinations of the batched kernel (normalized copies with row stride hstride halves; (normed) outputs
// with row stride E halves). nof32 bit q: output q is only wanted in fp16.
struct MixJob { uint nS, nQ; uint s_off[6]; int q_off; uint out_off[4]; int norm_off[4]; float eps; uint tstride; int ncopy[4][3]; uint pad_t;
                int hcopy[4][3]; int hout[4]; uint hstride, nof32; };

kernel void k_mix(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant MixJob * jobs [[buffer(5)]],
                  uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    // 768 threads, 3 channels per thread; all register arrays statically indexed (no spills)
    constant MixJob & J = jobs[tgp.x];
    const uint toff = tgp.y * J.tstride;
    threadgroup float part[24][32];
    threadgroup float red[32];
    threadgroup float sh2[24][4];
    const uint nS = J.nS, nQ = J.nQ;
    float s[6][3];
    #pragma unroll
    for (uint i = 0; i < 6; i++) {
        #pragma unroll
        for (int k = 0; k < 3; k++) s[i][k] = i < nS ? A[J.s_off[i] + toff + tid + k * 768] : 0.0f;
    }
    float o[4][3];
    if (J.q_off < 0) {
        #pragma unroll
        for (int k = 0; k < 3; k++) { o[0][k] = s[0][k]; o[1][k] = o[2][k] = o[3][k] = 0; }
    } else {
        device const float * Q = (device const float *) (PB + J.q_off);
        float qv[4][3];
        #pragma unroll
        for (uint q = 0; q < 4; q++) {
            #pragma unroll
            for (int k = 0; k < 3; k++) qv[q][k] = q < nQ ? Q[q * E + tid + k * 768] : 0.0f;
        }
        // v index: i*5 + 0 -> ss_i ; i*5 + 1 + q -> dot_iq
        #pragma unroll
        for (uint i = 0; i < 6; i++) {
            float ss = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            #pragma unroll
            for (int k = 0; k < 3; k++) { ss += s[i][k] * s[i][k]; d0 += s[i][k] * qv[0][k]; d1 += s[i][k] * qv[1][k]; d2 += s[i][k] * qv[2][k]; d3 += s[i][k] * qv[3][k]; }
            ss = simd_sum(ss); d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
            if (lane == 0) { part[sg][i * 5 + 0] = ss; part[sg][i * 5 + 1] = d0; part[sg][i * 5 + 2] = d1; part[sg][i * 5 + 3] = d2; part[sg][i * 5 + 4] = d3; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 30) { float t = 0; for (int g = 0; g < 24; g++) t += part[g][tid]; red[tid] = t; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rr[6];
        #pragma unroll
        for (uint i = 0; i < 6; i++) rr[i] = rsqrt(red[i * 5] / E + 1e-6f);
        #pragma unroll
        for (uint q = 0; q < 4; q++) {
            float sc[6], mx = -INFINITY;
            #pragma unroll
            for (uint i = 0; i < 6; i++) { sc[i] = i < nS ? rr[i] * red[i * 5 + 1 + q] * (1.0f / 48.0f) : -INFINITY; mx = max(mx, sc[i]); }
            float den = 0;
            #pragma unroll
            for (uint i = 0; i < 6; i++) { sc[i] = i < nS ? exp(sc[i] - mx) : 0.0f; den += sc[i]; }
            #pragma unroll
            for (uint i = 0; i < 6; i++) sc[i] = sc[i] / den * rr[i];
            #pragma unroll
            for (int k = 0; k < 3; k++) { float acc = 0; for (uint i = 0; i < 6; i++) acc += sc[i] * s[i][k]; o[q][k] = acc; }
        }
    }
    const uint nOut = J.q_off < 0 ? 1 : nQ;
    // optional RMSNorm per output: one reduction round for all outputs
    float ssq[4];
    #pragma unroll
    for (uint q = 0; q < 4; q++) { float t = 0; for (int k = 0; k < 3; k++) t += o[q][k] * o[q][k]; ssq[q] = simd_sum(t); }
    if (lane == 0) for (int q = 0; q < 4; q++) sh2[sg][q] = ssq[q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint q = 0; q < 4; q++) {
        if (q >= nOut) break;
        float t = 0; for (int g = 0; g < 24; g++) t += sh2[g][q];
        // plain RMSNorm copies (eps 1e-6, no weight) for the loop controllers' inputs
        const float rn = rsqrt(t / E + 1e-6f);
        for (int c = 0; c < 3; c++) if (J.ncopy[q][c] >= 0) for (int k = 0; k < 3; k++) A[J.ncopy[q][c] + toff + tid + k * 768] = o[q][k] * rn;
        if (J.norm_off[q] >= 0) {
            const float r = rsqrt(t / E + J.eps);
            device const float * w = (device const float *) (PB + J.norm_off[q]);
            for (int k = 0; k < 3; k++) A[J.out_off[q] + toff + tid + k * 768] = o[q][k] * r * w[tid + k * 768];
        } else {
            for (int k = 0; k < 3; k++) A[J.out_off[q] + toff + tid + k * 768] = o[q][k];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// generic segmented matvec: several weight matrices (possibly different quant types and inputs) in one dispatch.
// x is staged into threadgroup memory with optional transforms: * sigmoid(in2), RMSNorm(norm weight).
// epilogue 1: causal short conv (post-conv) with state + optional residual.

struct Seg {
    uint w_lo, w_hi, row_bytes, wtype;
    uint nrows, K, tg_start, in_off;
    int  in2_off, norm_off, ss_off, ss_n;
    uint out_off, epi; int res_off; uint cs_off;
    uint convw_off, pad0, pad1, pad2;
};
struct MvArgs { uint nseg; float eps; uint post, ctr; uint bias_off, sel_off; float wscale; uint pad; Seg seg[12]; };

// top-k over NEXP router logits (sigmoid scores + selection bias), normalized weights * wscale. one simdgroup.
inline void topk_sg(device float * A, device const uchar * PB, uint logits_off, uint bias_off, uint sel_off, float wscale, ushort lane) {
    device const float * lg = A + logits_off;
    device const float * bias = (device const float *) (PB + bias_off);
    float pr[4], sel[4];
    for (int i = 0; i < 4; i++) {
        const int e = lane * 4 + i;
        const float p = sigm(lg[e]), score = p + bias[e];
        // Inspect bits because these kernels are compiled with fast math.
        const bool finite = (as_type<uint>(score) & 0x7f800000u) != 0x7f800000u;
        pr[i] = finite ? p : 0.0f;
        sel[i] = finite ? score : as_type<float>(0xff7fffffu);
    }
    float wsum = 0, myw = 0; int myid = 0;
    for (int k = 0; k < TOPK; k++) {
        float bv = sel[0]; int bi = 0;
        for (int i = 1; i < 4; i++) if (sel[i] > bv) { bv = sel[i]; bi = i; }
        const float mv = simd_max(bv);
        const int candidate = simd_min((bv == mv) ? (int) (lane * 4 + bi) : NEXP);
        const int win = candidate < NEXP ? candidate : k;
        const float pw = simd_shuffle(pr[win % 4], (ushort) (win / 4));
        if ((int) (lane * 4 + bi) == win) sel[bi] = -INFINITY;
        if (lane == (ushort) k) { myid = win; myw = pw; }
        wsum += pw;
    }
    if (lane < TOPK) {
        A[sel_off + lane] = (float) myid;
        A[sel_off + TOPK + lane] = myw / (wsum + 1e-20f) * wscale;
    }
}

#define MV_NSG 4
#define MV_NR  2
#define MV_ROWS (MV_NSG * MV_NR)

inline void mv_epilogue(constant Seg & S, uint row, float y, device float * A, device float * CS, device const uchar * PB) {
    if (S.epi == 0) { A[S.out_off + row] = y; return; }
    // post short-conv (static taps) over channel `row` (S.epi == 2: dry-state, read CS without updating)
    device float * h = CS + S.cs_off + row * 3;
    device const float * w = (device const float *) (PB + S.convw_off) + row * 4;
    const float h0 = h[0], h1 = h[1], h2 = h[2];
    const float conv = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * y;
    if (S.epi == 1) { h[0] = h1; h[1] = h2; h[2] = y; }
    float o = (y + conv) * 0.4472135954999579f;  // 1/sqrt(5)
    if (S.res_off >= 0) o += A[S.res_off + row];
    A[S.out_off + row] = o;
}

// inputs are pre-normalized; x is read straight from device memory (L1/L2 cached), no staging/barriers
kernel void k_mv(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                 device float * CS [[buffer(4)]], constant MvArgs & args [[buffer(5)]], device atomic_uint * CTR [[buffer(7)]],
                 uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint flag;
    uint si = 0;
    while (si + 1 < args.nseg && tg >= args.seg[si + 1].tg_start) si++;
    constant Seg & S = args.seg[si];
    const uint row0 = (tg - S.tg_start) * MV_ROWS + sg * MV_NR;
    if (row0 < S.nrows) {
    const ulong wbase = ulong(S.w_hi) << 32 | S.w_lo;
    device const uchar * base = (S.wtype == 30) ? PB : W;
    device const uchar * wp[MV_NR];
    for (int rr = 0; rr < MV_NR; rr++) wp[rr] = base + wbase + (ulong) min(row0 + rr, S.nrows - 1) * S.row_bytes;
    float sumf[MV_NR] = {0, 0};
    device const float * x = A + S.in_off;
    if (S.wtype == 12) dot_q4k<MV_NR>(wp, S.K / 256, x, lane, sumf);
    else if (S.wtype == 14) dot_q6k<MV_NR>(wp, S.K / 256, x, lane, sumf);
    else if (S.wtype == 8) dot_q8<MV_NR>(wp, S.K / 32, x, lane, sumf);
    else dot_bf16<MV_NR>(wp, S.K, x, lane, sumf);
    for (int rr = 0; rr < MV_NR; rr++) {
        const float y = simd_sum(sumf[rr]);
        if (lane == 0 && row0 + rr < S.nrows) mv_epilogue(S, row0 + rr, y, A, CS, PB);
    }
    }
    if (args.post == 1 && last_tg(CTR + args.ctr, ntg, sg * 32 + lane, &flag) && sg == 0)
        topk_sg(A, PB, args.seg[0].out_off, args.bias_off, args.sel_off, args.wscale, lane);
}

// k_mv over the same weights for nv.x vectors at a stride of nv.y floats (the lanes of a decode batch): each weight row
// is read from memory once per step and applied to every vector; each vector's dot product is the same code, in the same
// order, as k_mv's. No epilogue (k_epi applies it per lane); the fused top-k, when asked, runs for every vector.
kernel void k_mvb(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                  constant MvArgs & args [[buffer(5)]], device atomic_uint * CTR [[buffer(7)]], constant uint2 & nv [[buffer(8)]],
                  uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint flag;
    uint si = 0;
    while (si + 1 < args.nseg && tg >= args.seg[si + 1].tg_start) si++;
    constant Seg & S = args.seg[si];
    const uint row0 = (tg - S.tg_start) * MV_ROWS + sg * MV_NR;
    if (row0 < S.nrows) {
        const ulong wbase = ulong(S.w_hi) << 32 | S.w_lo;
        device const uchar * base = (S.wtype == 30) ? PB : W;
        device const uchar * wp[MV_NR];
        for (int rr = 0; rr < MV_NR; rr++) wp[rr] = base + wbase + (ulong) min(row0 + rr, S.nrows - 1) * S.row_bytes;
        for (uint v = 0; v < nv.x; v++) {
            float sumf[MV_NR] = {0, 0};
            device const float * x = A + S.in_off + v * nv.y;
            if (S.wtype == 12) dot_q4k<MV_NR>(wp, S.K / 256, x, lane, sumf);
            else if (S.wtype == 14) dot_q6k<MV_NR>(wp, S.K / 256, x, lane, sumf);
            else if (S.wtype == 8) dot_q8<MV_NR>(wp, S.K / 32, x, lane, sumf);
            else dot_bf16<MV_NR>(wp, S.K, x, lane, sumf);
            for (int rr = 0; rr < MV_NR; rr++) {
                const float y = simd_sum(sumf[rr]);
                if (lane == 0 && row0 + rr < S.nrows) A[S.out_off + v * nv.y + row0 + rr] = y;
            }
        }
    }
    if (args.post == 1 && last_tg(CTR + args.ctr, ntg, sg * 32 + lane, &flag) && sg == 0)
        for (uint v = 0; v < nv.x; v++) topk_sg(A, PB, args.seg[0].out_off + v * nv.y, args.bias_off, args.sel_off + v * nv.y, args.wscale, lane);
}

// mv_epilogue for one lane, after k_mvb wrote the raw projection y to out_off: the post short-conv over each channel
// from the lane's state, the residual, the result written back to out_off. Same arithmetic as mv_epilogue.
struct EpiArgs { uint out_off; int res_off; uint cs_off; uint convw_off; uint n; uint pad0, pad1, pad2; };
kernel void k_epi(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device float * CS [[buffer(4)]],
                  constant EpiArgs & a [[buffer(5)]], uint row [[thread_position_in_grid]]) {
    if (row >= a.n) return;
    const float y = A[a.out_off + row];
    device float * h = CS + a.cs_off + row * 3;
    device const float * w = (device const float *) (PB + a.convw_off) + row * 4;
    const float h0 = h[0], h1 = h[1], h2 = h[2];
    const float conv = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * y;
    h[0] = h1; h[1] = h2; h[2] = y;
    float o = (y + conv) * 0.4472135954999579f;  // 1/sqrt(5)
    if (a.res_off >= 0) o += A[a.res_off + row];
    A[a.out_off + row] = o;
}

// RMSNorm of up to 3 vectors from precomputed partial sums of squares (written by k_ctrl_apply). 1 TG per vector.
struct NormJob { uint in_off, out_off, w_off, ss_off; };
struct NormArgs { uint n, nparts; float eps; uint tstride; NormJob job[4]; };
kernel void k_norm(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant NormArgs & a [[buffer(5)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constant NormJob & J = a.job[tgp.x];
    const uint tok = tgp.y, toff = tok * a.tstride;
    float ss = 0;
    for (uint i = 0; i < a.nparts; i++) ss += A[J.ss_off + tok * 128 + i];
    const float r = rsqrt(ss / E + a.eps);
    device const float * w = (device const float *) (PB + J.w_off);
    for (uint i = tid; i < E; i += 256) A[J.out_off + toff + i] = A[J.in_off + toff + i] * r * w[i];
}

// ---------------------------------------------------------------------------------------------
// loop controller: g = D . [rmsnorm(u); rmsnorm(m)]  (D: 64 x 4608 bf16). grid: nC*16 TGs of 128 threads (1 row / simdgroup)

struct CtrlJob { uint u_off, m_off, d_off, out_off, gain_off, forget_off, res_out, ss_off; int norm_w; uint ctr; };
struct CtrlArgs { uint nC; uint tstride; uint pad1, pad2; CtrlJob job[4]; };

kernel void k_ctrl_down(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant CtrlArgs & a [[buffer(5)]],
                        uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[2 * E];
    threadgroup float sh[32];
    const uint tg = tgp.x, tok = tgp.y, toff = tok * a.tstride;
    const uint c = tg / 16;
    constant CtrlJob & J = a.job[c];
    float su = 0, sm = 0;
    for (uint i = tid; i < E; i += 128) { float u = A[J.u_off + toff + i], m = A[J.m_off + toff + i]; su += u * u; sm += m * m; }
    su = tg_sum(su, sh, sg, lane, 4);
    sm = tg_sum(sm, sh, sg, lane, 4);
    const float ru = rsqrt(su / E + 1e-6f), rm = rsqrt(sm / E + 1e-6f);
    for (uint i = tid; i < E; i += 128) { xs[i] = A[J.u_off + toff + i] * ru; xs[E + i] = A[J.m_off + toff + i] * rm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = (tg % 16) * 4 + sg;
    device const uchar * wr = PB + J.d_off + row * (2 * E) * 2;
    float acc = 0;
    for (int i = lane * 8; i < 2 * E; i += 256) {
        const float4 xa = *(threadgroup const float4 *) (xs + i), xb = *(threadgroup const float4 *) (xs + i + 4);
        const ushort4 p = *(device const ushort4 *) (wr + 2 * i), q = *(device const ushort4 *) (wr + 2 * i + 8);
        acc += dot(xa, float4(bf16f(p.x), bf16f(p.y), bf16f(p.z), bf16f(p.w))) + dot(xb, float4(bf16f(q.x), bf16f(q.y), bf16f(q.z), bf16f(q.w)));
    }
    const float y = simd_sum(acc);
    if (lane == 0) A[J.out_off + tok * 64 + row] = y;
}

// out = u*(1 - tanh(F.g)) + tanh(G.g)*m ; G,F: E x 64 bf16. 8 lanes per channel; 256-thread TG = 32 channels.
// writes per-TG partial sum of squares of out to ss_off[tg] (for a downstream RMSNorm).
kernel void k_ctrl_apply(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant CtrlArgs & a [[buffer(5)]],
                         device atomic_uint * CTR [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ntg [[threadgroups_per_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    threadgroup uint flag;
    constant CtrlJob & J = a.job[tg.y];
    const uint tok = tg.z, toff = tok * a.tstride;
    const uint ch = tg.x * 32 + sg * 4 + lane / 8;
    const uint p = lane % 8;
    float g[8];
    for (int i = 0; i < 8; i++) g[i] = A[J.out_off + tok * 64 + p * 8 + i];
    ushort4 ga = *(device const ushort4 *) (PB + J.gain_off + ch * 128 + p * 16);
    ushort4 gb = *(device const ushort4 *) (PB + J.gain_off + ch * 128 + p * 16 + 8);
    ushort4 fa = *(device const ushort4 *) (PB + J.forget_off + ch * 128 + p * 16);
    ushort4 fb = *(device const ushort4 *) (PB + J.forget_off + ch * 128 + p * 16 + 8);
    float dg = bf16f(ga.x) * g[0] + bf16f(ga.y) * g[1] + bf16f(ga.z) * g[2] + bf16f(ga.w) * g[3] +
               bf16f(gb.x) * g[4] + bf16f(gb.y) * g[5] + bf16f(gb.z) * g[6] + bf16f(gb.w) * g[7];
    float df = bf16f(fa.x) * g[0] + bf16f(fa.y) * g[1] + bf16f(fa.z) * g[2] + bf16f(fa.w) * g[3] +
               bf16f(fb.x) * g[4] + bf16f(fb.y) * g[5] + bf16f(fb.z) * g[6] + bf16f(fb.w) * g[7];
    dg += simd_shuffle_xor(dg, 1); dg += simd_shuffle_xor(dg, 2); dg += simd_shuffle_xor(dg, 4);
    df += simd_shuffle_xor(df, 1); df += simd_shuffle_xor(df, 2); df += simd_shuffle_xor(df, 4);
    float o = 0;
    if (p == 0) {
        const float u = A[J.u_off + toff + ch], m = A[J.m_off + toff + ch];
        o = u - tanh(df) * u + tanh(dg) * m;
        A[J.res_out + toff + ch] = o;
    }
    const float ss = tg_sum(o * o, sh, sg, lane, 8);
    if (tid == 0) A[J.ss_off + tok * 128 + tg.x] = ss;
    if (J.norm_w >= 0 && last_tg(CTR + J.ctr, ntg.x, tid, &flag)) {
        float t = 0;
        for (uint i = 0; i < ntg.x; i++) t += A[J.ss_off + tok * 128 + i];
        const float r = rsqrt(t / E + 1e-6f);
        device const float * w = (device const float *) (PB + J.norm_w);
        for (uint i = tid; i < E; i += 256) A[J.res_out + toff + i] *= r * w[i];
    }
}

// ---------------------------------------------------------------------------------------------
// decode attention for one token, one kv head per TG column, keys split across TG rows (flash-decoding).
// fuses: K/V short conv (static + dynamic taps) with state update, K clip, V RMSNorm, Q clip, R clip,
// log-scale tau, relative-position bias, per-head sink, partial softmax.

struct AttnArgs {
    uint pos, nkeys, first_pos, ring, cap, k_base, v_base, cs_off;
    uint q_off, k_off, v_off, r_off, sink_off; int kdyn_off, vdyn_off; uint part_off;
    uint kconv_w, vconv_w, qclip_w, relclip_w, relbias_w; float tau; uint kps, nsplit;
    uint ctr, gate_off, out_off, pad;
    // sp0: index of this dispatch's first split among the layer's nsplit partials; noown: keys only (another dispatch computes
    // and stores the token's own K/V); row (g, idx) of the bound K/V lives at g * gstr + idx * istr; idx = pos - idx0 if !ring
    uint sp0, noown, gstr, istr, idx0;
    float lastbias;      // dev (key eviction experiments): added to the score of the dispatch's last key
};
#define KVROW(g, idx) ((g) * a.gstr + (idx) * a.istr)
#define KVIDX(p) (a.ring ? ((p) % a.cap) : ((p) - a.idx0))

#define KPS_MAX 256

#pragma clang fp contract(off) reassociate(off)
kernel void k_attn(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                   device float * CS [[buffer(4)]], constant AttnArgs & a [[buffer(5)]], device atomic_uint * CTR [[buffer(7)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    // one TG = (key split, kv head g) serving q heads 3g..3g+2; 128 threads; keys processed in chunks of 32
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float kc[HD], vc[HD];
    threadgroup float red[4][8];
    const uint sp = tgp.x, g = tgp.y;
    const uint p_lo = a.first_pos;
    const uint p_hi = a.first_pos + a.nkeys;
    const uint split_start = ((a.pad & 4u) != 0u) ? (p_lo + sp * a.kps) : ((p_lo / a.kps + sp) * a.kps);
    const uint split_end = split_start + a.kps;
    const uint p1 = min(p_hi, split_end);
    const bool owns = (p1 == p_hi) && a.noown == 0;
    const bool dry = (a.pad & 2u) != 0u;
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    // ---- phase 1: raw values + partial norms
    float q[3];
    for (int h = 0; h < 3; h++) { const uint H = g * 3 + h; q[h] = A[a.q_off + H * HD + tid] * qclip[H * HD + tid]; }
    float kx = 0, vx = 0;
    if (owns) {
        const uint c = g * HD + tid;
        const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            const uint ch = which * (NKV * HD) + c;
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_w : a.vconv_w)) + c * 4;
            device float * hh = CS + a.cs_off + ch * 3;
            const float x = A[(which == 0 ? a.k_off : a.v_off) + c], h0 = hh[0], h1 = hh[1], h2 = hh[2];
            float acc = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * x;
            const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
            if (dyn >= 0) { device const float * t = A + dyn + g * 4; acc += t[0] * x + t[1] * h2 + t[2] * h1 + t[3] * h0; }
            if (!dry) { hh[0] = h1; hh[1] = h2; hh[2] = x; }
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[a.r_off + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        const float s5 = simd_max(fabs(kx)), s6 = simd_max(fabs(vx));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    // logit = (q.k)/sqrt(HD) + r.B[d] - sink  (bias + sink are unscaled, as llama.cpp folds them into the mask)
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * a.tau * 0.08838834764831845f;
        ((threadgroup float *) qs4[h])[tid] = q[h] * f;
    }
    if (tid < 48) rs[tid / 16][tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * a.tau;
    if (owns) {
        const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
        const float k = kx * kf, v = vx * vf;
        const uint idx = KVIDX(a.pos);
#ifdef KV8
        const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
        const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
        const float sk = kv_scale(ck), sv = kv_scale(cv);
        const float qk = kv_quant(k, sk, ck, tid), qv = kv_quant(v, sv, cv, tid);
        kc[tid] = qk * sk; vc[tid] = qv * sv;
        if (!dry) {
            KV[a.k_base + KVROW(g, idx) + tid] = (kv_t) (int) qk;
            KV[a.v_base + KVROW(g, idx) + tid] = (kv_t) (int) qv;
        }
#else
        kc[tid] = k; vc[tid] = v;
        if (!dry) {
            KV[a.k_base + KVROW(g, idx) + tid] = half(k);
            KV[a.v_base + KVROW(g, idx) + tid] = half(v);
        }
#endif
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float * relb = (device const float *) (PB + a.relbias_w);
    // ---- barrier-free per-simdgroup online softmax over 8-key blocks (4 lanes per key for the scores)
    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, l[3] = {0, 0, 0};
    float4 o[3] = {0, 0, 0};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);
    for (uint pb = split_start + sg * 8; pb < p1; pb += 32) {
        if (pb + 8 <= p_lo) continue;
        const uint raw_p = pb + kk;
        const bool valid = (raw_p >= p_lo) && (raw_p < p1);
        const uint p = min(max(raw_p, p_lo), p1 - 1);
        const uint idx = KVIDX(p);
        float acc[3] = {0, 0, 0};
        if (p == a.pos) {
            for (int c = 0; c < 8; c++) {
                const float4 kf = *(threadgroup const float4 *) (&kc[pp * 32 + c * 4]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
        } else {
#ifdef KV8
            device const kv_t * krow = KV + a.k_base + KVROW(g, idx);
            device const packed_char4 * kr = (device const packed_char4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(char4(kr[c]));
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
            const float ks = kv_vscale(krow);
            for (int h = 0; h < 3; h++) acc[h] *= ks;
#else
            device const half4 * kr = (device const half4 *) (KV + a.k_base + KVROW(g, idx) + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
#endif
        }
        const uint d = a.pos - p;
        if (d < EXT) {
            const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4);
            for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
        }
        if (a.lastbias != 0.0f && valid && p == p_hi - 1) for (int h = 0; h < 3; h++) acc[h] += a.lastbias * 0.25f;
        float pr[3], sc_[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? t - sink[h] : -INFINITY;
            const float mx = max(m[h], simd_max(t));
            sc_[h] = (m[h] == -INFINITY) ? 0.0f : exp(m[h] - mx);
            pr[h] = valid ? exp(t - mx) : 0.0f;
            l[h] = l[h] * sc_[h] + simd_sum(pr[h]) * 0.25f;
            m[h] = mx;
            o[h] *= sc_[h];
        }
#ifdef KV8
        // each stored value vector's scale goes into its key's probability (the token's own vector is already scaled)
        { const float vsc = (p == a.pos) ? 1.0f : kv_vscale(KV + a.v_base + KVROW(g, idx)); for (int h = 0; h < 3; h++) pr[h] *= vsc; }
#endif
        // V: lane owns dims [4*lane, 4*lane+4); 8 keys of this block
        for (ushort q = 0; q < 8; q++) {
            const uint pq = pb + q;
            if (pq < p_lo) continue;
            if (pq >= p1) break;
            const uint iq = KVIDX(pq);
            const float4 v = (pq == a.pos) ? *(threadgroup const float4 *) (&vc[lane * 4])
#ifdef KV8
                                           : float4(char4(*(device const packed_char4 *) (KV + a.v_base + KVROW(g, iq) + lane * 4)));
#else
                                           : float4(*(device const half4 *) (KV + a.v_base + KVROW(g, iq) + lane * 4));
#endif
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q * 4) * v;
        }
    }
    // ---- merge the 4 simdgroups' partial softmax states
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int h = 0; h < 3; h++) {
        float M = -INFINITY;
        for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
        float L = 0, O = 0;
        for (int s2 = 0; s2 < 4; s2++) {
            const float w = ms_[s2][h] == -INFINITY ? 0.0f : exp(ms_[s2][h] - M);
            L += w * ls_[s2][h];
            O += w * ((threadgroup const float *) os_[s2][h])[tid];
        }
        device float * P = A + a.part_off + ((g * 3 + h) * a.nsplit + a.sp0 + sp) * (HD + 2);
        if (tid == 0) { P[0] = M; P[1] = L; }
        P[2 + tid] = O;
    }
    // the last split of this kv group combines all splits (+ the zero-logit sink) and applies the sigmoid gate
    threadgroup uint flag;
    if ((a.pad & 1u) == 0u && last_tg(CTR + a.ctr + g, a.nsplit, tid, &flag)) {
        for (int h = 0; h < 3; h++) {
            const uint H = g * 3 + h;
            device const float * P = A + a.part_off + H * a.nsplit * (HD + 2);
            float M = 0.0f;
            for (uint s2 = 0; s2 < a.nsplit; s2++) M = max(M, P[s2 * (HD + 2)]);
            float L = exp(-M), O = 0;
            for (uint s2 = 0; s2 < a.nsplit; s2++) {
                const float w = exp(P[s2 * (HD + 2)] - M);
                L += w * P[s2 * (HD + 2) + 1];
                O += w * P[s2 * (HD + 2) + 2 + tid];
            }
            const uint i = H * HD + tid;
            A[a.out_off + i] = O / L * sigm(A[a.gate_off + i]);
        }
    }
}

#pragma clang fp contract(fast) reassociate(on)

// combine split partials (+ zero-logit sink) and apply the sigmoid output gate. grid: NH TGs x 128 threads
// hout != 0: write half directly to ((device half *) (A + a.out_off))[i] instead of f32.
struct CombineArgs { uint part_off, nsplit, gate_off, out_off; uint hout, pad0, pad1, pad2; };
kernel void k_attn_combine(device float * A [[buffer(2)]], constant CombineArgs & a [[buffer(5)]],
                           uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    device const float * P = A + a.part_off + h * a.nsplit * (HD + 2);
    float M = 0.0f;  // sink logit
    for (uint s = 0; s < a.nsplit; s++) M = max(M, P[s * (HD + 2)]);
    float L = exp(-M), o = 0;
    for (uint s = 0; s < a.nsplit; s++) {
        const float w = exp(P[s * (HD + 2)] - M);
        L += w * P[s * (HD + 2) + 1];
        o += w * P[s * (HD + 2) + 2 + tid];
    }
    const uint i = h * HD + tid;
    const float val = o / L * sigm(A[a.gate_off + i]);
    if (a.hout) ((device half *) (A + a.out_off))[i] = half(val);
    else A[a.out_off + i] = val;
}

// 2D-batched split combine across all lanes in a decode batch. grid: (NH, nspans, 1) x 128 threads
// hout != 0: write half directly to ((device half *) (A + a.out_off))[i] instead of f32.
struct CombineDecArgs { uint desc_off, gate_off, out_off, swa; uint hout, pad0, pad1, pad2; };
kernel void k_attn_combine_dec(device float * A [[buffer(2)]], constant CombineDecArgs & a [[buffer(5)]],
                               uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint h = tgp.x, si = tgp.y;
    device const DecodeSpanDesc & d = ((device const DecodeSpanDesc *) (A + a.desc_off))[si];
    const uint nsplit = a.swa ? d.swa_nsplit : d.full_nsplit;
    device const float * P = A + d.part_off + h * nsplit * (HD + 2);
    float M = 0.0f;  // sink logit
    for (uint s = 0; s < nsplit; s++) M = max(M, P[s * (HD + 2)]);
    float L = exp(-M), o = 0;
    for (uint s = 0; s < nsplit; s++) {
        const float w = exp(P[s * (HD + 2)] - M);
        L += w * P[s * (HD + 2) + 1];
        o += w * P[s * (HD + 2) + 2 + tid];
    }
    const uint i = si * (NH * HD) + h * HD + tid;
    const float val = o / L * sigm(A[a.gate_off + i]);
    if (a.hout) ((device half *) (A + a.out_off))[i] = half(val);
    else A[a.out_off + i] = val;
}

// ---------------------------------------------------------------------------------------------
// MoE: top-k (sigmoid + selection bias), gate_up (Q4_K, fused [gate; up] rows), SiTU activation.
// grid: (FF / (NSG*2), TOPK) TGs of NSG simdgroups; each simdgroup does 2 gate rows + 2 up rows.

struct MoeArgs {
    uint logits_off, bias_off, sel_off, x_off;
    int  norm_off, ss_off, ss_n; float eps;
    uint gu_lo, gu_hi, gu_expert_bytes, gu_row_bytes;
    uint act_off; float wscale, beta, lbeta;
    uint dn_lo, dn_hi, dn_expert_bytes, dn_row_bytes;
    uint dn_type, out_off; int res_off; uint cs_off;
    uint convw_off, gu_type, gu_gs_lo, gu_gs_hi;
    uint dn_gs_lo, dn_gs_hi, no_state_write, pad1;
};

#define GU_NSG 4

kernel void k_topk(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant MoeArgs & a [[buffer(5)]],
                   ushort lane [[thread_index_in_simdgroup]]) {
    topk_sg(A, PB, a.logits_off, a.bias_off, a.sel_off, a.wscale, lane);
}

// grid: (FF / GU_NSG, TOPK); each simdgroup computes gate row i and up row i of one expert -> SiTU
kernel void k_moe_gu(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                     constant MoeArgs & a [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint k = tgp.y;
    const int e = (int) A[a.sel_off + k];
    const uint i = tgp.x * GU_NSG + sg;
    const ulong base = (ulong(a.gu_hi) << 32 | a.gu_lo) + (ulong) e * a.gu_expert_bytes;
    device const uchar * wp[2] = { W + base + (ulong) i * a.gu_row_bytes, W + base + (ulong) (FF + i) * a.gu_row_bytes };
    float sumf[2] = {0, 0};
    if (a.gu_type == 200) dot_nv4<2>(wp, E / 32, A + a.x_off, lane, sumf);
    else if (a.gu_type == 201) dot_mx8<2>(wp, E / 32, A + a.x_off, lane, sumf);
    else if (a.gu_type == 8) dot_q8<2>(wp, E / 32, A + a.x_off, lane, sumf);
    else dot_q4k<2>(wp, E / 256, A + a.x_off, lane, sumf);
    const float g = a.gu_type == 200 ? gscale_at(W, a.gu_gs_lo, a.gu_gs_hi, e) * NV4_K : 1.0f;
    const float gt = simd_sum(sumf[0]) * g, up = simd_sum(sumf[1]) * g;
    if (lane == 0) {
        const float act = a.beta * tanh(gt / a.beta) * sigm(gt) * (a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up);
        A[a.act_off + k * FF + i] = act;
    }
}

// down projection for all TOPK experts at once + weighted sum + post-conv + residual. Q8_0, Q5_0, NV4, or MX8 rows (K = FF).
// lane handles (expert, block) items f = lane + 32*s (TOPK*36 = 288 = 9*32); packed (unaligned) loads.
#define DN_NSG 2
#define DN_NR  2

template <int NR>
inline void down_rows(device const uchar * eb, uint row0, uint row_bytes, uint type, int b, thread const float4 * xv, float wk, thread float * acc) {
    for (int rr = 0; rr < NR; rr++) {
        device const uchar * rowp = eb + (ulong) (row0 + rr) * row_bytes;
        float sum = 0;
        if (type == 200) {
            device const uchar * bq = rowp + b * 18;
            device const packed_uchar4 * q4 = (device const packed_uchar4 *) (bq + 2);
            float al = 0, ah = 0;
            for (int i = 0; i < 4; i++) { const uchar4 q = uchar4(q4[i]); al += dot(xv[i], fp4f4(q & 0xF)); ah += dot(xv[4 + i], fp4f4(q >> 4)); }
            sum = e4m3f(bq[0]) * al + e4m3f(bq[1]) * ah;
        } else if (type == 201) {
            device const uchar * bq = rowp + b * 34;
            device const packed_uchar4 * q4 = (device const packed_uchar4 *) (bq + 2);
            for (int i = 0; i < 8; i++) sum += dot(xv[i], e4m3f4(uchar4(q4[i])));
            sum *= float(*(device const half *) bq) * MX8_K;
        } else if (type == 8) {
            device const block_q8_0 * bq = (device const block_q8_0 *) rowp + b;
            device const packed_char4 * q = (device const packed_char4 *) bq->qs;
            for (int i = 0; i < 8; i++) sum += dot(xv[i], float4(char4(q[i])));
            sum *= float(bq->d);
        } else {
            device const block_q5_0 * bq = (device const block_q5_0 *) rowp + b;
            const uint qh = as_type<uint>(uchar4(*(device const packed_uchar4 *) bq->qh));
            device const packed_uchar4 * qs4 = (device const packed_uchar4 *) bq->qs;
            for (int i = 0; i < 4; i++) {
                const uchar4 q = uchar4(qs4[i]);
                const uint hl = qh >> (4 * i), hh = qh >> (4 * i + 16);
                const float4 lo = float4(int4(uint4(q & 0xF) | ((uint4(hl, hl >> 1, hl >> 2, hl >> 3) & 1) << 4))) - 16.0f;
                const float4 hi = float4(int4(uint4(q >> 4) | ((uint4(hh, hh >> 1, hh >> 2, hh >> 3) & 1) << 4))) - 16.0f;
                sum += dot(xv[i], lo) + dot(xv[4 + i], hi);
            }
            sum *= float(bq->d);
        }
        acc[rr] += wk * sum;
    }
}

kernel void k_moe_down(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                       device float * CS [[buffer(4)]], constant MoeArgs & a [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * DN_NSG + sg) * DN_NR;
    const ulong base = ulong(a.dn_hi) << 32 | a.dn_lo;
    float acc[DN_NR] = {0, 0};
#ifdef NV4_BAL
    if (a.dn_type == 200) {
        const short ix = lane / 4, il = lane % 4;
        for (int s = 0; s < 36; s++) {
            const int f = ix + 8 * s, k = f / 36, b = f % 36;
            device const float * xb = A + a.act_off + k * FF + b * 32 + il * 4;
            const float4 xl = *(device const float4 *) xb, xh = *(device const float4 *) (xb + 16);
            const int e = (int) A[a.sel_off + k];
            const float wk = A[a.sel_off + TOPK + k] * gscale_at(W, a.dn_gs_lo, a.dn_gs_hi, e) * NV4_K;
            device const uchar * eb = W + base + (ulong) e * a.dn_expert_bytes + b * 18;
            for (int rr = 0; rr < DN_NR; rr++) {
                device const ushort * p = (device const ushort *) (eb + (ulong) (row0 + rr) * a.dn_row_bytes);
                const float2 sc = e4m3f2(p[0]);
                float4 lo, hi; fp4x8(uint(p[1 + 2 * il]) | (uint(p[2 + 2 * il]) << 16), lo, hi);
                acc[rr] += wk * (sc.x * dot(xl, lo) + sc.y * dot(xh, hi));
            }
        }
    } else if (a.dn_type == 201) {
        const short ix = lane / 4, il = lane % 4;
        for (int s = 0; s < 36; s++) {
            const int f = ix + 8 * s, k = f / 36, b = f % 36;
            device const float4 * x4 = (device const float4 *) (A + a.act_off + k * FF + b * 32 + il * 8);
            const float4 xa = x4[0], xb = x4[1];
            const int e = (int) A[a.sel_off + k];
            const float wk = A[a.sel_off + TOPK + k] * MX8_K;
            device const uchar * eb = W + base + (ulong) e * a.dn_expert_bytes + b * 34;
            for (int rr = 0; rr < DN_NR; rr++) {
                device const ushort * p = (device const ushort *) (eb + (ulong) (row0 + rr) * a.dn_row_bytes);
                float4 q0, q1; e4m3x8(p + 1 + 4 * il, q0, q1);
                acc[rr] += wk * float(as_type<half>(p[0])) * (dot(xa, q0) + dot(xb, q1));
            }
        }
    } else
#endif
    for (int s = 0; s < 9; s++) {
        const int f = lane + 32 * s, k = f / 36, b = f % 36;
        device const float4 * x4 = (device const float4 *) (A + a.act_off + k * FF + b * 32);
        float4 xv[8];
        for (int i = 0; i < 8; i++) xv[i] = x4[i];
        const int e = (int) A[a.sel_off + k];
        const float g = a.dn_type == 200 ? gscale_at(W, a.dn_gs_lo, a.dn_gs_hi, e) * NV4_K : 1.0f;
        down_rows<DN_NR>(W + base + (ulong) e * a.dn_expert_bytes, row0, a.dn_row_bytes, a.dn_type, b, xv, A[a.sel_off + TOPK + k] * g, acc);
    }
    for (int rr = 0; rr < DN_NR; rr++) {
        const float y = simd_sum(acc[rr]);
        if (lane == 0) {
            const uint row = row0 + rr;
            device float * h = CS + a.cs_off + row * 3;
            device const float * w = (device const float *) (PB + a.convw_off) + row * 4;
            const float h0 = h[0], h1 = h[1], h2 = h[2];
            const float conv = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * y;
            if (a.no_state_write == 0u) { h[0] = h1; h[1] = h2; h[2] = y; }
            A[a.out_off + row] = A[a.res_off + row] + (y + conv) * 0.4472135954999579f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// argmax over logits (two stages). stage 1: grid G TGs x 1024 threads -> partial (val, idx). stage 2: 1 TG.
struct ArgmaxArgs { uint logits_off, n, part_off, nparts, out_index; uint pad0, pad1, pad2; };

kernel void k_argmax1(device float * A [[buffer(2)]], constant ArgmaxArgs & a [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32]; threadgroup int bi[32];
    float best = -INFINITY; int idx = 0;
    for (uint i = tg * 1024 + tid; i < a.n; i += ntg * 1024) { const float v = A[a.logits_off + i]; if (v > best) { best = v; idx = i; } }
    const float m = simd_max(best);
    int c = best == m ? idx : 0x7fffffff; c = simd_min(c);
    if (lane == 0) { bv[sg] = m; bi[sg] = c; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = bv[lane]; int ii = bi[lane];
        const float m2 = simd_max(v);
        int c2 = v == m2 ? ii : 0x7fffffff; c2 = simd_min(c2);
        if (lane == 0) { A[a.part_off + 2 * tg] = m2; A[a.part_off + 2 * tg + 1] = as_type<float>(c2); }
    }
}

kernel void k_argmax2(device float * A [[buffer(2)]], constant ArgmaxArgs & a [[buffer(5)]], device int * toks [[buffer(6)]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    float best = -INFINITY; int idx = 0x7fffffff;
    for (uint i = lane; i < a.nparts; i += 32) {
        const float v = A[a.part_off + 2 * i]; const int ii = as_type<int>(A[a.part_off + 2 * i + 1]);
        if (v > best || (v == best && ii < idx)) { best = v; idx = ii; }
    }
    const float m = simd_max(best);
    int c = best == m ? idx : 0x7fffffff; c = simd_min(c);
    if (lane == 0) toks[a.out_index] = uint(c) < a.n ? c : 0;
}

// GPU Gumbel-max sampling (chadlyb/erisinky): argmax(softcap(l) / T + G), G = -log(-log(u)),
// draws from softmax(softcap(l) / T) when top_k == 0 && top_p >= 1.0.
// ban_tok != 0xFFFFFFFFu masks that token to -INFINITY (used by hard thinking_budget close).
struct SampArgs { uint logits_off, n, seed_lo, seed_hi, pos, ban_tok; float temp, softcap; };
inline uint mix32(uint h) { h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16; return h; }
kernel void k_gumbel(device float * A [[buffer(2)]], constant SampArgs & a [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    float x = A[a.logits_off + i];
    if (i == a.ban_tok) { A[a.logits_off + i] = -INFINITY; return; }
    if (a.softcap > 0.0f) x = a.softcap * precise::tanh(x / a.softcap);
    if (a.temp > 0.0f) {
        const uint h = mix32(mix32(mix32(mix32(i ^ 0x9e3779b9u) ^ a.pos) ^ a.seed_lo) ^ a.seed_hi);
        // u in (0, 1), never 1: 23 random bits plus a half step are exact in binary32
        const float u = (float(h >> 9) + 0.5f) * (1.0f / 8388608.0f);
        x = x / a.temp - precise::log(-precise::log(u));
    }
    A[a.logits_off + i] = x;
}

// Adaptive LoopCD-Logits (Liu et al., 2026, arXiv:2610.02185, Eqs. 3-4 & Appendix A.1-A.3)
// with exact logit softcapping: computes s3 = softcap(Z3), s1 = softcap(Z1), exact global
// top-2 probabilities (p_{3,(1)}, p_{3,(2)}) of softmax(s3), adaptive weight
//   omega_t = omega_max * clamp(1 - (p_{3,(1)} - p_{3,(2)}), 0, 1)
// and overwrites Z3[v] <- s3(v) + omega_t * (s3(v) - s1(v)) in-place.
// Downstream k_gumbel / CPU Sampler skips softcap only for rows with omega_max > 0.
// Grid: (1, n_lanes, 1), 1024 threads (32 simdgroups).
struct LoopCdArgs { uint logits_off, n, stride, pad0; float softcap; uint pad1, pad2, pad3; };
inline void top2_merge(thread float & m1, thread float & m2, float b1, float b2) {
    if (b1 >= m1) { m2 = max(m1, b2); m1 = b1; }
    else if (b1 > m2) { m2 = b1; }
}
kernel void k_loopcd_blend(device float * A [[buffer(2)]],
                           constant LoopCdArgs & a [[buffer(5)]],
                           constant float * OMEGA [[buffer(8)]],
                           device const float * LG_WEAK [[buffer(9)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    if (a.n == 0u) return;
    const uint si = tgp.y;
    const ulong row_off = ulong(si) * ulong(a.stride);
    device float * Z3 = A + ulong(a.logits_off) + row_off;
    device const float * Z1 = LG_WEAK + row_off;
    const float omega_max = OMEGA[si];
    const float c = a.softcap;

    threadgroup float sm1[32], sm2[32], ssum[32];
    threadgroup float g_shared[3];

    // Disabled lanes (omega_max <= 0) keep exact raw logits untouched so mixed batches
    // never alter an unguided lane's arithmetic relative to standalone omega == 0 decode.
    if (omega_max <= 0.0f) return;

    // Pass 1: exact global top-2 of softcapped strong logits s3 across all 1024 threads.
    float m1 = -INFINITY, m2 = -INFINITY;
    for (uint v = tid; v < a.n; v += 1024u) {
        const float raw = Z3[v];
        const float s3 = c > 0.0f ? c * precise::tanh(raw / c) : raw;
        if (s3 >= m1) { m2 = m1; m1 = s3; }
        else if (s3 > m2) { m2 = s3; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float o1 = simd_shuffle_xor(m1, off);
        const float o2 = simd_shuffle_xor(m2, off);
        top2_merge(m1, m2, o1, o2);
    }
    if (lane == 0) { sm1[sg] = m1; sm2[sg] = m2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float bm1 = sm1[lane], bm2 = sm2[lane];
        for (ushort off = 16; off > 0; off >>= 1) {
            const float o1 = simd_shuffle_xor(bm1, off);
            const float o2 = simd_shuffle_xor(bm2, off);
            top2_merge(bm1, bm2, o1, o2);
        }
        if (lane == 0) { g_shared[0] = bm1; g_shared[1] = bm2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: exact softmax partition function Z = sum_v exp(s3(v) - max1) and adaptive omega_t.
    const float max1 = g_shared[0];
    const float max2 = g_shared[1];
    float sum_e = 0.0f;
    for (uint v = tid; v < a.n; v += 1024u) {
        const float raw = Z3[v];
        const float s3 = c > 0.0f ? c * precise::tanh(raw / c) : raw;
        sum_e += precise::exp(s3 - max1);
    }
    const float sg_sum = simd_sum(sum_e);
    if (lane == 0) ssum[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float z = simd_sum(ssum[lane]);
        if (lane == 0) {
            const bool z_ok = ((as_type<uint>(z) & 0x7f800000u) != 0x7f800000u) && (z > 0.0f);
            const float p1 = z_ok ? (1.0f / z) : 1.0f;
            const float p2 = (z_ok && max2 > -1e30f) ? (precise::exp(max2 - max1) / z) : 0.0f;
            g_shared[2] = z_ok ? (omega_max * clamp(1.0f - (p1 - p2), 0.0f, 1.0f)) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: write guided post-softcap logits s3 + omega_t * (s3 - s1) back to Z3.
    const float omega = g_shared[2];
    for (uint v = tid; v < a.n; v += 1024u) {
        const float raw3 = Z3[v], raw1 = Z1[v];
        if ((as_type<uint>(raw3) & 0x7f800000u) == 0x7f800000u ||
            (as_type<uint>(raw1) & 0x7f800000u) == 0x7f800000u) {
            Z3[v] = as_type<float>(0x7fc00000u);
            continue;
        }
        const float s3 = c > 0.0f ? c * precise::tanh(raw3 / c) : raw3;
        if (omega > 0.0f) {
            const float s1 = c > 0.0f ? c * precise::tanh(raw1 / c) : raw1;
            Z3[v] = s3 + omega * (s3 - s1);
        } else {
            Z3[v] = s3;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Layer 3 block-addressed KV cache helpers and decode split-K attention (`k_attn_blk`).
// In-block layout matches `src/blocks.rs::key_row` and `specs/InBlock.tla`:
//   [pass][K|V][head g in 0..NKV][pos in 0..64][d in 0..HD]
// Each 64-bit block-table entry stores `(slab_index << 62) | within_slab_byte_offset`.

#define KV_BLK 64u
#define KV_BLK_MASK 63u
#define KV_SLAB_MASK ((1UL << 62) - 1UL)
#define KV_HEAD_BYTES  (ulong(KV_BLK) * ulong(HD) * ulong(sizeof(kv_t)))
#define KV_PLANE_BYTES (ulong(NKV) * KV_HEAD_BYTES)
#define KV_PASS_BYTES  (2UL * KV_PLANE_BYTES)

inline device kv_t * kv_blk_ptr(
    device uchar * s0, device uchar * s1, device uchar * s2, device uchar * s3,
    device const ulong * bt, uint blk, ulong in_slot_off)
{
    const ulong entry = bt[blk];
    const uint si = uint(entry >> 62);
    const ulong off = entry & KV_SLAB_MASK;
    device uchar * base = (si == 0) ? s0 : (si == 1 ? s1 : (si == 2 ? s2 : s3));
    return (device kv_t *) (base + off + in_slot_off);
}

#pragma clang fp contract(off) reassociate(off)
// Block-addressed split-K decode attention: identical arithmetic to `k_attn`, reading/writing
// through the lane's 64-bit block table `BT` into the pool slabs `KV0..KV3`. `a.k_base` is `pass`.
kernel void k_attn_blk(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                       device uchar * KV0 [[buffer(3)]], device float * CS [[buffer(4)]],
                       constant AttnArgs & a [[buffer(5)]], device atomic_uint * CTR [[buffer(7)]],
                       device const ulong * BT [[buffer(8)]],
                       device uchar * KV1 [[buffer(10)]], device uchar * KV2 [[buffer(11)]], device uchar * KV3 [[buffer(12)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float kc[HD], vc[HD];
    threadgroup float red[4][8];
    const uint sp = tgp.x, g = tgp.y;
    const uint p_lo = a.first_pos;
    const uint p_hi = a.first_pos + a.nkeys;
    const uint split_start = ((a.pad & 4u) != 0u) ? (p_lo + sp * a.kps) : ((p_lo / a.kps + sp) * a.kps);
    const uint split_end = split_start + a.kps;
    const uint p1 = min(p_hi, split_end);
    const bool owns = (p1 == p_hi) && a.noown == 0;
    const bool dry = (a.pad & 2u) != 0u;
    const ulong koff = ulong(a.k_base) * KV_PASS_BYTES + ulong(g) * KV_HEAD_BYTES;
    const ulong voff = koff + KV_PLANE_BYTES;
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) { const uint H = g * 3 + h; q[h] = A[a.q_off + H * HD + tid] * qclip[H * HD + tid]; }
    float kx = 0, vx = 0;
    if (owns) {
        const uint c = g * HD + tid;
        const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            const uint ch = which * (NKV * HD) + c;
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_w : a.vconv_w)) + c * 4;
            device float * hh = CS + a.cs_off + ch * 3;
            const float x = A[(which == 0 ? a.k_off : a.v_off) + c], h0 = hh[0], h1 = hh[1], h2 = hh[2];
            float acc = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * x;
            const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
            if (dyn >= 0) { device const float * t = A + dyn + g * 4; acc += t[0] * x + t[1] * h2 + t[2] * h1 + t[3] * h0; }
            if (!dry) { hh[0] = h1; hh[1] = h2; hh[2] = x; }
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[a.r_off + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        const float s5 = simd_max(fabs(kx)), s6 = simd_max(fabs(vx));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * a.tau * 0.08838834764831845f;
        ((threadgroup float *) qs4[h])[tid] = q[h] * f;
    }
    if (tid < 48) rs[tid / 16][tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * a.tau;
    if (owns) {
        const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
        const float k = kx * kf, v = vx * vf;
#ifdef KV8
        const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
        const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
        const float sk = kv_scale(ck), sv = kv_scale(cv);
        const float qk = kv_quant(k, sk, ck, tid), qv = kv_quant(v, sv, cv, tid);
        kc[tid] = qk * sk; vc[tid] = qv * sv;
        if (!dry) {
            const uint blk = a.pos >> 6, ib = a.pos & KV_BLK_MASK;
            device kv_t * kp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, koff) + ib * HD;
            device kv_t * vp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, voff) + ib * HD;
            kp[tid] = (kv_t) (int) qk;
            vp[tid] = (kv_t) (int) qv;
        }
#else
        kc[tid] = k; vc[tid] = v;
        if (!dry) {
            const uint blk = a.pos >> 6, ib = a.pos & KV_BLK_MASK;
            device kv_t * kp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, koff) + ib * HD;
            device kv_t * vp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, voff) + ib * HD;
            kp[tid] = half(k);
            vp[tid] = half(v);
        }
#endif
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, l[3] = {0, 0, 0};
    float4 o[3] = {0, 0, 0};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);
    for (uint pb = split_start + sg * 8; pb < p1; pb += 32) {
        if (pb + 8 <= p_lo) continue;
        const uint raw_p = pb + kk;
        const bool valid = (raw_p >= p_lo) && (raw_p < p1);
        const uint p = min(max(raw_p, p_lo), p1 - 1);
        float acc[3] = {0, 0, 0};
        if (p == a.pos) {
            for (int c = 0; c < 8; c++) {
                const float4 kf = *(threadgroup const float4 *) (&kc[pp * 32 + c * 4]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
        } else {
            device const kv_t * krow = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, p >> 6, koff) + (p & KV_BLK_MASK) * HD;
#ifdef KV8
            device const packed_char4 * kr = (device const packed_char4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(char4(kr[c]));
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
            const float ks = kv_vscale(krow);
            for (int h = 0; h < 3; h++) acc[h] *= ks;
#else
            device const half4 * kr = (device const half4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
#endif
        }
        const uint d = a.pos - p;
        if (d < EXT) {
            const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4);
            for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
        }
        if (a.lastbias != 0.0f && valid && p == p_hi - 1) for (int h = 0; h < 3; h++) acc[h] += a.lastbias * 0.25f;
        float pr[3], sc_[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? t - sink[h] : -INFINITY;
            const float mx = max(m[h], simd_max(t));
            sc_[h] = (m[h] == -INFINITY) ? 0.0f : exp(m[h] - mx);
            pr[h] = valid ? exp(t - mx) : 0.0f;
            l[h] = l[h] * sc_[h] + simd_sum(pr[h]) * 0.25f;
            m[h] = mx;
            o[h] *= sc_[h];
        }
#ifdef KV8
        {
            const float vsc = (p == a.pos) ? 1.0f : kv_vscale(kv_blk_ptr(KV0, KV1, KV2, KV3, BT, p >> 6, voff) + (p & KV_BLK_MASK) * HD);
            for (int h = 0; h < 3; h++) pr[h] *= vsc;
        }
#endif
        for (ushort q = 0; q < 8; q++) {
            const uint pq = pb + q;
            if (pq < p_lo) continue;
            if (pq >= p1) break;
            const float4 v = (pq == a.pos) ? *(threadgroup const float4 *) (&vc[lane * 4])
#ifdef KV8
                                           : float4(char4(*(device const packed_char4 *) (kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pq >> 6, voff) + (pq & KV_BLK_MASK) * HD + lane * 4)));
#else
                                           : float4(*(device const half4 *) (kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pq >> 6, voff) + (pq & KV_BLK_MASK) * HD + lane * 4));
#endif
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q * 4) * v;
        }
    }
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int h = 0; h < 3; h++) {
        float M = -INFINITY;
        for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
        float L = 0, O = 0;
        for (int s2 = 0; s2 < 4; s2++) {
            const float w = ms_[s2][h] == -INFINITY ? 0.0f : exp(ms_[s2][h] - M);
            L += w * ls_[s2][h];
            O += w * ((threadgroup const float *) os_[s2][h])[tid];
        }
        device float * P = A + a.part_off + ((g * 3 + h) * a.nsplit + a.sp0 + sp) * (HD + 2);
        if (tid == 0) { P[0] = M; P[1] = L; }
        P[2 + tid] = O;
    }
    threadgroup uint flag;
    if ((a.pad & 1u) == 0u && last_tg(CTR + a.ctr + g, a.nsplit, tid, &flag)) {
        for (int h = 0; h < 3; h++) {
            const uint H = g * 3 + h;
            device const float * P = A + a.part_off + H * a.nsplit * (HD + 2);
            float M = 0.0f;
            for (uint s2 = 0; s2 < a.nsplit; s2++) M = max(M, P[s2 * (HD + 2)]);
            float L = exp(-M), O = 0;
            for (uint s2 = 0; s2 < a.nsplit; s2++) {
                const float w = exp(P[s2 * (HD + 2)] - M);
                L += w * P[s2 * (HD + 2) + 1];
                O += w * P[s2 * (HD + 2) + 2 + tid];
            }
            const uint i = H * HD + tid;
            A[a.out_off + i] = O / L * sigm(A[a.gate_off + i]);
        }
    }
}

// `swa` bit 0: sliding-window layer; bit 1: no_state_write (LoopCD dry weak pass: compute exact kc/vc in TG memory
// without mutating CS or global KV).
struct AttnBlkDecArgs {
    uint desc_off, pass_idx, cs_slot, swa;
    uint q_off, k_off, v_off, r_off;
    uint sink_off; int kdyn_off, vdyn_off; uint kconv_w;
    uint vconv_w, qclip_w, relclip_w, relbias_w;
};

// 3D-batched block-addressed split-K decode attention across all lanes in a decode batch:
// grid `(max_nsplit, NKV, nspans)`, 128 threads/TG. Arithmetic is bit-identical to `k_attn_blk`.
kernel void k_attn_blk_dec(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                           device uchar * KV0 [[buffer(3)]], device float * CS [[buffer(4)]],
                           constant AttnBlkDecArgs & a [[buffer(5)]],
                           device const ulong * BT_ALL [[buffer(8)]],
                           device uchar * KV1 [[buffer(10)]], device uchar * KV2 [[buffer(11)]], device uchar * KV3 [[buffer(12)]],
                           uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float kc[HD], vc[HD];
    threadgroup float red[4][8];
    const uint sp = tgp.x, g = tgp.y, si = tgp.z;
    const bool is_swa = (a.swa & 1u) != 0u;
    const bool dry = (a.swa & 2u) != 0u;
    device const DecodeSpanDesc & d = ((device const DecodeSpanDesc *) (A + a.desc_off))[si];
    const uint nsplit = is_swa ? d.swa_nsplit : d.full_nsplit;
    if (sp >= nsplit) return;
    const uint pos = d.pos;
    const uint nkeys = is_swa ? d.swa_nkeys : d.full_nkeys;
    const uint first_pos = is_swa ? d.swa_first_pos : 0u;
    const uint kps = is_swa ? d.swa_kps : d.full_kps;
    const float tau = is_swa ? d.swa_tau : d.full_tau;
    const uint cs_off = d.cs_base + a.cs_slot;
    device const ulong * BT = BT_ALL + (is_swa ? d.bt_win_base : d.bt_full_base);
    const uint q_off = a.q_off + si * (NH * HD);
    const uint k_off = a.k_off + si * (NKV * HD);
    const uint v_off = a.v_off + si * (NKV * HD);
    const uint r_off = a.r_off + si * (NH * NREL);
    const uint sink_off = a.sink_off + si * NH;
    const int kdyn_off = a.kdyn_off >= 0 ? (a.kdyn_off + int(si * 32u)) : -1;
    const int vdyn_off = a.vdyn_off >= 0 ? (a.vdyn_off + int(si * 32u)) : -1;
    const uint part_off = d.part_off;
    const uint p_lo = first_pos;
    const uint p_hi = pos + 1u;
    const uint split_start = ((p_lo / kps) + sp) * kps;
    const uint split_end = split_start + kps;
    const uint p1 = min(p_hi, split_end);
    const bool owns = (p1 == p_hi);
    const ulong koff = ulong(a.pass_idx) * KV_PASS_BYTES + ulong(g) * KV_HEAD_BYTES;
    const ulong voff = koff + KV_PLANE_BYTES;
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) { const uint H = g * 3 + h; q[h] = A[q_off + H * HD + tid] * qclip[H * HD + tid]; }
    float kx = 0, vx = 0;
    if (owns) {
        const uint c = g * HD + tid;
        const float scl = kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            const uint ch = which * (NKV * HD) + c;
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_w : a.vconv_w)) + c * 4;
            device float * hh = CS + cs_off + ch * 3;
            const float x = A[(which == 0 ? k_off : v_off) + c], h0 = hh[0], h1 = hh[1], h2 = hh[2];
            float acc = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * x;
            const int dyn = which == 0 ? kdyn_off : vdyn_off;
            if (dyn >= 0) { device const float * t = A + dyn + g * 4; acc += t[0] * x + t[1] * h2 + t[2] * h1 + t[3] * h0; }
            if (!dry) { hh[0] = h1; hh[1] = h2; hh[2] = x; }
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[r_off + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        const float s5 = simd_max(fabs(kx)), s6 = simd_max(fabs(vx));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        ((threadgroup float *) qs4[h])[tid] = q[h] * f;
    }
    if (tid < 48) rs[tid / 16][tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    if (owns) {
        const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
        const float k = kx * kf, v = vx * vf;
#ifdef KV8
        const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
        const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
        const float sk = kv_scale(ck), sv = kv_scale(cv);
        const float qk = kv_quant(k, sk, ck, tid), qv = kv_quant(v, sv, cv, tid);
        kc[tid] = qk * sk; vc[tid] = qv * sv;
        if (!dry) {
            const uint blk = pos >> 6, ib = pos & KV_BLK_MASK;
            device kv_t * kp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, koff) + ib * HD;
            device kv_t * vp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, voff) + ib * HD;
            kp[tid] = (kv_t) (int) qk;
            vp[tid] = (kv_t) (int) qv;
        }
#else
        kc[tid] = k; vc[tid] = v;
        if (!dry) {
            const uint blk = pos >> 6, ib = pos & KV_BLK_MASK;
            device kv_t * kp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, koff) + ib * HD;
            device kv_t * vp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, voff) + ib * HD;
            kp[tid] = half(k);
            vp[tid] = half(v);
        }
#endif
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[sink_off + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, l[3] = {0, 0, 0};
    float4 o[3] = {0, 0, 0};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);
    for (uint pb = split_start + sg * 8; pb < p1; pb += 32) {
        if (pb + 8 <= p_lo) continue;
        const uint raw_p = pb + kk;
        const bool valid = (raw_p >= p_lo) && (raw_p < p1);
        const uint p = min(max(raw_p, p_lo), p1 - 1);
        float acc[3] = {0, 0, 0};
        if (p == pos) {
            for (int c = 0; c < 8; c++) {
                const float4 kf = *(threadgroup const float4 *) (&kc[pp * 32 + c * 4]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
        } else {
            device const kv_t * krow = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, p >> 6, koff) + (p & KV_BLK_MASK) * HD;
#ifdef KV8
            device const packed_char4 * kr = (device const packed_char4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(char4(kr[c]));
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
            const float ks = kv_vscale(krow);
            for (int h = 0; h < 3; h++) acc[h] *= ks;
#else
            device const half4 * kr = (device const half4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
#endif
        }
        const uint d_pos = pos - p;
        if (d_pos < EXT) {
            const float4 bb = *(device const float4 *) (relb + d_pos * NREL + pp * 4);
            for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
        }
        float pr[3], sc_[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? t - sink[h] : -INFINITY;
            const float mx = max(m[h], simd_max(t));
            sc_[h] = (m[h] == -INFINITY) ? 0.0f : exp(m[h] - mx);
            pr[h] = valid ? exp(t - mx) : 0.0f;
            l[h] = l[h] * sc_[h] + simd_sum(pr[h]) * 0.25f;
            m[h] = mx;
            o[h] *= sc_[h];
        }
#ifdef KV8
        {
            const float vsc = (p == pos) ? 1.0f : kv_vscale(kv_blk_ptr(KV0, KV1, KV2, KV3, BT, p >> 6, voff) + (p & KV_BLK_MASK) * HD);
            for (int h = 0; h < 3; h++) pr[h] *= vsc;
        }
#endif
        for (ushort q = 0; q < 8; q++) {
            const uint pq = pb + q;
            if (pq < p_lo) continue;
            if (pq >= p1) break;
            const float4 v = (pq == pos) ? *(threadgroup const float4 *) (&vc[lane * 4])
#ifdef KV8
                                         : float4(char4(*(device const packed_char4 *) (kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pq >> 6, voff) + (pq & KV_BLK_MASK) * HD + lane * 4)));
#else
                                         : float4(*(device const half4 *) (kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pq >> 6, voff) + (pq & KV_BLK_MASK) * HD + lane * 4));
#endif
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q * 4) * v;
        }
    }
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int h = 0; h < 3; h++) {
        float M = -INFINITY;
        for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
        float L = 0, O = 0;
        for (int s2 = 0; s2 < 4; s2++) {
            const float w = ms_[s2][h] == -INFINITY ? 0.0f : exp(ms_[s2][h] - M);
            L += w * ls_[s2][h];
            O += w * ((threadgroup const float *) os_[s2][h])[tid];
        }
        device float * P = A + part_off + ((g * 3 + h) * nsplit + sp) * (HD + 2);
        if (tid == 0) { P[0] = M; P[1] = L; }
        P[2 + tid] = O;
    }
}

// ---------------------------------------------------------------------------------------------
// Layer 3 Multi-Query Shared-KV Grouped Decode (`k_kvprep_dec` + `k_attn_grp_dec`, G_MAX = 8).
// `DecAttnGroup` (24 bytes, mirrors Rust `DecAttnGroup`): groups up to 8 decode lanes sharing
// identical `(slot, generation)` BlockEntries across an interior cache-fixed absolute split.

#define DEC_GRP_MAX 8u

struct DecAttnGroup {
    uint   abs_sp;
    uint   count;
    ushort span_idx[DEC_GRP_MAX];
};

// `swa` bit 0: sliding-window layer; bit 1: dry_state (LoopCD weak pass: compute exact kc/vc into
// `kv_cur_off` without mutating CS or BlockPool KV).
struct KvPrepDecArgs {
    uint desc_off, pass_idx, cs_slot, swa;
    uint q_off, k_off, v_off, r_off;
    uint qp_off, rp_off, kv_cur_off; int kdyn_off;
    int  vdyn_off; uint kconv_w, vconv_w, qclip_w;
    uint relclip_w, pad0, pad1, pad2;
};

struct AttnGrpDecArgs {
    uint desc_off, grp_off, qp_off, rp_off;
    uint kv_cur_off, sink_off, relbias_w, pass_idx;
    uint kps, swa, pad0, pad1;
};

// Runs ONCE per `(kv_head g, span si)` per attention layer (`grid = (NKV, nspans, 1)`, 128 threads/TG).
// Arithmetic is bit-identical to `k_attn` / `k_attn_blk` phase 1 + KV write.
kernel void k_kvprep_dec(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                         device uchar * KV0 [[buffer(3)]], device float * CS [[buffer(4)]],
                         constant KvPrepDecArgs & a [[buffer(5)]],
                         device const ulong * BT_ALL [[buffer(8)]],
                         device uchar * KV1 [[buffer(10)]], device uchar * KV2 [[buffer(11)]], device uchar * KV3 [[buffer(12)]],
                         uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4][8];
    const uint g = tgp.x, si = tgp.y;
    const bool is_swa = (a.swa & 1u) != 0u;
    const bool dry = (a.swa & 2u) != 0u;
    device const DecodeSpanDesc & d = ((device const DecodeSpanDesc *) (A + a.desc_off))[si];
    const uint pos = d.pos;
    const float tau = is_swa ? d.swa_tau : d.full_tau;
    const uint cs_off = d.cs_base + a.cs_slot;
    device const ulong * BT = BT_ALL + (is_swa ? d.bt_win_base : d.bt_full_base);
    const uint q_off = a.q_off + si * (NH * HD);
    const uint k_off = a.k_off + si * (NKV * HD);
    const uint v_off = a.v_off + si * (NKV * HD);
    const uint r_off = a.r_off + si * (NH * NREL);
    const uint qp_off = a.qp_off + si * (NH * HD);
    const uint rp_off = a.rp_off + si * (NH * NREL);
    const uint kc_off = a.kv_cur_off + si * (2 * NKV * HD) + g * HD;
    const uint vc_off = a.kv_cur_off + si * (2 * NKV * HD) + NKV * HD + g * HD;
    const int kdyn_off = a.kdyn_off >= 0 ? (a.kdyn_off + int(si * 32u)) : -1;
    const int vdyn_off = a.vdyn_off >= 0 ? (a.vdyn_off + int(si * 32u)) : -1;
    const ulong koff = ulong(a.pass_idx) * KV_PASS_BYTES + ulong(g) * KV_HEAD_BYTES;
    const ulong voff = koff + KV_PLANE_BYTES;
    device const float * qclip = (device const float *) (PB + a.qclip_w);

    float q[3];
    for (int h = 0; h < 3; h++) { const uint H = g * 3 + h; q[h] = A[q_off + H * HD + tid] * qclip[H * HD + tid]; }

    float kx = 0, vx = 0;
    {
        const uint c = g * HD + tid;
        const float scl = kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            const uint ch = which * (NKV * HD) + c;
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_w : a.vconv_w)) + c * 4;
            device float * hh = CS + cs_off + ch * 3;
            const float x = A[(which == 0 ? k_off : v_off) + c], h0 = hh[0], h1 = hh[1], h2 = hh[2];
            float acc = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * x;
            const int dyn = which == 0 ? kdyn_off : vdyn_off;
            if (dyn >= 0) { device const float * t = A + dyn + g * 4; acc += t[0] * x + t[1] * h2 + t[2] * h1 + t[3] * h0; }
            if (!dry) { hh[0] = h1; hh[1] = h2; hh[2] = x; }
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }

    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[r_off + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        const float s5 = simd_max(fabs(kx)), s6 = simd_max(fabs(vx));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];

    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        A[qp_off + H * HD + tid] = q[h] * f;
    }
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        A[rp_off + H * NREL + i] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    }

    const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
    const float k = kx * kf, v = vx * vf;
#ifdef KV8
    const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
    const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
    const float sk = kv_scale(ck), sv = kv_scale(cv);
    const float qk = kv_quant(k, sk, ck, tid), qv = kv_quant(v, sv, cv, tid);
    A[kc_off + tid] = qk * sk;
    A[vc_off + tid] = qv * sv;
    if (!dry) {
        const uint blk = pos >> 6, ib = pos & KV_BLK_MASK;
        device kv_t * kp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, koff) + ib * HD;
        device kv_t * vp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, voff) + ib * HD;
        kp[tid] = (kv_t) (int) qk;
        vp[tid] = (kv_t) (int) qv;
    }
#else
    A[kc_off + tid] = k;
    A[vc_off + tid] = v;
    if (!dry) {
        const uint blk = pos >> 6, ib = pos & KV_BLK_MASK;
        device kv_t * kp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, koff) + ib * HD;
        device kv_t * vp = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, blk, voff) + ib * HD;
        kp[tid] = half(k);
        vp[tid] = half(v);
    }
#endif
}

// Singleton Layer 3 decode attention (`count == 1`: boundary/partial/divergent splits or `INK_NO_GRP_DEC=1`).
// 128 threads/TG, ~7.6 KiB TG memory, low register footprint for full occupancy.
kernel void k_attn_dec_solo(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                            device uchar * KV0 [[buffer(3)]],
                            constant AttnGrpDecArgs & a [[buffer(5)]],
                            device const ulong * BT_ALL [[buffer(8)]],
                            device uchar * KV1 [[buffer(10)]], device uchar * KV2 [[buffer(11)]], device uchar * KV3 [[buffer(12)]],
                            uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float  rs[3][NREL];
    threadgroup float  ms_[4][3];
    threadgroup float  ls_[4][3];
    threadgroup float4 os_[4][3][32];

    const uint gi = tgp.x, g = tgp.y;
    const bool is_swa = (a.swa & 1u) != 0u;
    device const DecAttnGroup & grp = ((device const DecAttnGroup *) (A + a.grp_off))[gi];
    const uint abs_sp = grp.abs_sp;
    const uint split_start = abs_sp * a.kps;
    const uint split_end = split_start + a.kps;
    const uint si0 = uint(grp.span_idx[0]);
    device const DecodeSpanDesc & d0 = ((device const DecodeSpanDesc *) (A + a.desc_off))[si0];

    const uint qp_off = a.qp_off + si0 * (NH * HD);
    const uint rp_off = a.rp_off + si0 * (NH * NREL);
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        ((threadgroup float *) qs4[h])[tid] = A[qp_off + H * HD + tid];
    }
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        rs[h][i] = A[rp_off + H * NREL + i];
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + si0 * NH + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const ulong * BT = BT_ALL + (is_swa ? d0.bt_win_base : d0.bt_full_base);
    const ulong koff = ulong(a.pass_idx) * KV_PASS_BYTES + ulong(g) * KV_HEAD_BYTES;
    const ulong voff = koff + KV_PLANE_BYTES;
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const ushort kk = lane / 4, pp = lane % 4;
    const uint pos = d0.pos;
    const uint p_lo = is_swa ? d0.swa_first_pos : 0u;
    const uint p1 = min(pos + 1u, split_end);
    const uint kc_off = a.kv_cur_off + si0 * (2 * NKV * HD) + g * HD;
    const uint vc_off = a.kv_cur_off + si0 * (2 * NKV * HD) + NKV * HD + g * HD;
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);
    float mv[3] = {-INFINITY, -INFINITY, -INFINITY}, lv[3] = {0, 0, 0};
    float4 ov[3] = {0, 0, 0};
    for (uint pb = split_start + sg * 8; pb < p1; pb += 32) {
        if (pb + 8 <= p_lo) continue;
        const uint raw_p = pb + kk;
        const bool valid = (raw_p >= p_lo) && (raw_p < p1);
        const uint p = min(max(raw_p, p_lo), p1 - 1);
        float acc[3] = {0, 0, 0};
        if (p == pos) {
            for (int c = 0; c < 8; c++) {
                const float4 kf = *(device const float4 *) (A + kc_off + pp * 32 + c * 4);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
        } else {
            device const kv_t * krow = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, p >> 6, koff) + (p & KV_BLK_MASK) * HD;
#ifdef KV8
            device const packed_char4 * kr = (device const packed_char4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(char4(kr[c]));
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
            const float ks = kv_vscale(krow);
            for (int h = 0; h < 3; h++) acc[h] *= ks;
#else
            device const half4 * kr = (device const half4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
#endif
        }
        const uint d_pos = pos - p;
        if (d_pos < EXT) {
            const float4 bb = *(device const float4 *) (relb + d_pos * NREL + pp * 4);
            for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
        }
        float pr[3], sc_[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? t - sink[h] : -INFINITY;
            const float mx = max(mv[h], simd_max(t));
            sc_[h] = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mx);
            pr[h] = valid ? exp(t - mx) : 0.0f;
            lv[h] = lv[h] * sc_[h] + simd_sum(pr[h]) * 0.25f;
            mv[h] = mx;
            ov[h] *= sc_[h];
        }
#ifdef KV8
        {
            const float vsc = (p == pos) ? 1.0f : kv_vscale(kv_blk_ptr(KV0, KV1, KV2, KV3, BT, p >> 6, voff) + (p & KV_BLK_MASK) * HD);
            for (int h = 0; h < 3; h++) pr[h] *= vsc;
        }
#endif
        for (ushort q = 0; q < 8; q++) {
            const uint pq = pb + q;
            if (pq < p_lo) continue;
            if (pq >= p1) break;
            const float4 v = (pq == pos) ? *(device const float4 *) (A + vc_off + lane * 4)
#ifdef KV8
                                         : float4(char4(*(device const packed_char4 *) (kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pq >> 6, voff) + (pq & KV_BLK_MASK) * HD + lane * 4)));
#else
                                         : float4(*(device const half4 *) (kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pq >> 6, voff) + (pq & KV_BLK_MASK) * HD + lane * 4));
#endif
            for (int h = 0; h < 3; h++) ov[h] += simd_shuffle(pr[h], q * 4) * v;
        }
    }
    if (lane == 0) for (int h = 0; h < 3; h++) { ms_[sg][h] = mv[h]; ls_[sg][h] = lv[h]; }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = ov[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsplit = is_swa ? d0.swa_nsplit : d0.full_nsplit;
    const uint local_sp = is_swa ? (abs_sp - (d0.swa_first_pos >> 6)) : abs_sp;
    for (int h = 0; h < 3; h++) {
        float M_val = -INFINITY;
        for (int s2 = 0; s2 < 4; s2++) M_val = max(M_val, ms_[s2][h]);
        float L = 0, O = 0;
        for (int s2 = 0; s2 < 4; s2++) {
            const float w = ms_[s2][h] == -INFINITY ? 0.0f : exp(ms_[s2][h] - M_val);
            L += w * ls_[s2][h];
            O += w * ((threadgroup const float *) os_[s2][h])[tid];
        }
        device float * P = A + d0.part_off + ((g * 3 + h) * nsplit + local_sp) * (HD + 2);
        if (tid == 0) { P[0] = M_val; P[1] = L; }
        P[2 + tid] = O;
    }
}

union GrpSharedTile {
    struct {
        kv_t k[32][HD];
        kv_t v[32][HD];
    } kv;
    struct {
        float4 os[4][3][32];
        float  ms[4][3];
        float  ls[4][3];
    } red;
};

// Multi-query shared-KV grouped decode attention (`count` in `2..=8`, `grid = (n_shared_groups, NKV, 1)`, 512 threads/TG).
// 16 SIMD groups = 4 key stripes (`ksg = sg & 3`, exact `k_attn` order) x 4 lane pairs (`lsg = sg >> 2`, <= 2 lanes/SG).
// Cooperatively stages 32 K + 32 V vectors from DRAM ONCE into TG SRAM (`16.00 KiB` fp16 / `8.25 KiB` KV8) for all `m <= 8`
// lanes, fitting 2 TGs/core (1024 threads/core) on M5 with <= 60 regs/thread (zero register spills).
// Bit-for-bit identical to `k_attn` / `k_attn_dec_solo` on every lane and every split.
kernel void k_attn_grp_dec(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                           device uchar * KV0 [[buffer(3)]],
                           constant AttnGrpDecArgs & a [[buffer(5)]],
                           device const ulong * BT_ALL [[buffer(8)]],
                           device uchar * KV1 [[buffer(10)]], device uchar * KV2 [[buffer(11)]], device uchar * KV3 [[buffer(12)]],
                           uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup GrpSharedTile sh;
#ifdef KV8
    threadgroup float ks_tile[32];
    threadgroup float vs_tile[32];
#endif

    const uint gi = tgp.x, g = tgp.y;
    const bool is_swa = (a.swa & 1u) != 0u;
    device const DecAttnGroup & grp = ((device const DecAttnGroup *) (A + a.grp_off))[gi];
    const uint m = grp.count;
    const uint abs_sp = grp.abs_sp;
    const uint split_start = abs_sp * a.kps;
    const uint split_end = split_start + a.kps;
    device const DecodeSpanDesc * descs = (device const DecodeSpanDesc *) (A + a.desc_off);

    const uint si0 = uint(grp.span_idx[0]);
    device const DecodeSpanDesc & d0 = descs[si0];
    device const ulong * BT = BT_ALL + (is_swa ? d0.bt_win_base : d0.bt_full_base);
    const ulong koff = ulong(a.pass_idx) * KV_PASS_BYTES + ulong(g) * KV_HEAD_BYTES;
    const ulong voff = koff + KV_PLANE_BYTES;
    device const float * relb = (device const float *) (PB + a.relbias_w);

    const ushort ksg = sg & 3;
    const ushort lsg = sg >> 2;
    const ushort kk = lane / 4, pp = lane % 4;
    const uint u0 = uint(lsg) * 2u;
    const uint u1 = u0 + 1u;
    const bool has0 = u0 < m;
    const bool has1 = u1 < m;

    const uint si_u0 = has0 ? uint(grp.span_idx[u0]) : 0u;
    const uint si_u1 = has1 ? uint(grp.span_idx[u1]) : 0u;
    device const float4 * qp0 = (device const float4 *) (A + a.qp_off + si_u0 * (NH * HD) + (g * 3) * HD);
    device const float4 * qp1 = (device const float4 *) (A + a.qp_off + si_u1 * (NH * HD) + (g * 3) * HD);
    device const float * rp0 = A + a.rp_off + si_u0 * (NH * NREL) + (g * 3) * NREL;
    device const float * rp1 = A + a.rp_off + si_u1 * (NH * NREL) + (g * 3) * NREL;

    float mv0[3] = {-INFINITY, -INFINITY, -INFINITY}, lv0[3] = {0, 0, 0};
    float4 ov0[3] = {0, 0, 0};
    float mv1[3] = {-INFINITY, -INFINITY, -INFINITY}, lv1[3] = {0, 0, 0};
    float4 ov1[3] = {0, 0, 0};

    const uint pos0 = has0 ? descs[si_u0].pos : 0u;
    const uint pos1 = has1 ? descs[si_u1].pos : 0u;
    float s0[3], s1[3];
    if (has0) for (int h = 0; h < 3; h++) s0[h] = A[a.sink_off + si_u0 * NH + g * 3 + h];
    if (has1) for (int h = 0; h < 3; h++) s1[h] = A[a.sink_off + si_u1 * NH + g * 3 + h];

    for (uint pb_base = split_start; pb_base < split_end; pb_base += 32) {
        if (pb_base > split_start) threadgroup_barrier(mem_flags::mem_threadgroup);
        device const kv_t * k_blk = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pb_base >> 6, koff) + (pb_base & KV_BLK_MASK) * HD;
        device const kv_t * v_blk = kv_blk_ptr(KV0, KV1, KV2, KV3, BT, pb_base >> 6, voff) + (pb_base & KV_BLK_MASK) * HD;
#ifdef KV8
        ((threadgroup char4 *) sh.kv.k)[tid] = char4(((device const packed_char4 *) k_blk)[tid]);
        ((threadgroup char4 *) sh.kv.k)[tid + 512] = char4(((device const packed_char4 *) k_blk)[tid + 512]);
        ((threadgroup char4 *) sh.kv.v)[tid] = char4(((device const packed_char4 *) v_blk)[tid]);
        ((threadgroup char4 *) sh.kv.v)[tid + 512] = char4(((device const packed_char4 *) v_blk)[tid + 512]);
        if (tid < 32) {
            ks_tile[tid] = kv_vscale(k_blk + tid * HD);
            vs_tile[tid] = kv_vscale(v_blk + tid * HD);
        }
#else
        ((threadgroup half4 *) sh.kv.k)[tid] = ((device const half4 *) k_blk)[tid];
        ((threadgroup half4 *) sh.kv.k)[tid + 512] = ((device const half4 *) k_blk)[tid + 512];
        ((threadgroup half4 *) sh.kv.v)[tid] = ((device const half4 *) v_blk)[tid];
        ((threadgroup half4 *) sh.kv.v)[tid + 512] = ((device const half4 *) v_blk)[tid + 512];
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (has0 && has1) {
            const uint row_k = ksg * 8 + kk;
            const uint p = pb_base + row_k;
            float acc0[3] = {0, 0, 0};
            float acc1[3] = {0, 0, 0};
#ifdef KV8
            threadgroup const char4 * kr = (threadgroup const char4 *) (&sh.kv.k[row_k][pp * 32]);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                const uint qidx = pp * 8 + c;
                for (int h = 0; h < 3; h++) {
                    acc0[h] += dot(qp0[h * 32 + qidx], kf);
                    acc1[h] += dot(qp1[h * 32 + qidx], kf);
                }
            }
            const float ks = ks_tile[row_k];
            for (int h = 0; h < 3; h++) { acc0[h] *= ks; acc1[h] *= ks; }
            const float vsc = vs_tile[row_k];
#else
            threadgroup const half4 * kr = (threadgroup const half4 *) (&sh.kv.k[row_k][pp * 32]);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                const uint qidx = pp * 8 + c;
                for (int h = 0; h < 3; h++) {
                    acc0[h] += dot(qp0[h * 32 + qidx], kf);
                    acc1[h] += dot(qp1[h * 32 + qidx], kf);
                }
            }
#endif
            const uint d0_pos = pos0 - p;
            if (d0_pos < EXT) {
                const float4 bb = *(device const float4 *) (relb + d0_pos * NREL + pp * 4);
                for (int h = 0; h < 3; h++) acc0[h] += dot(*(device const float4 *) (rp0 + h * NREL + pp * 4), bb);
            }
            const uint d1_pos = pos1 - p;
            if (d1_pos < EXT) {
                const float4 bb = *(device const float4 *) (relb + d1_pos * NREL + pp * 4);
                for (int h = 0; h < 3; h++) acc1[h] += dot(*(device const float4 *) (rp1 + h * NREL + pp * 4), bb);
            }
            float pr0[3], pr1[3];
            for (int h = 0; h < 3; h++) {
                float t0 = acc0[h];
                t0 += simd_shuffle_xor(t0, 1); t0 += simd_shuffle_xor(t0, 2);
                t0 -= s0[h];
                const float mx0 = max(mv0[h], simd_max(t0));
                const float sc0 = (mv0[h] == -INFINITY) ? 0.0f : exp(mv0[h] - mx0);
                float p0v = exp(t0 - mx0);
                lv0[h] = lv0[h] * sc0 + simd_sum(p0v) * 0.25f;
                mv0[h] = mx0;
                ov0[h] *= sc0;
#ifdef KV8
                p0v *= vsc;
#endif
                pr0[h] = p0v;

                float t1 = acc1[h];
                t1 += simd_shuffle_xor(t1, 1); t1 += simd_shuffle_xor(t1, 2);
                t1 -= s1[h];
                const float mx1 = max(mv1[h], simd_max(t1));
                const float sc1 = (mv1[h] == -INFINITY) ? 0.0f : exp(mv1[h] - mx1);
                float p1v = exp(t1 - mx1);
                lv1[h] = lv1[h] * sc1 + simd_sum(p1v) * 0.25f;
                mv1[h] = mx1;
                ov1[h] *= sc1;
#ifdef KV8
                p1v *= vsc;
#endif
                pr1[h] = p1v;
            }
            for (ushort q = 0; q < 8; q++) {
#ifdef KV8
                const float4 v = float4(*(threadgroup const char4 *) (&sh.kv.v[ksg * 8 + q][lane * 4]));
#else
                const float4 v = float4(*(threadgroup const half4 *) (&sh.kv.v[ksg * 8 + q][lane * 4]));
#endif
                for (int h = 0; h < 3; h++) {
                    ov0[h] += simd_shuffle(pr0[h], q * 4) * v;
                    ov1[h] += simd_shuffle(pr1[h], q * 4) * v;
                }
            }
        } else if (has0) {
            const uint row_k = ksg * 8 + kk;
            const uint p = pb_base + row_k;
            float acc0[3] = {0, 0, 0};
#ifdef KV8
            threadgroup const char4 * kr = (threadgroup const char4 *) (&sh.kv.k[row_k][pp * 32]);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                const uint qidx = pp * 8 + c;
                for (int h = 0; h < 3; h++) acc0[h] += dot(qp0[h * 32 + qidx], kf);
            }
            const float ks = ks_tile[row_k];
            for (int h = 0; h < 3; h++) acc0[h] *= ks;
            const float vsc = vs_tile[row_k];
#else
            threadgroup const half4 * kr = (threadgroup const half4 *) (&sh.kv.k[row_k][pp * 32]);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                const uint qidx = pp * 8 + c;
                for (int h = 0; h < 3; h++) acc0[h] += dot(qp0[h * 32 + qidx], kf);
            }
#endif
            const uint d0_pos = pos0 - p;
            if (d0_pos < EXT) {
                const float4 bb = *(device const float4 *) (relb + d0_pos * NREL + pp * 4);
                for (int h = 0; h < 3; h++) acc0[h] += dot(*(device const float4 *) (rp0 + h * NREL + pp * 4), bb);
            }
            float pr0[3];
            for (int h = 0; h < 3; h++) {
                float t0 = acc0[h];
                t0 += simd_shuffle_xor(t0, 1); t0 += simd_shuffle_xor(t0, 2);
                t0 -= s0[h];
                const float mx0 = max(mv0[h], simd_max(t0));
                const float sc0 = (mv0[h] == -INFINITY) ? 0.0f : exp(mv0[h] - mx0);
                float p0v = exp(t0 - mx0);
                lv0[h] = lv0[h] * sc0 + simd_sum(p0v) * 0.25f;
                mv0[h] = mx0;
                ov0[h] *= sc0;
#ifdef KV8
                p0v *= vsc;
#endif
                pr0[h] = p0v;
            }
            for (ushort q = 0; q < 8; q++) {
#ifdef KV8
                const float4 v = float4(*(threadgroup const char4 *) (&sh.kv.v[ksg * 8 + q][lane * 4]));
#else
                const float4 v = float4(*(threadgroup const half4 *) (&sh.kv.v[ksg * 8 + q][lane * 4]));
#endif
                for (int h = 0; h < 3; h++) ov0[h] += simd_shuffle(pr0[h], q * 4) * v;
            }
        }
    }

    // Epilogue: reuse `sh.red` (aliasing `sh.kv`) across lanes `u = 0..m` with exact 4-stripe reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = 0; u < m; u++) {
        if (u > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uint(lsg) == (u >> 1)) {
            if ((u & 1u) == 0u) {
                if (lane == 0) for (int h = 0; h < 3; h++) { sh.red.ms[ksg][h] = mv0[h]; sh.red.ls[ksg][h] = lv0[h]; }
                for (int h = 0; h < 3; h++) sh.red.os[ksg][h][lane] = ov0[h];
            } else {
                if (lane == 0) for (int h = 0; h < 3; h++) { sh.red.ms[ksg][h] = mv1[h]; sh.red.ls[ksg][h] = lv1[h]; }
                for (int h = 0; h < 3; h++) sh.red.os[ksg][h][lane] = ov1[h];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128u) {
            const uint si = uint(grp.span_idx[u]);
            device const DecodeSpanDesc & du = descs[si];
            const uint nsplit = is_swa ? du.swa_nsplit : du.full_nsplit;
            const uint local_sp = is_swa ? (abs_sp - (du.swa_first_pos >> 6)) : abs_sp;
            for (int h = 0; h < 3; h++) {
                float M_val = -INFINITY;
                for (int s2 = 0; s2 < 4; s2++) M_val = max(M_val, sh.red.ms[s2][h]);
                float L = 0, O = 0;
                for (int s2 = 0; s2 < 4; s2++) {
                    const float w = sh.red.ms[s2][h] == -INFINITY ? 0.0f : exp(sh.red.ms[s2][h] - M_val);
                    L += w * sh.red.ls[s2][h];
                    O += w * ((threadgroup const float *) sh.red.os[s2][h])[tid];
                }
                device float * P = A + du.part_off + ((g * 3 + h) * nsplit + local_sp) * (HD + 2);
                if (tid == 0) { P[0] = M_val; P[1] = L; }
                P[2 + tid] = O;
            }
        }
    }
}
#pragma clang fp contract(fast) reassociate(on)
