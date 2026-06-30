/* tensor/neon_ops.c
 * Freestanding NEON SIMD kernels for Q4_K dequant + f32 matmul.
 * No libc. No memcpy (we write our own). Direct ARM intrinsics.
 * Target: aarch64 Apple M1 / any ARMv8.2-A
 */

#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>

/* ── Freestanding memset/memcpy replacements ─────────────────────────── */

void* titan_memset(void* dst, int c, size_t n) {
    uint8_t* p = (uint8_t*)dst;
    uint8_t val = (uint8_t)c;
    size_t i = 0;
    // NEON bulk clear for aligned regions
    if (n >= 16 && ((uintptr_t)p & 15) == 0) {
        uint8x16_t v = vdupq_n_u8(val);
        for (; i + 16 <= n; i += 16) vst1q_u8(p + i, v);
    }
    for (; i < n; i++) p[i] = val;
    return dst;
}

void* titan_memcpy(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t i = 0;
    if (n >= 16 && ((uintptr_t)d & 15) == 0 && ((uintptr_t)s & 15) == 0) {
        for (; i + 16 <= n; i += 16)
            vst1q_u8(d + i, vld1q_u8(s + i));
    }
    for (; i < n; i++) d[i] = s[i];
    return dst;
}

/* ── Q4_K Dequantize ─────────────────────────────────────────────────── */
/* Q4_K super-block: 256 weights, 2 scales, 2 mins, packed 4-bit nibbles  */

typedef struct {
    uint16_t d;        // super-block scale (fp16)
    uint16_t dmin;     // super-block min  (fp16)
    uint8_t  scales[12]; // 6-bit sub-block scales/mins packed
    uint8_t  qs[128];  // 256 x 4-bit weights
} block_q4_K;

static inline float fp16_to_f32(uint16_t h) {
    // ARM has __fp16 but we stay freestanding-safe with manual conversion
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    uint32_t s = (h >> 15) << 31;
    if (e == 0) { uint32_t v = s | (m << 13); float f; __builtin_memcpy(&f, &v, 4); return f; }
    if (e == 31) { uint32_t v = s | 0x7f800000 | (m << 13); float f; __builtin_memcpy(&f, &v, 4); return f; }
    uint32_t v = s | ((e + 112) << 23) | (m << 13);
    float f; __builtin_memcpy(&f, &v, 4); return f;
}

/* Dequantize one Q4_K block into dst[256] floats */
void dequant_q4_k_block(const block_q4_K* b, float* dst) {
    const float d    = fp16_to_f32(b->d);
    const float dmin = fp16_to_f32(b->dmin);

    // Extract 8 sub-block scales and mins from 12 packed bytes (6-bit each)
    uint8_t sc[8], m[8];
    sc[0] =  b->scales[0] & 0x3f;
    sc[1] =  b->scales[1] & 0x3f;
    sc[2] =  b->scales[2] & 0x3f;
    sc[3] =  b->scales[3] & 0x3f;
    sc[4] = (b->scales[8] & 0x0f) | ((b->scales[4] >> 2) & 0x30);
    sc[5] = (b->scales[9] & 0x0f) | ((b->scales[5] >> 2) & 0x30);
    sc[6] = (b->scales[10]& 0x0f) | ((b->scales[6] >> 2) & 0x30);
    sc[7] = (b->scales[11]& 0x0f) | ((b->scales[7] >> 2) & 0x30);
    m[0]  =  b->scales[4] & 0x3f;
    m[1]  =  b->scales[5] & 0x3f;
    m[2]  =  b->scales[6] & 0x3f;
    m[3]  =  b->scales[7] & 0x3f;
    m[4]  = (b->scales[8]  >> 4) | ((b->scales[4] >> 2) & 0x30);
    m[5]  = (b->scales[9]  >> 4) | ((b->scales[5] >> 2) & 0x30);
    m[6]  = (b->scales[10] >> 4) | ((b->scales[6] >> 2) & 0x30);
    m[7]  = (b->scales[11] >> 4) | ((b->scales[7] >> 2) & 0x30);

    // 8 sub-blocks of 32 weights each
    for (int sub = 0; sub < 8; sub++) {
        const float scale = d * sc[sub];
        const float min_v = dmin * m[sub];
        const uint8_t* qs = b->qs + sub * 16;
        float* out = dst + sub * 32;

        // Unpack 16 bytes → 32 nibbles via NEON
        uint8x16_t raw = vld1q_u8(qs);
        uint8x16_t lo  = vandq_u8(raw, vdupq_n_u8(0x0F));
        uint8x16_t hi  = vshrq_n_u8(raw, 4);

        // Convert to float32 x4 and apply scale/min
        for (int k = 0; k < 16; k++) {
            out[k]      = scale * vgetq_lane_u8(lo, k) - min_v;
            out[k + 16] = scale * vgetq_lane_u8(hi, k) - min_v;
        }
    }
}

/* ── F32 Matrix-Vector Multiply (NEON) ───────────────────────────────── */
/* Computes: out[rows] = mat[rows x cols] * vec[cols]                     */

void matmul_f32_neon(
    const float* __restrict__ mat,  // [rows * cols]
    const float* __restrict__ vec,  // [cols]
    float*       __restrict__ out,  // [rows]
    int rows, int cols
) {
    for (int r = 0; r < rows; r++) {
        const float* row = mat + (size_t)r * cols;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t a = vld1q_f32(row + c);
            float32x4_t b = vld1q_f32(vec + c);
            acc = vmlaq_f32(acc, a, b);
        }
        // Horizontal sum of acc
        float32x2_t s = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        float sum = vget_lane_f32(vpadd_f32(s, s), 0);
        // Scalar tail
        for (; c < cols; c++) sum += row[c] * vec[c];
        out[r] = sum;
    }
}

/* ── Softmax (in-place, f32) ─────────────────────────────────────────── */

void softmax_f32(float* x, int n) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];

    // exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = __builtin_expf(x[i] - max_val);
        sum += x[i];
    }
    // Normalize
    float inv = 1.0f / sum;
    float32x4_t inv4 = vdupq_n_f32(inv);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(x + i, vmulq_f32(v, inv4));
    }
    for (; i < n; i++) x[i] *= inv;
}

/* ── RMS Norm ────────────────────────────────────────────────────────── */

void rmsnorm_f32(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.0f;
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    ss = vget_lane_f32(vpadd_f32(s2, s2), 0);
    for (; i < n; i++) ss += x[i] * x[i];

    ss = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

    for (i = 0; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, vdupq_n_f32(ss)), wv));
    }
    for (; i < n; i++) out[i] = x[i] * ss * w[i];
}
