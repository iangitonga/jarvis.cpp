#pragma once


#include <cstdint>
#include <cmath>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#define JARVIS_ASSERT(condition)  \
    if (!(condition)) { \
        std::fprintf(stderr, "\nJARVIS_ASSERT: %s:%d: %s.\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    }



typedef uint16_t Float16;


namespace fpcvt {

// FP32 <-> FP16 Conversions.
inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

inline float fp16_to_fp32(Float16 h) noexcept
{
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

inline Float16 fp32_to_fp16(float f) noexcept
{
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}


static float* init_fpcvt_cache() {
    // TODO: fix memory leak.
    float* cache = new float[65536];
    Float16 idx = 0;
    for (int i = 0; i < 65536; i++) {
        cache[i] = fp16_to_fp32(idx);
        idx += 1;
    }
    return cache;
}

// Global lookup table for fp16->fp32 to avoid recomputations.
static const float* G_fp16_to_fp32_table = init_fpcvt_cache();

} // namespace fpcvt


// Convert 16-bit float to 32-bit float.
[[nodiscard]]
inline float fp16_to_fp32(Float16 half) {
#if defined(__F16C__)
    return _cvtsh_ss(half);
#else 
    return fpcvt::G_fp16_to_fp32_table[half];
#endif
}

// Convert 32-bit float to 16-bit float.
[[nodiscard]]
inline Float16 fp32_to_fp16(float flt) {
#if defined(__F16C__)
    return _cvtss_sh(flt, 0);
#else
    return fpcvt::fp32_to_fp16(flt);
#endif
}
