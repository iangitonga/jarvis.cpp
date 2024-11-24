#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#include "jarvis_types.h"
#include "utils.h"

#if defined(__AVX__)
#include <immintrin.h>

#define SIMD_AVX_LANES 8
#endif



namespace ops {

struct OpsMetrics {
    int64_t matmul_ms = 0;
    int64_t non_matmul_ms = 0;
};

static OpsMetrics ops_metrics;

void embed(const int* tokens, Float16* emb_table, Float16* out, int n_vocab, int n_ctx, int d_embd, int start_pos)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_ctx; i++) {
        const int emb_table_idx = tokens[i];
        const void* src = emb_table + emb_table_idx * d_embd;
        void* dest = out + i * d_embd;
        const size_t cpy_size = d_embd * sizeof(Float16);
        memcpy(dest, src, cpy_size);
    }
}

// rms_norm(x_i) = x_i * 1/rms(x) * weight(i) where rms(x) = sqrt(1/n * sum(x*x))
// inp   : (n_ctx, d_embd)
// weight: (d_embd)
// out   : (n_ctx, d_embd)
void rms_norm(const Float16* inp, const Float16* weight, Float16* out, int n_ctx, int d_embd, float eps, int start_pos)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_ctx; i++) {
        // compute mean of val squared.
        float sum_squares = 0.0f;
        for (int j = 0; j < d_embd; j++) {
            /// TODO: Use predefined pow fn.
            sum_squares += fp16_to_fp32(inp[i * d_embd + j]) * fp16_to_fp32(inp[i * d_embd + j]);
        }
        const float rms = sqrtf(sum_squares / (float)d_embd);
        const float rsqrt = 1.0f / (rms + eps);
        
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = fp32_to_fp16(fp16_to_fp32(inp[i * d_embd + j]) * rsqrt * fp16_to_fp32(weight[j]));
        }
        // x = xi / (root_mean_sq + 1e-6f) * wi
        // x = x / (rms+eps) * weight
    }
}


void add(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_embd, int start_pos=0)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = fp32_to_fp16(fp16_to_fp32(inp0[i * d_embd + j]) + fp16_to_fp32(inp1[i * d_embd + j]));
        }
    }
}


// inp0: (n_ctx, d_embd)
// inp1: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void mul_inplace(Float16* inp0, const Float16* inp1, int n_ctx, int d_in, int start_pos)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_in; j++) {
            inp0[i * d_in + j] = fp32_to_fp16(fp16_to_fp32(inp0[i * d_in + j]) * fp16_to_fp32(inp1[i * d_in + j]));
        }
    }
}


// inp: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void silu_inplace(Float16* inp, int n_ctx, int d_in, int start_pos)
{
    Timer timer{&ops_metrics.non_matmul_ms};

     for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_in; j++) {
            const float x = fp16_to_fp32(inp[i * d_in + j]);
            inp[i * d_in + j] = fp32_to_fp16(x / (1.0f + expf(-x)));
        }
    }
}


#if defined(__AVX__)
__m256 vec_f32x8_load(const Float16* src_ptr) {
#if defined(__F16C__)
    return _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(const_cast<Float16*>(src_ptr))));
#else
    float f32[SIMD_AVX_LANES];
    for (int i = 0; i < SIMD_AVX_LANES; ++i) {
        f32[i] = fp16_to_fp32(src_ptr[i]);
    }
    return _mm256_loadu_ps(f32);
#endif
}

float avx_reduce_sum(const __m256 x)
{
    const __m128 hi_quad = _mm256_extractf128_ps(x, 1);
    const __m128 lo_quad = _mm256_castps256_ps128(x);
    const __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
    const __m128 lo_dual = sum_quad;
    const __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
    const __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
    const __m128 lo = sum_dual;
    const __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
#endif


static float vec_dot_product(const Float16* vec_a, const Float16* vec_b, int vec_size)
{
#if defined(__AVX__)
    const int simd_vec_size = (int)(vec_size / SIMD_AVX_LANES) * SIMD_AVX_LANES;
    
    __m256 dot_prod_accum = _mm256_setzero_ps();
    for (int i = 0; i < simd_vec_size; i += SIMD_AVX_LANES) {
        const __m256 x0 = vec_f32x8_load(vec_a + i);
        const __m256 x1 = vec_f32x8_load(vec_b + i);
        dot_prod_accum = _mm256_add_ps(_mm256_mul_ps(x0, x1), dot_prod_accum);
    }
    
    // const float* f = (float *)(&dot_prod_accum);
    /// TODO:  Improve this: use simd to reduce sum.
    // float dot_prod = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
    float dot_prod = avx_reduce_sum(dot_prod_accum);

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = fp16_to_fp32(vec_a[i]);
        const float x1 = fp16_to_fp32(vec_b[i]);
        dot_prod += x0 * x1;
    }

#else
    float dot_prod = 0.0f;

    for (int i = 0; i < vec_size; i += 1) {
        dot_prod += fp16_to_fp32(vec_a[i]) * fp16_to_fp32(vec_b[i]);
    }

#endif

    return dot_prod;
}

// Computes logits for next-token pred only.
// inp   : n_ctx, d_embd
// weight: n_vocab, d_embd
// out   : n_vocab 
void lm_head_proj(const Float16* inp, const Float16* weight, float* out, int n_vocab, int n_ctx, int d_embd)
{
    Timer timer{&ops_metrics.matmul_ms};

#if defined(_OPENMP)
        #pragma omp parallel for collapse(2)
#endif
    for (int i = n_ctx - 1; i < n_ctx; i++) {
        for (int j = 0; j < n_vocab; j++) {
            const float dot_prod = vec_dot_product(inp + i * d_embd, weight + j*d_embd, d_embd);
            // for (int k = 0; k < d_embd; k++) {
            //     dot_prod += fp16_to_fp32(inp[i * d_embd + k]) * fp16_to_fp32(weight[j * d_embd + k]);
            // }
            out[j] = dot_prod;
        }
    }
}


// inp0: (n_ctx, d_in)
// inp1: (d_out, d_in)
// out : (n_ctx, d_out)
void matmul_2d(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos=0)
{
    Timer timer{&ops_metrics.matmul_ms};

#if defined(_OPENMP)
        #pragma omp parallel for collapse(2)
#endif
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_out; j++) {
            const float dot_prod = vec_dot_product(inp0 + i*d_in, inp1 + j*d_in, d_in);
            out[i * d_out + j] = fp32_to_fp16(dot_prod);
        }
    }   
}

void linear_2d(const Float16* inp0, const Float16* inp1, const Float16* bias, Float16* out, int n_ctx, int d_in, int d_out, int start_pos=0)
{
    matmul_2d(inp0, inp1, out, n_ctx, d_in, d_out, start_pos);

    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_out; j++) {
            out[i * d_out + j] = fp32_to_fp16(fp16_to_fp32(out[i * d_out + j]) + fp16_to_fp32(bias[j]));
        }
    }   
}

// inp: [n_ctx, d_in]
// inp1 [d_out, d_in]
// out: [d_out, n_ctx]
void linear_2d_transpose_out(const Float16* inp0, const Float16* inp1, const Float16* bias, Float16* out, int n_ctx, int d_in, int d_out)
{
    Timer timer{&ops_metrics.matmul_ms};

    for (int i = 0; i < n_ctx; i++) {
        for (int j = 0; j < d_out; j++) {
            const float dot_prod = vec_dot_product(inp0 + i*d_in, inp1 + j*d_in, d_in);
            out[j*n_ctx + i] = fp32_to_fp16(dot_prod + fp16_to_fp32(bias[j]));
        }
    }   
}

// q: (n_ctx, qn_embd) - (n_ctx, q_heads, d_head)[phy] -> (q_heads, n_ctx, d_head)[virt]
// k: (n_ctx, kn_embd) - (n_ctx, k_heads, d_head)[phy] -> (k_heads, n_ctx, d_head)[virt]
// out: (q_heads, n_ctx, n_ctx)
void qk_masked(const Float16* q, const Float16* k, Float16* out, int n_ctx, int q_heads, int kv_heads, int d_head, int start_pos=0)
{
    Timer timer{&ops_metrics.matmul_ms};

    const int k_heads = kv_heads;
    // Note: In qroup query attn, we divide queries together into groups,
    // each of which share a single key and value.
    const int q_group_size = (int)(q_heads / k_heads);

    const float qk_scaler = 1.0 / sqrtf(d_head);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif
    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            // Compute the dot products which are not subsequently masked (if masked=true).
            const int end_non_masked = i + 1; 
            for (int j = 0; j < end_non_masked; j++) {
                const int hk = h / q_group_size;
                const float dot_prod = vec_dot_product(q + h * d_head + i * q_heads*d_head, k + hk*d_head + j * k_heads*d_head, d_head);
                out[h * n_ctx * n_ctx + i * n_ctx + j] = fp32_to_fp16(dot_prod * qk_scaler);
            }
        }
    }
}

void qk(const Float16* q, const Float16* k, Float16* out, int n_ctx0, int n_ctx1, int q_heads, int kv_heads, int d_head, int start_pos=0)
{
    Timer timer{&ops_metrics.matmul_ms};

    const int k_heads = kv_heads;
    // Note: In qroup query attn, we divide queries together into groups,
    // each of which share a single key and value.
    const int q_group_size = (int)(q_heads / k_heads);

    const float qk_scaler = 1.0 / sqrtf(d_head);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx0; i++) {
            for (int j = 0; j < n_ctx1; j++) {
                const int hk = h / q_group_size;
                const float dot_prod = vec_dot_product(q + h * d_head + i * q_heads*d_head, k + hk*d_head + j * k_heads*d_head, d_head);
                out[h * n_ctx0 * n_ctx1 + i * n_ctx1 + j] = fp32_to_fp16(dot_prod * qk_scaler);
            }
        }
    }
}

// inp: (n_heads, n_ctx, n_ctx)
void attn_mask_inplace(Float16* inp, int n_heads, int n_ctx, int start_pos)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_heads; i++) {
        for (int j = 0; j < n_ctx; j++) {
            const int start_ix = j + 1;
            for (int k = start_ix; k < n_ctx; k++) {
                inp[i * n_ctx * n_ctx + j * n_ctx + k] = fp32_to_fp16(-INFINITY);
            }
        }
    }
}

// inp: [n_heads, n_ctx, n_ctz]
void softmax_inplace(Float16* inp, int n_heads, int n_ctx0, int n_ctx1, int start_pos=0)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int h = 0; h < n_heads; h++) {
        for (int i = start_pos; i < n_ctx0; i++) {
            float max = -INFINITY;
            for (int j = 0; j < n_ctx1; j++) {
                const float val = fp16_to_fp32(inp[h * n_ctx0 * n_ctx1 + i * n_ctx1 + j]);
                if (val > max) {
                    max = val;
                }
            }

            float sum_exp = 0;
            for (int j = 0; j < n_ctx1; j++) {
                const int idx = h * n_ctx0 * n_ctx1 + i * n_ctx1 + j;
                const float res = expf(fp16_to_fp32(inp[idx]) - max);
                sum_exp += res;
                inp[idx] = fp32_to_fp16(res);
            }

            for (int j = 0; j < n_ctx1; j++) {
                const int idx = h * n_ctx0 * n_ctx1 + i * n_ctx1 + j;
                inp[idx] = fp32_to_fp16(fp16_to_fp32(inp[idx]) / sum_exp);
            }
        }
    }
}

// qk: (n_heads, n_ctx0, n_ctx1)
//  v: (n_ctx1, vn_embd) - (n_ctx1, v_heads, d_heads)[phy] - (v_heads, d_heads, n_ctx)[virt]
// out: (n_ctx0, q_heads, d_head)
void qkv(const Float16* qk, const Float16* v, Float16* out, int n_ctx0, int n_ctx1, int q_heads, int kv_heads, int d_head, int start_pos=0)
{
    Timer timer{&ops_metrics.matmul_ms};

    const int v_heads = kv_heads;
    const int qk_group_size = (int)(q_heads / v_heads);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx0; i++) {
            for (int j = 0; j < d_head; j++) {
                float dot_prod = 0.0f;
                for (int k = 0; k < n_ctx1; k++) {
                    // index of the current head in v.
                    const int hv = h / qk_group_size;
                    dot_prod += fp16_to_fp32(qk[h * n_ctx0*n_ctx1 + i * n_ctx1 + k]) * fp16_to_fp32(v[hv * d_head + j + k * v_heads*d_head]);
                }
                out[i * q_heads*d_head + h*d_head + j] = fp32_to_fp16(dot_prod);
            } 
        }
    }
}

/*
2, 6

1, 3, 5, 7, 9, a
2, 4, 6, 8, 0, b

n, h, d
1, 3
5, 7
9, a

2, 4
6, 8
0, b

6, 2 -> 2, 3, 2

1, 2
3, 4
5, 6

7, 8
9, 0
a, b

*/


// qk: (n_heads, n_ctx0, n_ctx1)
//  v: (vn_embd, n_ctx1) - (v_heads, d_heads, n_ctx1)
// out: (n_ctx0, q_heads, d_head)
void qkv_transposed(const Float16* qk, const Float16* v, Float16* out, int n_ctx0, int n_ctx1, int q_heads, int kv_heads, int d_head, int start_pos=0)
{
    Timer timer{&ops_metrics.matmul_ms};

    const int v_heads = kv_heads;
    const int qk_group_size = (int)(q_heads / v_heads);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx0; i++) {
            for (int j = 0; j < d_head; j++) {
                const Float16* qk_vec = qk + h * n_ctx0*n_ctx1 + i * n_ctx1;
                const int hv = h / qk_group_size;
                const Float16* v_vec = v + hv * d_head * n_ctx1 + j * n_ctx1;
                const float dot_prod = vec_dot_product(qk_vec, v_vec, n_ctx1);
                out[i * q_heads*d_head + h*d_head + j] = fp32_to_fp16(dot_prod);
            } 
        }
    }
}

// inp: [n_ctx, n_head, d_head]
void rotary_emb(Float16* inp, int n_ctx, int n_heads, int d_head, float theta, int start_pos)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_ctx; ++i) {
       for (int h = 0; h < n_heads; ++h) {
            Float16* inp_vec = inp + i*n_heads*d_head + h*d_head;

            const int d_half = d_head / 2;
            for (int j = 0; j < d_half; ++j) {
                const float x0 = fp16_to_fp32(inp_vec[j]);
                const float x1 = fp16_to_fp32(inp_vec[j + d_half]);
                
                const float d = (float)(d_head);
                const float base_theta = theta;

                float inv_freq = powf(base_theta, -(2.0f*j/d));

                const float m = (float)(i);
                const float m_theta_i = m * inv_freq;

                const float o0 = x0 * cosf(m_theta_i) - x1 * sinf(m_theta_i);
                const float o1 = x0 * sinf(m_theta_i) + x1 * cosf(m_theta_i);

                inp_vec[j] = fp32_to_fp16(o0);
                inp_vec[j + d_half] = fp32_to_fp16(o1);
            }
        }
    }
}


void copy_tensors(const Float16* src, Float16* dest, int n_ctx, int d_in, int start_pos=0)
{
    Timer timer{&ops_metrics.non_matmul_ms};

    for (int i = start_pos; i < n_ctx; i++) {
        memcpy(dest + i * d_in, src + i * d_in, d_in*sizeof(Float16));
    }
}

/*
1D convolution illustration: (no bias)
inp: (6, 2)
1,  2
3,  4
5,  6
7,  8
8,  10
11, 12

weight: (1, 3, 2)
a, d
b, e
c, f

      | i=0  | i=1  | i=2  | i=3  | i=4  | i=5
==================================================
0,  0 | a, d | -> [Left padded row: is skipped because it always zero]
1,  2 | b, e | a, d |
3,  4 | c, f | b, e | a, d
5,  6        | c, f | b, e | a, d
7,  8               | c, f | b, e | a, d
8,  10                     | c, f | b, e | a, b 
11, 12                            | c, f | d, e
0,  0                                    | c, f |-> [Right padded row]

*/

// 1D convolution where kernel_size=3, stride=1 and padding=1(center)
// input:  [in_frames, in_channels]
// weight: [out_channels, kernel_size, in_channels]
// bias:   [out_channels]
// out :   [in_frames, out_channels]
void conv_1d_stride1(const Float16* inp, const Float16* weight, const Float16* bias, Float16* out, int in_frames, int in_channels, int out_channels)
{
    /// TODO: improve implementation.
    const int kernel_size = 3;
    const int out_frames = in_frames;

    for (int c = 0; c < out_channels; c++) {        
        for (int i = 0; i < out_frames; i++) {
            // The first and the last dot products i.e when i=0 or i=out_frames, we are computing
            // dot products where input should be padded. Since the padding should be done with 
            // zeros, we do not need to compute those parts of the dot product. However, we shift
            // the kernel indices to account for the padding.
            const int kernel_size_end = (i == 0 || i == out_frames-1) ? kernel_size - 1 : kernel_size; 
            const int j_offs = i == 0 ? 1 : 0; 
            float dot_prod = 0.0f;
            for (int j = 0; j < kernel_size_end; j++) {
                for (int k = 0; k < in_channels; k++) {
                    dot_prod += fp16_to_fp32(inp[i * in_channels + j * in_channels + k])
                                * fp16_to_fp32(weight[c * kernel_size*in_channels + (j+j_offs) * in_channels + k]);
                }
            }
            
            out[c + i*out_channels] = fp32_to_fp16(dot_prod + fp16_to_fp32(bias[c]));
        }
    }
}

// 1D convolution where kernel_size=3, stride=2 and padding=1(center)
// input:  [in_frames, in_channels]
// weight: [out_channels, kernel_size, in_channels]
// bias:   [out_channels]
// out :   [n_frames, out_channels]
void conv_1d_stride2(const Float16* inp, const Float16* weight, const Float16* bias, Float16* out, int in_frames, int in_channels, int out_channels)
{
    const int kernel_size = 3;
    const int out_frames = in_frames / 2;

    for (int c = 0; c < out_channels; c++) {
        for (int i = 0, out_i = 0; i < in_frames; i += 2, out_i += 1) {
            const int kernel_size_end = i == 0 ? kernel_size - 1 : kernel_size; 
            const int j_offs = i == 0 ? 1 : 0;
            float dot_prod = 0.0f;
            for (int j = 0; j < kernel_size_end; j++) {
                for (int k = 0; k < in_channels; k++) {
                    dot_prod += fp16_to_fp32(inp[i * in_channels + j * in_channels + k]) * fp16_to_fp32(weight[c * kernel_size*in_channels + (j+j_offs) * in_channels + k]);
                }
            }
            out[out_i*out_channels + c] = fp32_to_fp16(dot_prod + fp16_to_fp32(bias[c]));
        }
    }
}


// TODO: Replace with lookup table.
void gelu(const Float16* inp, Float16* out, int n_ctx, int d_in, int start_pos=0)
{
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_in; j++){
            const float x = fp16_to_fp32(inp[i * d_in + j]);
            const float res = 0.5 * x 
                            * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                            * (x + 0.044715f * std::pow(x, 3.0f))));
            out[i * d_in + j] = fp32_to_fp16(res);
        }
    }
}

void layer_norm(const Float16* inp, const Float16* weight, const Float16* bias, Float16* out, int n_ctx, int d_in, int start_pos=0)
{
    for (int i = 0; i < n_ctx; i++) {
        // Mean calculation.
        float mean_accum = 0.0f;
        for (int j = 0; j < d_in; j++) {
            mean_accum += fp16_to_fp32(inp[i * d_in + j]);
        }
        const float mean = mean_accum / (float)d_in;

        // Standard deviation calculation.
        float variance_accum = 0.0f;
        for (int j = 0; j < d_in; j++) {
            float x = fp16_to_fp32(inp[i * d_in + j]);
            variance_accum += (x - mean) * (x - mean);
        }
        const float std_dev = std::sqrt(variance_accum / (float)d_in);

        // Normalization.
        for (int j = 0; j < d_in; j++) {
            const float x = fp16_to_fp32(inp[i * d_in + j]);
            const float w = fp16_to_fp32(weight[j]);
            const float b = fp16_to_fp32(bias[j]);
            // Epsilon added to standard deviation prevents div by zero.
            const float eps = 1e-05f;
            float normalized = ((x - mean) / (std_dev + eps)) * w + b;
            out[i * d_in + j] = fp32_to_fp16(normalized);
        }
    }
}

} // namespace ops.
