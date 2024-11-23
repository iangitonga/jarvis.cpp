#include <cstdlib>
#include <vector>
#include <unordered_map>

#include "ops.h"



#define WHISPER_ASSERT(condition)  \
    if (!(condition)) { \
        std::fprintf(stderr, "\nLLAMA32_ASSERT: %s:%d: %s.\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    }

// English models only.
enum class WhisperType {
    Tiny,
    Base,
    Small,
    Medium
};


struct WhisperConfig {
    int n_mels;  // encoder input_channels: Number of audio mel-frequency bins.
    int kernel_size;
    int n_vocab;
    int n_audio_ctx;  // Number of frames(context length) in the encoder output representation.
    int n_audio_embd;  // Dimensionality of each frame of the encoder output representation.
    int n_audio_head;  // Number of heads in the audio encoder multi-head self-attention layers.
    int n_audio_layer;  // Number of blocks in the encoder.
    int n_audio_mlp;  // embed * 4
    int n_audio_dhead;
    int n_text_ctx;   // Max number of tokens to be used as a context in the decoder.
    int n_text_embd; // # Dimensionality of each token embedding.
    int n_text_head;  // Number of heads in the text decoder multi-head attention layers.
    int n_text_layer; // # Number of blocks in the decoder.
    int n_text_mlp;
    int n_text_dhead;
};


WhisperConfig get_whisper_config(WhisperType type)
{
    switch (type) {
        case WhisperType::Tiny: {
            WhisperConfig config = {
                .n_mels = 80,
                .kernel_size = 3,
                .n_vocab = 51864,
                .n_audio_ctx = 1500,
                .n_audio_embd = 384,
                .n_audio_head = 6,
                .n_audio_layer = 4,
                .n_audio_mlp = 384 * 4,
                .n_audio_dhead = 64,
                .n_text_ctx = 448,
                .n_text_embd = 384,
                .n_text_head = 6,
                .n_text_layer = 4,
                .n_text_mlp = 384*4,
                .n_text_dhead = 64
            };
            return config;
        }
        case WhisperType::Base: {
            WhisperConfig config = {
                .n_mels = 80,
                .kernel_size = 3,
                .n_vocab = 51864,
                .n_audio_ctx = 1500,
                .n_audio_embd = 512,
                .n_audio_head = 8,
                .n_audio_layer = 6,
                .n_audio_mlp = 512*4,
                .n_audio_dhead = 64,
                .n_text_ctx = 448,
                .n_text_embd = 512,
                .n_text_head = 8,
                .n_text_layer = 6,
                .n_text_mlp = 512*4,
                .n_text_dhead = 64
            };
            return config;
        }
        case WhisperType::Small: {
            WhisperConfig config = {
                .n_mels = 80,
                .kernel_size = 3,
                .n_vocab = 51864,
                .n_audio_ctx = 1500,
                .n_audio_embd = 768,
                .n_audio_head = 12,
                .n_audio_layer = 12,
                .n_audio_mlp = 768*4,
                .n_audio_dhead = 64,
                .n_text_ctx = 448,
                .n_text_embd = 768,
                .n_text_head = 12,
                .n_text_layer = 12,
                .n_text_mlp = 768*4,
                .n_text_dhead = 64
            };
            return config;
        }
        case WhisperType::Medium: {
            WhisperConfig config = {
                .n_mels = 80,
                .kernel_size = 3,
                .n_vocab = 51864,
                .n_audio_ctx = 1500,
                .n_audio_embd = 1024,
                .n_audio_head = 16,
                .n_audio_layer = 24,
                .n_audio_mlp = 1024*4,
                .n_audio_dhead = 64,
                .n_text_ctx = 448,
                .n_text_embd = 1024,
                .n_text_head = 16,
                .n_text_layer = 24,
                .n_text_mlp = 1024*4,
                .n_text_dhead = 64
            };
            return config;
        }
    }
    // Just to prevent warning: control reaches end of non-void function.
    return WhisperConfig{};
}


struct EncoderLayerWeights {
    Float16* attn_normw;
    Float16* attn_normb;
    Float16* attn_qw;
    Float16* attn_qb;
    Float16* attn_kw;
    Float16* attn_vw;
    Float16* attn_vb;
    Float16* attn_ow;
    Float16* attn_ob;
    Float16* mlp_normw;
    Float16* mlp_normb;
    Float16* mlp_upw;
    Float16* mlp_upb;
    Float16* mlp_downw;
    Float16* mlp_downb;
};


#define MAX_ENCODER_LAYERS 24
struct EncoderWeights {
    Float16* conv1w;
    Float16* conv1b;
    Float16* conv2w;
    Float16* conv2b; 
    Float16* pos_emb;
    EncoderLayerWeights layers[MAX_ENCODER_LAYERS];
    Float16* ln_postw;
    Float16* ln_postb;
};


struct EncoderAcvs {
    Float16* conv1_acv;
    Float16* conv2_acv;
    Float16* q_acv;
    Float16* k_acv;
    Float16* v_acv;
    Float16* qk_acv;
    Float16* qkv_acv;
    Float16* o_acv;
    Float16* mlp_up_acv;
    Float16* mlp_down_acv;
    Float16* residual0_acv;
    Float16* residual1_acv;
};


struct DecoderWeightsLayer {
    Float16* attn_normw;
    Float16* attn_normb;
    Float16* attn_qw;
    Float16* attn_qb;
    Float16* attn_kw;
    Float16* attn_vw;
    Float16* attn_vb;
    Float16* attn_ow;
    Float16* attn_ob;
    Float16* xattn_normw;
    Float16* xattn_normb;
    Float16* xattn_qw;
    Float16* xattn_qb;
    Float16* xattn_kw;
    Float16* xattn_vw;
    Float16* xattn_vb;
    Float16* xattn_ow;
    Float16* xattn_ob;
    Float16* mlp_normw;
    Float16* mlp_normb;
    Float16* mlp_upw;
    Float16* mlp_upb;
    Float16* mlp_downw;
    Float16* mlp_downb;
};

#define MAX_DECODER_LAYERS 24
struct DecoderWeights {
    Float16* tok_emb;
    Float16* pos_emb;
    DecoderWeightsLayer layers[MAX_DECODER_LAYERS];
    Float16* out_normw;
    Float16* out_normb;
};


struct DecoderAcvsLayer {
    Float16* residual0_acv;
    Float16* residual1_acv;
    Float16* attn_norm_acv;
    Float16* attn_q_acv;
    Float16* attn_k_acv;
    Float16* attn_v_acv;
    Float16* attn_qk_acv;
    Float16* attn_qkv_acv;
    Float16* attn_o_acv;
    Float16* xattn_norm_acv;
    Float16* xattn_q_acv;
    Float16* xattn_k_acv;
    Float16* xattn_v_acv;
    Float16* xattn_qk_acv;
    Float16* xattn_qkv_acv;
    Float16* xattn_o_acv;
    Float16* mlp_norm_acv;
    Float16* mlp_up_acv;
    Float16* mlp_down_acv;
};

struct DecoderAcvs {
    Float16* emb_acv;
    DecoderAcvsLayer layers[MAX_DECODER_LAYERS];
    Float16* out_norm_acv;
    float* logits;
};

struct Encoder {
    EncoderWeights w;
    EncoderAcvs a;
};

struct Decoder {
    DecoderWeights w;
    DecoderAcvs a;
};

struct Whisper {
    WhisperConfig config;
    Encoder enc;
    Decoder dec;
};

// init


Float16* f16_malloc(int size)
{
    void* ptr = malloc(size * sizeof(Float16));
    if (!ptr) {
        fprintf(stderr, "Failed malloc.");
        exit(-1);
    }
    return (Float16*)(ptr);
}

float* f32_malloc(int size)
{
    void* ptr = malloc(size * sizeof(float));
    if (!ptr) {
        fprintf(stderr, "Failed malloc.");
        exit(-1);
    }
    return (float*)(ptr);
}


void alloc_encoder_weights(EncoderWeights& w, const WhisperConfig& c)
{
    w.conv1w = f16_malloc(c.n_audio_embd * c.kernel_size * c.n_mels);
    w.conv1b = f16_malloc(c.n_audio_embd);
    w.conv2w = f16_malloc(c.n_audio_embd * c.kernel_size * c.n_audio_embd);
    w.conv2b = f16_malloc(c.n_audio_embd);
    w.pos_emb = f16_malloc(c.n_audio_ctx * c.n_audio_embd);

    for (int i = 0; i < c.n_audio_layer; i++) {
        w.layers[i].attn_normw = f16_malloc(c.n_audio_embd);
        w.layers[i].attn_normb = f16_malloc(c.n_audio_embd);
        w.layers[i].attn_qw    = f16_malloc(c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_qb    = f16_malloc(c.n_audio_embd);
        w.layers[i].attn_kw    = f16_malloc(c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_vw    = f16_malloc(c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_vb    = f16_malloc(c.n_audio_embd);
        w.layers[i].attn_ow    = f16_malloc(c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_ob    = f16_malloc(c.n_audio_embd);
        w.layers[i].mlp_normw  = f16_malloc(c.n_audio_embd);
        w.layers[i].mlp_normb  = f16_malloc(c.n_audio_embd);
        w.layers[i].mlp_upw    = f16_malloc(c.n_audio_embd * c.n_audio_mlp);
        w.layers[i].mlp_upb    = f16_malloc(c.n_audio_mlp);
        w.layers[i].mlp_downw  = f16_malloc(c.n_audio_mlp * c.n_audio_embd);
        w.layers[i].mlp_downb  = f16_malloc(c.n_audio_embd);
    }

    w.ln_postw = f16_malloc(c.n_audio_embd);
    w.ln_postb = f16_malloc(c.n_audio_embd);
}

void free_encoder_weights(EncoderWeights& w, const WhisperConfig& c)
{
    free(w.conv1w);
    free(w.conv1b);
    free(w.conv2w);
    free(w.conv2b);
    free(w.pos_emb);

    for (int i = 0; i < c.n_audio_layer; i++) {
        free(w.layers[i].attn_normw);
        free(w.layers[i].attn_normb);
        free(w.layers[i].attn_qw);
        free(w.layers[i].attn_qb);
        free(w.layers[i].attn_kw);
        free(w.layers[i].attn_vw);
        free(w.layers[i].attn_vb);
        free(w.layers[i].attn_ow); 
        free(w.layers[i].attn_ob); 
        free(w.layers[i].mlp_normw);
        free(w.layers[i].mlp_normb);
        free(w.layers[i].mlp_upw); 
        free(w.layers[i].mlp_upb);  
        free(w.layers[i].mlp_downw);
        free(w.layers[i].mlp_downb);
    }

    free(w.ln_postw);
    free(w.ln_postb);
}


void alloc_encoder_acvs(EncoderAcvs& a, const WhisperConfig& c)
{
    const int inp_frames = 3000;

    a.conv1_acv = f16_malloc(inp_frames * c.n_audio_embd);
    a.conv2_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.q_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.k_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.v_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.qk_acv    = f16_malloc(c.n_audio_head * c.n_audio_ctx * c.n_audio_ctx);
    a.qkv_acv   = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.o_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.mlp_up_acv = f16_malloc(c.n_audio_ctx * c.n_audio_mlp);
    a.mlp_down_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.residual0_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
    a.residual1_acv = f16_malloc(c.n_audio_ctx * c.n_audio_embd);
}

void free_encoder_acvs(EncoderAcvs& a)
{
    free(a.conv1_acv);
    free(a.conv2_acv);
    free(a.q_acv);
    free(a.k_acv);
    free(a.v_acv);
    free(a.qk_acv);
    free(a.qkv_acv);
    free(a.o_acv);
    free(a.mlp_up_acv);
    free(a.mlp_down_acv );
    free(a.residual0_acv );
    free(a.residual1_acv );
}

// inp: (n_frames=3000, n_channels=80)
// out: (n_audio_ctx=1500, n_audio_embd)
Float16* encoder_forward(const Float16* inp, Encoder& e, const WhisperConfig& c)
{
    const int inp_frames = 3000;
    ops::conv_1d_stride1(inp, e.w.conv1w, e.w.conv1b, e.a.conv1_acv, inp_frames, c.n_mels, c.n_audio_embd);
    ops::gelu(e.a.conv1_acv, e.a.conv1_acv, inp_frames, c.n_audio_embd);
    ops::conv_1d_stride2(e.a.conv1_acv, e.w.conv2w, e.w.conv2b, e.a.conv2_acv, inp_frames, c.n_audio_embd, c.n_audio_embd);
    ops::gelu(e.a.conv2_acv, e.a.conv2_acv, c.n_audio_ctx, c.n_audio_embd);

    ops::add(e.a.conv2_acv, e.w.pos_emb, e.a.conv2_acv, c.n_audio_ctx, c.n_audio_embd);

    Float16* layer_inp = e.a.conv2_acv;

    for (int i = 0; i < c.n_audio_layer; i++) {
        ops::copy_tensors(layer_inp, e.a.residual0_acv, c.n_audio_ctx, c.n_audio_embd);
        ops::layer_norm(layer_inp, e.w.layers[i].attn_normw, e.w.layers[i].attn_normb, layer_inp, c.n_audio_ctx, c.n_audio_embd);

        ops::linear_2d(layer_inp, e.w.layers[i].attn_qw, e.w.layers[i].attn_qb, e.a.q_acv, c.n_audio_ctx, c.n_audio_embd, c.n_audio_embd);
        ops::matmul_2d(layer_inp, e.w.layers[i].attn_kw, e.a.k_acv, c.n_audio_ctx, c.n_audio_embd, c.n_audio_embd);
        ops::linear_2d_transpose_out(layer_inp, e.w.layers[i].attn_vw, e.w.layers[i].attn_vb, e.a.v_acv, c.n_audio_ctx, c.n_audio_embd, c.n_audio_embd);

        ops::qk(e.a.q_acv, e.a.k_acv, e.a.qk_acv, c.n_audio_ctx,  c.n_audio_ctx, c.n_audio_head, c.n_audio_head, c.n_audio_dhead);

        ops::softmax_inplace(e.a.qk_acv, c.n_audio_head, c.n_audio_ctx, c.n_audio_ctx);
        ops::qkv_transposed(e.a.qk_acv, e.a.v_acv, e.a.qkv_acv, c.n_audio_ctx, c.n_audio_ctx, c.n_audio_head, c.n_audio_head, c.n_audio_dhead);
        ops::linear_2d(e.a.qkv_acv, e.w.layers[i].attn_ow, e.w.layers[i].attn_ob, e.a.o_acv, c.n_audio_ctx, c.n_audio_embd, c.n_audio_embd);
        ops::add(e.a.residual0_acv, e.a.o_acv, e.a.residual0_acv, c.n_audio_ctx, c.n_audio_embd);

        ops::copy_tensors(e.a.residual0_acv, e.a.residual1_acv, c.n_audio_ctx, c.n_audio_embd);

        ops::layer_norm(e.a.residual0_acv, e.w.layers[i].mlp_normw, e.w.layers[i].mlp_normb, e.a.residual0_acv, c.n_audio_ctx, c.n_audio_embd);
        ops::linear_2d(e.a.residual0_acv, e.w.layers[i].mlp_upw, e.w.layers[i].mlp_upb, e.a.mlp_up_acv, c.n_audio_ctx, c.n_audio_embd, c.n_audio_mlp);
        ops::gelu(e.a.mlp_up_acv, e.a.mlp_up_acv, c.n_audio_ctx, c.n_audio_mlp);
        ops::linear_2d(e.a.mlp_up_acv, e.w.layers[i].mlp_downw, e.w.layers[i].mlp_downb, e.a.mlp_down_acv, c.n_audio_ctx, c.n_audio_mlp, c.n_audio_embd);
        ops::add(e.a.residual1_acv, e.a.mlp_down_acv, e.a.residual1_acv, c.n_audio_ctx, c.n_audio_embd);

        layer_inp = e.a.residual1_acv;
    }

    ops::layer_norm(layer_inp, e.w.ln_postw, e.w.ln_postb, layer_inp, c.n_audio_ctx, c.n_audio_embd);

    return layer_inp;
}


void alloc_decoder_acvs(DecoderAcvs& a, const WhisperConfig& c)
{
    a.emb_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);

    for (int i = 0; i < c.n_text_layer; i++) {
        a.layers[i].residual0_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].residual1_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_norm_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_q_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_k_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_v_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_qk_acv = f16_malloc(c.n_text_head * c.n_text_ctx * c.n_text_ctx);
        a.layers[i].attn_qkv_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_o_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_norm_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_q_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_k_acv = f16_malloc(c.n_audio_ctx * c.n_text_embd);
        a.layers[i].xattn_v_acv = f16_malloc(c.n_audio_ctx * c.n_text_embd);
        a.layers[i].xattn_qk_acv = f16_malloc(c.n_audio_head * c.n_text_embd * c.n_audio_ctx);
        a.layers[i].xattn_qkv_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_o_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].mlp_norm_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
        a.layers[i].mlp_up_acv = f16_malloc(c.n_text_ctx * c.n_text_mlp);
        a.layers[i].mlp_down_acv = f16_malloc(c.n_text_ctx * c.n_text_mlp);
    }

    a.out_norm_acv = f16_malloc(c.n_text_ctx * c.n_text_embd);
    a.logits = f32_malloc(c.n_vocab);
} 

void free_decoder_acvs(DecoderAcvs& a, const WhisperConfig& c)
{
    free(a.emb_acv);

    for (int i = 0; i < c.n_text_layer; i++) {
        free(a.layers[i].residual0_acv);
        free(a.layers[i].residual1_acv);
        free(a.layers[i].attn_norm_acv);
        free(a.layers[i].attn_q_acv);
        free(a.layers[i].attn_k_acv);
        free(a.layers[i].attn_v_acv);
        free(a.layers[i].attn_qk_acv);
        free(a.layers[i].attn_qkv_acv);
        free(a.layers[i].attn_o_acv);
        free(a.layers[i].xattn_norm_acv);
        free(a.layers[i].xattn_q_acv);
        free(a.layers[i].xattn_k_acv);
        free(a.layers[i].xattn_v_acv);
        free(a.layers[i].xattn_qk_acv);
        free(a.layers[i].xattn_qkv_acv);
        free(a.layers[i].xattn_o_acv);
        free(a.layers[i].mlp_norm_acv);
        free(a.layers[i].mlp_up_acv);
        free(a.layers[i].mlp_down_acv);
    }

    free(a.out_norm_acv);
    free(a.logits);
} 


void alloc_decoder_weights(DecoderWeights& w, const WhisperConfig& c)
{
    w.tok_emb = f16_malloc(c.n_vocab * c.n_text_embd);
    w.pos_emb = f16_malloc(c.n_text_ctx * c.n_text_embd);

    for (int i = 0; i < c.n_text_layer; i++) {
        w.layers[i].attn_normw = f16_malloc(c.n_text_embd);
        w.layers[i].attn_normb = f16_malloc(c.n_text_embd);
        w.layers[i].attn_qw    = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_qb    = f16_malloc(c.n_text_embd);
        w.layers[i].attn_kw    = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_vw    = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_vb    = f16_malloc(c.n_text_embd);
        w.layers[i].attn_ow    = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_ob    = f16_malloc(c.n_text_embd);
        w.layers[i].xattn_normw= f16_malloc(c.n_text_embd);
        w.layers[i].xattn_normb= f16_malloc(c.n_text_embd);
        w.layers[i].xattn_qw   = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_qb   = f16_malloc(c.n_text_embd);
        w.layers[i].xattn_kw   = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_vw   = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_vb   = f16_malloc(c.n_text_embd);
        w.layers[i].xattn_ow   = f16_malloc(c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_ob   = f16_malloc(c.n_text_embd);
        w.layers[i].mlp_normw  = f16_malloc(c.n_text_embd);
        w.layers[i].mlp_normb  = f16_malloc(c.n_text_embd);
        w.layers[i].mlp_upw    = f16_malloc(c.n_text_embd * c.n_text_mlp);
        w.layers[i].mlp_upb    = f16_malloc(c.n_text_mlp);
        w.layers[i].mlp_downw  = f16_malloc(c.n_text_mlp * c.n_text_embd);
        w.layers[i].mlp_downb  = f16_malloc(c.n_text_embd);
    }

    w.out_normw = f16_malloc(c.n_text_embd);
    w.out_normb = f16_malloc(c.n_text_embd);
}

void free_decoder_weights(DecoderWeights& w, const WhisperConfig& c)
{
    free(w.tok_emb);
    free(w.pos_emb);

    for (int i = 0; i < c.n_text_layer; i++) {
        free(w.layers[i].attn_normw);
        free(w.layers[i].attn_normb);
        free(w.layers[i].attn_qw);
        free(w.layers[i].attn_qb);
        free(w.layers[i].attn_kw);
        free(w.layers[i].attn_vw);
        free(w.layers[i].attn_vb);
        free(w.layers[i].attn_ow);
        free(w.layers[i].attn_ob);
        free(w.layers[i].xattn_normw);
        free(w.layers[i].xattn_normb);
        free(w.layers[i].xattn_qw);
        free(w.layers[i].xattn_qb);
        free(w.layers[i].xattn_kw);
        free(w.layers[i].xattn_vw);
        free(w.layers[i].xattn_vb);
        free(w.layers[i].xattn_ow);
        free(w.layers[i].xattn_ob);
        free(w.layers[i].mlp_normw);
        free(w.layers[i].mlp_normb);
        free(w.layers[i].mlp_upw);
        free(w.layers[i].mlp_upb);
        free(w.layers[i].mlp_downw);
        free(w.layers[i].mlp_downb);
    }

    free(w.out_normw);
    free(w.out_normb);
}


void alloc_whisper(Whisper& model)
{
    alloc_encoder_weights(model.enc.w, model.config);
    alloc_encoder_acvs(model.enc.a, model.config);
    alloc_decoder_weights(model.dec.w, model.config);
    alloc_decoder_acvs(model.dec.a, model.config);
}

void free_whisper(Whisper& model)
{
    free_encoder_weights(model.enc.w, model.config);
    free_encoder_acvs(model.enc.a);
    free_decoder_weights(model.dec.w, model.config);
    free_decoder_acvs(model.dec.a, model.config);
}

float* decoder_forward(int* tokens, int n_ctx, Float16* xa, Decoder& d, const WhisperConfig& c, int start_pos)
{
    ops::embed(tokens, d.w.tok_emb, d.a.emb_acv, c.n_vocab, n_ctx, c.n_text_embd, start_pos);
    ops::add(d.a.emb_acv, d.w.pos_emb, d.a.emb_acv, n_ctx, c.n_text_embd, start_pos);

    Float16* layer_inp = d.a.emb_acv;

    for (int i = 0; i < c.n_text_layer; i++) {
        DecoderAcvsLayer& al = d.a.layers[i];
        DecoderWeightsLayer wl = d.w.layers[i];

        ops::layer_norm(layer_inp, wl.attn_normw, wl.attn_normb, al.attn_norm_acv, n_ctx, c.n_text_embd, start_pos);

        // Self Attention
        ops::linear_2d(al.attn_norm_acv, wl.attn_qw, wl.attn_qb, al.attn_q_acv, n_ctx, c.n_text_embd, c.n_text_embd, start_pos);
        ops::matmul_2d(al.attn_norm_acv, wl.attn_kw, al.attn_k_acv, n_ctx, c.n_text_embd, c.n_text_embd, start_pos);
        ops::linear_2d(al.attn_norm_acv, wl.attn_vw, wl.attn_vb, al.attn_v_acv, n_ctx, c.n_text_embd, c.n_text_embd, start_pos);
        ops::qk_masked(al.attn_q_acv, al.attn_k_acv, al.attn_qk_acv, n_ctx, c.n_text_head, c.n_text_head, c.n_text_dhead, start_pos);
        ops::attn_mask_inplace(al.attn_qk_acv, c.n_text_head, n_ctx, start_pos);
        ops::softmax_inplace(al.attn_qk_acv, c.n_text_head, n_ctx, n_ctx, start_pos);
        ops::qkv(al.attn_qk_acv, al.attn_v_acv, al.attn_qkv_acv, n_ctx, n_ctx, c.n_text_head, c.n_text_head, c.n_text_dhead, start_pos);
        ops::linear_2d(al.attn_qkv_acv, wl.attn_ow, wl.attn_ob, al.attn_o_acv, n_ctx, c.n_text_embd, c.n_text_embd, start_pos);

        ops::add(layer_inp, al.attn_o_acv, al.residual0_acv, n_ctx, c.n_text_embd, start_pos);
        ops::layer_norm(al.residual0_acv, wl.xattn_normw, wl.xattn_normb, al.xattn_norm_acv, n_ctx, c.n_text_embd, start_pos);

        // Cross Attention
        ops::linear_2d(al.xattn_norm_acv, wl.xattn_qw, wl.xattn_qb, al.xattn_q_acv, n_ctx, c.n_text_embd, c.n_text_embd, start_pos);
        // Compute and cache in the first iteration.
        if (start_pos == 0) {
            ops::matmul_2d(xa, wl.xattn_kw, al.xattn_k_acv, c.n_audio_ctx, c.n_audio_embd, c.n_audio_embd);
            ops::linear_2d(xa, wl.xattn_vw, wl.xattn_vb, al.xattn_v_acv, c.n_audio_ctx, c.n_audio_embd, c.n_audio_embd);
        }
        ops::qk(al.xattn_q_acv, al.xattn_k_acv, al.xattn_qk_acv, n_ctx, c.n_audio_ctx, c.n_text_head, c.n_audio_head, c.n_text_dhead, start_pos);
        ops::softmax_inplace(al.xattn_qk_acv, c.n_text_head, n_ctx, c.n_audio_ctx, start_pos);
        /// TODO: Transpose v and implement linear qkv.
        ops::qkv(al.xattn_qk_acv, al.xattn_v_acv, al.attn_qkv_acv, n_ctx, c.n_audio_ctx, c.n_text_head, c.n_audio_head, c.n_text_dhead, start_pos);
        ops::linear_2d(al.attn_qkv_acv, wl.xattn_ow, wl.xattn_ob, al.xattn_o_acv, n_ctx, c.n_text_embd, c.n_text_embd, start_pos);

        ops::add(al.residual0_acv, al.xattn_o_acv, al.residual1_acv, n_ctx, c.n_text_embd, start_pos);
        ops::layer_norm(al.residual1_acv, wl.mlp_normw, wl.mlp_normb, al.mlp_norm_acv, n_ctx, c.n_text_embd, start_pos);

        // MLP
        ops::linear_2d(al.mlp_norm_acv, wl.mlp_upw, wl.mlp_upb, al.mlp_up_acv, n_ctx, c.n_text_embd, c.n_text_mlp, start_pos);
        ops::gelu(al.mlp_up_acv, al.mlp_up_acv, n_ctx, c.n_text_mlp, start_pos);
        ops::linear_2d(al.mlp_up_acv, wl.mlp_downw, wl.mlp_downb, al.mlp_down_acv, n_ctx, c.n_text_mlp, c.n_text_embd, start_pos);

        ops::add(al.residual1_acv, al.mlp_down_acv, al.residual1_acv, n_ctx, c.n_text_embd, start_pos);

        layer_inp = al.residual1_acv;
    }

    ops::layer_norm(layer_inp, d.w.out_normw, d.w.out_normb, d.a.out_norm_acv, n_ctx, c.n_text_embd, start_pos);
    ops::lm_head_proj(d.a.out_norm_acv, d.w.tok_emb, d.a.logits, c.n_vocab, n_ctx, c.n_text_embd);

    return d.a.logits;
}


void read_into(Float16* dst, int size, FILE* stream) {
    WHISPER_ASSERT(fread(dst, sizeof(Float16), size, stream) == size);
}

void load_whisper(const char* fpath, Whisper& m)
{
    std::FILE* fin = std::fopen(fpath, "rb");

    if (!fin) {
        std::fprintf(stderr, "%s: failed to open %s.\n", __func__, fpath);
        std::exit(-1);
    }

    const int64_t true_magic_no = 0x6672657073696877; // Hex for ASCII string: whisperf
    int64_t magic_no;
    WHISPER_ASSERT(fread(&magic_no, sizeof(int64_t), 1, fin) == 1);

    if (magic_no != true_magic_no) {
        fprintf(stderr, "Magic number: %ld failed to match the expected one: %ld.\n", magic_no, true_magic_no);
        fclose(fin);
        exit(-1);
    }
    const WhisperConfig& c = m.config;

    read_into(m.enc.w.conv1w, c.n_audio_embd * c.kernel_size * c.n_mels, fin);
    read_into(m.enc.w.conv1b, c.n_audio_embd, fin);
    read_into(m.enc.w.conv2w, c.n_audio_embd * c.kernel_size * c.n_audio_embd, fin);
    read_into(m.enc.w.conv2b, c.n_audio_embd, fin);
    read_into(m.enc.w.pos_emb, c.n_audio_ctx * c.n_audio_embd, fin);

    for (int i = 0; i < c.n_audio_layer; i++) {
        read_into(m.enc.w.layers[i].attn_normw, c.n_audio_embd, fin);
        read_into(m.enc.w.layers[i].attn_normb, c.n_audio_embd, fin);
        read_into(m.enc.w.layers[i].attn_qw,    c.n_audio_embd * c.n_audio_embd, fin);
        read_into(m.enc.w.layers[i].attn_qb,    c.n_audio_embd,                  fin);
        read_into(m.enc.w.layers[i].attn_kw,    c.n_audio_embd * c.n_audio_embd, fin);
        read_into(m.enc.w.layers[i].attn_vw,    c.n_audio_embd * c.n_audio_embd, fin);
        read_into(m.enc.w.layers[i].attn_vb,    c.n_audio_embd,                  fin);
        read_into(m.enc.w.layers[i].attn_ow,    c.n_audio_embd * c.n_audio_embd, fin);
        read_into(m.enc.w.layers[i].attn_ob,    c.n_audio_embd,                  fin);
        read_into(m.enc.w.layers[i].mlp_normw,  c.n_audio_embd,                  fin);
        read_into(m.enc.w.layers[i].mlp_normb,  c.n_audio_embd,                  fin);
        read_into(m.enc.w.layers[i].mlp_upw,    c.n_audio_embd * c.n_audio_mlp,  fin);
        read_into(m.enc.w.layers[i].mlp_upb,    c.n_audio_mlp,                   fin);
        read_into(m.enc.w.layers[i].mlp_downw,  c.n_audio_mlp * c.n_audio_embd,  fin);
        read_into(m.enc.w.layers[i].mlp_downb,  c.n_audio_embd,                  fin);
    }

    read_into(m.enc.w.ln_postw, c.n_audio_embd, fin);
    read_into(m.enc.w.ln_postb, c.n_audio_embd, fin);

    // DECODER
    read_into(m.dec.w.tok_emb, c.n_vocab * c.n_text_embd, fin);
    read_into(m.dec.w.pos_emb, c.n_text_ctx * c.n_text_embd, fin);

    for (int i = 0; i < c.n_text_layer; i++) {
        read_into(m.dec.w.layers[i].attn_normw, c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].attn_normb, c.n_text_embd, fin);

        read_into(m.dec.w.layers[i].attn_qw, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].attn_qb, c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].attn_kw, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].attn_vw, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].attn_vb, c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].attn_ow, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].attn_ob, c.n_text_embd, fin);

        read_into(m.dec.w.layers[i].xattn_normw, c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].xattn_normb, c.n_text_embd, fin);

        read_into(m.dec.w.layers[i].xattn_qw, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].xattn_qb, c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].xattn_kw, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].xattn_vw, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].xattn_vb, c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].xattn_ow, c.n_text_embd * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].xattn_ob, c.n_text_embd, fin);

        read_into(m.dec.w.layers[i].mlp_normw, c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].mlp_normb, c.n_text_embd, fin);

        read_into(m.dec.w.layers[i].mlp_upw, c.n_text_embd * c.n_text_mlp, fin);
        read_into(m.dec.w.layers[i].mlp_upb, c.n_text_mlp, fin);
        read_into(m.dec.w.layers[i].mlp_downw, c.n_text_mlp * c.n_text_embd, fin);
        read_into(m.dec.w.layers[i].mlp_downb, c.n_text_embd, fin);
    }

    read_into(m.dec.w.out_normw, c.n_text_embd, fin);
    read_into(m.dec.w.out_normb, c.n_text_embd, fin);
}


void init_whisper(Whisper& model, WhisperType type, const char* path) {
    model.config = get_whisper_config(type);
    alloc_whisper(model);
    load_whisper(path, model);
}

void uninit_whisper(Whisper& model) {
     free_whisper(model);
}

class WhisperTokenizer {
public:
    int sot{50257};
    int eot{50256};
    int transcribe{50358};
    int translate{50357};
    int no_speech{50361};
    int no_timestamps{50362};
    int timestamp_begin{50363};
    int timestamp_end{51863};
    int english_token{50258};

    WhisperTokenizer() {
        std::ifstream fin{m_vocab_path, std::ios_base::binary};
        if(!fin.is_open()) {
            std::fprintf(stderr, "Failed to open vocab file: %s\n", m_vocab_path.c_str());
            std::exit(EXIT_FAILURE);
        };

        std::string word;
        for (int i = 0; i < m_n_vocab; i++) {
            int32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            m_token_to_str[i] = word;
        }     
    }
    
    const std::string& decode_token(int token) {
        if (token < m_n_vocab) {
            return m_token_to_str[token];
        }
        else {
            // std::cerr << "Unknown token: " << token << "\n";
            return m_special_token;
        }
    }

private:
    std::string m_vocab_path{"assets/whisper_vocab.bin"};
    int m_n_vocab{50256};
    std::unordered_map<int, std::string> m_token_to_str;
    const std::string m_special_token = "<special_token>";
};


std::string whisper_decode(Whisper& model, WhisperTokenizer& tokenizer, Float16* audio_embed, bool stream = false)
{
    std::string prompt;
    std::vector<int> tokens = {tokenizer.sot, tokenizer.no_timestamps};
    if (stream) {
        printf("STT Decoded: ");
    }
    for (int i = 0; i < model.config.n_text_ctx/2; i++) {
        const int start_pos = i == 0 ? 0 : tokens.size() - 1;
        float* logits = decoder_forward(tokens.data(), tokens.size(), audio_embed, model.dec, model.config, start_pos);

        /// TODO: Add logits suppression.
        int pred_token = -1;
        float max_prob = -INFINITY;
        // Only consider non_special tokens, i.e timestamps and tags.
        const int vocab_size = 50257;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_prob) {
                max_prob = logits[i];
                pred_token = i;
            }
        }

        if (pred_token == tokenizer.eot) {
            break;
        }

        std::string decoded = tokenizer.decode_token(pred_token);
        if (stream) {
            printf("%s", decoded.c_str());
            fflush(stdout);
        }

        prompt.append(decoded);

        tokens.push_back(pred_token);
    }
    if (stream) {
        printf("\n");
    }

    // Remove a leading whitespace which is produced by whisper tokenizer.
    if (prompt[0] == ' ') {
        prompt = prompt.substr(1, prompt.size());
    }

    return prompt;
}

void read_test_spectrogram(Float16* spectrogram_out)
{
    FILE* spec_file = fopen("assets/test_spectrogram.bin", "rb");
    JARVIS_ASSERT(spec_file);
    JARVIS_ASSERT(fread(spectrogram_out, sizeof(Float16), 3000*80, spec_file) == 3000*80);
    fclose(spec_file);
}

// int main(int argc, char const *argv[])
// {
//     const Config config{};
//     Whisper model{};
//     alloc_whisper(model, config);
//     load_whisper("whisper-tiny.en.bin", model, config);

//     //  [n_frames, in_channels]
//     float* spectrogram_f32 = new float[80 * 3000];
//     Float16* spectrogram = new Float16[80 * 3000];
//     FILE* fin = fopen("spectrogram.bin", "rb");
//     WHISPER_ASSERT(fin);

//     WHISPER_ASSERT(fread(spectrogram_f32, sizeof(float), 80*3000, fin) == 80*3000);

//     fclose(fin);

//     // 3000, 80
//     for (int i = 0; i < 80; i++) {
//         for (int j = 0; j < 3000; j++) {
//             spectrogram[i + j * 80] = fp32_to_fp16(spectrogram_f32[i * 3000 + j]);
//         }
//     }
    
//     Float16* xa = encoder_forward(spectrogram, model.enc, config);


//     Tokenizer tokenizer;

//     printf("Prediction:\n\n");
//     std::vector<int> tokens = {tokenizer.sot, tokenizer.no_timestamps};
//     for (int i = 0; i < config.n_text_ctx/2; i++) {
//         const int start_pos = i == 0 ? 0 : tokens.size() - 1;
//         float* logits = decoder_forward(tokens.data(), tokens.size(), xa, model.dec, config, start_pos);

//         int pred_token = -1;
//         float max_prob = -INFINITY;
//         for (int i = 0; i < 50257; i++) {
//             if (logits[i] > max_prob) {
//                 max_prob = logits[i];
//                 pred_token = i;
//             }
//         }

//         if (pred_token == tokenizer.eot) {
//             break;
//         }

//         printf("%s", tokenizer.decode_token(pred_token).c_str());
//         fflush(stdout);

//         tokens.push_back(pred_token);
//     }
//     printf("\n");

//     free_whisper(model, config);

//     printf("Matmul ms: %ld\n", globals::metrics.matmul_ms);
//     printf("NonMatmul ms: %ld\n", globals::metrics.non_matmul_ms);

//     return 0;
// }

