#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <fstream>

#include "ops.h"



#define WHISPER_ASSERT(condition)  \
    if (!(condition)) { \
        std::fprintf(stderr, "\nLLAMA32_ASSERT: %s:%d: %s.\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    }


namespace stt {

class WhisperTokenizer {
public:
    const int sot = 50257;
    const int eot = 50256;
    const int transcribe = 50358;
    const int translate = 50357;
    const int no_speech = 50361;
    const int no_timestamps = 50362;
    const int timestamp_begin = 50363;
    const int timestamp_end = 51863;
    const int english_token = 50258;

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
            return m_special_token;
        }
    }

private:
    std::string m_vocab_path{"assets/whisper_vocab.bin"};
    int m_n_vocab{50256};
    std::unordered_map<int, std::string> m_token_to_str;
    const std::string m_special_token = "<special_token>";
};

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
    WhisperTokenizer tokenizer;
    WhisperConfig config;
    Encoder enc;
    Decoder dec;
};

Float16* assign_mem(Float16** memptr, size_t advance_size) {
    Float16* out_ptr = *memptr;
    *memptr = *memptr + advance_size;
    return out_ptr;
}

void alloc_encoder_weights(EncoderWeights& w, const WhisperConfig& c, Float16* memptr)
{
    w.conv1w = assign_mem(&memptr, c.n_audio_embd * c.kernel_size * c.n_mels);
    w.conv1b = assign_mem(&memptr, c.n_audio_embd);
    w.conv2w = assign_mem(&memptr, c.n_audio_embd * c.kernel_size * c.n_audio_embd);
    w.conv2b = assign_mem(&memptr, c.n_audio_embd);
    w.pos_emb = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);

    for (int i = 0; i < c.n_audio_layer; i++) {
        w.layers[i].attn_normw = assign_mem(&memptr, c.n_audio_embd);
        w.layers[i].attn_normb = assign_mem(&memptr, c.n_audio_embd);
        w.layers[i].attn_qw    = assign_mem(&memptr, c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_qb    = assign_mem(&memptr, c.n_audio_embd);
        w.layers[i].attn_kw    = assign_mem(&memptr, c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_vw    = assign_mem(&memptr, c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_vb    = assign_mem(&memptr, c.n_audio_embd);
        w.layers[i].attn_ow    = assign_mem(&memptr, c.n_audio_embd * c.n_audio_embd);
        w.layers[i].attn_ob    = assign_mem(&memptr, c.n_audio_embd);
        w.layers[i].mlp_normw  = assign_mem(&memptr, c.n_audio_embd);
        w.layers[i].mlp_normb  = assign_mem(&memptr, c.n_audio_embd);
        w.layers[i].mlp_upw    = assign_mem(&memptr, c.n_audio_embd * c.n_audio_mlp);
        w.layers[i].mlp_upb    = assign_mem(&memptr, c.n_audio_mlp);
        w.layers[i].mlp_downw  = assign_mem(&memptr, c.n_audio_mlp * c.n_audio_embd);
        w.layers[i].mlp_downb  = assign_mem(&memptr, c.n_audio_embd);
    }

    w.ln_postw = assign_mem(&memptr, c.n_audio_embd);
    w.ln_postb = memptr;
}

size_t get_encoder_weights_size(const WhisperConfig& c)
{
    size_t nbytes = 0;

    nbytes += c.n_audio_embd * c.kernel_size * c.n_mels;
    nbytes += c.n_audio_embd;
    nbytes += c.n_audio_embd * c.kernel_size * c.n_audio_embd;
    nbytes += c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;

    for (int i = 0; i < c.n_audio_layer; i++) {
        nbytes += c.n_audio_embd;
        nbytes += c.n_audio_embd;
        nbytes += c.n_audio_embd * c.n_audio_embd;
        nbytes += c.n_audio_embd;
        nbytes += c.n_audio_embd * c.n_audio_embd;
        nbytes += c.n_audio_embd * c.n_audio_embd;
        nbytes += c.n_audio_embd;
        nbytes += c.n_audio_embd * c.n_audio_embd;
        nbytes += c.n_audio_embd;
        nbytes += c.n_audio_embd;
        nbytes += c.n_audio_embd;
        nbytes += c.n_audio_embd * c.n_audio_mlp;
        nbytes += c.n_audio_mlp;
        nbytes += c.n_audio_mlp * c.n_audio_embd;
        nbytes += c.n_audio_embd;
    }

    nbytes += c.n_audio_embd;
    nbytes += c.n_audio_embd;

    nbytes = nbytes * sizeof(Float16);
    return nbytes;
}

void alloc_encoder_acvs(EncoderAcvs& a, const WhisperConfig& c, Float16* memptr)
{
    const int inp_frames = 3000;
    a.conv1_acv     = assign_mem(&memptr, inp_frames * c.n_audio_embd);
    a.conv2_acv     = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.q_acv         = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.k_acv         = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.v_acv         = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.qk_acv        = assign_mem(&memptr, c.n_audio_head * c.n_audio_ctx * c.n_audio_ctx);
    a.qkv_acv       = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.o_acv         = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.mlp_up_acv    = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_mlp);
    a.mlp_down_acv  = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.residual0_acv = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
    a.residual1_acv = assign_mem(&memptr, c.n_audio_ctx * c.n_audio_embd);
}

size_t get_encoder_acvs_size(const WhisperConfig& c)
{
    size_t nbytes = 0;

    const int inp_frames = 3000;
    nbytes += inp_frames * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_head * c.n_audio_ctx * c.n_audio_ctx;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_mlp;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;
    nbytes += c.n_audio_ctx * c.n_audio_embd;

    nbytes = nbytes * sizeof(Float16);
    return nbytes;
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



void alloc_decoder_weights(DecoderWeights& w, const WhisperConfig& c, Float16* memptr)
{
    w.tok_emb = assign_mem(&memptr, c.n_vocab * c.n_text_embd);
    w.pos_emb = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);

    for (int i = 0; i < c.n_text_layer; i++) {
        w.layers[i].attn_normw = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].attn_normb = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].attn_qw    = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_qb    = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].attn_kw    = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_vw    = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_vb    = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].attn_ow    = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].attn_ob    = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].xattn_normw= assign_mem(&memptr, c.n_text_embd);
        w.layers[i].xattn_normb= assign_mem(&memptr, c.n_text_embd);
        w.layers[i].xattn_qw   = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_qb   = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].xattn_kw   = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_vw   = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_vb   = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].xattn_ow   = assign_mem(&memptr, c.n_text_embd * c.n_text_embd);
        w.layers[i].xattn_ob   = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].mlp_normw  = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].mlp_normb  = assign_mem(&memptr, c.n_text_embd);
        w.layers[i].mlp_upw    = assign_mem(&memptr, c.n_text_embd * c.n_text_mlp);
        w.layers[i].mlp_upb    = assign_mem(&memptr, c.n_text_mlp);
        w.layers[i].mlp_downw  = assign_mem(&memptr, c.n_text_mlp * c.n_text_embd);
        w.layers[i].mlp_downb  = assign_mem(&memptr, c.n_text_embd);
    }

    w.out_normw = assign_mem(&memptr, c.n_text_embd);
    w.out_normb = memptr;
}

size_t get_decoder_weights_size(const WhisperConfig& c)
{
    size_t nbytes = 0;

    nbytes += c.n_vocab * c.n_text_embd;
    nbytes += c.n_text_ctx * c.n_text_embd;

    for (int i = 0; i < c.n_text_layer; i++) {
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd;
        nbytes += c.n_text_embd * c.n_text_mlp;
        nbytes += c.n_text_mlp;
        nbytes += c.n_text_mlp * c.n_text_embd;
        nbytes += c.n_text_embd;
    }

    nbytes += c.n_text_embd;
    nbytes += c.n_text_embd;

    nbytes = nbytes * sizeof(Float16);
    return nbytes;
}


void alloc_decoder_acvs(DecoderAcvs& a, const WhisperConfig& c, Float16* memptr)
{
    a.emb_acv = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);

    for (int i = 0; i < c.n_text_layer; i++) {
        a.layers[i].residual0_acv  = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].residual1_acv  = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_norm_acv  = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_q_acv     = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_k_acv     = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_v_acv     = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_qk_acv    = assign_mem(&memptr, c.n_text_head * c.n_text_ctx * c.n_text_ctx);
        a.layers[i].attn_qkv_acv   = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].attn_o_acv     = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_norm_acv = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_q_acv    = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_k_acv    = assign_mem(&memptr, c.n_audio_ctx * c.n_text_embd);
        a.layers[i].xattn_v_acv    = assign_mem(&memptr, c.n_audio_ctx * c.n_text_embd);
        a.layers[i].xattn_qk_acv   = assign_mem(&memptr, c.n_audio_head * c.n_text_embd * c.n_audio_ctx);
        a.layers[i].xattn_qkv_acv  = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].xattn_o_acv    = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].mlp_norm_acv   = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
        a.layers[i].mlp_up_acv     = assign_mem(&memptr, c.n_text_ctx * c.n_text_mlp);
        a.layers[i].mlp_down_acv   = assign_mem(&memptr, c.n_text_ctx * c.n_text_mlp);
    }

    a.out_norm_acv = assign_mem(&memptr, c.n_text_ctx * c.n_text_embd);
    a.logits = (float*)memptr;
}

size_t get_decoder_acvs_size(const WhisperConfig& c)
{
    size_t nbytes = 0;

    nbytes += c.n_text_ctx * c.n_text_embd;

    for (int i = 0; i < c.n_text_layer; i++) {
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_head * c.n_text_ctx * c.n_text_ctx;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_audio_ctx * c.n_text_embd;
        nbytes += c.n_audio_ctx * c.n_text_embd;
        nbytes += c.n_audio_head * c.n_text_embd * c.n_audio_ctx;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_embd;
        nbytes += c.n_text_ctx * c.n_text_mlp;
        nbytes += c.n_text_ctx * c.n_text_mlp;
    }

    nbytes += c.n_text_ctx * c.n_text_embd;
    nbytes = nbytes * sizeof(Float16);

    nbytes += c.n_vocab * sizeof(float); // logits

    return nbytes;
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

    const size_t encoder_nbytes = get_encoder_weights_size(c);
    WHISPER_ASSERT(fread(m.enc.w.conv1w, encoder_nbytes, 1, fin) == 1);

    const size_t decoder_nbytes = get_decoder_weights_size(c);
    WHISPER_ASSERT(fread(m.dec.w.tok_emb, decoder_nbytes, 1, fin) == 1);
}


void* whisper_malloc(size_t nbytes) {
    void* allocated = malloc(nbytes);
    if (!allocated) {
        fprintf(stderr, "whisper_alloc: Failed to allocate %ld bytes.", nbytes);
        exit(-1);
    }
    return allocated;
}

void whisper_mfree(void* memptr) {
    free(memptr);
}


void whisper_alloc(Whisper& model)
{
    const size_t enc_weights_nbytes = get_encoder_weights_size(model.config);
    const size_t dec_weights_nbytes = get_decoder_weights_size(model.config);
    size_t weights_nbytes = enc_weights_nbytes + dec_weights_nbytes;
    const size_t enc_acvs_nbytes = get_encoder_acvs_size(model.config);
    const size_t dec_acvs_nbytes = get_decoder_acvs_size(model.config);
    size_t acvs_nbytes = enc_acvs_nbytes + dec_acvs_nbytes;

    size_t total_nbytes = weights_nbytes + acvs_nbytes;

    char* memptr = (char*)whisper_malloc(total_nbytes);
    printf("whisper alloc: %ldMB\n", total_nbytes / 1000000);
    char* enc_weights_ptr = memptr;
    char* dec_weights_ptr = enc_weights_ptr + enc_weights_nbytes;
    char* enc_acvs_ptr = dec_weights_ptr + dec_weights_nbytes;
    char* dec_acvs_ptr = enc_acvs_ptr + enc_acvs_nbytes;

    alloc_encoder_weights(model.enc.w, model.config, (Float16*)enc_weights_ptr);
    alloc_decoder_weights(model.dec.w, model.config, (Float16*)dec_weights_ptr);
    alloc_encoder_acvs(model.enc.a, model.config, (Float16*)enc_acvs_ptr);
    alloc_decoder_acvs(model.dec.a, model.config, (Float16*)dec_acvs_ptr);
}

void whisper_free(Whisper& model)
{
    whisper_mfree(model.enc.w.conv1w);
}


void whisper_init(Whisper& model, WhisperType type, const std::string& path) {
    model.config = get_whisper_config(type);
    whisper_alloc(model);
    load_whisper(path.c_str(), model);
}

void whisper_uninit(Whisper& model) {
     whisper_free(model);
}


// All non-language tokens that are suppressed during decoding.
static const int ENGLISH_VOCAB_BAD_TOKENS[98] = {
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92,
    93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377,
    1391, 1635, 1782, 1875, 1906, 2162, 2361, 2488, 3467, 3880, 4008, 4211, 4600, 4808, 5299,
    5855, 6329, 7203, 8864, 9609, 9959, 10221, 10563, 10786, 11420, 11709, 11907, 13163,
    13697, 13700, 14808, 15306, 16410, 16791, 17174, 17992, 19203, 19510, 20368, 20724,
    22305, 22935, 23090, 27007, 29113, 30109, 30420, 30906, 33409, 34949, 40283, 40493, 
    40549, 41906, 46111, 47282, 49146, 49704
};

void suppress_bad_tokens(float* logits, int n_logits)
{
    const int n_bad_tokens = sizeof(ENGLISH_VOCAB_BAD_TOKENS) / sizeof(int);
    for (int i = 0; i < n_bad_tokens; i++) {
        logits[ENGLISH_VOCAB_BAD_TOKENS[i]] = -INFINITY;
    }
}

std::string whisper_decode(Whisper& model, Float16* audio_embed, bool stream = false)
{
    std::string prompt;
    std::vector<int> tokens = {model.tokenizer.sot, model.tokenizer.no_timestamps};
    if (stream) {
        printf("STT Decoded: ");
        fflush(stdout);
    }
    for (int i = 0; i < model.config.n_text_ctx/2; i++) {
        const int start_pos = i == 0 ? 0 : tokens.size() - 1;
        float* logits = decoder_forward(tokens.data(), tokens.size(), audio_embed, model.dec, model.config, start_pos);

        // Vocab size without special tokens, i.e timestamps and tags.
        const int vocab_size = 50257;
        suppress_bad_tokens(logits, vocab_size);

        int pred_token = -1;
        float max_prob = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_prob) {
                max_prob = logits[i];
                pred_token = i;
            }
        }

        if (pred_token == model.tokenizer.eot) {
            break;
        }

        std::string decoded = model.tokenizer.decode_token(pred_token);
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

