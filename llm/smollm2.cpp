#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <vector>

#include "ops.h"
#include "utils.h"


#define LLAMA32_ASSERT(condition)  \
    if (!(condition)) { \
        std::fprintf(stderr, "\nLLAMA32_ASSERT: %s:%d: %s.\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    }


enum class ModelType {
    Small,
    Medium,
    Large
};


struct SmolLM2Config
{
    int n_vocab;
    int n_layers;
    int d_embd;
    int n_heads;
    int n_kv_heads;
    int d_head;
    int d_mlp;
    float rope_theta;
    float rms_eps;
};

struct LayerWeights {
    Float16* attn_norm;
    Float16* q_proj;
    Float16* k_proj;
    Float16* v_proj;
    Float16* o_proj;
    Float16* mlp_norm;
    Float16* gate_proj;
    Float16* up_proj;
    Float16* down_proj;
};

#define MAX_NUM_LAYERS 32
struct SmolLM2Weights
{
    Float16* emb_table;
    LayerWeights layers[MAX_NUM_LAYERS];
    Float16* out_norm;
};

struct LayerAcvs
{
    Float16* attn_norm_acv;
    Float16* res_0_acv;
    Float16* res_1_acv;
    Float16* q_proj_acv;
    Float16* k_proj_acv;
    Float16* v_proj_acv;
    Float16* o_proj_acv;
    Float16* qk_acv;
    Float16* qkv_acv;
    Float16* mlp_norm_acv;
    Float16* mlp_gate_acv;
    Float16* mlp_up_acv;
    Float16* mlp_down_acv;
};

struct SmolLM2Acvs
{
    Float16* emb_acv;
    LayerAcvs layers[MAX_NUM_LAYERS];
    Float16* out_norm_acv;
    float* logits_acv;
};


struct SmolLM2
{
    int max_ctx;
    SmolLM2Config config;
    SmolLM2Weights w;
    SmolLM2Acvs a;
};


size_t get_smollm2_weights_nbytes(SmolLM2& t)
{
    size_t nbytes = 0;

    nbytes += t.config.n_vocab * t.config.d_embd;
    for (int i = 0; i < (int)t.config.n_layers; i++) {
        nbytes += t.config.d_embd;
        nbytes += t.config.n_heads    * t.config.d_head * t.config.d_embd;
        nbytes += t.config.n_kv_heads * t.config.d_head * t.config.d_embd;
        nbytes += t.config.n_kv_heads * t.config.d_head * t.config.d_embd;
        nbytes += t.config.n_heads    * t.config.d_head * t.config.d_embd;
        nbytes += t.config.d_embd;
        nbytes += t.config.d_mlp * t.config.d_embd;
        nbytes += t.config.d_mlp * t.config.d_embd;
        nbytes += t.config.d_mlp * t.config.d_embd;
    }
    nbytes += t.config.d_embd;

    nbytes = nbytes * sizeof(Float16);

    return nbytes;
}


size_t get_smollm2_acvs_nbytes(SmolLM2& t)
{
    size_t nbytes = 0;

    nbytes += t.max_ctx * t.config.d_embd;
    for (int i = 0; i < (int)t.config.n_layers; i++) {
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.config.n_heads * t.max_ctx * t.max_ctx;
        nbytes += t.max_ctx * t.config.n_heads * t.config.d_head;
        nbytes += t.max_ctx * t.config.d_mlp;
        nbytes += t.max_ctx * t.config.d_mlp;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
    }
    nbytes += t.max_ctx * t.config.d_embd;

    nbytes = nbytes * sizeof(Float16);

    nbytes += t.config.n_vocab * sizeof(float); // Logits always float

    return nbytes;
}


void alloc_smollm2_weights(Float16* ptr, SmolLM2& t)
{
    const SmolLM2Config& c = t.config;

    t.w.emb_table = ptr;

    Float16* prev_layer_ptr = ptr + c.n_vocab * c.d_embd;
    for (int i = 0; i < (int)c.n_layers; i++) {
        t.w.layers[i].attn_norm = prev_layer_ptr;
        t.w.layers[i].q_proj    = t.w.layers[i].attn_norm + c.d_embd;
        t.w.layers[i].k_proj    = t.w.layers[i].q_proj    + c.n_heads * c.d_head * c.d_embd;
        t.w.layers[i].v_proj    = t.w.layers[i].k_proj    + c.n_kv_heads * c.d_head * c.d_embd;
        t.w.layers[i].o_proj    = t.w.layers[i].v_proj    + c.n_kv_heads * c.d_head * c.d_embd;
        t.w.layers[i].mlp_norm  = t.w.layers[i].o_proj    + c.n_heads * c.d_head * c.d_embd;
        t.w.layers[i].gate_proj = t.w.layers[i].mlp_norm  + c.d_embd;
        t.w.layers[i].up_proj   = t.w.layers[i].gate_proj + c.d_mlp * c.d_embd;
        t.w.layers[i].down_proj = t.w.layers[i].up_proj   + c.d_mlp * c.d_embd;

        prev_layer_ptr = t.w.layers[i].down_proj + c.d_mlp * c.d_embd;
    }
    
    t.w.out_norm = prev_layer_ptr;
}


void alloc_smollm2_acvs(Float16* ptr, SmolLM2& t)
{
    const SmolLM2Config& c = t.config;

    t.a.emb_acv = ptr;

    Float16* prev_layer_ptr = ptr + t.max_ctx * c.d_embd;

    for (int i = 0; i < (int)c.n_layers; i++) {
        t.a.layers[i].attn_norm_acv = prev_layer_ptr;
        t.a.layers[i].res_0_acv     = t.a.layers[i].attn_norm_acv + t.max_ctx * c.d_embd;
        t.a.layers[i].res_1_acv     = t.a.layers[i].res_0_acv     + t.max_ctx * c.d_embd;
        t.a.layers[i].q_proj_acv    = t.a.layers[i].res_1_acv     + t.max_ctx * c.d_embd;
        t.a.layers[i].k_proj_acv    = t.a.layers[i].q_proj_acv    + t.max_ctx * c.d_embd;
        t.a.layers[i].v_proj_acv    = t.a.layers[i].k_proj_acv    + t.max_ctx * c.d_embd;
        t.a.layers[i].o_proj_acv    = t.a.layers[i].v_proj_acv    + t.max_ctx * c.d_embd;
        t.a.layers[i].qk_acv        = t.a.layers[i].o_proj_acv    + t.max_ctx * c.d_embd;
        t.a.layers[i].qkv_acv       = t.a.layers[i].qk_acv        + c.n_heads * t.max_ctx * t.max_ctx;
        t.a.layers[i].mlp_gate_acv  = t.a.layers[i].qkv_acv       + t.max_ctx * c.n_heads * c.d_head;
        t.a.layers[i].mlp_up_acv    = t.a.layers[i].mlp_gate_acv  + t.max_ctx * c.d_mlp;
        t.a.layers[i].mlp_down_acv  = t.a.layers[i].mlp_up_acv    + t.max_ctx * c.d_mlp;
        t.a.layers[i].mlp_norm_acv  = t.a.layers[i].mlp_down_acv  + t.max_ctx * c.d_embd;

        prev_layer_ptr = t.a.layers[i].mlp_norm_acv + t.max_ctx * c.d_embd;
    }

    t.a.out_norm_acv  = prev_layer_ptr;
    t.a.logits_acv    = (float*)(t.a.out_norm_acv + t.max_ctx * c.d_embd); // Always float
}


void alloc_smollm2(SmolLM2& t)
{
    const size_t weights_nbytes = get_smollm2_weights_nbytes(t);
    const size_t acvs_nbytes = get_smollm2_acvs_nbytes(t);
    const size_t total_nbytes = weights_nbytes + acvs_nbytes;

    globals::metrics.weights_nbytes = weights_nbytes;
    globals::metrics.acvs_nbytes = acvs_nbytes;
    globals::metrics.model_nbytes = total_nbytes;

    char* memptr = reinterpret_cast<char*>(std::malloc(total_nbytes));
    if (!memptr) {
        std::fprintf(stderr, "%s: Failed to allocate %ld bytes.\n", __func__, total_nbytes);
        std::exit(-1);
    }

    Float16* weights_ptr = reinterpret_cast<Float16*>(memptr);
    Float16* acvs_ptr = reinterpret_cast<Float16*>(memptr + weights_nbytes);
    alloc_smollm2_weights(weights_ptr, t);
    alloc_smollm2_acvs(acvs_ptr, t);
}

void uninit_smollm2(SmolLM2& t)
{
    std::free(t.w.emb_table);
}


void init_smollm2(SmolLM2& t, int max_ctx, ModelType model_type)
{
    switch (model_type) {
        case ModelType::Small: {
            t.config = {
                .n_vocab = 49152,
                .n_layers = 30,
                .d_embd = 576,
                .n_heads = 9,
                .n_kv_heads = 3,
                .d_head = 64,
                .d_mlp = 1536,
                .rope_theta = 100000.0f,
                .rms_eps = 1e-05f
            };
            break;
        }
        case ModelType::Medium: {
            t.config = {
                .n_vocab = 49152,
                .n_layers = 32,
                .d_embd = 960,
                .n_heads = 15,
                .n_kv_heads = 5,
                .d_head = 64,
                .d_mlp = 2560,
                .rope_theta = 100000.0f,
                .rms_eps = 1e-05f
            };
            break;
        }
        case ModelType::Large: {
            t.config = {
                .n_vocab = 49152,
                .n_layers = 24,
                .d_embd = 2048,
                .n_heads = 32,
                .n_kv_heads = 32,
                .d_head = 64,
                .d_mlp = 8192,
                .rope_theta = 130000.0f,
                .rms_eps = 1e-05f
            };
            break;
        }
    }

    t.max_ctx = max_ctx;
    alloc_smollm2(t);
}

void load_smollm2_checkpoint(SmolLM2& t, const char* ckpt_path)
{
    Timer timer{&globals::metrics.load_time_ms};

    std::FILE* fin = std::fopen(ckpt_path, "rb");

    if (!fin) {
        std::fprintf(stderr, "%s: failed to open %s.\n", __func__, ckpt_path);
        std::exit(-1);
    }

    const int64_t true_magic_no = 0x66326d6c6c6f6d73; // Hex for ASCII string: smollm2f
    int64_t magic_no;
    LLAMA32_ASSERT(fread(&magic_no, sizeof(int64_t), 1, fin) == 1);

    if (magic_no != true_magic_no) {
        fprintf(stderr, "Magic number: %ld failed to match the expected one: %ld.\n", magic_no, true_magic_no);
        fclose(fin);
        exit(-1);
    }

    const size_t weights_nbytes = get_smollm2_weights_nbytes(t);

    LLAMA32_ASSERT(fread(t.w.emb_table, weights_nbytes, 1, fin) == 1);

    fclose(fin);
}


float* forward(SmolLM2& t, const int* tokens, int n_ctx, int start_pos)
{
    Timer timer{&globals::metrics.inference_time_ms};

    // Shorthands
    const SmolLM2Config& c = t.config;
    const LayerWeights* wl = t.w.layers;
    LayerAcvs* al = t.a.layers;


    ops::embed(tokens, t.w.emb_table, t.a.emb_acv, c.n_vocab, n_ctx, c.d_embd, start_pos);

    Float16* next_layer_inp = t.a.emb_acv;

    for (int i = 0; i < t.config.n_layers; i++) {
        ops::copy_tensors(next_layer_inp, al[i].res_0_acv, n_ctx, c.d_embd, start_pos);

        ops::rms_norm(next_layer_inp, wl[i].attn_norm, al[i].attn_norm_acv, n_ctx, c.d_embd, c.rms_eps, start_pos);

        // ATTN
        const int q_dim = c.n_heads * c.d_head;
        const int kv_dim = c.n_kv_heads * c.d_head;
        ops::matmul_2d(al[i].attn_norm_acv, wl[i].q_proj, al[i].q_proj_acv, n_ctx, c.d_embd, q_dim, start_pos);
        ops::matmul_2d(al[i].attn_norm_acv, wl[i].k_proj, al[i].k_proj_acv, n_ctx, c.d_embd, kv_dim, start_pos);
        ops::matmul_2d(al[i].attn_norm_acv, wl[i].v_proj, al[i].v_proj_acv, n_ctx, c.d_embd, kv_dim, start_pos);
        ops::rotary_emb(al[i].q_proj_acv, n_ctx, c.n_heads, c.d_head, c.rope_theta, start_pos);
        ops::rotary_emb(al[i].k_proj_acv, n_ctx, c.n_kv_heads, c.d_head, c.rope_theta, start_pos);

        ops::qk(al[i].q_proj_acv, al[i].k_proj_acv, al[i].qk_acv, n_ctx, c.n_heads, c.n_kv_heads, c.d_head, start_pos);
        ops::attn_mask_inplace(al[i].qk_acv, c.n_heads, n_ctx, start_pos);
        ops::softmax_inplace(al[i].qk_acv, c.n_heads, n_ctx, start_pos);
        ops::qkv(al[i].qk_acv, al[i].v_proj_acv, al[i].qkv_acv, n_ctx, c.n_heads, c.n_kv_heads, c.d_head, start_pos);
        ops::matmul_2d(al[i].qkv_acv, wl[i].o_proj, al[i].o_proj_acv, n_ctx, c.d_embd, c.d_embd, start_pos);

        ops::residual(al[i].o_proj_acv, al[i].res_0_acv, al[i].res_1_acv, n_ctx, c.d_embd, start_pos);

        // MLP:: down(silu(gate(x)) * up(x))
        ops::rms_norm(al[i].res_1_acv, wl[i].mlp_norm, al[i].mlp_norm_acv, n_ctx, c.d_embd, c.rms_eps, start_pos);
        ops::matmul_2d(al[i].mlp_norm_acv, wl[i].gate_proj, al[i].mlp_gate_acv, n_ctx, c.d_embd, c.d_mlp, start_pos);
        ops::matmul_2d(al[i].mlp_norm_acv, wl[i].up_proj, al[i].mlp_up_acv, n_ctx, c.d_embd, c.d_mlp, start_pos);

        ops::silu_inplace(al[i].mlp_gate_acv, n_ctx, c.d_mlp, start_pos);
        ops::mul_inplace(al[i].mlp_gate_acv, al[i].mlp_up_acv, n_ctx, c.d_mlp, start_pos);
        ops::matmul_2d(al[i].mlp_gate_acv, wl[i].down_proj, al[i].mlp_down_acv, n_ctx, c.d_mlp, c.d_embd, start_pos);

        ops::residual(al[i].res_1_acv, al[i].mlp_down_acv, al[i].res_1_acv, n_ctx, c.d_embd, start_pos);

        next_layer_inp = t.a.layers[i].res_1_acv;
    }

    ops::rms_norm(next_layer_inp, t.w.out_norm, t.a.out_norm_acv, n_ctx, c.d_embd, c.rms_eps, start_pos);
    
    ops::lm_head_proj(t.a.out_norm_acv, t.w.emb_table, t.a.logits_acv, c.n_vocab, n_ctx, c.d_embd);

    return t.a.logits_acv;
}


class SmolLMTokenizer {
public:
    const int eot_id = 2;

public:

SmolLMTokenizer(const std::string& vocab_path, int n_vocab)
    : m_n_vocab{n_vocab}
{
    std::ifstream fin{vocab_path, std::ios_base::binary};
    if(!fin.is_open()) {
        std::fprintf(stderr, "Failed to open vocab file: %s\n", vocab_path.c_str());
        std::exit(EXIT_FAILURE);
    };

    std::string word;
    for (int i = 0; i < n_vocab; i++)
    {
        int32_t len;
        fin.read((char *) &len, sizeof(len));

        word.resize(len);
        fin.read((char *) word.data(), len);

        token_to_id_[word] = i;
        id_to_token_[i] = word;
    }
}

// Convert a single token id into text.
const char* decode(int32_t token_id) {
    return id_to_token_[token_id].c_str();
}

// Convert a string of arbitrary text to a sequence of tokens ids.
std::vector<int32_t> encode(const std::string& text) {
    std::vector<std::string> words;

    // first split the text into words
    std::string str = text;
    std::regex re(m_pat);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
        for (auto x : m) {
            words.push_back(x);
        }
        str = m.suffix();
    }

    // find the longest tokens that form the words:
    std::vector<int32_t> tokens;
    tokens.reserve(encode_prefix.size() + words.size() + encode_suffix.size());
    // prepend prefix.
    tokens.insert(tokens.end(), encode_prefix.begin(), encode_prefix.end());

    for (const auto & word : words)
    {
        if (word.size() == 0) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            while (j > i)
            {
                auto it = token_to_id_.find(word.substr(i, j-i));
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    break;
                }
                --j;
            }
            if (i == n)
                break;
            if (j == i)
            {
                auto sub = word.substr(i, 1);
                if (token_to_id_.find(sub) != token_to_id_.end())
                    tokens.push_back(token_to_id_.at(sub));
                else
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                ++i;
            }
        }
    }

    // append suffix.
    tokens.reserve(tokens.size() + encode_suffix.size());
    tokens.insert(tokens.end(), encode_suffix.begin(), encode_suffix.end());

    return tokens;
}

private:
    const std::string m_pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    // Prefix: <|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n
    const std::vector<int> encode_prefix = {1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093, 198};
    // Suffix: <|im_end|>\n<|im_start|>assistant\n
    const std::vector<int> encode_suffix = {2, 198, 1, 520, 9531, 198};
    std::map<std::string, int32_t> token_to_id_;
    std::map<int32_t, std::string> id_to_token_;
    int m_n_vocab;
};


int topk_sample(SmolLM2& t, SmolLMTokenizer& tokenizer, const std::string& prompt, int top_k, float top_p, float temp)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> tokens = tokenizer.encode(prompt);
    if ((int)tokens.size() >= t.max_ctx) {
        std::fprintf(stderr, "Prompt is too large: %d for max context size: %d\n", (int)tokens.size(), t.max_ctx);
        return 0;
    }

    const int logits_size = t.config.n_vocab;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int n_pred_tokens = t.max_ctx - tokens.size();
    for (int i = 0; i < n_pred_tokens; i++) {
        const int start_pos = i == 0 ? 0 : tokens.size()-1;

        const float* logits = forward(t, tokens.data(), tokens.size(), start_pos);

        Timer sample_timer{&globals::metrics.sample_time_ms};

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j) {
            logits_probs.push_back(std::make_pair((double)logits[j] / temp, j));
        }
        
        // Select top k elements.
        std::partial_sort(
                logits_probs.begin(),
                logits_probs.begin() + top_k,
                logits_probs.end(),
                [](const std::pair<double, int> &rhs, const std::pair<double, int> &lhs) {
            return rhs.first > lhs.first;
        });
        logits_probs.resize(top_k);
        
        // compute softmax
        double sum_exp = 0;
        for (int j = 0; j < top_k; ++j) {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j) {
            logits_probs[j].first = logits_probs[j].first / sum_exp;
        }

        // top_p selection
        int top_p_count = top_k;
        double cumulative_prob = 0.0f;
        for (int j = 0; j < top_k; j++) {
            cumulative_prob += logits_probs[j].first;
            if (cumulative_prob >= top_p) {
                top_p_count = j + 1;
                break;
            }
        }

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_p_count; j++) {
            const auto &prob_pair = logits_probs[j];
            probs[prob_pair.second] = prob_pair.first;
        }

        std::discrete_distribution dist(probs.begin(), probs.end());
        const int pred_token = dist(gen);
        if (pred_token == tokenizer.eot_id) {
            break;
        }

        std::printf("%s", tokenizer.decode(pred_token));
        std::fflush(stdout);

        tokens.push_back(pred_token);
    }
    printf("\n");

    return tokens.size();
}


static const char *usage_message = R"(
USAGE:
./llama32 [options] -p PROMPT  for a single prompt or
./llama32 [options] for a chat interface. 

Optional args. 
-f16 :     Use float-16 model (2.3GB). [default]
--npred  N : Max context size. Minimum is 128 and max is 8192 [default=512]. Higher values consume more memory.
)";


int main(int argc, char const *argv[])
{
    Timer timer{&globals::metrics.total_runtime_ms};

    const char* model_path = "models/smollm2-sm.bin";
    const ModelType model_type = ModelType::Small;
    int max_ctx = 512;
    std::string prompt = "";

    for (int i = 1; i < argc; i++) {
        std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            fprintf(stderr, "%s\n.", usage_message);
            return 0;
        }
        else if (arg == "-f16") {
            continue;
        }
        else if (arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[i + 1];
                i += 1; // fast-forward
            } else {
                fprintf(stderr, "error: Prompt not provided.\n");
                fprintf(stderr, "%s\n.", usage_message);
                return -1;
            }
        }
        else if (arg == "--npred") {
            if (argc <= i+1) {
                fprintf(stderr, "npred value is missing.\n");
                return -1;
            }
            int npred;
            try {
                npred = std::stoi(argv[i+1]);
            } catch (...) {
                fprintf(stderr, "Invalid npred value.\n");
                return -1;
            }
            if (npred < 128 || npred > 8192) {
                fprintf(stderr, "npred must be greater than 128 and less than 2048.\n");
                return -1;
            }
            max_ctx = npred;
            i += 1; // skip len param
        }
        else {
            fprintf(stderr, "error: Unknown argument: %s\n", arg.data());
            fprintf(stderr, "%s\n.", usage_message);
            return -1;
        }
    }

// #ifdef _WIN32
//     int res = std::system("python model_dl.py");
// #else
//     int res = std::system("python3 model_dl.py");
// #endif
//     if (res != 0) {
//         fprintf(stderr, "Error: Failed to download the model. Check your network connectivity.\n");
//         return -1;
//     }

    SmolLM2 model;
    init_smollm2(model, max_ctx, model_type);
    load_smollm2_checkpoint(model, model_path);

    // size of the vocab without special tokens eg <|start_of_text|>.
    const int vocab_tok_size = model.config.n_vocab;
    SmolLMTokenizer tokenizer{"tokenizer.bin", vocab_tok_size};

    const int top_k = 40;
    const float top_p = 0.95;
    const float temp = 0.8;

    if (prompt == "") {
        printf("Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n");
        std::string prompt;
        while (true) {
            printf("\n\n[You]: "); fflush(stdout);

            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            printf("\n\n[SmolLM2]: \n"); fflush(stdout);
            
            topk_sample(model, tokenizer, prompt, top_k, top_p, temp);
        } 
    } else {
        printf("\n[PROMPT]:\n%s\n\n[SmolLM2]: ", prompt.c_str());
        std::fflush(stdout);

        const int processed_toks = topk_sample(model, tokenizer, prompt, top_k, top_p, temp);
        timer.stop();
        print_metrics(globals::metrics, processed_toks);
    }

    uninit_smollm2(model);

    return 0;
}


/*

audio = get_audio()
prompt = speech_to_text(audio);
output = llm_stream_answer(speech)

*/
