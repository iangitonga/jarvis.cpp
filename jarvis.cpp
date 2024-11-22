#include <cstdlib>
#include <cstdio>
#include <vector>
#include <iostream>

#include "audio.h"
#include "stt/whisper.h"
#include "llm/smollm2.h"


static const char *usage_message = R"(
USAGE:
./jarvis [options]

Optional args. 
-llm :  The LLM to use to respond to prompt. Options are (small, medium, large)[default=small].
)";
// -stt :  The Speech-to-Text model to use. Options are (tiny, base, small, medium)[default=tiny].

// TODO: Add memory footprint and metrics.


int main(int argc, char const *argv[])
{
    const char* stt_model_path = "models/whisper-tiny.en.bin";
    
    const char* llm_model_path = "models/smollm2-sm.bin";
    ModelType llm_model_size = ModelType::Small;

    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            printf("%s\n", usage_message);
            return 0;
        }
        else if (arg == "--llm") {
            if (i + 1 < argc) {
                const std::string_view llm_arg{argv[i + 1]};
                if (llm_arg == "small") {
                    llm_model_path = "models/smollm2-sm.bin";
                    llm_model_size = ModelType::Small;
                }
                else if (llm_arg == "medium") {
                    llm_model_path = "models/smollm2-md.bin";
                    llm_model_size = ModelType::Medium;
                }
                else if (llm_arg == "large") {
                    llm_model_path = "models/smollm2-lg.bin";
                    llm_model_size = ModelType::Large;
                } else {
                    printf("error: invalid llm option: %s.\n", llm_arg.data());
                    printf("%s\n", usage_message);
                    return -1;
                }
                i += 1; // fast-forward
            } else {
                printf("error: llm option is not provided.\n");
                printf("%s\n", usage_message);
                return -1;
            }
        } else {
            printf("error: unknown option: %s\n", arg.data());
            return -1;
        }
    }

    
#ifdef _WIN32
    const std::string cmd_download_command = std::string("python model_dl.py ") + stt_model_path + " " + llm_model_path;
#else
    const std::string cmd_download_command = std::string("python3 model_dl.py ") + stt_model_path + " " + llm_model_path;
#endif

    int res = std::system(cmd_download_command.c_str());
    if (res != 0) {
        fprintf(stderr, "Error: Failed to download the models. Check your network connectivity.\n");
        return -1;
    }

    AudioStream stream;
    AudioPreprocessor apreproc;

    WhisperTokenizer whisper_tokenizer;
    const Config stt_config{};
    Whisper whisper{};
    alloc_whisper(whisper, stt_config);
    load_whisper(stt_model_path, whisper, stt_config);

    // TODO: ALLOW MAX_CTX AS OPTION.
    int max_ctx = 256;
    SmolLM2 smollm2;
    alloc_smollm2(smollm2, max_ctx, llm_model_size);
    load_smollm2_checkpoint(smollm2, llm_model_path);

    // size of the vocab without special tokens eg <|start_of_text|>.
    const int vocab_tok_size = smollm2.config.n_vocab;
    SmolLMTokenizer smollm2_tokenizer{"assets/smollm2_tokenizer.bin", vocab_tok_size};

    std::string cmd_input;
    while (true) {
        printf("Press enter to begin recording (enter q to quit) ...");
        std::getline(std::cin, cmd_input);
        if (cmd_input == "q")
            break;

        stream.start_recording();
        printf("\nRecording. Enter to stop the recording ...");
        std::cin.get();

        std::vector<float>& signal = stream.stop_recording();
        printf("\nRecording complete. Converting Audio...\n\n");

        // SPEECH-TO-TEXT.
        const Float16* spectrogram = apreproc.get_mel_spectrogram(signal);
        Float16* xa = encoder_forward(spectrogram, whisper.enc, stt_config);
        std::string prompt;
        std::vector<int> tokens = {whisper_tokenizer.sot, whisper_tokenizer.no_timestamps};
        for (int i = 0; i < stt_config.n_text_ctx/2; i++) {
            const int start_pos = i == 0 ? 0 : tokens.size() - 1;
            float* logits = decoder_forward(tokens.data(), tokens.size(), xa, whisper.dec, stt_config, start_pos);

            /// TODO: Add logits suppression.
            int pred_token = -1;
            float max_prob = -INFINITY;
            // Only consider non_special tokens.
            for (int i = 0; i < 50257; i++) {
                if (logits[i] > max_prob) {
                    max_prob = logits[i];
                    pred_token = i;
                }
            }

            if (pred_token == whisper_tokenizer.eot) {
                break;
            }

            // printf("%s", tokenizer.decode_token(pred_token).c_str());
            // fflush(stdout);

            prompt.append(whisper_tokenizer.decode_token(pred_token));

            tokens.push_back(pred_token);
        }
        // Remove a leading whitespace which is produced by whisper tokenizer.
        if (prompt[0] == ' ') {
            prompt = prompt.substr(1, prompt.size());
        }
        printf("PROMPT: %s\n", prompt.c_str());

        // PROMPT Answering.
        printf("\n\n[LLM]: \n\n"); fflush(stdout);
        topk_sample(smollm2, smollm2_tokenizer, prompt);
        printf("\n\n");
    }

    free_whisper(whisper, stt_config);
    free_smollm2(smollm2);

    // printf("Matmul ms: %ld\n", globals::metrics.matmul_ms);
    // printf("NonMatmul ms: %ld\n", globals::metrics.non_matmul_ms);
    
    return 0;
}

