#include <cstdlib>
#include <cstdio>
#include <vector>
#include <iostream>
#include <memory>

#include "audio.h"
#include "stt/stt.h"
#include "llm/llm.h"



static const char *usage_message = R"(
USAGE:
./jarvis [options]

Optional args. 
--stt MODEL_SIZE:  The Speech-to-Text model to use. MODEL_SIZE options are (tiny, base, small, medium)[default=base].
--llm MODEL_SIZE:  The LLM to use to respond to prompt. MODEL_SIZE options are (small, medium, large)[default=medium].

Optional flags.
-testna: Runs a test in an environment with no microphone, e.g colab with test spectrogram in assets/test_spectrogram.

)";


// TODO: Add memory footprint and metrics.

std::string get_model_path(const char* model_name) {
    return std::string("models/") + model_name;
}

int main(int argc, char const *argv[])
{
    const char* stt_model_name = "whisper-base.en.bin";
    stt::WhisperType stt_model_type = stt::WhisperType::Base;
    
    const char* llm_model_name = "smollm2-md.bin";
    llm::SmolLM2Type llm_model_type = llm::SmolLM2Type::Medium;

    bool testrun_no_audio_inp = false;

    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            printf("%s\n", usage_message);
            return 0;
        }
        else if (arg == "-testna") {
            testrun_no_audio_inp = true;
        }
        else if (arg == "--stt") {
            if (i + 1 < argc) {
                const std::string_view stt_arg{argv[i + 1]};
                if (stt_arg == "tiny") {
                    stt_model_name = "whisper-tiny.en.bin";
                    stt_model_type = stt::WhisperType::Tiny;
                }
                else if (stt_arg == "base") {
                    stt_model_name = "whisper-base.en.bin";
                    stt_model_type = stt::WhisperType::Base;
                }
                else if (stt_arg == "small") {
                    stt_model_name = "whisper-small.en.bin";
                    stt_model_type = stt::WhisperType::Small;
                }
                else if (stt_arg == "medium") {
                    stt_model_name = "whisper-medium.en.bin";
                    stt_model_type = stt::WhisperType::Medium;
                }
                else {
                    printf("error: invalid stt option: %s.\n", stt_arg.data());
                    printf("%s\n", usage_message);
                    return -1;
                }
                i += 1; // fast-forward
            } else {
                printf("error: llm option is not provided.\n");
                printf("%s\n", usage_message);
                return -1;
            }
        }
        else if (arg == "--llm") {
            if (i + 1 < argc) {
                const std::string_view llm_arg{argv[i + 1]};
                if (llm_arg == "small") {
                    llm_model_name = "smollm2-sm.bin";
                    llm_model_type = llm::SmolLM2Type::Small;
                }
                else if (llm_arg == "medium") {
                    llm_model_name = "smollm2-md.bin";
                    llm_model_type = llm::SmolLM2Type::Medium;
                }
                else if (llm_arg == "large") {
                    llm_model_name = "smollm2-lg.bin";
                    llm_model_type = llm::SmolLM2Type::Large;
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
        }
        else {
            printf("error: unknown option: %s\n", arg.data());
            return -1;
        }
    }

    
#ifdef _WIN32
    const std::string cmd_download_command = std::string("python model_dl.py ") + stt_model_name + " " + llm_model_name;
#else
    const std::string cmd_download_command = std::string("python3 model_dl.py ") + stt_model_name + " " + llm_model_name;
#endif

    int res = std::system(cmd_download_command.c_str());
    if (res != 0) {
        fprintf(stderr, "Error: Failed to download the models. Check your network connectivity.\n");
        return -1;
    }
    stt::Whisper whisper;
    stt::whisper_init(whisper, stt_model_type, get_model_path(stt_model_name));

    int max_ctx = 512;
    llm::SmolLM2 smollm2;
    llm::smollm2_init(smollm2, llm_model_type, max_ctx, get_model_path(llm_model_name));

    if (testrun_no_audio_inp) {
        std::unique_ptr<Float16> spectrogram{new Float16[3000*80]};
        stt::read_test_spectrogram(spectrogram.get());

        Float16* xa = encoder_forward(spectrogram.get(), whisper.enc, whisper.config);
        std::string prompt = whisper_decode(whisper, xa, /*stream*/true);

        // PROMPT Answering.
        printf("\n[LLM]: \n\n"); fflush(stdout);
        topk_sample(smollm2, prompt);
        printf("\n\n");

        printf("Matmul ms: %ld\n", globals::metrics.matmul_ms);
        printf("NonMatmul ms: %ld\n", globals::metrics.non_matmul_ms);
    }
    else {
        AudioStream stream;
        AudioPreprocessor apreproc;

        std::string cmd_input;
        printf("\n");
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
            Float16* xa = encoder_forward(spectrogram, whisper.enc, whisper.config);
            
            std::string prompt = whisper_decode(whisper, xa);
            printf("PROMPT: %s\n", prompt.c_str());

            // PROMPT Answering.
            printf("\n\n[LLM]: \n\n"); fflush(stdout);
            topk_sample(smollm2, prompt);
            printf("\n\n");
        }
    }

    whisper_uninit(whisper);
    smollm2_uninit(smollm2);
    
    return 0;
}

