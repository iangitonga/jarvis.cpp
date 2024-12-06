#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio.h"
#include "llm/llm.h"
#include "stt/stt.h"


void audio_callback_capture(ma_device* device, void* output, const void* input, ma_uint32 frame_count)
{
    // In capture mode read data from input

    std::vector<float>* signal = (std::vector<float>*)device->pUserData;

    const float* buffer = (float*)input;
    for (int i = 0; i < frame_count; i++) { 
        signal->push_back(buffer[i]);
    }
    
    (void)output;
}


class AudioStream {
public:
    AudioStream() {
        m_device_config = ma_device_config_init(ma_device_type_capture);
        m_device_config.capture.format   = ma_format_f32;
        m_device_config.capture.channels = 1;
        m_device_config.sampleRate       = 16000;
        m_device_config.dataCallback     = audio_callback_capture;
        m_device_config.pUserData        = &m_signal;

        const ma_result result = ma_device_init(NULL, &m_device_config, &m_device);
        if (result != MA_SUCCESS) {
            fprintf(stderr, "Failed to initialize capture device.\n");
            exit(-1);
        }

        // Reserve 30s n_samples.
        const int sample_rate = 16000;
        m_signal.reserve(sample_rate * 30);
    }
    ~AudioStream() {
        ma_device_uninit(&m_device);
    }

    void start_recording() {
        m_signal.clear();

        const ma_result result = ma_device_start(&m_device);
        if (result != MA_SUCCESS) {
            ma_device_uninit(&m_device);
            fprintf(stderr, "Failed to start capture device.\n");
            exit(-1);
        }
    }

    std::vector<float>& stop_recording() {
        ma_device_stop(&m_device);
        return m_signal;
    }

private:
    ma_device_config m_device_config;
    ma_device m_device;
    std::vector<float> m_signal;
};



static const char *usage_message = R"(
USAGE:
./jarvis [options]

Optional args. 
--stt MODEL_SIZE:  The Speech-to-Text model to use. MODEL_SIZE options are (tiny, base, small, medium)[default=base].
--llm MODEL_SIZE:  The LLM to use to respond to prompt. MODEL_SIZE options are (small, medium, large)[default=medium].
--time DURATION :  The maximum number of seconds to record your query. DURATION options are (10, 20, 30)[default=10].

Optional flags.
-testrun: Runs a test in an environment with no microphone, e.g colab with test spectrogram in assets/test_spectrogram.

)";


// TODO: Add memory footprint and metrics.

int main(int argc, char const *argv[])
{
    const char* stt_model_name = "whisper-base.en.bin";
    stt::WhisperType stt_model_type = stt::WhisperType::Base;
    
    const char* llm_model_name = "smollm2-md.bin";
    llm::SmolLM2Type llm_model_type = llm::SmolLM2Type::Medium;

    int record_duration_secs = 10;
    bool testrun_no_audio_inp = false;

    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            printf("%s\n", usage_message);
            return 0;
        }
        else if (arg == "-testrun") {
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
                printf("error: stt option is not provided.\n");
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
        else if (arg == "--time") {
            if (argc <= i+1) {
                fprintf(stderr, "time value is missing.\n");
                return -1;
            }
            int time;
            try {
                time = std::stoi(argv[i+1]);
            } catch (...) {
                fprintf(stderr, "Invalid time value.\n");
                return -1;
            }
            if (!(time == 10 || time == 20 || time == 30)) {
                fprintf(stderr, "time must be one of (10, 20, 30). You provided: %d\n", time);
                return -1;
            }
            record_duration_secs = time;
            i += 1; // skip len param
        }
        else {
            printf("error: unknown option: %s\n", arg.data());
            return -1;
        }
    }

    const std::string download_command = get_model_download_command(stt_model_name, llm_model_name);
    const int res = std::system(download_command.c_str());
    if (res != 0) {
        fprintf(stderr, "Error: Failed to download the models. Check your network connectivity.\n");
        return -1;
    }

    printf("JARVIS\nMax query record duration is %d secs.\n\n", record_duration_secs);

    int64_t load_time_ms = 0;

    Timer load_timer{&load_time_ms};
    stt::Whisper whisper;
    stt::whisper_init(whisper, stt_model_type, get_model_path(stt_model_name));

    int max_ctx = 1024;
    llm::SmolLM2 smollm2;
    llm::smollm2_init(smollm2, llm_model_type, max_ctx, get_model_path(llm_model_name));

    load_timer.stop();

    if (testrun_no_audio_inp) {
        std::unique_ptr<Float16> spectrogram{new Float16[3000*80]};
        stt::read_test_spectrogram(spectrogram.get());
        const int n_spec_frames = record_duration_secs * 100;

        int64_t encoder_time_ms;
        int64_t decoder_time_ms;
        int64_t llm_time_ms;

        Timer enc_timer{&encoder_time_ms};
        Float16* xa = encoder_forward(spectrogram.get(), n_spec_frames, whisper.enc, whisper.config);
        enc_timer.stop();

        Timer dec_timer{&decoder_time_ms};
        std::string prompt = whisper_decode(whisper, xa, n_spec_frames, /*stream*/true);
        dec_timer.stop();
 
        // PROMPT Answering.
        printf("\n[LLM]: \n\n"); fflush(stdout);
        Timer llm_timer{&llm_time_ms};
        const int n_llm_tokens = topk_sample(smollm2, prompt);
        llm_timer.stop();
        printf("\n\n");

        // TODO: Add stt toks/sec.
        const float llm_toks_per_sec = (float)n_llm_tokens / ((float)llm_time_ms/1000.0f);
        printf("timer: llm_toks/sec: %4.2f\n", llm_toks_per_sec);
        printf("timer: encoder_time: %ldms\n", encoder_time_ms);
        printf("timer: decoder_time: %ldms\n", decoder_time_ms);
        printf("timer: load_time   : %ldms\n", load_time_ms);
        printf("timer: matmul_ops  : %ldms\n", ops::ops_metrics.matmul_ms);
        printf("timer: other_ops   : %ldms\n", ops::ops_metrics.non_matmul_ms);
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

            // TODO: Add recorder timeout.
            stream.start_recording();
            printf("\nRecording. Enter to stop the recording ...");
            std::cin.get();

            std::vector<float>& signal = stream.stop_recording();
            printf("\nRecording complete. Converting Audio...\n\n");

            // pad to record_duration_secs. audio less than 10 secs is not decoded without severe repetition.
            if (signal.size() < kSampleRate*record_duration_secs) {
                const int deficit = kSampleRate*record_duration_secs - signal.size();
                for (int i = 0; i < deficit; i++) {
                    signal.push_back(0);
                }
            }

            // SPEECH-TO-TEXT.
            int n_spec_frames;
            const Float16* spectrogram = apreproc.get_mel_spectrogram(signal, &n_spec_frames);
            Float16* xa = encoder_forward(spectrogram, n_spec_frames, whisper.enc, whisper.config);
            
            std::string prompt = whisper_decode(whisper, xa, n_spec_frames, /*stream*/true);
            // printf("PROMPT: %s\n", prompt.c_str());

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

