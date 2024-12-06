#include <cstdio>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <memory>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio.h"
#include "stt.h"


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
        m_signal.reserve(sample_rate * 60);
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
public:
    std::vector<float> m_signal;
};


static const char *usage_message = R"(
USAGE:
./vadtest [options]

Optional args. 
--stt MODEL_SIZE:  The Speech-to-Text model to use. MODEL_SIZE options are (tiny, base, small)[default=base].
--speech_threshold T: The vad speech threshold. Energy above threshold is speech and less is silence.
)";

int main(int argc, char const *argv[])
{
    stt::WhisperType stt_model_type = stt::WhisperType::Tiny;
    const char* stt_model_name = "acft-whisper-tiny.en.bin";
    float speech_threshold = -90.0f;

    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        if (arg == "--stt") {
            if (i + 1 < argc) {
                const std::string_view stt_arg{argv[i + 1]};
                if (stt_arg == "tiny") {
                    stt_model_name = "acft-whisper-tiny.en.bin";
                    stt_model_type = stt::WhisperType::Tiny;
                }
                else if (stt_arg == "base") {
                    stt_model_name = "acft-whisper-base.en.bin";
                    stt_model_type = stt::WhisperType::Base;
                }
                else if (stt_arg == "small") {
                    stt_model_name = "acft-whisper-small.en.bin";
                    stt_model_type = stt::WhisperType::Small;
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
        else if (arg == "--speech_threshold") {
            if (argc <= i+1) {
                fprintf(stderr, "speech_threshold value is missing.\n");
                return -1;
            }
            float threshold;
            try {
                threshold = std::stof(argv[i+1]);
            } catch (...) {
                fprintf(stderr, "Invalid threshold value.\n");
                return -1;
            }
            speech_threshold = threshold;
            i += 1; // skip len param
        }
        else {
            printf("error: unknown option: %s\n", arg.data());
            return -1;
        }
    }

    stt::Whisper whisper;
    stt::whisper_init(whisper, stt_model_type, get_model_path(stt_model_name));

    int audio_capture_time = 2;
    const int audio_block_size  = audio_capture_time * kSampleRate;

    AudioStream stream;
    AudioPreprocessor apreproc;
    VoiceActivityDetector vad(speech_threshold);

    std::unique_ptr<float> audio_sig(new float[10*kSampleRate]);
    int n_sig = 0;

    std::string cmd_input;
    while (true) {
        printf("Press enter to begin recording (enter q to quit) ...");
        std::getline(std::cin, cmd_input);
        if (cmd_input == "q") {
            break;
        }

        stream.start_recording();
        printf("\nRecording ...\n");
        
        int block_ctr = 0;
        bool first_speech = false;
        bool prev_is_silent = false;
        while (stream.m_signal.size() < 30*kSampleRate) {
            if (stream.m_signal.size() >= block_ctr*audio_block_size + audio_block_size) {
                const float* signal_data = stream.m_signal.data() + block_ctr*audio_block_size;

                const bool has_speech = vad.signal_has_speech(signal_data, audio_block_size);
                if (has_speech) {
                    printf("[SPEECH]: ADD FRAME\n");
                    memcpy(audio_sig.get() + n_sig, signal_data, audio_block_size*sizeof(float));
                    n_sig += audio_block_size;
                    first_speech = true;
                    prev_is_silent = false;
                } else if (!has_speech && first_speech && !prev_is_silent) {
                    printf("[NO_SPEECH]: ADD FRAME\n");
                    memcpy(audio_sig.get() + n_sig, signal_data, audio_block_size*sizeof(float));
                    n_sig += audio_block_size;
                    prev_is_silent = true;
                }
                else if (!has_speech && first_speech && prev_is_silent) {
                    printf("[NO SPEECH]: BREAK\n");
                    break;
                } else { // no_speech, !first_speech || no_speech, !first_speech
                    printf("[NO_SPEECH]: DROP FRAME.\n");
                    prev_is_silent = true;
                }

                block_ctr += 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        stream.stop_recording();

        if (first_speech) {
            int n_spec_frames;
            const int n_sig_samples = block_ctr*audio_block_size;
            const Float16* spectrogram = apreproc.get_mel_spectrogram(audio_sig.get(), n_sig_samples, &n_spec_frames);
            Float16* xa = encoder_forward(spectrogram, n_spec_frames, whisper.enc, whisper.config);
            
            std::string prompt = whisper_decode(whisper, xa, n_spec_frames, true);
        }
    }

    return 0;
}

