#include <cstdio>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio.h"
#include "stt.h"

/*
TODO:
    - Catch ctrl+C event and perform shutdown.
    - Add a form of VAD so that silent speech is not forwarded to the model. We could also
      use the model's no_speech recognition as second option.
    - Add translation models to perform live-translation.
*/

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



int main(int argc, char const *argv[])
{
    float speech_threshold = -90.0f;

    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        if (arg == "--speech_threshold") {
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

    stt::WhisperType stt_model_type = stt::WhisperType::Tiny;
    const char* stt_model_name = "acft-whisper-tiny.en.bin";
    int audio_capture_time = 2;

    stt::Whisper whisper;
    stt::whisper_init(whisper, stt_model_type, get_model_path(stt_model_name));

    const int samplerate = 16000;
    const int buffer_size = 30 * samplerate;
    const int audio_block_size  = audio_capture_time * samplerate;
    const int audio_block_capture_size = audio_capture_time * samplerate;
    AudioStream stream;
    AudioPreprocessor apreproc(10);
    VoiceActivityDetector vad(audio_block_size, speech_threshold);

    printf("Speak now (ctrl+c to quit)\n");

    float* audio_sig = new float[10*16000];
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
        while (stream.m_signal.size() < 30*16000) {
            if (stream.m_signal.size() >= block_ctr*audio_block_size + audio_block_size) {
                const float* signal_data = stream.m_signal.data() + block_ctr*audio_block_size;

                const bool has_speech = vad.signal_has_speech(signal_data);
                if (has_speech) {
                    printf("[SPEECH]: ADD FRAME\n");
                    memcpy(audio_sig + n_sig, signal_data, audio_block_size*sizeof(float));
                    n_sig += audio_block_size;
                    first_speech = true;
                    prev_is_silent = false;
                } else if (!has_speech && first_speech && !prev_is_silent) {
                    printf("[NO_SPEECH]: ADD FRAME\n");
                    memcpy(audio_sig + n_sig, signal_data, audio_block_size*sizeof(float));
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
            // TODO: Correct this.
            const Float16* spectrogram = apreproc.get_mel_spectrogram(audio_sig, 10*16000);
            const int n_spec_frames = apreproc.m_out_frames;
            Float16* xa = encoder_forward(spectrogram, n_spec_frames, whisper.enc, whisper.config);
            
            std::string prompt = whisper_decode(whisper, xa, n_spec_frames, true);
        }
    }

    return 0;
}

