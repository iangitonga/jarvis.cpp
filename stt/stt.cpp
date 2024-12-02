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


class AudioBuffer {
public:
    /// @brief Create an audio data ring-buffer.
    /// @param capacity The maximum number of frames the buffer should hold before wrapping to the start.
    /// @param capture_block_size The number of captured audio frames in audio block.
    /// @param out_block_size The total number of frames in audio block. If it is greater than
    ///  `capture_block_size`, it is padded with zeros.
    /// @param overlap_frames The number of frames an audio block overlaps with previous audio block. This
    /// allows for speech that is cut-off from the current block to be recovered in the next block.
    AudioBuffer(int capacity, int capture_block_size, int out_block_size, int overlap_frames) {
        m_data_capacity = capacity;
        m_data_size.store(0);
        m_block_ctr = 0;
        m_capture_block_size = capture_block_size;
        m_out_block_size = out_block_size;
        m_overlap_frames = overlap_frames;
        m_data_internal = (float*)jarvis_malloc(m_data_capacity * sizeof(float));
        m_data_external = (float*)jarvis_malloc(m_out_block_size * sizeof(float));
        memset(m_data_external, 0, m_out_block_size*sizeof(float));
    }
    ~AudioBuffer() {
        jarvis_free(m_data_internal);
        jarvis_free(m_data_external);
    }

    void add_audio_block(const float* buffer, ma_uint32 frame_count) {
        // NOTE: this is a real-time function. It must run in less than
        // frame_count/samplerate secs to avoid dropping frames.
        // frame_count=1024 ~60ms.
        // frame_count=1600 ~100ms.
        const int offs = m_data_size.load();
        for (int i = 0; i < frame_count; i++) { 
            m_data_internal[(offs + i) % m_data_capacity] = buffer[i];
        }
        m_data_size.fetch_add(frame_count); 
    }

    const float* get_next_audio_block() {
        const int next_block_pos =  m_block_ctr * m_capture_block_size - m_block_ctr * m_overlap_frames;
        const int check_size = next_block_pos + m_capture_block_size;
        // wait until data for the next block is available.
        while (m_data_size.load() < check_size) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        for (int i = 0; i < m_capture_block_size; i++) {
            m_data_external[i] = m_data_internal[(next_block_pos + i) % m_data_capacity];
        }
        
        m_block_ctr += 1;

        return m_data_external;
    }
private:
    int m_data_capacity; 
    // size of written data, including the overwritten data.
    std::atomic_int m_data_size;
    int m_block_ctr;
    int m_capture_block_size;
    int m_out_block_size;
    int m_overlap_frames;
    float* m_data_internal;
    float* m_data_external;
};


// Called from audio thread.
void audio_stream_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count)
{
    const float* buffer = (float*)input;
    AudioBuffer* audio_buffer = (AudioBuffer*)device->pUserData; 
    audio_buffer->add_audio_block(buffer, frame_count);
    
    (void)output;
}

void capture(AudioBuffer* audio_buffer) {
    ma_device_config m_device_config;
    ma_device m_device;
    ma_result result;

    m_device_config = ma_device_config_init(ma_device_type_capture);
    m_device_config.capture.format   = ma_format_f32;
    m_device_config.capture.channels = 1;
    m_device_config.sampleRate       = 16000;
    m_device_config.dataCallback     = audio_stream_callback;
    m_device_config.pUserData = audio_buffer;

    result = ma_device_init(NULL, &m_device_config, &m_device);
    if (result != MA_SUCCESS) {
        fprintf(stderr, "Failed to initialize capture device.\n");
        exit(-1);
    }

    result = ma_device_start(&m_device);
    if (result != MA_SUCCESS) {
        ma_device_uninit(&m_device);
        fprintf(stderr, "Failed to start capture device.\n");
        exit(-1);
    }
    printf("recording ...\n");

    // TODO: We should wait for a conditional variable to uninit device.
    std::cin.get();
    // ma_device_uninit(&dev);
}

static const char *usage_message = R"(
USAGE:
./whisper [options]

Optional args. 
--stt MODEL_SIZE:  The Speech-to-Text model to use. MODEL_SIZE options are (tiny, base, small)[default=base].
)";


int main(int argc, char const *argv[])
{
    stt::WhisperType stt_model_type = stt::WhisperType::Base;
    const char* stt_model_name = "acft-whisper-base.en.bin";
    int audio_capture_time = 2;

    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            printf("%s\n", usage_message);
            return 0;
        }
        else if (arg == "--stt") {
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
        else if (arg == "--capsize") {
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
            if (time < 1 || time > 3) {
                fprintf(stderr, "time must be greater than 1 and less than or eq to 3. You provided: %d\n", time);
                return -1;
            }
            audio_capture_time = time;
            i += 1; // skip len param
        }
        else {
            printf("error: unknown option: %s\n", arg.data());
            return -1;
        }
    }

    const std::string download_command = get_model_download_command(stt_model_name);
    const int res = std::system(download_command.c_str());
    if (res != 0) {
        fprintf(stderr, "Error: Failed to download the models. Check your network connectivity.\n");
        return -1;
    }

    stt::Whisper whisper;
    stt::whisper_init(whisper, stt_model_type, get_model_path(stt_model_name));

    const int samplerate = 16000;
    const int buffer_size = 30 * samplerate;
    const int audio_block_size  = 3 * samplerate;
    const int audio_block_capture_size = audio_capture_time * samplerate;
    AudioBuffer audio_buffer(buffer_size, audio_block_capture_size, audio_block_size, /*overlap_frames=*/1000);
    AudioPreprocessor apreproc{audio_block_size/samplerate};

    std::thread capture_thread(capture, &audio_buffer);
    capture_thread.detach();

    printf("Speak now (ctrl+c to quit)\n");

    // TODO: Catch ctrl+c.
    while (true) {
        const float* signal_data = audio_buffer.get_next_audio_block();
        const Float16* spectrogram = apreproc.get_mel_spectrogram(signal_data, audio_block_size);
        const int n_spec_frames = apreproc.m_out_frames;
        Float16* xa = encoder_forward(spectrogram, n_spec_frames, whisper.enc, whisper.config);
        
        whisper_decode_stream(whisper, xa, n_spec_frames);
    }

    return 0;
}

