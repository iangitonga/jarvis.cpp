#pragma once

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "common.h"


/*
audio.h [const k = 0]
audio.cpp [global::k]
jarvis.cpp [global::k]
maximum audio block/chunk lenth in seconds
during linking [error: multiple definitions of symbol k]
*/

inline constexpr int kSampleRate = 16000;
// Maximum length (in secs) of an audio chunk.
inline constexpr int kMaxChunkLength = 30; // in secs
inline constexpr int kMaxChunkFrames = kMaxChunkLength*kSampleRate;


// Implements simple energy-based thresholding VAD. Fast but not robust.
class VoiceActivityDetector {
public:
    VoiceActivityDetector(float speech_threshold = -90.0f) {
		speech_threshold_ = speech_threshold;
		const int max_out_frames = get_num_out_frames(kMaxChunkFrames);
		fft_out_ = jarvis_malloc_f32(max_out_frames * n_freqs_);
		fft_energy_ = jarvis_malloc_f32(max_out_frames);
		sin_cache_ = jarvis_malloc_f32(n_fft_ * n_freqs_);
		cos_cache_ = jarvis_malloc_f32(n_fft_ * n_freqs_);

		for (int f = 0; f < n_freqs_; ++f) {
			for (int t = 0; t < n_fft_; ++t) {
				cos_cache_[f * n_fft_ + t] = std::cos((2.0f * 3.141592653589793f * f * t) / static_cast<float>(n_fft_));
				sin_cache_[f * n_fft_ + t] = std::sin((2.0f * 3.141592653589793f * f * t) / static_cast<float>(n_fft_));
			}
		}
    }
	~VoiceActivityDetector() {
		jarvis_free(fft_out_);
		jarvis_free(fft_energy_);
		jarvis_free(sin_cache_);
		jarvis_free(cos_cache_);
	}

	bool signal_has_speech(const float* sig, int n_sig_frames) {
		JARVIS_ASSERT(n_sig_frames < kMaxChunkFrames);

		float* vad_out = spectral_vad(sig, n_sig_frames);

		const int n_out_frames = get_num_out_frames(n_sig_frames);
		// float min = INFINITY;
		// float max = -INFINITY;
		float mean_energy = 0.0f;
		for (int i = 0; i < n_out_frames; i++) {
			float val = vad_out[i];
			mean_energy += val;
			// if (val < min) { min = val; }
			// if (val > max) {max = val;}
		}
		mean_energy = mean_energy / n_out_frames;
		// printf("(min=%f, max=%f, mean=%f)\n", min, max, mean_energy);

		// for (int i = 0; i < n_out_frames; i++) {
		// 	const float frame_energy = vad_out[i];
		// 	if (frame_energy > speech_threshold_) {
		// 		return true;
		// 	}
		// }
		// return false;
		if (mean_energy > speech_threshold_) {
			return true;
		}
		return false;
	}

	float* spectral_vad(const float* sig, int n_sig_frames) {
		const int n_out_frames = get_num_out_frames(n_sig_frames);

		// Compute frequency magnitudes: abs(rfft(x))**2
		for (int i = 0; i < n_out_frames; ++i) {
			for (int f = 0; f < n_freqs_; ++f) {
				float sum_real = 0.0f;
				float sum_imag = 0.0f;
				for (int t = 0; t < n_fft_; ++t) {
					sum_real += sig[i * fft_hop_length_ + t] * cos_cache_[f * n_fft_ + t];
					sum_imag += sig[i * fft_hop_length_ + t] * -1.0 * sin_cache_[f * n_fft_ + t];
				}
				fft_out_[i * n_freqs_ + f] = sum_real * sum_real + sum_imag * sum_imag;
			}
		}

		// COMPUTE ENERGY
		for (int i = 0; i < n_out_frames; i++){
			float sum = 0.0f;
			for (int j = 0; j < n_freqs_; j++){
				sum += fft_out_[i * n_freqs_ + j];
			}
			fft_energy_[i] = sum / (n_out_frames*n_out_frames);
		}
		
		// normalize energy to 0 dB then filter
		for (int i = 0; i < n_out_frames; i++) {
			fft_energy_[i] = 10.0f * log10f(fft_energy_[i] / energy_threshold_);
		}

		const int n_filter = 5;
		const int pad_size = 4;
		float sorted[n_filter];
		for (int i = 0; i < n_out_frames-pad_size; i++) {
			// sort the 5
			// find medium, middle guy
			memcpy(sorted, fft_energy_ + i, n_filter*sizeof(float));
			sort_array(sorted, n_filter);
			const float median = sorted[2];
			fft_energy_[i] = median;
		}
		
		// speech_frames = where(fft_energy > speech_threshold)
		return fft_energy_;
	}

private:
	int get_num_out_frames(int n_in_frames) {
		// not-padded.
		return ((n_in_frames - n_fft_) / fft_hop_length_) + 1;
	}

	// fast for small arrays (n=5).
	void sort_array(float* inp, int size) {
		for (int i = 1; i < size; i++) {
			for (int j = i; j >= 1; j--) {
				const float cur_val = inp[j];
				const float prev_val = inp[j - 1];
				if (prev_val > cur_val) {
					inp[j] = prev_val;  
					inp[j-1] = cur_val;
				}
			}
		}
	}
private:
    const int n_fft_ = 400; // 25ms
    const int fft_hop_length_ = 400;
    const int n_freqs_ = n_fft_ / 2 + 1;
    // size of the median filter window.
    const int n_filter_ = 5;
    // Energy value characterizing the silence to speech energy ratio.
    const float energy_threshold_ = 1e7f;
    // Threshold where energy above is considered speech and energy below is silence.
    float speech_threshold_;
    float* fft_out_;
    float* fft_energy_;
    float* sin_cache_;
    float* cos_cache_;
};


// Hanning window as computed by pytorch. size=400, mem=1.6Kb.
const float HANN_WINDOW[] = {
	0.0000e+00, 6.1691e-05, 2.4673e-04, 5.5507e-04, 9.8664e-04, 1.5413e-03,
    2.2190e-03, 3.0195e-03, 3.9426e-03, 4.9882e-03, 6.1558e-03, 7.4453e-03,
    8.8564e-03, 1.0389e-02, 1.2042e-02, 1.3815e-02, 1.5708e-02, 1.7721e-02,
    1.9853e-02, 2.2103e-02, 2.4472e-02, 2.6957e-02, 2.9560e-02, 3.2278e-02,
    3.5112e-02, 3.8060e-02, 4.1123e-02, 4.4298e-02, 4.7586e-02, 5.0986e-02,
    5.4497e-02, 5.8117e-02, 6.1847e-02, 6.5684e-02, 6.9629e-02, 7.3680e-02,
    7.7836e-02, 8.2096e-02, 8.6460e-02, 9.0925e-02, 9.5491e-02, 1.0016e-01,
    1.0492e-01, 1.0978e-01, 1.1474e-01, 1.1980e-01, 1.2494e-01, 1.3018e-01,
    1.3552e-01, 1.4094e-01, 1.4645e-01, 1.5204e-01, 1.5773e-01, 1.6349e-01,
    1.6934e-01, 1.7528e-01, 1.8129e-01, 1.8738e-01, 1.9355e-01, 1.9979e-01,
    2.0611e-01, 2.1250e-01, 2.1896e-01, 2.2549e-01, 2.3209e-01, 2.3875e-01,
    2.4548e-01, 2.5227e-01, 2.5912e-01, 2.6604e-01, 2.7300e-01, 2.8003e-01,
    2.8711e-01, 2.9424e-01, 3.0143e-01, 3.0866e-01, 3.1594e-01, 3.2326e-01,
    3.3063e-01, 3.3804e-01, 3.4549e-01, 3.5298e-01, 3.6050e-01, 3.6806e-01,
    3.7566e-01, 3.8328e-01, 3.9093e-01, 3.9861e-01, 4.0631e-01, 4.1404e-01,
    4.2178e-01, 4.2955e-01, 4.3733e-01, 4.4513e-01, 4.5295e-01, 4.6077e-01,
    4.6860e-01, 4.7645e-01, 4.8429e-01, 4.9215e-01, 5.0000e-01, 5.0785e-01,
    5.1571e-01, 5.2355e-01, 5.3140e-01, 5.3923e-01, 5.4705e-01, 5.5487e-01,
    5.6267e-01, 5.7045e-01, 5.7822e-01, 5.8596e-01, 5.9369e-01, 6.0139e-01,
    6.0907e-01, 6.1672e-01, 6.2435e-01, 6.3194e-01, 6.3950e-01, 6.4702e-01,
    6.5451e-01, 6.6196e-01, 6.6937e-01, 6.7674e-01, 6.8406e-01, 6.9134e-01,
    6.9857e-01, 7.0576e-01, 7.1289e-01, 7.1997e-01, 7.2700e-01, 7.3396e-01,
    7.4088e-01, 7.4773e-01, 7.5452e-01, 7.6125e-01, 7.6791e-01, 7.7451e-01,
    7.8104e-01, 7.8750e-01, 7.9389e-01, 8.0021e-01, 8.0645e-01, 8.1262e-01,
    8.1871e-01, 8.2472e-01, 8.3066e-01, 8.3651e-01, 8.4227e-01, 8.4796e-01,
    8.5355e-01, 8.5906e-01, 8.6448e-01, 8.6982e-01, 8.7506e-01, 8.8020e-01,
    8.8526e-01, 8.9022e-01, 8.9508e-01, 8.9984e-01, 9.0451e-01, 9.0907e-01,
    9.1354e-01, 9.1790e-01, 9.2216e-01, 9.2632e-01, 9.3037e-01, 9.3432e-01,
    9.3815e-01, 9.4188e-01, 9.4550e-01, 9.4901e-01, 9.5241e-01, 9.5570e-01,
    9.5888e-01, 9.6194e-01, 9.6489e-01, 9.6772e-01, 9.7044e-01, 9.7304e-01,
    9.7553e-01, 9.7790e-01, 9.8015e-01, 9.8228e-01, 9.8429e-01, 9.8618e-01,
    9.8796e-01, 9.8961e-01, 9.9114e-01, 9.9255e-01, 9.9384e-01, 9.9501e-01,
    9.9606e-01, 9.9698e-01, 9.9778e-01, 9.9846e-01, 9.9901e-01, 9.9944e-01,
    9.9975e-01, 9.9994e-01, 1.0000e+00, 9.9994e-01, 9.9975e-01, 9.9944e-01,
    9.9901e-01, 9.9846e-01, 9.9778e-01, 9.9698e-01, 9.9606e-01, 9.9501e-01,
    9.9384e-01, 9.9255e-01, 9.9114e-01, 9.8961e-01, 9.8796e-01, 9.8618e-01,
    9.8429e-01, 9.8228e-01, 9.8015e-01, 9.7790e-01, 9.7553e-01, 9.7304e-01,
    9.7044e-01, 9.6772e-01, 9.6489e-01, 9.6194e-01, 9.5888e-01, 9.5570e-01,
    9.5241e-01, 9.4901e-01, 9.4550e-01, 9.4188e-01, 9.3815e-01, 9.3432e-01,
    9.3037e-01, 9.2632e-01, 9.2216e-01, 9.1790e-01, 9.1354e-01, 9.0907e-01,
    9.0451e-01, 8.9984e-01, 8.9508e-01, 8.9022e-01, 8.8526e-01, 8.8020e-01,
    8.7506e-01, 8.6982e-01, 8.6448e-01, 8.5906e-01, 8.5355e-01, 8.4796e-01,
    8.4227e-01, 8.3651e-01, 8.3066e-01, 8.2472e-01, 8.1871e-01, 8.1262e-01,
    8.0645e-01, 8.0021e-01, 7.9389e-01, 7.8750e-01, 7.8104e-01, 7.7451e-01,
    7.6791e-01, 7.6125e-01, 7.5452e-01, 7.4773e-01, 7.4088e-01, 7.3396e-01,
    7.2700e-01, 7.1997e-01, 7.1289e-01, 7.0576e-01, 6.9857e-01, 6.9134e-01,
    6.8406e-01, 6.7674e-01, 6.6937e-01, 6.6196e-01, 6.5451e-01, 6.4702e-01,
    6.3950e-01, 6.3194e-01, 6.2434e-01, 6.1672e-01, 6.0907e-01, 6.0139e-01,
    5.9369e-01, 5.8596e-01, 5.7822e-01, 5.7045e-01, 5.6267e-01, 5.5487e-01,
    5.4705e-01, 5.3923e-01, 5.3140e-01, 5.2355e-01, 5.1571e-01, 5.0785e-01,
    5.0000e-01, 4.9215e-01, 4.8429e-01, 4.7645e-01, 4.6860e-01, 4.6077e-01,
    4.5295e-01, 4.4513e-01, 4.3733e-01, 4.2955e-01, 4.2178e-01, 4.1404e-01,
    4.0631e-01, 3.9861e-01, 3.9093e-01, 3.8328e-01, 3.7565e-01, 3.6806e-01,
    3.6050e-01, 3.5298e-01, 3.4549e-01, 3.3804e-01, 3.3063e-01, 3.2326e-01,
    3.1594e-01, 3.0866e-01, 3.0143e-01, 2.9424e-01, 2.8711e-01, 2.8003e-01,
    2.7300e-01, 2.6604e-01, 2.5912e-01, 2.5227e-01, 2.4548e-01, 2.3875e-01,
    2.3209e-01, 2.2549e-01, 2.1896e-01, 2.1250e-01, 2.0611e-01, 1.9979e-01,
    1.9355e-01, 1.8738e-01, 1.8129e-01, 1.7528e-01, 1.6934e-01, 1.6349e-01,
    1.5773e-01, 1.5204e-01, 1.4645e-01, 1.4094e-01, 1.3552e-01, 1.3018e-01,
    1.2494e-01, 1.1980e-01, 1.1474e-01, 1.0978e-01, 1.0492e-01, 1.0016e-01,
    9.5491e-02, 9.0925e-02, 8.6460e-02, 8.2096e-02, 7.7836e-02, 7.3680e-02,
    6.9629e-02, 6.5684e-02, 6.1847e-02, 5.8117e-02, 5.4497e-02, 5.0986e-02,
    4.7586e-02, 4.4298e-02, 4.1123e-02, 3.8060e-02, 3.5112e-02, 3.2278e-02,
    2.9560e-02, 2.6957e-02, 2.4472e-02, 2.2103e-02, 1.9853e-02, 1.7721e-02,
    1.5708e-02, 1.3815e-02, 1.2042e-02, 1.0389e-02, 8.8564e-03, 7.4453e-03,
    6.1558e-03, 4.9882e-03, 3.9426e-03, 3.0195e-03, 2.2190e-03, 1.5413e-03,
    9.8664e-04, 5.5507e-04, 2.4673e-04, 6.1691e-05
};

static_assert(sizeof(HANN_WINDOW) / sizeof(float) == 400);

// Computes mel-spectrogram of the raw audio signal.
class AudioPreprocessor {
public:
	AudioPreprocessor() {
		const int max_out_frames = kMaxChunkFrames / hop_length_;

		stft_out_     = jarvis_malloc_f32(max_out_frames * n_freqs_);
		cos_cache_    = jarvis_malloc_f32(n_fft_ * n_freqs_);
		sin_cache_    = jarvis_malloc_f32(n_fft_ * n_freqs_);
		padded_sig_   = jarvis_malloc_f32(kMaxChunkFrames + n_fft_);
		mel_filters_  = jarvis_malloc_f32(n_mels_ * n_freqs_);
		mel_spec_f32_ = jarvis_malloc_f32(max_out_frames * n_freqs_);
		mel_spec_f16_ = jarvis_malloc_f16(max_out_frames * n_freqs_);
		// Load mel filters.
		const char* mel_filters_path = "assets/mel_filters.bin";
		std::ifstream fin_mf{mel_filters_path};
		if (!fin_mf.is_open()) {
			fprintf(stderr, "Failed to open: %s\n", mel_filters_path);
			std::exit(-1);
		}
		
		fin_mf.read(reinterpret_cast<char*>(mel_filters_), n_mels_*n_freqs_*sizeof(float));

		// Init the caches.
		for (int f = 0; f < 201; ++f) {
			for (int t = 0; t < n_fft_; ++t) {
				cos_cache_[f * n_fft_ + t] = std::cos( (2.0f * 3.141592653589793f * f * t) / (float)n_fft_);
				sin_cache_[f * n_fft_ + t] = std::sin( (2.0f * 3.141592653589793f * f * t) / (float)n_fft_);
			}
		}
	}

	~AudioPreprocessor() {
		jarvis_free(stft_out_);
		jarvis_free(cos_cache_);
		jarvis_free(sin_cache_);
		jarvis_free(padded_sig_);
		jarvis_free(mel_filters_);
		jarvis_free(mel_spec_f32_);
		jarvis_free(mel_spec_f16_);
	}

	// signal: Audio recording signal where samplerate=16000. If the signal recording
	// time is less than `nsecs`, it is padded with zeros. If it is greater than  `nsecs`,
	// we only use the last  `nsecs` samples. The new input has nsamples=(16000*`nsecs`).
	// We then compute a short-time fourier transform (stft) of the new input to obtain
	// a spectrogram output of shape (n_samples/hop_length, n_freqs).
	// We then perform a matmul between the spectrogram and a mel-filterbank to obtain
	// a mel-spectrogram of shape (out_frames, n_mels).
	// Returns a spectrogram of shape (out_frames, n_mels).
	Float16* get_mel_spectrogram(std::vector<float>& sig, int* n_out_frames) {
		// PAD SIGNAL
		const int n_samples = sig.size();
		float* signal_ptr = sig.data();
		if (n_samples > kMaxChunkFrames) {
			// If the number of samples exceed 30 secs, offset the ptr to the start of last n_secs-secs block.
			signal_ptr = sig.data() + n_samples - kMaxChunkFrames;
		}

		return get_mel_spectrogram(sig.data(), n_samples, n_out_frames);
	}

	Float16* get_mel_spectrogram(const float* sig, int n_samples, int* n_out_frames) {
		// PAD SIGNAL
		JARVIS_ASSERT(n_samples <= kMaxChunkFrames);

		const float* spectrogram_f32 = compute_mel_spectrogram(sig, n_samples);

		const int out_frames = n_samples / hop_length_;
		*n_out_frames = out_frames;
		const int out_channels = n_mels_;
		for (int i = 0; i < out_frames; i++) {
			for (int j = 0; j < out_channels; j++) {
				mel_spec_f16_[i * out_channels + j] = fp32_to_fp16(spectrogram_f32[i * out_channels + j]);
			}
		}

		return mel_spec_f16_;
	}


private:
	// Return (n_frames, n_channels)
	float* compute_mel_spectrogram(const float* sig, int n_samples) {
		JARVIS_ASSERT(n_samples <= kMaxChunkFrames);

		const int n_fft = n_fft_;
		const int hop_length = hop_length_;

		// Compute magnitudes. shape=[out_frames, n_freqs]
		float* magnitudes = stft(sig, n_samples, n_fft, hop_length, HANN_WINDOW);

		// mel_spec = filters @ magnitudes. shape=[out_frames, n_mels]
		// Stores the result of matrix multiplication between the mel_filters and the
		// magnitudes. magnitudes is in transposed order so that we have a good cache
		// locality with respect to both matrices when performing the multiplication.
		const int out_channels = n_mels_;
		const int nfreqs = n_freqs_;
		const int n_out_frames = n_samples / hop_length;

		for (int i = 0; i < out_channels; ++i) {
			for (int j = 0; j < n_out_frames; ++j) {
				float dotprod = 0.0f;
				for (int k = 0; k < nfreqs; ++k) {
					dotprod += mel_filters_[i * nfreqs + k] * magnitudes[j * nfreqs + k];
				}
				mel_spec_f32_[i + j * out_channels] = dotprod;
			}
		}

		const int n_mel_spec = n_mels_*n_out_frames;
		// log_spec = torch.clamp(mel_spec, min=1e-10).log10()
		for (int i = 0; i < n_mel_spec; ++i) {
			if (mel_spec_f32_[i] < 1e-10) {
				mel_spec_f32_[i] = 1e-10;
			}
			mel_spec_f32_[i] = std::log10(mel_spec_f32_[i]);
		}

		// torch.maximum(log_spec, log_spec.max() - 8.0)
		float mel_spec_max = -std::numeric_limits<float>::infinity();
		for (int i = 0; i < n_mel_spec; ++i) {
			if (mel_spec_f32_[i] > mel_spec_max) {
				mel_spec_max = mel_spec_f32_[i];
			}
		}
		for (int i = 0; i < n_mel_spec; ++i) {
			if (mel_spec_f32_[i] < mel_spec_max - 8.0) {
				mel_spec_f32_[i] = mel_spec_max - 8.0;
			}
		}

		// (log_spec + 4.0) / 4.0
		for (int i = 0; i < n_mel_spec; ++i) {
			mel_spec_f32_[i] = (mel_spec_f32_[i] + 4.0f) * 1.0f/4.0f;
		}

		return mel_spec_f32_;
	}

	float* stft(const float* sig, int nsamples, int nfft, int hop_length, const float* window)
	{
		const int nframes = nsamples / hop_length;
		const int nfreqs = nfft / 2 + 1;

		// PAD_ARRAY
		const int pad_size = nfft / 2;
		float* padded_sig = pad_sig(sig, nsamples, pad_size);

		// COMPUTE_STFT
		for (int i = 0; i < nframes; ++i) {
			float* frame_sig = padded_sig + i * hop_length;
			float* out_sig = stft_out_ + nfreqs * i;
			fourier_transform(frame_sig, nfft, window, out_sig);
		}

		return stft_out_;
	}

	// Pad array by pad_size both on the right and left by reflecting. E.g for pad_size=3
	// input:           [1, 2, 3, 4, 5, 6]
    // output: [4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3]
	float* pad_sig(const float* sig, int nsamples, int pad_size)
	{
		memcpy(padded_sig_+pad_size, sig, nsamples * 4);

		// Pad left
		const int pad_cnt = pad_size + 1;
		for (int i = 1; i < pad_cnt; ++i) {
			padded_sig_[pad_size - i] = sig[i];
		}

		// Pad right.
		const int offset = nsamples - pad_size - 1;
		const int padded_nsamples = nsamples + pad_size + pad_size;
		for (int i = 0; i < pad_size; ++i) {
			padded_sig_[padded_nsamples - 1 - i] = sig[i + offset];
		}

		return padded_sig_;
	}

	void fourier_transform(const float* sig, int nsamples, const float* window, float* out)
	{
		// Number of frequencies for which magnitudes are calculated.
		int n_freqs = nsamples / 2 + 1;

		for (int f = 0; f < n_freqs; ++f) {
			float sum_real = 0.0f;
			float sum_imag = 0.0f;
			for (int t = 0; t < nsamples; ++t) {
				sum_real += window[t] * sig[t] * cos_cache_[f * n_fft_ + t];
				sum_imag += window[t] * sig[t] * -1.0 * sin_cache_[f * n_fft_ + t];
			}
			out[f] = sum_real * sum_real + sum_imag * sum_imag;
		}
	}

private:
	const int n_fft_ = 400;
	const int hop_length_ = 160;
	const int n_mels_ = 80;
	const int n_freqs_ = n_fft_ / 2 + 1;
	float* stft_out_;
	float* cos_cache_;
	float* sin_cache_;
	float* padded_sig_;
	float* mel_filters_;
	float* mel_spec_f32_;
	Float16* mel_spec_f16_;
};
