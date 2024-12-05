#pragma once

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "common.h"


// Implements simple energy-based thresholding VAD. Fast but not robust.
class VoiceActivityDetector {
public:
    VoiceActivityDetector(int n_inp_frames, float speech_threshold = -90.0f) {
		speech_threshold_ = speech_threshold;
		n_inp_frames_ = n_inp_frames;
		n_out_frames_ =  ((n_inp_frames_ - n_fft_) / fft_hop_length_) + 1;
		fft_out_ = jarvis_malloc_f32(n_out_frames_ * n_freqs_);
		fft_energy_ = jarvis_malloc_f32(n_out_frames_);
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

	bool signal_has_speech(const float* sig) {
		float* vad_out = vad(sig);

		// float min = INFINITY;
		// float max = -INFINITY;
		float mean_energy = 0.0f;
		for (int i = 0; i < n_out_frames_; i++) {
			float val = vad_out[i];
			mean_energy += val;
			// if (val < min) { min = val; }
			// if (val > max) {max = val;}
		}
		mean_energy = mean_energy / n_out_frames_;
		// printf("(min=%f, max=%f, mean=%f)\n", min, max, mean_energy);

		// for (int i = 0; i < n_out_frames_; i++) {
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

	float* vad(const float* sig) {
		// Compute frequency magnitudes: abs(rfft(x))**2
		for (int i = 0; i < n_out_frames_; ++i) {
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
		for (int i = 0; i < n_out_frames_; i++){
			float sum = 0.0f;
			for (int j = 0; j < n_freqs_; j++){
				sum += fft_out_[i * n_freqs_ + j];
			}
			fft_energy_[i] = sum / (n_out_frames_*n_out_frames_);
		}
		
		// normalize energy to 0 dB then filter
		for (int i = 0; i < n_out_frames_; i++) {
			fft_energy_[i] = 10.0f * log10f(fft_energy_[i] / energy_threshold_);
		}

		const int n_filter = 5;
		const int pad_size = 4;
		float sorted[n_filter];
		for (int i = 0; i < n_out_frames_-pad_size; i++) {
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
	int n_inp_frames_;
	int n_out_frames_;
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
	const int m_samplerate = 16000;
	const int m_nfft = 400;
	const int m_hop_length = 160;
	const int m_nmels = 80;
	const int m_nfreqs = m_nfft / 2 + 1;
	int m_record_duration;
	int m_out_frames;
public:
	AudioPreprocessor(int record_duration_secs = 10) {
		m_record_duration = record_duration_secs;
		m_out_frames = (record_duration_secs * m_samplerate) / m_hop_length;

		m_stft_out     = jarvis_malloc_f32(m_out_frames * m_nfreqs);
		m_cos_cache    = jarvis_malloc_f32(m_nfft * m_nfreqs);
		m_sin_cache    = jarvis_malloc_f32(m_nfft * m_nfreqs);
		m_padded_sig   = jarvis_malloc_f32(m_samplerate * record_duration_secs + m_nfft);
		m_mel_filters  = jarvis_malloc_f32(m_nmels * m_nfreqs);
		m_mel_spec_f32 = jarvis_malloc_f32(m_out_frames * m_nfreqs);
		m_mel_spec_f16 = jarvis_malloc_f16(m_out_frames * m_nfreqs);
		//)* Load mel filters.
		const char* mel_filters_path = "assets/mel_filters.bin";
		std::ifstream fin_mf{mel_filters_path};
		if (!fin_mf.is_open()) {
			fprintf(stderr, "Failed to open: %s\n", mel_filters_path);
			std::exit(-1);
		}
		
		fin_mf.read(reinterpret_cast<char*>(m_mel_filters), m_nmels*m_nfreqs*sizeof(float));

		// Init the caches.
		for (int f = 0; f < 201; ++f) {
			for (int t = 0; t < m_nfft; ++t) {
				m_cos_cache[f*m_nfft + t] = std::cos( (2.0f * 3.141592653589793f * f * t) / (float)m_nfft );
				m_sin_cache[f*m_nfft + t] = std::sin( (2.0f * 3.141592653589793f * f * t) / (float)m_nfft );
			}
		}
	}

	~AudioPreprocessor() {
		jarvis_free(m_stft_out);
		jarvis_free(m_cos_cache);
		jarvis_free(m_sin_cache);
		jarvis_free(m_padded_sig);
		jarvis_free(m_mel_filters);
		jarvis_free(m_mel_spec_f32);
		jarvis_free(m_mel_spec_f16);
	}

	// signal: Audio recording signal where samplerate=16000. If the signal recording
	// time is less than `nsecs`, it is padded with zeros. If it is greater than  `nsecs`,
	// we only use the last  `nsecs` samples. The new input has nsamples=(16000*`nsecs`).
	// We then compute a short-time fourier transform (stft) of the new input to obtain
	// a spectrogram output of shape (n_samples/hop_length, n_freqs).
	// We then perform a matmul between the spectrogram and a mel-filterbank to obtain
	// a mel-spectrogram of shape (out_frames, n_mels).
	// Returns a spectrogram of shape (out_frames, n_mels).
	Float16* get_mel_spectrogram(std::vector<float>& signal) {
		// PAD SIGNAL
		const int n_samples = m_samplerate * m_record_duration; // number of samples in 30s where samplerate=16000
		if (signal.size() < n_samples) {
			const int rems = n_samples - signal.size();
			for (int i = 0; i < rems; i++) {
				signal.push_back(0);
			}
		}
		// If the number of samples exceed 30 secs, offset the ptr to the start of last n_secs-secs block.
		const float* signal_ptr = signal.data() + signal.size() - n_samples;

		return get_mel_spectrogram(signal.data(), n_samples);
	}

	Float16* get_mel_spectrogram(const float* signal, int inp_samples) {
		// PAD SIGNAL
		const int n_samples = m_samplerate * m_record_duration; // number of samples in 30s where samplerate=16000
		JARVIS_ASSERT(n_samples == inp_samples);

		const float* spectrogram_f32 = compute_mel_spectrogram(signal, n_samples);
		
		const int out_frames = m_out_frames;
		const int out_channels = m_nmels;
		for (int i = 0; i < out_frames; i++) {
			for (int j = 0; j < out_channels; j++) {
				m_mel_spec_f16[i * out_channels + j] = fp32_to_fp16(spectrogram_f32[i * out_channels + j]);
			}
		}

		return m_mel_spec_f16;
	}


private:
	// Return (n_frames, n_channels)
	float* compute_mel_spectrogram(const float* sig, int nsamples) {
		JARVIS_ASSERT(nsamples == m_samplerate*m_record_duration);

		const int nfft = m_nfft;
		const int hop_length = m_hop_length;

		// Compute magnitudes. shape=[out_frames, n_freqs]
		float* magnitudes = stft(sig, nsamples, nfft, hop_length, HANN_WINDOW);

		// mel_spec = filters @ magnitudes. shape=[out_frames, n_mels]
		// Stores the result of matrix multiplication between the mel_filters and the
		// magnitudes. magnitudes is in transposed order so that we have a good cache
		// locality with respect to both matrices when performing the multiplication.
		const int n_mel_spec = m_nmels*m_out_frames;

		const int out_channels = m_nmels;
		const int nfreqs = m_nfreqs;
		const int out_frames = m_out_frames;
		for (int i = 0; i < out_channels; ++i) {
			for (int j = 0; j < out_frames; ++j) {
				float dotprod = 0.0f;
				for (int k = 0; k < nfreqs; ++k) {
					dotprod += m_mel_filters[i * nfreqs + k] * magnitudes[j * nfreqs + k];
				}
				m_mel_spec_f32[i + j * out_channels] = dotprod;
			}
		}

		// log_spec = torch.clamp(mel_spec, min=1e-10).log10()
		for (int i = 0; i < n_mel_spec; ++i) {
			if (m_mel_spec_f32[i] < 1e-10) {
				m_mel_spec_f32[i] = 1e-10;
			}
			m_mel_spec_f32[i] = std::log10(m_mel_spec_f32[i]);
		}

		// torch.maximum(log_spec, log_spec.max() - 8.0)
		float mel_spec_max = -std::numeric_limits<float>::infinity();
		for (int i = 0; i < n_mel_spec; ++i) {
			if (m_mel_spec_f32[i] > mel_spec_max) {
				mel_spec_max = m_mel_spec_f32[i];
			}
		}
		for (int i = 0; i < n_mel_spec; ++i) {
			if (m_mel_spec_f32[i] < mel_spec_max - 8.0) {
				m_mel_spec_f32[i] = mel_spec_max - 8.0;
			}
		}

		// (log_spec + 4.0) / 4.0
		for (int i = 0; i < n_mel_spec; ++i) {
			m_mel_spec_f32[i] = (m_mel_spec_f32[i] + 4.0f) * 1.0f/4.0f;
		}

		return m_mel_spec_f32;
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
			float* out_sig = m_stft_out + nfreqs * i;
			fourier_transform(frame_sig, nfft, window, out_sig);
		}

		return m_stft_out;
	}

	// Pad array by pad_size both on the right and left by reflecting. E.g for pad_size=3
	// input:           [1, 2, 3, 4, 5, 6]
    // output: [4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3]
	float* pad_sig(const float* sig, int nsamples, int pad_size)
	{
		memcpy(m_padded_sig+pad_size, sig, nsamples * 4);

		// Pad left
		const int pad_cnt = pad_size + 1;
		for (int i = 1; i < pad_cnt; ++i) {
			m_padded_sig[pad_size - i] = sig[i];
		}

		// Pad right.
		const int offset = nsamples - pad_size - 1;
		const int padded_nsamples = nsamples + pad_size + pad_size;
		for (int i = 0; i < pad_size; ++i) {
			m_padded_sig[padded_nsamples - 1 - i] = sig[i + offset];
		}

		return m_padded_sig;
	}

	void fourier_transform(const float* sig, int nsamples, const float* window, float* out)
	{
		// Number of frequencies for which magnitudes are calculated.
		int n_freqs = nsamples / 2 + 1;

		for (int f = 0; f < n_freqs; ++f) {
			float sum_real = 0.0f;
			float sum_imag = 0.0f;
			for (int t = 0; t < nsamples; ++t) {
				sum_real += window[t] * sig[t] * m_cos_cache[f * m_nfft + t];
				sum_imag += window[t] * sig[t] * -1.0 * m_sin_cache[f * m_nfft + t];
			}
			out[f] = sum_real * sum_real + sum_imag * sum_imag;
		}
	}

private:
	float* m_stft_out;
	float* m_cos_cache;
	float* m_sin_cache;
	float* m_padded_sig;
	float* m_mel_filters;
	float* m_mel_spec_f32;
	Float16* m_mel_spec_f16;
};
