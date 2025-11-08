import numpy as np
import soundfile as sf
from scipy.fftpack import fft, ifft
from scipy.signal import get_window
import matplotlib.pyplot as plt

def noise_reduction(input_file, output_file, frame_size=4096, overlap=0.5, noise_start=0, noise_end=100,
                    suppression_factor=2, protection_factor=0.01):
    audio, sample_rate = sf.read(input_file)
    original_signal = audio.copy()

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio))

    hop_size = int(frame_size * (1 - overlap))
    window = get_window('hann', frame_size)

    noise_frames = []
    for i in range(0, min(noise_end * sample_rate // 1000, len(audio) - frame_size), hop_size):
        if i >= noise_start * sample_rate // 1000:
            frame = audio[i:i + frame_size] * window
            noise_fft = fft(frame)
            noise_frames.append(np.abs(noise_fft))

    # print(frame)

    noise_profile = np.mean(noise_frames, axis=0) if noise_frames else np.zeros(frame_size)

    output_signal = np.zeros(len(audio))
    window_sum = np.zeros(len(audio))

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size] * window
        frame_fft = fft(frame)

        magnitude = np.abs(frame_fft)
        phase = np.angle(frame_fft)

        clean_magnitude = np.maximum(magnitude - suppression_factor * noise_profile, protection_factor * magnitude)

        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_frame = np.real(ifft(clean_fft))

        output_signal[i:i + frame_size] += clean_frame
        window_sum[i:i + frame_size] += window

    window_sum[window_sum == 0] = 1
    output_signal /= window_sum

    output_signal = output_signal * 32767
    output_signal = np.clip(output_signal, -32768, 32767)
    output_signal = output_signal.astype(np.int16)

    # print(np.mean(np.abs(original_signal - output_signal)))
    # print("Original:", original_signal[:100])
    # print("Cleaned:", output_signal[:100])

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.specgram(audio, Fs=sample_rate, NFFT=frame_size, noverlap=hop_size)
    plt.colorbar()
    plt.title('Original Audio Spectrogram')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(2, 1, 2)
    plt.specgram(output_signal, Fs=sample_rate, NFFT=frame_size, noverlap=hop_size)
    plt.colorbar()
    plt.title('Cleaned Audio Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    sf.write(output_file, output_signal, sample_rate)

    plt.tight_layout()
    plt.savefig('spectrograms_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

noise_reduction('input2.wav', 'output_clean.wav')