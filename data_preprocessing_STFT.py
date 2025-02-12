import pandas as pd
import numpy as np
from scipy.signal import welch, stft, istft, butter, filtfilt
from scipy.stats import skew, kurtosis
import matplotlib

matplotlib.use("TkAgg")  # 解决 PyCharm 绘图问题
import matplotlib.pyplot as plt

# ==========================
# Load data
# ==========================

file_paths = {
    "default": "C:/Users/c1257/Desktop/data/default.csv",
    "steel": "C:/Users/c1257/Desktop/data/steel.csv",
    "roasted_steel": "C:/Users/c1257/Desktop/data/roasted_steel.csv",
    "aluminum": "C:/Users/c1257/Desktop/data/aluminum.csv",
    "brass": "C:/Users/c1257/Desktop/data/brass.csv",
}

# read all the data
dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Sampling Information
fs = 50000  # Sampling rate: 50 kHz
window_size = 10000  # 0.2 second window
step_size = 5000  # 50% overlap

# ============================================================
# Extract background noise from default.csv and preprocessing
# ============================================================

# get the background noise from default file
default_signal = dataframes["default"].values.flatten()
default_signal = (default_signal - np.mean(default_signal)) / np.std(default_signal)  # Standardization

# design bandpass filter
nyquist = 0.5 * fs
low, high = 500 / nyquist, 20000 / nyquist
b, a = butter(4, [low, high], btype='band')
default_signal = filtfilt(b, a, default_signal)  # Bandpass Filtering of background noise

# Compute STFT of background noise
f_noise, t_noise, Zxx_noise = stft(default_signal, fs=fs, nperseg=4096)
# 计算每个频率的噪声功率谱（对所有时刻取均值）
noise_psd = np.mean(np.abs(Zxx_noise) ** 2, axis=1)


# ================================================
# Function to compute Power Spectral Density (PSD)
# ================================================

def compute_psd(signal, fs=50000):
    frequencies, psd = welch(signal, fs=fs, nperseg=4096)
    psd = np.maximum(psd, 1e-10)  # 避免对数计算错误
    psd_log = 10 * np.log10(psd)  # 归一化
    return frequencies, psd_log


# Function to segment signal using sliding window
def segment_signal(signal, window_size=10000, step_size=5000):
    segments = []
    total_length = len(signal)
    for start in range(0, total_length - window_size, step_size):
        segment = signal[start:start + window_size]
        segments.append(segment)
    return np.array(segments)


# Function to remove anomalies based on RMS threshold
def filter_anomalies(signal_segments, threshold_low=0.1, threshold_high=5.0):
    rms_values = [np.sqrt(np.mean(segment ** 2)) for segment in signal_segments]
    filtered_segments = [seg for seg, rms in zip(signal_segments, rms_values) if threshold_low < rms < threshold_high]
    return np.array(filtered_segments)


# Function to extract features from signal segment
def extract_features(signal_segment, fs=50000):
    mean_val = np.mean(signal_segment)
    std_val = np.std(signal_segment)
    max_val = np.max(signal_segment)
    min_val = np.min(signal_segment)
    rms_val = np.sqrt(np.mean(signal_segment ** 2))
    skewness = skew(signal_segment)
    kurt = kurtosis(signal_segment)

    freqs, psd = welch(signal_segment, fs=fs, nperseg=4096)
    peak_freq = freqs[np.argmax(psd)]
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    return [mean_val, std_val, max_val, min_val, rms_val, skewness, kurt, peak_freq, spectral_entropy]


# ================================================
# Process and store segmented signals and features
# ================================================

denoised_signals = {}
stats_summary = {}
feature_list = []

for material, df in dataframes.items():
    if material == "default":
        continue

    signal = df.values.flatten()

    # Preprocessing
    signal = (signal - np.mean(signal)) / np.std(signal)  # Standardization
    signal = filtfilt(b, a, signal)  # Bandpass Filtering

    # STFT 与 Wiener 滤波降噪
    f_signal, t_signal, Zxx_signal = stft(signal, fs=fs, nperseg=4096)
    # 计算信号的时频功率谱（幅值平方）
    signal_psd = np.abs(Zxx_signal) ** 2

    # 估计 Wiener 滤波器系数：
    # 对每个频率和时间点，SNR = signal_psd / noise_psd，其中 noise_psd 是按频率估计的噪声功率
    # 为使维度匹配，将 noise_psd 扩展为二维数组（每列相同）
    noise_psd_2d = noise_psd[:, np.newaxis]
    snr = signal_psd / (noise_psd_2d + 1e-10)
    H_wiener = snr / (snr + 1)  # Wiener 滤波器传递函数

    # 应用 Wiener 滤波器
    Zxx_denoised = H_wiener * Zxx_signal

    # 通过 ISTFT 得到降噪后的时域信号
    _, denoised_signal = istft(Zxx_denoised, fs=fs)

    # =====================================
    # visualization
    # =====================================

    # Compare time-domain signal before and after denoising
    plt.figure(figsize=(12, 6))
    plt.plot(signal[:5000], label="Original Signal", alpha=0.7)
    plt.plot(denoised_signal[:5000], label="Denoised Signal", alpha=0.7)
    plt.title(f"Time Domain Signal Comparison for {material}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(
        f"C:/Users/c1257/Desktop/data_diagram/processed_data_comparison/3rd_trail_Wiener_Filter_4096/{material}_time_comparison_wiener.png")
    plt.show()

    # Compute and visualize STFT
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t_signal, f_signal, np.abs(Zxx_signal), shading='gouraud')
    plt.title(f"STFT of Original {material} Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.ylim(0, 20000)
    plt.savefig(
        f"C:/Users/c1257/Desktop/data_diagram/processed_data_comparison/3rd_trail_Wiener_Filter_4096/{material}_original_stft.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t_signal, f_signal, np.abs(Zxx_denoised), shading='gouraud')
    plt.title(f"STFT of Denoised {material} Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.ylim(0, 20000)
    plt.savefig(
        f"C:/Users/c1257/Desktop/data_diagram/processed_data_comparison/3rd_trail_Wiener_Filter_4096/{material}_denoised_stft_wiener.png")
    plt.show()

    # Compute and visualize PSD before and after denoising
    frequencies_signal, psd_signal = compute_psd(signal, fs)
    frequencies_denoised, psd_denoised = compute_psd(denoised_signal, fs)

    plt.figure(figsize=(10, 5))
    plt.plot(frequencies_signal, psd_signal, label="Original Signal (PSD)")
    plt.plot(frequencies_denoised, psd_denoised, linestyle="dashed", label="Denoised Signal (PSD)")
    plt.title(f"Power Spectral Density Before and After Denoising for {material}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.legend()
    plt.savefig(
        f"C:/Users/c1257/Desktop/data_diagram/processed_data_comparison/3rd_trail_Wiener_Filter_4096/{material}_psd_comparison_wiener.png")
    plt.show()

    # Segment signal
    segmented_signals = segment_signal(denoised_signal, window_size, step_size)
    segmented_signals = filter_anomalies(segmented_signals)  # Remove anomalies

    # Extract features
    features = [extract_features(seg, fs) for seg in segmented_signals]
    feature_list.extend([[material] + feat for feat in features])

    # Save segmented denoised signals
    df_denoised = pd.DataFrame(segmented_signals)
    df_denoised.to_csv(f"C:/Users/c1257/Desktop/processed_data/denoised_{material}.csv", index=False)

    # Compute statistics for visualization
    stats_summary[material] = {
        "Mean": np.mean(segmented_signals),
        "Std Dev": np.std(segmented_signals),
        "Max": np.max(segmented_signals),
        "Min": np.min(segmented_signals),
        "Skewness": skew(segmented_signals.flatten()),
        "Kurtosis": kurtosis(segmented_signals.flatten())
    }

# Save extracted features

df_features = pd.DataFrame(feature_list,
                           columns=["Material", "Mean", "Std Dev", "Max", "Min", "RMS", "Skewness", "Kurtosis",
                                    "Peak Freq", "Spectral Entropy"])
df_features.to_csv("C:/Users/c1257/Desktop/processed_data/feature_data.csv", index=False)

# Convert statistics summary to DataFrame and visualize
stats_df = pd.DataFrame(stats_summary).T
print(stats_df)
stats_df.to_csv("C:/Users/c1257/Desktop/processed_data/signal_statistics.csv")

# Plot statistical summary
stats_df.plot(kind='bar', figsize=(12, 6), title="Statistical Features of Denoised Signals")
plt.ylabel("Value")
plt.savefig(
    "C:/Users/c1257/Desktop/data_diagram/processed_data_comparison/3rd_trail_Wiener_Filter_4096/statistical_features_wiener.png")
plt.show()
