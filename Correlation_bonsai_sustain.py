import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

# Function to load Bonsai binary file
def load_bonsai_data(file_path, num_channels):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file)
    samples_per_channel = len(data) // num_channels
    reshaped_data = data.reshape((samples_per_channel, num_channels))
    return reshaped_data

# Low-pass filter function
def low_pass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# File paths
pid_signal_file = r"C:\\Work\\Bonsai Recordings\\Data\\PID_EB_10%_2"
sensor_signal_file = r"C:\\Work\\Bonsai Recordings\\Data\\MICS_27K_EB_10%"

# Load signals
num_channels = 4
pid_data = load_bonsai_data(pid_signal_file, num_channels)
sensor_data = load_bonsai_data(sensor_signal_file, num_channels)

# Select channels (adjust channel indices if necessary)
pid_signal = pid_data[:, 2]
sensor_signal = sensor_data[:, 3]
pid_trigger_signal = pid_data[:, 0]
sensor_trigger_signal = sensor_data[:, 0]

# Sampling rate
sampling_rate = 1000  # Hz

# Apply low-pass filter
cutoff_frequency = 50  # Hz
pid_signal = low_pass_filter(pid_signal, cutoff_frequency, sampling_rate)
sensor_signal = low_pass_filter(sensor_signal, cutoff_frequency, sampling_rate)

def improved_kernel(t, tau1, tau2, tau_sustain):
    """
    Improved kernel function with rise, sustain, and decay phases
    
    Parameters:
    - t: time array
    - tau1: decay time constant
    - tau2: rise time constant
    - tau_sustain: sustain time constant (controls how long the peak is maintained)
    
    Returns:
    Kernel signal with rise, sustain, and decay phases
    """
    # Rise phase: exponential rise
    rise = np.exp(-t / tau2)
    
    # Sustain phase: maintained peak with gradual decay
    sustain = np.exp(-t / tau_sustain)
    
    # Decay phase: exponential decay
    decay = np.exp(-t / tau1)
    
    # Combine phases with weighted contributions
    kernel_signal = rise * decay * sustain
    
    # Normalize the kernel
    kernel_signal /= np.max(np.abs(kernel_signal))
    
    return kernel_signal


# Define kernel parameters (experiment with these values)
tau1, tau2, tau_sustain = 0.5, 0.002, 1.2

# Create kernel time array
kernel_time = np.linspace(0, 1, len(sensor_signal))
kernel_signal = improved_kernel(kernel_time, tau1, tau2, tau_sustain)

# FFT of the sensor signal and kernel
fft_sensor = np.fft.fft(sensor_signal)
fft_kernel = np.fft.fft(kernel_signal, n=len(sensor_signal))

# Deconvolution
fft_deconvolved = fft_sensor / (fft_kernel + 1e-10)  # Avoid division by zero
deconvolved_signal = np.fft.ifft(fft_deconvolved).real

# Normalize signals
sensor_signal_normalized = sensor_signal / np.max(np.abs(sensor_signal))
deconvolved_signal_normalized = deconvolved_signal / np.max(np.abs(deconvolved_signal))
pid_signal_normalized = pid_signal / np.max(np.abs(pid_signal))

# Smooth deconvolved signal
deconvolved_signal_smoothed = -gaussian_filter1d(deconvolved_signal_normalized, sigma=5)

# Center and calibrate signals
pid_signal_centered = pid_signal_normalized - np.mean(pid_signal_normalized)
deconvolved_signal_centered = deconvolved_signal_smoothed - np.mean(deconvolved_signal_smoothed)
scaling_factor = np.max(np.abs(pid_signal_centered)) / np.max(np.abs(deconvolved_signal_centered))
deconvolved_signal_calibrated = deconvolved_signal_centered * scaling_factor

# Trigger parameters
trigger_threshold = 0.8
trigger_duration = 3  # seconds
offset = 0.01  # offset after trigger

# Deduplicate trigger indices for PID
pid_trigger_indices = np.where(pid_trigger_signal > trigger_threshold)[0]
pid_deduplicated_indices = [pid_trigger_indices[0]] + [
    idx for i, idx in enumerate(pid_trigger_indices[1:]) if idx - pid_trigger_indices[i] > int(0.1 * sampling_rate)
]

sensor_trigger_indices = np.where(sensor_trigger_signal > trigger_threshold)[0]
sensor_deduplicated_indices = [sensor_trigger_indices[0]] + [
    idx for i, idx in enumerate(sensor_trigger_indices[1:]) if idx - sensor_trigger_indices[i] > int(0.1 * sampling_rate)
]


samples_to_include = int(trigger_duration * sampling_rate)
offset_samples = int(offset * sampling_rate)

# Extract repeats for PID signals
pid_repeats = {
    f"repeat_{i + 1}": pid_signal_centered[idx + offset_samples:idx + offset_samples + samples_to_include]
    for i, idx in enumerate(pid_deduplicated_indices)
    if idx + offset_samples + samples_to_include <= len(pid_signal)
}

# Extract repeats for deconvolved signals
deconvolved_repeats = {
    f"repeat_{i + 1}": deconvolved_signal_calibrated[idx + offset_samples:idx + offset_samples + samples_to_include]
    for i, idx in enumerate(sensor_deduplicated_indices)
    if idx + offset_samples + samples_to_include <= len(deconvolved_signal_calibrated)
}

# Normalize repeats
pid_normalized_repeats = {
    key: (data - np.min(data)) / (np.max(data) - np.min(data)) for key, data in pid_repeats.items()
}
deconvolved_normalized_repeats = {
    key: (data - np.min(data)) / (np.max(data) - np.min(data)) for key, data in deconvolved_repeats.items()
}

# Correlations between each pair of PID and deconvolved repeats
correlations_pid_mics = {
    (pid_key, mics_key): pearsonr(pid_normalized_repeats[pid_key][:len(deconvolved_normalized_repeats[mics_key])],
                                  deconvolved_normalized_repeats[mics_key][:len(pid_normalized_repeats[pid_key])])[0]
    for pid_key in pid_normalized_repeats.keys()
    for mics_key in deconvolved_normalized_repeats.keys()
}

# Plotting
time = np.arange(samples_to_include) / sampling_rate
# Plot 1: Overlay of PID and deconvolved signals (All repeats together)
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
for key, data in pid_normalized_repeats.items():
    axs[0].plot(time, data, label=f"{key}", linewidth=1.5)
axs[0].set_title("Overlay of PID Signals")
axs[0].set_ylabel("Normalized Amplitude")
axs[0].legend()
axs[0].grid()

for key, data in deconvolved_normalized_repeats.items():
    axs[1].plot(time, data, label=f"{key}")
axs[1].set_title(f"Overlay of Deconvolved Signals(27K) (tau1={tau1}, tau2={tau2})")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Normalized Amplitude")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

# Plot 2: Overlay of individual repeats (PID and deconvolved signals)
fig, axs = plt.subplots(7, 1, figsize=(12, 15))  # One subplot for each repeat
for i, (pid_key, mics_key) in enumerate(zip(pid_normalized_repeats.keys(), deconvolved_normalized_repeats.keys())):
    if i >= len(axs):  # Handle cases with fewer than 7 repeats
        break
    pid_repeat = pid_normalized_repeats[pid_key]
    deconvolved_repeat = deconvolved_normalized_repeats[mics_key]
    correlation = correlations_pid_mics[(pid_key, mics_key)]
    
    axs[i].plot(time, pid_repeat, label="PID Signal", color="blue")
    axs[i].plot(time, deconvolved_repeat, label="Deconvolved Signal(10k)", color="orange")
    axs[i].set_title(f"Repeat {i + 1}: Correlation = {correlation:.4f}")
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel("Normalized Amplitude")
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()
