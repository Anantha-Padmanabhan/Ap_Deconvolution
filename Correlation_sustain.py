import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

def explore_hdf5(file_path):
    """
    Recursively explore the structure of an HDF5 file (.mat v7.3 format).
    
    Parameters:
        file_path (str): Path to the HDF5 (.mat) file.
    """
    def print_hdf5_structure(name, obj):
        print(f"{name}: {obj}")

    with h5py.File(file_path, 'r') as mat_file:
        print(f"Structure of {file_path}:")
        mat_file.visititems(print_hdf5_structure)

def get_hdf5_variable(file_path, variable_path):
    """
    Access a variable in an HDF5 file, even if nested.

    Parameters:
        file_path (str): Path to the HDF5 file.
        variable_path (str): Full path to the variable within the file (e.g., "group/subgroup/variable").
        
    Returns:
        np.array: The signal data.
    """
    with h5py.File(file_path, 'r') as mat_file:
        obj = mat_file
        for part in variable_path.split('/'):
            obj = obj[part]
        return np.array(obj).squeeze() 

pid_signal_file = r"C:\\Work\\New folder\\PID_aTP_10-2000ms_3.mat"
sensor_signal_file = r"C:\\Work\\New folder\\MICS_10Kresistance_aTP_10-2000ms_3.mat"

print("Exploring PID Signal File:")
explore_hdf5(pid_signal_file)

print("\nExploring Sensor Signal File:")
explore_hdf5(sensor_signal_file)

# Variable paths for PID, sensor, and their respective trigger channels
pid_variable_path = 'PID_aTP_10_2000ms_3_Ch4/values'  
sensor_variable_path = 'MICS_10Kresistance_aTP_10_2000ms_3_Ch5/values'  
pid_trigger_variable_path = 'PID_aTP_10_2000ms_3_Ch8/values'
sensor_trigger_variable_path = 'MICS_10Kresistance_aTP_10_2000ms_3_Ch8/values'

# Load signals
pid_signal = get_hdf5_variable(pid_signal_file, pid_variable_path)
sensor_signal = get_hdf5_variable(sensor_signal_file, sensor_variable_path)
pid_trigger_signal = get_hdf5_variable(pid_signal_file, pid_trigger_variable_path)
sensor_trigger_signal = get_hdf5_variable(sensor_signal_file, sensor_trigger_variable_path)

# Sampling rate
sampling_rate = 10000 

# Identify trigger spikes for PID and MICS (threshold-based detection)
pid_trigger_threshold = 0.8 * np.max(pid_trigger_signal)
sensor_trigger_threshold = 0.8 * np.max(sensor_trigger_signal)

pid_trigger_spikes = np.where(pid_trigger_signal > pid_trigger_threshold)[0]
sensor_trigger_spikes = np.where(sensor_trigger_signal > sensor_trigger_threshold)[0]

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
tau1, tau2, tau_sustain = 0.5, 0.002, 0.2

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
deconvolved_signal_smoothed = -gaussian_filter1d(deconvolved_signal_normalized, sigma=5)  # Adjust sigma for smoothing strength

# Center and calibrate signals
pid_signal_centered = pid_signal_normalized - np.mean(pid_signal_normalized)
deconvolved_signal_centered = deconvolved_signal_smoothed - np.mean(deconvolved_signal_smoothed)
scaling_factor = np.max(np.abs(pid_signal_centered)) / np.max(np.abs(deconvolved_signal_centered))
deconvolved_signal_calibrated = deconvolved_signal_centered * scaling_factor

# Parameters
trigger_threshold = 0.8
trigger_duration = 3 # seconds
offset = 0.01  # offset after trigger
Tmax = 0.7

# File paths
output_folder = r"C:\Work\Output"

# Deduplicate trigger indices for PID and sensor
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

# Correlations between each pair of PID and MICS repeats
correlations_pid_mics = {
    (pid_key, mics_key): pearsonr(pid_normalized_repeats[pid_key][:len(deconvolved_normalized_repeats[mics_key])],
                                  deconvolved_normalized_repeats[mics_key][:len(pid_normalized_repeats[pid_key])])[0]
    for pid_key in pid_normalized_repeats.keys() 
    for mics_key in deconvolved_normalized_repeats.keys()
}

# Calculate correlations between the first repeat of each signal
reference_pid_signal = list(pid_repeats.values())[0]  # Use the first PID repeat as reference
reference_pid_normalized = (reference_pid_signal - np.min(reference_pid_signal)) / (np.max(reference_pid_signal) - np.min(reference_pid_signal))

correlations_with_first = {
    key: pearsonr(data[:len(reference_pid_normalized)], reference_pid_normalized[:len(data)])[0]
    for key, data in deconvolved_normalized_repeats.items()
}

# Time axis for plotting
time = np.arange(samples_to_include) / sampling_rate

# Plot 1: Overlay of PID and deconvolved signals
fig1, axs = plt.subplots(2, 1, figsize=(12, 10))
for key, data in pid_normalized_repeats.items():
    axs[0].plot(time, data, label=f"{key}", linewidth=1.5)
axs[0].set_title(f"Overlay of PID Signals", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Normalized Amplitude")
axs[0].legend(fontsize=10, loc="upper right")
axs[0].grid(color='lightgrey', linestyle='--', linewidth=0.5)

for key, data in deconvolved_normalized_repeats.items():
    axs[1].plot(time, data, label=f"{key} (Corr: {correlations_pid_mics.get((list(pid_normalized_repeats.keys())[0], key), 'N/A'):.2f})")
axs[1].set_title(f"Overlay of Deconvolved Signals (RE sensor) ,tau1 = {tau1}, tau2 = {tau2}", fontsize=14, fontweight='bold')
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Normalized Amplitude")
axs[1].legend(fontsize=10)
axs[1].grid(color='lightgrey', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"PID_vs_Deconvolved_TMax_{Tmax}_Overlay.png"), dpi=300)
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


print("\nCorrelations between PID and Deconvolved Signals:")
for (pid_key, mics_key), corr in correlations_pid_mics.items():
    print(f"{pid_key} vs {mics_key}: {corr:.4f}")

