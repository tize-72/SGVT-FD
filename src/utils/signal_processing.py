"""
Signal processing utilities for vibration signal analysis.
STFT, spectrogram generation, fault characteristic frequency computation.
"""
import numpy as np
from scipy import signal
import torch


def compute_stft(sig, fs=12000, window='hann', nperseg=256, noverlap=128, nfft=512):
    """Compute Short-Time Fourier Transform of a vibration signal.

    Args:
        sig: 1D numpy array, vibration signal
        fs: Sampling frequency (Hz)
        window: Window function name
        nperseg: Window length
        noverlap: Overlap length
        nfft: FFT length

    Returns:
        f: Frequency array
        t: Time array
        Zxx: Complex STFT result (freq x time)
    """
    f, t, Zxx = signal.stft(sig, fs=fs, window=window, nperseg=nperseg,
                             noverlap=noverlap, nfft=nfft)
    return f, t, Zxx


def generate_spectrogram(sig, fs=12000, target_size=(224, 224)):
    """Generate a spectrogram image from a vibration signal.

    Args:
        sig: 1D numpy array, vibration signal
        fs: Sampling frequency
        target_size: Output image size (H, W)

    Returns:
        spec_db: Spectrogram in dB scale, shape (H, W)
    """
    f, t, Zxx = compute_stft(sig, fs=fs)
    magnitude = np.abs(Zxx)
    # Convert to dB scale
    spec_db = 20 * np.log10(magnitude + 1e-8)
    # Normalize to [0, 1]
    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
    # Resize to target size
    from PIL import Image
    spec_img = Image.fromarray((spec_db * 255).astype(np.uint8))
    spec_img = spec_img.resize(target_size, Image.BILINEAR)
    spec_array = np.array(spec_img).astype(np.float32) / 255.0
    return spec_array


def compute_fault_frequencies(rpm, nb=9, bd=7.94, pd=39.04, ca=0):
    """Compute bearing fault characteristic frequencies.

    Args:
        rpm: Rotational speed (RPM)
        nb: Number of balls
        bd: Ball diameter (mm)
        pd: Pitch diameter (mm)
        ca: Contact angle (degrees)

    Returns:
        Dictionary with BPFO, BPFI, FTF, BSF frequencies
    """
    fr = rpm / 60.0  # Rotational frequency
    bd_pd = bd / pd
    cos_ca = np.cos(np.radians(ca))

    bpfo = (nb / 2) * fr * (1 - bd_pd * cos_ca)
    bpfi = (nb / 2) * fr * (1 + bd_pd * cos_ca)
    ftf = (fr / 2) * (1 - bd_pd * cos_ca)
    bsf = (pd / (2 * bd)) * fr * (1 - (bd_pd * cos_ca) ** 2)

    return {"BPFO": bpfo, "BPFI": bpfi, "FTF": ftf, "BSF": bsf}


def segment_signal(sig, segment_length=8192, overlap=0.5):
    """Segment a long vibration signal into fixed-length segments.

    Args:
        sig: 1D numpy array
        segment_length: Length of each segment
        overlap: Overlap ratio (0 to 1)

    Returns:
        List of 1D numpy arrays
    """
    step = int(segment_length * (1 - overlap))
    segments = []
    for start in range(0, len(sig) - segment_length + 1, step):
        segments.append(sig[start:start + segment_length])
    return segments


def add_noise(sig, snr_db):
    """Add Gaussian noise to a signal at a specified SNR.

    Args:
        sig: 1D numpy array
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy signal
    """
    sig_power = np.mean(sig ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(sig))
    return sig + noise


def normalize_signal(sig):
    """Zero-mean, unit-variance normalization."""
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)


def add_impulse_noise(sig, density=0.01, amplitude=3.0):
    """Add impulse (salt-and-pepper) noise.

    Args:
        sig: 1D numpy array
        density: Fraction of samples to corrupt
        amplitude: Noise amplitude relative to signal std

    Returns:
        Noisy signal
    """
    noisy = sig.copy()
    sig_std = np.std(sig)
    n_impulses = int(len(sig) * density)
    positions = np.random.choice(len(sig), n_impulses, replace=False)
    signs = np.random.choice([-1, 1], n_impulses)
    noisy[positions] += signs * amplitude * sig_std
    return noisy


def add_colored_noise(sig, color='pink', snr_db=20):
    """Add colored noise (pink/brown/blue) at specified SNR.

    Args:
        sig: 1D numpy array
        color: 'pink', 'brown', or 'blue'
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy signal
    """
    n = len(sig)
    white = np.random.randn(n)

    # Generate colored noise via shaping in frequency domain
    f = np.fft.rfftfreq(n, d=1.0)
    f[0] = f[1] * 0.5  # Avoid DC division by zero

    spectrum = np.fft.rfft(white)

    if color == 'pink':
        spectrum = spectrum / np.sqrt(f)  # 1/f noise
    elif color == 'brown':
        spectrum = spectrum / f  # 1/f^2 noise (brownian)
    elif color == 'blue':
        spectrum = spectrum * np.sqrt(f)  # f noise (blue)
    else:
        raise ValueError(f"Unknown color: {color}")

    colored = np.fft.irfft(spectrum, n=n)
    colored = colored / (np.std(colored) + 1e-8)

    sig_power = np.mean(sig ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * colored

    return sig + noise


def add_mechanical_noise(sig, rpm=1750, snr_db=20):
    """Simulate mechanical noise (rotational harmonics + vibration harmonics).

    Args:
        sig: 1D numpy array
        rpm: Rotational speed in RPM for harmonic simulation
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy signal
    """
    n = len(sig)
    t = np.arange(n)
    fr = rpm / 60.0  # Rotational frequency

    # Generate mechanical noise: harmonics of rotation + random vibration
    noise = np.zeros(n)
    # Rotational harmonics (1x, 2x, 3x, 5x)
    for mult in [1, 2, 3, 5]:
        amplitude = 1.0 / mult
        noise += amplitude * np.sin(2 * np.pi * fr * mult * t / 12000)  # fs=12000

    # Add broadband vibration
    noise += 0.3 * np.random.randn(n)

    # Normalize
    noise = noise / (np.std(noise) + 1e-8)

    sig_power = np.mean(sig ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * noise

    return sig + noise


def add_multiple_noise_types(sig, noise_type, intensity):
    """Apply different noise types at specified intensity.

    Args:
        sig: Clean 1D signal
        noise_type: 'gaussian', 'impulse', 'pink', 'brown', 'blue', 'mechanical'
        intensity: SNR in dB for continuous noise, density for impulse

    Returns:
        Noisy signal
    """
    if noise_type == 'gaussian':
        return add_noise(sig, snr_db=intensity)
    elif noise_type == 'impulse':
        densities = {30: 0.001, 20: 0.005, 10: 0.01, 5: 0.02, 0: 0.05}
        amp = {30: 1.0, 20: 2.0, 10: 3.0, 5: 5.0, 0: 8.0}
        d = densities.get(intensity, intensity / 1000)
        a = amp.get(intensity, 3.0)
        return add_impulse_noise(sig, density=d, amplitude=a)
    elif noise_type in ('pink', 'brown', 'blue'):
        return add_colored_noise(sig, color=noise_type, snr_db=intensity)
    elif noise_type == 'mechanical':
        return add_mechanical_noise(sig, snr_db=intensity)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
