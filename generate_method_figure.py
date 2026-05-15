"""
Generate method overview figure for SGVT-FD paper.
6-panel figure showing the pipeline.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
from scipy import signal as sig
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def generate_method_overview(save_path='figs/method_overview.pdf'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel (a): Raw vibration signal
    ax = axes[0, 0]
    t = np.linspace(0, 0.5, 1000)
    # Simulate bearing fault signal
    np.random.seed(42)
    sig_raw = np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 107 * t) + 0.3 * np.random.randn(len(t))
    ax.plot(t, sig_raw, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('(a) Raw Vibration Signal', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel (b): STFT Spectrogram
    ax = axes[0, 1]
    f, t_stft, Zxx = sig.stft(sig_raw, fs=2000, nperseg=128, noverlap=64)
    spec = np.abs(Zxx)
    spec_db = 20 * np.log10(spec + 1e-8)
    ax.pcolormesh(t_stft, f, spec_db, shading='gouraud', cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('(b) STFT Spectrogram', fontweight='bold')

    # Panel (c): Patch Tokenization
    ax = axes[0, 2]
    # Show grid of patches
    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
    spec_resized = np.array(Image.fromarray((spec_norm * 255).astype(np.uint8)).resize((224, 224))) / 255.0 if False else spec_norm
    # Simple visualization of patches
    patch_grid = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            patch_grid[i, j] = spec_norm[min(i*2, spec_norm.shape[0]-1), min(j*2, spec_norm.shape[1]-1)]
    im = ax.imshow(patch_grid, cmap='viridis', aspect='equal')
    ax.set_xlabel('Patch Column')
    ax.set_ylabel('Patch Row')
    ax.set_title('(c) Patch Tokens (N=196)', fontweight='bold')
    # Add patch grid lines
    for i in range(15):
        ax.axhline(i-0.5, color='white', linewidth=0.5, alpha=0.5)
        ax.axvline(i-0.5, color='white', linewidth=0.5, alpha=0.5)

    # Panel (d): Semantic Grouping
    ax = axes[1, 0]
    # Show grouped patches with colors
    np.random.seed(123)
    group_map = np.random.randint(0, 8, size=(14, 14))
    # Make it more structured (frequency-based grouping)
    for i in range(14):
        group_map[i, :] = i // 2  # Group by frequency bands
    colors = plt.cm.Set3(np.linspace(0, 1, 8))
    grouped_img = np.zeros((14, 14, 3))
    for i in range(14):
        for j in range(14):
            grouped_img[i, j] = colors[group_map[i, j]][:3]
    ax.imshow(grouped_img, aspect='equal')
    ax.set_xlabel('Patch Column')
    ax.set_ylabel('Patch Row')
    ax.set_title('(d) Semantic Grouping (K=8)', fontweight='bold')
    # Add legend for groups
    patches = [mpatches.Patch(color=colors[k][:3], label=f'Group {k+1}') for k in range(4)]
    ax.legend(handles=patches, loc='upper right', fontsize=10, ncol=2)

    # Panel (e): Token Merging
    ax = axes[1, 1]
    # Show merged tokens as a smaller grid
    merged = np.zeros((4, 8))
    for k in range(8):
        row, col = k // 4, k % 4
        merged[row, col] = np.mean(patch_grid[group_map == k])
    im = ax.imshow(merged, cmap='viridis', aspect='equal')
    ax.set_xlabel('Merged Token Column')
    ax.set_ylabel('Merged Token Row')
    ax.set_title('(e) Merged Tokens (K=32)', fontweight='bold')
    for i in range(5):
        ax.axhline(i-0.5, color='white', linewidth=1, alpha=0.7)
        ax.axvline(i-0.5, color='white', linewidth=1, alpha=0.7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel (f): Classification
    ax = axes[1, 2]
    classes = ['Normal', 'Ball', 'Inner\nRace', 'Outer\nRace']
    probs = [0.05, 0.03, 0.87, 0.05]
    colors_bar = plt.cm.Set2(np.linspace(0, 1, 4))
    bars = ax.barh(classes, probs, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Probability')
    ax.set_title('(f) Fault Classification', fontweight='bold')
    ax.set_xlim(0, 1)
    # Highlight the predicted class
    bars[2].set_edgecolor('red')
    bars[2].set_linewidth(2.5)
    ax.text(0.89, 2, 'Predicted', fontsize=12, color='red', fontweight='bold',
            ha='left', va='center')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Method overview saved to {save_path}")


if __name__ == "__main__":
    generate_method_overview()
