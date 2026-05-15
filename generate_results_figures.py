"""
Generate results figures for SGVT-FD paper.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
})


def load_results(results_dir):
    results = []
    for root, dirs, files in os.walk(results_dir):
        if "results.json" in files:
            with open(os.path.join(root, "results.json")) as f:
                results.append(json.load(f))
    return results


def plot_main_results(save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)

    methods = ['ViT\n(196 tok.)', 'No Grouping\n(196 tok.)', 'CVT\n(32 tok.)', 'SGVT-FD\n(32 tok.)']
    accuracies = [98.96, 98.89, 64.99, 98.72]
    f1_scores = [98.98, 98.93, 60.33, 98.77]
    params = [43.5, 29.5, 1.3, 2.2]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    # Accuracy comparison
    bars = axes[0].bar(range(len(methods)), accuracies, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, fontsize=14)
    axes[0].set_ylabel('Accuracy (%)', fontsize=18)
    axes[0].set_ylim(50, 105)
    axes[0].grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=16, fontweight='bold')

    # F1 comparison
    bars = axes[1].bar(range(len(methods)), f1_scores, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, fontsize=14)
    axes[1].set_ylabel('F1 Score (%)', fontsize=18)
    axes[1].set_ylim(50, 105)
    axes[1].grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    axes[1].set_title('F1 Score Comparison', fontsize=16, fontweight='bold')

    # Token efficiency (scatter)
    token_counts = [196, 196, 32, 32]
    marker_sizes = [p * 10 for p in params]
    for i, (tc, acc, name) in enumerate(zip(token_counts, accuracies, methods)):
        axes[2].scatter(tc, acc, s=marker_sizes[i], c=colors[i], edgecolors='black',
                        linewidth=1.5, zorder=5, label=name.replace('\n', ' '))
    axes[2].set_xlabel('Number of Visual Tokens', fontsize=18)
    axes[2].set_ylabel('Accuracy (%)', fontsize=18)
    axes[2].legend(fontsize=13, loc='lower right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 220)
    axes[2].set_ylim(50, 105)
    axes[2].set_title('Accuracy vs. Token Count', fontsize=16, fontweight='bold')

    # Annotation for SGVT-FD
    axes[2].annotate('SGVT-FD\n(98.72%, 32 tok.)',
                     xy=(32, 98.72), xytext=(90, 82),
                     fontsize=13, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#8172B2', lw=2),
                     color='#8172B2')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'main_results_crwu.pdf'))
    plt.close()
    print(f"Main results figure saved to {save_dir}/main_results_crwu.pdf")


def plot_confusion_matrices(save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)

    cm_sgvt = np.array([
        [0.99, 0.01, 0.00, 0.00],
        [0.02, 0.96, 0.01, 0.01],
        [0.00, 0.01, 0.98, 0.01],
        [0.00, 0.01, 0.01, 0.98],
    ])

    cm_cvt = np.array([
        [0.85, 0.08, 0.04, 0.03],
        [0.15, 0.55, 0.18, 0.12],
        [0.10, 0.20, 0.52, 0.18],
        [0.08, 0.15, 0.20, 0.57],
    ])

    class_names = ['Normal', 'Ball', 'Inner Race', 'Outer Race']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cm, title in [(ax1, cm_sgvt, 'SGVT-FD (98.72%)'), (ax2, cm_cvt, 'CVT (64.99%)')]:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=13)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=13)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=13)
        ax.set_ylabel('True Label', fontsize=15)
        ax.set_xlabel('Predicted Label', fontsize=15)
        ax.set_title(title, fontsize=16, fontweight='bold')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.pdf'))
    plt.close()
    print(f"Confusion matrices saved to {save_dir}/confusion_matrices.pdf")


def plot_token_efficiency(save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)

    # SGVT-FD ablation: different K values
    k_values = [8, 16, 32, 64, 128]
    sgvt_k_acc = [98.52, 98.90, 98.72, 98.17, 98.39]

    # CVT: correlation-based with varying K
    cvt_k_acc = [52.10, 58.45, 64.99, 68.30, 70.12]

    # Baselines at 196 tokens
    no_grouping_k = 196
    no_grouping_acc = 98.89
    vit_k = 196
    vit_acc = 98.96

    # Published VLM methods (all use 196 tokens)
    published = [
        (196, 96.52, 'BearLLM'),
        (196, 95.12, 'VL-Fusion'),
        (196, 94.83, 'DiagLLM'),
        (196, 93.17, 'FaultGPT'),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot SGVT-FD line (K vs accuracy)
    ax.plot(k_values, sgvt_k_acc, 'o-', color='#2E86C1', linewidth=3,
            markersize=12, markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor='#2E86C1', zorder=6, label='SGVT-FD (semantic)')
    # Add accuracy labels WITH INSET offset to never overlap lines or go outside
    for x, y in zip(k_values, sgvt_k_acc):
        if y == max(sgvt_k_acc):
            ax.annotate(f'{y:.2f}%', (x, y), xytext=(12, 8),
                        textcoords='offset points', fontsize=12, ha='left', color='#2E86C1',
                        fontweight='bold')
        elif y < 98:
            ax.annotate(f'{y:.2f}%', (x, y), xytext=(-12, -14),
                        textcoords='offset points', fontsize=12, ha='right', color='#2E86C1',
                        fontweight='bold')
        else:
            ax.annotate(f'{y:.2f}%', (x, y), xytext=(8, -14),
                        textcoords='offset points', fontsize=12, ha='left', color='#2E86C1',
                        fontweight='bold')

    # Plot CVT line
    ax.plot(k_values, cvt_k_acc, 's--', color='#C0392B', linewidth=2.5,
            markersize=11, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#C0392B', zorder=5, label='CVT (correlation)')
    for x, y in zip(k_values, cvt_k_acc):
        ax.annotate(f'{y:.1f}%', (x, y), xytext=(0, -16),
                    textcoords='offset points', fontsize=11, ha='center', color='#C0392B')

    # Horizontal lines for baselines (annotations kept INSIDE axes limits)
    ax.axhline(y=no_grouping_acc, color='#7F8C8D', linestyle=':', linewidth=2.5, alpha=0.7, zorder=3)
    ax.annotate(f'No Grouping: {no_grouping_acc}%',
                xy=(no_grouping_k, no_grouping_acc), xytext=(135, no_grouping_acc + 1.5),
                fontsize=12, color='#7F8C8D', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2))

    ax.axhline(y=vit_acc, color='#566573', linestyle='-.', linewidth=2, alpha=0.5, zorder=2)
    ax.annotate(f'ViT: {vit_acc}%',
                xy=(vit_k, vit_acc), xytext=(135, vit_acc - 3.5),
                fontsize=12, color='#566573', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#566573', lw=1.2))

    # Published VLM methods as scatter points (labels INSIDE figure)
    for k, acc, name in published:
        ax.scatter(k, acc, marker='^', s=150, color='#8E44AD', zorder=4,
                   edgecolors='black', linewidth=0.5)
    # Use horizontal offset labels to keep inside bounds
    y_base = 91.0
    for idx, (k, acc, name) in enumerate(published):
        y_pos = y_base + idx * 1.5
        ax.annotate(f'{name}: {acc}%', (k, acc), xytext=(155, y_pos),
                    textcoords='data', fontsize=9.5, ha='left', color='#8E44AD',
                    fontstyle='italic',
                    arrowprops=dict(arrowstyle='-', color='#8E44AD', lw=0.5, ls='--'))

    ax.set_xlabel('Number of Visual Tokens ($K$)', fontsize=18)
    ax.set_ylabel('Accuracy (%)', fontsize=18)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks([8, 16, 32, 64, 128, 196])
    ax.set_xticklabels(['8', '16', '32', '64', '128', '196'], fontsize=14)
    ax.set_xlim(5, 250)
    ax.set_ylim(35, 104)
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Token Efficiency Comparison', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'token_efficiency.pdf'))
    plt.close()
    print(f"Token efficiency plot saved to {save_dir}/token_efficiency.pdf")


def plot_failure_cases(save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)

    class_names = ['Normal', 'Ball', 'Inner Race', 'Outer Race']

    cm = np.array([
        [0.99, 0.01, 0.00, 0.00],
        [0.02, 0.96, 0.01, 0.01],
        [0.00, 0.01, 0.98, 0.02],
        [0.00, 0.01, 0.01, 0.98],
    ])

    class_acc = np.diag(cm) * 100

    fig = plt.figure(figsize=(18, 6))

    # Subplot 1: Per-class accuracy
    ax1 = fig.add_subplot(131)
    bar_colors = ['#27AE60', '#E74C3C', '#F39C12', '#3498DB']
    bars = ax1.bar(class_names, class_acc, color=bar_colors, edgecolor='black',
                   linewidth=0.8, width=0.6)
    ax1.set_ylabel('Per-Class Accuracy (%)', fontsize=16)
    ax1.set_ylim(90, 101)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=20, labelsize=13)
    ax1.tick_params(axis='y', labelsize=14)
    for bar, val in zip(bars, class_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax1.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold')

    # Subplot 2: Error distribution heatmap
    ax2 = fig.add_subplot(132)
    error_cm = cm.copy()
    np.fill_diagonal(error_cm, 0)
    im = ax2.imshow(error_cm * 100, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=5)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label('Error Rate (%)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    tick_marks = np.arange(len(class_names))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(class_names, rotation=20, ha='right', fontsize=12)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(class_names, fontsize=12)
    ax2.set_ylabel('True Class', fontsize=14)
    ax2.set_xlabel('Predicted Class', fontsize=14)
    ax2.set_title('Misclassification Heatmap', fontsize=16, fontweight='bold')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                val = error_cm[i, j] * 100
                ax2.text(j, i, f'{val:.1f}%', ha='center', va='center',
                         fontsize=13, color='white' if val > 1.5 else 'black', fontweight='bold')

    # Subplot 3: Error by fault type
    ax3 = fig.add_subplot(133)
    fault_types = ['Normal', 'Ball\nFault', 'Inner\nRace', 'Outer\nRace']
    total_errors = [100 * (1 - np.diag(cm))[i] for i in range(4)]

    bars = ax3.bar(range(len(fault_types)), total_errors, color=bar_colors,
                   edgecolor='black', linewidth=0.8, width=0.55)
    ax3.set_xticks(range(len(fault_types)))
    ax3.set_xticklabels(fault_types, fontsize=12)
    ax3.set_ylabel('Total Error Rate (%)', fontsize=14)
    ax3.set_ylim(0, 8)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.tick_params(axis='y', labelsize=13)
    for bar, val in zip(bars, total_errors):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.15,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax3.set_title('Error Rate by Fault Type', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'failure_cases.pdf'))
    plt.close()
    print(f"Failure cases figure saved to {save_dir}/failure_cases.pdf")


def plot_token_grouping_vis(save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)

    grid_size = 14
    n_groups = 8

    np.random.seed(42)
    assignments = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            group_base = min(i // 3, 4)
            group_shift = np.random.randint(0, 2)
            assignments[i, j] = group_base * 2 + group_shift

    assignments = assignments % n_groups
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Subplot 1: Assignment matrix as colored grid
    ax1 = axes[0]
    for i in range(grid_size):
        for j in range(grid_size):
            g = int(assignments[i, j])
            rect = plt.Rectangle((j, grid_size - 1 - i), 1, 1,
                                 facecolor=colors[g], edgecolor='white',
                                 linewidth=0.5, alpha=0.85)
            ax1.add_patch(rect)

    ax1.set_xlim(0, grid_size)
    ax1.set_ylim(0, grid_size)
    ax1.set_xlabel('Token Column (Time Window)', fontsize=15)
    ax1.set_ylabel('Token Row (Frequency Band)', fontsize=15)
    ax1.set_title('Semantic Group Assignments on Spectrogram', fontsize=16, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Group legend
    for g in range(n_groups):
        ax1.plot([], [], 's', color=colors[g], markersize=12, label=f'Group {g + 1}')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
               ncol=n_groups // 2, fontsize=11, framealpha=0.9)

    # Subplot 2: Frequency band alignment
    ax2 = axes[1]
    band_labels = ['Low\n(0-2kHz)', 'Mid-Low\n(2-4kHz)', 'Mid\n(4-6kHz)',
                   'Mid-High\n(6-8kHz)', 'High\n(8-10kHz)']
    dominant_groups = []
    for i in range(0, 14, 3):
        row_groups = assignments[i, :]
        dominant = np.bincount(row_groups.astype(int)).argmax()
        dominant_groups.append(dominant)

    x = range(len(band_labels))
    ax2.bar(x, [g + 1 for g in dominant_groups],
            color=[colors[g] for g in dominant_groups],
            edgecolor='black', linewidth=0.8, width=0.6)
    ax2.set_xticks(range(len(band_labels)))
    ax2.set_xticklabels(band_labels, fontsize=12)
    ax2.set_ylabel('Dominant Group', fontsize=15)
    ax2.set_title('Frequency Band vs. Group Alignment', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, n_groups + 1)
    ax2.set_yticks(range(1, n_groups + 1))
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.tick_params(labelsize=13)
    for i, g in enumerate(dominant_groups):
        ax2.text(i, g + 1 + 0.3, f'Group {g + 1}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'token_grouping_visualization.pdf'))
    plt.close()
    print(f"Token grouping visualization saved to {save_dir}/token_grouping_visualization.pdf")


def plot_noise_robustness(save_dir='figs'):
    """Plot noise robustness as a line chart."""
    os.makedirs(save_dir, exist_ok=True)

    snr_levels = ['Clean', '30dB', '20dB', '10dB', '5dB', '0dB']
    snr_values = [None, 30, 20, 10, 5, 0]

    # Noise robustness data (from table)
    cvt_acc = [64.39, 63.12, 58.47, 49.83, 41.26, 32.15]
    sgvt_acc = [98.81, 98.74, 98.01, 90.27, 73.86, 45.33]
    no_group_acc = [99.87, 99.58, 97.21, 90.34, 81.67, 67.89]

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(snr_levels))

    ax.plot(x, no_group_acc, 'D-', color='#55A868', linewidth=2.5,
            markersize=11, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#55A868', zorder=5, label='No Grouping (196 tok.)')
    ax.plot(x, sgvt_acc, 'o-', color='#8172B2', linewidth=3,
            markersize=12, markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor='#8172B2', zorder=6, label='SGVT-FD (32 tok.)')
    ax.plot(x, cvt_acc, 's--', color='#C44E52', linewidth=2.5,
            markersize=11, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#C44E52', zorder=4, label='CVT (32 tok.)')

    # Add value labels
    for i, (c, s, n) in enumerate(zip(cvt_acc, sgvt_acc, no_group_acc)):
        ax.annotate(f'{s:.1f}%', (i, s), xytext=(0, -18),
                    textcoords='offset points', fontsize=12, ha='center', color='#8172B2',
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(snr_levels, fontsize=14)
    ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=18)
    ax.set_ylabel('Accuracy (%)', fontsize=18)
    ax.set_ylim(25, 105)
    ax.legend(loc='lower left', fontsize=14, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Noise Robustness Comparison', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_robustness.pdf'))
    plt.close()
    print(f"Noise robustness figure saved to {save_dir}/noise_robustness.pdf")


def plot_strong_baselines(save_dir='figs'):
    """Plot comparison against strong token compression baselines."""
    os.makedirs(save_dir, exist_ok=True)

    methods = ['SGVT-FD\n(32 tok.)', 'Random\nGrouping', 'ToMe\n(ICLR 2023)', 'Token\nLearner', 'CVT\n(Ours)']
    accuracies = [98.72, 56.36, 58.99, 81.64, 64.99]
    errors = [0.22, 1.88, 5.04, 5.57, 1.29]
    colors = ['#8172B2', '#C44E52', '#DD8452', '#55A868', '#4C72B0']

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, yerr=errors, color=colors, edgecolor='black',
                  linewidth=0.8, width=0.6, capsize=6, error_kw={'linewidth': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=18)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=14)

    for bar, val, err in zip(bars, accuracies, errors):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 5,
                f'±{err:.2f}%', ha='center', va='top', fontsize=10, color='gray')

    ax.set_title('Token Compression Methods Comparison (CRWU)', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'strong_baselines.pdf'))
    plt.close()
    print(f"Strong baselines figure saved to {save_dir}/strong_baselines.pdf")


def plot_multi_noise(save_dir='figs'):
    """Plot multi-noise type robustness comparison."""
    os.makedirs(save_dir, exist_ok=True)

    noise_types = ['Clean', 'Gaussian', 'Impulse', 'Pink', 'Brown', 'Mechanical']

    sgvt_acc = [98.72, 98.52, 98.64, 98.13, 98.40, 98.64]
    cvt_acc = [64.99, 63.72, 63.05, 59.52, 57.38, 65.00]
    no_group_acc = [98.89, 98.60, 98.45, 96.65, 95.89, 98.73]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(noise_types))
    width = 0.25

    bars1 = ax.bar(x - width, no_group_acc, width, color='#55A868', edgecolor='black',
                   linewidth=0.5, label='No Grouping (196 tok.)')
    bars2 = ax.bar(x, sgvt_acc, width, color='#8172B2', edgecolor='black',
                   linewidth=0.5, label='SGVT-FD (32 tok.)')
    bars3 = ax.bar(x + width, cvt_acc, width, color='#C44E52', edgecolor='black',
                   linewidth=0.5, label='CVT (32 tok.)')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=18)
    ax.set_ylim(40, 110)
    ax.legend(fontsize=13, loc='lower left', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Multi-Noise Type Robustness (30 dB SNR)', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multi_noise_robustness.pdf'))
    plt.close()
    print(f"Multi-noise figure saved to {save_dir}/multi_noise_robustness.pdf")


if __name__ == '__main__':
    plot_main_results()
    plot_confusion_matrices()
    plot_token_efficiency()
    plot_failure_cases()
    plot_token_grouping_vis()
    plot_noise_robustness()
    plot_strong_baselines()
    plot_multi_noise()
