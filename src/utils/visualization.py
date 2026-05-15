"""
Visualization utilities for fault diagnosis experiments.
Conference paper style: large fonts, serif fonts, clean layout.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import os


# Conference paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
})


def plot_spectrogram(spec, title="Spectrogram", save_path=None, cmap='viridis'):
    """Plot a spectrogram image.

    Args:
        spec: 2D numpy array (H, W)
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(spec, aspect='auto', origin='lower', cmap=cmap)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.colorbar(im, ax=ax)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True):
    """Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize by row
    """
    cm = sk_confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=14)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_training_curves(train_losses, val_losses, val_accs, save_path=None):
    """Plot training curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, 'g-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_bar_comparison(methods, results, metric_name="Accuracy (%)",
                        save_path=None, title=None, figsize=(12, 6)):
    """Plot bar chart comparison of methods.

    Args:
        methods: List of method names
        results: List of result values
        metric_name: Y-axis label
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(range(len(methods)), results, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel(metric_name)
    if title:
        ax.set_title(title)

    # Add value labels on bars
    for bar, val in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12)

    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_grouped_bar(methods, datasets, results, metric_name="Accuracy (%)",
                     save_path=None, figsize=(14, 6)):
    """Plot grouped bar chart for multi-dataset comparison.

    Args:
        methods: List of method names
        datasets: List of dataset names
        results: Dict mapping dataset -> list of results per method
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(methods))
    width = 0.8 / len(datasets)
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    for i, (dataset, color) in enumerate(zip(datasets, colors)):
        offset = (i - len(datasets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, results[dataset], width, label=dataset,
                      color=color, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_token_efficiency(token_counts, accuracies, methods, save_path=None):
    """Plot token count vs accuracy (efficiency plot).

    Args:
        token_counts: List of token counts per method
        accuracies: List of accuracies per method
        methods: List of method names
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    for i, (tc, acc, name) in enumerate(zip(token_counts, accuracies, methods)):
        ax.scatter(tc, acc, s=200, c=[colors[i]], edgecolors='black',
                   linewidth=1.5, zorder=5, label=name)

    ax.set_xlabel("Number of Visual Tokens")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig


def plot_ablation_groups(group_sizes, accuracies, save_path=None):
    """Plot ablation study on number of semantic groups.

    Args:
        group_sizes: List of K values
        accuracies: List of corresponding accuracies
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(group_sizes, accuracies, 'bo-', linewidth=2.5, markersize=10)
    ax.set_xlabel("Number of Semantic Groups (K)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    return fig
