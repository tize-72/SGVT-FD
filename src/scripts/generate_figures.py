"""
Generate all figures for the SGVT-FD paper.
Conference paper style: large fonts, serif fonts, clean layout.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import *
from src.utils.visualization import (
    plot_bar_comparison, plot_grouped_bar, plot_token_efficiency,
    plot_ablation_groups, plot_confusion_matrix
)


def load_all_results(results_dir):
    """Load all experiment results."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        if "results.json" in files:
            with open(os.path.join(root, "results.json")) as f:
                results.append(json.load(f))
    return results


def generate_main_results_table(results, dataset, output_dir):
    """Generate main results comparison table."""
    # Filter results for this dataset
    ds_results = [r for r in results if r["dataset"] == dataset]
    if not ds_results:
        return

    # Group by model, average across seeds
    model_results = {}
    for r in ds_results:
        model = r["model"]
        if model not in model_results:
            model_results[model] = {"acc": [], "f1": []}
        model_results[model]["acc"].append(r["test_accuracy"])
        model_results[model]["f1"].append(r["test_f1"])

    # Model display names
    model_names = {
        "sgvt": "SGVT-FD (Ours)",
        "sgvt_domain": "SGVT-FD + Domain",
        "sgvt_mi": "SGVT-FD + MI",
        "gvt": "CVT",
        "no_group": "No Grouping (ViT)",
        "vit": "ViT Baseline",
    }

    methods = []
    accs = []
    f1s = []
    for model in ["vit", "no_group", "gvt", "sgvt", "sgvt_domain", "sgvt_mi"]:
        if model in model_results:
            methods.append(model_names.get(model, model))
            accs.append(np.mean(model_results[model]["acc"]) * 100)
            f1s.append(np.mean(model_results[model]["f1"]) * 100)

    # Plot
    plot_bar_comparison(
        methods, accs,
        metric_name="Accuracy (%)",
        title=None,
        save_path=os.path.join(output_dir, f"main_results_{dataset}.png"),
        figsize=(12, 6)
    )

    # Print table
    print(f"\n{'='*60}")
    print(f"Table: Main Results on {dataset.upper()} Dataset")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Accuracy (%)':<15} {'F1 (%)':<15}")
    print(f"{'-'*60}")
    for m, a, f in zip(methods, accs, f1s):
        print(f"{m:<30} {a:<15.2f} {f:<15.2f}")


def generate_grouping_comparison(results, output_dir):
    """Generate figure comparing different grouping strategies."""
    ds_results = [r for r in results if r["dataset"] == "crwu"]
    if not ds_results:
        return

    model_results = {}
    for r in ds_results:
        model = r["model"]
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(r["test_accuracy"])

    # Token counts for efficiency plot
    token_counts = {
        "vit": 196,        # All patches
        "no_group": 196,   # All patches
        "gvt": 32,         # K=32 groups
        "sgvt": 32,
        "sgvt_domain": 32,
        "sgvt_mi": 32,
    }

    model_names = {
        "vit": "ViT",
        "no_group": "No Grouping",
        "gvt": "CVT",
        "sgvt": "SGVT-FD",
        "sgvt_domain": "SGVT+Domain",
        "sgvt_mi": "SGVT+MI",
    }

    methods = []
    tcs = []
    accs = []
    for model in ["vit", "no_group", "gvt", "sgvt", "sgvt_domain", "sgvt_mi"]:
        if model in model_results:
            methods.append(model_names.get(model, model))
            tcs.append(token_counts.get(model, 196))
            accs.append(np.mean(model_results[model]) * 100)

    plot_token_efficiency(tcs, accs, methods,
                          save_path=os.path.join(output_dir, "token_efficiency.png"))


def generate_dataset_comparison(results, output_dir):
    """Generate cross-dataset comparison figure."""
    model_results = {}
    for r in results:
        model = r["model"]
        dataset = r["dataset"]
        key = (model, dataset)
        if key not in model_results:
            model_results[key] = []
        model_results[key].append(r["test_accuracy"])

    model_names = {
        "sgvt": "SGVT-FD",
        "sgvt_mi": "SGVT+MI",
        "gvt": "CVT",
        "no_group": "No Grouping",
        "vit": "ViT",
    }

    methods = [m for m in ["vit", "no_group", "gvt", "sgvt", "sgvt_mi"]
               if any((m, d) in model_results for d in ["crwu", "mfpt"])]
    method_names = [model_names.get(m, m) for m in methods]
    datasets = ["crwu", "mfpt"]
    dataset_names = {"crwu": "CRWU", "mfpt": "MFPT"}

    data = {}
    for ds in datasets:
        data[dataset_names.get(ds, ds)] = [
            np.mean(model_results.get((m, ds), [0])) * 100 for m in methods
        ]

    plot_grouped_bar(
        method_names,
        [dataset_names.get(d, d) for d in datasets],
        data,
        metric_name="Accuracy (%)",
        save_path=os.path.join(output_dir, "dataset_comparison.png"),
    )


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    output_dir = FIGS_ROOT
    os.makedirs(output_dir, exist_ok=True)

    results = load_all_results(results_dir)
    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"Loaded {len(results)} results")

    # Generate figures
    generate_main_results_table(results, "crwu", output_dir)
    generate_main_results_table(results, "mfpt", output_dir)
    generate_grouping_comparison(results, output_dir)
    generate_dataset_comparison(results, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
