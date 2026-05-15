"""
Run all experiments for SGVT-FD paper.
Generates results for all tables and figures.
"""
import os
import sys
import json
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_command(cmd, desc=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    return result.returncode


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(base_dir, "src", "scripts", "train_sgvt.py")

    datasets = ["crwu", "mfpt"]
    models = ["sgvt", "sgvt_domain", "sgvt_mi", "gvt", "no_group", "vit"]
    seeds = [42, 123, 456]

    all_results = []

    for dataset in datasets:
        for model in models:
            for seed in seeds:
                cmd = (f"CUDA_VISIBLE_DEVICES=0 python {script} "
                       f"--dataset {dataset} "
                       f"--model {model} "
                       f"--seed {seed} "
                       f"--epochs 50 "
                       f"--batch_size 16 "
                       f"--num_groups 32")
                desc = f"{model} on {dataset} (seed={seed})"
                run_command(cmd, desc)

    # Collect all results
    results_dir = os.path.join(base_dir, "results")
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                result_path = os.path.join(
                    results_dir, dataset, model,
                    f"groups_32_seed_{seed}", "results.json"
                )
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        all_results.append(json.load(f))

    # Save combined results
    combined_path = os.path.join(results_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All experiments complete! Results saved to {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
