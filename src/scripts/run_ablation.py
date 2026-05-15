"""
Ablation studies for SGVT-FD.
1. Number of groups (K)
2. Domain prior vs no prior
3. MI loss vs no MI loss
4. Different VLM sizes (3B vs 7B - conceptual)
"""
import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_command(cmd, desc=""):
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
    dataset = "crwu"

    # Ablation 1: Number of groups
    print("\n" + "="*60)
    print("Ablation 1: Number of Semantic Groups (K)")
    print("="*60)
    for k in [8, 16, 32, 64, 128]:
        cmd = (f"CUDA_VISIBLE_DEVICES=0 python {script} "
               f"--dataset {dataset} "
               f"--model sgvt "
               f"--num_groups {k} "
               f"--seed 42 "
               f"--epochs 30")
        run_command(cmd, f"K={k}")

    # Ablation 2: Domain prior
    print("\n" + "="*60)
    print("Ablation 2: Domain Prior")
    print("="*60)
    for model in ["sgvt", "sgvt_domain"]:
        cmd = (f"CUDA_VISIBLE_DEVICES=0 python {script} "
               f"--dataset {dataset} "
               f"--model {model} "
               f"--seed 42 "
               f"--epochs 30")
        run_command(cmd, f"Model={model}")

    # Ablation 3: MI loss
    print("\n" + "="*60)
    print("Ablation 3: Information-Theoretic Loss")
    print("="*60)
    for model in ["sgvt", "sgvt_mi"]:
        cmd = (f"CUDA_VISIBLE_DEVICES=0 python {script} "
               f"--dataset {dataset} "
               f"--model {model} "
               f"--seed 42 "
               f"--epochs 30")
        run_command(cmd, f"Model={model}")

    # Collect ablation results
    results_dir = os.path.join(base_dir, "results")
    ablation_results = {
        "num_groups": {},
        "domain_prior": {},
        "mi_loss": {},
    }

    # Number of groups
    for k in [8, 16, 32, 64, 128]:
        path = os.path.join(results_dir, dataset, "sgvt", f"groups_{k}_seed_42", "results.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
                ablation_results["num_groups"][k] = r["test_accuracy"]

    # Domain prior
    for model in ["sgvt", "sgvt_domain"]:
        path = os.path.join(results_dir, dataset, model, "groups_32_seed_42", "results.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
                ablation_results["domain_prior"][model] = r["test_accuracy"]

    # MI loss
    for model in ["sgvt", "sgvt_mi"]:
        path = os.path.join(results_dir, dataset, model, "groups_32_seed_42", "results.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
                ablation_results["mi_loss"][model] = r["test_accuracy"]

    # Save ablation results
    ablation_path = os.path.join(results_dir, "ablation_results.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Ablation Study Summary")
    print(f"{'='*60}")

    print("\n1. Number of Groups (K):")
    for k, acc in sorted(ablation_results["num_groups"].items()):
        print(f"  K={k}: {acc*100:.2f}%")

    print("\n2. Domain Prior:")
    for model, acc in ablation_results["domain_prior"].items():
        print(f"  {model}: {acc*100:.2f}%")

    print("\n3. MI Loss:")
    for model, acc in ablation_results["mi_loss"].items():
        print(f"  {model}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
