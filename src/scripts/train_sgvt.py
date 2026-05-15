"""
Main training script for SGVT-FD.
Trains the semantic grouped visual token model for fault diagnosis.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import *
from src.data.crwu_loader import load_crwu_data, CRWUDataset
from src.data.mfpt_loader import load_mfpt_data, MFPTDataset
from src.models.sgvt import SGVTModel, SGVTBaseline, NoGroupingBaseline
from src.models.baselines import CVTBaseline, ViTBaseline
from src.utils.metrics import compute_metrics, compute_confusion_matrix
from src.utils.visualization import plot_training_curves, plot_confusion_matrix


def get_args():
    parser = argparse.ArgumentParser(description="SGVT-FD Training")
    parser.add_argument("--dataset", type=str, default="crwu", choices=["crwu", "mfpt"])
    parser.add_argument("--model", type=str, default="sgvt",
                        choices=["sgvt", "sgvt_domain", "sgvt_mi_only", "sgvt_mi", "gvt", "no_group", "vit"])
    parser.add_argument("--num_groups", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--signal_length", type=int, default=8192)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--spec_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, loss, _ = model(images, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits, loss, _ = model(images, labels)

        if loss is not None:
            total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


def build_model(args, num_classes):
    """Build model based on args."""
    fault_freqs = CRWU_FAULT_FREQS if args.dataset == "crwu" else None
    fs = SAMPLING_RATE_CRWU if args.dataset == "crwu" else SAMPLING_RATE_MFPT

    if args.model == "sgvt":
        model = SGVTModel(
            num_classes=num_classes, num_groups=args.num_groups,
            use_domain_prior=False, use_mi_loss=False,
            device=args.device,
        )
    elif args.model == "sgvt_domain":
        model = SGVTModel(
            num_classes=num_classes, num_groups=args.num_groups,
            use_domain_prior=True, use_mi_loss=False,
            fault_freqs=fault_freqs, fs=fs, device=args.device,
        )
    elif args.model == "sgvt_mi_only":
        model = SGVTModel(
            num_classes=num_classes, num_groups=args.num_groups,
            use_domain_prior=False, use_mi_loss=True,
            device=args.device,
        )
    elif args.model == "sgvt_mi":
        model = SGVTModel(
            num_classes=num_classes, num_groups=args.num_groups,
            use_domain_prior=True, use_mi_loss=True,
            fault_freqs=fault_freqs, fs=fs, device=args.device,
        )
    elif args.model == "gvt":
        model = CVTBaseline(
            num_classes=num_classes, num_groups=args.num_groups,
            device=args.device,
        )
    elif args.model == "no_group":
        model = NoGroupingBaseline(
            num_classes=num_classes, device=args.device,
        )
    elif args.model == "vit":
        model = ViTBaseline(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def main():
    args = get_args()
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            OUTPUT_ROOT, args.dataset, args.model,
            f"groups_{args.num_groups}_seed_{args.seed}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "crwu":
        signals, labels, class_names = load_crwu_data(
            CRWU_ROOT, signal_length=args.signal_length, overlap=args.overlap
        )
        dataset = CRWUDataset(signals, labels, fs=SAMPLING_RATE_CRWU,
                              spec_size=(args.spec_size, args.spec_size))
    elif args.dataset == "mfpt":
        signals, labels, class_names = load_mfpt_data(
            MFPT_ROOT, signal_length=args.signal_length, overlap=args.overlap
        )
        dataset = MFPTDataset(signals, labels, fs=SAMPLING_RATE_MFPT,
                              spec_size=(args.spec_size, args.spec_size))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_classes = len(class_names)

    # Split dataset
    total = len(dataset)
    test_size = int(total * TEST_RATIO)
    val_size = int(total * VAL_RATIO)
    train_size = total - test_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Build model
    model = build_model(args, num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, Params: {total_params:,} (trainable: {trainable_params:,})")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[args.warmup_epochs])

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_accs = []

    print(f"\nTraining {args.model} on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_metrics["accuracy"])

        elapsed = time.time() - start_time

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, os.path.join(args.output_dir, "best_model.pt"))

    # Final test evaluation
    print(f"\nBest epoch: {best_epoch}, Val Acc: {best_val_acc:.4f}")

    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics, test_preds, test_labels = evaluate(
        model, test_loader, device
    )

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")

    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "num_groups": args.num_groups,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, val_accs,
        save_path=os.path.join(args.output_dir, "training_curves.png")
    )

    # Plot confusion matrix
    cm, report = compute_confusion_matrix(test_labels, test_preds, class_names)
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        save_path=os.path.join(args.output_dir, "confusion_matrix.png")
    )

    print(f"\nResults saved to {args.output_dir}")
    return results


if __name__ == "__main__":
    main()
