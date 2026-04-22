from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from attack_defense.data import build_cifar10_loaders
from attack_defense.engine import evaluate_model, train_one_epoch
from attack_defense.models import build_resnet18
from attack_defense.utils import ensure_dir, resolve_device, save_csv, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clean ResNet18 baseline on CIFAR-10.")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-method",
        type=str,
        default="auto",
        choices=("auto", "wget", "torchvision"),
        help="Dataset bootstrap method. 'auto' prefers wget when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")

    train_loader, test_loader = build_cifar10_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        download_method=args.download_method,
    )

    model = build_resnet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float | int]] = []
    best_acc = 0.0
    best_path = checkpoints_dir / "resnet18_clean.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate_model(model, test_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)

        if test_metrics["acc"] >= best_acc:
            best_acc = test_metrics["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "resnet18_clean",
                    "num_classes": 10,
                    "epoch": epoch,
                    "test_acc": test_metrics["acc"],
                    "config": vars(args),
                },
                best_path,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_acc={train_metrics['acc']:.4f} | test_acc={test_metrics['acc']:.4f}"
        )

    summary = {
        "run_name": "clean_train",
        "model_name": "resnet18_clean",
        "best_checkpoint": str(best_path),
        "best_test_acc": best_acc,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "epochs": args.epochs,
        "config": vars(args),
    }
    save_json(metrics_dir / "clean_train_summary.json", summary)
    save_csv(metrics_dir / "clean_train_history.csv", history)


if __name__ == "__main__":
    main()
