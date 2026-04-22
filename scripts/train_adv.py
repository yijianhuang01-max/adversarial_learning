from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from attack_defense.data import build_cifar10_loaders, get_normalization_tensors
from attack_defense.engine import evaluate_model, evaluate_under_attack, train_one_epoch
from attack_defense.models import build_resnet18
from attack_defense.utils import ensure_dir, resolve_device, save_csv, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 with adversarial training.")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--attack", type=str, default="fgsm", choices=("fgsm", "pgd"))
    parser.add_argument("--attack-epsilon", type=float, default=0.001)
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--pgd-alpha", type=float, default=None)
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
    mean, std = get_normalization_tensors(device)

    model = build_resnet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float | int]] = []
    best_adv_acc = 0.0
    best_path = checkpoints_dir / f"resnet18_adv_{args.attack}.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            attack_name=args.attack,
            attack_epsilon=args.attack_epsilon,
            mean=mean,
            std=std,
            pgd_steps=args.pgd_steps,
            pgd_alpha=args.pgd_alpha,
        )
        clean_test = evaluate_model(model, test_loader, device)
        adv_test = evaluate_under_attack(
            model,
            test_loader,
            device,
            attack_name=args.attack,
            epsilon=args.attack_epsilon,
            mean=mean,
            std=std,
            pgd_steps=args.pgd_steps,
            pgd_alpha=args.pgd_alpha,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_acc": clean_test["acc"],
            "adv_test_acc": adv_test["acc"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)

        if adv_test["acc"] >= best_adv_acc:
            best_adv_acc = adv_test["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": f"resnet18_adv_{args.attack}",
                    "num_classes": 10,
                    "epoch": epoch,
                    "test_acc": clean_test["acc"],
                    "adv_test_acc": adv_test["acc"],
                    "config": vars(args),
                },
                best_path,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"test_acc={clean_test['acc']:.4f} | "
            f"adv_test_acc={adv_test['acc']:.4f}"
        )

    summary = {
        "run_name": "adv_train",
        "model_name": f"resnet18_adv_{args.attack}",
        "best_checkpoint": str(best_path),
        "best_adv_test_acc": best_adv_acc,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "final_adv_test_acc": history[-1]["adv_test_acc"],
        "epochs": args.epochs,
        "config": vars(args),
    }
    save_json(metrics_dir / "adv_train_summary.json", summary)
    save_csv(metrics_dir / "adv_train_history.csv", history)


if __name__ == "__main__":
    main()
