from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from attack_defense.attacks import make_adversarial_examples
from attack_defense.data import build_cifar10_loaders, denormalize_batch, get_normalization_tensors
from attack_defense.engine import evaluate_model, evaluate_under_attack
from attack_defense.models import build_resnet18
from attack_defense.utils import CIFAR10_CLASSES, ensure_dir, resolve_device, save_csv, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 models under adversarial attack.")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--clean-checkpoint", type=str, default="outputs/checkpoints/resnet18_clean.pt")
    parser.add_argument("--adv-checkpoint", type=str, default="outputs/checkpoints/resnet18_adv_fgsm.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--attack", type=str, default="fgsm", choices=("fgsm", "pgd"))
    parser.add_argument("--epsilons", type=str, default="0,0.001,0.005,0.01")
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
    parser.add_argument("--num-figure-examples", type=int, default=6)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_resnet18(num_classes=checkpoint.get("num_classes", 10)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def save_example_grid(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epsilon: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    attack_name: str,
    pgd_steps: int,
    pgd_alpha: float | None,
    output_path: Path,
    num_examples: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            clean_predictions = model(inputs).argmax(dim=1)
        adv_inputs = make_adversarial_examples(
            model,
            inputs,
            targets,
            attack_name=attack_name,
            epsilon=epsilon,
            mean=mean,
            std=std,
            pgd_steps=pgd_steps,
            pgd_alpha=pgd_alpha,
        )
        with torch.no_grad():
            adv_predictions = model(adv_inputs).argmax(dim=1)

        candidate_indices = torch.where((clean_predictions == targets) & (adv_predictions != targets))[0]
        if candidate_indices.numel() == 0:
            candidate_indices = torch.arange(min(inputs.size(0), num_examples), device=device)
        else:
            candidate_indices = candidate_indices[:num_examples]

        clean_pixels = denormalize_batch(inputs[candidate_indices], mean, std).clamp(0.0, 1.0).cpu()
        adv_pixels = denormalize_batch(adv_inputs[candidate_indices], mean, std).clamp(0.0, 1.0).cpu()
        clean_pred_labels = clean_predictions[candidate_indices].cpu().tolist()
        adv_pred_labels = adv_predictions[candidate_indices].cpu().tolist()
        true_labels = targets[candidate_indices].cpu().tolist()

        figure, axes = plt.subplots(2, len(candidate_indices), figsize=(2.2 * len(candidate_indices), 4.0))
        if len(candidate_indices) == 1:
            axes = axes.reshape(2, 1)

        for column in range(len(candidate_indices)):
            axes[0, column].imshow(clean_pixels[column].permute(1, 2, 0).numpy())
            axes[0, column].set_title(
                f"Orig\nT:{CIFAR10_CLASSES[true_labels[column]]}\nP:{CIFAR10_CLASSES[clean_pred_labels[column]]}",
                fontsize=8,
            )
            axes[1, column].imshow(adv_pixels[column].permute(1, 2, 0).numpy())
            axes[1, column].set_title(
                f"Adv\nP:{CIFAR10_CLASSES[adv_pred_labels[column]]}",
                fontsize=8,
            )
            axes[0, column].axis("off")
            axes[1, column].axis("off")

        figure.suptitle(f"{attack_name.upper()} examples at epsilon={epsilon:.5f}", fontsize=10)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        return

    raise RuntimeError("Could not extract examples from the evaluation loader.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    epsilons = [float(value) for value in args.epsilons.split(",") if value.strip()]

    output_dir = ensure_dir(args.output_dir)
    metrics_dir = ensure_dir(output_dir / "metrics")
    figures_dir = ensure_dir(output_dir / "figures")

    _, test_loader = build_cifar10_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        download_method=args.download_method,
    )
    mean, std = get_normalization_tensors(device)

    checkpoints = [
        ("clean", Path(args.clean_checkpoint)),
        ("adv_train", Path(args.adv_checkpoint)),
    ]
    rows: list[dict[str, float | str]] = []
    summary: dict[str, object] = {"attack": args.attack, "epsilons": epsilons, "models": {}}

    for label, checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue

        model, checkpoint = load_model(checkpoint_path, device)
        clean_metrics = evaluate_model(model, test_loader, device)
        summary["models"][label] = {
            "checkpoint": str(checkpoint_path),
            "clean_acc": clean_metrics["acc"],
            "model_name": checkpoint.get("model_name", label),
        }
        for epsilon in epsilons:
            attack_metrics = evaluate_under_attack(
                model,
                test_loader,
                device,
                attack_name=args.attack,
                epsilon=epsilon,
                mean=mean,
                std=std,
                pgd_steps=args.pgd_steps,
                pgd_alpha=args.pgd_alpha,
            )
            row = {
                "model_label": label,
                "model_name": checkpoint.get("model_name", label),
                "attack_name": args.attack,
                "epsilon": epsilon,
                "clean_acc": clean_metrics["acc"],
                "adv_test_acc": attack_metrics["acc"],
            }
            rows.append(row)
            print(
                f"{label:>9s} | attack={args.attack:<4s} | epsilon={epsilon:.5f} | "
                f"clean_acc={clean_metrics['acc']:.4f} | adv_acc={attack_metrics['acc']:.4f}"
            )

        if label == "clean" and epsilons:
            save_example_grid(
                model,
                test_loader,
                device,
                epsilon=max(epsilons),
                mean=mean,
                std=std,
                attack_name=args.attack,
                pgd_steps=args.pgd_steps,
                pgd_alpha=args.pgd_alpha,
                output_path=figures_dir / "adversarial_examples.png",
                num_examples=args.num_figure_examples,
            )

    save_csv(
        metrics_dir / "attack_eval.csv",
        rows,
        fieldnames=["model_label", "model_name", "attack_name", "epsilon", "clean_acc", "adv_test_acc"],
    )
    save_json(metrics_dir / "attack_eval_summary.json", summary)


if __name__ == "__main__":
    main()
