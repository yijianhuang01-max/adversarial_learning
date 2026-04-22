from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .attacks import make_adversarial_examples
from .utils import AverageMeter, accuracy_from_logits


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attack_name: str | None = None,
    attack_epsilon: float = 0.0,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    pgd_steps: int = 7,
    pgd_alpha: float | None = None,
) -> dict[str, float]:
    if attack_name is not None and (mean is None or std is None):
        raise ValueError("mean and std tensors are required for adversarial training.")

    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if attack_name is not None and attack_epsilon > 0:
            batch_inputs = make_adversarial_examples(
                model,
                inputs,
                targets,
                attack_name=attack_name,
                epsilon=attack_epsilon,
                mean=mean,
                std=std,
                pgd_steps=pgd_steps,
                pgd_alpha=pgd_alpha,
            )
        else:
            batch_inputs = inputs

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy_from_logits(logits, targets), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy_from_logits(logits, targets), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def _predict(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model(inputs).argmax(dim=1)


def evaluate_under_attack(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    attack_name: str,
    epsilon: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    pgd_steps: int = 7,
    pgd_alpha: float | None = None,
) -> dict[str, Any]:
    model.eval()
    acc_meter = AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
        predictions = _predict(model, adv_inputs)
        batch_accuracy = predictions.eq(targets).float().mean().item()
        acc_meter.update(batch_accuracy, targets.size(0))

    return {"acc": acc_meter.avg}
