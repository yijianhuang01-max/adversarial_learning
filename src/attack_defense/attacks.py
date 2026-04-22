from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import denormalize_batch, normalize_batch


def _project_to_valid_range(
    normalized_inputs: torch.Tensor,
    original_inputs: torch.Tensor,
    epsilon: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    original_pixels = denormalize_batch(original_inputs, mean, std)
    perturbed_pixels = denormalize_batch(normalized_inputs, mean, std)
    lower = (original_pixels - epsilon).clamp(0.0, 1.0)
    upper = (original_pixels + epsilon).clamp(0.0, 1.0)
    clipped_pixels = perturbed_pixels.clamp(lower, upper).clamp(0.0, 1.0)
    return normalize_batch(clipped_pixels, mean, std)


def fgsm_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    adv_inputs = inputs.detach().clone().requires_grad_(True)
    logits = model(adv_inputs)
    loss = F.cross_entropy(logits, targets)
    model.zero_grad(set_to_none=True)
    loss.backward()

    scaled_step = epsilon / std
    adv_inputs = adv_inputs + scaled_step * adv_inputs.grad.sign()
    adv_inputs = _project_to_valid_range(adv_inputs.detach(), inputs.detach(), epsilon, mean, std)

    if was_training:
        model.train()
    return adv_inputs.detach()


def pgd_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    steps: int = 7,
    alpha: float | None = None,
    random_start: bool = True,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    alpha = alpha if alpha is not None else epsilon / max(steps, 1) * 1.25
    scaled_alpha = alpha / std
    scaled_epsilon = epsilon / std

    original_inputs = inputs.detach()
    if random_start:
        noise = torch.empty_like(original_inputs).uniform_(-1.0, 1.0) * scaled_epsilon
        adv_inputs = original_inputs + noise
        adv_inputs = _project_to_valid_range(adv_inputs, original_inputs, epsilon, mean, std)
    else:
        adv_inputs = original_inputs.clone()

    for _ in range(steps):
        adv_inputs = adv_inputs.detach().clone().requires_grad_(True)
        logits = model(adv_inputs)
        loss = F.cross_entropy(logits, targets)
        model.zero_grad(set_to_none=True)
        loss.backward()
        adv_inputs = adv_inputs + scaled_alpha * adv_inputs.grad.sign()
        delta = (adv_inputs - original_inputs).clamp(-scaled_epsilon, scaled_epsilon)
        adv_inputs = original_inputs + delta
        adv_inputs = _project_to_valid_range(adv_inputs.detach(), original_inputs, epsilon, mean, std)

    if was_training:
        model.train()
    return adv_inputs.detach()


def make_adversarial_examples(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    attack_name: str,
    epsilon: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    pgd_steps: int = 7,
    pgd_alpha: float | None = None,
) -> torch.Tensor:
    attack_name = attack_name.lower()
    if epsilon <= 0:
        return inputs.detach()
    if attack_name == "fgsm":
        return fgsm_attack(model, inputs, targets, epsilon, mean, std)
    if attack_name == "pgd":
        return pgd_attack(
            model,
            inputs,
            targets,
            epsilon,
            mean,
            std,
            steps=pgd_steps,
            alpha=pgd_alpha,
        )
    raise ValueError(f"Unsupported attack: {attack_name}")
