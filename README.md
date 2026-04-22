# CDS521 Model Attack and Defense Project

This repository contains a course-dissertation starter focused on adversarial attack and defense for image classification. The codebase targets remote execution with PyTorch and `torchvision`.

## Project Structure

- `scripts/train_clean.py`: train a standard ResNet18 baseline on CIFAR-10.
- `scripts/train_adv.py`: train the same model with FGSM adversarial training.
- `scripts/evaluate_attack.py`: evaluate clean and adversarial robustness under FGSM and optional PGD.
- `scripts/generate_report_assets.py`: convert experiment outputs into figures and LaTeX fragments.`r`n- `scripts/download_cifar10.sh`: fetch and extract CIFAR-10 with `wget` on Linux servers.
- `src/attack_defense/`: shared data, model, attack, and training utilities.

## Remote Environment

Create a virtual environment on the remote server, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Experiment Workflow

1. Train the clean baseline:

```powershell
python scripts/train_clean.py --dataset-root data --epochs 20 --download --download-method wget
```

2. Train the defended model:

```powershell
python scripts/train_adv.py --dataset-root data --epochs 20 --attack-epsilon 0.03137 --download-method wget
```

3. Evaluate both checkpoints:

```powershell
python scripts/evaluate_attack.py --dataset-root data --attack fgsm --epsilons 0,0.00784,0.01569,0.03137 --download-method wget
```
