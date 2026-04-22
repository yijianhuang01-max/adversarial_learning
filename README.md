# CDS521 Model Attack and Defense Project

This repository contains a course-dissertation starter focused on adversarial attack and defense for image classification. The codebase targets remote execution with PyTorch and `torchvision`, while generated report assets live under `report/`.

## Project Structure

- `scripts/train_clean.py`: train a standard ResNet18 baseline on CIFAR-10.
- `scripts/train_adv.py`: train the same model with FGSM adversarial training.
- `scripts/evaluate_attack.py`: evaluate clean and adversarial robustness under FGSM and optional PGD.
- `scripts/generate_report_assets.py`: convert experiment outputs into figures and LaTeX fragments.`r`n- `scripts/download_cifar10.sh`: fetch and extract CIFAR-10 with `wget` on Linux servers.
- `src/attack_defense/`: shared data, model, attack, and training utilities.
- `CDS521_Attack_Defense_Report.tex`: course dissertation source.
- `report/generated/`: auto-generated LaTeX fragments consumed by the report.
- `output/pdf/`: recommended location for the final compiled PDF.

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

4. Turn metrics into report-ready assets:

```powershell
python scripts/generate_report_assets.py
```

5. Compile the dissertation:

```powershell
pdflatex CDS521_Attack_Defense_Report.tex
bibtex CDS521_Attack_Defense_Report
pdflatex CDS521_Attack_Defense_Report.tex
pdflatex CDS521_Attack_Defense_Report.tex
Copy-Item CDS521_Attack_Defense_Report.pdf output\pdf\cds521_attack_defense_report.pdf
```

## Notes

- The local workspace does not need to match the remote training environment.`r`n- On Linux servers, `--download-method wget` uses the official CIFAR-10 archive directly and avoids the slower `torchvision` downloader.
- `scripts/generate_report_assets.py` creates placeholder figures and tables when experiment outputs are missing, so the LaTeX report can still compile before remote runs finish.
- Replace the name and student ID placeholders in `CDS521_Attack_Defense_Report.tex` before submission.

