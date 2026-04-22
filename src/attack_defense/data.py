from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"


def get_normalization_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def normalize_batch(batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (batch - mean) / std


def denormalize_batch(batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return batch * std + mean


def _cifar10_ready(root: Path) -> bool:
    return (root / CIFAR10_EXTRACTED_DIR).is_dir()


def _download_with_wget(root: Path) -> None:
    archive_path = root / CIFAR10_ARCHIVE
    if shutil.which("wget") is None:
        raise RuntimeError(
            "download_method='wget' was requested, but wget is not installed or not on PATH."
        )
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-O", str(archive_path), CIFAR10_URL], check=True)
    with tarfile.open(archive_path, "r:gz") as handle:
        handle.extractall(path=root)


def ensure_cifar10_available(
    dataset_root: str | Path,
    download: bool,
    download_method: str = "auto",
) -> str:
    root = Path(dataset_root)
    if _cifar10_ready(root):
        return "local"
    if not download:
        raise RuntimeError(
            "CIFAR-10 was not found under the dataset root. Re-run with --download "
            "or prepare the dataset manually."
        )

    resolved_method = download_method
    if resolved_method == "auto":
        resolved_method = "wget" if shutil.which("wget") else "torchvision"

    if resolved_method == "wget":
        _download_with_wget(root)
        if not _cifar10_ready(root):
            raise RuntimeError("wget download finished, but CIFAR-10 was not extracted correctly.")
        return "wget"
    if resolved_method == "torchvision":
        return "torchvision"
    raise ValueError(f"Unsupported download method: {download_method}")


def build_cifar10_loaders(
    dataset_root: str | Path,
    batch_size: int,
    num_workers: int,
    download: bool = True,
    download_method: str = "auto",
) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    root = Path(dataset_root)
    resolved_method = ensure_cifar10_available(root, download=download, download_method=download_method)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    use_torchvision_download = resolved_method == "torchvision"
    train_dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=use_torchvision_download,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=use_torchvision_download,
        transform=test_transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
