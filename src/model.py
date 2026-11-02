"""Pembuatan dan pemuatan model klasifikasi."""

from pathlib import Path

import timm
import torch
import torch.nn as nn


def build_model(model_name: str, num_classes: int = 16, pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """Membuat model CNN dari library timm dengan kepala klasifikasi baru.

    Args:
        model_name: Nama arsitektur pada timm, misalnya `resnet18` atau `convnext_tiny`.
        num_classes: Jumlah kelas keluaran.
        pretrained: Menggunakan bobot pre-trained ImageNet jika True.
        drop_rate: Dropout sebelum lapisan klasifikasi.

    Returns:
        Model PyTorch.
    """
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def count_parameters(model: nn.Module) -> int:
    """Menghitung jumlah parameter yang dapat dilatih.

    Args:
        model: Model PyTorch.

    Returns:
        Jumlah parameter.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Memuat model beserta konfigurasi dari file checkpoint.

    Args:
        path: Path file `.pth` hasil training.
        device: Perangkat tujuan.

    Returns:
        Tuple (model dalam mode eval, konfigurasi saat training).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model_name"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg
