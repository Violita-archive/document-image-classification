"""Arsitektur model klasifikasi beserta fungsi pembuatan dan pemuatannya."""

from pathlib import Path

import timm
import torch
import torch.nn as nn


def build_model(model_name: str, num_classes: int = 16, pretrained: bool = True, drop_rate: float = 0.0,
                drop_path_rate: float = 0.0) -> nn.Module:
    """Membuat backbone CNN dari library timm dengan kepala klasifikasi baru.

    Args:
        model_name: Nama arsitektur pada timm, misalnya `resnet18` atau `convnext_tiny`.
        num_classes: Jumlah kelas keluaran.
        pretrained: Menggunakan bobot pre-trained ImageNet jika True.
        drop_rate: Dropout sebelum lapisan klasifikasi.
        drop_path_rate: Peluang stochastic depth (melewati blok secara acak) untuk regularisasi.

    Returns:
        Model PyTorch.
    """
    kwargs = {"drop_path_rate": drop_path_rate} if drop_path_rate > 0 else {}
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate, **kwargs)


class DocumentClassifier(nn.Module):
    """Model klasifikasi citra dokumen berbasis CNN pre-trained (transfer learning).

    Backbone diambil dari timm dengan bobot ImageNet, lalu lapisan klasifikasi terakhir
    diganti agar menghasilkan 16 kelas. Seluruh lapisan di-fine-tune (tidak dibekukan)
    karena citra dokumen hasil scan cukup berbeda dengan citra alami pada ImageNet.

    Args:
        model_name: Nama arsitektur pada timm.
        num_classes: Jumlah kelas keluaran.
        pretrained: Menggunakan bobot pre-trained ImageNet jika True.
        drop_rate: Dropout sebelum lapisan klasifikasi.
        drop_path_rate: Peluang stochastic depth.
    """

    def __init__(self, model_name: str, num_classes: int = 16, pretrained: bool = True,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        """Membuat backbone timm dengan kepala klasifikasi baru.

        Args:
            model_name: Nama arsitektur pada timm.
            num_classes: Jumlah kelas keluaran.
            pretrained: Menggunakan bobot pre-trained ImageNet jika True.
            drop_rate: Dropout sebelum lapisan klasifikasi.
            drop_path_rate: Peluang stochastic depth.
        """
        super().__init__()
        self.backbone = build_model(model_name, num_classes, pretrained, drop_rate, drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Menghitung logit kelas.

        Args:
            x: Tensor gambar berukuran (N, 3, H, W).

        Returns:
            Tensor logit berukuran (N, jumlah kelas).
        """
        return self.backbone(x)

    def freeze(self) -> None:
        """Membekukan seluruh lapisan kecuali lapisan klasifikasi."""
        head = self.backbone.get_classifier()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

    def unfreeze(self) -> None:
        """Mengaktifkan kembali pelatihan pada seluruh lapisan."""
        for p in self.backbone.parameters():
            p.requires_grad = True


def model_from_config(cfg: dict, pretrained: bool | None = None) -> DocumentClassifier:
    """Membuat `DocumentClassifier` berdasarkan konfigurasi eksperimen.

    Args:
        cfg: Konfigurasi eksperimen.
        pretrained: Menimpa nilai `pretrained` pada konfigurasi jika diberikan.

    Returns:
        Objek `DocumentClassifier`.
    """
    return DocumentClassifier(
        cfg["model_name"], cfg["num_classes"],
        cfg.get("pretrained", True) if pretrained is None else pretrained,
        cfg.get("drop_rate", 0.0), cfg.get("drop_path_rate", 0.0),
    )


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

    Checkpoint lama yang disimpan tanpa pembungkus `DocumentClassifier` juga dapat dimuat.

    Args:
        path: Path file `.pth` hasil training.
        device: Perangkat tujuan.

    Returns:
        Tuple (model dalam mode eval, konfigurasi saat training).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = model_from_config(cfg, pretrained=False)
    state = ckpt["model_state"]
    if any(k.startswith("backbone.") for k in state):
        model.load_state_dict(state)
    else:
        model.backbone.load_state_dict(state)
    return model.to(device).eval(), cfg
