"""Grad-CAM untuk melihat bagian gambar yang menjadi dasar prediksi model."""

import numpy as np
import torch
import torch.nn.functional as F


def get_target_layer(model) -> torch.nn.Module:
    """Mengambil lapisan konvolusi terakhir yang umum dipakai untuk Grad-CAM.

    Args:
        model: `DocumentClassifier` atau model timm (ResNet, ConvNeXt, EfficientNet, dan sejenisnya).

    Returns:
        Modul lapisan target.
    """
    model = getattr(model, "backbone", model)
    for name in ("stages", "layer4", "blocks", "features"):
        if hasattr(model, name):
            layer = getattr(model, name)
            return layer[-1] if isinstance(layer, torch.nn.Sequential) or hasattr(layer, "__getitem__") else layer
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1]


class GradCAM:
    """Implementasi Grad-CAM (Selvaraju dkk., 2017).

    Peta aktivasi dihitung dari rata-rata gradien kelas target terhadap feature map
    lapisan target, lalu dijumlahkan berbobot dan dilewatkan ReLU.

    Args:
        model: Model PyTorch dalam mode eval.
        target_layer: Lapisan yang feature map-nya digunakan.
    """

    def __init__(self, model, target_layer):
        """Memasang hook forward dan backward pada lapisan target.

        Args:
            model: Model PyTorch dalam mode eval.
            target_layer: Lapisan yang feature map-nya digunakan.
        """
        self.model = model
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._save_activation)
        self._h2 = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        """Menyimpan feature map hasil forward."""
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        """Menyimpan gradien hasil backward."""
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        """Menghitung peta Grad-CAM untuk satu gambar.

        Args:
            x: Tensor gambar berukuran (1, 3, H, W).
            class_idx: Kelas target. Jika None, kelas prediksi yang digunakan.

        Returns:
            Tuple (peta panas berukuran (H, W) bernilai 0-1, kelas, probabilitas kelas).
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        prob = logits.softmax(1)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx, float(prob[0, class_idx])

    def remove(self) -> None:
        """Melepas hook dari model."""
        self._h1.remove()
        self._h2.remove()
