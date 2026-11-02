"""Prediksi dan perhitungan metrik evaluasi."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from src.utils import CLASS_NAMES


@torch.no_grad()
def predict_proba(model, loader, device, use_amp: bool = True) -> np.ndarray:
    """Menghitung probabilitas kelas untuk seluruh data pada loader.

    Args:
        model: Model PyTorch.
        loader: DataLoader.
        device: Perangkat komputasi.
        use_amp: Menggunakan mixed precision jika tersedia GPU.

    Returns:
        Array probabilitas berukuran (N, jumlah kelas).
    """
    model.eval()
    use_amp = use_amp and device.type == "cuda"
    out = []
    for x, _ in tqdm(loader, desc="predict", leave=False):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out.append(model(x.to(device)).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def evaluation_report(y_true, y_pred) -> tuple[float, pd.DataFrame]:
    """Menghitung akurasi serta precision, recall, dan F1 per kelas.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.

    Returns:
        Tuple (akurasi, DataFrame classification report).
    """
    rep = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return accuracy_score(y_true, y_pred), pd.DataFrame(rep).T


def plot_confusion_matrix(y_true, y_pred, normalize: bool = True, save_path=None) -> np.ndarray:
    """Menampilkan confusion matrix.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        normalize: Menampilkan proporsi per baris (recall) jika True.
        save_path: Lokasi penyimpanan gambar. Tidak disimpan jika None.

    Returns:
        Matriks confusion (dinormalisasi jika `normalize=True`).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
    ax.set(xlabel="Prediksi", ylabel="Sebenarnya", title="Confusion matrix" + (" (proporsi per kelas)" if normalize else ""))
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return cm


def top_confusions(y_true, y_pred, k: int = 10) -> pd.DataFrame:
    """Mencari pasangan kelas yang paling sering tertukar.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        k: Jumlah pasangan yang ditampilkan.

    Returns:
        DataFrame berisi kelas sebenarnya, kelas prediksi, dan jumlah kesalahan.
    """
    df = pd.DataFrame({"sebenarnya": y_true, "prediksi": y_pred})
    df = df[df["sebenarnya"] != df["prediksi"]]
    out = df.value_counts().head(k).reset_index(name="jumlah")
    out["sebenarnya"] = out["sebenarnya"].map(lambda i: CLASS_NAMES[i])
    out["prediksi"] = out["prediksi"].map(lambda i: CLASS_NAMES[i])
    return out
