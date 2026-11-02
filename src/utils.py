"""Fungsi bantu umum: seed, konfigurasi, path, dan visualisasi riwayat training."""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

CLASS_NAMES = [
    "advertisement", "budget", "email", "file_folder", "form", "handwritten",
    "invoice", "letter", "memo", "news_article", "presentation", "questionnaire",
    "resume", "scientific_publication", "scientific_report", "specification",
]


def set_seed(seed: int = 42) -> None:
    """Mengatur seed agar hasil eksperimen dapat direproduksi.

    Args:
        seed: Nilai seed untuk random, numpy, dan torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str | Path, **overrides) -> dict:
    """Membaca file konfigurasi YAML dan menimpa nilai tertentu bila diberikan.

    Args:
        path: Path file YAML.
        **overrides: Pasangan kunci-nilai yang menggantikan isi file.

    Returns:
        Dictionary konfigurasi.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    return cfg


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Menentukan lokasi folder data, baik di laptop maupun di Kaggle.

    Jika `data_dir` tidak ditemukan, folder `/kaggle/input` ditelusuri untuk
    mencari direktori yang berisi `train.csv`.

    Args:
        data_dir: Path data yang tertulis di konfigurasi.

    Returns:
        Path folder yang berisi `train.csv`.

    Raises:
        FileNotFoundError: Jika `train.csv` tidak ditemukan.
    """
    p = Path(data_dir)
    if (p / "train.csv").exists():
        return p
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob("train.csv"), key=lambda x: len(x.parts))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(f"train.csv tidak ditemukan di {p.resolve()} maupun /kaggle/input")


def get_device() -> torch.device:
    """Mengembalikan GPU bila tersedia, selain itu CPU.

    Returns:
        Objek `torch.device`.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*dirs: str | Path) -> None:
    """Membuat folder bila belum ada.

    Args:
        *dirs: Daftar path folder.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def plot_history(history: pd.DataFrame, title: str, save_path: str | Path | None = None) -> None:
    """Menampilkan kurva loss dan akurasi train vs validation per epoch.

    Args:
        history: DataFrame dengan kolom `epoch`, `train_loss`, `val_loss`, `train_acc`, `val_acc`.
        title: Judul grafik.
        save_path: Lokasi penyimpanan gambar. Tidak disimpan jika None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    axes[1].plot(history["epoch"], history["train_acc"], marker="o", label="train")
    axes[1].plot(history["epoch"], history["val_acc"], marker="o", label="validation")
    axes[1].set(title="Akurasi", xlabel="epoch", ylabel="akurasi")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
