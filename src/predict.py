"""Prediksi data test dan pembuatan file submission Kaggle.

Dapat dijalankan dari terminal:
    python -m src.predict --checkpoint checkpoint/convnext_tiny_512_best.pth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import load_metadata, make_loader
from src.evaluate import predict_proba
from src.model import load_checkpoint
from src.transforms import get_eval_transforms
from src.utils import ensure_dirs, get_device, resolve_data_dir


def predict_test(model, cfg: dict, test_df: pd.DataFrame) -> np.ndarray:
    """Menghitung probabilitas kelas untuk seluruh data test.

    Args:
        model: Model terlatih.
        cfg: Konfigurasi yang dipakai saat training (untuk ukuran gambar dan batch).
        test_df: DataFrame test dengan kolom `path`.

    Returns:
        Array probabilitas berukuran (N, jumlah kelas).
    """
    device = next(model.parameters()).device
    loader = make_loader(test_df, get_eval_transforms(cfg["img_size"]), cfg["batch_size"] * 2, False, cfg["num_workers"])
    return predict_proba(model, loader, device)


def save_submission(probs: np.ndarray, test_df: pd.DataFrame, save_path: str | Path) -> pd.DataFrame:
    """Menyimpan prediksi dalam format submission Kaggle (kolom `id`, `label`).

    Args:
        probs: Probabilitas kelas berukuran (N, jumlah kelas).
        test_df: DataFrame test dengan kolom `id`.
        save_path: Lokasi file CSV.

    Returns:
        DataFrame submission.
    """
    sub = pd.DataFrame({"id": test_df["id"].values, "label": probs.argmax(1)})
    ensure_dirs(Path(save_path).parent)
    sub.to_csv(save_path, index=False)
    return sub


def make_submission(model, cfg: dict, test_df: pd.DataFrame, save_path: str | Path) -> pd.DataFrame:
    """Memprediksi data test dan menyimpan hasilnya dalam format submission.

    Args:
        model: Model terlatih.
        cfg: Konfigurasi yang dipakai saat training (untuk ukuran gambar dan batch).
        test_df: DataFrame test dengan kolom `id` dan `path`.
        save_path: Lokasi file CSV submission.

    Returns:
        DataFrame submission dengan kolom `id` dan `label`.
    """
    return save_submission(predict_test(model, cfg, test_df), test_df, save_path)


def validate_submission(sub: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Memastikan format submission sesuai ketentuan kompetisi.

    Args:
        sub: DataFrame submission.
        test_df: DataFrame test asli.

    Raises:
        AssertionError: Jika format tidak sesuai.
    """
    assert list(sub.columns) == ["id", "label"], "Kolom harus id,label"
    assert len(sub) == len(test_df), "Jumlah baris tidak sama dengan test.csv"
    assert sub["id"].tolist() == test_df["id"].tolist(), "Urutan id berbeda dengan test.csv"
    assert sub["label"].between(0, 15).all(), "Label harus 0-15"


def main() -> None:
    """Membuat submission dari terminal menggunakan checkpoint yang sudah dilatih."""
    parser = argparse.ArgumentParser(description="Prediksi data test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/submissions/submission.csv")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    model, cfg = load_checkpoint(args.checkpoint, get_device())
    _, test_df = load_metadata(resolve_data_dir(args.data_dir))
    sub = make_submission(model, cfg, test_df, args.output)
    validate_submission(sub, test_df)
    print(f"Submission disimpan ke {args.output} ({len(sub)} baris)")


if __name__ == "__main__":
    main()
