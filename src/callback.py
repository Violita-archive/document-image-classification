"""Callback training: pencatatan per epoch, penyimpanan model terbaik, dan early stopping."""

import time
from pathlib import Path

import pandas as pd
import torch

from src.utils import ensure_dirs, plot_history


class Callback:
    """Mencatat hasil tiap epoch, menyimpan model terbaik, dan menghentikan training lebih awal.

    Mengikuti pola `Callback` pada praktikum. Model terbaik dipilih berdasarkan akurasi
    validation dan disimpan ke `checkpoint/<exp_name>_best.pth` bersama konfigurasinya,
    sedangkan riwayat per epoch disimpan ke `results/metrics/<exp_name>_history.csv`.

    Args:
        model: Model yang dilatih.
        config: Konfigurasi eksperimen.
        exp_name: Nama eksperimen untuk penamaan file.
        patience: Jumlah epoch tanpa peningkatan sebelum training dihentikan.
        root: Folder utama repository.
    """

    def __init__(self, model, config: dict, exp_name: str, patience: int = 3, root: str | Path = "."):
        """Menyiapkan folder output dan variabel pemantauan.

        Args:
            model: Model yang dilatih.
            config: Konfigurasi eksperimen.
            exp_name: Nama eksperimen untuk penamaan file.
            patience: Jumlah epoch tanpa peningkatan sebelum training dihentikan.
            root: Folder utama repository.
        """
        root = Path(root)
        self.model = model
        self.config = config
        self.exp_name = exp_name
        self.patience = patience
        self.ckpt_path = root / "checkpoint" / f"{exp_name}_best.pth"
        self.log_path = root / "results" / "metrics" / f"{exp_name}_history.csv"
        ensure_dirs(self.ckpt_path.parent, self.log_path.parent)

        self.records = []
        self.best_score = -1.0
        self.best_epoch = 0
        self.best_probs = None
        self.counter = 0
        self.start = time.time()
        self._t0 = time.time()

    @property
    def epochs(self) -> int:
        """Jumlah epoch yang sudah dicatat."""
        return len(self.records)

    @property
    def history(self) -> pd.DataFrame:
        """Riwayat training dalam bentuk DataFrame."""
        return pd.DataFrame(self.records)

    def log(self, train_cost: float, test_cost: float, train_score: float, test_score: float, lr: float) -> None:
        """Mencatat hasil satu epoch dan menyimpannya ke file CSV.

        Args:
            train_cost: Rata-rata loss data latih.
            test_cost: Rata-rata loss data validation.
            train_score: Akurasi data latih.
            test_score: Akurasi data validation.
            lr: Learning rate di akhir epoch.
        """
        self.records.append({"epoch": self.epochs + 1, "train_loss": train_cost, "train_acc": train_score,
                             "val_loss": test_cost, "val_acc": test_score, "lr": lr,
                             "detik": time.time() - self._t0})
        self.history.to_csv(self.log_path, index=False)

    def early_stopping(self, model, test_score: float, test_probs=None) -> bool:
        """Menyimpan model jika akurasi validation membaik dan memeriksa kondisi berhenti.

        Args:
            model: Model yang sedang dilatih.
            test_score: Akurasi validation epoch terakhir.
            test_probs: Probabilitas prediksi validation epoch terakhir.

        Returns:
            True jika training harus dihentikan.
        """
        r = self.records[-1]
        improved = test_score > self.best_score
        if improved:
            self.best_score, self.best_epoch, self.best_probs, self.counter = test_score, r["epoch"], test_probs, 0
            torch.save({"model_state": model.state_dict(), "config": self.config,
                        "epoch": r["epoch"], "val_acc": test_score}, self.ckpt_path)
        else:
            self.counter += 1
        print(f"epoch {r['epoch']:2d} | train loss {r['train_loss']:.4f} acc {r['train_acc']:.4f} | "
              f"val loss {r['val_loss']:.4f} acc {r['val_acc']:.4f} | {r['detik']:.0f}s{' *' if improved else ''}")
        self._t0 = time.time()
        if self.counter >= self.patience:
            print(f"Early stopping pada epoch {r['epoch']}.")
            return True
        return False

    def plot_final(self, title: str, save_path: str | Path | None = None) -> None:
        """Menampilkan kurva loss dan akurasi train vs validation.

        Args:
            title: Judul grafik.
            save_path: Lokasi penyimpanan gambar.
        """
        plot_history(self.history, title, save_path)

    def summary(self) -> dict:
        """Merangkum hasil training.

        Returns:
            Dictionary berisi riwayat, akurasi dan epoch terbaik, probabilitas validation
            terbaik, durasi (menit), dan path checkpoint.
        """
        return {
            "history": self.history,
            "best_acc": self.best_score,
            "best_epoch": self.best_epoch,
            "val_probs": self.best_probs,
            "minutes": (time.time() - self.start) / 60,
            "ckpt_path": self.ckpt_path,
        }
