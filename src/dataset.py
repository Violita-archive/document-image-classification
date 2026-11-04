"""Pembacaan metadata, pembersihan data, pembagian train/validation, dan Dataset PyTorch."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.utils import CLASS_NAMES


def resolve_image_path(data_dir: Path, filename: str) -> Path:
    """Mencari lokasi file gambar dari nilai kolom `filename` pada CSV.

    Hasil ekstrak Kaggle dapat berbentuk `train/<kelas>/x.tif` atau
    `train/train/<kelas>/x.tif`, sehingga kedua kemungkinan diperiksa.

    Args:
        data_dir: Folder data utama.
        filename: Path relatif dari CSV, misalnya `train/resume/123.tif`.

    Returns:
        Path absolut file gambar.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
    """
    rel = Path(filename.replace("\\", "/"))
    for cand in (data_dir / rel, data_dir / rel.parts[0] / rel):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"File tidak ditemukan: {filename}")


def load_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membaca `train.csv` dan `test.csv` lalu menambahkan kolom path gambar.

    Args:
        data_dir: Folder data utama.

    Returns:
        Tuple (train_df, test_df). train_df memiliki kolom `id`, `filename`,
        `label`, `class_name`, dan `path`; test_df memiliki `id`, `filename`, dan `path`.
    """
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df["path"] = [resolve_image_path(data_dir, f) for f in train_df["filename"]]
    test_df["path"] = [resolve_image_path(data_dir, f) for f in test_df["filename"]]
    if "class_name" not in train_df.columns:
        train_df["class_name"] = train_df["label"].map(lambda i: CLASS_NAMES[i])
    return train_df, test_df


def _file_info(path: Path) -> tuple[bool, str]:
    """Memeriksa apakah gambar dapat dibuka dan menghitung hash MD5 isinya.

    Args:
        path: Path gambar.

    Returns:
        Tuple (dapat_dibuka, md5).
    """
    h = hashlib.md5(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as im:
            im.size
        return True, h
    except Exception:
        return False, h


def clean_train_data(train_df: pd.DataFrame, n_workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """Membersihkan data train berdasarkan temuan EDA.

    Langkah: (1) membuang file yang tidak dapat dibuka, (2) membuang seluruh grup
    gambar identik yang memiliki label berbeda, (3) menyisakan satu file dari
    setiap grup gambar identik berlabel sama.

    Args:
        train_df: DataFrame hasil `load_metadata`.
        n_workers: Jumlah thread untuk membaca file.

    Returns:
        Tuple (DataFrame bersih, ringkasan jumlah data yang dibuang).
    """
    with ThreadPoolExecutor(max(1, n_workers)) as ex:
        info = list(tqdm(ex.map(_file_info, train_df["path"]), total=len(train_df), desc="Cek file"))
    df = train_df.copy()
    df["readable"] = [i[0] for i in info]
    df["md5"] = [i[1] for i in info]

    n0 = len(df)
    df = df[df["readable"]]
    n_unreadable = n0 - len(df)

    n_label = df.groupby("md5")["label"].transform("nunique")
    n_conflict = int((n_label > 1).sum())
    df = df[n_label == 1]

    n1 = len(df)
    df = df.drop_duplicates("md5", keep="first")
    n_dup = n1 - len(df)

    summary = {
        "awal": n0,
        "tidak_terbaca": n_unreadable,
        "duplikat_label_bertentangan": n_conflict,
        "duplikat_label_sama": n_dup,
        "akhir": len(df),
    }
    return df.drop(columns=["readable"]).reset_index(drop=True), summary


def split_train_val(df: pd.DataFrame, val_size: float = 0.1, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membagi data menjadi train dan validation secara stratified.

    Args:
        df: DataFrame data bersih.
        val_size: Proporsi data validation.
        seed: Seed pembagian.

    Returns:
        Tuple (train_df, val_df).
    """
    tr, va = train_test_split(df, test_size=val_size, stratify=df["label"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


class DocumentDataset(Dataset):
    """Dataset citra dokumen untuk PyTorch.

    Gambar grayscale dibaca lalu dikonversi ke 3 kanal agar sesuai dengan model
    pre-trained ImageNet.

    Args:
        df: DataFrame dengan kolom `path` dan (opsional) `label`.
        transform: Transformasi torchvision yang diterapkan pada gambar.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """Menyimpan daftar path, label, dan transformasi.

        Args:
            df: DataFrame dengan kolom `path` dan (opsional) `label`.
            transform: Transformasi torchvision yang diterapkan pada gambar.
        """
        self.paths = [str(p) for p in df["path"]]
        self.labels = df["label"].tolist() if "label" in df.columns else [-1] * len(df)
        self.transform = transform

    def __len__(self) -> int:
        """Mengembalikan jumlah sampel."""
        return len(self.paths)

    def __getitem__(self, idx: int):
        """Mengambil satu sampel.

        Args:
            idx: Indeks sampel.

        Returns:
            Tuple (tensor gambar, label).
        """
        with Image.open(self.paths[idx]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def make_loader(df: pd.DataFrame, transform, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    """Membuat DataLoader dari DataFrame.

    Args:
        df: DataFrame dengan kolom `path` dan (opsional) `label`.
        transform: Transformasi gambar.
        batch_size: Ukuran batch.
        shuffle: Apakah data diacak setiap epoch.
        num_workers: Jumlah proses pemuat data.

    Returns:
        Objek DataLoader.
    """
    return DataLoader(
        DocumentDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
