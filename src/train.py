"""Loop training dan validasi (`loop_fn`) serta fungsi `fit` untuk menjalankan satu eksperimen.

Dapat dijalankan dari terminal:
    python -m src.train --config configs/config.yaml
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from src.callback import Callback
from src.dataset import clean_train_data, load_metadata, make_loader, split_train_val
from src.model import count_parameters, model_from_config
from src.transforms import get_eval_transforms, get_train_transforms
from src.utils import get_device, load_config, resolve_data_dir, set_seed


def build_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Membuat scheduler learning rate: warmup linear lalu cosine decay.

    Args:
        optimizer: Optimizer PyTorch.
        total_steps: Jumlah total langkah (batch) selama training.
        warmup_steps: Jumlah langkah warmup.

    Returns:
        Objek LambdaLR.
    """
    def lr_lambda(step):
        """Menghitung faktor pengali learning rate pada langkah tertentu.

        Args:
            step: Indeks langkah training.

        Returns:
            Faktor pengali learning rate (0-1).
        """
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def loop_fn(mode: str, dataloader, model, criterion, optimizer, device, scheduler=None, scaler=None,
            use_amp: bool = False) -> tuple[float, float, np.ndarray | None]:
    """Menjalankan satu epoch training atau evaluasi (mengikuti pola `loop_fn` pada praktikum).

    Pada mode `train`, bobot model diperbarui setiap batch dan scheduler dijalankan per batch.
    Pada mode `test`, fungsi ini dipanggil di dalam `torch.no_grad()` dan probabilitas
    prediksi dikembalikan.

    Args:
        mode: `train` atau `test`.
        dataloader: DataLoader yang digunakan.
        model: Model PyTorch.
        criterion: Fungsi loss.
        optimizer: Optimizer.
        device: Perangkat komputasi.
        scheduler: Scheduler learning rate (hanya mode train).
        scaler: GradScaler untuk mixed precision (hanya mode train).
        use_amp: Menggunakan mixed precision jika True.

    Returns:
        Tuple (rata-rata loss, akurasi, probabilitas prediksi). Probabilitas bernilai None pada mode train.
    """
    model.train() if mode == "train" else model.eval()
    cost, correct, n, probs = 0.0, 0, 0, []
    for feature, target in tqdm(dataloader, desc=mode, leave=False):
        feature, target = feature.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(feature)
            loss = criterion(output, target)
        if mode == "train":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            probs.append(output.float().softmax(1).cpu().numpy())
        cost += loss.item() * target.size(0)
        correct += (output.argmax(1) == target).sum().item()
        n += target.size(0)
    return cost / n, correct / n, (np.concatenate(probs) if probs else None)


def fit(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, exp_name: str, root: str | Path = ".") -> dict:
    """Menjalankan seluruh tahapan training untuk satu eksperimen.

    Tahapan sama dengan yang ditulis langsung pada `02_training.ipynb`: DataLoader,
    MCOC (Model, Criterion, Optimizer, Callback), lalu loop training dengan early stopping.

    Args:
        cfg: Konfigurasi eksperimen.
        train_df: Data latih.
        val_df: Data validation.
        exp_name: Nama eksperimen, dipakai untuk penamaan file.
        root: Folder utama repository.

    Returns:
        Ringkasan hasil dari `Callback.summary()`.
    """
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    trainloader = make_loader(train_df, get_train_transforms(cfg["img_size"]), cfg["batch_size"], True, cfg["num_workers"])
    testloader = make_loader(val_df, get_eval_transforms(cfg["img_size"]), cfg["batch_size"] * 2, False, cfg["num_workers"])

    model = model_from_config(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg["epochs"] * len(trainloader), int(cfg.get("warmup_epochs", 1) * len(trainloader)))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    callback = Callback(model, cfg, exp_name, patience=cfg.get("patience", 99), root=root)
    print(f"{exp_name}: {cfg['model_name']} | {count_parameters(model) / 1e6:.1f} juta parameter | "
          f"img {cfg['img_size']} | device {device} | AMP {use_amp}")

    for _ in range(cfg["epochs"]):
        train_cost, train_score, _ = loop_fn("train", trainloader, model, criterion, optimizer, device, scheduler, scaler, use_amp)
        with torch.no_grad():
            test_cost, test_score, test_probs = loop_fn("test", testloader, model, criterion, optimizer, device, use_amp=use_amp)
        callback.log(train_cost, test_cost, train_score, test_score, optimizer.param_groups[0]["lr"])
        if callback.early_stopping(model, test_score, test_probs):
            break
    return callback.summary()


def main() -> None:
    """Menjalankan training dari terminal menggunakan file konfigurasi."""
    parser = argparse.ArgumentParser(description="Training klasifikasi citra dokumen")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--name", default=None, help="Nama eksperimen (default: nama model)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve_data_dir(cfg["data_dir"])
    train_df, _ = load_metadata(data_dir)
    clean_df, summary = clean_train_data(train_df, cfg["num_workers"])
    print("Pembersihan data:", summary)
    tr, va = split_train_val(clean_df, cfg["val_size"], cfg["seed"])
    out = fit(cfg, tr, va, args.name or cfg["model_name"])
    print(f"Akurasi validation terbaik {out['best_acc']:.4f} (epoch {out['best_epoch']})")


if __name__ == "__main__":
    main()
