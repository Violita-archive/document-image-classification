"""Loop training, validasi, dan penyimpanan model terbaik.

Dapat dijalankan dari terminal:
    python -m src.train --config configs/config.yaml
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from src.dataset import clean_train_data, load_metadata, make_loader, split_train_val
from src.model import build_model, count_parameters
from src.transforms import get_eval_transforms, get_train_transforms
from src.utils import ensure_dirs, get_device, load_config, resolve_data_dir, set_seed


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


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp: bool) -> tuple[float, float]:
    """Melatih model selama satu epoch.

    Args:
        model: Model PyTorch.
        loader: DataLoader data latih.
        criterion: Fungsi loss.
        optimizer: Optimizer.
        scheduler: Scheduler learning rate (diperbarui setiap batch).
        scaler: GradScaler untuk mixed precision.
        device: Perangkat komputasi.
        use_amp: Menggunakan mixed precision jika True.

    Returns:
        Tuple (rata-rata loss, akurasi) pada data latih.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp: bool) -> tuple[float, float, np.ndarray]:
    """Mengevaluasi model pada data validation.

    Args:
        model: Model PyTorch.
        loader: DataLoader data validation.
        criterion: Fungsi loss.
        device: Perangkat komputasi.
        use_amp: Menggunakan mixed precision jika True.

    Returns:
        Tuple (rata-rata loss, akurasi, probabilitas prediksi berukuran (N, kelas)).
    """
    model.eval()
    total_loss, correct, n, probs = 0.0, 0, 0, []
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
        probs.append(logits.float().softmax(1).cpu().numpy())
    return total_loss / n, correct / n, np.concatenate(probs)


def fit(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, exp_name: str, root: str | Path = ".") -> dict:
    """Menjalankan training lengkap untuk satu eksperimen.

    Model dengan akurasi validation tertinggi disimpan ke `checkpoint/<exp_name>_best.pth`
    dan riwayat per epoch disimpan ke `results/metrics/<exp_name>_history.csv`.

    Args:
        cfg: Konfigurasi eksperimen.
        train_df: Data latih.
        val_df: Data validation.
        exp_name: Nama eksperimen, dipakai untuk penamaan file.
        root: Folder utama repository.

    Returns:
        Dictionary berisi riwayat training, akurasi terbaik, epoch terbaik,
        probabilitas validation terbaik, durasi, dan path checkpoint.
    """
    root = Path(root)
    ckpt_dir, metric_dir = root / "checkpoint", root / "results" / "metrics"
    ensure_dirs(ckpt_dir, metric_dir)
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    train_loader = make_loader(train_df, get_train_transforms(cfg["img_size"]), cfg["batch_size"], True, cfg["num_workers"])
    val_loader = make_loader(val_df, get_eval_transforms(cfg["img_size"]), cfg["batch_size"] * 2, False, cfg["num_workers"])

    model = build_model(cfg["model_name"], cfg["num_classes"], cfg.get("pretrained", True), cfg.get("drop_rate", 0.0),
                        cfg.get("drop_path_rate", 0.0)).to(device)
    print(f"{exp_name}: {cfg['model_name']} | {count_parameters(model) / 1e6:.1f} juta parameter | "
          f"img {cfg['img_size']} | device {device} | AMP {use_amp}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, int(cfg.get("warmup_epochs", 1) * len(train_loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_path = ckpt_dir / f"{exp_name}_best.pth"
    history, best_acc, best_epoch, best_probs, wait = [], -1.0, 0, None, 0
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, use_amp)
        va_loss, va_acc, va_probs = validate(model, val_loader, criterion, device, use_amp)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                        "val_acc": va_acc, "lr": optimizer.param_groups[0]["lr"], "detik": time.time() - t0})
        flag = ""
        if va_acc > best_acc:
            best_acc, best_epoch, best_probs, wait = va_acc, epoch, va_probs, 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch, "val_acc": va_acc}, ckpt_path)
            flag = " *"
        else:
            wait += 1
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s{flag}")
        if wait >= cfg.get("patience", 99):
            print(f"Early stopping pada epoch {epoch}.")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(metric_dir / f"{exp_name}_history.csv", index=False)
    return {
        "history": hist_df,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "val_probs": best_probs,
        "minutes": (time.time() - start) / 60,
        "ckpt_path": ckpt_path,
    }


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
