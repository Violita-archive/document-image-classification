"""Transformasi dan augmentasi gambar."""

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_hw(img_size) -> tuple[int, int]:
    """Mengubah ukuran input menjadi pasangan (tinggi, lebar).

    Args:
        img_size: Bilangan bulat untuk ukuran persegi, atau list/tuple [tinggi, lebar].

    Returns:
        Tuple (tinggi, lebar).
    """
    if isinstance(img_size, (list, tuple)):
        return int(img_size[0]), int(img_size[1])
    return int(img_size), int(img_size)


def get_train_transforms(img_size) -> T.Compose:
    """Transformasi untuk data latih dengan augmentasi ringan.

    Augmentasi meniru variasi hasil scan: rotasi kecil, pergeseran, skala, serta
    perubahan kecerahan dan kontras. Flip tidak digunakan karena dokumen yang
    terbalik tidak realistis.

    Args:
        img_size: Ukuran gambar setelah resize, berupa bilangan bulat (persegi) atau [tinggi, lebar].

    Returns:
        Objek `transforms.Compose`.
    """
    return T.Compose([
        T.Resize(to_hw(img_size), antialias=True),
        T.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms(img_size) -> T.Compose:
    """Transformasi untuk data validation dan test (tanpa augmentasi).

    Args:
        img_size: Ukuran gambar setelah resize, berupa bilangan bulat (persegi) atau [tinggi, lebar].

    Returns:
        Objek `transforms.Compose`.
    """
    return T.Compose([
        T.Resize(to_hw(img_size), antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Mengembalikan tensor ternormalisasi ke rentang 0-1 untuk ditampilkan.

    Args:
        tensor: Tensor gambar berukuran (3, H, W).

    Returns:
        Tensor berukuran (H, W, 3) dengan nilai 0-1.
    """
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)
