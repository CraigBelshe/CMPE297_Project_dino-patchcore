"""MVTec AD loading.

Expected layout (the official MVTec AD release, extracted)::

    <root>/
      carpet/
        train/good/000.png ...
        test/good/000.png ...
        test/<defect_type>/000.png ...
        ground_truth/<defect_type>/000_mask.png ...
      bottle/ ...

`root` can be a Google Drive folder, but reading thousands of PNGs from Drive is
slow, so use :func:`copy_to_local` once per Colab session to copy the categories
you need onto the VM's local disk (``/content``).
"""
from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# The project's 6-category subset, split by structural taxonomy.
TEXTURE_CATEGORIES = ["carpet", "tile", "wood"]
OBJECT_CATEGORIES = ["bottle", "cable", "transistor"]
DOMAINS = {"texture": TEXTURE_CATEGORIES, "object": OBJECT_CATEGORIES}

# Full MVTec AD taxonomy, in case you want to extend the study later.
ALL_TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
ALL_OBJECTS = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill",
               "screw", "toothbrush", "transistor", "zipper"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def get_categories(domain: str) -> list[str]:
    """Return the categories for ``"texture"``, ``"object"`` or ``"all"`` (both)."""
    domain = domain.lower()
    if domain == "all":
        return TEXTURE_CATEGORIES + OBJECT_CATEGORIES
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {list(DOMAINS) + ['all']}, got {domain!r}")
    return list(DOMAINS[domain])


def domain_of(category: str) -> str:
    """Map a category name to ``"texture"`` or ``"object"``."""
    if category in ALL_TEXTURES:
        return "texture"
    if category in ALL_OBJECTS:
        return "object"
    return "unknown"


@dataclass
class Sample:
    image_path: Path
    mask_path: Path | None
    label: int          # 0 = good, 1 = anomalous
    defect_type: str


class MVTecAD(Dataset):
    """One MVTec AD category, train or test split.

    Each item is a dict with:
      * ``image``  – float tensor (3, S, S), ImageNet-normalised
      * ``mask``   – float tensor (1, S, S) in {0, 1} (all zeros for good images)
      * ``label``  – int, 0 good / 1 anomalous
      * ``defect_type`` – str (``"good"`` for normal images)
      * ``path``   – str path to the source image

    Args:
        root: dataset root containing one folder per category.
        category: e.g. ``"carpet"``.
        split: ``"train"`` (only good images) or ``"test"``.
        image_size: images are resized to ``image_size x image_size``. For a
            ViT this must be a multiple of the patch size (14 for DINOv2:
            224 -> 16x16 patches, 448 -> 32x32 patches).
    """

    def __init__(self, root: str | Path, category: str, split: str = "train",
                 image_size: int = 448):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.samples = self._index()
        self.image_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR,
                     antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def _index(self) -> list[Sample]:
        split_dir = self.root / self.category / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split_dir} not found. Check that `root` points at the folder that "
                f"contains '{self.category}/'.")
        samples = []
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect = defect_dir.name
            for img in sorted(p for p in defect_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
                if defect == "good":
                    samples.append(Sample(img, None, 0, defect))
                else:
                    mask = (self.root / self.category / "ground_truth" / defect
                            / f"{img.stem}_mask.png")
                    samples.append(Sample(img, mask if mask.exists() else None, 1, defect))
        if not samples:
            raise RuntimeError(f"No images found under {split_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = self.image_tf(Image.open(s.image_path).convert("RGB"))
        if s.mask_path is not None:
            mask = (self.mask_tf(Image.open(s.mask_path).convert("L")) > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        return {"image": image, "mask": mask, "label": s.label,
                "defect_type": s.defect_type, "path": str(s.image_path)}


def make_loader(dataset: Dataset, batch_size: int = 16, num_workers: int = 2,
                shuffle: bool = False) -> DataLoader:
    """Standard DataLoader (no shuffling: order is kept so scores line up with paths)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def verify_mvtec_root(root: str | Path, categories: list[str]) -> dict[str, dict[str, int]]:
    """Count images per category/split and fail early with a clear message.

    Returns ``{category: {"train": n, "test_good": n, "test_anomalous": n}}``.
    """
    root = Path(root)
    report = {}
    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"Missing category folder: {cat_dir}")
        train = len(MVTecAD(root, cat, "train", 32))
        test = MVTecAD(root, cat, "test", 32)
        n_bad = sum(s.label for s in test.samples)
        n_missing_masks = sum(1 for s in test.samples if s.label == 1 and s.mask_path is None)
        if n_missing_masks:
            print(f"[warn] {cat}: {n_missing_masks} anomalous test images have no mask")
        report[cat] = {"train": train, "test_good": len(test) - n_bad, "test_anomalous": n_bad}
    return report


def copy_to_local(src_root: str | Path, dst_root: str | Path, categories: list[str]) -> Path:
    """Copy categories from (slow) Google Drive to (fast) local disk.

    For each category this accepts either an extracted folder ``src_root/<cat>/`` or
    a per-category archive ``src_root/<cat>.tar.xz`` (the format MVTec distributes).
    Already-copied categories are skipped, so it is safe to re-run.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        dst = dst_root / cat
        if (dst / "train").is_dir():
            print(f"{cat}: already local")
            continue
        folder = src_root / cat
        archive = next((src_root / f"{cat}{ext}" for ext in (".tar.xz", ".tar.gz", ".tar")
                        if (src_root / f"{cat}{ext}").exists()), None)
        if folder.is_dir():
            print(f"{cat}: copying folder ...")
            shutil.copytree(folder, dst, dirs_exist_ok=True)
        elif archive is not None:
            print(f"{cat}: extracting {archive.name} ...")
            with tarfile.open(archive) as tf:
                tf.extractall(dst_root, filter="data")
        else:
            raise FileNotFoundError(f"Neither {folder} nor {cat}.tar.xz found in {src_root}")
    return dst_root
