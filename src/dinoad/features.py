"""Turn backbone outputs into PatchCore patch features, and cache them to disk.

Pipeline per image batch (:func:`embed_batch`)::

    extractor(x) -> {layer: (B, C_l, h_l, w_l)}
        -> local_aggregate (3x3 neighbourhood average, per layer)
        -> fuse_layers (resize to the finest grid, concatenate channels)
        -> (B, C, h, w)

Feature caching matters on Colab: extract once per (backbone, layer, resolution,
category), save fp16 tensors to Drive, and every later sweep (coreset ratio,
projection dim, ...) just loads them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def local_aggregate(fmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """PatchCore's locally-aware patch features: average each patch with its
    ``k x k`` neighbourhood (stride 1, same spatial size). ``kernel_size=1`` disables it."""
    if kernel_size <= 1:
        return fmap
    return F.avg_pool2d(fmap, kernel_size, stride=1, padding=kernel_size // 2,
                        count_include_pad=False)


def fuse_layers(fmaps: dict, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Concatenate per-layer maps along channels after resizing to a common grid
    (default: the largest grid, e.g. ResNet layer2). For ViT layers the grids
    already match, so this is just a concatenation."""
    maps = [fmaps[k] for k in sorted(fmaps, key=str)]
    if size is None:
        size = max((m.shape[-2:] for m in maps), key=lambda s: s[0] * s[1])
    maps = [m if tuple(m.shape[-2:]) == tuple(size)
            else F.interpolate(m, size=size, mode="bilinear", align_corners=False) for m in maps]
    return torch.cat(maps, dim=1)


def to_patches(fmap: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B*h*w, C) patch matrix."""
    return fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])


@torch.no_grad()
def embed_batch(extractor: torch.nn.Module, x: torch.Tensor, agg_kernel: int = 3,
                amp: bool = True) -> torch.Tensor:
    """Images (B, 3, H, W) on device -> fused patch features (B, C, h, w), float32."""
    use_amp = amp and x.is_cuda
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        fmaps = extractor(x)
    fmaps = {k: local_aggregate(v.float(), agg_kernel) for k, v in fmaps.items()}
    return fuse_layers(fmaps)


@dataclass
class FeatureSet:
    """Features for one split of one category (kept on CPU)."""
    features: torch.Tensor                    # (N, C, h, w), usually fp16
    labels: torch.Tensor                      # (N,) 0/1
    masks: torch.Tensor | None = None         # (N, H, W) bool (test only)
    paths: list[str] = field(default_factory=list)
    defect_types: list[str] = field(default_factory=list)
    extract_seconds: float = 0.0

    @property
    def grid(self) -> tuple[int, int]:
        return tuple(self.features.shape[-2:])

    @property
    def dim(self) -> int:
        return self.features.shape[1]


@torch.no_grad()
def extract_features(extractor: torch.nn.Module, loader, device: torch.device,
                     agg_kernel: int = 3, amp: bool = True, keep_masks: bool = True,
                     out_dtype: torch.dtype = torch.float16, desc: str = "extract") -> FeatureSet:
    """Run the extractor over a DataLoader of :class:`~dinoad.data.MVTecAD` items."""
    import time
    feats, labels, masks, paths, defects = [], [], [], [], []
    t0 = time.perf_counter()
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device, non_blocking=True)
        feats.append(embed_batch(extractor, x, agg_kernel, amp).to("cpu", out_dtype))
        labels.append(torch.as_tensor(batch["label"]))
        if keep_masks:
            masks.append(batch["mask"][:, 0].bool())
        paths += list(batch["path"])
        defects += list(batch["defect_type"])
    return FeatureSet(torch.cat(feats), torch.cat(labels),
                      torch.cat(masks) if keep_masks else None, paths, defects,
                      time.perf_counter() - t0)


def feature_cache_path(cache_dir: str | Path, backbone: str, image_size: int, layers,
                       category: str, split: str, agg_kernel: int = 3,
                       apply_norm: bool = False) -> Path:
    """Deterministic cache filename, e.g.
    ``cache/dinov2_vitb14/448/carpet/train_L6-9_k3.pt``."""
    tag = "-".join(str(l).replace("layer", "") for l in layers)
    norm = "_norm" if apply_norm else ""
    return Path(cache_dir) / backbone / str(image_size) / category / f"{split}_L{tag}_k{agg_kernel}{norm}.pt"


def save_features(fs: FeatureSet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fs.__dict__, path)


def load_features(path: str | Path) -> FeatureSet:
    return FeatureSet(**torch.load(path, map_location="cpu", weights_only=False))


def get_or_extract(cache_path: str | Path | None, extractor, loader, device, **kwargs) -> FeatureSet:
    """Load cached features if present, otherwise extract (and save if a path is given)."""
    if cache_path is not None and Path(cache_path).exists():
        return load_features(cache_path)
    fs = extract_features(extractor, loader, device, **kwargs)
    if cache_path is not None:
        save_features(fs, cache_path)
    return fs
