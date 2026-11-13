"""Efficiency metrics: memory-bank footprint, feature dimensionality, throughput.

Throughput rules (so numbers are comparable across runs):
* time on the same GPU type (Colab assigns GPUs dynamically – record ``gpu_name``);
* warm up first, then ``torch.cuda.synchronize()`` around every timed region;
* report per-stage time (backbone / feature post-processing / kNN / map) and end to end.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .features import embed_batch


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def memory_bank_report(patchcore) -> dict:
    """Storage footprint of a fitted PatchCore memory bank (plus any reducer matrix)."""
    bank = patchcore.memory_bank
    n, d = bank.shape
    reducer_bytes = sum(tensor_bytes(getattr(patchcore.reducer, a))
                        for a in ("P", "components", "mean") if hasattr(patchcore.reducer, a))
    return {
        "bank_patches": n,
        "bank_dim": d,
        "bank_dtype": str(bank.dtype).replace("torch.", ""),
        "bank_mb": tensor_bytes(bank) / 2**20,
        "bank_mb_fp32": n * d * 4 / 2**20,
        "reducer_mb": reducer_bytes / 2**20,
        "bytes_per_patch": d * bank.element_size(),
        "train_patches": patchcore.fit_stats.get("n_train_patches"),
        "keep_ratio": n / max(1, patchcore.fit_stats.get("n_train_patches", n)),
    }


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_pipeline(extractor, patchcore, images: torch.Tensor, agg_kernel: int = 3,
                       n_warmup: int = 3, n_iters: int = 10, amp: bool = True) -> dict:
    """Time test-time inference on one batch of images, stage by stage.

    Args:
        images: (B, 3, S, S) normalised batch (moved to the PatchCore device).
    Returns per-image milliseconds for each stage, end-to-end FPS, peak GPU memory.
    """
    device = patchcore.device
    x = images.to(device)
    B, S = x.shape[0], x.shape[-1]
    stages = {"backbone_ms": [], "postproc_ms": [], "knn_ms": [], "map_ms": []}

    from .features import fuse_layers, local_aggregate

    def one_pass(record: bool):
        _sync(device); t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and x.is_cuda):
            fmaps = extractor(x)
        _sync(device); t1 = time.perf_counter()
        feats = fuse_layers({k: local_aggregate(v.float(), agg_kernel) for k, v in fmaps.items()})
        _sync(device); t2 = time.perf_counter()
        ps = patchcore.score_patches(feats)
        _sync(device); t3 = time.perf_counter()
        patchcore.anomaly_map(ps, S)
        ps.flatten(1).max(1)
        _sync(device); t4 = time.perf_counter()
        if record:
            for key, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                stages[key].append(dt * 1000 / B)

    for _ in range(n_warmup):
        one_pass(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(n_iters):
        one_pass(True)

    out = {k: float(np.median(v)) for k, v in stages.items()}
    out["total_ms"] = sum(out.values())
    out["fps"] = 1000.0 / out["total_ms"]
    out["batch_size"] = B
    out["gpu"] = gpu_name()
    out["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda"
                          else float("nan"))
    out["compute_fraction"] = getattr(extractor, "compute_fraction", float("nan"))
    return out


@torch.no_grad()
def benchmark_backbone_only(extractor, images: torch.Tensor, n_warmup: int = 3,
                            n_iters: int = 10, amp: bool = True) -> dict:
    """Backbone-only latency (isolates the effect of layer depth / truncation)."""
    device = images.device
    for _ in range(n_warmup):
        embed_batch(extractor, images, 1, amp)
    times = []
    for _ in range(n_iters):
        _sync(device); t0 = time.perf_counter()
        embed_batch(extractor, images, 1, amp)
        _sync(device); times.append(time.perf_counter() - t0)
    ms = float(np.median(times)) * 1000 / images.shape[0]
    return {"backbone_ms": ms, "backbone_fps": 1000 / ms, "gpu": gpu_name()}
