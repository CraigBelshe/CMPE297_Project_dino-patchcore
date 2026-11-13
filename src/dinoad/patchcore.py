"""PatchCore: memory bank, coreset compression, nearest-neighbour scoring.

Memory-bank size = (#patches kept) x (feature dim) x (bytes per value), so the
three knobs that shrink storage are:

1. ``coreset_ratio``  – greedy k-center coreset keeps 10% / 5% / 1% of patches;
2. ``reducer`` + ``reducer_dim`` – random projection or PCA to fewer dims;
3. ``bank_dtype``     – fp16 halves fp32 storage.

Choosing a shallower ViT layer does *not* change the dimension (all ViT-B blocks
emit 768-d tokens); it changes accuracy and compute (see backbones.ViTLayerExtractor).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from .features import to_patches


# --------------------------------------------------------------------------- #
# Dimension reduction
# --------------------------------------------------------------------------- #
def _chunked_matmul(X: torch.Tensor, P: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    return torch.cat([X[i:i + chunk].float() @ P for i in range(0, X.shape[0], chunk)])


class Reducer:
    """Base class: identity."""
    out_dim: int | None = None

    def fit(self, X: torch.Tensor) -> "Reducer":
        self.out_dim = X.shape[1]
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return X.float()


class RandomProjection(Reducer):
    """Gaussian random projection (Johnson-Lindenstrauss). Data-independent and free to fit."""

    def __init__(self, out_dim: int, seed: int = 0):
        self.out_dim, self.seed = out_dim, seed

    def fit(self, X):
        g = torch.Generator().manual_seed(self.seed)
        self.P = (torch.randn(X.shape[1], self.out_dim, generator=g)
                  / math.sqrt(self.out_dim)).to(X.device)
        return self

    def transform(self, X):
        return _chunked_matmul(X, self.P)


class PCAReducer(Reducer):
    """PCA fitted on (a subsample of) the normal training patches."""

    def __init__(self, out_dim: int, max_fit_samples: int = 100_000, seed: int = 0):
        self.out_dim, self.max_fit, self.seed = out_dim, max_fit_samples, seed

    def fit(self, X):
        g = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(X.shape[0], generator=g)[: self.max_fit].to(X.device)
        Xs = X[idx].float()
        self.mean = Xs.mean(0, keepdim=True)
        _, S, V = torch.pca_lowrank(Xs - self.mean, q=self.out_dim, center=False, niter=4)
        self.components = V[:, : self.out_dim]
        self.explained_variance = (S[: self.out_dim] ** 2 / (Xs.shape[0] - 1)).cpu()
        return self

    def transform(self, X):
        return _chunked_matmul(X, self.components) - self.mean @ self.components


def make_reducer(kind: str, out_dim: int | None, seed: int = 0) -> Reducer:
    if kind in (None, "none") or out_dim is None:
        return Reducer()
    if kind == "random":
        return RandomProjection(out_dim, seed)
    if kind == "pca":
        return PCAReducer(out_dim, seed=seed)
    raise ValueError("reducer must be 'none', 'random' or 'pca'")


# --------------------------------------------------------------------------- #
# Coreset selection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_coreset_indices(X: torch.Tensor, n_select: int, proj_dim: int | None = 128,
                           seed: int = 0) -> torch.Tensor:
    """Greedy k-center coreset (PatchCore's approximate version).

    Repeatedly adds the patch farthest from everything selected so far, so the kept
    set covers the feature space. Distances are computed in a ``proj_dim``-dim
    random projection for speed (selection only; the stored bank keeps full features).
    Cost is O(N * n_select * proj_dim): run it on the GPU.
    """
    N = X.shape[0]
    n_select = int(min(max(n_select, 1), N))
    g = torch.Generator().manual_seed(seed)
    if proj_dim is not None and proj_dim < X.shape[1]:
        P = (torch.randn(X.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)).to(X.device)
        Z = _chunked_matmul(X, P)
    else:
        Z = X.float()
    start = int(torch.randint(N, (1,), generator=g))
    selected = torch.empty(n_select, dtype=torch.long, device=X.device)
    selected[0] = start
    min_d = ((Z - Z[start]) ** 2).sum(1)
    for j in range(1, n_select):
        i = torch.argmax(min_d)          # stays on device: no host sync per step
        selected[j] = i
        min_d = torch.minimum(min_d, ((Z - Z[i]) ** 2).sum(1))
    return selected


def random_coreset_indices(N: int, n_select: int, seed: int = 0) -> torch.Tensor:
    """Uniform random subsample – the naive baseline to compare greedy against."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(N, generator=g)[: int(min(max(n_select, 1), N))]


# --------------------------------------------------------------------------- #
# PatchCore
# --------------------------------------------------------------------------- #
@dataclass
class PatchCoreConfig:
    coreset_ratio: float = 0.1          # fraction of training patches kept (1.0 = all)
    coreset_method: str = "greedy"      # "greedy" | "random"
    coreset_proj_dim: int | None = 128  # projection used only for coreset distance computations
    reducer: str = "none"               # "none" | "random" | "pca"
    reducer_dim: int | None = None      # output dim when reducer != "none"
    bank_dtype: str = "float16"         # storage dtype of the memory bank
    k_neighbors: int = 1                # patch score = mean distance to k nearest bank entries
    blur_sigma: float = 4.0             # Gaussian smoothing of the upsampled anomaly map
    query_chunk: int = 4096
    bank_chunk: int = 65536
    seed: int = 0


class PatchCore:
    """Fit on normal-image features, score test features.

    >>> pc = PatchCore(PatchCoreConfig(coreset_ratio=0.1), device)
    >>> pc.fit(train_fs.features)                       # (N, C, h, w)
    >>> img_scores, maps = pc.predict(test_fs.features, image_size=448)
    """

    def __init__(self, config: PatchCoreConfig | None = None,
                 device: str | torch.device = "cpu"):
        self.cfg = config or PatchCoreConfig()
        self.device = torch.device(device)
        self.reducer: Reducer | None = None
        self.memory_bank: torch.Tensor | None = None
        self.fit_stats: dict = {}

    # ---- fitting ----------------------------------------------------------
    @torch.no_grad()
    def fit(self, train_features: torch.Tensor) -> "PatchCore":
        cfg = self.cfg
        t0 = time.perf_counter()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        X = to_patches(train_features).to(self.device, dtype)
        n_train, in_dim = X.shape

        self.reducer = make_reducer(cfg.reducer, cfg.reducer_dim, cfg.seed).fit(X)
        X = self.reducer.transform(X)                       # float32 (N, D)

        t1 = time.perf_counter()
        n_keep = max(1, int(round(n_train * cfg.coreset_ratio)))
        if cfg.coreset_ratio >= 1.0:
            idx = torch.arange(n_train, device=self.device)
        elif cfg.coreset_method == "greedy":
            idx = greedy_coreset_indices(X, n_keep, cfg.coreset_proj_dim, cfg.seed)
        elif cfg.coreset_method == "random":
            idx = random_coreset_indices(n_train, n_keep, cfg.seed).to(self.device)
        else:
            raise ValueError("coreset_method must be 'greedy' or 'random'")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        self.memory_bank = X[idx].to(getattr(torch, cfg.bank_dtype)).contiguous()
        self._bank_f32 = self.memory_bank.float()          # compute copy (not counted as storage)
        self._bank_sq = (self._bank_f32 ** 2).sum(1)
        self.fit_stats = {"n_train_patches": n_train, "input_dim": in_dim,
                          "bank_patches": self.memory_bank.shape[0],
                          "bank_dim": self.memory_bank.shape[1],
                          "reduce_seconds": t1 - t0, "coreset_seconds": t2 - t1,
                          "fit_seconds": time.perf_counter() - t0}
        return self

    # ---- scoring ------------------------------------------------------------
    @torch.no_grad()
    def nearest_distances(self, Q: torch.Tensor) -> torch.Tensor:
        """Euclidean distance from each query row (already reduced) to its nearest
        bank entry (mean over ``k_neighbors``). Chunked so huge banks fit in memory."""
        k = self.cfg.k_neighbors
        out = []
        for qs in range(0, Q.shape[0], self.cfg.query_chunk):
            q = Q[qs:qs + self.cfg.query_chunk].to(self.device, torch.float32)
            q_sq = (q ** 2).sum(1, keepdim=True)
            best = torch.full((q.shape[0], k), float("inf"), device=self.device)
            for bs in range(0, self._bank_f32.shape[0], self.cfg.bank_chunk):
                b = self._bank_f32[bs:bs + self.cfg.bank_chunk]
                d = (q_sq + self._bank_sq[bs:bs + self.cfg.bank_chunk][None] - 2 * q @ b.T)
                kk = min(k, d.shape[1])
                cand = torch.cat([best, d.clamp_min_(0).topk(kk, dim=1, largest=False).values], 1)
                best = cand.topk(k, dim=1, largest=False).values
            out.append(best.sqrt().mean(1))
        return torch.cat(out)

    @torch.no_grad()
    def score_patches(self, features: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) features -> (B, h, w) patch anomaly scores."""
        B, _, h, w = features.shape
        Q = self.reducer.transform(to_patches(features).to(self.device))
        return self.nearest_distances(Q).reshape(B, h, w)

    @torch.no_grad()
    def anomaly_map(self, patch_scores: torch.Tensor, image_size: int) -> torch.Tensor:
        """Upsample (B, h, w) patch scores to (B, S, S) and Gaussian-smooth (sigma=4 as in PatchCore)."""
        m = F.interpolate(patch_scores[:, None], size=(image_size, image_size),
                          mode="bilinear", align_corners=False)
        if self.cfg.blur_sigma > 0:
            s = self.cfg.blur_sigma
            ks = 2 * int(4 * s + 0.5) + 1
            m = gaussian_blur(m, kernel_size=[ks, ks], sigma=[s, s])
        return m[:, 0]

    @torch.no_grad()
    def predict(self, features: torch.Tensor, image_size: int,
                batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``image_scores`` (N,) = max patch score per image and
        ``anomaly_maps`` (N, S, S) float32 numpy arrays."""
        scores, maps = [], []
        for i in range(0, features.shape[0], batch_size):
            ps = self.score_patches(features[i:i + batch_size])
            scores.append(ps.flatten(1).max(1).values.cpu())
            maps.append(self.anomaly_map(ps, image_size).cpu())
        return torch.cat(scores).numpy(), torch.cat(maps).numpy().astype(np.float32)

    # ---- persistence ---------------------------------------------------------
    def save(self, path) -> None:
        torch.save({"config": asdict(self.cfg), "memory_bank": self.memory_bank.cpu(),
                    "reducer": self.reducer, "fit_stats": self.fit_stats}, path)

    @classmethod
    def load(cls, path, device="cpu") -> "PatchCore":
        d = torch.load(path, map_location="cpu", weights_only=False)
        pc = cls(PatchCoreConfig(**d["config"]), device)
        pc.reducer, pc.fit_stats = d["reducer"], d["fit_stats"]
        for attr in ("P", "mean", "components"):
            if hasattr(pc.reducer, attr):
                setattr(pc.reducer, attr, getattr(pc.reducer, attr).to(pc.device))
        pc.memory_bank = d["memory_bank"].to(pc.device)
        pc._bank_f32 = pc.memory_bank.float()
        pc._bank_sq = (pc._bank_f32 ** 2).sum(1)
        return pc
