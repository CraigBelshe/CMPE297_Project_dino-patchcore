"""Fast CPU tests: run with `pytest -q` (no dataset or pretrained weights needed)."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from dinoad import (MVTecAD, PatchCore, PatchCoreConfig, ViTLayerExtractor, compute_metrics,
                    extract_features, greedy_coreset_indices, make_loader, memory_bank_report,
                    pro_score, benchmark_pipeline)
from dinoad.synthetic import make_synthetic_mvtec


class TinyViT(nn.Module):
    """Mimics the DINOv2 interface: patch_embed, cls token, blocks, norm, patch_size."""

    def __init__(self, dim=32, depth=4, patch=8):
        super().__init__()
        self.patch_size = patch
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, 4, 64, batch_first=True) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.calls = 0

    def forward_features(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls_token.expand(len(t), -1, -1), t], 1)
        for b in self.blocks:
            self.calls += 1
            t = b(t)
        return self.norm(t)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    return make_synthetic_mvtec(tmp_path_factory.mktemp("mvtec"), ("carpet", "bottle"), size=64)


def test_dataset(data_root):
    ds = MVTecAD(data_root, "carpet", "test", image_size=64)
    item = ds[len(ds) - 1]
    assert item["image"].shape == (3, 64, 64) and item["mask"].shape == (1, 64, 64)
    assert item["label"] == 1 and item["mask"].sum() > 0


def test_extractor_shapes_and_truncation():
    m = TinyViT().eval()
    ex = ViTLayerExtractor(m, [2], truncate=True)
    out = ex(torch.randn(2, 3, 64, 64))
    assert out[2].shape == (2, 32, 8, 8)
    assert m.calls == 2                       # stopped after block 2 of 4
    assert ex.compute_fraction == 0.5
    # hooks are removed after each call
    assert all(len(b._forward_hooks) == 0 for b in m.blocks)


def test_extractor_matches_manual_forward():
    m = TinyViT().eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        t = m.patch_embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([m.cls_token, t], 1)
        t = m.blocks[1](m.blocks[0](t))
    ref = t[:, 1:].transpose(1, 2).reshape(1, 32, 4, 4)
    out = ViTLayerExtractor(m, [2])(x)[2]
    assert torch.allclose(out, ref, atol=1e-5)


def test_greedy_coreset_covers_clusters():
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(5, 16, generator=g) * 20
    X = torch.cat([c + torch.randn(200, 16, generator=g) for c in centers])
    idx = greedy_coreset_indices(X, 5, proj_dim=None)
    assert len(set((idx // 200).tolist())) == 5   # one point per cluster


def test_pro_sanity():
    rng = np.random.default_rng(0)
    masks = np.zeros((4, 64, 64), bool)
    masks[:, 10:20, 10:20] = True
    masks[:, 40:50, 30:45] = True
    perfect = masks.astype(np.float32) + rng.random(masks.shape) * 0.1
    assert pro_score(masks, perfect) > 0.99
    rand = rng.random(masks.shape).astype(np.float32)
    assert abs(pro_score(masks, rand) - 0.15) < 0.02


def test_end_to_end(data_root):
    torch.manual_seed(0)
    m = TinyViT().eval()
    ex = ViTLayerExtractor(m, [1, 2])
    dev = torch.device("cpu")
    tr = extract_features(ex, make_loader(MVTecAD(data_root, "carpet", "train", 64), 4, 0), dev,
                          keep_masks=False)
    te_ds = MVTecAD(data_root, "carpet", "test", 64)
    te = extract_features(ex, make_loader(te_ds, 4, 0), dev)
    assert tr.features.shape[1:] == (64, 8, 8)     # 2 layers x 32 dims

    for reducer, dim in [("none", None), ("random", 16), ("pca", 16)]:
        pc = PatchCore(PatchCoreConfig(coreset_ratio=0.25, reducer=reducer, reducer_dim=dim), dev)
        pc.fit(tr.features)
        scores, maps = pc.predict(te.features, 64)
        assert maps.shape == (len(te_ds), 64, 64)
        met = compute_metrics(te.labels.numpy(), scores, te.masks.numpy(), maps)
        assert all(0 <= v <= 1 for v in met.values())
        rep = memory_bank_report(pc)
        assert rep["bank_patches"] == round(tr.features.shape[0] * 64 * 0.25)
        assert rep["bank_dim"] == (dim or 64)

    bench = benchmark_pipeline(ex, pc, torch.stack([te_ds[i]["image"] for i in range(2)]),
                               n_warmup=1, n_iters=2)
    assert bench["fps"] > 0 and math.isclose(bench["compute_fraction"], 0.5)


def test_knn_matches_cdist():
    pc = PatchCore(PatchCoreConfig(coreset_ratio=1.0, bank_dtype="float32", bank_chunk=7,
                                   query_chunk=5, blur_sigma=0), "cpu")
    train = torch.randn(3, 8, 4, 4)
    pc.fit(train)
    test = torch.randn(2, 8, 4, 4)
    got = pc.score_patches(test).flatten()
    Q = test.permute(0, 2, 3, 1).reshape(-1, 8)
    ref = torch.cdist(Q, train.permute(0, 2, 3, 1).reshape(-1, 8)).min(1).values
    assert torch.allclose(got, ref, atol=1e-4)
