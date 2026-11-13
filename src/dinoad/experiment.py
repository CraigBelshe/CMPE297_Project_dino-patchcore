"""One-call experiment runner that strings the helpers together.

The notebook walks through every step by hand for one category; use
:func:`run_category` / :func:`run_domain` once you trust the pipeline and want to
sweep layers, coreset ratios, projection dims, etc.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch

from .backbones import build_extractor
from .data import MVTecAD, domain_of, get_categories, make_loader
from .efficiency import benchmark_pipeline, memory_bank_report
from .features import feature_cache_path, get_or_extract
from .metrics import compute_metrics
from .patchcore import PatchCore, PatchCoreConfig


@dataclass
class ExperimentConfig:
    data_root: str
    backbone: str = "dinov2_vitb14"
    layers: tuple = (12,)                 # ViT: 1-indexed blocks; CNN: ("layer2", "layer3")
    image_size: int = 448
    agg_kernel: int = 3
    apply_norm: bool = False
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    batch_size: int = 16
    num_workers: int = 2
    cache_dir: str | None = None          # feature cache (e.g. on Drive); None = no caching
    compute_pro: bool = True
    max_metric_pixels: int | None = None  # subsample pixels for pixel AUROC (None = exact)
    benchmark: bool = True
    benchmark_batch: int = 8
    benchmark_iters: int = 10
    tag: str = ""                         # free-text label stored in the results row


def run_category(cfg: ExperimentConfig, category: str, model: torch.nn.Module,
                 device: torch.device, return_artifacts: bool = False):
    """Extract features -> fit PatchCore -> score -> metrics -> efficiency.

    Returns a flat dict (one results-CSV row). With ``return_artifacts=True`` also
    returns ``(row, artifacts)`` where artifacts holds the PatchCore object, scores,
    maps and test FeatureSet for visualisation.
    """
    extractor = build_extractor(model, cfg.backbone, cfg.layers, cfg.apply_norm, truncate=True)
    train_ds = MVTecAD(cfg.data_root, category, "train", cfg.image_size)
    test_ds = MVTecAD(cfg.data_root, category, "test", cfg.image_size)
    loader_kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    def cache(split):
        if cfg.cache_dir is None:
            return None
        return feature_cache_path(cfg.cache_dir, cfg.backbone, cfg.image_size, cfg.layers,
                                  category, split, cfg.agg_kernel, cfg.apply_norm)

    train_fs = get_or_extract(cache("train"), extractor, make_loader(train_ds, **loader_kw),
                              device, agg_kernel=cfg.agg_kernel, keep_masks=False,
                              desc=f"{category} train")
    test_fs = get_or_extract(cache("test"), extractor, make_loader(test_ds, **loader_kw),
                             device, agg_kernel=cfg.agg_kernel, desc=f"{category} test")

    pc = PatchCore(cfg.patchcore, device).fit(train_fs.features)
    t0 = time.perf_counter()
    scores, maps = pc.predict(test_fs.features, cfg.image_size)
    score_seconds = time.perf_counter() - t0

    acc = compute_metrics(test_fs.labels.numpy(), scores, test_fs.masks.numpy(), maps,
                          compute_pro=cfg.compute_pro, max_pixels=cfg.max_metric_pixels)
    mem = memory_bank_report(pc)

    row = {"category": category, "domain": domain_of(category), "backbone": cfg.backbone,
           "layers": "+".join(map(str, cfg.layers)), "image_size": cfg.image_size,
           "agg_kernel": cfg.agg_kernel, "apply_norm": cfg.apply_norm,
           "feature_grid": "x".join(map(str, test_fs.grid)), "feature_dim": train_fs.dim,
           **{f"pc_{k}": v for k, v in asdict(cfg.patchcore).items()},
           **acc, **mem, **pc.fit_stats, "score_seconds_total": score_seconds,
           "tag": cfg.tag}

    if cfg.benchmark:
        n = min(cfg.benchmark_batch, len(test_ds))
        imgs = torch.stack([test_ds[i]["image"] for i in range(n)])
        row.update(benchmark_pipeline(extractor, pc, imgs, cfg.agg_kernel,
                                      n_iters=cfg.benchmark_iters))
    if return_artifacts:
        return row, {"patchcore": pc, "scores": scores, "maps": maps, "test_fs": test_fs,
                     "test_ds": test_ds}
    return row


def run_domain(cfg: ExperimentConfig, domain: str, model: torch.nn.Module,
               device: torch.device, results_csv: str | None = None) -> list[dict]:
    """Run :func:`run_category` on every category of ``"texture"`` / ``"object"`` / ``"all"``,
    appending each row to ``results_csv`` as soon as it finishes (safe if Colab disconnects)."""
    from .results import append_result
    rows = []
    for cat in get_categories(domain):
        row = run_category(cfg, cat, model, device)
        rows.append(row)
        if results_csv:
            append_result(results_csv, row)
        print(f"{cat:12s} img-AUROC {row['image_auroc']:.4f}  pix-AUROC {row.get('pixel_auroc', float('nan')):.4f}"
              f"  PRO {row.get('pro', float('nan')):.4f}  bank {row['bank_mb']:.1f} MB"
              + (f"  {row['fps']:.1f} FPS" if 'fps' in row else ""))
    return rows
