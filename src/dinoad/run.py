"""Command-line experiment runner (sweeps without the notebook).

Every combination of --layers x --coreset x --reducer-dim is run on every category of
--domain, and each result is appended to --results as soon as it finishes.

Examples (in a Colab cell prefix with `!`):

    # E1 layer sweep on textures, full bank and 10% coreset
    python -m dinoad.run --data-root /content/mvtec --domain texture \
        --layers 3 6 7 8 9 12 --coreset 1.0 0.1 --tag E1 \
        --cache-dir /content/drive/MyDrive/CMPE297_final_project/feature_cache \
        --results  /content/drive/MyDrive/CMPE297_final_project/results/results.csv

    # E2 multi-layer fusion: join layers with '+'
    python -m dinoad.run ... --layers 6+9 8+12 6+9+12 --coreset 0.1 --tag E2

    # E0 CNN baseline
    python -m dinoad.run ... --backbone wide_resnet50_2 --layers layer2+layer3 \
        --image-size 224 --coreset 0.1 --tag E0

    # pipeline smoke test with fake data and random weights (no downloads of data)
    python -m dinoad.run --synthetic --data-root /tmp/fake --domain texture --layers 6 \
        --image-size 224 --no-pretrained --results /tmp/results.csv
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import torch

from .backbones import load_backbone
from .data import copy_to_local, get_categories
from .experiment import ExperimentConfig, run_category
from .patchcore import PatchCoreConfig
from .results import append_result


def _parse_layers(spec: str):
    parts = spec.split("+")
    return tuple(p if p.startswith("layer") else int(p) for p in parts)


def _none_or_int(s: str):
    return None if s.lower() == "none" else int(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data-root", required=True, help="folder containing <category>/ folders")
    g.add_argument("--drive-root", default=None,
                   help="if given, copy categories from here to --data-root first (Drive -> local)")
    g.add_argument("--domain", default="texture", choices=["texture", "object", "all"])
    g.add_argument("--categories", nargs="*", default=None, help="override --domain with explicit categories")
    g.add_argument("--synthetic", action="store_true", help="generate fake data in --data-root (testing only)")
    g.add_argument("--image-size", type=int, default=448)

    g = p.add_argument_group("backbone")
    g.add_argument("--backbone", default="dinov2_vitb14")
    g.add_argument("--layers", nargs="+", default=["12"],
                   help="one or more layer specs; '+' concatenates, e.g. 6 9 12 6+9 layer2+layer3")
    g.add_argument("--dino-source", default="hub", choices=["hub", "drive"])
    g.add_argument("--weights-path", default=None, help="DINOv2/v3 or CNN state dict (.pth)")
    g.add_argument("--repo-dir", default=None, help="local clone of the dinov2/dinov3 repo")
    g.add_argument("--hub-dir", default=None, help="torch.hub cache dir (e.g. on Drive)")
    g.add_argument("--no-pretrained", action="store_true", help="random weights (testing only)")
    g.add_argument("--apply-norm", action="store_true")
    g.add_argument("--agg-kernel", type=int, default=3)

    g = p.add_argument_group("patchcore")
    g.add_argument("--coreset", type=float, nargs="+", default=[0.1], help="coreset ratios, e.g. 1.0 0.1 0.05 0.01")
    g.add_argument("--coreset-method", default="greedy", choices=["greedy", "random"])
    g.add_argument("--reducer", default="none", choices=["none", "random", "pca"])
    g.add_argument("--reducer-dim", type=_none_or_int, nargs="+", default=[None],
                   help="output dims for --reducer, e.g. 384 256 128 (use 'none' for no reduction)")
    g.add_argument("--k-neighbors", type=int, default=1)
    g.add_argument("--seeds", type=int, nargs="+", default=[0])

    g = p.add_argument_group("run")
    g.add_argument("--results", required=True, help="results CSV to append to")
    g.add_argument("--cache-dir", default=None, help="feature cache dir (strongly recommended)")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--num-workers", type=int, default=2)
    g.add_argument("--no-pro", action="store_true", help="skip PRO (faster)")
    g.add_argument("--no-benchmark", action="store_true", help="skip FPS benchmark")
    g.add_argument("--bench-batch", type=int, default=8)
    g.add_argument("--bench-iters", type=int, default=20)
    g.add_argument("--tag", default="")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cats = a.categories or get_categories(a.domain)

    if a.synthetic:
        from .synthetic import make_synthetic_mvtec
        make_synthetic_mvtec(a.data_root, cats)
    elif a.drive_root:
        copy_to_local(a.drive_root, a.data_root, cats)

    if a.backbone.startswith("dinov2"):
        kw = dict(source=a.dino_source, weights_path=a.weights_path, repo_dir=a.repo_dir,
                  hub_dir=a.hub_dir, pretrained=not a.no_pretrained)
    elif a.backbone.startswith("dinov3"):
        kw = dict(repo_dir=a.repo_dir, weights_path=a.weights_path)
    else:
        kw = dict(weights_path=a.weights_path)
    model = load_backbone(a.backbone, device=device, **kw)
    print(f"device={device} backbone={a.backbone} categories={cats}")

    base = ExperimentConfig(
        data_root=a.data_root, backbone=a.backbone, image_size=a.image_size,
        agg_kernel=a.agg_kernel, apply_norm=a.apply_norm, batch_size=a.batch_size,
        num_workers=a.num_workers, cache_dir=None if a.synthetic else a.cache_dir,
        compute_pro=not a.no_pro, benchmark=not a.no_benchmark,
        benchmark_batch=a.bench_batch, benchmark_iters=a.bench_iters, tag=a.tag)

    combos = list(itertools.product(a.layers, a.coreset, a.reducer_dim, a.seeds))
    print(f"{len(combos)} configurations x {len(cats)} categories = {len(combos) * len(cats)} runs")
    for layer_spec, ratio, rdim, seed in combos:
        pc = PatchCoreConfig(coreset_ratio=ratio, coreset_method=a.coreset_method,
                             reducer=a.reducer if rdim is not None else "none", reducer_dim=rdim,
                             k_neighbors=a.k_neighbors, seed=seed)
        cfg = replace(base, layers=_parse_layers(layer_spec), patchcore=pc)
        for cat in cats:
            row = run_category(cfg, cat, model, device)
            append_result(a.results, row)
            print(f"L={layer_spec:8s} coreset={ratio:<5} dim={rdim or row['bank_dim']:<4} seed={seed} "
                  f"{cat:11s} img={row['image_auroc']:.4f} pix={row['pixel_auroc']:.4f} "
                  f"pro={row['pro']:.4f} bank={row['bank_mb']:.2f}MB"
                  + (f" fps={row['fps']:.1f}" if "fps" in row else ""), flush=True)


if __name__ == "__main__":
    main()
