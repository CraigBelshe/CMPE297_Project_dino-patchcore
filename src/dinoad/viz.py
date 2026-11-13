"""Plots: anomaly heatmaps and accuracy-vs-efficiency trade-offs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img) -> np.ndarray:
    """(3, H, W) normalised tensor -> (H, W, 3) float image in [0, 1]."""
    x = img.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(x * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)


def show_anomaly_maps(dataset, maps, scores, indices=None, n: int = 4, vmax=None, save_path=None):
    """Rows of [image | ground truth | heatmap overlay] for selected test images.
    Defaults to the ``n`` highest-scoring anomalous images."""
    labels = np.array([s.label for s in dataset.samples])
    if indices is None:
        bad = np.where(labels == 1)[0]
        indices = bad[np.argsort(-np.asarray(scores)[bad])][:n]
    vmax = vmax if vmax is not None else float(np.percentile(maps, 99.5))
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)), squeeze=False)
    for r, i in enumerate(indices):
        item = dataset[int(i)]
        img = denormalize(item["image"])
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{item['defect_type']}  score={scores[i]:.2f}")
        axes[r, 1].imshow(item["mask"][0], cmap="gray"); axes[r, 1].set_title("ground truth")
        axes[r, 2].imshow(img); axes[r, 2].imshow(maps[i], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        axes[r, 2].set_title("anomaly map")
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_score_histogram(labels, scores, title: str = ""):
    """Image-score distributions for good vs anomalous test images."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores[labels == 0], bins=30, alpha=0.6, label="good")
    ax.hist(scores[labels == 1], bins=30, alpha=0.6, label="anomalous")
    ax.set_xlabel("image anomaly score"); ax.set_ylabel("count"); ax.legend(); ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_tradeoff(df, x: str = "bank_mb", y: str = "pixel_auroc", hue: str = "layers",
                  logx: bool = True, save_path=None):
    """Accuracy vs efficiency scatter, one panel per domain (texture / object)."""
    domains = [d for d in ("texture", "object") if d in set(df["domain"])]
    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 4), squeeze=False)
    for ax, dom in zip(axes[0], domains):
        sub = df[df["domain"] == dom]
        # one point per configuration (averaged over the domain's categories)
        cfg_cols = [c for c in ("pc_coreset_ratio", "pc_reducer", "pc_reducer_dim") if c in sub]
        agg = (sub.groupby([hue] + cfg_cols, dropna=False)[[x, y]].mean().reset_index()
               .sort_values(x))
        for key, g in agg.groupby(hue):
            ax.plot(g[x], g[y], "o-", label=f"{hue}={key}")
        ax.set_title(dom); ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=0.3)
        ax.ticklabel_format(axis="y", useOffset=False)
        if logx:
            ax.set_xscale("log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
