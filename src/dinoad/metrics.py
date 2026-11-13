"""Accuracy metrics: image AUROC, pixel AUROC, PRO (per-region overlap).

PRO follows the MVTec AD definition: for each threshold, the mean over all ground
truth connected components (8-connectivity) of the fraction of that component
predicted anomalous, plotted against the false-positive rate on normal pixels;
the area under that curve up to FPR = 0.3 is normalised to [0, 1].
A random scorer gets ~0.15, a perfect one 1.0.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def image_auroc(labels, image_scores) -> float:
    """Detection AUROC over test images (label 1 = anomalous)."""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(image_scores)))


def _flatten(masks, maps, max_pixels: int | None, seed: int = 0):
    m = np.asarray(masks).astype(bool).ravel()
    s = np.asarray(maps, dtype=np.float32).ravel()
    if max_pixels is not None and m.size > max_pixels:
        idx = np.random.default_rng(seed).choice(m.size, max_pixels, replace=False)
        m, s = m[idx], s[idx]
    return m, s


def pixel_auroc(masks, anomaly_maps, max_pixels: int | None = None) -> float:
    """Segmentation AUROC over all test pixels. ``max_pixels`` randomly subsamples
    pixels to save time/memory (e.g. 5_000_000); leave None for the exact value."""
    m, s = _flatten(masks, anomaly_maps, max_pixels)
    if m.all() or not m.any():
        return float("nan")
    return float(roc_auc_score(m, s))


def pro_score(masks, anomaly_maps, fpr_limit: float = 0.3, num_thresholds: int = 300) -> float:
    """Normalised area under the PRO curve up to ``fpr_limit``.

    Implementation is histogram-based (O(#pixels)), so it is fast even at 448x448:
    thresholds are placed at quantiles of normal-pixel scores so they are spread
    evenly over FPR in [0, fpr_limit].
    """
    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    if not masks.any():
        return float("nan")

    # Global connected-component ids across all images (0 = background).
    structure = np.ones((3, 3), dtype=int)
    region_ids = np.zeros(masks.shape, dtype=np.int64)
    n_regions = 0
    for i in range(masks.shape[0]):
        lab, n = ndimage.label(masks[i], structure=structure)
        region_ids[i][lab > 0] = lab[lab > 0] + n_regions
        n_regions += n

    normal = maps[~masks]
    normal_sorted = np.sort(normal)
    fpr_grid = np.linspace(0.0, fpr_limit, num_thresholds)
    thr = np.quantile(normal_sorted, 1.0 - fpr_grid)
    thr[0] = np.nextafter(normal_sorted[-1], np.float32(np.inf))   # FPR exactly 0
    thr = np.unique(thr)                                           # ascending

    # Exact FPR at each threshold: share of normal pixels with score >= t.
    fpr = 1.0 - np.searchsorted(normal_sorted, thr, side="left") / normal_sorted.size

    # Per-region coverage via a (region, bin) histogram.
    rid = region_ids[masks]
    rs = maps[masks]
    K = thr.size
    bins = np.searchsorted(thr, rs, side="right")        # #thresholds <= score, in [0, K]
    hist = np.bincount(rid * (K + 1) + bins, minlength=(n_regions + 1) * (K + 1))
    hist = hist.reshape(n_regions + 1, K + 1)[1:]        # drop background row
    # pixels with score >= thr[k]  <=>  bins > k
    count_ge = np.cumsum(hist[:, ::-1], axis=1)[:, ::-1][:, 1:]  # (R, K)
    sizes = hist.sum(1, keepdims=True)
    pro = (count_ge / sizes).mean(0)                     # (K,)

    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]
    keep = fpr <= fpr_limit
    x, y = fpr[keep], pro[keep]
    if x.size and x[-1] < fpr_limit and keep.sum() < fpr.size:   # interpolate to the limit
        j = keep.sum()
        y_lim = np.interp(fpr_limit, [fpr[j - 1], fpr[j]], [pro[j - 1], pro[j]])
        x, y = np.append(x, fpr_limit), np.append(y, y_lim)
    return float(np.trapezoid(y, x) / fpr_limit)


def compute_metrics(labels, image_scores, masks=None, anomaly_maps=None, compute_pro: bool = True,
                    max_pixels: int | None = None) -> dict[str, float]:
    """All accuracy metrics in one dict: ``image_auroc``, ``pixel_auroc``, ``pro``."""
    out = {"image_auroc": image_auroc(labels, image_scores)}
    if masks is not None and anomaly_maps is not None:
        out["pixel_auroc"] = pixel_auroc(masks, anomaly_maps, max_pixels)
        out["pro"] = pro_score(masks, anomaly_maps) if compute_pro else float("nan")
    return out
