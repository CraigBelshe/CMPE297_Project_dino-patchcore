# Experiment plan

**Research question:** which DINO layer gives the best trade-off between memory-bank size,
speed, and anomaly-detection accuracy, for texture vs. object categories?

**Categories:** texture = carpet, tile, wood; object = bottle, cable, transistor.
**Metrics for every run:** image AUROC, pixel AUROC, PRO; bank MB, feature dim, FPS (per stage).
Every run appends one row to the shared `results.csv`.

## A note on the hypothesis

All ViT-B blocks output 768-d tokens, so **choosing layer 6 instead of 12 does not shrink the
memory bank.** The layer changes *accuracy* and *compute*: with truncation, layer 8 runs 8 of 12
blocks. Storage only shrinks through coreset ratio, dimension reduction (random projection or PCA),
dtype, and resolution. The experiments below test these factors separately, then combine them for
the final hypothesis check.

Suggested restatement: *"An intermediate layer (6–8) with coreset ≤ X% and dimension ≤ D keeps
texture pixel AUROC ≥ 95% at ≤ 50% of the layer-12 baseline's bank size and ≥ 1.25× its FPS."*

## Experiments

| ID | Experiment | Varies | Fixed | Runs | Question answered |
|---|---|---|---|---|---|
| **E0** | Baseline / pipeline check | WRN-50 (and/or ResNet-50), `layer2+layer3` | 224 px, coreset 10%, 6 categories | 6–12 | Does our PatchCore reproduce published numbers (≈99% image AUROC)? **Gate: do not trust DINO results until this passes.** |
| **E1** | Layer sweep (core result) | DINOv2-B/14 single layer ∈ {3, 6, 7, 8, 9, 12}; coreset ∈ {100%, 10%} | 448 px, no reducer, k=3 aggregation | 72 | Which layer is most accurate, per domain? Does the best layer differ between texture and object? |
| **E2** | Multi-layer fusion | {6+9, 8+12, 6+9+12} | as E1, coreset 10% | 18 | Does concatenating layers beat the best single layer, and is it worth 2–3× storage? |
| **E3** | Compression sweep | best 2 layers from E1 + layer 12; coreset ∈ {10, 5, 1}%; reducer ∈ {none, random, PCA} × dim ∈ {384, 256, 128} | 448 px | ~350 (fast: cached features) | Accuracy vs bank MB curve per domain; where does accuracy drop off? |
| **E4** | Throughput | layer ∈ {3, 6, 8, 9, 12}; batch ∈ {1, 8} | **one GPU type (e.g. L4), one session**, best E3 compression | 10–20 | Does truncating at block 6–8 give ≥ 25% higher end-to-end FPS? Report backbone-only and end-to-end, since kNN cost does not shrink with depth. |
| **E5** | Resolution | 224 vs 448 | best config from E3 | 12 | 448 stores 4× the patches; is the localisation gain worth it? |
| **E6** | DINOv3 replication | repeat E1 (and E3 for its best layer) with `dinov3_vitb16` | 448 px (multiple of 16) | ~100 | Does the layer–domain pattern hold for DINOv3? |
| **E7** | Ablations (if time) | `apply_norm` on/off; aggregation kernel 1/3/5; greedy vs random coreset; k_neighbors 1 vs 3 | best config | ~40 | Which design choices matter? |
| **E8** | Final hypothesis check + variance | layer-12 baseline vs best intermediate config; seeds 0/1/2 | as chosen | ~36 | Pass/fail on the hypothesis thresholds, with mean ± std |

## Order and dependencies

```
E0 (gate) ─► E1 ─┬─► E2
                 ├─► E3 ─► E4, E5 ─► E8
                 └─► E6
E7 any time after E3
```

E1 and E6 are the expensive steps (backbone forward passes). E2, E3, E5 (per resolution), E7 and
E8 mostly reuse cached features, so they are cheap and can run in parallel.

## Practical notes

* Cache size at 448 px: ~1.5 MB per image per 768-d layer, so ~3 GB per layer for the 6
  categories (train + test). Cache only the layers you are sweeping, or explore at 224 px
  (4× smaller) first.
* Log the GPU name with every FPS number, and only compare FPS within one GPU type.
* Use `tag=` in `ExperimentConfig` to mark which experiment (E1, E3, ...) a row belongs to.
* Report texture and object means separately (`domain_summary`), as well as per category.
