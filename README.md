# dino-patchcore: finding the best DINO layer for fast anomaly detection

CMPE297 final project: Craig Belshe, Esther Hammon, Raymond Leung.

PatchCore anomaly detection on MVTec AD using intermediate-layer features from self-supervised
DINOv2 / DINOv3 ViTs. Compares accuracy (image AUROC, pixel AUROC, PRO) against memory-bank size
and throughput, for texture (carpet, tile, wood) vs. object (bottle, cable, transistor) categories.

See **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** for the experiment plan.

## Repo layout

```
src/dinoad/
  data.py        MVTec AD Dataset, texture/object domains, Drive -> local copy
  backbones.py   load_dinov2 (hub or Drive weights), load_dinov3, load_cnn (ResNet baseline)
                 ViTLayerExtractor: forward hooks on chosen blocks + early stop after the deepest one
  features.py    3x3 neighbourhood aggregation, layer fusion, FeatureSet, feature caching
  patchcore.py   PatchCore memory bank, greedy coreset, random projection / PCA, chunked kNN
  metrics.py     image AUROC, pixel AUROC, PRO (fast histogram implementation)
  efficiency.py  memory-bank footprint, per-stage latency / FPS benchmark
  experiment.py  ExperimentConfig, run_category, run_domain
  results.py     append-only results CSV, domain summaries
  viz.py         heatmaps, score histograms, accuracy-vs-efficiency plots
  synthetic.py   tiny fake MVTec-style dataset for smoke tests
notebooks/
  01_patchcore_mvtec.ipynb   documented walkthrough of every helper, then domain runs and sweeps
tests/
  test_pipeline.py           CPU tests (no data or weights needed): pytest -q
docs/EXPERIMENTS.md
```

## Google Drive layout (shared team folder)

```
MyDrive/CMPE297_final_project/
  mvtec_anomaly_detection/   carpet/ tile/ wood/ bottle/ cable/ transistor/   (or <cat>.tar.xz)
  weights/                   dinov2_vitb14_pretrain.pth   (optional; only for DINO_SOURCE="drive")
  torch_hub/                 torch.hub cache (created automatically)
  feature_cache/             cached fp16 features (created automatically)
  results/results.csv        shared results log (created automatically)
  outputs/                   figures
```

MVTec AD is distributed by MVTec under a non-commercial licence. Download it from their website
and put it in the shared folder. Everyone on the team should add a shortcut to the shared folder
in their own "My Drive" so the path above works in each person's Colab.

## Quick start (Colab)

1. Push this repo to GitHub and set `REPO_URL` in the notebook's first code cell.
2. Open `notebooks/01_patchcore_mvtec.ipynb` in Colab and select a GPU runtime.
3. Edit the **Config** cell (`DOMAIN = "texture"` or `"object"`, `LAYERS`, `CORESET_RATIO`, ...) and run all.
   To check the pipeline before the data is on Drive, set `USE_SYNTHETIC = True`.

## Local use

```bash
pip install -e ".[dev]"
pytest -q
```

## Design notes

* **Layer numbering:** ViT layers are 1-indexed block numbers (ViT-B: 1..12, 12 = last block).
* **Hooks + truncation:** `ViTLayerExtractor` attaches forward hooks only during each call and stops
  the forward pass after the deepest requested block, so shallow layers really are faster.
  Its output was checked to match DINOv2's `get_intermediate_layers` exactly (with and without the final norm).
* **Storage:** every ViT-B block emits 768-d tokens. The memory bank shrinks only through
  `coreset_ratio`, `reducer`/`reducer_dim`, `bank_dtype` (fp16 by default) and image resolution.
* **Image preprocessing:** images are resized (no centre crop) to `image_size`, which must be a
  multiple of the patch size (14 for DINOv2, 16 for DINOv3).
* **Image score:** max patch distance. PatchCore's optional re-weighting is not implemented.
* **PRO:** MVTec definition (8-connected regions, FPR ≤ 0.3, normalised). Sanity check: random ≈ 0.15, perfect = 1.0.
