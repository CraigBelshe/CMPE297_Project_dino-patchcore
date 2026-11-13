# CMPE297_Project_dino-patchcore: finding the best DINO layer for fast anomaly detection

CMPE297 final project: Craig Belshe, Esther Hammon, Raymond Leung.

PatchCore anomaly detection on MVTec AD using intermediate-layer features from self-supervised
DINOv2 / DINOv3 Vision Transformers. For texture categories (carpet, tile, wood) and object
categories (bottle, cable, transistor), it measures how layer depth affects **accuracy** (image
AUROC, pixel AUROC, PRO), **memory-bank size**, and **throughput**.

**Contents**

1. [Repo layout](#1-repo-layout)
2. [One-time setup](#2-one-time-setup-do-this-once-per-team)
3. [Run the notebook in Colab](#3-run-the-notebook-in-colab)
4. [Run experiments from the command line](#4-run-experiments-from-the-command-line)
5. [Recipes for each experiment (E0–E8)](#5-recipes-for-each-experiment-e0e8)
6. [Use the Python API directly](#6-use-the-python-api-directly)
7. [Outputs: results CSV, cache, figures](#7-outputs-results-csv-feature-cache-figures)
8. [Analyse results](#8-analyse-results)
9. [Local development and tests](#9-local-development-and-tests)
10. [Design notes](#10-design-notes)
11. [Troubleshooting](#11-troubleshooting)
12. [What has been verified](#12-what-has-been-verified)

The experiment plan (what to run and why) is in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

---

## 1. Repo layout

```
CMPE297_Project_dino-patchcore/
├── README.md                 <- you are here
├── docs/EXPERIMENTS.md       experiment plan E0–E8
├── notebooks/
│   └── 01_patchcore_mvtec.ipynb   documented walkthrough of every helper + domain runs + sweeps
├── src/dinoad/               the Python package (import dinoad)
│   ├── data.py          MVTec AD Dataset, texture/object domains, Drive -> local copy
│   ├── backbones.py     load_dinov2 / load_dinov3 / load_cnn, ViTLayerExtractor (forward hooks)
│   ├── features.py      3x3 neighbourhood aggregation, layer fusion, FeatureSet, feature cache
│   ├── patchcore.py     memory bank, greedy coreset, random projection / PCA, chunked kNN
│   ├── metrics.py       image AUROC, pixel AUROC, PRO
│   ├── efficiency.py    memory-bank footprint, per-stage latency / FPS
│   ├── experiment.py    ExperimentConfig, run_category, run_domain
│   ├── run.py           command-line sweep runner (python -m dinoad.run)
│   ├── results.py       append-only results CSV, domain summaries
│   ├── viz.py           heatmaps, score histograms, accuracy-vs-efficiency plots
│   └── synthetic.py     tiny fake MVTec-style dataset for smoke tests
├── tests/test_pipeline.py    CPU tests, no data or weights needed
├── pyproject.toml / requirements.txt
└── .gitignore                data, caches, results, *.pt, *.pth are never committed
```

---

## 2. One-time setup (do this once per team)

### 2.1 The GitHub repo

The code lives at **https://github.com/CraigBelshe/CMPE297_Project_dino-patchcore** (private).
The repo owner adds the other teammates under **Settings → Collaborators**. Each teammate accepts the
email invite, then clones the repo:

```bash
git clone https://github.com/CraigBelshe/CMPE297_Project_dino-patchcore.git
```

Because the repo is **private**, Colab needs a token to clone it. Each person:

1. Creates a fine-grained personal access token (GitHub → **Settings → Developer settings →
   Fine-grained tokens**) with **Contents: Read** on this repo.
2. Saves it in Colab's **Secrets** panel (key icon in the left sidebar) as `GH_TOKEN`, with notebook access turned on.
3. Replaces `REPO_URL` in the notebook's first code cell with:

```python
from google.colab import userdata
REPO_URL = f"https://{userdata.get('GH_TOKEN')}@github.com/CraigBelshe/CMPE297_Project_dino-patchcore.git"
```

Alternatively, make the repo public (**Settings → General → Danger Zone**); it contains no data or weights.

### 2.2 Create the shared Google Drive folder

One person creates this folder and shares it with the others (Editor access):

```
MyDrive/CMPE297_final_project/
├── mvtec_anomaly_detection/   <- dataset (section 2.3)
├── weights/                   <- optional DINOv2 weights (section 2.4)
├── torch_hub/                 <- created automatically
├── feature_cache/             <- created automatically
├── results/results.csv        <- created automatically: the team's shared results log
└── outputs/                   <- created automatically: figures
```

**Each teammate**: open "Shared with me" in Drive, right-click the folder, and choose
**Organize → Add shortcut → My Drive**. The path `/content/drive/MyDrive/CMPE297_final_project`
then works in everyone's Colab, and nobody has to edit paths.

### 2.3 Get MVTec AD

MVTec AD is free for non-commercial research. Download it from MVTec's website (search
"MVTec AD download"; it requires a short form). You only need the six project categories.

**Option A (recommended): per-category archives.** Download `carpet.tar.xz`, `tile.tar.xz`,
`wood.tar.xz`, `bottle.tar.xz`, `cable.tar.xz`, `transistor.tar.xz` and upload them unchanged to
`mvtec_anomaly_detection/`. Each session, `copy_to_local` extracts them onto the Colab machine's
fast local disk (about a minute per category).

**Option B: extracted folders.** If you have the full `mvtec_anomaly_detection.tar.xz`, extract it
once in Colab and copy only the six categories to Drive:

```python
!mkdir -p /content/mvtec_full && tar -xf "/content/drive/MyDrive/CMPE297_final_project/mvtec_anomaly_detection.tar.xz" -C /content/mvtec_full
!for c in carpet tile wood bottle cable transistor; do cp -r /content/mvtec_full/$c "/content/drive/MyDrive/CMPE297_final_project/mvtec_anomaly_detection/"; done
```

Either way, the code expects the standard MVTec layout per category:

```
carpet/
├── train/good/*.png
├── test/good/*.png
├── test/<defect_type>/*.png
└── ground_truth/<defect_type>/*_mask.png
```

If the archive creates an extra top-level folder, point `MVTEC_DRIVE` (notebook) or
`--drive-root` (CLI) at the folder that directly contains `carpet/`, `tile/`, and so on.

### 2.4 Backbone weights

| Backbone | How to get it | Settings |
|---|---|---|
| **DINOv2** (public) | Nothing to do. It downloads from torch.hub the first time and is cached in `torch_hub/` on Drive. | `DINO_SOURCE = "hub"` |
| DINOv2 from a file | Download `dinov2_vitb14_pretrain.pth` (linked from the DINOv2 GitHub README) into `weights/`. | `DINO_SOURCE = "drive"`, `DINO_WEIGHTS = ".../weights/dinov2_vitb14_pretrain.pth"` |
| **DINOv3** (gated) | Request access from Meta (DINOv3 GitHub README). Clone `facebookresearch/dinov3` to Drive and save the ViT-B/16 checkpoint in `weights/`. | `BACKBONE = "dinov3_vitb16"`, `DINO_REPO_DIR`, `DINO_WEIGHTS`, `IMAGE_SIZE` a multiple of 16 |
| **ResNet baseline** | Nothing to do (torchvision downloads ImageNet weights). | `BACKBONE = "wide_resnet50_2"` or `"resnet50"`, `LAYERS = ["layer2", "layer3"]` |

Request DINOv3 access early, since approval can take time.

---

## 3. Run the notebook in Colab

1. Open `notebooks/01_patchcore_mvtec.ipynb` in Colab (**File → Open notebook → GitHub**, or upload it).
2. **Runtime → Change runtime type → GPU.** An L4 or T4 is enough; an A100 is not needed.
3. Because the repo is private, set `REPO_URL` in the first code cell to the token URL (section 2.1).
4. Edit the **Config** cell (reference below). On a fresh setup, run once with
   `USE_SYNTHETIC = True` to check that everything installs and runs before you touch real data.
5. **Runtime → Run all.**

The setup cell mounts Drive, clones the repo to `/content/CMPE297_Project_dino-patchcore` (or runs `git pull` if
it's already there), and installs it with `pip install -e`. After you push code changes, rerun
that cell to pick them up, and restart the runtime if a module was already imported.

### What each section does

| § | Section | Helpers called |
|---|---|---|
| 1 | Setup: mount Drive, clone, install | – |
| 2 | Config | – |
| 3 | Data: copy Drive → local, verify counts, build datasets | `copy_to_local`, `verify_mvtec_root`, `MVTecAD`, `make_loader` |
| 4 | Load backbone, attach layer extractor, check output shapes | `load_backbone`, `build_extractor` |
| 5 | Extract patch features (or load them from the cache) | `feature_cache_path`, `get_or_extract` |
| 6 | Build the memory bank, report its size | `PatchCoreConfig`, `PatchCore.fit`, `memory_bank_report` |
| 7 | Score test images | `PatchCore.predict` |
| 8 | Accuracy | `image_auroc`, `pixel_auroc`, `pro_score`, `compute_metrics` |
| 9 | Speed | `benchmark_pipeline`, `benchmark_backbone_only` |
| 10 | Heatmaps and score histogram (saved to `outputs/`) | `show_anomaly_maps`, `plot_score_histogram` |
| 11 | Append the walkthrough run to `results.csv` | `append_result` |
| 12 | Run every category in the domain | `ExperimentConfig`, `run_domain` |
| 13 | Optional layer × coreset sweep, summary table, trade-off plot | `run_category`, `load_results`, `domain_summary`, `plot_tradeoff` |

Sections 3–11 go through **one** category (`CATEGORY`, which defaults to the first in the domain)
step by step. Section 12 repeats all of it for every category in `DOMAIN`.

### Config cell reference

| Variable | Default | Meaning |
|---|---|---|
| `DRIVE_ROOT` | `/content/drive/MyDrive/CMPE297_final_project` | shared team folder |
| `MVTEC_DRIVE` | `{DRIVE_ROOT}/mvtec_anomaly_detection` | dataset on Drive (folders or `<cat>.tar.xz`) |
| `LOCAL_DATA` | `/content/mvtec` | fast local copy (re-created each session) |
| `CACHE_DIR` | `{DRIVE_ROOT}/feature_cache` | feature cache; `None` disables it |
| `RESULTS_CSV` | `{DRIVE_ROOT}/results/results.csv` | shared results log |
| `OUTPUT_DIR` | `{DRIVE_ROOT}/outputs` | saved figures |
| `USE_SYNTHETIC` | `False` | `True` = fake data, for pipeline testing only (nothing is logged) |
| `DOMAIN` | `"texture"` | `"texture"` (carpet, tile, wood) or `"object"` (bottle, cable, transistor) |
| `CATEGORY` | `None` | category for the step-by-step walkthrough (`None` = first in the domain) |
| `BACKBONE` | `"dinov2_vitb14"` | also `dinov3_vitb16`, `wide_resnet50_2`, `resnet50` |
| `DINO_SOURCE` | `"hub"` | `"hub"` or `"drive"` (load `DINO_WEIGHTS`) |
| `DINO_WEIGHTS` | `{DRIVE_ROOT}/weights/dinov2_vitb14_pretrain.pth` | used when `DINO_SOURCE = "drive"` (and for DINOv3) |
| `DINO_REPO_DIR` | `None` | local clone of the dinov2/dinov3 repo (needed for DINOv3) |
| `HUB_DIR` | `{DRIVE_ROOT}/torch_hub` | torch.hub cache so weights download only once |
| `DINO_PRETRAINED` | `True` | `False` = random weights (testing only) |
| `LAYERS` | `[8]` | **1-indexed** blocks, e.g. `[6]`, `[12]`, `[6, 9]` (concatenated). CNN: `["layer2", "layer3"]` |
| `IMAGE_SIZE` | `448` | multiple of the patch size: 14 for DINOv2 (224 → 16×16 grid, 448 → 32×32), 16 for DINOv3 |
| `AGG_KERNEL` | `3` | neighbourhood averaging size (1 = off) |
| `APPLY_NORM` | `False` | apply the final LayerNorm to intermediate tokens |
| `CORESET_RATIO` | `0.1` | fraction of training patches kept (1.0 / 0.1 / 0.05 / 0.01) |
| `CORESET_METHOD` | `"greedy"` | `"greedy"` (k-center, PatchCore) or `"random"` |
| `REDUCER`, `REDUCER_DIM` | `"none"`, `None` | `"random"` projection or `"pca"` to `REDUCER_DIM` dims |
| `SEED` | `0` | coreset start point and projection seed |
| `BATCH_SIZE`, `NUM_WORKERS` | `16`, `2` | feature-extraction loader settings |
| `BENCH_BATCH`, `BENCH_ITERS` | `8`, `20` | throughput benchmark batch and timed iterations |
| `RUN_SWEEP` | `False` | section 13: run the layer × coreset sweep |
| `TAG` | `"walkthrough"` | label stored in every results row (use the experiment ID: `E1`, `E3`, ...) |

---

## 4. Run experiments from the command line

For sweeps, the command-line runner is simpler than editing the notebook. It runs every
combination of `--layers` × `--coreset` × `--reducer-dim` × `--seeds` on each category of
`--domain`, and appends each row to `--results` **as soon as it finishes**, so a disconnect loses
at most one run.

In Colab, after running the setup cell (section 1 of the notebook), or in any Python environment
with the package installed:

```python
D = "/content/drive/MyDrive/CMPE297_final_project"
!python -m dinoad.run \
    --drive-root {D}/mvtec_anomaly_detection --data-root /content/mvtec \
    --domain texture --layers 6 8 12 --coreset 1.0 0.1 \
    --cache-dir {D}/feature_cache --hub-dir {D}/torch_hub \
    --results {D}/results/results.csv --tag E1
```

`python -m dinoad.run --help` lists every option. The main ones:

| Option | Default | Notes |
|---|---|---|
| `--data-root` | required | local folder with `<category>/` folders |
| `--drive-root` | – | copy/extract categories from here into `--data-root` first |
| `--domain` / `--categories` | `texture` | `texture`, `object`, `all`, or an explicit list of categories |
| `--backbone` | `dinov2_vitb14` | as in the notebook |
| `--layers` | `12` | several specs; `+` concatenates: `3 6 9 12 6+9 layer2+layer3` |
| `--image-size` | `448` | |
| `--coreset` | `0.1` | several ratios: `1.0 0.1 0.05 0.01` |
| `--reducer` / `--reducer-dim` | `none` / `none` | e.g. `--reducer pca --reducer-dim none 384 256 128` (`none` = keep the full dimension) |
| `--seeds` | `0` | e.g. `0 1 2` for variance |
| `--dino-source`, `--weights-path`, `--repo-dir`, `--hub-dir` | | backbone loading, as in the notebook |
| `--cache-dir` | – | strongly recommended: the backbone runs once per layer/category |
| `--results` | required | CSV to append to |
| `--tag` | `""` | experiment ID |
| `--no-pro`, `--no-benchmark` | | skip PRO or FPS measurement for faster exploration |
| `--synthetic --no-pretrained` | | smoke test with fake data and random weights |

Long sweeps: a Colab Pro session can stay up for many hours, but keep the browser tab open. Rows
are saved continuously, and features are cached, so you can just rerun the same command
after a disconnect. Finished configurations will be duplicated in the CSV; deduplicate during
analysis (keep the latest row per configuration).

---

## 5. Recipes for each experiment (E0–E8)

`D` is the Drive root as above. Add `--drive-root {D}/mvtec_anomaly_detection --data-root /content/mvtec --cache-dir {D}/feature_cache --hub-dir {D}/torch_hub --results {D}/results/results.csv` to each command (shown as `...`). Run each one with `--domain texture` and again with `--domain object`, or use `--domain all`.

```bash
# E0 – baseline / pipeline check (must reach roughly the published PatchCore numbers before moving on)
python -m dinoad.run ... --domain all --backbone wide_resnet50_2 --layers layer2+layer3 --image-size 224 --coreset 0.1 --tag E0
python -m dinoad.run ... --domain all --backbone resnet50        --layers layer2+layer3 --image-size 224 --coreset 0.1 --tag E0

# E1 – single-layer sweep (core result)
python -m dinoad.run ... --domain all --layers 3 6 7 8 9 12 --coreset 1.0 0.1 --tag E1

# E2 – multi-layer fusion
python -m dinoad.run ... --domain all --layers 6+9 8+12 6+9+12 --coreset 0.1 --tag E2

# E3 – compression sweep (replace 6 8 with the two best layers from E1); fast because features are cached
python -m dinoad.run ... --domain all --layers 6 8 12 --coreset 0.1 0.05 0.01 --reducer random --reducer-dim none 384 256 128 --no-benchmark --tag E3
python -m dinoad.run ... --domain all --layers 6 8 12 --coreset 0.1 0.05 0.01 --reducer pca    --reducer-dim 384 256 128      --no-benchmark --tag E3

# E4 – throughput: ONE session, ONE GPU type, best compression from E3 (example: 5% coreset, PCA 256)
python -m dinoad.run ... --domain all --layers 3 6 8 9 12 --coreset 0.05 --reducer pca --reducer-dim 256 --bench-batch 8 --bench-iters 50 --tag E4
python -m dinoad.run ... --domain all --layers 3 6 8 9 12 --coreset 0.05 --reducer pca --reducer-dim 256 --bench-batch 1 --bench-iters 50 --tag E4_b1

# E5 – resolution
python -m dinoad.run ... --domain all --layers 8 --image-size 224 --coreset 0.1 --tag E5
python -m dinoad.run ... --domain all --layers 8 --image-size 448 --coreset 0.1 --tag E5

# E6 – DINOv3 replication (patch 16 -> image size 448 still works)
python -m dinoad.run ... --domain all --backbone dinov3_vitb16 --repo-dir {D}/dinov3 \
    --weights-path {D}/weights/<dinov3_vitb16_checkpoint>.pth --layers 3 6 7 8 9 12 --coreset 1.0 0.1 --tag E6

# E7 – ablations (examples)
python -m dinoad.run ... --domain all --layers 8 --coreset 0.1 --apply-norm --tag E7_norm
python -m dinoad.run ... --domain all --layers 8 --coreset 0.1 --agg-kernel 1 --tag E7_agg1
python -m dinoad.run ... --domain all --layers 8 --coreset 0.1 --coreset-method random --tag E7_random_coreset
python -m dinoad.run ... --domain all --layers 8 --coreset 0.1 --k-neighbors 3 --tag E7_k3

# E8 – final hypothesis check with 3 seeds (layer-12 baseline vs chosen intermediate config)
python -m dinoad.run ... --domain all --layers 12 --coreset 0.1 --seeds 0 1 2 --tag E8_baseline
python -m dinoad.run ... --domain all --layers 8  --coreset 0.05 --reducer pca --reducer-dim 256 --seeds 0 1 2 --tag E8_candidate
```

In a Colab cell, prefix each command with `!` and use `{D}` for the path (as in section 4).
Note that the `--apply-norm` and `--agg-kernel` ablations write separate cache files, so they
re-extract features.

---

## 6. Use the Python API directly

Everything the notebook and CLI do is available as plain function calls:

```python
import torch, dinoad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = dinoad.load_dinov2("dinov2_vitb14", source="hub", device=device)
extractor = dinoad.build_extractor(model, "dinov2_vitb14", layers=[8])   # hooks block 8, stops there

train_ds = dinoad.MVTecAD("/content/mvtec", "carpet", "train", image_size=448)
test_ds  = dinoad.MVTecAD("/content/mvtec", "carpet", "test",  image_size=448)
train_fs = dinoad.extract_features(extractor, dinoad.make_loader(train_ds), device, keep_masks=False)
test_fs  = dinoad.extract_features(extractor, dinoad.make_loader(test_ds), device)

pc = dinoad.PatchCore(dinoad.PatchCoreConfig(coreset_ratio=0.1), device).fit(train_fs.features)
scores, maps = pc.predict(test_fs.features, image_size=448)

print(dinoad.compute_metrics(test_fs.labels.numpy(), scores, test_fs.masks.numpy(), maps))
print(dinoad.memory_bank_report(pc))
imgs = torch.stack([test_ds[i]["image"] for i in range(8)])
print(dinoad.benchmark_pipeline(extractor, pc, imgs))
```

Or the whole thing in one call:

```python
cfg = dinoad.ExperimentConfig(data_root="/content/mvtec", layers=(8,),
                              patchcore=dinoad.PatchCoreConfig(coreset_ratio=0.1),
                              cache_dir="/content/drive/MyDrive/CMPE297_final_project/feature_cache",
                              tag="E1")
row = dinoad.run_category(cfg, "carpet", model, device)          # one results row (dict)
rows = dinoad.run_domain(cfg, "texture", model, device, results_csv=".../results.csv")
```

### Helper reference

| Helper | Returns / does |
|---|---|
| `get_categories(domain)` | category list for `"texture"`, `"object"`, `"all"` |
| `copy_to_local(src, dst, categories)` | copies folders or extracts `<cat>.tar.xz` from Drive to local disk; skips categories already copied |
| `verify_mvtec_root(root, categories)` | `{cat: {train, test_good, test_anomalous}}`; raises if a folder is missing |
| `MVTecAD(root, category, split, image_size)` | Dataset of dicts: `image`, `mask`, `label`, `defect_type`, `path` |
| `make_loader(ds, batch_size, num_workers)` | unshuffled DataLoader |
| `load_dinov2(name, source, weights_path, repo_dir, hub_dir, pretrained, device)` | frozen DINOv2 in eval mode |
| `load_dinov3(name, repo_dir, weights_path, device)` | frozen DINOv3 |
| `load_cnn(arch, weights_path, device)` | frozen ImageNet ResNet |
| `load_backbone(name, device, **kw)` | dispatches to one of the three loaders above |
| `build_extractor(model, name, layers, apply_norm, truncate)` | `ViTLayerExtractor` / `CNNLayerExtractor`; call it on images to get `{layer: (B, C, h, w)}`; `.compute_fraction` |
| `extract_features(extractor, loader, device, agg_kernel)` | `FeatureSet(features, labels, masks, paths, defect_types)` |
| `feature_cache_path(...)`, `get_or_extract(path, ...)`, `save_features`, `load_features` | feature caching |
| `PatchCoreConfig(...)` | coreset / reducer / dtype / k / blur settings |
| `PatchCore(cfg, device).fit(features)` | builds the memory bank; `.fit_stats` holds timings |
| `PatchCore.predict(features, image_size)` | `(image_scores (N,), anomaly_maps (N, S, S))` |
| `PatchCore.save(path)` / `PatchCore.load(path)` | persist a fitted memory bank |
| `image_auroc`, `pixel_auroc`, `pro_score`, `compute_metrics` | accuracy metrics |
| `memory_bank_report(pc)` | patches, dim, dtype, MB, fp32 MB, bytes/patch, keep ratio |
| `benchmark_pipeline(extractor, pc, images)` | per-image ms per stage, `fps`, `peak_gpu_mb`, `gpu`, `compute_fraction` |
| `benchmark_backbone_only(extractor, images)` | backbone-only latency |
| `append_result(csv, row)`, `load_results(csv)`, `domain_summary(df)` | results log |
| `viz.show_anomaly_maps`, `viz.plot_score_histogram`, `viz.plot_tradeoff` | figures |

Every function has a docstring; `help(dinoad.PatchCore)` and the like work in Colab.

---

## 7. Outputs: results CSV, feature cache, figures

**`results/results.csv`**: one row per (configuration, category), with these columns:

| Group | Columns |
|---|---|
| identity | `timestamp`, `tag`, `category`, `domain`, `backbone`, `layers`, `image_size`, `agg_kernel`, `apply_norm` |
| features | `feature_grid` (e.g. `32x32`), `feature_dim` |
| PatchCore config | `pc_coreset_ratio`, `pc_coreset_method`, `pc_reducer`, `pc_reducer_dim`, `pc_bank_dtype`, `pc_k_neighbors`, `pc_blur_sigma`, `pc_seed`, ... |
| **accuracy** | `image_auroc`, `pixel_auroc`, `pro` |
| **storage** | `bank_patches`, `bank_dim`, `bank_dtype`, `bank_mb`, `bank_mb_fp32`, `reducer_mb`, `bytes_per_patch`, `train_patches`, `keep_ratio` |
| fit timing | `reduce_seconds`, `coreset_seconds`, `fit_seconds`, `score_seconds_total` |
| **speed** | `backbone_ms`, `postproc_ms`, `knn_ms`, `map_ms`, `total_ms` (all per image), `fps`, `batch_size`, `gpu`, `peak_gpu_mb`, `compute_fraction` |

**`feature_cache/<backbone>/<image_size>/<category>/<split>_L<layers>_k<agg>[_norm].pt`**: fp16
patch features, for example `feature_cache/dinov2_vitb14/448/carpet/train_L8_k3.pt`. At 448 px,
budget about 1.5 MB per image per 768-d layer, or roughly 3 GB per layer for all six categories.
Delete layers you no longer need.

**`outputs/`**: heatmap figures from the notebook (`<category>_<backbone>_L<layers>_heatmaps.png`)
and trade-off plots.

---

## 8. Analyse results

```python
import dinoad
from dinoad.viz import plot_tradeoff
df = dinoad.load_results(".../results/results.csv")
df = df.drop_duplicates(subset=["tag", "category", "backbone", "layers", "image_size",
                                "pc_coreset_ratio", "pc_reducer", "pc_reducer_dim", "pc_seed"], keep="last")

dinoad.domain_summary(df[df.tag == "E1"])                                    # mean per config x domain
plot_tradeoff(df[df.tag == "E3"], x="bank_mb", y="pixel_auroc", hue="layers")   # accuracy vs storage
plot_tradeoff(df[df.tag == "E4"], x="fps", y="pixel_auroc", hue="layers", logx=False)
```

Only compare FPS between rows with the same `gpu` value.

---

## 9. Local development and tests

```bash
pip install -e ".[dev]"
pytest -q          # 7 tests, CPU only, ~5 s; uses synthetic data and a tiny fake ViT
```

Smoke-test the whole pipeline (fake data, random weights) without a GPU:

```bash
python -m dinoad.run --synthetic --data-root /tmp/fake --domain texture --layers 6 \
    --image-size 224 --no-pretrained --num-workers 0 --results /tmp/results.csv
```

Team workflow: work on branches, open pull requests, and run `pytest -q` before merging. Never
commit data, caches, weights or results (`.gitignore` already excludes them); those live on Drive.

---

## 10. Design notes

* **Layer numbering.** ViT layers are 1-indexed block numbers (ViT-B: 1..12, where 12 = last).
* **Hooks + truncation.** `ViTLayerExtractor` attaches forward hooks only while it runs and
  stops the forward pass right after the deepest requested block. Layer 6 executes 6 of 12
  blocks, so shallow layers really are faster. Its output matches DINOv2's own
  `get_intermediate_layers` exactly, with and without the final norm.
* **Storage vs. layer.** Every ViT-B block outputs 768-d tokens, so **layer choice alone does
  not shrink the memory bank.** Storage shrinks only through `coreset_ratio`,
  `reducer`/`reducer_dim`, `bank_dtype` (fp16 by default) and image resolution. See
  `docs/EXPERIMENTS.md` for how the hypothesis is tested with this in mind.
* **Preprocessing.** Images are resized (no centre crop) to `image_size`, then ImageNet-normalised.
  Masks are resized with nearest-neighbour interpolation.
* **Patch features.** 3×3 neighbourhood average (`AGG_KERNEL`). Multiple layers are resized to
  the finest grid and concatenated.
* **Coreset.** Greedy k-center (PatchCore's approximate version). Distances for selection are
  computed in a 128-d random projection; the stored bank keeps the full (or reduced) features.
* **Scoring.** Patch score = Euclidean distance to the nearest bank entry (mean of k if
  `k_neighbors > 1`). Image score = max patch score. Anomaly map = bilinear upsampling + Gaussian
  blur (σ = 4). PatchCore's optional score re-weighting is not implemented.
* **PRO.** MVTec definition: 8-connected regions, integrated up to FPR = 0.3, normalised to [0, 1].
  The implementation is histogram-based and fast at 448 px. Sanity values: random ≈ 0.15, perfect = 1.0.
* **Timing.** Warmup runs, median of N iterations, `torch.cuda.synchronize()` around each stage,
  fp16 autocast for the backbone on GPU.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ... carpet/ not found` | `MVTEC_DRIVE` / `--drive-root` must point at the folder that directly contains `carpet/`. Check the Drive shortcut (section 2.2). |
| Data loading is very slow | You're reading from Drive. Make sure `copy_to_local` ran and that `DATA_ROOT` is `/content/mvtec`. |
| `image size ... must be a multiple of patch size` | Use 224 or 448 for DINOv2 (patch 14) and DINOv3 (patch 16). |
| torch.hub error / GitHub rate limit (HTTP 403) | Set `HUB_DIR` to Drive so it only downloads once, or `git clone https://github.com/facebookresearch/dinov2` to Drive and set `DINO_REPO_DIR` / `--repo-dir`. |
| CUDA out of memory | Lower `BATCH_SIZE`; lower `PatchCoreConfig(query_chunk=..., bank_chunk=...)`; use `coreset_ratio < 1.0` for 448 px full banks; use 224 px for the ResNet baseline. |
| Colab disconnected mid-sweep | Rerun the same command. Cached features are reused, and finished rows are already in the CSV. |
| Different FPS between sessions | Colab assigned a different GPU. Filter by the `gpu` column, and run E4 in one session. |
| `ModuleNotFoundError: dinoad` | Rerun the setup cell. If it still fails, restart the runtime and run all. |
| Changed code isn't picked up | `git pull` (setup cell), then **Runtime → Restart session**. |
| Numbers far below published PatchCore in E0 | Check that the masks load (`verify_mvtec_root` warns about missing masks), that pretrained weights loaded (`DINO_PRETRAINED=True`, no `weights=None`), and the image size. |

---

## 12. What has been verified

Verified in development (CPU, no internet access to the weight servers):

* `pytest`: dataset indexing, hook extraction and truncation, a manual-forward equivalence
  check, greedy coreset coverage, kNN vs. `torch.cdist`, PRO sanity values, and an end-to-end
  pipeline run with all three reducers.
* With the **real DINOv2 ViT-B/14 code** (random weights): hook outputs at blocks 3/6/9/12 match
  `get_intermediate_layers` exactly; at 448 px, stopping at block 6 halves backbone time compared to block 12.
* Both notebook domains, the command-line runner, CSV logging, the feature cache, the summary and
  plots all ran end to end on synthetic data.

**Not yet verified:** pretrained weights, real MVTec data, GPU execution, the DINOv3 loader, and
the torchvision ImageNet weight download. Run E0 first to cover these.
