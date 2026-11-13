"""dinoad – which DINO layer is best for fast PatchCore anomaly detection?

Modules
-------
data        MVTec AD dataset, texture/object domains, Drive -> local copy
backbones   DINOv2 / DINOv3 / ResNet loading + forward-hook layer extractors
features    neighbourhood aggregation, layer fusion, feature caching
patchcore   memory bank, greedy coreset, random-projection/PCA, kNN scoring
metrics     image AUROC, pixel AUROC, PRO
efficiency  memory-bank footprint, per-stage latency / FPS
experiment  run_category / run_domain convenience runners
run         command-line sweep runner: python -m dinoad.run --help
results     append-only CSV log + summaries
viz         heatmaps and trade-off plots
synthetic   fake MVTec-style data for smoke tests
"""
from .data import (DOMAINS, OBJECT_CATEGORIES, TEXTURE_CATEGORIES, MVTecAD, copy_to_local,
                   domain_of, get_categories, make_loader, verify_mvtec_root)
from .backbones import (CNNLayerExtractor, ViTLayerExtractor, build_extractor, load_backbone,
                        load_cnn, load_dinov2, load_dinov3)
from .features import (FeatureSet, embed_batch, extract_features, feature_cache_path,
                       get_or_extract, load_features, save_features)
from .patchcore import (PatchCore, PatchCoreConfig, greedy_coreset_indices, make_reducer,
                        random_coreset_indices)
from .metrics import compute_metrics, image_auroc, pixel_auroc, pro_score
from .efficiency import (benchmark_backbone_only, benchmark_pipeline, gpu_name,
                         memory_bank_report)
from .experiment import ExperimentConfig, run_category, run_domain
from .results import append_result, domain_summary, load_results

__version__ = "0.1.0"
