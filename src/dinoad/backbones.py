"""Backbone loading and intermediate-layer feature extraction.

* :func:`load_dinov2` – DINOv2 from torch.hub (public) or from weights on Drive.
* :func:`load_dinov3` – DINOv3 from a local clone of its repo + gated weights file.
* :func:`load_cnn`    – ImageNet ResNet-50 / WideResNet-50 baseline (PatchCore).
* :class:`ViTLayerExtractor` – PyTorch forward hooks on transformer blocks that
  return spatial patch-token maps, and (optionally) *stop the forward pass* right
  after the deepest requested block so shallower layers are genuinely cheaper.
* :class:`CNNLayerExtractor` – same interface for ResNet stages.

Layer numbering for ViTs is **1-indexed block number**: for ViT-B (12 blocks),
layer 12 is the last block, layer 6 is the middle.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

DINOV2_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024,
               "dinov2_vitg14": 1536}


def _finalize(model: nn.Module, device: str | torch.device) -> nn.Module:
    model.eval().requires_grad_(False)
    return model.to(device)


def load_dinov2(model_name: str = "dinov2_vitb14", source: str = "hub",
                weights_path: str | Path | None = None, repo_dir: str | Path | None = None,
                hub_dir: str | Path | None = None, pretrained: bool = True,
                device: str | torch.device = "cpu") -> nn.Module:
    """Load a DINOv2 backbone.

    Args:
        model_name: e.g. ``"dinov2_vitb14"`` (ViT-B/14, 12 blocks, 768-d tokens).
        source:
            * ``"hub"``   – ``torch.hub.load("facebookresearch/dinov2", ...)``; code from
              GitHub and weights from Meta's public URL. Pass ``hub_dir`` (e.g. a Drive
              folder) to cache both there so later sessions skip the download.
            * ``"drive"`` – architecture from the hub (or ``repo_dir``), weights from
              ``weights_path`` (e.g. ``/content/drive/MyDrive/.../dinov2_vitb14_pretrain.pth``).
        weights_path: ``.pth`` state dict, required when ``source="drive"``.
        repo_dir: local clone of github.com/facebookresearch/dinov2 (optional; avoids
            needing GitHub at load time).
        hub_dir: torch.hub cache directory.
        pretrained: set False for random weights (only for testing the pipeline).
    """
    if hub_dir is not None:
        torch.hub.set_dir(str(hub_dir))
    repo = str(repo_dir) if repo_dir else "facebookresearch/dinov2"
    hub_source = "local" if repo_dir else "github"

    if source == "hub":
        model = torch.hub.load(repo, model_name, source=hub_source, pretrained=pretrained)
    elif source == "drive":
        if weights_path is None:
            raise ValueError("source='drive' needs weights_path")
        model = torch.hub.load(repo, model_name, source=hub_source, pretrained=False)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError("source must be 'hub' or 'drive'")
    return _finalize(model, device)


def load_dinov3(model_name: str = "dinov3_vitb16", repo_dir: str | Path | None = None,
                weights_path: str | Path | None = None,
                device: str | torch.device = "cpu") -> nn.Module:
    """Load DINOv3 (weights are gated: request access from Meta first).

    Follows the DINOv3 README: clone github.com/facebookresearch/dinov3 (e.g. to
    Drive) and pass the downloaded checkpoint path. Note DINOv3 ViTs use patch size
    16, so pick image sizes that are multiples of 16 (224, 448).
    """
    if repo_dir is None or weights_path is None:
        raise ValueError("load_dinov3 needs repo_dir (local clone) and weights_path")
    model = torch.hub.load(str(repo_dir), model_name, source="local", weights=str(weights_path))
    return _finalize(model, device)


def load_cnn(arch: str = "wide_resnet50_2", weights_path: str | Path | None = None,
             device: str | torch.device = "cpu") -> nn.Module:
    """ImageNet-pretrained CNN baseline. Original PatchCore used ``wide_resnet50_2``;
    the project proposal lists ``resnet50`` – both work."""
    import torchvision.models as tvm
    if weights_path is not None:
        model = getattr(tvm, arch)(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    else:
        model = getattr(tvm, arch)(weights="DEFAULT")
    return _finalize(model, device)


def load_backbone(name: str, device: str | torch.device = "cpu", **kwargs) -> nn.Module:
    """Dispatch on name: ``dinov2_*``, ``dinov3_*`` or a torchvision CNN name."""
    if name.startswith("dinov2"):
        return load_dinov2(name, device=device, **kwargs)
    if name.startswith("dinov3"):
        return load_dinov3(name, device=device, **kwargs)
    return load_cnn(name, device=device, **kwargs)


# --------------------------------------------------------------------------- #
# Feature extractors
# --------------------------------------------------------------------------- #
class _StopForward(Exception):
    """Raised from a hook to end the forward pass early."""


def get_patch_size(model: nn.Module) -> int:
    ps = getattr(model, "patch_size", None)
    if ps is None:
        ps = model.patch_embed.patch_size
    return int(ps[0] if isinstance(ps, (tuple, list)) else ps)


def flatten_blocks(model: nn.Module) -> list[nn.Module]:
    """Transformer blocks in order (handles DINOv2's optional BlockChunk wrapping)."""
    blocks = []
    for b in model.blocks:
        if isinstance(b, nn.ModuleList):           # BlockChunk
            blocks.extend(m for m in b if not isinstance(m, nn.Identity))
        else:
            blocks.append(b)
    return blocks


class ViTLayerExtractor(nn.Module):
    """Return patch-token maps from chosen ViT blocks via forward hooks.

    ``forward(x)`` returns ``{layer: tensor (B, C, H/p, W/p)}`` with CLS / register
    tokens removed. Hooks are attached only for the duration of each forward call,
    so several extractors can share one backbone safely.

    Args:
        model: DINOv2 / DINOv3 ViT.
        layers: 1-indexed block numbers, e.g. ``[6]`` or ``[6, 9, 12]``.
        apply_norm: apply the backbone's final LayerNorm to intermediate tokens
            (what DINOv2's ``get_intermediate_layers(norm=True)`` does). Worth
            ablating; raw intermediate tokens have very different scales per layer.
        truncate: stop the forward pass after the deepest requested block. This is
            what makes shallow layers faster – keep it True for throughput numbers.
    """

    def __init__(self, model: nn.Module, layers: list[int] | tuple[int, ...],
                 apply_norm: bool = False, truncate: bool = True):
        super().__init__()
        self.model = model
        self._blocks = flatten_blocks(model)       # plain list, not registered twice
        self.num_blocks = len(self._blocks)
        self.layers = sorted({int(l) for l in layers})
        bad = [l for l in self.layers if not 1 <= l <= self.num_blocks]
        if bad:
            raise ValueError(f"layers {bad} out of range 1..{self.num_blocks}")
        self.patch_size = get_patch_size(model)
        self.apply_norm = apply_norm
        self.truncate = truncate

    @property
    def compute_fraction(self) -> float:
        """Fraction of transformer blocks executed per image."""
        return (self.layers[-1] if self.truncate else self.num_blocks) / self.num_blocks

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        B, _, H, W = x.shape
        p = self.patch_size
        if H % p or W % p:
            raise ValueError(f"image size {H}x{W} must be a multiple of patch size {p}")
        h, w = H // p, W // p
        buf: dict[int, torch.Tensor] = {}
        deepest = self.layers[-1]

        def make_hook(layer: int):
            def hook(_module, _inp, out):
                if isinstance(out, (list, tuple)):   # DINOv3 blocks may pass lists
                    out = out[0]
                buf[layer] = out
                if self.truncate and layer == deepest:
                    raise _StopForward
            return hook

        handles = [self._blocks[l - 1].register_forward_hook(make_hook(l)) for l in self.layers]
        try:
            fwd = getattr(self.model, "forward_features", self.model)
            fwd(x)
        except _StopForward:
            pass
        finally:
            for hd in handles:
                hd.remove()

        out = {}
        for l in self.layers:
            t = buf[l]
            if self.apply_norm:
                t = self.model.norm(t)
            t = t[:, -h * w:, :]                     # drop prefix (CLS/register) tokens
            out[l] = t.transpose(1, 2).reshape(B, t.shape[-1], h, w)
        return out


class CNNLayerExtractor(nn.Module):
    """ResNet-stage extractor with the same interface (baseline PatchCore uses
    ``layer2`` + ``layer3``)."""

    STAGES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, model: nn.Module, layers=("layer2", "layer3"), truncate: bool = True):
        super().__init__()
        self.model = model
        self.layers = [l for l in self.STAGES if l in set(layers)]
        if not self.layers:
            raise ValueError(f"layers must be a subset of {self.STAGES}")
        self.truncate = truncate
        self.patch_size = 1   # no constraint on image size

    @property
    def compute_fraction(self) -> float:
        last = self.STAGES.index(self.layers[-1]) + 1 if self.truncate else 4
        return last / 4

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        out = {}
        for name in self.STAGES:
            x = getattr(m, name)(x)
            if name in self.layers:
                out[name] = x
            if self.truncate and name == self.layers[-1]:
                break
        return out


def build_extractor(model: nn.Module, backbone_name: str, layers, apply_norm: bool = False,
                    truncate: bool = True) -> nn.Module:
    """Pick the right extractor class for a backbone name."""
    if backbone_name.startswith("dino"):
        return ViTLayerExtractor(model, layers, apply_norm=apply_norm, truncate=truncate)
    return CNNLayerExtractor(model, layers, truncate=truncate)
