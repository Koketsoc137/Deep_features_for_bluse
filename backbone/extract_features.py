#!/usr/bin/env python3
"""Patch-level feature extraction from radio spectrogram cutouts.

Frozen ImageNet ResNet18, layer1 + layer2 only, PatchCore-style local
aggregation. No training, no memory bank, no scoring.

Input tiles are fed at native 57x120 -- ResNet is fully convolutional and the
classifier head is never touched, so the 224x224 convention does not apply.

Usage:
    python extract_features.py /path/to/mk_sample_hits.h5

writes mk_sample_hits_resnet18_intermediate_features.npy and its _meta.json
alongside the input file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

BATCH_SIZE = 256
MAD_EPS = 1e-6
CLIP = 10.0
LAYERS = ("layer1", "layer2")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUFFIX = "_resnet18_intermediate_features"


def normalise_tiles(x):
    """Per-tile robust normalisation: (x - median) / MAD, clipped. x: (B, H, W)."""
    flat = x.flatten(1)
    med = flat.median(dim=1, keepdim=True).values
    mad = (flat - med).abs().median(dim=1, keepdim=True).values.clamp(min=MAD_EPS)
    return ((flat - med) / mad).clamp(-CLIP, CLIP).view_as(x)


def to_model_input(x, mean, std):
    """(B, H, W) -> (B, 3, H, W): replicate to 3 channels, ImageNet normalise."""
    return (x.unsqueeze(1).expand(-1, 3, -1, -1) - mean) / std


def build_backbone(device):
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def forward_features(model, x):
    """Forward-slice the stem + layer1/layer2 only; layer3/4 are never run."""
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    f1 = model.layer1(x)
    f2 = model.layer2(f1)
    return {"layer1": f1, "layer2": f2}


def aggregate(feats):
    """Local avg-pool, upsample layer2 onto the layer1 grid, concat channels."""
    pooled = {k: F.avg_pool2d(v, kernel_size=3, stride=1, padding=1)
              for k, v in feats.items()}
    grid = pooled["layer1"].shape[-2:]
    up = F.interpolate(pooled["layer2"], size=grid, mode="bilinear", align_corners=False)
    return torch.cat([pooled["layer1"], up], dim=1)


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("input", type=Path, help="input .h5 file of cutouts")
parser.add_argument("--dataset", default="data",
                    help="dataset key inside the input file (default: data)")
parser.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: alongside the input file)")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
args = parser.parse_args()

out_dir = args.out_dir if args.out_dir is not None else args.input.parent
features_path = out_dir / f"{args.input.stem}{SUFFIX}.npy"
meta_path = out_dir / f"{args.input.stem}{SUFFIX}_meta.json"

# --------------------------------------------------------------------------
# >>> your load line here: `data` must be float32, shape (N, 57, 120)
with h5py.File(args.input, "r") as fin:
    data = fin[args.dataset][:]
data = data.squeeze()
# --------------------------------------------------------------------------

data = np.ascontiguousarray(data, dtype=np.float32)
n_tiles, tile_h, tile_w = data.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"input:  {args.input}")
print(f"device: {device}   tiles: {n_tiles}  ({tile_h}x{tile_w})")

model = build_backbone(device)
mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

# --- sanity check on a couple of tiles before committing to the full set ---
with torch.no_grad():
    probe = torch.from_numpy(data[: min(4, n_tiles)]).to(device)
    probe = to_model_input(normalise_tiles(probe), mean, std)
    print(f"  input          {tuple(probe.shape)}")
    probe_feats = forward_features(model, probe)
    for name in LAYERS:
        print(f"  {name:<14} {tuple(probe_feats[name].shape)}")
    probe_out = aggregate(probe_feats)
    print(f"  concatenated   {tuple(probe_out.shape)}")

_, feat_dim, grid_h, grid_w = probe_out.shape
n_patches = grid_h * grid_w
n_bytes = n_tiles * n_patches * feat_dim * 4
print(f"  -> {n_patches} patches x {feat_dim} dims per tile "
      f"({n_bytes / 1e9:.2f} GB on disk)")

meta = {
    "feature_type": "intermediate",
    "source_file": str(args.input),
    "feature_dim": feat_dim,
    "grid_shape": [grid_h, grid_w],
    "grid_axes": ["time", "frequency"],
    "num_patches": n_patches,
    "layers": list(LAYERS),
    "layer_channels": {k: int(probe_feats[k].shape[1]) for k in LAYERS},
    "input_shape": [tile_h, tile_w],
    "output_shape": [n_tiles, n_patches, feat_dim],
    "normalisation": {"per_tile": "median/MAD", "mad_eps": MAD_EPS, "clip": CLIP,
                      "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD},
    "patch_index": {
        "order": "row-major: patch = time_idx * grid_w + freq_idx",
        "time_idx": f"patch // {grid_w}",
        "freq_idx": f"patch % {grid_w}",
        "backbone_stride": 4,
        "pixel_centre_approx": "time_px = time_idx * 4 + 1.5, freq_px = freq_idx * 4 + 1.5",
        "map": [[p // grid_w, p % grid_w] for p in range(n_patches)],
    },
}

# --- full pass; written straight to the .npy to avoid holding it all in RAM ---
out_dir.mkdir(parents=True, exist_ok=True)
patches = np.lib.format.open_memmap(
    features_path, mode="w+", dtype=np.float32,
    shape=(n_tiles, n_patches, feat_dim),
)

with torch.no_grad():
    for start in tqdm(range(0, n_tiles, args.batch_size),
                      desc="extracting", unit="batch"):
        stop = min(start + args.batch_size, n_tiles)
        batch = torch.from_numpy(data[start:stop]).to(device, non_blocking=True)
        batch = to_model_input(normalise_tiles(batch), mean, std)
        out = aggregate(forward_features(model, batch))
        # (B, C, H, W) -> (B, H*W, C), row-major: patch = time_idx * grid_w + freq_idx
        patches[start:stop] = out.flatten(2).permute(0, 2, 1).cpu().numpy()

patches.flush()

meta_path.write_text(json.dumps(meta, indent=2))
print(f"wrote {features_path}")
print(f"wrote {meta_path}")
