#!/usr/bin/env python3
"""Penultimate-layer feature extraction from radio spectrogram cutouts.

Frozen ImageNet ResNet18, full backbone through layer4 followed by global
average pooling -- the standard 512-d descriptor used for clustering and
classification. One vector per tile, no spatial grid.

Companion to extract_features.py, which keeps patch-level features from
layer1+layer2 instead. Same input, same normalisation, different head, so the
two feature sets are directly comparable.

Input tiles are fed at native 57x120. Note that layer3/layer4 have nominal
receptive fields larger than the tile, so those activations are substantially
determined by zero-padding -- see the shape printout below.

Usage:
    python extract_features_penultimate.py /path/to/mk_sample_hits.h5

writes mk_sample_hits_resnet18_penultimate_features.npy and its _meta.json
alongside the input file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
import torch
import torchvision
from tqdm import tqdm

BATCH_SIZE = 256
MAD_EPS = 1e-6
CLIP = 10.0
LAYERS = ("layer1", "layer2", "layer3", "layer4")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUFFIX = "_resnet18_penultimate_features"


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


def forward_stages(model, x):
    """Run the stem and all four stages; return each stage's feature map."""
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    feats = {}
    for name in LAYERS:
        x = getattr(model, name)(x)
        feats[name] = x
    return feats


def penultimate(model, x):
    """Global average pool over layer4 -> (B, 512), the pre-classifier vector."""
    f4 = forward_stages(model, x)["layer4"]
    return torch.flatten(model.avgpool(f4), 1)


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
    probe_feats = forward_stages(model, probe)
    for name in LAYERS:
        print(f"  {name:<14} {tuple(probe_feats[name].shape)}")
    probe_out = torch.flatten(model.avgpool(probe_feats["layer4"]), 1)
    print(f"  avgpool        {tuple(probe_out.shape)}")

feat_dim = probe_out.shape[1]
n_bytes = n_tiles * feat_dim * 4
print(f"  -> {feat_dim} dims per tile ({n_bytes / 1e6:.1f} MB on disk)")

# --- full pass; written straight to the .npy for consistency with the patch run ---
out_dir.mkdir(parents=True, exist_ok=True)
feats = np.lib.format.open_memmap(
    features_path, mode="w+", dtype=np.float32, shape=(n_tiles, feat_dim),
)

with torch.no_grad():
    for start in tqdm(range(0, n_tiles, args.batch_size),
                      desc="extracting", unit="batch"):
        stop = min(start + args.batch_size, n_tiles)
        batch = torch.from_numpy(data[start:stop]).to(device, non_blocking=True)
        batch = to_model_input(normalise_tiles(batch), mean, std)
        feats[start:stop] = penultimate(model, batch).cpu().numpy()

feats.flush()

meta = {
    "feature_type": "penultimate",
    "source_file": str(args.input),
    "feature_dim": feat_dim,
    "source_layer": "layer4 -> global average pool (pre-fc)",
    "stage_shapes": {k: list(probe_feats[k].shape[1:]) for k in LAYERS},
    "input_shape": [tile_h, tile_w],
    "output_shape": list(feats.shape),
    "normalisation": {"per_tile": "median/MAD", "mad_eps": MAD_EPS, "clip": CLIP,
                      "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD},
    "note": ("One descriptor per tile; all spatial information is pooled away. "
             "Row i corresponds to row i of the intermediate feature file and to "
             "row i of the hits table."),
}
meta_path.write_text(json.dumps(meta, indent=2))
print(f"wrote {features_path}")
print(f"wrote {meta_path}")
