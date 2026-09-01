"""
PatchCore anomaly scoring for spectrogram patch features.

Input:  patches, shape (N, 450, 192)  -- N tiles, 15x30 grid, 192-d descriptors
Output: per-tile scores (N,) and per-patch score maps (N, 15, 30)

Requires: numpy, torch. faiss optional but much faster for large banks.
"""

import numpy as np
import torch
from pathlib import Path
import time

GRID_T, GRID_F = 15, 30
DIM_L1 = 64  # first 64 channels are layer1, remainder layer2
DATA_DIR = Path("/home/michelle/BigData/BLUSE")
OUT_DIR = DATA_DIR / "features"
FEATURES_PATH = OUT_DIR / "patch_features.npy"
META_PATH = OUT_DIR / "patch_features_meta.json"
DATA_PATH = DATA_DIR / "mk_sample_hits.h5"


# ---------------------------------------------------------------- preprocessing

def normalise(patches):
    """L2-normalise layer1 and layer2 blocks separately, then the whole vector.

    The two blocks have different activation scales; normalising jointly lets
    layer2 dominate the distance for no good reason.
    """
    l1 = patches[..., :DIM_L1]
    l2 = patches[..., DIM_L1:]
    l1 = l1 / (np.linalg.norm(l1, axis=-1, keepdims=True) + 1e-8)
    l2 = l2 / (np.linalg.norm(l2, axis=-1, keepdims=True) + 1e-8)
    x = np.concatenate([l1, l2], axis=-1)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def standardise(patches, eps=1e-8):
    """Per-channel z-scoring across the whole dataset.

    Equalises the two layer blocks (layer1 and layer2 activations sit on
    different scales) without destroying within-patch magnitude, which is
    what distinguishes a signal patch from empty background.
    """
    flat = patches.reshape(-1, patches.shape[-1])
    mu = flat.mean(axis=0, keepdims=True)
    sd = flat.std(axis=0, keepdims=True) + eps
    return ((flat - mu) / sd).reshape(patches.shape)

# ---------------------------------------------------------------- coreset

def greedy_coreset(X, n_select, projection_dim=64, seed=0, device="cuda"):
    """Greedy furthest-point subsampling (k-center greedy).

    Picks points that maximise minimum distance to those already chosen, so the
    memory bank covers the feature manifold rather than over-representing dense
    regions. Random projection first, as in the paper -- it barely changes the
    selection and makes the distance updates much cheaper.
    """
    print("Running greedy_coreset")
    rng = np.random.default_rng(seed)
    dev = device if torch.cuda.is_available() else "cpu"

    Xt = torch.from_numpy(X).float().to(dev)
    P = torch.from_numpy(
        rng.normal(size=(X.shape[1], projection_dim)).astype(np.float32)
    ).to(dev)
    Z = Xt @ P

    idx = [int(rng.integers(len(Z)))]
    min_d = torch.cdist(Z, Z[idx[0]:idx[0] + 1]).squeeze(1)

    t1 = time.perf_counter()
    for i in range(n_select - 1):
        if i%100 == 0:
            print(f"Completed {i} of {n_select-1} in {(time.perf_counter()-t1)/60:.2f}min")
        nxt = int(torch.argmax(min_d))
        idx.append(nxt)
        d = torch.cdist(Z, Z[nxt:nxt + 1]).squeeze(1)
        min_d = torch.minimum(min_d, d)

    print("Completed greedy_coreset")
    return np.array(idx)


# ---------------------------------------------------------------- scoring

# def knn_scores(queries, bank, k=1, batch=8192, device="cuda"):
#     """Mean cosine distance to the k nearest bank entries.

#     Inputs must already be L2-normalised, so inner product is cosine similarity
#     and distance is 1 - similarity.
#     """
#     dev = device if torch.cuda.is_available() else "cpu"
#     B = torch.from_numpy(bank).float().to(dev)
#     out = np.empty(len(queries), dtype=np.float32)

#     for i in range(0, len(queries), batch):
#         Q = torch.from_numpy(queries[i:i + batch]).float().to(dev)
#         sim = Q @ B.T
#         topk = torch.topk(sim, k=k, dim=1).values
#         out[i:i + batch] = (1.0 - topk.mean(dim=1)).cpu().numpy()

#     return out

def knn_scores(queries, bank, k=1, batch=8192, device="cuda"):
    """Mean Euclidean distance to the k nearest bank entries.

    Inputs should be per-channel standardised, NOT per-patch normalised.
    Magnitude carries the signal here, so cosine throws away what we want.
    """
    dev = device if torch.cuda.is_available() else "cpu"
    B = torch.from_numpy(bank).float().to(dev)
    out = np.empty(len(queries), dtype=np.float32)

    for i in range(0, len(queries), batch):
        Q = torch.from_numpy(queries[i:i + batch]).float().to(dev)
        d = torch.cdist(Q, B)
        topk = torch.topk(d, k=k, dim=1, largest=False).values
        out[i:i + batch] = topk.mean(dim=1).cpu().numpy()

    return out

def tile_scores(patch_scores, top_k=5):
    """Aggregate per-patch scores to one score per tile.

    Mean of the top-k patches. Plain max is the PatchCore default but is a
    single-patch statistic and noisy; top-5 is more stable and still sensitive
    to a tile containing one unusual region.
    """
    s = np.sort(patch_scores, axis=1)[:, ::-1]
    return s[:, :top_k].mean(axis=1)


# ---------------------------------------------------------------- main

def run(patches, coreset_frac=0.01, k=1, top_k=5, seed=0):
    N = patches.shape[0]

    # X = normalise(patches)                      # (N, 450, 192)
    X = standardise(patches)                      # (N, 450, 192)
    flat = X.reshape(-1, X.shape[-1])           # (N*450, 192)

    n_select = max(1000, int(len(flat) * coreset_frac))
    print(f"{len(flat):,} patches -> selecting {n_select:,} for the bank")

    sel = greedy_coreset(flat, n_select, seed=seed)
    bank = flat[sel]

    scores = knn_scores(flat, bank, k=k).reshape(N, GRID_T * GRID_F)

    return tile_scores(scores, top_k), scores.reshape(N, GRID_T, GRID_F)


if __name__ == "__main__":
    patches = np.load(FEATURES_PATH)            # (N, 450, 192)
    tile, maps = run(patches, coreset_frac=0.01)

    np.save(OUT_DIR / "tile_scores.npy", tile)
    np.save(OUT_DIR / "score_maps.npy", maps)

    print(f"tile scores: mean {tile.mean():.4f}  max {tile.max():.4f}")
    print("top 10 tiles:", np.argsort(tile)[::-1][:10])

    # Diagnostic: mean score by grid position. Should be roughly flat. If the
    # centre frequency column lights up, the scores are being driven by the
    # always-present central signal rather than by anything anomalous.
    print("mean score by frequency column:")
    print(np.round(maps.mean(axis=(0, 1)), 4))