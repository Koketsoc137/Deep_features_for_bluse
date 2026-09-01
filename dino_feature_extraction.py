#!/usr/bin/env python3
"""
DINO feature extraction script for BLUSE radio spectral dataset.

Loads DINOv3 ViT-G/14 model (largest), extracts features from H5 spectral data,
and saves features with IDs as a parquet file.

Usage:
    python dino_feature_extraction.py [h5_file] [output_dir]

Default:
    h5_file: C:/Users/koket/Desktop/BLUSE/data/mk_sample_hits.h5
    output_dir: C:/Users/koket/Desktop/BLUSE/features/

Requires DINOv3 weights file. Set DINOV3_WEIGHTS path below.
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import h5py

# Add backbone module to path
base_dir = Path(__file__).parent
backbone_dir = base_dir / 'backbone'
if backbone_dir.exists():
    sys.path.insert(0, str(base_dir))
    sys.path.insert(0, str(backbone_dir))

from BLUSE_dataset import ImageDataset


def load_model(device='cuda', weights_path=None):
    """Load DINOv3 ViT-G/14 model."""
    print('Loading DINOv3 ViT-G/14...')
    weights_path = Path('/path/to/dinov3_vitg14_weights.pth')  # Edit this path
    model = torch.hub.load(str(weights_path.parent), 'dinov3_vitg14', source='local', weights=str(weights_path))
    model = model.to(device).eval()
    print(f'Model loaded from: {weights_path}')
    return model


def load_data(h5_path):
    """Load data from H5 file and normalize."""
    print(f'Loading data from {h5_path}...')
    h5_path = Path(h5_path)
    dataset = ImageDataset(str(h5_path))
    print(f'Dataset size: {len(dataset)} samples')
    
    all_tensors = []
    for i in range(len(dataset)):
        all_tensors.append(dataset[i])
    
    all_data = torch.stack(all_tensors)
    print(f'Data shape: {all_data.shape}')
    
    # Normalize each image individually
    all_data = torch.stack([
        (img - img.min()) / (img.max() - img.min())
        for img in all_data
    ])
    
    # Convert to RGB by repeating channels
    all_rgb = all_data.repeat(1, 3, 1, 1)
    print(f'RGB data shape: {all_rgb.shape}')
    
    return all_rgb, h5_path


def extract_features(model, data, batch_size=32, input_size=224):
    """Extract DINO features from data."""
    device = next(model.parameters()).device
    all_features = []
    
    print(f'Extracting features (batch_size={batch_size})...')
    for batch_idx in range(0, len(data), batch_size):
        batch = data[batch_idx:batch_idx+batch_size]
        batch = batch.float().to(device)
        batch = F.interpolate(batch, size=(input_size, input_size),
                              mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, dict):
                out = out['x_norm_clstoken']
            all_features.append(out.cpu().numpy())
        
        current_batch = batch_idx // batch_size + 1
        total_batches = (len(data) - 1) // batch_size + 1
        print(f'  Batch {current_batch}/{total_batches}')
    
    features = np.concatenate(all_features, axis=0)
    print(f'Features shape: {features.shape}')
    return features


def save_features(features, h5_path, output_dir):
    """Save features as parquet with IDs from H5 file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load IDs from H5 file
    with h5py.File(h5_path, 'r') as f:
        ids = f['id'][:]
    
    # Create DataFrame
    features_df = pd.DataFrame(index=ids, data=features)
    features_df.index.name = 'id'
    
    # Save
    h5_basename = Path(h5_path).stem
    output_file = output_dir / f'{h5_basename}_dino_features.parquet'
    features_df.to_parquet(output_file)
    print(f'Features saved to: {output_file}')
    print(f'Features DataFrame shape: {features_df.shape}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('h5_file', nargs='?',
                        default='C:/Users/koket/Desktop/BLUSE/data/mk_sample_hits.h5',
                        help='Path to H5 file')
    parser.add_argument('-o', '--output-dir', 
                        default='C:/Users/koket/Desktop/BLUSE/features/',
                        help='Output directory for parquet file')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')
    
    # Pipeline
    model = load_model(device)
    data, h5_path = load_data(args.h5_file)
    features = extract_features(model, data)
    save_features(features, h5_path, args.output_dir)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
