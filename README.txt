The following datasets, with their shapes, are available in HDF5 format in the "data" folder:

mk_sample_hits.h5
(15119, 1, 57, 120)
lband_short.h5
(866002, 1, 24, 120)
sband_long.h5
(36132, 1, 59, 120)
sband_short.h5
(38576, 1, 24, 120)
uhf_long.h5
(299878, 1, 36, 120)
uhf_short.h5
(208774, 1, 15, 120)
lband_long.h5
(557690, 1, 57, 120)

I have extracted features for the mk_sample_hits.h5 file in the features folder:
mk_sample_resnet18_patch_features_meta.json  
mk_sample_resnet18_patch_features.npy  
mk_sample_resnet18_score_maps.npy
mk_sample_resnet18_tile_scores.npy
mk_sample_resnet18_penultimate_features_meta.json  
mk_sample_resnet18_penultimate_features.npy        

I suggest using the patch features (the same which are used by an algorithm called PatchCore) with this code:

X = np.load("features/mk_sample_resnet18_patch_features.npy", mmap_mode="r")   # 0 bytes read

# X: (15119, 450, 192)
# normalise the two blocks separately first — layer1 and layer2
# activations sit on different scales
l1 = X[..., :64]  / np.linalg.norm(X[..., :64],  axis=-1, keepdims=True)
l2 = X[..., 64:] / np.linalg.norm(X[..., 64:], axis=-1, keepdims=True)
Xn = np.concatenate([l1, l2], axis=-1)
    
features = np.concatenate([Xn.mean(axis=1), Xn.max(axis=1)], axis=-1)  # (15119, 384)
