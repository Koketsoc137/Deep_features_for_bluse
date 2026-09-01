import h5py
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Loads data from a list of BLUSE .h5 files.

    Usage:
        dataset = H5Dataset(['file1.h5', 'file2.h5'])
        sample = dataset[0]          # returns a single (1, w, h) tensor
        len(dataset)                 # total number of observations across all files
    """

    def __init__(self, file_path):
        self.file_path = file_path

        with h5py.File(self.file_path, 'r') as f:
            self.n_samples = f['data'].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.n_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.n_samples}")

        # Open file safely per access
        with h5py.File(self.file_path, 'r') as f:
            sample = f['data'][index]

        return torch.from_numpy(sample).float()