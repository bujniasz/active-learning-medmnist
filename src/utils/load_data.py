# === IMPORTS ===
# General
import os
import numpy as np
import random

# Torch
import torch
from torch.utils.data import Dataset, DataLoader

# Custom
from src.utils.labels_mapping import map_labels, get_valid_indices

binary_mapping_required = {"bloodmnist", "octmnist", "pathmnist"}
class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        # Ensure that the image is in (C, H, W) format
        if x.shape == (28, 28):
            x = np.expand_dims(x, axis=0)
        elif x.shape == (28, 28, 3):
            x = np.transpose(x, (2, 0, 1))
        elif x.shape in [(1, 28, 28), (3, 28, 28)]:
            pass
        else:
            raise ValueError(f"Unknown image format: {x.shape}")

        x = torch.tensor(x, dtype=torch.float32) / 255.0  # normalisation
        y = torch.tensor(y, dtype=torch.long).squeeze()   # (N,1) -> (N,)

        if self.transform:
            x = self.transform(x)

        return x, y

def load_npz_split(data_dir, split):
    """
    Loads images and labels for a given partition (train/val/test)
    """
    images = np.load(os.path.join(data_dir, f"{split}_images.npy"))
    labels = np.load(os.path.join(data_dir, f"{split}_labels.npy"))
    return images, labels

def seed_worker(worker_id):
    """
    Sets the seed for randomness within the worker process:
    - torch.initial_seed() -> worker seed (comes from the DataLoader generator)
    - we pass this seed to numpy.random and random
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_split_supervised(data_dir, batch_size=64, num_workers=2, seed: int | None = None):
    """
    Returns dataloaders for train, validation, and test sets.
    
    Applies binary label mapping and filtering where applicable.
    If a `seed` is provided, the training loader uses a deterministic generator 
    and workers are seeded via `worker_init_fn` to ensure reproducibility.
    """
    splits = ['train', 'val', 'test']
    dataloaders = {}

    dataset_name = os.path.basename(os.path.normpath(data_dir))

    gen = None
    worker_fn = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        worker_fn = seed_worker

    for split in splits:
        images, labels = load_npz_split(data_dir, split)

        if dataset_name in binary_mapping_required:
            valid_indices = get_valid_indices(dataset_name, labels)
            images = images[valid_indices]
            labels = labels[valid_indices]
            labels = map_labels(dataset_name, labels)

        dataset = MedMNISTDataset(images, labels)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            generator=gen if split == 'train' else None,
            worker_init_fn=worker_fn if split == 'train' else None,
        )

    return dataloaders['train'], dataloaders['val'], dataloaders['test']

def prepare_split_active(data_dir: str, split: str = "train", to_nchw: bool = True):
    """
    Prepares a specific dataset split for Active Learning workflows.
    
    Loads data from NPZ files, applies optional binary mapping, and returns 
    raw arrays along with dataset metadata (channels and number of classes).
    Supports converting images to NCHW format for PyTorch compatibility.
    """
    dataset_name = os.path.basename(os.path.normpath(data_dir))

    X, y = load_npz_split(data_dir, split)

    if dataset_name in binary_mapping_required:
        idx = get_valid_indices(dataset_name, y)
        X, y = X[idx], y[idx]
        y = map_labels(dataset_name, y).astype(np.int64)
    else:
        y = y.squeeze().astype(np.int64)

    if to_nchw and X.ndim == 4 and X.shape[-1] in (1, 3):
        X = np.transpose(X, (0, 3, 1, 2))

    in_channels = 1 if X.ndim == 3 else (X.shape[1] if X.ndim == 4 else 1)
    num_classes = int(np.unique(y).size)
    return X, y, in_channels, num_classes

