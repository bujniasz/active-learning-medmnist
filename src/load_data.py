import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from labels_mapping import map_labels, get_valid_indices

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
            raise ValueError(f"Nieznany format obrazu: {x.shape}")

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


def get_dataloaders(data_dir, batch_size=64, num_workers=2):
    """
    Returns dataloaders for train/val/test sets.
    it applies binary label mapping and filtering (if needed).
    """
    splits = ['train', 'val', 'test']
    dataloaders = {}

    dataset_name = os.path.basename(os.path.normpath(data_dir))

    for split in splits:
        images, labels = load_npz_split(data_dir, split)

        if dataset_name in binary_mapping_required:
            valid_indices = get_valid_indices(dataset_name, labels) # it is for filtering purposes - nothing happens if that's not a pathmnist
            images = images[valid_indices]
            labels = labels[valid_indices]

            labels = map_labels(dataset_name, labels)

        dataset = MedMNISTDataset(images, labels)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers
        )

    return dataloaders['train'], dataloaders['val'], dataloaders['test']

# TEST 
# if __name__ == "__main__":
#     data_dir = "data/pneumoniamnist"

#     train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=8)

#     for x, y in train_loader:
#         print("x shape:", x.shape)  # expected: [8, 1, 28, 28] or [8, 3, 28, 28]
#         print("y shape:", y.shape)  # expected: [8]
#         print("x dtype:", x.dtype)  # expected: torch.float32
#         print("y dtype:", y.dtype)  # expected: torch.int64
#         print("y batch:", y.tolist())  # np. [0, 3, 1, 1, 2, 0, 3, 3] NOW [0, 1, 0 ....]
#         break
