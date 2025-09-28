# ======= Imports =======

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple, List

import os
import numpy as np
import torch
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling
from libact.labelers import IdealLabeler
from libact.base.interfaces import ProbabilisticModel

from load_data import get_dataloaders, load_npz_split
from labels_mapping import map_labels, get_valid_indices
from metrics import evaluate_predictions

# === Consts and Seed ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Custom wrapper for libact <-> resnet to work ===
# === https://github.com/ntucllab/libact/blob/master/libact/base/interfaces.py ===
class TorchModelWrapper(ProbabilisticModel):
    def __init__(self, in_channels: int, num_classes: int, lr: float = 1e-3, epochs_per_cycle: int = 1):
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs_per_cycle = epochs_per_cycle

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader
        self.model.eval()
        if isinstance(X, list):
            X = np.stack(X, axis=0)
        X_t = torch.from_numpy(X)
        if X_t.dtype == torch.uint8:
            X_t = X_t.float() / 255.0
        if X_t.ndim == 3:  # (N,H,W) -> (N,1,H,W)
            X_t = X_t.unsqueeze(1)
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(DEVICE, non_blocking=True)
                logits = self.model(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def train_on_numpy(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 64):
        from torch.utils.data import TensorDataset, DataLoader
        self.model.train()
        X_t = torch.from_numpy(X)
        if X_t.dtype == torch.uint8:
            X_t = X_t.float() / 255.0
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(1)
        y_t = torch.from_numpy(y).long()
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()

    def train(self, dataset):
        X_l, y_l = dataset.get_labeled_entries()
        if len(y_l) == 0:
            return
        X_arr = np.stack(X_l)
        y_arr = np.asarray(y_l, dtype=np.int64)
        self.train_on_numpy(X_arr, y_arr, epochs=getattr(self, "epochs_per_cycle", 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
    
    def score(self, dataset) -> float:
        X_l, y_l = dataset.get_labeled_entries()
        if len(y_l) == 0:
            return 0.0
        X_arr = np.stack(X_l)
        y_arr = np.asarray(y_l, dtype=np.int64)
        y_pred = self.predict(X_arr)
        return float((y_pred == y_arr).mean())

# === Args to parse (baseline.py should also be extended like that) ===
def parse_args():
    p = argparse.ArgumentParser(description="Active Learning on bloodmnist with libact")
    p.add_argument("--data-dir", type=str, default="data/bloodmnist", help="MedMNIST root dir")
    p.add_argument("--init-size", type=int, default=100)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--batch", type=int, default=10, help="queries per AL cycle")
    p.add_argument("--epochs-per-cycle", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--method", type=str, default="lc", choices=["lc", "sm", "entropy"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def prepare_bloodmnist_numpy(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads train split from .npz and applies the same rules as get_dataloaders():
    - filtering (for pathmnist) by get_valid_indices()
    - binary label mapping by map_labels()
    Returns: X (e.g. uint8 or float), y (int64) with shapes (N, H, W[, C?]) and (N,)
    """
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    X, y = load_npz_split(data_dir, "train")             # (N, 28, 28[, 3]), (N[,1])
    valid_idx = get_valid_indices(dataset_name, y)        # no-op dla bloodmnist
    X, y = X[valid_idx], y[valid_idx]
    y_bin = map_labels(dataset_name, y).astype(np.int64)  # 0/1 zgodnie z naszym mappingiem
    return X, y_bin

# === AL start === 
def init_libact(X: np.ndarray, y: np.ndarray, init_size: int, method: str, wrapper: TorchModelWrapper, seed: int = 42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = max(1, int(round(init_size * len(pos) / len(y))))
    n_neg = max(0, init_size - n_pos)
    init_idx = np.concatenate([
        rng.choice(pos, min(n_pos, len(pos)), replace=False),
        rng.choice(neg, min(n_neg, len(neg)), replace=False)
    ])
    init_set = set(init_idx.tolist())
    y_masked = [int(y[i]) if i in init_set else None for i in range(len(y))]
    active_ds = Dataset(X, y_masked)
    oracle = IdealLabeler(Dataset(X, y))
    qs = UncertaintySampling(active_ds, method=method, model=wrapper)
    return active_ds, oracle, qs, init_idx


# === TEST BELOW ===
# python3 src/train_active.py --data-dir data/bloodmnist --init-size 100 --epochs-per-cycle 1 --method lc
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    X, y = prepare_bloodmnist_numpy(args.data_dir)
    if X.ndim == 4 and X.shape[-1] in (1, 3):
        X = np.transpose(X, (0, 3, 1, 2))
    in_channels = 1 if X.ndim == 3 else X.shape[1]

    wrapper = TorchModelWrapper(in_channels=in_channels, num_classes=2, lr=args.lr, epochs_per_cycle=args.epochs_per_cycle)
    wrapper.epochs_per_cycle = args.epochs_per_cycle

    active_ds, oracle, qs, init_idx = init_libact(X, y, args.init_size, args.method, wrapper, seed=args.seed)
    print(f"Start: labeled={len(init_idx)}, unlabeled={len(y)-len(init_idx)}")

    wrapper.train(active_ds)
    for t in range(2):
        ask_id = qs.make_query()
        x_feat = active_ds.data[ask_id][0]
        y_new = oracle.label(x_feat)
        active_ds.update(ask_id, y_new)
        wrapper.train(active_ds)
        labeled_cnt = sum(lbl is not None for _, lbl in active_ds.data)
        print(f"Cycle {t+1}: labeled={labeled_cnt}")

    probs = wrapper.predict_proba(X[:8])
    print("probs shape:", probs.shape, "row sums:", probs.sum(axis=1))