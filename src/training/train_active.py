# === IMPORTS ===
# General
import os
import numpy as np
import argparse
import random
from pathlib import Path
import json

# Torch
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# Sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Libact
from libact.base.dataset import Dataset
from libact.query_strategies import RandomSampling
from libact.labelers import IdealLabeler
from libact.base.interfaces import ProbabilisticModel

# Custom
from src.utils.load_data import prepare_split_active
from src.utils.shared import load_config, get_predictions, class_report_conf_matrix, fmt, append_row_to_csv, ResNet18EmbedDropout

# === DEVICE ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def apply_config(args):
    if args.config is None:
        return args

    cfg = load_config(args.config)

    args.data_dir = cfg.get("data_dir")
    args.model_path = cfg.get("model_path")
    args.results_path = cfg.get("results_path")

    args.seed = cfg.get("seed", args.seed)
    args.strategy = cfg.get("strategy", args.strategy)

    al_mode = cfg.get("al_mode", "percent")

    if al_mode == "percent":
        args.init_size_pct = cfg.get("init_size_pct")
        args.budget_pct = cfg.get("budget_pct")
        args.batch_pct_of_budget = cfg.get("batch_pct_of_budget")

        args.init_size = None
        args.budget = None
        args.batch = None

    elif al_mode == "absolute":
        args.init_size = cfg.get("init_size")
        args.budget = cfg.get("budget")
        args.batch = cfg.get("batch")

        args.init_size_pct = None
        args.budget_pct = None
        args.batch_pct_of_budget = None

    else:
        raise ValueError(f"Unknown al_mode: {al_mode}")

    args.epochs_per_cycle = cfg.get("epochs_per_cycle", args.epochs_per_cycle)

    return args

def build_batch_schedule_from_budget(budget: int, n_cycles: int) -> list[int]:
    if budget <= 0:
        raise ValueError("budget must be > 0")
    if n_cycles <= 0:
        raise ValueError("n_cycles must be > 0")

    base = budget // n_cycles
    remainder = budget % n_cycles

    if base <= 0:
        raise ValueError(
            f"Planned number of cycles ({n_cycles}) is too large for budget ({budget}); "
            "would create empty batches."
        )

    schedule = [base] * n_cycles
    schedule[-1] += remainder
    return schedule

def _pct_to_count(pct: float, total: int) -> int:
    return max(1, int(round(total * float(pct) / 100.0)))

def resolve_active_params(args, train_size: int) -> dict:
    # init_size: absolute XOR percent
    if args.init_size is not None and args.init_size_pct is not None:
        raise ValueError("Use either --init-size or --init-size-pct, not both.")
    if args.init_size is None and args.init_size_pct is None:
        raise ValueError("One of --init-size / --init-size-pct is required.")

    # budget: absolute XOR percent
    if args.budget is not None and args.budget_pct is not None:
        raise ValueError("Use either --budget or --budget-pct, not both.")
    if args.budget is None and args.budget_pct is None:
        raise ValueError("One of --budget / --budget-pct is required.")

    # batch: absolute XOR percent-of-budget
    if args.batch is not None and args.batch_pct_of_budget is not None:
        raise ValueError("Use either --batch or --batch-pct-of-budget, not both.")
    if args.batch is None and args.batch_pct_of_budget is None:
        raise ValueError("One of --batch / --batch-pct-of-budget is required.")

    # resolve init
    if args.init_size_pct is not None:
        init_size = _pct_to_count(args.init_size_pct, train_size)
        init_size_pct = float(args.init_size_pct)
    else:
        init_size = int(args.init_size)
        init_size_pct = 100.0 * init_size / train_size

    # resolve budget
    if args.budget_pct is not None:
        budget = _pct_to_count(args.budget_pct, train_size)
        budget_pct = float(args.budget_pct)
    else:
        budget = int(args.budget)
        budget_pct = 100.0 * budget / train_size

    if init_size <= 0:
        raise ValueError("Resolved init_size must be > 0")
    if budget <= 0:
        raise ValueError("Resolved budget must be > 0")
    if init_size >= train_size:
        raise ValueError(f"Resolved init_size ({init_size}) must be smaller than train size ({train_size})")
    if init_size + budget > train_size:
        raise ValueError(
            f"Resolved init_size + budget = {init_size + budget}, "
            f"which exceeds train size ({train_size})"
        )

    # resolve batch / schedule
    if args.batch_pct_of_budget is not None:
        batch_pct_of_budget = float(args.batch_pct_of_budget)

        if batch_pct_of_budget <= 0:
            raise ValueError("--batch-pct-of-budget must be > 0")
        if batch_pct_of_budget > 100:
            raise ValueError("--batch-pct-of-budget must be <= 100")

        n_cycles_planned = max(1, int(round(100.0 / batch_pct_of_budget)))
        batch_schedule = build_batch_schedule_from_budget(budget, n_cycles_planned)
        batch = int(batch_schedule[0])  # nominal / first batch for metadata
    else:
        batch = int(args.batch)
        if batch <= 0:
            raise ValueError("Resolved batch must be > 0")
        if batch > budget:
            raise ValueError(f"Resolved batch ({batch}) cannot be larger than budget ({budget})")

        n_cycles_planned = int(np.ceil(budget / batch))
        batch_schedule = [batch] * (budget // batch)
        remainder = budget % batch
        if remainder > 0:
            batch_schedule[-1] += remainder

        batch_pct_of_budget = 100.0 * batch / budget

    return {
        "init_size": int(init_size),
        "budget": int(budget),
        "batch": int(batch),
        "init_size_pct": float(init_size_pct),
        "budget_pct": float(budget_pct),
        "batch_pct_of_budget": float(batch_pct_of_budget),
        "final_labeled_target": int(init_size + budget),
        "n_cycles_planned": int(n_cycles_planned),
        "batch_schedule": [int(x) for x in batch_schedule],
    }

def get_candidate_pool(
    active_ds: Dataset,
    candidate_size: int,
    rng_seed: int,
):
    """
    Returns (cand_entry_ids, X_cand) where cand_entry_ids are entry IDs usable in active_ds.update().
    candidate_size:
      -1 => full unlabeled pool
      >0 => uniform subsample without replacement if pool bigger than candidate_size
    """
    unlabeled_entry_ids, X_pool = active_ds.get_unlabeled_entries()
    n = len(unlabeled_entry_ids)
    if n == 0:
        return np.array([], dtype=int), np.asarray(X_pool)

    unlabeled_entry_ids = np.asarray(unlabeled_entry_ids)

    if candidate_size == -1 or candidate_size >= n:
        return unlabeled_entry_ids.astype(int), np.asarray(X_pool)

    rng = np.random.default_rng(int(rng_seed))
    idx = rng.choice(n, size=int(candidate_size), replace=False)
    return unlabeled_entry_ids[idx].astype(int), np.asarray(X_pool)[idx]

def get_least_confident_scores(X_u, wrapper):
    probs = wrapper.predict_proba(X_u)
    return 1.0 - np.max(probs, axis=1)

def margin_rows(probs: np.ndarray) -> np.ndarray:
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top2 = np.sort(top2, axis=1)[:, ::-1]
    return -(top2[:, 0] - top2[:, 1])

def get_margin_scores(X_u, wrapper):
    probs = wrapper.predict_proba(X_u)
    return margin_rows(probs)

def entropy_rows(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    probs: (..., C)
    returns entropy over last axis: (...)
    """
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)

def get_entropy_scores(X_u, wrapper):
    probs = wrapper.predict_proba(X_u)
    return entropy_rows(probs)

# def select_uncertainty_entropy_full_pool(active_ds: Dataset, wrapper, k: int) -> list[int]:
#     """Select top-k unlabeled samples by predictive entropy on the FULL pool.

#     Implements:
#         x* = argmax_{x in U} ( - sum_i p(y_i|x) log p(y_i|x) )

#     Notes:
#       - Uses wrapper.predict_proba on the full unlabeled pool.
#       - Deterministic forward pass assumed (dropout disabled in wrapper.predict_proba).
#       - Returns libact entry ids (integers compatible with active_ds.update()).
#     """
#     if k <= 0:
#         return []

#     # Official libact API: returns IDs and the corresponding feature matrix/array
#     unlabeled_entry_ids, X_pool = active_ds.get_unlabeled_entries()
#     if len(unlabeled_entry_ids) == 0:
#         return []

#     probs = wrapper.predict_proba(X_pool)  # shape: (N, C)

#     # Entropy: -sum p log p
#     eps = 1e-12
#     p = np.clip(probs, eps, 1.0)
#     scores = -np.sum(p * np.log(p), axis=1)  # shape: (N,)

#     k = min(k, len(unlabeled_entry_ids))

#     # Top-k by entropy (descending)
#     top_idx = np.argpartition(scores, -k)[-k:]                # fast top-k (unordered)
#     top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]      # sort those k desc

#     return [int(unlabeled_entry_ids[int(j)]) for j in top_idx]

def bald_score(probs_T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    probs_T: (T, N, C)
    returns: (N,)  BALD = H(mean_p) - mean_t H(p_t)
    """
    mean_p = np.mean(probs_T, axis=0)              # (N, C)
    H_mean = entropy_rows(mean_p, eps=eps)         # (N,)
    H_each = entropy_rows(probs_T, eps=eps)        # (T, N)
    return H_mean - np.mean(H_each, axis=0)        # (N,)

def k_center_greedy(emb: np.ndarray, k: int, seed: int = 42, first: int | None = None) -> list[int]:
    """
    Select k points using farthest-first traversal (k-center greedy).
    emb: (M, D)
    returns indices in [0..M-1]
    """
    rng = np.random.default_rng(seed)
    M = emb.shape[0]
    if k >= M:
        return list(range(M))

    if first is None:
        first = int(rng.integers(0, M))
    else:
        first = int(first)
        if first < 0 or first >= M:
            raise ValueError(f"`first` must be in [0, {M-1}], got {first}")

    selected = [first]

    # distances to closest selected
    d = np.linalg.norm(emb - emb[first], axis=1)

    for _ in range(1, k):
        nxt = int(np.argmax(d))
        selected.append(nxt)
        d = np.minimum(d, np.linalg.norm(emb - emb[nxt], axis=1))

    return selected

def egl_fc_score(emb: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    EGL for last linear layer (fc) under softmax+CE.
    emb: (N, D) embeddings
    probs: (N, C) predicted probabilities
    returns: (N,) scores (higher = more informative)
    """
    emb_norm2 = np.sum(emb * emb, axis=1)          # (N,)
    sum_p2 = np.sum(probs * probs, axis=1)         # (N,)
    return (emb_norm2 + 1.0) * (1.0 - sum_p2)      # (N,)


# === CUSTOM WRAPPER FOR libact <-> resnet TO WORK ===
# === https://github.com/ntucllab/libact/blob/master/libact/base/interfaces.py ===
class TorchModelWrapper(ProbabilisticModel):
    def __init__(self, in_channels: int, num_classes: int, lr: float = 1e-3, epochs_per_cycle: int = 1, seed: int | None = None, dropout_p: float = 0.5):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model = ResNet18EmbedDropout(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=dropout_p
        ).to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs_per_cycle = epochs_per_cycle
        self.seed = seed

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
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
        all_probs = []
        with torch.inference_mode():
            for (inputs,) in dl:
                inputs = inputs.to(DEVICE)
                logits = self.model(inputs, backbone_mode="eval", enable_dropout=False)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.detach().cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def mc_predict_proba(self, X: np.ndarray, T: int = 10, base_seed: int | None = None, batch_size: int = 256) -> np.ndarray:
        """
        MC Dropout predictive distribution: mean of T stochastic forward passes.
        Returns mean probabilities of shape (N, C).
        """
        if isinstance(X, list):
            X = np.stack(X, axis=0)

        X_t = torch.from_numpy(X)
        if X_t.dtype == torch.uint8:
            X_t = X_t.float() / 255.0
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(1)

        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        self.model.eval()

        probs_T = []
        for t in range(T):
            if base_seed is not None:
                torch.manual_seed(int(base_seed) + t)
            all_probs = []
            with torch.inference_mode():
                for (inputs,) in dl:
                    inputs = inputs.to(DEVICE)
                    logits = self.model(inputs, backbone_mode="eval", enable_dropout=True)
                    all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs_T.append(np.concatenate(all_probs, axis=0))

        return np.mean(np.stack(probs_T, axis=0), axis=0)


    def mc_predict_proba_T(self, X: np.ndarray, T: int = 10, base_seed: int | None = None, batch_size: int = 256) -> np.ndarray:
        """
        Returns probabilities for each MC pass.
        Shape: (T, N, C)
        """
        if isinstance(X, list):
            X = np.stack(X, axis=0)

        X_t = torch.from_numpy(X)
        if X_t.dtype == torch.uint8:
            X_t = X_t.float() / 255.0
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(1)

        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        probs_T = []
        for t in range(T):
            if base_seed is not None:
                torch.manual_seed(int(base_seed) + t)

            all_probs = []
            with torch.inference_mode():
                for (inputs,) in dl:
                    inputs = inputs.to(DEVICE)
                    logits = self.model(inputs, backbone_mode="eval", enable_dropout=True)
                    all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

            probs_T.append(np.concatenate(all_probs, axis=0))

        return np.stack(probs_T, axis=0)


    def extract_embeddings(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Returns embeddings from the penultimate layer (after avgpool).
        Output shape: (N, D), e.g. D=512 for ResNet18.
        """
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

        embeddings = []

        def hook_fn(module, input, output):
            # output shape: (N, 512, 1, 1)
            embeddings.append(output.detach().cpu())

        #handle = self.model.avgpool.register_forward_hook(hook_fn)

        handle = self.model.backbone.avgpool.register_forward_hook(hook_fn)

        with torch.inference_mode():
            for (inputs,) in dl:
                inputs = inputs.to(DEVICE)
                _ = self.model(inputs, backbone_mode="eval", enable_dropout=False)

        handle.remove()

        feats = torch.cat(embeddings, dim=0)
        feats = feats.view(feats.size(0), -1)  # (N, 512)
        return feats.numpy()



    def train_on_numpy(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 64, verbose: bool = False):
        self.model.train()
        X_t = torch.from_numpy(X)
        if X_t.dtype == torch.uint8:
            X_t = X_t.float() / 255.0
        else:
            X_t = X_t.float()
            if X_t.max() > 1.5:
                X_t = X_t / 255.0
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(1)
        y_t = torch.from_numpy(y).long()

        ds = TensorDataset(X_t, y_t)
        n = len(ds)
        if n < 2:
            return None  # explicit

        eff_bs = min(batch_size, n)
        gen = None
        if self.seed is not None:
            gen = torch.Generator()
            gen.manual_seed(self.seed)

        dl = DataLoader(ds, batch_size=eff_bs, shuffle=True, drop_last=True, generator=gen, num_workers=0)

        epoch_losses: list[float] = []
        for _ in range(int(epochs)):
            total_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                logits = self.model(xb, backbone_mode="train", enable_dropout=True)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())

            avg_loss = total_loss / max(1, len(dl))
            epoch_losses.append(avg_loss)
            # if verbose:
            #     print(f"   🔹 Training loss: {avg_loss:.4f}")

        # train_loss per cycle = mean loss across epochs
        return float(np.mean(epoch_losses))

    def train(self, dataset, verbose: bool = False):
        X_l, y_l = dataset.get_labeled_entries()
        if len(y_l) == 0:
            return
        X_arr = np.stack(X_l)
        y_arr = np.asarray(y_l, dtype=np.int64)
        return self.train_on_numpy(X_arr, y_arr, epochs=getattr(self, "epochs_per_cycle", 1), verbose=verbose)

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

# === ARGUMENTS ===
def parse_args():
    p = argparse.ArgumentParser(description="Active Learning on bloodmnist with libact")
    p.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--eval-only", action="store_true", help="Skip training of the model - just evaluate the existing one")
    p.add_argument("-d", "--data-dir", type=str, help="Path to data folder")
    p.add_argument("-mp", "--model-path", type=str, default=None, help="Path to the .pth model file (new or existing one)")
    p.add_argument("-r", "--results-path", type=str, default="results/test-exps-pt3.csv", help="Path to the .csv file with evaluation results (if none provided it's the same as model-path)")
    p.add_argument("--init-size", type=int, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="queries per AL cycle")
    p.add_argument("--init-size-pct", type=float, default=None, help="Initial labeled set as percent of train set size")
    p.add_argument("--budget-pct", type=float, default=None, help="AL budget as percent of train set size")
    p.add_argument("--batch-pct-of-budget", type=float, default=None, help="Batch size as percent of resolved budget")
    p.add_argument("--epochs-per-cycle", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--select-metric", type=str, default="mean", choices=["mean", "acc", "f1", "auc", "ap"], help="Metric used to select the best checkpoint (mean = average of acc,f1,auc,ap)")
    p.add_argument("--select-delta", type=float, default=1e-4, help="Minimum improvement required to save a new best checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategy", type=str, default="entropy", choices=["least_confident", "margin", "entropy", "random", "mc_entropy", "mc_bald", "mc_entropy_diverse", "mc_bald_diverse", "entropy_diverse", "egl_fc"])
    p.add_argument("--mc-T", type=int, default=5, help="Number of MC Dropout forward passes")
    p.add_argument("--candidate-size", type=int, default=-1, help="Unlabeled candidates to score each query (mc strategies)")
    p.add_argument("--top-m-mult", type=int, default=10, help="For diverse batch: candidates = top_m_mult * batch")
    return p.parse_args()

# === SEED ===
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception as e:
        print(f"[WARN] Could not set torch threads: {e}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[WARN] torch.use_deterministic_algorithms(True) not supported: {e}")

    os.environ["PYTHONHASHSEED"] = str(seed)

# === AL START === 
def init_libact( X: np.ndarray, y: np.ndarray, init_size: int, seed: int = 42, strategy="entropy"):
    rng = np.random.default_rng(seed)

    y = np.asarray(y)
    n_total = len(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos_total = len(pos_idx)
    n_neg_total = len(neg_idx)

    if n_pos_total == 0 or n_neg_total == 0:
        n_init = min(init_size, n_total)
        all_idx = np.arange(n_total)
        init_idx = rng.choice(all_idx, size=n_init, replace=False)

        init_set = set(init_idx.tolist())
        y_masked = [int(y[i]) if i in init_set else None for i in range(n_total)]

        active_ds = Dataset(X, y_masked)
        oracle = IdealLabeler(Dataset(X, y))
        if strategy == "random":
            qs = RandomSampling(active_ds, random_state=seed)
        else:
            qs = None
        return active_ds, oracle, qs, init_idx


    frac_pos = n_pos_total / n_total

    raw_n_pos = int(round(init_size * frac_pos))
    min_per_class = 1

    n_pos = max(min_per_class, raw_n_pos)
    n_neg = max(min_per_class, init_size - n_pos)

    total_requested = n_pos + n_neg
    if total_requested > init_size:
        overflow = total_requested - init_size
        if n_pos >= n_neg:
            n_pos = max(min_per_class, n_pos - overflow)
        else:
            n_neg = max(min_per_class, n_neg - overflow)

    n_pos_sample = min(n_pos, n_pos_total)
    n_neg_sample = min(n_neg, n_neg_total)

    chosen_pos = rng.choice(pos_idx, size=n_pos_sample, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)

    init_idx = np.concatenate([chosen_pos, chosen_neg])

    rng.shuffle(init_idx)

    init_set = set(init_idx.tolist())
    y_masked = [int(y[i]) if i in init_set else None for i in range(n_total)]

    active_ds = Dataset(X, y_masked)
    oracle = IdealLabeler(Dataset(X, y))
    if strategy == "random":
        qs = RandomSampling(active_ds, random_state=seed)
    else:
        qs = None

    return active_ds, oracle, qs, init_idx

# === TRAINING + VALIDATION LOOP ===
def run_active_loop(
    active_ds,
    oracle,
    qs,
    wrapper,
    X_val,
    y_val,
    batch_schedule: list[int],
    model_path: str,
    select_metric: str = "acc",
    data_dir: str | None = None,
    *,
    resolved_init_size: int,
    resolved_budget: int,
    resolved_batch: int,
    resolved_init_size_pct: float,
    resolved_budget_pct: float,
    resolved_batch_pct_of_budget: float,
    final_labeled_target: int,
    n_cycles_planned: int,
    initial_best_sel: float,
) -> str:

    # CHANGED: asked nadal trzymamy, ale cycle kontrolujemy przez enumerate(batch_schedule)
    asked = 0
    best_sel = float(initial_best_sel)
    ask_log = []

    Path(Path(model_path).parent).mkdir(parents=True, exist_ok=True)

    # CHANGED: zamiast while asked < budget używamy jawnego harmonogramu batchy
    for cycle_idx, k in enumerate(batch_schedule, start=1):
        if k <= 0:
            continue

        cycle = cycle_idx  # CHANGED: numer cyklu pochodzi z harmonogramu

        cand_seed = int(args.seed + 10_000 * cycle + asked)
        mc_seed = int(args.seed + 20_000 * cycle + asked)

        # ============================================================
        # 0) Random strategy (libact) – sequential querying
        # ============================================================
        if args.strategy == "random":

            cycle_ask_ids = []

            for _ in range(k):
                ask_id = int(qs.make_query())
                cycle_ask_ids.append(ask_id)

                y_new = oracle.label(active_ds.data[ask_id][0])
                active_ds.update(ask_id, y_new)

            ask_log.append({
                "cycle": cycle,
                "planned_batch_size": int(k),  # CHANGED: zapisujemy planowany rozmiar batcha
                "actual_batch_size": int(len(cycle_ask_ids)),
                "ask_ids": cycle_ask_ids,
            })

        else:

            # ============================================================
            # 1) Candidate pool (computed ONCE per cycle)
            # ============================================================
            cand_ids, X_u = get_candidate_pool(
                active_ds,
                candidate_size=args.candidate_size,
                rng_seed=cand_seed,
            )

            if len(cand_ids) == 0:
                cycle_ask_ids = []

            else:

                # ============================================================
                # 2) Compute uncertainty scores
                # ============================================================
                if args.strategy == "least_confident":
                    scores = get_least_confident_scores(X_u, wrapper)

                elif args.strategy == "margin":
                    scores = get_margin_scores(X_u, wrapper)

                elif args.strategy in ("entropy", "entropy_diverse"):
                    scores = get_entropy_scores(X_u, wrapper)

                elif args.strategy in ("mc_entropy", "mc_entropy_diverse"):
                    probs = wrapper.mc_predict_proba(
                        X_u,
                        T=args.mc_T,
                        base_seed=mc_seed,
                    )
                    scores = entropy_rows(probs)

                elif args.strategy in ("mc_bald", "mc_bald_diverse"):
                    probs_T = wrapper.mc_predict_proba_T(
                        X_u,
                        T=args.mc_T,
                        base_seed=mc_seed,
                    )
                    scores = bald_score(probs_T)

                elif args.strategy == "egl_fc":
                    probs = wrapper.predict_proba(X_u)
                    emb = wrapper.extract_embeddings(X_u)
                    scores = egl_fc_score(emb, probs)

                else:
                    raise ValueError(f"Unknown strategy: {args.strategy}")

                # ============================================================
                # 3) Select batch
                # ============================================================

                # ---------- Diverse strategies ----------
                if args.strategy.endswith("_diverse"):

                    M = min(len(cand_ids), args.top_m_mult * k)

                    top_local = np.argpartition(scores, -M)[-M:]
                    top_local = top_local[np.argsort(scores[top_local])[::-1]]

                    X_top = X_u[top_local]
                    E_top = wrapper.extract_embeddings(X_top)

                    diverse_local = k_center_greedy(
                        E_top,
                        k=min(k, len(top_local)),
                        seed=args.seed,
                        first=0,
                    )

                    batch_local = top_local[diverse_local]
                    cycle_ask_ids = [int(cand_ids[int(j)]) for j in batch_local]

                # ---------- Non-diverse strategies ----------
                else:
                    k_eff = min(k, len(cand_ids))

                    top_local = np.argpartition(scores, -k_eff)[-k_eff:]
                    top_local = top_local[np.argsort(scores[top_local])[::-1]]

                    cycle_ask_ids = [int(cand_ids[int(j)]) for j in top_local]

                # ============================================================
                # 4) Query oracle + update dataset
                # ============================================================
                for ask_id in cycle_ask_ids:
                    y_new = oracle.label(active_ds.data[int(ask_id)][0])
                    active_ds.update(int(ask_id), y_new)

                ask_log.append({
                    "cycle": cycle,
                    "planned_batch_size": int(k),  # CHANGED: zapisujemy planowany rozmiar batcha
                    "actual_batch_size": int(len(cycle_ask_ids)),
                    "ask_ids": cycle_ask_ids,
                })

        val_loss = wrapper.train(active_ds, verbose=True)

        # CHANGED: liczymy faktycznie zadane próbki, ale nie sterujemy już tym pętlą
        asked += len(cycle_ask_ids)

        _, val_y_pred = get_predictions(
            wrapper.model,
            (X_val, y_val),
            DEVICE,
            forward_kwargs={"backbone_mode": "eval", "enable_dropout": False},
        )

        val_acc = accuracy_score(y_val, val_y_pred)
        val_f1 = f1_score(y_val, val_y_pred, average="macro")
        val_proba = None
        val_auc = float("nan")
        val_ap = float("nan")

        if wrapper.num_classes == 2:
            val_proba = wrapper.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            val_ap = average_precision_score(y_val, val_proba)

        vals = np.array([val_acc, val_f1, val_auc, val_ap], dtype=float)
        val_mean = float(np.nanmean(vals))
        if np.isnan(val_mean):
            val_mean = float("-inf")

        labeled_cnt = sum(lbl is not None for _, lbl in active_ds.data)

        # CHANGED: używamy cycle/n_cycles_planned z harmonogramu
        print(
            f"[cycle {cycle}/{n_cycles_planned}] "
            f"planned_batch={k} actual_batch={len(cycle_ask_ids)} "
            f"labeled={labeled_cnt} train loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"val_auc={val_auc:.4f} val_ap={val_ap:.4f} val_mean={val_mean:.4f}"
        )

        sel_map = {
            "mean": val_mean,
            "acc": val_acc,
            "f1": val_f1,
            "auc": val_auc,
            "ap": val_ap,
        }
        sel = float(sel_map[select_metric])

        if np.isnan(sel):
            sel = float("-inf")

        delta = getattr(args, "select_delta", 0.0)
        is_best = 1 if (sel > best_sel + delta) else 0

        append_row_to_csv({
            "dataset": os.path.basename(os.path.normpath(data_dir)) if data_dir else "",
            "phase": "active",
            "strategy": args.strategy,
            "seed": int(args.seed),
            "model": os.path.basename(os.path.normpath(model_path)),

            "init_size": int(resolved_init_size),
            "batch": int(resolved_batch),  # nominalny batch do metadanych
            "budget": int(resolved_budget),
            "init_size_pct": float(resolved_init_size_pct),
            "budget_pct": float(resolved_budget_pct),
            "batch_pct_of_budget": float(resolved_batch_pct_of_budget),
            "epc": int(args.epochs_per_cycle),
            "final_labeled_target": int(final_labeled_target),

            "step_type": "cycle",
            "step": int(cycle),
            "labeled_count": int(labeled_cnt),
            "split": "val",
            "train_loss": fmt(val_loss),

            "acc": fmt(val_acc),
            "f1_macro": fmt(val_f1),
            "auc": fmt(val_auc),
            "ap": fmt(val_ap),

            "val_mean": fmt(val_mean),
            "select_metric": select_metric,
            "is_best": int(is_best),
            "tp": -1,
            "fp": -1,
            "tn": -1,
            "fn": -1,
        }, RESULTS_PATH)

        if sel > best_sel + delta:
            best_sel = sel

            torch.save({
                "model_state_dict": wrapper.model.state_dict(),
                "in_channels": wrapper.in_channels,
                "num_classes": wrapper.num_classes,
                "data_dir": data_dir,
                "strategy": args.strategy,

                "init_size": int(resolved_init_size),
                "batch": int(resolved_batch),  # nominalny batch do metadanych
                "budget": int(resolved_budget),
                "init_size_pct": float(resolved_init_size_pct),
                "budget_pct": float(resolved_budget_pct),
                "batch_pct_of_budget": float(resolved_batch_pct_of_budget),
                "epc": int(args.epochs_per_cycle),
                "final_labeled_target": int(final_labeled_target),

                # CHANGED: zapisujemy harmonogram batchy do checkpointu
                "batch_schedule": [int(x) for x in batch_schedule],
                "n_cycles_planned": int(n_cycles_planned),

                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
                "val_ap": float(val_ap),
                "val_mean": float(val_mean),
                "select_metric": select_metric,
                "best_metric": float(sel),
                "best_cycle": int(cycle),
                "labeled_count": int(labeled_cnt),
                "seed": int(args.seed),
            }, model_path)

            print(f"✅ NEW BEST (by {args.select_metric}) → {best_sel:.4f}")

    # Save ask log to logs/ instead of models/
    ask_log_path = Path("logs") / Path(model_path).with_suffix(".asklog.json").name

    # ensure logs directory exists
    ask_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ask_log_path, "w") as f:
        json.dump(ask_log, f, indent=2)

    print(f"📝 Ask log saved to: {ask_log_path}")
    return model_path

# === MAIN LOOP ===
if __name__ == "__main__":
    args = parse_args()
    args = apply_config(args)

    set_seed(args.seed)

    if args.model_path is None:
        raise SystemExit("model_path must be provided either via CLI or config")
    
    if args.results_path is not None:
        RESULTS_PATH = args.results_path
    else:
        model_filename = os.path.basename(args.model_path)
        model_name = os.path.splitext(model_filename)[0]
        RESULTS_PATH = os.path.join("results", model_name + ".csv")

    if args.eval_only:
        print("🔍 Mode: evaluation only (ACTIVE)")

        ckpt = torch.load(args.model_path, map_location=DEVICE)
        data_dir = ckpt["data_dir"]
        in_channels, num_classes = ckpt["in_channels"], ckpt["num_classes"]
        print(f"📊 Detected: {num_classes} classes, {in_channels} channels\n")

        wrapper = TorchModelWrapper(in_channels=in_channels, num_classes=num_classes, lr=args.lr, epochs_per_cycle=args.epochs_per_cycle)
        wrapper.model.load_state_dict(ckpt["model_state_dict"])

        X_val, y_val, _, _   = prepare_split_active(data_dir, split="val",  to_nchw=True)
        X_test, y_test, _, _ = prepare_split_active(data_dir, split="test", to_nchw=True)

    else:
        print("🚀 Mode: training + evaluation (ACTIVE)")

        X, y, in_channels, num_classes = prepare_split_active(args.data_dir, split="train", to_nchw=True)
        print(f"📊 Detected: {num_classes} classes, {in_channels} channels\n")

        train_size = int(len(y))
        resolved = resolve_active_params(args, train_size=train_size)

        resolved_init_size = int(resolved["init_size"])
        resolved_budget = int(resolved["budget"])
        resolved_batch = int(resolved["batch"])

        resolved_init_size_pct = float(resolved["init_size_pct"])
        resolved_budget_pct = float(resolved["budget_pct"])
        resolved_batch_pct_of_budget = float(resolved["batch_pct_of_budget"])

        final_labeled_target = int(resolved["final_labeled_target"])
        n_cycles_planned = int(resolved["n_cycles_planned"])
        resolved_batch_schedule = list(resolved["batch_schedule"])

        print(
            f"📐 Resolved AL params | "
            f"train_size={train_size} | "
            f"init_size={resolved_init_size} ({resolved_init_size_pct:.1f}%) | "
            f"budget={resolved_budget} ({resolved_budget_pct:.1f}%) | "
            f"batch={resolved_batch} ({resolved_batch_pct_of_budget:.1f}% of budget) | "
            f"final_target={final_labeled_target} | "
            f"planned_cycles={n_cycles_planned} | "
            f"batch_schedule={resolved_batch_schedule}"
        )

        wrapper = TorchModelWrapper(in_channels=in_channels, num_classes=num_classes, lr=args.lr, epochs_per_cycle=args.epochs_per_cycle, seed=args.seed)

        active_ds, oracle, qs, init_idx = init_libact(X, y, resolved_init_size, seed=args.seed, strategy=args.strategy)

        y = np.asarray(y)
        init_labels = y[init_idx]
        unique, counts = np.unique(init_labels, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        class0 = class_dist.get(0, 0)
        class1 = class_dist.get(1, 0)

        print(
            f"Start: labeled={len(init_idx)}, unlabeled={len(y) - len(init_idx)}, "
            f"class 0 = {class0}, class 1 = {class1} "
            f"Strategy = {args.strategy}"
        )

        # First training (before anotations)
        print("\n🔸 Initial training on starting labeled set")
        start_loss = wrapper.train(active_ds, verbose=True)

        X_val, y_val, _, _ = prepare_split_active(args.data_dir, split="val", to_nchw=True)
        _, val_y_pred = get_predictions(wrapper.model, (X_val, y_val), DEVICE, forward_kwargs={"backbone_mode": "eval", "enable_dropout": False})

        start_val_acc = accuracy_score(y_val, val_y_pred)
        start_val_f1 = f1_score(y_val, val_y_pred, average='macro')

        start_val_proba = None
        start_val_auc = float("nan")
        start_val_ap = float("nan")
        if wrapper.num_classes == 2:
            start_val_proba = wrapper.predict_proba(X_val)[:, 1]
            start_val_auc = roc_auc_score(y_val, start_val_proba)
            start_val_ap = average_precision_score(y_val, start_val_proba)

        start_vals = np.array([start_val_acc, start_val_f1, start_val_auc, start_val_ap], dtype=float)
        start_mean = float(np.nanmean(start_vals))
        if np.isnan(start_mean):
            start_mean = float("-inf")

        labeled_cnt = sum(lbl is not None for _, lbl in active_ds.data)
        print(
            f"[cycle 0] labeled={labeled_cnt} "
            f"val_loss={start_loss:.4f} "
            f"val_acc={start_val_acc:.4f} val_f1={start_val_f1:.4f} "
            f"val_auc={start_val_auc:.4f} val_ap={start_val_ap:.4f} "
            f"val_mean={start_mean:.4f}"
        )

        append_row_to_csv({
            "dataset": os.path.basename(os.path.normpath(args.data_dir)),
            "phase": "active",
            "strategy": args.strategy,
            "seed": int(args.seed),
            "model": os.path.basename(os.path.normpath(args.model_path)),

            "init_size": int(resolved_init_size),
            "batch": int(resolved_batch),
            "budget": int(resolved_budget),
            "init_size_pct": float(resolved_init_size_pct),
            "budget_pct": float(resolved_budget_pct),
            "batch_pct_of_budget": float(resolved_batch_pct_of_budget),
            "epc": int(args.epochs_per_cycle),
            "final_labeled_target": int(final_labeled_target),

            "step_type": "cycle",
            "step": 0,
            "labeled_count": int(labeled_cnt),
            "split": "val",
            "train_loss": fmt(start_loss),

            "acc": fmt(start_val_acc),
            "f1_macro": fmt(start_val_f1),
            "auc": fmt(start_val_auc),
            "ap": fmt(start_val_ap),

            "val_mean": fmt(start_mean),
            "select_metric": args.select_metric,
            "is_best": 1,
            "tp": -1, "fp": -1, "tn": -1, "fn": -1,
        }, RESULTS_PATH)

        best_sel = {"mean": start_mean, "acc": start_val_acc, "f1": start_val_f1, "auc": start_val_auc, "ap": start_val_ap}[args.select_metric]
        torch.save({
            "model_state_dict": wrapper.model.state_dict(),
            "in_channels": wrapper.in_channels,
            "num_classes": wrapper.num_classes,
            "data_dir": args.data_dir,
            "strategy": args.strategy,
            "init_size": int(resolved_init_size),
            "batch": int(resolved_batch),
            "budget": int(resolved_budget),
            "init_size_pct": float(resolved_init_size_pct),
            "budget_pct": float(resolved_budget_pct),
            "batch_pct_of_budget": float(resolved_batch_pct_of_budget),
            "epc": int(args.epochs_per_cycle),
            "final_labeled_target": int(final_labeled_target),
            "val_acc": float(start_val_acc),
            "val_f1": float(start_val_f1),
            "val_auc": float(start_val_auc),
            "val_ap": float(start_val_ap),
            "val_mean": float(start_mean),
            "select_metric": args.select_metric,
            "best_metric": float(best_sel),
            "best_cycle": 0,
            "labeled_count": int(labeled_cnt),
            "seed": int(args.seed),
        }, args.model_path)
        print(f"💾 Saved initial (cycle 0) model → {args.model_path}")

        # Validation
        best_ckpt = run_active_loop(
            active_ds,
            oracle,
            qs,
            wrapper,
            X_val,
            y_val,
            batch_schedule=resolved_batch_schedule,
            model_path=args.model_path,
            select_metric=args.select_metric,
            data_dir=args.data_dir,
            resolved_init_size=resolved_init_size,
            resolved_budget=resolved_budget,
            resolved_batch=resolved_batch,
            resolved_init_size_pct=resolved_init_size_pct,
            resolved_budget_pct=resolved_budget_pct,
            resolved_batch_pct_of_budget=resolved_batch_pct_of_budget,
            final_labeled_target=final_labeled_target,
            n_cycles_planned=n_cycles_planned,
            initial_best_sel=best_sel,
        )

        # Test
        X_test, y_test, _, _ = prepare_split_active(args.data_dir, split="test", to_nchw=True)
        checkpoint = torch.load(best_ckpt, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            wrapper.model.load_state_dict(checkpoint["model_state_dict"])

        if isinstance(checkpoint, dict) and "best_cycle" in checkpoint:
            print(f"🏁 Best checkpoint from cycle {checkpoint['best_cycle']} "
                f"(labeled={checkpoint.get('labeled_count','?')}), "
                f"val_acc={checkpoint.get('val_acc','?'):.4f}, "
                f"val_f1={checkpoint.get('val_f1','?'):.4f}, "
                f"val_auc={checkpoint.get('val_auc','?'):.4f}, "
                f"val_ap={checkpoint.get('val_ap','?'):.4f} "
                f"val_mean={checkpoint.get('val_mean','?'):.4f} "
                f"[select_metric={checkpoint.get('select_metric','?')}, "
                f"best_metric={checkpoint.get('best_metric','?'):.4f}]")

    print(f"📦 MODEL NAME: {args.model_path}")
    wrapper.model.eval()

    #val double check
    val_y_true_check, val_y_pred_check = get_predictions(wrapper.model, (X_val, y_val), DEVICE, forward_kwargs={"backbone_mode": "eval", "enable_dropout": False})
    val_acc_check = accuracy_score(val_y_true_check, val_y_pred_check)
    print(f"VAL ACC DOUBLE CHECK = {val_acc_check:.4f}")

    with torch.inference_mode():
        _, test_y_pred = get_predictions(wrapper.model, (X_test, y_test), DEVICE, forward_kwargs={"backbone_mode": "eval", "enable_dropout": False})
        test_proba = wrapper.predict_proba(X_test)[:, 1] if getattr(wrapper, "num_classes", None) == 2 else None
    
    tp, fp, tn, fn = class_report_conf_matrix(y_test, test_y_pred)

    test_acc = accuracy_score(y_test, test_y_pred)
    test_f1  = f1_score(y_test, test_y_pred, average="macro")

    test_auc = float("nan")
    test_ap  = float("nan")
    if test_proba is not None:
        try:
            test_auc = roc_auc_score(y_test, test_proba)
            test_ap  = average_precision_score(y_test, test_proba)
        except Exception:
            pass

    print(f"[TEST] acc={test_acc:.4f} "
        f"f1={test_f1:.4f} "
        f"auc={test_auc:.4f} "
        f"ap={test_ap:.4f} "
        f"(ckpt: {args.model_path})")

    if not args.eval_only:
        append_row_to_csv({
            "dataset": os.path.basename(os.path.normpath(args.data_dir)),
            "phase": "active",
            "strategy": args.strategy,
            "seed": int(args.seed),
            "model": os.path.basename(os.path.normpath(args.model_path)),

            "init_size": int(resolved_init_size),
            "batch": int(resolved_batch),
            "budget": int(resolved_budget),
            "init_size_pct": float(resolved_init_size_pct),
            "budget_pct": float(resolved_budget_pct),
            "batch_pct_of_budget": float(resolved_batch_pct_of_budget),
            "epc": int(args.epochs_per_cycle),
            "final_labeled_target": int(final_labeled_target),

            "step_type": "final",
            "step": -1,
            "labeled_count": checkpoint.get("labeled_count", ""),
            "split": "test",
            "train_loss": -1,

            "acc": fmt(test_acc),
            "f1_macro": fmt(test_f1),
            "auc": fmt(test_auc),
            "ap": fmt(test_ap),

            "val_mean": "",
            "select_metric": checkpoint.get("select_metric", args.select_metric),
            "is_best": -1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }, RESULTS_PATH)