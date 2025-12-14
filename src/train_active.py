# === IMPORTS ===
# General
import os
import numpy as np
import argparse
import random
from pathlib import Path

# Torch
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights   

# Sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Libact
from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling
from libact.labelers import IdealLabeler
from libact.base.interfaces import ProbabilisticModel

# Custom
from load_data import prepare_split_active
from metrics import get_predictions, evaluate_predictions, save_metrics_to_csv

# === DEVICE ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CUSTOM WRAPPER FOR libact <-> resnet TO WORK ===
# === https://github.com/ntucllab/libact/blob/master/libact/base/interfaces.py ===
class TorchModelWrapper(ProbabilisticModel):
    def __init__(self, in_channels: int, num_classes: int, lr: float = 1e-3, epochs_per_cycle: int = 1, seed: int | None = None):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(DEVICE)
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
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.detach().cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def train_on_numpy(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 64, verbose: bool = False):
        self.model.train()
        X_t = torch.from_numpy(X)
        if X_t.dtype == torch.uint8:
            X_t = X_t.float() / 255.0
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(1)
        y_t = torch.from_numpy(y).long()
        ds = TensorDataset(X_t, y_t)         
        n = len(ds)
        if n < 2:
            return  # skipping if not enough instances
        eff_bs = min(batch_size, n)
        gen = None
        if self.seed is not None:
            gen = torch.Generator()
            gen.manual_seed(self.seed)

        dl = DataLoader(ds, batch_size=eff_bs, shuffle=True, drop_last=True, generator=gen, num_workers=0)
        for _ in range(epochs):
            total_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dl)
            if verbose:
                print(f"   🔹 Training loss: {avg_loss:.4f}")

    def train(self, dataset, verbose: bool = False):
        X_l, y_l = dataset.get_labeled_entries()
        if len(y_l) == 0:
            return
        X_arr = np.stack(X_l)
        y_arr = np.asarray(y_l, dtype=np.int64)
        self.train_on_numpy(X_arr, y_arr, epochs=getattr(self, "epochs_per_cycle", 1), verbose=verbose)

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
    p.add_argument("--eval-only", action="store_true", help="Skip training of the model - just evaluate the existing one")
    p.add_argument("-d", "--data-dir", type=str, help="Path to data folder")
    p.add_argument("-m", "--model-path", type=str, required=True, help="Path to the .pth model file (new or existing one)")
    p.add_argument("-r", "--results-path", type=str, default=None,
               help="Path to the .csv file with evaluation results (if none provided it's the same as model-path)")
    p.add_argument("--init-size", type=int, default=500)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--batch", type=int, default=10, help="queries per AL cycle")
    p.add_argument("--epochs-per-cycle", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--method", type=str, default="lc", choices=["lc", "sm", "entropy"])
    p.add_argument("--select-metric", type=str, default="acc", choices=["acc", "f1", "auc", "ap"])
    p.add_argument("--seed", type=int, default=42)
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
def init_libact( X: np.ndarray, y: np.ndarray, init_size: int, method: str, wrapper: TorchModelWrapper, seed: int = 42):
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
        qs = UncertaintySampling(active_ds, method=method, model=wrapper)
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
    qs = UncertaintySampling(active_ds, method=method, model=wrapper)

    return active_ds, oracle, qs, init_idx

# === TRAINING + VALIDATION LOOP === 
def run_budget_loop_val(active_ds, oracle, qs, wrapper, X_val, y_val,
                        budget: int, batch: int, model_path: str,
                        select_metric: str = "acc",
                        data_dir: str | None = None) -> str:
    
    asked = 0; cycle = 0; best_sel = float("-inf")
    Path(Path(model_path).parent).mkdir(parents=True, exist_ok=True)
    while asked < budget:
        k = min(batch, budget - asked)
        for _ in range(k):
            ask_id = qs.make_query()
            y_new = oracle.label(active_ds.data[ask_id][0])
            active_ds.update(ask_id, y_new)
        wrapper.train(active_ds, verbose=True)
        cycle += 1; asked += k
        _, y_pred=get_predictions(wrapper.model, (X_val, y_val), DEVICE)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        proba = None
        auc = float("nan")
        ap = float("nan")
        if wrapper.num_classes == 2:
            proba = wrapper.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            ap = average_precision_score(y_val, proba)
        labeled_cnt = sum(lbl is not None for _, lbl in active_ds.data)
        print(f"[cycle {cycle}/{int(budget / batch)}] labeled={labeled_cnt} val_acc={acc:.4f} val_f1={f1:.4f} val_auc={auc:.4f} val_ap={ap:.4f}")
        sel = {"acc": acc, "f1": f1, "auc": auc, "ap": ap}[select_metric]
        if sel > best_sel:
            best_sel = sel
            torch.save({
                "model_state_dict": wrapper.model.state_dict(),
                "in_channels": wrapper.in_channels,
                "num_classes": wrapper.num_classes,
                "data_dir": data_dir,
                "val_acc": float(acc),
                "val_f1": float(f1),
                "val_auc": float(auc),
                "val_ap": float(ap),
                "select_metric": select_metric,
                "best_metric": float(sel),
                "best_cycle": int(cycle),
                "labeled_count": int(labeled_cnt),
                "seed": int(args.seed),
            }, model_path)
            print(f"✅ NEW BEST (by {args.select_metric}) → {best_sel:.4f}")
    return model_path

# === MAIN LOOP ===
if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    
    if args.results_path is not None:
        RESULTS_PATH = args.results_path
    else:
        model_filename = os.path.basename(args.model_path)
        model_name = os.path.splitext(model_filename)[0]
        RESULTS_PATH = os.path.join("results", model_name + ".csv")
    
    if args.batch <= 0: raise ValueError("--batch must be > 0")
    if args.budget <= 0: raise ValueError("--budget must be > 0")
    if args.budget % args.batch != 0:
        raise ValueError(
            f"Invalid combination: budget ({args.budget}) not divisible by batch ({args.batch}). "
            "Each cycle must have the same number of samples."
        )

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

        wrapper = TorchModelWrapper(in_channels=in_channels, num_classes=num_classes, lr=args.lr, epochs_per_cycle=args.epochs_per_cycle, seed=args.seed)

        active_ds, oracle, qs, init_idx = init_libact(X, y, args.init_size, args.method, wrapper, seed=args.seed)

        y = np.asarray(y)
        init_labels = y[init_idx]
        unique, counts = np.unique(init_labels, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        class0 = class_dist.get(0, 0)
        class1 = class_dist.get(1, 0)

        print(
            f"Start: labeled={len(init_idx)}, unlabeled={len(y) - len(init_idx)}, "
            f"class 0 = {class0}, class 1 = {class1}"
        )

        ### FIRST TRAINING BEFORE ANOTATIONS
        #wrapper.train(active_ds)
        print("\n🔸 Initial training on starting labeled set")
        wrapper.train(active_ds, verbose=True)

        # === Initial evaluation (cycle 0) ===
        print("🔍 Evaluating initial model (cycle 0)...")
        X_val, y_val, _, _ = prepare_split_active(args.data_dir, split="val", to_nchw=True)
        yv_t, yv_p = get_predictions(wrapper.model, (X_val, y_val), DEVICE)

        acc0 = accuracy_score(y_val, yv_p)
        f10 = f1_score(y_val, yv_p, average='macro')

        proba0 = None
        auc0 = float("nan")
        ap0 = float("nan")
        if wrapper.num_classes == 2:
            proba0 = wrapper.predict_proba(X_val)[:, 1]
            auc0 = roc_auc_score(y_val, proba0)
            ap0 = average_precision_score(y_val, proba0)

        labeled_cnt = sum(lbl is not None for _, lbl in active_ds.data)
        print(
            f"[cycle 0] labeled={labeled_cnt} "
            f"val_acc={acc0:.4f} val_f1={f10:.4f} "
            f"val_auc={auc0:.4f} val_ap={ap0:.4f}"
        )

        # === OPTIONAL: Save cycle-0 model as current best ===
        best_sel = {"acc": acc0, "f1": f10, "auc": auc0, "ap": ap0}[args.select_metric]
        torch.save({
            "model_state_dict": wrapper.model.state_dict(),
            "in_channels": wrapper.in_channels,
            "num_classes": wrapper.num_classes,
            "data_dir": args.data_dir,
            "val_acc": float(acc0),
            "val_f1": float(f10),
            "val_auc": float(auc0),
            "val_ap": float(ap0),
            "select_metric": args.select_metric,
            "best_metric": float(best_sel),
            "best_cycle": 0,
            "labeled_count": int(labeled_cnt),
            "seed": int(args.seed),
        }, args.model_path)
        print(f"💾 Saved initial (cycle 0) model → {args.model_path}")

        # Validation
        #X_val, y_val, _, _ = prepare_split_active(args.data_dir, split="val", to_nchw=True)
        best_ckpt = run_budget_loop_val(active_ds, oracle, qs, wrapper,
                                    X_val, y_val,
                                    budget=args.budget, batch=args.batch,
                                    model_path=args.model_path,
                                    select_metric=args.select_metric,
                                    data_dir=args.data_dir)

        # Test
        X_test, y_test, _, _ = prepare_split_active(args.data_dir, split="test", to_nchw=True)
        _ckpt = torch.load(best_ckpt, map_location=DEVICE)
        if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt:
            wrapper.model.load_state_dict(_ckpt["model_state_dict"])

        if isinstance(_ckpt, dict) and "best_cycle" in _ckpt:
            print(f"🏁 Best checkpoint from cycle {_ckpt['best_cycle']} "
                f"(labeled={_ckpt.get('labeled_count','?')}), "
                f"val_acc={_ckpt.get('val_acc','?'):.4f}, "
                f"val_f1={_ckpt.get('val_f1','?'):.4f}, "
                f"val_auc={_ckpt.get('val_auc','?'):.4f}, "
                f"val_ap={_ckpt.get('val_ap','?'):.4f} "
                f"[select_metric={_ckpt.get('select_metric','?')}, "
                f"best_metric={_ckpt.get('best_metric','?'):.4f}]")

    print(f"📦 MODEL NAME: {args.model_path}")
    wrapper.model.eval()

    #val double check
    yv_t, yv_p = get_predictions(wrapper.model, (X_val, y_val), DEVICE)
    val_acc_check = accuracy_score(yv_t, yv_p)
    print(f"VAL ACC DOUBLE CHECK = {val_acc_check:.4f}")

    with torch.inference_mode():
        y_true, y_pred = get_predictions(wrapper.model, (X_test, y_test), DEVICE)
        proba = wrapper.predict_proba(X_test)[:, 1] if getattr(wrapper, "num_classes", None) == 2 else None
    
    metrics = evaluate_predictions(y_true, y_pred, y_proba=proba)
    print(f"[TEST] acc={metrics['accuracy']:.4f} "
        f"f1={metrics['f1_macro']:.4f} "
        f"auc={metrics.get('auc', float('nan')):.4f} "
        f"ap={metrics.get('ap', float('nan')):.4f} "
        f"(ckpt: {args.model_path})")
    
    metrics["seed"] = int(args.seed)
    save_metrics_to_csv(metrics, RESULTS_PATH)
    print(f"💾 Metrics saved to: {RESULTS_PATH}")    