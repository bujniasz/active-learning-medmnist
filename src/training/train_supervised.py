# === IMPORTS ===
# General
import os
import argparse
import random
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Custom
from src.utils.load_data import prepare_split_supervised
from src.utils.shared import load_config, get_predictions, class_report_conf_matrix, fmt, append_row_to_csv, ResNet18EmbedDropout

"""
train_supervised.py

Trains or evaluates a medical image classifier (ResNet18) on a selected MedMNIST subset.
In training mode, it saves the best model based on validation accuracy.
In evaluation mode, it loads the saved model and computes test metrics.

Uses:
- load_data.py: for loading and preparing datasets (via DataLoader)
- metrics.py: for calculating classification metrics and saving them to CSV

Example usage:
    # Train a model on the dermamnist dataset:
    python train_supervised.py -d data/dermamnist -m models/dermamnist_model.pth
    # Evaluate a previously saved model:
    python train_supervised.py --eval-only -m models/dermamnist_model.pth
"""

# === ARGUMENTS ===
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--eval-only", action="store_true", help="Skip training of the model - just evaluate the existing one")
    p.add_argument("-d", "--data-dir", type=str, help="Path to data folder")
    p.add_argument("-mp", "--model-path", type=str, default=None, help="Path to the .pth model file (new or existing one)")
    p.add_argument("-r", "--results-path", type=str, default="results/test-exps-pt3.csv", help="Global CSV log path")
    p.add_argument("--select-metric", type=str, default="mean", choices=["mean", "acc", "f1", "auc", "ap"],
                        help="Metric used to select the best checkpoint (mean = average of acc,f1,auc,ap)")
    p.add_argument("--select-delta", type=float, default=1e-4, help="Minimum improvement required to save a new best checkpoint")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    p.add_argument("--max-epochs", type=int, default=33, help="Maximum number of training epochs")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
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

# === DEVICE ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_config(args):
    if args.config is None:
        return args

    cfg = load_config(args.config)

    args.data_dir = cfg.get("data_dir", args.data_dir)
    args.model_path = cfg.get("model_path", args.model_path)
    args.results_path = cfg.get("results_path", args.results_path)
    args.seed = cfg.get("seed", args.seed)

    return args

# === MODEL BUILDER ===
def get_model(num_classes, in_channels, dropout_p: float = 0.2):
    return ResNet18EmbedDropout(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)

def get_num_classes(model) -> int | None:
    # plain resnet
    if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        return int(model.fc.out_features)
    # your wrapper model
    if hasattr(model, "backbone") and hasattr(model.backbone, "fc") and hasattr(model.backbone.fc, "out_features"):
        return int(model.backbone.fc.out_features)
    return None

# === TRAINING + VALIDATION LOOP === 
def run_supervised_loop(model, train_loader, val_loader, *,
                        max_epochs: int, patience: int, 
                        select_metric: str,
                        model_path: str, data_dir: str,
                        in_channels: int, num_classes: int) -> str:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_sel = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # --- train ---
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, backbone_mode="train", enable_dropout=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0); correct += (predicted == targets).sum().item()
        train_acc = correct / total
        avg_train_loss = running_loss / len(train_loader)

        # --- val ---
        model.eval()
        val_y_true, val_y_pred = get_predictions(model, val_loader, DEVICE, forward_kwargs={"backbone_mode": "eval", "enable_dropout": False})
        val_acc = accuracy_score(val_y_true, val_y_pred)
        val_y_proba = None
        if get_num_classes(model) == 2:
            val_y_proba = []
            with torch.inference_mode():
                for inputs, _ in val_loader:
                    inputs = inputs.to(DEVICE)
                    probs = torch.softmax(model(inputs, backbone_mode="eval", enable_dropout=False), dim=1)[:, 1]
                    val_y_proba.extend(probs.detach().cpu().tolist())
        val_f1  = f1_score(val_y_true, val_y_pred, average='macro')
        val_auc = roc_auc_score(val_y_true, val_y_proba) if val_y_proba is not None else float('nan')
        val_ap  = average_precision_score(val_y_true, val_y_proba) if val_y_proba is not None else float('nan')
        
        # --- mean score ---
        vals = np.array([val_acc, val_f1, val_auc, val_ap], dtype=float)
        val_mean = float(np.nanmean(vals))
        if np.isnan(val_mean):
            val_mean = float("-inf")

        print(f"Epoch [{epoch+1}/{max_epochs}] - Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f} "
        f"- Val acc={val_acc:.4f} Val f1={val_f1:.4f} Val auc={val_auc:.4f} Val ap={val_ap:.4f} "
        f"Val mean={val_mean:.4f}")

        sel_map = {"mean": val_mean, "acc": val_acc, "f1": val_f1, "auc": val_auc, "ap": val_ap}
        sel = float(sel_map[select_metric])

        if np.isnan(sel):
            sel = float("-inf")

        delta = getattr(args, "select_delta", 0.0)  
        is_best = 1 if (sel > best_sel + delta) else 0
        append_row_to_csv({
            "dataset": os.path.basename(os.path.normpath(data_dir)),
            "phase": "supervised",
            "strategy": "full_train_set",
            "seed": int(args.seed),
            "model": os.path.basename(os.path.normpath(model_path)),

            "init_size": -1,
            "batch": -1,
            "budget": -1,
            "init_size_pct": -1,
            "budget_pct": -1,
            "batch_pct_of_budget": -1,
            "epc": -1,
            "final_labeled_target": -1,
            
            "step_type": "epoch",
            "step": int(epoch + 1),
            "labeled_count": int(len(train_loader.dataset)),
            "split": "val",
            "train_loss": fmt(avg_train_loss),

            "acc": fmt(val_acc),
            "f1_macro": fmt(val_f1),
            "auc": fmt(val_auc),
            "ap": fmt(val_ap),

            "val_mean": fmt(val_mean),
            "select_metric": select_metric,
            "is_best": int(is_best),
            "tp": -1, "fp": -1, "tn": -1, "fn": -1,
        }, RESULTS_PATH)

        if sel > best_sel + delta:
            best_sel = sel
            epochs_without_improvement = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'in_channels': in_channels,
                'num_classes': num_classes,
                'data_dir': data_dir,
                'train_size': int(len(train_loader.dataset)),
                'val_acc': float(val_acc),
                'val_f1': float(val_f1),
                'val_auc': float(val_auc),
                'val_ap': float(val_ap),
                'val_mean': float(val_mean),
                'select_metric': select_metric,
                'best_metric': float(sel),
                'best_epoch': int(epoch + 1),
                'max_epochs': int(max_epochs),
                'patience': int(patience),
                "seed": int(args.seed),
            }, model_path)

            print(f"✅ NEW BEST (by {select_metric}) → {best_sel:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"⏳ No improvement for {epochs_without_improvement}/{patience} epoch(s)"
            )

            if epochs_without_improvement >= patience:
                print(
                    f"🛑 Early stopping at epoch {epoch + 1}. "
                    f"Best {select_metric}: {best_sel:.4f}"
                )
                break
    return model_path

# === MAIN LOOP ===
if __name__ == "__main__":

    args = parse_args()

    args = apply_config(args)

    if args.model_path is None:
        raise SystemExit("model_path must be provided either via CLI or config")

    if not args.eval_only and args.data_dir is None:
        raise SystemExit("data_dir must be provided either via CLI or config")

    set_seed(args.seed)

    if args.results_path is not None:
        RESULTS_PATH = args.results_path
    else:
        model_filename = os.path.basename(args.model_path)
        model_name = os.path.splitext(model_filename)[0]
        RESULTS_PATH = os.path.join("results", model_name + ".csv")

    # ======= Evaluation mode (without training) =======
    if args.eval_only:
        print("🔍 Mode: evaluation only (SUPERVISED)")

        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        num_classes = checkpoint['num_classes']
        in_channels = checkpoint['in_channels']

        DATA_DIR = checkpoint['data_dir']

        # This call now also handles binary label mapping and filtering
        _, val_loader, test_loader = prepare_split_supervised(DATA_DIR, batch_size=args.batch_size)

        model = get_model(num_classes, in_channels)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

    # ======= TRAINING + VALIDATION LOOP =======
    else:
        print("🚀 Mode: training + evaluation (SUPERVISED)")
        DATA_DIR = args.data_dir

        # This call now also handles binary label mapping and filtering
        train_loader, val_loader, test_loader = prepare_split_supervised(DATA_DIR, batch_size=args.batch_size, seed=args.seed)

        sample_x, _ = next(iter(train_loader))
        in_channels = sample_x.shape[1]
        num_classes = len(torch.unique(torch.cat([y for _, y in train_loader])))

        print(f"📊 Detected: {num_classes} classes, {in_channels} channels\n")

        model = get_model(num_classes, in_channels).to(DEVICE)

        best_ckpt = run_supervised_loop(model, train_loader, val_loader,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        select_metric=args.select_metric,
                        model_path=args.model_path,
                        data_dir=DATA_DIR,
                        in_channels=in_channels,
                        num_classes=num_classes)

        checkpoint = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

    # ======= EVALUATION =======
    print(f"📦 MODEL NAME: {args.model_path}")
    model.eval()

    with torch.inference_mode():
        val_y_true_check, val_y_pred_check = get_predictions(model, val_loader, DEVICE, forward_kwargs={"backbone_mode": "eval", "enable_dropout": False})
        val_acc_check = accuracy_score(val_y_true_check, val_y_pred_check)
        test_y_true, test_y_pred = get_predictions(model, test_loader, DEVICE, forward_kwargs={"backbone_mode": "eval", "enable_dropout": False})
    print(f"📈 Validation check: val_acc = {val_acc_check:.4f}")

    test_proba = None
    if get_num_classes(model) == 2:
        y_proba_list = []
        with torch.inference_mode():
            for inputs, _ in test_loader:
                inputs = inputs.to(DEVICE)
                logits = model(inputs, backbone_mode="eval", enable_dropout=False)
                probs = torch.softmax(logits, dim=1)[:, 1]
                y_proba_list.extend(probs.detach().cpu().tolist())
        test_proba = y_proba_list

    tp, fp, tn, fn = class_report_conf_matrix(test_y_true, test_y_pred)

    test_acc = accuracy_score(test_y_true, test_y_pred)
    test_f1  = f1_score(test_y_true, test_y_pred, average="macro")

    test_auc = float("nan")
    test_ap  = float("nan")
    if test_proba is not None:
        try:
            test_auc = roc_auc_score(test_y_true, test_proba)
            test_ap  = average_precision_score(test_y_true, test_proba)
        except Exception:
            pass

    print(f"[TEST] acc={test_acc:.4f} "
        f"f1={test_f1:.4f} "
        f"auc={test_auc:.4f} "
        f"ap={test_ap:.4f} "
        f"(ckpt: {args.model_path})")

    if not args.eval_only:
        append_row_to_csv({
            "dataset": os.path.basename(os.path.normpath(DATA_DIR)),
            "phase": "supervised",
            "strategy": "full_train_set",
            "seed": int(args.seed),
            "model": os.path.basename(os.path.normpath(args.model_path)),

            "init_size": -1,
            "batch": -1,
            "budget": -1,
            "init_size_pct": -1,
            "budget_pct": -1,
            "batch_pct_of_budget": -1,
            "epc": -1,
            "final_labeled_target": -1,

            "step_type": "final",
            "step": -1,
            "labeled_count": checkpoint.get("train_size", ""),
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
