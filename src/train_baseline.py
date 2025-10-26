# === IMPORTS ===
# General
import os
import argparse

# Torch
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

# Sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Custom
from load_data import prepare_split_baseline
from metrics import get_predictions, evaluate_predictions, save_metrics_to_csv

"""
train_baseline.py

Trains or evaluates a medical image classifier (ResNet18) on a selected MedMNIST subset.
In training mode, it saves the best model based on validation accuracy.
In evaluation mode, it loads the saved model and computes test metrics.

Uses:
- load_data.py: for loading and preparing datasets (via DataLoader)
- metrics.py: for calculating classification metrics and saving them to CSV

Example usage:
    # Train a model on the dermamnist dataset:
    python train_baseline.py -d data/dermamnist -m models/dermamnist_model.pth
    # Evaluate a previously saved model:
    python train_baseline.py --eval-only -m models/dermamnist_model.pth
"""

# ======= Args to parse =======
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-only", action="store_true", help="Skip training of the model - just evaluate the existing one")
    p.add_argument("-d", "--data-dir", type=str, help="Path to data folder")
    p.add_argument("-m", "--model-path", type=str, required=True, help="Path to the .pth model file (new or existing one)")
    p.add_argument("-r", "--results-path", type=str, default=None, 
                        help="Path to the .csv file with evaluation results (if none provided it's the same as model-path)")
    p.add_argument("--select-metric", type=str, default="acc",
                        choices=["acc", "f1", "auc", "ap"],
                        help="Metric used to select the best checkpoint")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    return p.parse_args()

# ======= Parameters =======
# BEST_MODEL_PATH = args.model_path
# BATCH_SIZE = 64
# EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if args.results_path is not None:
#     RESULTS_PATH = args.results_path
# else:
#     model_filename = os.path.basename(args.model_path)
#     model_name = os.path.splitext(model_filename)[0]
#     RESULTS_PATH = os.path.join("results", model_name + ".csv")

# ======= Model builder =======
def get_model(num_classes, in_channels):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ======= Evaluation function =======
def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


if __name__ == "__main__":

    args = parse_args()

    if args.results_path is not None:
        RESULTS_PATH = args.results_path
    else:
        model_filename = os.path.basename(args.model_path)
        model_name = os.path.splitext(model_filename)[0]
        RESULTS_PATH = os.path.join("results", model_name + ".csv")

    # ======= Evaluation mode (without training) =======
    if args.eval_only:
        print("🔍 Mode: evaluation only (BASELINE)")

        checkpoint = torch.load(args.model_path)
        num_classes = checkpoint['num_classes']
        in_channels = checkpoint['in_channels']

        DATA_DIR = checkpoint['data_dir']

        # This call now also handles binary label mapping and filtering
        _, val_loader, test_loader = prepare_split_baseline(DATA_DIR, batch_size=args.batch_size)

        model = get_model(num_classes, in_channels)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

    # ======= Full training mode =======
    else:
        print("🚀 Mode: training + evaluation (BASELINE)")
        DATA_DIR = args.data_dir

        # This call now also handles binary label mapping and filtering
        train_loader, val_loader, test_loader = prepare_split_baseline(DATA_DIR, batch_size=args.batch_size)

        sample_x, _ = next(iter(train_loader))
        in_channels = sample_x.shape[1]
        num_classes = len(torch.unique(torch.cat([y for _, y in train_loader])))

        print(f"📊 Detected: {num_classes} classes, {in_channels} channels\n")

        model = get_model(num_classes, in_channels).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_val_acc = 0.0

        for epoch in range(args.epochs):
            model.train()
            running_loss, total, correct = 0.0, 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            train_acc = correct / total
            avg_train_loss = running_loss / len(train_loader)
            #val_acc_old = evaluate(model, val_loader)

            # === Validation ===
            model.eval()
            yv_true, yv_pred = get_predictions(model, val_loader, DEVICE)
            val_acc = accuracy_score(yv_true, yv_pred)
            # proba only for binary case
            yv_proba = None
            if getattr(model, "fc", None) is not None and getattr(model.fc, "out_features", None) == 2:
                yv_proba = []
                with torch.inference_mode():
                    for inputs, _ in val_loader:
                        inputs = inputs.to(DEVICE)
                        probs = torch.softmax(model(inputs), dim=1)[:, 1]
                        yv_proba.extend(probs.detach().cpu().tolist())
            # if model.fc.out_features == 2:
            #     yv_proba = []
            #     model.eval()
            #     with torch.no_grad():
            #         for inputs, _ in val_loader:
            #             probs = F.softmax(model(inputs.to(DEVICE)), dim=1)[:, 1]
            #             yv_proba.extend(probs.detach().cpu().tolist())
            val_f1  = f1_score(yv_true, yv_pred, average='macro')
            val_auc = roc_auc_score(yv_true, yv_proba) if yv_proba is not None else float('nan')
            val_ap  = average_precision_score(yv_true, yv_proba) if yv_proba is not None else float('nan')
            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f} "
                    f"- Val acc={val_acc:.4f} Val f1={val_f1:.4f} Val auc={val_auc:.4f} Val ap={val_ap:.4f}")

            sel = {"acc": val_acc, "f1": val_f1, "auc": val_auc, "ap": val_ap}[args.select_metric]
            if epoch == 0:
                best_sel = sel
            if sel >= best_sel:
                best_sel = sel
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'in_channels': in_channels,
                    'num_classes': num_classes,
                    'data_dir': DATA_DIR,
                    'val_acc': float(val_acc),
                    'val_f1': float(val_f1),
                    'val_auc': float(val_auc),
                    'val_ap': float(val_ap),
                    'select_metric': args.select_metric,
                    'best_metric': float(sel),
                    'best_epoch': int(epoch + 1),
                }, args.model_path)
                print(f"✅ NEW BEST (by {args.select_metric}) → {best_sel:.4f}")

            # print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     torch.save({
            #         'model_state_dict': model.state_dict(),
            #         'in_channels': in_channels,
            #         'num_classes': num_classes,
            #         'val_acc': val_acc,
            #         'data_dir': DATA_DIR
            #     }, BEST_MODEL_PATH)
            #     print(f"✅ NEW BEST MODEL FOUND (val_acc = {val_acc:.4f})")

        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

    # ======= Evaluation on test set + metrics =======
    print(f"📦 MODEL NAME: {args.model_path}")
    model.eval()

    # test_acc = evaluate(model, test_loader)
    # print(f"\n✅ Test accuracy: {test_acc:.4f}")

    # val_acc_check = evaluate(model, val_loader)
    # print(f"📈 Validation check: val_acc = {val_acc_check:.4f}")

    with torch.inference_mode():
        y_val_true, y_val_pred = get_predictions(model, val_loader, DEVICE)
        val_acc_check = accuracy_score(y_val_true, y_val_pred)
    print(f"📈 Validation check: val_acc = {val_acc_check:.4f}")

    y_true, y_pred = get_predictions(model, test_loader, DEVICE)

    proba = None
    if getattr(model, "fc", None) is not None and getattr(model.fc, "out_features", None) == 2:
        y_proba_list = []
        with torch.inference_mode():
            for inputs, _ in test_loader:
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                y_proba_list.extend(probs.detach().cpu().tolist())
        proba = y_proba_list
    # if model.fc.out_features == 2: #it makes sense only for binary labels but the mechanism could be better
    #     y_proba_list = []
    #     with torch.no_grad():
    #         for inputs, _ in test_loader:
    #             inputs = inputs.to(DEVICE)
    #             probs = F.softmax(model(inputs), dim=1)[:, 1]
    #             y_proba_list.extend(probs.cpu().tolist())
    #     proba = y_proba_list

    metrics = evaluate_predictions(y_true, y_pred, y_proba=proba)
    print(f"[TEST] acc={metrics['accuracy']:.4f} "
        f"f1={metrics['f1_macro']:.4f} "
        f"auc={metrics.get('auc', float('nan')):.4f} "
        f"ap={metrics.get('ap', float('nan')):.4f} "
        f"(ckpt: {args.model_path})")
    save_metrics_to_csv(metrics, RESULTS_PATH)
