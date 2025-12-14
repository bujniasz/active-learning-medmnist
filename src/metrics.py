import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset

def get_predictions(model, data, device, batch_size: int = 256):
    """
    Returns y_true, y_pred using a single interface:
    - data can be either DataLoaderEM, or
    - a tuple (X, y) as numpy arrays or torch.Tensors
    """
    model.eval()
    y_true, y_pred = [], []

    #works for baseline and active training
    if isinstance(data, DataLoader):
        loader = data
    else:
       X, y = data
       X_t = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
       # uint8 -> float/255, otherwise float
       # if the range looks like 0..255, divide by 255 as well
       if X_t.dtype == torch.uint8:
           X_t = X_t.float() / 255.0
       else:
           X_t = X_t.float()
           if X_t.max() > 1.5:  #if looks like 0..255
               X_t = X_t / 255.0
       if X_t.ndim == 3:  # (N,H,W) -> (N,1,H,W)
           X_t = X_t.unsqueeze(1)
       y_t = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
       loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred

def evaluate_predictions(y_true, y_pred, y_proba=None):
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

    print("\n📋 Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("📊 Confusion Matrix:")
    print(cm)
    
    if y_proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
            metrics["ap"] = average_precision_score(y_true, y_proba)
        except Exception as e:
            print(f"evaluate_predictions() - Skipping AUC/AP: {e}")

    return metrics

def save_metrics_to_csv(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame([metrics])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)