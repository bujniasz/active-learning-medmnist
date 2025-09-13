import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from load_data import get_dataloaders
from metrics import get_predictions, evaluate_predictions, save_metrics_to_csv
import argparse

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
parser = argparse.ArgumentParser()
parser.add_argument("--eval-only", action="store_true", help="Skip training of the model - just evaluate the existing one")
parser.add_argument("-d", "--data-dir", type=str, help="Path to data folder")
parser.add_argument("-m", "--model-path", type=str, required=True, help="Path to the .pth model file (new or existing one)")
parser.add_argument("-r", "--results-path", type=str, required=True, help="Path to the .csv file with evaluation results")
args = parser.parse_args()

# ======= Parameters =======
BEST_MODEL_PATH = args.model_path
BATCH_SIZE = 64
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ======= Evaluation mode (without training) =======
if args.eval_only:
    print("🔍 Mode: evaluation only")

    checkpoint = torch.load(BEST_MODEL_PATH)
    num_classes = checkpoint['num_classes']
    in_channels = checkpoint['in_channels']

    DATA_DIR = checkpoint['data_dir']

    _, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    model = get_model(num_classes, in_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

# ======= Full training mode =======
else:
    print("🚀 Mode: training + evaluation")
    DATA_DIR = args.data_dir

    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    sample_x, _ = next(iter(train_loader))
    in_channels = sample_x.shape[1]
    num_classes = len(torch.unique(torch.cat([y for _, y in train_loader])))

    print(f"📊 Detected: {num_classes} classes, {in_channels} channels\n")

    model = get_model(num_classes, in_channels).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
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
        val_acc = evaluate(model, val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'in_channels': in_channels,
                'num_classes': num_classes,
                'val_acc': val_acc,
                'data_dir': DATA_DIR
            }, BEST_MODEL_PATH)
            print(f"✅ NEW BEST MODEL FOUND (val_acc = {val_acc:.4f})")

    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

# ======= Evaluation on test set + metrics =======
print(f"📦 MODEL NAME: {BEST_MODEL_PATH}")

test_acc = evaluate(model, test_loader)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

val_acc_check = evaluate(model, val_loader)
print(f"📈 Validation check: val_acc = {val_acc_check:.4f}")

y_true, y_pred = get_predictions(model, test_loader, DEVICE)
metrics = evaluate_predictions(y_true, y_pred)
save_metrics_to_csv(metrics, args.results_path)
