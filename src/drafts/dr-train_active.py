import torch
import numpy as np
from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling
from libact.labelers import IdealLabeler
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn, optim
from load_data import get_dataloaders
from metrics import evaluate_predictions
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 1. Model wrapper zgodny z libact ====
class TorchModelWrapper:
    def __init__(self, in_channels, num_classes):
        self.in_channels = in_channels  # <-- MUSI BYĆ TUTAJ

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, X, y):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)

        # 🔁 Przywróć (C, H, W)
        X_tensor = X_tensor.view(-1, self.in_channels, 28, 28)

        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = self.loss_fn(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            X_tensor = X_tensor.view(-1, self.in_channels, 28, 28)

            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            X_tensor = X_tensor.view(-1, self.in_channels, 28, 28)

            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()


# ==== 2. Przygotowanie danych ====
def prepare_active_learning_data(dataset_name="bloodmnist", initial_label_size=100):
    train_loader, val_loader, test_loader = get_dataloaders(f"data/{dataset_name}", batch_size=512)

    X_all, y_all = [], []
    for x, y in train_loader:
        X_all.append(x.numpy())
        y_all.append(y.numpy())
    X_all = np.concatenate(X_all, axis=0)  # (N, C, H, W)
    y_all = np.concatenate(y_all, axis=0)  # (N,)

    # Spłaszcz dane + rzutuj na float32
    X_all_flat = X_all.reshape(X_all.shape[0], -1).astype(np.float32)
    y_all = y_all.astype(np.int32)

    # Podziel na labeled i unlabeled
    idxs = list(range(len(X_all_flat)))
    random.shuffle(idxs)
    labeled_idxs = idxs[:initial_label_size]
    unlabeled_idxs = idxs[initial_label_size:]

    X_labeled = X_all_flat[labeled_idxs]
    y_labeled = y_all[labeled_idxs]
    X_unlabeled = X_all_flat[unlabeled_idxs]

    # Datasety libact
    full_dataset = Dataset(X_all_flat, y_all)
    labeled_data = list(zip(X_labeled, y_labeled))
    unlabeled_data = [(x, None) for x in X_unlabeled]
    labeled_dataset = Dataset(labeled_data + unlabeled_data)

    return labeled_dataset, full_dataset, val_loader, test_loader, X_all.shape[1]


# ==== 3. Główna pętla aktywnego uczenia ====
def run_active_learning():
    labeled_dataset, full_dataset, val_loader, test_loader, in_channels = prepare_active_learning_data()

    labeler = IdealLabeler(full_dataset)
    learner_model = TorchModelWrapper(in_channels=in_channels, num_classes=2)
    strategy = UncertaintySampling(labeled_dataset, model=learner_model)

    ITERATIONS = 3
    for i in range(ITERATIONS):
        print(f"\n🔁 Iteracja {i+1}/{ITERATIONS}")

        query_idx, _ = strategy.make_query()
        x, _ = labeled_dataset.data[query_idx]
        y = labeler.label(x)
        labeled_dataset.update(query_idx, y)

        # Pobierz labeled dane do trenowania
        labeled_X = np.array([x for x, y in labeled_dataset.data if y is not None])
        labeled_y = np.array([y for x, y in labeled_dataset.data if y is not None])
        learner_model.fit(labeled_X, labeled_y)

    # Ewaluacja końcowa
    y_true, y_pred = [], []
    learner_model.model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            outputs = learner_model.model(x)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    evaluate_predictions(y_true, y_pred)


if __name__ == "__main__":
    run_active_learning()