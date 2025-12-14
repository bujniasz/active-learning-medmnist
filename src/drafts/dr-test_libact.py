import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.query_strategies import UncertaintySampling
from libact.models import SklearnProbaAdapter

from load_data import get_dataloaders  # Twój kod

from typing import Optional

def main():
    # ==== 1. Wczytaj dane ====
    data_path = "data/bloodmnist"
    train_loader, _, test_loader = get_dataloaders(data_path, batch_size=99999)

    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))

    X_train = X_train.numpy().reshape(len(X_train), -1)
    y_train = y_train.numpy()

    X_test = X_test.numpy().reshape(len(X_test), -1)
    y_test = y_test.numpy()

    # ==== 2. Parametry eksperymentu ====
    n_initial = 20
    n_queries = 100
    random_state = 42
    np.random.seed(random_state)

    # ==== 3. Przygotuj dataset ====
    initial_indices = np.random.choice(len(X_train), size=n_initial, replace=False)

    # lepiej: Optional[int], bo potem wstawiamy liczby
    labeled: list[Optional[int]] = [None] * len(y_train)

    for idx in initial_indices:
        labeled[idx] = int(y_train[idx])


    # Dataset aktywnego uczenia
    active_dataset = Dataset(X_train.tolist(), labeled)

    # Oracle zna wszystkie etykiety
    fully_labeled_dataset = Dataset(X_train.tolist(), y_train.tolist())
    oracle = IdealLabeler(fully_labeled_dataset)

    # Strategia + model
    model = SklearnProbaAdapter(LogisticRegression(max_iter=1000))
    qs = UncertaintySampling(active_dataset, model=model)

    # ==== 4. Pętla aktywnego uczenia ====
    accuracies = []

    for i in range(n_queries):
        ask_id = qs.make_query()

        if ask_id is None:
            print("🔁 Brak nieoznaczonych przykładów — kończymy pętlę.")
            break

        lbl = oracle.label(ask_id)
        active_dataset.update(ask_id, lbl)

        model.train(active_dataset)
        preds = model.predict(X_test.tolist())
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)

        if i % 10 == 0 or i == n_queries - 1:
            print(f"[{i+1:03d}] Test accuracy: {acc:.4f}")

    # ==== 5. Wykres ====
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker="o")
    plt.title("Active Learning on BloodMNIST (Logistic Regression)")
    plt.xlabel("Number of Queries")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()