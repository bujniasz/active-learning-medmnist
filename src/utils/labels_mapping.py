import numpy as np

def map_labels(dataset_name: str, labels: np.ndarray) -> np.ndarray:
    """
    Maps original multi-class labels to binary labels for selected MedMNIST datasets.

    Args:
        dataset_name (str): name of the dataset, e.g., 'bloodmnist'
        labels (np.ndarray): original label array (shape: [N] or [N, 1])

    Returns:
        np.ndarray: binary label array (shape: [N])
    """
    # Ensure labels are in flat shape [N]
    labels = labels.squeeze()

    if dataset_name.lower() == "bloodmnist":
        """
            Original labels -> https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py#L185-L210
            Patology: labels 2-3
            Physiology: Everything except 2 and 3 is physiologically present in the blood
        """
        patho_labels = {2, 3}
    
    elif dataset_name.lower() == "octmnist":
        """
            Original labels -> https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py#L112-L133
            Patology: labels 0-2 
            Physiology: label 3 (normal)
        """
        patho_labels = {0, 1, 2}
    
    elif dataset_name.lower() == "pathmnist":
        """
            Original labels ->  https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py#L40-L54
            Pathology: labels 2, 7, 8
            Physiology: labels 0, 4, 5, 6
            Discarded: 1 = background, 3 = lymphocytes
        """
        patho_labels = {2, 7, 8}

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    binary_labels = np.array([1 if label in patho_labels else 0 for label in labels], dtype=np.int64)
    return binary_labels

def get_valid_indices(dataset_name: str, labels: np.ndarray) -> list[int]:
    """
    Returns list of indices for which labels are valid (i.e., not to be discarded).
    Makes changes only for datasets that require filtering (e.g. pathmnist).
    """
    labels = labels.squeeze()

    if dataset_name.lower() == "pathmnist":
        discard_labels = {1, 3}
        return [i for i, label in enumerate(labels) if label not in discard_labels]

    else:
        return list(range(len(labels)))