import numpy as np
from collections import Counter
from utils.labels_mapping import map_labels, get_valid_indices
import os

def analyze_split(split_name: str, data_dir: str):
    dataset_name = os.path.basename(os.path.normpath(data_dir))

    print(f"\n🔍 SUBSET: {split_name.upper()}")

    # data import
    images = np.load(f"{data_dir}/{split_name}_images.npy")
    labels = np.load(f"{data_dir}/{split_name}_labels.npy").squeeze()

    # images and labels info
    print(f"[IMAGES] shape: {images.shape}, dtype: {images.dtype}, min: {images.min()}, max: {images.max()}")
    unique_shapes = set([img.shape for img in images])
    print("[IMAGES] Unique shapes", unique_shapes)

    print(f"[LABELS] shape: {labels.shape}, dtype: {labels.dtype}, unique values: {np.unique(labels)}")
    print(f"[LABELS] examples:\n{labels[:5]}")

    # before binary encoding
    print("\n🎯 LABELS - BEFORE BINARY ENCODING:")
    class_counts_before = Counter(labels)
    total_before = len(labels)
    for cls, count in sorted(class_counts_before.items()):
        print(f"class {cls} - {count} elements")
    print(f"total - {total_before} elements")

    if dataset_name in binary_mapping_required:
        # binary encoding (and optional filtering)
        valid_indices = get_valid_indices(dataset_name, labels)
        labels = labels[valid_indices]
        binary_labels = map_labels(dataset_name, labels)

        # after binary encoding
        print("\n✅  LABELS - after BINARY ENCODING")
        class_counts_after = Counter(binary_labels)
        total_after = len(binary_labels)
        for cls, count in sorted(class_counts_after.items()):
            print(f"class {cls} - {count} elements")
        print(f"total - {total_after} elements")

        print(f"[LABELS] shape: {binary_labels.shape}, dtype: {binary_labels.dtype}, unique values: {np.unique(binary_labels)}")
        print(f"[LABELS] examples:\n{binary_labels[:5]}")


data_dir = "data/bloodmnist"
binary_mapping_required = {"bloodmnist", "octmnist", "pathmnist"}

for split in ["train", "val", "test"]:
    analyze_split(split, data_dir)
