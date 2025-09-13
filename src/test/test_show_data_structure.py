import numpy as np

# path to data
path = "data/dermamnist"

# load images and labels
images = np.load(f"{path}/train_images.npy")
labels = np.load(f"{path}/train_labels.npy")

# check the pixel value range
print(f"[IMAGES] dtype: {images.dtype}")
print(f"[IMAGES] min: {images.min()}, max: {images.max()}")
print(f"[IMAGES] shape: {images.shape}")
print(images[0].shape)

unique_shapes = set([img.shape for img in images])
print("Unikalne kształty obrazów:", unique_shapes)

# check labels
print(f"[LABELS] dtype: {labels.dtype}")
print(f"[LABELS] shape: {labels.shape}")
print(f"[LABELS] unikalne wartości: {np.unique(labels)}")
print(f"[LABELS] przykład(y):\n{labels[:5]}")