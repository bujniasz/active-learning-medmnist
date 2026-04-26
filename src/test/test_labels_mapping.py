import numpy as np
from utils.labels_mapping import map_labels

original = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
binary = map_labels("bloodmnist", original)

print("Original:", original.squeeze())
print("Binary  :", binary)