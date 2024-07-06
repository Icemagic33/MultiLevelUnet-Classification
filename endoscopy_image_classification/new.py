import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import all_labels, all_preds, labels

# Select 2 images from each class for visualization
selected_images = []
selected_labels = []
selected_preds = []

# Assuming `all_labels` and `all_preds` are lists of ground truth and predicted labels respectively
for label in labels:
    count = 0
    for i in range(len(all_labels)):
        if labels[all_labels[i]] == label:
            img = X_test[i].permute(1, 2, 0).numpy() * 255.0
            selected_images.append(img.astype(np.uint8))
            selected_labels.append(labels[all_labels[i]])
            selected_preds.append(labels[all_preds[i]])
            count += 1
        if count == 2:
            break

# Plotting function
def plot_images(images, gt_labels, pred_labels, n_cols=4):
    n_rows = len(images) // n_cols + (1 if len(images) % n_cols > 0 else 0)
    plt.figure(figsize=(15, n_rows * 5))

    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i])
        plt.title(f"GT: {gt_labels[i]}\nPred: {pred_labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_images(selected_images, selected_labels, selected_preds)
