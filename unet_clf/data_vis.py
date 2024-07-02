import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
from check_data import im_list_dcm

# Access to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# File location
fd = "/home/adam/data"
train = pd.read_csv(f'{fd}/train.csv')

# Define mean and std (you need to compute these for your uint16 dataset)
mean = 0.6116309762001038
std = 0.24248747527599335

# Define transformation pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])


# Define the SpineDataset class
class SpineDataset(Dataset):
    def __init__(self, im_list_dcm, transform=None):
        self.im_list_dcm = im_list_dcm
        self.study_ids = list(im_list_dcm.keys())
        self.transform = transform

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        data = self.im_list_dcm[study_id]
        
        series_images = []
        for image_dict in data['images']:
            img = image_dict['dicom'].pixel_array
            img = img.astype(np.float32)  # Work with float32 to preserve precision
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255  # Normalize to 0-255
            img = img.astype(np.uint8)  # Convert to uint8 for the transformation
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            img = np.repeat(img, 3, axis=-1)  # Repeat the channel to convert to 3 channels
            if self.transform:
                img = self.transform(img)
            series_images.append(img)

        if len(series_images) == 0:
            raise RuntimeError(f"No valid images found for study ID {study_id}")

        all_images = torch.stack(series_images, dim=0)
        labels = torch.tensor(0)  # Placeholder for labels if not available
        
        return all_images, labels

# Create SpineDataset instance with the updated transform
dataset = SpineDataset(im_list_dcm, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Save the first image in the loader as a PNG file
for images, _ in loader:
    img = images[0][0].numpy().transpose(1, 2, 0)  # Convert from Tensor (C, H, W) to numpy (H, W, C)
    img = (img * std + mean).squeeze()  # Denormalize for visualization

    plt.figure()  # Create a new figure for the image plot
    plt.imshow(img, cmap='gray')
    plt.title('Example Image after Normalization')
    plt.axis('off')
    plt.savefig('example_image_after_normalization.png', bbox_inches='tight')  # Save the image as PNG
    plt.show()
    break
