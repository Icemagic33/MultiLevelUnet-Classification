import torch 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from check_data import im_list_dcm
from data_vis import SpineDataset


# Define transformation 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Creating SpineDataset instance
dataset = SpineDataset(im_list_dcm, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size 1 to handle varying series lengths

# Function to compute mean and standard deviation
def compute_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for batch in loader:
        images_batch = batch[0]  # batch is a list of tensors, batch[0] to get the tensor itself
        num_images_in_batch = images_batch.size(0)  # First dimension is the number of images
        images_batch = images_batch.view(num_images_in_batch, -1)  # Flatten the image pixels
        mean += images_batch.mean(1).sum(0)  # Compute mean per image and sum them up
        std += images_batch.std(1).sum(0)  # Compute std per image and sum them up
        total_images_count += num_images_in_batch
    
    mean /= total_images_count
    std /= total_images_count
    return mean, std

# Calculate mean and std
mean, std = compute_mean_std(loader)
print(f"Mean: {mean}, Std: {std}")
