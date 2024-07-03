import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths for data
csv_file = "/kaggle/input/processed-rsna/processed_csv/df_train_OneHot.csv"
root = "/kaggle/input/processed-rsna/rsna_dataset_dcm2Image/train_images"

# Data preprocessing with augmentation
transform = transforms.Compose([
    transforms.Resize((576, 576)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Load the CSV file containing image paths and labels
        self.transform = transform  # Store the transformations to be applied on images
    
    def __len__(self):
        return len(self.data)  # Return the number of samples in the dataset
    
    def __getitem__(self, idx):
        # Construct the image file path based on the CSV file data
        image_id = os.path.join(root,
                                self.data.iloc[idx, 0].astype(str),
                                self.data.iloc[idx, 1].astype(str),
                                f"{self.data.iloc[idx, 2]}.jpg")
        image = Image.open(image_id).convert('L')  # Open the image file and convert to grayscale
        
        # Convert grayscale image to three-channel image by duplicating the channel
        image = image.convert('RGB')
        
        # Apply transformations if any are specified
        if self.transform:
            image = self.transform(image)
        
        # Load the labels, assuming the 4th column is the label index
        label = self.data.iloc[idx, 3:].values.astype(float)  # Ensure label is an integer representing the class index
        
        return image, label  # Return the image and its corresponding label