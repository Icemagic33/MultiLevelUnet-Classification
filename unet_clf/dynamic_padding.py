import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
from data_vis import SpineDataset, transform
from model_multi_level import DoubleConv, MultiLevelUnet
from check_data import im_list_dcm

# Define the dynamic padding class
class DynamicPad:
    def __init__(self, max_width, max_height):
        self.max_width = max_width
        self.max_height = max_height

    def __call__(self, img):
        width, height = img.size
        pad_width = self.max_width - width
        pad_height = self.max_height - height
        padding = (0, 0, pad_width, pad_height)  # (left, top, right, bottom)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

# Calculate max dimensions
def calculate_max_dimensions(dataset):
    max_width, max_height = 0, 0
    max_images = 0
    for i in range(len(dataset)):
        images, _ = dataset[i]
        num_images = images.size(0)
        if num_images > max_images:
            max_images = num_images
        for img in images:
            img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
            width, height = img.size
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
    return max_width, max_height, max_images

class SpineDataset(Dataset):
    def __init__(self, im_list_dcm, transform=None, max_images=54):
        self.im_list_dcm = im_list_dcm
        self.transform = transform
        self.study_ids = list(im_list_dcm.keys())
        self.max_images = max_images

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        data = self.im_list_dcm[study_id]

        series_images = []
        for image_info in data['images']:
            img = image_info['dicom'].pixel_array
            img = Image.fromarray(img).convert('L')  # Convert to grayscale

            if self.transform:
                img = self.transform(img)

            series_images.append(img)  # Append PIL image

        # Pad the sequence of images to have the same length
        if len(series_images) < self.max_images:
            img_size = series_images[0].size if len(series_images) > 0 else (0, 0)
            padding = [Image.new('L', img_size)] * (self.max_images - len(series_images))
            series_images.extend(padding)

        # Convert all PIL images to tensors
        series_images = [transforms.ToTensor()(img) for img in series_images]

        all_images = torch.stack(series_images, dim=0)
        labels = torch.zeros((25,), dtype=torch.long)  # Dummy labels

        return all_images, labels

# Define your data directory and load the dataset
fd = '/home/adam/data'
train = pd.read_csv(f'{fd}/train.csv')

# Step 1: Calculate maximum dimensions without any transformation
raw_dataset = SpineDataset(im_list_dcm, transform=None)
max_width, max_height, max_images = calculate_max_dimensions(raw_dataset)
print(f"Max Width: {max_width}, Max Height: {max_height}, Max Images: {max_images}")

# Step 2: Apply transformations including dynamic padding
transform = transforms.Compose([
    DynamicPad(max_width, max_height),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6116], std=[0.2425])
])

# Create the dataset and dataloader with transformations
dataset = SpineDataset(im_list_dcm, transform=transform, max_images=max_images)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLevelUnet(in_channels=1, num_heads=25, num_classes_per_head=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = sum(criterion(output, labels[:, i]) for i, output in enumerate(outputs))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

print('Finished Training')
torch.save(model.state_dict(), 'multi_level_unet.pth')
