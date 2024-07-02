import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
from PIL import Image
import numpy as np
from check_data import im_list_dcm

# Define mean and std (you need to compute these for your uint16 dataset)
mean = 0.6116309762001038
std = 0.24248747527599335

# Define transformation pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std])
])

# Define the SpineDataset class
class SpineDataset(Dataset):
    def __init__(self, im_list_dcm, transform=None, max_images=54):
        self.im_list_dcm = im_list_dcm
        self.study_ids = list(im_list_dcm.keys())
        self.transform = transform
        self.max_images = max_images

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

        # Pad with zeros if the number of images is less than max_images
        while len(series_images) < self.max_images:
            series_images.append(torch.zeros_like(series_images[0]))

        all_images = torch.stack(series_images, dim=0)
        labels = torch.tensor(0)  # Placeholder for labels if not available

        return all_images, labels

# Initialize the model with pretrained weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the last layer of the model to match the number of output classes
num_classes = 25 * 3  # Assuming 25 heads each predicting 3 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create dataset and dataloader
dataset = SpineDataset(im_list_dcm, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define loss and optimizer
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
        outputs = outputs.view(-1, 3)  # Reshape outputs to match the number of classes
        labels = labels.view(-1)  # Flatten labels
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

print('Finished Training')
torch.save(model.state_dict(), 'resnet50_spine.pth')
